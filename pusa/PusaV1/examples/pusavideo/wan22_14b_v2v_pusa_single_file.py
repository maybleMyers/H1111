from PIL import Image
import torch
import os
import sys
import argparse
from diffsynth import ModelManagerWan22, Wan22VideoPusaV2VPipeline, save_video
import datetime
import cv2

def process_video_frames(video_path, target_width=832, target_height=480):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []
        # Get original video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate scaling and cropping parameters
    target_ratio = target_width / target_height
    original_ratio = width / height

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize maintaining aspect ratio
        if original_ratio > target_ratio:
            # Video is wider than target
            new_width = int(height * target_ratio)
            # Crop width from center
            start_x = (width - new_width) // 2
            frame = frame[:, start_x:start_x + new_width]
        else:
            # Video is taller than target
            new_height = int(width / target_ratio)
            # Crop height from center
            start_y = (height - new_height) // 2
            frame = frame[start_y:start_y + new_height]

        # Resize to target dimensions
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()
    return frames

def main():
    # Enable PyTorch memory optimization settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

    parser = argparse.ArgumentParser(description="Pusa V2V: Video-to-Video Generation with dual DiT models")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the conditioning video.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--negative_prompt", type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", help="Negative text prompt for video generation.")
    parser.add_argument("--cond_position", type=str, default=None, help="Comma-separated list of frame indices for conditioning.")
    parser.add_argument("--extend_from_end", type=int, default=None, help="Number of frames from the end of `--video_path` to use for conditioning the start of the new video. Mutually exclusive with `--cond_position`.")
    parser.add_argument("--noise_multipliers", type=str, required=True, help="Comma-separated noise multipliers for conditioning frames.")

    # Model file arguments - now support both single files and directories
    parser.add_argument("--high_model", type=str, default="model_zoo/PusaV1/Wan2.2-T2V-A14B/high_noise_model",
                       help="Path to high noise DiT model - can be a single .safetensors file or a directory containing multiple files.")
    parser.add_argument("--low_model", type=str, default="model_zoo/PusaV1/Wan2.2-T2V-A14B/low_noise_model",
                       help="Path to low noise DiT model - can be a single .safetensors file or a directory containing multiple files.")
    parser.add_argument("--base_dir", type=str, default="model_zoo/PusaV1/Wan2.2-T2V-A14B",
                       help="Directory of the T2V model components (T5, VAE).")

    # LoRA arguments
    parser.add_argument("--high_lora_path", type=str, default="", help="Path(s) to the LoRA checkpoint file(s) for high noise model. Multiple paths separated by comma. Optional.")
    parser.add_argument("--high_lora_alpha", type=str, default="1.4", help="Alpha value(s) for high noise LoRA. Multiple values separated by comma.")
    parser.add_argument("--low_lora_path", type=str, default="", help="Path(s) to the LoRA checkpoint file(s) for low noise model. Multiple paths separated by comma. Optional.")
    parser.add_argument("--low_lora_alpha", type=str, default="1.4", help="Alpha value(s) for low noise LoRA. Multiple values separated by comma.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--switch_DiT_boundary", type=float, default=0.875, help="Boundary to switch between DiT models.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the output video.")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale.")
    parser.add_argument("--shift", type=float, default=5.0, help="Sigma shift parameter for flow matching scheduler. Default: 5.0")
    parser.add_argument("--lightx2v", action="store_true", help="Use lightx2v for acceleration.")
    parser.add_argument("--concatenate", action="store_true", help="Automatically concatenate the original video with the generated video for a final extended output. Only works with `--extend_from_end`.")
    parser.add_argument("--num_persistent_params", type=float, default=6e9, help="Number of persistent parameters in DiT for VRAM management. Use scientific notation (e.g., 6e9 for 6 billion).")
    parser.add_argument("--width", type=int, default=832, help="Width of the output video. Default: 832")
    parser.add_argument("--height", type=int, default=480, help="Height of the output video. Default: 480")
    parser.add_argument("--fps", type=int, default=24, help="fps to save video in")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation. Default: 0")
    parser.add_argument("--preview", type=int, default=0, help="Preview interval in steps (0 to disable)")
    args = parser.parse_args()

    # --- Argument Validation and Input Preparation ---
    if args.extend_from_end is not None and args.cond_position is not None:
        raise ValueError("Cannot use both `--extend_from_end` and `--cond_position`. Please choose one.")
    if args.extend_from_end is None and args.cond_position is None:
        raise ValueError("Either `--extend_from_end` or `--cond_position` must be specified for conditioning.")
    if args.concatenate and args.extend_from_end is None:
        raise ValueError("`--concatenate` can only be used with `--extend_from_end` for video extension.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Models ---
    print("Loading models...")
    model_manager = ModelManagerWan22(device="cpu")

    # Helper function to handle both single file and directory inputs
    def get_model_path(path):
        """Returns either the single file path or a list of safetensors files from directory"""
        if os.path.isfile(path) and path.endswith('.safetensors'):
            # Single file case
            print(f"  Using single model file: {path}")
            return path
        elif os.path.isdir(path):
            # Directory case - get all safetensors files
            files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.safetensors')])
            if not files:
                raise ValueError(f"No .safetensors files found in directory: {path}")
            print(f"  Found {len(files)} model files in: {path}")
            return files
        else:
            raise ValueError(f"Invalid model path: {path} (must be a .safetensors file or directory containing .safetensors files)")

    # Get model paths (can be single files or lists)
    high_model_path = get_model_path(args.high_model)
    low_model_path = get_model_path(args.low_model)

    # Load all models with proper model type tagging for LoRA matching
    model_manager.load_models(
        [
            (high_model_path, "high"),  # Tag as high noise model
            (low_model_path, "low"),    # Tag as low noise model
            os.path.join(args.base_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(args.base_dir, "Wan2.1_VAE.pth"),
        ],
        torch_dtype=torch.bfloat16,
    )

    if args.lightx2v:
        # Lightx2v for acceleration
        high_lora_path = "./model_zoo/PusaV1/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors"
        model_manager.load_loras_wan22_lightx2v(high_lora_path, model_type="high")
        low_lora_path = "./model_zoo/PusaV1/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors"
        model_manager.load_loras_wan22_lightx2v(low_lora_path, model_type="low")

    # Parse multiple LoRA paths and alphas (handle empty strings)
    if args.high_lora_path and args.high_lora_path.strip():
        high_lora_paths = [path.strip() for path in args.high_lora_path.split(',') if path.strip()]
        high_lora_alphas = [float(alpha.strip()) for alpha in args.high_lora_alpha.split(',') if alpha.strip()]

        # Validate matching counts
        if len(high_lora_paths) != len(high_lora_alphas):
            raise ValueError(f"Number of high LoRA paths ({len(high_lora_paths)}) must match number of high LoRA alphas ({len(high_lora_alphas)})")

        # Load multiple LoRAs for high noise model
        for lora_path, lora_alpha in zip(high_lora_paths, high_lora_alphas):
            print(f"Loading high noise LoRA: {lora_path} with alpha={lora_alpha}")
            model_manager.load_loras_wan22(lora_path, lora_alpha=lora_alpha, model_type="high")
    else:
        print("No high noise LoRAs specified, using base model")

    if args.low_lora_path and args.low_lora_path.strip():
        low_lora_paths = [path.strip() for path in args.low_lora_path.split(',') if path.strip()]
        low_lora_alphas = [float(alpha.strip()) for alpha in args.low_lora_alpha.split(',') if alpha.strip()]

        # Validate matching counts
        if len(low_lora_paths) != len(low_lora_alphas):
            raise ValueError(f"Number of low LoRA paths ({len(low_lora_paths)}) must match number of low LoRA alphas ({len(low_lora_alphas)})")

        # Load multiple LoRAs for low noise model
        for lora_path, lora_alpha in zip(low_lora_paths, low_lora_alphas):
            print(f"Loading low noise LoRA: {lora_path} with alpha={lora_alpha}")
            model_manager.load_loras_wan22(lora_path, lora_alpha=lora_alpha, model_type="low")
    else:
        print("No low noise LoRAs specified, using base model")
    
    # Create pipeline with CPU device first to avoid loading to GPU
    pipe = Wan22VideoPusaV2VPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cpu")

    # Now set the actual device and enable VRAM management
    pipe.device = device
    pipe.enable_vram_management(num_persistent_param_in_dit=int(args.num_persistent_params))

    # Clear any cached memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Models loaded successfully with VRAM management enabled")
    print(f"  Persistent parameters in DiT: {args.num_persistent_params/1e9:.2f}B")
    
    # --- Prepare Conditioning Inputs ---
    all_video_frames = process_video_frames(args.video_path, target_width=args.width, target_height=args.height)
    noise_mult_list = [float(x.strip()) for x in args.noise_multipliers.split(',')]

    # Initialize preview handler if enabled
    preview_handler = None
    if args.preview > 0:
        from diffsynth.utils.latent_preview import LatentPreviewHandler
        timestamp_preview = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        preview_suffix = f"pusa_v2v_{timestamp_preview}"
        preview_handler = LatentPreviewHandler(
            save_path=args.output_dir,
            fps=args.fps // 4,  # Lower FPS for preview
            preview_suffix=preview_suffix
        )
        print(f"Preview enabled: generating preview every {args.preview} steps")

    if args.extend_from_end:
        print(f"Video extension mode: Using last {args.extend_from_end} frames for conditioning.")
        if args.extend_from_end > len(all_video_frames):
            raise ValueError(f"`--extend_from_end` ({args.extend_from_end}) is greater than the number of frames in the video ({len(all_video_frames)}).")
        if len(noise_mult_list) != args.extend_from_end:
            raise ValueError(f"Number of noise multipliers ({len(noise_mult_list)}) must match `--extend_from_end` ({args.extend_from_end}).")

        # Use the last N frames of the input video
        conditioning_video = all_video_frames[-args.extend_from_end:]
        # Condition the first N frames of the output video
        cond_pos_list = list(range(args.extend_from_end))
    else:
        # Standard conditioning mode
        print(f"Standard conditioning mode: Using full video with specified positions.")
        cond_pos_list = [int(x.strip()) for x in args.cond_position.split(',')]
        if len(noise_mult_list) != len(cond_pos_list):
            raise ValueError(f"Number of noise multipliers ({len(noise_mult_list)}) must match the number of conditioning positions ({len(cond_pos_list)}).")
        conditioning_video = all_video_frames

    # --- Run Pipeline ---
    print("Generating new video frames...")
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        conditioning_video=conditioning_video,
        conditioning_indices=cond_pos_list,
        conditioning_noise_multipliers=noise_mult_list,
        num_inference_steps=args.num_inference_steps,
        height=args.height, width=args.width, num_frames=81,
        seed=args.seed, tiled=True,
        switch_DiT_boundary=args.switch_DiT_boundary,
        cfg_scale=args.cfg_scale,
        sigma_shift=args.shift,
        preview_handler=preview_handler,
        preview_interval=args.preview
    )
    print("Video generation complete.")

    # --- Save Output ---
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename_base = os.path.basename(args.video_path).split('.')[0]

    # Create a descriptive filename stem
    if args.extend_from_end:
        mode_str = f"extend_{args.extend_from_end}frames"
    else:
        mode_str = f"cond_{str(cond_pos_list)}"

    base_video_filename = f"wan22_v2v_{output_filename_base}_{timestamp}_{mode_str}_noise_{str(noise_mult_list)}_cfg_{args.cfg_scale}_steps_{args.num_inference_steps}"
    if args.lightx2v:
        base_video_filename += "_lightx2v"

    # Decide which frames to save based on the --concatenate flag
    if args.concatenate and args.extend_from_end:
        print("Concatenating original video with the generated video...")
        final_video_frames = all_video_frames + video
        final_filename_stem = base_video_filename.replace(f"extend_{args.extend_from_end}frames", f"extended_total_{len(final_video_frames)}frames")
        video_to_save = final_video_frames
        video_filename = os.path.join(args.output_dir, final_filename_stem + ".mp4")
    else:
        video_to_save = video
        video_filename = os.path.join(args.output_dir, base_video_filename + ".mp4")

    print(f"Saving video to {video_filename}")
    save_video(video_to_save, video_filename, fps=args.fps, quality=5)
    print("Video saved successfully.")

if __name__ == "__main__":
    main()