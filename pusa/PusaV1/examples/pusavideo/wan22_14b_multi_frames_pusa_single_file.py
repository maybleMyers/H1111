from PIL import Image
import torch
import os
import sys
import argparse
from diffsynth import ModelManagerWan22, Wan22VideoPusaMultiFramesPipeline, save_video
import datetime

def main():
    parser = argparse.ArgumentParser(description="Pusa Conditional Video Generation from one or more images using dual DiT models.")
    parser.add_argument("--image_paths", type=str, nargs='+', required=True, help="Paths to one or more conditioning image frames.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--negative_prompt", type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", help="Negative text prompt for video generation.")
    parser.add_argument("--cond_position", type=str, required=True, help="Comma-separated list of frame indices for conditioning. You can use any position from 0 to 20.")
    parser.add_argument("--noise_multipliers", type=str, required=True, help="Comma-separated noise multipliers for conditioning frames. A value of 0 means the condition image is used as totally clean, higher value means add more noise.")
    parser.add_argument("--base_dir", type=str, default="model_zoo/PusaV1/Wan2.2-T2V-A14B", help="Directory of the T2V model components (T5, VAE).")

    # Model file arguments - now support both single files and directories
    parser.add_argument("--high_model", type=str, default="model_zoo/PusaV1/Wan2.2-T2V-A14B/high_noise_model",
                       help="Path to high noise DiT model - can be a single .safetensors file or a directory containing multiple files.")
    parser.add_argument("--low_model", type=str, default="model_zoo/PusaV1/Wan2.2-T2V-A14B/low_noise_model",
                       help="Path to low noise DiT model - can be a single .safetensors file or a directory containing multiple files.")

    parser.add_argument("--high_lora_path", type=str, required=True, help="Path to the high noise LoRA checkpoint file.")
    parser.add_argument("--high_lora_alpha", type=float, default=1.4, help="Alpha value for high noise LoRA.")
    parser.add_argument("--low_lora_path", type=str, required=True, help="Path to the low noise LoRA checkpoint file.")
    parser.add_argument("--low_lora_alpha", type=float, default=1.4, help="Alpha value for low noise LoRA.")

    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--switch_DiT_boundary", type=float, default=0.875, help="Boundary to switch from high noise DiT to low noise DiT.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the output video.")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale.")
    parser.add_argument("--shift", type=float, default=5.0, help="Sigma shift parameter for flow matching scheduler. Default: 5.0")
    parser.add_argument("--lightx2v", action="store_true", help="Use lightx2v for acceleration.")
    parser.add_argument("--num_persistent_params", type=float, default=6e9, help="Number of persistent parameters in DiT for VRAM management. Use scientific notation (e.g., 6e9 for 6 billion).")
    parser.add_argument("--width", type=int, default=1280, help="Width of the output video. Default: 1280")
    parser.add_argument("--height", type=int, default=720, help="Height of the output video. Default: 720")
    parser.add_argument("--fps", type=int, default=24, help="FPS to save video in. Default: 24")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate. Default: 81")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation. Default: 0")
    parser.add_argument("--preview", type=int, default=0, help="Preview interval in steps (0 to disable)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
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
        high_lora_path = "pusa/pusa_lora/high_noise_model.safetensors"
        model_manager.load_loras_wan22_lightx2v(high_lora_path, model_type="high")
        low_lora_path = "pusa/pusa_lora/low_noise_model.safetensors"
        model_manager.load_loras_wan22_lightx2v(low_lora_path, model_type="low")

    # Load LoRAs
    model_manager.load_loras_wan22(args.high_lora_path, lora_alpha=args.high_lora_alpha, model_type="high")
    model_manager.load_loras_wan22(args.low_lora_path, lora_alpha=args.low_lora_alpha, model_type="low")

    # Create pipeline with CPU device first to avoid loading to GPU
    pipe = Wan22VideoPusaMultiFramesPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cpu")

    # Now set the actual device and enable VRAM management
    pipe.device = device
    pipe.enable_vram_management(num_persistent_param_in_dit=int(args.num_persistent_params))

    # Clear any cached memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Models loaded successfully with VRAM management enabled")
    print(f"  Persistent parameters in DiT: {args.num_persistent_params/1e9:.2f}B")

    cond_pos_list = [int(x.strip()) for x in args.cond_position.split(',')]
    noise_mult_list = [float(x.strip()) for x in args.noise_multipliers.split(',')]

    # Initialize preview handler if enabled
    preview_handler = None
    if args.preview > 0:
        from diffsynth.utils.latent_preview import LatentPreviewHandler
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        preview_suffix = f"pusa_i2v_{timestamp}"
        preview_handler = LatentPreviewHandler(
            save_path=args.output_dir,
            fps=args.fps // 4,  # Lower FPS for preview
            preview_suffix=preview_suffix
        )
        print(f"Preview enabled: generating preview every {args.preview} steps")

    images = []
    target_w, target_h = args.width, args.height
    for p in args.image_paths:
        img = Image.open(p).convert("RGB")
        original_w, original_h = img.size

        ratio = min(target_w / original_w, target_h / original_h)
        new_w = int(original_w * ratio)
        new_h = int(original_h * ratio)

        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        background = Image.new('RGB', (target_w, target_h), (0, 0, 0))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        background.paste(img_resized, (paste_x, paste_y))
        images.append(background)

    if len(images) != len(cond_pos_list) or len(images) != len(noise_mult_list):
        raise ValueError("The number of --image_paths, --cond_position, and --noise_multipliers must be the same.")

    multi_frame_images = {
        cond_pos: (img, noise_mult)
        for cond_pos, img, noise_mult in zip(cond_pos_list, images, noise_mult_list)
    }

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        multi_frame_images=multi_frame_images,
        num_inference_steps=args.num_inference_steps,
        height=args.height, width=args.width, num_frames=args.num_frames,
        seed=args.seed, tiled=True,
        switch_DiT_boundary=args.switch_DiT_boundary,
        cfg_scale=args.cfg_scale,
        sigma_shift=args.shift,
        preview_handler=preview_handler,
        preview_interval=args.preview
    )

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.lightx2v:
        video_filename= os.path.join(args.output_dir, f"wan22_multi_frame_output_{timestamp}_cond_{str(cond_pos_list)}_noise_{str(noise_mult_list)}_high_alpha_{args.high_lora_alpha}_low_alpha_{args.low_lora_alpha}_cfg_{args.cfg_scale}_steps_{args.num_inference_steps}_lightx2v.mp4")
    else:
        video_filename = os.path.join(args.output_dir, f"wan22_multi_frame_output_{timestamp}_cond_{str(cond_pos_list)}_noise_{str(noise_mult_list)}_high_alpha_{args.high_lora_alpha}_low_alpha_{args.low_lora_alpha}_cfg_{args.cfg_scale}_steps_{args.num_inference_steps}.mp4")

    print(f"Saved to {video_filename}")
    save_video(video, video_filename, fps=args.fps, quality=5)

if __name__ == "__main__":
    main()