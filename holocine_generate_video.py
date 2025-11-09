"""
HoloCine Multi-Shot Video Generation Script

This script uses the HoloCine DiffSynth pipeline for multi-shot video generation.
It provides a command-line interface for the WanVideoHoloCinePipeline.

Usage:
    python holocine_generate_video.py --global_caption "..." --shot_captions "..." "..." --num_frames 241 --save_path output.mp4

Based on HoloCine reference implementation.
"""

import torch
import argparse
import logging
from pathlib import Path

# Import HoloCine pipeline
try:
    from HoloCine.diffsynth import save_video
    from HoloCine.diffsynth.pipelines.wan_video_holocine import WanVideoHoloCinePipeline, ModelConfig
except ImportError as e:
    print(f"Error importing HoloCine modules: {e}")
    print("Make sure the HoloCine directory is present in the working directory.")
    raise

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def enforce_4t_plus_1(n: int) -> int:
    """Forces an integer 'n' to the closest 4t+1 form."""
    t = round((n - 1) / 4)
    return 4 * t + 1

def prepare_multishot_inputs(
    global_caption: str,
    shot_captions: list[str],
    total_frames: int,
    custom_shot_cut_frames: list[int] = None
) -> dict:
    """
    Prepares the inference parameters from user-friendly segmented inputs.
    Returns dict with 'prompt', 'shot_cut_frames', 'num_frames'.
    """

    num_shots = len(shot_captions)

    # 1. Prepare 'prompt' in HoloCine format
    if "This scene contains" not in global_caption:
        global_caption = global_caption.strip() + f" This scene contains {num_shots} shots."
    per_shot_string = " [shot cut] ".join(shot_captions)
    prompt = f"[global caption] {global_caption} [per shot caption] {per_shot_string}"

    # 2. Prepare 'num_frames' (enforce 4t+1)
    processed_total_frames = enforce_4t_plus_1(total_frames)

    # 3. Prepare 'shot_cut_frames'
    num_cuts = num_shots - 1
    processed_shot_cuts = []

    if custom_shot_cut_frames:
        # User provided custom cuts
        logger.info(f"Using {len(custom_shot_cut_frames)} user-defined shot cuts (enforcing 4t+1).")
        for frame in custom_shot_cut_frames:
            processed_shot_cuts.append(enforce_4t_plus_1(frame))
    else:
        # Auto-calculate cuts
        logger.info(f"Auto-calculating {num_cuts} shot cuts.")
        if num_cuts > 0:
            ideal_step = processed_total_frames / num_shots
            for i in range(1, num_shots):
                approx_cut_frame = i * ideal_step
                processed_shot_cuts.append(enforce_4t_plus_1(round(approx_cut_frame)))

    processed_shot_cuts = sorted(list(set(processed_shot_cuts)))
    processed_shot_cuts = [f for f in processed_shot_cuts if f > 0 and f < processed_total_frames]

    return {
        "prompt": prompt,
        "shot_cut_frames": processed_shot_cuts,
        "num_frames": processed_total_frames
    }

def parse_args():
    parser = argparse.ArgumentParser(description="HoloCine Multi-Shot Video Generation")

    # Model paths
    parser.add_argument("--dit_high_noise", type=str, required=True,
                       help="Path to high noise DiT model (safetensors)")
    parser.add_argument("--dit_low_noise", type=str, required=True,
                       help="Path to low noise DiT model (safetensors)")
    parser.add_argument("--vae", type=str, required=True,
                       help="Path to VAE model (pth)")
    parser.add_argument("--t5", type=str, required=True,
                       help="Path to T5 encoder model (pth or safetensors)")

    # Input mode: Structured multi-shot OR raw prompt
    parser.add_argument("--global_caption", type=str, default=None,
                       help="Global scene caption (for structured multi-shot)")
    parser.add_argument("--shot_captions", type=str, nargs="*", default=None,
                       help="List of per-shot captions (for structured multi-shot)")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Raw HoloCine format prompt (alternative to global_caption + shot_captions)")

    # Generation parameters
    parser.add_argument("--num_frames", type=int, default=None,
                       help="Number of frames to generate (will be adjusted to 4t+1)")
    parser.add_argument("--shot_cut_frames", type=int, nargs="*", default=None,
                       help="Frame indices where shots change (for custom shot cuts)")
    parser.add_argument("--negative_prompt", type=str, default=None,
                       help="Negative prompt")

    # Video parameters
    parser.add_argument("--height", type=int, default=480,
                       help="Video height (default: 480)")
    parser.add_argument("--width", type=int, default=832,
                       help="Video width (default: 832)")
    parser.add_argument("--fps", type=int, default=15,
                       help="Output video FPS (default: 15)")

    # Inference parameters
    parser.add_argument("--infer_steps", type=int, default=50,
                       help="Number of inference steps (default: 50)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed (default: 0)")
    parser.add_argument("--tiled", action="store_true", default=True,
                       help="Use tiled VAE (default: True)")

    # Output
    parser.add_argument("--save_path", type=str, required=True,
                       help="Path to save output video")
    parser.add_argument("--quality", type=int, default=5,
                       help="Output video quality (default: 5)")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (default: cuda)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16"],
                       help="Model dtype (default: bfloat16)")

    # Memory management
    parser.add_argument("--offload_device", type=str, default="cpu",
                       help="Device to offload models to (default: cpu)")
    parser.add_argument("--enable_vram_management", action="store_true", default=True,
                       help="Enable VRAM management (default: True)")

    # Alias for num_frames (compatibility)
    parser.add_argument("--video_length", type=int, default=None,
                       help="Alias for --num_frames")

    # Compatibility arguments (ignored but accepted)
    parser.add_argument("--blocks_to_swap", type=int, default=None,
                       help="(Ignored - for compatibility only)")

    return parser.parse_args()

def main():
    args = parse_args()

    # Handle video_length alias
    if args.video_length is not None and args.num_frames is None:
        args.num_frames = args.video_length

    logger.info("=== HoloCine Multi-Shot Video Generation ===")

    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype = dtype_map[args.dtype]

    # Initialize pipeline
    logger.info("Loading HoloCine pipeline...")
    logger.info(f"Device: {args.device}, Dtype: {args.dtype}")

    try:
        pipe = WanVideoHoloCinePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=args.device,
            model_configs=[
                ModelConfig(path=args.t5, offload_device=args.offload_device),
                ModelConfig(path=args.dit_high_noise, offload_device=args.offload_device),
                ModelConfig(path=args.dit_low_noise, offload_device=args.offload_device),
                ModelConfig(path=args.vae, offload_device=args.offload_device),
            ],
        )

        if args.enable_vram_management:
            pipe.enable_vram_management()
            logger.info("VRAM management enabled")

        pipe.to(args.device)
        logger.info("Pipeline loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        raise

    # Prepare generation parameters
    pipe_kwargs = {
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "tiled": args.tiled,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.infer_steps
    }

    # Determine input mode and prepare inputs
    if args.global_caption and args.shot_captions:
        # Mode 1: Structured multi-shot input
        logger.info("=== Mode 1: Structured Multi-Shot Input ===")

        if args.num_frames is None:
            raise ValueError("Must provide --num_frames for structured multi-shot input")

        logger.info(f"Global caption: {args.global_caption[:100]}...")
        logger.info(f"Number of shots: {len(args.shot_captions)}")

        inputs = prepare_multishot_inputs(
            global_caption=args.global_caption,
            shot_captions=args.shot_captions,
            total_frames=args.num_frames,
            custom_shot_cut_frames=args.shot_cut_frames
        )
        pipe_kwargs.update(inputs)

        logger.info(f"Generated prompt: {inputs['prompt'][:150]}...")
        logger.info(f"Frames: {inputs['num_frames']}, Shot cuts: {inputs['shot_cut_frames']}")

    elif args.prompt:
        # Mode 2: Raw HoloCine format prompt
        logger.info("=== Mode 2: Raw HoloCine Format Prompt ===")

        pipe_kwargs["prompt"] = args.prompt

        # Process num_frames if provided
        if args.num_frames is not None:
            processed_frames = enforce_4t_plus_1(args.num_frames)
            if args.num_frames != processed_frames:
                logger.info(f"Adjusted num_frames: {args.num_frames} -> {processed_frames}")
            pipe_kwargs["num_frames"] = processed_frames

        # Process shot_cut_frames if provided
        if args.shot_cut_frames is not None:
            processed_cuts = [enforce_4t_plus_1(f) for f in args.shot_cut_frames]
            pipe_kwargs["shot_cut_frames"] = processed_cuts
            logger.info(f"Shot cuts: {processed_cuts}")

        logger.info(f"Prompt: {args.prompt[:150]}...")

    else:
        raise ValueError(
            "Must provide either:\n"
            "  1) --global_caption and --shot_captions (structured multi-shot), OR\n"
            "  2) --prompt (raw HoloCine format)"
        )

    # Filter out None values
    final_pipe_kwargs = {k: v for k, v in pipe_kwargs.items() if v is not None}

    # Run generation
    logger.info("=== Starting Video Generation ===")
    try:
        video = pipe(**final_pipe_kwargs)
        logger.info("Generation complete!")

        # Save video
        logger.info(f"Saving video to: {args.save_path}")
        save_video(video, args.save_path, fps=args.fps, quality=args.quality)
        logger.info(f"✅ Video successfully saved to: {args.save_path}")

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise

if __name__ == "__main__":
    main()
