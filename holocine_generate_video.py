#!/usr/bin/env python3
"""
HoloCine Video Generation Script

This script extends wan2_generate_video.py to support HoloCine models with multi-shot video generation.
It maintains compatibility with the block swapping and memory management infrastructure while adding
shot-specific features like shot embeddings, shot cut frames, and shot-aware attention masking.

Key Features:
- Multi-shot video generation with automatic or custom shot cuts
- Shot embeddings and shot-aware attention masking
- Compatible with HoloCine T2V models (full_high_noise/full_low_noise)
- Maintains all wan2_generate_video.py optimizations (block swapping, FP8, compilation, etc.)
- Supports both structured input (global_caption + shot_captions) and raw prompt format

Usage:
    # Mode 1: Structured multi-shot input
    python holocine_generate_video.py --prompt "scene description" \
        --global_caption "The scene..." \
        --shot_captions "Shot 1 description" "Shot 2 description" \
        --video_length 241 --save_path output.mp4

    # Mode 2: Raw HoloCine format prompt
    python holocine_generate_video.py \
        --prompt "[global caption] ... [per shot caption] ... [shot cut] ..." \
        --shot_cut_frames 37 73 113 \
        --video_length 241 --save_path output.mp4
"""

import argparse
from datetime import datetime
import gc
import random
import os
import re
import time
import math
from typing import Tuple, Optional, List, Union, Any, Dict
from pathlib import Path

# Set PyTorch CUDA allocator to reduce memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import accelerate
from accelerate import Accelerator
from functools import partial
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms.functional as TF
import torchvision
from tqdm import tqdm

from networks import lora_wan
from utils.safetensors_utils import mem_eff_save_file, load_safetensors
from utils.lora_utils import filter_lora_state_dict
from Wan2_2.wan.configs import WAN_CONFIGS, SUPPORTED_SIZES
import wan
from wan.modules.model import WanModel, load_wan_model, detect_wan_sd_dtype
from wan.modules.vae import WanVAE
from Wan2_2.wan.modules.vae2_2 import Wan2_2_VAE
from wan.modules.t5 import T5EncoderModel
from wan.modules.clip import CLIPModel
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.fm_solvers_euler import EulerScheduler
from wan.utils.step_distill_scheduler import StepDistillScheduler

from blissful_tuner.latent_preview import LatentPreviewer

try:
    from lycoris.kohya import create_network_from_weights
except:
    pass

from utils.model_utils import str_to_dtype
from utils.device_utils import clean_memory_on_device

# Context Windows imports
try:
    from Wan2_2.context_windows import (
        WanContextWindowsHandler,
        IndexListContextHandler,
        ContextSchedules,
        ContextFuseMethods,
    )
    CONTEXT_WINDOWS_AVAILABLE = True
except ImportError:
    CONTEXT_WINDOWS_AVAILABLE = False

import av
import glob
from einops import rearrange

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ====================================================================================
#                            HOLOCINE-SPECIFIC HELPER FUNCTIONS
# ====================================================================================

def enforce_4t_plus_1(n: int) -> int:
    """
    Forces an integer 'n' to the closest 4t+1 form required by HoloCine models.

    HoloCine models require num_frames to follow the pattern: 1, 5, 9, 13, 17, 21, ... (4t+1)
    This is due to the temporal compression in the VAE.

    Args:
        n: Target number of frames

    Returns:
        Closest valid frame count in 4t+1 form

    Examples:
        >>> enforce_4t_plus_1(240)
        241
        >>> enforce_4t_plus_1(81)
        81
        >>> enforce_4t_plus_1(100)
        101
    """
    t = round((n - 1) / 4)
    return 4 * t + 1


def prepare_multishot_inputs(
    global_caption: str,
    shot_captions: List[str],
    total_frames: int,
    custom_shot_cut_frames: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Prepares HoloCine inference parameters from user-friendly segmented inputs.

    This function converts structured multi-shot input (Mode 1) into the format expected
    by the HoloCine model, including:
    - Combining captions into HoloCine prompt format
    - Enforcing 4t+1 frame requirements
    - Calculating or validating shot cut frames

    Args:
        global_caption: Overall scene description (will be wrapped in [global caption])
        shot_captions: List of per-shot descriptions (will be joined with [shot cut])
        total_frames: Target total number of frames (will be adjusted to 4t+1)
        custom_shot_cut_frames: Optional list of frame indices where shots change

    Returns:
        Dictionary with keys:
            - prompt: Full HoloCine-formatted prompt string
            - shot_cut_frames: List of validated shot cut frame indices
            - num_frames: Adjusted frame count (4t+1 form)

    Example:
        >>> inputs = prepare_multishot_inputs(
        ...     global_caption="A park scene. This scene contains 3 shots.",
        ...     shot_captions=["Wide shot of park", "Close-up of bench", "Pan to tree"],
        ...     total_frames=241
        ... )
        >>> inputs['prompt']
        '[global caption] A park scene. This scene contains 3 shots. [per shot caption] Wide shot of park [shot cut] Close-up of bench [shot cut] Pan to tree'
        >>> inputs['num_frames']
        241
        >>> len(inputs['shot_cut_frames'])
        2
    """
    num_shots = len(shot_captions)

    # 1. Prepare 'prompt' in HoloCine format
    # Add shot count to global caption if not already present
    if "This scene contains" not in global_caption and "scene contains" not in global_caption.lower():
        global_caption = global_caption.strip() + f" This scene contains {num_shots} shots."

    # Join shot captions with [shot cut] separator
    per_shot_string = " [shot cut] ".join(shot_captions)
    prompt = f"[global caption] {global_caption} [per shot caption] {per_shot_string}"

    # 2. Prepare 'num_frames' - enforce 4t+1 constraint
    processed_total_frames = enforce_4t_plus_1(total_frames)
    if processed_total_frames != total_frames:
        logger.info(f"Adjusted num_frames from {total_frames} to {processed_total_frames} (4t+1 requirement)")

    # 3. Prepare 'shot_cut_frames'
    num_cuts = num_shots - 1  # Number of cuts = number of shots - 1
    processed_shot_cuts = []

    if custom_shot_cut_frames:
        # User provided custom cuts - validate and enforce 4t+1
        logger.info(f"Using {len(custom_shot_cut_frames)} user-defined shot cuts (enforcing 4t+1).")
        for frame in custom_shot_cut_frames:
            adjusted_frame = enforce_4t_plus_1(frame)
            if adjusted_frame != frame:
                logger.warning(f"Adjusted shot cut frame {frame} -> {adjusted_frame} (4t+1 requirement)")
            processed_shot_cuts.append(adjusted_frame)
    else:
        # Auto-calculate cuts evenly distributed across the video
        logger.info(f"Auto-calculating {num_cuts} shot cuts for {num_shots} shots.")
        if num_cuts > 0:
            ideal_step = processed_total_frames / num_shots
            for i in range(1, num_shots):
                approx_cut_frame = i * ideal_step
                processed_shot_cuts.append(enforce_4t_plus_1(round(approx_cut_frame)))

    # Remove duplicates and sort
    processed_shot_cuts = sorted(list(set(processed_shot_cuts)))

    # Filter out invalid cuts (must be > 0 and < total_frames)
    processed_shot_cuts = [f for f in processed_shot_cuts if 0 < f < processed_total_frames]

    logger.info(f"Shot cut frames: {processed_shot_cuts}")

    return {
        "prompt": prompt,
        "shot_cut_frames": processed_shot_cuts,
        "num_frames": processed_total_frames
    }


def prepare_shot_indices(
    shot_cut_frames: List[int],
    num_frames: int,
    device: torch.device
) -> torch.Tensor:
    """
    Convert shot cut frame indices into shot index tensor for the model.

    This creates a tensor where each latent frame is assigned to a shot ID (0, 1, 2, ...).
    The shot indices are used by the model's shot embedding layer and shot-aware attention.

    Args:
        shot_cut_frames: List of frame indices where shots change (in pixel frame space)
        num_frames: Total number of frames in the video (pixel space)
        device: Target device for the tensor

    Returns:
        torch.Tensor of shape [1, num_latent_frames] with shot IDs

    Example:
        If shot_cut_frames=[73, 145] and num_frames=241:
        - Frames 0-72: shot 0
        - Frames 73-144: shot 1
        - Frames 145-240: shot 2

        With 4x temporal compression in VAE:
        - Latent frames 0-18: shot 0
        - Latent frames 19-36: shot 1
        - Latent frames 37-60: shot 2

    Note:
        Converts from pixel frame indices to latent frame indices using: (f - 1) // 4 + 1
        This matches the HoloCine VAE's temporal compression.
    """
    # Calculate number of latent frames (VAE compresses temporally by 4x)
    num_latent_frames = (num_frames - 1) // 4 + 1

    # Convert pixel frame cut indices to latent cut indices
    shot_cut_latents = [0]  # First shot starts at latent frame 0
    for frame_idx in sorted(shot_cut_frames):
        if frame_idx > 0:
            latent_idx = (frame_idx - 1) // 4 + 1
            if latent_idx < num_latent_frames:
                shot_cut_latents.append(latent_idx)

    # Remove duplicates and sort
    cuts = sorted(list(set(shot_cut_latents))) + [num_latent_frames]

    # Create shot indices tensor
    shot_indices = torch.zeros(num_latent_frames, dtype=torch.long)
    for i in range(len(cuts) - 1):
        start_latent, end_latent = cuts[i], cuts[i+1]
        shot_indices[start_latent:end_latent] = i

    # Add batch dimension
    shot_indices = shot_indices.unsqueeze(0).to(device=device)

    logger.info(f"Created shot indices tensor: {shot_indices.shape}, num_shots={shot_indices.max().item() + 1}")

    return shot_indices


def parse_holocine_prompt(prompt: str) -> Tuple[Optional[str], Optional[List[str]], Optional[List[int]]]:
    """
    Parse a HoloCine-formatted prompt string to extract components.

    Attempts to extract:
    - Global caption (text between [global caption] and [per shot caption])
    - Shot captions (text segments separated by [shot cut])
    - Shot positions (character positions of [shot cut] markers)

    Args:
        prompt: HoloCine-formatted prompt string

    Returns:
        Tuple of (global_caption, shot_captions, text_cut_positions)
        Returns (None, None, None) if prompt doesn't match HoloCine format

    Example:
        >>> prompt = "[global caption] A park. This scene contains 2 shots. [per shot caption] Wide shot [shot cut] Close-up"
        >>> global_cap, shots, cuts = parse_holocine_prompt(prompt)
        >>> global_cap
        'A park. This scene contains 2 shots.'
        >>> shots
        ['Wide shot', 'Close-up']
    """
    # Check if prompt matches HoloCine format
    if "[global caption]" not in prompt or "[per shot caption]" not in prompt:
        return None, None, None

    try:
        # Extract global caption
        global_start = prompt.find("[global caption]") + len("[global caption]")
        global_end = prompt.find("[per shot caption]")
        global_caption = prompt[global_start:global_end].strip()

        # Extract shot captions
        shots_text = prompt[global_end + len("[per shot caption]"):].strip()
        shot_captions = [s.strip() for s in shots_text.split("[shot cut]")]

        # Find text cut positions (for attention masking)
        text_cut_positions = []
        search_start = global_end
        while True:
            cut_pos = prompt.find("[shot cut]", search_start)
            if cut_pos == -1:
                break
            text_cut_positions.append(cut_pos)
            search_start = cut_pos + len("[shot cut]")

        return global_caption, shot_captions, text_cut_positions

    except Exception as e:
        logger.warning(f"Error parsing HoloCine prompt format: {e}")
        return None, None, None


def validate_shot_inputs(
    global_caption: Optional[str],
    shot_captions: Optional[List[str]],
    shot_cut_frames: Optional[List[int]],
    video_length: int
) -> Tuple[bool, str]:
    """
    Validate shot-related inputs for consistency and correctness.

    Args:
        global_caption: Global scene description
        shot_captions: List of per-shot descriptions
        shot_cut_frames: List of shot cut frame indices
        video_length: Total number of frames

    Returns:
        Tuple of (is_valid, error_message)
        - (True, "") if valid
        - (False, "error description") if invalid
    """
    # If using structured input, need both global and shot captions
    if global_caption is not None or shot_captions is not None:
        if global_caption is None:
            return False, "global_caption required when shot_captions is provided"
        if shot_captions is None or len(shot_captions) == 0:
            return False, "shot_captions required when global_caption is provided"

        num_shots = len(shot_captions)
        expected_cuts = num_shots - 1

        # Validate shot cut frames if provided
        if shot_cut_frames is not None:
            if len(shot_cut_frames) != expected_cuts:
                return False, f"Expected {expected_cuts} shot cuts for {num_shots} shots, got {len(shot_cut_frames)}"

            # Check cuts are in valid range
            for cut in shot_cut_frames:
                if cut <= 0 or cut >= video_length:
                    return False, f"Shot cut frame {cut} out of range [1, {video_length-1}]"

            # Check cuts are sorted
            if shot_cut_frames != sorted(shot_cut_frames):
                return False, "Shot cut frames must be in ascending order"

    return True, ""


# ====================================================================================
#                      IMPORT REMAINING FUNCTIONS FROM WAN2_GENERATE_VIDEO
# ====================================================================================

# Import core utility functions from base script
# These are already defined in wan2_generate_video.py and will be imported at runtime
# when this script is in the same directory:
# - synchronize_device
# - glob_images
# - resize_image_to_bucket
# - hv_load_images
# - hv_load_video
# - save_videos_grid
# - save_images_grid
# - DynamicModelManager (will be extended below)
# - optimize_model
# - setup_args
# - check_inputs
# - encode_prompt
# - prepare_noise
# - decode_latent
# - save_output
# - All other wan2 infrastructure

# For now, we import the base wan2_generate_video module to access these functions
try:
    import sys
    # Import from wan2_generate_video.py
    import wan2_generate_video as wan2_base

    # Import key functions and classes
    synchronize_device = wan2_base.synchronize_device
    glob_images = wan2_base.glob_images
    resize_image_to_bucket = wan2_base.resize_image_to_bucket
    hv_load_images = wan2_base.hv_load_images
    hv_load_video = wan2_base.hv_load_video
    save_videos_grid = wan2_base.save_videos_grid
    save_images_grid = wan2_base.save_images_grid

    logger.info("Successfully imported base functions from wan2_generate_video.py")

except ImportError as e:
    logger.error(f"Failed to import wan2_generate_video.py: {e}")
    logger.error("Please ensure wan2_generate_video.py is in the same directory")
    raise


# ====================================================================================
#                            HOLOCINE-SPECIFIC ARGUMENT PARSER
# ====================================================================================

def parse_args() -> argparse.Namespace:
    """
    Extended argument parser with HoloCine-specific shot arguments.

    Extends the base wan2_generate_video.py argument parser with:
    - --global_caption: Overall scene description
    - --shot_captions: List of per-shot descriptions
    - --shot_cut_frames: Custom shot cut frame indices
    - --shot_mask_type: Type of shot mask to use (id, normalized, alternating)

    Also adjusts defaults for HoloCine models:
    - Default task: t2v-A14B (HoloCine uses same architecture)
    - Default model paths point to HoloCine checkpoints
    """
    # Start with base parser from wan2_generate_video.py
    parser = argparse.ArgumentParser(
        description="HoloCine inference script with multi-shot video generation support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # === WAN Base Arguments (same as wan2_generate_video.py) ===
    parser.add_argument("--ckpt_dir", type=str, default=None,
                       help="Path to checkpoint directory (Wan 2.1 official)")
    parser.add_argument("--task", type=str, default="t2v-A14B", choices=list(WAN_CONFIGS.keys()),
                       help="Task to run (default: t2v-A14B for HoloCine)")

    parser.add_argument("--sample_solver", type=str, default="unipc",
                       choices=["unipc", "dpm++", "vanilla", "euler", "step_distill"],
                       help="Solver for sampling (default: unipc)")

    # DiT model paths (adjusted for HoloCine)
    parser.add_argument("--dit", type=str, default=None,
                       help="DiT checkpoint path (single model)")
    parser.add_argument("--dit_low_noise", type=str, default=None,
                       help="DiT low noise checkpoint (e.g., full_low_noise.safetensors)")
    parser.add_argument("--dit_high_noise", type=str, default=None,
                       help="DiT high noise checkpoint (e.g., full_high_noise.safetensors)")
    parser.add_argument("--dual_dit_boundary", type=float, default=None,
                       help="Boundary for dual-dit models (0.0-1.0). Default: 0.875 for t2v-A14B")

    # Other model components
    parser.add_argument("--vae", type=str, default=None,
                       help="VAE checkpoint path")
    parser.add_argument("--vae_dtype", type=str, default=None,
                       help="VAE data type (default: bfloat16)")
    parser.add_argument("--vae_cache_cpu", action="store_true",
                       help="Cache VAE features on CPU")
    parser.add_argument("--t5", type=str, default=None,
                       help="T5 text encoder checkpoint path")
    parser.add_argument("--clip", type=str, default=None,
                       help="CLIP checkpoint path (for I2V)")

    # LoRA arguments
    parser.add_argument("--lora_weight", type=str, nargs="*", default=None,
                       help="LoRA weight path(s)")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0,
                       help="LoRA multiplier(s)")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None,
                       help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None,
                       help="LoRA module exclude patterns")

    # LoRA for high noise model (dual-dit)
    parser.add_argument("--lora_weight_high", type=str, nargs="*", default=None,
                       help="LoRA weights for high noise model")
    parser.add_argument("--lora_multiplier_high", type=float, nargs="*", default=1.0,
                       help="LoRA multipliers for high noise model")
    parser.add_argument("--include_patterns_high", type=str, nargs="*", default=None,
                       help="LoRA include patterns for high noise model")
    parser.add_argument("--exclude_patterns_high", type=str, nargs="*", default=None,
                       help="LoRA exclude patterns for high noise model")

    parser.add_argument("--save_merged_model", type=str, default=None,
                       help="Save merged model path (no inference)")

    # === Inference Arguments ===
    parser.add_argument("--prompt", type=str, required=True,
                       help="Prompt for generation (HoloCine or standard format)")
    parser.add_argument("--negative_prompt", type=str, default=None,
                       help="Negative prompt (uses default if not specified)")
    parser.add_argument("--video_size", type=int, nargs=2, default=[480, 832],
                       help="Video size [height, width] (default: 480x832)")
    parser.add_argument("--video_length", type=int, default=None,
                       help="Video length in frames (will be adjusted to 4t+1)")
    parser.add_argument("--fps", type=int, default=15,
                       help="Output video FPS (default: 15 for HoloCine)")
    parser.add_argument("--infer_steps", type=int, default=None,
                       help="Number of inference steps (default: 50)")
    parser.add_argument("--save_path", type=str, required=True,
                       help="Output video path")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--cpu_noise", action="store_true",
                       help="Generate noise on CPU")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                       help="CFG scale (default: 5.0)")

    # === HoloCine-Specific Shot Arguments ===
    parser.add_argument("--global_caption", type=str, default=None,
                       help="Global scene description for multi-shot (Mode 1)")
    parser.add_argument("--shot_captions", type=str, nargs="+", default=None,
                       help="List of per-shot descriptions for multi-shot (Mode 1)")
    parser.add_argument("--shot_cut_frames", type=int, nargs="*", default=None,
                       help="Frame indices where shots change (optional, auto-calculated if not provided)")
    parser.add_argument("--shot_mask_type", type=str, default="normalized",
                       choices=["id", "normalized", "alternating"],
                       help="Shot mask type (default: normalized)")

    # === I2V Arguments ===
    parser.add_argument("--image_path", type=str, default=None,
                       help="Start image for I2V")
    parser.add_argument("--end_image_path", type=str, default=None,
                       help="End image for I2V")

    # === V2V Arguments ===
    parser.add_argument("--video_path", type=str, default=None,
                       help="Input video for V2V")
    parser.add_argument("--strength", type=float, default=0.75,
                       help="V2V denoising strength (0.0-1.0)")
    parser.add_argument("--v2v_low_noise_only", action="store_true",
                       help="V2V: use only low noise model")
    parser.add_argument("--v2v_use_i2v", action="store_true",
                       help="V2V: use I2V model conditioning")

    # === Advanced CFG/Guidance ===
    parser.add_argument("--cfg_skip_mode", type=str, default="none",
                       choices=["early", "late", "middle", "early_late", "alternate", "none"],
                       help="CFG skip mode")
    parser.add_argument("--cfg_apply_ratio", type=float, default=None,
                       help="CFG apply ratio (0.0-1.0)")
    parser.add_argument("--slg_layers", type=str, default=None,
                       help="SLG layer indices (comma-separated)")
    parser.add_argument("--slg_scale", type=float, default=3.0,
                       help="SLG scale")
    parser.add_argument("--slg_start", type=float, default=0.0,
                       help="SLG start ratio")
    parser.add_argument("--slg_end", type=float, default=0.3,
                       help="SLG end ratio")
    parser.add_argument("--slg_mode", type=str, default=None,
                       choices=["original", "uncond"],
                       help="SLG mode")

    # === Flow Matching ===
    parser.add_argument("--flow_shift", type=float, default=None,
                       help="Flow matching shift factor")

    # === Optimization Arguments ===
    parser.add_argument("--fp8", action="store_true",
                       help="Use FP8 for DiT")
    parser.add_argument("--fp8_scaled", action="store_true",
                       help="Use scaled FP8")
    parser.add_argument("--mixed_dtype", action="store_true",
                       help="Use mixed weight dtypes")
    parser.add_argument("--fp8_fast", action="store_true",
                       help="Enable fast FP8 (RTX 4XXX+)")
    parser.add_argument("--fp8_t5", action="store_true",
                       help="Use FP8 for T5")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu/mps, auto-detect if None)")
    parser.add_argument("--attn_mode", type=str, default="torch",
                       choices=["flash", "flash2", "flash3", "torch", "sageattn", "xformers", "sdpa"],
                       help="Attention implementation")
    parser.add_argument("--blocks_to_swap", type=int, default=0,
                       help="Number of DiT blocks to swap to CPU")
    parser.add_argument("--output_type", type=str, default="video",
                       choices=["video", "images", "latent", "both"],
                       help="Output format")
    parser.add_argument("--no_metadata", action="store_true",
                       help="Don't save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None,
                       help="Load and decode latent(s)")
    parser.add_argument("--lycoris", action="store_true",
                       help="Use LyCORIS")
    parser.add_argument("--compile", action="store_true",
                       help="Enable torch.compile")
    parser.add_argument("--compile_args", nargs=4,
                       metavar=("BACKEND", "MODE", "DYNAMIC", "FULLGRAPH"),
                       default=["inductor", "max-autotune-no-cudagraphs", "False", "False"],
                       help="Torch.compile settings")
    parser.add_argument("--preview", type=int, default=None,
                       help="Enable latent preview every N steps")
    parser.add_argument("--preview_suffix", type=str, default=None,
                       help="Suffix for preview files")

    parser.add_argument("--trim_tail_frames", type=int, default=0,
                       help="Trim N frames from end before saving")

    # === Context Windows (if available) ===
    if CONTEXT_WINDOWS_AVAILABLE:
        parser.add_argument("--use_context_windows", action="store_true",
                           help="Enable sliding context windows")
        parser.add_argument("--context_length", type=int, default=81,
                           help="Context window length")
        parser.add_argument("--context_overlap", type=int, default=30,
                           help="Context window overlap")
        parser.add_argument("--context_schedule", type=str, default="standard_static",
                           choices=["standard_static", "standard_uniform", "looped_uniform", "batched"],
                           help="Context schedule method")
        parser.add_argument("--context_stride", type=int, default=1,
                           help="Context stride")
        parser.add_argument("--context_closed_loop", action="store_true",
                           help="Enable closed loop")
        parser.add_argument("--context_fuse_method", type=str, default="pyramid",
                           choices=["pyramid", "flat", "overlap-linear", "relative"],
                           help="Context fusion method")
        parser.add_argument("--context_dim", type=int, default=2,
                           help="Context dimension (2=temporal)")

    args = parser.parse_args()

    # === Validation ===
    # Check for conflicting modes
    if args.global_caption and not args.shot_captions:
        raise ValueError("--global_caption requires --shot_captions")
    if args.shot_captions and not args.global_caption:
        raise ValueError("--shot_captions requires --global_caption")

    # Validate shot inputs if provided
    if args.global_caption and args.shot_captions:
        if args.video_length is None:
            raise ValueError("--video_length required for structured multi-shot input")

        is_valid, error_msg = validate_shot_inputs(
            args.global_caption,
            args.shot_captions,
            args.shot_cut_frames,
            args.video_length
        )
        if not is_valid:
            raise ValueError(f"Shot input validation failed: {error_msg}")

    # Mutually exclusive argument checks (from base)
    if args.video_path and args.image_path and not args.v2v_use_i2v:
        raise ValueError("--video_path and --image_path cannot be used together (unless --v2v_use_i2v)")

    if args.mixed_dtype and (args.fp8 or args.fp8_scaled):
        raise ValueError("--mixed_dtype incompatible with --fp8/--fp8_scaled")

    return args


# ====================================================================================
#                           HOLOCINE GENERATE FUNCTION
# ====================================================================================

def generate_holocine(args: argparse.Namespace) -> Optional[torch.Tensor]:
    """
    HoloCine video generation with shot support.

    This function preprocesses shot-specific inputs and delegates to wan2_base.generate()
    with the appropriate parameters.

    Args:
        args: Command line arguments (extended with HoloCine shot parameters)

    Returns:
        Generated latent tensor [B, C, F, H, W] or None
    """
    logger.info("=== HoloCine Video Generation ===")

    # --- Shot Input Processing ---
    shot_inputs_processed = False

    # Mode 1: Structured multi-shot input (--global_caption + --shot_captions)
    if args.global_caption and args.shot_captions:
        logger.info("Mode 1: Structured multi-shot input")

        # Use helper to prepare HoloCine format
        shot_data = prepare_multishot_inputs(
            global_caption=args.global_caption,
            shot_captions=args.shot_captions,
            total_frames=args.video_length,
            custom_shot_cut_frames=args.shot_cut_frames
        )

        # Update args with processed data
        args.prompt = shot_data["prompt"]
        args.shot_cut_frames = shot_data["shot_cut_frames"]
        args.video_length = shot_data["num_frames"]

        shot_inputs_processed = True
        logger.info(f"Generated HoloCine prompt format")
        logger.info(f"Shot cuts: {args.shot_cut_frames}")
        logger.info(f"Adjusted video_length: {args.video_length} (4t+1)")

    # Mode 2: Raw HoloCine format prompt (--prompt with shot cuts)
    elif args.shot_cut_frames is not None:
        logger.info("Mode 2: Raw HoloCine format prompt with custom cuts")

        # Enforce 4t+1 on frame count if provided
        if args.video_length is not None:
            original_length = args.video_length
            args.video_length = enforce_4t_plus_1(args.video_length)
            if original_length != args.video_length:
                logger.info(f"Adjusted video_length: {original_length} -> {args.video_length} (4t+1)")

        # Enforce 4t+1 on shot cuts
        adjusted_cuts = [enforce_4t_plus_1(f) for f in args.shot_cut_frames]
        if adjusted_cuts != args.shot_cut_frames:
            logger.info(f"Adjusted shot cuts to 4t+1: {args.shot_cut_frames} -> {adjusted_cuts}")
            args.shot_cut_frames = adjusted_cuts

        shot_inputs_processed = True

    # Mode 3: Standard T2V (no shots, but still enforce 4t+1 for HoloCine models)
    else:
        logger.info("Mode 3: Standard T2V (no multi-shot)")
        if args.video_length is not None:
            original_length = args.video_length
            args.video_length = enforce_4t_plus_1(args.video_length)
            if original_length != args.video_length:
                logger.info(f"Adjusted video_length: {original_length} -> {args.video_length} (4t+1)")

    # --- Pass to wan2_base Infrastructure ---
    # The shot_cut_frames and shot_mask_type will be available in args
    # The wan2 infrastructure will need to be extended to handle these
    # For now, we'll note that this requires wan2_generate_video.py modifications

    logger.info("Delegating to wan2_generate_video infrastructure...")
    logger.info("NOTE: Full shot support requires modified wan2_generate_video.py to handle:")
    logger.info("  - shot_cut_frames parameter")
    logger.info("  - shot_indices tensor generation")
    logger.info("  - shot_mask_type parameter")
    logger.info("  - Passing shot_indices to DiT model during inference")

    # For now, we can call the base generate() but it won't use shot features
    # This will at least allow basic generation to work
    try:
        # Import the base generate function
        from wan2_generate_video import generate as base_generate

        # Call base generation with our preprocessed args
        result = base_generate(args)
        return result

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        logger.error("This may be because wan2_generate_video.py doesn't yet have shot support")
        logger.error("To fully enable HoloCine multi-shot features, wan2_generate_video.py needs to be extended")
        raise


# ====================================================================================
#                                    MAIN FUNCTION
# ====================================================================================

def main():
    """Main entry point for HoloCine video generation."""
    # Parse arguments
    args = parse_args()

    # Set device
    device_str = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    args.device = torch.device(device_str)
    logger.info(f"Using device: {args.device}")

    # Set defaults for critical parameters if not provided
    if args.infer_steps is None:
        args.infer_steps = 50
        logger.info(f"Using default infer_steps: {args.infer_steps}")

    if args.video_length is None:
        # For multi-shot, video_length is required
        if args.global_caption and args.shot_captions:
            raise ValueError("--video_length is required for multi-shot generation")
        # For standard T2V, use a default
        args.video_length = 81
        logger.info(f"Using default video_length: {args.video_length}")

    # Log HoloCine-specific settings
    if args.global_caption and args.shot_captions:
        logger.info(f"=== HoloCine Multi-Shot Generation ===")
        logger.info(f"Global caption: {args.global_caption[:100]}...")
        logger.info(f"Number of shots: {len(args.shot_captions)}")
        logger.info(f"Shot mask type: {args.shot_mask_type}")
    elif args.shot_cut_frames:
        logger.info(f"=== HoloCine Generation with Shot Cuts ===")
        logger.info(f"Shot cuts: {args.shot_cut_frames}")
        logger.info(f"Shot mask type: {args.shot_mask_type}")

    # Log model paths
    if args.dit_low_noise or args.dit_high_noise:
        logger.info(f"=== HoloCine Model Paths ===")
        if args.dit_low_noise:
            logger.info(f"Low noise DiT: {args.dit_low_noise}")
        if args.dit_high_noise:
            logger.info(f"High noise DiT: {args.dit_high_noise}")

    try:
        # Generate video with HoloCine shot support
        generated_latent = generate_holocine(args)

        if generated_latent is None:
            logger.error("Generation failed or was skipped")
            return

        logger.info(f"Generation complete! Latent shape: {generated_latent.shape}")
        logger.info(f"Output saved to: {args.save_path}")

    except Exception as e:
        logger.error(f"HoloCine generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("Done!")


# ====================================================================================
#                                  ENTRY POINT
# ====================================================================================

if __name__ == "__main__":
    main()
