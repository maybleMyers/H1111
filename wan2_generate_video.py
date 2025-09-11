import argparse
from datetime import datetime
import gc
import random
import os
import re
import time
import math
import types
from typing import Tuple, Optional, List, Union, Any
from pathlib import Path # Added for glob_images in V2V

# Set PyTorch CUDA allocator to reduce memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.distributed as dist
import accelerate
from accelerate import Accelerator
from functools import partial
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from PIL import Image
import cv2 # Added for V2V video loading/resizing
import numpy as np # Added for V2V video processing
import torchvision.transforms.functional as TF
import torchvision
from tqdm import tqdm

from networks import lora_wan
from utils.safetensors_utils import mem_eff_save_file, load_safetensors
from utils.lora_utils import filter_lora_state_dict
from Wan2_2.wan.configs import WAN_CONFIGS, SUPPORTED_SIZES
import wan
from wan.modules.model import WanModel, load_wan_model, detect_wan_sd_dtype
from wan.distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from wan.distributed.util import init_distributed_group, get_world_size, get_rank
from Wan2_2.wan.distributed.fsdp import shard_model, free_model
from torch.distributed.fsdp import ShardingStrategy
from wan.modules.vae import WanVAE
from Wan2_2.wan.modules.vae2_2 import Wan2_2_VAE
from wan.modules.t5 import T5EncoderModel
from wan.modules.clip import CLIPModel
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.pusa.flowmatch_pusa_scheduler import FlowMatchSchedulerPusa
from wan.pusa.flowmatch_pusa_v2v_scheduler import FlowMatchSchedulerPusaV2V
from wan.pusa.utils import (
    process_conditioning_images,
    process_conditioning_video,
    parse_conditioning_positions,
    parse_noise_multipliers,
    create_conditioning_dict,
    validate_conditioning_parameters,
    create_frame_noise_mapping,
    log_conditioning_info,
)

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
    logger = logging.getLogger(__name__)
    logger.info("Context windows module not available. Install required dependencies to enable context windows.")
# Local implementations to avoid xformers/flash-attention dependency
import av
import cv2
import glob
from PIL import Image
from einops import rearrange

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def glob_images(directory, base="*"):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # remove duplicates
    img_paths.sort()
    return img_paths

def resize_image_to_bucket(image, bucket_reso):
    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image

    bucket_width, bucket_height = bucket_reso
    if bucket_width == image_width or bucket_height == image_height:
        image = np.array(image) if is_pil_image else image
    else:
        # resize the image to the bucket resolution to match the short side
        scale_width = bucket_width / image_width
        scale_height = bucket_height / image_height
        scale = max(scale_width, scale_height)
        image_width = int(image_width * scale + 0.5)
        image_height = int(image_height * scale + 0.5)

        if scale > 1:
            image = Image.fromarray(image) if not is_pil_image else image
            image = image.resize((image_width, image_height), Image.LANCZOS)
            image = np.array(image)
        else:
            image = np.array(image) if is_pil_image else image
            image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)

    # crop the image to the bucket resolution
    crop_left = (image_width - bucket_width) // 2
    crop_top = (image_height - bucket_height) // 2
    image = image[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]
    return image

def hv_load_images(image_dir, video_length, bucket_reso):
    image_files = glob_images(image_dir)
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {image_dir}")
    if len(image_files) < video_length:
        raise ValueError(f"Number of images in {image_dir} is less than {video_length}")

    image_files.sort()
    images = []
    for image_file in image_files[:video_length]:
        image = Image.open(image_file)
        image = resize_image_to_bucket(image, bucket_reso)  # returns a numpy array
        images.append(image)

    return images

def hv_load_video(video_path, start_frame, end_frame, bucket_reso):
    container = av.open(video_path)
    video = []
    for i, frame in enumerate(container.decode(video=0)):
        if start_frame is not None and i < start_frame:
            continue
        if end_frame is not None and i >= end_frame:
            break
        frame = frame.to_image()

        if bucket_reso is not None:
            frame = resize_image_to_bucket(frame, bucket_reso)
        else:
            frame = np.array(frame)

        video.append(frame)
    container.close()
    return video

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    from einops import rearrange  # Local import to avoid scope issues
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    height, width, _ = outputs[0].shape

    # create output container
    container = av.open(path, mode="w")

    # create video stream
    codec = "libx264"
    pixel_format = "yuv420p"
    stream = container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = pixel_format
    stream.bit_rate = 4000000  # 4Mbit/s

    for frame_array in outputs:
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        packets = stream.encode(frame)
        for packet in packets:
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()

def save_images_grid(videos: torch.Tensor, parent_dir: str, image_name: str, rescale: bool = False, n_rows: int = 1, save_individually=True):
    from einops import rearrange  # Local import to avoid scope issues
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    if save_individually:
        output_dir = os.path.join(parent_dir, image_name)
    else:
        output_dir = parent_dir

    os.makedirs(output_dir, exist_ok=True)
    for i, x in enumerate(outputs):
        image_path = os.path.join(output_dir, f"{image_name}_{i:03d}.png")
        image = Image.fromarray(x)
        image.save(image_path)

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="Wan 2.2 inference script with new model architecture support")

    # Distributed launch arguments (added by torch.distributed.launch/torchrun)
    # Note: --local-rank gets converted to args.local_rank by argparse (replaces hyphen with underscore)
    parser.add_argument("--local-rank", type=int, default=0, dest='local_rank', 
                       help="Local rank for distributed training (auto-set by torchrun)")
    parser.add_argument("--local_rank", type=int, default=0, 
                       help="Local rank for distributed training (alternative format)")
    
    # WAN arguments
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory (Wan 2.1 official).")
    parser.add_argument("--task", type=str, default="t2v-A14B", choices=list(WAN_CONFIGS.keys()), help="The task to run.")
    parser.add_argument(
        "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++", "vanilla", "pusa"], help="The solver used to sample."
    )

    parser.add_argument("--dit", type=str, default=None, help="DiT checkpoint path")
    parser.add_argument("--dit_low_noise", type=str, default=None, help="DiT low noise checkpoint path (for dual-dit models)")
    parser.add_argument("--dit_high_noise", type=str, default=None, help="DiT high noise checkpoint path (for dual-dit models)")
    parser.add_argument("--dual_dit_boundary", type=float, default=None, help="Override boundary for dual-dit models (0.0-1.0). Low noise model used above after threshold. Default: 0.875 for t2v-A14B, 0.900 for i2v-A14B")
    parser.add_argument("--vae", type=str, default=None, help="VAE checkpoint path")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default is bfloat16")
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
    parser.add_argument("--t5", type=str, default=None, help="text encoder (T5) checkpoint path")
    parser.add_argument("--clip", type=str, default=None, help="text encoder (CLIP) checkpoint path")
    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")
    # LoRA for high noise model (dual-dit models only)
    parser.add_argument("--lora_weight_high", type=str, nargs="*", required=False, default=None, 
                       help="LoRA weight path for high noise model (dual-dit models only)")
    parser.add_argument("--lora_multiplier_high", type=float, nargs="*", default=1.0, 
                       help="LoRA multiplier for high noise model")
    parser.add_argument("--include_patterns_high", type=str, nargs="*", default=None, help="LoRA module include patterns for high noise model")
    parser.add_argument("--exclude_patterns_high", type=str, nargs="*", default=None, help="LoRA module exclude patterns for high noise model")
    parser.add_argument(
        "--save_merged_model",
        type=str,
        default=None,
        help="Save merged model to path. If specified, no inference will be performed.",
    )

    # inference
    parser.add_argument("--prompt", type=str, required=True, help="prompt for generation")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="negative prompt for generation, use default negative prompt if not specified",
    )
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256], help="video size, height and width")
    parser.add_argument("--video_length", type=int, default=None, help="video length, Default depends on task")
    parser.add_argument("--fps", type=int, default=16, help="video fps, Default is 16")
    parser.add_argument("--infer_steps", type=int, default=None, help="number of inference steps")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument(
        "--cpu_noise", action="store_true", help="Use CPU to generate noise (compatible with trash). Default is False."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for classifier free guidance. Default is 5.0.",
    )
    # V2V arguments
    parser.add_argument("--video_path", type=str, default=None, help="path to video for video2video inference (standard Wan V2V)")
    parser.add_argument("--strength", type=float, default=0.75, help="Strength for video2video inference (0.0-1.0)")
    parser.add_argument("--v2v_low_noise_only", action="store_true", help="For V2V with dual-dit models, use only the low noise model")
    parser.add_argument(
        "--v2v_use_i2v", action="store_true", 
        help="Use i2v model for V2V (extracts first frame for CLIP conditioning). Recommended for i2v-A14B."
    )
    # I2V arguments
    parser.add_argument("--image_path", type=str, default=None, help="path to image for image2video inference")
    parser.add_argument("--end_image_path", type=str, default=None, help="path to end image for image2video inference")
    # Fun-Control arguments (NEW/MODIFIED)
    parser.add_argument(
        "--control_path", # Keep this argument name
        type=str,
        default=None,
        help="path to control video for inference with Fun-Control model. video file or directory with images",
    )
    parser.add_argument(
        "--control_start",
        type=float,
        default=0.0,
        help="Start point (0.0-1.0) in the timeline where control influence is full (after fade-in)",
    )
    parser.add_argument(
        "--control_end",
        type=float,
        default=1.0,
        help="End point (0.0-1.0) in the timeline where control influence starts to fade out",
    )
    parser.add_argument(
        "--control_falloff_percentage", # NEW name
        type=float,
        default=0.3,
        help="Falloff percentage (0.0-0.49) for smooth transitions at start/end of control influence region",
    )
    parser.add_argument(
        "--control_weight", # NEW name
        type=float,
        default=1.0,
        help="Overall weight/strength of control video influence for Fun-Control (0.0 to high values)",
    )
    parser.add_argument("--trim_tail_frames", type=int, default=0, help="trim tail N frames from the video before saving")
    parser.add_argument(
        "--cfg_skip_mode",
        type=str,
        default="none",
        choices=["early", "late", "middle", "early_late", "alternate", "none"],
        help="CFG skip mode. each mode skips different parts of the CFG. "
        " early: initial steps, late: later steps, middle: middle steps, early_late: both early and late, alternate: alternate, none: no skip (default)",
    )
    parser.add_argument(
        "--cfg_apply_ratio",
        type=float,
        default=None,
        help="The ratio of steps to apply CFG (0.0 to 1.0). Default is None (apply all steps).",
    )
    parser.add_argument(
        "--slg_layers", type=str, default=None, help="Skip block (layer) indices for SLG (Skip Layer Guidance), comma separated"
    )
    parser.add_argument(
        "--slg_scale",
        type=float,
        default=3.0,
        help="scale for SLG classifier free guidance. Default is 3.0. Ignored if slg_mode is None or uncond",
    )
    parser.add_argument("--slg_start", type=float, default=0.0, help="start ratio for inference steps for SLG. Default is 0.0.")
    parser.add_argument("--slg_end", type=float, default=0.3, help="end ratio for inference steps for SLG. Default is 0.3.")
    parser.add_argument(
        "--slg_mode",
        type=str,
        default=None,
        choices=["original", "uncond"],
        help="SLG mode. original: same as SD3, uncond: replace uncond pred with SLG pred",
    )

    # Flow Matching
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=None,
        help="Shift factor for flow matching schedulers. Default depends on task.",
    )
    
    # Pusa Extension arguments
    parser.add_argument(
        "--pusa_noise_multipliers",
        type=float,
        default=0.0,
        help="Noise multipliers for Pusa extension (0.0-100.0). Only used with --sample_solver pusa.",
    )
    parser.add_argument(
        "--pusa_noisy_steps",
        type=int,
        default=-1,
        help="Number of steps to apply Pusa noise (-1 for all steps). Only used with --sample_solver pusa.",
    )
    
    # Pusa Multi-Frame Conditioning Arguments  
    parser.add_argument(
        "--cond_images",
        type=str,
        nargs='+',
        help="Paths to conditioning images for multi-frame Pusa (e.g., start/end frames). Use with --cond_positions and --cond_noise_multipliers.",
    )
    parser.add_argument(
        "--cond_video", 
        type=str,
        help="Path to conditioning video for video-to-video Pusa generation. Use with --cond_positions and --cond_noise_multipliers.",
    )
    parser.add_argument(
        "--cond_positions",
        type=str,
        help="Comma-separated frame positions for conditioning (e.g., '0,20,40'). Must match length of --cond_images or correspond to --cond_video frames.",
    )
    parser.add_argument(
        "--cond_noise_multipliers",
        type=str,
        help="Comma-separated noise multipliers for conditioning frames (e.g., '0.0,0.4,0.6'). 0.0 = clean conditioning, higher values = more noise.",
    )
    parser.add_argument(
        "--end_image",
        type=str,
        help="Path to end frame image for start-end interpolation. Use with --input_image as start frame.",
    )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8")
    parser.add_argument("--mixed_dtype", action="store_true", help="use model with mixed weight dtypes (preserves original dtypes, e.g. mixed fp16/fp32)")
    parser.add_argument("--fp8_fast", action="store_true", help="Enable fast FP8 arithmetic (RTX 4XXX+), only for fp8_scaled")
    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "flash2", "flash3", "torch", "sageattn", "xformers", "sdpa"],
        help="attention mode",
    )
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="number of blocks to swap in the model")
    parser.add_argument(
        "--output_type", type=str, default="video", choices=["video", "images", "latent", "both"], help="output type"
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")
    parser.add_argument("--lycoris", action="store_true", help="use lycoris for inference")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument(
        "--compile_args",
        nargs=4,
        metavar=("BACKEND", "MODE", "DYNAMIC", "FULLGRAPH"),
        default=["inductor", "max-autotune-no-cudagraphs", "False", "False"],
        help="Torch.compile settings",
    )
    parser.add_argument("--preview", type=int, default=None, metavar="N",
        help="Enable latent preview every N steps. Generates previews in 'previews' subdirectory.",
    )
    parser.add_argument("--preview_suffix", type=str, default=None,
        help="Unique suffix for preview files to avoid conflicts in concurrent runs.",
    )

    # Sequence Parallelism arguments
    parser.add_argument("--use_sequence_parallel", action="store_true", 
        help="Enable sequence parallelism across multiple GPUs for faster inference")
    parser.add_argument("--use_dual_gpu", action="store_true", 
        help="[DEPRECATED] Use --use_sequence_parallel instead")
    parser.add_argument("--debug_sequence_parallel", action="store_true",
        help="Enable debugging output for sequence parallelism")
    
    # FSDP (Fully Sharded Data Parallel) arguments
    parser.add_argument("--dit_fsdp", action="store_true",
        help="Enable FSDP sharding for DiT models to reduce VRAM per GPU")
    parser.add_argument("--t5_fsdp", action="store_true", 
        help="Enable FSDP sharding for T5 text encoder to reduce VRAM")
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="FULL_SHARD",
        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"],
        help="FSDP sharding strategy (default: FULL_SHARD for maximum memory savings)")
    parser.add_argument("--fsdp_mixed_precision", type=str, default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Mixed precision type for FSDP (default: bf16)")
    parser.add_argument("--fsdp_cpu_offload", action="store_true",
        help="Enable CPU offloading for FSDP (uses ~28GB RAM per GPU for BF16/FP16, ~56GB for FP32). Not recommended with blocks_to_swap.")

    # Video extension arguments (multitalk-style)
    parser.add_argument("--extend_video", type=str, default=None, help="Path to video to extend using multitalk-style iterative generation")
    parser.add_argument("--extend_frames", type=int, default=200, help="Total number of frames to generate when extending video")
    parser.add_argument("--frames_to_check", type=int, default=30, help="Number of frames from the end to analyze for best transition point (clean i2v-based extension)")
    parser.add_argument("--motion_frames", type=int, default=25, help="Number of frames to use for motion conditioning in each chunk")
    # Model selection for extension
    parser.add_argument("--force_low_noise", action="store_true", help="Force use of low noise model for video extension")
    parser.add_argument("--force_high_noise", action="store_true", help="Force use of high noise model for video extension")
    parser.add_argument("--extension_dual_dit_boundary", type=float, default=None, help="Custom dual-dit boundary for video extension (0.0-1.0). Overrides force_low_noise/force_high_noise")
    # Latent injection timing controls
    parser.add_argument("--inject_motion_timesteps", type=str, default="all", choices=["all", "high_only", "low_only", "none"], 
                       help="When to inject motion frames: 'all'=every timestep, 'high_only'=high noise timesteps only, 'low_only'=low noise timesteps only, 'none'=no injection")
    parser.add_argument("--injection_strength", type=float, default=1.0, help="Strength of motion frame injection (0.0-1.0, 1.0=full replacement)")
    parser.add_argument("--motion_noise_ratio", type=float, default=0.3, 
                       help="Noise ratio for motion frames in extension (0.0-1.0, lower=less noise/more preservation)")
    parser.add_argument("--color_match", type=str, default="hm", 
                       choices=["disabled", "hm", "mkl", "reinhard", "mvgd", "hm-mvgd-hm", "hm-mkl-hm"],
                       help="Color matching method for video extension (default: histogram matching)")
    
    # Context Windows Arguments
    parser.add_argument("--use_context_windows", action="store_true", 
                       help="Enable sliding context windows for long video generation")
    parser.add_argument("--context_length", type=int, default=81, 
                       help="Length of context window in frames (default: 81)")
    parser.add_argument("--context_overlap", type=int, default=30, 
                       help="Overlap between context windows in frames (default: 30)")
    parser.add_argument("--context_schedule", type=str, default="standard_static",
                       choices=["standard_static", "standard_uniform", "looped_uniform", "batched"],
                       help="Context window scheduling method (default: standard_static)")
    parser.add_argument("--context_stride", type=int, default=1,
                       help="Stride for uniform context schedules (default: 1)")
    parser.add_argument("--context_closed_loop", action="store_true",
                       help="Enable closed loop for cyclic videos")
    parser.add_argument("--context_fuse_method", type=str, default="pyramid",
                       choices=["pyramid", "flat", "overlap-linear", "relative"],
                       help="Method for fusing context window results (default: pyramid)")
    parser.add_argument("--context_dim", type=int, default=2,
                       help="Dimension to apply context windows (2=temporal for video, default: 2)")

    args = parser.parse_args()

    assert (args.latent_path is None or len(args.latent_path) == 0) or (
        args.output_type == "images" or args.output_type == "video"
    ), "latent_path is only supported for images or video output"

    # Add checks for mutually exclusive arguments
    if args.video_path is not None and args.image_path is not None and not args.v2v_use_i2v:
        raise ValueError("--video_path and --image_path cannot be used together unless --v2v_use_i2v is specified.")
    if args.v2v_use_i2v and args.video_path is None:
        raise ValueError("--v2v_use_i2v requires --video_path to be specified.")
    if args.v2v_use_i2v and "i2v" not in args.task:
        logger.warning("--v2v_use_i2v is recommended for i2v models. Current task: %s", args.task)
    if args.video_path is not None and args.control_path is not None:
        raise ValueError("--video_path (standard V2V) and --control_path (Fun-Control) cannot be used together.")
    if args.image_path is not None and "t2v" in args.task:
         logger.warning("--image_path is provided, but task is set to t2v. Task type does not directly affect I2V mode.")
    if args.control_path is not None and not WAN_CONFIGS[args.task].is_fun_control:
        raise ValueError("--control_path is provided, but the selected task does not support Fun-Control.")
    if not (0.0 <= args.control_falloff_percentage <= 0.49):
        raise ValueError("--control_falloff_percentage must be between 0.0 and 0.49")
    if args.mixed_dtype and args.fp8:
        raise ValueError("--mixed_dtype and --fp8 cannot be used together")
    if args.mixed_dtype and args.fp8_scaled:
        raise ValueError("--mixed_dtype and --fp8_scaled cannot be used together")
    if args.mixed_dtype and args.lora_weight:
        logger.warning("--mixed_dtype with LoRA: LoRA weights will be merged at the model's original precision")
    if args.task == "i2v-14B-FC-1.1" and args.image_path is None:
         logger.warning(f"Task '{args.task}' typically uses --image_path as the reference image for ref_conv. Proceeding without it.")    
    return args

class DynamicModelManager:
    """Manages dynamic loading and unloading of models during inference."""
    
    def __init__(self, config, device, dit_dtype, dit_weight_dtype, args):
        self.config = config
        self.device = device
        self.dit_dtype = dit_dtype
        self.dit_weight_dtype = dit_weight_dtype
        self.args = args
        self.current_model = None
        self.current_model_type = None  # 'low' or 'high'
        self.model_paths = {}
        self.lora_weights_list_low = None
        self.lora_multipliers_low = None
        self.lora_weights_list_high = None
        self.lora_multipliers_high = None
        
    def has_model_loaded(self):
        """Check if any model is currently loaded."""
        return self.current_model is not None
        
    def set_model_paths(self, low_path: str, high_path: str):
        """Set the paths for low and high noise models."""
        self.model_paths['low'] = low_path
        self.model_paths['high'] = high_path
        
    def set_lora_weights(self, lora_weights_list_low, lora_multipliers_low, 
                        lora_weights_list_high, lora_multipliers_high):
        """Save LoRA weights to apply to dynamically loaded models."""
        self.lora_weights_list_low = lora_weights_list_low
        self.lora_multipliers_low = lora_multipliers_low
        self.lora_weights_list_high = lora_weights_list_high
        self.lora_multipliers_high = lora_multipliers_high
        
    def get_model(self, model_type: str) -> WanModel:
        """Load the requested model if not already loaded."""
        if self.current_model_type == model_type:
            return self.current_model
            
        # Unload current model if exists
        if self.current_model is not None:
            logger.info(f"Unloading {self.current_model_type} noise model (GPU + CPU blocks)...")
            
            # Get memory usage before unloading (GPU + CPU monitoring)
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated(self.device) / 1024**3
                logger.info(f"GPU memory before unload: {memory_before:.2f} GB")
            
            # Monitor CPU memory for debugging model deletion
            import psutil
            cpu_memory_before = psutil.Process().memory_info().rss / 1024**3
            logger.info(f"CPU memory before model unload: {cpu_memory_before:.2f} GB")
            
            # Handle block swapping cleanup if enabled
            if hasattr(self.current_model, 'blocks_to_swap') and self.current_model.blocks_to_swap is not None:
                if self.current_model.blocks_to_swap > 0:
                    logger.info(f"Cleaning up block swapping for {self.current_model_type} model...")
                    
                    # First, ensure all blocks are moved back from swap
                    if hasattr(self.current_model, 'offloader') and self.current_model.offloader is not None:
                        # Wait for any pending operations on all blocks
                        for idx in range(len(self.current_model.blocks)):
                            try:
                                self.current_model.offloader.wait_for_block(idx)
                            except Exception as e:
                                logger.warning(f"Error waiting for block {idx}: {e}")
                        
                        # Move all blocks back to CPU to free GPU memory
                        for idx in range(len(self.current_model.blocks)):
                            try:
                                # Move block to CPU if it's not already there
                                self.current_model.blocks[idx] = self.current_model.blocks[idx].cpu()
                            except Exception as e:
                                logger.warning(f"Error moving block {idx} to CPU: {e}")
                        
                        # Clean up the offloader properly - FIX BACKWARD HOOK CLOSURE LEAK
                        try:
                            # 1. Clear backward hook handles (prevents closure reference leak)
                            if hasattr(self.current_model.offloader, 'remove_handles'):
                                for handle in self.current_model.offloader.remove_handles:
                                    handle.remove()
                                self.current_model.offloader.remove_handles.clear()
                            
                            # 2. Shutdown ThreadPoolExecutor and clear futures to prevent memory leaks
                            if hasattr(self.current_model.offloader, 'thread_pool'):
                                self.current_model.offloader.thread_pool.shutdown(wait=True)
                            if hasattr(self.current_model.offloader, 'futures'):
                                self.current_model.offloader.futures.clear()
                            del self.current_model.offloader
                            self.current_model.offloader = None
                        except Exception as e:
                            logger.warning(f"Error cleaning up offloader: {e}")
            
            # Enhanced block reference clearing before deletion
            try:
                # 1. Clear torch.compile cache if model was compiled
                if self.args.compile:
                    logger.info("Clearing torch.compile cache for model deletion")
                    torch._dynamo.reset()
                
                # 2. Clear individual block references (prevents compiled block retention)
                if hasattr(self.current_model, 'blocks') and self.current_model.blocks is not None:
                    for i in range(len(self.current_model.blocks)):
                        self.current_model.blocks[i] = None
                    self.current_model.blocks.clear()
                    self.current_model.blocks = None
                
                # 3. Move any remaining parameters to CPU
                self.current_model = self.current_model.cpu()
            except Exception as e:
                logger.warning(f"Error during enhanced cleanup: {e}")
            
            # 4. Force model deletion with verification
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            
            # Aggressive cleanup for both GPU and CPU memory
            torch.cuda.empty_cache()  # Clear GPU cache
            torch.cuda.synchronize()  # Wait for all GPU operations to complete
            gc.collect()              # Force Python garbage collection (clears CPU memory)
            torch.cuda.empty_cache()  # Second GPU cache clear
            clean_memory_on_device(self.device)
            
            # Additional cleanup for fragmented memory
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
            
            # Log memory usage after unloading (GPU + CPU verification)
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated(self.device) / 1024**3
                logger.info(f"GPU memory after unload: {memory_after:.2f} GB (freed: {memory_before - memory_after:.2f} GB)")
            
            # Verify CPU memory cleanup
            cpu_memory_after = psutil.Process().memory_info().rss / 1024**3
            cpu_freed = cpu_memory_before - cpu_memory_after
            logger.info(f"CPU memory after model unload: {cpu_memory_after:.2f} GB (freed: {cpu_freed:.2f} GB)")
            
        # Load new model
        logger.info(f"Loading {model_type} noise model...")
        loading_device = "cpu"
        if self.args.blocks_to_swap == 0 and self.lora_weights_list_low is None and not self.args.fp8_scaled:
            loading_device = self.device
            
        loading_weight_dtype = self.dit_weight_dtype
        if self.args.mixed_dtype:
            # For mixed dtype, load weights as-is without conversion
            loading_weight_dtype = None
        elif self.args.fp8_scaled or self.args.lora_weight is not None:
            loading_weight_dtype = self.dit_dtype
            
        # Select appropriate LoRA weights for this model type
        lora_weights_list = None
        lora_multipliers = None
        if model_type == 'low':
            lora_weights_list = self.lora_weights_list_low
            lora_multipliers = self.lora_multipliers_low
        else:  # 'high'
            lora_weights_list = self.lora_weights_list_high
            lora_multipliers = self.lora_multipliers_high
            
        # DEBUG: Print full LoRA list and weights being applied to this DiT model
        if lora_weights_list is not None:
            logger.info(f"DEBUG: Loading {model_type} noise DiT model with {len(lora_weights_list)} LoRA(s)")
            for i, lora_sd in enumerate(lora_weights_list):
                multiplier = lora_multipliers[i] if lora_multipliers and i < len(lora_multipliers) else 1.0
                lora_keys = list(lora_sd.keys())[:5]  # Show first 5 keys
                logger.info(f"DEBUG: LoRA {i+1}/{len(lora_weights_list)} for {model_type} noise model - Multiplier: {multiplier}, Keys sample: {lora_keys}")
        else:
            logger.info(f"DEBUG: Loading {model_type} noise DiT model with NO LoRA weights")
            
        # Load model with LoRA weights if available
        model = load_wan_model(
            self.config, self.device, self.model_paths[model_type], 
            self.args.attn_mode, False, loading_device, loading_weight_dtype, False,
            lora_weights_list=lora_weights_list, lora_multipliers=lora_multipliers
        )
        
        # Optimize model
        optimize_model(model, self.args, self.device, self.dit_dtype, self.dit_weight_dtype)
        
        self.current_model = model
        self.current_model_type = model_type
        return model
        
            
    def cleanup(self):
        """Clean up any loaded models."""
        if self.current_model is not None:
            logger.info(f"Final cleanup of {self.current_model_type} noise model...")
            
            # Handle block swapping cleanup
            if hasattr(self.current_model, 'blocks_to_swap') and self.current_model.blocks_to_swap is not None:
                if self.current_model.blocks_to_swap > 0 and hasattr(self.current_model, 'offloader'):
                    if self.current_model.offloader is not None:
                        # Wait for all blocks using the correct method
                        for idx in range(len(self.current_model.blocks)):
                            try:
                                self.current_model.offloader.wait_for_block(idx)
                            except Exception as e:
                                logger.warning(f"Error waiting for block {idx}: {e}")
                        
                        # Move all blocks to CPU
                        for idx in range(len(self.current_model.blocks)):
                            try:
                                self.current_model.blocks[idx] = self.current_model.blocks[idx].cpu()
                            except Exception as e:
                                logger.warning(f"Error moving block {idx} to CPU: {e}")
                        
                        try:
                            # 1. Clear backward hook handles (prevents closure reference leak)
                            if hasattr(self.current_model.offloader, 'remove_handles'):
                                for handle in self.current_model.offloader.remove_handles:
                                    handle.remove()
                                self.current_model.offloader.remove_handles.clear()
                            
                            # 2. Shutdown ThreadPoolExecutor and clear futures to prevent memory leaks
                            if hasattr(self.current_model.offloader, 'thread_pool'):
                                self.current_model.offloader.thread_pool.shutdown(wait=True)
                            if hasattr(self.current_model.offloader, 'futures'):
                                self.current_model.offloader.futures.clear()
                            del self.current_model.offloader
                            self.current_model.offloader = None
                        except Exception as e:
                            logger.warning(f"Error cleaning up offloader: {e}")
            
            # Enhanced block reference clearing before final deletion
            try:
                # 1. Clear torch.compile cache if model was compiled
                if self.args.compile:
                    logger.info("Final cleanup: Clearing torch.compile cache")
                    torch._dynamo.reset()
                
                # 2. Clear individual block references (prevents compiled block retention)
                if hasattr(self.current_model, 'blocks') and self.current_model.blocks is not None:
                    for i in range(len(self.current_model.blocks)):
                        self.current_model.blocks[i] = None
                    self.current_model.blocks.clear()
                    self.current_model.blocks = None
                
                # 3. Move model to CPU before deletion
                self.current_model = self.current_model.cpu()
            except Exception as e:
                logger.warning(f"Error during final enhanced cleanup: {e}")
            
            # 4. Force final model deletion
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            
            # Aggressive cleanup for both GPU and CPU
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            clean_memory_on_device(self.device)
            
    def unload_all(self):
        """Alias for cleanup method to ensure compatibility."""
        self.cleanup()

class PipelineDynamicModelManager(DynamicModelManager):
    """Extended model manager that supports pipeline parallelism across two GPUs."""
    
    def __init__(self, config, device, dit_dtype, dit_weight_dtype, args):
        super().__init__(config, device, dit_dtype, dit_weight_dtype, args)
        
        # Setup dual GPU devices
        self.gpu0 = torch.device(f"cuda:{args.pipeline_gpu0}")
        self.gpu1 = torch.device(f"cuda:{args.pipeline_gpu1}")
        self.split_ratio = args.gpu_split_ratio
        
        # Pipeline components (will be set after model loading)
        self.blocks_gpu0 = None
        self.blocks_gpu1 = None
        self.split_point = None
        
        # CUDA streams for async transfers
        self.stream_gpu0 = torch.cuda.Stream(device=self.gpu0)
        self.stream_gpu1 = torch.cuda.Stream(device=self.gpu1)
        
        logger.info(f"Pipeline parallelism enabled: GPU0={self.gpu0}, GPU1={self.gpu1}, split_ratio={self.split_ratio}")
        
    def _split_model_blocks(self):
        """Split the current model's transformer blocks across two GPUs."""
        if self.current_model is None:
            return
            
        num_blocks = len(self.current_model.blocks)
        self.split_point = int(num_blocks * self.split_ratio)
        
        # Ensure at least one block on each GPU
        if self.split_point == 0:
            self.split_point = 1
        elif self.split_point >= num_blocks:
            self.split_point = num_blocks - 1
            
        logger.info(f"Splitting {num_blocks} blocks: {self.split_point} on GPU0, {num_blocks - self.split_point} on GPU1")
        
        # Move embeddings and time projections to GPU0
        self.current_model.patch_embedding = self.current_model.patch_embedding.to(self.gpu0)
        self.current_model.time_embedding = self.current_model.time_embedding.to(self.gpu0)
        self.current_model.time_projection = self.current_model.time_projection.to(self.gpu0)
        if hasattr(self.current_model, 'text_embedding'):
            self.current_model.text_embedding = self.current_model.text_embedding.to(self.gpu0)
        
        # Move CLIP projection if present (for I2V models)
        if hasattr(self.current_model, 'clip_projection'):
            self.current_model.clip_projection = self.current_model.clip_projection.to(self.gpu0)
        if hasattr(self.current_model, 'clip_norm'):
            self.current_model.clip_norm = self.current_model.clip_norm.to(self.gpu0)
            
        # Split blocks between GPUs
        for i in range(self.split_point):
            self.current_model.blocks[i] = self.current_model.blocks[i].to(self.gpu0)
        for i in range(self.split_point, num_blocks):
            self.current_model.blocks[i] = self.current_model.blocks[i].to(self.gpu1)
            
        # Move output head to GPU1 (unpatchify is a method, not a module)
        self.current_model.head = self.current_model.head.to(self.gpu1)
        
        # Move ref_conv if present (for Fun-Control models)
        if hasattr(self.current_model, 'ref_conv') and self.current_model.ref_conv is not None:
            self.current_model.ref_conv = self.current_model.ref_conv.to(self.gpu1)
        
        # Store references for easier access
        self.blocks_gpu0 = self.current_model.blocks[:self.split_point]
        self.blocks_gpu1 = self.current_model.blocks[self.split_point:]
        
        # Clear cache on both GPUs
        torch.cuda.empty_cache()
        
    def get_model(self, model_type: str) -> WanModel:
        """Load the requested model and split it across GPUs."""
        # Check if we need to switch models
        if self.current_model_type == model_type:
            return self.current_model
            
        # Unload current model if exists (with proper dual GPU cleanup)
        if self.current_model is not None:
            logger.info(f"Unloading {self.current_model_type} noise model from dual GPUs...")
            
            # Get memory usage before unloading
            if torch.cuda.is_available():
                memory_before_gpu0 = torch.cuda.memory_allocated(self.gpu0) / 1024**3
                memory_before_gpu1 = torch.cuda.memory_allocated(self.gpu1) / 1024**3
                logger.info(f"GPU0 memory before unload: {memory_before_gpu0:.2f} GB")
                logger.info(f"GPU1 memory before unload: {memory_before_gpu1:.2f} GB")
            
            # Use our custom cleanup that handles both GPUs
            self.cleanup()
            
            # Log memory after cleanup
            if torch.cuda.is_available():
                memory_after_gpu0 = torch.cuda.memory_allocated(self.gpu0) / 1024**3
                memory_after_gpu1 = torch.cuda.memory_allocated(self.gpu1) / 1024**3
                logger.info(f"GPU0 memory after unload: {memory_after_gpu0:.2f} GB (freed: {memory_before_gpu0 - memory_after_gpu0:.2f} GB)")
                logger.info(f"GPU1 memory after unload: {memory_after_gpu1:.2f} GB (freed: {memory_before_gpu1 - memory_after_gpu1:.2f} GB)")
            
        # Now load the new model using parent's loading logic (but not its unloading)
        logger.info(f"Loading {model_type} noise model...")
        loading_device = "cpu"
        if self.args.blocks_to_swap == 0 and self.lora_weights_list_low is None and not self.args.fp8_scaled:
            loading_device = self.device
            
        loading_weight_dtype = self.dit_weight_dtype
        if self.args.mixed_dtype:
            loading_weight_dtype = None
        elif self.args.fp8_scaled or self.args.lora_weight is not None:
            loading_weight_dtype = self.dit_dtype
            
        # Select appropriate LoRA weights
        lora_weights_list = None
        lora_multipliers = None
        if model_type == 'low':
            lora_weights_list = self.lora_weights_list_low
            lora_multipliers = self.lora_multipliers_low
        else:  # 'high'
            lora_weights_list = self.lora_weights_list_high
            lora_multipliers = self.lora_multipliers_high
            
        # Load model with LoRA weights if available
        model = load_wan_model(
            self.config, self.device, self.model_paths[model_type], 
            self.args.attn_mode, False, loading_device, loading_weight_dtype, False,
            lora_weights_list=lora_weights_list, lora_multipliers=lora_multipliers
        )
        
        # Optimize model
        optimize_model(model, self.args, self.device, self.dit_dtype, self.dit_weight_dtype)
        
        self.current_model = model
        self.current_model_type = model_type
        
        # Split the loaded model across GPUs
        self._split_model_blocks()
        
        return model
        
    def pipeline_forward(self, x, t, context, seq_len, clip_fea=None, y=None, skip_block_indices=None, fun_ref=None):
        """
        Custom forward pass that handles pipeline parallelism.
        This replaces the standard model.forward() call when using dual GPUs.
        """
        model = self.current_model
        
        # Handle list input format (x is a list of tensors for WAN model)
        if isinstance(x, list):
            x = [tensor.to(self.gpu0) for tensor in x]
        else:
            x = x.to(self.gpu0)
        
        # === Stage 1: Embeddings and preprocessing on GPU0 ===
        with torch.cuda.stream(self.stream_gpu0):
            # Move other inputs to GPU0
            t = t.to(self.gpu0)
            if isinstance(context, list):
                context = [c.to(self.gpu0) for c in context]
            else:
                context = context.to(self.gpu0)
            if clip_fea is not None:
                clip_fea = clip_fea.to(self.gpu0)
            if y is not None:
                if isinstance(y, list):
                    y = [tensor.to(self.gpu0) for tensor in y]
                else:
                    y = y.to(self.gpu0)
            if fun_ref is not None:
                fun_ref = fun_ref.to(self.gpu0)
                
            # Patch embedding
            if model.model_type == 'i2v':
                assert y is not None
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            
            x = [model.patch_embedding(u.unsqueeze(0)) for u in x]
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            
            # Padding to seq_len
            x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ])
            
            # Time embeddings - match the original model's processing
            with torch.amp.autocast('cuda', dtype=torch.float32):
                from wan.modules.model import sinusoidal_embedding_1d
                e = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, t).float())
                e0 = model.time_projection(e).unflatten(1, (6, model.dim))
            
            # Text embeddings
            context = model.text_embedding(
                torch.stack([
                    torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))
            
            # Process CLIP features if present
            if clip_fea is not None:
                clip_fea = model.clip_projection(clip_fea).view(-1, seq_len, model.dim)
                clip_fea = model.clip_norm(clip_fea)
                x = x + clip_fea
                
            # Setup kwargs for blocks
            kwargs = dict(
                e=e0, 
                seq_lens=seq_lens.to(self.gpu0), 
                grid_sizes=grid_sizes.to(self.gpu0), 
                freqs=model.freqs.to(self.gpu0), 
                context=context, 
                context_lens=None
            )
            
            # Process through first half of blocks on GPU0
            for block_idx, block in enumerate(self.blocks_gpu0):
                is_block_skipped = skip_block_indices is not None and block_idx in skip_block_indices
                if not is_block_skipped:
                    x = block(x, **kwargs)
        
        # === Transfer to GPU1 ===
        torch.cuda.synchronize(self.gpu0)  # Ensure GPU0 work is complete
        
        with torch.cuda.stream(self.stream_gpu1):
            # Move intermediate tensors to GPU1
            x = x.to(self.gpu1)
            e = e.to(self.gpu1)
            e0 = e0.to(self.gpu1)
            context = context.to(self.gpu1)
            seq_lens = seq_lens.to(self.gpu1)
            grid_sizes = grid_sizes.to(self.gpu1)
            
            # Update kwargs for GPU1
            kwargs['e'] = e0
            kwargs['seq_lens'] = seq_lens
            kwargs['grid_sizes'] = grid_sizes
            kwargs['freqs'] = model.freqs.to(self.gpu1)
            kwargs['context'] = context
            
            # Process through second half of blocks on GPU1
            for block_idx, block in enumerate(self.blocks_gpu1):
                actual_idx = self.split_point + block_idx
                is_block_skipped = skip_block_indices is not None and actual_idx in skip_block_indices
                if not is_block_skipped:
                    x = block(x, **kwargs)
            
            # Apply ref_conv if present (Fun-Control)
            if model.ref_conv is not None and fun_ref is not None:
                fun_ref = fun_ref.to(self.gpu1)
                ref_emb = model.ref_conv(fun_ref.to(model.ref_conv.weight.dtype))
                ref_emb = ref_emb.reshape(x.shape[0], x.shape[1], -1)
                x = x + ref_emb
            
            # Final head processing
            x = model.head(x, e)
            
            # Unpatchify
            x = model.unpatchify(x, grid_sizes)
        
        # Synchronize before returning
        torch.cuda.synchronize(self.gpu1)
        
        return [u.float() for u in x]
        
    def cleanup(self):
        """Clean up models on both GPUs."""
        if self.current_model is not None:
            logger.info(f"Cleaning up pipeline model from both GPUs...")
            
            # Move all model components to CPU before deletion
            # Move embeddings from GPU0 to CPU
            if hasattr(self.current_model, 'patch_embedding'):
                self.current_model.patch_embedding = self.current_model.patch_embedding.cpu()
            if hasattr(self.current_model, 'time_embedding'):
                self.current_model.time_embedding = self.current_model.time_embedding.cpu()
            if hasattr(self.current_model, 'time_projection'):
                self.current_model.time_projection = self.current_model.time_projection.cpu()
            if hasattr(self.current_model, 'text_embedding'):
                self.current_model.text_embedding = self.current_model.text_embedding.cpu()
            if hasattr(self.current_model, 'clip_projection'):
                self.current_model.clip_projection = self.current_model.clip_projection.cpu()
            if hasattr(self.current_model, 'clip_norm'):
                self.current_model.clip_norm = self.current_model.clip_norm.cpu()
                
            # Move blocks from both GPUs to CPU
            if self.blocks_gpu0 is not None:
                for i, block in enumerate(self.blocks_gpu0):
                    self.current_model.blocks[i] = self.current_model.blocks[i].cpu()
                    
            if self.blocks_gpu1 is not None:
                for i, block in enumerate(self.blocks_gpu1):
                    idx = self.split_point + i
                    self.current_model.blocks[idx] = self.current_model.blocks[idx].cpu()
                    
            # Move head from GPU1 to CPU
            if hasattr(self.current_model, 'head'):
                self.current_model.head = self.current_model.head.cpu()
            if hasattr(self.current_model, 'ref_conv') and self.current_model.ref_conv is not None:
                self.current_model.ref_conv = self.current_model.ref_conv.cpu()
                
            # Clear references
            self.blocks_gpu0 = None
            self.blocks_gpu1 = None
            
            # Clear cache on both GPUs
            with torch.cuda.device(self.gpu0):
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self.gpu0)
            with torch.cuda.device(self.gpu1):
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self.gpu1)
                
            # Delete the model and reset state
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Final cache clear
            with torch.cuda.device(self.gpu0):
                torch.cuda.empty_cache()
            with torch.cuda.device(self.gpu1):
                torch.cuda.empty_cache()

class SequenceParallelModelManager(DynamicModelManager):
    """Extended model manager that supports sequence parallelism across multiple GPUs using distributed PyTorch."""
    
    def __init__(self, config, device, dit_dtype, dit_weight_dtype, args):
        super().__init__(config, device, dit_dtype, dit_weight_dtype, args)
        
        # Initialize distributed group if not already done
        if not dist.is_initialized():
            # Check if we should avoid libuv (Windows compatibility)
            import os
            if os.environ.get('USE_LIBUV') == '0':
                # On Windows, use gloo backend instead of nccl
                import platform
                if platform.system() == 'Windows':
                    dist.init_process_group(backend='gloo')
                else:
                    init_distributed_group()
            else:
                init_distributed_group()
            
        self.world_size = get_world_size()
        self.rank = get_rank()
        
        # Set device based on rank
        self.device = torch.device(f"cuda:{self.rank}")
        
        # Enable debugging if requested
        self.debug = getattr(args, 'debug_sequence_parallel', False)
        
        logger.info(f"Sequence parallelism enabled: rank={self.rank}, world_size={self.world_size}, device={self.device}")
        
    def _apply_sequence_parallel(self, model):
        """Apply sequence parallel modifications to the model."""
        # Replace attention forward methods with sequence parallel version
        for block in model.blocks:
            block.self_attn.forward = types.MethodType(sp_attn_forward, block.self_attn)
        
        # Replace model forward with sequence parallel version
        model.forward = types.MethodType(sp_dit_forward, model)
        
        logger.info(f"Applied sequence parallelism to model on rank {self.rank}")
        
        if self.debug:
            # Add hooks to monitor tensor values
            self._add_debug_hooks(model)
        
    def get_model(self, model_type: str) -> WanModel:
        """Load the requested model and apply sequence parallelism."""
        # Check if we need to switch models
        if self.current_model_type == model_type:
            return self.current_model
            
        # Unload current model if exists
        if self.current_model is not None:
            logger.info(f"Rank {self.rank}: Unloading {self.current_model_type} noise model...")
            
            # Get memory usage before unloading
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated(self.device) / 1024**3
                logger.info(f"Rank {self.rank}: Memory before unload: {memory_before:.2f} GB")
            
            # Use parent's cleanup
            self.cleanup()
            
            # Log memory after cleanup
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated(self.device) / 1024**3
                logger.info(f"Rank {self.rank}: Memory after unload: {memory_after:.2f} GB (freed: {memory_before - memory_after:.2f} GB)")
            
        # Now load the new model using parent's loading logic
        logger.info(f"Rank {self.rank}: Loading {model_type} noise model...")
        loading_device = "cpu"
        if self.args.blocks_to_swap == 0 and self.lora_weights_list_low is None and not self.args.fp8_scaled:
            loading_device = self.device
            
        loading_weight_dtype = self.dit_weight_dtype
        if self.args.mixed_dtype:
            loading_weight_dtype = None
        elif self.args.fp8_scaled or self.args.lora_weight is not None:
            loading_weight_dtype = self.dit_dtype
            
        # Select appropriate LoRA weights
        lora_weights_list = None
        lora_multipliers = None
        if model_type == 'low':
            lora_weights_list = self.lora_weights_list_low
            lora_multipliers = self.lora_multipliers_low
        else:  # 'high'
            lora_weights_list = self.lora_weights_list_high
            lora_multipliers = self.lora_multipliers_high
            
        # Load model with LoRA weights if available
        model = load_wan_model(
            self.config, self.device, self.model_paths[model_type], 
            self.args.attn_mode, False, loading_device, loading_weight_dtype, False,
            lora_weights_list=lora_weights_list, lora_multipliers=lora_multipliers
        )
        
        # Optimize model
        optimize_model(model, self.args, self.device, self.dit_dtype, self.dit_weight_dtype)
        
        # Apply sequence parallelism
        self._apply_sequence_parallel(model)
        
        # Move model to correct device
        model = model.to(self.device)
        
        self.current_model = model
        self.current_model_type = model_type
        
        # Synchronize across all ranks
        if dist.is_initialized():
            dist.barrier()
        
        return model
    
    def cleanup(self):
        """Clean up resources with distributed synchronization."""
        if dist.is_initialized():
            dist.barrier()  # Ensure all ranks cleanup together
        super().cleanup()
    
    def _add_debug_hooks(self, model):
        """Add debug hooks to monitor tensor values during forward pass."""
        def check_tensor(name, tensor):
            if tensor is None:
                return
            if torch.isnan(tensor).any():
                logger.error(f"Rank {self.rank}: {name} contains NaN values!")
            if torch.isinf(tensor).any():
                logger.error(f"Rank {self.rank}: {name} contains Inf values!")
            logger.debug(f"Rank {self.rank}: {name} - shape={tensor.shape}, min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")
        
        def forward_hook(module, input, output, name):
            if isinstance(output, tuple):
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        check_tensor(f"{name}_output_{i}", out)
            elif isinstance(output, torch.Tensor):
                check_tensor(f"{name}_output", output)
        
        # Add hooks to key modules
        model.patch_embedding.register_forward_hook(lambda m, i, o: forward_hook(m, i, o, "patch_embedding"))
        model.time_embedding.register_forward_hook(lambda m, i, o: forward_hook(m, i, o, "time_embedding"))
        model.head.register_forward_hook(lambda m, i, o: forward_hook(m, i, o, "head"))
        
        # Add hooks to first and last transformer blocks
        if len(model.blocks) > 0:
            model.blocks[0].register_forward_hook(lambda m, i, o: forward_hook(m, i, o, "block_0"))
            model.blocks[-1].register_forward_hook(lambda m, i, o: forward_hook(m, i, o, f"block_{len(model.blocks)-1}"))
        
        logger.info(f"Rank {self.rank}: Debug hooks added to model")

class FSDPModelManager(DynamicModelManager):
    """Extended model manager that supports FSDP (Fully Sharded Data Parallel) to reduce VRAM per GPU."""
    
    def __init__(self, config, device, dit_dtype, dit_weight_dtype, args):
        # Initialize distributed group if not already done
        if not dist.is_initialized():
            # Check if we should avoid libuv (Windows compatibility)
            import os
            if os.environ.get('USE_LIBUV') == '0':
                # On Windows, use gloo backend instead of nccl
                import platform
                if platform.system() == 'Windows':
                    dist.init_process_group(backend='gloo')
                else:
                    init_distributed_group()
            else:
                init_distributed_group()
            
        self.world_size = get_world_size() if dist.is_initialized() else 1
        self.rank = get_rank() if dist.is_initialized() else 0
        
        # Set device based on LOCAL rank, not global rank
        # Use environment variable or args to get local rank
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        self.device = torch.device(f"cuda:{local_rank}")
        self.local_rank = local_rank
        
        # Store FSDP configuration
        self.dit_fsdp = args.dit_fsdp
        self.t5_fsdp = args.t5_fsdp
        self.fsdp_strategy = self._get_sharding_strategy(args.fsdp_sharding_strategy)
        self.fsdp_mixed_precision = args.fsdp_mixed_precision
        self.fsdp_cpu_offload = getattr(args, 'fsdp_cpu_offload', False)
        
        # Store memory optimization settings
        self.blocks_to_swap = getattr(args, 'blocks_to_swap', 0)
        self.fp8_scaled = getattr(args, 'fp8_scaled', False)
        
        # Call parent constructor with the rank-specific device
        super().__init__(config, self.device, dit_dtype, dit_weight_dtype, args)
        
        # Warn about memory usage with CPU offload
        if self.fsdp_cpu_offload:
            # Calculate expected RAM usage based on dtype
            param_bytes = 2  # Default to FP16/BF16
            if dit_weight_dtype == torch.float32:
                param_bytes = 4
            elif dit_weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                param_bytes = 1
            model_size_gb = 14 * param_bytes  # 14B parameters
            total_ram_gb = model_size_gb * self.world_size
            logger.warning(f"FSDP CPU offload enabled: will use ~{total_ram_gb}GB system RAM ({model_size_gb}GB per GPU)")
            
        # Warn if both optimizations are enabled
        if self.blocks_to_swap > 0 and self.fsdp_cpu_offload:
            logger.warning("Both blocks_to_swap and fsdp_cpu_offload are enabled. This will use excessive RAM.")
            logger.warning("Recommend using only one: either blocks_to_swap OR fsdp_cpu_offload")
        
        logger.info(f"FSDP enabled: global_rank={self.rank}, local_rank={self.local_rank}, world_size={self.world_size}, device={self.device}")
        logger.info(f"FSDP config: dit_fsdp={self.dit_fsdp}, t5_fsdp={self.t5_fsdp}, strategy={args.fsdp_sharding_strategy}")
        logger.info(f"Memory optimizations: blocks_to_swap={self.blocks_to_swap}, fp8_scaled={self.fp8_scaled}, cpu_offload={self.fsdp_cpu_offload}")
        
    def _get_sharding_strategy(self, strategy_name):
        """Convert string strategy name to ShardingStrategy enum."""
        strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD if hasattr(ShardingStrategy, 'HYBRID_SHARD') else ShardingStrategy.FULL_SHARD,
        }
        return strategy_map.get(strategy_name, ShardingStrategy.FULL_SHARD)
    
    def _get_fsdp_mixed_precision_dtype(self):
        """Get the dtype for FSDP mixed precision."""
        if self.fsdp_mixed_precision == "bf16":
            return torch.bfloat16
        elif self.fsdp_mixed_precision == "fp16":
            return torch.float16
        else:
            return torch.float32
            
    def get_model(self, model_type: str) -> WanModel:
        """Load the requested model and apply FSDP sharding."""
        # Check if we need to switch models
        if self.current_model_type == model_type:
            return self.current_model
            
        # Unload current model if exists
        if self.current_model is not None:
            logger.info(f"Rank {self.rank}: Unloading {self.current_model_type} noise model...")
            
            # Get memory usage before unloading
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated(self.device) / 1024**3
                logger.info(f"Rank {self.rank}: Memory before unload: {memory_before:.2f} GB")
            
            # Free FSDP model if it was sharded
            if self.dit_fsdp and hasattr(self, 'current_model'):
                # With CPU offloading, we can't use free_model due to pinned memory
                # Instead, delete the model and clear cache
                if self.fsdp_cpu_offload:
                    del self.current_model
                    self.current_model = None
                    self.current_model_type = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                else:
                    # Without CPU offload, we can use the free_model function
                    free_model(self.current_model)
            else:
                # Use parent's cleanup
                self.cleanup()
            
            # Log memory after cleanup
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated(self.device) / 1024**3
                logger.info(f"Rank {self.rank}: Memory after unload: {memory_after:.2f} GB (freed: {memory_before - memory_after:.2f} GB)")
            
        # Now load the new model
        logger.info(f"Rank {self.rank}: Loading {model_type} noise model...")
        
        # Load on CPU first if using FSDP (for memory efficiency)
        loading_device = "cpu" if self.dit_fsdp else self.device
        
        # Adjust weight dtype - respect user settings for FSDP
        loading_weight_dtype = self.dit_weight_dtype
        if self.args.mixed_dtype:
            loading_weight_dtype = None
        elif self.args.fp8_scaled or self.args.lora_weight is not None:
            loading_weight_dtype = self.dit_dtype
        # Note: We now respect the original dtype for FSDP instead of forcing FP32
        # This significantly reduces RAM usage (e.g., BF16 uses 28GB instead of 56GB per GPU)
            
        # Select appropriate LoRA weights
        lora_weights_list = None
        lora_multipliers = None
        if model_type == 'low':
            lora_weights_list = self.lora_weights_list_low
            lora_multipliers = self.lora_multipliers_low
        else:  # 'high'
            lora_weights_list = self.lora_weights_list_high
            lora_multipliers = self.lora_multipliers_high
            
        # Load model with LoRA weights if available
        model = load_wan_model(
            self.config, self.device, self.model_paths[model_type], 
            self.args.attn_mode, False, loading_device, loading_weight_dtype, False,
            lora_weights_list=lora_weights_list, lora_multipliers=lora_multipliers
        )
        
        # Apply FP8 optimization if enabled (before FSDP)
        if self.fp8_scaled:
            logger.info(f"Rank {self.rank}: Applying FP8 optimization...")
            state_dict = model.state_dict()
            # Apply FP8 optimization on CPU to save memory
            state_dict = model.fp8_optimization(state_dict, torch.device('cpu'), move_to_device=False, use_scaled_mm=self.args.fp8_fast)
            info = model.load_state_dict(state_dict, strict=True, assign=True)
            logger.info(f"Rank {self.rank}: Loaded FP8 optimized weights: {info}")
        
        # Apply FSDP sharding if enabled
        if self.dit_fsdp:
            logger.info(f"Rank {self.rank}: Applying FSDP sharding to {model_type} model...")
            
            # Import necessary FSDP components
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision, CPUOffload
            from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
            from functools import partial
            
            # Get mixed precision dtype first
            param_dtype = self._get_fsdp_mixed_precision_dtype()
            
            # Setup CPU offload strategy based on settings
            cpu_offload = None
            if self.fsdp_cpu_offload:
                # Use CPU offloading only if explicitly enabled
                # Calculate actual RAM usage based on dtype
                param_bytes = 2 if param_dtype in [torch.float16, torch.bfloat16] else 4
                ram_per_gpu = 14 * param_bytes  # 14B parameters
                cpu_offload = CPUOffload(offload_params=True)
                logger.info(f"Rank {self.rank}: CPU offloading enabled - will use ~{ram_per_gpu}GB RAM per GPU")
            elif self.blocks_to_swap > 0:
                # For block swapping, use FSDP sharding without CPU offload
                logger.info(f"Rank {self.rank}: Using FSDP sharding without CPU offload for block swapping")
                # Don't enable CPU offload - let FSDP handle memory via sharding only
            
            # Apply FSDP with proper configuration
            model = FSDP(
                module=model,
                process_group=None,
                sharding_strategy=self.fsdp_strategy,
                auto_wrap_policy=partial(
                    lambda_auto_wrap_policy,
                    lambda_fn=lambda m: m in model.blocks
                ),
                mixed_precision=MixedPrecision(
                    param_dtype=param_dtype,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32
                ),
                device_id=self.rank,
                sync_module_states=True,
                cpu_offload=cpu_offload,  # Enable CPU offloading
                limit_all_gathers=True,  # Limit concurrent all-gathers to save memory
                use_orig_params=False  # Use flattened parameters for better memory efficiency
            )
            
            logger.info(f"Rank {self.rank}: FSDP sharding applied successfully")
        else:
            # Regular optimization without FSDP
            if not self.fp8_scaled:  # Only optimize if not already done
                optimize_model(model, self.args, self.device, self.dit_dtype, self.dit_weight_dtype)
            
            # Move to device if not using block swapping
            if self.blocks_to_swap == 0:
                model = model.to(self.device)
        
        self.current_model = model
        self.current_model_type = model_type
        
        # Synchronize across all ranks
        if dist.is_initialized():
            dist.barrier()
        
        return model
    
    def cleanup(self):
        """Clean up resources with FSDP-aware cleanup."""
        if self.current_model is not None and self.dit_fsdp:
            # With CPU offloading, we can't use free_model due to pinned memory
            if self.fsdp_cpu_offload:
                del self.current_model
                self.current_model = None
                self.current_model_type = None
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            else:
                # Without CPU offload, we can use the free_model function
                free_model(self.current_model)
        else:
            # Use parent's cleanup
            super().cleanup()
            
        # Synchronize cleanup across all ranks
        if dist.is_initialized():
            dist.barrier()
    
    def load_text_encoder(self, config) -> T5EncoderModel:
        """Load T5 text encoder with optional FSDP sharding."""
        logger.info(f"Rank {self.rank}: Loading T5 text encoder (FSDP={self.t5_fsdp})...")
        
        checkpoint_path = None if self.args.ckpt_dir is None else os.path.join(self.args.ckpt_dir, config.t5_checkpoint)
        tokenizer_path = None if self.args.ckpt_dir is None else os.path.join(self.args.ckpt_dir, config.t5_tokenizer)
        
        # Load on CPU first if using FSDP
        loading_device = torch.device('cpu') if self.t5_fsdp else self.device
        
        # Create text encoder
        text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=loading_device,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            shard_fn=None  # We'll apply FSDP manually below
        )
        
        # Apply FSDP sharding if enabled
        if self.t5_fsdp:
            logger.info(f"Rank {self.rank}: Applying FSDP sharding to T5 encoder...")
            
            # Apply sharding to T5 encoder
            from functools import partial
            shard_fn = partial(
                shard_model,
                device_id=self.rank,
                param_dtype=param_dtype,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
                process_group=None,
                sharding_strategy=self.fsdp_strategy,
                sync_module_states=True
            )
            
            # Apply sharding to the encoder model
            if hasattr(text_encoder, 'encoder') and text_encoder.encoder is not None:
                text_encoder.encoder = shard_fn(text_encoder.encoder)
                logger.info(f"Rank {self.rank}: T5 encoder FSDP sharding applied successfully")
        else:
            # Move to device without FSDP
            text_encoder = text_encoder.to(self.device)
        
        # Synchronize across all ranks
        if dist.is_initialized():
            dist.barrier()
        
        return text_encoder

def create_funcontrol_conditioning_latent(
    args: argparse.Namespace,
    config,
    vae: WanVAE,
    device: torch.device,
    lat_f: int,
    lat_h: int,
    lat_w: int,
    pixel_height: int, # Actual pixel height for resizing
    pixel_width: int   # Actual pixel width for resizing
) -> Optional[torch.Tensor]:
    """
    Creates the conditioning latent tensor 'y' for FunControl models,
    replicating the logic from WanWeightedControlToVideo node.

    Args:
        args: Command line arguments.
        config: Model configuration.
        vae: Loaded VAE model instance.
        device: Target computation device.
        lat_f: Number of latent frames.
        lat_h: Latent height.
        lat_w: Latent width.
        pixel_height: Target pixel height for image/video processing.
        pixel_width: Target pixel width for image/video processing.

    Returns:
        torch.Tensor: The final conditioning latent 'y' [1, 32, lat_f, lat_h, lat_w],
                      or None if VAE is missing when required.
    """
    logger.info("Creating FunControl conditioning latent 'y'...")
    if vae is None:
         # Should not happen if called correctly, but check anyway
         logger.error("VAE is required to create FunControl conditioning latent but was not provided.")
         return None

    batch_size = 1 # Hardcoded for script execution
    total_latent_frames = lat_f
    vae_dtype = vae.dtype # Use VAE's dtype for encoding

    # Initialize the two parts of the concat latent
    # Control part (first 16 channels) - will be filled later
    control_latent_part = torch.zeros([batch_size, 16, total_latent_frames, lat_h, lat_w],
                                    device=device, dtype=vae_dtype).contiguous()
    # Image guidance part (last 16 channels)
    image_guidance_latent = torch.zeros([batch_size, 16, total_latent_frames, lat_h, lat_w],
                                      device=device, dtype=vae_dtype).contiguous()

    # --- Image Guidance Processing (Start/End Images) ---
    timeline_mask = torch.zeros([1, 1, total_latent_frames], device=device, dtype=torch.float32).contiguous()
    has_start_image = args.image_path is not None
    has_end_image = args.end_image_path is not None

    # Process start image if provided
    start_latent = None
    if has_start_image:
        logger.info(f"Processing start image: {args.image_path}")
        try:
            img = Image.open(args.image_path).convert("RGB")
            img_np = np.array(img)
            # Resize to target pixel dimensions
            interpolation = cv2.INTER_AREA if pixel_height < img_np.shape[0] else cv2.INTER_CUBIC
            img_resized_np = cv2.resize(img_np, (pixel_width, pixel_height), interpolation=interpolation)
            # Convert to tensor CFHW, range [-1, 1]
            img_tensor = TF.to_tensor(img_resized_np).sub_(0.5).div_(0.5).to(device)
            img_tensor = img_tensor.unsqueeze(1) # Add frame dim: C,F,H,W

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vae_dtype):
                # vae.encode expects a list, returns a list. Take first element.
                # Result shape [C', F', H', W'] - needs batch dim for processing here
                start_latent = vae.encode([img_tensor])[0].unsqueeze(0).to(device).contiguous() # [1, 16, 1, lat_h, lat_w]

            # Calculate influence and falloff
            start_frames_influence = min(start_latent.shape[2], total_latent_frames) # Usually 1
            if start_frames_influence > 0:
                 # Use falloff_percentage for smooth transition *away* from start image
                 falloff_len_frames = max(1, int(total_latent_frames * args.control_falloff_percentage))
                 start_influence_mask = torch.ones([1, 1, total_latent_frames], device=device, dtype=torch.float32).contiguous()

                 # Apply falloff starting *after* the first frame
                 if total_latent_frames > 1 + falloff_len_frames:
                     # Falloff from frame 1 to 1+falloff_len_frames
                     t = torch.linspace(0, 1, falloff_len_frames, device=device)
                     falloff = 0.5 + 0.5 * torch.cos(t * math.pi) # 1 -> 0
                     start_influence_mask[0, 0, 1:1+falloff_len_frames] = falloff
                     # Set influence to 0 after falloff
                     start_influence_mask[0, 0, 1+falloff_len_frames:] = 0.0
                 elif total_latent_frames > 1:
                     # Shorter falloff if video is too short
                     t = torch.linspace(0, 1, total_latent_frames - 1, device=device)
                     falloff = 0.5 + 0.5 * torch.cos(t * math.pi) # 1 -> 0
                     start_influence_mask[0, 0, 1:] = falloff

                 # Place start latent in the image guidance part, weighted by mask
                 # Since start_latent is only frame 0, we just place it there.
                 # The mask influences how other elements (like end image) blend *in*.
                 image_guidance_latent[:, :, 0:1, :, :] = start_latent[:, :, 0:1, :, :] # Take first frame

                 # Update the main timeline mask
                 timeline_mask = torch.max(timeline_mask, start_influence_mask) # Start image dominates beginning
                 logger.info(f"Start image processed. Latent shape: {start_latent.shape}")

        except Exception as e:
            logger.error(f"Error processing start image: {e}")
            # Continue without start image guidance

    # Process end image if provided
    end_latent = None
    if has_end_image:
        logger.info(f"Processing end image: {args.end_image_path}")
        try:
            img = Image.open(args.end_image_path).convert("RGB")
            img_np = np.array(img)
            # Resize to target pixel dimensions
            interpolation = cv2.INTER_AREA if pixel_height < img_np.shape[0] else cv2.INTER_CUBIC
            img_resized_np = cv2.resize(img_np, (pixel_width, pixel_height), interpolation=interpolation)
            # Convert to tensor CFHW, range [-1, 1]
            img_tensor = TF.to_tensor(img_resized_np).sub_(0.5).div_(0.5).to(device)
            img_tensor = img_tensor.unsqueeze(1) # Add frame dim: C,F,H,W

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vae_dtype):
                # vae.encode expects a list, returns a list. Take first element.
                # Result shape [C', F', H', W'] - needs batch dim for processing here
                end_latent = vae.encode([img_tensor])[0].unsqueeze(0).to(device).contiguous() # [1, 16, 1, lat_h, lat_w]

            # Calculate end image influence transition (S-curve / cubic)
            end_influence_mask = torch.zeros([1, 1, total_latent_frames], device=device, dtype=torch.float32).contiguous()
            falloff_len_frames = max(1, int(total_latent_frames * args.control_falloff_percentage))

            # Determine when the end image influence should start ramping up
            # More sophisticated start point based on control_end if control video exists
            if args.control_path and args.control_end < 1.0:
                 # Start fade-in just before control video fades out significantly
                 influence_start_frame = max(0, int(total_latent_frames * args.control_end) - falloff_len_frames // 2)
            else:
                 # Default: start influence around 60% mark if no control or control runs full length
                 influence_start_frame = max(0, int(total_latent_frames * 0.6))

            # Ensure start frame isn't too close to the beginning if start image exists
            if has_start_image:
                influence_start_frame = max(influence_start_frame, 1 + falloff_len_frames) # Ensure it starts after start_img falloff

            transition_length = total_latent_frames - influence_start_frame
            if transition_length > 0:
                 logger.info(f"End image influence transition: frames {influence_start_frame} to {total_latent_frames-1}")
                 curve_positions = torch.linspace(0, 1, transition_length, device=device)
                 for i, pos in enumerate(curve_positions):
                     idx = influence_start_frame + i
                     if idx < total_latent_frames:
                         # Cubic ease-in-out curve (smoother than cosine)
                         if pos < 0.5: influence = 4 * pos * pos * pos
                         else: p = pos - 1; influence = 1 + 4 * p * p * p
                         # Ensure full influence near the end
                         if idx >= total_latent_frames - 3: influence = 1.0
                         end_influence_mask[0, 0, idx] = influence

                 # Blending logic (similar to base_nodes)
                 blend_start_frame = influence_start_frame
                 blend_length = total_latent_frames - blend_start_frame
                 if blend_length > 0:
                     # Create reference end latent (just the single frame repeated conceptually)
                     # Blend existing content with end latent based on influence weight
                     for i in range(blend_length):
                         idx = blend_start_frame + i
                         if idx < total_latent_frames:
                             weight = end_influence_mask[0, 0, idx].item()
                             if weight > 0:
                                 # Blend: (1-w)*current + w*end_latent
                                 image_guidance_latent[:, :, idx] = (
                                     (1.0 - weight) * image_guidance_latent[:, :, idx] +
                                     weight * end_latent[:, :, 0] # Use the single frame end_latent
                                 )

                 # Ensure final frames are exactly the end image latent
                 last_frames_exact = min(3, total_latent_frames) # Ensure at least last 3 frames are end image
                 if last_frames_exact > 0:
                     end_offset = total_latent_frames - last_frames_exact
                     if end_offset >= 0:
                         image_guidance_latent[:, :, end_offset:] = end_latent[:, :, 0:1].repeat(1, 1, last_frames_exact, 1, 1)

                 # Update the main timeline mask
                 timeline_mask = torch.max(timeline_mask, end_influence_mask)
                 logger.info(f"End image processed. Latent shape: {end_latent.shape}")

        except Exception as e:
            logger.error(f"Error processing end image: {e}")
            # Continue without end image guidance

    # --- Control Video Processing ---
    control_video_latent = None
    if args.control_path:
        logger.info(f"Processing control video: {args.control_path}")
        try:
            # Load control video frames (use helper from hv_generate_video for consistency)
            # Use args.video_length for the number of frames
            if os.path.isfile(args.control_path):
                video_frames_np = hv_load_video(args.control_path, 0, args.video_length, bucket_reso=(pixel_width, pixel_height))
            elif os.path.isdir(args.control_path):
                video_frames_np = hv_load_images(args.control_path, args.video_length, bucket_reso=(pixel_width, pixel_height))
            else:
                 raise FileNotFoundError(f"Control path not found: {args.control_path}")

            if not video_frames_np:
                raise ValueError("No frames loaded from control path.")

            num_control_frames_loaded = len(video_frames_np)
            if num_control_frames_loaded < args.video_length:
                 logger.warning(f"Control video loaded {num_control_frames_loaded} frames, less than target {args.video_length}. Padding with last frame.")
                 # Pad with the last frame
                 last_frame = video_frames_np[-1]
                 padding = [last_frame] * (args.video_length - num_control_frames_loaded)
                 video_frames_np.extend(padding)

            # Stack and convert to tensor: F, H, W, C -> B, C, F, H, W, range [-1, 1]
            video_frames_np = np.stack(video_frames_np[:args.video_length], axis=0) # Ensure correct length
            control_tensor = torch.from_numpy(video_frames_np).permute(0, 3, 1, 2).float() / 127.5 - 1.0 # F,C,H,W
            control_tensor = control_tensor.permute(1, 0, 2, 3) # C,F,H,W
            control_tensor = control_tensor.unsqueeze(0).to(device) # B,C,F,H,W

            # Encode control video
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vae_dtype):
                # vae.encode expects list of [C, F, H, W], returns list of [C', F', H', W']
                control_video_latent = vae.encode([control_tensor[0]])[0].unsqueeze(0).to(device).contiguous() # [1, 16, lat_f, lat_h, lat_w]

            # Calculate weighted control mask (replicating base_nodes logic)
            control_frames_latent = control_video_latent.shape[2] # Should match total_latent_frames
            control_mask = torch.zeros([1, 1, control_frames_latent], device=device, dtype=torch.float32).contiguous()

            start_frame_idx = max(0, min(control_frames_latent - 1, int(control_frames_latent * args.control_start)))
            end_frame_idx = max(start_frame_idx + 1, min(control_frames_latent, int(control_frames_latent * args.control_end)))
            falloff_len_frames = max(2, int(control_frames_latent * args.control_falloff_percentage))

            # Main active region
            if start_frame_idx < end_frame_idx:
                control_mask[:, :, start_frame_idx:end_frame_idx] = 1.0

            # Fall-on at the start
            if start_frame_idx > 0:
                fallon_start = max(0, start_frame_idx - falloff_len_frames)
                fallon_len = start_frame_idx - fallon_start
                if fallon_len > 0:
                    t = torch.linspace(0, 1, fallon_len, device=device)
                    smooth_t = 0.5 - 0.5 * torch.cos(t * math.pi) # 0 -> 1
                    control_mask[:, :, fallon_start:start_frame_idx] = smooth_t.reshape(1, 1, -1)

            # Fall-off at the end (interacting with end_image influence)
            if end_frame_idx < control_frames_latent:
                falloff_start = end_frame_idx
                falloff_end = min(control_frames_latent, falloff_start + falloff_len_frames)
                falloff_actual_len = falloff_end - falloff_start
                if falloff_actual_len > 0:
                    # Check for end image influence in this region
                    if has_end_image:
                        for i in range(falloff_start, falloff_end):
                            # Calculate original falloff (1 -> 0)
                            fade_pos = (i - falloff_start) / falloff_actual_len
                            original_falloff = 0.5 + 0.5 * math.cos(fade_pos * math.pi)
                            # Get end image influence (already calculated in timeline_mask)
                            end_influence_here = timeline_mask[0, 0, i].item()
                            # Adjust control falloff: decrease faster if end image is taking over
                            # Use a factor (e.g., 0.8) to control how much end image preempts control
                            adjusted_falloff = original_falloff * (1.0 - (end_influence_here * 0.8))
                            control_mask[0, 0, i] = adjusted_falloff
                        logger.info("Applied end-image interaction to control falloff.")
                    else:
                        # Standard falloff if no end image
                        t = torch.linspace(0, 1, falloff_actual_len, device=device)
                        smooth_t = 0.5 + 0.5 * torch.cos(t * math.pi) # 1 -> 0
                        control_mask[:, :, falloff_start:falloff_end] = smooth_t.reshape(1, 1, -1)

            # Apply final control weight
            control_mask = control_mask * args.control_weight

            # Expand mask and apply to control latent
            control_mask_expanded = control_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, 1, 1, F, 1, 1] ? -> needs [1, 1, F, 1, 1]
            control_mask_expanded = control_mask.unsqueeze(-1).unsqueeze(-1) # Shape: [1, 1, F, 1, 1]

            # Apply weighting to the control_video_latent
            weighted_control_latent = control_video_latent * control_mask_expanded # [1, 16, F, H, W]

            # Place into the first 16 channels of the final latent
            control_latent_part = weighted_control_latent

            # Log mask pattern
            mask_pattern = "".join(["#" if v > 0.8*args.control_weight else "+" if v > 0.4*args.control_weight else "." if v > 0.1*args.control_weight else " "
                                   for v in control_mask[0, 0, :].tolist()])
            logger.info(f"Control mask pattern (weight={args.control_weight:.2f}): |{mask_pattern}|")
            logger.info(f"Control video processed. Latent shape: {control_video_latent.shape}")

        except Exception as e:
            logger.error(f"Error processing control video: {e}")
            # Continue without control video guidance (control_latent_part remains zeros)

    # --- Final Assembly ---
    # Concatenate the control part and the image guidance part
    final_y = torch.cat([control_latent_part, image_guidance_latent], dim=1) # Concat along channel dim: [1, 16+16, F, H, W]
    final_y = final_y.contiguous()

    logger.info(f"FunControl conditioning latent 'y' created. Final shape: {final_y.shape}")

    # Optional: Clean up intermediate tensors explicitly if memory is tight
    del start_latent, end_latent, control_video_latent, control_latent_part, image_guidance_latent
    del timeline_mask, control_mask
    if 'control_tensor' in locals(): del control_tensor
    if 'img_tensor' in locals(): del img_tensor
    clean_memory_on_device(device) # Be cautious with frequent cache clearing

    return final_y

def get_task_defaults(task: str, size: Optional[Tuple[int, int]] = None) -> Tuple[int, float, int, bool]:
    """Return default values for each task

    Args:
        task: task name (t2v, t2i, i2v etc.)
        size: size of the video (width, height)

    Returns:
        Tuple[int, float, int, bool]: (infer_steps, flow_shift, video_length, needs_clip)
    """
    width, height = size if size else (0, 0)

    if "t2i" in task:
        return 50, 5.0, 1, False
    elif task == "i2v-A14B":
        # New Wan2.2 i2v-A14B model defaults
        return 40, 5.0, 81, True
    elif task == "t2v-A14B":
        # New Wan2.2 t2v-A14B model defaults
        return 40, 1.0, 81, False
    elif task == "ti2v-5B":
        # New Wan2.2 ti2v-5B model defaults (supports 121 frames at 24 FPS)
        return 50, 5.0, 121, False
    elif "i2v" in task:
        flow_shift = 3.0 if (width == 832 and height == 480) or (width == 480 and height == 832) else 5.0
        return 40, flow_shift, 81, True
    else:  # t2v or default
        return 50, 5.0, 81, False


def setup_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate and set default values for optional arguments

    Args:
        args: command line arguments

    Returns:
        argparse.Namespace: updated arguments
    """
    # Get default values for the task
    infer_steps, flow_shift, video_length, _ = get_task_defaults(args.task, tuple(args.video_size))

    # Apply default values to unset arguments
    if args.infer_steps is None:
        args.infer_steps = infer_steps
    if args.flow_shift is None:
        args.flow_shift = flow_shift
    # For V2V, video_length might be determined by the input video later if not set
    if args.video_length is None and args.video_path is None:
        args.video_length = video_length
    elif args.video_length is None and args.video_path is not None:
        # Delay setting default if V2V and length not specified
        pass
    elif args.video_length is not None:
        # Use specified length
        pass
    
    # Set FPS defaults for specific models
    if args.task == "ti2v-5B" and args.fps == 16:  # Only override if using default FPS
        args.fps = 24  # ti2v-5B model designed for 24 FPS

    # Force video_length to 1 for t2i tasks
    if "t2i" in args.task:
        assert args.video_length == 1, f"video_length should be 1 for task {args.task}"

    # parse slg_layers
    if args.slg_layers is not None:
        args.slg_layers = list(map(int, args.slg_layers.split(",")))
    
    # Validate dual_dit_boundary
    if args.dual_dit_boundary is not None:
        if not (0.0 <= args.dual_dit_boundary <= 1.0):
            raise ValueError(f"--dual_dit_boundary must be between 0.0 and 1.0, got {args.dual_dit_boundary}")
        # Only applicable for dual-dit models
        if "A14B" not in args.task:
            logger.warning(f"--dual_dit_boundary specified but task '{args.task}' is not a dual-dit model. This setting will be ignored.")
    
    # Validate video extension arguments
    if args.extend_video is not None:
        # Check for conflicting model selection options
        conflicting_options = sum([args.force_low_noise, args.force_high_noise, args.extension_dual_dit_boundary is not None])
        if conflicting_options > 1:
            raise ValueError("Only one of --force_low_noise, --force_high_noise, or --extension_dual_dit_boundary can be specified")
        
        # Validate extension_dual_dit_boundary
        if args.extension_dual_dit_boundary is not None:
            if not (0.0 <= args.extension_dual_dit_boundary <= 1.0):
                raise ValueError(f"--extension_dual_dit_boundary must be between 0.0 and 1.0, got {args.extension_dual_dit_boundary}")
            if "A14B" not in args.task:
                logger.warning(f"--extension_dual_dit_boundary specified but task '{args.task}' is not a dual-dit model. This setting will be ignored.")
        
        # Validate injection_strength
        if not (0.0 <= args.injection_strength <= 1.0):
            raise ValueError(f"--injection_strength must be between 0.0 and 1.0, got {args.injection_strength}")

    return args


def check_inputs(args: argparse.Namespace) -> Tuple[int, int, Optional[int]]:
    """Validate video size and potentially length (if not V2V auto-detect)

    Args:
        args: command line arguments

    Returns:
        Tuple[int, int, Optional[int]]: (height, width, video_length)
    """
    height = args.video_size[0]
    width = args.video_size[1]
    size = f"{width}*{height}"

    # Only check supported sizes if not doing V2V (V2V might use custom sizes from input)
    # Or if it's FunControl, which might have different size constraints
    if args.video_path is None and not WAN_CONFIGS[args.task].is_fun_control:
        if size not in SUPPORTED_SIZES[args.task]:
            logger.warning(f"Size {size} is not officially reccomended for task {args.task}. Supported sizes are {SUPPORTED_SIZES[args.task]}.")

    video_length = args.video_length # Might be None if V2V auto-detect

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    return height, width, video_length


def calculate_dimensions(video_size: Tuple[int, int], video_length: int, config, task: str = None) -> Tuple[Tuple[int, int, int, int], int]:
    """calculate dimensions for the generation

    Args:
        video_size: video frame size (height, width)
        video_length: number of frames in the video
        config: model configuration

    Returns:
        Tuple[Tuple[int, int, int, int], int]:
            ((channels, frames, height, width), seq_len)
    """
    height, width = video_size
    frames = video_length

    # calculate latent space dimensions
    lat_f = (frames - 1) // config.vae_stride[0] + 1
    lat_h = height // config.vae_stride[1]
    lat_w = width // config.vae_stride[2]

    # calculate sequence length
    seq_len = math.ceil((lat_h * lat_w) / (config.patch_size[1] * config.patch_size[2]) * lat_f)
    
    # Determine channels based on task
    channels = 48 if task == "ti2v-5B" else 16
    return ((channels, lat_f, lat_h, lat_w), seq_len)


# Modified function (replace the original)
def load_vae(args: argparse.Namespace, config, device: torch.device, dtype: torch.dtype):
    """load VAE model with robust path handling and automatic model selection

    Args:
        args: command line arguments
        config: model configuration
        device: device to use
        dtype: data type for the model

    Returns:
        WanVAE or Wan2_2_VAE: loaded VAE model
    """
    vae_override_path = args.vae
    # Assume models are in 'wan' dir relative to script if not otherwise specified
    vae_base_dir = "wan"

    final_vae_path = None

    # 1. Check if args.vae is a valid *existing file path*
    if vae_override_path and isinstance(vae_override_path, str) and \
       (vae_override_path.endswith(".pth") or vae_override_path.endswith(".safetensors")) and \
       os.path.isfile(vae_override_path):
        final_vae_path = vae_override_path
        logger.info(f"Using VAE override path from --vae: {final_vae_path}")

    # 2. If override is invalid or not provided, select VAE based on task type
    if final_vae_path is None:
        # Select correct VAE based on model type
        if args.task == "ti2v-5B":
            # 5B model uses the new Wan2.2_VAE.pth
            vae_filename = "Wan2.2_VAE.pth"
            logger.info(f"Detected ti2v-5B task, using new VAE: {vae_filename}")
        elif args.task in ["i2v-A14B", "t2v-A14B"]:
            # 14B models use the older Wan2.1_VAE.pth
            vae_filename = "Wan2.1_VAE.pth"
            logger.info(f"Detected 14B task ({args.task}), using older VAE: {vae_filename}")
        else:
            # Fallback to config default for other tasks
            vae_filename = config.vae_checkpoint
            logger.info(f"Using config default VAE for task {args.task}: {vae_filename}")

        constructed_path = os.path.join(vae_base_dir, vae_filename)
        if os.path.isfile(constructed_path):
            final_vae_path = constructed_path
            logger.info(f"Constructed VAE path: {final_vae_path}")
            if vae_override_path:
                 logger.warning(f"Ignoring potentially invalid --vae argument: {vae_override_path}")
        else:
             # 3. Fallback using ckpt_dir if provided and default construction failed
             if args.ckpt_dir:
                 fallback_path = os.path.join(args.ckpt_dir, vae_filename)
                 if os.path.isfile(fallback_path):
                     final_vae_path = fallback_path
                     logger.info(f"Using VAE path from --ckpt_dir fallback: {final_vae_path}")
                 else:
                     # If all attempts fail, raise error
                     raise FileNotFoundError(f"Cannot find VAE. Checked override '{vae_override_path}', constructed '{constructed_path}', and fallback '{fallback_path}'")
             else:
                 raise FileNotFoundError(f"Cannot find VAE. Checked override '{vae_override_path}' and constructed '{constructed_path}'. No --ckpt_dir provided for fallback.")

    # At this point, final_vae_path should be valid
    logger.info(f"Loading VAE model from final path: {final_vae_path}")
    cache_device = torch.device("cpu") if args.vae_cache_cpu else None
    
    # Use different VAE classes based on task type
    if args.task == "ti2v-5B":
        # Use the new Wan2_2_VAE for 5B model
        logger.info(f"Using Wan2_2_VAE for ti2v-5B model")
        vae = Wan2_2_VAE(vae_pth=final_vae_path, device=device, dtype=dtype)
    else:
        # Use the original WanVAE for 14B models
        logger.info(f"Using WanVAE for {args.task} model")
        vae = WanVAE(vae_path=final_vae_path, device=device, dtype=dtype, cache_device=cache_device)
    
    return vae


def load_text_encoder(args: argparse.Namespace, config, device: torch.device) -> T5EncoderModel:
    """load text encoder (T5) model

    Args:
        args: command line arguments
        config: model configuration
        device: device to use

    Returns:
        T5EncoderModel: loaded text encoder model
    """
    checkpoint_path = None if args.ckpt_dir is None else os.path.join(args.ckpt_dir, config.t5_checkpoint)
    tokenizer_path = None if args.ckpt_dir is None else os.path.join(args.ckpt_dir, config.t5_tokenizer)

    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=device,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        weight_path=args.t5,
        fp8=args.fp8_t5,
    )

    return text_encoder


def load_clip_model(args: argparse.Namespace, config, device: torch.device) -> CLIPModel:
    """load CLIP model (for I2V only)

    Args:
        args: command line arguments
        config: model configuration
        device: device to use

    Returns:
        CLIPModel: loaded CLIP model
    """
    checkpoint_path = None if args.ckpt_dir is None else os.path.join(args.ckpt_dir, config.clip_checkpoint)
    tokenizer_path = None if args.ckpt_dir is None else os.path.join(args.ckpt_dir, config.clip_tokenizer)

    clip = CLIPModel(
        dtype=config.clip_dtype,
        device=device,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        weight_path=args.clip,
    )

    return clip


def load_dit_model(
    args: argparse.Namespace,
    config,
    device: torch.device,
    dit_dtype: torch.dtype,
    dit_weight_dtype: Optional[torch.dtype] = None,
    is_i2v: bool = False, # is_i2v might influence model loading specifics in some versions
    dynamic_loading: bool = False
) -> Union[WanModel, Tuple[WanModel, WanModel]]:
    """load DiT model(s) - modified to support dynamic loading

    Args:
        args: command line arguments
        config: model configuration
        device: device to use
        dit_dtype: data type for the model
        dit_weight_dtype: data type for the model weights. None for as-is
        is_i2v: I2V mode (might affect some model config details)
        dynamic_loading: whether to use dynamic loading for dual-dit models

    Returns:
        WanModel or Tuple[WanModel, WanModel]: loaded DiT model(s)
    """
    
    # Load LoRA weights BEFORE model loading for efficient merging
    lora_weights_list_low = None
    lora_multipliers_low = None
    lora_weights_list_high = None
    lora_multipliers_high = None
    
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        lora_weights_list_low = []
        
        for i, lora_path in enumerate(args.lora_weight):
            logger.info(f"Loading LoRA weight from: {lora_path}")
            lora_sd = load_file(lora_path, device="cpu")  # Load to CPU for efficiency
            
            # Apply include/exclude patterns if specified
            include_pattern = None
            exclude_pattern = None
            
            if args.include_patterns is not None and i < len(args.include_patterns):
                include_pattern = args.include_patterns[i]
            if args.exclude_patterns is not None and i < len(args.exclude_patterns):
                exclude_pattern = args.exclude_patterns[i]
            
            if include_pattern or exclude_pattern:
                lora_sd = filter_lora_state_dict(lora_sd, include_pattern, exclude_pattern)
            
            lora_weights_list_low.append(lora_sd)
        
        # Set up multipliers
        if isinstance(args.lora_multiplier, list):
            lora_multipliers_low = args.lora_multiplier
        else:
            lora_multipliers_low = [args.lora_multiplier] * len(lora_weights_list_low)
    
    # Load high noise model LoRA weights if specified
    if hasattr(args, 'lora_weight_high') and args.lora_weight_high is not None and len(args.lora_weight_high) > 0:
        lora_weights_list_high = []
        
        for i, lora_path in enumerate(args.lora_weight_high):
            logger.info(f"Loading LoRA weight for high noise model from: {lora_path}")
            lora_sd = load_file(lora_path, device="cpu")  # Load to CPU for efficiency
            
            # Apply include/exclude patterns if specified
            include_pattern = None
            exclude_pattern = None
            
            if hasattr(args, 'include_patterns_high') and args.include_patterns_high is not None and i < len(args.include_patterns_high):
                include_pattern = args.include_patterns_high[i]
            if hasattr(args, 'exclude_patterns_high') and args.exclude_patterns_high is not None and i < len(args.exclude_patterns_high):
                exclude_pattern = args.exclude_patterns_high[i]
            
            if include_pattern or exclude_pattern:
                lora_sd = filter_lora_state_dict(lora_sd, include_pattern, exclude_pattern)
            
            lora_weights_list_high.append(lora_sd)
        
        # Set up multipliers
        if hasattr(args, 'lora_multiplier_high') and isinstance(args.lora_multiplier_high, list):
            lora_multipliers_high = args.lora_multiplier_high
        else:
            lora_multipliers_high = [args.lora_multiplier_high if hasattr(args, 'lora_multiplier_high') else 1.0] * len(lora_weights_list_high)
    
    # Check if this is a dual-dit model (A14B models)
    is_dual_dit = "A14B" in args.task
    
    if is_dual_dit and args.dit_low_noise and args.dit_high_noise:
        # Check if high noise LoRA is provided but it's not a dual-dit model
        if lora_weights_list_high and not is_dual_dit:
            logger.warning("High noise LoRA weights specified but model is not dual-dit. These will be ignored.")
            
        # Always use dynamic loading for dual-dit models to save RAM
        logger.info(f"Using dynamic loading for dual-dit models (default behavior)")
        # Return paths and LoRA weights for dynamic loading
        return (args.dit_low_noise, args.dit_high_noise, 
                lora_weights_list_low, lora_multipliers_low,
                lora_weights_list_high, lora_multipliers_high)
    elif is_dual_dit:
        logger.warning(f"Task {args.task} expects dual-dit models but dit_low_noise/dit_high_noise not specified. Using single model.")
    
    # Single model loading (standard path)
    dit_path = args.dit
    if dit_path is None:
        raise ValueError("No DiT checkpoint path specified")
        
    loading_device = "cpu"
    if args.blocks_to_swap == 0 and args.lora_weight is None and not args.fp8_scaled:
        loading_device = device

    loading_weight_dtype = dit_weight_dtype
    if args.mixed_dtype:
        # For mixed dtype, load weights as-is without conversion
        loading_weight_dtype = None
    elif args.fp8_scaled or args.lora_weight is not None:
        loading_weight_dtype = dit_dtype
        
    # DEBUG: Print full LoRA list and weights being applied to single DiT model
    if lora_weights_list_low is not None:
        logger.info(f"DEBUG: Loading single DiT model with {len(lora_weights_list_low)} LoRA(s)")
        for i, lora_sd in enumerate(lora_weights_list_low):
            multiplier = lora_multipliers_low[i] if lora_multipliers_low and i < len(lora_multipliers_low) else 1.0
            lora_keys = list(lora_sd.keys())[:5]  # Show first 5 keys
            logger.info(f"DEBUG: LoRA {i+1}/{len(lora_weights_list_low)} for single DiT model - Multiplier: {multiplier}, Keys sample: {lora_keys}")
    else:
        logger.info(f"DEBUG: Loading single DiT model with NO LoRA weights")
        
    model = load_wan_model(
        config, device, dit_path, args.attn_mode, False, 
        loading_device, loading_weight_dtype, False,
        lora_weights_list=lora_weights_list_low, lora_multipliers=lora_multipliers_low
    )
    return model


def merge_lora_weights(model: torch.nn.Module, args: argparse.Namespace, device: torch.device):
    """
    Merges LoRA weights by directly modifying the model's parameters in-place,
    ensuring all device placements are correct.
    """
    if not hasattr(args, 'lora_weight') or not args.lora_weight:
        return
    param_dict = {name: param for name, param in model.named_parameters()}

    for i, lora_path in enumerate(args.lora_weight):
        lora_multiplier = args.lora_multiplier[i] if hasattr(args, 'lora_multiplier') and i < len(args.lora_multiplier) else 1.0
        if lora_multiplier == 0:
            continue

        logging.info(f"Loading and merging LoRA from {lora_path} with multiplier {lora_multiplier}")
        lora_sd = load_file(lora_path, device="cpu") # Load LoRA to CPU

        applied_count = 0
        matched_blocks = set()

        for key, value in lora_sd.items():
            lora_prefix = "diffusion_model."
            if not key.startswith(lora_prefix):
                continue
            
            target_key_base = key[len(lora_prefix):]
            
            # 1. Handle traditional lora_down/lora_up pairs for Linear layers
            if key.endswith(".lora_down.weight"):
                up_key = key.replace(".lora_down.weight", ".lora_up.weight")
                if up_key not in lora_sd:
                    continue

                target_param_name = target_key_base.replace(".lora_down.weight", ".weight")
                if target_param_name not in param_dict:
                    continue
                
                target_param = param_dict[target_param_name]
                
                lora_down_weight = value.to(torch.float32)
                lora_up_weight = lora_sd[up_key].to(torch.float32)
                
                update_matrix = (lora_up_weight @ lora_down_weight) * lora_multiplier
                
                with torch.no_grad():
                    # Move the final update to the SAME device as the target parameter
                    target_param.add_(update_matrix.to(target_param.device, dtype=target_param.dtype))
                
                # Track which blocks were matched
                if 'blocks.' in target_param_name:
                    block_num = target_param_name.split('blocks.')[1].split('.')[0]
                    matched_blocks.add(f"block_{block_num}")
                applied_count += 1
            
            # 2. Handle 'diff' keys (for norm weights)
            elif key.endswith(".diff"):
                target_param_name = target_key_base.replace(".diff", ".weight")
                if target_param_name not in param_dict:
                    continue
                    
                target_param = param_dict[target_param_name]
                update = value.to(torch.float32) * lora_multiplier
                
                with torch.no_grad():
                    target_param.add_(update.to(target_param.device, dtype=target_param.dtype))
                if 'blocks.' in target_param_name:
                    block_num = target_param_name.split('blocks.')[1].split('.')[0]
                    matched_blocks.add(f"block_{block_num}")
                applied_count += 1
                
            # 3. Handle 'diff_b' keys (for biases)
            elif key.endswith(".diff_b"):
                target_param_name = target_key_base.replace(".diff_b", ".bias")
                if target_param_name not in param_dict:
                    continue
                    
                target_param = param_dict[target_param_name]
                update = value.to(torch.float32) * lora_multiplier

                with torch.no_grad():
                    target_param.add_(update.to(target_param.device, dtype=target_param.dtype))
                if 'blocks.' in target_param_name:
                    block_num = target_param_name.split('blocks.')[1].split('.')[0]
                    matched_blocks.add(f"block_{block_num}")
                applied_count += 1

        if applied_count > 0:
            logging.info(f"SUCCESS: Merged {applied_count} LoRA tensors from {os.path.basename(lora_path)} into the model.")
            if matched_blocks:
                logging.info(f"Matched DiT blocks: {sorted(matched_blocks, key=lambda x: int(x.split('_')[1]))}")
        else:
            lora_multiplier = args.lora_multiplier[i] if hasattr(args, 'lora_multiplier') and i < len(args.lora_multiplier) else 1.0
            logging.info(f"Loading and merging LoRA with alt strat from {lora_path} with multiplier {lora_multiplier}")
            weights_sd = load_file(lora_path, device="cpu")
            network = lora_wan.create_arch_network_from_weights(
                multiplier=lora_multiplier,
                weights_sd=weights_sd,
                unet=model,
                for_inference=True
            )
            network.merge_to(text_encoders=None, unet=model, weights_sd=weights_sd, device=device)
            logging.info(f"Successfully merged LoRA: {os.path.basename(lora_path)}")
            del network, weights_sd


def apply_context_windows(args: argparse.Namespace, model_options: dict = None) -> Optional[dict]:
    """Apply context windows configuration to model options if enabled.
    
    Args:
        args: Command line arguments containing context window settings
        model_options: Existing model options dict (will be created if None)
    
    Returns:
        Updated model options dict with context handler, or None if not enabled
    """
    if not args.use_context_windows:
        return model_options
    
    if not CONTEXT_WINDOWS_AVAILABLE:
        logger.warning("Context windows requested but module not available. Proceeding without context windows.")
        return model_options
    
    # Create model options if not provided
    if model_options is None:
        model_options = {}
    
    # Initialize transformer options
    if "transformer_options" not in model_options:
        model_options["transformer_options"] = {}
    
    # Create WAN context windows handler
    try:
        context_handler = WanContextWindowsHandler(
            context_length=args.context_length,
            context_overlap=args.context_overlap,
            context_schedule=args.context_schedule,
            context_stride=args.context_stride,
            closed_loop=args.context_closed_loop,
            fuse_method=args.context_fuse_method
        )
        
        # Store handler in model options
        model_options["context_handler"] = context_handler.handler
        model_options["transformer_options"]["context_handler"] = context_handler.handler
        
        logger.info(f"Context windows enabled: length={args.context_length} frames, "
                   f"overlap={args.context_overlap} frames, schedule={args.context_schedule}, "
                   f"fuse={args.context_fuse_method}")
        
        return model_options
        
    except Exception as e:
        logger.error(f"Failed to initialize context windows: {e}")
        return model_options


def optimize_model(
    model: WanModel, args: argparse.Namespace, device: torch.device, dit_dtype: torch.dtype, dit_weight_dtype: torch.dtype
) -> None:
    """optimize the model (FP8 conversion, device move etc.)

    Args:
        model: dit model
        args: command line arguments
        device: device to use
        dit_dtype: dtype for the model
        dit_weight_dtype: dtype for the model weights
    """
    if args.fp8_scaled:
        # load state dict as-is and optimize to fp8
        state_dict = model.state_dict()

        # if no blocks to swap, we can move the weights to GPU after optimization on GPU (omit redundant CPU->GPU copy)
        move_to_device = args.blocks_to_swap == 0  # if blocks_to_swap > 0, we will keep the model on CPU
        state_dict = model.fp8_optimization(state_dict, device, move_to_device, use_scaled_mm=args.fp8_fast)

        info = model.load_state_dict(state_dict, strict=True, assign=True)
        logger.info(f"Loaded FP8 optimized weights: {info}")

        if args.blocks_to_swap == 0:
            model.to(device)  # make sure all parameters are on the right device (e.g. RoPE etc.)
    else:
        # simple cast to dit_dtype
        target_dtype = None  # load as-is (dit_weight_dtype == dtype of the weights in state_dict)
        target_device = None

        if args.mixed_dtype:
            # Skip dtype conversion for mixed dtype models
            logger.info("Using mixed dtype model - preserving original weight dtypes")
            target_dtype = None
        else:
            if dit_weight_dtype is not None:  # in case of args.fp8 and not args.fp8_scaled
                logger.info(f"Convert model to {dit_weight_dtype}")
                target_dtype = dit_weight_dtype

        if args.blocks_to_swap == 0:
            logger.info(f"Move model to device: {device}")
            target_device = device

        model.to(target_device, target_dtype)  # move and cast  at the same time. this reduces redundant copy operations

    if args.compile:
        compile_backend, compile_mode, compile_dynamic, compile_fullgraph = args.compile_args
        logger.info(
            f"Torch Compiling[Backend: {compile_backend}; Mode: {compile_mode}; Dynamic: {compile_dynamic}; Fullgraph: {compile_fullgraph}]"
        )
        torch._dynamo.config.cache_size_limit = 32
        for i in range(len(model.blocks)):
            model.blocks[i] = torch.compile(
                model.blocks[i],
                backend=compile_backend,
                mode=compile_mode,
                dynamic=compile_dynamic.lower() in "true",
                fullgraph=compile_fullgraph.lower() in "true",
            )

    if args.blocks_to_swap > 0:
        logger.info(f"Enable swap {args.blocks_to_swap} blocks to CPU from device: {device}")
        model.enable_block_swap(args.blocks_to_swap, device, supports_backward=False)
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    else:
        # make sure the model is on the right device
        model.to(device)

    model.eval().requires_grad_(False)
    clean_memory_on_device(device)
    # Verify mixed dtype preservation if requested
    if args.mixed_dtype:
        import collections
        dtype_counts = collections.defaultdict(int)
        dtype_params = collections.defaultdict(int)
        
        for name, param in model.named_parameters():
            dtype_counts[str(param.dtype)] += 1
            dtype_params[str(param.dtype)] += param.numel()
        
        logger.info("Mixed dtype verification - Parameter distribution:")
        total_params = sum(dtype_params.values())
        for dtype_str, count in sorted(dtype_counts.items()):
            param_count = dtype_params[dtype_str]
            size_gb = param_count * torch.finfo(getattr(torch, dtype_str.split('.')[-1])).bits / 8 / 1024**3
            logger.info(f"  {dtype_str}: {count} tensors, {param_count:,} params ({param_count/total_params*100:.1f}%), {size_gb:.2f} GB")
        
        if len(dtype_counts) == 1:
            logger.warning("WARNING: Only one dtype found! Mixed dtype preservation may have failed.")

def prepare_t2v_inputs(
    args: argparse.Namespace, config, accelerator: Accelerator, device: torch.device, vae: Optional[WanVAE] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
    """Prepare inputs for T2V (including Fun-Control variation)

    Args:
        args: command line arguments
        config: model configuration
        accelerator: Accelerator instance
        device: device to use
        vae: VAE model, required only for Fun-Control

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
            (noise, context, context_null, (arg_c, arg_null))
    """
    # Prepare inputs for T2V
    # calculate dimensions and sequence length
    height, width = args.video_size
    frames = args.video_length # Should be set by now
    (ch, lat_f, lat_h, lat_w), seq_len = calculate_dimensions(args.video_size, args.video_length, config, args.task)
    target_shape = (ch, lat_f, lat_h, lat_w) # Will be (48, lat_f, lat_h, lat_w) for ti2v-5B, (16, lat_f, lat_h, lat_w) for others

    # configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

    # set seed
    seed = args.seed # Seed should be set in generate()
    if not args.cpu_noise:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
    else:
        # trash compatible noise
        seed_g = torch.manual_seed(seed)

    # load text encoder
    text_encoder = load_text_encoder(args, config, device)
    text_encoder.model.to(device)

    # encode prompt
    with torch.no_grad():
        if args.fp8_t5:
            with torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype):
                context = text_encoder([args.prompt], device)
                context_null = text_encoder([n_prompt], device)
        else:
            context = text_encoder([args.prompt], device)
            context_null = text_encoder([n_prompt], device)

    # free text encoder and clean memory
    del text_encoder
    clean_memory_on_device(device)
    # Always unload to save memory
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Unloaded T5 model from memory")

    # Initialize 'y' (conditioning latent) to None
    y = None

    # Handle Fun-Control T2V case
    if config.is_fun_control:
        logger.info("Preparing inputs for Fun-Control T2V.")
        if vae is None:
            raise ValueError("VAE is required for Fun-Control T2V input preparation.")

        # Calculate pixel dimensions needed for encoding helper
        pixel_height = lat_h * config.vae_stride[1]
        pixel_width = lat_w * config.vae_stride[2]

        # Create the conditioning latent 'y'
        # This function handles control video encoding (if path provided)
        # and creates the [1, 32, F, H, W] tensor.
        # If no control path, it creates the control part as zeros.
        # Since this is T2V, image_path and end_image_path are None in args,
        # so the image guidance part will also be zeros.
        vae.to_device(device) # Ensure VAE is on device
        y = create_funcontrol_conditioning_latent(
            args, config, vae, device, lat_f, lat_h, lat_w, pixel_height, pixel_width
        )
        # Move VAE back after use
        vae.to_device(args.vae_cache_cpu if args.vae_cache_cpu else "cpu")
        clean_memory_on_device(device)

        if y is None:
            raise RuntimeError("Failed to create FunControl conditioning latent 'y'.")

    # generate noise (base latent noise, shape [16, F, H, W])
    noise = torch.randn(target_shape, dtype=torch.float32, generator=seed_g, device=device if not args.cpu_noise else "cpu")
    noise = noise.to(device)

    # prepare model input arguments
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}

    # Add 'y' ONLY if it was created (i.e., for Fun-Control)
    if y is not None:
        arg_c["y"] = [y] # Model expects y as a list
        arg_null["y"] = [y]
        logger.info(f"Added FunControl conditioning 'y' (shape: {y.shape}) to model inputs.")
    elif config.is_fun_control:
        # This case should technically be handled by y being zeros, but double-check
         logger.warning("FunControl task but 'y' tensor was not generated. Model might error.")
         # Create a zero tensor as fallback if y generation failed somehow?
         # y = torch.zeros([1, 32, lat_f, lat_h, lat_w], device=device, dtype=noise.dtype)
         # arg_c["y"] = [y]
         # arg_null["y"] = [y]


    return noise, context, context_null, (arg_c, arg_null)

# ========================================================================= #
# START OF MODIFIED FUNCTION prepare_i2v_inputs
# ========================================================================= #
def prepare_i2v_inputs(
    args: argparse.Namespace, config, accelerator: Accelerator, device: torch.device, vae: WanVAE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
    """Prepare inputs for I2V (including Fun-Control I2V variation)

    Args:
        args: command line arguments
        config: model configuration
        accelerator: Accelerator instance
        device: device to use
        vae: VAE model, used for image encoding

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
            (noise, context, context_null, y, (arg_c, arg_null))
            'y' is the conditioning latent ([1, 32, F, H, W] for FunControl,
             [1, C+4, F, H, W] for standard I2V with mask).
    """
    if vae is None:
        raise ValueError("VAE must be provided for I2V input preparation.")
    
    # Note: FSDP memory optimization is now handled at a higher level
    # This function assumes VAE/T5/CLIP are loaded sequentially as in single GPU mode

    # --- Prepare Conditioning Latent 'y' ---
    # This check MUST come first to decide the entire logic path
    if config.is_fun_control:
        # --- FunControl I2V Path ---
        logger.info("Preparing inputs for Fun-Control I2V.")

        # Calculate dimensions (FunControl might use different aspect logic)
        height, width = args.video_size
        frames = args.video_length # Should be set by now
        (_, lat_f, lat_h, lat_w), seq_len = calculate_dimensions(args.video_size, args.video_length, config, args.task)
        pixel_height = lat_h * config.vae_stride[1]
        pixel_width = lat_w * config.vae_stride[2]
        noise_channels = 16 # FunControl DiT denoises 16 channels

        logger.info(f"FunControl I2V target pixel resolution: {pixel_height}x{pixel_width}, latent shape: ({lat_f}, {lat_h}, {lat_w}), seq_len: {seq_len}")

        # set seed
        seed = args.seed
        if not args.cpu_noise:
            seed_g = torch.Generator(device=device)
            seed_g.manual_seed(seed)
        else:
            seed_g = torch.manual_seed(seed)

        # generate noise (for the part being denoised by the DiT)
        noise = torch.randn(
            noise_channels, lat_f, lat_h, lat_w,
            dtype=torch.float32, generator=seed_g,
            device=device if not args.cpu_noise else "cpu",
        )
        noise = noise.to(device)

        # configure negative prompt
        n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

        # Unload VAE before loading T5 to save memory (sequential loading pattern)
        logger.info("Unloading VAE before loading T5...")
        vae.to_device("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        # load text encoder & encode prompts
        text_encoder = load_text_encoder(args, config, device)
        text_encoder.model.to(device)
        with torch.no_grad():
            if args.fp8_t5:
                with torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype):
                    context = text_encoder([args.prompt], device)
                    context_null = text_encoder([n_prompt], device)
            else:
                context = text_encoder([args.prompt], device)
                context_null = text_encoder([n_prompt], device)
        
        # Debug: Check what T5 encoder returned
        logger.info(f"T5 Encoder output - context type: {type(context)}, context_null type: {type(context_null)}")
        if isinstance(context, list):
            logger.info(f"Context is list with {len(context)} items, first item shape: {context[0].shape}")
        else:
            logger.info(f"Context tensor shape: {context.shape}")
        
        del text_encoder
        clean_memory_on_device(device)
        # Always unload to save memory
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Unloaded T5 model from memory")
        
        # Reload VAE after T5 is unloaded
        logger.info("Reloading VAE after T5 unload...")
        vae.to_device(device)

        # load CLIP model & encode image
        clip = load_clip_model(args, config, device)
        clip.model.to(device)
        if not args.image_path:
             raise ValueError("--image_path is required for FunControl I2V mode.")
        img_clip = Image.open(args.image_path).convert("RGB")
        img_tensor_clip = TF.to_tensor(img_clip).sub_(0.5).div_(0.5).to(device) # CHW, [-1, 1]
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
            clip_context = clip.visual([img_tensor_clip.unsqueeze(1)]) # Add Frame dim
        del clip, img_clip, img_tensor_clip
        clean_memory_on_device(device)
        # Always unload to save memory
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Unloaded CLIP model from memory")

        fun_ref_latent = None
        
        # In FSDP mode, each rank performs its own VAE encoding to avoid broadcast issues
        # The results will be identical since they use the same input
        
        # Check if the task requires ref_conv and if a reference image is provided via --image_path
        if args.task == "i2v-14B-FC-1.1" and args.image_path is not None:
            logger.info(f"Task {args.task} requires ref_conv. Encoding reference image from --image_path: {args.image_path}")
            try:
                ref_img = Image.open(args.image_path).convert("RGB")
                ref_img_np = np.array(ref_img)
                # Resize ref image to target pixel dimensions
                interpolation = cv2.INTER_AREA if pixel_height < ref_img_np.shape[0] else cv2.INTER_CUBIC
                ref_img_resized_np = cv2.resize(ref_img_np, (pixel_width, pixel_height), interpolation=interpolation)
                # Convert to tensor CFHW, range [-1, 1]
                ref_img_tensor = TF.to_tensor(ref_img_resized_np).sub_(0.5).div_(0.5).to(device)
                ref_img_tensor = ref_img_tensor.unsqueeze(1) # Add frame dim: C,F,H,W

                vae.to_device(device) # Ensure VAE is on device for encoding
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vae.dtype):
                    # Encode the single reference frame
                    # vae.encode returns list, take first element. Result shape [C', 1, H', W']
                    fun_ref_latent = vae.encode([ref_img_tensor])[0]
                    # Squeeze the frame dimension for Conv2d in the model: [C', H', W']
                    fun_ref_latent = fun_ref_latent.squeeze(1)
                logger.info(f"Encoded fun_ref latent. Shape: {fun_ref_latent.shape}")
                # Keep VAE on device for main conditioning latent creation below

            except Exception as e:
                logger.error(f"Error processing reference image for fun_ref: {e}")
                fun_ref_latent = None # Continue without ref if encoding fails

            # **IMPORTANT**: Since --image_path is now used for fun_ref,
            # temporarily set it to None *before* calling create_funcontrol_conditioning_latent
            # so it doesn't get processed *again* as a start image inside that function.
            original_image_path = args.image_path
            args.image_path = None
        else:
            original_image_path = None

        # Use the FunControl helper function to create the 32-channel 'y'
        vae.to_device(device) # Ensure VAE is on compute device
        y = create_funcontrol_conditioning_latent(
            args, config, vae, device, lat_f, lat_h, lat_w, pixel_height, pixel_width
        )
        if args.task == "i2v-14B-FC-1.1" and original_image_path:
             args.image_path = original_image_path        
        if y is None:
            raise RuntimeError("Failed to create FunControl conditioning latent 'y'.")
        vae.to_device(args.vae_cache_cpu if args.vae_cache_cpu else "cpu") # Move VAE back
        clean_memory_on_device(device)

        # Prepare Model Input Arguments for FunControl
        y_for_model = y[0] # Shape becomes [32, F, H, W]
        # A14B models don't have img_emb layer, so don't pass clip_fea
        use_clip_fea = clip_context if not ("A14B" in args.task) else None
        
        arg_c = {
            "context": context,
            "clip_fea": use_clip_fea,
            "seq_len": seq_len,
            "y": [y_for_model], # Pass the 4D tensor in the list
        }
        arg_null = {
            "context": context_null,
            "clip_fea": use_clip_fea,
            "seq_len": seq_len,
            "y": [y_for_model], # Pass the 4D tensor in the list
        }
        
        if fun_ref_latent is not None:
            # Model forward expects fun_ref directly, not in a list like 'y'
            arg_c["fun_ref"] = fun_ref_latent
            arg_null["fun_ref"] = fun_ref_latent # Pass to both cond and uncond
            logger.info("Added fun_ref latent to model inputs.")    

        # Return noise, context, context_null, y (for potential debugging), (arg_c, arg_null)
        return noise, context, context_null, y, (arg_c, arg_null)

    else:
        # --- Standard I2V Path (Logic copied/adapted from original wan_generate_video.py) ---
        logger.info("Preparing inputs for standard I2V.")

        # get video dimensions
        height, width = args.video_size
        frames = args.video_length # Should be set by now
        max_area = width * height

        # load image
        if not args.image_path:
            raise ValueError("--image_path is required for standard I2V mode.")
        img = Image.open(args.image_path).convert("RGB")
        img_cv2 = np.array(img)  # PIL to numpy
        img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device) # For CLIP

        # end frame image
        end_img = None
        end_img_cv2 = None
        if args.end_image_path is not None:
            end_img = Image.open(args.end_image_path).convert("RGB")
            end_img_cv2 = np.array(end_img)  # PIL to numpy
        has_end_image = end_img is not None

        # calculate latent dimensions: keep aspect ratio (Original Method)
        img_height, img_width = img.size[::-1] # PIL size is W,H
        aspect_ratio = img_height / img_width
        lat_h = round(np.sqrt(max_area * aspect_ratio) / config.vae_stride[1] / config.patch_size[1]) * config.patch_size[1]
        lat_w = round(np.sqrt(max_area / aspect_ratio) / config.vae_stride[2] / config.patch_size[2]) * config.patch_size[2]
        target_height = lat_h * config.vae_stride[1]
        target_width = lat_w * config.vae_stride[2]

        # --- CRITICAL ORIGINAL LOGIC DIFFERENCE #1: Frame Dimension ---
        lat_f_base = (frames - 1) // config.vae_stride[0] + 1  # size of latent frames
        lat_f_effective = lat_f_base + (1 if has_end_image else 0) # Adjust frame dim if end image exists

        # --- CRITICAL ORIGINAL LOGIC DIFFERENCE #2: Sequence Length ---
        max_seq_len = math.ceil(lat_f_effective * lat_h * lat_w / (config.patch_size[1] * config.patch_size[2]))

        logger.info(f"Standard I2V target pixel resolution: {target_height}x{target_width}, latent shape: ({lat_f_effective}, {lat_h}, {lat_w}), seq_len: {max_seq_len}")

        # set seed
        seed = args.seed
        if not args.cpu_noise:
            seed_g = torch.Generator(device=device)
            seed_g.manual_seed(seed)
        else:
            seed_g = torch.manual_seed(seed)

        # --- CRITICAL ORIGINAL LOGIC DIFFERENCE #3: Noise Shape ---
        noise = torch.randn(
            16, # Channel dim for latent
            lat_f_effective, # Use adjusted frame dim
            lat_h, lat_w,
            dtype=torch.float32, generator=seed_g,
            device=device if not args.cpu_noise else "cpu",
        )
        noise = noise.to(device)

        # configure negative prompt
        n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

        # Unload VAE before loading T5 to save memory (sequential loading pattern)
        logger.info("Unloading VAE before loading T5...")
        vae.to_device("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        # load text encoder & encode prompts
        text_encoder = load_text_encoder(args, config, device)
        text_encoder.model.to(device)
        with torch.no_grad():
            if args.fp8_t5:
                with torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype):
                    context = text_encoder([args.prompt], device)
                    context_null = text_encoder([n_prompt], device)
            else:
                context = text_encoder([args.prompt], device)
                context_null = text_encoder([n_prompt], device)
        
        # Debug: Check what T5 encoder returned
        logger.info(f"T5 Encoder output - context type: {type(context)}, context_null type: {type(context_null)}")
        if isinstance(context, list):
            logger.info(f"Context is list with {len(context)} items, first item shape: {context[0].shape}")
        else:
            logger.info(f"Context tensor shape: {context.shape}")
        
        del text_encoder
        clean_memory_on_device(device)
        # Always unload to save memory
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Unloaded T5 model from memory")
        
        # Reload VAE after T5 is unloaded
        logger.info("Reloading VAE after T5 unload...")
        vae.to_device(device)

        # load CLIP model & encode image
        clip = load_clip_model(args, config, device)
        clip.model.to(device)
        logger.info(f"Encoding image to CLIP context")
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
            # Use the [-1, 1] tensor directly if clip.visual expects that format
            # clip_context = clip.visual([img_tensor[:, None, :, :]]).squeeze(1) # Original had [img_tensor[:, None, :, :]] which adds frame dim
            # Use unsqueeze(1) which seems more consistent with other parts
            clip_context = clip.visual([img_tensor.unsqueeze(1)]) # Add Frame dim
        logger.info(f"CLIP Encoding complete")
        del clip
        clean_memory_on_device(device)
        # Always unload to save memory
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Unloaded CLIP model from memory")

        # --- CRITICAL ORIGINAL LOGIC DIFFERENCE #4: VAE Encoding and 'y' construction ---
        logger.info(f"Encoding image(s) to latent space (Standard I2V method)")
        
        vae.to_device(device)
        
        # Perform VAE encoding on all ranks
        # Resize image(s) for VAE
        interpolation = cv2.INTER_AREA if target_height < img_cv2.shape[0] else cv2.INTER_CUBIC
        img_resized_np = cv2.resize(img_cv2, (target_width, target_height), interpolation=interpolation)
        img_resized = TF.to_tensor(img_resized_np).sub_(0.5).div_(0.5).to(device)  # [-1, 1], CHW
        img_resized = img_resized.unsqueeze(1)  # Add frame dimension -> CFHW, Shape [C, 1, H, W]

        end_img_resized = None
        if has_end_image and end_img_cv2 is not None:
            interpolation_end = cv2.INTER_AREA if target_height < end_img_cv2.shape[0] else cv2.INTER_CUBIC
            end_img_resized_np = cv2.resize(end_img_cv2, (target_width, target_height), interpolation=interpolation_end)
            end_img_resized = TF.to_tensor(end_img_resized_np).sub_(0.5).div_(0.5).to(device) # [-1, 1], CHW
            end_img_resized = end_img_resized.unsqueeze(1) # Add frame dimension -> CFHW, Shape [C, 1, H, W]

        # --- CRITICAL ORIGINAL LOGIC DIFFERENCE #5: Mask Shape ---
        msk = torch.zeros(4, lat_f_effective, lat_h, lat_w, device=device, dtype=vae.dtype) # Use adjusted frame dim
        msk[:, 0] = 1 # Mask first frame
        if has_end_image:
            msk[:, -1] = 1 # Mask last frame (the lat_f+1'th frame)

        # Encode image(s) using VAE (Padded Method)
        with accelerator.autocast(), torch.no_grad():
            # Pad the *start* image tensor temporally before encoding
            # Calculate padding needed to reach base frame count (before adding end frame)
            padding_frames_needed = frames - 1 # Number of frames to generate *after* the first
            if padding_frames_needed < 0: padding_frames_needed = 0

            img_padded = img_resized # Start with [C, 1, H, W]
            if padding_frames_needed > 0:
                 # Create padding tensor [C, padding_frames_needed, H, W]
                 padding_tensor = torch.zeros(
                     img_resized.shape[0], padding_frames_needed, img_resized.shape[2], img_resized.shape[3],
                     device=device, dtype=img_resized.dtype
                 )
                 # Concatenate along frame dimension (dim=1)
                 img_padded = torch.cat([img_resized, padding_tensor], dim=1)
                 # Shape should now be [C, 1 + padding_frames_needed, H, W] = [C, frames, H, W]

            # Encode the padded start image tensor. VAE output matches latent frame count.
            # vae.encode expects [C, F, H, W]
            y_latent_base = vae.encode([img_padded])[0] # Shape [C', lat_f_base, H, W]

            if has_end_image and end_img_resized is not None:
                 # Encode the single end frame
                 y_end = vae.encode([end_img_resized])[0] # Shape [C', 1, H, W]
                 # Concatenate along frame dimension (dim=1)
                 y_latent_combined = torch.cat([y_latent_base, y_end], dim=1) # Shape [C', lat_f_base + 1, H, W] = [C', lat_f_effective, H, W]
            else:
                 y_latent_combined = y_latent_base # Shape [C', lat_f_base, H, W] = [C', lat_f_effective, H, W]

        # Concatenate mask and the combined latent
        # --- CRITICAL ORIGINAL LOGIC DIFFERENCE #6: Final 'y' Tensor ---
        y = torch.cat([msk, y_latent_combined], dim=0) # Shape [4+C', lat_f_effective, H, W]
        
        # y = y.unsqueeze(0) # Add batch dimension? Check model input requirements. Assume model forward handles list/batching.

        logger.info(f"Standard I2V conditioning 'y' constructed. Shape: {y.shape}")
        logger.info(f"Image encoding complete")

        # Move VAE back
        vae.to_device(args.vae_cache_cpu if args.vae_cache_cpu else "cpu")
        clean_memory_on_device(device)

        # Prepare model input arguments for Standard I2V
        # A14B models don't have img_emb layer, so don't pass clip_fea
        use_clip_fea = clip_context if not ("A14B" in args.task) else None
        
        arg_c = {
            "context": context, # Model expects batch dim? Assuming yes.
            "clip_fea": use_clip_fea,
            "seq_len": max_seq_len, # Use original seq len calculation
            "y": [y], # Use the 'original method' y
        }
        arg_null = {
            "context": context_null,
            "clip_fea": use_clip_fea,
            "seq_len": max_seq_len,
            "y": [y], # Use the 'original method' y
        }

        # Return noise, context, context_null, y (for debugging), (arg_c, arg_null)
        return noise, context, context_null, y, (arg_c, arg_null)
# ========================================================================= #
# END OF MODIFIED FUNCTION prepare_i2v_inputs
# ========================================================================= #


def prepare_ti2v_inputs(
    args: argparse.Namespace, config, accelerator: Accelerator, device: torch.device, vae: WanVAE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare inputs for TI2V (Text+Image-to-Video) inference for ti2v-5B model
    
    Args:
        args: command line arguments
        config: model configuration
        accelerator: Accelerator instance
        device: device to use
        vae: VAE model, used for image encoding
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
            (noise, context, context_null, inputs)
    """
    if vae is None:
        raise ValueError("VAE must be provided for TI2V input preparation.")
        
    logger.info("Preparing inputs for Text+Image-to-Video (TI2V) inference.")
    
    # Note: FSDP memory optimization is now handled at a higher level
    # This function assumes VAE/T5/CLIP are loaded sequentially as in single GPU mode
    
    # Load and process the input image
    image = Image.open(args.image_path).convert('RGB')
    logger.info(f"Loaded input image: {args.image_path}")
    
    # Get dimensions
    height, width = args.video_size
    frames = args.video_length
    
    # Resize image to match video dimensions
    image = image.resize((width, height), Image.LANCZOS)
    
    # Calculate latent dimensions following VAE strides (needed for expansion)
    lat_f = (frames - 1) // config.vae_stride[0] + 1
    lat_h = height // config.vae_stride[1] 
    lat_w = width // config.vae_stride[2]
    
    # Convert to tensor and normalize to [0, 1]
    # Follow official implementation format: [3, 1, H, W]
    image_tensor = TF.to_tensor(image).sub_(0.5).div_(0.5).to(device, dtype=vae.dtype).unsqueeze(1)  # [3, 1, H, W]
    
    # Encode image to latent space
    logger.info("Encoding input image to latent space...")
    with torch.no_grad():
        # Ensure VAE is on correct device
        vae.to_device(device)
        
        # Encode image - handle different VAE types
        if hasattr(vae, 'model') and hasattr(vae, 'scale'):
            # Wan2_2_VAE type - expects list input with frame dimension [C, F, H, W]
            # image_tensor is already [3, 1, H, W] from official format
            encoded_latents = vae.encode([image_tensor])  # Returns list of encoded latents
            image_latent = encoded_latents[0]  # Get the first (and only) latent [C, 1, H', W']
            
            # CRITICAL: Expand the single image frame to all temporal positions to match noise
            # This matches the official implementation where z[0] has same temporal dims as noise
            image_latent = image_latent.expand(-1, lat_f, -1, -1)  # [C, F, H', W'] - expand to all frames
            
        else:
            # Original WanVAE type - expects tensor input
            image_latent = vae.encode(image_tensor).latent_dist.sample()  # [1, C, H', W']
            # Scale by VAE scaling factor
            image_latent = image_latent * vae.config.scaling_factor
            
            # For TI2V, we need to expand the single image frame to all temporal positions
            # This matches the official implementation where z[0] has full temporal dimensions
            image_latent = image_latent.expand(-1, lat_f, -1, -1)  # [C, F, H', W'] - expand to all frames
        
    logger.info(f"Image latent shape: {image_latent.shape}")
    
    # Move VAE to cache/CPU for memory management
    if args.vae_cache_cpu:
        vae.to_device("cpu")
        clean_memory_on_device(device)
    
    # Prepare text context
    t5 = load_text_encoder(args, config, device)
    
    # Encode positive prompt - follow official API
    context = t5([args.prompt], device)[0]  # [1, seq_len, dim]
    
    # Encode negative prompt (use default if not specified)
    negative_prompt = args.negative_prompt if args.negative_prompt else ""
    context_null = t5([negative_prompt], device)[0]  # [1, seq_len, dim]
    
    # Move T5 to CPU for memory management - follow official pattern
    t5.model.cpu()
    clean_memory_on_device(device)
    # Always unload to save memory
    del t5.model
    del t5
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Unloaded T5 model from memory")
    
    # Generate noise matching the latent dimensions - follow official pattern
    # Create noise tensor [z_dim, F, H, W] - 4D like official implementation  
    # For ti2v-5B: z_dim=48 (VAE latent channels)
    vae_z_dim = 48 if args.task == "ti2v-5B" else 16  # 48 for 5B, 16 for others
    noise = torch.randn(
        vae_z_dim, lat_f, lat_h, lat_w,
        device=device if not args.cpu_noise else "cpu", dtype=torch.float32,
        generator=torch.Generator(device=device if not args.cpu_noise else "cpu").manual_seed(args.seed)
    )
    
    # Calculate sequence length for model
    seq_len = lat_f * lat_h * lat_w // (config.patch_size[1] * config.patch_size[2])
    
    # Prepare arguments for ti2v model following standard format (no image_latent param)
    arg_c = {
        "context": [context],
        "seq_len": seq_len
    }
    
    arg_null = {
        "context": [context_null], 
        "seq_len": seq_len
    }
    
    logger.info(f"TI2V inputs prepared: noise {noise.shape}, context {context.shape}, image_latent {image_latent.shape}")
    
    # Store image_latent in arg_c and arg_null for access during sampling (not as model param)
    arg_c["_image_latent"] = image_latent  # Store for later use, not passed to model
    arg_null["_image_latent"] = image_latent
    
    return noise, context, context_null, (arg_c, arg_null)


# --- V2V Helper Functions ---

def load_video(video_path, start_frame=0, num_frames=None, bucket_reso=(256, 256)):
    """Load video frames and resize them to the target resolution for V2V.

    Args:
        video_path (str): Path to the video file
        start_frame (int): First frame to load (0-indexed)
        num_frames (int, optional): Number of frames to load. If None, load all frames from start_frame.
        bucket_reso (tuple): Target resolution (height, width)

    Returns:
        list: List of numpy arrays containing video frames in RGB format, resized.
        int: Actual number of frames loaded.
    """
    logger.info(f"Loading video for V2V from {video_path}, target reso {bucket_reso}, frames {start_frame}-{start_frame+num_frames if num_frames else 'end'}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    # Get total frame count and FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Input video has {total_frames} frames, {fps} FPS")

    # Calculate how many frames to load
    if num_frames is None:
        frames_to_load = total_frames - start_frame
    else:
        # Make sure we don't try to load more frames than exist
        frames_to_load = min(num_frames, total_frames - start_frame)

    if frames_to_load <= 0:
        cap.release()
        logger.warning(f"No frames to load (start_frame={start_frame}, num_frames={num_frames}, total_frames={total_frames})")
        return [], 0

    # Skip to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read frames
    frames = []
    target_h, target_w = bucket_reso
    for i in range(frames_to_load):
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Could only read {len(frames)} frames out of {frames_to_load} requested.")
            break

        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame
        # Use INTER_AREA for downscaling, INTER_LANCZOS4/CUBIC for upscaling
        current_h, current_w = frame_rgb.shape[:2]
        if target_h * target_w < current_h * current_w:
             interpolation = cv2.INTER_AREA
        else:
             interpolation = cv2.INTER_LANCZOS4 # Higher quality for upscaling
        frame_resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=interpolation)

        frames.append(frame_resized)

    cap.release()
    actual_frames_loaded = len(frames)
    logger.info(f"Successfully loaded and resized {actual_frames_loaded} frames for V2V.")

    return frames, actual_frames_loaded


def encode_video_to_latents(video_tensor: torch.Tensor, vae: WanVAE, device: torch.device, vae_dtype: torch.dtype, args: argparse.Namespace) -> torch.Tensor: # Added args parameter
    """Encode video tensor to latent space using VAE for V2V.

    Args:
        video_tensor (torch.Tensor): Video tensor with shape [B, C, F, H, W], values in [0, 1].
        vae (WanVAE): VAE model instance.
        device (torch.device): Device to perform encoding on.
        vae_dtype (torch.dtype): Target dtype for the output latents.
        args (argparse.Namespace): Command line arguments (needed for vae_cache_cpu). # Added args description

    Returns:
        torch.Tensor: Encoded latents with shape [B, C', F', H', W'].
    """
    if vae is None:
        raise ValueError("VAE must be provided for video encoding.")

    logger.info(f"Encoding video tensor to latents: input shape {video_tensor.shape}")

    # Ensure VAE is on the correct device
    vae.to_device(device)

    # Prepare video tensor: move to device, ensure float32, scale to [-1, 1]
    video_tensor = video_tensor.to(device=device, dtype=torch.float32)
    video_tensor = video_tensor * 2.0 - 1.0 # Scale from [0, 1] to [-1, 1]

    # WanVAE expects input as a list of [C, F, H, W] tensors (no batch dim)
    # Process each video in the batch if batch size > 1 (usually 1 here)
    latents_list = []
    batch_size = video_tensor.shape[0]
    for i in range(batch_size):
        video_single = video_tensor[i] # Shape [C, F, H, W]
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vae.dtype): # Use VAE's internal dtype for autocast
            # vae.encode expects a list containing the tensor
            encoded_latent = vae.encode([video_single])[0] # Returns tensor [C', F', H', W']
            latents_list.append(encoded_latent)

    # Stack results back into a batch
    latents = torch.stack(latents_list, dim=0) # Shape [B, C', F', H', W']

    # Move VAE back to CPU (or cache device)
    # Use the passed args object here
    vae_target_device = torch.device("cpu") if not args.vae_cache_cpu else torch.device("cpu") # Default to CPU, TODO: check if cache device needs specific name
    if args.vae_cache_cpu:
        # Determine the actual cache device if needed, for now, CPU is safe fallback
        logger.info("Moving VAE to CPU for caching (as configured by --vae_cache_cpu).")
    else:
        logger.info("Moving VAE to CPU after encoding.")
    vae.to_device(vae_target_device) # Use args to decide target device
    clean_memory_on_device(device) # Clean the GPU memory

    # Convert latents to the desired final dtype (e.g., bfloat16)
    latents = latents.to(dtype=vae_dtype)
    logger.info(f"Encoded video latents shape: {latents.shape}, dtype: {latents.dtype}")

    return latents


def prepare_v2v_inputs(args: argparse.Namespace, config, accelerator: Accelerator, device: torch.device, video_latents: torch.Tensor):
    """Prepare inputs for Video2Video inference based on encoded video latents.

    Args:
        args (argparse.Namespace): Command line arguments.
        config: Model configuration.
        accelerator: Accelerator instance.
        device (torch.device): Device to use.
        video_latents (torch.Tensor): Encoded latent representation of input video [B, C', F', H', W'].

    Returns:
        Tuple containing noise, context, context_null, (arg_c, arg_null).
    """
    # Get dimensions directly from the video latents
    if len(video_latents.shape) != 5:
        raise ValueError(f"Expected video_latents to have 5 dimensions [B, C, F, H, W], but got shape {video_latents.shape}")

    batch_size, latent_channels, lat_f, lat_h, lat_w = video_latents.shape
    if batch_size != 1:
        logger.warning(f"V2V input preparation currently assumes batch size 1, but got {batch_size}. Using first item.")
        video_latents = video_latents[0:1] # Keep batch dim

    target_shape = video_latents.shape[1:] # Get shape without batch dim: [C', F', H', W']

    # Calculate the sequence length based on actual latent dimensions
    patch_h, patch_w = config.patch_size[1], config.patch_size[2]
    spatial_tokens_per_frame = (lat_h * lat_w) // (patch_h * patch_w)
    seq_len = spatial_tokens_per_frame * lat_f
    logger.info(f"V2V derived latent shape: {target_shape}, seq_len: {seq_len}")

    # Configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

    # Set seed (already set in generate(), just need generator)
    seed = args.seed
    if not args.cpu_noise:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
    else:
        seed_g = torch.manual_seed(seed)

    # Load text encoder
    text_encoder = load_text_encoder(args, config, device)
    text_encoder.model.to(device)

    # Encode prompt
    with torch.no_grad():
        if args.fp8_t5:
            with torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype):
                context = text_encoder([args.prompt], device)
                context_null = text_encoder([n_prompt], device)
        else:
            context = text_encoder([args.prompt], device)
            context_null = text_encoder([n_prompt], device)

    # Free text encoder and clean memory
    del text_encoder
    clean_memory_on_device(device)
    # Always unload to save memory
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Unloaded T5 model from memory")

    # Generate noise with the same shape as video_latents (including batch dimension)
    noise = torch.randn(
        video_latents.shape, # [B, C', F', H', W']
        dtype=torch.float32,
        device=device if not args.cpu_noise else "cpu",
        generator=seed_g
    )
    noise = noise.to(device) # Ensure noise is on the target device

    # Prepare model input arguments (context needs to match batch size of latents)
    # Assuming batch size 1 for now based on implementation
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}

    # V2V does not use 'y' or 'clip_fea' in the standard Wan model case
    # If a specific V2V variant *did* need them, they would be added here.

    return noise, context, context_null, (arg_c, arg_null)


def prepare_v2v_i2v_inputs(
    args: argparse.Namespace, 
    config, 
    accelerator: Accelerator, 
    device: torch.device, 
    vae: WanVAE,
    video_frames_np: List[np.ndarray]  # Pass in loaded video frames
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
    """Prepare V2V inputs for i2v models (combines V2V video encoding with I2V conditioning).
    
    Args:
        args: command line arguments
        config: model configuration
        accelerator: Accelerator instance
        device: device to use
        vae: VAE model instance
        video_frames_np: List of video frames as numpy arrays (HWC, 0-255)
        
    Returns:
        Tuple containing noise, context, context_null, clip_context, video_latents, (arg_c, arg_null)
    """
    if vae is None:
        raise ValueError("VAE must be provided for V2V-I2V input preparation.")
        
    logger.info("Preparing V2V inputs for i2v model (with CLIP conditioning)")
    
    # Get dimensions from args
    height, width = args.video_size
    frames = args.video_length
    
    # Convert frames to tensor and encode to latents
    video_tensor = torch.from_numpy(np.stack(video_frames_np, axis=0))  # [F,H,W,C]
    video_tensor = video_tensor.permute(0, 3, 1, 2).float() / 255.0  # [F,C,H,W], [0,1]
    video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1,C,F,H,W]
    
    # Encode video to latents
    video_latents = encode_video_to_latents(video_tensor, vae, device, vae.dtype, args)
    logger.info(f"Encoded video to latents: {video_latents.shape}")
    
    # Extract first frame for CLIP conditioning (i2v requirement)
    first_frame_np = video_frames_np[0]  # HWC, 0-255
    first_frame_pil = Image.fromarray(first_frame_np)
    
    # Calculate dimensions from latents
    _, _, lat_f, lat_h, lat_w = video_latents.shape
    seq_len = (lat_h * lat_w) // (config.patch_size[1] * config.patch_size[2]) * lat_f
    
    # Configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt
    
    # Set seed
    seed = args.seed
    if not args.cpu_noise:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
    else:
        seed_g = torch.manual_seed(seed)
    
    # Load text encoder and encode prompts
    text_encoder = load_text_encoder(args, config, device)
    text_encoder.model.to(device)
    
    with torch.no_grad():
        if args.fp8_t5:
            with torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype):
                context = text_encoder([args.prompt], device)
                context_null = text_encoder([n_prompt], device)
        else:
            context = text_encoder([args.prompt], device)
            context_null = text_encoder([n_prompt], device)
    
    # Free text encoder
    del text_encoder
    clean_memory_on_device(device)
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Unloaded T5 model from memory")
    
    # Load CLIP model and encode first frame
    clip = load_clip_model(args, config, device)
    clip.model.to(device)
    
    # Convert first frame for CLIP
    img_tensor_clip = TF.to_tensor(first_frame_pil).sub_(0.5).div_(0.5).to(device)  # CHW, [-1, 1]
    
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
        clip_context = clip.visual([img_tensor_clip.unsqueeze(1)])  # Add Frame dim
    
    logger.info("Encoded first frame with CLIP for i2v conditioning")
    
    # Free CLIP model
    del clip
    clean_memory_on_device(device)
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Unloaded CLIP model from memory")
    
    # Generate noise matching video latents shape
    noise = torch.randn(
        video_latents.shape,  # [B, C', F', H', W']
        dtype=torch.float32,
        device=device if not args.cpu_noise else "cpu",
        generator=seed_g
    )
    noise = noise.to(device)
    
    # Prepare model input arguments
    # A14B models don't have img_emb layer, so don't pass clip_fea
    use_clip_fea = clip_context if not ("A14B" in args.task) else None
    
    # For i2v models, we need to prepare 'y' tensor (mask + latent)
    # For V2V with i2v, we want to preserve the first frame and generate the rest
    # This is like standard I2V where the first frame is given
    msk = torch.zeros(4, lat_f, lat_h, lat_w, device=device, dtype=vae.dtype)
    msk[:, 0] = 1  # Mask (preserve) the first frame only
    
    # Concatenate mask with video latents to create 'y'
    y = torch.cat([msk, video_latents.squeeze(0)], dim=0)  # [4+C', F', H', W']
    
    arg_c = {
        "context": context,
        "clip_fea": use_clip_fea,
        "seq_len": seq_len,
        "y": [y],  # i2v models expect y as a list
    }
    
    arg_null = {
        "context": context_null,
        "clip_fea": use_clip_fea,
        "seq_len": seq_len,
        "y": [y],
    }
    
    return noise, context, context_null, clip_context, video_latents, (arg_c, arg_null)


# --- End V2V Helper Functions ---

def load_control_video(control_path: str, frames: int, height: int, width: int, args=None) -> torch.Tensor:
    """Load control video to pixel space for Fun-Control model with enhanced control.

    Args:
        control_path: path to control video
        frames: number of frames in the video
        height: height of the video
        width: width of the video
        args: command line arguments (optional, for logging)

    Returns:
        torch.Tensor: control video tensor, CFHW, range [-1, 1]
    """
    logger = logging.getLogger(__name__)
    msg = f"Load control video for Fun-Control from {control_path}"
    if args:
        # Use the correct argument names from wanFUN_generate_video.py
        msg += f" (weight={args.control_weight}, start={args.control_start}, end={args.control_end})"
    logger.info(msg)

    # Use the original helper from hv_generate_video for consistency
    if os.path.isfile(control_path):
        # Use hv_load_video which returns list of numpy arrays (HWC, 0-255)
        video_frames_np = hv_load_video(control_path, 0, frames, bucket_reso=(width, height))
    elif os.path.isdir(control_path):
         # Use hv_load_images which returns list of numpy arrays (HWC, 0-255)
        video_frames_np = hv_load_images(control_path, frames, bucket_reso=(width, height))
    else:
        raise FileNotFoundError(f"Control path not found: {control_path}")

    if not video_frames_np:
         raise ValueError(f"No frames loaded from control path: {control_path}")
    if len(video_frames_np) < frames:
        logger.warning(f"Control video has {len(video_frames_np)} frames, less than requested {frames}. Using available frames.")
        # Optionally, could repeat last frame or loop, but using available is simplest
        frames = len(video_frames_np) # Adjust frame count

    # Stack and convert to tensor: F, H, W, C (0-255) -> F, C, H, W (-1 to 1)
    video_frames_np = np.stack(video_frames_np, axis=0)
    video_tensor = torch.from_numpy(video_frames_np).permute(0, 3, 1, 2).float() / 127.5 - 1.0 # Normalize to [-1, 1]

    # Permute to C, F, H, W
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    logger.info(f"Loaded Fun-Control video tensor shape: {video_tensor.shape}")

    return video_tensor

def setup_scheduler(args: argparse.Namespace, config, device: torch.device) -> Tuple[Any, torch.Tensor]:
    """setup scheduler for sampling

    Args:
        args: command line arguments
        config: model configuration
        device: device to use

    Returns:
        Tuple[Any, torch.Tensor]: (scheduler, timesteps)
    """
    if args.sample_solver == "unipc":
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(args.infer_steps, device=device, shift=args.flow_shift)
        timesteps = scheduler.timesteps
    elif args.sample_solver == "dpm++":
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        sampling_sigmas = get_sampling_sigmas(args.infer_steps, args.flow_shift)
        timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sampling_sigmas)
    elif args.sample_solver == "vanilla":
        scheduler = FlowMatchDiscreteScheduler(num_train_timesteps=config.num_train_timesteps, shift=args.flow_shift)
        scheduler.set_timesteps(args.infer_steps, device=device)
        timesteps = scheduler.timesteps

        # FlowMatchDiscreteScheduler does not support generator argument in step method
        org_step = scheduler.step

        def step_wrapper(
            model_output: torch.Tensor,
            timestep: Union[int, torch.Tensor],
            sample: torch.Tensor,
            return_dict: bool = True,
            generator=None, # Add generator argument here
        ):
            # Call original step, ignoring generator if it doesn't accept it
            try:
                # Try calling with generator if the underlying class was updated
                return org_step(model_output, timestep, sample, return_dict=return_dict, generator=generator)
            except TypeError:
                 # Fallback to calling without generator
                 logger.warning("Scheduler step does not support generator argument, proceeding without it.")
                 return org_step(model_output, timestep, sample, return_dict=return_dict)


        scheduler.step = step_wrapper
    elif args.sample_solver == "pusa":
        # Choose scheduler based on whether V2V conditioning is being used
        if hasattr(args, 'pusa_use_v2v_scheduler') and args.pusa_use_v2v_scheduler:
            logger.info("Using Pusa V2V scheduler for multi-frame conditioning")
            scheduler = FlowMatchSchedulerPusaV2V(
                num_train_timesteps=config.num_train_timesteps,
                shift=args.flow_shift,
                extra_one_step=True
            )
        else:
            scheduler = FlowMatchSchedulerPusa(
                num_train_timesteps=config.num_train_timesteps, 
                shift=args.flow_shift,
                extra_one_step=True  # Following ComfyUI Pusa implementation
            )
        
        scheduler.set_timesteps(
            num_inference_steps=args.infer_steps,
            device=device,
            shift=args.flow_shift
        )
        timesteps = scheduler.timesteps
    else:
        raise NotImplementedError(f"Unsupported solver: {args.sample_solver}")

    logger.info(f"Using scheduler: {args.sample_solver}, timesteps shape: {timesteps.shape}")
    return scheduler, timesteps


def run_sampling(
    model: WanModel,
    noise: torch.Tensor, # This might be pure noise (T2V/I2V) or mixed noise+latent (V2V)
    scheduler: Any,
    timesteps: torch.Tensor, # Might be a subset for V2V
    args: argparse.Namespace,
    inputs: Tuple[dict, dict],
    device: torch.device,
    seed_g: torch.Generator,
    accelerator: Accelerator,
    is_ti2v: bool = False, # Flag for TI2V (Text+Image-to-Video) mode
    model_manager: Optional[DynamicModelManager] = None,  # Dynamic model manager for dual-dit
    previewer: Optional[LatentPreviewer] = None, # Add previewer argument
    use_cpu_offload: bool = True, # Example parameter, adjust as needed
    preview_suffix: Optional[str] = None, # <<< ADD suffix argument
    model_options: Optional[dict] = None  # Context windows and other model options
) -> torch.Tensor:
    """run sampling loop (Denoising)
    Args:
        model: dit model
        noise: initial latent state (pure noise or mixed noise/video latent)
        scheduler: scheduler for sampling
        timesteps: time steps for sampling (can be subset for V2V)
        args: command line arguments
        inputs: model input dictionaries (arg_c, arg_null) containing context etc.
        device: device to use
        seed_g: random generator
        accelerator: Accelerator instance
        previewer: LatentPreviewer instance or None # Added description
        use_cpu_offload: Whether to offload tensors to CPU during processing (example)
        preview_suffix: Unique suffix for preview files to avoid conflicts in concurrent runs.
        model_options: Optional model options including context window handler
    Returns:
        torch.Tensor: generated latent
    """
    arg_c, arg_null = inputs

    latent = noise # Initialize latent state
    
    # Check if we should use context windows
    context_handler = None
    if model_options and "context_handler" in model_options:
        context_handler = model_options["context_handler"]
        if context_handler and context_handler.should_use_context(model, [], latent, torch.tensor(0), model_options):
            logger.info("Context windows will be used for this generation")
        else:
            context_handler = None  # Don't use if not needed
    # Determine storage device (CPU if offloading, otherwise compute device)
    latent_storage_device = torch.device("cpu") if use_cpu_offload else device
    latent = latent.to(latent_storage_device) # Move initial state to storage device

    # cfg skip logic
    apply_cfg_array = []
    num_timesteps = len(timesteps)

    # ... (keep existing cfg skip logic) ...
    if args.cfg_skip_mode != "none" and args.cfg_apply_ratio is not None:
        # Calculate thresholds based on cfg_apply_ratio
        apply_steps = int(num_timesteps * args.cfg_apply_ratio)

        if args.cfg_skip_mode == "early":
            start_index = num_timesteps - apply_steps; end_index = num_timesteps
        elif args.cfg_skip_mode == "late":
            start_index = 0; end_index = apply_steps
        elif args.cfg_skip_mode == "early_late":
            start_index = (num_timesteps - apply_steps) // 2; end_index = start_index + apply_steps
        elif args.cfg_skip_mode == "middle":
            skip_steps = num_timesteps - apply_steps
            middle_start = (num_timesteps - skip_steps) // 2; middle_end = middle_start + skip_steps
        else: # Includes "alternate" - handled inside loop
             start_index = 0; end_index = num_timesteps # Default range for alternate

        w = 0.0 # For alternate mode
        for step_idx in range(num_timesteps):
            apply = True # Default
            if args.cfg_skip_mode == "alternate":
                w += args.cfg_apply_ratio; apply = w >= 1.0
                if apply: w -= 1.0
            elif args.cfg_skip_mode == "middle":
                apply = not (step_idx >= middle_start and step_idx < middle_end)
            elif args.cfg_skip_mode != "none": # early, late, early_late
                apply = step_idx >= start_index and step_idx < end_index

            apply_cfg_array.append(apply)

        pattern = ["A" if apply else "S" for apply in apply_cfg_array]
        pattern = "".join(pattern)
        logger.info(f"CFG skip mode: {args.cfg_skip_mode}, apply ratio: {args.cfg_apply_ratio}, steps: {num_timesteps}, pattern: {pattern}")
    else:
        # Apply CFG on all steps
        apply_cfg_array = [True] * num_timesteps

    # SLG (Skip Layer Guidance) setup
    apply_slg_global = args.slg_layers is not None and args.slg_mode is not None
    slg_start_step = int(args.slg_start * num_timesteps)
    slg_end_step = int(args.slg_end * num_timesteps)

    logger.info(f"Starting sampling loop for {num_timesteps} steps.")
    for i, t in enumerate(tqdm(timesteps)):
        # Prepare input for the model (move latent to compute device)
        # Latent should be [B, C, F, H, W] or [C, F, H, W]
        latent_on_device = latent.to(device)

        # FIX: Check if latent_on_device has too many dimensions and fix it
        # The model expects input x as a list of tensors with shape [C, F, H, W]
        # This adjustment seems specific to a potential bug elsewhere, keep it if needed.
        if len(latent_on_device.shape) > 5:
            while len(latent_on_device.shape) > 5:
                latent_on_device = latent_on_device.squeeze(0)
            logger.debug(f"Adjusted latent shape for model input: {latent_on_device.shape}")

        # The model expects the latent input 'x' as a list: [tensor]
        # If batch dimension is present, we need to split the tensor into a list of tensors
        if len(latent_on_device.shape) == 5:
            # Has batch dimension [B, C, F, H, W]
            latent_model_input_list = [latent_on_device[i] for i in range(latent_on_device.shape[0])]
        elif len(latent_on_device.shape) == 4:
            # No batch dimension [C, F, H, W]
            latent_model_input_list = [latent_on_device]
        else:
            # Handle unexpected shape
            raise ValueError(f"Latent tensor has unexpected shape {latent_on_device.shape} for model input.")

        # Prepare timestep - use official TI2V timestep processing if applicable
        if is_ti2v and "_ti2v_mask2" in inputs[0]:
            # Official TI2V timestep processing with mask-based spatial-temporal modulation
            # This is critical for proper image conditioning
            timestep_base = torch.stack([t]).to(device)
            
            # Get sequence length from args
            seq_len = inputs[0]["seq_len"]
            
            # Use stored masks from official implementation
            ti2v_mask2 = inputs[0]["_ti2v_mask2"]
            
            # Official timestep processing: temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
            # mask2[0] is the first tensor in the mask list, [0] is the first channel
            temp_ts = (ti2v_mask2[0][0][:, ::2, ::2] * timestep_base).flatten()
            temp_ts = torch.cat([
                temp_ts,
                temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep_base
            ])
            # TEMPORARY: Use standard timestep while we debug the expanded timestep issue
            # TODO: Fix the tensor dimension mismatch when using expanded timesteps
            timestep = timestep_base  # Keep original timestep [1] for now
        else:
            # Standard timestep for T2V, I2V, V2V
            timestep = torch.stack([t]).to(device) # Ensure timestep is a tensor on device

        with accelerator.autocast(), torch.no_grad():
            # --- Select appropriate model for dual-dit architectures ---
            if model_manager is not None:  # Always true for dual-dit models now
                # Dynamic loading mode
                cfg = WAN_CONFIGS[args.task]
                # Force low noise model for V2V if requested
                if hasattr(args, 'v2v_low_noise_only') and args.v2v_low_noise_only:
                    current_model = model_manager.get_model('low')
                else:
                    # Normal boundary logic
                    if args.dual_dit_boundary is not None:
                        boundary = args.dual_dit_boundary * 1000  # Custom boundary
                    else:
                        boundary = cfg.boundary * 1000  # Default: 0.875 * 1000 = 875 for t2v-A14B, 0.900 * 1000 = 900 for i2v-A14B
                    
                    if t.item() >= boundary:
                        current_model = model_manager.get_model('high')
                    else:
                        current_model = model_manager.get_model('low')
            else:
                # Single model mode
                current_model = model
            
            # --- (Keep existing prediction logic: cond, uncond, slg, cfg) ---
            # Define helper function for model calls with optional context windows
            def calc_cond_batch(model_to_use, conds_list, x_input, ts, opts):
                """Helper to calculate conditional predictions, optionally with context windows."""
                # Debug what we're receiving
                logger.debug(f"calc_cond_batch received conds_list type: {type(conds_list)}")
                logger.debug(f"calc_cond_batch x_input shape: {x_input.shape if isinstance(x_input, torch.Tensor) else type(x_input)}")
                if isinstance(conds_list, list):
                    logger.debug(f"  Length: {len(conds_list)}")
                    if len(conds_list) > 0:
                        logger.debug(f"  First element type: {type(conds_list[0])}")
                        if isinstance(conds_list[0], list) and len(conds_list[0]) > 0:
                            logger.debug(f"    First-first element type: {type(conds_list[0][0])}")
                
                # Handle different input formats
                # The context handler passes a list of resized condition dicts
                # We need to extract the actual condition dict
                cond_dict = {}
                if isinstance(conds_list, dict):
                    cond_dict = conds_list
                elif isinstance(conds_list, list) and len(conds_list) > 0:
                    # Unwrap nested lists until we find a dict
                    current = conds_list[0]
                    while isinstance(current, list) and len(current) > 0:
                        current = current[0]
                    if isinstance(current, dict):
                        cond_dict = current
                    else:
                        logger.error(f"Could not find dict in conds_list structure")
                        cond_dict = {}
                
                # The model needs context and seq_len - these should NOT be filtered out
                # Extract required arguments
                context = cond_dict.get('context')
                seq_len = cond_dict.get('seq_len')
                
                # Ensure seq_len is not None
                if seq_len is None:
                    logger.error(f"seq_len is None in calc_cond_batch - context window resizing may have failed")
                    logger.error(f"cond_dict keys: {cond_dict.keys()}")
                    raise ValueError("seq_len is None in calc_cond_batch - context window resizing failed")
                
                # Filter out context and seq_len from the dict to get other parameters
                model_cond_dict = {k: v for k, v in cond_dict.items() 
                                 if k not in ['context', 'seq_len'] and not k.startswith('_')}
                
                # Prepare input for model - it expects a list of tensors
                if isinstance(x_input, torch.Tensor):
                    if x_input.dim() == 5:  # [B, C, F, H, W]
                        # Split batch into list
                        x_input_list = [x_input[i] for i in range(x_input.shape[0])]
                    elif x_input.dim() == 4:  # [C, F, H, W]
                        x_input_list = [x_input]
                    else:
                        x_input_list = x_input if isinstance(x_input, list) else [x_input]
                else:
                    x_input_list = x_input
                
                # Call model with required positional arguments
                logger.debug(f"Calling model with x_input_list length: {len(x_input_list) if isinstance(x_input_list, list) else 'not list'}")
                if isinstance(x_input_list, list) and len(x_input_list) > 0:
                    logger.debug(f"  First tensor shape: {x_input_list[0].shape}")
                # Direct forward call (sequence parallel modifies the model's forward method)
                result = model_to_use(x_input_list, t=ts, context=context, seq_len=seq_len, **model_cond_dict)
                # Return just the tensor for WAN models (not wrapped in list)
                # The context handler expects a single tensor for single condition
                logger.debug(f"Model returned shape: {result[0].shape if isinstance(result, tuple) else result.shape if isinstance(result, torch.Tensor) else 'unknown'}")
                return result[0] if isinstance(result, tuple) else result
            
            # 1. Predict conditional noise estimate
            # Filter out non-model parameters before passing to model
            model_arg_c = {k: v for k, v in arg_c.items() if not k.startswith('_')}
            
            # Debug logging for seq_len
            if 'seq_len' in model_arg_c:
                logger.debug(f"Original seq_len before context windows: {model_arg_c['seq_len']}")
            
            if context_handler is not None:
                # Use context windows for processing
                # Prepare inputs for context handler
                conds_for_handler = [model_arg_c]  # Context handler expects list of dicts
                
                # Get the latent tensor for context window processing
                # Context handler will slice this into windows
                # Keep it without batch dimension for proper slicing
                if len(latent_model_input_list) == 1:
                    context_input = latent_model_input_list[0]  # [C, F, H, W]
                    # Don't add batch dimension - context handler handles windowing on frame dimension
                else:
                    # Stack batch elements [B x [C, F, H, W]] -> [B, C, F, H, W]
                    context_input = torch.stack(latent_model_input_list)
                
                # Execute with context windows
                noise_pred_results = context_handler.execute(
                    calc_cond_batch,
                    current_model,
                    [conds_for_handler],  # List of condition lists
                    context_input,
                    timestep,
                    model_options or {}
                )
                noise_pred_cond = noise_pred_results[0]
                # Squeeze batch dimension if present (context windows may add it)
                if noise_pred_cond.dim() == 5 and noise_pred_cond.shape[0] == 1:
                    noise_pred_cond = noise_pred_cond.squeeze(0)
                    logger.debug(f"Squeezed batch dimension from noise_pred_cond, new shape: {noise_pred_cond.shape}")
            else:
                # Standard model call without context windows
                # Direct forward call (sequence parallel modifies the model's forward method)
                noise_pred_cond = current_model(latent_model_input_list, t=timestep, **model_arg_c)[0]
            
            # Move result to storage device early if offloading to potentially save VRAM during uncond/slg pred
            noise_pred_cond = noise_pred_cond.to(latent_storage_device)

            # 2. Predict unconditional noise estimate (potentially with SLG)
            apply_cfg = apply_cfg_array[i]
            if apply_cfg:
                apply_slg_step = apply_slg_global and (i >= slg_start_step and i < slg_end_step)
                slg_indices_for_call = args.slg_layers if apply_slg_step else None
                # Filter out non-model parameters for uncond args too
                model_arg_null = {k: v for k, v in arg_null.items() if not k.startswith('_')}

                if context_handler is not None:
                    # Use context windows for unconditional predictions
                    conds_null_for_handler = [model_arg_null]
                    
                    # Prepare context input - keep consistent with conditional prediction
                    # Don't add batch dimension for proper windowing
                    if len(latent_model_input_list) == 1:
                        context_input = latent_model_input_list[0]  # [C, F, H, W]
                    else:
                        context_input = torch.stack(latent_model_input_list)  # [B, C, F, H, W]
                    
                    if apply_slg_step and args.slg_mode == "original":
                        # Uncond prediction
                        noise_pred_uncond_results = context_handler.execute(
                            calc_cond_batch, current_model, [conds_null_for_handler],
                            context_input,
                            timestep, model_options or {}
                        )
                        noise_pred_uncond = noise_pred_uncond_results[0]
                        # Squeeze batch dimension if present
                        if noise_pred_uncond.dim() == 5 and noise_pred_uncond.shape[0] == 1:
                            noise_pred_uncond = noise_pred_uncond.squeeze(0)
                        noise_pred_uncond = noise_pred_uncond.to(latent_storage_device)
                        
                        # SLG prediction (with skip layers)
                        def calc_slg_batch(model_to_use, conds_list, x_input, ts, opts):
                            # Handle different input formats
                            if isinstance(conds_list, dict):
                                cond_dict = conds_list
                            elif isinstance(conds_list, list) and len(conds_list) > 0:
                                cond_dict = conds_list[0] if isinstance(conds_list[0], dict) else {}
                            else:
                                cond_dict = {}
                            # Extract required arguments
                            context = cond_dict.get('context')
                            seq_len = cond_dict.get('seq_len')
                            # Ensure seq_len is not None
                            if seq_len is None:
                                logger.error(f"seq_len is None in calc_slg_batch")
                                raise ValueError("seq_len is None in calc_slg_batch")
                            # Filter for other parameters
                            model_cond_dict = {k: v for k, v in cond_dict.items() 
                                             if k not in ['context', 'seq_len'] and not k.startswith('_')}
                            # Prepare input for model
                            if isinstance(x_input, torch.Tensor):
                                if x_input.dim() == 5:  # [B, C, F, H, W]
                                    x_input_list = [x_input[i] for i in range(x_input.shape[0])]
                                elif x_input.dim() == 4:  # [C, F, H, W]
                                    x_input_list = [x_input]
                                else:
                                    x_input_list = x_input if isinstance(x_input, list) else [x_input]
                            else:
                                x_input_list = x_input
                            result = model_to_use(x_input_list, t=ts, context=context, seq_len=seq_len, 
                                                skip_block_indices=slg_indices_for_call, **model_cond_dict)
                            return [result[0]] if isinstance(result, tuple) else [result]
                        
                        skip_layer_results = context_handler.execute(
                            calc_slg_batch, current_model, [conds_null_for_handler],
                            context_input,
                            timestep, model_options or {}
                        )
                        skip_layer_out = skip_layer_results[0].to(latent_storage_device)
                        
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        noise_pred = noise_pred + args.slg_scale * (noise_pred_cond - skip_layer_out)
                    
                    elif apply_slg_step and args.slg_mode == "uncond":
                        # Uncond with SLG
                        def calc_slg_batch(model_to_use, conds_list, x_input, ts, opts):
                            # Handle different input formats
                            if isinstance(conds_list, dict):
                                cond_dict = conds_list
                            elif isinstance(conds_list, list) and len(conds_list) > 0:
                                cond_dict = conds_list[0] if isinstance(conds_list[0], dict) else {}
                            else:
                                cond_dict = {}
                            # Extract required arguments
                            context = cond_dict.get('context')
                            seq_len = cond_dict.get('seq_len')
                            # Ensure seq_len is not None
                            if seq_len is None:
                                logger.error(f"seq_len is None in calc_slg_batch")
                                raise ValueError("seq_len is None in calc_slg_batch")
                            # Filter for other parameters
                            model_cond_dict = {k: v for k, v in cond_dict.items() 
                                             if k not in ['context', 'seq_len'] and not k.startswith('_')}
                            # Prepare input for model
                            if isinstance(x_input, torch.Tensor):
                                if x_input.dim() == 5:  # [B, C, F, H, W]
                                    x_input_list = [x_input[i] for i in range(x_input.shape[0])]
                                elif x_input.dim() == 4:  # [C, F, H, W]
                                    x_input_list = [x_input]
                                else:
                                    x_input_list = x_input if isinstance(x_input, list) else [x_input]
                            else:
                                x_input_list = x_input
                            result = model_to_use(x_input_list, t=ts, context=context, seq_len=seq_len,
                                                skip_block_indices=slg_indices_for_call, **model_cond_dict)
                            return [result[0]] if isinstance(result, tuple) else [result]
                        
                        noise_pred_uncond_results = context_handler.execute(
                            calc_slg_batch, current_model, [conds_null_for_handler],
                            context_input,
                            timestep, model_options or {}
                        )
                        noise_pred_uncond = noise_pred_uncond_results[0]
                        # Squeeze batch dimension if present
                        if noise_pred_uncond.dim() == 5 and noise_pred_uncond.shape[0] == 1:
                            noise_pred_uncond = noise_pred_uncond.squeeze(0)
                        noise_pred_uncond = noise_pred_uncond.to(latent_storage_device)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    else:  # Regular CFG with context windows
                        noise_pred_uncond_results = context_handler.execute(
                            calc_cond_batch, current_model, [conds_null_for_handler],
                            context_input,
                            timestep, model_options or {}
                        )
                        noise_pred_uncond = noise_pred_uncond_results[0]
                        # Squeeze batch dimension if present
                        if noise_pred_uncond.dim() == 5 and noise_pred_uncond.shape[0] == 1:
                            noise_pred_uncond = noise_pred_uncond.squeeze(0)
                        noise_pred_uncond = noise_pred_uncond.to(latent_storage_device)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    # Standard calls without context windows (original code)
                    if apply_slg_step and args.slg_mode == "original":
                        noise_pred_uncond = current_model(latent_model_input_list, t=timestep, **model_arg_null)[0].to(latent_storage_device)
                        skip_layer_out = current_model(latent_model_input_list, t=timestep, skip_block_indices=slg_indices_for_call, **model_arg_null)[0].to(latent_storage_device)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        noise_pred = noise_pred + args.slg_scale * (noise_pred_cond - skip_layer_out)

                    elif apply_slg_step and args.slg_mode == "uncond":
                        noise_pred_uncond = current_model(latent_model_input_list, t=timestep, skip_block_indices=slg_indices_for_call, **model_arg_null)[0].to(latent_storage_device)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    else: # Regular CFG
                        noise_pred_uncond = current_model(latent_model_input_list, t=timestep, **model_arg_null)[0].to(latent_storage_device)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # CFG is skipped, use conditional prediction directly
                noise_pred = noise_pred_cond
            # --- End prediction logic ---

            # 3. Compute previous sample state with the scheduler
            # Ensure noise_pred and latent_on_device have matching batch dimensions for scheduler
            if len(noise_pred.shape) < len(latent_on_device.shape):
                 noise_pred = noise_pred.unsqueeze(0) # Add batch dim if missing ([C,F,H,W]->[1,C,F,H,W])
            elif len(noise_pred.shape) > len(latent_on_device.shape):
                 # This shouldn't happen if latent_on_device handles batch correctly
                 logger.warning(f"Noise pred shape {noise_pred.shape} has more dims than latent {latent_on_device.shape}")

            # Scheduler expects noise_pred [B, C, F, H, W] and sample [B, C, F, H, W]
            # latent_on_device should already have the batch dim handled by the logic above
            
            # Prepare V2V conditioning parameters for Pusa scheduler
            scheduler_kwargs = {
                "return_dict": False,
                "generator": seed_g
            }
            
            # Add V2V conditioning if using Pusa V2V scheduler
            if (hasattr(args, 'pusa_use_v2v_scheduler') and args.pusa_use_v2v_scheduler and 
                hasattr(args, 'pusa_cond_positions') and hasattr(args, 'pusa_frame_noise_mapping')):
                scheduler_kwargs.update({
                    "cond_frame_latent_indices": args.pusa_cond_positions,
                    "noise_multipliers": args.pusa_frame_noise_mapping
                })
                logger.debug(f"Step {i}: Applying V2V conditioning to positions {args.pusa_cond_positions}")
                
                # Create 2D timestep tensor for V2V scheduler (expects shape [B, F])
                # Convert scalar timestep to frame-aware tensor
                num_frames = latent_on_device.shape[2] if len(latent_on_device.shape) == 5 else latent_on_device.shape[1]
                # Ensure t is a tensor (sometimes it might be extracted as a Python scalar)
                if not isinstance(t, torch.Tensor):
                    t_tensor = torch.tensor(t, device=device)
                else:
                    t_tensor = t.to(device)
                timestep_2d = t_tensor.unsqueeze(0).unsqueeze(1).repeat(1, num_frames)
                
                # Apply frame-specific timestep modifications for conditioning frames
                for frame_idx in args.pusa_cond_positions:
                    if frame_idx < num_frames:
                        noise_mult = args.pusa_frame_noise_mapping.get(frame_idx, 1.0)
                        timestep_2d[:, frame_idx] = timestep_2d[:, frame_idx] * noise_mult
                
                # Use 2D timestep for V2V scheduler
                timestep_for_scheduler = timestep_2d
            else:
                # Use scalar timestep for regular schedulers
                timestep_for_scheduler = t
            
            scheduler_output = scheduler.step(
                noise_pred.to(device), # Ensure noise_pred is on compute device for step
                timestep_for_scheduler,
                latent_on_device, # Pass the tensor (with batch dim) on compute device
                **scheduler_kwargs
            )
            prev_latent = scheduler_output[0] # Get the new latent state [B, C, F, H, W]

            # 4. Update latent state (move back to storage device)
            latent = prev_latent.to(latent_storage_device)
            
            # 5. Apply image conditioning for TI2V (if image_latent is available)
            # CRITICAL: This must happen after EVERY timestep, not just once - following official implementation
            if "_image_latent" in arg_c and "_ti2v_mask2" in arg_c:
                image_latent = arg_c["_image_latent"].to(latent_storage_device)
                ti2v_mask2 = arg_c["_ti2v_mask2"]
                
                # Apply mask-based conditioning using stored masks: latent = (1. - mask2[0]) * z[0] + mask2[0] * latent
                # This matches the official implementation exactly
                latent = (1. - ti2v_mask2[0]) * image_latent + ti2v_mask2[0] * latent
            
            # 6. Apply Pusa multi-frame conditioning (inject conditioning frames)
            if (hasattr(args, 'pusa_conditioning_dict') and args.pusa_conditioning_dict and 
                hasattr(args, 'pusa_frame_noise_mapping')):
                logger.debug(f"Step {i}: Applying Pusa frame conditioning")
                
                # Get current timestep value for noise calculation
                current_t = t.item() if hasattr(t, 'item') else t
                sigma_ratio = current_t / 1000.0  # Convert timestep to sigma ratio
                
                # Apply conditioning to specific frames
                for frame_idx, (cond_image, noise_mult) in args.pusa_conditioning_dict.items():
                    if frame_idx < latent.shape[2]:  # Ensure frame index is valid
                        # Convert conditioning image to latent space if needed
                        if not hasattr(args, '_pusa_cond_latents_cache'):
                            args._pusa_cond_latents_cache = {}
                        
                        if frame_idx not in args._pusa_cond_latents_cache:
                            # This is a placeholder - in a real implementation, you'd encode the image to latent space
                            # For now, we'll apply a simple noise-based conditioning
                            logger.debug(f"Caching conditioning latent for frame {frame_idx}")
                            # TODO: Add VAE encoding of conditioning images to latent space
                            args._pusa_cond_latents_cache[frame_idx] = None
                        
                        # Apply conditioning based on noise multiplier and current timestep
                        if noise_mult < 0.1:  # Clean conditioning (noise_mult ~ 0.0)
                            logger.debug(f"Applying clean conditioning to frame {frame_idx}")
                            # For clean conditioning, we would inject the clean latent
                            # This is simplified - real implementation would use encoded image latents
                        else:
                            # Noisy conditioning - apply partial conditioning based on timestep
                            conditioning_strength = 1.0 - (sigma_ratio * noise_mult)
                            conditioning_strength = max(0.0, min(1.0, conditioning_strength))
                            logger.debug(f"Applying noisy conditioning to frame {frame_idx}, strength: {conditioning_strength:.3f}")

            # --- Latent Preview Call ---
            # Preview the state *after* step 'i' is completed
            if previewer is not None and (i + 1) % args.preview == 0 and (i + 1) < num_timesteps:
                 try:
                      #logger.debug(f"Generating preview for step {i + 1}")
                      # Pass the *resulting* latent from this step (prev_latent).
                      # Ensure it's on the compute device for the previewer call.
                      # LatentPreviewer handles internal device management.
                      # Need to pass without batch dim if previewer expects [C, F, H, W]
                      # Check LatentPreviewer.preview expects [C, F, H, W]
                      if len(prev_latent.shape) == 5:
                          preview_latent_input = prev_latent.squeeze(0) # Remove batch dim
                      else:
                          preview_latent_input = prev_latent # Assume already [C, F, H, W]

                      # Pass the latent on the main compute device
                      #print(f"DEBUG run_sampling: Step {i}, prev_latent shape: {prev_latent.shape}, preview_latent_input shape: {preview_latent_input.shape}")
                      previewer.preview(preview_latent_input.to(device), i, preview_suffix=preview_suffix) # Pass 0-based index 'i'
                 except Exception as e:
                      logger.error(f"Error during latent preview generation at step {i + 1}: {e}", exc_info=True)
                      # Optional: Disable previewer after first error to avoid repeated logs/errors
                      # logger.warning("Disabling latent preview due to error.")
                      # previewer = None

    # Return the final denoised latent (should be on storage device)
    logger.info("Sampling loop finished.")
    return latent

def prepare_video_extension_inputs(
    args: argparse.Namespace,
    config,
    accelerator: Accelerator,
    device: torch.device,
    vae: WanVAE,
    cond_frames: torch.Tensor,  # Last frames from previous generation [C, F, H, W]
    clip_context: torch.Tensor,  # CLIP encoding of first frame
    frame_num: int,  # Number of frames to generate
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
    """Prepare inputs for video extension (multitalk-style iterative generation)
    
    Args:
        args: command line arguments
        config: model configuration
        accelerator: Accelerator instance
        device: device to use
        vae: VAE model instance
        cond_frames: Conditioning frames from previous generation [C, F, H, W]
        clip_context: CLIP context from first frame
        frame_num: Number of frames to generate in this chunk
        
    Returns:
        Tuple containing noise, context, context_null, y, (arg_c, arg_null)
    """
    logger.info(f"Preparing video extension inputs for {frame_num} frames")
    
    # Get dimensions
    _, cond_f, pixel_h, pixel_w = cond_frames.shape
    lat_h = pixel_h // config.vae_stride[1]
    lat_w = pixel_w // config.vae_stride[2]
    lat_f = (frame_num - 1) // config.vae_stride[0] + 1
    
    # Calculate sequence length
    seq_len = lat_f * lat_h * lat_w // (config.patch_size[1] * config.patch_size[2])
    
    # Configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt
    
    # Set seed
    seed = args.seed
    if not args.cpu_noise:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
    else:
        seed_g = torch.manual_seed(seed)
    
    # Generate noise for new frames
    noise = torch.randn(
        16, lat_f, lat_h, lat_w,
        dtype=torch.float32, generator=seed_g,
        device=device if not args.cpu_noise else "cpu"
    )
    noise = noise.to(device)
    
    # Load text encoder and encode prompts
    text_encoder = load_text_encoder(args, config, device)
    text_encoder.model.to(device)
    
    # Generate three contexts for proper CFG (multitalk-style)
    with torch.no_grad():
        if args.fp8_t5:
            with torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype):
                context = text_encoder([args.prompt], device)  # Full conditional
                context_text_dropped = text_encoder([""], device)  # Text-dropped (empty prompt)
                context_null = text_encoder([n_prompt], device)  # Unconditional (negative prompt)
        else:
            context = text_encoder([args.prompt], device)  # Full conditional
            context_text_dropped = text_encoder([""], device)  # Text-dropped (empty prompt)
            context_null = text_encoder([n_prompt], device)  # Unconditional (negative prompt)
    
    # Free text encoder
    del text_encoder
    clean_memory_on_device(device)
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Unloaded T5 model from memory")
    
    # Prepare zero padding for future frames
    padding_frames = frame_num - cond_f
    if padding_frames > 0:
        padding_tensor = torch.zeros(
            cond_frames.shape[0], padding_frames, pixel_h, pixel_w,
            device=device, dtype=cond_frames.dtype
        )
        padded_frames = torch.cat([cond_frames, padding_tensor], dim=1)
    else:
        padded_frames = cond_frames[:, :frame_num]
    
    # Encode conditioning frames with VAE
    vae.to_device(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vae.dtype):
        y_latent = vae.encode([padded_frames])[0]  # [C', lat_f, lat_h, lat_w]
    
    # Create mask for conditioning frames
    motion_frames_latent_num = (cond_f - 1) // config.vae_stride[0] + 1
    msk = torch.zeros(4, lat_f, lat_h, lat_w, device=device, dtype=vae.dtype)
    msk[:, :motion_frames_latent_num] = 1  # Mask the conditioning frames
    
    # Concatenate mask and latent
    y = torch.cat([msk, y_latent], dim=0)  # [4+C', lat_f, lat_h, lat_w]
    
    # Move VAE back to CPU/cache
    vae.to_device(args.vae_cache_cpu if args.vae_cache_cpu else "cpu")
    clean_memory_on_device(device)
    
    # Prepare model arguments
    # A14B models don't have img_emb layer
    use_clip_fea = clip_context if not ("A14B" in args.task) else None
    
    arg_c = {
        "context": context,
        "clip_fea": use_clip_fea,
        "seq_len": seq_len,
        "y": [y],
    }
    
    arg_text_dropped = {
        "context": context_text_dropped,
        "clip_fea": use_clip_fea,  # Keep CLIP for text-dropped
        "seq_len": seq_len,
        "y": [y],
    }
    
    arg_null = {
        "context": context_null,
        "clip_fea": use_clip_fea,  # Keep CLIP for unconditional too
        "seq_len": seq_len,
        "y": [y],
    }
    
    return noise, context, context_null, y, (arg_c, arg_text_dropped, arg_null)

def timestep_transform(
    t: torch.Tensor,
    shift: float = 5.0,
    num_timesteps: int = 1000,
) -> torch.Tensor:
    """Transform timesteps with shift parameter for better temporal dynamics"""
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t

def add_noise_for_extension(
    original_samples: torch.FloatTensor,
    noise: torch.FloatTensor,
    timestep: torch.FloatTensor,
    num_timesteps: int = 1000
) -> torch.FloatTensor:
    """Add noise using the MultiTalk approach (linear interpolation)"""
    # Critical: Use timestep VALUE, not index
    timesteps = timestep.float() / num_timesteps
    if len(timesteps.shape) == 0:
        timesteps = timesteps.unsqueeze(0)
    timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape)-1))
    return (1 - timesteps) * original_samples + timesteps * noise

def run_extension_sampling(
    model: WanModel,
    noise: torch.Tensor,
    scheduler: Any,
    timesteps: torch.Tensor,
    args: argparse.Namespace,
    inputs: Tuple[dict, dict, dict],
    device: torch.device,
    seed_g: torch.Generator,
    accelerator: Accelerator,
    motion_latent: torch.Tensor,
    motion_frames: int,
    model_manager: Optional[DynamicModelManager] = None,
) -> torch.Tensor:
    """Run sampling loop for video extension with motion frame injection"""
    
    arg_c, arg_text_dropped, arg_null = inputs
    latent = noise
    
    # Calculate motion frame latent count
    motion_frames_latent_num = (motion_frames - 1) // 4 + 1  # VAE stride of 4 in temporal dimension
    
    logger.info(f"Starting extension sampling loop for {len(timesteps)} steps with {motion_frames} motion frames")
    
    # Inject motion frames at the beginning (standard approach)
    if motion_latent is not None:
        motion_add_noise = torch.randn_like(motion_latent).to(device)
        noised_motion = add_noise_for_extension(
            motion_latent,
            motion_add_noise, 
            timesteps[0],
            scheduler.config.num_train_timesteps if hasattr(scheduler.config, 'num_train_timesteps') else 1000
        )
        latent[:, :motion_frames_latent_num] = noised_motion[:, :motion_frames_latent_num]
        logger.info(f"Injected {motion_frames_latent_num} motion frames at timestep {timesteps[0]}")
    
    for i, t in enumerate(tqdm(timesteps)):
        
        # Prepare input for the model
        latent_on_device = latent.to(device)
        
        # The model expects the latent input 'x' as a list: [tensor]
        if len(latent_on_device.shape) == 5:
            # Has batch dimension [B, C, F, H, W]
            latent_model_input_list = [latent_on_device[i] for i in range(latent_on_device.shape[0])]
        elif len(latent_on_device.shape) == 4:
            # No batch dimension [C, F, H, W]
            latent_model_input_list = [latent_on_device]
        else:
            raise ValueError(f"Latent tensor has unexpected shape {latent_on_device.shape} for model input.")
        
        timestep = torch.stack([t]).to(device)
        
        with accelerator.autocast(), torch.no_grad():
            # Select appropriate model for dual-dit architectures
            if model_manager is not None:
                cfg = WAN_CONFIGS[args.task]
                if args.dual_dit_boundary is not None:
                    boundary = args.dual_dit_boundary * 1000
                else:
                    boundary = cfg.boundary * 1000
                
                if t.item() >= boundary:
                    current_model = model_manager.get_model('high')
                else:
                    current_model = model_manager.get_model('low')
            else:
                current_model = model
            
            # Filter out non-model parameters
            model_arg_c = {k: v for k, v in arg_c.items() if not k.startswith('_')}
            model_arg_text_dropped = {k: v for k, v in arg_text_dropped.items() if not k.startswith('_')}
            model_arg_null = {k: v for k, v in arg_null.items() if not k.startswith('_')}
            
            # Three-way CFG (multitalk-style)
            # Use pipeline forward if available
            # Direct forward calls (sequence parallel modifies the model's forward method)
            # 1. Full conditional (text + CLIP)
            noise_pred_cond = current_model(latent_model_input_list, t=timestep, **model_arg_c)[0]
            # 2. Text-dropped (CLIP only)
            noise_pred_text_dropped = current_model(latent_model_input_list, t=timestep, **model_arg_text_dropped)[0]
            # 3. Unconditional (negative prompt)
            noise_pred_uncond = current_model(latent_model_input_list, t=timestep, **model_arg_null)[0]
            
            # Apply three-way CFG formula (simplified from multitalk, no audio)
            # Formula: uncond + text_scale * (cond - text_dropped)
            # Since we don't have audio, we simplify the formula
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_text_dropped)
            
            # CRITICAL: Negate the noise prediction (from multitalk)
            noise_pred = -noise_pred
            
            # Ensure noise_pred and latent_on_device have matching batch dimensions
            if len(noise_pred.shape) < len(latent_on_device.shape):
                noise_pred = noise_pred.unsqueeze(0)
            
            # Manual latent update (from multitalk) instead of scheduler.step
            if i < len(timesteps) - 1:
                # Calculate dt based on timestep difference
                dt = (timesteps[i] - timesteps[i + 1]) / 1000.0  # Assuming num_timesteps = 1000
                dt = dt.view(dt.shape + (1,) * (len(noise_pred.shape) - 1))  # Reshape for broadcasting
                latent = latent_on_device + noise_pred * dt
            else:
                # Last step - just add the final noise prediction
                dt = timesteps[i] / 1000.0
                dt = dt.view(dt.shape + (1,) * (len(noise_pred.shape) - 1))
                latent = latent_on_device + noise_pred * dt
            
    logger.info("Extension sampling loop finished.")
    return latent

def variance_of_laplacian(image):
    """Calculate image sharpness using Laplacian variance"""
    import cv2
    return cv2.Laplacian(image, cv2.CV_64F).var()

def extract_best_transition_frame(video_path: str, frames_to_check: int = 30) -> int:
    """Extract the sharpest frame from the last N frames for smooth transition"""
    import cv2
    from tqdm import tqdm
    
    logger.info(f"Extracting best transition frame from last {frames_to_check} frames of {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video file")
        return -1
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, total_frames - frames_to_check)
    
    best_frame_idx = -1
    max_sharpness = -1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_idx in range(start_frame, total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate sharpness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = variance_of_laplacian(gray)
        
        if sharpness > max_sharpness:
            max_sharpness = sharpness
            best_frame_idx = frame_idx
    
    cap.release()
    
    logger.info(f"Best transition frame: {best_frame_idx} with sharpness {max_sharpness:.2f}")
    return best_frame_idx

def blend_video_transition(video1: torch.Tensor, video2: torch.Tensor, blend_frames: int = 8) -> torch.Tensor:
    """Simple concatenation of video segments without blending to avoid glitches"""
    # Simply concatenate the videos without any blending
    return torch.cat([video1, video2], dim=2)

def generate_extended_video_i2v_based(
    args: argparse.Namespace,
    initial_video_path: str,
    total_frames: int,
) -> torch.Tensor:
    """Generate extended video using clean i2v approach with smooth blending"""
    import tempfile
    import cv2
    
    device = torch.device(args.device)
    logger.info(f"Starting clean i2v-based video extension from {initial_video_path} to {total_frames} frames")
    
    # Extract the best transition frame
    best_frame_idx = extract_best_transition_frame(initial_video_path, frames_to_check=args.frames_to_check)
    
    # Load initial video up to the best frame
    if best_frame_idx > 0:
        video_frames_np, initial_frames = load_video(
            initial_video_path, 0, best_frame_idx + 1, bucket_reso=tuple(args.video_size)
        )
    else:
        # Use entire video as fallback
        video_frames_np, initial_frames = load_video(
            initial_video_path, 0, None, bucket_reso=tuple(args.video_size)
        )
    
    # Convert initial video to tensor [1, C, F, H, W]
    initial_video = torch.from_numpy(np.stack(video_frames_np, axis=0))
    initial_video = initial_video.permute(0, 3, 1, 2).float() / 255.0  # [F,C,H,W], [0,1]
    initial_video = initial_video.permute(1, 0, 2, 3).unsqueeze(0)  # [1,C,F,H,W]
    
    logger.info(f"Initial video loaded: {initial_video.shape[2]} frames")
    
    if initial_frames >= total_frames:
        logger.info("Video already has desired length")
        return initial_video[:, :, :total_frames]
    
    # Store original arguments
    original_image_path = args.image_path
    original_video_length = args.video_length
    original_extend_video = args.extend_video
    
    # Generate extension chunks using i2v
    all_videos = [initial_video]
    current_frames = initial_frames
    chunk_size = 81  # Standard i2v length
    
    # Get the best transition frame as starting image
    best_frame_tensor = initial_video[0, :, -1]  # [C, H, W] - last frame
    best_frame_np = best_frame_tensor.permute(1, 2, 0).cpu().numpy() * 255
    best_frame_np = best_frame_np.astype(np.uint8)
    
    try:
        while current_frames < total_frames:
            remaining_frames = total_frames - current_frames
            frames_to_generate = min(chunk_size, remaining_frames)
            
            logger.info(f"Generating chunk: {frames_to_generate} frames (progress: {current_frames}/{total_frames})")
            
            # Save the frame as temporary image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, cv2.cvtColor(best_frame_np, cv2.COLOR_RGB2BGR))
                temp_image_path = tmp_file.name
            
            # Modify args for i2v generation
            args.image_path = temp_image_path
            args.video_length = frames_to_generate
            args.extend_video = None  # Prevent recursive extension calls
            
            # Generate new chunk using the main generation function
            new_chunk = generate(args)
            
            # Clean up temp file
            os.unlink(temp_image_path)
            
            if new_chunk is not None:
                logger.info(f"Generated chunk shape: {new_chunk.shape}")
                # Decode the latent chunk to pixel space for blending
                decoded_chunk = decode_latent(new_chunk, args, WAN_CONFIGS[args.task])
                logger.info(f"Decoded chunk shape: {decoded_chunk.shape}")
                all_videos.append(decoded_chunk)
                current_frames += frames_to_generate
                
                # Use the last frame of the decoded chunk as the next starting frame
                best_frame_tensor = decoded_chunk[0, :, -1]
                best_frame_np = best_frame_tensor.permute(1, 2, 0).cpu().numpy() * 255
                best_frame_np = best_frame_np.astype(np.uint8)
            else:
                logger.error("Failed to generate video chunk")
                break
        
        # Blend all video segments smoothly
        logger.info(f"Blending {len(all_videos)} video segments")
        result = all_videos[0]
        
        for i in range(1, len(all_videos)):
            result = blend_video_transition(result, all_videos[i], blend_frames=8)
            logger.info(f"Blended segment {i+1}, current length: {result.shape[2]} frames")
        
        logger.info(f"Final extended video: {result.shape}")
        return result
        
    finally:
        # Restore original arguments
        args.image_path = original_image_path
        args.video_length = original_video_length
        args.extend_video = original_extend_video

def generate_extended_video(
    args: argparse.Namespace,
    initial_video_path: str,
    total_frames: int,
    motion_frames: int = 25,
) -> torch.Tensor:
    """Generate extended video using multitalk-style iterative generation
    
    Args:
        args: Command line arguments
        initial_video_path: Path to initial video to extend
        total_frames: Total number of frames to generate
        motion_frames: Number of frames to use for conditioning each chunk
        
    Returns:
        torch.Tensor: Extended video tensor [1, C, F, H, W]
    """
    device = torch.device(args.device)
    cfg = WAN_CONFIGS[args.task]
    
    # Create accelerator for autocast support
    accelerator = Accelerator()
    
    # Ensure we're using i2v-A14B model
    if args.task != "i2v-A14B":
        raise ValueError(f"Video extension requires i2v-A14B task, got {args.task}")
    
    # Load initial video
    logger.info(f"Loading initial video from {initial_video_path}")
    video_frames_np, initial_frames = load_video(
        initial_video_path, 0, None, bucket_reso=tuple(args.video_size)
    )
    
    if initial_frames < motion_frames:
        raise ValueError(f"Initial video has {initial_frames} frames, need at least {motion_frames}")
    
    # Convert to tensor [1, C, F, H, W]
    video_tensor = torch.from_numpy(np.stack(video_frames_np, axis=0))
    video_tensor = video_tensor.permute(0, 3, 1, 2).float() / 255.0  # [F,C,H,W], [0,1]
    video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1,C,F,H,W]
    
    # Get dimensions
    _, _, _, height, width = video_tensor.shape
    
    # Load models
    vae_dtype = torch.float16
    vae = load_vae(args, cfg, device, vae_dtype)
    
    # Load CLIP and encode first frame
    clip = load_clip_model(args, cfg, device)
    clip.model.to(device)
    first_frame = video_tensor[0, :, 0]  # [C, H, W]
    # Avoid in-place modification - use explicit operation
    first_frame_normalized = (first_frame - 0.5) / 0.5  # [-1, 1]
    # Convert to correct dtype and device for CLIP
    first_frame_normalized = first_frame_normalized.to(device=device, dtype=torch.float16)
    
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
        clip_context = clip.visual([first_frame_normalized.unsqueeze(1)])
    
    del clip
    clean_memory_on_device(device)
    
    # Configure model selection for video extension
    if args.extension_dual_dit_boundary is not None:
        # Use custom boundary for extension with BOTH models
        args.dual_dit_boundary = args.extension_dual_dit_boundary
        logger.info(f"Using custom extension dual-dit boundary: {args.extension_dual_dit_boundary} (both models will be used)")
    elif args.force_high_noise:
        # Force high noise model ONLY (boundary = 0.0 means high noise model used for all timesteps)
        args.dual_dit_boundary = 0.0
        logger.info("Forcing high noise model ONLY for video extension")
    elif args.force_low_noise:
        # Force low noise model ONLY (boundary = 1.0 means low noise model used for all timesteps)
        args.dual_dit_boundary = 1.0
        logger.info("Forcing low noise model ONLY for video extension")
    else:
        # Default: use both models with default boundary (better for quality)
        args.dual_dit_boundary = cfg.boundary  # Use default i2v-A14B boundary (0.9)
        logger.info(f"Using default dual-dit boundary for extension: {cfg.boundary} (both models will be used)")
    
    model_result = load_dit_model(args, cfg, device, torch.float16, torch.float16, True)
    
    # Handle dual-dit dynamic loading - DO NOT force single model selection for extension
    is_dual_dit = isinstance(model_result, tuple)
    if is_dual_dit:
        model_low_path, model_high_path, *lora_weights = model_result
        # Select appropriate model manager based on configuration
        if args.dit_fsdp or args.t5_fsdp:
            # Use FSDP manager for memory-efficient distributed training
            if not dist.is_initialized() and torch.cuda.device_count() < 2:
                logger.warning(f"FSDP requested but only {torch.cuda.device_count()} GPU(s) available. Falling back to single GPU.")
                model_manager = DynamicModelManager(cfg, device, torch.float16, torch.float16, args)
            else:
                logger.info(f"Using FSDP Model Manager for extension")
                model_manager = FSDPModelManager(cfg, device, torch.float16, torch.float16, args)
                # Use the rank-specific device from the model manager
                device = model_manager.device
        elif args.use_sequence_parallel or args.use_dual_gpu:
            if args.use_dual_gpu:
                logger.warning("--use_dual_gpu is deprecated. Using --use_sequence_parallel instead.")
            if not dist.is_initialized() and torch.cuda.device_count() < 2:
                logger.warning(f"Sequence parallelism requested but only {torch.cuda.device_count()} GPU(s) available. Falling back to single GPU.")
                model_manager = DynamicModelManager(cfg, device, torch.float16, torch.float16, args)
            else:
                logger.info(f"Using Sequence Parallel Model Manager for extension")
                model_manager = SequenceParallelModelManager(cfg, device, torch.float16, torch.float16, args)
        else:
            model_manager = DynamicModelManager(cfg, device, torch.float16, torch.float16, args)
        model_manager.set_model_paths(model_low_path, model_high_path)
        if len(lora_weights) >= 4:
            model_manager.set_lora_weights(*lora_weights)
        # For extension, start with high noise model but allow dynamic switching
        model = model_manager.get_model('high')  # Initial model, will switch dynamically
    else:
        model = model_result
    
    optimize_model(model, args, device, torch.float16, torch.float16)
    
    # Setup scheduler
    scheduler, _ = setup_scheduler(args, cfg, device)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(args.seed if args.seed else 42)
    
    # Initialize with initial video
    all_frames = video_tensor.squeeze(0).permute(1, 0, 2, 3)  # [F, C, H, W]
    all_frames = (all_frames * 2.0 - 1.0)  # Scale to [-1, 1], keep on CPU to save GPU memory
    
    # Compute color statistics of initial video for normalization
    initial_mean = all_frames.mean(dim=(0, 2, 3), keepdim=True)  # [1, C, 1, 1]
    initial_std = all_frames.std(dim=(0, 2, 3), keepdim=True)  # [1, C, 1, 1]
    logger.info(f"Initial video statistics - Mean: {initial_mean.squeeze().tolist()}, Std: {initial_std.squeeze().tolist()}")
    
    # Generate chunks iteratively
    frames_per_chunk = args.video_length if args.video_length else 81
    current_frame = initial_frames
    
    # Start with initial video, convert to [C, F, H, W] format to match other chunks
    initial_chunk = all_frames[:initial_frames].permute(1, 0, 2, 3)  # [C, F, H, W]
    generated_chunks = [initial_chunk]  # Start with initial video
    
    while current_frame < total_frames:
        logger.info(f"Generating chunk: frames {current_frame} to {min(current_frame + frames_per_chunk, total_frames)}")
        
        # Get conditioning frames (last motion_frames from previous generation)
        cond_frames = all_frames[-motion_frames:].clone()  # [F, C, H, W]
        cond_frames = cond_frames.permute(1, 0, 2, 3).to(device)  # [C, F, H, W], move to device
        
        # Encode conditioning frames to latent
        vae.to_device(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vae.dtype):
            cond_latent = vae.encode([cond_frames])[0]
        
        # Prepare inputs for this chunk
        chunk_frames = min(frames_per_chunk, total_frames - current_frame + motion_frames)
        noise, context, context_null, y, inputs = prepare_video_extension_inputs(
            args, cfg, None, device, vae,
            cond_frames, clip_context, chunk_frames
        )
        
        # Get motion frames latent for injection
        motion_latent = cond_latent[:, :(motion_frames-1)//cfg.vae_stride[0]+1]
        
        # Run sampling with motion frame injection
        scheduler.set_timesteps(args.infer_steps, device=device)
        timesteps = scheduler.timesteps
        
        # Apply timestep transformation for better temporal dynamics
        # Use flow_shift parameter (default 5.0, or 3.0 for 480p as per multitalk)
        shift = args.flow_shift if args.flow_shift else (3.0 if "480" in str(args.video_size) else 5.0)
        transformed_timesteps = []
        for t in timesteps:
            transformed_t = timestep_transform(t, shift=shift, num_timesteps=1000)
            transformed_timesteps.append(transformed_t)
        timesteps = torch.stack(transformed_timesteps) if transformed_timesteps else timesteps
        
        final_latent = run_extension_sampling(
            model, noise, scheduler, timesteps, args, inputs,
            device, seed_g, accelerator,
            motion_latent, motion_frames,
            model_manager=model_manager
        )
        
        # Decode the generated latent
        vae.to_device(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vae.dtype):
            decoded_chunk = vae.decode([final_latent.squeeze(0)])[0]
        
        if args.color_match != "disabled":
            try:
                from color_matcher import ColorMatcher
                cm = ColorMatcher()
                
                # Convert decoded chunk to numpy format [F, H, W, C] for ColorMatcher
                # decoded_chunk is [C, F, H, W] in range [-1, 1]
                decoded_np = decoded_chunk.permute(1, 2, 3, 0).cpu().float().numpy()  # [F, H, W, C]
                
                # Get reference frame from initial video (first frame)
                # Use the very first frame of the initial video as reference for ALL chunks
                # This prevents drift accumulation
                ref_frame = initial_chunk[:, 0].permute(1, 2, 0).cpu().float().numpy()  # [H, W, C]
                
                # Apply color matching frame by frame
                matched_frames = []
                for frame_idx in range(decoded_np.shape[0]):
                    frame = decoded_np[frame_idx]  # [H, W, C]
                    # ColorMatcher expects values in range [0, 1] or [0, 255]
                    # Convert from [-1, 1] to [0, 1]
                    frame_01 = (frame + 1.0) / 2.0
                    ref_01 = (ref_frame + 1.0) / 2.0
                    
                    # Apply color matching
                    matched_01 = cm.transfer(src=frame_01, ref=ref_01, method=args.color_match)
                    
                    # Convert back to [-1, 1]
                    matched = matched_01 * 2.0 - 1.0
                    matched_frames.append(torch.from_numpy(matched))
                
                # Stack and convert back to [C, F, H, W]
                decoded_chunk_normalized = torch.stack(matched_frames).permute(3, 0, 1, 2).to(device)
                
                logger.info(f"Applied {args.color_match} color matching to chunk {len(generated_chunks)+1}")
            
            except ImportError:
                logger.warning("color_matcher library not installed. Install with: pip install color-matcher")
                logger.warning("Falling back to internal statistics matching")
                
                # Fallback to internal normalization
                decoded_mean = decoded_chunk.mean(dim=(1, 2, 3), keepdim=True)
                decoded_std = decoded_chunk.std(dim=(1, 2, 3), keepdim=True)
                decoded_chunk_normalized = (decoded_chunk - decoded_mean) / (decoded_std + 1e-6)
                initial_std_reshaped = initial_std.view(-1, 1, 1, 1).to(device)
                initial_mean_reshaped = initial_mean.view(-1, 1, 1, 1).to(device)
                decoded_chunk_normalized = decoded_chunk_normalized * initial_std_reshaped + initial_mean_reshaped
                decoded_chunk_normalized = torch.clamp(decoded_chunk_normalized, -1.0, 1.0)
        else:
            # No color matching - keep decoded chunk as is
            decoded_chunk_normalized = decoded_chunk
        
        # Ensure values stay in valid range
        decoded_chunk_normalized = torch.clamp(decoded_chunk_normalized, -1.0, 1.0)
        
        # Apply chunk blending for smooth transition
        if len(generated_chunks) > 0 and motion_frames > 0:
            # Blend overlapping motion frames region
            blend_frames = min(8, motion_frames // 2)  # Blend up to 8 frames
            if blend_frames > 0:
                # Get last frames from all_frames (previous chunk)
                prev_motion_end = all_frames[-blend_frames:].permute(1, 0, 2, 3).to(device)  # [C, blend_frames, H, W]
                # Get corresponding frames from new chunk
                new_motion_start = decoded_chunk_normalized[:, motion_frames:motion_frames+blend_frames]  # [C, blend_frames, H, W]
                
                # Create blending weights using cosine interpolation for smoother transition
                # Cosine blending provides smoother transition than linear
                t = torch.linspace(0, np.pi, blend_frames).to(device)
                blend_weights = (1.0 - torch.cos(t)) / 2.0  # Smoothstep from 0 to 1
                blend_weights = (1.0 - blend_weights)  # Reverse to go from 1 to 0 for prev frames
                blend_weights = blend_weights.view(1, blend_frames, 1, 1)  # [1, blend_frames, 1, 1]
                
                # Blend the overlapping region
                blended_region = prev_motion_end * blend_weights + new_motion_start * (1.0 - blend_weights)
                
                # Replace the beginning of new frames with blended version
                decoded_chunk_normalized[:, motion_frames:motion_frames+blend_frames] = blended_region
        
        # Log statistics for debugging
        logger.info(f"Chunk {len(generated_chunks)+1} - After processing mean: {decoded_chunk_normalized.mean(dim=(1,2,3)).cpu().tolist()}")
        
        # Append new frames (skip the motion frames that were conditioning)
        new_frames = decoded_chunk_normalized[:, motion_frames:]  # [C, remaining_frames, H, W]
        generated_chunks.append(new_frames.cpu())  # Move to CPU to save GPU memory
        
        # Convert new_frames to [F, C, H, W] format to match all_frames and move to CPU
        new_frames_transposed = new_frames.permute(1, 0, 2, 3).cpu()  # [remaining_frames, C, H, W], move to CPU
        
        # Update all_frames for next iteration
        all_frames = torch.cat([all_frames, new_frames_transposed], dim=0)
        current_frame += (frames_per_chunk - motion_frames)
        
        # Clean up GPU memory after each chunk
        del cond_frames, cond_latent, noise, context, context_null, y, inputs, motion_latent, final_latent, decoded_chunk, new_frames, new_frames_transposed
        clean_memory_on_device(device)
        torch.cuda.empty_cache()
        logger.info(f"Cleaned up GPU memory after chunk {current_frame // frames_per_chunk}")
    
    # Combine all chunks
    final_video = torch.cat(generated_chunks, dim=1)  # [C, F, H, W]
    final_video = final_video.unsqueeze(0)  # [1, C, F, H, W]
    
    # Cleanup
    if is_dual_dit:
        model_manager.cleanup()
    else:
        del model
    del vae
    
    clean_memory_on_device(device)
    
    # Scale to [0, 1] for saving
    final_video = (final_video + 1.0) / 2.0
    final_video = torch.clamp(final_video, 0.0, 1.0)
    
    logger.info(f"Generated extended video: {final_video.shape}")
    return final_video

def broadcast_prepared_inputs(noise, context, context_null, y, inputs, device):
    """Broadcast prepared inputs from rank 0 to all other ranks.
    
    Args:
        noise: Noise tensor
        context: Text context tensor
        context_null: Negative text context tensor
        y: Conditioning latent (can be None)
        inputs: Tuple of (arg_c, arg_null) dicts
        device: Target device for tensors after broadcast
    
    Returns:
        Tuple of broadcasted tensors on the target device
    """
    import torch.distributed as dist
    
    if not dist.is_initialized():
        # No distribution, return as-is
        return noise, context, context_null, y, inputs
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Step 1: Broadcast shapes from rank 0
    if rank == 0:
        # Handle context as list or tensor
        context_is_list = isinstance(context, list)
        context_tensor = context[0] if context_is_list else context
        context_null_is_list = isinstance(context_null, list)
        context_null_tensor = context_null[0] if context_null_is_list else context_null
        
        logger.info(f"Rank {rank}: Context is_list={context_is_list}, shape={context_tensor.shape}, dtype={context_tensor.dtype}")
        logger.info(f"Rank {rank}: Context_null is_list={context_null_is_list}, shape={context_null_tensor.shape}, dtype={context_null_tensor.dtype}")
        
        shapes_info = {
            'noise_shape': list(noise.shape),
            'noise_dtype': str(noise.dtype),
            'context_is_list': context_is_list,
            'context_shape': list(context_tensor.shape),
            'context_dtype': str(context_tensor.dtype),
            'context_null_is_list': context_null_is_list,
            'context_null_shape': list(context_null_tensor.shape),
            'context_null_dtype': str(context_null_tensor.dtype),
            'has_y': y is not None,
            'y_shape': list(y.shape) if y is not None else None,
            'y_dtype': str(y.dtype) if y is not None else None,
        }
    else:
        shapes_info = None
    
    # Broadcast shapes info as a list (NCCL doesn't support dict broadcast directly)
    shapes_list = [shapes_info]
    dist.broadcast_object_list(shapes_list, src=0)
    shapes_info = shapes_list[0]
    
    # Step 2: Allocate tensors on non-rank-0 processes
    if rank != 0:
        # Convert string dtype back to torch dtype
        dtype_map = {
            'torch.float32': torch.float32,
            'torch.float16': torch.float16,
            'torch.bfloat16': torch.bfloat16,
        }
        
        # Allocate on CUDA device for NCCL broadcast
        noise = torch.empty(shapes_info['noise_shape'], 
                           dtype=dtype_map.get(shapes_info['noise_dtype'], torch.float32),
                           device=device)
        context = torch.empty(shapes_info['context_shape'],
                             dtype=dtype_map.get(shapes_info['context_dtype'], torch.float32),
                             device=device)
        context_null = torch.empty(shapes_info['context_null_shape'],
                                   dtype=dtype_map.get(shapes_info['context_null_dtype'], torch.float32),
                                   device=device)
        
        if shapes_info['has_y']:
            y = torch.empty(shapes_info['y_shape'],
                           dtype=dtype_map.get(shapes_info['y_dtype'], torch.float32),
                           device=device)
        else:
            y = None
    
    # Step 3: Ensure tensors are on CUDA for NCCL broadcast
    if rank == 0:
        # Move to device if not already there
        noise_cuda = noise.to(device) if noise.device != device else noise
        # Handle list contexts
        if isinstance(context, list):
            context_cuda = context[0].to(device) if context[0].device != device else context[0]
        else:
            context_cuda = context.to(device) if context.device != device else context
        if isinstance(context_null, list):
            context_null_cuda = context_null[0].to(device) if context_null[0].device != device else context_null[0]
        else:
            context_null_cuda = context_null.to(device) if context_null.device != device else context_null
        y_cuda = y.to(device) if y is not None and y.device != device else y
    else:
        noise_cuda = noise
        context_cuda = context
        context_null_cuda = context_null
        y_cuda = y
    
    # Step 4: Broadcast tensors (using CUDA tensors for NCCL)
    dist.broadcast(noise_cuda, src=0)
    dist.broadcast(context_cuda, src=0)
    dist.broadcast(context_null_cuda, src=0)
    if shapes_info['has_y']:
        dist.broadcast(y_cuda, src=0)
    
    # Step 5: Broadcast inputs dict
    if rank == 0:
        inputs_list = [inputs]
    else:
        inputs_list = [None]
    
    dist.broadcast_object_list(inputs_list, src=0)
    inputs = inputs_list[0]
    
    # Step 6: Restore list format if needed (tensors already on device)
    noise = noise_cuda
    
    # Restore list format for context if needed
    context = [context_cuda] if shapes_info['context_is_list'] else context_cuda
    context_null = [context_null_cuda] if shapes_info['context_null_is_list'] else context_null_cuda
    
    if y_cuda is not None:
        y = y_cuda
    
    # Synchronize to ensure broadcast is complete
    dist.barrier()
    
    logger.info(f"Rank {rank}: Broadcast complete. Noise: {noise.shape}, Context shape: {context_cuda.shape}")
    logger.info(f"Rank {rank}: Final context type: {type(context)}, context_null type: {type(context_null)}")
    if isinstance(context, list):
        logger.info(f"Rank {rank}: Context is list with {len(context)} items")
    
    return noise, context, context_null, y, inputs

def generate(args: argparse.Namespace) -> Optional[torch.Tensor]:
    """main function for generation pipeline (T2V, I2V, V2V)

    Args:
        args: command line arguments

    Returns:
        Optional[torch.Tensor]: generated latent tensor [B, C, F, H, W], or None if only saving merged model.
    """
    device = torch.device(args.device)
    cfg = WAN_CONFIGS[args.task]

    # --- Determine Mode ---
    is_i2v = args.image_path is not None and "i2v" in args.task
    is_ti2v = args.image_path is not None and "ti2v" in args.task  # Text+Image-to-Video
    is_v2v_i2v = args.video_path is not None and args.v2v_use_i2v  # V2V using i2v model
    is_v2v = args.video_path is not None
    is_fun_control = args.control_path is not None and cfg.is_fun_control
    # For ti2v-5B without image, treat as T2V mode (matches official implementation)
    is_extension = args.extend_video is not None
    is_t2v = not is_i2v and not is_ti2v and not is_v2v and not is_fun_control

    if is_v2v: logger.info(f"Running Video-to-Video (V2V) inference with strength {args.strength}")
    elif is_ti2v: logger.info(f"Running Text+Image-to-Video (TI2V) inference")
    elif is_i2v: logger.info(f"Running Image-to-Video (I2V) inference")
    elif is_v2v_i2v: logger.info(f"Running Video-to-Video (V2V) using i2v model")
    elif is_fun_control: logger.info(f"Running Text-to-Video with Fun-Control") # Note: FunControl can also be I2V if image_path is given
    elif is_extension: logger.info(f"Running Video Extension (multitalk-style) to {args.extend_frames} frames")
    else: 
        if args.task == "ti2v-5B" and args.image_path is None:
            logger.info(f"Running Text-to-Video (T2V) inference for ti2v-5B (no image provided)")
        else:
            logger.info(f"Running Text-to-Video (T2V) inference")

    # --- Data Types ---
    # Default to fp16 for new Wan2.2 models, detect from checkpoint if available
    if args.dit is not None:
        dit_dtype = detect_wan_sd_dtype(args.dit)
    elif args.dit_low_noise is not None:
        dit_dtype = detect_wan_sd_dtype(args.dit_low_noise)
    else:
        dit_dtype = torch.float16  # Default to fp16 for new models
    
    if args.mixed_dtype:
        # For mixed dtype, keep using fp16 for activations/computation
        # This keeps memory usage low while preserving the precision of fp32 weights
        # PyTorch will handle mixed precision ops automatically
        dit_dtype = torch.float16
        logger.info("Mixed dtype mode: Using fp16 for activations, preserving original weight dtypes")
        dit_weight_dtype = None  # Will be set to None in load_dit_model
    elif dit_dtype.itemsize == 1: # FP8 weights loaded
        dit_dtype = torch.bfloat16 # Use bfloat16 for computation
        if args.fp8_scaled:
            raise ValueError("Cannot use --fp8_scaled with pre-quantized FP8 weights.")
        dit_weight_dtype = torch.float8_e4m3fn
    elif args.fp8_scaled:
        dit_weight_dtype = None # Optimize later
    elif args.fp8:
        dit_weight_dtype = torch.float8_e4m3fn
    else:
        dit_weight_dtype = dit_dtype # Use compute dtype for weights
    # Format weight dtype for logging
    if args.mixed_dtype:
        weight_dtype_str = "Mixed (Original)"
    elif args.fp8_scaled:
        weight_dtype_str = "Mixed (FP8 Scaled)"
    else:
        weight_dtype_str = str(dit_weight_dtype)
    vae_dtype = str_to_dtype(args.vae_dtype) if args.vae_dtype is not None else (torch.bfloat16 if dit_dtype == torch.bfloat16 else torch.float16)
    # Format weight dtype for logging
    if args.mixed_dtype:
        weight_dtype_str = "Mixed (Original)"
    elif args.fp8_scaled:
        weight_dtype_str = "Mixed (FP8 Scaled)"
    else:
        weight_dtype_str = str(dit_weight_dtype)
    
    logger.info(
        f"Using device: {device}, DiT compute: {dit_dtype}, DiT weight: {weight_dtype_str}, VAE: {vae_dtype}, T5 FP8: {args.fp8_t5}"
     )

    # --- Accelerator ---
    mixed_precision = "bf16" if dit_dtype == torch.bfloat16 else "fp16"
    accelerator = accelerate.Accelerator(mixed_precision=mixed_precision)

    # --- Seed ---
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # Store seed back for metadata
    logger.info(f"Using seed: {seed}")

    # --- Load VAE (if needed for input processing) ---
    vae = None
    # VAE is needed early for V2V, I2V, TI2V, and FunControl T2V
    needs_vae_early = is_v2v or is_i2v or is_ti2v or is_v2v_i2v or (is_fun_control and is_t2v) or (is_fun_control and is_i2v) # Refined condition
    
    # Check if we're using FSDP
    use_fsdp = args.dit_fsdp or args.t5_fsdp
    rank = 0
    if use_fsdp and dist.is_initialized():
        rank = dist.get_rank()
    
    # Only load VAE on rank 0 when using FSDP
    if needs_vae_early:
        if use_fsdp and rank != 0:
            # Other ranks don't need VAE for input prep
            vae = None
        else:
            vae = load_vae(args, cfg, device, vae_dtype)
            # Keep VAE on specified device for now, will be moved as needed

    # Handle video extension mode
    if is_extension:
        logger.info(f"Extending video from {args.extend_video} to {args.extend_frames} frames")
        
        # Use clean i2v-based extension approach (much better than multitalk-style)
        # This approach extracts the best frame and generates smooth extensions
        try:
            extended_video = generate_extended_video_i2v_based(
                args, args.extend_video, args.extend_frames
            )
            return extended_video  # Return the extended video directly
        except Exception as e:
            logger.error(f"i2v-based extension failed: {e}")
            logger.info("Falling back to multitalk-style extension")
            # Fallback to original method
            extended_video = generate_extended_video(
                args, args.extend_video, args.extend_frames, args.motion_frames
            )
            return extended_video

        vae = load_vae(args, cfg, device, vae_dtype)
        # Keep VAE on specified device for now, will be moved as needed

    # --- Prepare Inputs ---
    noise = None
    context = None
    context_null = None
    inputs = None
    video_latents = None # For V2V mixing
    
    # Check if we're using FSDP
    use_fsdp = args.dit_fsdp or args.t5_fsdp
    rank = 0
    if use_fsdp and dist.is_initialized():
        rank = dist.get_rank()

    if is_v2v_i2v:
        # V2V using i2v model - combines video encoding with CLIP conditioning
        # 1. Load video frames
        video_frames_np, actual_frames_loaded = load_video(
            args.video_path,
            start_frame=0,
            num_frames=args.video_length,
            bucket_reso=tuple(args.video_size)
        )
        if actual_frames_loaded == 0:
            raise ValueError(f"Could not load any frames from video: {args.video_path}")
            
        # Update video_length if needed
        if args.video_length is None or actual_frames_loaded < args.video_length:
            logger.info(f"Updating video_length based on loaded frames: {actual_frames_loaded}")
            args.video_length = actual_frames_loaded
            height, width, video_length = check_inputs(args)
            args.video_size = [height, width]
        else:
            video_length = args.video_length
            
        # 2. Prepare V2V-I2V inputs (handles both video encoding and CLIP conditioning)
        noise, context, context_null, clip_context, video_latents, inputs = prepare_v2v_i2v_inputs(
            args, cfg, accelerator, device, vae, video_frames_np
        )
        
        # Force low noise model for V2V if requested
        if args.v2v_low_noise_only and args.dual_dit_boundary is None:
            logger.info("V2V with --v2v_low_noise_only: forcing dual_dit_boundary to 1.0 (low noise model only)")
            args.dual_dit_boundary = 1.0
            
        # Adjust default strength for i2v V2V
        if args.strength == 0.75:  # Default value
            args.strength = 0.9
            logger.info(f"Using recommended V2V strength for i2v model: {args.strength}")
        
    elif is_v2v and not is_v2v_i2v:
        # Standard V2V path (mutually exclusive with FunControl)
        # 1. Load and prepare video
        video_frames_np, actual_frames_loaded = load_video(
            args.video_path,
            start_frame=0,
            num_frames=args.video_length, # Can be None
            bucket_reso=tuple(args.video_size)
        )
        if actual_frames_loaded == 0:
             raise ValueError(f"Could not load any frames from video: {args.video_path}")

        # Update video_length if it was None or if fewer frames were loaded
        if args.video_length is None or actual_frames_loaded < args.video_length:
            logger.info(f"Updating video_length based on loaded frames: {actual_frames_loaded}")
            args.video_length = actual_frames_loaded
            # Re-check height/width/length now that length is known
            height, width, video_length = check_inputs(args)
            args.video_size = [height, width] # Update args
        else:
            video_length = args.video_length # Use the specified length

        # Convert frames to tensor [1, C, F, H, W], range [0, 1]
        video_tensor = torch.from_numpy(np.stack(video_frames_np, axis=0)) #[F,H,W,C]
        video_tensor = video_tensor.permute(0, 3, 1, 2).float() / 255.0 #[F,C,H,W]
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0) #[1,C,F,H,W]

        # 2. Encode video to latents
        video_latents = encode_video_to_latents(video_tensor, vae, device, dit_dtype, args) # Use DiT dtype for latents
        del video_tensor # Free pixel video memory
        clean_memory_on_device(device)

        # 3. Prepare V2V inputs (noise matching latent shape, context, etc.)
        noise, context, context_null, inputs = prepare_v2v_inputs(args, cfg, accelerator, device, video_latents)

    elif is_ti2v:
        # TI2V path - use official WanTI2V implementation  
        logger.info("Using official WanTI2V implementation for ti2v-5B")
        
        # Import the official TI2V class
        from Wan2_2.wan.textimage2video import WanTI2V
        from Wan2_2.wan.configs.wan_ti2v_5B import ti2v_5B
        
        # The official implementation expects a diffusers checkpoint directory
        # We need to create a temporary directory structure that matches what it expects
        import os
        import tempfile
        import shutil
        from safetensors.torch import save_file
        
        # Create temporary directory with expected structure
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Load our single safetensor file
            from safetensors.torch import load_file
            state_dict = load_file(args.dit)
            
            # Create the expected checkpoint structure 
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save the model file
            save_file(state_dict, os.path.join(temp_dir, "diffusion_pytorch_model.safetensors"))
            
            # Create config.json with full model configuration matching WanModel
            import json
            config_dict = {
                "_class_name": "WanModel",
                "_diffusers_version": "0.21.0",
                "model_type": "ti2v",
                "patch_size": [1, 2, 2],
                "text_len": 512,
                "in_dim": 48,
                "dim": 3072,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 48,
                "num_heads": 24,
                "num_layers": 30,
                "window_size": [-1, -1],
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6
            }
            with open(os.path.join(temp_dir, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)
            
            # Copy VAE and T5 files to temp directory
            import shutil
            vae_filename = os.path.basename(args.vae)
            t5_filename = os.path.basename(args.t5)
            
            shutil.copy2(args.vae, os.path.join(temp_dir, vae_filename))
            shutil.copy2(args.t5, os.path.join(temp_dir, t5_filename))
            
            # Update the config to use relative paths within temp directory
            from easydict import EasyDict
            ti2v_5B_config = EasyDict(ti2v_5B)
            ti2v_5B_config.vae_checkpoint = vae_filename
            ti2v_5B_config.t5_checkpoint = t5_filename 
            
            # Create a custom WanTI2V instance that handles tokenizer path correctly
            class CustomWanTI2V(WanTI2V):
                def __init__(self, config, checkpoint_dir, **kwargs):
                    # Override just the T5 initialization to handle tokenizer path correctly
                    from functools import partial
                    from Wan2_2.wan.distributed.fsdp import shard_model
                    from Wan2_2.wan.modules.t5 import T5EncoderModel
                    from Wan2_2.wan.modules.vae2_2 import Wan2_2_VAE
                    from Wan2_2.wan.modules.model import WanModel
                    
                    # Initialize base attributes
                    self.device = torch.device(f"cuda:{kwargs.get('device_id', 0)}")
                    self.config = config
                    self.rank = kwargs.get('rank', 0)
                    self.t5_cpu = kwargs.get('t5_cpu', False)
                    self.init_on_cpu = kwargs.get('init_on_cpu', True)
                    
                    self.num_train_timesteps = config.num_train_timesteps
                    self.param_dtype = config.param_dtype
                    
                    if kwargs.get('t5_fsdp') or kwargs.get('dit_fsdp') or kwargs.get('use_sp'):
                        self.init_on_cpu = False
                    
                    # Create text encoder with correct tokenizer path
                    shard_fn = partial(shard_model, device_id=kwargs.get('device_id', 0))
                    self.text_encoder = T5EncoderModel(
                        text_len=config.text_len,
                        dtype=config.t5_dtype,
                        device=torch.device('cpu'),
                        checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
                        tokenizer_path="google/umt5-xxl",  # Use HF repo ID directly
                        shard_fn=shard_fn if kwargs.get('t5_fsdp') else None)
                    
                    # Initialize VAE and model normally
                    self.vae_stride = config.vae_stride
                    self.patch_size = config.patch_size
                    self.vae = Wan2_2_VAE(
                        vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
                        device=self.device)
                    
                    import logging
                    logging.info(f"Creating WanModel from {checkpoint_dir}")
                    self.model = WanModel.from_pretrained(checkpoint_dir)
                    self.model = self._configure_model(
                        model=self.model,
                        use_sp=kwargs.get('use_sp', False),
                        dit_fsdp=kwargs.get('dit_fsdp', False),
                        shard_fn=shard_fn,
                        convert_model_dtype=kwargs.get('convert_model_dtype', False))
                    
                    if kwargs.get('use_sp'):
                        from Wan2_2.wan.distributed.util import get_world_size
                        self.sp_size = get_world_size()
                    else:
                        self.sp_size = 1
                    
                    self.sample_neg_prompt = config.sample_neg_prompt
                
                def i2v(self, *args, **kwargs):
                    # Call the original i2v method but with explicit model unloading before VAE decode
                    from tqdm import tqdm
                    from contextlib import contextmanager
                    import torch
                    import random
                    import sys
                    import math
                    from PIL import Image
                    import torchvision.transforms.functional as TF
                    from Wan2_2.wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
                    from Wan2_2.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
                    from Wan2_2.wan.utils.utils import best_output_size, masks_like
                    
                    # Extract parameters
                    input_prompt = args[0] if args else kwargs['input_prompt']
                    img = args[1] if len(args) > 1 else kwargs['img']
                    max_area = kwargs.get('max_area', 704 * 1280)
                    frame_num = kwargs.get('frame_num', 121)
                    shift = kwargs.get('shift', 5.0)
                    sample_solver = kwargs.get('sample_solver', 'unipc')
                    sampling_steps = kwargs.get('sampling_steps', 40)
                    guide_scale = kwargs.get('guide_scale', 5.0)
                    n_prompt = kwargs.get('n_prompt', "")
                    seed = kwargs.get('seed', -1)
                    offload_model = kwargs.get('offload_model', True)
                    
                    # Reproduce the original i2v logic but with better memory management
                    # preprocess
                    ih, iw = img.height, img.width
                    dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[2] * self.vae_stride[2]
                    ow, oh = best_output_size(iw, ih, dw, dh, max_area)

                    scale = max(ow / iw, oh / ih)
                    img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

                    # center-crop
                    x1 = (img.width - ow) // 2
                    y1 = (img.height - oh) // 2
                    img = img.crop((x1, y1, x1 + ow, y1 + oh))
                    assert img.width == ow and img.height == oh

                    # to tensor
                    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

                    F = frame_num
                    seq_len = ((F - 1) // self.vae_stride[0] + 1) * (oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (self.patch_size[1] * self.patch_size[2])
                    seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

                    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
                    seed_g = torch.Generator(device=self.device)
                    seed_g.manual_seed(seed)
                    noise = torch.randn(
                        self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        oh // self.vae_stride[1], ow // self.vae_stride[2],
                        dtype=torch.float32, generator=seed_g, device=self.device)

                    if n_prompt == "":
                        n_prompt = self.sample_neg_prompt

                    # preprocess
                    if not self.t5_cpu:
                        self.text_encoder.model.to(self.device)
                        context = self.text_encoder([input_prompt], self.device)
                        context_null = self.text_encoder([n_prompt], self.device)
                        if offload_model:
                            self.text_encoder.model.cpu()
                    else:
                        context = self.text_encoder([input_prompt], torch.device('cpu'))
                        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                        context = [t.to(self.device) for t in context]
                        context_null = [t.to(self.device) for t in context_null]

                    z = self.vae.encode([img])

                    @contextmanager
                    def noop_no_sync():
                        yield

                    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

                    # evaluation mode
                    with (
                            torch.amp.autocast('cuda', dtype=self.param_dtype),
                            torch.no_grad(),
                            no_sync(),
                    ):

                        if sample_solver == 'unipc':
                            sample_scheduler = FlowUniPCMultistepScheduler(
                                num_train_timesteps=self.num_train_timesteps,
                                shift=1,
                                use_dynamic_shifting=False)
                            sample_scheduler.set_timesteps(
                                sampling_steps, device=self.device, shift=shift)
                            timesteps = sample_scheduler.timesteps
                        elif sample_solver == 'dpm++':
                            sample_scheduler = FlowDPMSolverMultistepScheduler(
                                num_train_timesteps=self.num_train_timesteps,
                                shift=1,
                                use_dynamic_shifting=False)
                            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                            timesteps, _ = retrieve_timesteps(
                                sample_scheduler,
                                device=self.device,
                                sigmas=sampling_sigmas)
                        else:
                            raise NotImplementedError("Unsupported solver.")

                        # sample videos
                        latent = noise
                        mask1, mask2 = masks_like([noise], zero=True)
                        latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

                        arg_c = {
                            'context': [context[0]],
                            'seq_len': seq_len,
                        }

                        arg_null = {
                            'context': context_null,
                            'seq_len': seq_len,
                        }

                        if offload_model or self.init_on_cpu:
                            self.model.to(self.device)
                            torch.cuda.empty_cache()

                        for _, t in enumerate(tqdm(timesteps)):
                            latent_model_input = [latent.to(self.device)]
                            timestep = [t]

                            timestep = torch.stack(timestep).to(self.device)

                            temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                            temp_ts = torch.cat([
                                temp_ts,
                                temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                            ])
                            timestep = temp_ts.unsqueeze(0)

                            noise_pred_cond = self.model(
                                latent_model_input, t=timestep, **arg_c)[0]
                            if offload_model:
                                torch.cuda.empty_cache()
                            noise_pred_uncond = self.model(
                                latent_model_input, t=timestep, **arg_null)[0]
                            if offload_model:
                                torch.cuda.empty_cache()
                            noise_pred = noise_pred_uncond + guide_scale * (
                                noise_pred_cond - noise_pred_uncond)

                            temp_x0 = sample_scheduler.step(
                                noise_pred.unsqueeze(0),
                                t,
                                latent.unsqueeze(0),
                                return_dict=False,
                                generator=seed_g)[0]
                            latent = temp_x0.squeeze(0)
                            latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

                            x0 = [latent]
                            del latent_model_input, timestep

                        # CRITICAL: Unload model BEFORE VAE decode to free GPU memory
                        if offload_model:
                            self.model.cpu()
                            torch.cuda.synchronize()
                        
                        # Delete intermediate tensors
                        del mask1, mask2, z, context, context_null, arg_c, arg_null
                        del noise_pred_cond, noise_pred_uncond, noise_pred
                        if 'temp_x0' in locals():
                            del temp_x0
                        
                        # Force memory cleanup
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        gc.collect()
                        
                        # Additional memory cleanup
                        import time
                        time.sleep(0.5)  # Give GPU time to free memory
                        torch.cuda.empty_cache()

                        # Return latent instead of decoded video to avoid OOM
                        # Let the main pipeline handle VAE decode with its own memory management
                        if self.rank == 0:
                            # Return the latent directly
                            return x0  # Return list of latents

                    del noise, latent
                    del sample_scheduler
                    if offload_model:
                        gc.collect()
                        torch.cuda.synchronize()
                    
                    return None
            
            # Create custom TI2V model instance that handles tokenizer path correctly
            wan_ti2v = CustomWanTI2V(
                config=ti2v_5B_config,
                checkpoint_dir=temp_dir,  # Use our temporary directory
                device_id=0,
                rank=0,  
                t5_fsdp=False,
                dit_fsdp=False,
                use_sp=False,
                t5_cpu=False,
                init_on_cpu=True,
                convert_model_dtype=True,
            )
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            # Force cleanup of any lingering GPU memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Load the input image
        from PIL import Image
        input_image = Image.open(args.image_path).convert('RGB')
        
        # Generate using official implementation but return latent instead of decoded video
        # We need to modify generate to return latent to avoid VAE decode inside
        result_latent = wan_ti2v.i2v(
            input_prompt=args.prompt,
            img=input_image,
            max_area=args.video_size[0] * args.video_size[1],
            frame_num=args.video_length,
            shift=args.flow_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.infer_steps,
            guide_scale=args.guidance_scale,
            n_prompt=args.negative_prompt or "",
            seed=args.seed,
            offload_model=True
        )
        
        # Explicitly unload all models to free GPU memory
        if hasattr(wan_ti2v, 'model'):
            wan_ti2v.model.cpu()
            del wan_ti2v.model
        if hasattr(wan_ti2v, 'text_encoder'):
            wan_ti2v.text_encoder.model.cpu()
            del wan_ti2v.text_encoder
        if hasattr(wan_ti2v, 'vae'):
            wan_ti2v.vae.model.cpu()
            del wan_ti2v.vae
        del wan_ti2v
        torch.cuda.empty_cache()
        gc.collect()
        
        # The result_latent is the generated latent, not the decoded video
        # We'll return it as is and let the main pipeline handle VAE decode
        if result_latent is not None:
            # Convert to expected format [1, C, F, H, W]
            if isinstance(result_latent, list):
                final_latent = result_latent[0].unsqueeze(0)
            else:
                final_latent = result_latent.unsqueeze(0)
        else:
            final_latent = None
            
        # Store a flag that we already have a VAE loaded
        args._vae_already_loaded = True
            
        logger.info("TI2V generation complete using official implementation (latent output)")
        return final_latent

    elif is_i2v:
        # I2V path (handles both standard and FunControl internally based on config)
        if args.video_length is None:
             raise ValueError("video_length must be specified for I2V mode.")
        
        if use_fsdp and rank == 0:
            # Only rank 0 prepares inputs
            noise, context, context_null, y, inputs = prepare_i2v_inputs(args, cfg, accelerator, device, vae)
            # Broadcast to other ranks
            noise, context, context_null, y, inputs = broadcast_prepared_inputs(
                noise, context, context_null, y, inputs, device
            )
        elif use_fsdp:
            # Other ranks receive broadcast
            noise, context, context_null, y, inputs = broadcast_prepared_inputs(
                None, None, None, None, None, device
            )
        else:
            # Single GPU or non-FSDP mode
            noise, context, context_null, y, inputs = prepare_i2v_inputs(args, cfg, accelerator, device, vae)
        # Note: prepare_i2v_inputs moves VAE to CPU/cache after use

    elif is_fun_control: # Pure FunControl T2V (no image input unless using start/end image)
        if args.video_length is None:
             raise ValueError("video_length must be specified for Fun-Control T2V mode.")
        
        if use_fsdp and rank == 0:
            # Only rank 0 prepares inputs
            noise, context, context_null, inputs = prepare_t2v_inputs(args, cfg, accelerator, device, vae)
            # Broadcast to other ranks (y is None for FunControl T2V)
            noise, context, context_null, _, inputs = broadcast_prepared_inputs(
                noise, context, context_null, None, inputs, device
            )
        elif use_fsdp:
            # Other ranks receive broadcast
            noise, context, context_null, _, inputs = broadcast_prepared_inputs(
                None, None, None, None, None, device
            )
        else:
            # Single GPU or non-FSDP mode
            noise, context, context_null, inputs = prepare_t2v_inputs(args, cfg, accelerator, device, vae)
        # Note: prepare_t2v_inputs moves VAE to CPU/cache if it used it

    elif is_t2v: # Standard T2V
        if args.video_length is None:
             raise ValueError("video_length must be specified for standard T2V mode.")
        
        if use_fsdp and rank == 0:
            # Only rank 0 prepares inputs
            noise, context, context_null, inputs = prepare_t2v_inputs(args, cfg, accelerator, device, None) # Pass None for VAE
            # Broadcast to other ranks (y is None for standard T2V)
            noise, context, context_null, _, inputs = broadcast_prepared_inputs(
                noise, context, context_null, None, inputs, device
            )
        elif use_fsdp:
            # Other ranks receive broadcast
            noise, context, context_null, _, inputs = broadcast_prepared_inputs(
                None, None, None, None, None, device
            )
        else:
            # Single GPU or non-FSDP mode
            noise, context, context_null, inputs = prepare_t2v_inputs(args, cfg, accelerator, device, None) # Pass None for VAE
    
    # Add synchronization barrier for FSDP to ensure all ranks have inputs before proceeding
    if use_fsdp and dist.is_initialized():
        dist.barrier()
        logger.info(f"Rank {rank}: Synchronized after input preparation")

    # At this point, VAE should be on CPU/cache unless still needed for decoding
    # If VAE wasn't loaded early (standard T2V), vae is still None

    # --- Load DiT Model(s) ---
    model_result = load_dit_model(args, cfg, device, dit_dtype, dit_weight_dtype, is_i2v)
    
    # Handle dual-dit models
    is_dual_dit = isinstance(model_result, tuple)
    model_manager = None
    
    if is_dual_dit:
        # Set up dynamic model manager
        if len(model_result) == 6:
            # New format with LoRA weights
            model_low_path, model_high_path, lora_weights_list_low, lora_multipliers_low, \
                lora_weights_list_high, lora_multipliers_high = model_result
        else:
            # Old format compatibility
            model_low_path, model_high_path = model_result
            lora_weights_list_low = lora_multipliers_low = None
            lora_weights_list_high = lora_multipliers_high = None
            
        # Select appropriate model manager based on configuration
        if args.dit_fsdp or args.t5_fsdp:
            # Use FSDP manager for memory-efficient distributed training
            if not dist.is_initialized() and torch.cuda.device_count() < 2:
                logger.warning(f"FSDP requested but only {torch.cuda.device_count()} GPU(s) available. Falling back to single GPU.")
                model_manager = DynamicModelManager(cfg, device, dit_dtype, dit_weight_dtype, args)
            else:
                logger.info(f"Using FSDP Model Manager")
                model_manager = FSDPModelManager(cfg, device, dit_dtype, dit_weight_dtype, args)
                # Use the rank-specific device from the model manager
                device = model_manager.device
                
                # Synchronize before starting sampling
                if dist.is_initialized():
                    dist.barrier()
                    logger.info(f"Rank {model_manager.rank}: Ready to start sampling with FSDP")
        elif args.use_sequence_parallel or args.use_dual_gpu:
            if args.use_dual_gpu:
                logger.warning("--use_dual_gpu is deprecated. Using --use_sequence_parallel instead.")
            if not dist.is_initialized() and torch.cuda.device_count() < 2:
                logger.warning(f"Sequence parallelism requested but only {torch.cuda.device_count()} GPU(s) available. Falling back to single GPU.")
                model_manager = DynamicModelManager(cfg, device, dit_dtype, dit_weight_dtype, args)
            else:
                logger.info(f"Using Sequence Parallel Model Manager")
                model_manager = SequenceParallelModelManager(cfg, device, dit_dtype, dit_weight_dtype, args)
        else:
            model_manager = DynamicModelManager(cfg, device, dit_dtype, dit_weight_dtype, args)
        model_manager.set_model_paths(model_low_path, model_high_path)
        
        # Set LoRA weights if available
        if lora_weights_list_low is not None or lora_weights_list_high is not None:
            model_manager.set_lora_weights(lora_weights_list_low, lora_multipliers_low,
                                          lora_weights_list_high, lora_multipliers_high)
        
        # Don't load any model initially - let the sampling loop load the appropriate one
        # This avoids loading low noise model just to immediately swap to high noise
        model = None
        model_low = model_high = None  # Not used anymore
        logger.info("Using dynamic model loading for dual-dit architecture (default)")
    else:
        model = model_result
        model_low = model_high = None

    # --- Verify LoRA Merge (if applicable) ---
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        # LoRA weights were already merged during model loading
        logger.info("LoRA weights were merged during model loading (efficient hook-based method)")
        
        # Optional: Verify the merge was successful by checking a few weights
        # This maintains compatibility with your dual checking process
        try:
            # Simple verification: check if model has expected parameter count
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"Model loaded with {param_count:,} parameters after LoRA merge")
        except Exception as e:
            logger.warning(f"Could not verify LoRA merge: {e}")
        
        # Handle save_merged_model if specified
        if args.save_merged_model:
            logger.info(f"Saving merged model to {args.save_merged_model}")
            try:
                mem_eff_save_file(model.state_dict(), args.save_merged_model)
                logger.info("Merged model saved successfully")
            except Exception as e:
                logger.error(f"Failed to save merged model: {e}")
            
            # Clean up and exit
            if 'model' in locals(): del model
            if 'vae' in locals() and vae is not None: del vae
            clean_memory_on_device(device)
            return None # Exit early

    # --- Optimize Model (FP8, Swapping, Compile) ---
    # Only optimize if we have a model loaded (not for dual-dit dynamic loading)
    if model is not None:
        optimize_model(model, args, device, dit_dtype, dit_weight_dtype)
    else:
        # For dual-dit, optimization will happen when each model is loaded
        logger.info("Model optimization will be performed during dynamic loading")

    # --- Setup Scheduler & Timesteps ---
    scheduler, timesteps = setup_scheduler(args, cfg, device)

    # --- Prepare for Sampling ---
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    # `latent` here is the initial state *before* the sampling loop starts
    latent = noise # Start with noise (already shaped correctly for T2V/I2V/V2V)
    
    # --- Apply Pusa Multi-Frame Conditioning Before Sampling ---
    # This must happen BEFORE the sampling loop to inject conditioning frames into the initial latent
    if hasattr(args, 'pusa_conditioning_dict') and args.pusa_conditioning_dict and vae is not None:
        logger.info("Encoding Pusa conditioning images to latent space...")
        
        # Ensure VAE is on the correct device
        vae.to_device(device)
        
        # Process each conditioning image
        for frame_idx, (cond_image, noise_mult) in args.pusa_conditioning_dict.items():
            # Determine the frame dimension based on tensor shape
            if latent.dim() == 5:  # [B, C, F, H, W]
                num_frames = latent.shape[2]
            elif latent.dim() == 4:  # [C, F, H, W]
                num_frames = latent.shape[1]
            else:
                logger.warning(f"Unexpected latent dimensions: {latent.shape}")
                continue
                
            if frame_idx < num_frames:  # Check frame index is valid
                logger.debug(f"Encoding conditioning image for frame {frame_idx} (total frames: {num_frames})")
                
                # Convert PIL image to tensor and prepare for VAE encoding
                # The cond_image is a PIL Image from the pusa utils
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),  # Converts to [C, H, W] with values [0, 1]
                ])
                img_tensor = transform(cond_image).unsqueeze(0).to(device)  # [1, 3, H, W]
                
                # Normalize to [-1, 1] for VAE
                img_tensor = 2.0 * img_tensor - 1.0
                
                # Encode image to latent space
                with torch.no_grad():
                    # VAE expects a list containing [C, F, H, W] tensor
                    # We'll treat each frame as a single-frame video
                    img_tensor_video = img_tensor.squeeze(0).unsqueeze(1)  # [3, 1, H, W] - remove batch, add frame dim
                    
                    # vae.encode expects a list, returns a list
                    cond_latent_list = vae.encode([img_tensor_video])  # Returns list with [C', 1, H_lat, W_lat]
                    cond_latent = cond_latent_list[0].to(device)  # Get first element from list
                    
                    # Remove the temporal dimension since we encoded a single frame
                    cond_latent = cond_latent.squeeze(1)  # [C', H_lat, W_lat]
                    
                    # Inject into the latent at the specified frame position
                    if latent.dim() == 4:  # [C, F, H, W]
                        latent[:, frame_idx, :, :] = cond_latent
                    elif latent.dim() == 5:  # [B, C, F, H, W]
                        latent[:, :, frame_idx, :, :] = cond_latent.unsqueeze(0)  # Add batch dim
                    
                    logger.info(f"Injected conditioning latent at frame {frame_idx} (noise_mult={noise_mult})")
            else:
                logger.warning(f"Skipping conditioning frame {frame_idx} - out of bounds (latent has {num_frames} frames)")
        
        # Move VAE back to CPU to save memory
        vae.to_device(args.vae_cache_cpu if args.vae_cache_cpu else "cpu")
        clean_memory_on_device(device)
        logger.info("Pusa conditioning images encoded and injected into initial latent")
    
    # Create masks for TI2V before sampling loop (following official implementation)
    ti2v_mask1, ti2v_mask2 = None, None
    if is_ti2v and "_image_latent" in inputs[0]:
        image_latent = inputs[0]["_image_latent"].to(latent.device)
        
        # Create masks following official implementation: masks_like([noise], zero=True)
        # These masks are created once and reused throughout the sampling process
        if latent.dim() == 4:  # [48, 21, 44, 80] format (no batch dim)
            # Create mask list like official implementation
            ti2v_mask1 = [torch.ones_like(latent)]
            ti2v_mask2 = [torch.ones_like(latent)]
            ti2v_mask2[0][:, 0, :, :] = 0  # First frame gets image conditioning
            
            # Apply initial mask-based conditioning: latent = (1. - mask2[0]) * z[0] + mask2[0] * latent
            latent = (1. - ti2v_mask2[0]) * image_latent + ti2v_mask2[0] * latent
            
        elif latent.dim() == 5:  # [1, 48, 21, 44, 80] format (with batch dim)
            # Ensure image_latent has batch dimension
            if image_latent.dim() == 4:
                image_latent = image_latent.unsqueeze(0)
            
            # Create mask list like official implementation  
            ti2v_mask1 = [torch.ones_like(latent)]
            ti2v_mask2 = [torch.ones_like(latent)]
            ti2v_mask2[0][:, :, 0, :, :] = 0  # First frame gets image conditioning
            
            # Apply initial mask-based conditioning
            latent = (1. - ti2v_mask2[0]) * image_latent + ti2v_mask2[0] * latent
        
        logger.info("Applied initial image conditioning for TI2V using mask-based blending")
        
        # Store masks in inputs for use during sampling
        inputs[0]["_ti2v_mask1"] = ti2v_mask1
        inputs[0]["_ti2v_mask2"] = ti2v_mask2

    if (is_v2v or is_v2v_i2v) and args.strength < 1.0:
        # Calculate how many steps to skip based on strength
        init_timestep_idx = int(args.infer_steps * (1.0 - args.strength))
        init_timestep_idx = min(init_timestep_idx, args.infer_steps - 1)
        
        # Get the actual timestep value
        init_timestep = timesteps[init_timestep_idx]
        
        # Use scheduler's add_noise method to properly add noise
        # This applies the correct alpha_t and sigma_t scaling
        latent = scheduler.add_noise(
            original_samples=video_latents,
            noise=latent,  # This is pure noise
            timesteps=torch.tensor([init_timestep], device=device)
        )
        
        # Skip the early timesteps
        timesteps = timesteps[init_timestep_idx:]
        
        logger.info(f"V2V: Starting from timestep {init_timestep.item():.0f} (skipping {init_timestep_idx} steps)")
        logger.info(f"Using {len(timesteps)} timesteps for V2V sampling")
    else:
         logger.info(f"Using full {len(timesteps)} timesteps for sampling.")
    previewer = None
    if LatentPreviewer is not None and args.preview is not None and args.preview > 0:
        logger.info(f"Initializing Latent Previewer (every {args.preview} steps)...")
        try:
             # Use the initial 'latent' state which might be pure noise or mixed V2V start
             # Pass without batch dim [C, F, H, W]
             initial_latent_for_preview = latent.clone().squeeze(0)
             previewer = LatentPreviewer(args, initial_latent_for_preview, timesteps, device, dit_dtype, model_type="wan")
             logger.info("Latent Previewer initialized successfully.")
        except Exception as e:
             logger.error(f"Failed to initialize Latent Previewer: {e}", exc_info=True)
             previewer = None # Ensure it's None if init fails

    # --- Apply Context Windows if Enabled ---
    model_options = apply_context_windows(args)
    
    # --- Run Sampling Loop ---
    logger.info("Starting denoising sampling loop...")
    
    # Log dual-dit boundary information if applicable
    if is_dual_dit and model_high is not None:
        if args.dual_dit_boundary is not None:
            boundary_percent = args.dual_dit_boundary * 100
            logger.info(f"Using custom dual-dit boundary: {boundary_percent:.1f}% (high noise model used above {args.dual_dit_boundary * 1000:.0f} timesteps)")
        else:
            default_boundary = cfg.boundary
            boundary_percent = default_boundary * 100
            logger.info(f"Using default dual-dit boundary: {boundary_percent:.1f}% (high noise model used above {default_boundary * 1000:.0f} timesteps)")
    
    final_latent = run_sampling(
        model,
        latent, # Initial state (noise or mixed)
        scheduler,
        timesteps, # Full or partial timesteps
        args,
        inputs, # Contains context etc.
        device,
        seed_g,
        accelerator,
        is_ti2v=is_ti2v,  # Pass TI2V flag for special processing
        model_manager=model_manager,  # New parameter
        previewer=previewer, # MODIFIED: Pass the previewer instance
        use_cpu_offload=(args.blocks_to_swap > 0), # Example: offload if swapping
        preview_suffix=args.preview_suffix, # Pass the preview suffix to run_sampling
        model_options=model_options  # Pass context window options
    )

    # --- Cleanup ---
    if model_manager:
        model_manager.cleanup()
    
    # Only delete model if it exists
    if model is not None:
        del model
        
    if 'scheduler' in locals(): del scheduler
    if 'context' in locals(): del context
    if 'context_null' in locals(): del context_null
    if 'inputs' in locals(): del inputs # Free memory from encoded inputs
    if video_latents is not None: del video_latents
    # previewer instance will be garbage collected

    synchronize_device(device)

    if args.blocks_to_swap > 0:
        logger.info("Waiting for 5 seconds to ensure block swap finishes...")
        time.sleep(5)

    gc.collect()
    clean_memory_on_device(device)

    # Store VAE instance in args for decoding function (if it exists)
    args._vae = vae # Store VAE instance (might be None if T2V)

    # Return latent with batch dimension [1, C, F, H, W]
    # final_latent is potentially on CPU if use_cpu_offload=True
    if len(final_latent.shape) == 4: # If run_sampling returned [C, F, H, W]
        final_latent = final_latent.unsqueeze(0)

    return final_latent

def decode_latent(latent: torch.Tensor, args: argparse.Namespace, cfg) -> torch.Tensor:
    """decode latent tensor to video frames

    Args:
        latent: latent tensor [B, C, F, H, W]
        args: command line arguments (contains _vae instance)
        cfg: model configuration

    Returns:
        torch.Tensor: decoded video tensor [B, C, F, H, W], range [0, 1], on CPU
    """
    device = torch.device(args.device)

    # Load VAE model or use the one from the generation pipeline
    vae = None
    if hasattr(args, "_vae") and args._vae is not None:
        vae = args._vae
        logger.info("Using VAE instance from generation pipeline for decoding.")
    else:
        # Need to load VAE if it wasn't used/stored (e.g., pure T2V or latent input mode)
        logger.info("Loading VAE for decoding...")
        # Attempt to detect DiT dtype even if DiT wasn't loaded (e.g., latent mode)
        # Fallback to bfloat16 if DiT path isn't available
        try:
            dit_dtype_ref = detect_wan_sd_dtype(args.dit) if args.dit else torch.float16
        except: # Handle cases where DiT path is invalid or missing in latent mode
            dit_dtype_ref = torch.bfloat16
            logger.warning("Could not detect DiT dtype for VAE decoding, defaulting to bfloat16.")

        vae_dtype_decode = str_to_dtype(args.vae_dtype) if args.vae_dtype is not None else (torch.bfloat16 if dit_dtype_ref == torch.bfloat16 else torch.float16)
        vae = load_vae(args, cfg, device, vae_dtype_decode)
        args._vae = vae # Store it in case needed again?

    # Ensure VAE is on device for decoding
    vae.to_device(device)

    logger.info(f"Decoding video from latents: shape {latent.shape}, dtype {latent.dtype}")
    # Ensure latent is on the correct device and expected dtype for VAE
    latent_decode = latent.to(device=device, dtype=vae.dtype)

    # Handle different VAE decode APIs
    videos = None
    with torch.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
        if hasattr(vae, 'model') and hasattr(vae, 'scale'):
            # Wan2_2_VAE type - expects list of [C, F, H, W] tensors
            # Convert [1, 48, 21, 44, 80] -> list of [48, 21, 44, 80]
            latent_list = [latent_decode.squeeze(0)]  # Remove batch dim for list
            decoded_list = vae.decode(latent_list)
            if decoded_list and len(decoded_list) > 0:
                # Stack list back into batch dimension: [1, C, F, H, W]
                videos = torch.stack(decoded_list, dim=0)
            else:
                raise RuntimeError("VAE decoding failed or returned empty list.")
        else:
            # Original WanVAE type - handles tensor input directly
            decoded_list = vae.decode(latent_decode)
            if decoded_list and len(decoded_list) > 0:
                videos = torch.stack(decoded_list, dim=0)
            else:
                raise RuntimeError("VAE decoding failed or returned empty list.")


    # Move VAE back to CPU/cache
    vae.to_device(args.vae_cache_cpu if args.vae_cache_cpu else "cpu")
    
    # Explicit cleanup to prevent memory fragmentation
    del latent_decode
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    clean_memory_on_device(device)

    logger.info(f"Decoded video shape: {videos.shape}")

    # Post-processing: trim tail frames, convert to float32 CPU, scale to [0, 1]
    if args.trim_tail_frames > 0:
        logger.info(f"Trimming last {args.trim_tail_frames} frames.")
        videos = videos[:, :, : -args.trim_tail_frames, :, :]

    # Scale from [-1, 1] (VAE output range) to [0, 1] (video save range)
    videos = (videos + 1.0) / 2.0
    videos = torch.clamp(videos, 0.0, 1.0)

    # Move to CPU and convert to float32 for saving
    video_final = videos.cpu().to(torch.float32)
    logger.info(f"Decoding complete. Final video tensor shape: {video_final.shape}")

    return video_final


def save_output(
    video_tensor: torch.Tensor, # Expects [B, C, F, H, W] range [0, 1]
    args: argparse.Namespace,
    original_base_names: Optional[List[str]] = None,
    latent_to_save: Optional[torch.Tensor] = None # Optional latent [B, C, F, H, W]
) -> None:
    """save output video, images, or latent

    Args:
        video_tensor: decoded video tensor [B, C, F, H, W], range [0, 1]
        args: command line arguments
        original_base_names: original base names (if latents are loaded from files)
        latent_to_save: optional raw latent tensor to save
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    seed = args.seed
    # Get dimensions from the *decoded* video tensor
    batch_size, channels, video_length, height, width = video_tensor.shape

    base_name = f"{time_flag}_{seed}"
    if original_base_names:
         # Use first original name if loading multiple latents (though currently unsupported)
         base_name += f"_{original_base_names[0]}"

    # --- Save Latent ---
    if (args.output_type == "latent" or args.output_type == "both") and latent_to_save is not None:
        latent_path = os.path.join(save_path, f"{base_name}_latent.safetensors")
        logger.info(f"Saving latent tensor shape: {latent_to_save.shape}")

        metadata = {}
        if not args.no_metadata:
            # Try to get model paths robustly for metadata
            cfg = WAN_CONFIGS.get(args.task) # Get config if task exists
            dit_path_meta = "N/A"
            if args.dit: dit_path_meta = args.dit
            elif cfg and cfg.dit_checkpoint and args.ckpt_dir: dit_path_meta = os.path.join(args.ckpt_dir, cfg.dit_checkpoint)
            elif cfg and cfg.dit_checkpoint: dit_path_meta = cfg.dit_checkpoint # Use relative path if no ckpt_dir

            vae_path_meta = "N/A"
            if args.vae: vae_path_meta = args.vae
            elif cfg and cfg.vae_checkpoint and args.ckpt_dir: vae_path_meta = os.path.join(args.ckpt_dir, cfg.vae_checkpoint)
            elif cfg and cfg.vae_checkpoint: vae_path_meta = cfg.vae_checkpoint # Use relative path if no ckpt_dir

            metadata = {
                "prompt": f"{args.prompt}",
                "negative_prompt": f"{args.negative_prompt or ''}",
                "seeds": f"{seed}",
                "height": f"{height}", # Use decoded height/width
                "width": f"{width}",
                "video_length": f"{video_length}", # Use decoded length
                "infer_steps": f"{args.infer_steps}",
                "guidance_scale": f"{args.guidance_scale}",
                "flow_shift": f"{args.flow_shift}",
                "task": f"{args.task}",
                "dit_model": f"{dit_path_meta}",
                "vae_model": f"{vae_path_meta}",
                # Add V2V/I2V specific info
                "mode": "V2V" if args.video_path else ("I2V" if args.image_path else ("FunControl" if args.control_path else "T2V")),
            }
            if args.video_path: metadata["v2v_strength"] = f"{args.strength}"
            if args.image_path: metadata["i2v_image"] = f"{os.path.basename(args.image_path)}"
            if args.end_image_path: metadata["i2v_end_image"] = f"{os.path.basename(args.end_image_path)}"
            if args.control_path:
                 metadata["funcontrol_video"] = f"{os.path.basename(args.control_path)}"
                 metadata["funcontrol_weight"] = f"{args.control_weight}"
                 metadata["funcontrol_start"] = f"{args.control_start}"
                 metadata["funcontrol_end"] = f"{args.control_end}"
                 metadata["funcontrol_falloff"] = f"{args.control_falloff_percentage}"
            # Add LoRA info if used
            if args.lora_weight:
                metadata["lora_weights"] = ", ".join([os.path.basename(p) for p in args.lora_weight])
                metadata["lora_multipliers"] = ", ".join(map(str, args.lora_multiplier))


        # Ensure latent is on CPU for saving
        sd = {"latent": latent_to_save.cpu()}
        try:
            save_file(sd, latent_path, metadata=metadata)
            logger.info(f"Latent saved to: {latent_path}")
        except Exception as e:
            logger.error(f"Failed to save latent file: {e}")


    # --- Save Video or Images ---
    if args.output_type == "video" or args.output_type == "both":
        video_path = os.path.join(save_path, f"{base_name}.mp4")
        # save_videos_grid expects [B, T, H, W, C], need to permute and rescale if needed
        # Input video_tensor is [B, C, T, H, W], range [0, 1]
        # save_videos_grid handles the rescale flag correctly if input is [0,1]
        try:
            save_videos_grid(video_tensor, video_path, fps=args.fps, rescale=False) # Pass rescale=False as tensor is already [0,1]
            logger.info(f"Video saved to: {video_path}")
        except Exception as e:
            logger.error(f"Failed to save video file: {e}")
            logger.error(f"Video tensor info: shape={video_tensor.shape}, dtype={video_tensor.dtype}, min={video_tensor.min()}, max={video_tensor.max()}")


    elif args.output_type == "images":
        image_save_dir = os.path.join(save_path, base_name)
        os.makedirs(image_save_dir, exist_ok=True)
        # save_images_grid expects [B, T, H, W, C], need to permute and rescale if needed
        # Input video_tensor is [B, C, T, H, W], range [0, 1]
        # save_images_grid handles the rescale flag correctly if input is [0,1]
        try:
             # Save as individual frames
             save_images_grid(video_tensor, image_save_dir, "frame", rescale=False, save_individually=True) # Pass rescale=False
             logger.info(f"Image frames saved to directory: {image_save_dir}")
        except Exception as e:
            logger.error(f"Failed to save image files: {e}")


def main():
    # --- Argument Parsing & Setup ---
    args = parse_args()
    
    # Handle distributed setup if using sequence parallelism or FSDP
    if args.use_sequence_parallel or args.dit_fsdp or args.t5_fsdp:
        # Get local rank - args.local_rank is set by the parser (converts --local-rank to local_rank)
        local_rank = args.local_rank
        
        # Set CUDA device based on local rank
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            
        # Initialize distributed environment (handled by model managers)
        if args.use_sequence_parallel:
            logger.info(f"Sequence parallel mode detected, local_rank={local_rank}")
        if args.dit_fsdp or args.t5_fsdp:
            logger.info(f"FSDP mode detected, local_rank={local_rank}, dit_fsdp={args.dit_fsdp}, t5_fsdp={args.t5_fsdp}")

    # Determine mode: generation or loading latents
    latents_mode = args.latent_path is not None and len(args.latent_path) > 0

    # Set device
    if args.use_sequence_parallel or args.dit_fsdp or args.t5_fsdp:
        # For distributed modes, use the rank-specific device
        local_rank = args.local_rank
        device_str = f"cuda:{local_rank}"
    else:
        device_str = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    args.device = torch.device(device_str) # Store device back in args
    logger.info(f"Using device: {args.device}")

    generated_latent = None # To hold the generated latent if not in latents_mode
    cfg = WAN_CONFIGS[args.task] # Get config early for potential use
    height, width, video_length = None, None, None # Initialize dimensions
    original_base_names = None # For naming output when loading latents

    if not latents_mode:
        # --- Generation Mode (T2V, I2V, V2V, Fun-Control) ---
        logger.info("Running in Generation Mode")
        # Setup arguments (defaults, etc.)
        args = setup_args(args)
        # Validate inputs (initial check, V2V might refine length later)
        height, width, video_length = check_inputs(args)
        args.video_size = [height, width] # Ensure args reflect checked dimensions
        args.video_length = video_length # May still be None for V2V

        # Determine specific mode string
        mode_str = "Unknown"
        if args.video_path: mode_str = "V2V"
        elif args.image_path and args.control_path: mode_str = "FunControl-I2V" # FunControl overrides if control_path is present
        elif args.control_path: mode_str = "FunControl-T2V"
        elif args.image_path: mode_str = "I2V"
        else: mode_str = "T2V"

        logger.info(f"Mode: {mode_str} (Task: {args.task})")
        logger.info(
            f"Initial settings: video size: {height}x{width}@{video_length or 'auto'} (HxW@F), fps: {args.fps}, "
            f"infer_steps: {args.infer_steps}, guidance: {args.guidance_scale}, flow_shift: {args.flow_shift}"
        )
        if mode_str == "V2V": logger.info(f"V2V Strength: {args.strength}")
        if "FunControl" in mode_str: logger.info(f"FunControl Weight: {args.control_weight}, Start: {args.control_start}, End: {args.control_end}, Falloff: {args.control_falloff_percentage}")

        # --- Pusa Multi-Frame Conditioning Processing ---
        if args.sample_solver == "pusa" and (args.cond_images or args.cond_video or args.end_image):
            logger.info("Processing Pusa multi-frame conditioning...")
            
            # Determine conditioning type and process accordingly
            conditioning_images = []
            cond_positions = []
            noise_multipliers = []
            
            if args.end_image and args.image_path:
                # Start-end frame interpolation mode
                logger.info("Pusa Start-End Frame Mode")
                conditioning_images = process_conditioning_images([args.image_path, args.end_image], width, height)
                
                # Convert pixel frame positions to latent frame positions (VAE has temporal stride)
                # The VAE downsamples temporally, typically by a factor of 4
                vae_temporal_stride = 4  # Standard for most video VAEs
                latent_frames = (video_length - 1) // vae_temporal_stride + 1 if video_length else 21
                
                # Position conditioning at start and end of LATENT space, not pixel space
                cond_positions = [0, latent_frames - 1]  # First and last latent frame
                noise_multipliers = [0.0, 0.2]  # Clean start, some noise on end
                mode_str += "-Pusa-StartEnd"
                
                logger.info(f"Pixel frames: {video_length}, Latent frames: {latent_frames}, Conditioning at latent positions: {cond_positions}")
                
            elif args.cond_images:
                # Multi-frame conditioning mode
                logger.info("Pusa Multi-Frame Conditioning Mode") 
                if not args.cond_positions:
                    raise ValueError("--cond_positions required when using --cond_images")
                if not args.cond_noise_multipliers:
                    raise ValueError("--cond_noise_multipliers required when using --cond_images")
                
                cond_positions = parse_conditioning_positions(args.cond_positions)
                noise_multipliers = parse_noise_multipliers(args.cond_noise_multipliers)
                conditioning_images = process_conditioning_images(args.cond_images, width, height)
                
                validate_conditioning_parameters(video_length or 81, cond_positions, noise_multipliers, conditioning_images)
                mode_str += "-Pusa-MultiFrame"
                
            elif args.cond_video:
                # Video-to-video conditioning mode
                logger.info("Pusa Video-to-Video Mode")
                if not args.cond_positions:
                    raise ValueError("--cond_positions required when using --cond_video")
                if not args.cond_noise_multipliers:
                    raise ValueError("--cond_noise_multipliers required when using --cond_video")
                
                cond_positions = parse_conditioning_positions(args.cond_positions)
                noise_multipliers = parse_noise_multipliers(args.cond_noise_multipliers)
                video_frames = process_conditioning_video(args.cond_video, width, height)
                
                # Automatically use first frame as input image if no input image provided
                if not args.image_path and len(video_frames) > 0:
                    import tempfile
                    import os
                    # Save first frame as temporary input image
                    temp_dir = tempfile.mkdtemp()
                    temp_image_path = os.path.join(temp_dir, "pusa_input_frame.png")
                    video_frames[0].save(temp_image_path)
                    args.image_path = temp_image_path
                    logger.info(f"Auto-extracted first frame as input image: {temp_image_path}")
                
                # Select specific frames based on positions
                conditioning_images = []
                for pos in cond_positions:
                    if pos < len(video_frames):
                        conditioning_images.append(video_frames[pos])
                    else:
                        logger.warning(f"Position {pos} exceeds video length {len(video_frames)}, using last frame")
                        conditioning_images.append(video_frames[-1])
                
                validate_conditioning_parameters(video_length or 81, cond_positions, noise_multipliers, conditioning_images)
                mode_str += "-Pusa-V2V"
            
            # Create conditioning dictionary and frame noise mapping
            args.pusa_conditioning_dict = create_conditioning_dict(cond_positions, conditioning_images, noise_multipliers)
            args.pusa_frame_noise_mapping = create_frame_noise_mapping(cond_positions, noise_multipliers)
            args.pusa_cond_positions = cond_positions
            
            # Log conditioning info
            log_conditioning_info(
                task_type=mode_str,
                cond_positions=cond_positions,
                noise_multipliers=noise_multipliers,
                num_images=len(conditioning_images),
                video_path=args.cond_video if hasattr(args, 'cond_video') else None
            )
            
            # Switch to V2V scheduler if multi-frame conditioning is used
            if len(cond_positions) > 1:
                logger.info("Using Pusa V2V scheduler for multi-frame conditioning")
                args.pusa_use_v2v_scheduler = True
            else:
                args.pusa_use_v2v_scheduler = False

        # Core generation pipeline
        generated_latent = generate(args) # Returns [B, C, F, H, W] or None

        if args.save_merged_model:
            logger.info("Exiting after saving merged model.")
            return # Exit if only saving model

        if generated_latent is None:
             logger.error("Generation failed or was skipped, exiting.")
             return

        # Update dimensions based on the *actual* generated latent
        # Latent shape might differ slightly from input request depending on VAE/model strides
        _, lat_c, lat_f, lat_h, lat_w = generated_latent.shape
        # Convert latent dimensions back to pixel dimensions for metadata/logging
        pixel_height = lat_h * cfg.vae_stride[1]
        pixel_width = lat_w * cfg.vae_stride[2]
        pixel_frames = (lat_f - 1) * cfg.vae_stride[0] + 1
        logger.info(f"Generation complete. Latent shape: {generated_latent.shape} -> Pixel Video: {pixel_height}x{pixel_width}@{pixel_frames}")
        # Use these derived pixel dimensions for saving metadata
        height, width, video_length = pixel_height, pixel_width, pixel_frames


    else:
        # --- Latents Mode (Load and Decode) ---
        logger.info("Running in Latent Loading Mode")
        original_base_names = []
        latents_list = []
        seeds = [] # Try to recover seed from metadata

        # Currently only supporting one latent file input
        if len(args.latent_path) > 1:
            logger.warning("Loading multiple latent files is not fully supported for metadata merging. Using first file's info.")

        latent_path = args.latent_path[0]
        original_base_names.append(os.path.splitext(os.path.basename(latent_path))[0])
        loaded_latent = None
        metadata = {}
        seed = args.seed if args.seed is not None else random.randint(0, 2**32-1) # Default seed if none in metadata

        try:
            if os.path.splitext(latent_path)[1].lower() != ".safetensors":
                logger.warning("Loading non-safetensors latent file. Metadata might be missing.")
                loaded_latent = torch.load(latent_path, map_location="cpu")
                # Attempt to handle different save formats (dict vs raw tensor)
                if isinstance(loaded_latent, dict):
                    if "latent" in loaded_latent:
                        loaded_latent = loaded_latent["latent"]
                    elif "state_dict" in loaded_latent: # Might be a full model checkpoint by mistake
                         raise ValueError("Loaded file appears to be a model checkpoint, not a latent tensor.")
                    else: # Try the first value if it's a tensor
                         first_key = next(iter(loaded_latent))
                         if isinstance(loaded_latent[first_key], torch.Tensor):
                              loaded_latent = loaded_latent[first_key]
                         else:
                              raise ValueError("Could not find latent tensor in loaded dictionary.")
                elif not isinstance(loaded_latent, torch.Tensor):
                     raise ValueError(f"Loaded file content is not a tensor or expected dictionary format: {type(loaded_latent)}")


            else:
                # Load latent tensor
                loaded_latent = load_file(latent_path, device="cpu")["latent"]
                # Load metadata
                with safe_open(latent_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata() or {}
                logger.info(f"Loaded metadata: {metadata}")

                # Restore args from metadata if available AND not overridden by command line
                # Command line args take precedence if provided
                if args.seed is None and "seeds" in metadata: seed = int(metadata["seeds"])
                if "prompt" in metadata: args.prompt = args.prompt or metadata["prompt"] # Keep command line if provided
                if "negative_prompt" in metadata: args.negative_prompt = args.negative_prompt or metadata["negative_prompt"]
                # We need height/width/length to decode, so always load if available
                if "height" in metadata: height = int(metadata["height"])
                if "width" in metadata: width = int(metadata["width"])
                if "video_length" in metadata: video_length = int(metadata["video_length"])
                # Restore other relevant args if not set by user
                if args.guidance_scale == 5.0 and "guidance_scale" in metadata: args.guidance_scale = float(metadata["guidance_scale"]) # Assuming 5.0 is default
                if args.infer_steps is None and "infer_steps" in metadata: args.infer_steps = int(metadata["infer_steps"])
                if args.flow_shift is None and "flow_shift" in metadata: args.flow_shift = float(metadata["flow_shift"])
                if "task" in metadata: args.task = args.task or metadata["task"] # Restore task if not specified
                # FunControl specific args
                if "funcontrol_weight" in metadata: args.control_weight = args.control_weight or float(metadata["funcontrol_weight"])
                if "funcontrol_start" in metadata: args.control_start = args.control_start or float(metadata["funcontrol_start"])
                if "funcontrol_end" in metadata: args.control_end = args.control_end or float(metadata["funcontrol_end"])
                if "funcontrol_falloff" in metadata: args.control_falloff_percentage = args.control_falloff_percentage or float(metadata["funcontrol_falloff"])
                # V2V specific args
                if "v2v_strength" in metadata: args.strength = args.strength or float(metadata["v2v_strength"])

                # Update config based on restored task
                cfg = WAN_CONFIGS[args.task]

            seeds.append(seed)
            latents_list.append(loaded_latent)
            logger.info(f"Loaded latent from {latent_path}. Shape: {loaded_latent.shape}, dtype: {loaded_latent.dtype}")

        except Exception as e:
            logger.error(f"Failed to load latent file {latent_path}: {e}")
            return

        if not latents_list:
            logger.error("No latent tensors were loaded.")
            return

        # Stack latents (currently just one) - ensure batch dimension
        generated_latent = torch.stack(latents_list, dim=0) # [B, C, F, H, W]
        if len(generated_latent.shape) != 5:
             # Maybe saved without batch dim? Try adding it.
             if len(generated_latent.shape) == 4:
                 logger.warning(f"Loaded latent has 4 dimensions {generated_latent.shape}. Adding batch dimension.")
                 generated_latent = generated_latent.unsqueeze(0)
             else:
                 raise ValueError(f"Loaded latent has incorrect shape: {generated_latent.shape}. Expected 4 or 5 dimensions.")

        # Set seed from metadata (or default)
        args.seed = seeds[0]

        # Infer pixel dimensions from latent shape and config if not available in metadata
        _, _, lat_f, lat_h, lat_w = generated_latent.shape # Get dimensions from loaded latent
        if height is None or width is None or video_length is None:
             logger.warning("Dimensions not found in metadata, inferring from latent shape.")
             height = lat_h * cfg.vae_stride[1]
             width = lat_w * cfg.vae_stride[2]
             video_length = (lat_f - 1) * cfg.vae_stride[0] + 1
             logger.info(f"Inferred pixel dimensions: {height}x{width}@{video_length}")
             # Store final dimensions in args for consistency
             args.video_size = [height, width]
             args.video_length = video_length

    # --- Decode and Save ---
    if generated_latent is not None:
        # Unload DiT models before VAE decode to free GPU memory (especially important for fp8_scaled)
        if args.fp8_scaled:
            logger.info("Unloading DiT models before VAE decode to free GPU memory...")
            
            # Unload main model
            if 'model' in locals() and model is not None:
                model.cpu()
                del model
            
            # Unload dual-dit models if present
            if 'model_low' in locals() and model_low is not None:
                model_low.cpu()
                del model_low
            if 'model_high' in locals() and model_high is not None:
                model_high.cpu()
                del model_high
            
            # Unload dynamic model manager if present
            if 'model_manager' in locals() and model_manager is not None:
                model_manager.unload_all()
                del model_manager
            
            # Force memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            clean_memory_on_device(args.device)
            
            # Give GPU time to free memory
            import time
            time.sleep(0.5)
            torch.cuda.empty_cache()
        
        # Decode latent to video tensor [B, C, F, H, W], range [0, 1]
        # Skip VAE decode for extension mode since it already returns decoded pixels
        if hasattr(args, 'extend_video') and args.extend_video is not None:
            logger.info("Extension mode detected - using already decoded video")
            decoded_video = generated_latent  # Already decoded pixels
        else:
            decoded_video = decode_latent(generated_latent, args, cfg)

        # Save the output (latent and/or video/images)
        # Don't save "latents" for extension mode since generated_latent contains pixels
        latent_to_save = None
        if not (hasattr(args, 'extend_video') and args.extend_video is not None):
            latent_to_save = generated_latent if (args.output_type == "latent" or args.output_type == "both") else None
        
        save_output(
            decoded_video,
            args,
            original_base_names=original_base_names,
            latent_to_save=latent_to_save
        )
    else:
        logger.error("No latent available for decoding and saving.")

    logger.info("Done!")


if __name__ == "__main__":
    main()