import argparse
from datetime import datetime
import gc
import random
import os
import re
import time
import math
from typing import Tuple, Optional, List, Union, Any
from pathlib import Path # Added for glob_images in V2V

import torch
import accelerate
from accelerate import Accelerator
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from PIL import Image
import cv2 # Added for V2V video loading/resizing
import numpy as np # Added for V2V video processing
import torchvision.transforms.functional as TF
from tqdm import tqdm

from networks import lora_wan
from utils.safetensors_utils import mem_eff_save_file, load_safetensors
from wan.configs import WAN_CONFIGS, SUPPORTED_SIZES
import wan
from wan.modules.model import WanModel, load_wan_model, detect_wan_sd_dtype
from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel
from wan.modules.clip import CLIPModel
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

try:
    from lycoris.kohya import create_network_from_weights
except:
    pass

from utils.model_utils import str_to_dtype
from utils.device_utils import clean_memory_on_device
# Original load_video/load_images are still needed for Fun-Control / image loading
from hv_generate_video import save_images_grid, save_videos_grid, synchronize_device, load_images as hv_load_images, load_video as hv_load_video

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="Wan 2.1 inference script")

    # WAN arguments
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory (Wan 2.1 official).")
    parser.add_argument("--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run.")
    parser.add_argument(
        "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++", "vanilla"], help="The solver used to sample."
    )

    parser.add_argument("--dit", type=str, default=None, help="DiT checkpoint path")
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
    parser.add_argument(
        "--save_merged_model",
        type=str,
        default=None,
        help="Save merged model to path. If specified, no inference will be performed.",
    )

    # inference
    parser.add_argument("--prompt", type=str, required=True, help="prompt for generation (describe the continuation for extension)")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="negative prompt for generation, use default negative prompt if not specified",
    )
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256], help="video size, height and width")
    parser.add_argument("--video_length", type=int, default=None, help="Total video length (input+generated) for diffusion processing. Default depends on task/mode.")
    parser.add_argument("--fps", type=int, default=16, help="video fps, Default is 16")
    parser.add_argument("--infer_steps", type=int, default=None, help="number of inference steps")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument(
        "--cpu_noise", action="store_true", help="Use CPU to generate noise (compatible with ComfyUI). Default is False."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for classifier free guidance. Default is 5.0.",
    )

    # Modes (mutually exclusive)
    parser.add_argument("--video_path", type=str, default=None, help="path to video for video2video inference (standard Wan V2V)")
    parser.add_argument("--image_path", type=str, default=None, help="path to image for image2video inference")
    parser.add_argument("--extend_video", type=str, default=None, help="path to video for extending it using initial frames")

    # Mode specific args
    parser.add_argument("--strength", type=float, default=0.75, help="Strength for video2video inference (0.0-1.0)")
    parser.add_argument("--end_image_path", type=str, default=None, help="path to end image for image2video or extension inference")
    parser.add_argument("--num_input_frames", type=int, default=4, help="Number of frames from start of --extend_video to use as input (min 1)")
    parser.add_argument("--extend_length", type=int, default=None, help="Number of frames to generate *after* the input frames for --extend_video. Default makes total length match task default (e.g., 81).")


    # Fun-Control argument (distinct from V2V/I2V/Extend)
    parser.add_argument(
        "--control_strength",
        type=float,
        default=1.0,
        help="Strength of control video influence for Fun-Control (1.0 = normal)",
    )
    parser.add_argument(
        "--control_path",
        type=str,
        default=None,
        help="path to control video for inference with controlnet (Fun-Control model only). video file or directory with images",
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

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8")
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

    args = parser.parse_args()

    assert (args.latent_path is None or len(args.latent_path) == 0) or (
        args.output_type == "images" or args.output_type == "video"
    ), "latent_path is only supported for images or video output"

    # --- Mode Exclusivity Checks ---
    modes = [args.video_path, args.image_path, args.extend_video, args.control_path]
    num_modes_set = sum(1 for mode in modes if mode is not None)

    if num_modes_set > 1:
        active_modes = []
        if args.video_path: active_modes.append("--video_path (V2V)")
        if args.image_path: active_modes.append("--image_path (I2V)")
        if args.extend_video: active_modes.append("--extend_video (Extend)")
        if args.control_path: active_modes.append("--control_path (Fun-Control)")
        # Allow Fun-Control + another mode conceptually, but the script logic needs adjustment
        if not (num_modes_set == 2 and args.control_path is not None):
             raise ValueError(f"Only one operation mode can be specified. Found: {', '.join(active_modes)}")
        # Special case: Fun-Control can technically be combined, but let's check task compatibility
        if args.control_path is not None and not WAN_CONFIGS[args.task].is_fun_control:
            raise ValueError("--control_path is provided, but the selected task does not support Fun-Control.")

    # --- Specific Mode Validations ---
    if args.extend_video is not None:
        if args.num_input_frames < 1:
            raise ValueError("--num_input_frames must be at least 1 for video extension.")
        if "t2v" in args.task:
            logger.warning("--extend_video provided, but task is t2v. Using I2V-like conditioning.")
        # We'll set video_length later based on num_input_frames and extend_length

    if args.image_path is not None:
         logger.warning("--image_path is provided. This is standard single-frame I2V.")
         if "t2v" in args.task:
              logger.warning("--image_path provided, but task is t2v. Using I2V conditioning.")

    if args.video_path is not None:
         logger.info("Running in V2V mode.")
         # V2V length is determined later if not specified

    if args.control_path is not None and not WAN_CONFIGS[args.task].is_fun_control:
        raise ValueError("--control_path is provided, but the selected task does not support Fun-Control.")

    return args


def get_task_defaults(task: str, size: Optional[Tuple[int, int]] = None, is_extend_mode: bool = False) -> Tuple[int, float, int, bool]:
    """Return default values for each task

    Args:
        task: task name (t2v, t2i, i2v etc.)
        size: size of the video (width, height)
        is_extend_mode: whether we are in video extension mode

    Returns:
        Tuple[int, float, int, bool]: (infer_steps, flow_shift, video_length, needs_clip)
    """
    width, height = size if size else (0, 0)

    # I2V and Extend mode share similar defaults
    is_i2v_like = "i2v" in task or is_extend_mode

    if "t2i" in task:
        return 50, 5.0, 1, False
    elif is_i2v_like:
        flow_shift = 3.0 if (width == 832 and height == 480) or (width == 480 and height == 832) else 5.0
        return 40, flow_shift, 81, True # Default total length 81
    else:  # t2v or default
        return 50, 5.0, 81, False # Default total length 81


def setup_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate and set default values for optional arguments

    Args:
        args: command line arguments

    Returns:
        argparse.Namespace: updated arguments
    """
    is_extend_mode = args.extend_video is not None

    # Get default values for the task
    default_infer_steps, default_flow_shift, default_video_length, _ = get_task_defaults(args.task, tuple(args.video_size), is_extend_mode)

    # Apply default values to unset arguments
    if args.infer_steps is None:
        args.infer_steps = default_infer_steps
    if args.flow_shift is None:
        args.flow_shift = default_flow_shift

    # --- Video Length Handling ---
    if is_extend_mode:
        if args.extend_length is None:
            # Calculate extend_length to reach the default total length
            args.extend_length = max(1, default_video_length - args.num_input_frames)
            logger.info(f"Defaulting --extend_length to {args.extend_length} to reach total length {default_video_length}")
        # Set the total video_length for processing
        args.video_length = args.num_input_frames + args.extend_length
        if args.video_length <= args.num_input_frames:
             raise ValueError(f"Total video length ({args.video_length}) must be greater than input frames ({args.num_input_frames}). Increase --extend_length.")
    elif args.video_length is None and args.video_path is None: # T2V, I2V (not extend)
        args.video_length = default_video_length
    elif args.video_length is None and args.video_path is not None: # V2V auto-detect
        pass # Delay setting default if V2V and length not specified
    elif args.video_length is not None: # User specified length
        pass

    # Force video_length to 1 for t2i tasks
    if "t2i" in task:
        assert args.video_length == 1, f"video_length should be 1 for task {args.task}"

    # parse slg_layers
    if args.slg_layers is not None:
        args.slg_layers = list(map(int, args.slg_layers.split(",")))

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

    is_extend_mode = args.extend_video is not None
    is_v2v_mode = args.video_path is not None

    # Check supported sizes unless it's V2V/Extend (input video dictates size) or FunControl
    if not is_v2v_mode and not is_extend_mode and not WAN_CONFIGS[args.task].is_fun_control:
        if size not in SUPPORTED_SIZES[args.task]:
            logger.warning(f"Size {size} is not supported for task {args.task}. Supported sizes are {SUPPORTED_SIZES[args.task]}.")

    video_length = args.video_length # Might be None if V2V auto-detect

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    return height, width, video_length


def calculate_dimensions(video_size: Tuple[int, int], video_length: int, config) -> Tuple[Tuple[int, int, int, int], int]:
    """calculate dimensions for the generation

    Args:
        video_size: video frame size (height, width)
        video_length: number of frames in the video being processed
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

    return ((16, lat_f, lat_h, lat_w), seq_len)


# Modified function (replace the original)
def load_vae(args: argparse.Namespace, config, device: torch.device, dtype: torch.dtype) -> WanVAE:
    """load VAE model with robust path handling

    Args:
        args: command line arguments
        config: model configuration
        device: device to use
        dtype: data type for the model

    Returns:
        WanVAE: loaded VAE model
    """
    vae_override_path = args.vae
    vae_filename = config.vae_checkpoint # Get expected filename, e.g., "Wan2.1_VAE.pth"
    # Assume models are in 'wan' dir relative to script if not otherwise specified
    vae_base_dir = "wan"

    final_vae_path = None

    # 1. Check if args.vae is a valid *existing file path*
    if vae_override_path and isinstance(vae_override_path, str) and \
       (vae_override_path.endswith(".pth") or vae_override_path.endswith(".safetensors")) and \
       os.path.isfile(vae_override_path):
        final_vae_path = vae_override_path
        logger.info(f"Using VAE override path from --vae: {final_vae_path}")

    # 2. If override is invalid or not provided, construct default path
    if final_vae_path is None:
        constructed_path = os.path.join(vae_base_dir, vae_filename)
        if os.path.isfile(constructed_path):
            final_vae_path = constructed_path
            logger.info(f"Constructed default VAE path: {final_vae_path}")
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
    """load CLIP model (for I2V / Extend only)

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
    is_i2v_like: bool = False, # Combined flag for I2V and Extend modes
) -> WanModel:
    """load DiT model

    Args:
        args: command line arguments
        config: model configuration
        device: device to use
        dit_dtype: data type for the model
        dit_weight_dtype: data type for the model weights. None for as-is
        is_i2v_like: I2V or Extend mode (might affect some model config details)

    Returns:
        WanModel: loaded DiT model
    """
    loading_device = "cpu"
    if args.blocks_to_swap == 0 and args.lora_weight is None and not args.fp8_scaled:
        loading_device = device

    loading_weight_dtype = dit_weight_dtype
    if args.fp8_scaled or args.lora_weight is not None:
        loading_weight_dtype = dit_dtype  # load as-is

    # do not fp8 optimize because we will merge LoRA weights
    # Pass the is_i2v_like flag if the underlying loading function uses it
    model = load_wan_model(config, device, args.dit, args.attn_mode, False, loading_device, loading_weight_dtype, is_i2v_like)

    return model


def merge_lora_weights(model: WanModel, args: argparse.Namespace, device: torch.device) -> None:
    """merge LoRA weights to the model

    Args:
        model: DiT model
        args: command line arguments
        device: device to use
    """
    if args.lora_weight is None or len(args.lora_weight) == 0:
        return

    for i, lora_weight in enumerate(args.lora_weight):
        if args.lora_multiplier is not None and len(args.lora_multiplier) > i:
            lora_multiplier = args.lora_multiplier[i]
        else:
            lora_multiplier = 1.0

        logger.info(f"Loading LoRA weights from {lora_weight} with multiplier {lora_multiplier}")
        weights_sd = load_file(lora_weight)

        # apply include/exclude patterns
        original_key_count = len(weights_sd.keys())
        if args.include_patterns is not None and len(args.include_patterns) > i:
            include_pattern = args.include_patterns[i]
            regex_include = re.compile(include_pattern)
            weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
            logger.info(f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}")
        if args.exclude_patterns is not None and len(args.exclude_patterns) > i:
            original_key_count_ex = len(weights_sd.keys())
            exclude_pattern = args.exclude_patterns[i]
            regex_exclude = re.compile(exclude_pattern)
            weights_sd = {k: v for k, v in weights_sd.items() if not regex_exclude.search(k)}
            logger.info(
                f"Filtered keys with exclude pattern {exclude_pattern}: {original_key_count_ex} -> {len(weights_sd.keys())}"
            )
        if len(weights_sd) != original_key_count:
            remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys()]))
            remaining_keys.sort()
            logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
            if len(weights_sd) == 0:
                logger.warning(f"No keys left after filtering.")

        if args.lycoris:
            lycoris_net, _ = create_network_from_weights(
                multiplier=lora_multiplier,
                file=None,
                weights_sd=weights_sd,
                unet=model,
                text_encoder=None,
                vae=None,
                for_inference=True,
            )
            lycoris_net.merge_to(None, model, weights_sd, dtype=None, device=device)
        else:
            network = lora_wan.create_arch_network_from_weights(lora_multiplier, weights_sd, unet=model, for_inference=True)
            network.merge_to(None, model, weights_sd, device=device, non_blocking=True)

        synchronize_device(device)
        logger.info("LoRA weights loaded")

    # save model here before casting to dit_weight_dtype
    if args.save_merged_model:
        logger.info(f"Saving merged model to {args.save_merged_model}")
        mem_eff_save_file(model.state_dict(), args.save_merged_model)  # save_file needs a lot of memory
        logger.info("Merged model saved")


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
    # T2V/FunControl length should be set by setup_args
    frames = args.video_length
    if frames is None:
         raise ValueError("video_length must be determined before calling prepare_t2v_inputs")

    (_, lat_f, lat_h, lat_w), seq_len = calculate_dimensions(args.video_size, frames, config)
    target_shape = (16, lat_f, lat_h, lat_w) # Latent channel dim is 16

    # configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

    # set seed
    seed = args.seed # Seed should be set in generate()
    if not args.cpu_noise:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
    else:
        
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

    # Fun-Control: encode control video to latent space
    y = None
    if config.is_fun_control and args.control_path:
         if vae is None:
             raise ValueError("VAE must be provided for Fun-Control input preparation.")
         logger.info(f"Encoding control video for Fun-Control")
         control_video = load_control_video(args.control_path, frames, height, width).to(device)
         vae.to_device(device)
         with accelerator.autocast(), torch.no_grad():
              y = vae.encode([control_video])[0] # Encode video
              y = y * args.control_strength # Apply strength
         vae.to_device("cpu" if args.vae_cache_cpu else "cpu") # Move VAE back
         clean_memory_on_device(device)
         logger.info(f"Fun-Control conditioning 'y' shape: {y.shape}")

    # generate noise
    noise = torch.randn(target_shape, dtype=torch.float32, generator=seed_g, device=device if not args.cpu_noise else "cpu")
    noise = noise.to(device)

    # prepare model input arguments
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}
    if y is not None: # Add 'y' only if Fun-Control generated it
        arg_c["y"] = [y]
        arg_null["y"] = [y]

    return noise, context, context_null, (arg_c, arg_null)


def load_video_frames(video_path: str, num_frames: int, target_reso: Tuple[int, int]) -> Tuple[List[np.ndarray], torch.Tensor]:
    """Load the first N frames from a video, resize, return numpy list and normalized tensor.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to load from the start.
        target_reso (Tuple[int, int]): Target resolution (height, width).

    Returns:
        Tuple[List[np.ndarray], torch.Tensor]:
            - List of numpy arrays (frames) in HWC, RGB, uint8 format.
            - Tensor of shape [C, F, H, W], float32, range [0, 1].
    """
    logger.info(f"Loading first {num_frames} frames from {video_path}, target reso {target_reso}")
    target_h, target_w = target_reso

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    # Get total frame count and check if enough frames exist
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        cap.release()
        raise ValueError(f"Video has only {total_frames} frames, but {num_frames} were requested for input.")

    # Read frames
    frames_np = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Could only read {len(frames_np)} frames out of {num_frames} requested from {video_path}.")
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        current_h, current_w = frame_rgb.shape[:2]
        interpolation = cv2.INTER_AREA if target_h * target_w < current_h * current_w else cv2.INTER_LANCZOS4
        frame_resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=interpolation)

        frames_np.append(frame_resized)

    cap.release()

    if len(frames_np) != num_frames:
         raise RuntimeError(f"Failed to load the required {num_frames} frames.")

    # Convert list of numpy arrays to tensor [F, H, W, C] -> [C, F, H, W], range [0, 1]
    frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).permute(0, 3, 1, 2).float() / 255.0
    frames_tensor = frames_tensor.permute(1, 0, 2, 3) # [C, F, H, W]

    logger.info(f"Loaded {len(frames_np)} input frames. Tensor shape: {frames_tensor.shape}")

    # Return both the original numpy frames (for saving later) and the normalized tensor
    return frames_np, frames_tensor


# Combined function for I2V and Extend modes
def prepare_i2v_or_extend_inputs(
    args: argparse.Namespace, config, accelerator: Accelerator, device: torch.device, vae: WanVAE,
    input_frames_tensor: Optional[torch.Tensor] = None # Required for Extend mode
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
    """Prepare inputs for I2V (single image) or Extend (multiple frames)."""
    if vae is None:
        raise ValueError("VAE must be provided for I2V/Extend input preparation.")

    is_extend_mode = input_frames_tensor is not None
    is_i2v_mode = args.image_path is not None

    # --- Get Dimensions and Frame Counts ---
    height, width = args.video_size
    frames = args.video_length # Total frames for diffusion process
    if frames is None:
         raise ValueError("video_length must be set before calling prepare_i2v_or_extend_inputs")

    num_input_frames = 0
    if is_extend_mode:
        num_input_frames = args.num_input_frames
        if num_input_frames >= frames:
             raise ValueError(f"Number of input frames ({num_input_frames}) must be less than total video length ({frames})")
    elif is_i2v_mode:
        num_input_frames = 1

    # --- Load Input Image(s) / Frames ---
    img_tensor_for_clip = None # Representative tensor for CLIP
    img_tensor_for_vae = None # Tensor containing all input frames/image for VAE

    if is_extend_mode:
        # Input frames tensor already provided (normalized [0,1])
        img_tensor_for_vae = input_frames_tensor.to(device)
        # Use first frame for CLIP
        img_tensor_for_clip = img_tensor_for_vae[:, 0:1, :, :] # [C, 1, H, W]
        logger.info(f"Preparing inputs for Extend mode with {num_input_frames} input frames.")

    elif is_i2v_mode:
        # Load single image
        img = Image.open(args.image_path).convert("RGB")
        img_cv2 = np.array(img)
        interpolation = cv2.INTER_AREA if height < img_cv2.shape[0] else cv2.INTER_CUBIC
        img_resized_np = cv2.resize(img_cv2, (width, height), interpolation=interpolation)
        # Normalized [0,1], shape [C, H, W]
        img_tensor_single = TF.to_tensor(img_resized_np).to(device)
        # Add frame dimension -> [C, 1, H, W]
        img_tensor_for_vae = img_tensor_single.unsqueeze(1)
        img_tensor_for_clip = img_tensor_for_vae
        logger.info("Preparing inputs for standard I2V mode.")

    else:
        raise ValueError("Neither extend_video nor image_path provided for I2V/Extend preparation.")

    # --- Optional End Frame ---
    has_end_image = args.end_image_path is not None
    end_img_tensor_vae = None # Normalized [-1, 1], shape [C, 1, H, W]
    if has_end_image:
        end_img = Image.open(args.end_image_path).convert("RGB")
        end_img_cv2 = np.array(end_img)
        interpolation_end = cv2.INTER_AREA if height < end_img_cv2.shape[0] else cv2.INTER_CUBIC
        end_img_resized_np = cv2.resize(end_img_cv2, (width, height), interpolation=interpolation_end)
        # Normalized [0,1], shape [C, H, W] -> [C, 1, H, W]
        end_img_tensor_load = TF.to_tensor(end_img_resized_np).unsqueeze(1).to(device)
        end_img_tensor_vae = (end_img_tensor_load * 2.0 - 1.0) # Scale to [-1, 1] for VAE
        logger.info(f"Loaded end image: {args.end_image_path}")

    # --- Calculate Latent Dimensions ---
    lat_f = (frames - 1) // config.vae_stride[0] + 1 # Total latent frames
    lat_h = height // config.vae_stride[1]
    lat_w = width // config.vae_stride[2]
    # Latent frames corresponding to the input pixel frames
    lat_input_f = (num_input_frames - 1) // config.vae_stride[0] + 1

    max_seq_len = math.ceil((lat_f + (1 if has_end_image else 0)) * lat_h * lat_w / (config.patch_size[1] * config.patch_size[2]))
    logger.info(f"Target latent shape: ({lat_f}, {lat_h}, {lat_w}), Input latent frames: {lat_input_f}, Seq len: {max_seq_len}")

    # --- Set Seed ---
    seed = args.seed
    seed_g = torch.Generator(device=device) if not args.cpu_noise else torch.manual_seed(seed)
    if not args.cpu_noise:
        seed_g.manual_seed(seed)

    # --- Generate Noise ---
    # Noise for the *entire* processing duration (including input frame slots)
    noise = torch.randn(
        16, lat_f + (1 if has_end_image else 0), lat_h, lat_w,
        dtype=torch.float32, generator=seed_g, device=device if not args.cpu_noise else "cpu"
    ).to(device)

    # --- Text Encoding ---
    n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt
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
    del text_encoder
    clean_memory_on_device(device)

    # --- CLIP Encoding ---
    clip = load_clip_model(args, config, device)
    clip.model.to(device)
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
        # Input needs to be [-1, 1], shape [C, 1, H, W] (or maybe [C, F, H, W] if model supports?)
        # Assuming visual encoder takes one frame: use the representative clip tensor
        clip_input = img_tensor_for_clip.sub_(0.5).div_(0.5) # Scale [0,1] -> [-1,1]
        clip_context = clip.visual([clip_input]) # Pass as list [tensor]
    del clip
    clean_memory_on_device(device)

    # --- VAE Encoding for Conditioning Tensor 'y' ---
    vae.to_device(device)
    y_latent_part = torch.zeros(config.latent_channels, lat_f + (1 if has_end_image else 0), lat_h, lat_w, device=device, dtype=vae.dtype)

    with accelerator.autocast(), torch.no_grad():
        # Encode the input frames/image (scale [0,1] -> [-1,1])
        input_frames_vae = (img_tensor_for_vae * 2.0 - 1.0).to(dtype=vae.dtype) # [-1, 1]
        # Pad with zeros if needed to match VAE chunking? Assume encode handles variable length for now.
        encoded_input_latents = vae.encode([input_frames_vae])[0] # [C', F_in', H', W']
        actual_encoded_input_f = encoded_input_latents.shape[1]
        if actual_encoded_input_f > lat_input_f:
            logger.warning(f"VAE encoded {actual_encoded_input_f} frames, expected {lat_input_f}. Truncating.")
            encoded_input_latents = encoded_input_latents[:, :lat_input_f, :, :]
        elif actual_encoded_input_f < lat_input_f:
             logger.warning(f"VAE encoded {actual_encoded_input_f} frames, expected {lat_input_f}. Padding needed for mask.")
             # This case shouldn't happen if lat_input_f calculation is correct, but handle defensively

        # Place encoded input latents into the full y tensor
        y_latent_part[:, :actual_encoded_input_f, :, :] = encoded_input_latents

        # Encode end image if present
        if has_end_image and end_img_tensor_vae is not None:
            encoded_end_latent = vae.encode([end_img_tensor_vae.to(dtype=vae.dtype)])[0] # [C', 1, H', W']
            y_latent_part[:, -1:, :, :] = encoded_end_latent # Place at the end

    # --- Create Mask ---
    msk = torch.zeros(4, lat_f + (1 if has_end_image else 0), lat_h, lat_w, device=device, dtype=vae.dtype)
    msk[:, :lat_input_f, :, :] = 1  # Mask the input frames
    if has_end_image:
        msk[:, -1:, :, :] = 1 # Mask the end frame

    # --- Combine Mask and Latent Part for 'y' ---
    y = torch.cat([msk, y_latent_part], dim=0) # Shape [4+C', F_total', H', W']
    logger.info(f"Constructed conditioning 'y' tensor shape: {y.shape}")

    # --- Fun-Control Integration (Optional, might need adjustment for Extend mode) ---
    if config.is_fun_control and args.control_path:
        logger.warning("Fun-Control with Extend mode is experimental. Control signal might conflict with input frames.")
        control_video = load_control_video(args.control_path, frames + (1 if has_end_image else 0), height, width).to(device)
        with accelerator.autocast(), torch.no_grad():
            control_latent = vae.encode([control_video])[0] # Encode control video
            control_latent = control_latent * args.control_strength # Apply strength

        # How to combine? Replace y? Add? For now, let's assume control replaces the VAE part of y
        y = torch.cat([msk, control_latent], dim=0) # Overwrite latent part with control
        logger.info(f"Replaced latent part of 'y' with Fun-Control latent. New 'y' shape: {y.shape}")


    vae.to_device("cpu" if args.vae_cache_cpu else "cpu") # Move VAE back
    clean_memory_on_device(device)

    # --- Prepare Model Input Dictionaries ---
    arg_c = {
        "context": [context[0]], # Needs list format? Check model forward
        "clip_fea": clip_context,
        "seq_len": max_seq_len,
        "y": [y], # Pass conditioning tensor y
    }

    arg_null = {
        "context": context_null,
        "clip_fea": clip_context,
        "seq_len": max_seq_len,
        "y": [y], # Pass conditioning tensor y
    }

    return noise, context, context_null, y, (arg_c, arg_null)


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
        current_h, current_w = frame_rgb.shape[:2]
        interpolation = cv2.INTER_AREA if target_h * target_w < current_h * current_w else cv2.INTER_LANCZOS4
        frame_resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=interpolation)

        frames.append(frame_resized)

    cap.release()
    actual_frames_loaded = len(frames)
    logger.info(f"Successfully loaded and resized {actual_frames_loaded} frames for V2V.")

    return frames, actual_frames_loaded


def encode_video_to_latents(video_tensor: torch.Tensor, vae: WanVAE, device: torch.device, vae_dtype: torch.dtype, args: argparse.Namespace) -> torch.Tensor:
    """Encode video tensor to latent space using VAE for V2V.

    Args:
        video_tensor (torch.Tensor): Video tensor with shape [B, C, F, H, W], values in [-1, 1].
        vae (WanVAE): VAE model instance.
        device (torch.device): Device to perform encoding on.
        vae_dtype (torch.dtype): Target dtype for the output latents.
        args (argparse.Namespace): Command line arguments (needed for vae_cache_cpu).

    Returns:
        torch.Tensor: Encoded latents with shape [B, C', F', H', W'].
    """
    if vae is None:
        raise ValueError("VAE must be provided for video encoding.")

    logger.info(f"Encoding video tensor to latents: input shape {video_tensor.shape}")

    # Ensure VAE is on the correct device
    vae.to_device(device)

    # Prepare video tensor: move to device, ensure correct dtype
    video_tensor = video_tensor.to(device=device, dtype=vae.dtype) # Use VAE's dtype

    # WanVAE expects input as a list of [C, F, H, W] tensors (no batch dim)
    latents_list = []
    batch_size = video_tensor.shape[0]
    for i in range(batch_size):
        video_single = video_tensor[i] # Shape [C, F, H, W]
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vae.dtype):
            encoded_latent = vae.encode([video_single])[0] # Returns tensor [C', F', H', W']
            latents_list.append(encoded_latent)

    # Stack results back into a batch
    latents = torch.stack(latents_list, dim=0) # Shape [B, C', F', H', W']

    # Move VAE back to CPU (or cache device)
    vae_target_device = torch.device("cpu") if not args.vae_cache_cpu else torch.device("cpu")
    if args.vae_cache_cpu: logger.info("Moving VAE to CPU for caching.")
    else: logger.info("Moving VAE to CPU after encoding.")
    vae.to_device(vae_target_device)
    clean_memory_on_device(device)

    # Convert latents to the desired final dtype (e.g., bfloat16 for DiT)
    latents = latents.to(dtype=vae_dtype) # Use the target vae_dtype passed to function
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

    # Calculate target shape and sequence length based on actual latent dimensions
    target_shape = video_latents.shape[1:] # [C', F', H', W']
    (_, _, _), seq_len = calculate_dimensions((args.video_size[0], args.video_size[1]), args.video_length, config) # Use original args to get seq_len
    # (_, _, _), seq_len = calculate_dimensions((lat_h * config.vae_stride[1], lat_w * config.vae_stride[2]), (lat_f-1)*config.vae_stride[0]+1, config) # Recalculate seq_len from latent dims

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

    # Generate noise with the same shape as video_latents (including batch dimension)
    noise = torch.randn(
        video_latents.shape, # [B, C', F', H', W']
        dtype=torch.float32,
        device=device if not args.cpu_noise else "cpu",
        generator=seed_g
    )
    noise = noise.to(device) # Ensure noise is on the target device

    # Prepare model input arguments (context needs to match batch size of latents)
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}

    # V2V does not use 'y' or 'clip_fea' in the standard Wan model case

    return noise, context, context_null, (arg_c, arg_null)


# --- End V2V Helper Functions ---

def load_control_video(control_path: str, frames: int, height: int, width: int) -> torch.Tensor:
    """load control video to pixel space for Fun-Control model

    Args:
        control_path: path to control video
        frames: number of frames in the video
        height: height of the video
        width: width of the video

    Returns:
        torch.Tensor: control video tensor, CFHW, range [-1, 1]
    """
    logger.info(f"Load control video for Fun-Control from {control_path}")

    # Use the original helper from hv_generate_video for consistency
    if os.path.isfile(control_path):
        # Use hv_load_video which returns list of numpy arrays (HWC, 0-255)
        # NOTE: hv_load_video takes (W, H) for bucket_reso!
        video_frames_np = hv_load_video(control_path, 0, frames, bucket_reso=(width, height))
    elif os.path.isdir(control_path):
         # Use hv_load_images which returns list of numpy arrays (HWC, 0-255)
         # NOTE: hv_load_images takes (W, H) for bucket_reso!
        video_frames_np = hv_load_images(control_path, frames, bucket_reso=(width, height))
    else:
        raise FileNotFoundError(f"Control path not found: {control_path}")

    if not video_frames_np:
         raise ValueError(f"No frames loaded from control path: {control_path}")
    if len(video_frames_np) < frames:
        logger.warning(f"Control video has {len(video_frames_np)} frames, less than requested {frames}. Using available frames and repeating last.")
        # Repeat last frame to match length
        last_frame = video_frames_np[-1]
        video_frames_np.extend([last_frame] * (frames - len(video_frames_np)))

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
                 # logger.warning("Scheduler step does not support generator argument, proceeding without it.") # Reduce noise
                 return org_step(model_output, timestep, sample, return_dict=return_dict)


        scheduler.step = step_wrapper
    else:
        raise NotImplementedError(f"Unsupported solver: {args.sample_solver}")

    logger.info(f"Using scheduler: {args.sample_solver}, timesteps shape: {timesteps.shape}")
    return scheduler, timesteps


def run_sampling(
    model: WanModel,
    noise: torch.Tensor, # This might be pure noise (T2V/I2V/Extend) or mixed noise+latent (V2V)
    scheduler: Any,
    timesteps: torch.Tensor, # Might be a subset for V2V
    args: argparse.Namespace,
    inputs: Tuple[dict, dict], # (arg_c, arg_null)
    device: torch.device,
    seed_g: torch.Generator,
    accelerator: Accelerator,
    use_cpu_offload: bool = True,
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
        use_cpu_offload: Whether to offload tensors to CPU during processing
    Returns:
        torch.Tensor: generated latent
    """
    arg_c, arg_null = inputs

    # Ensure inputs (context, y, etc.) are correctly formatted (e.g., lists if model expects list input)
    # Example: ensure context is list [tensor] if model expects list
    if isinstance(arg_c.get("context"), torch.Tensor):
        arg_c["context"] = [arg_c["context"]]
    if isinstance(arg_null.get("context"), torch.Tensor):
        arg_null["context"] = [arg_null["context"]]
    # Similar checks/conversions for other keys like 'y' if needed based on WanModel.forward signature


    latent = noise # Initialize latent state [B, C, F, H, W]
    latent_storage_device = device if not use_cpu_offload else "cpu"
    latent = latent.to(latent_storage_device) # Move initial state to storage device

    # cfg skip logic
    apply_cfg_array = []
    num_timesteps = len(timesteps)

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
        # Latent should be [B, C, F, H, W]
        # Model expects latent input 'x' as list: [tensor]
        latent_on_device = latent.to(device)
        latent_model_input_list = [latent_on_device] # Wrap in list
        timestep = torch.stack([t]).to(device) # Ensure timestep is a tensor on device

        with accelerator.autocast(), torch.no_grad():
            # 1. Predict conditional noise estimate
            noise_pred_cond = model(x=latent_model_input_list, t=timestep, **arg_c)[0]
            noise_pred_cond = noise_pred_cond.to(latent_storage_device)

            # 2. Predict unconditional noise estimate (potentially with SLG)
            apply_cfg = apply_cfg_array[i]
            if apply_cfg:
                apply_slg_step = apply_slg_global and (i >= slg_start_step and i < slg_end_step)
                slg_indices_for_call = args.slg_layers if apply_slg_step else None
                uncond_input_args = arg_null

                if apply_slg_step and args.slg_mode == "original":
                    # Standard uncond prediction first
                    noise_pred_uncond = model(x=latent_model_input_list, t=timestep, **uncond_input_args)[0].to(latent_storage_device)
                    # SLG prediction (skipping layers in uncond)
                    skip_layer_out = model(x=latent_model_input_list, t=timestep, skip_block_indices=slg_indices_for_call, **uncond_input_args)[0].to(latent_storage_device)
                    # Combine: scaled = uncond + scale * (cond - uncond) + slg_scale * (cond - skip)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    noise_pred = noise_pred + args.slg_scale * (noise_pred_cond - skip_layer_out)

                elif apply_slg_step and args.slg_mode == "uncond":
                    # SLG prediction (skipping layers in uncond) replaces standard uncond
                    noise_pred_uncond = model(x=latent_model_input_list, t=timestep, skip_block_indices=slg_indices_for_call, **uncond_input_args)[0].to(latent_storage_device)
                    # Combine: scaled = slg_uncond + scale * (cond - slg_uncond)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                else:
                    # Regular CFG (no SLG or SLG not active this step)
                    noise_pred_uncond = model(x=latent_model_input_list, t=timestep, **uncond_input_args)[0].to(latent_storage_device)
                    # Combine: scaled = uncond + scale * (cond - uncond)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # CFG is skipped for this step, use conditional prediction directly
                noise_pred = noise_pred_cond

            # 3. Compute previous sample state with the scheduler
            # Scheduler expects noise_pred [B, C, F, H, W] and latent [B, C, F, H, W]
            scheduler_output = scheduler.step(
                noise_pred.to(device), # Ensure noise_pred is on compute device
                t,
                latent_on_device, # Pass the tensor directly
                return_dict=False,
                generator=seed_g # Pass generator
            )
            prev_latent = scheduler_output[0] # Get the new latent state [B, C, F, H, W]

            # 4. Update latent state (move back to storage device)
            latent = prev_latent.to(latent_storage_device)

    # Return the final denoised latent (should be on storage device)
    logger.info("Sampling loop finished.")
    return latent


def generate(args: argparse.Namespace) -> Tuple[Optional[torch.Tensor], Optional[List[np.ndarray]]]:
    """main function for generation pipeline (T2V, I2V, V2V, Extend)

    Args:
        args: command line arguments

    Returns:
        Tuple[Optional[torch.Tensor], Optional[List[np.ndarray]]]:
            - generated latent tensor [B, C, F, H, W], or None if error/skipped.
            - list of original input frames (numpy HWC RGB uint8) if in Extend mode, else None.
    """
    device = torch.device(args.device)
    cfg = WAN_CONFIGS[args.task]

    # --- Determine Mode ---
    is_extend_mode = args.extend_video is not None
    is_i2v_mode = args.image_path is not None and not is_extend_mode
    is_v2v_mode = args.video_path is not None
    is_fun_control = args.control_path is not None and cfg.is_fun_control # Can overlap
    is_t2v_mode = not is_extend_mode and not is_i2v_mode and not is_v2v_mode and not is_fun_control

    mode_str = ("Extend" if is_extend_mode else
                "I2V" if is_i2v_mode else
                "V2V" if is_v2v_mode else
                "T2V" + ("+FunControl" if is_fun_control else ""))
    if is_fun_control and not is_t2v_mode: # If funcontrol combined with other modes
         mode_str += "+FunControl"
    logger.info(f"Running in {mode_str} mode")

    # --- Data Types ---
    dit_dtype = detect_wan_sd_dtype(args.dit) if args.dit is not None else torch.bfloat16
    if dit_dtype.itemsize == 1:
        dit_dtype = torch.bfloat16
        if args.fp8_scaled: raise ValueError("Cannot use --fp8_scaled with pre-quantized FP8 weights.")
        dit_weight_dtype = None
    elif args.fp8_scaled: dit_weight_dtype = None
    elif args.fp8: dit_weight_dtype = torch.float8_e4m3fn
    else: dit_weight_dtype = dit_dtype

    vae_dtype = str_to_dtype(args.vae_dtype) if args.vae_dtype is not None else (torch.bfloat16 if dit_dtype == torch.bfloat16 else torch.float16)
    logger.info(
        f"Using device: {device}, DiT compute: {dit_dtype}, DiT weight: {dit_weight_dtype or 'Mixed (FP8 Scaled)' if args.fp8_scaled else dit_dtype}, VAE: {vae_dtype}, T5 FP8: {args.fp8_t5}"
    )

    # --- Accelerator ---
    mixed_precision = "bf16" if dit_dtype == torch.bfloat16 else "fp16"
    accelerator = accelerate.Accelerator(mixed_precision=mixed_precision)

    # --- Seed ---
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed
    logger.info(f"Using seed: {seed}")

    # --- Load VAE (if needed for input processing) ---
    vae = None
    needs_vae_early = is_extend_mode or is_i2v_mode or is_v2v_mode or is_fun_control
    if needs_vae_early:
        vae = load_vae(args, cfg, device, vae_dtype)

    # --- Prepare Inputs ---
    noise = None
    context = None
    context_null = None
    inputs = None
    video_latents = None # For V2V mixing
    original_input_frames_np = None # For Extend mode saving

    if is_extend_mode:
        # 1. Load initial frames (numpy list and normalized tensor)
        original_input_frames_np, input_frames_tensor = load_video_frames(
            args.extend_video, args.num_input_frames, tuple(args.video_size)
        )
        # 2. Prepare inputs using the loaded frames tensor
        noise, context, context_null, _, inputs = prepare_i2v_or_extend_inputs(
            args, cfg, accelerator, device, vae, input_frames_tensor=input_frames_tensor
        )
        del input_frames_tensor # Free memory
        clean_memory_on_device(device)

    elif is_i2v_mode:
        # Prepare I2V inputs (single image)
        noise, context, context_null, _, inputs = prepare_i2v_or_extend_inputs(
            args, cfg, accelerator, device, vae
        )

    elif is_v2v_mode:
        # 1. Load and prepare video
        video_frames_np, actual_frames_loaded = load_video(
            args.video_path, start_frame=0, num_frames=args.video_length, bucket_reso=tuple(args.video_size)
        )
        if actual_frames_loaded == 0: raise ValueError(f"Could not load frames from video: {args.video_path}")
        if args.video_length is None or actual_frames_loaded < args.video_length:
            logger.info(f"Updating video_length based on loaded V2V frames: {actual_frames_loaded}")
            args.video_length = actual_frames_loaded
        height, width, video_length = check_inputs(args) # Re-check

        # Convert frames np [F,H,W,C] uint8 -> tensor [1,C,F,H,W] float32 [-1, 1]
        video_tensor = torch.from_numpy(np.stack(video_frames_np, axis=0))
        video_tensor = video_tensor.permute(0, 3, 1, 2).float() # F,C,H,W
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0) # 1,C,F,H,W
        video_tensor = video_tensor / 127.5 - 1.0 # Normalize to [-1, 1]

        # 2. Encode video to latents (pass vae_dtype for DiT compatibility)
        video_latents = encode_video_to_latents(video_tensor, vae, device, vae_dtype, args)
        del video_tensor, video_frames_np
        clean_memory_on_device(device)

        # 3. Prepare V2V inputs (noise, context, etc.)
        noise, context, context_null, inputs = prepare_v2v_inputs(args, cfg, accelerator, device, video_latents)

    elif is_t2v_mode or is_fun_control: # Should handle T2V+FunControl here
        # Prepare T2V inputs (passes VAE if is_fun_control)
        if args.video_length is None:
             raise ValueError("video_length must be specified for T2V/Fun-Control.")
        noise, context, context_null, inputs = prepare_t2v_inputs(args, cfg, accelerator, device, vae if is_fun_control else None)

    # At this point, VAE should be on CPU/cache unless still needed for decoding

    # --- Load DiT Model ---
    is_i2v_like = is_i2v_mode or is_extend_mode
    model = load_dit_model(args, cfg, device, dit_dtype, dit_weight_dtype, is_i2v_like)

    # --- Merge LoRA ---
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        merge_lora_weights(model, args, device)
        if args.save_merged_model:
            logger.info("Merged model saved. Exiting without generation.")
            return None, None

    # --- Optimize Model ---
    optimize_model(model, args, device, dit_dtype, dit_weight_dtype)

    # --- Setup Scheduler & Timesteps ---
    scheduler, timesteps = setup_scheduler(args, cfg, device)

    # --- Prepare for Sampling ---
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    latent = noise # Start with noise (correctly shaped for T2V/I2V/Extend)

    # --- V2V Strength Adjustment ---
    if is_v2v_mode and args.strength < 1.0:
        if video_latents is None: raise RuntimeError("video_latents not available for V2V strength.")
        num_inference_steps = max(1, int(args.infer_steps * args.strength))
        logger.info(f"V2V Strength: {args.strength}, adjusting inference steps to {num_inference_steps}")
        t_start_idx = len(timesteps) - num_inference_steps
        if t_start_idx < 0: t_start_idx = 0
        t_start = timesteps[t_start_idx]
        # Use scheduler.add_noise for proper mixing
        video_latents = video_latents.to(device=noise.device, dtype=noise.dtype)
        latent = scheduler.add_noise(video_latents, noise, t_start.unsqueeze(0).expand(noise.shape[0])) # Add noise based on start time
        latent = latent.to(noise.dtype) # Ensure correct dtype after add_noise
        logger.info(f"Mixed noise and video latents using scheduler.add_noise at timestep {t_start.item():.1f}")
        timesteps = timesteps[t_start_idx:] # Use subset of timesteps
        logger.info(f"Using last {len(timesteps)} timesteps for V2V sampling.")
    else:
         logger.info(f"Using full {len(timesteps)} timesteps for sampling.")
         # Latent remains the initial noise (already handles I2V/Extend via 'y' conditioning)


    # --- Run Sampling Loop ---
    logger.info("Starting denoising sampling loop...")
    final_latent = run_sampling(
        model, latent, scheduler, timesteps, args, inputs, device, seed_g, accelerator,
        use_cpu_offload=(args.blocks_to_swap > 0)
    )

    # --- Cleanup ---
    del model, scheduler, context, context_null, inputs
    if video_latents is not None: del video_latents
    synchronize_device(device)
    if args.blocks_to_swap > 0:
        logger.info("Waiting 5 seconds for block swap cleanup...")
        time.sleep(5)
    gc.collect()
    clean_memory_on_device(device)

    # Store VAE instance for decoding
    args._vae = vae

    # Return latent [B, C, F, H, W] and original frames if extending
    if len(final_latent.shape) == 4: final_latent = final_latent.unsqueeze(0)
    return final_latent, original_input_frames_np


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
    vae = None
    if hasattr(args, "_vae") and args._vae is not None:
        vae = args._vae
        logger.info("Using VAE instance from generation pipeline for decoding.")
    else:
        logger.info("Loading VAE for decoding...")
        vae_dtype_decode = str_to_dtype(args.vae_dtype) if args.vae_dtype is not None else torch.bfloat16 # Default bfloat16 if not specified
        vae = load_vae(args, cfg, device, vae_dtype_decode)
        args._vae = vae

    vae.to_device(device)
    logger.info(f"Decoding video from latents: shape {latent.shape}, dtype {latent.dtype}")
    latent_decode = latent.to(device=device, dtype=vae.dtype)

    videos = None
    with torch.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
        # Assuming vae.decode handles batch tensor [B, C, F, H, W] and returns list of [C, F, H, W]
        decoded_list = vae.decode(latent_decode)
        if decoded_list and len(decoded_list) > 0:
             videos = torch.stack(decoded_list, dim=0) # Stack list back into batch: B, C, F, H, W
        else:
             raise RuntimeError("VAE decoding failed or returned empty list.")

    vae.to_device("cpu" if args.vae_cache_cpu else "cpu") # Move back VAE
    clean_memory_on_device(device)
    logger.info(f"Decoded video shape: {videos.shape}")

    # Post-processing: scale [-1, 1] -> [0, 1], clamp, move to CPU float32
    videos = (videos + 1.0) / 2.0
    videos = torch.clamp(videos, 0.0, 1.0)
    video_final = videos.cpu().to(torch.float32)

    # Apply trim tail frames *after* decoding
    if args.trim_tail_frames > 0:
        logger.info(f"Trimming last {args.trim_tail_frames} frames from decoded video.")
        video_final = video_final[:, :, : -args.trim_tail_frames, :, :]

    logger.info(f"Decoding complete. Final video tensor shape: {video_final.shape}")
    return video_final


def save_output(
    video_tensor: torch.Tensor, # Full decoded video [B, C, F, H, W], range [0, 1]
    args: argparse.Namespace,
    original_base_names: Optional[List[str]] = None,
    latent_to_save: Optional[torch.Tensor] = None, # Full latent [B, C, F, H, W]
    original_input_frames_np: Optional[List[np.ndarray]] = None # For Extend mode
) -> None:
    """save output video, images, or latent, handling concatenation for Extend mode"""
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    seed = args.seed
    is_extend_mode = original_input_frames_np is not None

    # --- Determine Final Video Tensor for Saving ---
    video_to_save = video_tensor # Default: save the full decoded tensor
    final_video_length = video_tensor.shape[2]
    final_height = video_tensor.shape[3]
    final_width = video_tensor.shape[4]

    if is_extend_mode:
        logger.info("Processing output for Extend mode: concatenating original frames with generated frames.")
        num_original_frames = len(original_input_frames_np)

        # 1. Prepare original frames tensor: list[HWC uint8] -> tensor[B, C, N, H, W] float32 [0,1]
        original_frames_np_stacked = np.stack(original_input_frames_np, axis=0) # [N, H, W, C]
        original_frames_tensor = torch.from_numpy(original_frames_np_stacked).permute(0, 3, 1, 2).float() / 255.0 # [N, C, H, W]
        original_frames_tensor = original_frames_tensor.permute(1, 0, 2, 3).unsqueeze(0) # [1, C, N, H, W]
        original_frames_tensor = original_frames_tensor.to(video_tensor.device, dtype=video_tensor.dtype) # Match decoded tensor attributes

        # 2. Extract the generated part from the decoded tensor
        # The decoded tensor includes reconstructed input frames + generated frames
        # We only want the part *after* the input frames.
        if video_tensor.shape[2] <= num_original_frames:
             logger.error(f"Decoded video length ({video_tensor.shape[2]}) is not longer than original frames ({num_original_frames}). Cannot extract generated part.")
             # Fallback to saving the full decoded video? Or raise error?
             # Let's save the full decoded video for inspection
             logger.warning("Saving the full decoded video instead of concatenating.")
        else:
             generated_part_tensor = video_tensor[:, :, num_original_frames:, :, :] # [B, C, M, H, W]

             # 3. Concatenate original pixel tensor + generated pixel tensor
             video_to_save = torch.cat((original_frames_tensor, generated_part_tensor), dim=2) # Concat along Frame dimension
             final_video_length = video_to_save.shape[2] # Update final length
             logger.info(f"Concatenated original {num_original_frames} frames with generated {generated_part_tensor.shape[2]} frames. Final shape: {video_to_save.shape}")

    # --- Determine Base Filename ---
    base_name = f"{time_flag}_{seed}"
    if original_base_names:
         base_name += f"_{original_base_names[0]}" # Use original name if from latent
    elif args.extend_video:
         input_video_name = os.path.splitext(os.path.basename(args.extend_video))[0]
         base_name += f"_ext_{input_video_name}"
    elif args.image_path:
         input_image_name = os.path.splitext(os.path.basename(args.image_path))[0]
         base_name += f"_i2v_{input_image_name}"
    elif args.video_path:
         input_video_name = os.path.splitext(os.path.basename(args.video_path))[0]
         base_name += f"_v2v_{input_video_name}"
    # Add prompt hint? Might be too long
    # prompt_hint = "".join(filter(str.isalnum, args.prompt))[:20]
    # base_name += f"_{prompt_hint}"


    # --- Save Latent ---
    if (args.output_type == "latent" or args.output_type == "both") and latent_to_save is not None:
        latent_path = os.path.join(save_path, f"{base_name}_latent.safetensors")
        logger.info(f"Saving latent tensor shape: {latent_to_save.shape}") # Save the full latent
        metadata = {}
        if not args.no_metadata:
             # Get metadata from final saved video dimensions
            metadata = {
                "prompt": f"{args.prompt}", "negative_prompt": f"{args.negative_prompt or ''}",
                "seeds": f"{seed}", "height": f"{final_height}", "width": f"{final_width}",
                "video_length": f"{final_video_length}", # Length of the *saved* video/latent
                "infer_steps": f"{args.infer_steps}", "guidance_scale": f"{args.guidance_scale}",
                "flow_shift": f"{args.flow_shift}", "task": f"{args.task}",
                "dit_model": f"{args.dit or os.path.join(args.ckpt_dir, cfg.dit_checkpoint) if args.ckpt_dir else 'N/A'}",
                "vae_model": f"{args.vae or os.path.join(args.ckpt_dir, cfg.vae_checkpoint) if args.ckpt_dir else 'N/A'}",
                "mode": ("Extend" if is_extend_mode else "I2V" if args.image_path else "V2V" if args.video_path else "T2V"),
            }
            if is_extend_mode:
                 metadata["extend_video"] = f"{os.path.basename(args.extend_video)}"
                 metadata["num_input_frames"] = f"{args.num_input_frames}"
                 metadata["extend_length"] = f"{args.extend_length}" # Generated part length
                 metadata["total_processed_length"] = f"{latent_to_save.shape[2]}" # Latent length
            # Add other mode details... (V2V strength, I2V image, etc.)
            if args.video_path: metadata["v2v_strength"] = f"{args.strength}"
            if args.image_path: metadata["i2v_image"] = f"{os.path.basename(args.image_path)}"
            if args.end_image_path: metadata["end_image"] = f"{os.path.basename(args.end_image_path)}"
            if args.control_path: metadata["funcontrol_video"] = f"{os.path.basename(args.control_path)}"
            if args.lora_weight:
                metadata["lora_weights"] = ", ".join([os.path.basename(p) for p in args.lora_weight])
                metadata["lora_multipliers"] = ", ".join(map(str, args.lora_multiplier))

        sd = {"latent": latent_to_save.cpu()}
        try:
            save_file(sd, latent_path, metadata=metadata)
            logger.info(f"Latent saved to: {latent_path}")
        except Exception as e:
            logger.error(f"Failed to save latent file: {e}")


    # --- Save Video or Images ---
    if args.output_type == "video" or args.output_type == "both":
        video_path = os.path.join(save_path, f"{base_name}.mp4")
        # save_videos_grid expects [B, T, H, W, C], input is [B, C, T, H, W] range [0, 1]
        try:
            # Ensure tensor is on CPU for saving function
            save_videos_grid(video_to_save.cpu(), video_path, fps=args.fps, rescale=False)
            logger.info(f"Video saved to: {video_path}")
        except Exception as e:
            logger.error(f"Failed to save video file: {e}")
            logger.error(f"Video tensor info: shape={video_to_save.shape}, dtype={video_to_save.dtype}, min={video_to_save.min()}, max={video_to_save.max()}")

    elif args.output_type == "images":
        image_save_dir = os.path.join(save_path, base_name)
        os.makedirs(image_save_dir, exist_ok=True)
        # save_images_grid expects [B, T, H, W, C]
        try:
             save_images_grid(video_to_save.cpu(), image_save_dir, "frame", rescale=False, save_individually=True)
             logger.info(f"Image frames saved to directory: {image_save_dir}")
        except Exception as e:
            logger.error(f"Failed to save image files: {e}")


def main():
    # --- Argument Parsing & Setup ---
    args = parse_args()

    latents_mode = args.latent_path is not None and len(args.latent_path) > 0
    device_str = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    args.device = torch.device(device_str)
    logger.info(f"Using device: {args.device}")

    generated_latent = None
    original_input_frames_np = None # Store original frames for extend mode
    cfg = WAN_CONFIGS[args.task]
    height, width, video_length = None, None, None
    original_base_names = None # For naming output when loading latents

    if not latents_mode:
        # --- Generation Mode ---
        logger.info("Running in Generation Mode")
        args = setup_args(args) # Sets defaults, calculates video_length for extend mode
        height, width, video_length = check_inputs(args) # Validate final dimensions
        args.video_size = [height, width]
        args.video_length = video_length # Ensure video_length is stored in args for processing

        mode_str = ("Extend" if args.extend_video else
                    "I2V" if args.image_path else
                    "V2V" if args.video_path else
                    "T2V" + ("+FunControl" if args.control_path else ""))
        if args.control_path and not (args.extend_video or args.image_path or args.video_path):
             pass # Already handled above
        elif args.control_path:
             mode_str += "+FunControl"

        logger.info(f"Mode: {mode_str}")
        logger.info(
            f"Settings: video size: {height}x{width}, processed length: {video_length} frames, fps: {args.fps}, "
            f"infer_steps: {args.infer_steps}, guidance: {args.guidance_scale}, flow_shift: {args.flow_shift}"
        )
        if args.extend_video:
             logger.info(f"  Extend details: Input video: {args.extend_video}, Input frames: {args.num_input_frames}, Generated frames: {args.extend_length}")

        # Core generation pipeline - returns latent and potentially original frames
        generated_latent, original_input_frames_np = generate(args)

        if args.save_merged_model:
            logger.info("Exiting after saving merged model.")
            return
        if generated_latent is None:
             logger.error("Generation failed or was skipped, exiting.")
             return

        # Get dimensions from the *generated latent* for logging/metadata consistency
        _, _, lat_f, lat_h, lat_w = generated_latent.shape
        processed_pixel_height = lat_h * cfg.vae_stride[1]
        processed_pixel_width = lat_w * cfg.vae_stride[2]
        processed_pixel_frames = (lat_f - 1) * cfg.vae_stride[0] + 1
        logger.info(f"Generation complete. Processed latent shape: {generated_latent.shape} -> Approx Pixel Video: {processed_pixel_height}x{processed_pixel_width}@{processed_pixel_frames}")
        # Note: Final saved dimensions might differ slightly due to concatenation in Extend mode

    else:
        # --- Latents Mode ---
        logger.info("Running in Latent Loading Mode")
        original_base_names = []
        latents_list = []
        seeds = []
        metadata = {}

        if len(args.latent_path) > 1:
            logger.warning("Loading multiple latent files is not fully supported. Using first file's info.")

        latent_path = args.latent_path[0]
        original_base_names.append(os.path.splitext(os.path.basename(latent_path))[0])
        loaded_latent = None
        seed = args.seed if args.seed is not None else 0

        try:
            if os.path.splitext(latent_path)[1] != ".safetensors":
                logger.warning("Loading non-safetensors latent file. Metadata might be missing.")
                loaded_latent = torch.load(latent_path, map_location="cpu")
                if isinstance(loaded_latent, dict):
                    if "latent" in loaded_latent: loaded_latent = loaded_latent["latent"]
                    elif "state_dict" in loaded_latent: raise ValueError("Loaded file appears to be a model checkpoint.")
                    else:
                         first_key = next(iter(loaded_latent)); loaded_latent = loaded_latent[first_key]
            else:
                loaded_latent = load_file(latent_path, device="cpu")["latent"]
                with safe_open(latent_path, framework="pt", device="cpu") as f: metadata = f.metadata() or {}
                logger.info(f"Loaded metadata: {metadata}")
                # Restore args from metadata if available
                if "seeds" in metadata: seed = int(metadata["seeds"])
                if "prompt" in metadata: args.prompt = metadata["prompt"]
                if "negative_prompt" in metadata: args.negative_prompt = metadata["negative_prompt"]
                # Use metadata dimensions if available, otherwise infer later
                if "height" in metadata and "width" in metadata:
                    height = int(metadata["height"]); width = int(metadata["width"])
                    args.video_size = [height, width]
                if "video_length" in metadata: # This is the length of the *saved* video/latent
                    video_length = int(metadata["video_length"])
                    args.video_length = video_length # Store the length of the latent data
                # Restore other relevant args...
                if "guidance_scale" in metadata: args.guidance_scale = float(metadata["guidance_scale"])
                if "infer_steps" in metadata: args.infer_steps = int(metadata["infer_steps"])
                if "flow_shift" in metadata: args.flow_shift = float(metadata["flow_shift"])
                if "mode" in metadata and metadata["mode"] == "Extend":
                     if "num_input_frames" in metadata: args.num_input_frames = int(metadata["num_input_frames"])
                     # Cannot reliably get original frames from latent, so concatenation won't work right

            seeds.append(seed)
            latents_list.append(loaded_latent)
            logger.info(f"Loaded latent from {latent_path}. Shape: {loaded_latent.shape}, dtype: {loaded_latent.dtype}")

        except Exception as e:
            logger.error(f"Failed to load latent file {latent_path}: {e}")
            return

        if not latents_list: logger.error("No latent tensors loaded."); return

        generated_latent = torch.stack(latents_list, dim=0) # [B, C, F, H, W]
        if len(generated_latent.shape) != 5: raise ValueError(f"Loaded latent shape error: {generated_latent.shape}")

        args.seed = seeds[0]
        # Infer pixel dimensions from latent if not fully set by metadata
        if height is None or width is None or video_length is None:
             logger.warning("Dimensions not fully found in metadata, inferring from latent shape.")
             _, _, lat_f, lat_h, lat_w = generated_latent.shape
             height = lat_h * cfg.vae_stride[1]; width = lat_w * cfg.vae_stride[2]
             video_length = (lat_f - 1) * cfg.vae_stride[0] + 1 # This is the length corresponding to the latent
             logger.info(f"Inferred pixel dimensions from latent: {height}x{width}@{video_length}")
             args.video_size = [height, width]; args.video_length = video_length

    # --- Decode and Save ---
    if generated_latent is not None:
        # Decode latent to video tensor [B, C, F, H, W], range [0, 1]
        # Note: args.video_length might be different from latent's frame dim if trimmed during decode
        decoded_video = decode_latent(generated_latent, args, cfg)

        # Save output (handles Extend mode concatenation inside)
        save_output(
            decoded_video, args,
            original_base_names=original_base_names,
            latent_to_save=generated_latent if (args.output_type in ["latent", "both"]) else None,
            original_input_frames_np=original_input_frames_np # Pass original frames if in Extend mode
        )
    else:
        logger.error("No latent available for decoding and saving.")

    logger.info("Done!")


if __name__ == "__main__":
    main()