# --- START OF FILE wan_generate_video.py ---

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
        "--cpu_noise", action="store_true", help="Use CPU to generate noise (compatible with ComfyUI). Default is False."
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
    # I2V arguments
    parser.add_argument("--image_path", type=str, default=None, help="path to image for image2video inference")
    parser.add_argument("--end_image_path", type=str, default=None, help="path to end image for image2video inference")
    # Fun-Control argument (distinct from V2V)
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

    # Add checks for mutually exclusive arguments
    if args.video_path is not None and args.image_path is not None:
        raise ValueError("--video_path and --image_path cannot be used together.")
    if args.video_path is not None and args.control_path is not None:
        raise ValueError("--video_path (standard V2V) and --control_path (Fun-Control) cannot be used together.")
    #if args.image_path is not None and args.control_path is not None:
    #    raise ValueError("--image_path (I2V) and --control_path (Fun-Control) cannot be used together.")
    #if args.video_path is not None and "i2v" in args.task:
    #     logger.warning("--video_path is provided, but task is set to i2v. Task type does not directly affect V2V mode.")
    if args.image_path is not None and "t2v" in args.task:
         logger.warning("--image_path is provided, but task is set to t2v. Task type does not directly affect I2V mode.")
    if args.control_path is not None and not WAN_CONFIGS[args.task].is_fun_control:
        raise ValueError("--control_path is provided, but the selected task does not support Fun-Control.")
    #if not args.control_path and WAN_CONFIGS[args.task].is_fun_control:
    #    raise ValueError("The selected task requires Fun-Control (--control_path), but it was not provided.")


    return args


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

    # Force video_length to 1 for t2i tasks
    if "t2i" in args.task:
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

    # Only check supported sizes if not doing V2V (V2V might use custom sizes from input)
    # Or if it's FunControl, which might have different size constraints
    if args.video_path is None and not WAN_CONFIGS[args.task].is_fun_control:
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
) -> WanModel:
    """load DiT model

    Args:
        args: command line arguments
        config: model configuration
        device: device to use
        dit_dtype: data type for the model
        dit_weight_dtype: data type for the model weights. None for as-is
        is_i2v: I2V mode (might affect some model config details)

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
    # The 'is_i2v' flag might be used internally by load_wan_model if needed by specific Wan versions
    model = load_wan_model(config, device, args.dit, args.attn_mode, False, loading_device, loading_weight_dtype, False)

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
    frames = args.video_length # Should be set by now
    (_, lat_f, lat_h, lat_w), seq_len = calculate_dimensions(args.video_size, args.video_length, config)
    target_shape = (16, lat_f, lat_h, lat_w)

    # configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

    # set seed
    seed = args.seed # Seed should be set in generate()
    if not args.cpu_noise:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
    else:
        # ComfyUI compatible noise
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
    y = None # Initialize y for standard T2V

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


def prepare_i2v_inputs(
    args: argparse.Namespace, config, accelerator: Accelerator, device: torch.device, vae: WanVAE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
    """Prepare inputs for I2V

    Args:
        args: command line arguments
        config: model configuration
        accelerator: Accelerator instance
        device: device to use
        vae: VAE model, used for image encoding

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
            (noise, context, context_null, y, (arg_c, arg_null))
    """
    if vae is None:
        raise ValueError("VAE must be provided for I2V input preparation.")

    # get video dimensions
    height, width = args.video_size
    frames = args.video_length # Should be set by now
    max_area = width * height

    # load image
    img = Image.open(args.image_path).convert("RGB")

    # convert to numpy
    img_cv2 = np.array(img)  # PIL to numpy

    # convert to tensor (-1 to 1)
    img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)

    # end frame image
    if args.end_image_path is not None:
        end_img = Image.open(args.end_image_path).convert("RGB")
        end_img_cv2 = np.array(end_img)  # PIL to numpy
    else:
        end_img = None
        end_img_cv2 = None
    has_end_image = end_img is not None

    # calculate latent dimensions: keep aspect ratio
    img_height, img_width = img_tensor.shape[1:]
    aspect_ratio = img_height / img_width
    lat_h = round(np.sqrt(max_area * aspect_ratio) // config.vae_stride[1] // config.patch_size[1] * config.patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect_ratio) // config.vae_stride[2] // config.patch_size[2] * config.patch_size[2])
    target_height = lat_h * config.vae_stride[1]
    target_width = lat_w * config.vae_stride[2]
    lat_f = (frames - 1) // config.vae_stride[0] + 1  # size of latent frames
    max_seq_len = math.ceil((lat_f + (1 if has_end_image else 0)) * lat_h * lat_w / (config.patch_size[1] * config.patch_size[2]))
    logger.info(f"I2V target resolution: {target_height}x{target_width}, latent shape: ({lat_f}, {lat_h}, {lat_w}), seq_len: {max_seq_len}")

    # set seed
    seed = args.seed # Seed should be set in generate()
    if not args.cpu_noise:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
    else:
        # ComfyUI compatible noise
        seed_g = torch.manual_seed(seed)

    # generate noise
    noise = torch.randn(
        16, # Channel dim for latent
        lat_f + (1 if has_end_image else 0),
        lat_h,
        lat_w,
        dtype=torch.float32,
        generator=seed_g,
        device=device if not args.cpu_noise else "cpu",
    )
    noise = noise.to(device)

    # configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

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

    # load CLIP model
    clip = load_clip_model(args, config, device)
    clip.model.to(device)

    # encode image to CLIP context
    logger.info(f"Encoding image to CLIP context")
    
    # Prepare image for CLIP in the format clip.visual expects
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
        # This is what works in fun_wan_generate_video.py
        clip_context = clip.visual([img_tensor[:, None, :, :]])
    
    logger.info(f"Encoding complete")

    # free CLIP model and clean memory
    del clip
    clean_memory_on_device(device)

    # encode image to latent space with VAE
    logger.info(f"Encoding image to latent space")
    vae.to_device(device)

    # CRITICAL FIX: following the exact pattern from fun_wan_generate_video.py
    # resize image to the calculated target resolution for VAE
    interpolation = cv2.INTER_AREA if target_height < img_cv2.shape[0] else cv2.INTER_CUBIC
    img_resized_np = cv2.resize(img_cv2, (target_width, target_height), interpolation=interpolation)
    img_resized = TF.to_tensor(img_resized_np).sub_(0.5).div_(0.5).to(device)  # -1 to 1, CHW
    img_resized = img_resized.unsqueeze(1)  # CFHW (add frame dimension)

    if has_end_image:
        interpolation_end = cv2.INTER_AREA if target_height < end_img_cv2.shape[0] else cv2.INTER_CUBIC
        end_img_resized_np = cv2.resize(end_img_cv2, (target_width, target_height), interpolation=interpolation_end)
        end_img_resized = TF.to_tensor(end_img_resized_np).sub_(0.5).div_(0.5).to(device)  # -1 to 1, CHW
        end_img_resized = end_img_resized.unsqueeze(1)  # CFHW (add frame dimension)

    # create mask for the first frame (and potentially last)
    msk = torch.zeros(4, lat_f + (1 if has_end_image else 0), lat_h, lat_w, device=device)
    msk[:, 0] = 1
    if has_end_image:
        msk[:, -1] = 1

    # encode image(s) to latent space
    with accelerator.autocast(), torch.no_grad():
        # CRITICAL FIX: add padding frames to match required video length
        padding_frames = frames - 1  # first frame is the image
        img_padded = torch.cat([img_resized, torch.zeros(3, padding_frames, target_height, target_width, device=device)], dim=1)
        y = vae.encode([img_padded])[0] # Encode with padding

        if has_end_image:
            y_end = vae.encode([end_img_resized])[0] # Encode end frame
            y = torch.cat([y, y_end], dim=1) # Add end frame latent

    # Concatenate mask and latent(s)
    y = torch.concat([msk, y]) # Shape [4+C, F(+1), H, W]
    logger.info(f"I2V conditioning 'y' shape: {y.shape}")
    logger.info(f"Encoding complete")

    # Fun-Control: encode control video to latent space if needed
    # In prepare_i2v_inputs function
    if config.is_fun_control and args.control_path:
        logger.info(f"Encoding control video to latent space for Fun-Control")
        # Load and process control video (C, F, H, W format)
        control_video = load_control_video(args.control_path, frames + (1 if has_end_image else 0), 
                                          target_height, target_width).to(device)

        with accelerator.autocast(), torch.no_grad():
            control_latent = vae.encode([control_video])[0]

        # Apply control strength scaling to control_latent
        # This is the key change - scale the control latent according to strength parameter
        control_strength = getattr(args, 'control_strength', 1.0)  # Default to 1.0 if not provided
        logger.info(f"Applying control strength factor: {control_strength}")
        if control_strength != 1.0:
            # Scale the control latent - increasing strength means more influence 
            control_latent = control_latent * control_strength

        # Existing code continues...
        y = y[msk.shape[0]:]  # remove mask because Fun-Control does not need it
        if has_end_image:
            y[:, 1:-1] = 0  # remove image latent except first and last frame
        else:
            y[:, 1:] = 0  # remove image latent except first frame
        y = torch.concat([control_latent, y], dim=0)

        logger.info(f"Fun-Control combined latent shape: {y.shape}")

    # move VAE to CPU (or cache device if specified)
    vae.to_device(args.vae_cache_cpu if args.vae_cache_cpu else "cpu")
    clean_memory_on_device(device)

    # prepare model input arguments
    arg_c = {
        "context": [context[0]],
        "clip_fea": clip_context,
        "seq_len": max_seq_len,
        "y": [y], # y contains mask and image latent(s)
    }

    arg_null = {
        "context": context_null,
        "clip_fea": clip_context,
        "seq_len": max_seq_len,
        "y": [y], # y contains mask and image latent(s)
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
    # is_i2v: bool = False, # No longer needed as logic is handled by inputs dict
    use_cpu_offload: bool = True, # Example parameter, adjust as needed
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
        use_cpu_offload: Whether to offload tensors to CPU during processing (example)
    Returns:
        torch.Tensor: generated latent
    """
    arg_c, arg_null = inputs

    latent = noise # Initialize latent state
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
        # Latent should be [B, C, F, H, W] or [C, F, H, W]
        # The model expects the latent input 'x' as a list: [tensor]
        latent_on_device = latent.to(device)
        latent_model_input_list = [latent_on_device] # <<< WRAP IN LIST
        timestep = torch.stack([t]).to(device) # Ensure timestep is a tensor on device

        with accelerator.autocast(), torch.no_grad():
            # 1. Predict conditional noise estimate
            # Pass the list here
            noise_pred_cond = model(latent_model_input_list, t=timestep, **arg_c)[0]
            noise_pred_cond = noise_pred_cond.to(latent_storage_device)

            # 2. Predict unconditional noise estimate (potentially with SLG)
            apply_cfg = apply_cfg_array[i]
            if apply_cfg:
                apply_slg_step = apply_slg_global and (i >= slg_start_step and i < slg_end_step)
                slg_indices_for_call = args.slg_layers if apply_slg_step else None
                uncond_input_args = arg_null

                if apply_slg_step and args.slg_mode == "original":
                    # Standard uncond prediction first - Pass the list here
                    noise_pred_uncond = model(latent_model_input_list, t=timestep, **uncond_input_args)[0].to(latent_storage_device)
                    # SLG prediction (skipping layers in uncond) - Pass the list here
                    skip_layer_out = model(latent_model_input_list, t=timestep, skip_block_indices=slg_indices_for_call, **uncond_input_args)[0].to(latent_storage_device)

                    # Combine using SD3 formula: scaled = uncond + scale * (cond - uncond) + slg_scale * (cond - skip)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    noise_pred = noise_pred + args.slg_scale * (noise_pred_cond - skip_layer_out)

                elif apply_slg_step and args.slg_mode == "uncond":
                     # SLG prediction (skipping layers in uncond) replaces standard uncond - Pass the list here
                    noise_pred_uncond = model(latent_model_input_list, t=timestep, skip_block_indices=slg_indices_for_call, **uncond_input_args)[0].to(latent_storage_device)
                    # Combine: scaled = slg_uncond + scale * (cond - slg_uncond)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                else:
                    # Regular CFG (no SLG or SLG not active this step) - Pass the list here
                    noise_pred_uncond = model(latent_model_input_list, t=timestep, **uncond_input_args)[0].to(latent_storage_device)
                    # Combine: scaled = uncond + scale * (cond - uncond)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # CFG is skipped for this step, use conditional prediction directly
                noise_pred = noise_pred_cond

            # 3. Compute previous sample state with the scheduler
            # Scheduler expects noise_pred [B, C, F, H, W] and latent [B, C, F, H, W]
            # Ensure latent is on the compute device for the step
            # latent_step_input = latent.to(device) # Already did this with latent_on_device
            scheduler_output = scheduler.step(
                noise_pred.to(device), # Ensure noise_pred is on compute device
                t,
                latent_on_device, # Pass the tensor directly to scheduler step
                return_dict=False,
                generator=seed_g # Pass generator
            )
            prev_latent = scheduler_output[0] # Get the new latent state

            # 4. Update latent state (move back to storage device)
            latent = prev_latent.to(latent_storage_device)

    # Return the final denoised latent (should be on storage device)
    logger.info("Sampling loop finished.")
    return latent


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
    is_i2v = args.image_path is not None
    is_v2v = args.video_path is not None
    is_fun_control = args.control_path is not None and cfg.is_fun_control
    is_t2v = not is_i2v and not is_v2v and not is_fun_control

    if is_v2v: logger.info(f"Running Video-to-Video (V2V) inference with strength {args.strength}")
    elif is_i2v: logger.info(f"Running Image-to-Video (I2V) inference")
    elif is_fun_control: logger.info(f"Running Text-to-Video with Fun-Control")
    else: logger.info(f"Running Text-to-Video (T2V) inference")

    # --- Data Types ---
    dit_dtype = detect_wan_sd_dtype(args.dit) if args.dit is not None else torch.bfloat16
    if dit_dtype.itemsize == 1: # FP8 weights loaded
        dit_dtype = torch.bfloat16 # Use bfloat16 for computation
        if args.fp8_scaled:
            raise ValueError("Cannot use --fp8_scaled with pre-quantized FP8 weights.")
        dit_weight_dtype = None # Weights are already FP8
    elif args.fp8_scaled:
        dit_weight_dtype = None # Optimize later
    elif args.fp8:
        dit_weight_dtype = torch.float8_e4m3fn
    else:
        dit_weight_dtype = dit_dtype # Use compute dtype for weights

    vae_dtype = str_to_dtype(args.vae_dtype) if args.vae_dtype is not None else (torch.bfloat16 if dit_dtype == torch.bfloat16 else torch.float16)
    logger.info(
        f"Using device: {device}, DiT compute: {dit_dtype}, DiT weight: {dit_weight_dtype or 'Mixed (FP8 Scaled)' if args.fp8_scaled else dit_dtype}, VAE: {vae_dtype}, T5 FP8: {args.fp8_t5}"
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
    needs_vae_early = is_i2v or is_v2v or is_fun_control
    if needs_vae_early:
        vae = load_vae(args, cfg, device, vae_dtype)
        # Keep VAE on specified device for now, will be moved as needed

    # --- Prepare Inputs ---
    noise = None
    context = None
    context_null = None
    inputs = None
    video_latents = None # For V2V mixing

    if is_v2v:
        # 1. Load and prepare video
        # Use args.video_size as the target resolution
        # If args.video_length is None, load_video determines actual length
        # If args.video_length is set, load_video loads that many frames
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

    elif is_i2v:
        # Prepare I2V inputs (requires VAE)
        noise, context, context_null, _, inputs = prepare_i2v_inputs(args, cfg, accelerator, device, vae)
        # Note: prepare_i2v_inputs moves VAE to CPU/cache after use

    elif is_fun_control or is_t2v:
        # Check if video_length was determined by V2V loading
        if args.video_length is None:
             raise ValueError("video_length must be specified for T2V/Fun-Control or derived during V2V loading.")
        # Prepare T2V inputs (Fun-Control variation passes VAE)
        noise, context, context_null, inputs = prepare_t2v_inputs(args, cfg, accelerator, device, vae if is_fun_control else None)
        # Note: prepare_t2v_inputs moves VAE to CPU/cache if it used it

    # At this point, VAE should be on CPU/cache unless still needed for decoding
    # If VAE wasn't loaded early (pure T2V), vae is still None

    # --- Load DiT Model ---
    model = load_dit_model(args, cfg, device, dit_dtype, dit_weight_dtype, is_i2v) # Pass is_i2v flag

    # --- Merge LoRA ---
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        merge_lora_weights(model, args, device)
        if args.save_merged_model:
            logger.info("Merged model saved. Exiting without generation.")
            return None # Exit early

    # --- Optimize Model (FP8, Swapping, Compile) ---
    optimize_model(model, args, device, dit_dtype, dit_weight_dtype)

    # --- Setup Scheduler & Timesteps ---
    scheduler, timesteps = setup_scheduler(args, cfg, device)

    # --- Prepare for Sampling ---
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    latent = noise # Start with noise (already shaped correctly for T2V/I2V/V2V)

    # --- V2V Strength Adjustment ---
    if is_v2v and args.strength < 1.0:
        if video_latents is None:
             raise RuntimeError("video_latents not available for V2V strength adjustment.")

        # Calculate number of inference steps based on strength
        num_inference_steps = max(1, int(args.infer_steps * args.strength))
        logger.info(f"V2V Strength: {args.strength}, adjusting inference steps from {args.infer_steps} to {num_inference_steps}")

        # Get starting timestep index and value
        t_start_idx = len(timesteps) - num_inference_steps
        if t_start_idx < 0: t_start_idx = 0 # Ensure non-negative index
        t_start = timesteps[t_start_idx] # Timestep value at the start of sampling

        # Mix noise and video latents based on starting timestep
        # Need to map timestep value to noise schedule (e.g., sigma) - scheduler might help
        # Simple linear interpolation for now, assuming t goes 0 to 1000 roughly
        # A more accurate approach would use scheduler.add_noise or sigmas
        mix_ratio = t_start.item() / scheduler.config.num_train_timesteps # Approximate ratio
        mix_ratio = max(0.0, min(1.0, mix_ratio)) # Clamp to [0, 1]
        logger.info(f"Mixing noise and video latents with ratio (noise={mix_ratio:.3f}, video={1.0-mix_ratio:.3f}) based on start timestep {t_start.item():.1f}")

        # Ensure video_latents are on the same device and dtype as noise for mixing
        video_latents = video_latents.to(device=noise.device, dtype=noise.dtype)
        latent = noise * mix_ratio + video_latents * (1.0 - mix_ratio)

        # Use only the required subset of timesteps
        timesteps = timesteps[t_start_idx:]
        logger.info(f"Using last {len(timesteps)} timesteps for V2V sampling.")
    else:
         logger.info(f"Using full {len(timesteps)} timesteps for sampling.")
         # Latent remains the initial noise


    # --- Run Sampling Loop ---
    logger.info("Starting denoising sampling loop...")
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
        use_cpu_offload=(args.blocks_to_swap > 0) # Example: offload if swapping
    )

    # --- Cleanup ---
    del model
    del scheduler
    del context, context_null, inputs # Free memory from encoded inputs
    if video_latents is not None: del video_latents
    synchronize_device(device)

    # Wait for potential block swap operations to finish
    if args.blocks_to_swap > 0:
        logger.info("Waiting for 5 seconds to ensure block swap finishes...")
        time.sleep(5)

    gc.collect()
    clean_memory_on_device(device)

    # Store VAE instance in args for decoding function (if it exists)
    args._vae = vae # Store VAE instance (might be None if T2V)

    # Return latent with batch dimension [1, C, F, H, W]
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
        vae_dtype_decode = str_to_dtype(args.vae_dtype) if args.vae_dtype is not None else detect_wan_sd_dtype(args.dit)
        vae = load_vae(args, cfg, device, vae_dtype_decode)
        args._vae = vae # Store it in case needed again?

    # Ensure VAE is on device for decoding
    vae.to_device(device)

    logger.info(f"Decoding video from latents: shape {latent.shape}, dtype {latent.dtype}")
    # Ensure latent is on the correct device and expected dtype for VAE
    latent_decode = latent.to(device=device, dtype=vae.dtype)

    # VAE decode expects list of [C, F, H, W] or a single [B, C, F, H, W]
    # WanVAE wrapper seems to handle the list internally now? Check its decode method.
    # Assuming it takes [B, C, F, H, W] directly or handles the list internally.
    videos = None
    with torch.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
        # WanVAE.decode returns a list of decoded videos [C, F, H, W]
        decoded_list = vae.decode(latent_decode) # Pass the batch tensor
        if decoded_list and len(decoded_list) > 0:
             # Stack list back into batch dimension: B, C, F, H, W
             videos = torch.stack(decoded_list, dim=0)
        else:
             raise RuntimeError("VAE decoding failed or returned empty list.")


    # Move VAE back to CPU/cache
    vae.to_device(args.vae_cache_cpu if args.vae_cache_cpu else "cpu")
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
                "dit_model": f"{args.dit or os.path.join(args.ckpt_dir, WAN_CONFIGS[args.task].dit_checkpoint) if args.ckpt_dir else 'N/A'}",
                "vae_model": f"{args.vae or os.path.join(args.ckpt_dir, WAN_CONFIGS[args.task].vae_checkpoint) if args.ckpt_dir else 'N/A'}",
                # Add V2V/I2V specific info
                "mode": "V2V" if args.video_path else ("I2V" if args.image_path else ("FunControl" if args.control_path else "T2V")),
            }
            if args.video_path: metadata["v2v_strength"] = f"{args.strength}"
            if args.image_path: metadata["i2v_image"] = f"{os.path.basename(args.image_path)}"
            if args.end_image_path: metadata["i2v_end_image"] = f"{os.path.basename(args.end_image_path)}"
            if args.control_path: metadata["funcontrol_video"] = f"{os.path.basename(args.control_path)}"
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

    # Determine mode: generation or loading latents
    latents_mode = args.latent_path is not None and len(args.latent_path) > 0

    # Set device
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

        mode_str = "V2V" if args.video_path else ("I2V" if args.image_path else ("FunControl" if args.control_path else "T2V"))
        logger.info(f"Mode: {mode_str}")
        logger.info(
            f"Initial settings: video size: {height}x{width}@{video_length or 'auto'} (HxW@F), fps: {args.fps}, "
            f"infer_steps: {args.infer_steps}, guidance: {args.guidance_scale}, flow_shift: {args.flow_shift}"
        )

        # Core generation pipeline
        generated_latent = generate(args) # Returns [B, C, F, H, W] or None

        if args.save_merged_model:
            logger.info("Exiting after saving merged model.")
            return # Exit if only saving model

        if generated_latent is None:
             logger.error("Generation failed or was skipped, exiting.")
             return

        # Update dimensions based on the *actual* generated latent
        _, _, video_length, height, width = generated_latent.shape
        # Convert latent dimensions back to pixel dimensions for metadata/logging
        pixel_height = height * cfg.vae_stride[1]
        pixel_width = width * cfg.vae_stride[2]
        pixel_frames = (video_length - 1) * cfg.vae_stride[0] + 1
        logger.info(f"Generation complete. Latent shape: {generated_latent.shape} -> Pixel Video: {pixel_height}x{pixel_width}@{pixel_frames}")
        height, width, video_length = pixel_height, pixel_width, pixel_frames # Update for saving


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
        seed = args.seed if args.seed is not None else 0 # Default seed

        try:
            if os.path.splitext(latent_path)[1] != ".safetensors":
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

            else:
                # Load latent tensor
                loaded_latent = load_file(latent_path, device="cpu")["latent"]
                # Load metadata
                with safe_open(latent_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata() or {}
                logger.info(f"Loaded metadata: {metadata}")

                # Restore args from metadata if available
                if "seeds" in metadata: seed = int(metadata["seeds"])
                if "prompt" in metadata: args.prompt = metadata["prompt"] # Overwrite prompt for context
                if "negative_prompt" in metadata: args.negative_prompt = metadata["negative_prompt"]
                if "height" in metadata and "width" in metadata:
                    height = int(metadata["height"]); width = int(metadata["width"])
                    args.video_size = [height, width]
                if "video_length" in metadata:
                    video_length = int(metadata["video_length"])
                    args.video_length = video_length
                if "guidance_scale" in metadata: args.guidance_scale = float(metadata["guidance_scale"])
                if "infer_steps" in metadata: args.infer_steps = int(metadata["infer_steps"])
                if "flow_shift" in metadata: args.flow_shift = float(metadata["flow_shift"])
                # Could restore more args if needed

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
             raise ValueError(f"Loaded latent has incorrect shape: {generated_latent.shape}. Expected 5 dimensions.")

        # Set seed from metadata (or default)
        args.seed = seeds[0]

        # Infer pixel dimensions from latent shape and config if not in metadata
        if height is None or width is None or video_length is None:
             logger.warning("Dimensions not found in metadata, inferring from latent shape.")
             _, _, lat_f, lat_h, lat_w = generated_latent.shape
             height = lat_h * cfg.vae_stride[1]
             width = lat_w * cfg.vae_stride[2]
             video_length = (lat_f - 1) * cfg.vae_stride[0] + 1
             logger.info(f"Inferred pixel dimensions: {height}x{width}@{video_length}")
             args.video_size = [height, width]
             args.video_length = video_length

    # --- Decode and Save ---
    if generated_latent is not None:
        # Decode latent to video tensor [B, C, F, H, W], range [0, 1]
        decoded_video = decode_latent(generated_latent, args, cfg)

        # Save the output (latent and/or video/images)
        save_output(
            decoded_video,
            args,
            original_base_names=original_base_names,
            latent_to_save=generated_latent if (args.output_type == "latent" or args.output_type == "both") else None
        )
    else:
        logger.error("No latent available for decoding and saving.")

    logger.info("Done!")


if __name__ == "__main__":
    main()
# --- END OF FILE wan_generate_video.py ---