import argparse
from datetime import datetime
import gc
import json
import random
import os
import re
import time
import math
import copy
from typing import Tuple, Optional, List, Union, Any, Dict
from rich.traceback import install as install_rich_tracebacks
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from transformers import LlamaModel
from tqdm import tqdm
from rich_argparse import RichHelpFormatter
from networks import lora_framepack
from hunyuan_model.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from frame_pack import hunyuan
from frame_pack.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked, load_packed_model
from frame_pack.utils import crop_or_pad_yield_mask, resize_and_center_crop, soft_append_bcthw
from frame_pack.bucket_tools import find_nearest_bucket
from frame_pack.clip_vision import hf_clip_vision_encode
from frame_pack.k_diffusion_hunyuan import sample_hunyuan
from dataset import image_video_dataset

try:
    from lycoris.kohya import create_network_from_weights
except:
    pass

from utils.device_utils import clean_memory_on_device
from base_hv_generate_video import save_images_grid, save_videos_grid, synchronize_device
from base_wan_generate_video import merge_lora_weights
from frame_pack.framepack_utils import load_vae, load_text_encoder1, load_text_encoder2, load_image_encoders
from dataset.image_video_dataset import load_video
from blissful_tuner.blissful_args import add_blissful_args, parse_blissful_args
from blissful_tuner.video_processing_common import save_videos_grid_advanced
from blissful_tuner.latent_preview import LatentPreviewer
import logging
from diffusers_helper.utils import save_bcthw_as_mp4

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GenerationSettings:
    def __init__(self, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None):
        self.device = device
        self.dit_weight_dtype = dit_weight_dtype


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    install_rich_tracebacks()
    parser = argparse.ArgumentParser(description="Framepack inference script", formatter_class=RichHelpFormatter)

    # WAN arguments
    # parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory (Wan 2.1 official).")
    parser.add_argument("--is_f1", action="store_true", help="Use the FramePack F1 model specific logic.")
    parser.add_argument(
        "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++", "vanilla"], help="The solver used to sample."
    )

    parser.add_argument("--dit", type=str, default=None, help="DiT directory or path. Overrides --model_version if specified.")
    parser.add_argument(
        "--model_version", type=str, default="original", choices=["original", "f1"], help="Select the FramePack model version to use ('original' or 'f1'). Ignored if --dit is specified."
    )
    parser.add_argument("--vae", type=str, default=None, help="VAE directory or path")
    parser.add_argument("--text_encoder1", type=str, required=True, help="Text Encoder 1 directory or path")
    parser.add_argument("--text_encoder2", type=str, required=True, help="Text Encoder 2 directory or path")
    parser.add_argument("--image_encoder", type=str, required=True, help="Image Encoder directory or path")
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
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="prompt for generation. If `;;;` is used, it will be split into sections. Example: `section_index:prompt` or "
        "`section_index:prompt;;;section_index:prompt;;;...`, section_index can be `0` or `-1` or `0-2`, `-1` means last section, `0-2` means from 0 to 2 (inclusive).",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="negative prompt for generation, default is empty string. should not change.",
    )
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256], help="video size, height and width")
    parser.add_argument("--video_seconds", type=float, default=5.0, help="video length, Default is 5.0 seconds")
    parser.add_argument("--fps", type=int, default=30, help="video fps, Default is 30")
    parser.add_argument("--infer_steps", type=int, default=25, help="number of inference steps, Default is 25")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=str, default=None, help="Seed for evaluation.")
    # parser.add_argument(
    #     "--cpu_noise", action="store_true", help="Use CPU to generate noise (compatible with ComfyUI). Default is False."
    # )
    parser.add_argument("--latent_window_size", type=int, default=9, help="latent window size, default is 9. should not change.")
    parser.add_argument(
        "--embedded_cfg_scale", type=float, default=10.0, help="Embeded CFG scale (distilled CFG Scale), default is 10.0"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for classifier free guidance. Default is 1.0, should not change.",
    )
    parser.add_argument("--guidance_rescale", type=float, default=0.0, help="CFG Re-scale, default is 0.0. Should not change.")
    # parser.add_argument("--video_path", type=str, default=None, help="path to video for video2video inference")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="path to image for image2video inference. If `;;;` is used, it will be used as section images. The notation is same as `--prompt`.",
    )
    parser.add_argument("--end_image_path", type=str, default=None, help="path to end image for image2video inference")
    # parser.add_argument(
    #     "--control_path",
    #     type=str,
    #     default=None,
    #     help="path to control video for inference with controlnet. video file or directory with images",
    # )
    # parser.add_argument("--trim_tail_frames", type=int, default=0, help="trim tail N frames from the video before saving")

    # # Flow Matching
    # parser.add_argument(
    #     "--flow_shift",
    #     type=float,
    #     default=None,
    #     help="Shift factor for flow matching schedulers. Default depends on task.",
    # )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8")
    parser.add_argument("--fp8_fast", action="store_true", help="Enable fast FP8 arithmetic (RTX 4XXX+), only for fp8_scaled mode and can degrade quality slightly but offers noticeable speedup")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for Text Encoder 1 (LLM)")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "torch", "sageattn", "xformers", "sdpa"],  #  "flash2", "flash3",
        help="attention mode",
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument(
        "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    )
    parser.add_argument("--bulk_decode", action="store_true", help="decode all frames at once")
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

    # New arguments for batch and interactive modes
    parser.add_argument("--from_file", type=str, default=None, help="Read prompts from a file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: read prompts from console")

    #parser.add_argument("--preview_latent_every", type=int, default=None, help="Preview latent every N sections")
    parser.add_argument("--preview_suffix", type=str, default=None, help="Unique suffix for preview files to avoid conflicts in concurrent runs.")
    parser.add_argument("--full_preview", action="store_true", help="Save full intermediate video previews instead of latent previews.")

    # TeaCache arguments
    parser.add_argument("--use_teacache", action="store_true", help="Enable TeaCache for faster generation.")
    parser.add_argument("--teacache_steps", type=int, default=25, help="Number of steps for TeaCache initialization (should match --infer_steps).")
    parser.add_argument("--teacache_thresh", type=float, default=0.15, help="Relative L1 distance threshold for TeaCache skipping.")

    parser.add_argument(
    "--video_sections",
    type=int,
    default=None,
    help="number of video sections, Default is None (auto calculate from video seconds). Overrides --video_seconds if set.",
    )

    parser = add_blissful_args(parser)
    args = parser.parse_args()
    args = parse_blissful_args(args)

    # Validate arguments
    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive at the same time")

    if args.prompt is None and not args.from_file and not args.interactive:
        raise ValueError("Either --prompt, --from_file or --interactive must be specified")

    return args


def parse_prompt_line(line: str) -> Dict[str, Any]:
    """Parse a prompt line into a dictionary of argument overrides

    Args:
        line: Prompt line with options

    Returns:
        Dict[str, Any]: Dictionary of argument overrides
    """
    # TODO common function with hv_train_network.line_to_prompt_dict
    parts = line.split(" --")
    prompt = parts[0].strip()

    # Create dictionary of overrides
    overrides = {"prompt": prompt}

    for part in parts[1:]:
        if not part.strip():
            continue
        option_parts = part.split(" ", 1)
        option = option_parts[0].strip()
        value = option_parts[1].strip() if len(option_parts) > 1 else ""

        # Map options to argument names
        if option == "w":
            overrides["video_size_width"] = int(value)
        elif option == "h":
            overrides["video_size_height"] = int(value)
        elif option == "f":
            overrides["video_seconds"] = float(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "s":
            overrides["infer_steps"] = int(value)
        elif option == "g" or option == "l":
            overrides["guidance_scale"] = float(value)
        # elif option == "fs":
        #     overrides["flow_shift"] = float(value)
        elif option == "i":
            overrides["image_path"] = value
        elif option == "cn":
            overrides["control_path"] = value
        elif option == "n":
            overrides["negative_prompt"] = value

    return overrides


def apply_overrides(args: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    """Apply overrides to args

    Args:
        args: Original arguments
        overrides: Dictionary of overrides

    Returns:
        argparse.Namespace: New arguments with overrides applied
    """
    args_copy = copy.deepcopy(args)

    for key, value in overrides.items():
        if key == "video_size_width":
            args_copy.video_size[1] = value
        elif key == "video_size_height":
            args_copy.video_size[0] = value
        else:
            setattr(args_copy, key, value)

    return args_copy


def check_inputs(args: argparse.Namespace) -> Tuple[int, int, float]:
    """Validate video size and length

    Args:
        args: command line arguments

    Returns:
        Tuple[int, int, float]: (height, width, video_seconds)
    """
    height = args.video_size[0]
    width = args.video_size[1]

    if args.video_sections is not None:
        video_seconds = (args.video_sections * (args.latent_window_size * 4) + 1) / args.fps
        logger.info(f"--video_sections is set to {args.video_sections}. Calculated video_seconds: {video_seconds:.2f}s")
        args.video_seconds = video_seconds
    else:
        video_seconds = args.video_seconds


    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    return height, width, video_seconds


# region DiT model


def get_dit_dtype(args: argparse.Namespace) -> torch.dtype:
    dit_dtype = torch.bfloat16
    if args.precision == "fp16":
        dit_dtype = torch.float16
    elif args.precision == "fp32":
        dit_dtype = torch.float32
    return dit_dtype


def load_dit_model(args: argparse.Namespace, device: torch.device) -> HunyuanVideoTransformer3DModelPacked:
    """load DiT model

    Args:
        args: command line arguments
        device: device to use

    Returns:
        HunyuanVideoTransformer3DModelPacked: DiT model
    """
    loading_device = "cpu"
    # Adjust loading device logic based on F1 requirements if necessary
    if args.blocks_to_swap == 0 and not args.fp8_scaled and args.lora_weight is None:
        loading_device = device

    # F1 model expects bfloat16 according to demo
    # However, load_packed_model might handle dtype internally based on checkpoint.
    # Let's keep the call as is for now.
    logger.info(f"Loading DiT model (Class: HunyuanVideoTransformer3DModelPacked) for {'F1' if args.is_f1 else 'Standard'} mode.")
    model = load_packed_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        loading_device=loading_device,
        # Pass fp8_scaled and split_attn if load_packed_model supports them directly
        # fp8_scaled=args.fp8_scaled, # Assuming load_packed_model handles this
        # split_attn=False, # F1 demo doesn't use split_attn
    )
    return model


def optimize_model(model: HunyuanVideoTransformer3DModelPacked, args: argparse.Namespace, device: torch.device) -> None:
    """optimize the model (FP8 conversion, device move etc.)

    Args:
        model: dit model
        args: command line arguments
        device: device to use
    """
    if args.fp8_scaled:
        # load state dict as-is and optimize to fp8
        state_dict = model.state_dict()

        # if no blocks to swap, we can move the weights to GPU after optimization on GPU (omit redundant CPU->GPU copy)
        move_to_device = args.blocks_to_swap == 0  # if blocks_to_swap > 0, we will keep the model on CPU
        state_dict = model.fp8_optimization(state_dict, device, move_to_device, use_scaled_mm=args.fp8_fast)  # args.fp8_fast)

        info = model.load_state_dict(state_dict, strict=True, assign=True)
        logger.info(f"Loaded FP8 optimized weights: {info}")

        if args.blocks_to_swap == 0:
            model.to(device)  # make sure all parameters are on the right device (e.g. RoPE etc.)
    else:
        # simple cast to dit_dtype
        target_dtype = None  # load as-is (dit_weight_dtype == dtype of the weights in state_dict)
        target_device = None

        if args.fp8:
            target_dtype = torch.float8e4m3fn

        if args.blocks_to_swap == 0:
            logger.info(f"Move model to device: {device}")
            target_device = device

        if target_device is not None and target_dtype is not None:
            model.to(target_device, target_dtype)  # move and cast  at the same time. this reduces redundant copy operations

    if args.compile:
        compile_backend, compile_mode, compile_dynamic, compile_fullgraph = args.compile_args
        logger.info(
            f"Torch Compiling[Backend: {compile_backend}; Mode: {compile_mode}; Dynamic: {compile_dynamic}; Fullgraph: {compile_fullgraph}]"
        )
        torch._dynamo.config.cache_size_limit = 32
        for i in range(len(model.transformer_blocks)):
            model.transformer_blocks[i] = torch.compile(
                model.transformer_blocks[i],
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


# endregion


def decode_latent(
    latent_window_size: int,
    total_latent_sections: int,
    bulk_decode: bool,
    vae: AutoencoderKLCausal3D,
    latent: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    logger.info(f"Decoding video...")
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)  # add batch dimension

    vae.to(device)
    if not bulk_decode:
        latent_window_size = latent_window_size  # default is 9
        # total_latent_sections = (args.video_seconds * 30) / (latent_window_size * 4)
        # total_latent_sections = int(max(round(total_latent_sections), 1))
        num_frames = latent_window_size * 4 - 3

        latents_to_decode = []
        latent_frame_index = 0
        for i in range(total_latent_sections - 1, -1, -1):
            is_last_section = i == total_latent_sections - 1
            generated_latent_frames = (num_frames + 3) // 4 + (1 if is_last_section else 0)
            section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)

            section_latent = latent[:, :, latent_frame_index : latent_frame_index + section_latent_frames, :, :]
            latents_to_decode.append(section_latent)

            latent_frame_index += generated_latent_frames

        latents_to_decode = latents_to_decode[::-1]  # reverse the order of latents to decode

        history_pixels = None
        for latent in tqdm(latents_to_decode):
            if history_pixels is None:
                history_pixels = hunyuan.vae_decode(latent, vae).cpu()
            else:
                overlapped_frames = latent_window_size * 4 - 3
                current_pixels = hunyuan.vae_decode(latent, vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
            clean_memory_on_device(device)
    else:
        # bulk decode
        logger.info(f"Bulk decoding")
        history_pixels = hunyuan.vae_decode(latent, vae).cpu()
    vae.to("cpu")

    logger.info(f"Decoded. Pixel shape {history_pixels.shape}")
    return history_pixels[0]  # remove batch dimension


def prepare_i2v_inputs(
    args: argparse.Namespace,
    device: torch.device,
    vae: AutoencoderKLCausal3D,
    encoded_context: Optional[Dict] = None,
    encoded_context_n: Optional[Dict] = None,
) -> Tuple[int, int, float, dict, dict, dict, torch.Tensor]: # Adjusted return type annotation
    """Prepare inputs for I2V

    Args:
        args: command line arguments
        device: device to use
        vae: VAE model, used for image encoding
        encoded_context: Pre-encoded text context
        encoded_context_n: Pre-encoded negative text context

    Returns:
        Tuple[int, int, float, dict, dict, dict, torch.Tensor]:
            (height, width, video_seconds, context, context_null, context_img, end_latent)
    """

    def parse_section_strings(input_string: str) -> dict[int, str]:
        section_strings = {}
        if not input_string: # Handle empty input string
            return {0: ""}
        if ";;;" in input_string:
            split_section_strings = input_string.split(";;;")
            for section_str in split_section_strings:
                if ":" not in section_str:
                    start = end = 0
                    section_str_val = section_str.strip()
                else:
                    index_str, section_str_val = section_str.split(":", 1)
                    index_str = index_str.strip()
                    section_str_val = section_str_val.strip()

                    m = re.match(r"^(-?\d+)(-\d+)?$", index_str)
                    if m:
                        start = int(m.group(1))
                        end = int(m.group(2)[1:]) if m.group(2) is not None else start
                    else:
                        start = end = 0 # Default to 0 if index format is invalid

                # Handle negative indices relative to a hypothetical 'last section' (-1)
                # This part is tricky without knowing the total sections beforehand.
                # For now, treat negative indices directly. A better approach might involve
                # resolving them later in the generation loop.
                for i in range(start, end + 1):
                    section_strings[i] = section_str_val
        else:
            # If no section specifiers, assume section 0
             section_strings[0] = input_string.strip()


        # Ensure section 0 exists if any sections are defined
        if section_strings and 0 not in section_strings:
            indices = list(section_strings.keys())
            # Prefer smallest non-negative index, otherwise smallest negative index
            try:
                first_positive_index = min(i for i in indices if i >= 0)
                section_index = first_positive_index
            except ValueError: # No non-negative indices
                 section_index = min(indices) if indices else 0 # Fallback to 0 if empty

            if section_index in section_strings:
                 section_strings[0] = section_strings[section_index]
            elif section_strings: # If section_index wasn't valid somehow, pick first available
                section_strings[0] = next(iter(section_strings.values()))
            else: # If section_strings was empty initially
                section_strings[0] = "" # Default empty prompt

        # If still no section 0 (e.g., empty input string initially)
        if 0 not in section_strings:
            section_strings[0] = ""

        return section_strings

    # prepare image preprocessing function
    def preprocess_image(image_path: str, target_height: int, target_width: int, is_f1: bool): # is_f1 is kept for signature, but not used differently here
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)  # PIL to numpy, HWC

        # Consistent image preprocessing for both F1 and standard mode,
        # using target_height/target_width which come from args.video_size
        image_np = image_video_dataset.resize_image_to_bucket(image_np, (target_width, target_height))
        processed_height, processed_width = image_np.shape[0], image_np.shape[1] # Get actual size after resize

        image_tensor = torch.from_numpy(image_np).float() / 127.5 - 1.0  # -1 to 1.0, HWC
        image_tensor = image_tensor.permute(2, 0, 1)[None, :, None]  # HWC -> CHW -> NCFHW, N=1, C=3, F=1
        return image_tensor, image_np, processed_height, processed_width

    # Initial height/width check. These dimensions will be used for image processing and generation.
    height, width, video_seconds = check_inputs(args)
    logger.info(f"Video dimensions for processing and generation set to: {height}x{width} (from --video_size or default).")

    section_image_paths = parse_section_strings(args.image_path)

    section_images = {}
    first_image_processed = False
    for index, image_path in section_image_paths.items():

        img_tensor, img_np, proc_h, proc_w = preprocess_image(image_path, height, width, args.is_f1)
        section_images[index] = (img_tensor, img_np)
        if not first_image_processed and image_path:
            default_video_size_used = (args.video_size[0] == 256 and args.video_size[1] == 256) # Check if default was used
            if default_video_size_used and (proc_h != height or proc_w != width):
                logger.info(f"Video dimensions updated to {proc_h}x{proc_w} based on first image processing (as default --video_size was used).")
                height, width = proc_h, proc_w
                args.video_size = [height, width] # Update args for consistency for downstream logging/metadata.
            elif not default_video_size_used and (proc_h != height or proc_w != width):
                logger.warning(f"User specified --video_size {height}x{width}, but first image processed to {proc_h}x{proc_w}. "
                               f"Generation will use {height}x{width}. Conditioning image aspect might differ.")
            first_image_processed = True


    # Process end image if provided
    if args.end_image_path is not None:
        end_img_tensor, end_img_np, _, _ = preprocess_image(args.end_image_path, height, width, args.is_f1)
    else:
        end_img_tensor, end_img_np = None, None

    # configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else ""

    if encoded_context is None or encoded_context_n is None: # Regenerate if either is missing
        # parse section prompts
        section_prompts = parse_section_strings(args.prompt)

        # load text encoder
        # Assuming load_text_encoder1/2 are compatible
        tokenizer1, text_encoder1 = load_text_encoder1(args, args.fp8_llm, device)
        tokenizer2, text_encoder2 = load_text_encoder2(args)
        text_encoder2.to(device)

        logger.info(f"Encoding prompts...")
        llama_vecs = {}
        llama_attention_masks = {}
        clip_l_poolers = {}
        # Use a common dtype for text encoders if possible, respecting fp8 flag
        text_encoder_dtype = torch.float8_e4m3fn if args.fp8_llm else torch.float16 # text_encoder1.dtype
        
        # Pre-allocate negative prompt tensors only if needed
        llama_vec_n, clip_l_pooler_n = None, None
        llama_attention_mask_n = None

        # Encode positive prompts first
        with torch.autocast(device_type=device.type, dtype=text_encoder_dtype), torch.no_grad():
             for index, prompt in section_prompts.items():
                 # Ensure prompt is not empty before encoding
                 current_prompt = prompt if prompt else "" # Use empty string if prompt is None or empty
                 llama_vec, clip_l_pooler = hunyuan.encode_prompt_conds(current_prompt, text_encoder1, text_encoder2, tokenizer1, tokenizer2)

                 # Pad/crop and store
                 llama_vec_padded, llama_attention_mask = crop_or_pad_yield_mask(llama_vec.cpu(), length=512) # Move to CPU before padding

                 llama_vecs[index] = llama_vec_padded
                 llama_attention_masks[index] = llama_attention_mask
                 clip_l_poolers[index] = clip_l_pooler.cpu() # Move to CPU

                 # Use the encoding of section 0 as fallback for negative if needed
                 if index == 0 and args.guidance_scale == 1.0:
                     llama_vec_n = torch.zeros_like(llama_vec_padded)
                     llama_attention_mask_n = torch.zeros_like(llama_attention_mask)
                     clip_l_pooler_n = torch.zeros_like(clip_l_poolers[0])

        # Encode negative prompt if needed
        if args.guidance_scale != 1.0:
             with torch.autocast(device_type=device.type, dtype=text_encoder_dtype), torch.no_grad():
                 current_n_prompt = n_prompt if n_prompt else ""
                 llama_vec_n_raw, clip_l_pooler_n_raw = hunyuan.encode_prompt_conds(
                     current_n_prompt, text_encoder1, text_encoder2, tokenizer1, tokenizer2
                 )
                 llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n_raw.cpu(), length=512) # Move to CPU
                 clip_l_pooler_n = clip_l_pooler_n_raw.cpu() # Move to CPU


        # Check if negative prompt was generated (handles guidance_scale=1.0 case)
        if llama_vec_n is None:
             logger.warning("Negative prompt tensors not generated (likely guidance_scale=1.0). Using zeros.")
             # Assuming section 0 exists and was processed
             llama_vec_n = torch.zeros_like(llama_vecs[0])
             llama_attention_mask_n = torch.zeros_like(llama_attention_masks[0])
             clip_l_pooler_n = torch.zeros_like(clip_l_poolers[0])


        # free text encoder and clean memory
        del text_encoder1, text_encoder2, tokenizer1, tokenizer2
        clean_memory_on_device(device)

        # load image encoder (Handles SigLIP via framepack_utils)
        feature_extractor, image_encoder = load_image_encoders(args)
        image_encoder.to(device)

        # encode image with image encoder
        logger.info(f"Encoding images with {'SigLIP' if args.is_f1 else 'Image Encoder'}...")
        section_image_encoder_last_hidden_states = {}
        img_encoder_dtype = image_encoder.dtype # Get dtype from loaded model
        end_image_embedding_for_f1 = None # Initialize for F1 end image
        with torch.autocast(device_type=device.type, dtype=img_encoder_dtype), torch.no_grad():
            for index, (img_tensor, img_np) in section_images.items():
                # Use hf_clip_vision_encode (works for SigLIP too)
                image_encoder_output = hf_clip_vision_encode(img_np, feature_extractor, image_encoder)
                image_encoder_last_hidden_state = image_encoder_output.last_hidden_state.cpu() # Move to CPU
                section_image_encoder_last_hidden_states[index] = image_encoder_last_hidden_state
            
            if args.is_f1 and end_img_np is not None: # end_img_np is from args.end_image_path
                logger.info("F1 Mode: Encoding end image for potential conditioning.")
                end_image_encoder_output_f1 = hf_clip_vision_encode(end_img_np, feature_extractor, image_encoder)
                end_image_embedding_for_f1 = end_image_encoder_output_f1.last_hidden_state.cpu()              

        # free image encoder and clean memory
        del image_encoder, feature_extractor
        clean_memory_on_device(device)

        # --- Store encoded contexts for potential reuse ---
        # Positive context (bundle per unique prompt string if needed, or just section 0)
        # For simplicity, let's assume we only cache based on args.prompt for now
        encoded_context = {
            "llama_vecs": llama_vecs,
            "llama_attention_masks": llama_attention_masks,
            "clip_l_poolers": clip_l_poolers,
            "image_encoder_last_hidden_states": section_image_encoder_last_hidden_states # Store all section states
        }
        # Negative context
        encoded_context_n = {
             "llama_vec": llama_vec_n,
             "llama_attention_mask": llama_attention_mask_n,
             "clip_l_pooler": clip_l_pooler_n,
        }
        # --- End context caching ---

    else:
        # Use pre-encoded context
        logger.info("Using pre-encoded context.")
        llama_vecs = encoded_context["llama_vecs"]
        llama_attention_masks = encoded_context["llama_attention_masks"]
        clip_l_poolers = encoded_context["clip_l_poolers"]
        section_image_encoder_last_hidden_states = encoded_context["image_encoder_last_hidden_states"] # Retrieve all sections
        llama_vec_n = encoded_context_n["llama_vec"]
        llama_attention_mask_n = encoded_context_n["llama_attention_mask"]
        clip_l_pooler_n = encoded_context_n["clip_l_pooler"]
        # Need to re-parse section prompts if using cached context
        section_prompts = parse_section_strings(args.prompt)


    # VAE encoding
    logger.info(f"Encoding image(s) to latent space...")
    vae.to(device)
    vae_dtype = vae.dtype # Get VAE dtype

    section_start_latents = {}
    with torch.autocast(device_type=device.type, dtype=vae_dtype), torch.no_grad():
        for index, (img_tensor, img_np) in section_images.items():
            start_latent = hunyuan.vae_encode(img_tensor, vae).cpu() # Move to CPU
            section_start_latents[index] = start_latent

        end_latent = hunyuan.vae_encode(end_img_tensor, vae).cpu() if end_img_tensor is not None else None # Move to CPU

    vae.to("cpu")  # move VAE to CPU to save memory
    clean_memory_on_device(device)

    # prepare model input arguments
    arg_c = {} # Positive text conditioning per section
    arg_c_img = {} # Positive image conditioning per section

    # Ensure section_prompts is available (parsed earlier)
    if 'section_prompts' not in locals():
         section_prompts = parse_section_strings(args.prompt)

    # Populate positive text args
    for index in llama_vecs.keys():
        # Get corresponding prompt, defaulting to empty string if index missing
        prompt_text = section_prompts.get(index, "")

        arg_c_i = {
            "llama_vec": llama_vecs[index],
            "llama_attention_mask": llama_attention_masks[index],
            "clip_l_pooler": clip_l_poolers[index],
            "prompt": prompt_text,  # Include the actual prompt text
        }
        arg_c[index] = arg_c_i

     # Populate negative text args (only one needed)
    arg_null = {
        "llama_vec": llama_vec_n,
        "llama_attention_mask": llama_attention_mask_n,
        "clip_l_pooler": clip_l_pooler_n,
        "prompt": n_prompt, # Include negative prompt text
    }

    # Populate positive image args
    for index in section_start_latents.keys(): # Use latents keys as reference
         # Check if corresponding hidden state exists, fallback to section 0 if needed
         image_encoder_last_hidden_state = section_image_encoder_last_hidden_states.get(index, section_image_encoder_last_hidden_states.get(0))
         if image_encoder_last_hidden_state is None and section_image_encoder_last_hidden_states:
              # Absolute fallback if index and 0 are missing but others exist
              image_encoder_last_hidden_state = next(iter(section_image_encoder_last_hidden_states.values()))
         elif image_encoder_last_hidden_state is None:
              raise ValueError(f"Cannot find image encoder state for section {index} or fallback section 0.")


         arg_c_img_i = {
             "image_encoder_last_hidden_state": image_encoder_last_hidden_state,
             "start_latent": section_start_latents[index]
         }
         arg_c_img[index] = arg_c_img_i

    # Ensure fallback section 0 exists in arg_c and arg_c_img if needed later
    if 0 not in arg_c and arg_c:
        arg_c[0] = next(iter(arg_c.values()))
    if 0 not in arg_c_img and arg_c_img:
        arg_c_img[0] = next(iter(arg_c_img.values()))

    # Final check for minimal context existence
    if not arg_c or not arg_c_img:
        raise ValueError("Failed to prepare conditioning arguments. Check prompts and image paths.")


    return height, width, video_seconds, arg_c, arg_null, arg_c_img, end_latent, end_image_embedding_for_f1


# def setup_scheduler(args: argparse.Namespace, config, device: torch.device) -> Tuple[Any, torch.Tensor]:
#     """setup scheduler for sampling

#     Args:
#         args: command line arguments
#         config: model configuration
#         device: device to use

#     Returns:
#         Tuple[Any, torch.Tensor]: (scheduler, timesteps)
#     """
#     if args.sample_solver == "unipc":
#         scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False)
#         scheduler.set_timesteps(args.infer_steps, device=device, shift=args.flow_shift)
#         timesteps = scheduler.timesteps
#     elif args.sample_solver == "dpm++":
#         scheduler = FlowDPMSolverMultistepScheduler(
#             num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False
#         )
#         sampling_sigmas = get_sampling_sigmas(args.infer_steps, args.flow_shift)
#         timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sampling_sigmas)
#     elif args.sample_solver == "vanilla":
#         scheduler = FlowMatchDiscreteScheduler(num_train_timesteps=config.num_train_timesteps, shift=args.flow_shift)
#         scheduler.set_timesteps(args.infer_steps, device=device)
#         timesteps = scheduler.timesteps

#         # FlowMatchDiscreteScheduler does not support generator argument in step method
#         org_step = scheduler.step

#         def step_wrapper(
#             model_output: torch.Tensor,
#             timestep: Union[int, torch.Tensor],
#             sample: torch.Tensor,
#             return_dict: bool = True,
#             generator=None,
#         ):
#             return org_step(model_output, timestep, sample, return_dict=return_dict)

#         scheduler.step = step_wrapper
#     else:
#         raise NotImplementedError("Unsupported solver.")

#     return scheduler, timesteps


# In fpack_generate_video.py

def generate(args: argparse.Namespace, gen_settings: GenerationSettings, shared_models: Optional[Dict] = None) -> Tuple[AutoencoderKLCausal3D, torch.Tensor]: # Return VAE too
    """main function for generation

    Args:
        args: command line arguments
        gen_settings: Generation settings object
        shared_models: dictionary containing pre-loaded models and encoded data

    Returns:
        Tuple[AutoencoderKLCausal3D, torch.Tensor]: vae, generated latent
    """
    device, dit_weight_dtype = (gen_settings.device, gen_settings.dit_weight_dtype)

    # prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    # Ensure seed is integer
    if isinstance(seed, str):
        try:
            seed = int(seed)
        except ValueError:
            logger.warning(f"Invalid seed string: {seed}. Generating random seed.")
            seed = random.randint(0, 2**32 - 1)
    elif not isinstance(seed, int):
        logger.warning(f"Invalid seed type: {type(seed)}. Generating random seed.")
        seed = random.randint(0, 2**32 - 1)

    args.seed = seed  # set seed to args for saving

    vae = None # Initialize VAE

    # Check if we have shared models
    if shared_models is not None:
        # Use shared models and encoded data
        vae = shared_models.get("vae")
        model = shared_models.get("model")

        # --- Retrieve cached context ---
        # Try to get context based on the full prompt string first
        prompt_key = args.prompt if args.prompt else ""
        n_prompt_key = args.negative_prompt if args.negative_prompt else ""

        encoded_context = shared_models.get("encoded_contexts", {}).get(prompt_key)
        encoded_context_n = shared_models.get("encoded_contexts", {}).get(n_prompt_key)

        # If not found, maybe the cache uses a simpler key (like just section 0?) - needs alignment with prepare_i2v_inputs caching logic
        # For now, assume prepare_i2v_inputs handles regeneration if cache miss
        if encoded_context is None or encoded_context_n is None:
             logger.info("Cached context not found or incomplete, preparing inputs.")
             # Need VAE for preparation if regenerating context
             if vae is None:
                 vae = load_vae(args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, device)
             height, width, video_seconds, context, context_null, context_img, end_latent = prepare_i2v_inputs(
                 args, device, vae # Pass VAE here
             )
             # Store newly generated context back? (Requires shared_models to be mutable and handled carefully)
             # shared_models["encoded_contexts"][prompt_key] = context # Simplified example
             # shared_models["encoded_contexts"][n_prompt_key] = context_null # Simplified example
        else:
             logger.info("Using cached context from shared models.")
             # Need VAE if decoding later, load if not present
             if vae is None:
                  vae = load_vae(args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, device)
             height, width, video_seconds, context, context_null, context_img, end_latent = prepare_i2v_inputs(
                 args, device, vae, encoded_context, encoded_context_n
             )
        # --- End context retrieval ---

    else:
        # prepare inputs without shared models
        vae = load_vae(args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, device)
        height, width, video_seconds, context, context_null, context_img, end_latent, end_image_embedding_for_f1 = prepare_i2v_inputs(args, device, vae)
        # load DiT model
        model = load_dit_model(args, device) # Handles F1 class loading implicitly

        # merge LoRA weights
        if args.lora_weight is not None and len(args.lora_weight) > 0:
             # Ensure merge_lora_weights can handle HunyuanVideoTransformer3DModelPacked
             # It might need adjustments depending on its implementation.
             logger.info("Merging LoRA weights...")
             # Assuming lora_framepack is the correct network type definition
             # Make sure merge_lora_weights exists and is imported
             try:
                 from base_wan_generate_video import merge_lora_weights # Example import path
                 merge_lora_weights(lora_framepack, model, args, device)
             except ImportError:
                  logger.error("merge_lora_weights function not found. Skipping LoRA merge.")
             except Exception as e:
                  logger.error(f"Error merging LoRA weights: {e}")

             # if we only want to save the model, we can skip the rest
             if args.save_merged_model:
                 # Implement saving logic here if merge_lora_weights doesn't handle it
                 logger.info(f"Saving merged model to {args.save_merged_model} and exiting.")
                 # Example: save_model(model, args.save_merged_model)
                 return None, None # Indicate no generation occurred


        # optimize model: fp8 conversion, block swap etc.
        optimize_model(model, args, device)
        if args.use_teacache:
            logger.info(f"Initializing TeaCache: steps={args.teacache_steps}, threshold={args.teacache_thresh}")
            # The model's initialize_teacache expects num_steps and rel_l1_thresh
            model.initialize_teacache(
                enable_teacache=True,
                num_steps=args.teacache_steps,
                rel_l1_thresh=args.teacache_thresh
            )
        else:
            logger.info("TeaCache is disabled.")
            # Ensure it's explicitly disabled in the model too, just in case
            model.initialize_teacache(enable_teacache=False)

    # --- Sampling ---
    latent_window_size = args.latent_window_size  # default is 9 (consistent with F1 demo)
    
    if args.video_sections is not None:
        total_latent_sections = args.video_sections
        logger.info(f"Using --video_sections: {total_latent_sections} sections.")
    else:
        total_latent_sections = (video_seconds * args.fps) / (latent_window_size * 4) 
        total_latent_sections = int(max(round(total_latent_sections), 1))
        logger.info(f"Calculated total_latent_sections from video_seconds: {total_latent_sections} sections.")

    # set random generator
    seed_g = torch.Generator(device="cpu") # Keep noise on CPU initially
    seed_g.manual_seed(seed)

    # F1 expects frames = latent_window_size * 4 - 3
    # Our script's default decode uses latent_window_size * 4 - 3 overlap
    # Let's calculate F1 frames per section explicitly
    f1_frames_per_section = latent_window_size * 4 - 3

    logger.info(
        f"Mode: {'F1' if args.is_f1 else 'Standard'}, "
        f"Video size: {height}x{width}@{video_seconds:.2f}s, fps: {args.fps}, num sections: {total_latent_sections}, "
        f"infer_steps: {args.infer_steps}, frames per generation step: {f1_frames_per_section}"
    )

    # Determine compute dtype based on model/args
    compute_dtype = model.dtype if hasattr(model, 'dtype') else torch.bfloat16 # Default for F1
    if args.fp8 or args.fp8_scaled:
        # FP8 might still use bfloat16/float16 for some operations
        logger.info("FP8 enabled, using bfloat16 for intermediate computations.")
        compute_dtype = torch.bfloat16 # Or potentially float16 depending on model/ops
    logger.info(f"Using compute dtype: {compute_dtype}")


# --- F1 Model Specific Sampling Logic ---
    if args.is_f1: # Renamed from args.f1 in simpler script to args.is_f1
        logger.info("Starting F1 model sampling process.")

        logger.info(f"F1 Mode: Using video dimensions {height}x{width} for latent operations and generation.")
        history_latents = torch.zeros((1, 16, 19, height // 8, width // 8), dtype=torch.float32, device='cpu')

        start_latent_0 = context_img.get(0, {}).get("start_latent")
        if start_latent_0 is None:
             raise ValueError("Cannot find start_latent for section 0 in context_img.")
        
        if start_latent_0.shape[3] != (height // 8) or start_latent_0.shape[4] != (width // 8):
            logger.error(f"Mismatch between start_latent_0 dimensions ({start_latent_0.shape[3]}x{start_latent_0.shape[4]}) "
                         f"and history_latents dimensions ({height//8}x{width//8}). This should not happen with current logic.")

        history_latents = torch.cat([history_latents, start_latent_0.cpu().float()], dim=2)
        
        history_pixels_for_preview_f1_cpu = None
        # Initialize history_pixels_for_preview_f1_cpu with the decoded start_latent_0
        # if full_preview is enabled. This happens *before* the section loop.
        if args.full_preview and args.preview_latent_every is not None: # No section_index check here
            if vae is None:
                logger.error("VAE not available for initial F1 preview setup.")
            else:
                logger.info("F1 Full Preview: Decoding initial start_latent for preview history.")
                vae.to(device)
                initial_latent_for_preview = start_latent_0.to(device, dtype=vae.dtype if hasattr(vae, 'dtype') else torch.float16)
                decoded_initial = hunyuan.vae_decode(initial_latent_for_preview, vae).cpu()
                if decoded_initial.ndim == 4: # CTHW
                    history_pixels_for_preview_f1_cpu = decoded_initial.unsqueeze(0) # BCTHW
                elif decoded_initial.ndim == 5: # BCTHW
                    history_pixels_for_preview_f1_cpu = decoded_initial
                else:
                    logger.error(f"Unexpected dimensions from initial VAE decode: {decoded_initial.shape}")
                    history_pixels_for_preview_f1_cpu = None # Ensure it's None if decode failed
                vae.to("cpu")
                clean_memory_on_device(device)

        total_generated_latent_frames = 1

        if args.preview_latent_every and not args.full_preview: 
            previewer = LatentPreviewer(args, vae, None, gen_settings.device, compute_dtype, model_type="framepack")
        else:
            previewer = None

        for section_index in range(total_latent_sections): 
            logger.info(f"--- F1 Section {section_index + 1} / {total_latent_sections} ---")
            f1_split_sizes = [1, 16, 2, 1, args.latent_window_size] 
            f1_indices = torch.arange(0, sum(f1_split_sizes)).unsqueeze(0).to(device)
            (
                f1_clean_latent_indices_start,
                f1_clean_latent_4x_indices,
                f1_clean_latent_2x_indices,
                f1_clean_latent_1x_indices,
                f1_latent_indices, 
            ) = f1_indices.split(f1_split_sizes, dim=1)
            f1_clean_latent_indices = torch.cat([f1_clean_latent_indices_start, f1_clean_latent_1x_indices], dim=1)

            current_image_context_section_idx = section_index if section_index in context_img else 0
            current_start_latent = context_img[current_image_context_section_idx]["start_latent"].to(device, dtype=torch.float32) 

            current_history_for_f1_clean = history_latents[:, :, -sum([16, 2, 1]):, :, :].to(device, dtype=torch.float32)
            f1_clean_latents_4x, f1_clean_latents_2x, f1_clean_latents_1x = current_history_for_f1_clean.split([16, 2, 1], dim=2)
            
            f1_clean_latents_combined = torch.cat([current_start_latent, f1_clean_latents_1x], dim=2)
            
            context_section_idx = section_index if section_index in context else 0
            llama_vec = context[context_section_idx]["llama_vec"].to(device, dtype=compute_dtype)
            llama_attention_mask = context[context_section_idx]["llama_attention_mask"].to(device)
            clip_l_pooler = context[context_section_idx]["clip_l_pooler"].to(device, dtype=compute_dtype)
            image_encoder_last_hidden_state = context_img[current_image_context_section_idx]["image_encoder_last_hidden_state"].to(device, dtype=compute_dtype)
            llama_vec_n = context_null["llama_vec"].to(device, dtype=compute_dtype)
            llama_attention_mask_n = context_null["llama_attention_mask"].to(device)
            clip_l_pooler_n = context_null["clip_l_pooler"].to(device, dtype=compute_dtype)

            # generated_latents_step is on GPU after sample_hunyuan
            generated_latents_step = sample_hunyuan(
                transformer=model, sampler=args.sample_solver, width=width, height=height,
                frames=f1_frames_per_section, real_guidance_scale=args.guidance_scale, 
                distilled_guidance_scale=args.embedded_cfg_scale, guidance_rescale=args.guidance_rescale,
                num_inference_steps=args.infer_steps, generator=seed_g,
                prompt_embeds=llama_vec, prompt_embeds_mask=llama_attention_mask, prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n, negative_prompt_embeds_mask=llama_attention_mask_n, negative_prompt_poolers=clip_l_pooler_n,
                device=device, dtype=compute_dtype, image_embeddings=image_encoder_last_hidden_state,
                latent_indices=f1_latent_indices, clean_latents=f1_clean_latents_combined, clean_latent_indices=f1_clean_latent_indices,
                clean_latents_2x=f1_clean_latents_2x, clean_latent_2x_indices=f1_clean_latent_2x_indices,
                clean_latents_4x=f1_clean_latents_4x, clean_latent_4x_indices=f1_clean_latent_4x_indices,
            )
            
            newly_generated_latent_frames_count_this_step = int(generated_latents_step.shape[2])
            history_latents = torch.cat([history_latents, generated_latents_step.cpu().float()], dim=2)
            total_generated_latent_frames += newly_generated_latent_frames_count_this_step

            if args.preview_latent_every is not None and (section_index + 1) % args.preview_latent_every == 0:
                if args.full_preview: # This is the block we are modifying
                    logger.info(f"Saving full F1 preview at section {section_index + 1}")
                    if vae is None:
                        logger.error("VAE not available for full F1 preview.")
                    else:
                        preview_dir = os.path.join(args.save_path, "previews")
                        os.makedirs(preview_dir, exist_ok=True)
                        preview_filename_full = os.path.join(preview_dir, f"latent_preview_{args.preview_suffix if args.preview_suffix else section_index + 1}.mp4")                        
                        num_latents_for_stitch_decode_preview = args.latent_window_size * 2 
                        num_latents_for_stitch_decode_preview = min(num_latents_for_stitch_decode_preview, history_latents.shape[2])

                        if num_latents_for_stitch_decode_preview > 0:
                            # Select the tail from the *total accumulated* latents for decoding this preview part
                            latents_for_preview_stitch_gpu = history_latents[:, :, -num_latents_for_stitch_decode_preview:].to(
                                device, dtype=vae.dtype if hasattr(vae, 'dtype') else torch.float16
                            )
                            
                            vae.to(device)
                            pixels_for_preview_stitch_cpu = hunyuan.vae_decode(latents_for_preview_stitch_gpu, vae).cpu()
                            vae.to("cpu")

                            if pixels_for_preview_stitch_cpu.ndim == 4: # Should be CTHW from vae_decode
                                pixels_for_preview_stitch_cpu = pixels_for_preview_stitch_cpu.unsqueeze(0) # Make it BCTHW
                            
                            if history_pixels_for_preview_f1_cpu is None:
                                # This case is for the very first preview update (e.g. after section 0 generation).
                                # The `history_pixels_for_preview_f1_cpu` should have been initialized with the decoded start_image.
                                # If it's None here, it means the initial start_image decode didn't happen or this is the first section *being previewed*.
                                if section_index == 0 or (args.preview_latent_every == 1 and section_index == 0) : # Check if it's truly the first previewable section
                                     history_pixels_for_preview_f1_cpu = pixels_for_preview_stitch_cpu
                                else: 
                                    # This is an unexpected state if initial history was set up.
                                    logger.warning("history_pixels_for_preview_f1_cpu was None unexpectedly after section 0; re-initializing with current stitch.")
                                    history_pixels_for_preview_f1_cpu = pixels_for_preview_stitch_cpu
                                            
                            elif pixels_for_preview_stitch_cpu.shape[2] > 0: # Ensure the new decoded part has frames
                                # `pixels_for_preview_stitch_cpu` now contains the new segment *plus* the overlapping tail from history.
                                # `history_pixels_for_preview_f1_cpu` is everything up to the start of that overlap.
                                
                                overlap_pixels_for_preview_blend = args.latent_window_size * 4 - 3 
                                actual_blend_overlap = min(overlap_pixels_for_preview_blend, 
                                                           history_pixels_for_preview_f1_cpu.shape[2], 
                                                           pixels_for_preview_stitch_cpu.shape[2]) 
                                
                                if actual_blend_overlap > 0:
                                    history_pixels_for_preview_f1_cpu = soft_append_bcthw(
                                        history_pixels_for_preview_f1_cpu,
                                        pixels_for_preview_stitch_cpu, # This 'current' now starts with the overlap
                                        overlap=actual_blend_overlap
                                    )
                                else: # Fallback to simple concatenation if no valid overlap
                                    history_pixels_for_preview_f1_cpu = torch.cat(
                                        [history_pixels_for_preview_f1_cpu, pixels_for_preview_stitch_cpu], dim=2
                                    )
                            
                            if history_pixels_for_preview_f1_cpu is not None and history_pixels_for_preview_f1_cpu.shape[2] > 0:
                                save_bcthw_as_mp4(history_pixels_for_preview_f1_cpu, preview_filename_full, fps=args.fps, crf=getattr(args, 'mp4_crf', 16))
                                logger.info(f"Full F1 preview saved to {preview_filename_full}")
                            else:
                                logger.warning(f"Skipping F1 preview save at section {section_index + 1} due to empty or invalid pixel history.")
                            
                            # Clean up tensors specific to this preview update step
                            del latents_for_preview_stitch_gpu
                            if 'pixels_for_preview_stitch_cpu' in locals(): del pixels_for_preview_stitch_cpu
                        else:
                            logger.warning(f"Not enough latents for stitch decode at F1 preview section {section_index + 1}. Current preview not updated.")
                        
                        clean_memory_on_device(device)
                elif previewer is not None: 
                    logger.info(f"Previewing latents at F1 section {section_index + 1}")
                    # `history_latents` is the full accumulation on CPU.
                    # `total_generated_latent_frames` should reflect the current length of the *meaningful* part of `history_latents`
                    preview_latents_f1_for_pv = history_latents[:, :, -total_generated_latent_frames:, :, :].to(gen_settings.device)
                    previewer.preview(preview_latents_f1_for_pv, section_index, preview_suffix=args.preview_suffix)
                    del preview_latents_f1_for_pv
                    clean_memory_on_device(gen_settings.device)
            
            del generated_latents_step, current_history_for_f1_clean, f1_clean_latents_combined
            del f1_clean_latents_1x, f1_clean_latents_2x, f1_clean_latents_4x, current_start_latent
            del llama_vec, llama_attention_mask, clip_l_pooler, image_encoder_last_hidden_state
            del llama_vec_n, llama_attention_mask_n, clip_l_pooler_n
            clean_memory_on_device(device)

        real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]
        # No resizing needed as generation happened at target dimensions.

    # --- Standard Model Sampling Logic ---
    else: # Standard mode
        logger.info("Starting standard model sampling process.")
        history_latents = torch.zeros((1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32, device='cpu')
        if end_latent is not None: 
            logger.info(f"Using end image: {args.end_image_path}")
            history_latents[:, :, 0:1] = end_latent.cpu().float()

        total_generated_latent_frames = 0
        
        history_pixels_for_preview_std_cpu = None # Initialize pixel history
        # For standard mode (backward generation), the first chunk generated is the "end" of the video.
        # If end_latent is provided and previews are on, we should decode it to start the preview history.
        if args.full_preview and args.preview_latent_every is not None and end_latent is not None:
            logger.info(f"Saving full preview at standard section {section_index + 1} (Reverse Index {section_index_reverse})")
            if vae is None:
                logger.error("VAE not available for full standard preview.")
            else:
                preview_filename_full_std = os.path.join(args.save_path, f"latent_preview_{args.preview_suffix if args.preview_suffix else section_index_reverse + 1}.mp4")
                
                # Decode ONLY the latents processed in THIS step for the preview update
                latents_this_step_for_decode = current_step_latents_cpu.to(device, dtype=vae.dtype if hasattr(vae, 'dtype') else torch.float16)
                
                vae.to(device)
                current_section_pixels_decoded_cpu = hunyuan.vae_decode(latents_this_step_for_decode, vae).cpu()
                vae.to("cpu")
                if current_section_pixels_decoded_cpu.ndim == 4:
                    current_section_pixels_decoded_cpu = current_section_pixels_decoded_cpu.unsqueeze(0)
                if history_pixels_for_preview_std_cpu is None:
                    # This is the first chunk generated (chronologically latest part of the video, or end_image decode)
                    history_pixels_for_preview_std_cpu = current_section_pixels_decoded_cpu
                elif current_section_pixels_decoded_cpu.shape[2] > 0 :
                    # Prepend the new (earlier in time) segment
                    history_pixels_for_preview_std_cpu = torch.cat(
                        [current_section_pixels_decoded_cpu, history_pixels_for_preview_std_cpu], dim=2
                    )
                
                if history_pixels_for_preview_std_cpu is not None and history_pixels_for_preview_std_cpu.shape[2] > 0:
                    save_bcthw_as_mp4(history_pixels_for_preview_std_cpu, preview_filename_full_std, fps=args.fps, crf=getattr(args, 'mp4_crf', 16))
                    logger.info(f"Full standard preview saved to {preview_filename_full_std}")
                else:
                    logger.warning(f"Skipping Standard mode preview save at section {section_index + 1} due to empty pixel history.")
                del latents_this_step_for_decode, current_section_pixels_decoded_cpu
                clean_memory_on_device(device)


        latent_paddings = list(reversed(range(total_latent_sections))) 
        if total_latent_sections > 4:
            logger.info("Using F1-style latent padding heuristic for > 4 sections.")
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        if args.preview_latent_every and not args.full_preview:
            previewer = LatentPreviewer(args, vae, None, gen_settings.device, compute_dtype, model_type="framepack")
        else:
            previewer = None

        for section_index_reverse, latent_padding in enumerate(latent_paddings):
            section_index = total_latent_sections - 1 - section_index_reverse
            section_index_from_last = -(section_index_reverse + 1) 
            logger.info(f"--- Standard Section {section_index + 1} / {total_latent_sections} (Reverse Index {section_index_reverse}, Padding {latent_padding}) ---")

            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            apply_section_image = False
            if section_index_from_last in context_img:
                image_index = section_index_from_last
                if not is_last_section: apply_section_image = True
            elif section_index in context_img:
                image_index = section_index
                if not is_last_section: apply_section_image = True
            else:
                image_index = 0 

            start_latent_section = context_img[image_index]["start_latent"].to(device, dtype=torch.float32) 
            if apply_section_image:
                latent_padding_size = 0
                logger.info(f"Applying experimental section image, forcing latent_padding_size = 0")

            split_sizes_std = [1, latent_padding_size, latent_window_size, 1, 2, 16]
            indices_std = torch.arange(0, sum(split_sizes_std)).unsqueeze(0).to(device)
            (
                clean_latent_indices_pre, blank_indices, latent_indices, 
                clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices,
            ) = indices_std.split(split_sizes_std, dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            current_history_std = history_latents[:, :, :19].to(device, dtype=torch.float32) 
            clean_latents_post, clean_latents_2x, clean_latents_4x = current_history_std.split([1, 2, 16], dim=2)
            clean_latents = torch.cat([start_latent_section, clean_latents_post], dim=2)

            if section_index_from_last in context: prompt_index = section_index_from_last
            elif section_index in context: prompt_index = section_index
            else: prompt_index = 0
            context_for_index = context[prompt_index]
            logger.info(f"Using prompt from section {prompt_index}: '{context_for_index['prompt'][:100]}...'")

            llama_vec = context_for_index["llama_vec"].to(device, dtype=compute_dtype)
            llama_attention_mask = context_for_index["llama_attention_mask"].to(device)
            clip_l_pooler = context_for_index["clip_l_pooler"].to(device, dtype=compute_dtype)
            image_encoder_last_hidden_state = context_img[image_index]["image_encoder_last_hidden_state"].to(device, dtype=compute_dtype)
            llama_vec_n = context_null["llama_vec"].to(device, dtype=compute_dtype)
            llama_attention_mask_n = context_null["llama_attention_mask"].to(device)
            clip_l_pooler_n = context_null["clip_l_pooler"].to(device, dtype=compute_dtype)

            sampler_to_use = args.sample_solver
            guidance_scale_to_use = args.guidance_scale
            embedded_cfg_scale_to_use = args.embedded_cfg_scale
            guidance_rescale_to_use = args.guidance_rescale

            # generated_latents_step is on GPU after sample_hunyuan
            generated_latents_step_gpu = sample_hunyuan(
                transformer=model, sampler=sampler_to_use, width=width, height=height,
                frames=f1_frames_per_section, real_guidance_scale=guidance_scale_to_use,
                distilled_guidance_scale=embedded_cfg_scale_to_use, guidance_rescale=guidance_rescale_to_use,
                num_inference_steps=args.infer_steps, generator=seed_g, 
                prompt_embeds=llama_vec, prompt_embeds_mask=llama_attention_mask, prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n, negative_prompt_embeds_mask=llama_attention_mask_n, negative_prompt_poolers=clip_l_pooler_n,
                device=device, dtype=compute_dtype, image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices,
            )
            
            # Move to CPU for history accumulation and potential preview decode
            generated_latents_step = generated_latents_step_gpu.cpu().float()

            if is_last_section: # This is the first iteration in reverse, corresponds to earliest part of generated video
                logger.info("Standard Mode: Last section (first in reverse loop), prepending start_latent_section for this chunk.")
                generated_latents_step = torch.cat([start_latent_section.cpu().float(), generated_latents_step], dim=2)
            
            current_step_latents_cpu = generated_latents_step.clone() # This is what was generated/prepended in this step

            total_generated_latent_frames += int(generated_latents_step.shape[2])
            history_latents = torch.cat([generated_latents_step, history_latents], dim=2) # Prepend to full latent history

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
            if args.preview_latent_every is not None and (section_index_reverse + 1) % args.preview_latent_every == 0:
                if args.full_preview:
                    logger.info(f"Saving full preview at standard section {section_index + 1} (Reverse Index {section_index_reverse})")
                    if vae is None:
                        logger.error("VAE not available for full standard preview.")
                    else:
                        preview_dir_std = os.path.join(args.save_path, "previews")
                        os.makedirs(preview_dir_std, exist_ok=True)
                        preview_filename_full_std = os.path.join(preview_dir_std, f"latent_preview_{args.preview_suffix if args.preview_suffix else section_index_reverse + 1}.mp4")
                        num_latents_to_decode_for_stitch = (args.latent_window_size * 2 + 1) if is_last_section else (args.latent_window_size * 2)
                        num_latents_to_decode_for_stitch = min(num_latents_to_decode_for_stitch, real_history_latents.shape[2])

                        if num_latents_to_decode_for_stitch > 0:
                            # Decode the relevant segment from the *beginning* of real_history_latents (chronologically earliest)
                            latents_for_preview_decode_gpu = real_history_latents[:, :, :num_latents_to_decode_for_stitch].to(
                                device, dtype=vae.dtype if hasattr(vae, 'dtype') else torch.float16
                            )
                            
                            vae.to(device)
                            pixels_decoded_for_preview_cpu = hunyuan.vae_decode(latents_for_preview_decode_gpu, vae).cpu()
                            vae.to("cpu")

                            if pixels_decoded_for_preview_cpu.ndim == 4: # Should be CTHW from vae_decode
                                pixels_decoded_for_preview_cpu = pixels_decoded_for_preview_cpu.unsqueeze(0) # Make it BCTHW
                            
                            if history_pixels_for_preview_std_cpu is None:
                                history_pixels_for_preview_std_cpu = pixels_decoded_for_preview_cpu
                            elif pixels_decoded_for_preview_cpu.shape[2] > 0:
                                overlap_pixels_for_preview_blend = args.latent_window_size * 4 - 3
                                
                                # pixels_decoded_for_preview_cpu is the NEW (chronologically EARLIER) segment
                                # history_pixels_for_preview_std_cpu is the OLD (chronologically LATER) segment
                                actual_blend_overlap = min(overlap_pixels_for_preview_blend, 
                                                           pixels_decoded_for_preview_cpu.shape[2], 
                                                           history_pixels_for_preview_std_cpu.shape[2])
                                
                                if actual_blend_overlap > 0:
                                    history_pixels_for_preview_std_cpu = soft_append_bcthw(
                                        pixels_decoded_for_preview_cpu,      # History for soft_append (chronologically earlier)
                                        history_pixels_for_preview_std_cpu,  # Current for soft_append (chronologically later)
                                        overlap=actual_blend_overlap
                                    )
                                else: # Fallback to simple concatenation if no valid overlap
                                    history_pixels_for_preview_std_cpu = torch.cat(
                                        [pixels_decoded_for_preview_cpu, history_pixels_for_preview_std_cpu], dim=2
                                    )
                            
                            if history_pixels_for_preview_std_cpu is not None and history_pixels_for_preview_std_cpu.shape[2] > 0:
                                save_bcthw_as_mp4(history_pixels_for_preview_std_cpu, preview_filename_full_std, fps=args.fps, crf=getattr(args, 'mp4_crf', 16))
                                logger.info(f"Full standard preview saved to {preview_filename_full_std}")
                            else:
                                logger.warning(f"Skipping Standard mode preview save at section {section_index + 1} due to empty or invalid pixel history.")
                            
                            del latents_for_preview_decode_gpu
                            if 'pixels_decoded_for_preview_cpu' in locals(): del pixels_decoded_for_preview_cpu
                        else:
                             logger.warning(f"Not enough latents for stitch decode at Standard mode preview section {section_index + 1}. Current preview not updated.")
                        clean_memory_on_device(device)
                elif previewer is not None: 
                    logger.info(f"Previewing latents at standard section {section_index + 1} (Reverse Index {section_index_reverse})")
                    preview_latents_std_for_pv = real_history_latents.to(gen_settings.device) 
                    previewer.preview(preview_latents_std_for_pv, section_index, preview_suffix=args.preview_suffix)
                    del preview_latents_std_for_pv
                    clean_memory_on_device(gen_settings.device)

            logger.info(f"Section {section_index + 1} finished. Total latent frames: {total_generated_latent_frames}. History shape: {history_latents.shape}")

            del generated_latents_step, current_history_std, clean_latents, clean_latents_post, clean_latents_2x, clean_latents_4x
            del llama_vec, llama_attention_mask, clip_l_pooler, image_encoder_last_hidden_state, start_latent_section
            del llama_vec_n, llama_attention_mask_n, clip_l_pooler_n
            # Explicitly delete the GPU tensor if it was created
            if 'generated_latents_step_gpu' in locals(): del generated_latents_step_gpu
            clean_memory_on_device(device)

    gc.collect()
    clean_memory_on_device(device)

    # Return the final generated latents (CPU tensor) and the VAE
    # The shape should be (B, C, T_total, H, W)
    logger.info(f"Generation complete. Final latent shape: {real_history_latents.shape}")
    return vae, real_history_latents # Return VAE along with latents


def save_latent(latent: torch.Tensor, args: argparse.Namespace, height: int, width: int, original_base_name: Optional[str] = None) -> str: # Add original_base_name
    """Save latent to file

    Args:
        latent: Latent tensor (CTHW expected)
        args: command line arguments
        height: height of frame
        width: width of frame
        original_base_name: Optional base name from loaded file

    Returns:
        str: Path to saved latent file
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}" # Use provided base name
    video_seconds = args.video_seconds
    latent_path = f"{save_path}/{time_flag}_{seed}{original_name}_latent.safetensors" # Add original name to file

    # Ensure latent is on CPU before saving
    latent = latent.detach().cpu()

    if args.no_metadata:
        metadata = None
    else:
        # (Metadata creation remains the same)
        metadata = {
            "seeds": f"{seed}",
            "prompt": f"{args.prompt}",
            "height": f"{height}",
            "width": f"{width}",
            "video_seconds": f"{video_seconds}",
            "infer_steps": f"{args.infer_steps}",
            "guidance_scale": f"{args.guidance_scale}",
            "latent_window_size": f"{args.latent_window_size}",
            "embedded_cfg_scale": f"{args.embedded_cfg_scale}",
            "guidance_rescale": f"{args.guidance_rescale}",
            "sample_solver": f"{args.sample_solver}",
            # "latent_window_size": f"{args.latent_window_size}", # Duplicate key
            "fps": f"{args.fps}",
            "is_f1": f"{args.is_f1}", # Add F1 flag to metadata
        }
        if args.negative_prompt is not None:
            metadata["negative_prompt"] = f"{args.negative_prompt}"
        # Add other relevant args like LoRA, compile settings, etc. if desired

    sd = {"latent": latent.contiguous()}
    save_file(sd, latent_path, metadata=metadata)
    logger.info(f"Latent saved to: {latent_path}")

    return latent_path


def save_video(
    video: torch.Tensor, args: argparse.Namespace, original_base_name: Optional[str] = None, latent_frames: Optional[int] = None
) -> str:
    """Save video to file

    Args:
        video: Video tensor
        args: command line arguments
        original_base_name: Original base name (if latents are loaded from files)

    Returns:
        str: Path to saved video file
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    latent_frames = "" if latent_frames is None else f"_{latent_frames}"
    video_path = f"{save_path}/{time_flag}_{seed}{original_name}{latent_frames}.mp4"

    video = video.unsqueeze(0)
    if args.codec is not None:
        save_videos_grid_advanced(video, video_path, args.codec, args.container, rescale=True, fps=args.fps, keep_frames=args.keep_pngs)
    else:
        save_videos_grid(video, video_path, fps=args.fps, rescale=True)
    logger.info(f"Video saved to: {video_path}")

    return video_path


def save_images(sample: torch.Tensor, args: argparse.Namespace, original_base_name: Optional[str] = None) -> str:
    """Save images to directory

    Args:
        sample: Video tensor
        args: command line arguments
        original_base_name: Original base name (if latents are loaded from files)

    Returns:
        str: Path to saved images directory
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    image_name = f"{time_flag}_{seed}{original_name}"
    sample = sample.unsqueeze(0)
    save_images_grid(sample, save_path, image_name, rescale=True)
    logger.info(f"Sample images saved to: {save_path}/{image_name}")

    return f"{save_path}/{image_name}"


# In fpack_generate_video.py

def save_output(
    args: argparse.Namespace,
    vae: AutoencoderKLCausal3D,
    latent: torch.Tensor,
    device: torch.device,
    original_base_names: Optional[List[str]] = None,
) -> None:
    """save output

    Args:
        args: command line arguments
        vae: VAE model
        latent: latent tensor (should be BCTHW or CTHW)
        device: device to use
        original_base_names: original base names (if latents are loaded from files)
    """
    if latent.ndim == 4: # Add batch dim if missing (CTHW -> BCTHW)
        latent = latent.unsqueeze(0)
    elif latent.ndim != 5:
        raise ValueError(f"Unexpected latent dimensions: {latent.ndim}. Expected 4 or 5.")

    # Latent shape is BCTHW
    batch_size, channels, latent_frames, latent_height, latent_width = latent.shape
    height = latent_height * 8
    width = latent_width * 8
    logger.info(f"Saving output. Latent shape: {latent.shape}; Target pixel shape: {height}x{width}")

    if args.output_type == "latent" or args.output_type == "both":
        # save latent (use first name if multiple originals)
        base_name = original_base_names[0] if original_base_names else None
        save_latent(latent[0], args, height, width, original_base_name=base_name) # Save first batch item if B > 1
    if args.output_type == "latent":
        return

    if args.video_sections is not None:
        total_latent_sections = args.video_sections
    else:
        total_latent_sections = (args.video_seconds * args.fps) / (args.latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))
    
    logger.info(f"Decoding using total_latent_sections = {total_latent_sections} (derived from {'--video_sections' if args.video_sections is not None else '--video_seconds'}).")

    # Decode (handle potential batch > 1?)
    # decode_latent expects BCTHW or CTHW, and returns CTHW
    # Currently process only the first item in the batch for saving video/images
    video = decode_latent(args.latent_window_size, total_latent_sections, args.bulk_decode, vae, latent[0], device)

    if args.output_type == "video" or args.output_type == "both":
        # save video
        original_name = original_base_names[0] if original_base_names else None
        save_video(video, args, original_name, latent_frames=latent_frames) # Pass latent frames count

    elif args.output_type == "images":
        # save images
        original_name = original_base_names[0] if original_base_names else None
        save_images(video, args, original_name)


def preprocess_prompts_for_batch(prompt_lines: List[str], base_args: argparse.Namespace) -> List[Dict]:
    """Process multiple prompts for batch mode

    Args:
        prompt_lines: List of prompt lines
        base_args: Base command line arguments

    Returns:
        List[Dict]: List of prompt data dictionaries
    """
    prompts_data = []

    for line in prompt_lines:
        line = line.strip()
        if not line or line.startswith("#"):  # Skip empty lines and comments
            continue

        # Parse prompt line and create override dictionary
        prompt_data = parse_prompt_line(line)
        logger.info(f"Parsed prompt data: {prompt_data}")
        prompts_data.append(prompt_data)

    return prompts_data


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
    device = torch.device(args.device)

    dit_weight_dtype = None  # default
    if args.fp8_scaled:
        dit_weight_dtype = None  # various precision weights, so don't cast to specific dtype
    elif args.fp8:
        dit_weight_dtype = torch.float8_e4m3fn

    logger.info(f"Using device: {device}, DiT weight weight precision: {dit_weight_dtype}")

    gen_settings = GenerationSettings(device=device, dit_weight_dtype=dit_weight_dtype)
    return gen_settings


# In fpack_generate_video.py

def main():
    # Parse arguments
    args = parse_args()

    # Check if latents are provided
    latents_mode = args.latent_path is not None and len(args.latent_path) > 0

    # Set device
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    args.device = device # Ensure args has the final device

    if latents_mode:
        # --- Latent Decode Mode ---
        # (Keep existing logic, but maybe add F1 flag reading from metadata?)
        original_base_names = []
        latents_list = []
        seeds = []
        is_f1_from_metadata = False # Default

        # Allow only one latent file for simplicity now
        if len(args.latent_path) > 1:
             logger.warning("Loading multiple latents is not fully supported for metadata consistency. Using first latent's metadata.")

        for i, latent_path in enumerate(args.latent_path):
             logger.info(f"Loading latent from: {latent_path}")
             base_name = os.path.splitext(os.path.basename(latent_path))[0]
             original_base_names.append(base_name)
             seed = 0 # Default seed

             if not latent_path.lower().endswith(".safetensors"):
                 logger.warning(f"Loading from non-safetensors file {latent_path}. Metadata might be missing.")
                 latents = torch.load(latent_path, map_location="cpu")
                 if isinstance(latents, dict) and "latent" in latents: # Handle potential dict structure
                     latents = latents["latent"]
             else:
                 try:
                     # Load latent tensor
                     loaded_data = load_file(latent_path, device="cpu") # Load to CPU
                     latents = loaded_data["latent"]

                     # Load metadata
                     metadata = {}
                     with safe_open(latent_path, framework="pt", device="cpu") as f:
                         metadata = f.metadata()
                     if metadata is None:
                         metadata = {}
                     logger.info(f"Loaded metadata: {metadata}")

                     # Apply metadata only from the first file for consistency
                     if i == 0:
                         if "seeds" in metadata:
                             try:
                                 seed = int(metadata["seeds"])
                             except ValueError:
                                 logger.warning(f"Could not parse seed from metadata: {metadata['seeds']}")
                         if "height" in metadata and "width" in metadata:
                             try:
                                 height = int(metadata["height"])
                                 width = int(metadata["width"])
                                 args.video_size = [height, width]
                                 logger.info(f"Set video size from metadata: {height}x{width}")
                             except ValueError:
                                 logger.warning(f"Could not parse height/width from metadata.")
                         if "video_seconds" in metadata:
                              try:
                                  args.video_seconds = float(metadata["video_seconds"])
                                  logger.info(f"Set video seconds from metadata: {args.video_seconds}")
                              except ValueError:
                                  logger.warning(f"Could not parse video_seconds from metadata.")
                         if "fps" in metadata:
                             try:
                                 args.fps = int(metadata["fps"])
                                 logger.info(f"Set fps from metadata: {args.fps}")
                             except ValueError:
                                  logger.warning(f"Could not parse fps from metadata.")
                         if "is_f1" in metadata:
                             is_f1_from_metadata = metadata["is_f1"].lower() == 'true'
                             if args.is_f1 != is_f1_from_metadata:
                                  logger.warning(f"Metadata indicates is_f1={is_f1_from_metadata}, overriding command line argument --is_f1={args.is_f1}")
                                  args.is_f1 = is_f1_from_metadata


                 except Exception as e:
                     logger.error(f"Error loading safetensors file {latent_path}: {e}")
                     continue # Skip this file

             # Use seed from first file for all if multiple latents are somehow processed
             if i == 0:
                 args.seed = seed
             seeds.append(seed) # Store all seeds read

             logger.info(f"Loaded latent shape: {latents.shape}")

             if latents.ndim == 5:  # [BCTHW]
                 if latents.shape[0] > 1:
                     logger.warning("Latent file contains batch size > 1. Using only the first item.")
                 latents = latents[0]  # Use first item -> [CTHW]
             elif latents.ndim != 4:
                 logger.error(f"Unexpected latent dimension {latents.ndim} in {latent_path}. Skipping.")
                 continue

             latents_list.append(latents)

        if not latents_list:
             logger.error("No valid latents loaded. Exiting.")
             return

        # Stack latents into a batch if multiple were loaded (BCTHW)
        # Note: Saving output currently only processes the first batch item.
        latent_batch = torch.stack(latents_list, dim=0)

        # Load VAE needed for decoding
        vae = load_vae(args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, device)
        # Call save_output with the batch
        save_output(args, vae, latent_batch, device, original_base_names)

    elif args.from_file:
        # Batch mode from file (Not Implemented)
        logger.error("Batch mode (--from_file) is not implemented yet.")
        # with open(args.from_file, "r", encoding="utf-8") as f:
        #     prompt_lines = f.readlines()
        # prompts_data = preprocess_prompts_for_batch(prompt_lines, args)
        # process_batch_prompts(prompts_data, args) # Needs implementation
        raise NotImplementedError("Batch mode is not implemented yet.")

    elif args.interactive:
        # Interactive mode (Not Implemented)
        logger.error("Interactive mode (--interactive) is not implemented yet.")
        # process_interactive(args) # Needs implementation
        raise NotImplementedError("Interactive mode is not implemented yet.")

    else:
        # --- Single prompt mode (original behavior + F1 support) ---
        gen_settings = get_generation_settings(args)

        # Generate returns (vae, latent)
        vae, latent = generate(args, gen_settings) # VAE might be loaded inside generate

        if latent is None: # Handle cases like --save_merged_model
             logger.info("Generation did not produce latents (e.g., --save_merged_model used). Exiting.")
             return

        # Ensure VAE is available (it should be returned by generate)
        if vae is None:
             logger.error("VAE not available after generation. Cannot save output.")
             return

        # Save output expects BCTHW or CTHW, generate returns BCTHW
        # save_output handles the batch dimension internally now.
        save_output(args, vae, latent, device)

        # Clean up VAE if it was loaded here
        del vae
        gc.collect()
        clean_memory_on_device(device)


    logger.info("Done!")


if __name__ == "__main__":
    main()
