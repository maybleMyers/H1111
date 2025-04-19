# Combined and Corrected Script
#!/usr/bin/env python3

import argparse
import os
import sys
import time
import random
import traceback
from datetime import datetime
from pathlib import Path
import re # For parsing section args

import einops
import numpy as np
import torch
import av # For saving video (used by save_bcthw_as_mp4)
from PIL import Image
from tqdm import tqdm
import cv2


# --- Dependencies from diffusers_helper ---
# Ensure this library is installed or in the PYTHONPATH
try:
    # from diffusers_helper.hf_login import login # Not strictly needed for inference if models public/cached
    from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode #, vae_decode_fake # vae_decode_fake not used here
    from diffusers_helper.utils import (save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw,
                                        resize_and_center_crop, generate_timestamp)
    from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
    from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
    from diffusers_helper.memory import (cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation,
                                         offload_model_from_device_for_memory_preservation, fake_diffusers_current_device,
                                         DynamicSwapInstaller, unload_complete_models, load_model_as_complete)
    from diffusers_helper.clip_vision import hf_clip_vision_encode
    from diffusers_helper.bucket_tools import find_nearest_bucket#, bucket_options # bucket_options no longer needed here
except ImportError:
    print("Error: Could not import modules from 'diffusers_helper'.")
    print("Please ensure the 'diffusers_helper' library is installed and accessible.")
    print("You might need to clone the repository and add it to your PYTHONPATH.")
    sys.exit(1)
# --- End Dependencies ---

from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from transformers import SiglipImageProcessor, SiglipVisionModel

# --- Constants ---
DIMENSION_MULTIPLE = 16 # VAE and model constraints often require divisibility by 8 or 16. 16 is safer.
SECTION_ARG_PATTERN = re.compile(r"^(\d+):([^:]+)(?::(.*))?$") # Regex for section arg: number:image_path[:prompt]

def parse_section_args(section_strings):
    """ Parses the --section arguments into a dictionary.
        Key: Section index (int, 0 = last section/video start, 1 = second last, ...)
        Value: Tuple (image_path: str, prompt: str or None)
    """
    section_data = {}
    if not section_strings:
        return section_data

    for section_str in section_strings:
        match = SECTION_ARG_PATTERN.match(section_str)
        if not match:
            print(f"Warning: Invalid section format: '{section_str}'. Expected 'number:image_path[:prompt]'. Skipping.")
            continue

        section_index_str, image_path, prompt_text = match.groups()
        section_index = int(section_index_str)
        prompt_text = prompt_text if prompt_text else None # Store None if prompt is missing/empty

        if not os.path.exists(image_path):
            print(f"Warning: Image path for section {section_index} ('{image_path}') not found. Skipping section.")
            continue

        if section_index in section_data:
             print(f"Warning: Duplicate section index {section_index}. Overwriting previous entry.")

        section_data[section_index] = (image_path, prompt_text)
        print(f"Parsed section {section_index}: Image='{image_path}', Prompt='{prompt_text}'")

    # Sort by section index just for clarity, though dict access is by key
    # return dict(sorted(section_data.items()))
    # No need to sort, dictionary stores it fine.
    return section_data


def parse_args():
    parser = argparse.ArgumentParser(description="FramePack HunyuanVideo inference script (CLI version with Advanced End Frame & Section Control)")

    # --- Model Paths ---
    parser.add_argument('--transformer_path', type=str, default='lllyasviel/FramePackI2V_HY', help="Path to the FramePack Transformer model")
    parser.add_argument('--vae_path', type=str, default='hunyuanvideo-community/HunyuanVideo', help="Path to the VAE model directory")
    parser.add_argument('--text_encoder_path', type=str, default='hunyuanvideo-community/HunyuanVideo', help="Path to the Llama text encoder directory")
    parser.add_argument('--text_encoder_2_path', type=str, default='hunyuanvideo-community/HunyuanVideo', help="Path to the CLIP text encoder directory")
    parser.add_argument('--image_encoder_path', type=str, default='lllyasviel/flux_redux_bfl', help="Path to the SigLIP image encoder directory")
    parser.add_argument('--hf_home', type=str, default='./hf_download', help="Directory to download/cache Hugging Face models")

    # --- Input ---
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image (start frame)")
    parser.add_argument("--end_frame", type=str, default=None, help="Path to the optional end frame image (video end)")
    parser.add_argument("--prompt", type=str, required=True, help="Default prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt for generation")
    # <<< START: Modified Arguments for End Frame >>>
    parser.add_argument("--end_frame_weight", type=float, default=0.6, help="End frame influence weight (0.0-1.0) for blending modes ('half', 'progressive'). Not used for sections.")
    parser.add_argument("--end_frame_influence", type=str, default="last", # Changed default to 'last' to align with `endframe.py`'s implicit behavior
                       choices=["last", "half", "progressive", "bookend"],
                       help="How to use the global end frame: 'last' (sets context latent for first generated section), 'half' (blends conditioning for second half), 'progressive' (gradually increases influence), 'bookend' (uses end frame embedding/latent only for first generated section). Section keyframes override this for their specific section.")
    # <<< END: Modified Arguments for End Frame >>>
    # <<< START: New Arguments for Section Control >>>
    parser.add_argument("--section", type=str, action='append',
                        help="Define a keyframe section. Format: 'index:image_path[:prompt]'. Index 0 is the last generated section (video start), 1 is second last, etc. Repeat for multiple sections. Example: --section 0:path/to/start_like.png:'A sunrise' --section 2:path/to/mid.png")
    # <<< END: New Arguments for Section Control >>>

    # --- Output Resolution (Choose ONE method) ---
    parser.add_argument("--target_resolution", type=int, default=None, help=f"Target resolution for the longer side for automatic aspect ratio calculation (bucketing). Used if --width and --height are not specified. Must be positive and ideally divisible by {DIMENSION_MULTIPLE}.")
    parser.add_argument("--width", type=int, default=None, help=f"Explicit target width for the output video. Overrides --target_resolution. Must be positive and ideally divisible by {DIMENSION_MULTIPLE}.")
    parser.add_argument("--height", type=int, default=None, help=f"Explicit target height for the output video. Overrides --target_resolution. Must be positive and ideally divisible by {DIMENSION_MULTIPLE}.")

    # --- Output ---
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save the generated video")
    parser.add_argument("--save_intermediate_sections", action='store_true', help="Save the video after each section is generated and decoded.")
    parser.add_argument("--save_section_final_frames", action='store_true', help="Save the final decoded frame of each generated section as a PNG image.")


    # --- Generation Parameters (Matching Gradio Demo Defaults where applicable) ---
    parser.add_argument("--seed", type=int, default=None, help="Seed for generation. Random if not set.")
    parser.add_argument("--total_second_length", type=float, default=5.0, help="Total desired video length in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps (changing not recommended)")
    parser.add_argument("--distilled_guidance_scale", "--gs", type=float, default=10.0, help="Distilled CFG Scale (gs)")
    parser.add_argument("--cfg", type=float, default=1.0, help="Classifier-Free Guidance Scale (fixed at 1.0 for FramePack usually)")
    parser.add_argument("--rs", type=float, default=0.0, help="CFG Rescale (fixed at 0.0 for FramePack usually)")
    parser.add_argument("--latent_window_size", type=int, default=9, help="Latent window size (changing not recommended)")
    # parser.add_argument("--trim_tail_frames", type=int, default=None, help="Number of frames to trim from the end of the *final* video (used with --end_frame). Note: May conflict with precise end-frame matching.") # Trim logic might be less relevant now.

    # --- Performance / Memory ---
    parser.add_argument('--high_vram', action='store_true', help="Force high VRAM mode (loads all models to GPU)")
    parser.add_argument('--low_vram', action='store_true', help="Force low VRAM mode (uses dynamic swapping)")
    parser.add_argument("--gpu_memory_preservation", type=float, default=6.0, help="GPU memory (GB) to preserve when offloading (low VRAM mode)")
    parser.add_argument('--use_teacache', action='store_true', default=True, help="Use TeaCache optimization (default: True)")
    parser.add_argument('--no_teacache', action='store_false', dest='use_teacache', help="Disable TeaCache optimization")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu'). Auto-detects if None.")

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
        print(f"Generated random seed: {args.seed}")

    # Resolution validation
    if args.width is not None and args.height is not None:
        if args.width <= 0 or args.height <= 0:
            print(f"Error: Explicit --width ({args.width}) and --height ({args.height}) must be positive.")
            sys.exit(1)
        if args.target_resolution is not None:
            print("Warning: Both --width/--height and --target_resolution specified. Using explicit --width and --height.")
            args.target_resolution = None # Ignore target_resolution
    elif args.target_resolution is not None:
        if args.target_resolution <= 0:
            print(f"Error: --target_resolution ({args.target_resolution}) must be positive.")
            sys.exit(1)
        if args.width is not None or args.height is not None:
            print("Error: Cannot specify --target_resolution with only one of --width or --height. Provide both or neither.")
            sys.exit(1)
    else:
        # Default resolution if none specified
        print(f"Warning: No resolution specified. Defaulting to --target_resolution 640.")
        args.target_resolution = 640
        #print(f"Error: You must specify the target resolution using either --target_resolution OR both --width and --height.")
        #sys.exit(1)

    # Validate end frame weight
    if args.end_frame_weight < 0.0 or args.end_frame_weight > 1.0:
        print(f"Error: --end_frame_weight must be between 0.0 and 1.0 (got {args.end_frame_weight}).")
        sys.exit(1)

    # Check divisibility later after calculation/rounding, but warn if initial inputs aren't ideal
    if args.width is not None and args.width % DIMENSION_MULTIPLE != 0:
         print(f"Warning: Specified --width ({args.width}) is not divisible by {DIMENSION_MULTIPLE}. It will be rounded down.")
    if args.height is not None and args.height % DIMENSION_MULTIPLE != 0:
         print(f"Warning: Specified --height ({args.height}) is not divisible by {DIMENSION_MULTIPLE}. It will be rounded down.")
    if args.target_resolution is not None and args.target_resolution % DIMENSION_MULTIPLE != 0:
         print(f"Warning: Specified --target_resolution ({args.target_resolution}) is not divisible by {DIMENSION_MULTIPLE}. The calculated dimensions will be rounded down.")

    # Validate end frame path if provided
    if args.end_frame and not os.path.exists(args.end_frame):
        print(f"Error: End frame image not found at '{args.end_frame}'.")
        sys.exit(1)

    # Parse section arguments
    args.section_data = parse_section_args(args.section)
    max_section_index = max(args.section_data.keys()) if args.section_data else -1

    # # Validate trim_tail_frames only if end_frame is used
    # if args.trim_tail_frames is not None and args.end_frame is None:
    #      print("Warning: --trim_tail_frames specified but --end_frame is not. --trim_tail_frames will be ignored.")
    #      args.trim_tail_frames = None # Ignore trim if no end frame

    # Set HF_HOME environment variable
    os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(args.hf_home))
    os.makedirs(os.environ['HF_HOME'], exist_ok=True)

    return args


def load_models(args):
    """Loads all necessary models."""
    print("Loading models...")
    # Login if needed
    # login() # Uncomment if you need to login for gated models

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(gpu if torch.cuda.is_available() else cpu)
    print(f"Using device: {device}")

    # Load models to CPU first
    print("  Loading Text Encoder 1 (Llama)...")
    text_encoder = LlamaModel.from_pretrained(args.text_encoder_path, subfolder='text_encoder', torch_dtype=torch.float16).cpu()
    print("  Loading Text Encoder 2 (CLIP)...")
    text_encoder_2 = CLIPTextModel.from_pretrained(args.text_encoder_2_path, subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
    print("  Loading Tokenizer 1 (Llama)...")
    tokenizer = LlamaTokenizerFast.from_pretrained(args.text_encoder_path, subfolder='tokenizer')
    print("  Loading Tokenizer 2 (CLIP)...")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.text_encoder_2_path, subfolder='tokenizer_2')
    print("  Loading VAE...")
    vae = AutoencoderKLHunyuanVideo.from_pretrained(args.vae_path, subfolder='vae', torch_dtype=torch.float16).cpu()
    print("  Loading Image Feature Extractor (SigLIP)...")
    feature_extractor = SiglipImageProcessor.from_pretrained(args.image_encoder_path, subfolder='feature_extractor')
    print("  Loading Image Encoder (SigLIP)...")
    image_encoder = SiglipVisionModel.from_pretrained(args.image_encoder_path, subfolder='image_encoder', torch_dtype=torch.float16).cpu()
    print("  Loading Transformer (FramePack)...")
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(args.transformer_path, torch_dtype=torch.bfloat16).cpu()

    # Set to evaluation mode
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()

    # Apply settings
    transformer.high_quality_fp32_output_for_inference = True
    print('transformer.high_quality_fp32_output_for_inference = True')

    # Disable gradients
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    print("Models loaded.")
    return {
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "vae": vae,
        "feature_extractor": feature_extractor,
        "image_encoder": image_encoder,
        "transformer": transformer,
        "device": device
    }

def adjust_to_multiple(value, multiple):
    """Rounds down value to the nearest multiple."""
    return (value // multiple) * multiple

def mix_latents(latent_a, latent_b, weight_b):
    """Mix two latents with the specified weight for latent_b."""
    # Ensure latents are on the same device, same dtype
    if latent_a.device != latent_b.device:
        latent_b = latent_b.to(latent_a.device)
    if latent_a.dtype != latent_b.dtype:
        latent_b = latent_b.to(dtype=latent_a.dtype)

    # Ensure weight is a scalar tensor or float
    if isinstance(weight_b, torch.Tensor):
        weight_b = weight_b.item() # Convert tensor to float if necessary

    # Check for valid weight
    if not 0.0 <= weight_b <= 1.0:
        print(f"Warning: mix_latents received weight {weight_b}, clamping to [0, 1]")
        weight_b = max(0.0, min(1.0, weight_b))

    # Interpolate between latents (weight_b is the strength of latent_b)
    return (1.0 - weight_b) * latent_a + weight_b * latent_b

def preprocess_image_for_generation(image_path, target_width, target_height, job_id, output_dir, frame_name="input"):
    """Loads, processes, and saves a single image."""
    try:
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
    except Exception as e:
        print(f"Error loading image '{image_path}': {e}")
        raise

    H_orig, W_orig, _ = image_np.shape
    print(f"  {frame_name.capitalize()} image loaded ({W_orig}x{H_orig}): '{image_path}'")

    image_resized_np = resize_and_center_crop(image_np, target_width=target_width, target_height=target_height)
    Image.fromarray(image_resized_np).save(output_dir / f'{job_id}_{frame_name}_resized_{target_width}x{target_height}.png')

    image_pt = torch.from_numpy(image_resized_np).float() / 127.5 - 1.0
    image_pt = image_pt.permute(2, 0, 1)[None, :, None] # B=1, C=3, T=1, H, W
    print(f"  {frame_name.capitalize()} image processed to tensor shape: {image_pt.shape}")

    return image_np, image_resized_np, image_pt


@torch.no_grad()
def generate_video(args, models):
    """Generates the video using the loaded models and arguments."""

    # Unpack models
    text_encoder = models["text_encoder"]
    text_encoder_2 = models["text_encoder_2"]
    tokenizer = models["tokenizer"]
    tokenizer_2 = models["tokenizer_2"]
    vae = models["vae"]
    feature_extractor = models["feature_extractor"]
    image_encoder = models["image_encoder"]
    transformer = models["transformer"]
    device = models["device"] # Use the device determined during loading

    # --- Determine Memory Mode ---
    if args.high_vram and args.low_vram:
        print("Warning: Both --high_vram and --low_vram specified. Defaulting to auto-detection.")
        force_high_vram = force_low_vram = False
    else:
        force_high_vram = args.high_vram
        force_low_vram = args.low_vram

    if force_high_vram:
        high_vram = True
    elif force_low_vram:
        high_vram = False
    else:
        # Use gpu directly if device is cuda, otherwise assume high_vram=False (cpu case)
        free_mem_gb = get_cuda_free_memory_gb(device) if device.type == 'cuda' else 0
        high_vram = free_mem_gb > 60 # Heuristic, adjust as needed
        print(f'Auto-detected Free VRAM {free_mem_gb:.2f} GB')

    print(f'High-VRAM Mode: {high_vram}')

    # --- Configure Models based on VRAM mode ---
    if not high_vram:
        print("Configuring for Low VRAM mode...")
        vae.enable_slicing()
        vae.enable_tiling()
        # DynamicSwapInstaller is like huggingface's enable_sequential_offload but faster
        print("  Installing DynamicSwap for Transformer...")
        DynamicSwapInstaller.install_model(transformer, device=device)
        print("  Installing DynamicSwap for Text Encoder 1...")
        DynamicSwapInstaller.install_model(text_encoder, device=device)
        # Initial unload in low VRAM mode
        print("Unloading models from GPU (Low VRAM setup)...")
        unload_complete_models(
            text_encoder, text_encoder_2, image_encoder, vae, transformer
        )
    else:
        print("Configuring for High VRAM mode (moving models to GPU)...")
        text_encoder.to(device)
        text_encoder_2.to(device)
        image_encoder.to(device)
        vae.to(device)
        transformer.to(device)
        print("  Models moved to GPU.")

    # --- Prepare Inputs ---
    print("Preparing inputs...")
    prompt = args.prompt
    n_prompt = args.negative_prompt
    seed = args.seed
    total_second_length = args.total_second_length
    latent_window_size = args.latent_window_size
    steps = args.steps
    cfg = args.cfg
    gs = args.distilled_guidance_scale
    rs = args.rs
    gpu_memory_preservation = args.gpu_memory_preservation
    use_teacache = args.use_teacache
    fps = args.fps
    end_frame_path = args.end_frame
    # trim_tail_frames = args.trim_tail_frames # Revisit if needed
    end_frame_influence = args.end_frame_influence # Global end frame influence (overridden by sections)
    end_frame_weight = args.end_frame_weight # Global end frame weight (overridden by sections)
    section_data = args.section_data # Dict: {index_from_end: (img_path, prompt_text)}
    save_intermediate = args.save_intermediate_sections
    save_section_frames = args.save_section_final_frames

    # Calculate total latent sections (Mimics Gradio Demo)
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    print(f"Calculated total latent sections: {total_latent_sections}")

    job_id = generate_timestamp() + f"_seed{seed}"
    output_dir = Path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_video_path = None # Track the path of the final complete video

    # --- Section Preprocessing Storage ---
    section_latents = {}            # {index: tensor}
    section_image_embeddings = {}   # {index: tensor}
    section_prompt_embeddings = {}  # {index: (llama_vec, clip_pooler)}

    try:
        # --- Text Encoding (Global Prompts) ---
        print("Encoding global text prompts...")
        if not high_vram:
            print("  Low VRAM mode: Loading Text Encoders to GPU...")
            fake_diffusers_current_device(text_encoder, device)
            load_model_as_complete(text_encoder_2, target_device=device)
            print("  Text Encoders loaded.")

        global_llama_vec, global_clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1.0:
             print("  CFG scale is 1.0, using zero negative embeddings.")
             global_llama_vec_n, global_clip_l_pooler_n = torch.zeros_like(global_llama_vec), torch.zeros_like(global_clip_l_pooler)
        else:
             print(f"  Encoding negative prompt: '{n_prompt}'")
             global_llama_vec_n, global_clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        global_llama_vec, global_llama_attention_mask = crop_or_pad_yield_mask(global_llama_vec, length=512)
        global_llama_vec_n, global_llama_attention_mask_n = crop_or_pad_yield_mask(global_llama_vec_n, length=512)
        print("  Global text encoded and processed.")

        # --- Section Text Encoding ---
        if section_data:
            print("Encoding section-specific prompts...")
            for section_index, (img_path, prompt_text) in section_data.items():
                if prompt_text:
                    print(f"  Encoding prompt for section {section_index}: '{prompt_text}'")
                    sec_llama_vec, sec_clip_pooler = encode_prompt_conds(prompt_text, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                    sec_llama_vec, _ = crop_or_pad_yield_mask(sec_llama_vec, length=512) # Mask not needed here
                    section_prompt_embeddings[section_index] = (sec_llama_vec, sec_clip_pooler)
                    print(f"    Section {section_index} prompt encoded.")
                else:
                     print(f"  Section {section_index} has no specific prompt, will use global prompt.")

        if not high_vram:
            print("  Low VRAM mode: Unloading Text Encoders from GPU...")
            unload_complete_models(text_encoder_2)
            print("  Text Encoder 2 unloaded.") # TE1 might still be needed by transformer swap

        # --- Input Image Processing & Dimension Calculation ---
        print("Processing input image and determining dimensions...")
        try:
            input_image_np_orig, _, _ = preprocess_image_for_generation(
                args.input_image, 1, 1, job_id, output_dir, "temp_input_orig"
            ) # Load just to get original dimensions
        except Exception as e:
             print(f"Error loading input image '{args.input_image}' for dimension check: {e}")
             raise
        H_orig, W_orig, _ = input_image_np_orig.shape
        print(f"  Input image original size: {W_orig}x{H_orig}")


        # Determine Target Dimensions
        if args.width is not None and args.height is not None:
            target_w = args.width
            target_h = args.height
            print(f"  Using explicit target dimensions: {target_w}x{target_h}")
        elif args.target_resolution is not None:
            print(f"  Calculating dimensions based on target resolution for longer side: {args.target_resolution}")
            target_h, target_w = find_nearest_bucket(H_orig, W_orig, resolution=args.target_resolution)
            print(f"  Calculated dimensions (before adjustment): {target_w}x{target_h}")
        else:
            raise ValueError("Internal Error: Resolution determination failed.") # Should not happen due to default

        # Adjust Dimensions to be Divisible by DIMENSION_MULTIPLE
        final_w = adjust_to_multiple(target_w, DIMENSION_MULTIPLE)
        final_h = adjust_to_multiple(target_h, DIMENSION_MULTIPLE)

        if final_w <= 0 or final_h <= 0:
            print(f"Error: Calculated dimensions ({target_w}x{target_h}) resulted in non-positive dimensions after adjusting to be divisible by {DIMENSION_MULTIPLE} ({final_w}x{final_h}).")
            raise ValueError("Adjusted dimensions are invalid.")

        if final_w != target_w or final_h != target_h:
            print(f"Warning: Adjusted dimensions from {target_w}x{target_h} to {final_w}x{final_h} to be divisible by {DIMENSION_MULTIPLE}.")
        else:
            print(f"  Final dimensions confirmed: {final_w}x{final_h}")

        width, height = final_w, final_h

        if width * height > 1024 * 1024:
             print(f"Warning: Target resolution {width}x{height} is large. Ensure you have sufficient VRAM.")

        # Process Start Frame
        _, input_image_resized_np, input_image_pt = preprocess_image_for_generation(
            args.input_image, width, height, job_id, output_dir, "input"
        )

        # Process End Frame (if provided)
        end_frame_resized_np = None
        end_frame_pt = None
        if end_frame_path:
            _, end_frame_resized_np, end_frame_pt = preprocess_image_for_generation(
                end_frame_path, width, height, job_id, output_dir, "end"
            )

        # Process Section Frames (if provided)
        section_images_resized_np = {} # {index: np_array}
        section_images_pt = {}         # {index: tensor}
        if section_data:
            print("Processing section keyframe images...")
            for section_index, (img_path, _) in section_data.items():
                _, sec_resized_np, sec_pt = preprocess_image_for_generation(
                    img_path, width, height, job_id, output_dir, f"section{section_index}"
                )
                section_images_resized_np[section_index] = sec_resized_np
                section_images_pt[section_index] = sec_pt


        # --- VAE Encoding ---
        print("VAE encoding initial frame...")
        if not high_vram:
            print("  Low VRAM mode: Loading VAE to GPU...")
            load_model_as_complete(vae, target_device=device)
            print("  VAE loaded.")

        input_image_pt_dev = input_image_pt.to(device=device, dtype=vae.dtype)
        start_latent = vae_encode(input_image_pt_dev, vae) # B, C, 1, H/8, W/8
        print(f"  Initial latent shape: {start_latent.shape}")
        print(f"  Start latent stats - Min: {start_latent.min().item():.4f}, Max: {start_latent.max().item():.4f}, Mean: {start_latent.mean().item():.4f}")

        end_frame_latent = None
        if end_frame_pt is not None:
            print("VAE encoding end frame...")
            end_frame_pt_dev = end_frame_pt.to(device=device, dtype=vae.dtype)
            end_frame_latent = vae_encode(end_frame_pt_dev, vae)
            print(f"  End frame latent shape: {end_frame_latent.shape}")
            print(f"  End frame latent stats - Min: {end_frame_latent.min().item():.4f}, Max: {end_frame_latent.max().item():.4f}, Mean: {end_frame_latent.mean().item():.4f}")
            if end_frame_latent.shape != start_latent.shape:
                print(f"Warning: End frame latent shape mismatch. Reshaping.")
                try:
                    end_frame_latent = end_frame_latent.reshape(start_latent.shape)
                except Exception as reshape_err:
                     print(f"Error reshaping end frame latent: {reshape_err}. Disabling end frame.")
                     end_frame_latent = None

        if section_images_pt:
             print("VAE encoding section keyframes...")
             for section_index, sec_pt in section_images_pt.items():
                 sec_pt_dev = sec_pt.to(device=device, dtype=vae.dtype)
                 sec_latent = vae_encode(sec_pt_dev, vae)
                 print(f"  Section {section_index} latent shape: {sec_latent.shape}")
                 if sec_latent.shape != start_latent.shape:
                     print(f"  Warning: Section {section_index} latent shape mismatch. Reshaping.")
                     try:
                         sec_latent = sec_latent.reshape(start_latent.shape)
                     except Exception as reshape_err:
                         print(f"  Error reshaping section {section_index} latent: {reshape_err}. Skipping section latent.")
                         continue # Skip adding this latent
                 section_latents[section_index] = sec_latent.cpu() # Store on CPU
                 print(f"  Section {section_index} latent encoded and stored on CPU.")


        if not high_vram:
            print("  Low VRAM mode: Unloading VAE from GPU...")
            unload_complete_models(vae)
            print("  VAE unloaded.")

        # Move essential latents to CPU after encoding
        start_latent = start_latent.cpu()
        if end_frame_latent is not None:
            end_frame_latent = end_frame_latent.cpu()


        # --- CLIP Vision Encoding ---
        print("CLIP Vision encoding image(s)...")
        if not high_vram:
            print("  Low VRAM mode: Loading Image Encoder to GPU...")
            load_model_as_complete(image_encoder, target_device=device)
            print("  Image Encoder loaded.")

        # Encode start frame
        image_encoder_output = hf_clip_vision_encode(input_image_resized_np, feature_extractor, image_encoder)
        start_image_embedding = image_encoder_output.last_hidden_state # B, SeqLen, Dim
        print(f"  Start image embedding shape: {start_image_embedding.shape}")

        # Encode end frame
        end_frame_embedding = None
        if end_frame_resized_np is not None:
            print("  Encoding end frame with CLIP Vision...")
            end_frame_output = hf_clip_vision_encode(end_frame_resized_np, feature_extractor, image_encoder)
            end_frame_embedding = end_frame_output.last_hidden_state
            print(f"  End frame embedding shape: {end_frame_embedding.shape}")

        # Encode section frames
        if section_images_resized_np:
             print("CLIP Vision encoding section keyframes...")
             for section_index, sec_resized_np in section_images_resized_np.items():
                 sec_output = hf_clip_vision_encode(sec_resized_np, feature_extractor, image_encoder)
                 sec_embedding = sec_output.last_hidden_state
                 section_image_embeddings[section_index] = sec_embedding.cpu() # Store on CPU
                 print(f"  Section {section_index} embedding shape: {sec_embedding.shape}. Stored on CPU.")


        if not high_vram:
            print("  Low VRAM mode: Unloading Image Encoder from GPU...")
            unload_complete_models(image_encoder)
            print("  Image Encoder unloaded.")

        # Move essential embeddings to CPU
        start_image_embedding = start_image_embedding.cpu()
        if end_frame_embedding is not None:
             end_frame_embedding = end_frame_embedding.cpu()


        # --- Prepare Embeddings for Transformer ---
        print("Preparing embeddings for Transformer...")
        target_dtype = transformer.dtype
        # Global prompts
        global_llama_vec = global_llama_vec.cpu().to(target_dtype)
        global_llama_vec_n = global_llama_vec_n.cpu().to(target_dtype)
        global_clip_l_pooler = global_clip_l_pooler.cpu().to(target_dtype)
        global_clip_l_pooler_n = global_clip_l_pooler_n.cpu().to(target_dtype)
        # Section prompts
        for idx in section_prompt_embeddings:
            sec_llama, sec_clip = section_prompt_embeddings[idx]
            section_prompt_embeddings[idx] = (sec_llama.cpu().to(target_dtype), sec_clip.cpu().to(target_dtype))
        # Image embeddings
        start_image_embedding = start_image_embedding.cpu().to(target_dtype)
        if end_frame_embedding is not None:
            end_frame_embedding = end_frame_embedding.cpu().to(target_dtype)
        for idx in section_image_embeddings:
             section_image_embeddings[idx] = section_image_embeddings[idx].cpu().to(target_dtype)
        print(f"  Embeddings prepared on CPU with dtype {target_dtype}.")


        # --- Sampling Setup ---
        print("Setting up sampling...")
        rnd = torch.Generator(cpu).manual_seed(seed)
        num_frames = latent_window_size * 4 - 3 # Latent frames generated per sampling step
        print(f"  Latent frames per sampling step (num_frames input): {num_frames}")

        latent_c, latent_h, latent_w = start_latent.shape[1], start_latent.shape[3], start_latent.shape[4]
        # history_latents stores the CONTEXT (downsampled previous outputs) for conditioning the next step
        context_latents = torch.zeros(size=(1, latent_c, 1 + 2 + 16, latent_h, latent_w), dtype=torch.float32).cpu()

        # accumulated_generated_latents stores the actual generated video latents in sequence
        accumulated_generated_latents = None # Will store B, C, T, H, W on CPU

        # history_pixels stores the decoded video frames
        history_pixels = None # Will store B, C, T, H, W on CPU

        # FramePack generates backward from the end. Paddings represent section index from the END.
        # Example: total_latent_sections=4 -> [3, 2, 1, 0]
        # Example: total_latent_sections=5 -> [3, 2, 2, 1, 0] (endframe.py logic)
        latent_paddings = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            print(f"  Using adjusted padding sequence for >4 sections: {latent_paddings}")
        else:
            print(f"  Using standard padding sequence: {latent_paddings}")

        # Initialize context buffer with end frame latent if available (matching endframe.py)
        if end_frame_latent is not None:
            print("  Initializing context buffer's first slot with end frame latent.")
            context_latents[:, :, 0:1, :, :] = end_frame_latent.float() # Use the 1x slot for the closest context

        # --- Main Sampling Loop (Generates Backward: End -> Start) ---
        start_time = time.time()

        for i_loop, latent_padding in enumerate(latent_paddings):
            section_start_time = time.time()
            current_section_index_from_end = latent_padding # Index 0 is last generated (video start)
            is_first_generation_step = (i_loop == 0) # Corresponds to video END
            is_last_generation_step = (latent_padding == 0) # Corresponds to video START

            print(f"\n--- Starting Generation Step {i_loop+1}/{len(latent_paddings)} (Section Index from End: {current_section_index_from_end}, Last Gen Step: {is_last_generation_step}) ---")
            latent_padding_size = latent_padding * latent_window_size
            print(f'  Padding size (latent frames): {latent_padding_size}, Window size (latent frames): {latent_window_size}')

            # --- Select Conditioning Inputs for this Section ---

            # 1. Conditioning Latent (clean_latents_pre = target appearance for this step)
            if current_section_index_from_end in section_latents:
                current_conditioning_latent = section_latents[current_section_index_from_end].cpu().float()
                print(f"  Using SECTION {current_section_index_from_end} latent as conditioning latent.")
            else:
                # Default: Use start frame latent as the target appearance
                # End frame latent is used for context init, not default target appearance
                current_conditioning_latent = start_latent.cpu().float()
                print(f"  Using START frame latent as conditioning latent.")

            # 2. Image Embedding (image_embeddings input to transformer)
            if current_section_index_from_end in section_image_embeddings:
                current_image_embedding = section_image_embeddings[current_section_index_from_end].cpu()
                print(f"  Using SECTION {current_section_index_from_end} image embedding.")
            else:
                 # Default: Use start frame embedding
                 current_image_embedding = start_image_embedding.cpu()
                 print(f"  Using START frame image embedding.")
                 # NOTE: Could re-introduce end_frame_influence logic here as a fallback if needed,
                 # but section keyframes are intended to override it.

            # 3. Text Embedding (prompt_embeds, prompt_poolers inputs)
            if current_section_index_from_end in section_prompt_embeddings:
                 current_llama_vec, current_clip_pooler = section_prompt_embeddings[current_section_index_from_end]
                 current_llama_vec = current_llama_vec.cpu()
                 current_clip_pooler = current_clip_pooler.cpu()
                 print(f"  Using SECTION {current_section_index_from_end} prompt embeddings.")
            else:
                 current_llama_vec = global_llama_vec.cpu()
                 current_clip_pooler = global_clip_l_pooler.cpu()
                 print(f"  Using GLOBAL prompt embeddings.")

            # Negative prompt is always global for simplicity
            current_llama_vec_n = global_llama_vec_n.cpu()
            current_clip_pooler_n = global_clip_l_pooler_n.cpu()
            current_llama_attention_mask = global_llama_attention_mask # Use global masks
            current_llama_attention_mask_n = global_llama_attention_mask_n


            # --- Prepare Sampler Inputs ---
            # Define indices
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = \
                indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            # Prepare conditioning latents from context buffer
            clean_latents_pre = current_conditioning_latent # Use the selected target appearance latent
            # Get context from the history buffer
            clean_latents_post, clean_latents_2x, clean_latents_4x = \
                context_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2) # context_latents is float32 cpu
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            print(f"  Conditioning shapes (CPU): clean={clean_latents.shape}, 2x={clean_latents_2x.shape}, 4x={clean_latents_4x.shape}")


            # Load Transformer (Low VRAM)
            if not high_vram:
                print("  Moving Transformer to GPU...")
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation)
                fake_diffusers_current_device(text_encoder, device) # Keep TE1 active for swap

            # Configure TeaCache
            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                print("  TeaCache enabled.")
            else:
                transformer.initialize_teacache(enable_teacache=False)
                print("  TeaCache disabled.")


            # --- Run Sampling ---
            print(f"  Starting sampling ({steps} steps) for {num_frames} latent frames...")
            sampling_step_start_time = time.time()

            pbar = tqdm(total=steps, desc=f"  Section {current_section_index_from_end} Sampling", leave=False)
            def callback(d):
                pbar.update(1)
                # Could add preview logic here if needed, like in endframe.py
                return

            current_device = transformer.device # Device where transformer currently resides

            # Move necessary tensors to the sampling device just before the call
            _prompt_embeds = current_llama_vec.to(text_encoder.device if not high_vram else device)
            _prompt_embeds_mask = current_llama_attention_mask.to(text_encoder.device if not high_vram else device)
            _prompt_poolers = current_clip_pooler.to(current_device)
            _negative_prompt_embeds = current_llama_vec_n.to(text_encoder.device if not high_vram else device)
            _negative_prompt_embeds_mask = current_llama_attention_mask_n.to(text_encoder.device if not high_vram else device)
            _negative_prompt_poolers = current_clip_pooler_n.to(current_device)
            _image_embeddings = current_image_embedding.to(current_device) # Use selected embedding

            _latent_indices = latent_indices.to(current_device)
            _clean_latents = clean_latents.to(current_device, dtype=transformer.dtype)
            _clean_latent_indices = clean_latent_indices.to(current_device)
            _clean_latents_2x = clean_latents_2x.to(current_device, dtype=transformer.dtype)
            _clean_latent_2x_indices = clean_latent_2x_indices.to(current_device)
            _clean_latents_4x = clean_latents_4x.to(current_device, dtype=transformer.dtype)
            _clean_latent_4x_indices = clean_latent_4x_indices.to(current_device)

            generated_latents_gpu = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames, # Generate this many *new* latent frames
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=_prompt_embeds,
                prompt_embeds_mask=_prompt_embeds_mask,
                prompt_poolers=_prompt_poolers,
                negative_prompt_embeds=_negative_prompt_embeds,
                negative_prompt_embeds_mask=_negative_prompt_embeds_mask,
                negative_prompt_poolers=_negative_prompt_poolers,
                device=current_device,
                dtype=transformer.dtype,
                image_embeddings=_image_embeddings, # Pass the selected embedding
                latent_indices=_latent_indices,
                clean_latents=_clean_latents,
                clean_latent_indices=_clean_latent_indices,
                clean_latents_2x=_clean_latents_2x,
                clean_latent_2x_indices=_clean_latent_2x_indices,
                clean_latents_4x=_clean_latents_4x,
                clean_latent_4x_indices=_clean_latent_4x_indices,
                callback=callback,
            )
            pbar.close()
            sampling_step_end_time = time.time()
            print(f"  Sampling finished in {sampling_step_end_time - sampling_step_start_time:.2f} seconds.")
            print(f"  Raw generated latent shape for this step: {generated_latents_gpu.shape}")
            print(f"  Generated latents stats (GPU) - Min: {generated_latents_gpu.min().item():.4f}, Max: {generated_latents_gpu.max().item():.4f}, Mean: {generated_latents_gpu.mean().item():.4f}")

            generated_latents_cpu = generated_latents_gpu.cpu().float()
            del generated_latents_gpu # Free GPU memory


            # Offload Transformer and TE1 (Low VRAM)
            if not high_vram:
                print("  Low VRAM mode: Offloading Transformer and Text Encoder from GPU...")
                offload_model_from_device_for_memory_preservation(transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation)
                offload_model_from_device_for_memory_preservation(text_encoder, target_device=device, preserved_memory_gb=gpu_memory_preservation)
                print("  Transformer and Text Encoder offloaded.")


            # --- History/Context Update (Matching endframe.py logic) ---

            # Prepend Start Frame Latent if this is the last generation step (video start)
            if is_last_generation_step:
                print("  Last generation step: Prepending start frame latent to generated latents.")
                generated_latents_cpu = torch.cat([start_latent.cpu().float(), generated_latents_cpu], dim=2)
                print(f"  Shape after prepending start latent: {generated_latents_cpu.shape}")

            # Update context buffer by prepending the newly generated section
            # This context is used for conditioning the *next* generation step (further back in time)
            context_latents = torch.cat([generated_latents_cpu.to(context_latents), context_latents], dim=2)
            print(f"  Context buffer updated. New shape: {context_latents.shape}")


            # Accumulate the generated latents for the final video output
            # Since generation is backward, prepend the new latents to the accumulator
            if accumulated_generated_latents is None:
                 accumulated_generated_latents = generated_latents_cpu
            else:
                 accumulated_generated_latents = torch.cat([generated_latents_cpu, accumulated_generated_latents], dim=2)

            current_total_latent_frames = accumulated_generated_latents.shape[2]
            print(f"  Accumulated generated latents updated. Total latent frames: {current_total_latent_frames}")
            print(f"  Accumulated latents stats - Min: {accumulated_generated_latents.min().item():.4f}, Max: {accumulated_generated_latents.max().item():.4f}, Mean: {accumulated_generated_latents.mean().item():.4f}")


            # --- VAE Decoding & Merging (Using logic from endframe.py) ---
            print("  Decoding generated latents and merging video...")
            decode_start_time = time.time()

            if not high_vram:
                print("    Moving VAE to GPU...")
                # Ensure other large models are offloaded before loading VAE
                offload_model_from_device_for_memory_preservation(transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation)
                unload_complete_models(text_encoder, text_encoder_2, image_encoder)
                load_model_as_complete(vae, target_device=device)
                print("    VAE loaded.")


            # Append smoothly (Prepending `current_pixels` to `history_pixels`)
            if history_pixels is None:
                 # First section generated (video end): Decode the generated latents directly
                 latents_to_decode_this_step = generated_latents_cpu.to(device=device, dtype=vae.dtype)
                 print(f"    First section: Decoding generated latents directly, shape: {latents_to_decode_this_step.shape}")
                 history_pixels = vae_decode(latents_to_decode_this_step, vae).cpu()
                 print(f"    Initialized history_pixels shape: {history_pixels.shape}")
                 print(f"    Initial pixels stats - Min: {history_pixels.min().item():.4f}, Max: {history_pixels.max().item():.4f}, Mean: {history_pixels.mean().item():.4f}")
            else:
                # Number of *pixel* frames corresponding to latent_window_size latents minus overlap adjustment
                overlapped_frames = latent_window_size * 4 - 3

                # Decode the necessary portion of the *accumulated* latents for smooth append
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_generation_step else (latent_window_size * 2)
                print(f"    Decoding {section_latent_frames} latent frames from *accumulated history start* for append.")
                latents_for_append = accumulated_generated_latents[:, :, :section_latent_frames].to(device=device, dtype=vae.dtype)
                # Decode this slice *to be used in the append*
                current_pixels = vae_decode(latents_for_append, vae).cpu()
                print(f"    Decoded pixels for append stats - Min: {current_pixels.min().item():.4f}, Max: {current_pixels.max().item():.4f}, Mean: {current_pixels.mean().item():.4f}")
                # Using variable name current_pixels_for_append to avoid confusion if debugging original variable
                current_pixels_for_append = current_pixels
                print(f"    Decoded pixels for append shape: {current_pixels_for_append.shape}")
                # --- End: Replicate endframe.py VAE decode logic ---

                print(f"    Appending section with pixel overlap: {overlapped_frames}")
                # Use the pixels decoded from the history slice for appending
                history_pixels = soft_append_bcthw(current_pixels_for_append, history_pixels, overlap=overlapped_frames)
                print(f"    Appended. New total pixel shape: {history_pixels.shape}")
                print(f"    Combined pixels stats - Min: {history_pixels.min().item():.4f}, Max: {history_pixels.max().item():.4f}, Mean: {history_pixels.mean().item():.4f}")

            # Unload VAE (Low VRAM)
            if not high_vram:
                print("    Low VRAM mode: Unloading VAE from GPU...")
                unload_complete_models(vae)
                print("    VAE unloaded.")

            decode_end_time = time.time()
            print(f"  Decoding and merging finished in {decode_end_time - decode_start_time:.2f} seconds.")


            # --- Save Intermediate/Section Output ---
            current_num_pixel_frames = history_pixels.shape[2]

            # Save final frame of the *decoded* section
            if save_section_frames:
                try:
                    last_frame_of_section = current_pixels[0, :, -1, :, :] # Get the last frame of the *newly decoded* part
                    last_frame_of_section = einops.rearrange(last_frame_of_section, 'c h w -> h w c')
                    last_frame_of_section_np = last_frame_of_section.cpu().numpy()
                    last_frame_of_section_np = np.clip((last_frame_of_section_np * 127.5 + 127.5), 0, 255).astype(np.uint8)
                    # Maybe resize back to original aspect for saving? Optional. Using target H/W for now.
                    # last_frame_of_section_np = resize_and_center_crop(last_frame_of_section_np, target_width=width, target_height=height)
                    section_frame_filename = output_dir / f'{job_id}_section_end_frame_idx{current_section_index_from_end}.png'
                    Image.fromarray(last_frame_of_section_np).save(section_frame_filename)
                    print(f"  Saved final frame of section {current_section_index_from_end} to: {section_frame_filename}")
                except Exception as e:
                     print(f"  [WARN] Error saving section {current_section_index_from_end} final frame image: {e}")

            # Save intermediate video
            if save_intermediate or is_last_generation_step: # Always save on last step
                output_filename = output_dir / f'{job_id}_step{i_loop+1}_idx{current_section_index_from_end}_frames{current_num_pixel_frames}_{width}x{height}.mp4'
                print(f"  Saving {'intermediate' if not is_last_generation_step else 'final'} video ({current_num_pixel_frames} frames) to: {output_filename}")
                try:
                    save_bcthw_as_mp4(history_pixels.float(), str(output_filename), fps=int(fps))
                    print(f"  Saved video using save_bcthw_as_mp4")
                    if not is_last_generation_step:
                        print(f"INTERMEDIATE_VIDEO_PATH:{output_filename}") # For external parsing
                    final_video_path = str(output_filename) # Keep track of last successful save
                except Exception as e:
                    print(f"  Error saving video using save_bcthw_as_mp4: {e}")
                    traceback.print_exc()
                    # Try saving first frame as fallback
                    try:
                        first_frame = history_pixels.float()[0, :, 0].permute(1, 2, 0).cpu().numpy()
                        first_frame = (first_frame * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
                        frame_path = str(output_filename).replace('.mp4', '_first_frame_ERROR.png')
                        Image.fromarray(first_frame).save(frame_path)
                        print(f"  Saved first frame as image to {frame_path} due to video saving error.")
                    except Exception as frame_err:
                        print(f"  Could not save first frame either: {frame_err}")


            section_end_time = time.time()
            print(f"--- Generation Step {i_loop+1} finished in {section_end_time - section_start_time:.2f} seconds ---")

            if is_last_generation_step:
                print("\nFinal generation step completed.")
                break # Exit loop


        # --- Final Video Saved During Last Step ---
        if final_video_path and os.path.exists(final_video_path):
            print(f"\nSuccessfully generated: {final_video_path}")
            print(f"ACTUAL_FINAL_PATH:{final_video_path}") # Distinct print for parsing
            return final_video_path
        else:
             print("\nError: Final video path not found or not saved correctly.")
             return None


    except Exception as e:
        print("\n--- ERROR DURING GENERATION ---")
        traceback.print_exc()
        print("-----------------------------")
        # Attempt to save partial video if history_pixels exists
        if 'history_pixels' in locals() and history_pixels is not None and history_pixels.shape[2] > 0:
             partial_output_name = output_dir / f"{job_id}_partial_ERROR_{history_pixels.shape[2]}_frames_{width}x{height}.mp4"
             print(f"Attempting to save partial video to: {partial_output_name}")
             try:
                 save_bcthw_as_mp4(history_pixels.float(), str(partial_output_name), fps=fps)
                 print(f"ACTUAL_FINAL_PATH:{partial_output_name}")
                 return str(partial_output_name)
             except Exception as save_err:
                 print(f"Error saving partial video during error handling: {save_err}")
                 traceback.print_exc()

        print("Status: Error occurred, no video saved.")
        return None

    finally:
        # Final cleanup
        print("Performing final model cleanup...")
        try:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        except Exception as e:
             print(f"Error during final model unload: {e}")
             pass

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")


def main():
    args = parse_args()
    models = load_models(args)
    final_path = generate_video(args, models)
    if final_path:
        print(f"\nVideo generation finished. Final path: {final_path}")
        sys.exit(0)
    else:
        print("\nVideo generation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()