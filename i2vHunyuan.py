import os
import time
import random
import argparse
import logging
import threading
import numpy as np
import torch
import accelerate
import torchvision
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from einops import rearrange
from PIL import Image
import math
import av

from hunyuan_model.text_encoder import TextEncoder
from hunyuan_model.text_encoder import PROMPT_TEMPLATE
from hunyuan_model.vae import load_vae
from hunyuan_model.models import load_transformer, get_rotary_pos_embed
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from transformers.models.llama import LlamaModel
from diffusers.utils.torch_utils import randn_tensor
import torchvision.transforms as transforms

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global stop event for batch processing
stop_event = threading.Event()

def clean_memory_on_device(device):
    """Clean up CUDA memory if available"""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "cpu":
        pass
    elif device.type == "mps":
        torch.mps.empty_cache()

def synchronize_device(device: torch.device):
    """Synchronize device operations"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def generate_crop_size_list(base_size, aligned_size):
    """Generate a list of crop sizes with different aspect ratios"""
    crop_size_list = []
    
    # Add square size
    crop_size_list.append((base_size, base_size))
    
    # Add portrait and landscape sizes
    for ratio in [4/3, 3/2, 16/9, 2/1]:
        h = int(base_size * ratio)
        h = (h // aligned_size) * aligned_size
        crop_size_list.append((h, base_size))  # Portrait
        
        w = int(base_size * ratio)
        w = (w // aligned_size) * aligned_size
        crop_size_list.append((base_size, w))  # Landscape
    
    return crop_size_list

def get_closest_ratio(h, w, aspect_ratios, crop_size_list):
    """Find the closest aspect ratio and corresponding size"""
    img_ratio = round(float(h) / float(w), 5)
    closest_idx = np.argmin(np.abs(aspect_ratios - img_ratio))
    closest_size = crop_size_list[closest_idx]
    closest_ratio = aspect_ratios[closest_idx]
    return closest_size, closest_ratio

def preprocess_image_for_i2v(image_path, resolution="720p"):
    """Process image for i2v model with proper aspect ratio handling"""
    # Set base size based on resolution
    if resolution == "720p":
        bucket_hw_base_size = 960
    elif resolution == "540p":
        bucket_hw_base_size = 720
    elif resolution == "360p":
        bucket_hw_base_size = 480
    else:
        raise ValueError(f"i2v_resolution: {resolution} must be in [360p, 540p, 720p]")
    
    # Open and convert image
    image = Image.open(image_path).convert('RGB')
    origin_size = image.size  # (width, height)
    
    # Generate crop size list and find closest aspect ratio
    crop_size_list = generate_crop_size_list(bucket_hw_base_size, 32)
    aspect_ratios = np.array([round(float(h)/float(w), 5) for h, w in crop_size_list])
    closest_size, closest_ratio = get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)
    
    # Transform image
    transform = transforms.Compose([
        transforms.Resize(closest_size),
        transforms.CenterCrop(closest_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Process the image
    processed_image = transform(image).unsqueeze(0).unsqueeze(2)  # Add batch and time dimensions
    
    return processed_image, closest_size

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """Save video tensor to a file"""
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

    # Create output container
    container = av.open(path, mode="w")

    # Create video stream
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

def encode_prompt(prompt: str, device: torch.device, num_videos_per_prompt: int, text_encoder: TextEncoder):
    """Encode the prompt into text encoder hidden states"""
    data_type = "video"  # video only for Hunyuan I2V

    text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

    with torch.no_grad():
        prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type, device=device)
    prompt_embeds = prompt_outputs.hidden_state

    attention_mask = prompt_outputs.attention_mask
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        bs_embed, seq_len = attention_mask.shape
        attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
        attention_mask = attention_mask.view(bs_embed * num_videos_per_prompt, seq_len)

    prompt_embeds_dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    if prompt_embeds.ndim == 2:
        bs_embed, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
    else:
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds, attention_mask

def encode_input_prompt(prompt, device, args, fp8_llm=False, accelerator=None):
    """Encode input prompt with text encoders"""
    # Constants from config
    # Check if the templates exist in PROMPT_TEMPLATE, if not use defaults
    prompt_template_video = "dit-llm-encode-video"  # Default template for video
    if hasattr(args, 'prompt_template_video') and args.prompt_template_video in PROMPT_TEMPLATE:
        prompt_template_video = args.prompt_template_video
    
    prompt_template = "dit-llm-encode"  # Default template
    if hasattr(args, 'prompt_template') and args.prompt_template in PROMPT_TEMPLATE:
        prompt_template = args.prompt_template
    
    text_encoder_dtype = torch.float16
    text_encoder_type = "llm"
    text_len = 256
    hidden_state_skip_layer = 2
    apply_final_norm = False
    reproduce = False

    text_encoder_2_type = "clipL"
    text_len_2 = 77

    num_videos = 1

    # Get crop start from the prompt template
    crop_start = PROMPT_TEMPLATE[prompt_template_video].get("crop_start", 0)
    max_length = text_len + crop_start

    # prompt_template
    prompt_template_obj = PROMPT_TEMPLATE[prompt_template]

    # prompt_template_video
    prompt_template_video_obj = PROMPT_TEMPLATE[prompt_template_video]
    
    # Load text encoders
    logger.info(f"Loading text encoder: {args.text_encoder1}")
    text_encoder = TextEncoder(
        text_encoder_type=text_encoder_type,
        max_length=max_length,
        text_encoder_dtype=text_encoder_dtype,
        text_encoder_path=args.text_encoder1,
        tokenizer_type=text_encoder_type,
        prompt_template=prompt_template_obj,
        prompt_template_video=prompt_template_video_obj,
        hidden_state_skip_layer=hidden_state_skip_layer,
        apply_final_norm=apply_final_norm,
        reproduce=reproduce,
        i2v_mode=True
    )
    text_encoder.eval()
    if fp8_llm:
        org_dtype = text_encoder.dtype
        logger.info(f"Moving and casting text encoder to {device} and torch.float8_e4m3fn")
        text_encoder.to(device=device, dtype=torch.float8_e4m3fn)

        # Prepare LLM for fp8
        def prepare_fp8(llama_model: LlamaModel, target_dtype):
            def forward_hook(module):
                def forward(hidden_states):
                    input_dtype = hidden_states.dtype
                    hidden_states = hidden_states.to(torch.float32)
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(variance + module.variance_epsilon)
                    return module.weight.to(input_dtype) * hidden_states.to(input_dtype)
                return forward

            for module in llama_model.modules():
                if module.__class__.__name__ in ["Embedding"]:
                    module.to(target_dtype)
                if module.__class__.__name__ in ["LlamaRMSNorm"]:
                    module.forward = forward_hook(module)

        prepare_fp8(text_encoder.model, org_dtype)

    logger.info(f"Loading text encoder 2: {args.text_encoder2}")
    text_encoder_2 = TextEncoder(
        text_encoder_type=text_encoder_2_type,
        max_length=text_len_2,
        text_encoder_dtype=text_encoder_dtype,
        text_encoder_path=args.text_encoder2,
        tokenizer_type=text_encoder_2_type,
        reproduce=reproduce,
    )
    text_encoder_2.eval()

    # Encode prompt
    logger.info(f"Encoding prompt with text encoder 1")
    text_encoder.to(device=device)
    if fp8_llm:
        with accelerator.autocast():
            prompt_embeds, prompt_mask = encode_prompt(prompt, device, num_videos, text_encoder)
    else:
        prompt_embeds, prompt_mask = encode_prompt(prompt, device, num_videos, text_encoder)
    text_encoder = None
    clean_memory_on_device(device)

    logger.info(f"Encoding prompt with text encoder 2")
    text_encoder_2.to(device=device)
    prompt_embeds_2, prompt_mask_2 = encode_prompt(prompt, device, num_videos, text_encoder_2)

    prompt_embeds = prompt_embeds.to("cpu")
    prompt_mask = prompt_mask.to("cpu")
    prompt_embeds_2 = prompt_embeds_2.to("cpu")
    prompt_mask_2 = prompt_mask_2.to("cpu")

    text_encoder_2 = None
    clean_memory_on_device(device)

    return prompt_embeds, prompt_mask, prompt_embeds_2, prompt_mask_2

def prepare_vae(args, device):
    """Prepare VAE model"""
    vae_dtype = torch.float16
    vae, _, s_ratio, t_ratio = load_vae(vae_dtype=vae_dtype, device=device, vae_path=args.vae)
    vae.eval()

    # Set chunk_size to CausalConv3d recursively
    chunk_size = args.vae_chunk_size
    if chunk_size is not None:
        vae.set_chunk_size_for_causal_conv_3d(chunk_size)
        logger.info(f"Set chunk_size to {chunk_size} for CausalConv3d")

    if args.vae_spatial_tile_sample_min_size is not None:
        vae.enable_spatial_tiling(True)
        vae.tile_sample_min_size = args.vae_spatial_tile_sample_min_size
        vae.tile_latent_min_size = args.vae_spatial_tile_sample_min_size // 8
    else:
        vae.enable_spatial_tiling(True)

    return vae, vae_dtype

def resize_image_keeping_aspect_ratio(image_path, max_width, max_height):
    """Resize image keeping aspect ratio with dimensions divisible by 16"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Calculate new dimensions while maintaining aspect ratio
        if width > height:
            new_width = min(max_width, width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(max_height, height)
            new_width = int(new_height * aspect_ratio)
        
        # Make dimensions divisible by 16
        new_width = math.floor(new_width / 16) * 16
        new_height = math.floor(new_height / 16) * 16
        
        # Ensure minimum size
        new_width = max(16, new_width)
        new_height = max(16, new_height)
        
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save to temporary file
        temp_path = f"temp_resized_{os.path.basename(image_path)}"
        resized_img.save(temp_path)
        
        return temp_path, (new_width, new_height)
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_random_image_from_folder(folder_path):
    """Get a random image from the specified folder"""
    import glob
    
    if not os.path.isdir(folder_path):
        return None, f"Error: {folder_path} is not a valid directory"
    
    # Get all image files in the folder
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    for ext in ('*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.WEBP'):
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_files:
        return None, f"Error: No image files found in {folder_path}"
    
    # Select a random image
    random_image = random.choice(image_files)
    return random_image, f"Selected: {os.path.basename(random_image)}"

def add_metadata_to_video(video_path, parameters):
    """Add generation parameters to video metadata using ffmpeg"""
    import json
    import subprocess

    # Convert parameters to JSON string
    params_json = json.dumps(parameters, indent=2)
    
    # Temporary output path
    temp_path = video_path.replace(".mp4", "_temp.mp4")
    
    # FFmpeg command to add metadata without re-encoding
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-metadata', f'comment={params_json}',
        '-codec', 'copy',
        temp_path
    ]
    
    try:
        # Execute FFmpeg command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Replace original file with the metadata-enhanced version
        os.replace(temp_path, video_path)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to add metadata: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False
    except Exception as e:
        logger.error(f"Error adding metadata: {str(e)}")
        return False

def decode_latents(args, latents, device, vae=None):
    """Decode latents into images"""
    if vae is None:
        vae, vae_dtype = prepare_vae(args, device)
    else:
        vae_dtype = vae.dtype

    expand_temporal_dim = False
    if len(latents.shape) == 4:
        latents = latents.unsqueeze(2)
        expand_temporal_dim = True
    elif len(latents.shape) == 5:
        pass
    else:
        raise ValueError(f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}.")

    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
        latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    else:
        latents = latents / vae.config.scaling_factor

    latents = latents.to(device=device, dtype=vae_dtype)
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]

    if expand_temporal_dim:
        image = image.squeeze(2)

    image = (image / 2 + 0.5).clamp(0, 1)
    # We always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().float()

    return image

def generate_video(
    args,
    device,
    accelerator,
    prompt,
    negative_prompt,
    image_path,
    batch_size=1,
    seed=None,
    height=None,
    width=None,
    callback=None
):
    """
    Generate video from image using Hunyuan I2V model
    
    Args:
        args: Command line arguments
        device: torch device
        accelerator: accelerate instance
        prompt: Text prompt
        negative_prompt: Negative text prompt
        image_path: Path to input image
        batch_size: Batch size
        seed: Random seed
        height: Target height (optional, will use image dimensions if None)
        width: Target width (optional, will use image dimensions if None)
        callback: Progress callback function
        
    Returns:
        List of generated video paths
    """
    # Stop generation if requested
    if stop_event.is_set():
        return []
        
    # Set the dtype
    dit_dtype = torch.bfloat16
    dit_weight_dtype = torch.float8_e4m3fn if args.fp8 else dit_dtype
    logger.info(f"Using device: {device}, DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}")
    
    # Prepare mixed precision
    mixed_precision = "bf16" if dit_dtype == torch.bfloat16 else "fp16"
    if accelerator is None:
        accelerator = accelerate.Accelerator(mixed_precision=mixed_precision)
    
    # If batch size > 1 and only one image provided, we'll use the same image for all batches
    if image_path is None:
        raise ValueError("Image path is required for I2V generation")
    
    # Process image for i2v
    logger.info(f"Processing image for i2v: {image_path}")
    processed_image, (target_height, target_width) = preprocess_image_for_i2v(
        image_path, args.i2v_resolution
    )
    
    # Override height and width with the processed image dimensions if not provided
    if height is None:
        height = target_height
    if width is None:
        width = target_width
        
    # Encode prompt
    logger.info(f"Encoding prompt: {prompt}")
    
    # Prepare prompt for classifier-free guidance
    do_classifier_free_guidance = args.guidance_scale != 1.0
    if do_classifier_free_guidance:
        if negative_prompt is None:
            logger.info("Using default i2v negative prompt")
            negative_prompt = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
        logger.info(f"Encoding negative prompt: {negative_prompt}")
        text_prompt = [negative_prompt, prompt]
    else:
        text_prompt = prompt
        if negative_prompt is not None:
            logger.warning("Negative prompt is provided but guidance_scale is 1.0, negative prompt will be ignored.")
    
    # Encode prompt with text encoders
    prompt_embeds, prompt_mask, prompt_embeds_2, prompt_mask_2 = encode_input_prompt(
        text_prompt, device, args, args.fp8_llm, accelerator
    )
    
    # Encode image to latents
    processed_image = processed_image.to(device=device, dtype=dit_dtype)
    with torch.no_grad():
        vae, vae_dtype = prepare_vae(args, device)
        vae.to(device=device, dtype=dit_dtype)
        img_latents = vae.encode(processed_image).latent_dist.mode()
        img_latents = img_latents * vae.config.scaling_factor
    
    clean_memory_on_device(device)
    
    # Load transformer model
    blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0
    loading_device = "cpu" if blocks_to_swap > 0 else device
    
    logger.info(f"Loading DiT model from {args.dit}")
    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"
        
    # For I2V mode we use 33 input channels (16*2+1)
    dit_in_channels = 33
    
    transformer = load_transformer(
        args.dit,
        args.attn_mode,
        args.split_attn,
        loading_device,
        dit_dtype,
        in_channels=dit_in_channels,
        i2v_mode=True
    )
    transformer.eval()
    
    # Cast model to appropriate dtype
    if blocks_to_swap > 0:
        logger.info(f"Casting model to {dit_weight_dtype}")
        transformer.to(dtype=dit_weight_dtype)
        logger.info(f"Enable swap {blocks_to_swap} blocks to CPU from device: {device}")
        transformer.enable_block_swap(blocks_to_swap, device, supports_backward=False)
        transformer.move_to_device_except_swap_blocks(device)
        transformer.prepare_block_swap_before_forward()
    else:
        logger.info(f"Moving and casting model to {device} and {dit_weight_dtype}")
        transformer.to(device=device, dtype=dit_weight_dtype)
    
    if args.img_in_txt_in_offloading:
        logger.info("Enable offloading img_in and txt_in to CPU")
        transformer.enable_img_in_txt_in_offloading()
    
    # Load scheduler
    logger.info(f"Loading scheduler")
    scheduler = FlowMatchDiscreteScheduler(shift=args.flow_shift, reverse=True, solver="euler")
    
    # Prepare timesteps
    num_inference_steps = args.infer_steps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    # Create video paths list to return
    video_paths = []
    
    # Process each batch
    for batch_idx in range(batch_size):
        if stop_event.is_set():
            break
            
        # Set seed for this batch
        if seed is None or seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        else:
            current_seed = seed + batch_idx
            
        generator = torch.Generator(device).manual_seed(current_seed)
        
        # Prepare input dimensions
        vae_ver = "xxx"  # This doesn't matter for latent_video_length calculation with I2V mode
        vae_scale_factor = 2 ** (4 - 1)
        
        # Use processed image dimensions
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        
        # For I2V mode, we determine the video length based on args
        latent_video_length = args.video_length
        
        # Generate initial noise
        num_channels_latents = 16
        shape_of_frame = (1, num_channels_latents, 1, latent_height, latent_width)
        
        # Create noise for each frame
        latents = []
        for i in range(latent_video_length):
            latents.append(randn_tensor(shape_of_frame, generator=generator, device=device, dtype=dit_dtype))
        latents = torch.cat(latents, dim=2)
        
        # Prepare image latents for I2V mode
        # Repeat the image latents for batch size if needed
        batch_size_local = latents.shape[0]
        img_latents = img_latents.repeat(batch_size_local, 1, 1, 1, 1)
        
        # Expand temporal dimension to match video length
        if img_latents.shape[2] < latent_video_length:
            img_latents = img_latents.repeat(1, 1, latent_video_length, 1, 1)
            # Zero out image content for frames after the first one
            img_latents[:, :, 1:, :, :] = 0
        
        # Create timestep channel (all ones)
        t_channel = torch.ones(
            (batch_size_local, 1, latent_video_length, latent_height, latent_width),
            device=device, dtype=dit_dtype
        )
        
        # Combine image latents, noise latents, and timestep channel
        combined_latents = torch.cat([img_latents, latents, t_channel], dim=1)
        latents = combined_latents
        
        # Prepare guidance scale
        embedded_guidance_scale = args.embedded_cfg_scale
        if embedded_guidance_scale is not None:
            guidance_expand = torch.tensor([embedded_guidance_scale * 1000.0] * latents.shape[0], 
                                          dtype=torch.float32, device="cpu")
            guidance_expand = guidance_expand.to(device=device, dtype=dit_dtype)
            if do_classifier_free_guidance:
                guidance_expand = torch.cat([guidance_expand, guidance_expand], dim=0)
        else:
            guidance_expand = None
            
        # Prepare rotary positional embeddings
        freqs_cos, freqs_sin = get_rotary_pos_embed(vae_ver, transformer, args.video_length, height, width)
        
        # Move and cast all inputs to device and dtype
        prompt_embeds = prompt_embeds.to(device=device, dtype=dit_dtype)
        prompt_mask = prompt_mask.to(device=device)
        prompt_embeds_2 = prompt_embeds_2.to(device=device, dtype=dit_dtype)
        prompt_mask_2 = prompt_mask_2.to(device=device)
        freqs_cos = freqs_cos.to(device=device, dtype=dit_dtype)
        freqs_sin = freqs_sin.to(device=device, dtype=dit_dtype)
        
        # Ensure split_uncond is enabled if split_attn is enabled
        if args.split_attn and do_classifier_free_guidance and not args.split_uncond:
            logger.warning("split_attn is enabled, split_uncond will be enabled as well.")
            args.split_uncond = True
            
        # Set up progress bar
        progress_bar = tqdm(total=num_inference_steps, desc=f"Generating batch {batch_idx+1}/{batch_size}")
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Check for stop event
            if stop_event.is_set():
                break
                
            # Scale model input
            latents = scheduler.scale_model_input(latents, t)
            
            # Predict noise residual
            with torch.no_grad(), accelerator.autocast():
                latents_input = latents if not do_classifier_free_guidance else torch.cat([latents, latents], dim=0)
                batch_size_step = 1 if args.split_uncond else latents_input.shape[0]
                
                noise_pred_list = []
                for j in range(0, latents_input.shape[0], batch_size_step):
                    noise_pred = transformer(
                        latents_input[j : j + batch_size_step],
                        t.repeat(batch_size_step).to(device=device, dtype=dit_dtype),
                        text_states=prompt_embeds[j : j + batch_size_step],
                        text_mask=prompt_mask[j : j + batch_size_step],
                        text_states_2=prompt_embeds_2[j : j + batch_size_step],
                        freqs_cos=freqs_cos,
                        freqs_sin=freqs_sin,
                        guidance=guidance_expand[j : j + batch_size_step] if guidance_expand is not None else None,
                        return_dict=True,
                    )["x"]
                    noise_pred_list.append(noise_pred)
                noise_pred = torch.cat(noise_pred_list, dim=0)
                
                # Perform classifier free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                # Compute previous noisy sample
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
            # Update progress
            if callback:
                callback(i, num_inference_steps, batch_idx, batch_size)
            progress_bar.update(1)
        
        # Clean up progress bar
        progress_bar.close()
        
        # Detach latents
        latents = latents.detach().cpu()
        
        # Clean up memory
        clean_memory_on_device(device)
        
        # Decode latents
        videos = decode_latents(args, latents, device, vae)
        
        # Save video
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
        video_path = f"{args.save_path}/{time_flag}_{batch_idx}_{current_seed}.mp4"
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        save_videos_grid(videos, video_path, fps=args.fps)
        logger.info(f"Video saved to: {video_path}")
        
        # Add metadata to video
        if not args.no_metadata:
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt if do_classifier_free_guidance else None,
                "seed": current_seed,
                "height": height,
                "width": width,
                "steps": num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "embedded_cfg_scale": args.embedded_cfg_scale,
                "i2v_mode": True,
                "i2v_resolution": args.i2v_resolution,
                "flow_shift": args.flow_shift,
                "input_image": image_path,
            }
            add_metadata_to_video(video_path, metadata)
        
        # Add to result list
        video_paths.append(video_path)
    
    # Clean up transformer
    transformer = None
    clean_memory_on_device(device)
    
    return video_paths

def batch_process_folder(
    args,
    device,
    accelerator,
    prompt,
    negative_prompt,
    input_folder,
    batch_size=1,
    seed=None,
    callback=None
):
    """Process multiple images from a folder"""
    # Reset stop event
    stop_event.clear()
    
    # Result videos
    video_paths = []
    
    # Process each image
    for i in range(batch_size):
        # Check for stop event
        if stop_event.is_set():
            break
            
        # Get random image
        random_image, status = get_random_image_from_folder(input_folder)
        if random_image is None:
            logger.error(f"Error getting random image: {status}")
            continue
            
        # Resize image if needed
        resized_image, size_info = resize_image_keeping_aspect_ratio(random_image, args.max_width, args.max_height)
        if resized_image is None:
            logger.error(f"Error resizing image: {size_info}")
            continue
            
        # Log processing
        logger.info(f"Processing image {i+1}/{batch_size}: {os.path.basename(random_image)}")
        
        # Calculate seed for this batch
        if seed is None or seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        else:
            current_seed = seed + i
            
        # Set dimensions from resized image
        if isinstance(size_info, tuple):
            width, height = size_info
            
        # Generate video
        batch_video_paths = generate_video(
            args=args,
            device=device,
            accelerator=accelerator,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_path=resized_image,
            batch_size=1,
            seed=current_seed,
            height=height,
            width=width,
            callback=callback
        )
        
        # Add to result
        video_paths.extend(batch_video_paths)
        
        # Clean up temporary file
        try:
            if os.path.exists(resized_image):
                os.remove(resized_image)
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")
            
        # Clear CUDA cache
        clean_memory_on_device(device)
        
    return video_paths

def get_args():
    parser = argparse.ArgumentParser(description="Hunyuan I2V Inference Script")
    
    # Input/output params
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt for generation")
    parser.add_argument("--image_path", type=str, default=None, help="Path to input image")
    parser.add_argument("--input_folder", type=str, default=None, help="Folder with input images for batch processing")
    parser.add_argument("--save_path", type=str, default="outputs", help="Path to save generated videos")
    
    # Model paths
    parser.add_argument("--dit", type=str, default="hunyuan/mp_rank_00_model_states.pt", help="DiT model path")
    parser.add_argument("--vae", type=str, default="hunyuan/pytorch_model.pt", help="VAE model path")
    parser.add_argument("--text_encoder1", type=str, default="hunyuan/llava_llama3_fp16.safetensors", help="Text Encoder 1 path")
    parser.add_argument("--text_encoder2", type=str, default="hunyuan/clip_l.safetensors", help="Text Encoder 2 path")
    
    # I2V specific params
    parser.add_argument("--i2v_resolution", type=str, default="720p", choices=["360p", "540p", "720p"], help="Resolution for I2V")
    parser.add_argument("--prompt_template_video", type=str, default="dit-llm-encode-video-i2v", help="Prompt template for video")
    parser.add_argument("--prompt_template", type=str, default="dit-llm-encode-i2v", help="Prompt template for text")
    
    # Generation params
    parser.add_argument("--video_length", type=int, default=25, help="Video length in frames")
    parser.add_argument("--fps", type=int, default=24, help="Video FPS")
    parser.add_argument("--infer_steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Classifier free guidance scale")
    parser.add_argument("--embedded_cfg_scale", type=float, default=1.0, help="Embedded classifier free guidance scale")
    parser.add_argument("--flow_shift", type=float, default=11.0, help="Flow shift parameter")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (-1 for random)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (number of videos to generate)")
    parser.add_argument("--max_width", type=int, default=1024, help="Maximum width for resizing images")
    parser.add_argument("--max_height", type=int, default=1024, help="Maximum height for resizing images")
    
    # Technical params
    parser.add_argument("--fp8", action="store_true", help="Use FP8 for DiT model")
    parser.add_argument("--fp8_llm", action="store_true", help="Use FP8 for LLM text encoder")
    parser.add_argument("--no_metadata", action="store_true", help="Don't add metadata to videos")
    parser.add_argument("--device", type=str, default=None, help="Device for inference (default: cuda if available)")
    parser.add_argument("--attn_mode", type=str, default="sdpa", choices=["sdpa", "flash", "torch", "xformers"], help="Attention mode")
    parser.add_argument("--split_attn", action="store_true", help="Use split attention (batch size 1)")
    parser.add_argument("--split_uncond", action="store_true", help="Split unconditional computation")
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="Number of blocks to swap to save memory")
    parser.add_argument("--img_in_txt_in_offloading", action="store_true", help="Offload img_in and txt_in to CPU")
    parser.add_argument("--vae_chunk_size", type=int, default=32, help="Chunk size for VAE")
    parser.add_argument("--vae_spatial_tile_sample_min_size", type=int, default=128, help="VAE tile sample min size")
    
    return parser.parse_args()

def create_progress_callback(progress_callback_fn=None):
    """Create a callback function to track progress"""
    def callback(current_step, total_steps, batch_idx, batch_size):
        progress_percent = int((current_step / total_steps) * a00)
        if progress_callback_fn:
            progress_callback_fn(progress_percent, current_step, total_steps, batch_idx, batch_size)
        else:
            sys.stdout.write(f"\rProgress: {progress_percent}% | Batch: {batch_idx+1}/{batch_size} | Step: {current_step}/{total_steps}")
            sys.stdout.flush()
    return callback

def main():
    # Parse arguments
    args = get_args()
    
    # Setup device
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Print configuration
    logger.info(f"Running Hunyuan I2V with device: {device}")
    logger.info(f"Model: {args.dit}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Guidance scale: {args.guidance_scale}")
    logger.info(f"Embedded CFG scale: {args.embedded_cfg_scale}")
    
    # Make sure save path exists
    os.makedirs(args.save_path, exist_ok=True)
    
    # Create accelerator
    mixed_precision = "bf16" 
    accelerator = accelerate.Accelerator(mixed_precision=mixed_precision)
    
    # Create progress callback
    callback = create_progress_callback()
    
    # Check if we're doing batch processing from folder
    if args.input_folder is not None:
        logger.info(f"Processing images from folder: {args.input_folder}")
        video_paths = batch_process_folder(
            args=args,
            device=device,
            accelerator=accelerator,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            input_folder=args.input_folder,
            batch_size=args.batch_size,
            seed=args.seed,
            callback=callback
        )
    else:
        # Single image or batch of same image
        if args.image_path is None:
            logger.error("Either --image_path or --input_folder must be specified")
            return
            
        logger.info(f"Processing image: {args.image_path}")
        video_paths = generate_video(
            args=args,
            device=device,
            accelerator=accelerator,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image_path=args.image_path,
            batch_size=args.batch_size,
            seed=args.seed,
            callback=callback
        )
    
    # Print results
    logger.info(f"Generated {len(video_paths)} videos:")
    for path in video_paths:
        logger.info(f"  - {path}")

if __name__ == "__main__":
    main()