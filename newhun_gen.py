import argparse
import os
import time
import random
from datetime import datetime
from pathlib import Path
import logging
import torch
import torchvision
import numpy as np
import math
from einops import rearrange
import av
from tqdm import tqdm

from PIL import Image
from safetensors.torch import save_file, load_file
from safetensors import safe_open

# Import necessary modules from hyvideo
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.constants import PROMPT_TEMPLATE, NEGATIVE_PROMPT, NEGATIVE_PROMPT_I2V
from hyvideo.utils.file_utils import save_videos_grid

# Try to import optional modules that might be present in the original setup
try:
    from utils.model_utils import str_to_dtype
    from utils.safetensors_utils import mem_eff_save_file
    from networks import lora
    try:
        from lycoris.kohya import create_network_from_weights
    except ImportError:
        pass
except ImportError:
    # Define fallback function
    def str_to_dtype(dtype_str):
        if dtype_str == "fp32":
            return torch.float32
        elif dtype_str == "fp16":
            return torch.float16
        elif dtype_str == "bf16":
            return torch.bfloat16
        else:
            raise ValueError(f"Unknown dtype: {dtype_str}")
    
    mem_eff_save_file = save_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanVideo inference script with official API")

    # Model paths
    parser.add_argument("--model_base", type=str, required=True, help="Base model directory containing T2V models")
    parser.add_argument("--dit", type=str, help="DiT checkpoint path (will override model_base if provided)")
    parser.add_argument("--vae", type=str, help="VAE checkpoint path (will override default if provided)")
    parser.add_argument("--text_encoder1", type=str, help="Text Encoder 1 path (will override default if provided)")
    parser.add_argument("--text_encoder2", type=str, help="Text Encoder 2 path (will override default if provided)")
    parser.add_argument("--dit_weight", type=str, default=None, help="DiT weight path (alternative to dit)")
    parser.add_argument("--i2v_dit_weight", type=str, default=None, help="DiT weight path for I2V mode")

    # Generation settings
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation")
    parser.add_argument("--neg_prompt", type=str, default=None, help="Negative prompt for generation")
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256], help="Video size (height, width)")
    parser.add_argument("--video_length", type=int, default=129, help="Video length in frames")
    parser.add_argument("--infer_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save generated video")
    parser.add_argument("--save_path_suffix", type=str, default="", help="Optional suffix for save path")
    parser.add_argument("--seed", type=int, default=None, help="Seed for generation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--num_videos", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--fps", type=int, default=24, help="Video FPS for saving")

    # CFG settings
    parser.add_argument("--cfg_scale", type=float, default=6.0, help="Classifier-free guidance scale")
    parser.add_argument("--embedded_cfg_scale", type=float, default=None, help="Embedded CFG scale (for CFG distilled models)")
    parser.add_argument("--flow_shift", type=float, default=7.0, help="Flow shift factor for flow matching schedulers")

    # Image to Video mode
    parser.add_argument("--i2v_mode", action="store_true", help="Enable image to video mode")
    parser.add_argument("--i2v_image_path", type=str, default=None, help="Path to input image for I2V mode")
    parser.add_argument("--i2v_resolution", type=str, default="720p", choices=["360p", "540p", "720p"], 
                       help="Resolution for I2V inference")

    # Video to Video mode
    parser.add_argument("--video_path", type=str, default=None, help="Path to input video for V2V mode")
    parser.add_argument("--strength", type=float, default=0.8, help="Strength for video-to-video inference")
    parser.add_argument("--image_path", type=str, default=None, 
                        help="Path to image for SkyReels-style Image2Video (not Hunyuan I2V)")
    
    # Precision settings
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], 
                       help="Precision for inference")
    parser.add_argument("--vae_precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"], 
                       help="Precision for VAE")
    parser.add_argument("--vae_dtype", type=str, default=None, help="Alternative way to specify VAE precision")
    parser.add_argument("--text_encoder_precision", type=str, default="fp16", 
                       choices=["fp32", "fp16", "bf16"], help="Precision for text encoder")
    parser.add_argument("--text_encoder_precision_2", type=str, default="fp16", 
                       choices=["fp32", "fp16", "bf16"], help="Precision for text encoder 2")

    # Advanced settings - Text Encoder & Tokenizer
    parser.add_argument("--text_encoder", type=str, default=None, help="Text encoder type")
    parser.add_argument("--text_encoder_2", type=str, default=None, help="Second text encoder type")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer type")
    parser.add_argument("--tokenizer_2", type=str, default=None, help="Tokenizer type for second text encoder")
    parser.add_argument("--prompt_template", type=str, default=None, help="Prompt template")
    parser.add_argument("--prompt_template_video", type=str, default=None, help="Prompt template for video")
    parser.add_argument("--hidden_state_skip_layer", type=int, default=2, help="Skip layers in hidden state")
    parser.add_argument("--apply_final_norm", action="store_true", help="Apply final normalization to text encoder output")
    parser.add_argument("--reproduce", action="store_true", help="Enable reproduction mode")
    parser.add_argument("--text_len", type=int, default=256, help="Maximum text length")
    parser.add_argument("--text_len_2", type=int, default=77, help="Maximum text length for second encoder")
    
    # Model and VAE config
    parser.add_argument("--latent_channels", type=int, default=16, help="Number of latent channels")
    parser.add_argument("--vae_tiling", action="store_true", help="Enable VAE tiling")
    parser.add_argument("--use_fp8", action="store_true", help="Use FP8 precision")
    parser.add_argument("--fp8", action="store_true", help="Alternative flag for FP8 precision")
    parser.add_argument("--fp8_llm", action="store_true", help="Use FP8 for Text Encoder 1 (LLM)")
    parser.add_argument("--model_resolution", type=str, default="normal", help="Model resolution")
    parser.add_argument("--load_key", type=str, default="ema", help="Key to load in state dict")
    parser.add_argument("--use_cpu_offload", action="store_true", help="Use CPU offloading")
    parser.add_argument("--denoise_type", type=str, default="flow", choices=["flow"], help="Denoising algorithm")
    parser.add_argument("--flow_reverse", action="store_true", help="Reverse flow matching direction")
    parser.add_argument("--flow_solver", type=str, default="euler", help="ODE solver for flow matching")
    
    # LoRA support
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for inference")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA scale factor")
    parser.add_argument("--lora_weight", type=str, nargs="*", default=None, help="Alternative LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument("--exclude_single_blocks", action="store_true", help="Exclude single blocks when loading LoRA weights")
    parser.add_argument("--save_merged_model", type=str, default=None, help="Save merged model to path, no inference performed")
    parser.add_argument("--lycoris", action="store_true", help="Use lycoris for inference")
    
    # Memory optimization settings
    parser.add_argument("--blocks_to_swap", type=int, default=None, help="Number of blocks to swap to CPU for memory optimization")
    parser.add_argument("--img_in_txt_in_offloading", action="store_true", help="Offload img_in and txt_in to CPU")
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="Chunk size for CausalConv3d in VAE")
    parser.add_argument("--vae_spatial_tile_sample_min_size", type=int, default=None, 
                        help="Spatial tile sample min size for VAE, default 256")
    parser.add_argument("--attn_mode", type=str, default="torch", 
                        choices=["flash", "torch", "sageattn", "xformers", "sdpa"], help="Attention mode")
    parser.add_argument("--split_attn", action="store_true", help="Use split attention, default is False")
    parser.add_argument("--split_uncond", action="store_true", 
                        help="Split unconditional call for classifier free guidance (slower but less memory usage)")
    
    # Distributed training args
    parser.add_argument("--ulysses_degree", type=int, default=1, help="Ulysses attention degree")
    parser.add_argument("--ring_degree", type=int, default=1, help="Ring attention degree")
    
    # Device selection
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use for inference. If None, use CUDA if available, otherwise CPU")
    
    # Output format
    parser.add_argument("--output_type", type=str, default="video", 
                       choices=["video", "images", "latent", "both"], help="Output type")
    parser.add_argument("--no_metadata", action="store_true", help="Do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, 
                        help="Path to latent for decode, no inference performed")

    args = parser.parse_args()
    
    # I2V mode requires an image path
    if args.i2v_mode and args.i2v_image_path is None:
        parser.error("--i2v_image_path is required when --i2v_mode is enabled")
    
    # Combine equivalent flags
    if args.fp8:
        args.use_fp8 = True
    
    # VAE dtype consistency
    if args.vae_dtype is not None and args.vae_precision is not None:
        logger.warning(f"Both vae_dtype ({args.vae_dtype}) and vae_precision ({args.vae_precision}) specified; "
                      f"using vae_dtype")
    
    # If latent_path is provided, check output_type compatibility
    if args.latent_path is not None and len(args.latent_path) > 0:
        if args.output_type not in ["video", "images"]:
            parser.error("latent_path is only supported for 'video' or 'images' output")
    
    return args

def clean_memory_on_device(device):
    """Helper function to clean GPU memory"""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "cpu":
        pass
    elif device.type == "mps":  # for Mac M1/M2
        torch.mps.empty_cache()

def synchronize_device(device):
    """Helper function to synchronize device operations"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def setup_block_swapping(transformer, num_blocks, device, dtype):
    """
    Set up block swapping on the transformer model for memory optimization.
    This implementation matches the approach in the original hv_generate_video.py
    """
    if num_blocks <= 0:
        logger.info("Block swapping requested with zero or negative blocks, skipping")
        return transformer
    
    logger.info(f"Setting up block swapping with {num_blocks} blocks on device {device}")
    
    # Cast model to appropriate precision
    transformer.to(dtype=dtype)
    
    # Enable block swapping with the specified number of blocks
    transformer.enable_block_swap(num_blocks, device, supports_backward=False)
    
    # Move model to device but keep blocks on CPU to reduce GPU memory usage
    transformer.move_to_device_except_swap_blocks(device)
    
    # Prepare blocks before forward pass
    transformer.prepare_block_swap_before_forward()
    
    # Switch to inference-only mode for block swapping
    if hasattr(transformer, 'switch_block_swap_for_inference'):
        transformer.switch_block_swap_for_inference()
    
    logger.info(f"Block swapping configured successfully with {num_blocks} blocks")
    
    return transformer

def save_videos_grid_local(videos, path, rescale=False, n_rows=1, fps=24):
    """Save videos as a grid, with customizable options"""
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

def save_images_grid(videos, parent_dir, image_name, rescale=False, n_rows=1, create_subdir=True):
    """Save individual frames from videos as images"""
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

    if create_subdir:
        output_dir = os.path.join(parent_dir, image_name)
    else:
        output_dir = parent_dir

    os.makedirs(output_dir, exist_ok=True)
    for i, x in enumerate(outputs):
        image_path = os.path.join(output_dir, f"{image_name}_{i:03d}.png")
        image = Image.fromarray(x)
        image.save(image_path)

def save_latent(latent, save_path, time_flag, seed, args):
    """Save latent to file with metadata"""
    height, width = args.video_size
    video_length = args.video_length
    
    latent_path = f"{save_path}/{time_flag}_{seed}_latent.safetensors"
    
    if args.no_metadata:
        metadata = None
    else:
        metadata = {
            "seeds": f"{seed}",
            "prompt": f"{args.prompt}",
            "height": f"{height}",
            "width": f"{width}",
            "video_length": f"{video_length}",
            "infer_steps": f"{args.infer_steps}",
            "guidance_scale": f"{args.cfg_scale}",
        }
        
        if args.embedded_cfg_scale is not None:
            metadata["embedded_cfg_scale"] = f"{args.embedded_cfg_scale}"
        if args.neg_prompt is not None:
            metadata["negative_prompt"] = f"{args.neg_prompt}"
        if args.i2v_mode:
            metadata["i2v_mode"] = "True"
            metadata["i2v_resolution"] = args.i2v_resolution
    
    # Use memory-efficient save if available
    if 'mem_eff_save_file' in globals():
        mem_eff_save_file({"latent": latent}, latent_path, metadata=metadata)
    else:
        save_file({"latent": latent}, latent_path, metadata=metadata)
    
    logger.info(f"Latent saved to: {latent_path}")
    return latent_path

def load_latents_from_path(latent_paths):
    """Load latents from saved files"""
    original_base_names = []
    latents_list = []
    seeds = []
    
    for latent_path in latent_paths:
        original_base_names.append(os.path.splitext(os.path.basename(latent_path))[0])
        seed = 0

        # Load based on file format
        if os.path.splitext(latent_path)[1] != ".safetensors":
            latents = torch.load(latent_path, map_location="cpu")
        else:
            latents = load_file(latent_path)["latent"]
            # Extract metadata if available
            with safe_open(latent_path, framework="pt") as f:
                metadata = f.metadata()
            logger.info(f"Loaded metadata: {metadata}")

            if metadata and "seeds" in metadata:
                seed = int(metadata["seeds"])

        seeds.append(seed)
        latents_list.append(latents)
        logger.info(f"Loaded latent from {latent_path}. Shape: {latents.shape}")
    
    latents = torch.stack(latents_list, dim=0)
    return latents, seeds, original_base_names

def apply_lora_weights(args, transformer, device):
    """Apply LoRA weights to the transformer model"""
    # Support both lora_weight and lora_path formats
    weights_to_load = []
    
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        weights_to_load.extend(args.lora_weight)
    
    if args.lora_path is not None:
        weights_to_load.append(args.lora_path)
    
    if not weights_to_load:
        logger.info("No LoRA weights to load")
        return transformer
    
    for i, lora_weight in enumerate(weights_to_load):
        # Determine multiplier
        if args.lora_multiplier is not None and isinstance(args.lora_multiplier, list) and len(args.lora_multiplier) > i:
            lora_multiplier = args.lora_multiplier[i]
        else:
            lora_multiplier = args.lora_scale if args.lora_path else 1.0
        
        logger.info(f"Loading LoRA weights from {lora_weight} with multiplier {lora_multiplier}")
        weights_sd = load_file(lora_weight)
        
        # Filter to exclude keys that are part of single_blocks if requested
        if args.exclude_single_blocks:
            filtered_weights = {k: v for k, v in weights_sd.items() if "single_blocks" not in k}
            weights_sd = filtered_weights
        
        # Use appropriate LoRA library
        if args.lycoris and 'create_network_from_weights' in globals():
            lycoris_net, _ = create_network_from_weights(
                multiplier=lora_multiplier,
                file=None,
                weights_sd=weights_sd,
                unet=transformer,
                text_encoder=None,
                vae=None,
                for_inference=True,
            )
            lycoris_net.merge_to(None, transformer, weights_sd, dtype=None, device=device)
        elif 'lora' in globals():
            network = lora.create_network_from_weights_hunyuan_video(
                lora_multiplier, weights_sd, unet=transformer, for_inference=True
            )
            network.merge_to(None, transformer, weights_sd, device=device, non_blocking=True)
        else:
            logger.warning("LoRA libraries not found, cannot apply LoRA weights")
            continue
        
        synchronize_device(device)
        logger.info(f"LoRA weights from {lora_weight} applied successfully")
    
    return transformer

def main():
    args = parse_args()
    
    # Set model paths if explicitly provided
    models_root_path = Path(args.model_base)
    
    if args.dit:
        args.dit_weight = args.dit
    if args.text_encoder1:
        args.text_encoder = args.text_encoder1
    if args.text_encoder2:
        args.text_encoder_2 = args.text_encoder2
    
    # Create save folder
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    os.makedirs(save_path, exist_ok=True)
    
    # Determine device and precision settings from original hv_generate_video.py
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    dit_dtype = str_to_dtype(args.precision) if hasattr(str_to_dtype, '__call__') else torch.bfloat16
    dit_weight_dtype = torch.float8_e4m3fn if args.use_fp8 else dit_dtype
    logger.info(f"Using device: {device}, precision: {dit_dtype}, weight precision: {dit_weight_dtype}")
    
    # Handle direct latent decoding if latent_path is provided
    if args.latent_path is not None and len(args.latent_path) > 0:
        logger.info("Latent path provided. Loading latents for direct decoding...")
        latents, seeds, original_base_names = load_latents_from_path(args.latent_path)
        
        # Create and load sampler for decoding
        hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
        
        # Decode latents to samples (will implement similar to the VAE decoding in hv_generate_video.py)
        logger.info("Decoding latents to visual samples...")
        latents = latents.to(device)
        
        # TODO: Implement proper decoding with VAE from hunyuan_video_sampler
        # For now we'll use a placeholder approach
        samples = hunyuan_video_sampler.pipeline.decode_latents(latents)
        
        # Save the decoded results
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        
        if args.output_type == "video":
            # Save as video
            for i, sample in enumerate(samples):
                sample = sample.unsqueeze(0)
                original_name = "" if original_base_names is None else f"_{original_base_names[i]}"
                video_path = f"{save_path}/{time_flag}_{i}_{seeds[i]}{original_name}.mp4"
                save_videos_grid_local(sample, video_path, fps=args.fps)
                logger.info(f"Sample video saved to: {video_path}")
                
        elif args.output_type == "images":
            # Save as image frames
            for i, sample in enumerate(samples):
                sample = sample.unsqueeze(0)
                original_name = "" if original_base_names is None else f"_{original_base_names[i]}"
                image_name = f"{time_flag}_{i}_{seeds[i]}{original_name}"
                save_images_grid(sample, save_path, image_name)
                logger.info(f"Sample images saved to: {save_path}/{image_name}")
                
        logger.info("Done processing latents!")
        return
    
    # Load models using the official HunyuanVideoSampler
    logger.info("Loading Hunyuan Video Sampler...")
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args
    
    # Apply block swapping if enabled - using approach from original hv_generate_video.py
    if args.blocks_to_swap and args.blocks_to_swap > 0:
        if hasattr(hunyuan_video_sampler.pipeline, 'transformer'):
            transformer = hunyuan_video_sampler.pipeline.transformer
            
            # Apply block swapping using the technique from original script
            transformer = setup_block_swapping(transformer, args.blocks_to_swap, device, dit_weight_dtype)
            
            # Apply image and text embedding offloading if enabled
            if args.img_in_txt_in_offloading:
                if hasattr(transformer, 'enable_img_in_txt_in_offloading'):
                    transformer.enable_img_in_txt_in_offloading()
                    logger.info("Enabled img_in and txt_in offloading")
            
            # Update the transformer in the pipeline
            hunyuan_video_sampler.pipeline.transformer = transformer
            
            # Clean GPU memory
            clean_memory_on_device(device)
    
    # Apply LoRA weights if enabled
    if (args.use_lora and args.lora_path) or (args.lora_weight and len(args.lora_weight) > 0):
        if hasattr(hunyuan_video_sampler.pipeline, 'transformer'):
            transformer = hunyuan_video_sampler.pipeline.transformer
            transformer = apply_lora_weights(args, transformer, device)
            
            # Save merged model if requested
            if args.save_merged_model:
                logger.info(f"Saving merged model to {args.save_merged_model}")
                if 'mem_eff_save_file' in globals():
                    mem_eff_save_file(transformer.state_dict(), args.save_merged_model)
                else:
                    save_file(transformer.state_dict(), args.save_merged_model)
                logger.info("Merged model saved")
                return
            
            # Update transformer in pipeline
            hunyuan_video_sampler.pipeline.transformer = transformer
    
    # Set output specific parameters
    if args.embedded_cfg_scale is None:
        embedded_guidance_scale = 6.0  # Default value if not specified
    else:
        embedded_guidance_scale = args.embedded_cfg_scale
    
    # Generate video
    logger.info(f"Generating video with prompt: {args.prompt}")
    
    # Configure VAE options if needed
    if args.vae_chunk_size is not None and hasattr(hunyuan_video_sampler.vae, 'set_chunk_size_for_causal_conv_3d'):
        hunyuan_video_sampler.vae.set_chunk_size_for_causal_conv_3d(args.vae_chunk_size)
        logger.info(f"Set VAE chunk size to {args.vae_chunk_size}")
    
    if args.vae_spatial_tile_sample_min_size is not None and hasattr(hunyuan_video_sampler.vae, 'enable_spatial_tiling'):
        hunyuan_video_sampler.vae.enable_spatial_tiling(True)
        hunyuan_video_sampler.vae.tile_sample_min_size = args.vae_spatial_tile_sample_min_size
        hunyuan_video_sampler.vae.tile_latent_min_size = args.vae_spatial_tile_sample_min_size // 8
        logger.info(f"Enabled VAE spatial tiling with sample min size {args.vae_spatial_tile_sample_min_size}")
    
    # Run generation
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=embedded_guidance_scale,
        i2v_mode=args.i2v_mode,
        i2v_resolution=args.i2v_resolution,
        i2v_image_path=args.i2v_image_path,
    )
    
    samples = outputs['samples']
    seeds = outputs['seeds']
    prompts = outputs['prompts']
    
    # Extract latents if possible (for saving)
    latents = None
    if 'latents' in outputs:
        latents = outputs['latents']
    
    # Clean up GPU memory before saving
    if device.type == "cuda":
        clean_memory_on_device(device)
    
    # Save outputs
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
    
    # Process each generated video
    for i, sample in enumerate(samples):
        sample_expanded = sample.unsqueeze(0)
        output_base_name = f"{time_flag}_seed{seeds[i]}_{prompts[i][:100].replace('/','').replace(' ','_')}"
        
        # Handle different output types
        if args.output_type in ["video", "both"]:
            video_path = f"{save_path}/{output_base_name}.mp4"
            
            # Use the appropriate video saving function
            if hasattr(save_videos_grid, '__call__'):
                save_videos_grid(sample_expanded, video_path, fps=args.fps)
            else:
                save_videos_grid_local(sample_expanded, video_path, fps=args.fps)
                
            logger.info(f"Video saved to: {video_path}")
        
        if args.output_type in ["images", "both"]:
            image_dir = f"{save_path}/{output_base_name}_frames"
            os.makedirs(image_dir, exist_ok=True)
            
            # Convert tensor to PIL images and save
            frames = sample.permute(1, 2, 3, 0).cpu().numpy()  # T, H, W, C
            for frame_idx, frame in enumerate(frames):
                frame = (frame * 255).astype('uint8')
                img = Image.fromarray(frame)
                img.save(f"{image_dir}/frame_{frame_idx:04d}.png")
            
            logger.info(f"Frame images saved to: {image_dir}")
        
        if args.output_type in ["latent", "both"]:
            if latents is not None and i < len(latents):
                save_latent(latents[i], save_path, time_flag, seeds[i], args)
            else:
                logger.warning("Latents not available for saving. This requires modification of HunyuanVideoSampler.")
    
    # Final memory cleanup
    if device.type == "cuda":
        clean_memory_on_device(device)
        
    logger.info("Generation completed successfully!")

if __name__ == "__main__":
    main()