"""
WAN Video Upscaling Helper
Implements video upscaling pipeline with tiled VAE operations
for efficient VRAM usage.
"""

import argparse
import gc
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import accelerate
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from tqdm import tqdm

# Import from existing codebase
from Wan2_2.wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel, load_wan_model, detect_wan_sd_dtype
from wan.modules.vae import WanVAE
from Wan2_2.wan.modules.vae2_2 import Wan2_2_VAE
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from utils.model_utils import str_to_dtype
from utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class WanUpscaler:
    """Main upscaling class for WAN video upscaling"""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the upscaler with command line arguments
        
        Args:
            args: Command line arguments including upscale parameters
        """
        self.args = args
        self.device = self._setup_device()
        self.cfg = WAN_CONFIGS["ti2v-5B"]  # Always use 5B model for upscaling
        
        # Set default tiling parameters if not provided
        if not hasattr(args, 'upscale_tile_width'):
            args.upscale_tile_width = 272
        if not hasattr(args, 'upscale_tile_height'):
            args.upscale_tile_height = 272
        if not hasattr(args, 'upscale_tile_stride_x'):
            args.upscale_tile_stride_x = 144
        if not hasattr(args, 'upscale_tile_stride_y'):
            args.upscale_tile_stride_y = 128
        if not hasattr(args, 'upscale_resize_mode'):
            args.upscale_resize_mode = 'lanczos'
            
        logger.info(f"WanUpscaler initialized with device: {self.device}")
        logger.info(f"Tiling config: {args.upscale_tile_width}x{args.upscale_tile_height}, "
                   f"stride: {args.upscale_tile_stride_x}x{args.upscale_tile_stride_y}")
    
    def _setup_device(self) -> torch.device:
        """Setup and return the device to use"""
        device_str = self.args.device if self.args.device else "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device_str)
    
    def load_input(self) -> Tuple[torch.Tensor, int, int, int]:
        """Load input video or latent and return video tensor with dimensions
        
        Returns:
            Tuple of (video_tensor, original_height, original_width, num_frames)
        """
        if self.args.video_path:
            return self._load_video()
        elif self.args.latent_path and len(self.args.latent_path) > 0:
            return self._load_latent_as_video()
        else:
            raise ValueError("Upscale mode requires either --video_path or --latent_path")
    
    def _load_video(self) -> Tuple[torch.Tensor, int, int, int]:
        """Load video from file
        
        Returns:
            Tuple of (video_tensor, original_height, original_width, num_frames)
        """
        logger.info(f"Loading video from {self.args.video_path}")
        
        cap = cv2.VideoCapture(self.args.video_path)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"Could not load any frames from video: {self.args.video_path}")
        
        # Convert to tensor [F, H, W, C] -> [1, C, F, H, W]
        video_np = np.stack(frames, axis=0)
        video_tensor = torch.from_numpy(video_np).float() / 255.0  # Normalize to [0, 1]
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # [F, C, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, F, H, W]
        
        logger.info(f"Loaded {len(frames)} frames at {original_height}x{original_width}, fps: {fps}")
        
        return video_tensor, original_height, original_width, len(frames)
    
    def _load_latent_as_video(self) -> Tuple[Optional[torch.Tensor], int, int, int]:
        """Load latent and decode it to video (for latent-to-latent upscaling)
        
        Returns:
            Tuple of (None, original_height, original_width, num_frames)
            Note: Returns None for video_tensor as we'll work directly with latents
        """
        from safetensors.torch import load_file
        from safetensors import safe_open
        
        loaded_data = load_file(self.args.latent_path[0], device="cpu")
        self.input_latent = loaded_data["latent"]
        
        # Load metadata to get original dimensions
        with safe_open(self.args.latent_path[0], framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
        
        original_height = int(metadata.get("height", 512))
        original_width = int(metadata.get("width", 512))
        num_frames = self.input_latent.shape[2] if len(self.input_latent.shape) == 5 else self.input_latent.shape[1]
        
        logger.info(f"Loaded latent from {self.args.latent_path[0]}, shape: {self.input_latent.shape}")
        logger.info(f"Original dimensions from metadata: {original_height}x{original_width}")
        
        return None, original_height, original_width, num_frames
    
    def resize_video(self, video_tensor: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """Resize video tensor at pixel level using specified interpolation
        
        Args:
            video_tensor: Input video [1, C, F, H, W]
            target_height: Target height in pixels
            target_width: Target width in pixels
            
        Returns:
            Resized video tensor [1, C, F, H, W]
        """
        B, C, num_frames, H, W = video_tensor.shape
        
        if H == target_height and W == target_width:
            return video_tensor
        
        logger.info(f"Resizing video from {H}x{W} to {target_height}x{target_width}")
        
        # Reshape to process all frames at once: [B*F, C, H, W]
        video_reshaped = video_tensor.permute(0, 2, 1, 3, 4).reshape(B * num_frames, C, H, W)
        
        # Determine interpolation mode
        if self.args.upscale_resize_mode == 'lanczos':
            # For lanczos, we need to use PIL or cv2 as PyTorch doesn't support it
            # We'll use bilinear with antialiasing as a good alternative
            mode = 'bilinear'
            antialias = True
        else:
            mode = self.args.upscale_resize_mode
            antialias = False
        
        # Resize using PyTorch
        video_resized = F.interpolate(
            video_reshaped,
            size=(target_height, target_width),
            mode=mode,
            align_corners=False,
            antialias=antialias
        )
        
        # Reshape back: [B, C, F, H, W]
        video_resized = video_resized.reshape(B, num_frames, C, target_height, target_width).permute(0, 2, 1, 3, 4)
        
        return video_resized
    
    def encode_with_tiling(self, video_tensor: torch.Tensor, vae, vae_dtype: torch.dtype) -> torch.Tensor:
        """Encode video to latents using the VAE's built-in tiling system
        
        Args:
            video_tensor: Video tensor [1, C, F, H, W] in range [0, 1]
            vae: VAE model instance
            vae_dtype: Data type for VAE operations
            
        Returns:
            Encoded latent tensor [1, C, F, H, W]
        """
        logger.info(f"Encoding video with tiling, input shape: {video_tensor.shape}")
        
        # Move VAE to device
        vae.to_device(self.device)
        
        # Convert video to VAE expected range [-1, 1] and dtype
        video_tensor = (video_tensor * 2.0 - 1.0).to(device=self.device, dtype=vae_dtype)
        
        # Use tiled encoding for memory efficiency
        try:
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=vae_dtype):
                latents = vae.encode(video_tensor, device=self.device, tiled=True, 
                                   tile_size=(self.args.upscale_tile_width//8, self.args.upscale_tile_height//8), 
                                   tile_stride=(self.args.upscale_tile_stride_x//8, self.args.upscale_tile_stride_y//8))
                if hasattr(vae, 'model'):
                    vae.model.clear_cache()
        except Exception as e:
            logger.warning(f"Tiled encoding failed: {e}, falling back to regular encoding")
            # Fallback to regular encoding
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=vae_dtype):
                if hasattr(vae, 'encode'):
                    # Wan2_2_VAE expects list input
                    latents = vae.encode([video_tensor.squeeze(0)])[0].unsqueeze(0)
                else:
                    latents = vae.encode_video(video_tensor)
        
        # Move VAE back to CPU
        vae.to_device("cpu")
        clean_memory_on_device(self.device)
        
        logger.info(f"Encoded to latent shape: {latents.shape}")
        
        return latents
    
    def decode_with_tiling(self, latent: torch.Tensor, vae, vae_dtype: torch.dtype) -> torch.Tensor:
        """Decode latents to video using the VAE's built-in tiling system
        
        Args:
            latent: Latent tensor [1, C, F, H, W]
            vae: VAE model instance
            vae_dtype: Data type for VAE operations
            
        Returns:
            Decoded video tensor [1, C, F, H, W] in range [0, 1]
        """
        logger.info(f"Decoding latent with tiling, input shape: {latent.shape}")
        
        # Move VAE to device
        vae.to_device(self.device)
        
        latent = latent.to(device=self.device, dtype=vae_dtype)
        
        # Manual tiled decoding since Wan2_2_VAE doesn't support tiled parameter
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=vae_dtype):
            decoded_video = self.manual_tiled_decode(vae, latent)
            if hasattr(vae, 'model'):
                vae.model.clear_cache()
        
        # Move to CPU and convert to [0, 1] range
        decoded_video = decoded_video.cpu().float()
        
        # Clamp and normalize to [0, 1]
        decoded_video.clamp_(-1.0, 1.0)
        decoded_video.add_(1.0).div_(2.0)
        decoded_video.clamp_(0.0, 1.0)
        
        # Move VAE back to CPU
        vae.to_device("cpu")
        clean_memory_on_device(self.device)
        
        logger.info(f"Decoded to video shape: {decoded_video.shape}")
        
        return decoded_video
    
    def manual_tiled_decode(self, vae, latent):
        """
        Manually perform tiled decoding by splitting the latent into tiles.
        Only tiles spatially, not temporally, for proper VAE decoding.
        
        Args:
            vae: The VAE model
            latent: Latent tensor [B, C, F, H, W] where F is temporal frames
            
        Returns:
            Decoded video tensor [B, C, F_out, H*scale, W*scale]
        """
        B, C, F, H, W = latent.shape
        
        # Calculate tile dimensions in latent space (only for spatial dimensions)
        # Wan2.2 VAE uses 16x spatial upsampling
        upsampling_factor = 16
        tile_h = self.args.upscale_tile_height // upsampling_factor
        tile_w = self.args.upscale_tile_width // upsampling_factor
        stride_h = self.args.upscale_tile_stride_y // upsampling_factor
        stride_w = self.args.upscale_tile_stride_x // upsampling_factor
        
        # Ensure minimum tile size
        tile_h = max(tile_h, 8)
        tile_w = max(tile_w, 8)
        stride_h = min(stride_h, tile_h - 2)  # Ensure overlap
        stride_w = min(stride_w, tile_w - 2)
        
        logger.info(f"Tiling with tile size ({tile_h}, {tile_w}), stride ({stride_h}, {stride_w})")
        
        # If the latent is small enough, decode without tiling
        if H <= tile_h and W <= tile_w:
            logger.info("Latent is small enough, decoding without tiling")
            if hasattr(vae, 'decode'):
                # Wan2_2_VAE expects list input
                decoded_list = vae.decode([latent.squeeze(0)])
                if decoded_list and len(decoded_list) > 0:
                    return torch.stack(decoded_list, dim=0)
            return vae.decode_video(latent)
        
        # Calculate temporal output dimension
        # WanVideo VAE temporal upscaling: out_T = T * 4 - 3
        # For Wan2.2, we need to determine the temporal upscale factor
        # From the logs: 11 latent frames -> 41 output frames
        # This is approximately 4x temporal upscaling
        out_T = F * 4 - 3  # Temporal frames after decoding
        
        # Calculate spatial output dimensions
        out_h = H * upsampling_factor
        out_w = W * upsampling_factor
        out_c = 3  # RGB channels
        
        # Initialize output tensor with correct temporal dimension
        output = torch.zeros((B, out_c, out_T, out_h, out_w), device=latent.device, dtype=torch.float32)
        weight_map = torch.zeros((B, 1, out_T, out_h, out_w), device=latent.device, dtype=torch.float32)
        
        # Create blend weights for smooth transitions (in latent space units)
        blend_h = min(4, (tile_h - stride_h) // 2)
        blend_w = min(4, (tile_w - stride_w) // 2)
        
        # Process tiles
        tiles_processed = 0
        for y in range(0, H, stride_h):
            for x in range(0, W, stride_w):
                # Calculate tile boundaries
                y_end = min(y + tile_h, H)
                x_end = min(x + tile_w, W)
                
                # Adjust start if we're at the edge
                if y_end == H and y_end - y < tile_h:
                    y = max(0, H - tile_h)
                    y_end = H
                if x_end == W and x_end - x < tile_w:
                    x = max(0, W - tile_w)
                    x_end = W
                
                # Extract tile (keep all temporal frames, only tile spatially)
                tile_latent = latent[:, :, :, y:y_end, x:x_end]
                
                # Decode tile
                if hasattr(vae, 'decode'):
                    # Wan2_2_VAE expects list input
                    decoded_list = vae.decode([tile_latent.squeeze(0)])
                    if decoded_list and len(decoded_list) > 0:
                        tile_decoded = torch.stack(decoded_list, dim=0)
                    else:
                        raise RuntimeError("VAE decoding failed or returned empty list.")
                else:
                    tile_decoded = vae.decode_video(tile_latent)
                
                # Convert to float32 for blending
                tile_decoded = tile_decoded.float()
                
                # Get actual decoded dimensions
                _, _, decoded_T, decoded_h, decoded_w = tile_decoded.shape
                
                # Calculate output position (only for spatial dimensions)
                out_y = y * upsampling_factor
                out_x = x * upsampling_factor
                out_y_end = min(out_y + decoded_h, out_h)
                out_x_end = min(out_x + decoded_w, out_w)
                
                # Crop tile if it exceeds boundaries
                if out_y_end - out_y < decoded_h:
                    tile_decoded = tile_decoded[:, :, :, :out_y_end - out_y, :]
                if out_x_end - out_x < decoded_w:
                    tile_decoded = tile_decoded[:, :, :, :, :out_x_end - out_x]
                
                # Create weight mask for blending
                tile_h_actual = out_y_end - out_y
                tile_w_actual = out_x_end - out_x
                tile_weight = torch.ones((1, 1, decoded_T, tile_h_actual, tile_w_actual), 
                                        device=latent.device, dtype=torch.float32)
                
                # Apply linear blending at edges (only spatial, not temporal)
                if y > 0 and blend_h > 0:
                    # Blend top edge
                    blend_size = min(blend_h * upsampling_factor, tile_h_actual)
                    for i in range(blend_size):
                        weight = i / float(blend_size)
                        tile_weight[:, :, :, i, :] *= weight
                
                if x > 0 and blend_w > 0:
                    # Blend left edge
                    blend_size = min(blend_w * upsampling_factor, tile_w_actual)
                    for i in range(blend_size):
                        weight = i / float(blend_size)
                        tile_weight[:, :, :, :, i] *= weight
                
                if y + tile_h < H and blend_h > 0:
                    # Blend bottom edge
                    blend_size = min(blend_h * upsampling_factor, tile_h_actual)
                    for i in range(blend_size):
                        weight = 1.0 - (i / float(blend_size))
                        tile_weight[:, :, :, -(i+1), :] *= weight
                
                if x + tile_w < W and blend_w > 0:
                    # Blend right edge
                    blend_size = min(blend_w * upsampling_factor, tile_w_actual)
                    for i in range(blend_size):
                        weight = 1.0 - (i / float(blend_size))
                        tile_weight[:, :, :, :, -(i+1)] *= weight
                
                # Add decoded tile to output with weighting
                output[:, :, :, out_y:out_y_end, out_x:out_x_end] += tile_decoded * tile_weight
                weight_map[:, :, :, out_y:out_y_end, out_x:out_x_end] += tile_weight
                
                tiles_processed += 1
                if tiles_processed % 10 == 0:
                    logger.info(f"Processed {tiles_processed} tiles...")
        
        # Normalize by weight map to handle overlapping regions
        output = output / (weight_map + 1e-8)
        
        logger.info(f"Tiled decoding complete, processed {tiles_processed} tiles")
        return output
    
    def prepare_v2v_sampling(self, latent: torch.Tensor, target_height: int, target_width: int, num_frames: int):
        """Prepare inputs for V2V sampling
        
        Args:
            latent: Input latent tensor [1, C, F, H, W]
            target_height: Target height in pixels
            target_width: Target width in pixels
            num_frames: Number of frames
            
        Returns:
            Dictionary with prepared inputs for sampling
        """
        # Import necessary functions from main pipeline
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Import functions we need
        wan2_module = __import__('wan2_generate_video')
        load_text_encoder = getattr(wan2_module, 'load_text_encoder')
        setup_scheduler = getattr(wan2_module, 'setup_scheduler')
        
        # Import WanModel and load_wan_model directly 
        from wan.modules.model import load_wan_model
        from utils.model_utils import str_to_dtype
        
        # Setup accelerator
        accelerator = accelerate.Accelerator(mixed_precision="bf16")
        
        # Load text encoder
        text_encoder = load_text_encoder(self.args, self.cfg, self.device)
        
        # Prepare text embeddings
        prompt = self.args.prompt if self.args.prompt else "high quality, detailed, sharp, 4k resolution, professional"
        negative_prompt = self.args.negative_prompt if self.args.negative_prompt else "blurry, low quality, pixelated, artifacts, distorted"
        
        # Manual text embedding preparation (simplified)
        with torch.no_grad():
            context = text_encoder([prompt], self.device)
            context_null = text_encoder([negative_prompt], self.device)
        
        # Unload text encoder
        del text_encoder
        clean_memory_on_device(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        
        # Load DiT model directly using load_wan_model like the main pipeline does
        dit_dtype = torch.bfloat16
        dit_weight_dtype = torch.float8_e4m3fn if self.args.fp8 or self.args.fp8_scaled else dit_dtype
        dit_path = self.args.upscale_model
        
        model = load_wan_model(
            self.cfg, self.device, dit_path, self.args.attn_mode, False, 
            self.device, dit_weight_dtype, False
        )
        
        # Convert model to correct dtype and device
        model.to(self.device, dtype=dit_dtype)
        
        # Apply Enhance-A-Video settings if requested
        if hasattr(self.args, 'enhance_weight') and self.args.enhance_weight > 0:
            logger.info(f"Applying Enhance-A-Video with weight {self.args.enhance_weight}")
            self.args.guidance_scale = self.args.guidance_scale * (1 + self.args.enhance_weight * 0.2)
        
        # Setup scheduler
        scheduler, timesteps = setup_scheduler(self.args, self.cfg, self.device)
        
        # Apply V2V strength to timesteps
        seed = self.args.seed if self.args.seed else random.randint(0, 2**32 - 1)
        seed_g = torch.Generator(device=self.device).manual_seed(seed)
        
        # Ensure latent is properly shaped and on device
        latent = latent.to(device=self.device)
        logger.info(f"Input latent shape before noise generation: {latent.shape}")
        
        # Prepare noise with V2V strength
        if self.args.upscale_strength < 1.0:
            init_timestep_idx = int(self.args.upscale_steps * (1.0 - self.args.upscale_strength))
            init_timestep_idx = min(init_timestep_idx, self.args.upscale_steps - 1)
            init_timestep = timesteps[init_timestep_idx]
            
            # Create noise with exact same shape as latent
            noise = torch.randn(latent.shape, generator=seed_g, device=self.device, dtype=latent.dtype)
            logger.info(f"Generated noise shape: {noise.shape}")
            
            # Mix with input latent based on strength
            latent_noised = scheduler.add_noise(
                original_samples=latent,
                noise=noise,
                timesteps=torch.tensor([init_timestep], device=self.device)
            )
            
            timesteps = timesteps[init_timestep_idx:]
            logger.info(f"V2V upscaling: Starting from timestep {init_timestep.item():.0f} (skipping {init_timestep_idx} steps)")
        else:
            noise = torch.randn(latent.shape, generator=seed_g, device=self.device, dtype=latent.dtype)
            latent_noised = noise
            
        logger.info(f"Final latent_noised shape: {latent_noised.shape}")
        
        # Calculate sequence length for the model
        B, C, lat_f, lat_h, lat_w = latent.shape
        seq_len = lat_f * lat_h * lat_w // (self.cfg.patch_size[1] * self.cfg.patch_size[2])
        
        # Prepare model input arguments in the correct format
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}
        
        return {
            'model': model,
            'latent': latent_noised,
            'scheduler': scheduler,
            'timesteps': timesteps,
            'context': context,
            'context_null': context_null,
            'inputs': (arg_c, arg_null),  # Should be tuple of (arg_c, arg_null) dictionaries
            'seed_g': seed_g,
            'accelerator': accelerator
        }
    
    def run_upscale(self):
        """Main upscaling pipeline for video enhancement"""
        logger.info("Starting WAN Video Upscaling Pipeline")
        
        # Step 1: Load input
        video_tensor, original_height, original_width, num_frames = self.load_input()
        
        # Calculate target dimensions
        if self.args.upscale_target_size:
            target_height, target_width = self.args.upscale_target_size
        else:
            target_height = int(original_height * self.args.upscale_factor)
            target_width = int(original_width * self.args.upscale_factor)
        
        # Round to nearest multiple of 32 for VAE+Transformer compatibility
        # VAE stride is 16, transformer patch_size is (1,2,2), so total spatial downsampling is 16*2=32
        target_height = (target_height // 32) * 32
        target_width = (target_width // 32) * 32
        
        logger.info(f"After rounding to multiple of 32: target_height={target_height}, target_width={target_width}")
        
        actual_factor = max(target_height / original_height, target_width / original_width)
        logger.info(f"Upscaling from {original_height}x{original_width} to {target_height}x{target_width} (factor: {actual_factor:.2f}x)")
        
        # Step 2: Load VAE
        from wan2_generate_video import load_vae
        vae_dtype = str_to_dtype(self.args.vae_dtype) if self.args.vae_dtype else torch.bfloat16
        vae = load_vae(self.args, self.cfg, self.device, vae_dtype)
        
        # Step 3: Process based on input type
        if video_tensor is not None:
            # Video input path: Resize at pixel level, then encode
            logger.info("Processing video input: resize then encode")
            
            # Resize video at pixel level
            video_resized = self.resize_video(video_tensor, target_height, target_width)
            
            # Encode resized video to latent
            latent = self.encode_with_tiling(video_resized, vae, vae_dtype)
            logger.info(f"Encoded latent shape after encoding resized video: {latent.shape}")
            
            # Clean up video tensors
            del video_tensor, video_resized
            torch.cuda.empty_cache()
            gc.collect()
        else:
            # Latent input path: Work directly with latents
            logger.info("Processing latent input directly")
            
            # Use the loaded latent
            latent = self.input_latent
            if len(latent.shape) == 4:
                latent = latent.unsqueeze(0)
            
            # Calculate current latent dimensions
            _, _, lat_f, lat_h, lat_w = latent.shape
            current_height = lat_h * 8
            current_width = lat_w * 8
            
            # If upscaling needed, decode -> resize -> encode
            if current_height != target_height or current_width != target_width:
                logger.info(f"Latent dimensions don't match target, decoding -> resizing -> encoding")
                
                # Decode current latent to video
                video_current = self.decode_with_tiling(latent, vae, vae_dtype)
                
                # Resize at pixel level
                video_resized = self.resize_video(video_current, target_height, target_width)
                
                # Encode back to latent
                latent = self.encode_with_tiling(video_resized, vae, vae_dtype)
                
                # Clean up
                del video_current, video_resized
                torch.cuda.empty_cache()
                gc.collect()
        
        # Step 4: Prepare V2V sampling
        logger.info("Preparing V2V sampling")
        sampling_inputs = self.prepare_v2v_sampling(latent, target_height, target_width, num_frames)
        
        # Step 5: Run sampling
        logger.info("Running V2V denoising")
        logger.info(f"Input latent to sampling: {sampling_inputs['latent'].shape}")
        from wan2_generate_video import run_sampling
        
        final_latent = run_sampling(
            sampling_inputs['model'],
            sampling_inputs['latent'],
            sampling_inputs['scheduler'],
            sampling_inputs['timesteps'],
            self.args,
            sampling_inputs['inputs'],
            self.device,
            sampling_inputs['seed_g'],
            sampling_inputs['accelerator'],
            is_ti2v=False
        )
        
        # Step 6: Clean up model to free VRAM
        logger.info("Unloading DiT model to free VRAM for VAE decoding")
        del sampling_inputs['model']
        del sampling_inputs['scheduler']
        del sampling_inputs['context']
        del sampling_inputs['context_null']
        del sampling_inputs['latent']
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.5)
        torch.cuda.empty_cache()
        
        # Step 7: Decode final latent to video
        logger.info("Decoding final result")
        if len(final_latent.shape) == 4:
            final_latent = final_latent.unsqueeze(0)
        
        decoded_video = self.decode_with_tiling(final_latent, vae, vae_dtype)
        
        # Step 8: Save output
        from wan2_generate_video import save_output
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
        seed = self.args.seed if self.args.seed else "auto"
        base_name = f"upscaled_{actual_factor:.1f}x_{time_flag}_{seed}"
        
        save_output(
            decoded_video,
            self.args,
            original_base_names=[base_name],
            latent_to_save=final_latent if self.args.output_type in ["latent", "both"] else None
        )
        
        logger.info(f"Upscaling complete! Output saved with prefix: {base_name}")
        
        # Final cleanup
        del vae, decoded_video, final_latent
        clean_memory_on_device(self.device)
        torch.cuda.empty_cache()
        gc.collect()


def run_upscale_from_args(args: argparse.Namespace):
    """Entry point for upscaling from command line arguments
    
    Args:
        args: Command line arguments
    """
    upscaler = WanUpscaler(args)
    upscaler.run_upscale()