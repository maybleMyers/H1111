#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent preview for Blissful Tuner extension (latent2rgb mode only)
License: Apache 2.0
Created on Mon Mar 10 16:47:29 2025

@author: blyss
"""
import os
import torch
import av
from PIL import Image
# Assuming load_torch_file is in utils.py in the same directory
# If it's elsewhere, adjust the import path.
try:
    from .utils import load_torch_file
except ImportError:
    try:
        from utils import load_torch_file # Try importing directly if not a package
    except ImportError:
        # Fallback to a basic implementation if utils not found
        print("Warning: Could not import load_torch_file from utils. Using basic torch.load.")
        def load_torch_file(path, safe_load=False, device="cpu"):
             # Basic fallback, ignoring safe_load for simplicity here
             return torch.load(path, map_location=device)

# Use standard logging
import logging
logger = logging.getLogger(__name__)
# Configure basic logging if not already configured by the main script
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class LatentPreviewer():
    @torch.inference_mode()
    def __init__(self, args, original_latents, timesteps, device, dtype, model_type="wan"):
        """
        Initializes the LatentPreviewer in latent2rgb mode.

        Args:
            args: Command line arguments from the main script (used for save_path, fps).
            original_latents (torch.Tensor): The initial noise tensor.
            timesteps (torch.Tensor): The timesteps used by the scheduler.
            device (torch.device): The compute device.
            dtype (torch.dtype): The data type for computation.
            model_type (str): The type of model ("wan" or "hunyuan"), used for latent factors.
        """
        self.args = args # Store args for later use (e.g., save_path, fps)
        logger.info("Initializing latent previewer (latent2rgb mode only)...")
        self.model_type = model_type
        self.device = device
        self.dtype = dtype if dtype != torch.float8_e4m3fn else torch.float16 # Use float16 if original was fp8
        self.original_latents = original_latents.to(self.device, dtype=self.dtype) # Ensure correct dtype and device

        # Calculate timesteps percentage relative to the scheduler's train steps (usually 1000)
        num_train_timesteps = 1000 # Assume standard 1000 steps
        self.timesteps_percent = timesteps.to(self.device) / num_train_timesteps

        if self.model_type not in ["hunyuan", "wan"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # --- Directly configure for latent2rgb mode ---
        self.decoder = self.decode_latent2rgb
        self.scale_factor = 8 # Standard VAE scale factor
        # Adjust fps for latent preview (often looks better slower)
        self.fps = max(1, int(args.fps / 4)) # Example: quarter speed
        logger.info(f"Using latent2rgb preview mode with scale factor {self.scale_factor} and fps {self.fps}.")
        # --- End configuration ---


    @torch.inference_mode()
    def preview(self, noisy_latents, current_step):
        """
        Generates and saves a preview of the current latent state.

        Args:
            noisy_latents (torch.Tensor): The current latent tensor from the sampling loop.
                                           Expected shape [C, F, H, W].
            current_step (int): The current step index in the sampling loop (0-based).
        """
        logger.debug(f"Generating preview for step {current_step + 1}...") # Log step number (1-based)
        # Ensure correct device and dtype for calculations
        noisy_latents = noisy_latents.to(self.device, dtype=self.dtype)

        # Add batch dim if needed (decoder expects batched input)
        if self.model_type == "wan":
             if len(noisy_latents.shape) == 4:
                  noisy_latents = noisy_latents.unsqueeze(0) # Add batch dim: [1, C, F, H, W]
        # Add elif for other model types if their input differs

        # Match original_latents batch dim if necessary
        original_latents_matched = self.original_latents
        if len(original_latents_matched.shape) == 4 and len(noisy_latents.shape) == 5:
             original_latents_matched = original_latents_matched.unsqueeze(0)

        # Denoise approximation
        denoisy_latents = self.subtract_original_and_normalize(noisy_latents, current_step, original_latents_matched)

        # Decode the denoised latents
        # Decoder function expects [B, C, F, H, W] and returns [F, C, H, W]
        decoded = self.decoder(denoisy_latents, current_step)

        # Upscale using the scale factor
        # Need to check channel dim location before interpolate
        # decoded should be [F, C, H, W]
        if decoded.shape[1] != 3:
             logger.error(f"Decoded tensor has unexpected channel dimension {decoded.shape[1]} before upscaling.")
             return # Cannot upscale correctly

        upscaled = torch.nn.functional.interpolate(
            decoded,
            scale_factor=self.scale_factor,
            mode="bicubic", # Use bicubic for better quality
            align_corners=False
        )

        # Ensure output is F, C, H, W before writing
        if len(upscaled.shape) != 4 or upscaled.shape[1] != 3:
             logger.error(f"Final upscaled tensor has unexpected shape {upscaled.shape}. Cannot write preview.")
             return

        _, _, h, w = upscaled.shape
        # Pass step number for potentially unique filenames
        self.write_preview(upscaled, w, h, current_step + 1) # Pass 1-based step number
        logger.debug(f"Preview for step {current_step + 1} written.")
        # Clean cache after preview generation
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def subtract_original_and_normalize(self, noisy_latents, current_step, original_latents_matched):
        """
        Approximates denoising by subtracting estimated original noise.

        Args:
            noisy_latents (torch.Tensor): Current latents [B, C, F, H, W].
            current_step (int): Current step index (0-based).
            original_latents_matched (torch.Tensor): Initial noise, matched shape [B, C, F, H, W].

        Returns:
            torch.Tensor: Normalized, approximated denoised latents [B, C, F, H, W].
        """
        if current_step >= len(self.timesteps_percent):
            logger.warning(f"Current step {current_step} exceeds timesteps_percent length {len(self.timesteps_percent)}. Using last value.")
            noise_remaining = self.timesteps_percent[-1]
        else:
            noise_remaining = self.timesteps_percent[current_step]

        # Ensure noise_remaining is scalar or broadcastable
        noise_remaining = noise_remaining.to(device=noisy_latents.device, dtype=noisy_latents.dtype) # Match device and dtype
        while len(noise_remaining.shape) < len(noisy_latents.shape):
            noise_remaining = noise_remaining.unsqueeze(-1) # Make broadcastable: [1, 1, 1, 1, 1]

        # Subtract the estimated portion of original noise
        original_latents_matched = original_latents_matched.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        denoisy_latents = noisy_latents - (original_latents_matched * noise_remaining)

        # Normalize latents to approx [-1, 1] range for better visualization
        max_val = torch.max(torch.abs(denoisy_latents)) + 1e-8 # Avoid division by zero
        normalized_denoisy_latents = denoisy_latents / max_val

        return normalized_denoisy_latents

    @torch.inference_mode()
    def write_preview(self, frames, width, height, step_num):
        """
        Writes the decoded frames to a video or image file.

        Args:
            frames (torch.Tensor): Tensor of frames [F, C, H, W], range approx [-1, 1] or [0, 1].
            width (int): Frame width.
            height (int): Frame height.
            step_num (int): The current step number (1-based) for filename.
        """
        # Create previews subdirectory if it doesn't exist
        preview_dir = os.path.join(self.args.save_path, "previews")
        try:
            os.makedirs(preview_dir, exist_ok=True)
        except OSError as e:
             logger.error(f"Failed to create preview directory {preview_dir}: {e}")
             return

        # Determine target filename (add step number)
        base_filename = f"latent_preview_step_{step_num:04d}"
        target_base = os.path.join(preview_dir, base_filename)

        # Clamp and scale frames to [0, 255] byte range
        # Input frames are normalized, assume roughly [-1, 1] from normalize step
        frames = ((frames + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0
        frames = frames.byte().cpu() # Move to CPU and convert to byte

        # Check if we only have a single frame.
        num_frames = frames.shape[0]
        if num_frames == 1:
            target_img = target_base + ".png"
            try:
                # Permute from (C, H, W) to (H, W, C) for PIL.
                frame_np = frames[0].permute(1, 2, 0).numpy()
                Image.fromarray(frame_np).save(target_img)
                logger.debug(f"Saved single frame preview: {target_img}")
            except Exception as e:
                logger.error(f"Failed to save preview image {target_img}: {e}")
            return

        # Otherwise, write out as a video.
        target_vid = target_base + ".mp4"
        try:
            container = av.open(target_vid, mode="w")
            stream = container.add_stream("libx264", rate=self.fps)
            stream.pix_fmt = "yuv420p" # Common format
            stream.width = width
            stream.height = height
            # Optional: Set CRF for quality (lower means better quality, larger file)
            # stream.options = {"crf": "18"}

            # Loop through each frame.
            for frame_idx in range(num_frames):
                frame = frames[frame_idx]
                # Permute from (C, H, W) -> (H, W, C) for AV.
                frame_np = frame.permute(1, 2, 0).numpy()
                video_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)

            # Flush out any remaining packets and close.
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            logger.debug(f"Saved video preview: {target_vid}")
        except Exception as e:
             logger.error(f"Failed to write preview video {target_vid}: {e}")


    # decode_taehv method is removed

    @torch.inference_mode()
    def decode_latent2rgb(self, latents, current_step):
        """
        Decodes latents to RGB using linear transform.
        Input: [B, C, F, H, W]
        Output: [F, 3, H, W]
        """
        model_params = {
            # Factors from comfyui_latent_preview.py
            "wan": {
                 "rgb_factors": [
                      [-0.1299, -0.1692,  0.2932], [ 0.0671,  0.0406,  0.0442],
                      [ 0.3568,  0.2548,  0.1747], [ 0.0372,  0.2344,  0.1420],
                      [ 0.0313,  0.0189, -0.0328], [ 0.0296, -0.0956, -0.0665],
                      [-0.3477, -0.4059, -0.2925], [ 0.0166,  0.1902,  0.1975],
                      [-0.0412,  0.0267, -0.1364], [-0.1293,  0.0740,  0.1636],
                      [ 0.0680,  0.3019,  0.1128], [ 0.0032,  0.0581,  0.0639],
                      [-0.1251,  0.0927,  0.1699], [ 0.0060, -0.0633,  0.0005],
                      [ 0.3477,  0.2275,  0.2950], [ 0.1984,  0.0913,  0.1861]
                  ],
                 "bias": [-0.1835, -0.0868, -0.3360],
            },
             "hunyuan": { # Added for completeness if needed later
                "rgb_factors": [
                    [-0.0395, -0.0331,  0.0445], [ 0.0696,  0.0795,  0.0518],
                    [ 0.0135, -0.0945, -0.0282], [ 0.0108, -0.0250, -0.0765],
                    [-0.0209,  0.0032,  0.0224], [-0.0804, -0.0254, -0.0639],
                    [-0.0991,  0.0271, -0.0669], [-0.0646, -0.0422, -0.0400],
                    [-0.0696, -0.0595, -0.0894], [-0.0799, -0.0208, -0.0375],
                    [ 0.1166,  0.1627,  0.0962], [ 0.1165,  0.0432,  0.0407],
                    [-0.2315, -0.1920, -0.1355], [-0.0270,  0.0401, -0.0821],
                    [-0.0616, -0.0997, -0.0727], [ 0.0249, -0.0469, -0.1703]
                ],
                "bias": [0.0259, -0.0192, -0.0761],
            }
        }

        if self.model_type not in model_params:
             logger.error(f"No latent2rgb factors defined for model type {self.model_type}")
             # Return dummy tensor
             b, c, f, h, w = latents.shape
             return torch.zeros([f, 3, h * self.scale_factor, w * self.scale_factor], device=self.device, dtype=self.dtype)

        latent_rgb_factors = model_params[self.model_type]["rgb_factors"]
        latent_rgb_factors_bias = model_params[self.model_type]["bias"]

        # Prepare linear transform factors
        # --- FIX: Remove the .t() ---
        factors = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype) # Shape: [C=16, 3]
        # --- END FIX ---
        bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype) # Shape: [3]

        # Input latents: [B, C, F, H, W]
        # We process one batch element (B=0)
        latents_single_batch = latents[0] # [C, F, H, W]

        # Reshape for linear operation: Treat F as batch dim temporary
        # Permute to [F, H, W, C]
        latents_permuted = latents_single_batch.permute(1, 2, 3, 0) # [F, H, W, C]
        f, h, w, c = latents_permuted.shape

        # Apply linear transformation: (F*H*W, C=16) @ (C=16, 3) -> (F*H*W, 3)
        pixels = torch.matmul(latents_permuted.view(-1, c), factors) + bias
        # Reshape back to [F, H, W, 3]
        rgb_frames = pixels.view(f, h, w, 3)

        # Permute to [F, 3, H, W] for standard format
        rgb_frames_final = rgb_frames.permute(0, 3, 1, 2)

        return rgb_frames_final