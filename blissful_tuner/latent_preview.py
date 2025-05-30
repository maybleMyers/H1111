# latent_preview.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent preview for Blissful Tuner extension
License: Apache 2.0
Created on Mon Mar 10 16:47:29 2025

@author: blyss
"""
import os
import torch
import av
from PIL import Image
from .taehv import TAEHV
from .utils import load_torch_file
from blissful_tuner.utils import BlissfulLogger

logger = BlissfulLogger(__name__, "#8e00ed")


class LatentPreviewer():
    @torch.inference_mode()
    def __init__(self, args, original_latents, timesteps, device, dtype, model_type="hunyuan"):
        #print(f"DEBUG LATENT_PREVIEW.PY: LatentPreviewer __init__ called from file: {__file__}")
        self.mode = "latent2rgb" if not hasattr(args, 'preview_vae') or args.preview_vae is None else "taehv"
        ##logger.info(f"Initializing latent previewer with mode {self.mode}...")
        # Correctly handle framepack - it should subtract noise like others unless specifically told otherwise
        self.subtract_noise = True # Default to True for all models now
        # If you specifically need framepack NOT to subtract noise, you'd add a condition here
        # Example: self.subtract_noise = False if model_type == "framepack" else True
        self.args = args
        self.model_type = model_type
        self.device = device
        self.dtype = dtype if dtype != torch.float8_e4m3fn else torch.float16
        if model_type != "framepack" and original_latents is not None and timesteps is not None:
            self.original_latents = original_latents.to(self.device)
            self.timesteps_percent = timesteps / 1000
        # Add Framepack check here too if needed for original_latents/timesteps later
        # elif model_type == "framepack" and ...

        if self.model_type not in ["hunyuan", "wan", "framepack"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        if self.mode == "taehv":
            #logger.info(f"Loading TAEHV: {args.preview_vae}...")
            if os.path.exists(args.preview_vae):
                tae_sd = load_torch_file(args.preview_vae, safe_load=True, device=args.device)
            else:
                raise FileNotFoundError(f"{args.preview_vae} was not found!")
            self.taehv = TAEHV(tae_sd).to("cpu", self.dtype)  # Offload for VRAM and match datatype
            self.decoder = self.decode_taehv
            self.scale_factor = None
            self.fps = args.fps
        elif self.mode == "latent2rgb":
            self.decoder = self.decode_latent2rgb
            self.scale_factor = 8
            # Adjust FPS for latent2rgb preview if necessary
            # Original code had / 4, but maybe match output FPS is better?
            # Let's keep the / 4 logic for now as it was there before.
            self.fps = int(args.fps / 4) if args.fps > 4 else 1 # Ensure fps is at least 1


    @torch.inference_mode()
    def write_preview(self, frames, width, height, preview_suffix=None):
        suffix_str = f"_{preview_suffix}" if preview_suffix else ""
        base_name = f"latent_preview{suffix_str}" # This is correct for the filename itself

        preview_dir = os.path.join(self.args.save_path, "previews")
        os.makedirs(preview_dir, exist_ok=True)

        target = os.path.join(preview_dir, f"{base_name}.mp4")
        target_img = os.path.join(preview_dir, f"{base_name}.png")
        
        logger.info(f"LatentPreviewer.write_preview: Input frames shape: {frames.shape}, dtype: {frames.dtype}, device: {frames.device}")
        logger.info(f"LatentPreviewer.write_preview: Target width={width}, height={height}, fps={self.fps}")


        # Check if we only have a single frame.
        if frames.shape[0] == 1:
            logger.info(f"LatentPreviewer.write_preview: Saving single frame to {target_img}")
            try:
                # Clamp, scale, convert to byte and move to CPU
                frame = frames[0].clamp(0, 1).mul(255).byte().cpu()
                # Permute from (3, H, W) to (H, W, 3) for PIL.
                frame_np = frame.permute(1, 2, 0).numpy()
                Image.fromarray(frame_np).save(target_img)
                logger.info(f"LatentPreviewer: Successfully saved single frame preview to {target_img}")
            except Exception as e:
                logger.error(f"LatentPreviewer: Error saving single frame preview to {target_img}: {e}", exc_info=True)
            return

        # Otherwise, write out as a video.
        output_fps = max(1, self.fps)
        logger.info(f"LatentPreviewer.write_preview: Attempting to write MP4 video to {target} at {output_fps} FPS with {frames.shape[0]} frames.")
        
        container = None # Initialize for finally block
        try:
            container = av.open(target, mode="w")
            stream = container.add_stream("libx264", rate=output_fps) 
            stream.pix_fmt = "yuv420p"
            stream.width = width
            stream.height = height
            stream.options = {'crf': '23', 'preset': 'fast'} # Reasonable defaults

            #logger.info(f"LatentPreviewer: AV container opened for {target}. Stream options: {stream.options}")

            for frame_idx, frame_tensor in enumerate(frames): # Renamed 'frame' to 'frame_tensor'
                #logger.debug(f"LatentPreviewer: Processing frame {frame_idx+1}/{frames.shape[0]}")
                # Clamp to [0,1], scale, convert to byte and move to CPU.
                frame_processed = frame_tensor.clamp(0, 1).mul(255).byte().cpu()
                # Permute from (3, H, W) -> (H, W, 3) for AV.
                frame_np = frame_processed.permute(1, 2, 0).numpy()
                
                if not frame_np.flags['C_CONTIGUOUS']: # Ensure C-contiguous
                    frame_np = np.ascontiguousarray(frame_np)

                try:
                    video_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                    for packet in stream.encode(video_frame):
                        container.mux(packet)
                except Exception as e_encode:
                     logger.error(f"LatentPreviewer: Error encoding frame {frame_idx} for {target}: {e_encode}", exc_info=True)
                     break # Stop trying to encode if one frame fails critically

            # Flush out any remaining packets and close.
            #logger.info(f"LatentPreviewer: Flushing stream for {target}")
            for packet in stream.encode(): # Flush stream
                container.mux(packet)
            
            container.close() # Close container
            container = None # Indicate successful close
            logger.info(f"LatentPreviewer: Successfully finished writing preview video: {target}")
            if not os.path.exists(target) or os.path.getsize(target) == 0:
                logger.error(f"LatentPreviewer: Video file {target} was NOT created or is empty after closing.")

        except Exception as e_container:
            logger.error(f"LatentPreviewer: Error opening/writing MP4 container {target}: {e_container}", exc_info=True)
        finally:
            if container is not None: # If container was opened but not closed due to error
                try:
                    logger.warning(f"LatentPreviewer: Closing container for {target} in finally block due to earlier error.")
                    container.close()
                except Exception as e_close_finally:
                    logger.error(f"LatentPreviewer: Error closing container in finally block for {target}: {e_close_finally}", exc_info=True)

    @torch.inference_mode()
    def subtract_original_and_normalize(self, noisy_latents, current_step):
        # Ensure original_latents and timesteps_percent were initialized
        if not hasattr(self, 'original_latents') or not hasattr(self, 'timesteps_percent'):
             logger.warning("Cannot subtract noise: original_latents or timesteps_percent not initialized.")
             return noisy_latents # Return original if we can't process

        # Compute what percent of original noise is remaining
        noise_remaining = self.timesteps_percent[current_step].to(device=noisy_latents.device)
        # Subtract the portion of original latents
        denoisy_latents = noisy_latents - (self.original_latents.to(device=noisy_latents.device) * noise_remaining)

        # Normalize
        normalized_denoisy_latents = (denoisy_latents - denoisy_latents.mean()) / (denoisy_latents.std() + 1e-8)
        return normalized_denoisy_latents

    @torch.inference_mode()
    def preview(self, noisy_latents, current_step=None, preview_suffix=None): # CORRECTED METHOD NAME
        # noisy_latents is input [C, F_input, H, W]
        # self.original_latents is stored [C, F_orig, H, W]

        if self.device == "cuda" or self.device == torch.device("cuda"):
            torch.cuda.empty_cache()
        
        processed_noisy_latents = noisy_latents # Placeholder for now, original logic was complex
        # The complex logic from the previous attempt to unsqueeze/trim noisy_latents
        # can be simplified or re-evaluated once this basic naming error is fixed.
        # For now, let's assume noisy_latents arrives in the correct shape from run_sampling.
        # The primary goal here is to fix the AttributeError.

        if noisy_latents.ndim == 4: # Expecting [C,F,H,W] from run_sampling
            processed_noisy_latents = noisy_latents.unsqueeze(0) # Add batch for subtract
        elif noisy_latents.ndim == 5: # Already [B,C,F,H,W]
            processed_noisy_latents = noisy_latents
        else:
            logger.error(f"LatentPreviewer.preview: noisy_latents has unexpected ndim {noisy_latents.ndim}.")
            return # Don't proceed if shape is wrong

        # Apply subtraction only if enabled AND necessary inputs are available
        if self.subtract_noise and hasattr(self, 'original_latents') and hasattr(self, 'timesteps_percent') and current_step is not None:
            # Defensive check for S2V-like temporal dimension mismatch before subtraction
            if processed_noisy_latents.shape[2] > self.original_latents.shape[1] and \
               self.model_type == "wan": 
                num_extra_frames = processed_noisy_latents.shape[2] - self.original_latents.shape[1]
                logger.warning(
                    f"LatentPreviewer.preview: Trimming {num_extra_frames} frames from processed_noisy_latents (F={processed_noisy_latents.shape[2]}) "
                    f"to match self.original_latents (F={self.original_latents.shape[1]}) for S2V-like preview."
                )
                processed_noisy_latents_for_sub = processed_noisy_latents[:, :, :-num_extra_frames, :, :]
            else:
                processed_noisy_latents_for_sub = processed_noisy_latents
            
            denoisy_latents = self.subtract_original_and_normalize(processed_noisy_latents_for_sub, current_step)
        else:
            denoisy_latents = processed_noisy_latents


        decoded = self.decoder(denoisy_latents)  # Expects F, C, H, W output from decoder

        # Upscale if we used latent2rgb so output is same size as expected
        if self.scale_factor is not None:
            upscaled = torch.nn.functional.interpolate(
                decoded,
                scale_factor=self.scale_factor,
                mode="bicubic",
                align_corners=False
            )
        else:
            upscaled = decoded

        _, _, h, w = upscaled.shape # This gets H, W of the *pixel* space frames
        self.write_preview(upscaled, w, h, preview_suffix=preview_suffix)

    @torch.inference_mode()
    def decode_taehv(self, latents):
        """
        Decodes latents with the TAEHV model, returns shape (F, C, H, W).
        """
        self.taehv.to(self.device)  # Onload
        # --- Adjust permute based on expected input dimension order ---
        # Assuming TAEHV expects B, C, F, H, W (check TAEHV implementation)
        # If input `latents` is B, F, C, H, W (like hunyuan/wan), permute is needed
        # If input `latents` is B, C, F, H, W (like framepack), permute might not be needed or different
        if self.model_type == "framepack": # Assuming framepack latents are B,C,T,H,W
             latents_permuted = latents # No permute needed if TAEHV handles B,C,T,H,W
        else: # Assuming hunyuan/wan are B,F,C,H,W -> need B,C,F,H,W for TAEHV?
             # Original permute was (0, 2, 1, 3, 4) - Check if this matches TAEHV's expectation
             # This permutes B, F, C, H, W -> B, C, F, H, W
             latents_permuted = latents.permute(0, 2, 1, 3, 4)

        latents_permuted = latents_permuted.to(device=self.device, dtype=self.dtype)
        decoded = self.taehv.decode_video(latents_permuted, parallel=False, show_progress_bar=False)
        self.taehv.to("cpu")  # Offload
        return decoded.squeeze(0)  # squeeze off batch dimension -> F, C, H, W

    @torch.inference_mode()
    def decode_latent2rgb(self, latents):
        """
        Decodes latents to RGB using linear transform, returns shape (F, 3, H, W).
        Handles different latent dimension orders (B,F,C,H,W or B,C,T,H,W).
        """
        model_params = {
            "hunyuan": {
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
            },
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
            # No 'framepack' key needed, will map to 'hunyuan' below
        }

        # --- FIX: Determine the correct parameter key ---
        # Use 'hunyuan' parameters if the model type is 'framepack'
        params_key = "hunyuan" if self.model_type == "framepack" else self.model_type
        if params_key not in model_params:
             logger.error(f"Unsupported model type '{self.model_type}' (key '{params_key}') for latent2rgb.")
             # Optionally return a black image or raise error
             # Returning black image of expected shape might prevent further crashes
             b, c_or_f, t_or_c, h, w = latents.shape # Get shape
             num_frames = t_or_c if self.model_type == "framepack" else c_or_f # Estimate frame dim
             return torch.zeros((num_frames, 3, h * self.scale_factor, w * self.scale_factor), device='cpu')
             # raise KeyError(f"Unsupported model type '{self.model_type}' (key '{params_key}') for latent2rgb decoding.")

        latent_rgb_factors_data = model_params[params_key]["rgb_factors"]
        latent_rgb_factors_bias_data = model_params[params_key]["bias"]
        # --- END FIX ---

        # Prepare linear transform
        latent_rgb_factors = torch.tensor(
            latent_rgb_factors_data, # Use data fetched with correct key
            device=latents.device,
            dtype=latents.dtype
        ).transpose(0, 1)
        latent_rgb_factors_bias = torch.tensor(
            latent_rgb_factors_bias_data, # Use data fetched with correct key
            device=latents.device,
            dtype=latents.dtype
        )

        # Handle different dimension orders
        # B, F, C, H, W (Hunyuan, Wan) vs B, C, T, H, W (Framepack)
        if self.model_type == "framepack":
            # Input: B, C, T, H, W
            # We need to iterate through T (time/frames) dimension
            num_frames = latents.shape[2]
            frame_dim_idx = 2
            channel_dim_idx = 1
        else: # Wan (and potentially Hunyuan if prepared similarly)
            # Input is expected as B, C, F, H, W after preview() method
            num_frames = latents.shape[2] # F (frame dimension)
            channel_dim_idx = 1           # C
            frame_dim_idx = 2             # F

        latent_images = []
        for t in range(num_frames):
            # Extract frame t, permute C to the end for linear layer
            if self.model_type == "framepack":
                 # Extract B, C, H, W for frame t -> squeeze B -> C, H, W -> permute -> H, W, C
                 extracted = latents[:, :, t, :, :].squeeze(0).permute(1, 2, 0)
            else:
                 # Extract B, C, H, W for frame t -> squeeze B -> C, H, W -> permute -> H, W, C
                 extracted = latents[:, :, t, :, :].squeeze(0).permute(1, 2, 0)

            # extracted should now be (H, W, C)
            rgb = torch.nn.functional.linear(extracted, latent_rgb_factors, bias=latent_rgb_factors_bias) # shape = (H, W, 3)
            latent_images.append(rgb)

        # Stack frames into (F, H, W, 3)
        if not latent_images: # Handle case where loop might not run
             logger.warning("No latent images generated in decode_latent2rgb.")
             b, c_or_f, t_or_c, h, w = latents.shape
             num_frames = t_or_c if self.model_type == "framepack" else c_or_f
             return torch.zeros((num_frames, 3, h * self.scale_factor, w * self.scale_factor), device='cpu')

        latent_images_stacked = torch.stack(latent_images, dim=0)

        # Normalize to [0..1]
        latent_images_min = latent_images_stacked.min()
        latent_images_max = latent_images_stacked.max()
        if latent_images_max > latent_images_min:
            normalized_images = (latent_images_stacked - latent_images_min) / (latent_images_max - latent_images_min)
        else:
            # Handle case where max == min (e.g., all black image)
            normalized_images = torch.zeros_like(latent_images_stacked)

        # Permute to (F, 3, H, W) before returning
        final_images = normalized_images.permute(0, 3, 1, 2)
        return final_images