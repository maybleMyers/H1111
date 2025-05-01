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
    def preview(self, noisy_latents, current_step=None):
        if self.device == "cuda" or self.device == torch.device("cuda"):
            torch.cuda.empty_cache()
        if self.model_type == "wan":
            noisy_latents = noisy_latents.unsqueeze(0)  # F, C, H, W -> B, F, C, H, W
        elif self.model_type == "hunyuan" or self.model_type == "framepack": # Handle framepack like hunyuan
            pass  # already B, F, C, H, W or expected format B, C, T, H, W
        
        # Check dimensions for framepack - it might be B,C,T,H,W not B,F,C,H,W
        if self.model_type == "framepack" and noisy_latents.ndim == 5: # B,C,T,H,W
             # Ensure latent shape is B, F, C, H, W for consistent processing below if needed
             # If decoder expects B,C,T,H,W, this permute might be wrong. Check decoder.
             # Assuming decoder handles B,C,T,H,W for framepack's latent2rgb
             pass # Keep as B, C, T, H, W if latent2rgb handles it
        
        # Apply subtraction only if enabled AND necessary inputs are available
        if self.subtract_noise and hasattr(self, 'original_latents') and hasattr(self, 'timesteps_percent') and current_step is not None:
             denoisy_latents = self.subtract_original_and_normalize(noisy_latents, current_step)
        else:
             # If not subtracting, maybe still normalize? Depends on desired preview quality.
             # For now, just pass through if subtraction isn't happening.
             denoisy_latents = noisy_latents


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

        _, _, h, w = upscaled.shape
        self.write_preview(upscaled, w, h)

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
    def write_preview(self, frames, width, height):
        target = os.path.join(self.args.save_path, "latent_preview.mp4")
        # Check if we only have a single frame.
        if frames.shape[0] == 1:
            # Clamp, scale, convert to byte and move to CPU
            frame = frames[0].clamp(0, 1).mul(255).byte().cpu()
            # Permute from (3, H, W) to (H, W, 3) for PIL.
            frame_np = frame.permute(1, 2, 0).numpy()
            # Change the target filename from .mp4 to .png
            target_img = target.replace(".mp4", ".png")
            Image.fromarray(frame_np).save(target_img)
            #logger.info(f"Saved single frame preview to {target_img}") # Add log
            return

        # Otherwise, write out as a video.
        # Make sure fps is at least 1
        output_fps = max(1, self.fps)
        #logger.info(f"Writing preview video to {target} at {output_fps} FPS") # Add log
        container = av.open(target, mode="w")
        stream = container.add_stream("libx264", rate=output_fps) # Use output_fps
        stream.pix_fmt = "yuv420p"
        stream.width = width
        stream.height = height
        # Add option for higher quality preview encoding if needed
        # stream.options = {'crf': '18'} # Example: Lower CRF = higher quality

        # Loop through each frame.
        for frame_idx, frame in enumerate(frames):
            # Clamp to [0,1], scale, convert to byte and move to CPU.
            frame = frame.clamp(0, 1).mul(255).byte().cpu()
            # Permute from (3, H, W) -> (H, W, 3) for AV.
            frame_np = frame.permute(1, 2, 0).numpy()
            try:
                video_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            except Exception as e:
                 logger.error(f"Error encoding frame {frame_idx}: {e}")
                 # Optionally break or continue if one frame fails
                 break


        # Flush out any remaining packets and close.
        try:
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            #logger.info(f"Finished writing preview video: {target}") # Add log
        except Exception as e:
             logger.error(f"Error finalizing preview video: {e}")
             # Clean up container if possible
             try: container.close()
             except: pass

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