"""
Latent Preview Handler for Pusa Video Generation
Provides CPU-based latent2rgb conversion for real-time previews during generation
"""

import os
import torch
import torch.nn.functional as F
import av
import numpy as np


class LatentPreviewHandler:
    """Handles latent to RGB conversion and video preview generation on CPU"""

    def __init__(self, save_path, fps, preview_suffix):
        """
        Initialize the preview handler

        Args:
            save_path: Base output directory
            fps: Frames per second for preview video (typically 1/4 of target)
            preview_suffix: Unique suffix for preview filename
        """
        self.preview_dir = os.path.join(save_path, "previews")
        os.makedirs(self.preview_dir, exist_ok=True)

        self.preview_path = os.path.join(self.preview_dir, f"latent_preview_{preview_suffix}.mp4")
        self.fps = max(1, fps)  # Ensure at least 1 FPS

        # Wan model RGB factors for latent2rgb conversion
        self.rgb_factors = torch.tensor([
            [-0.1299, -0.1692,  0.2932], [ 0.0671,  0.0406,  0.0442],
            [ 0.3568,  0.2548,  0.1747], [ 0.0372,  0.2344,  0.1420],
            [ 0.0313,  0.0189, -0.0328], [ 0.0296, -0.0956, -0.0665],
            [-0.3477, -0.4059, -0.2925], [ 0.0166,  0.1902,  0.1975],
            [-0.0412,  0.0267, -0.1364], [-0.1293,  0.0740,  0.1636],
            [ 0.0680,  0.3019,  0.1128], [ 0.0032,  0.0581,  0.0639],
            [-0.1251,  0.0927,  0.1699], [ 0.0060, -0.0633,  0.0005],
            [ 0.3477,  0.2275,  0.2950], [ 0.1984,  0.0913,  0.1861]
        ], dtype=torch.float32).transpose(0, 1)  # [3, 16] for matrix multiplication

        self.rgb_bias = torch.tensor([-0.1835, -0.0868, -0.3360], dtype=torch.float32)

    @torch.no_grad()
    def decode_latent2rgb(self, latents):
        """
        Convert latents to RGB using linear transformation

        Args:
            latents: Tensor of shape [C, F, H, W] where C=16 channels

        Returns:
            RGB frames tensor of shape [F, 3, H, W]
        """
        # Ensure we're working on CPU
        if latents.device.type != 'cpu':
            latents = latents.cpu()

        # Convert to float32 for processing
        latents = latents.float()

        C, F, H, W = latents.shape
        rgb_frames = []

        # Process each frame
        for f in range(F):
            # Extract frame: [C, H, W]
            frame_latent = latents[:, f, :, :]

            # Reshape for linear transformation: [H, W, C]
            frame_latent = frame_latent.permute(1, 2, 0)

            # Apply linear transformation: [H, W, C] @ [C, 3] -> [H, W, 3]
            rgb = torch.nn.functional.linear(frame_latent, self.rgb_factors, self.rgb_bias)

            # Permute back to [3, H, W]
            rgb = rgb.permute(2, 0, 1)
            rgb_frames.append(rgb)

        # Stack frames: [F, 3, H, W]
        rgb_tensor = torch.stack(rgb_frames, dim=0)

        # Normalize to [0, 1]
        rgb_min = rgb_tensor.min()
        rgb_max = rgb_tensor.max()
        if rgb_max > rgb_min:
            rgb_tensor = (rgb_tensor - rgb_min) / (rgb_max - rgb_min)
        else:
            rgb_tensor = torch.zeros_like(rgb_tensor)

        return rgb_tensor

    @torch.no_grad()
    def process_preview(self, latents, step_num):
        """
        Process latents and save preview video

        Args:
            latents: Tensor of shape [C, F, H, W] on GPU or CPU
            step_num: Current denoising step number
        """
        try:
            print(f"[Preview] Processing preview at step {step_num + 1}")
            print(f"[Preview] Latents shape: {latents.shape}, device: {latents.device}")

            # Decode latents to RGB on CPU
            rgb_frames = self.decode_latent2rgb(latents)  # [F, 3, H, W]
            print(f"[Preview] RGB frames shape after decode: {rgb_frames.shape}")

            # Upscale 8x using bicubic interpolation
            upscaled = F.interpolate(
                rgb_frames,
                scale_factor=8,
                mode='bicubic',
                align_corners=False
            )
            print(f"[Preview] Upscaled shape: {upscaled.shape}")

            # Save to video file
            self.save_video(upscaled)
            print(f"[Preview] Preview saved to: {self.preview_path}")

        except Exception as e:
            print(f"[Preview Error] Error generating preview at step {step_num}: {e}")
            import traceback
            traceback.print_exc()

    @torch.no_grad()
    def save_video(self, frames):
        """
        Save frames as MP4 video

        Args:
            frames: Tensor of shape [F, 3, H, W] with values in [0, 1]
        """
        F_num, _, H, W = frames.shape

        # Convert to uint8 and move to CPU if needed
        frames = frames.clamp(0, 1).mul(255).byte().cpu()

        container = None
        try:
            # Open video container
            container = av.open(self.preview_path, mode='w')
            stream = container.add_stream('libx264', rate=self.fps)
            stream.pix_fmt = 'yuv420p'
            stream.width = W
            stream.height = H
            stream.options = {'crf': '23', 'preset': 'fast'}

            # Write frames
            for i in range(F_num):
                # Get frame and convert to numpy: [3, H, W] -> [H, W, 3]
                frame = frames[i].permute(1, 2, 0).numpy()

                # Ensure C-contiguous for av
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)

                # Create video frame and encode
                video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                for packet in stream.encode(video_frame):
                    container.mux(packet)

            # Flush stream
            for packet in stream.encode():
                container.mux(packet)

            container.close()
            container = None

        except Exception as e:
            print(f"Error saving preview video: {e}")
        finally:
            if container is not None:
                try:
                    container.close()
                except:
                    pass