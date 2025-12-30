"""VACE Video and Mask Processing Module

This module provides functions for processing control videos and masks
for VACE-based video generation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class VaceVideoProcessor:
    """Process control videos for VACE models"""

    def __init__(
        self,
        vae_stride: Tuple[int, int, int] = (4, 8, 8),
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        device: str = "cuda"
    ):
        """Initialize VACE video processor

        Args:
            vae_stride: VAE downsampling stride
            patch_size: Model patch size
            device: Processing device
        """
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.device = device
        self.downsample = tuple([x * y for x, y in zip(vae_stride, patch_size)])

    def resize_and_crop_video(
        self,
        video: Union[torch.Tensor, np.ndarray],
        target_height: int,
        target_width: int
    ) -> torch.Tensor:
        """Resize and center crop video to target dimensions

        Args:
            video: Input video tensor or numpy array
            target_height: Target height
            target_width: Target width

        Returns:
            Processed video tensor
        """
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)

        # video shape: [T, H, W, C] or [C, T, H, W]
        if video.dim() == 4:
            if video.shape[-1] in [1, 3, 4]:  # [T, H, W, C]
                video = video.permute(3, 0, 1, 2)  # -> [C, T, H, W]

        C, T, H, W = video.shape

        # Calculate resize dimensions to maintain aspect ratio
        scale = max(target_height / H, target_width / W)
        new_h = int(H * scale)
        new_w = int(W * scale)

        # Resize
        video = video.float()
        video = F.interpolate(
            video.view(C * T, 1, H, W),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).view(C, T, new_h, new_w)

        # Center crop
        start_h = (new_h - target_height) // 2
        start_w = (new_w - target_width) // 2
        video = video[:, :, start_h:start_h + target_height, start_w:start_w + target_width]

        return video

    def encode_frames(
        self,
        vae,
        frames: List[torch.Tensor],
        ref_images: Optional[List[torch.Tensor]] = None,
        masks: Optional[List[torch.Tensor]] = None,
        tile_size: int = 0
    ) -> List[torch.Tensor]:
        """Encode frames with VACE processing

        Args:
            vae: VAE model for encoding
            frames: List of frame tensors
            ref_images: Optional reference images
            masks: Optional masks for inpainting
            tile_size: Tile size for memory-efficient processing

        Returns:
            List of encoded latents
        """
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            # Simple encoding without masks
            latents = vae.encode(frames, tile_size=tile_size)
        else:
            # Encode with masks for inpainting
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]

            inactive_latents = vae.encode(inactive, tile_size=tile_size)
            reactive_latents = vae.encode(reactive, tile_size=tile_size)

            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive_latents, reactive_latents)]

        # Concatenate reference images if provided
        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = vae.encode(refs, tile_size=tile_size)
                else:
                    ref_latent = vae.encode(refs, tile_size=tile_size)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]

                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)

            cat_latents.append(latent)

        return cat_latents

    def encode_masks(
        self,
        masks: List[torch.Tensor],
        ref_images: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """Encode masks for VACE processing

        Args:
            masks: List of mask tensors
            ref_images: Optional reference images

        Returns:
            List of encoded masks
        """
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // self.vae_stride[0])
            height = 2 * (int(height) // (self.vae_stride[1] * 2))
            width = 2 * (int(width) // (self.vae_stride[2] * 2))

            # Reshape mask
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, self.vae_stride[1], width, self.vae_stride[2]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(64, depth, height, width)
            mask = F.avg_pool3d(
                mask[None, ...],
                kernel_size=(self.vae_stride[0], 1, 1),
                stride=(self.vae_stride[0], 1, 1)
            )[:, :, :new_depth, :, :]

            # Add reference frames if provided
            if refs is not None:
                # Add reference frame dimensions
                ref_count = 1 if isinstance(refs, torch.Tensor) else len(refs)
                ref_mask = torch.ones(
                    1, 64, ref_count, height, width,
                    dtype=mask.dtype, device=mask.device
                )
                mask = torch.cat([ref_mask, mask], dim=2)

            result_masks.append(mask[0])

        return result_masks

    def create_vace_latent(
        self,
        encoded_frames: List[torch.Tensor],
        encoded_masks: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Combine encoded frames and masks into VACE latent

        Args:
            encoded_frames: List of encoded frame tensors
            encoded_masks: List of encoded mask tensors

        Returns:
            List of VACE latent tensors
        """
        vace_latents = []

        for frame_latent, mask_latent in zip(encoded_frames, encoded_masks):
            # Concatenate frame and mask latents
            # frame_latent shape: [C, F, H, W] where C=16 or 32 (with mask)
            # mask_latent shape: [64, F, H, W]

            # Ensure same spatial dimensions
            if frame_latent.shape[-2:] != mask_latent.shape[-2:]:
                mask_latent = F.interpolate(
                    mask_latent,
                    size=frame_latent.shape[-2:],
                    mode='nearest'
                )

            # Combine to create 96-channel input (16 + 64 + 16 padding)
            if frame_latent.shape[0] == 16:
                # No mask in frame encoding
                padding = torch.zeros_like(frame_latent)
                vace_latent = torch.cat([frame_latent, mask_latent, padding], dim=0)
            else:
                # Frame already includes mask channels
                vace_latent = torch.cat([frame_latent, mask_latent], dim=0)

            # Ensure we have 96 channels total
            if vace_latent.shape[0] < 96:
                padding_size = 96 - vace_latent.shape[0]
                padding = torch.zeros(
                    padding_size, *vace_latent.shape[1:],
                    dtype=vace_latent.dtype, device=vace_latent.device
                )
                vace_latent = torch.cat([vace_latent, padding], dim=0)
            elif vace_latent.shape[0] > 96:
                vace_latent = vace_latent[:96]

            vace_latents.append(vace_latent)

        return vace_latents


def process_control_video(
    video_path: str,
    target_frames: int,
    target_height: int,
    target_width: int
) -> torch.Tensor:
    """Load and process a control video

    Args:
        video_path: Path to control video
        target_frames: Number of frames to extract
        target_height: Target height
        target_width: Target width

    Returns:
        Processed video tensor [C, F, H, W]
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < target_frames:
        ret, frame = cap.read()
        if not ret:
            if len(frames) == 0:
                raise ValueError(f"Could not read any frames from {video_path}")
            # Repeat last frame if we run out
            frame = frames[-1]
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    cap.release()

    # Convert to tensor
    frames = np.stack(frames[:target_frames], axis=0)  # [F, H, W, C]
    frames = torch.from_numpy(frames).float() / 255.0  # Normalize to [0, 1]
    frames = frames.permute(3, 0, 1, 2)  # [C, F, H, W]

    # Resize and crop
    processor = VaceVideoProcessor()
    frames = processor.resize_and_crop_video(frames, target_height, target_width)

    # Normalize to [-1, 1] for VAE
    frames = frames * 2.0 - 1.0

    return frames


def process_reference_image(
    image_path: str,
    target_height: int,
    target_width: int,
    remove_background: bool = False
) -> torch.Tensor:
    """Load and process a reference image for VACE

    Args:
        image_path: Path to reference image
        target_height: Target height
        target_width: Target width
        remove_background: Whether to remove background

    Returns:
        Processed image tensor [C, 1, H, W]
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)

    if remove_background:
        # Placeholder for background removal
        # In production, use a library like rembg
        logger.info("Background removal requested but not implemented")

    # Convert to tensor
    image = torch.from_numpy(image).float() / 255.0  # [H, W, C]
    image = image.permute(2, 0, 1).unsqueeze(1)  # [C, 1, H, W]

    # Resize and crop
    processor = VaceVideoProcessor()
    image = processor.resize_and_crop_video(image, target_height, target_width)

    # Normalize to [-1, 1]
    image = image * 2.0 - 1.0

    return image


def create_control_mask(
    num_frames: int,
    control_weight: float = 1.0,
    control_start: float = 0.0,
    control_end: float = 1.0,
    falloff_percentage: float = 0.1,
    device: str = "cuda"
) -> torch.Tensor:
    """Create a control weight mask for frame-level control

    Args:
        num_frames: Number of frames
        control_weight: Overall control weight
        control_start: Start position (0-1)
        control_end: End position (0-1)
        falloff_percentage: Percentage of frames for falloff
        device: Device for tensor

    Returns:
        Control mask tensor [1, 1, F]
    """
    mask = torch.zeros([1, 1, num_frames], device=device, dtype=torch.float32)

    start_idx = max(0, min(num_frames - 1, int(num_frames * control_start)))
    end_idx = max(start_idx + 1, min(num_frames, int(num_frames * control_end)))
    falloff_len = max(2, int(num_frames * falloff_percentage))

    # Main active region
    if start_idx < end_idx:
        mask[:, :, start_idx:end_idx] = 1.0

    # Smooth falloff at start
    if start_idx > 0:
        fallon_start = max(0, start_idx - falloff_len)
        fallon_len = start_idx - fallon_start
        if fallon_len > 0:
            t = torch.linspace(0, 1, fallon_len, device=device)
            smooth_t = 0.5 - 0.5 * torch.cos(t * torch.pi)
            mask[:, :, fallon_start:start_idx] = smooth_t.reshape(1, 1, -1)

    # Smooth falloff at end
    if end_idx < num_frames:
        falloff_start = end_idx
        falloff_end = min(num_frames, falloff_start + falloff_len)
        falloff_actual_len = falloff_end - falloff_start
        if falloff_actual_len > 0:
            t = torch.linspace(0, 1, falloff_actual_len, device=device)
            smooth_t = 0.5 + 0.5 * torch.cos(t * torch.pi)
            mask[:, :, falloff_start:falloff_end] = smooth_t.reshape(1, 1, -1)

    # Apply overall weight
    mask = mask * control_weight

    return mask