"""
Adaptive Projected Guidance (APG) utilities for Wan 2.2
Based on MultiTalk's APG implementation: https://arxiv.org/abs/2410.02416
"""

import torch
import torch.nn as nn
from typing import Optional


class MomentumBuffer:
    """
    Momentum buffer for tracking running averages in APG.
    Used to stabilize guidance updates across denoising steps.
    """
    def __init__(self, momentum: float = -0.75):
        """
        Args:
            momentum: Momentum factor for exponential moving average.
                     Negative values provide dampening effect.
        """
        self.momentum = momentum
        self.running_average = 0
    
    def update(self, update_value: torch.Tensor):
        """Update the running average with new value."""
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average
    
    def reset(self):
        """Reset the running average to zero."""
        self.running_average = 0


def project(
    v0: torch.Tensor,  # [B, C, T, H, W] or [B, C, H, W]
    v1: torch.Tensor,  # [B, C, T, H, W] or [B, C, H, W]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project vector v0 onto v1 to get parallel and orthogonal components.
    
    Args:
        v0: Vector to project
        v1: Vector to project onto
        
    Returns:
        Tuple of (parallel_component, orthogonal_component)
    """
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    
    # Determine dimensions based on tensor shape
    if v0.dim() == 5:  # Video tensors [B, C, T, H, W]
        dims = [-1, -2, -3, -4]
    else:  # Image tensors [B, C, H, W]
        dims = [-1, -2, -3]
    
    v1 = torch.nn.functional.normalize(v1, dim=dims)
    v0_parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def adaptive_projected_guidance(
    diff: torch.Tensor,  # [B, C, T, H, W] or [B, C, H, W]
    pred_cond: torch.Tensor,  # [B, C, T, H, W] or [B, C, H, W]
    momentum_buffer: Optional[MomentumBuffer] = None,
    eta: float = 0.0,
    norm_threshold: float = 55.0,
    verbose: bool = False
) -> torch.Tensor:
    """
    Apply Adaptive Projected Guidance to modify the noise prediction.
    
    APG decomposes the guidance update into parallel and orthogonal components
    relative to the conditional prediction, allowing for more controlled guidance.
    
    Args:
        diff: Difference between conditional and unconditional predictions
        pred_cond: Conditional noise prediction
        momentum_buffer: Optional momentum buffer for stabilization
        eta: Weight for parallel component (0 = pure orthogonal guidance)
        norm_threshold: Maximum norm for the update (0 = no clipping)
        verbose: Whether to print debug information
        
    Returns:
        Modified guidance update
    """
    # Apply momentum if buffer provided
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    
    # Apply norm clipping if threshold specified
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        
        # Determine dimensions based on tensor shape
        if diff.dim() == 5:  # Video tensors
            dims = [-1, -2, -3, -4]
        else:  # Image tensors
            dims = [-1, -2, -3]
        
        diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
        
        if verbose:
            print(f"APG diff_norm: {diff_norm.mean().item():.2f}")
        
        scale_factor = torch.minimum(ones, norm_threshold / (diff_norm + 1e-8))
        diff = diff * scale_factor
    
    # Project diff onto pred_cond to get parallel and orthogonal components
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    
    # Combine components with eta weighting
    normalized_update = diff_orthogonal + eta * diff_parallel
    
    return normalized_update


class ChunkedVideoGenerator:
    """
    Helper class for managing chunked video generation with frame conditioning.
    """
    
    def __init__(
        self,
        frames_per_chunk: int = 25,
        motion_frames: int = 5,
        blend_overlap: bool = True,
        blend_mode: str = "linear"  # "linear", "smooth", or "cosine"
    ):
        """
        Args:
            frames_per_chunk: Number of frames to generate per chunk
            motion_frames: Number of frames to use as conditioning from previous chunk
            blend_overlap: Whether to blend overlapping frames between chunks
        """
        self.frames_per_chunk = frames_per_chunk
        self.motion_frames = min(motion_frames, frames_per_chunk // 2)  # Sanity check
        self.blend_overlap = blend_overlap
        self.blend_mode = blend_mode
        self.generated_frames = []
        self.current_chunk = 0
    
    def get_chunk_params(self, total_frames: int) -> list[tuple[int, int, int]]:
        """
        Calculate chunk parameters for a given total frame count.
        
        Returns:
            List of tuples (start_frame, end_frame, condition_frames)
        """
        chunks = []
        current_pos = 0
        
        while current_pos < total_frames:
            # First chunk has no conditioning
            if current_pos == 0:
                chunk_end = min(current_pos + self.frames_per_chunk, total_frames)
                chunks.append((0, chunk_end, 0))
                current_pos = chunk_end
            else:
                # Subsequent chunks use motion_frames for conditioning
                chunk_start = current_pos - self.motion_frames
                chunk_end = min(current_pos + self.frames_per_chunk - self.motion_frames, total_frames)
                chunks.append((chunk_start, chunk_end, self.motion_frames))
                current_pos = chunk_end
        
        return chunks
    
    def blend_frames(
        self,
        prev_frames: torch.Tensor,  # [C, T1, H, W]
        new_frames: torch.Tensor,   # [C, T2, H, W]
        overlap: int
    ) -> torch.Tensor:
        """
        Blend overlapping frames between chunks for smooth transitions.
        
        Args:
            prev_frames: Previous chunk's frames
            new_frames: Current chunk's frames
            overlap: Number of overlapping frames
            
        Returns:
            Blended frames tensor
        """
        if overlap <= 0 or not self.blend_overlap:
            return new_frames
        
        # Create blend weights based on blend mode
        t = torch.linspace(0, 1, overlap, device=new_frames.device)
        
        if self.blend_mode == "linear":
            blend_weights = t
        elif self.blend_mode == "smooth":
            # Smooth step function: 3t² - 2t³
            blend_weights = 3 * t**2 - 2 * t**3
        elif self.blend_mode == "cosine":
            # Cosine interpolation
            blend_weights = (1 - torch.cos(t * torch.pi)) / 2
        else:
            # Default to linear
            blend_weights = t
        
        blend_weights = blend_weights.view(1, -1, 1, 1)  # [1, T, 1, 1]
        
        # Blend the overlapping region
        blended_overlap = (1 - blend_weights) * prev_frames[:, -overlap:] + blend_weights * new_frames[:, :overlap]
        
        # Concatenate non-overlapping part with blended overlap
        if new_frames.shape[1] > overlap:
            result = torch.cat([blended_overlap, new_frames[:, overlap:]], dim=1)
        else:
            result = blended_overlap
        
        return result
    
    def add_chunk(self, frames: torch.Tensor, is_first: bool = False):
        """
        Add a generated chunk to the full video.
        
        Args:
            frames: Generated frames tensor [C, T, H, W]
            is_first: Whether this is the first chunk
        """
        if is_first or not self.generated_frames:
            self.generated_frames = frames
        else:
            # Blend with previous chunk if applicable
            if self.blend_overlap and self.motion_frames > 0:
                frames = self.blend_frames(
                    self.generated_frames,
                    frames,
                    self.motion_frames
                )
            
            # Append non-overlapping frames
            self.generated_frames = torch.cat([
                self.generated_frames,
                frames[:, self.motion_frames:] if self.motion_frames > 0 else frames
            ], dim=1)
        
        self.current_chunk += 1
    
    def get_conditioning_frames(self) -> Optional[torch.Tensor]:
        """
        Get the last motion_frames from the generated video for conditioning.
        
        Returns:
            Conditioning frames or None if no frames generated yet
        """
        if self.generated_frames is None or self.generated_frames.shape[1] == 0:
            return None
        
        return self.generated_frames[:, -self.motion_frames:]
    
    def reset(self):
        """Reset the generator for a new video."""
        self.generated_frames = []
        self.current_chunk = 0


def apply_apg_to_cfg(
    noise_pred_cond: torch.Tensor,
    noise_pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: Optional[MomentumBuffer] = None,
    norm_threshold: float = 55.0,
    eta: float = 0.0,
    verbose: bool = False
) -> torch.Tensor:
    """
    Apply APG to Classifier-Free Guidance.
    
    Args:
        noise_pred_cond: Conditional noise prediction
        noise_pred_uncond: Unconditional noise prediction
        guidance_scale: CFG scale
        momentum_buffer: Optional momentum buffer
        norm_threshold: APG norm threshold
        eta: APG eta parameter
        verbose: Whether to print debug info
        
    Returns:
        Modified noise prediction with APG
    """
    # Calculate standard CFG difference
    diff = noise_pred_cond - noise_pred_uncond
    
    # Apply APG to the difference
    modified_diff = adaptive_projected_guidance(
        diff=diff,
        pred_cond=noise_pred_cond,
        momentum_buffer=momentum_buffer,
        eta=eta,
        norm_threshold=norm_threshold,
        verbose=verbose
    )
    
    # Apply modified guidance
    noise_pred = noise_pred_uncond + guidance_scale * modified_diff
    
    return noise_pred