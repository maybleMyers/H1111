"""
Color matching algorithms for video frame consistency
Implements various color transfer methods for smooth video extension
"""

import torch
import numpy as np
from typing import Optional, Literal

def histogram_matching(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Match the histogram of source frames to reference frames.
    
    Args:
        source: Source tensor [C, F, H, W] in range [-1, 1]
        reference: Reference tensor [C, F_ref, H, W] in range [-1, 1]
    
    Returns:
        Color-matched source tensor in range [-1, 1]
    """
    device = source.device
    dtype = source.dtype
    
    # Convert to [0, 255] range for histogram operations
    source_255 = ((source + 1.0) * 127.5).clamp(0, 255).byte()
    reference_255 = ((reference + 1.0) * 127.5).clamp(0, 255).byte()
    
    # Process each channel separately
    matched = torch.zeros_like(source_255)
    
    for c in range(source.shape[0]):  # For each color channel
        # Compute histograms
        src_hist = torch.histc(source_255[c].float(), bins=256, min=0, max=255)
        ref_hist = torch.histc(reference_255[c].float(), bins=256, min=0, max=255)
        
        # Compute CDFs
        src_cdf = src_hist.cumsum(0)
        src_cdf = src_cdf / src_cdf[-1]  # Normalize
        
        ref_cdf = ref_hist.cumsum(0)
        ref_cdf = ref_cdf / ref_cdf[-1]  # Normalize
        
        # Create lookup table
        lut = torch.zeros(256, dtype=torch.uint8, device=device)
        ref_indices = torch.arange(256, dtype=torch.float32, device=device)
        
        for i in range(256):
            # Find closest match in reference CDF
            diff = torch.abs(ref_cdf - src_cdf[i])
            lut[i] = torch.argmin(diff).byte()
        
        # Apply lookup table
        src_flat = source_255[c].flatten()
        matched_flat = lut[src_flat.long()]
        matched[c] = matched_flat.reshape(source_255[c].shape)
    
    # Convert back to [-1, 1] range
    matched_float = matched.float() / 127.5 - 1.0
    return matched_float.to(dtype)


def reinhard_color_transfer(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Reinhard et al. color transfer in LAB color space.
    
    Args:
        source: Source tensor [C, F, H, W] in range [-1, 1]
        reference: Reference tensor [C, F_ref, H, W] in range [-1, 1]
    
    Returns:
        Color-matched source tensor in range [-1, 1]
    """
    # Convert to [0, 1] range
    source_01 = (source + 1.0) / 2.0
    reference_01 = (reference + 1.0) / 2.0
    
    # Calculate channel-wise statistics
    src_mean = source_01.mean(dim=(1, 2, 3), keepdim=True)
    src_std = source_01.std(dim=(1, 2, 3), keepdim=True)
    
    ref_mean = reference_01.mean(dim=(1, 2, 3), keepdim=True)
    ref_std = reference_01.std(dim=(1, 2, 3), keepdim=True)
    
    # Apply Reinhard transfer
    # Normalize source
    normalized = (source_01 - src_mean) / (src_std + 1e-8)
    
    # Scale and shift to match reference
    transferred = normalized * ref_std + ref_mean
    
    # Clamp and convert back to [-1, 1]
    transferred = transferred.clamp(0, 1)
    return transferred * 2.0 - 1.0


def mkl_color_matching(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Monge-Kantorovich linear (MKL) color mapping.
    
    Args:
        source: Source tensor [C, F, H, W] in range [-1, 1]
        reference: Reference tensor [C, F_ref, H, W] in range [-1, 1]
    
    Returns:
        Color-matched source tensor in range [-1, 1]
    """
    # Reshape to [C, N] where N is total pixels
    C, F, H, W = source.shape
    source_flat = source.reshape(C, -1)
    reference_flat = reference.reshape(C, -1)
    
    # Compute covariance matrices
    src_mean = source_flat.mean(dim=1, keepdim=True)
    ref_mean = reference_flat.mean(dim=1, keepdim=True)
    
    src_centered = source_flat - src_mean
    ref_centered = reference_flat - ref_mean
    
    # Covariance matrices
    src_cov = torch.mm(src_centered, src_centered.t()) / (source_flat.shape[1] - 1)
    ref_cov = torch.mm(ref_centered, ref_centered.t()) / (reference_flat.shape[1] - 1)
    
    # Add small epsilon for numerical stability
    eps = 1e-5
    src_cov = src_cov + eps * torch.eye(C, device=source.device)
    ref_cov = ref_cov + eps * torch.eye(C, device=reference.device)
    
    # Compute transformation matrix using eigendecomposition
    # A = ref_cov^(1/2) @ src_cov^(-1/2)
    
    # Eigendecomposition of source covariance
    src_eigvals, src_eigvecs = torch.linalg.eigh(src_cov)
    src_eigvals = src_eigvals.clamp(min=eps)
    src_cov_isqrt = src_eigvecs @ torch.diag(1.0 / torch.sqrt(src_eigvals)) @ src_eigvecs.t()
    
    # Eigendecomposition of reference covariance  
    ref_eigvals, ref_eigvecs = torch.linalg.eigh(ref_cov)
    ref_eigvals = ref_eigvals.clamp(min=eps)
    ref_cov_sqrt = ref_eigvecs @ torch.diag(torch.sqrt(ref_eigvals)) @ ref_eigvecs.t()
    
    # Transformation matrix
    transform = ref_cov_sqrt @ src_cov_isqrt
    
    # Apply transformation
    matched_flat = transform @ src_centered + ref_mean
    
    # Reshape back and clamp
    matched = matched_flat.reshape(C, F, H, W)
    return matched.clamp(-1, 1)


def mvgd_color_transfer(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Mean-Variance Gaussian Distribution color transfer.
    Simplified version of MKL for faster computation.
    
    Args:
        source: Source tensor [C, F, H, W] in range [-1, 1]
        reference: Reference tensor [C, F_ref, H, W] in range [-1, 1]
    
    Returns:
        Color-matched source tensor in range [-1, 1]
    """
    # Calculate statistics per channel
    src_mean = source.mean(dim=(1, 2, 3), keepdim=True)
    src_std = source.std(dim=(1, 2, 3), keepdim=True)
    
    ref_mean = reference.mean(dim=(1, 2, 3), keepdim=True)
    ref_std = reference.std(dim=(1, 2, 3), keepdim=True)
    
    # Normalize and transfer
    normalized = (source - src_mean) / (src_std + 1e-8)
    transferred = normalized * ref_std + ref_mean
    
    return transferred.clamp(-1, 1)


def apply_color_match(
    source: torch.Tensor,
    reference: torch.Tensor,
    method: Literal['disabled', 'hm', 'mkl', 'reinhard', 'mvgd', 'hm-mvgd-hm', 'hm-mkl-hm'] = 'hm'
) -> torch.Tensor:
    """
    Apply selected color matching method.
    
    Args:
        source: Source tensor [C, F, H, W] in range [-1, 1]
        reference: Reference tensor [C, F_ref, H, W] in range [-1, 1]
        method: Color matching algorithm to use
    
    Returns:
        Color-matched source tensor
    """
    if method == 'disabled':
        return source
    elif method == 'hm':
        return histogram_matching(source, reference)
    elif method == 'mkl':
        return mkl_color_matching(source, reference)
    elif method == 'reinhard':
        return reinhard_color_transfer(source, reference)
    elif method == 'mvgd':
        return mvgd_color_transfer(source, reference)
    elif method == 'hm-mvgd-hm':
        # Apply HM, then MVGD, then HM again
        result = histogram_matching(source, reference)
        result = mvgd_color_transfer(result, reference)
        result = histogram_matching(result, reference)
        return result
    elif method == 'hm-mkl-hm':
        # Apply HM, then MKL, then HM again
        result = histogram_matching(source, reference)
        result = mkl_color_matching(result, reference)
        result = histogram_matching(result, reference)
        return result
    else:
        raise ValueError(f"Unknown color matching method: {method}")


def blend_overlap_region(
    prev_frames: torch.Tensor,
    new_frames: torch.Tensor,
    overlap_frames: int,
    blend_type: str = 'linear'
) -> torch.Tensor:
    """
    Blend overlapping frames between chunks for smooth transition.
    
    Args:
        prev_frames: Previous chunk's last frames [C, overlap_frames, H, W]
        new_frames: New chunk's first frames [C, F, H, W]
        overlap_frames: Number of frames to blend
        blend_type: Type of blending ('linear', 'cosine', 'sigmoid')
    
    Returns:
        Blended new frames tensor
    """
    if overlap_frames <= 0:
        return new_frames
    
    device = new_frames.device
    C, F, H, W = new_frames.shape
    
    # Create blending weights
    if blend_type == 'linear':
        weights = torch.linspace(0, 1, overlap_frames, device=device)
    elif blend_type == 'cosine':
        weights = 0.5 * (1 - torch.cos(torch.linspace(0, np.pi, overlap_frames, device=device)))
    elif blend_type == 'sigmoid':
        x = torch.linspace(-6, 6, overlap_frames, device=device)
        weights = 1 / (1 + torch.exp(-x))
    else:
        weights = torch.linspace(0, 1, overlap_frames, device=device)
    
    # Reshape weights for broadcasting
    weights = weights.view(1, overlap_frames, 1, 1)
    
    # Blend the overlapping region
    blended = new_frames.clone()
    blend_frames = min(overlap_frames, F, prev_frames.shape[1])
    
    if blend_frames > 0:
        prev_overlap = prev_frames[:, -blend_frames:]
        new_overlap = new_frames[:, :blend_frames]
        
        # Apply weighted blending
        w = weights[:, :blend_frames]
        blended[:, :blend_frames] = (1 - w) * prev_overlap + w * new_overlap
    
    return blended