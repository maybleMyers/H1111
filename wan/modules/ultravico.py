"""
UltraViCo: Ultra-extrapolated Video via Attention Concentration

Based on the paper: "UltraViCo: Breaking Extrapolation Limits in Video Diffusion Transformers"
https://arxiv.org/abs/2511.20123

This module implements attention decay for video length extrapolation, addressing:
1. Quality degradation (universal) - caused by attention dispersion
2. Content repetition (model-specific) - caused by harmonic RoPE frequencies

The key insight is that tokens beyond the training window dilute learned attention patterns.
By applying a decay factor to out-of-window attention, we restore focus on reliable context.

USAGE:
    This is ONLY activated when --ultravico flag is passed to wan2_generate_video.py
    Existing code paths remain completely unchanged without the flag.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class UltraViCoConfig:
    """Configuration for UltraViCo attention decay."""
    enabled: bool = False
    # Training window size in latent frames (5s@24fps = ~31 latent frames for Wan2.2)
    training_frames: int = 21  # Default for Wan2.2 (5s at 24fps with 4x temporal compression)
    # Decay factor for out-of-window attention (0.85-0.95 recommended)
    alpha: float = 0.9
    # Decay factor for harmonic risk positions (only if repetition occurs)
    beta: float = 0.6
    # Whether to apply harmonic suppression (set True if you see repetition)
    suppress_harmonics: bool = False
    # Frames around harmonic peaks to suppress
    gamma: int = 4
    # Harmonic period in latent frames (auto-detected or manual)
    harmonic_period: Optional[int] = None


def compute_temporal_positions(seq_len: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    """
    Compute temporal position for each token in flattened sequence.

    Tokens are flattened in (T, H, W) order, so:
    - token i has temporal position: t_i = i // (H * W)

    Args:
        seq_len: Total sequence length (T * H * W)
        height: Spatial height in latent space
        width: Spatial width in latent space
        device: Device for tensor

    Returns:
        Tensor of shape [seq_len] with temporal position for each token
    """
    hw = height * width
    temporal_pos = torch.arange(seq_len, device=device) // hw
    return temporal_pos


def create_ultravico_bias(
    seq_len: int,
    height: int,
    width: int,
    config: UltraViCoConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create attention bias for UltraViCo decay.

    The bias is added to attention logits before softmax:
    - 0 for in-window pairs (no change)
    - log(alpha) for out-of-window pairs (multiplicative decay after softmax)
    - log(beta) for harmonic risk positions (stronger decay)

    Args:
        seq_len: Total sequence length
        height: Spatial height in latent space
        width: Spatial width in latent space
        config: UltraViCo configuration
        device: Device for tensor
        dtype: Data type for tensor

    Returns:
        Attention bias tensor of shape [seq_len, seq_len]
    """
    # Get temporal positions
    t_pos = compute_temporal_positions(seq_len, height, width, device)

    # Compute temporal distance matrix
    # t_dist[i, j] = |t_i - t_j|
    t_dist = torch.abs(t_pos.unsqueeze(1) - t_pos.unsqueeze(0))

    # Training window radius (half of training frames)
    window_radius = config.training_frames // 2

    # Create bias tensor (0 = no change, negative = decay)
    bias = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)

    # Apply alpha decay to out-of-window positions
    out_of_window = t_dist > window_radius
    if config.alpha < 1.0:
        # log(alpha) as additive bias = multiplicative alpha after exp
        bias[out_of_window] = math.log(config.alpha)

    # Apply beta decay to harmonic risk positions (if enabled)
    if config.suppress_harmonics and config.beta < config.alpha:
        period = config.harmonic_period
        if period is None:
            # Auto-detect: use training_frames as period
            period = config.training_frames

        # Find positions near harmonic alignment: |t_i - t_j| ≈ m * period
        max_dist = t_dist.max().item()
        for m in range(1, int(max_dist // period) + 2):
            harmonic_center = m * period
            near_harmonic = (t_dist >= harmonic_center - config.gamma) & \
                           (t_dist <= harmonic_center + config.gamma)
            # Only apply to out-of-window positions
            risk_positions = near_harmonic & out_of_window
            bias[risk_positions] = math.log(config.beta)

    return bias


def create_ultravico_bias_for_shape(
    shape: Tuple[int, int, int],
    config: UltraViCoConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create UltraViCo bias given visual shape (T, H, W).

    Args:
        shape: Visual shape tuple (duration, height, width) in latent space
        config: UltraViCo configuration
        device: Device for tensor
        dtype: Data type

    Returns:
        Attention bias tensor
    """
    duration, height, width = shape
    seq_len = duration * height * width
    return create_ultravico_bias(seq_len, height, width, config, device, dtype)


# Global config instance (can be set from CLI)
_ultravico_config: Optional[UltraViCoConfig] = None
_cached_bias: Optional[torch.Tensor] = None
_cached_shape: Optional[Tuple[int, int, int]] = None
_current_visual_shape: Optional[Tuple[int, int, int]] = None

# Sage UltraViCo parameters (for memory-efficient Triton kernel path)
_sage_ultravico_enabled: bool = False
_sage_multi_factor: float = 0.9  # UltraViCo decay factor (alpha)
_sage_frame_tokens: int = 1560  # Tokens per latent frame (resolution-dependent)
_sage_training_frames: int = 21  # Training window in latent frames


def set_ultravico_config(config: UltraViCoConfig):
    """Set global UltraViCo configuration."""
    global _ultravico_config, _cached_bias, _cached_shape
    _ultravico_config = config
    _cached_bias = None
    _cached_shape = None


def get_ultravico_config() -> Optional[UltraViCoConfig]:
    """Get global UltraViCo configuration."""
    return _ultravico_config


def is_ultravico_enabled() -> bool:
    """Check if UltraViCo is enabled."""
    return _ultravico_config is not None and _ultravico_config.enabled


def set_current_visual_shape(shape: Tuple[int, int, int]):
    """Set current visual shape for attention bias computation."""
    global _current_visual_shape, _cached_bias, _cached_shape
    _current_visual_shape = shape
    # Invalidate cache when shape changes
    if _cached_shape != shape:
        _cached_bias = None
        _cached_shape = None


def get_current_visual_shape() -> Optional[Tuple[int, int, int]]:
    """Get current visual shape."""
    return _current_visual_shape


def get_ultravico_bias(shape: Tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """
    Get cached UltraViCo bias for given shape.

    Caches the bias tensor to avoid recomputation.
    """
    global _cached_bias, _cached_shape

    config = _ultravico_config
    if config is None or not config.enabled:
        return None

    # Check if we can use cached bias
    if _cached_bias is not None and _cached_shape == shape:
        if _cached_bias.device == device:
            return _cached_bias.to(dtype)

    # Create new bias
    _cached_bias = create_ultravico_bias_for_shape(shape, config, device, dtype)
    _cached_shape = shape

    return _cached_bias


def get_ultravico_bias_auto(seq_len: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """
    Get UltraViCo bias using stored visual shape.

    This is called from attention when we don't have explicit shape info.
    Returns None if UltraViCo is disabled or shape doesn't match.
    """
    if not is_ultravico_enabled():
        return None

    shape = _current_visual_shape
    if shape is None:
        return None

    # Verify sequence length matches shape
    expected_len = shape[0] * shape[1] * shape[2]
    if seq_len != expected_len:
        # Shape mismatch - might be text attention or partial sequence
        return None

    return get_ultravico_bias(shape, device, dtype)


def identify_harmonic_period(
    base: float = 10000.0,
    dim: int = 16,
    training_frames: int = 21
) -> Tuple[int, int]:
    """
    Identify the intrinsic frequency and its period for RoPE.

    Based on RIFLEx paper Eq. (4) and (7).

    Args:
        base: RoPE base frequency (theta)
        dim: Dimension of temporal RoPE
        training_frames: Training window in latent frames

    Returns:
        Tuple of (k, period) where k is frequency index and period is in frames
    """
    periods = []
    for j in range(dim // 2):
        theta_j = 1.0 / (base ** (2 * j / dim))
        period_j = int(round(2 * math.pi / theta_j))
        periods.append(period_j)

    # Find frequency with period closest to training_frames
    diffs = [abs(p - training_frames) for p in periods]
    k = diffs.index(min(diffs))

    return k, periods[k]


def clear_ultravico_cache():
    """Clear the cached bias tensor."""
    global _cached_bias, _cached_shape, _current_visual_shape
    _cached_bias = None
    _cached_shape = None
    _current_visual_shape = None


# Sage UltraViCo functions (for memory-efficient Triton kernel path)

def set_sage_ultravico_config(
    enabled: bool,
    multi_factor: float = 0.9,
    frame_tokens: int = 1560,
    training_frames: int = 21
):
    """
    Set configuration for sage_ultravico attention mode.

    This is the memory-efficient path that uses Triton kernels with inline decay
    instead of materializing a full attention bias matrix.

    Args:
        enabled: Whether sage_ultravico is active
        multi_factor: UltraViCo decay factor (alpha), typically 0.85-0.95
        frame_tokens: Tokens per latent frame (resolution-dependent)
            - 720x1280: 3600
            - 480x854: ~1605
        training_frames: Training window in latent frames (default: 21 for Wan)
    """
    global _sage_ultravico_enabled, _sage_multi_factor, _sage_frame_tokens, _sage_training_frames
    _sage_ultravico_enabled = enabled
    _sage_multi_factor = multi_factor
    _sage_frame_tokens = frame_tokens
    _sage_training_frames = training_frames


def is_sage_ultravico_enabled() -> bool:
    """Check if sage_ultravico mode is enabled."""
    return _sage_ultravico_enabled


def get_sage_ultravico_params() -> Tuple[float, int, int]:
    """
    Get sage_ultravico parameters.

    Returns:
        Tuple of (multi_factor, frame_tokens, training_frames)
    """
    return _sage_multi_factor, _sage_frame_tokens, _sage_training_frames


def calculate_frame_tokens(height: int, width: int, vae_stride: Tuple[int, int, int] = (4, 8, 8), patch_size: Tuple[int, int, int] = (1, 2, 2)) -> int:
    """
    Calculate frame_tokens from video resolution.

    Args:
        height: Video height in pixels
        width: Video width in pixels
        vae_stride: VAE temporal/spatial stride (default: Wan2.2 values)
        patch_size: Patch size for patchification

    Returns:
        Number of tokens per latent frame
    """
    lat_h = height // vae_stride[1]
    lat_w = width // vae_stride[2]
    frame_tokens = (lat_h * lat_w) // (patch_size[1] * patch_size[2])
    return frame_tokens
