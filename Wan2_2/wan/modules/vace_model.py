"""VACE Model Components for Wan 2.2

This module provides VACE-specific model components including:
- VaceWanAttentionBlock: Extended attention block with control input processing
- adapt_vace_model: Function to map VACE blocks to transformer layers
- load_vace_model: Function to load VACE weights
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from safetensors.torch import load_file
import logging

from wan.modules.model import WanAttentionBlock

logger = logging.getLogger(__name__)


class VaceWanAttentionBlock(WanAttentionBlock):
    """VACE-specific attention block that extends WanAttentionBlock with control processing"""

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        block_id: int = 0
    ):
        super().__init__(
            cross_attn_type, dim, ffn_dim, num_heads,
            window_size, qk_norm, cross_attn_norm, eps
        )
        self.block_id = block_id

        # Special initialization for first block
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        # After projection for all VACE blocks
        self.after_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, hints: List[Optional[torch.Tensor]], x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with control hint processing

        Args:
            hints: List containing control tensor at index 0
            x: Input tensor
            **kwargs: Additional arguments for parent forward

        Returns:
            Processed tensor
        """
        # Extract control hint
        c = hints[0]
        hints[0] = None  # Clear to avoid reprocessing

        if self.block_id == 0 and c is not None:
            c = self.before_proj(c)
            bz = x.shape[0]
            # Repeat control if batch size mismatch
            if bz > c.shape[0]:
                c = c.repeat(bz, 1, 1)
            x = x + c
        elif c is not None:
            x = x + c

        # Process through parent attention block
        x = super().forward(x, **kwargs)

        # Apply after projection
        if c is not None:
            x = x + self.after_proj(c)

        # Store processed control for next layer
        hints[0] = x if c is not None else None

        return x


def adapt_vace_model(model: nn.Module) -> None:
    """Adapt a Wan model to support VACE by mapping VACE blocks to transformer layers

    Args:
        model: The Wan model to adapt
    """
    if not hasattr(model, 'vace_layers_mapping'):
        logger.warning("Model does not have vace_layers_mapping, skipping adaptation")
        return

    modules_dict = {k: m for k, m in model.named_modules()}

    # Map VACE blocks to their corresponding transformer layers
    for model_layer, vace_layer in model.vace_layers_mapping.items():
        vace_block_name = f"vace_blocks.{vace_layer}"
        target_block_name = f"blocks.{model_layer}"

        if vace_block_name in modules_dict and target_block_name in modules_dict:
            vace_module = modules_dict[vace_block_name]
            target = modules_dict[target_block_name]
            setattr(target, "vace", vace_module)
            logger.info(f"Mapped VACE block {vace_layer} to transformer layer {model_layer}")
        else:
            logger.warning(f"Could not map VACE block {vace_block_name} to {target_block_name}")

    # Remove the separate vace_blocks module to save memory
    if hasattr(model, "vace_blocks"):
        delattr(model, "vace_blocks")
        logger.info("Removed vace_blocks module after mapping")


def load_vace_weights(model: nn.Module, vace_weights_path: str) -> None:
    """Load VACE-specific weights into the model

    Args:
        model: The Wan model with VACE adaptation
        vace_weights_path: Path to the VACE weights safetensors file
    """
    logger.info(f"Loading VACE weights from {vace_weights_path}")

    try:
        vace_weights = load_file(vace_weights_path)

        # Filter and load VACE-specific weights
        vace_state_dict = {}
        for key, value in vace_weights.items():
            if "vace" in key or "before_proj" in key or "after_proj" in key:
                vace_state_dict[key] = value

        # Load the weights
        missing_keys, unexpected_keys = model.load_state_dict(vace_state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys when loading VACE weights: {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading VACE weights: {unexpected_keys[:5]}...")

        logger.info(f"Successfully loaded {len(vace_state_dict)} VACE weight tensors")

    except Exception as e:
        logger.error(f"Failed to load VACE weights: {e}")
        raise


def create_vace_patch_embedding(in_dim: int, out_dim: int, patch_size: Tuple[int, int, int]) -> nn.Module:
    """Create VACE-specific patch embedding layer for 96-channel input

    Args:
        in_dim: Input dimension (96 for VACE)
        out_dim: Output dimension
        patch_size: Patch size tuple

    Returns:
        Conv3d patch embedding layer
    """
    patch_embedding = nn.Conv3d(
        in_dim, out_dim,
        kernel_size=patch_size,
        stride=patch_size
    )
    # Initialize with small values for stability
    nn.init.xavier_uniform_(patch_embedding.weight, gain=0.02)
    if patch_embedding.bias is not None:
        nn.init.zeros_(patch_embedding.bias)

    return patch_embedding


def initialize_vace_blocks(
    num_layers: int,
    vace_layers: List[int],
    dim: int,
    ffn_dim: int,
    num_heads: int,
    window_size: Tuple[int, int] = (-1, -1),
    qk_norm: bool = True,
    cross_attn_norm: bool = False,
    eps: float = 1e-6
) -> nn.ModuleList:
    """Initialize VACE control blocks

    Args:
        num_layers: Total number of transformer layers
        vace_layers: List of layer indices that will have VACE control
        dim: Model dimension
        ffn_dim: FFN dimension
        num_heads: Number of attention heads
        window_size: Attention window size
        qk_norm: Whether to use QK normalization
        cross_attn_norm: Whether to use cross attention normalization
        eps: Epsilon for layer normalization

    Returns:
        ModuleList of VACE blocks
    """
    vace_blocks = nn.ModuleList([
        VaceWanAttentionBlock(
            't2v_cross_attn', dim, ffn_dim, num_heads,
            window_size, qk_norm, cross_attn_norm, eps,
            block_id=i
        )
        for i in range(len(vace_layers))
    ])

    return vace_blocks