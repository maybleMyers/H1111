"""
SageAttention with UltraViCo support for memory-efficient attention decay.

This module provides a Triton-based INT8 quantized attention implementation
that can apply UltraViCo decay inline without materializing a full bias matrix.
"""

from .core import sage_attention

__all__ = ['sage_attention']
