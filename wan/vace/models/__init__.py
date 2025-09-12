"""VACE model components."""

from .transformer_vace import VaceWanTransformer3DModel, VaceWanAttentionBlock, BaseWanAttentionBlock
from .vae_vace import AutoencoderKLWan, AutoencoderKLWan_
from .vae3_8 import AutoencoderKLWan3_8, AutoencoderKLWan2_2_
from .text_encoder import WanT5EncoderModel
from .cache_utils import get_teacache_coefficients

# For compatibility with transformers AutoTokenizer
from transformers import AutoTokenizer

__all__ = [
    'VaceWanTransformer3DModel',
    'VaceWanAttentionBlock',
    'BaseWanAttentionBlock',
    'AutoencoderKLWan',
    'AutoencoderKLWan_',
    'AutoencoderKLWan3_8',
    'AutoencoderKLWan2_2_',
    'WanT5EncoderModel',
    'get_teacache_coefficients',
    'AutoTokenizer'
]