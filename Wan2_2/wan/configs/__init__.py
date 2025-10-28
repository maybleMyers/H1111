# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_i2v_A14B import i2v_A14B
from .wan_t2v_A14B import t2v_A14B
from .wan_ti2v_5B import ti2v_5B

# LongCat configuration (based on Wan 2.1)
LONGCAT_CONFIG = {
    'longcat-t2v-13.6B': {
        'in_channels': 16,
        'out_channels': 16,
        'hidden_size': 4096,
        'depth': 48,
        'num_heads': 32,
        'mlp_ratio': 4,
        'patch_size': (1, 2, 2),
        'vae_type': 'wan2.1',  # Uses Wan 2.1 VAE (AutoencoderKLWan)
        'vae_stride': (4, 8, 8),  # Temporal, Height, Width strides
        'text_encoder': 'umt5-xxl',  # UMT5-XXL instead of T5-XXL
        'max_seq_length': 512,
        'vae_scale_factor_temporal': 4,
        'vae_scale_factor_spatial': 8,
        'default_frames': 93,
        'fps': 15,  # 15fps for 480p base generation
        'default_height': 480,
        'default_width': 832,
        'is_fun_control': False,  # LongCat doesn't support FunControl
        'i2v': False,  # This is T2V only for now
    }
}

WAN_CONFIGS = {
    't2v-A14B': t2v_A14B,
    'i2v-A14B': i2v_A14B,
    'ti2v-5B': ti2v_5B,
}

# Merge LongCat configs into WAN_CONFIGS
WAN_CONFIGS.update(LONGCAT_CONFIG)

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '704*1280': (704, 1280),
    '1280*704': (1280, 704)
}

MAX_AREA_CONFIGS = {
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
    '704*1280': 704 * 1280,
    '1280*704': 1280 * 704,
}

SUPPORTED_SIZES = {
    't2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
    'i2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
    'ti2v-5B': ('704*1280', '1280*704'),
    'longcat-t2v-13.6B': ('480*832', '832*480'),  # LongCat base resolution
}
