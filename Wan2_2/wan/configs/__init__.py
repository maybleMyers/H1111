# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from easydict import EasyDict

from .wan_i2v_A14B import i2v_A14B
from .wan_t2v_A14B import t2v_A14B
from .wan_ti2v_5B import ti2v_5B

# LongCat configuration (based on Wan 2.1)
longcat_t2v_13_6B = EasyDict(__name__='Config: LongCat T2V 13.6B')
longcat_t2v_13_6B.in_channels = 16
longcat_t2v_13_6B.out_channels = 16
longcat_t2v_13_6B.hidden_size = 4096
longcat_t2v_13_6B.depth = 48
longcat_t2v_13_6B.num_heads = 32
longcat_t2v_13_6B.mlp_ratio = 4
longcat_t2v_13_6B.patch_size = (1, 2, 2)
longcat_t2v_13_6B.vae_type = 'wan2.1'  # Uses Wan 2.1 VAE (AutoencoderKLWan)
longcat_t2v_13_6B.vae_stride = (4, 8, 8)  # Temporal, Height, Width strides
longcat_t2v_13_6B.vae_checkpoint = 'vae'  # Subfolder name in checkpoint dir
longcat_t2v_13_6B.text_encoder = 'umt5-xxl'  # UMT5-XXL instead of T5-XXL
longcat_t2v_13_6B.max_seq_length = 512
longcat_t2v_13_6B.vae_scale_factor_temporal = 4
longcat_t2v_13_6B.vae_scale_factor_spatial = 8
longcat_t2v_13_6B.default_frames = 93
longcat_t2v_13_6B.fps = 15  # 15fps for 480p base generation
longcat_t2v_13_6B.default_height = 480
longcat_t2v_13_6B.default_width = 832
longcat_t2v_13_6B.is_fun_control = False  # LongCat doesn't support FunControl
longcat_t2v_13_6B.i2v = False  # This is T2V only for now

LONGCAT_CONFIG = {
    'longcat-t2v-13.6B': longcat_t2v_13_6B
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
