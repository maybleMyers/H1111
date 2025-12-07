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

# --- Parameters for WanModel initialization ---
longcat_t2v_13_6B.dim = 4096                  # RENAMED from hidden_size
longcat_t2v_13_6B.num_layers = 48             # RENAMED from depth
longcat_t2v_13_6B.ffn_dim = 4096 * 4          # CALCULATED from hidden_size * mlp_ratio
longcat_t2v_13_6B.in_dim = 16                 # RENAMED from in_channels
longcat_t2v_13_6B.out_dim = 16                # RENAMED from out_channels
longcat_t2v_13_6B.text_len = 512              # RENAMED from max_seq_length
longcat_t2v_13_6B.freq_dim = 256              # ADDED missing parameter
longcat_t2v_13_6B.eps = 1e-6                  # ADDED missing parameter
longcat_t2v_13_6B.num_train_timesteps = 1000  # ADDED missing parameter for scheduler
longcat_t2v_13_6B.num_heads = 32              # This one was correct
longcat_t2v_13_6B.patch_size = (1, 2, 2)      # This one was correct

# --- Other required config values ---
longcat_t2v_13_6B.vae_stride = (4, 8, 8)
longcat_t2v_13_6B.is_fun_control = False
longcat_t2v_13_6B.i2v = False

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
