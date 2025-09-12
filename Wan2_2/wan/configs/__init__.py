# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_i2v_A14B import i2v_A14B
from .wan_t2v_A14B import t2v_A14B
from .wan_ti2v_5B import ti2v_5B
from .wan_vace_A14B import vace_t2v_A14B, vace_i2v_A14B, vace_v2v_A14B, vace_s2v_A14B

WAN_CONFIGS = {
    't2v-A14B': t2v_A14B,
    'i2v-A14B': i2v_A14B,
    'ti2v-5B': ti2v_5B,
    # VACE configurations
    'vace-t2v-A14B': vace_t2v_A14B,
    'vace-i2v-A14B': vace_i2v_A14B,
    'vace-v2v-A14B': vace_v2v_A14B,
    'vace-s2v-A14B': vace_s2v_A14B,
}

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
    # VACE models support same sizes as base A14B
    'vace-t2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
    'vace-i2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
    'vace-v2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
    'vace-s2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
}
