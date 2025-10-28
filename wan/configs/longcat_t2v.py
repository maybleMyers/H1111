# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import wan_shared_cfg

# ------------------------ LongCat T2V ------------------------#

longcat_t2v = EasyDict(__name__="Config: LongCat T2V")
longcat_t2v.update(wan_shared_cfg)
longcat_t2v.i2v = False
longcat_t2v.is_fun_control = False

# t5
longcat_t2v.t5_checkpoint = "models_t5_umt5-xxl-enc-bf16.pth"
longcat_t2v.t5_tokenizer = "google/umt5-xxl"
longcat_t2v.text_len = 512

# vae (uses Wan 2.1 VAE)
longcat_t2v.vae_checkpoint = "Wan2.1_VAE.pth"
longcat_t2v.vae_stride = (4, 8, 8)

# transformer (from LongCat dit/config.json)
longcat_t2v.patch_size = (1, 2, 2)
longcat_t2v.dim = 4096  # hidden_size
longcat_t2v.ffn_dim = 16384  # hidden_size * mlp_ratio (4096 * 4)
longcat_t2v.freq_dim = 256  # frequency_embedding_size
longcat_t2v.in_dim = 16  # in_channels
longcat_t2v.num_heads = 32
longcat_t2v.num_layers = 48  # depth
longcat_t2v.window_size = (-1, -1)
longcat_t2v.qk_norm = True
longcat_t2v.cross_attn_norm = True
longcat_t2v.eps = 1e-6
