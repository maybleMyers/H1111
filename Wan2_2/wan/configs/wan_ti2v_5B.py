# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan TI2V 5B ------------------------#

ti2v_5B = EasyDict(__name__='Config: Wan TI2V 5B')
ti2v_5B.update(wan_shared_cfg)

# Model type identification
ti2v_5B.i2v = False  # This is a TI2V model (text+image to video, not pure I2V)

# t5
ti2v_5B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
ti2v_5B.t5_tokenizer = 'google/umt5-xxl'

# vae
ti2v_5B.vae_checkpoint = 'Wan2.2_VAE.pth'
ti2v_5B.vae_stride = (4, 16, 16)

# transformer
ti2v_5B.patch_size = (1, 2, 2)
ti2v_5B.dim = 3072
ti2v_5B.ffn_dim = 14336
ti2v_5B.freq_dim = 256
ti2v_5B.num_heads = 24
ti2v_5B.num_layers = 30
ti2v_5B.window_size = (-1, -1)
ti2v_5B.qk_norm = True
ti2v_5B.cross_attn_norm = True
ti2v_5B.eps = 1e-6
ti2v_5B.in_channels = 48  # TI2V 5B model uses 48 input channels (VAE latent + image conditioning)
ti2v_5B.out_channels = 48  # Output latent channels to match VAE expectations (head outputs 192 = 48*4)  
ti2v_5B.in_dim = 48  # Same as in_channels for compatibility
ti2v_5B.out_dim = 48  # Same as out_channels for compatibility

# inference
ti2v_5B.sample_fps = 24
ti2v_5B.sample_shift = 5.0
ti2v_5B.sample_steps = 50
ti2v_5B.sample_guide_scale = 5.0
ti2v_5B.frame_num = 121
