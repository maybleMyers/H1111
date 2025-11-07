# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan T2V A14B ------------------------#

t2v_A14B = EasyDict(__name__='Config: Wan T2V A14B')
t2v_A14B.update(wan_shared_cfg)

# Model type identification
t2v_A14B.i2v = False  # This is a T2V model

# t5
t2v_A14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
t2v_A14B.t5_tokenizer = 'google/umt5-xxl'

# vae
t2v_A14B.vae_checkpoint = 'Wan2.1_VAE.pth'
t2v_A14B.vae_stride = (4, 8, 8)

# transformer
t2v_A14B.patch_size = (1, 2, 2)
t2v_A14B.dim = 5120
t2v_A14B.ffn_dim = 13824
t2v_A14B.freq_dim = 256
t2v_A14B.num_heads = 40
t2v_A14B.num_layers = 40
t2v_A14B.window_size = (-1, -1)
t2v_A14B.qk_norm = True
t2v_A14B.cross_attn_norm = True
t2v_A14B.eps = 1e-6
t2v_A14B.low_noise_checkpoint = 'low_noise_model'
t2v_A14B.high_noise_checkpoint = 'high_noise_model'
t2v_A14B.in_channels = 16  # Standard latent channels for T2V (update when weights available)
t2v_A14B.out_channels = 16  # Output latent channels
t2v_A14B.in_dim = 16  # Same as in_channels for compatibility
t2v_A14B.out_dim = 16  # Same as out_channels for compatibility

# inference
t2v_A14B.sample_shift = 12.0
t2v_A14B.sample_steps = 40
t2v_A14B.boundary = 0.875
t2v_A14B.sample_guide_scale = (3.0, 4.0)  # low noise, high noise
