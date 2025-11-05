# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan I2V A14B ------------------------#

i2v_A14B = EasyDict(__name__='Config: Wan I2V A14B')
i2v_A14B.update(wan_shared_cfg)

# Model type identification - CORRECTED based on layer analysis
i2v_A14B.i2v = False  # No separate image embedding layers, uses input channel conditioning

i2v_A14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
i2v_A14B.t5_tokenizer = 'google/umt5-xxl'

# clip (required for I2V models)
i2v_A14B.clip_model = "clip_xlm_roberta_vit_h_14"
i2v_A14B.clip_dtype = torch.float16
i2v_A14B.clip_checkpoint = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
i2v_A14B.clip_tokenizer = "xlm-roberta-large"

# vae
i2v_A14B.vae_checkpoint = 'Wan2.1_VAE.pth'
i2v_A14B.vae_stride = (4, 8, 8)

# transformer
i2v_A14B.patch_size = (1, 2, 2)
i2v_A14B.dim = 5120
i2v_A14B.ffn_dim = 13824
i2v_A14B.freq_dim = 256
i2v_A14B.num_heads = 40
i2v_A14B.num_layers = 40
i2v_A14B.window_size = (-1, -1)
i2v_A14B.qk_norm = True
i2v_A14B.cross_attn_norm = True
i2v_A14B.eps = 1e-6
i2v_A14B.low_noise_checkpoint = 'low_noise_model'
i2v_A14B.high_noise_checkpoint = 'high_noise_model'
i2v_A14B.in_channels = 36  # Confirmed by model weights: 16 latent + 16 image + 4 mask
i2v_A14B.out_channels = 16  # Output latent channels
i2v_A14B.in_dim = 36  # Same as in_channels for compatibility
i2v_A14B.out_dim = 16  # Same as out_channels for compatibility

# inference
i2v_A14B.sample_shift = 5.0
i2v_A14B.sample_steps = 40
i2v_A14B.boundary = 0.900
i2v_A14B.sample_guide_scale = (3.5, 3.5)  # low noise, high noise
