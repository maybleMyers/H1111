# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# HuMo 17B Model Configuration - Audio-driven talking head synthesis
# Based on Wan 2.2 A14B architecture with audio cross-attention

import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ HuMo 17B TIA (Text+Image+Audio) ------------------------#

humo_17B_TIA = EasyDict(__name__='Config: HuMo 17B TIA')
humo_17B_TIA.update(wan_shared_cfg)

# Model type identification
humo_17B_TIA.i2v = False  # Uses input channel conditioning like Wan 2.2 i2v
humo_17B_TIA.humo = True  # HuMo model flag
humo_17B_TIA.humo_mode = 'TIA'  # Text + Image + Audio mode

# t5 (same as Wan 2.2)
humo_17B_TIA.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
humo_17B_TIA.t5_tokenizer = 'google/umt5-xxl'

# clip - Not used in HuMo, conditioning is via audio
humo_17B_TIA.clip_model = None
humo_17B_TIA.clip_dtype = None
humo_17B_TIA.clip_checkpoint = None
humo_17B_TIA.clip_tokenizer = None

# vae (same as Wan 2.2)
humo_17B_TIA.vae_checkpoint = 'Wan2.1_VAE.pth'
humo_17B_TIA.vae_stride = (4, 8, 8)

# transformer (HuMo 17B architecture - same as Wan 2.2 A14B with audio)
humo_17B_TIA.patch_size = (1, 2, 2)
humo_17B_TIA.dim = 5120  # Same as Wan 2.2 A14B
humo_17B_TIA.ffn_dim = 13824  # Same as Wan 2.2 A14B
humo_17B_TIA.freq_dim = 256
humo_17B_TIA.num_heads = 40
humo_17B_TIA.num_layers = 40
humo_17B_TIA.window_size = (-1, -1)
humo_17B_TIA.qk_norm = True
humo_17B_TIA.cross_attn_norm = True
humo_17B_TIA.eps = 1e-6
humo_17B_TIA.in_channels = 36  # mask(4) + latent(16) + ref_latent(16)
humo_17B_TIA.out_channels = 16
humo_17B_TIA.in_dim = 36
humo_17B_TIA.out_dim = 16

# Audio settings
humo_17B_TIA.audio_token_num = 16  # Audio tokens per frame
humo_17B_TIA.insert_audio = True  # Enable audio cross-attention

# inference (HuMo specific)
humo_17B_TIA.sample_shift = 5.0
humo_17B_TIA.sample_steps = 50  # HuMo default
humo_17B_TIA.sample_guide_scale = 5.0  # Single model, not dual-dit
humo_17B_TIA.scale_a = 5.5  # Audio guidance scale
humo_17B_TIA.scale_t = 5.0  # Text guidance scale
humo_17B_TIA.step_change = 980  # Timestep to change CFG formula

# Additional required attributes
humo_17B_TIA.text_len = 512  # Max text token length
humo_17B_TIA.is_fun_control = False  # Not FunControl model
humo_17B_TIA.sample_neg_prompt = ""  # Default negative prompt

# Zero VAE paths (for conditioning)
humo_17B_TIA.zero_vae_480p = 'zero_vae_129frame.pt'
humo_17B_TIA.zero_vae_720p = 'zero_vae_720p_161frame.pt'


#------------------------ HuMo 17B TA (Text+Audio) ------------------------#

humo_17B_TA = EasyDict(__name__='Config: HuMo 17B TA')
humo_17B_TA.update(humo_17B_TIA)  # Start from TIA config

# Override mode
humo_17B_TA.humo_mode = 'TA'  # Text + Audio mode (no image)
humo_17B_TA.in_channels = 36  # Still same architecture
humo_17B_TA.in_dim = 36
