# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan Animate 14B ------------------------#

animate_14B = EasyDict(__name__='Config: Wan Animate 14B')
animate_14B.update(wan_shared_cfg)

# Model type identification - Animate model for character animation/replacement
animate_14B.i2v = True  # Uses image conditioning for reference character
animate_14B.animate = True  # Special flag for animate model

# T5 text encoder
animate_14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
animate_14B.t5_tokenizer = 'google/umt5-xxl'

# CLIP model (required for reference image encoding)
animate_14B.clip_model = "clip_xlm_roberta_vit_h_14"
animate_14B.clip_dtype = torch.float16
animate_14B.clip_checkpoint = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
animate_14B.clip_tokenizer = "xlm-roberta-large"

# VAE configuration
animate_14B.vae_checkpoint = 'Wan2.1_VAE.pth'
animate_14B.vae_stride = (4, 8, 8)

# Transformer architecture (14B model)
animate_14B.patch_size = (1, 2, 2)
animate_14B.dim = 5120
animate_14B.ffn_dim = 13824
animate_14B.freq_dim = 256
animate_14B.num_heads = 40
animate_14B.num_layers = 40
animate_14B.window_size = (-1, -1)
animate_14B.qk_norm = True
animate_14B.cross_attn_norm = True
animate_14B.eps = 1e-6

# Model checkpoints
animate_14B.animate_checkpoint = 'animate_model'  # Main animate model
animate_14B.face_encoder_checkpoint = 'face_encoder.pth'  # Face encoder
animate_14B.face_adapter_checkpoint = 'face_adapter.pth'  # Face adapter
animate_14B.motion_encoder_checkpoint = 'motion_encoder.pth'  # Motion encoder

# Channel configuration for animate model
animate_14B.in_channels = 52  # 16 latent + 16 reference + 16 drive + 4 mask
animate_14B.out_channels = 16  # Output latent channels
animate_14B.in_dim = 52  # Same as in_channels for compatibility
animate_14B.out_dim = 16  # Same as out_channels for compatibility

# Inference parameters
animate_14B.sample_shift = 5.0
animate_14B.sample_steps = 40
animate_14B.boundary = 0.875  # Not used for animate but kept for compatibility
animate_14B.sample_guide_scale = (3.5, 3.5)  # Guidance scale

# Animate-specific parameters
animate_14B.refert_num = 1  # Temporal guidance frames (1 or 5)
animate_14B.clip_len = 77  # Frames per clip (must be 4n+1)
animate_14B.replace_mode = False  # Default to animation mode
animate_14B.use_relighting_lora = False  # Relighting LoRA for replacement

# Preprocessing pipeline parameters
animate_14B.preprocess = False  # Run preprocessing by default
animate_14B.retarget = False  # Pose retargeting
animate_14B.use_flux = False  # Use FLUX for advanced editing

# Default resolution support (flexible from 256 to 1280)
animate_14B.default_height = 480
animate_14B.default_width = 832
animate_14B.min_resolution = 256
animate_14B.max_resolution = 1280
animate_14B.resolution_step = 32