# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# VACE (Visual Attention Context Enhancement) configuration for Wan 2.2 14B models
from easydict import EasyDict
import torch

vace_A14B = EasyDict({
    # Model architecture
    'model_type': 'vace',
    'pretrained_model_path': None,
    'dit_model_name': 'WanModel',
    'dit_hidden_size': 5120,
    'dit_num_heads': 40,
    'dit_depth': 40,
    'dit_patch_size': 2,
    'dit_learn_sigma': True,
    'dit_enable_flashattn': False,
    'dit_mixed_precision': False,
    'dit_dtype': torch.bfloat16,
    'dit_use_fp8': False,
    'dit_use_fp8_scaled': False,
    
    # VACE-specific parameters
    'vace_layers': [0, 5, 10, 15, 20, 25, 30, 35],  # From config-low.json
    'vace_in_dim': 96,
    'vace_context_scale': 1.0,
    'model_class': 'VaceWanModel',
    
    # VAE
    'vae_type': 'AutoencoderKLWan',
    'vae_checkpoint': 'Wan2.1_VAE.pth',
    'vae_dtype': torch.bfloat16,
    'vae_path': None,
    'patch_size': 2,
    'latent_channels': 16,
    'temporal_compression_ratio': 4,
    
    # Text encoders
    't5_checkpoint': 'models_t5_umt5-xxl-enc-bf16.pth',
    't5_tokenizer': 'google/umt5-xxl',
    't5_dtype': torch.bfloat16,
    't5_path': None,
    
    # Model checkpoints
    'low_noise_checkpoint': 'low_noise_model',
    'high_noise_checkpoint': 'high_noise_model',
    
    # CLIP for image conditioning
    'clip_checkpoint': 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
    'clip_tokenizer': None,
    'clip_dtype': torch.bfloat16,
    'clip_path': None,
    
    # Dual-DiT configuration
    'transformer_combination_type': 'moe',  # mixture of experts for low/high noise
    'transformer_low_noise_model_subpath': 'transformer',
    'transformer_high_noise_model_subpath': 'transformer_high',
    'dual_dit_boundary': 0.875,  # Switch point between low and high noise models
    
    # Default generation params
    'default_steps': 50,
    'default_cfg': 7.0,
    'default_flow_shift': 5.0,
    'use_linear_quadratic_schedule': False,
    'schedule_shift': 1.0,
    
    # Memory optimization
    'enable_teacache': True,
    'teacache_threshold': 0.10,
    'enable_riflex': False,
    'riflex_k': 6,
    'cfg_skip_ratio': 0.0,
    
    # Model dimensions (matching A14B model)
    'dim': 5120,
    'in_dim': 16,  # From config-low.json
    'out_dim': 16,
    'ffn_dim': 13824,
    'freq_dim': 256,
    'num_heads': 40,
    'num_layers': 40,
    'text_len': 512,
    'eps': 1e-06,
    'cross_attn_norm': True,
    'in_channels': 16,
    'out_channels': 16,
    
    # Additional transformer kwargs
    'transformer_additional_kwargs': {
        'model_type': 'vace',
        'boundary': 0.875,
        'transformer_combination_type': 'moe',
        'transformer_low_noise_model_subpath': 'transformer',
        'transformer_high_noise_model_subpath': 'transformer_high',
        'vace_layers': [0, 5, 10, 15, 20, 25, 30, 35],
        'vace_in_dim': 96,
    },
    
    # VAE kwargs
    'vae_kwargs': {
        'vae_type': 'AutoencoderKLWan',
        'vae_subpath': 'vae',
        'temporal_compression_ratio': 4,
    },
    
    # Text encoder kwargs
    'text_encoder_kwargs': {
        'text_encoder_subpath': 'text_encoder',
        'tokenizer_subpath': 'tokenizer',
    },
    
    # Scheduler kwargs
    'scheduler_kwargs': {
        'shift': 5.0,
        'use_dynamic_shifting': False,
    },
    
    # Required attributes for compatibility
    'is_fun_control': False,  # Not a Fun-Control model
    'i2v': False,  # Base is T2V
    'temporal_compression_ratio': 4,
    'vae_stride': (4, 8, 8),
    'patch_size': (1, 2, 2),
    'qk_norm': True,
    'window_size': (-1, -1),
    
    # Scheduler configuration
    'num_train_timesteps': 1000,
    'sample_fps': 16,
    'frame_num': 81,
    'text_len': 512,
    
    # Boundary for dual-dit switching
    'boundary': 0.875,
})

# Create variations for different VACE modes
# Ensure all configs are EasyDict objects for attribute access
vace_t2v_A14B = EasyDict(vace_A14B.copy())
vace_t2v_A14B['mode'] = 't2v'
vace_t2v_A14B['i2v'] = False

vace_i2v_A14B = EasyDict(vace_A14B.copy())
vace_i2v_A14B['mode'] = 'i2v'
vace_i2v_A14B['i2v'] = True
vace_i2v_A14B['dual_dit_boundary'] = 0.900  # Different boundary for I2V

vace_v2v_A14B = EasyDict(vace_A14B.copy())
vace_v2v_A14B['mode'] = 'v2v'
vace_v2v_A14B['i2v'] = False

vace_s2v_A14B = EasyDict(vace_A14B.copy())
vace_s2v_A14B['mode'] = 's2v'
vace_s2v_A14B['i2v'] = True  # S2V uses image conditioning