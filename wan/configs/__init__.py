# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import os
import torch
from easydict import EasyDict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from .shared_config import wan_shared_cfg
from .wan_i2v_14B import i2v_14B
from .wan_t2v_1_3B import t2v_1_3B
from .wan_t2v_14B import t2v_14B

# the config of t2i_14B is the same as t2v_14B
t2i_14B = copy.deepcopy(t2v_14B)
t2i_14B.__name__ = "Config: Wan T2I 14B"

# ================== START: Add New 1.3B I2V Model Config ==================
i2v_1_3B_new = EasyDict(__name__="Config: Wan I2V 1.3B New")
i2v_1_3B_new.update(wan_shared_cfg) # Start with shared defaults

# --- Core Model Parameters from your config.json ---
i2v_1_3B_new.dim = 1536
i2v_1_3B_new.ffn_dim = 8960
i2v_1_3B_new.num_heads = 12
i2v_1_3B_new.num_layers = 30
i2v_1_3B_new.in_dim = 36  # From config.json (latent + mask)
i2v_1_3B_new.out_dim = 16 # From config.json
i2v_1_3B_new.freq_dim = 256 # From config.json
i2v_1_3B_new.text_len = 512 # From config.json
i2v_1_3B_new.eps = 1e-06 # From config.json

# --- I2V Specific Settings ---
i2v_1_3B_new.i2v = True # Mark as I2V
i2v_1_3B_new.is_fun_control = False # This is NOT a FunControl model

# --- Assumed Component Checkpoints & Settings (ADJUST IF NEEDED) ---
# Assume it uses the same components as other models unless specified
# DiT: User MUST provide this path via --dit
# VAE: Assume standard VAE, user can override with --vae
i2v_1_3B_new.vae_checkpoint = "Wan2.1_VAE.pth" # Or specific VAE if different
i2v_1_3B_new.vae_stride = (4, 8, 8) # Standard stride

# T5: Assume standard T5, user can override with --t5
i2v_1_3B_new.t5_checkpoint = "models_t5_umt5-xxl-enc-bf16.pth" # Or smaller T5 if available
i2v_1_3B_new.t5_tokenizer = "google/umt5-xxl"
i2v_1_3B_new.t5_dtype = torch.bfloat16 # Default T5 dtype

# CLIP: Needed for I2V, assume standard CLIP, user can override with --clip
i2v_1_3B_new.clip_model = "clip_xlm_roberta_vit_h_14"
i2v_1_3B_new.clip_dtype = torch.float16 # Default CLIP dtype
i2v_1_3B_new.clip_checkpoint = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
i2v_1_3B_new.clip_tokenizer = "xlm-roberta-large"

# Transformer structure (Assume standard based on WanModel)
i2v_1_3B_new.patch_size = (1, 2, 2) # Standard patch size
i2v_1_3B_new.window_size = (-1, -1) # Global attention
i2v_1_3B_new.qk_norm = True # Standard norm
i2v_1_3B_new.cross_attn_norm = True # Standard norm (often True for I2V)

# Default sample prompts (can be kept or adjusted)
i2v_1_3B_new.sample_prompts = ["cinematic video of a sports car"]
i2v_1_3B_new.sample_neg_prompt = "text, watermark, copyright, blurry, low quality, noisy"
i2v_1_3B_new.num_train_timesteps = 1000 # Standard diffusion timesteps

# ================== END: Add New 1.3B I2V Model Config ==================

# support Fun models: deepcopy and change some configs. FC denotes Fun Control
t2v_1_3B_FC = copy.deepcopy(t2v_1_3B)
t2v_1_3B_FC.__name__ = "Config: Wan-Fun-Control T2V 1.3B"
t2v_1_3B_FC.in_dim = 48
i2v_14B.is_fun_control = False
t2v_14B_FC = copy.deepcopy(t2v_14B)
t2v_14B_FC.__name__ = "Config: Wan-Fun-Control T2V 14B"
t2v_14B_FC.i2v = True  # this is strange, but Fun-Control model needs this because it has img cross-attention
t2v_14B_FC.in_dim = 48  # same as i2v_14B, use zeros for image latents
t2v_14B_FC.is_fun_control = True
i2v_14B_FC = copy.deepcopy(i2v_14B)
i2v_14B_FC.__name__ = "Config: Wan-Fun-Control I2V 14B"
i2v_14B_FC.in_dim = 48
i2v_14B_FC.is_fun_control = True

i2v_14B_FC_1_1 = copy.deepcopy(i2v_14B_FC) # Copy the existing FunControl I2V 14B config
i2v_14B_FC_1_1.__name__ = "Config: Wan-Fun-Control I2V 14B v1.1"
# Explicitly add the flag for clarity, though loading logic will derive it
# i2v_14B_FC_1_1.add_ref_conv = True # This flag isn't directly used by the Python config struct, but good for documentation
# The key is that the loaded weights for this model WILL contain 'ref_conv.weight'
# All other parameters are inherited from i2v_14B_FC (in_dim=48, is_fun_control=True, etc.)

WAN_CONFIGS = {
    "t2v-14B": t2v_14B,
    "t2v-1.3B": t2v_1_3B,
    "i2v-14B": i2v_14B,
    "t2i-14B": t2i_14B,
    "i2v-1.3B-new": i2v_1_3B_new,
    # Fun Control models
    "t2v-1.3B-FC": t2v_1_3B_FC,
    "t2v-14B-FC": t2v_14B_FC,
    "i2v-14B-FC": i2v_14B_FC,
    "i2v-14B-FC-1.1": i2v_14B_FC_1_1,
}

SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
    "1024*1024": (1024, 1024),
    "512*512": (512, 512),     # <--- Example: Added 512x512 if used
    "672*352": (672, 352),     # <--- Added from your command line example
    "352*672": (352, 672),     # <--- Added from your command line example (vertical)
}
# --- ^^^ MODIFY THIS DICTIONARY ^^^ ---


# --- vvv MODIFY THIS DICTIONARY vvv ---
MAX_AREA_CONFIGS = {
    "720*1280": 720 * 1280,
    "1280*720": 1280 * 720,
    "480*832": 480 * 832,
    "832*480": 832 * 480,
    "1024*1024": 1024 * 1024,
    "512*512": 512 * 512,         # <--- Added 512x512 if used
    "672*352": 672 * 352,         # <--- Added from your command line example
    "352*672": 352 * 672,         # <--- Added from your command line example (vertical)
}
# --- ^^^ MODIFY THIS DICTIONARY ^^^ ---


# --- vvv MODIFY THIS DICTIONARY vvv ---
SUPPORTED_SIZES = {
    "t2v-14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "t2v-1.3B": ("480*832", "832*480"),
    "i2v-14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "t2i-14B": tuple(SIZE_CONFIGS.keys()),
    # Fun Control models
    "t2v-1.3B-FC": ("480*832", "832*480"),
    "t2v-14B-FC": ("720*1280", "1280*720", "480*832", "832*480"),
    "i2v-14B-FC": ("720*1280", "1280*720", "480*832", "832*480"),
    "i2v-14B-FC-1.1": ("720*1280", "1280*720", "480*832", "832*480"),
    # Add supported sizes for the new model
    "i2v-1.3B-new": ("480*832", "832*480", "512*512", "672*352", "352*672"), 
    
}
