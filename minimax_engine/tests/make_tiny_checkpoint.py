"""Build a tiny fake MiniMax-H3 checkpoint dir (random weights, real layout) for the
dummy-weight end-to-end tests. All checkpoint-tied geometry (17n+5 chunking, 40 audio
latents/s at 32 kHz, patch (1,2,2)) is kept real; only the widths/depths are tiny.

    env/bin/python minimax_engine/tests/make_tiny_checkpoint.py <out_dir>
"""

import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_ENGINE_DIR)
for _p in (_ENGINE_DIR, _REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _env_compat  # noqa: F401,E402

import torch  # noqa: E402

TOKENIZER_DIR = os.path.join(_REPO_ROOT, "cosmos_engine", "cosmos_hf_indexes", "Cosmos3-Nano", "text_tokenizer")

TINY_TRANSFORMER_CONFIG = {
    "_class_name": "MiniMaxH3Transformer3DModel",
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "hidden_size": 24,
    "num_layers": 2,
    "num_refiner_layers": 2,
    "ffn_dim": 32,
    "in_channels": 4,  # tiny video VAE latent channels
    "audio_in_channels": 8,  # tiny audio VAE latent channels
    "patch_size": [1, 2, 2],
    "text_dim": 32,  # tiny conditioner hidden size
    "freq_dim": 8,
    "time_embed_hidden_dim": 24,
    "time_embed_dim": 16,
    "rope_freq_dim": 2,
}

# Checkpoint-tied fields (clip_length 17 / token_drop 3 -> 17n+5 frames to 5n+2 latents) real;
# widths tiny. Spatial ratio is 4 instead of 16 to keep CPU encodes cheap.
TINY_VIDEO_VAE_CONFIG = {
    "_class_name": "AutoencoderKLMiniMaxH3",
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "block_out_channels": [16, 32],
    "layers_per_block": 1,
    "spatial_downsample_factors": [2, 2],
    "temporal_downsample_factors": [2, 2],
    "norm_num_groups": 8,
    "norm_eps": 1e-06,
    "spatial_padding_mode": "reflect",
    "decoder_num_layers": 2,
    "decoder_num_attention_heads": 2,
    "decoder_attention_head_dim": 16,
    "decoder_num_register_tokens": 4,
    "decoder_ffn_mult": 2,
    "decoder_rope_theta": 100.0,
    "decoder_rope_dim_ratio": 0.75,
    "decoder_norm_eps": 1e-05,
    "clip_length": 17,
    "token_drop": 3,
    "latents_mean": [0.0] * 4,
    "latents_std": [1.0] * 4,
}

# Real hop length (2*4*4*5*5 = 800 -> 40 latents/s at 32 kHz); tiny widths.
TINY_AUDIO_VAE_CONFIG = {
    "_class_name": "AutoencoderKLMiniMaxH3Audio",
    "encoder_dim": 4,
    "encoder_rates": [2, 4, 4, 5, 5],
    "latent_dim": 32,
    "latent_channels": 8,
    "num_attention_heads": 2,
    "decoder_dim": 128,
    "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
    "decoder_kernel_sizes": [9, 9, 4, 4, 4, 4, 4],
    "resblock_kernel_sizes": [3, 7],
    "resblock_dilation_sizes": [[1, 3], [1, 3]],
    "sampling_rate": 32000,
    "latents_mean": [0.0] * 8,
    "latents_std": [1.0] * 8,
}

TINY_TEXT_CONFIG = {
    "vocab_size": 151936,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 52,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    "attention_bias": False,
    "hidden_act": "silu",
}
TINY_VISION_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 4,
    "depth": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "in_channels": 3,
    "out_hidden_size": 32,
    "num_position_embeddings": 16,
    "deepstack_visual_indexes": [0, 1],
}


def _save(module: torch.nn.Module, folder: str, config: dict, key_prefix: str = "") -> None:
    import safetensors.torch

    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    state = {key_prefix + k: v.to(torch.float32).contiguous().clone() for k, v in module.state_dict().items()}
    safetensors.torch.save_file(state, os.path.join(folder, "model.safetensors"))


def build_tiny_checkpoint(out_dir: str, seed: int = 0) -> str:
    from minimax_video.qwen3vl_text import Qwen3VLTruncatedTextModel
    from minimax_video.qwen3vl_vision import build_vision_tower
    from minimax_video.transformer import MiniMaxH3Transformer3DModel
    from minimax_video.vae_audio import AutoencoderKLMiniMaxH3Audio
    from minimax_video.vae_video import AutoencoderKLMiniMaxH3

    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    def strip(config):
        return {k: v for k, v in config.items() if not k.startswith("_")}

    transformer = MiniMaxH3Transformer3DModel(**strip(TINY_TRANSFORMER_CONFIG))
    _save(transformer, os.path.join(out_dir, "transformer"), TINY_TRANSFORMER_CONFIG)
    _save(transformer, os.path.join(out_dir, "transformer_ref"), TINY_TRANSFORMER_CONFIG)

    vae = AutoencoderKLMiniMaxH3(**strip(TINY_VIDEO_VAE_CONFIG))
    _save(vae, os.path.join(out_dir, "vae"), TINY_VIDEO_VAE_CONFIG)

    audio_vae = AutoencoderKLMiniMaxH3Audio(**strip(TINY_AUDIO_VAE_CONFIG))
    _save(audio_vae, os.path.join(out_dir, "audio_vae"), TINY_AUDIO_VAE_CONFIG)

    for folder, shift in (("scheduler", 12.0), ("audio_scheduler", 3.0)):
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)
        with open(os.path.join(out_dir, folder, "scheduler_config.json"), "w") as f:
            json.dump({"_class_name": "MiniMaxH3Scheduler", "shift": shift}, f)

    # Conditioner: full HF-style state dict (all 52 layers + norm + lm_head, model.* prefixes).
    text = Qwen3VLTruncatedTextModel(TINY_TEXT_CONFIG, num_read_layers=TINY_TEXT_CONFIG["num_hidden_layers"])
    vision = build_vision_tower(TINY_VISION_CONFIG)
    import safetensors.torch

    encoder_dir = os.path.join(out_dir, "text_encoder")
    os.makedirs(encoder_dir, exist_ok=True)
    with open(os.path.join(encoder_dir, "config.json"), "w") as f:
        json.dump(
            {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "image_token_id": 151655,
                "video_token_id": 151656,
                "text_config": TINY_TEXT_CONFIG,
                "vision_config": TINY_VISION_CONFIG,
            },
            f,
        )
    state = {f"model.language_model.{k}": v.to(torch.float32).clone() for k, v in text.state_dict().items()}
    state["model.language_model.norm.weight"] = torch.ones(TINY_TEXT_CONFIG["hidden_size"])
    state["lm_head.weight"] = torch.zeros(8, TINY_TEXT_CONFIG["hidden_size"])
    state.update({f"model.visual.{k}": v.to(torch.float32).clone() for k, v in vision.state_dict().items()})
    safetensors.torch.save_file(state, os.path.join(encoder_dir, "model.safetensors"))

    tokenizer_dir = os.path.join(out_dir, "tokenizer")
    if not os.path.isdir(tokenizer_dir):
        shutil.copytree(TOKENIZER_DIR, tokenizer_dir)

    processor_dir = os.path.join(out_dir, "processor")
    os.makedirs(processor_dir, exist_ok=True)
    with open(os.path.join(processor_dir, "preprocessor_config.json"), "w") as f:
        json.dump(
            {
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "size": {"shortest_edge": 1024, "longest_edge": 16384},
            },
            f,
        )
    with open(os.path.join(processor_dir, "video_preprocessor_config.json"), "w") as f:
        json.dump(
            {
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "size": {"shortest_edge": 1024, "longest_edge": 65536},
            },
            f,
        )
    return out_dir


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "minimax_tiny_checkpoint"
    build_tiny_checkpoint(out)
    print(f"tiny checkpoint written to {out}")
