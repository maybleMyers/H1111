#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Convert a Cosmos3 DCP checkpoint to diffusers format.

Example:
CUDA_VISIBLE_DEVICES=0 python scripts/convert_cosmos3_to_diffusers.py \
    --checkpoint-path Cosmos3-Nano \
    --output converted/cosmos3-nano-pipeline \
    --save-pipeline

A raw Cosmos3 Edge DCP checkpoint is detected automatically. The Edge path
loads its EMA denoiser weights directly from DCP, packages the matching
Wan 2.2 VAE, and writes a shared-weight Omni checkpoint: the Diffusers
transformer is reused by the reasoner and only the missing vision tower is
stored separately.

Edge checkpoints trained with action generation are also detected
automatically: the `action2llm`/`llm2action` domain-aware projections and the
action modality embedding are exported onto the transformer's
`action_proj_in`/`action_proj_out`/`action_modality_embed` modules, with
`action_dim` and `num_embodiment_domains` derived from the DCP tensor shapes.
"""

import argparse
import contextlib
import json
import pathlib
import re
import shutil

import torch

from diffusers.models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer


DEFAULT_SOUND_TOKENIZER_CONFIG = {
    "sampling_rate": 48000,
    "vocoder_input_dim": 64,
    "dec_dim": 320,
    "dec_c_mults": [1, 2, 4, 8, 16],
    "dec_strides": [2, 4, 5, 6, 8],
    "dec_out_channels": 2,
}


COSMOS3_EDGE_REASONER = "nvidia/Cosmos3-Edge-Reasoner"
COSMOS3_EDGE_REASONER_REVISION = "590c1c0f1cd7146162d478a2180556055c1a252b"
COSMOS3_EDGE_VAE = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Keep the Transformers metadata, but avoid copying its two source shards:
# Edge's language tower is already byte-identical to the base part of the
# converted Diffusers transformer. Only the visual tower and projector have
# to be extracted from source weights.
COSMOS3_EDGE_REASONER_METADATA_FILES = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
)
COSMOS3_EDGE_REASONER_INDEX_FILE = "model.safetensors.index.json"
COSMOS3_EDGE_REASONER_VISION_SHARD = "00000.safetensors"
COSMOS3_EDGE_REASONER_FILES = (
    *COSMOS3_EDGE_REASONER_METADATA_FILES,
    COSMOS3_EDGE_REASONER_INDEX_FILE,
    COSMOS3_EDGE_REASONER_VISION_SHARD,
)


_EDGE_ATTN_KEY_REMAP = (
    (".q_proj_moe_gen.", ".add_q_proj."),
    (".k_proj_moe_gen.", ".add_k_proj."),
    (".v_proj_moe_gen.", ".add_v_proj."),
    (".o_proj_moe_gen.", ".to_add_out."),
    (".q_norm_moe_gen.", ".norm_added_q."),
    (".k_norm_moe_gen.", ".norm_added_k."),
    (".q_proj.", ".to_q."),
    (".k_proj.", ".to_k."),
    (".v_proj.", ".to_v."),
    (".o_proj.", ".to_out."),
)

_TIME_EMBEDDER_REMAP = {
    "mlp.0.weight": "linear_1.weight",
    "mlp.0.bias": "linear_1.bias",
    "mlp.2.weight": "linear_2.weight",
    "mlp.2.bias": "linear_2.bias",
}


def _get_config_value(*configs, name, default=None):
    for config in configs:
        if config is None:
            continue
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
        if isinstance(config, dict) and config.get(name) is not None:
            return config[name]
    return default


def _find_edge_dcp_model_dir(checkpoint_path: pathlib.Path) -> pathlib.Path | None:
    candidates = [checkpoint_path, checkpoint_path / "model"]
    if checkpoint_path.is_dir():
        candidates.extend(metadata_path.parent for metadata_path in checkpoint_path.glob("iter_*/model/.metadata"))
        candidates.extend(metadata_path.parent for metadata_path in checkpoint_path.glob("*/iter_*/model/.metadata"))

    for candidate in candidates:
        if not (candidate / ".metadata").is_file():
            continue

        from torch.distributed.checkpoint.filesystem import FileSystemReader

        state_dict_metadata = FileSystemReader(str(candidate)).read_metadata().state_dict_metadata
        has_edge_mlp = "net.language_model.model.layers.0.mlp.up_proj.weight" in state_dict_metadata
        has_qwen_mlp = "net.language_model.model.layers.0.mlp.gate_proj.weight" in state_dict_metadata
        if has_edge_mlp and not has_qwen_mlp:
            return candidate

    return None


def _remap_edge_dcp_key(key: str) -> str:
    if key.startswith("language_model."):
        key = key.removeprefix("language_model.").removeprefix("model.")
        for old, new in _EDGE_ATTN_KEY_REMAP:
            if old in key:
                return key.replace(old, new)
        return key
    if key.startswith("vae2llm."):
        return f"proj_in.{key.removeprefix('vae2llm.')}"
    if key.startswith("llm2vae."):
        return f"proj_out.{key.removeprefix('llm2vae.')}"
    if key.startswith("time_embedder."):
        time_embedder_key = key.removeprefix("time_embedder.")
        return f"time_embedder.{_TIME_EMBEDDER_REMAP[time_embedder_key]}"
    if key.startswith("action2llm."):
        return f"action_proj_in.{key.removeprefix('action2llm.')}"
    if key.startswith("llm2action."):
        return f"action_proj_out.{key.removeprefix('llm2action.')}"
    if key == "action_modality_embed":
        return key
    raise ValueError(f"Unsupported Cosmos3 Edge DCP key: {key!r}.")


def _detect_edge_action_config(checkpoint_path: pathlib.Path, use_ema: bool) -> dict | None:
    from torch.distributed.checkpoint.filesystem import FileSystemReader

    prefix = "net_ema." if use_ema else "net."
    metadata = FileSystemReader(str(checkpoint_path)).read_metadata().state_dict_metadata
    action_suffixes = (
        "action2llm.fc.weight",
        "action2llm.bias.weight",
        "llm2action.fc.weight",
        "llm2action.bias.weight",
        "action_modality_embed",
    )
    action_metadata = {suffix: metadata.get(f"{prefix}{suffix}") for suffix in action_suffixes}
    present_suffixes = [suffix for suffix, value in action_metadata.items() if value is not None]
    if not present_suffixes:
        return None

    # `action2llm`/`llm2action` are DomainAwareLinear modules: `fc` is an
    # Embedding(num_domains, output_size * input_size) and `bias` an
    # Embedding(num_domains, output_size), so the shapes fix every config value.
    missing_suffixes = [suffix for suffix, value in action_metadata.items() if value is None]
    if missing_suffixes:
        raise ValueError(
            f"Cosmos3 Edge {'EMA' if use_ema else 'regular'} action weights are incomplete: "
            f"present={present_suffixes}, missing={missing_suffixes}."
        )

    action_shapes = {suffix: tuple(value.size) for suffix, value in action_metadata.items()}
    expected_ranks = {
        "action2llm.fc.weight": 2,
        "action2llm.bias.weight": 2,
        "llm2action.fc.weight": 2,
        "llm2action.bias.weight": 2,
        "action_modality_embed": 1,
    }
    invalid_shapes = {
        suffix: shape for suffix, shape in action_shapes.items() if len(shape) != expected_ranks[suffix]
    }
    if invalid_shapes:
        raise ValueError(f"Cosmos3 Edge action tensors have invalid ranks: {invalid_shapes}.")

    num_embodiment_domains, proj_in_flat = action_shapes["action2llm.fc.weight"]
    proj_in_domains, hidden_size = action_shapes["action2llm.bias.weight"]
    proj_out_domains, proj_out_flat = action_shapes["llm2action.fc.weight"]
    proj_out_bias_domains, action_dim = action_shapes["llm2action.bias.weight"]
    (action_modality_embed_size,) = action_shapes["action_modality_embed"]
    domain_counts = {
        "action2llm.fc.weight": num_embodiment_domains,
        "action2llm.bias.weight": proj_in_domains,
        "llm2action.fc.weight": proj_out_domains,
        "llm2action.bias.weight": proj_out_bias_domains,
    }
    if len(set(domain_counts.values())) != 1:
        raise ValueError(f"Cosmos3 Edge action tensors disagree on embodiment domains: {domain_counts}.")
    if min(num_embodiment_domains, hidden_size, action_dim) < 1:
        raise ValueError(f"Cosmos3 Edge action tensor dimensions must be positive: {action_shapes}.")

    expected_projection_size = hidden_size * action_dim
    if proj_in_flat != expected_projection_size or proj_out_flat != expected_projection_size:
        raise ValueError(
            "Cosmos3 Edge action projection shapes are inconsistent: "
            f"action2llm.fc.weight={action_shapes['action2llm.fc.weight']}, "
            f"llm2action.fc.weight={action_shapes['llm2action.fc.weight']}; both must factor into "
            f"hidden_size={hidden_size} x action_dim={action_dim}."
        )
    if action_modality_embed_size != hidden_size:
        raise ValueError(
            "Cosmos3 Edge action modality embedding has the wrong width: "
            f"action_modality_embed={action_shapes['action_modality_embed']}, expected ({hidden_size},)."
        )

    return {
        "action_gen": True,
        "action_dim": action_dim,
        "num_embodiment_domains": num_embodiment_domains,
    }


def _build_edge_transformer(dtype: torch.dtype, action_config: dict | None = None):
    import inspect

    from accelerate import init_empty_weights

    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer

    action_kwargs = dict(action_config or {})
    init_params = inspect.signature(Cosmos3OmniTransformer.__init__).parameters
    edge_required_params = {
        "hidden_act",
        "qk_norm_for_text",
        "backbone_type",
        "temporal_compression_factor",
    }
    action_required_params = set(action_kwargs)
    missing_edge_params = sorted(edge_required_params - set(init_params))
    missing_action_params = sorted(action_required_params - set(init_params))
    if missing_edge_params or missing_action_params:
        missing_descriptions = []
        if missing_edge_params:
            missing_descriptions.append(f"Edge backbone parameters {missing_edge_params}")
        if missing_action_params:
            missing_descriptions.append(f"action-generation parameters {missing_action_params}")
        raise RuntimeError(
            "The installed diffusers build cannot construct this Cosmos3 Edge checkpoint; it is missing "
            f"{', '.join(missing_descriptions)}. Use a build that combines the Edge Nemotron-dense backbone "
            "with upstream action generation support (commit bcc20e452)."
        )

    with init_empty_weights():
        transformer = Cosmos3OmniTransformer(
            **action_kwargs,
            attention_bias=False,
            attention_dropout=0.0,
            base_fps=24,
            enable_fps_modulation=True,
            head_dim=128,
            hidden_act="relu2",
            hidden_size=2048,
            intermediate_size=9216,
            latent_channel=48,
            latent_patch_size=2,
            num_attention_heads=16,
            num_hidden_layers=28,
            num_key_value_heads=8,
            patch_latent_dim=192,
            qk_norm_for_text=False,
            rms_norm_eps=1e-5,
            rope_scaling={"mrope_section": [24, 20, 20]},
            rope_theta=100_000_000.0,
            timestep_scale=0.001,
            unified_3d_mrope_reset_spatial_ids=True,
            unified_3d_mrope_temporal_modality_margin=15000,
            vocab_size=131072,
            backbone_type="cosmos3_edge_nemotron_dense",
            temporal_compression_factor=4,
        )
        transformer = transformer.to(dtype=dtype)
        transformer.time_embedder.to(dtype=torch.float32)
    return transformer.to_empty(device="cpu")


def _validate_edge_action_pipeline_support() -> None:
    import inspect

    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline

    pipeline_params = inspect.signature(Cosmos3OmniPipeline.__call__).parameters
    required_params = {"action", "action_latents"}
    missing_params = sorted(required_params - set(pipeline_params))
    if missing_params:
        raise RuntimeError(
            "The checkpoint has action generation weights and --save-pipeline was requested, but this diffusers "
            f"build's Cosmos3OmniPipeline does not accept {missing_params}. Use a build that also includes the "
            "Cosmos3 action pipeline support from commit bcc20e452."
        )


def _load_edge_dcp_weights(transformer, checkpoint_path: pathlib.Path, use_ema: bool) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.filesystem import FileSystemReader

    prefix = "net_ema." if use_ema else "net."
    state_dict_metadata = FileSystemReader(str(checkpoint_path)).read_metadata().state_dict_metadata
    target_state_dict = transformer.state_dict()
    dcp_state_dict: dict[str, torch.Tensor] = {}
    for key in state_dict_metadata:
        if not key.startswith(prefix):
            continue
        target_key = _remap_edge_dcp_key(key.removeprefix(prefix))
        if target_key not in target_state_dict:
            raise RuntimeError(f"Cosmos3 Edge DCP key {key!r} maps to unknown transformer key {target_key!r}.")
        dcp_state_dict[key] = target_state_dict[target_key]

    mapped_keys = {_remap_edge_dcp_key(key.removeprefix(prefix)) for key in dcp_state_dict}
    missing_keys = sorted(set(target_state_dict) - mapped_keys)
    if missing_keys:
        raise RuntimeError(f"Cosmos3 Edge DCP is missing transformer weights: {missing_keys}.")

    print(f"Loading {'EMA' if use_ema else 'regular'} Cosmos3 Edge weights from {checkpoint_path} …")
    dcp.load(state_dict=dcp_state_dict, storage_reader=FileSystemReader(str(checkpoint_path)), no_dist=True)
    del dcp_state_dict, target_state_dict


def _resolve_edge_reasoner_path(args) -> pathlib.Path:
    if args.reasoner_path is not None:
        reasoner_path = pathlib.Path(args.reasoner_path).expanduser().absolute()
    else:
        from huggingface_hub import snapshot_download

        print(
            "Downloading the pinned Cosmos3 Edge reasoner snapshot "
            f"({args.reasoner_repo_id}@{args.reasoner_revision}) …"
        )
        reasoner_path = pathlib.Path(
            snapshot_download(
                repo_id=args.reasoner_repo_id,
                revision=args.reasoner_revision,
                allow_patterns=list(COSMOS3_EDGE_REASONER_FILES),
            )
        )

    if not reasoner_path.is_dir():
        raise FileNotFoundError(f"Cosmos3 Edge reasoner directory not found: {reasoner_path}")

    missing_files = [filename for filename in COSMOS3_EDGE_REASONER_FILES if not (reasoner_path / filename).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Cosmos3 Edge reasoner at {reasoner_path} is missing required files: {missing_files}"
        )
    return reasoner_path


def _load_json(path: pathlib.Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(payload: dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _edge_reasoner_language_to_generator_key_map() -> dict[str, str]:
    """Map source Nemotron reasoner keys to their shared Diffusers tensor keys."""
    mapping = {
        "lm_head.weight": "lm_head.weight",
        "model.language_model.embeddings.weight": "embed_tokens.weight",
        "model.language_model.norm_f.weight": "norm.weight",
    }
    attention_key_map = {
        "q_proj": "to_q",
        "k_proj": "to_k",
        "v_proj": "to_v",
        "o_proj": "to_out",
    }
    for layer_idx in range(28):
        attention_layer_idx = 2 * layer_idx
        mlp_layer_idx = attention_layer_idx + 1
        reasoner_attention_prefix = f"model.language_model.layers.{attention_layer_idx}"
        generator_attention_prefix = f"layers.{layer_idx}"
        mapping[f"{reasoner_attention_prefix}.norm.weight"] = f"{generator_attention_prefix}.input_layernorm.weight"
        for source_name, generator_name in attention_key_map.items():
            mapping[f"{reasoner_attention_prefix}.mixer.{source_name}.weight"] = (
                f"{generator_attention_prefix}.self_attn.{generator_name}.weight"
            )

        reasoner_mlp_prefix = f"model.language_model.layers.{mlp_layer_idx}"
        generator_mlp_prefix = f"layers.{layer_idx}"
        mapping[f"{reasoner_mlp_prefix}.norm.weight"] = f"{generator_mlp_prefix}.post_attention_layernorm.weight"
        for source_name in ("up_proj", "down_proj"):
            mapping[f"{reasoner_mlp_prefix}.mixer.{source_name}.weight"] = (
                f"{generator_mlp_prefix}.mlp.{source_name}.weight"
            )
    return mapping


def _native_edge_text_config(source_config: dict) -> dict:
    """Strip inactive hybrid fields and express the dense reasoner layout explicitly."""
    layers_block_type = source_config.get("layers_block_type")
    if layers_block_type is None:
        legacy_pattern = source_config.get("hybrid_override_pattern")
        if legacy_pattern is None:
            num_hidden_layers = source_config.get("num_hidden_layers", 56)
            if num_hidden_layers % 2:
                raise ValueError("Cosmos3 Edge requires an even number of alternating attention/MLP layers.")
            layers_block_type = ["full_attention", "mlp"] * (num_hidden_layers // 2)
        else:
            layer_type_map = {"*": "full_attention", "-": "mlp"}
            try:
                layers_block_type = [layer_type_map[layer] for layer in legacy_pattern]
            except KeyError as exc:
                raise ValueError(f"Unsupported Cosmos3 Edge legacy layer type: {exc.args[0]!r}.") from exc
    else:
        layers_block_type = list(layers_block_type)

    if set(layers_block_type) - {"full_attention", "mlp"}:
        raise ValueError(f"Cosmos3 Edge only supports dense attention/MLP layers, got {layers_block_type!r}.")

    native_config = {"model_type": "cosmos3_edge_text", "layers_block_type": layers_block_type}
    for key in (
        "attention_bias",
        "attention_dropout",
        "bos_token_id",
        "dtype",
        "eos_token_id",
        "head_dim",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "layer_norm_epsilon",
        "max_position_embeddings",
        "mlp_bias",
        "mlp_hidden_act",
        "num_attention_heads",
        "num_key_value_heads",
        "num_logits_to_keep",
        "pad_token_id",
        "rope_theta",
        "use_cache",
        "vocab_size",
    ):
        if key in source_config:
            native_config[key] = source_config[key]
    native_config["num_hidden_layers"] = len(layers_block_type)
    native_config["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": native_config.get("rope_theta", 100_000_000.0),
        "mrope_section": source_config.get("mrope_section", [24, 20, 20]),
    }
    return native_config


def _native_edge_projector_config(source_config: dict) -> dict:
    merger_intermediate_size = source_config.get("merger_intermediate_size")
    if merger_intermediate_size is None:
        merger_intermediate_size = source_config["merger_intermedia"]
    native_config = {
        "model_type": "cosmos3_edge_projector",
        "input_hidden_size": source_config["input_hidden_size"],
        "merger_intermediate_size": merger_intermediate_size,
        "out_hidden_size": source_config["out_hidden_size"],
        "spatial_merge_size": source_config["spatial_merge_size"],
        "use_postshuffle_norm": source_config["use_postshuffle_norm"],
    }
    return native_config


def _native_edge_vision_config(source_config: dict) -> dict:
    native_config = {"model_type": "cosmos3_edge_vision"}
    for key in (
        "attention_dropout",
        "hidden_act",
        "hidden_size",
        "intermediate_size",
        "layer_norm_eps",
        "num_attention_heads",
        "num_channels",
        "num_hidden_layers",
        "num_patches",
        "patch_size",
        "spatial_merge_size",
    ):
        if key in source_config:
            native_config[key] = source_config[key]
    return native_config


def _native_edge_config(source_config: dict) -> dict:
    native_config = {
        key: value
        for key, value in source_config.items()
        if key not in {"architectures", "auto_map", "model_type", "projector_config", "text_config", "vision_config"}
    }
    native_config["architectures"] = ["Cosmos3EdgeForConditionalGeneration"]
    native_config["model_type"] = "cosmos3_edge"
    native_config["allow_patterns_overrides"] = ["*/*.safetensors"]
    native_config["text_config"] = _native_edge_text_config(source_config["text_config"])
    native_config["vision_config"] = _native_edge_vision_config(source_config["vision_config"])
    native_config["projector_config"] = _native_edge_projector_config(source_config["projector_config"])
    return native_config


def _native_edge_image_processor_config(source_config: dict) -> dict:
    native_config = dict(source_config)
    native_config.pop("auto_map", None)
    native_config["processor_class"] = "Cosmos3EdgeProcessor"
    native_config["image_processor_type"] = "Cosmos3EdgeImageProcessor"
    return native_config


def _native_edge_video_processor_config(source_config: dict) -> dict:
    native_config = dict(source_config)
    native_config.pop("auto_map", None)
    native_config["processor_class"] = "Cosmos3EdgeProcessor"
    native_config["video_processor_type"] = "Cosmos3EdgeVideoProcessor"
    return native_config


def _validate_edge_reasoner_mapping(source_weight_map: dict[str, str], generator_weight_map: dict[str, str]) -> dict[str, str]:
    language_keys = {
        key for key in source_weight_map if key == "lm_head.weight" or key.startswith("model.language_model.")
    }
    vision_keys = {
        key for key in source_weight_map if key.startswith(("model.visual.", "model.projector."))
    }
    unexpected_keys = set(source_weight_map) - language_keys - vision_keys
    if unexpected_keys:
        raise RuntimeError(f"Unexpected Cosmos3 Edge reasoner keys: {sorted(unexpected_keys)}")

    mapping = _edge_reasoner_language_to_generator_key_map()
    if set(mapping) != language_keys:
        missing = sorted(language_keys - set(mapping))
        extra = sorted(set(mapping) - language_keys)
        raise RuntimeError(f"Incomplete Edge language mapping: missing={missing}, extra={extra}")
    missing_generator_keys = sorted(set(mapping.values()) - set(generator_weight_map))
    if missing_generator_keys:
        raise RuntimeError(f"Diffusers transformer is missing shared reasoner tensors: {missing_generator_keys}")
    if len(vision_keys) != 443:
        raise RuntimeError(f"Expected 443 Edge vision/projector tensors, found {len(vision_keys)}.")
    return mapping


def _write_edge_vision_encoder(
    reasoner_path: pathlib.Path,
    output_dir: pathlib.Path,
    source_weight_map: dict[str, str],
) -> set[str]:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError("Saving the Cosmos3 Edge vision encoder requires safetensors.") from exc

    vision_keys = {
        key for key in source_weight_map if key.startswith(("model.visual.", "model.projector."))
    }
    non_vision_shards = {source_weight_map[key] for key in vision_keys} - {COSMOS3_EDGE_REASONER_VISION_SHARD}
    if non_vision_shards:
        raise RuntimeError(f"Edge visual tensors unexpectedly span shards: {sorted(non_vision_shards)}")

    vision_dir = output_dir / "vision_encoder"
    vision_dir.mkdir(parents=True, exist_ok=True)
    vision_path = vision_dir / "model.safetensors"
    print(f"Extracting {len(vision_keys)} Cosmos3 Edge vision/projector tensors to {vision_path} …")
    with safe_open(reasoner_path / COSMOS3_EDGE_REASONER_VISION_SHARD, framework="pt", device="cpu") as source_file:
        state_dict = {key: source_file.get_tensor(key).contiguous() for key in sorted(vision_keys)}
    save_file(state_dict, str(vision_path), metadata={"format": "pt"})
    del state_dict
    # This is an auxiliary shard consumed through the root reasoner manifest;
    # it is not a standalone Transformers model. Do not leave a stale legacy
    # config beside it when converting into an existing output directory.
    (vision_dir / "config.json").unlink(missing_ok=True)
    return vision_keys


def _write_edge_reasoner_manifest(reasoner_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    source_index = _load_json(reasoner_path / COSMOS3_EDGE_REASONER_INDEX_FILE)
    source_weight_map = source_index["weight_map"]
    source_config = _load_json(reasoner_path / "config.json")
    generator_index = _load_json(output_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json")
    generator_weight_map = generator_index["weight_map"]
    language_mapping = _validate_edge_reasoner_mapping(source_weight_map, generator_weight_map)
    vision_keys = _write_edge_vision_encoder(reasoner_path, output_dir, source_weight_map)

    weight_map = {
        generator_key: f"transformer/{generator_weight_map[generator_key]}"
        for generator_key in language_mapping.values()
    }
    weight_map.update({key: "vision_encoder/model.safetensors" for key in vision_keys})
    if len(weight_map) != len(source_weight_map):
        raise RuntimeError(
            f"Edge unified weight index has {len(weight_map)} entries, expected {len(source_weight_map)}."
        )
    _save_json(
        {"metadata": source_index["metadata"], "weight_map": weight_map},
        output_dir / COSMOS3_EDGE_REASONER_INDEX_FILE,
    )

    _save_json(_native_edge_config(source_config), output_dir / "config.json")


def _copy_edge_reasoner_metadata(reasoner_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    print(f"Writing the shared Cosmos3 Edge reasoner into {output_dir} …")
    for filename in COSMOS3_EDGE_REASONER_METADATA_FILES:
        if filename == "config.json":
            continue
        if filename == "preprocessor_config.json":
            _save_json(
                _native_edge_image_processor_config(_load_json(reasoner_path / filename)),
                output_dir / filename,
            )
            continue
        if filename == "video_preprocessor_config.json":
            _save_json(
                _native_edge_video_processor_config(_load_json(reasoner_path / filename)),
                output_dir / filename,
            )
            continue
        shutil.copy2(reasoner_path / filename, output_dir / filename)

    for filename in (
        "configuration_nemotron_siglip2_h.py",
        "modeling_cosmos3_edge_omni.py",
        "modeling_nemotron_siglip2_h.py",
        "processing.py",
    ):
        (output_dir / filename).unlink(missing_ok=True)
    _write_edge_reasoner_manifest(reasoner_path, output_dir)
    for filename in ("00000.safetensors", "00001.safetensors"):
        (output_dir / filename).unlink(missing_ok=True)


def _copy_edge_conversion_script(output_dir: pathlib.Path) -> None:
    source = pathlib.Path(__file__).resolve()
    destination = output_dir / source.name
    if source != destination:
        shutil.copy2(source, destination)


def _add_edge_reasoner_to_pipeline(args) -> None:
    output_dir = pathlib.Path(args.output).expanduser().absolute()
    expected_paths = ("model_index.json", "scheduler", "text_tokenizer", "transformer", "vae")
    missing_paths = [str(output_dir / path) for path in expected_paths if not (output_dir / path).exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Expected an existing Cosmos3 Edge Diffusers pipeline before adding its reasoner; "
            f"missing paths: {missing_paths}"
        )

    reasoner_path = _resolve_edge_reasoner_path(args)
    _copy_edge_reasoner_metadata(reasoner_path, output_dir)
    _copy_edge_conversion_script(output_dir)
    print("Done.")


def _convert_edge_dcp(args, checkpoint_path: pathlib.Path, dtype: torch.dtype) -> None:
    from transformers import PreTrainedTokenizerFast

    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline

    if args.include_sound_tokenizer or args.sound_tokenizer_path is not None:
        raise ValueError("The supplied Cosmos3 Edge checkpoint is video-only and cannot include a sound tokenizer.")
    if args.include_reasoner and not args.save_pipeline:
        raise ValueError(
            "A Cosmos3 Edge reasoner can only be included with --save-pipeline because its root Transformers "
            "config.json would conflict with a transformer-only Diffusers save. Use --no-include-reasoner instead."
        )

    action_config = _detect_edge_action_config(checkpoint_path, args.use_ema)
    if action_config is not None:
        print(
            "Detected Cosmos3 Edge action generation weights "
            f"(action_dim={action_config['action_dim']}, "
            f"num_embodiment_domains={action_config['num_embodiment_domains']})."
        )
    transformer = _build_edge_transformer(dtype, action_config)
    if action_config is not None and args.save_pipeline:
        _validate_edge_action_pipeline_support()
    _load_edge_dcp_weights(transformer, checkpoint_path, args.use_ema)

    reasoner_path = _resolve_edge_reasoner_path(args) if args.include_reasoner else None

    output_dir = pathlib.Path(args.output).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.save_pipeline:
        print(f"Saving Cosmos3 Edge transformer to {output_dir} …")
        transformer.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="5GB")
        print("Done.")
        return

    tokenizer_source = str(reasoner_path) if reasoner_path is not None else args.reasoner_repo_id
    tokenizer_kwargs = {}
    if reasoner_path is None:
        tokenizer_kwargs["revision"] = args.reasoner_revision
    text_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    for token in ("<|vision_start|>", "<|vision_end|>"):
        token_id = text_tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0 or token_id >= transformer.config.vocab_size:
            raise ValueError(
                f"Cosmos3 Edge tokenizer token {token!r} has invalid ID {token_id!r} for "
                f"vocab_size={transformer.config.vocab_size}."
            )

    diffusers_vae = AutoencoderKLWan.from_pretrained(COSMOS3_EDGE_VAE, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000,
        solver_order=2,
        predict_x0=True,
        solver_type="bh2",
        lower_order_final=True,
        final_sigmas_type="zero",
        use_flow_sigmas=True,
        prediction_type="flow_prediction",
        flow_shift=5.0,
    )
    pipeline = Cosmos3OmniPipeline(
        transformer=transformer,
        text_tokenizer=text_tokenizer,
        vae=diffusers_vae,
        scheduler=scheduler,
        enable_safety_checker=False,
        default_use_system_prompt=False,
        use_native_flow_schedule=True,
    )
    print(f"Saving Cosmos3 Edge pipeline to {output_dir} …")
    pipeline.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="5GB")
    if reasoner_path is not None:
        _copy_edge_reasoner_metadata(reasoner_path, output_dir)
    _copy_edge_conversion_script(output_dir)
    print("Done.")


def _load_sound_tokenizer_state_dict(checkpoint_path: pathlib.Path) -> dict[str, torch.Tensor]:
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Loading AVAE .safetensors checkpoints requires safetensors.") from exc
        checkpoint = load_file(str(checkpoint_path), device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError(f"AVAE checkpoint must be a dict, got {type(checkpoint)!r}.")

    for key in ("generator", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break

    state_dict = {
        key: value.detach().cpu().contiguous() for key, value in checkpoint.items() if isinstance(value, torch.Tensor)
    }
    if not state_dict:
        raise RuntimeError(f"No tensor state dict found in AVAE checkpoint keys: {list(checkpoint.keys())[:16]}")
    return state_dict


def _load_sound_tokenizer_config(config_path: pathlib.Path | None, fallback_config_path: pathlib.Path) -> dict:
    selected_config_path = config_path
    if selected_config_path is None and fallback_config_path.exists():
        selected_config_path = fallback_config_path
    if selected_config_path is None:
        return dict(DEFAULT_SOUND_TOKENIZER_CONFIG)
    with open(selected_config_path, encoding="utf-8") as f:
        return json.load(f)


_SOUND_TOKENIZER_PER_KEY_PREFIXES = ("module.", "generator.", "model.", "state_dict.")
_SOUND_TOKENIZER_RES_UNIT_INNER_NAMES = {0: "snake1", 1: "conv1", 2: "snake2", 3: "conv2"}


def _sound_tokenizer_strip_per_key_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(state_dict)
    changed = True
    while changed:
        changed = False
        for prefix in _SOUND_TOKENIZER_PER_KEY_PREFIXES:
            if any(key.startswith(prefix) for key in out):
                out = {(key[len(prefix) :] if key.startswith(prefix) else key): value for key, value in out.items()}
                changed = True
                break
        if any(key.startswith(("decoder.", "encoder.", "bottleneck.")) for key in out):
            break
    return out


def _sound_tokenizer_filter_decoder(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state_dict.items() if key.startswith("decoder.")}


def _sound_tokenizer_infer_num_blocks(state_dict: dict[str, torch.Tensor]) -> int:
    block_indices: set[int] = set()
    for key in state_dict:
        match = re.match(r"decoder\.layers\.(\d+)\.layers\.\d+\.", key)
        if match:
            block_indices.add(int(match.group(1)))
    return len(block_indices)


def _sound_tokenizer_remap_flat_layout(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert legacy AVAE `decoder.layers.*` keys to OobleckDecoder attribute keys."""
    if not any(re.match(r"decoder\.layers\.\d+\.", key) for key in state_dict):
        return state_dict

    num_blocks = _sound_tokenizer_infer_num_blocks(state_dict)
    if num_blocks == 0:
        raise RuntimeError("Detected flat `decoder.layers.*` layout but no decoder blocks were found; cannot remap.")
    snake1_idx = num_blocks + 1
    conv2_idx = num_blocks + 2

    def _remap(key: str) -> str:
        match = re.fullmatch(r"decoder\.layers\.(\d+)\.layers\.(\d+)\.layers\.(\d+)\.(.+)", key)
        if match:
            block_n, res_n, inner_n, rest = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                match.group(4),
            )
            if res_n not in (2, 3, 4):
                raise RuntimeError(f"Unexpected residual position {res_n} in {key!r}.")
            inner_name = _SOUND_TOKENIZER_RES_UNIT_INNER_NAMES.get(inner_n)
            if inner_name is None:
                raise RuntimeError(f"Unexpected residual inner index {inner_n} in {key!r}.")
            return f"decoder.block.{block_n - 1}.res_unit{res_n - 1}.{inner_name}.{rest}"

        match = re.fullmatch(r"decoder\.layers\.(\d+)\.layers\.(\d+)\.(.+)", key)
        if match:
            block_n, sub_n, rest = int(match.group(1)), int(match.group(2)), match.group(3)
            block_idx = block_n - 1
            if sub_n == 0:
                return f"decoder.block.{block_idx}.snake1.{rest}"
            if sub_n == 1:
                return f"decoder.block.{block_idx}.conv_t1.{rest}"
            raise RuntimeError(f"Unexpected decoder block sub-index {sub_n} in {key!r}.")

        match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", key)
        if match:
            layer_n, rest = int(match.group(1)), match.group(2)
            if layer_n == 0:
                return f"decoder.conv1.{rest}"
            if layer_n == snake1_idx:
                return f"decoder.snake1.{rest}"
            if layer_n == conv2_idx:
                return f"decoder.conv2.{rest}"
            raise RuntimeError(
                f"Unexpected decoder leaf layer index {layer_n} (expected 0, {snake1_idx}, or {conv2_idx}) in {key!r}."
            )

        return key

    return {_remap(key): value for key, value in state_dict.items()}


def _sound_tokenizer_reshape_snake_params(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if (key.endswith(".alpha") or key.endswith(".beta")) and value.ndim == 1:
            value = value.unsqueeze(0).unsqueeze(-1).contiguous()
        out[key] = value
    return out


def _sound_tokenizer_reapply_weight_norm(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reconstruct weight-norm parameters if the source checkpoint has folded conv weights."""
    out = dict(state_dict)
    candidate_keys = [
        key
        for key in state_dict
        if key.endswith(".weight") and any(f".{layer}." in key for layer in ("conv1", "conv2", "conv_t1"))
    ]
    for key in candidate_keys:
        stem = key[: -len(".weight")]
        weight_g_key = f"{stem}.weight_g"
        weight_v_key = f"{stem}.weight_v"
        if weight_g_key in state_dict or weight_v_key in state_dict:
            continue
        weight = state_dict[key]
        norm_dims = tuple(range(1, weight.ndim))
        out.pop(key)
        out[weight_g_key] = weight.norm(p=2, dim=norm_dims, keepdim=True).contiguous()
        out[weight_v_key] = weight.contiguous()
    return out


def _remap_avae_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a legacy AVAE state dict into the Cosmos3AVAEAudioTokenizer state dict."""
    state_dict = _sound_tokenizer_strip_per_key_prefixes(state_dict)
    state_dict = _sound_tokenizer_filter_decoder(state_dict)
    if not state_dict:
        raise RuntimeError("Sound tokenizer state dict has no `decoder.*` keys after prefix stripping.")
    state_dict = _sound_tokenizer_remap_flat_layout(state_dict)
    state_dict = _sound_tokenizer_reshape_snake_params(state_dict)
    state_dict = _sound_tokenizer_reapply_weight_norm(state_dict)
    if any(re.match(r"decoder\.layers\.\d+", key) for key in state_dict):
        raise RuntimeError("Flat `decoder.layers.*` keys remain after remap; conversion is incomplete.")
    return state_dict


def _build_sound_tokenizer(
    checkpoint_path: pathlib.Path,
    config_path: pathlib.Path | None,
) -> Cosmos3AVAEAudioTokenizer:
    config = _load_sound_tokenizer_config(config_path, fallback_config_path=pathlib.Path())
    print(f"Loading AVAE sound tokenizer weights from {checkpoint_path} …")
    raw_state_dict = _load_sound_tokenizer_state_dict(checkpoint_path)
    state_dict = _remap_avae_state_dict(raw_state_dict)
    print(f"  Remapped {len(raw_state_dict)} → {len(state_dict)} decoder keys.")

    sound_tokenizer = Cosmos3AVAEAudioTokenizer(
        sampling_rate=config.get("sampling_rate", DEFAULT_SOUND_TOKENIZER_CONFIG["sampling_rate"]),
        vocoder_input_dim=config.get("vocoder_input_dim", DEFAULT_SOUND_TOKENIZER_CONFIG["vocoder_input_dim"]),
        dec_dim=config.get("dec_dim", DEFAULT_SOUND_TOKENIZER_CONFIG["dec_dim"]),
        dec_c_mults=tuple(config.get("dec_c_mults", DEFAULT_SOUND_TOKENIZER_CONFIG["dec_c_mults"])),
        dec_strides=tuple(config.get("dec_strides", DEFAULT_SOUND_TOKENIZER_CONFIG["dec_strides"])),
        dec_out_channels=config.get("dec_out_channels", DEFAULT_SOUND_TOKENIZER_CONFIG["dec_out_channels"]),
    )
    load_result = sound_tokenizer.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Cosmos3 AVAE sound tokenizer load did not match strictly: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}."
        )
    return sound_tokenizer


@contextlib.contextmanager
def _skip_source_sound_tokenizer_load(omni_mot_model_cls):
    original_set_up_tokenizers = omni_mot_model_cls.set_up_tokenizers

    def set_up_tokenizers_without_sound(self):
        if not getattr(self.config, "sound_gen", False):
            return original_set_up_tokenizers(self)

        sound_gen = self.config.sound_gen
        self.config.sound_gen = False
        try:
            return original_set_up_tokenizers(self)
        finally:
            self.config.sound_gen = sound_gen

    omni_mot_model_cls.set_up_tokenizers = set_up_tokenizers_without_sound
    try:
        yield
    finally:
        omni_mot_model_cls.set_up_tokenizers = original_set_up_tokenizers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        default="Cosmos3-Nano",
        help="Named checkpoint (e.g. 'Cosmos3-Nano') or path to a DCP checkpoint directory.",
    )
    parser.add_argument("--output", required=True, help="Directory to save the converted diffusers model.")
    parser.add_argument(
        "--save-pipeline",
        action="store_true",
        help="Save the full pipeline (transformer + VAE + tokenizer + scheduler).",
    )
    parser.add_argument(
        "--dtype", default="bf16", choices=["fp32", "fp16", "bf16"], help="Dtype to save the transformer in."
    )
    parser.add_argument(
        "--sound-tokenizer-path", help="Optional AVAE sound tokenizer checkpoint to save under sound_tokenizer/."
    )
    parser.add_argument(
        "--sound-tokenizer-config-path", help="Optional AVAE config JSON to save under sound_tokenizer/config.json."
    )
    parser.add_argument(
        "--include-sound-tokenizer",
        action="store_true",
        help="Require saving sound_tokenizer/ even if the source transformer is video-only.",
    )
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use `net_ema` weights when converting a raw Cosmos3 Edge DCP checkpoint.",
    )
    parser.add_argument(
        "--include-reasoner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add the pinned Cosmos3 Edge reasoner in shared-weight form so the pipeline output is usable by "
            "Transformers as well as Diffusers."
        ),
    )
    parser.add_argument(
        "--reasoner-repo-id",
        default=COSMOS3_EDGE_REASONER,
        help="Hugging Face repository containing the Cosmos3 Edge reasoner checkpoint.",
    )
    parser.add_argument(
        "--reasoner-revision",
        default=COSMOS3_EDGE_REASONER_REVISION,
        help="Pinned revision of the Cosmos3 Edge reasoner checkpoint.",
    )
    parser.add_argument(
        "--reasoner-path",
        help="Optional local Cosmos3 Edge reasoner snapshot, used instead of downloading --reasoner-repo-id.",
    )
    parser.add_argument(
        "--copy-edge-reasoner",
        action="store_true",
        help=(
            "Add the pinned shared-weight reasoner and this converter to an existing Cosmos3 Edge Diffusers "
            "pipeline at --output."
        ),
    )
    args = parser.parse_args()

    if args.copy_edge_reasoner:
        _add_edge_reasoner_to_pipeline(args)
        return

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    raw_checkpoint_path = pathlib.Path(args.checkpoint_path).expanduser()
    edge_dcp_model_dir = _find_edge_dcp_model_dir(raw_checkpoint_path) if raw_checkpoint_path.exists() else None
    if edge_dcp_model_dir is not None:
        _convert_edge_dcp(args, edge_dcp_model_dir, dtype)
        return

    from cosmos3.common.init import init_script

    init_script()

    from accelerate import init_empty_weights
    from cosmos3.args import _CHECKPOINTS
    from cosmos3.model import Cosmos3OmniModel
    from projects.cosmos3.vfm.models.omni_mot_model import OmniMoTModel
    from transformers import AutoTokenizer

    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline

    sound_tokenizer_path = (
        pathlib.Path(args.sound_tokenizer_path).expanduser().absolute() if args.sound_tokenizer_path else None
    )
    sound_tokenizer_config_path = (
        pathlib.Path(args.sound_tokenizer_config_path).expanduser().absolute()
        if args.sound_tokenizer_config_path
        else None
    )
    if args.include_sound_tokenizer and sound_tokenizer_path is None:
        raise ValueError("Sound tokenizer output was requested, but --sound-tokenizer-path was not provided.")
    if sound_tokenizer_path is not None and not sound_tokenizer_path.exists():
        raise FileNotFoundError(f"Sound tokenizer checkpoint not found: {sound_tokenizer_path}")
    if sound_tokenizer_config_path is not None and not sound_tokenizer_config_path.exists():
        raise FileNotFoundError(f"Sound tokenizer config not found: {sound_tokenizer_config_path}")

    checkpoint_name = args.checkpoint_path
    if checkpoint_name in _CHECKPOINTS:
        checkpoint_path = pathlib.Path(_CHECKPOINTS[checkpoint_name].download())
    else:
        checkpoint_path = pathlib.Path(checkpoint_name).expanduser().absolute()
    print(f"Resolved checkpoint path: {checkpoint_path}")

    print("Instantiating model and loading weights from DCP checkpoint …")
    print("Skipping source AVAE tokenizer instantiation during converter-only model load …")
    with _skip_source_sound_tokenizer_load(OmniMoTModel):
        _tmp = Cosmos3OmniModel.from_pretrained_dcp(checkpoint_path).model

    # Extract network components and architecture config from DCP model
    language_model = _tmp.net.language_model
    vae2llm = _tmp.net.vae2llm
    llm2vae = _tmp.net.llm2vae
    time_embedder = _tmp.net.time_embedder
    lm_cfg = _tmp.net.language_model.config
    net_cfg = _tmp.net.config
    model_cfg = _tmp.config
    patch_latent_dim = _tmp.net.patch_latent_dim
    hidden_size = _tmp.net.hidden_size
    num_attention_heads = _tmp.net.num_heads
    num_key_value_heads = _tmp.net.num_kv_heads
    head_dim = _tmp.net.head_dim
    num_hidden_layers = _tmp.net.num_hidden_layers
    latent_patch_size = _tmp.net.latent_patch_size
    latent_channel = _tmp.net.latent_channel
    timestep_scale = _tmp.net.timestep_scale
    base_fps = int(net_cfg.base_fps)
    enable_fps_modulation = net_cfg.enable_fps_modulation
    unified_3d_mrope_reset_spatial_ids = _tmp.config.diffusion_expert_config.unified_3d_mrope_reset_spatial_ids
    unified_3d_mrope_temporal_modality_margin = (
        _tmp.config.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin
    )
    sound2llm = getattr(_tmp.net, "sound2llm", None)
    llm2sound = getattr(_tmp.net, "llm2sound", None)
    sound_modality_embed = getattr(_tmp.net, "sound_modality_embed", None)
    has_sound_projection_weights = any(module is not None for module in (sound2llm, llm2sound, sound_modality_embed))
    sound_gen = bool(
        _get_config_value(net_cfg, model_cfg, name="sound_gen", default=False) or has_sound_projection_weights
    )
    sound_dim = _get_config_value(net_cfg, model_cfg, name="sound_dim", default=None)
    if sound_dim is None and sound2llm is not None:
        sound_dim = sound2llm.in_features
    sound_latent_fps = _get_config_value(net_cfg, model_cfg, name="sound_latent_fps", default=25.0)
    if sound_gen:
        missing_sound_modules = [
            name
            for name, module in (
                ("sound2llm", sound2llm),
                ("llm2sound", llm2sound),
                ("sound_modality_embed", sound_modality_embed),
            )
            if module is None
        ]
        if missing_sound_modules:
            raise RuntimeError(
                "Source checkpoint is configured for sound generation but is missing "
                f"sound projection weights: {missing_sound_modules}."
            )
        if sound_dim is None:
            raise RuntimeError("Source checkpoint is configured for sound generation but sound_dim is missing.")
    del _tmp
    torch.cuda.empty_cache()

    # Init diffusers Cosmos3OmniTransformer with full architecture config from DCP
    with init_empty_weights():
        transformer = Cosmos3OmniTransformer(
            attention_bias=lm_cfg.attention_bias,
            attention_dropout=lm_cfg.attention_dropout,
            base_fps=base_fps,
            enable_fps_modulation=enable_fps_modulation,
            head_dim=head_dim,
            hidden_size=hidden_size,
            intermediate_size=lm_cfg.intermediate_size,
            latent_channel=latent_channel,
            latent_patch_size=latent_patch_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            patch_latent_dim=patch_latent_dim,
            rms_norm_eps=lm_cfg.rms_norm_eps,
            rope_scaling=lm_cfg.rope_scaling,
            rope_theta=lm_cfg.rope_theta,
            sound_dim=sound_dim,
            sound_gen=sound_gen,
            sound_latent_fps=sound_latent_fps,
            timestep_scale=timestep_scale,
            unified_3d_mrope_reset_spatial_ids=unified_3d_mrope_reset_spatial_ids,
            unified_3d_mrope_temporal_modality_margin=unified_3d_mrope_temporal_modality_margin,
            vocab_size=lm_cfg.vocab_size,
            backbone_type="cosmos3_edge_nemotron_dense",

        )
    # The source language_model nests its transformer stack under a `model.` attribute
    # (HF Qwen-style). Diffusers Cosmos3OmniTransformer holds those layers flat, so
    # strip the leading `model.` prefix from the language-model state-dict keys.
    state_dict = {
        (k[len("model.") :] if k.startswith("model.") else k): v for k, v in language_model.state_dict().items()
    }
    # Remap PackedAttentionMoT attribute names from the source (Qwen-style q_proj/k_proj/...
    # plus cosmos-specific *_moe_gen) to the diffusers AttentionModuleMixin canonical names.
    # Order matters: the *_moe_gen substrings must be substituted before the plain ones.
    _ATTN_KEY_REMAP = [
        (".q_proj_moe_gen.", ".add_q_proj."),
        (".k_proj_moe_gen.", ".add_k_proj."),
        (".v_proj_moe_gen.", ".add_v_proj."),
        (".o_proj_moe_gen.", ".to_add_out."),
        (".q_norm_moe_gen.", ".norm_added_q."),
        (".k_norm_moe_gen.", ".norm_added_k."),
        (".q_proj.", ".to_q."),
        (".k_proj.", ".to_k."),
        (".v_proj.", ".to_v."),
        (".o_proj.", ".to_out."),
        (".q_norm.", ".norm_q."),
        (".k_norm.", ".norm_k."),
    ]
    remapped_state_dict: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        for old, new in _ATTN_KEY_REMAP:
            if old in k:
                k = k.replace(old, new)
                break
        remapped_state_dict[k] = v
    state_dict = remapped_state_dict
    for k, v in vae2llm.state_dict().items():
        state_dict[f"proj_in.{k}"] = v
    for k, v in llm2vae.state_dict().items():
        state_dict[f"proj_out.{k}"] = v
    _TIME_EMBEDDER_REMAP = {
        "mlp.0.weight": "linear_1.weight",
        "mlp.0.bias": "linear_1.bias",
        "mlp.2.weight": "linear_2.weight",
        "mlp.2.bias": "linear_2.bias",
    }
    for k, v in time_embedder.state_dict().items():
        state_dict[f"time_embedder.{_TIME_EMBEDDER_REMAP[k]}"] = v
    if sound_gen:
        for k, v in sound2llm.state_dict().items():
            state_dict[f"audio_proj_in.{k}"] = v
        for k, v in llm2sound.state_dict().items():
            state_dict[f"audio_proj_out.{k}"] = v
        state_dict["audio_modality_embed"] = sound_modality_embed
    transformer.load_state_dict(state_dict, strict=True, assign=True)
    del (
        language_model,
        vae2llm,
        llm2vae,
        time_embedder,
        sound2llm,
        llm2sound,
        sound_modality_embed,
        state_dict,
    )
    torch.cuda.empty_cache()

    transformer = transformer.to(dtype=dtype)

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_sound_tokenizer = (
        args.include_sound_tokenizer or sound_tokenizer_path is not None or (sound_gen and args.save_pipeline)
    )
    if include_sound_tokenizer and sound_tokenizer_path is None:
        raise ValueError(
            "The source checkpoint is configured for sound generation, so --sound-tokenizer-path "
            "is required when saving a full pipeline."
        )

    if args.save_pipeline:
        text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        diffusers_vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=torch.bfloat16
        )
        sound_tokenizer = None
        if include_sound_tokenizer:
            assert sound_tokenizer_path is not None
            sound_tokenizer = _build_sound_tokenizer(sound_tokenizer_path, sound_tokenizer_config_path)

        # Karras schedule approximating FlowUniPCMultistepScheduler with shift=5, 35 steps.
        # Measured from that schedule: first flow-sigma=0.9998, last flow-sigma=0.1281.
        # EDM sigma = flow_sigma / (1 - flow_sigma), so:
        #   sigma_max = 0.9998 / 0.0002 = 4999  (but capped at 200 to avoid duplicate
        #               integer timesteps from Karras clustering near the top)
        #   sigma_min = 0.1281 / (1 - 0.1281)  = 0.1281 / 0.8719 ≈ 0.147
        scheduler = UniPCMultistepScheduler(
            use_karras_sigmas=True,
            use_flow_sigmas=True,
            prediction_type="flow_prediction",
            sigma_max=200.0,
            sigma_min=0.147,
        )

        pipeline = Cosmos3OmniPipeline(
            transformer=transformer,
            text_tokenizer=text_tokenizer,
            vae=diffusers_vae,
            scheduler=scheduler,
            sound_tokenizer=sound_tokenizer,
        )
        print(f"Saving full pipeline to {output_dir} …")
        pipeline.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="5GB")
    else:
        print(f"Saving transformer to {output_dir} …")
        transformer.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="5GB")
        if include_sound_tokenizer:
            print("Skipping sound_tokenizer/ save because --save-pipeline was not set.")

    print("Done.")


if __name__ == "__main__":
    main()
