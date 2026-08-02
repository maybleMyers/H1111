# Component loaders for MiniMax-H3 HF checkpoints (diffusers-layout snapshot dirs) with H1111
# memory management: lazy shard streaming, LoRA merge at load, fp8. Mirrors
# cosmos_engine/cosmos_video/model_loader.py.
#
# Checkpoint layout (MiniMaxAI/MiniMax-H3): transformer/ (t2va + fl2va), transformer_ref/
# (ref2va), vae/, audio_vae/, scheduler/, audio_scheduler/, text_encoder/, tokenizer/,
# processor/. The task selects which transformer partition loads; the other is never touched.
import glob
import json
import logging
import os
from typing import List, Optional

import torch
from accelerate import init_empty_weights

logger = logging.getLogger(__name__)

# fp8 targets the block stack only. AdaLN projections (`adaln_proj.linear`, ~40% of the
# weights) are bfloat16 in the checkpoint and are *included* — required to fit 33B on a 48GB
# card — unless `fp8_exclude_adaln` asks for the quality escape hatch (+~13 GB resident).
FP8_TARGET_KEYS = ["transformer_blocks."]
FP8_EXCLUDE_KEYS = [
    "norm",
    "proj_in",
    "proj_out",
    "audio_proj_in",
    "audio_proj_out",
    "time_embedder",
    "time_proj",
    "context_embedder",
    "token_refiner",
    "rope",
]

# MiniMax-H3 ships a mixed-precision checkpoint: these top-level modules are float32 and the
# forward's dtype-alignment casts depend on them staying float32 (transformer.py
# `_keep_in_fp32_modules`). The cast loops below must skip them instead of blanket-casting.
FP32_KEY_PREFIXES = ("proj_in.", "audio_proj_in.", "time_embedder.", "proj_out.", "audio_proj_out.")


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _component_dir(ckpt_dir, name):
    d = os.path.join(ckpt_dir, name)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"checkpoint dir {ckpt_dir} has no '{name}/' subfolder; expected a diffusers-layout "
            f"MiniMax-H3 snapshot (e.g. huggingface-cli download MiniMaxAI/MiniMax-H3)"
        )
    return d


def _shard_files(component_dir):
    files = sorted(glob.glob(os.path.join(component_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors shards in {component_dir}")
    return files


def _from_config(cls, config_path, torch_dtype):
    config = _read_json(config_path)
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)
    with init_empty_weights():
        model = cls.from_config(config)
    if torch_dtype is not None:
        model = model.to(torch_dtype)
    return model


def _load_sharded_state_dict(files, device, dtype=None):
    from utils.safetensors_utils import load_safetensors

    sd = {}
    for f in files:
        sd.update(load_safetensors(f, device=device, disable_mmap=True, dtype=dtype))
    return sd


def _is_fp32_key(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in FP32_KEY_PREFIXES)


def load_transformer(
    ckpt_dir: str,
    device: torch.device,
    task: str = "t2va",
    dit_dtype: torch.dtype = torch.bfloat16,
    fp8: bool = False,
    fp8_scaled: bool = False,
    fp8_fast: bool = False,
    fp8_exclude_adaln: bool = False,
    lora_weights_list: Optional[List[dict]] = None,
    lora_multipliers: Optional[List[float]] = None,
    dit_path: Optional[str] = None,
):
    """Build MiniMaxH3Transformer3DModel and load the task's partition: transformer/ for
    t2va/fl2va, transformer_ref/ for ref2va (or an explicit dir / merged file via dit_path).
    LoRA merges during the streaming load; fp8_scaled quantizes on the fly and monkey-patches
    the Linears. The float32 modules of the mixed-precision checkpoint stay float32."""
    from .transformer import MiniMaxH3Transformer3DModel

    subfolder = "transformer_ref" if task == "ref2va" else "transformer"
    tdir = dit_path if dit_path else _component_dir(ckpt_dir, subfolder)
    if os.path.isdir(tdir):
        config_path = os.path.join(tdir, "config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(_component_dir(ckpt_dir, subfolder), "config.json")
        files = _shard_files(tdir)
    else:
        config_path = os.path.join(_component_dir(ckpt_dir, subfolder), "config.json")
        files = [tdir]

    exclude_keys = FP8_EXCLUDE_KEYS + (["adaln_proj"] if fp8_exclude_adaln else [])
    model = _from_config(MiniMaxH3Transformer3DModel, config_path, dit_dtype)

    if fp8_scaled:
        from modules.fp8_optimization_utils import (
            apply_fp8_monkey_patch,
            optimize_state_dict_with_fp8,
            optimize_state_dict_with_fp8_on_the_fly,
        )

        if lora_weights_list:
            from utils.lora_utils import load_safetensors_with_lora_and_fp8

            sd = load_safetensors_with_lora_and_fp8(
                model_files=files,
                lora_weights_list=lora_weights_list,
                lora_multipliers=lora_multipliers,
                fp8_optimization=False,
                calc_device=device,
                move_to_device=False,
            )
            sd = optimize_state_dict_with_fp8(
                sd, device, target_layer_keys=FP8_TARGET_KEYS, exclude_layer_keys=exclude_keys
            )
        else:
            sd = optimize_state_dict_with_fp8_on_the_fly(
                files,
                calc_device=device,
                target_layer_keys=FP8_TARGET_KEYS,
                exclude_layer_keys=exclude_keys,
                move_to_device=False,
            )
        for k in list(sd.keys()):
            if _is_fp32_key(k) and sd[k].dtype != torch.float32:
                sd[k] = sd[k].to(torch.float32)
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=fp8_fast)
        info = model.load_state_dict(sd, strict=True, assign=True)
        logger.info(f"fp8-scaled transformer load ({subfolder}): {info}")
    else:
        from utils.lora_utils import load_safetensors_with_lora_and_fp8

        sd = load_safetensors_with_lora_and_fp8(
            model_files=files,
            lora_weights_list=lora_weights_list if lora_weights_list else None,
            lora_multipliers=lora_multipliers,
            fp8_optimization=False,
            calc_device=device,
            move_to_device=False,
        )
        for k in list(sd.keys()):
            if _is_fp32_key(k):
                if sd[k].dtype != torch.float32:
                    sd[k] = sd[k].to(torch.float32)
            elif fp8 and (
                k.endswith(".weight")
                and any(t in k for t in FP8_TARGET_KEYS)
                and not any(e in k for e in exclude_keys)
                and sd[k].dtype in (torch.float16, torch.bfloat16, torch.float32)
            ):
                # plain e4m3 weight cast for eligible linear weights
                sd[k] = sd[k].to(torch.float8_e4m3fn)
            elif sd[k].dtype not in (torch.float8_e4m3fn,):
                sd[k] = sd[k].to(dit_dtype)
        info = model.load_state_dict(sd, strict=True, assign=True)
        logger.info(f"transformer load ({subfolder}): {info}")

    model.eval().requires_grad_(False)
    return model


def load_vae(ckpt_dir: str, device: torch.device, vae_dtype: torch.dtype = torch.float32, vae_path: Optional[str] = None):
    """Video VAE. The released checkpoint is float32 and the verified decode recipe is fp16
    autocast over float32 weights, so `vae_dtype` should stay float32."""
    from .vae_video import AutoencoderKLMiniMaxH3

    vdir = vae_path if vae_path else _component_dir(ckpt_dir, "vae")
    config_path = os.path.join(vdir if os.path.isdir(vdir) else _component_dir(ckpt_dir, "vae"), "config.json")
    vae = _from_config(AutoencoderKLMiniMaxH3, config_path, vae_dtype)
    files = _shard_files(vdir) if os.path.isdir(vdir) else [vdir]
    sd = _load_sharded_state_dict(files, device="cpu", dtype=vae_dtype)
    info = vae.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"vae load: {info}")
    vae.eval().requires_grad_(False)
    return vae.to(device)


def load_audio_vae(
    ckpt_dir: str, device: torch.device, dtype: torch.dtype = torch.float32, audio_vae_path: Optional[str] = None
):
    from .vae_audio import AutoencoderKLMiniMaxH3Audio

    adir = audio_vae_path if audio_vae_path else _component_dir(ckpt_dir, "audio_vae")
    config_path = os.path.join(
        adir if os.path.isdir(adir) else _component_dir(ckpt_dir, "audio_vae"), "config.json"
    )
    vae = _from_config(AutoencoderKLMiniMaxH3Audio, config_path, dtype)
    files = _shard_files(adir) if os.path.isdir(adir) else [adir]
    sd = _load_sharded_state_dict(files, device="cpu", dtype=dtype)
    info = vae.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"audio vae load: {info}")
    vae.eval().requires_grad_(False)
    return vae.to(device)


def load_schedulers(ckpt_dir: str, flow_shift: Optional[float] = None, audio_flow_shift: Optional[float] = None):
    """The two MiniMaxH3Scheduler instances (video shift=12.0, audio shift=3.0 in the release)."""
    from .scheduler import MiniMaxH3Scheduler

    schedulers = []
    for folder, override in (("scheduler", flow_shift), ("audio_scheduler", audio_flow_shift)):
        config = _read_json(os.path.join(_component_dir(ckpt_dir, folder), "scheduler_config.json"))
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)
        scheduler = MiniMaxH3Scheduler(**config)
        if override is not None:
            scheduler.set_shift(override)
        schedulers.append(scheduler)
    return tuple(schedulers)


def read_component_geometry(ckpt_dir: str) -> dict:
    """The cheap config-only facts the setup stage needs before any weights load."""
    vae_config = _read_json(os.path.join(_component_dir(ckpt_dir, "vae"), "config.json"))
    audio_config = _read_json(os.path.join(_component_dir(ckpt_dir, "audio_vae"), "config.json"))
    spatial = 1
    for f in vae_config.get("spatial_downsample_factors", (2, 2, 2, 2, 1, 1)):
        spatial *= int(f)
    return {
        "vae_latent_channels": int(vae_config.get("latent_channels", 24)),
        "vae_spatial_compression_ratio": spatial,
        "audio_latent_channels": int(audio_config.get("latent_channels", 32)),
        "audio_sampling_rate": int(audio_config.get("sampling_rate", 32000)),
    }
