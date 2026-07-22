# Component loaders for Cosmos3 HF checkpoints (diffusers-layout snapshot dirs)
# with H1111 memory management: lazy shard streaming, LoRA merge at load, fp8.
import glob
import json
import logging
import os
from typing import List, Optional

import torch
from accelerate import init_empty_weights

logger = logging.getLogger(__name__)

# fp8 must never touch these (norms, embeddings, modality projections, heads).
FP8_EXCLUDE_KEYS = [
    "norm",
    "embed_tokens",
    "lm_head",
    "time_embedder",
    "time_proj",
    "proj_in",
    "proj_out",
    "action_proj",
    "audio_proj",
    "modality_embed",
    "rotary",
]
FP8_TARGET_KEYS = ["layers."]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _component_dir(ckpt_dir, name):
    d = os.path.join(ckpt_dir, name)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"checkpoint dir {ckpt_dir} has no '{name}/' subfolder; expected a diffusers-layout "
            f"Cosmos3 snapshot (e.g. huggingface-cli download nvidia/Cosmos3-Nano)"
        )
    return d


def _shard_files(component_dir):
    files = sorted(glob.glob(os.path.join(component_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors shards in {component_dir}")
    return files


def _from_config(cls, config_path, torch_dtype):
    config = _read_json(config_path)
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


def load_transformer(
    ckpt_dir: str,
    device: torch.device,
    dit_dtype: torch.dtype = torch.bfloat16,
    fp8: bool = False,
    fp8_scaled: bool = False,
    fp8_fast: bool = False,
    lora_weights_list: Optional[List[dict]] = None,
    lora_multipliers: Optional[List[float]] = None,
    dit_path: Optional[str] = None,
):
    """Build Cosmos3OmniTransformer and load weights from <ckpt_dir>/transformer/
    (or a merged single file / directory via dit_path). LoRA is merged during the
    streaming load; fp8_scaled quantizes on the fly and monkey-patches Linears."""
    from .transformer import Cosmos3OmniTransformer

    tdir = dit_path if dit_path else _component_dir(ckpt_dir, "transformer")
    if os.path.isdir(tdir):
        config_path = os.path.join(tdir, "config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(_component_dir(ckpt_dir, "transformer"), "config.json")
        files = _shard_files(tdir)
    else:
        config_path = os.path.join(_component_dir(ckpt_dir, "transformer"), "config.json")
        files = [tdir]

    model = _from_config(Cosmos3OmniTransformer, config_path, dit_dtype)

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
                sd, device, target_layer_keys=FP8_TARGET_KEYS, exclude_layer_keys=FP8_EXCLUDE_KEYS
            )
        else:
            sd = optimize_state_dict_with_fp8_on_the_fly(
                files,
                calc_device=device,
                target_layer_keys=FP8_TARGET_KEYS,
                exclude_layer_keys=FP8_EXCLUDE_KEYS,
                move_to_device=False,
            )
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=fp8_fast)
        info = model.load_state_dict(sd, strict=True, assign=True)
        logger.info(f"fp8-scaled transformer load: {info}")
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
        if fp8:
            # plain e4m3 weight cast for eligible linear weights
            for k in list(sd.keys()):
                if (
                    k.endswith(".weight")
                    and any(t in k for t in FP8_TARGET_KEYS)
                    and not any(e in k for e in FP8_EXCLUDE_KEYS)
                    and sd[k].dtype in (torch.float16, torch.bfloat16, torch.float32)
                ):
                    sd[k] = sd[k].to(torch.float8_e4m3fn)
        else:
            for k in list(sd.keys()):
                if sd[k].dtype not in (torch.float8_e4m3fn,):
                    sd[k] = sd[k].to(dit_dtype)
        info = model.load_state_dict(sd, strict=True, assign=True)
        logger.info(f"transformer load: {info}")

    model.eval().requires_grad_(False)
    return model


def load_vae(ckpt_dir: str, device: torch.device, vae_dtype: torch.dtype = torch.float32, vae_path: Optional[str] = None):
    from .vae import AutoencoderKLWan

    vdir = vae_path if vae_path else _component_dir(ckpt_dir, "vae")
    config_path = os.path.join(vdir if os.path.isdir(vdir) else _component_dir(ckpt_dir, "vae"), "config.json")
    vae = _from_config(AutoencoderKLWan, config_path, vae_dtype)
    files = _shard_files(vdir) if os.path.isdir(vdir) else [vdir]
    sd = _load_sharded_state_dict(files, device="cpu", dtype=vae_dtype)
    info = vae.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"vae load: {info}")
    vae.eval().requires_grad_(False)
    return vae.to(device)


def load_sound_tokenizer(ckpt_dir: str, device: torch.device, dtype: torch.dtype = torch.float32):
    """Returns None if the checkpoint has no sound tokenizer (e.g. Cosmos3-Edge)."""
    from .sound_tokenizer import Cosmos3AVAEAudioTokenizer

    sdir = os.path.join(ckpt_dir, "sound_tokenizer")
    if not os.path.isdir(sdir):
        return None
    tok = _from_config(Cosmos3AVAEAudioTokenizer, os.path.join(sdir, "config.json"), dtype)
    sd = _load_sharded_state_dict(_shard_files(sdir), device="cpu", dtype=dtype)
    info = tok.load_state_dict(sd, strict=False, assign=True)  # encoder keys may be absent
    logger.info(f"sound tokenizer load: {info}")
    tok.eval().requires_grad_(False)
    return tok.to(device)


def load_text_tokenizer(ckpt_dir: str):
    """AutoTokenizer plus the chat template (transformers 4.46 does not auto-load
    a separate chat_template.jinja, so return it for explicit passing)."""
    from transformers import AutoTokenizer

    tok_dir = os.path.join(ckpt_dir, "text_tokenizer")
    if not os.path.isdir(tok_dir):
        tok_dir = ckpt_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        for candidate in (os.path.join(tok_dir, "chat_template.jinja"), os.path.join(ckpt_dir, "chat_template.jinja")):
            if os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    chat_template = f.read()
                tokenizer.chat_template = chat_template
                break
    return tokenizer, chat_template


def load_scheduler(ckpt_dir: str, flow_shift: Optional[float] = None, sigma_max: Optional[float] = None):
    from .scheduler_unipc import UniPCMultistepScheduler

    config = _read_json(os.path.join(_component_dir(ckpt_dir, "scheduler"), "scheduler_config.json"))
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)
    if flow_shift is not None:
        config["flow_shift"] = flow_shift
    if sigma_max is not None:
        config["sigma_max"] = sigma_max
    return UniPCMultistepScheduler(**config)


def load_distilled_scheduler(ckpt_dir: str):
    """FlowMatchEuler scheduler + fixed sigma list for DMD2 4-step checkpoints.
    Returns (scheduler, distilled_sigmas) or (None, None) when not distilled."""
    from .scheduler_flow_euler import FlowMatchEulerDiscreteScheduler

    mmi = os.path.join(ckpt_dir, "modular_model_index.json")
    sigmas = None
    if os.path.exists(mmi):
        data = _read_json(mmi)

        def find_sigmas(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "distilled_sigmas" and isinstance(v, list):
                        return v
                    found = find_sigmas(v)
                    if found is not None:
                        return found
            elif isinstance(node, list):
                for v in node:
                    found = find_sigmas(v)
                    if found is not None:
                        return found
            return None

        sigmas = find_sigmas(data)
    if sigmas is None:
        return None, None
    sched_cfg = _read_json(os.path.join(_component_dir(ckpt_dir, "scheduler"), "scheduler_config.json"))
    if sched_cfg.get("_class_name") == "FlowMatchEulerDiscreteScheduler":
        sched_cfg.pop("_class_name", None)
        sched_cfg.pop("_diffusers_version", None)
        scheduler = FlowMatchEulerDiscreteScheduler(**sched_cfg)
    else:
        scheduler = FlowMatchEulerDiscreteScheduler()
    return scheduler, sigmas


def detect_variant(ckpt_dir: str) -> str:
    """super | nano | edge, best-effort from config.json / folder name."""
    name = os.path.basename(os.path.normpath(ckpt_dir)).lower()
    for v in ("super", "nano", "edge"):
        if v in name:
            return v
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(cfg_path):
        model_type = str(_read_json(cfg_path).get("model_type", ""))
        for v in ("super", "nano", "edge"):
            if v in model_type:
                return v
    return "nano"
