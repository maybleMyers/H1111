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


def convert_peft_lora_to_native(lora_sd: dict, expected_shapes: Optional[dict] = None) -> dict:
    """Convert an ai-toolkit MiniMax-H3 LoRA (PEFT `lora_A`/`lora_B` keys over the original
    fused module names) into `lora_down`/`lora_up` pairs on this port's diffusers-layout keys,
    so the generic merge in `load_safetensors_with_lora_and_fp8` can consume it.

    ai-toolkit vendors the reference model, whose in-memory layout is what
    `_convert_minimax_h3_upstream.py` consumes: `qkv_proj` rows are `[q_all; k_all; v_all]`
    and `fc1` rows are `[gate; up]`, while the port splits QKV into `to_q`/`to_k`/`to_v` and
    stores SwiGLU as `[up; gate]`. Splitting/reordering the `lora_B` rows applies the identical
    transform to the low-rank delta (`delta_W = B @ A`; a row permutation of `delta_W` is the
    same row permutation of `B`). The PEFT format carries no alpha keys and ai-toolkit's
    runtime scale is alpha/rank == 1.0, which matches the merge's `alpha = dim` fallback.

    `expected_shapes` (model key -> weight shape) drops converted pairs that cannot apply to
    this checkpoint. Concretely: LoRAs trained on the pruned single-file variant carry
    `adaln_proj.linear` deltas over its 8-dim `adaln_t_table` temb, which do not exist in the
    full checkpoint's 2688-dim AdaLN input space and would otherwise crash the merge.
    """
    if not any(key.endswith((".lora_A.weight", ".lora_B.weight")) for key in lora_sd):
        return lora_sd

    converted = {}
    for key, value in lora_sd.items():
        if not key.endswith((".lora_A.weight", ".lora_B.weight")):
            converted[key] = value
            continue
        is_up = key.endswith(".lora_B.weight")
        suffix = ".lora_up.weight" if is_up else ".lora_down.weight"
        base = key[: -len(".lora_A.weight")]
        for prefix in ("diffusion_model.", "transformer."):
            if base.startswith(prefix):
                base = base[len(prefix):]
                break
        if base.startswith("token_refiner.blocks."):
            base = "token_refiner.refiner_blocks." + base[len("token_refiner.blocks."):]
        elif base.startswith("blocks."):
            base = "transformer_blocks." + base[len("blocks."):]

        if base.endswith(".attn.qkv_proj"):
            stem = base[: -len("qkv_proj")]
            if is_up:
                for name, part in zip(("to_q", "to_k", "to_v"), value.chunk(3, dim=0)):
                    converted[stem + name + suffix] = part.contiguous()
            else:
                for name in ("to_q", "to_k", "to_v"):
                    converted[stem + name + suffix] = value
        elif base.endswith(".attn.out_proj"):
            converted[base[: -len("out_proj")] + "to_out.0" + suffix] = value
        elif base.endswith(".mlp.fc1"):
            new_base = base[: -len("mlp.fc1")] + "ff.net.0.proj"
            if is_up:
                gate, up = value.chunk(2, dim=0)
                converted[new_base + suffix] = torch.cat([up, gate], dim=0).contiguous()
            else:
                converted[new_base + suffix] = value
        elif base.endswith(".mlp.fc2"):
            converted[base[: -len("mlp.fc2")] + "ff.net.2" + suffix] = value
        else:
            # 1:1 modules: adaln_proj.linear and anything already in diffusers naming.
            converted[base + suffix] = value

    if expected_shapes is not None:
        dropped = []
        for down_key in [k for k in converted if k.endswith(".lora_down.weight")]:
            base = down_key[: -len(".lora_down.weight")]
            up_key = base + ".lora_up.weight"
            down, up = converted[down_key], converted.get(up_key)
            weight_shape = expected_shapes.get(base + ".weight")
            if (
                up is None
                or weight_shape is None
                or len(weight_shape) != 2
                or down.shape[-1] != weight_shape[1]
                or up.shape[0] != weight_shape[0]
            ):
                converted.pop(down_key, None)
                converted.pop(up_key, None)
                dropped.append(base)
        if dropped:
            logger.warning(
                f"Dropped {len(dropped)} LoRA modules whose shapes do not match this "
                f"checkpoint (e.g. {dropped[0]}). LoRAs trained on the pruned single-file "
                f"checkpoint carry adaln_proj deltas over its 8-dim adaln_t_table input, "
                f"which the full checkpoint's 2688-dim AdaLN cannot consume; those modules "
                f"are skipped."
            )
    return converted


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
    int8_use_int_mm: bool = False,
):
    """Build MiniMaxH3Transformer3DModel and load the task's partition: transformer/ for
    t2va/fl2va, transformer_ref/ for ref2va (or an explicit dir / merged file via dit_path).
    LoRA merges during the streaming load; fp8_scaled quantizes on the fly and monkey-patches
    the Linears. The float32 modules of the mixed-precision checkpoint stay float32.

    A single-file `dit_path` in the int8 convrot / adaln-curve pruned export layout
    (detected from the file header) takes the int8 load path instead: keys convert to this
    port's diffusers layout, int8 weights stay int8 with their per-row scales, and the
    quantized Linears are monkey-patched (`int8_use_int_mm` picks torch._int_mm over
    dequantize-per-forward)."""
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

    from .int8_quant import is_int8_checkpoint

    if dit_path and not os.path.isdir(dit_path) and is_int8_checkpoint(dit_path):
        if fp8 or fp8_scaled:
            logger.warning("int8 checkpoint detected: the weights are already quantized, fp8 flags are ignored")
        return _load_int8_transformer(
            config_path=config_path,
            checkpoint_path=dit_path,
            device=device,
            dit_dtype=dit_dtype,
            lora_weights_list=lora_weights_list,
            lora_multipliers=lora_multipliers,
            int8_use_int_mm=int8_use_int_mm,
        )

    exclude_keys = FP8_EXCLUDE_KEYS + (["adaln_proj"] if fp8_exclude_adaln else [])
    model = _from_config(MiniMaxH3Transformer3DModel, config_path, dit_dtype)

    if lora_weights_list:
        expected_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        lora_weights_list = [
            convert_peft_lora_to_native(sd, expected_shapes) for sd in lora_weights_list
        ]

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


def _merge_loras_into_int8_sd(
    sd: dict,
    quant_map: dict,
    lora_weights_list: List[dict],
    lora_multipliers: Optional[List[float]],
    calc_device: torch.device,
) -> None:
    """Merge converted-native LoRAs (`lora_down`/`lora_up` pairs, `scale = alpha/dim` with the
    `alpha = dim` fallback, matching utils/lora_utils) into the converted int8 state dict.
    Quantized layers dequantize + un-rotate to the original basis at float32, take the delta,
    and re-quantize in the same rotated basis — the only added error is the final int8
    rounding. Unquantized layers merge in place at float32."""
    from .int8_quant import marker_groupsize, merge_lora_deltas_into_int8

    if lora_multipliers is None:
        lora_multipliers = [1.0] * len(lora_weights_list)

    consumed = [set() for _ in lora_weights_list]
    merged_layers = 0
    for key in [k for k in sd if k.endswith(".weight")]:
        base = key[: -len(".weight")]
        deltas = []
        for i, (lora_sd, multiplier) in enumerate(zip(lora_weights_list, lora_multipliers)):
            down = lora_sd.get(base + ".lora_down.weight")
            up = lora_sd.get(base + ".lora_up.weight")
            if down is None or up is None:
                continue
            dim = down.shape[0]
            alpha = lora_sd.get(base + ".alpha", dim)
            alpha = float(alpha.item()) if isinstance(alpha, torch.Tensor) else float(alpha)
            delta = (
                up.to(device=calc_device, dtype=torch.float32)
                @ down.to(device=calc_device, dtype=torch.float32)
            ) * (multiplier * alpha / dim)
            deltas.append(delta)
            consumed[i].update({base + ".lora_down.weight", base + ".lora_up.weight", base + ".alpha"})
        if not deltas:
            continue
        if base in quant_map:
            sd[key], sd[base + ".weight_scale"] = merge_lora_deltas_into_int8(
                sd[key], sd[base + ".weight_scale"], marker_groupsize(quant_map[base]), deltas, calc_device
            )
        else:
            weight = sd[key].to(device=calc_device, dtype=torch.float32)
            for delta in deltas:
                weight += delta
            sd[key] = weight.to(device=sd[key].device, dtype=sd[key].dtype)
        merged_layers += 1

    for i, lora_sd in enumerate(lora_weights_list):
        leftover = [k for k in lora_sd if k.endswith((".lora_down.weight", ".lora_up.weight")) and k not in consumed[i]]
        if leftover:
            logger.warning(
                f"LoRA #{i}: {len(leftover)} tensors did not match any model weight (e.g. {leftover[0]})"
            )
    logger.info(f"int8 load: LoRA merged into {merged_layers} layers")


def _load_int8_transformer(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    dit_dtype: torch.dtype,
    lora_weights_list: Optional[List[dict]],
    lora_multipliers: Optional[List[float]],
    int8_use_int_mm: bool,
):
    """Load a single-file MiniMax-H3 export: int8 convrot weights and/or the adaln-curve
    pruned form. Keys convert to the port's diffusers layout via pure renames and row
    permutations (exact for int8 rows + per-row scales), the curve table and the tiny
    curve-form AdaLN projections stay float32 (matching the export's own runtime), and
    quantized Linears are monkey-patched to dequantize per forward (or run torch._int_mm)."""
    from utils.safetensors_utils import MemoryEfficientSafeOpen

    from .int8_quant import (
        apply_int8_monkey_patch,
        convert_int8_dit_state_dict,
        read_safetensors_header,
    )
    from .transformer import MiniMaxH3Transformer3DModel

    config = _read_json(config_path)
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)
    header = read_safetensors_header(checkpoint_path)
    curve = "adaln_t_table" in header
    if curve:
        # the curve table's shape decides the AdaLN geometry; everything else follows config.json
        grid, curve_dim = header["adaln_t_table"]["shape"]
        config["adaln_curve_grid"] = grid
        config["time_embed_dim"] = curve_dim
    with init_empty_weights():
        model = MiniMaxH3Transformer3DModel.from_config(config)
    # int8 tensors cannot carry grads; must be flipped before load_state_dict(assign=True)
    model.requires_grad_(False)

    def stream_tensors():
        with MemoryEfficientSafeOpen(checkpoint_path) as f:
            for key in f.keys():
                yield key, f.get_tensor(key)

    sd, quant_map = convert_int8_dit_state_dict(stream_tensors())

    # dtype policy: int8 weights and their float32 scales stay untouched; the mixed-precision
    # float32 islands (patch projections, output heads) stay float32; the curve table and the
    # curve-form AdaLN projections (stored float16, 8-dim input) are promoted to float32 to
    # match the export runtime's compute; everything else takes the block-stack dtype.
    for key in list(sd.keys()):
        value = sd[key]
        if value.dtype == torch.int8 or key.endswith(".weight_scale"):
            continue
        if _is_fp32_key(key) or key == "adaln_t_table":
            target = torch.float32
        elif curve and (".adaln_proj.linear." in key or key.startswith("norm_out.linear.")):
            target = torch.float32
        else:
            target = dit_dtype
        if value.dtype != target:
            sd[key] = value.to(target)

    if lora_weights_list:
        expected_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        lora_weights_list = [convert_peft_lora_to_native(l, expected_shapes) for l in lora_weights_list]
        _merge_loras_into_int8_sd(sd, quant_map, lora_weights_list, lora_multipliers, device)

    apply_int8_monkey_patch(model, quant_map, use_int_mm=int8_use_int_mm, state_dict=sd)
    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(
        f"int8 transformer load ({os.path.basename(checkpoint_path)}, "
        f"{len(quant_map)} quantized layers, curve_adaln={curve}): {info}"
    )
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
