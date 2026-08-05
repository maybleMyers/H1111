# INT8 "ConvRot" checkpoint support (the `int8_tensorwise` single-file export format).
#
# Storage per quantized Linear: `<layer>.weight` int8 [out, in], `<layer>.weight_scale`
# float32 [out, 1] (amax/127 per row), and a `<layer>.comfy_quant` uint8 JSON marker, e.g.
# `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`. When `convrot`
# is set, the weight was rotated *before* quantization by a block-diagonal regular Hadamard
# over groups of `convrot_groupsize` input channels: `W_rot = grouped(W) @ H^T`. H is
# symmetric and orthogonal, so the same grouped transform inverts itself: dequantization is
# `grouped(q * scale) @ H^T` again.
#
# Two runtime paths, mirroring the fp8 monkey patch split:
#   * default  — dequantize the weight per forward at float32 (un-rotating it back to the
#     original basis), cast to the activation dtype, and run a plain F.linear. Weight-only
#     quantization error, no activation quantization.
#   * int8 mm  — keep the weight rotated, rotate the activation online, quantize it per row
#     and use torch._int_mm (how the export's own runtime computes). Faster, adds
#     activation error.
import json
import logging
import re
from typing import Iterable, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

INT8_TENSORWISE_FORMAT = "int8_tensorwise"
# the export format's per-layer marker tensor is literally named `<layer>.comfy_quant`
QUANT_MARKER_SUFFIX = ".comfy_quant"
WEIGHT_SCALE_SUFFIX = ".weight_scale"

_HADAMARD_CACHE: dict = {}


# ---------------------------------------------------------------------------- rotation math


def build_regular_hadamard(size: int, device="cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalized *regular* Hadamard matrix, the ConvRot construction: Kronecker powers of a
    symmetric 4x4 seed, scaled by `size**-0.5`. Symmetric + orthogonal, hence involutory —
    applying the rotation twice is the identity, which is what makes de-rotation the same
    grouped matmul as rotation."""
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached

    if size < 4 or (size & (size - 1)) != 0 or (size.bit_length() - 1) % 2 != 0:
        raise ValueError(f"regular Hadamard size must be a power of 4, got {size}")

    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    h = h / (size**0.5)
    _HADAMARD_CACHE[key] = h
    return h


def rotate_weight_rows(mat: torch.Tensor, group_size: int) -> torch.Tensor:
    """`grouped(mat) @ H^T` over the last dimension. Used both to rotate (before
    quantization) and to un-rotate (after dequantization)."""
    out_f, in_f = mat.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by convrot group size {group_size}")
    h = build_regular_hadamard(group_size, device=mat.device, dtype=mat.dtype)
    rotated = mat.reshape(out_f, in_f // group_size, group_size) @ h.T
    return rotated.reshape(out_f, in_f)


def rotate_activation(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """`grouped(x) @ H` over the last dimension (the online activation-side rotation)."""
    features = x.shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by convrot group size {group_size}")
    h = build_regular_hadamard(group_size, device=x.device, dtype=x.dtype)
    rotated = x.reshape(*x.shape[:-1], features // group_size, group_size) @ h
    return rotated.reshape(x.shape)


# ---------------------------------------------------------------------------- quant / dequant


def quantize_int8_rowwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric int8: `scale = amax/127` (float32, clamped), `q = round(w/scale)`."""
    scale = w.abs().amax(dim=-1, keepdim=True).float().div_(127.0).clamp_(min=1e-30)
    q = torch.round(w / scale.to(w.dtype)).clamp_(-128, 127).to(torch.int8)
    return q, scale


def quantize_int8_convrot_weight(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate then quantize — the storage form of a convrot layer (used to re-quantize after
    a LoRA merge, at float32 for the rotation and scale math)."""
    w = w.to(torch.float32)
    if group_size:
        w = rotate_weight_rows(w, group_size)
    return quantize_int8_rowwise(w)


def dequantize_int8_weight(
    q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 0,
    out_dtype: torch.dtype = torch.float32,
    row_chunk: int = 8192,
) -> torch.Tensor:
    """Dequantize (and un-rotate, if `group_size`) back to the original basis. Math runs at
    float32 in row chunks so the transient never exceeds ~2 chunk-sized float32 buffers."""
    out = torch.empty(q.shape, dtype=out_dtype, device=q.device)
    per_row = scale.numel() == q.shape[0]
    scale = scale.reshape(-1, 1) if per_row else scale.reshape(())
    for start in range(0, q.shape[0], row_chunk):
        stop = min(start + row_chunk, q.shape[0])
        s = scale[start:stop] if per_row else scale
        w = q[start:stop].to(torch.float32) * s.to(torch.float32)
        if group_size:
            w = rotate_weight_rows(w, group_size)
        out[start:stop] = w.to(out_dtype)
    return out


# ---------------------------------------------------------------------------- marker parsing


def parse_quant_marker(marker: torch.Tensor) -> dict:
    """The per-layer quant marker is a uint8 tensor holding a JSON object."""
    info = json.loads(bytes(marker.cpu().to(torch.uint8).tolist()).decode("utf-8"))
    if "format" not in info:
        raise ValueError(f"quant marker without 'format': {info}")
    return info


def marker_groupsize(info: dict) -> int:
    """0 when the layer is not rotated; the Hadamard group size otherwise."""
    if info.get("format") != INT8_TENSORWISE_FORMAT:
        raise ValueError(f"unsupported quant format {info.get('format')!r} (only {INT8_TENSORWISE_FORMAT})")
    if info.get("convrot", False):
        return int(info.get("convrot_groupsize", 256))
    return 0


def collect_quant_markers(sd: dict) -> dict[str, dict]:
    """Pop every per-layer quant marker tensor out of `sd`; return {layer_path: marker dict}."""
    markers = {}
    for key in [k for k in sd if k.endswith(QUANT_MARKER_SUFFIX)]:
        markers[key[: -len(QUANT_MARKER_SUFFIX)]] = parse_quant_marker(sd.pop(key))
    return markers


# ---------------------------------------------------------------------------- runtime patch


# Row-chunk budget for the int8 GEMM's int32 accumulator. The full accumulator would be
# [seq_len, out_features] int32 — ~11 GB for the 28672-wide fc1 on a 100k-row packed
# sequence — and the float32 scaling epilogue would transiently double that. Chunking caps
# both while the (much smaller) int8/output tensors stay whole.
_INT8_MM_ACC_BYTES = 256 * 1024 * 1024


def _int8_mm_chunked(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """`(xq @ weight_t) * (xs * weight_scale)` in row chunks: the int32 accumulator and its
    float32 scaling exist only chunk-at-a-time; results land directly in `out_dtype`."""
    m = xq.shape[0]
    n = weight_t.shape[1]
    ws = weight_scale.reshape(1, -1).to(torch.float32)
    out = torch.empty(m, n, dtype=out_dtype, device=xq.device)
    chunk = max(32, _INT8_MM_ACC_BYTES // (n * 4))
    start = 0
    while start < m:
        stop = min(start + chunk, m)
        if 0 < m - stop <= 16:  # torch._int_mm needs > 16 rows; fold a tiny tail in
            stop = m
        acc = torch._int_mm(xq[start:stop], weight_t)
        out[start:stop] = (acc.to(torch.float32) * (xs[start:stop].to(torch.float32) * ws)).to(out_dtype)
        del acc
        start = stop
    return out


def _int8_mm_2d(
    x2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """True int8 GEMM: rotate + row-quantize the activation, `torch._int_mm`, scale back by
    `x_scale * weight_scale`. Returns None when the shape constraints of the cuBLAS IMMA
    path are not met (caller falls back to the dequant path)."""
    m, k = x2d.shape
    n = weight.shape[0]
    if x2d.device.type != "cuda" or m <= 16 or k % 8 != 0 or n % 8 != 0:
        return None
    if group_size:
        x2d = rotate_activation(x2d, group_size)
    xq, xs = quantize_int8_rowwise(x2d)
    del x2d
    return _int8_mm_chunked(xq, xs, weight.t().contiguous(), weight_scale, out_dtype)


def int8_linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    group_size = self.int8_convrot_groupsize
    if self.int8_use_int_mm:
        out = _int8_mm_2d(x.reshape(-1, x.shape[-1]), self.weight, self.weight_scale, group_size, x.dtype)
        if out is not None:
            out = out.reshape(*x.shape[:-1], self.weight.shape[0])
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return out
    w = dequantize_int8_weight(self.weight, self.weight_scale, group_size, out_dtype=x.dtype)
    return F.linear(x, w, self.bias)


def int8_embedding_forward(self: nn.Embedding, input: torch.Tensor) -> torch.Tensor:
    # Row gather first, then dequantize only the gathered rows. ConvRot embeddings would
    # need an un-rotate here; none of the MiniMax-H3 exports rotate the embedding table, so
    # it is rejected at patch time.
    rows = F.embedding(input, self.weight).to(torch.float32)
    if self.weight_scale.dim() >= 2:
        rows = rows * F.embedding(input, self.weight_scale).to(torch.float32)
    else:
        rows = rows * self.weight_scale.to(torch.float32)
    return rows.to(self.int8_output_dtype)


def apply_int8_monkey_patch(
    model: nn.Module,
    quant_layers: dict[str, dict],
    use_int_mm: bool = False,
    embedding_output_dtype: torch.dtype = torch.bfloat16,
    state_dict: Optional[dict] = None,
) -> int:
    """Prepare `model` for an int8 state dict: for every layer path in `quant_layers`
    (mapping module path -> quant marker dict), register the `weight_scale` buffer the
    checkpoint provides and bind the int8 forward. Call BEFORE `load_state_dict(assign=True)`
    (and make sure `requires_grad_(False)` ran first — int8 tensors cannot carry grads).
    `state_dict` (when given) sizes each scale buffer from the checkpoint's own
    `<layer>.weight_scale` — scales are `[out, 1]` for Linears but scalar for the tensor-wise
    embedding table, and load_state_dict rejects size mismatches even with strict=False."""
    patched = 0
    for path, info in quant_layers.items():
        module = model.get_submodule(path)
        group_size = marker_groupsize(info)
        scale_key = path + WEIGHT_SCALE_SUFFIX
        if state_dict is not None and scale_key in state_dict:
            scale_shape = state_dict[scale_key].shape
        elif isinstance(module, nn.Linear):
            scale_shape = (module.out_features, 1)
        else:
            scale_shape = ()
        if isinstance(module, nn.Linear):
            module.register_buffer("weight_scale", torch.empty(scale_shape, dtype=torch.float32))
            module.int8_convrot_groupsize = group_size
            module.int8_use_int_mm = use_int_mm
            module.forward = int8_linear_forward.__get__(module, type(module))
        elif isinstance(module, nn.Embedding):
            if group_size:
                raise ValueError(f"{path}: convrot embeddings are not supported (no GEMM to fold the un-rotate into)")
            module.register_buffer("weight_scale", torch.empty(scale_shape, dtype=torch.float32))
            module.int8_output_dtype = embedding_output_dtype
            module.forward = int8_embedding_forward.__get__(module, type(module))
        else:
            raise ValueError(f"{path}: quant marker on unsupported module type {type(module).__name__}")
        patched += 1
    logger.info(f"int8 monkey patch applied to {patched} modules ({'int8 mm' if use_int_mm else 'dequant'} path)")
    return patched


# ---------------------------------------------------------------------------- LoRA merge


def merge_lora_deltas_into_int8(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    deltas: Iterable[torch.Tensor],
    calc_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge already-scaled LoRA deltas into a quantized layer: dequantize + un-rotate to the
    original basis at float32, add, re-rotate and re-quantize. The only added error over the
    shipped checkpoint is the final int8 rounding."""
    original_device = weight.device
    w = dequantize_int8_weight(weight.to(calc_device), weight_scale.to(calc_device), group_size, torch.float32)
    for delta in deltas:
        w += delta.to(device=calc_device, dtype=torch.float32)
    q, s = quantize_int8_convrot_weight(w, group_size)
    return q.to(original_device), s.to(original_device)


# ---------------------------------------------------------------------------- DiT conversion
#
# The int8 exports keep the upstream reference's *in-memory* naming: fused `qkv_proj` with
# rows already de-interleaved to `[q_all; k_all; v_all]` (the export's runtime splits with a
# plain `.split(heads*head_dim)`), and fused `mlp.fc1` with rows `[gate; value]` (its swiglu
# is `silu(first_half) * second_half`). The port splits QKV into `to_q/to_k/to_v` and stores
# SwiGLU as `[value; gate]` (diffusers order), so QKV converts by a plain 3-way row split and
# fc1 by swapping the two row halves — pure row permutations, which apply identically to the
# int8 rows and their per-row scales with no requantization. NOTE: the *raw* upstream shards
# are per-head interleaved and need `_convert_minimax_h3_upstream.reorder_interleaved_qkv`;
# these single-file exports do NOT.

_DIT_TOP_RENAMES = {
    "video_patch_proj.weight": "proj_in.weight",
    "video_patch_proj.bias": "proj_in.bias",
    "audio_patch_proj.weight": "audio_proj_in.weight",
    "audio_patch_proj.bias": "audio_proj_in.bias",
    "condition_proj.weight": "context_embedder.weight",
    "condition_proj.bias": "context_embedder.bias",
    "final_layer.norm.weight": "norm_out.norm.weight",
    "final_layer.adaln_proj.linear.weight": "norm_out.linear.weight",
    "final_layer.adaln_proj.linear.bias": "norm_out.linear.bias",
    "final_layer.video_out.weight": "proj_out.weight",
    "final_layer.video_out.bias": "proj_out.bias",
    "final_layer.audio_out.weight": "audio_proj_out.weight",
    "final_layer.audio_out.bias": "audio_proj_out.bias",
    "token_refiner.final_norm.weight": "token_refiner.final_norm.weight",
    "adaln_t_table": "adaln_t_table",
}

# `rope.inv_freq` is recomputed by MiniMaxH3RotaryPosEmbed (bitwise equal, see the upstream
# converter); time_embedder keys cannot occur in a curve-form checkpoint.
_DIT_DROPPED_KEYS = ("rope.inv_freq",)

_DIT_BLOCK_PREFIXES = (
    ("blocks.", "transformer_blocks."),
    ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
)

_DIT_BLOCK_RENAMES = {
    "attn.q_norm.weight": "attn.norm_q.weight",
    "attn.k_norm.weight": "attn.norm_k.weight",
    "attn.out_proj": "attn.to_out.0",
    "mlp.fc2": "ff.net.2",
    "norm1.weight": "norm1.weight",
    "norm2.weight": "norm2.weight",
    "adaln_proj.linear.weight": "adaln_proj.linear.weight",
    "adaln_proj.linear.bias": "adaln_proj.linear.bias",
}


def _split_block_key(key: str) -> Optional[tuple[str, str]]:
    """`blocks.7.attn.out_proj.weight` -> (`transformer_blocks.7.`, `attn.out_proj.weight`)."""
    for source_prefix, target_prefix in _DIT_BLOCK_PREFIXES:
        match = re.match(re.escape(source_prefix) + r"(\d+)\.", key)
        if match:
            return f"{target_prefix}{match.group(1)}.", key[match.end():]
    return None


def convert_int8_dit_tensor(key: str, value: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """Convert one single-file-export DiT tensor (weight, weight_scale or quant marker) to
    the port's diffusers-layout key(s). Returns [] for dropped keys. Raises on unknown keys so
    a layout drift fails loudly instead of silently missing weights."""
    if key in _DIT_DROPPED_KEYS:
        return []
    if key in _DIT_TOP_RENAMES:
        return [(_DIT_TOP_RENAMES[key], value)]

    split = _split_block_key(key)
    if split is None:
        raise KeyError(f"unrecognized MiniMax-H3 single-file checkpoint key: {key}")
    target_block, rest = split

    if rest in _DIT_BLOCK_RENAMES:
        return [(target_block + _DIT_BLOCK_RENAMES[rest], value)]

    stem, _, leaf = rest.rpartition(".")  # e.g. ("attn.qkv_proj", ".", "weight")
    if leaf not in ("weight", "weight_scale", "comfy_quant"):
        raise KeyError(f"unrecognized MiniMax-H3 single-file checkpoint key: {key}")

    if stem == "attn.qkv_proj":
        names = ("attn.to_q", "attn.to_k", "attn.to_v")
        if leaf == "comfy_quant":
            return [(f"{target_block}{name}.{leaf}", value) for name in names]
        parts = value.chunk(3, dim=0)
        return [(f"{target_block}{name}.{leaf}", part.contiguous()) for name, part in zip(names, parts)]
    if stem == "attn.out_proj":
        return [(f"{target_block}attn.to_out.0.{leaf}", value)]
    if stem == "mlp.fc1":
        if leaf == "comfy_quant":
            return [(f"{target_block}ff.net.0.proj.{leaf}", value)]
        gate, up = value.chunk(2, dim=0)
        return [(f"{target_block}ff.net.0.proj.{leaf}", torch.cat([up, gate], dim=0))]
    if stem == "mlp.fc2":
        return [(f"{target_block}ff.net.2.{leaf}", value)]
    raise KeyError(f"unrecognized MiniMax-H3 single-file checkpoint key: {key}")


def convert_int8_dit_state_dict(tensors: Iterator[tuple[str, torch.Tensor]]) -> tuple[dict, dict[str, dict]]:
    """Stream-convert a single-file MiniMax-H3 export into (native state dict, quant map).
    The quant map holds {native module path: quant marker dict}."""
    sd: dict[str, torch.Tensor] = {}
    for key, value in tensors:
        for native_key, tensor in convert_int8_dit_tensor(key, value):
            sd[native_key] = tensor
    markers = collect_quant_markers(sd)
    return sd, markers


def read_safetensors_header(path: str) -> dict:
    """Header-only read: {key: {dtype, shape, ...}} without touching tensor data."""
    import struct

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header


def is_int8_checkpoint(path: str) -> bool:
    """True when `path` is a single-file MiniMax-H3 export (int8 and/or adaln-curve pruned
    form) that must go through the conversion path."""
    import os

    if not path or not path.endswith(".safetensors") or not os.path.isfile(path):
        return False
    header = read_safetensors_header(path)
    return "adaln_t_table" in header or any(k.endswith(QUANT_MARKER_SUFFIX) for k in header)
