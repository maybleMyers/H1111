# Attention dispatch + mixin shims for the vendored MiniMax-H3 model code.
# Copied from cosmos_engine/cosmos_video/attention.py.
#
# Replaces diffusers-main `diffusers.models.attention_dispatch.dispatch_attention_fn` and the
# `AttentionMixin` / `AttentionModuleMixin` classes, neither of which exist in the pinned
# diffusers==0.33.1. Tensor layout contract (matches diffusers-main dispatch and the call sites in
# minimax_video/transformer.py): query/key/value are [batch, seq, heads, head_dim] and the output is
# returned in the same [batch, seq, heads, head_dim] layout.

import inspect

import torch
import torch.nn.functional as F

from .compile_config import maybe_compile

_VALID_BACKENDS = ("torch", "sdpa", "flash", "flashattn", "flash2", "flash3", "sageattn", "xformers")

# Module-level default backend, used when dispatch_attention_fn is called with backend=None.
_ATTENTION_BACKEND = "torch"


def set_attention_backend(mode: str) -> None:
    """Set the module-level default attention backend for Cosmos3 attention dispatch."""
    global _ATTENTION_BACKEND
    if mode not in _VALID_BACKENDS:
        raise ValueError(f"Unknown attention backend {mode!r}. Valid options: {_VALID_BACKENDS}")
    _ATTENTION_BACKEND = mode


def get_attention_backend() -> str:
    return _ATTENTION_BACKEND


def _repeat_kv(key: torch.Tensor, value: torch.Tensor, num_q_heads: int, heads_dim: int):
    """Repeat KV heads along `heads_dim` so KV head count matches the query head count (manual GQA)."""
    num_kv_heads = key.shape[heads_dim]
    if num_kv_heads == num_q_heads:
        return key, value
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"Query heads ({num_q_heads}) must be a multiple of KV heads ({num_kv_heads}) for GQA."
        )
    n_rep = num_q_heads // num_kv_heads
    return key.repeat_interleave(n_rep, dim=heads_dim), value.repeat_interleave(n_rep, dim=heads_dim)


@maybe_compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def _sdpa_core(q, k, v, attn_mask, dropout_p, is_causal):
    """Compiled SDPA core ([B, H, S, D] layout), used by the non-GQA paths."""
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
    )


def _sdpa_attention(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa):
    # [B, S, H, D] -> [B, H, S, D]
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    if enable_gqa and k.shape[1] != q.shape[1]:
        try:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, enable_gqa=True
            )
        except TypeError:
            # Older torch without enable_gqa kwarg: repeat KV heads manually.
            k, v = _repeat_kv(k, v, q.shape[1], heads_dim=1)
            out = _sdpa_core(q, k, v, attn_mask, dropout_p, is_causal)
    else:
        out = _sdpa_core(q, k, v, attn_mask, dropout_p, is_causal)
    return out.transpose(1, 2)


def _flash_attention(query, key, value, dropout_p, is_causal, backend):
    flash_attn_func = None
    if backend == "flash3":
        try:
            from flash_attn_interface import flash_attn_func  # FlashAttention-3
        except ImportError:
            flash_attn_func = None
    if flash_attn_func is None:
        try:
            from flash_attn import flash_attn_func
        except ImportError:
            raise RuntimeError(
                "Attention backend requires the `flash-attn` package (`pip install flash-attn`), "
                "but it is not installed."
            )
    # flash_attn expects [B, S, H, D] and supports GQA natively (kv heads may differ).
    out = flash_attn_func(query, key, value, dropout_p=dropout_p, causal=is_causal)
    if isinstance(out, tuple):  # flash_attn_interface may return (out, lse)
        out = out[0]
    return out


def _sage_attention(query, key, value, dropout_p, is_causal, enable_gqa):
    try:
        from sageattention import sageattn
    except ImportError:
        raise RuntimeError(
            "Attention backend 'sageattn' requires the `sageattention` package "
            "(`pip install sageattention`), but it is not installed."
        )
    params = inspect.signature(sageattn).parameters
    if is_causal and "is_causal" not in params:
        # This sageattention version cannot do causal masking; fall back to SDPA for correctness.
        return _sdpa_attention(query, key, value, None, dropout_p, is_causal, enable_gqa)
    # sageattention has no GQA support: repeat KV heads manually ([B, S, H, D] -> heads dim 2).
    k, v = _repeat_kv(key, value, query.shape[2], heads_dim=2)
    kwargs = {}
    if "is_causal" in params:
        kwargs["is_causal"] = is_causal
    if "tensor_layout" in params:
        return sageattn(query, k, v, tensor_layout="NHD", **kwargs)
    # Older versions only accept [B, H, S, D].
    out = sageattn(query.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), **kwargs)
    return out.transpose(1, 2)


def _xformers_attention(query, key, value, dropout_p, is_causal):
    try:
        import xformers.ops as xops
    except ImportError:
        raise RuntimeError(
            "Attention backend 'xformers' requires the `xformers` package (`pip install xformers`), "
            "but it is not installed."
        )
    # xformers expects [B, S, H, D]; repeat KV heads manually for GQA.
    k, v = _repeat_kv(key, value, query.shape[2], heads_dim=2)
    attn_bias = xops.LowerTriangularMask() if is_causal else None
    return xops.memory_efficient_attention(query, k, v, attn_bias=attn_bias, p=dropout_p)


def dispatch_attention_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    enable_gqa: bool = False,
    backend: str | None = None,
    parallel_config=None,
) -> torch.Tensor:
    """Attention with layout [batch, seq, heads, head_dim] for q/k/v and the returned output.

    `backend=None` uses the module-level default set via `set_attention_backend` ("torch" = SDPA).
    """
    if parallel_config is not None:
        raise NotImplementedError("parallel_config is not supported by the H1111 Cosmos3 attention dispatch.")
    if backend is None:
        backend = _ATTENTION_BACKEND
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Unknown attention backend {backend!r}. Valid options: {_VALID_BACKENDS}")

    if backend in ("torch", "sdpa"):
        return _sdpa_attention(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa)

    if attn_mask is not None:
        raise ValueError(f"Attention backend {backend!r} does not support an explicit attn_mask.")

    if backend in ("flash", "flashattn", "flash2", "flash3"):
        return _flash_attention(query, key, value, dropout_p, is_causal, backend)
    if backend == "sageattn":
        return _sage_attention(query, key, value, dropout_p, is_causal, enable_gqa)
    if backend == "xformers":
        return _xformers_attention(query, key, value, dropout_p, is_causal)
    raise ValueError(f"Unhandled attention backend {backend!r}.")


class AttentionModuleMixin:
    """Minimal shim of diffusers-main AttentionModuleMixin: processor storage only."""

    _default_processor_cls = None
    _available_processors = []
    _supports_qkv_fusion = False
    _attention_backend = None
    _parallel_config = None

    def set_processor(self, processor) -> None:
        # If the current processor is an nn.Module but the new one is not, drop the submodule entry
        # so it does not linger in the module tree / state dict.
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            self._modules.pop("processor")
        self.processor = processor

    def get_processor(self):
        return self.processor


class AttentionMixin:
    """Minimal shim of diffusers-main AttentionMixin: per-model attention backend selection."""

    def set_attention_backend(self, backend: str) -> None:
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"Unknown attention backend {backend!r}. Valid options: {_VALID_BACKENDS}")
        for module in self.modules():
            if isinstance(module, AttentionModuleMixin):
                processor = getattr(module, "processor", None)
                if processor is not None:
                    processor._attention_backend = backend

    def reset_attention_backend(self) -> None:
        for module in self.modules():
            if isinstance(module, AttentionModuleMixin):
                processor = getattr(module, "processor", None)
                if processor is not None:
                    processor._attention_backend = None
