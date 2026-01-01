"""
SageAttention core implementation with UltraViCo support.

Modified from DiT-Extrapolation ultra-wan branch to support:
- Dynamic frame_tokens (resolution-dependent)
- Dynamic training_frames
- Configurable multi_factor (alpha decay)
"""
import torch
import triton
import triton.language as tl

from .quant_per_block import per_block_int8
from .attn_qk_int8_per_block import forward as attn_forward
from .flashattention import forward as fp16_attn
from typing import Optional


def sage_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    xpos_xi: float = 0.9999934149894527,
    flags: Optional[torch.Tensor] = None,
    block_bias: Optional[torch.Tensor] = None,
    sigmoid_a: float = 1.0,
    alpha_xpos_xi: float = 0.97,
    beta_xpos_xi: float = 0.8,
    decay_mask: Optional[torch.Tensor] = None,
    sink_width: int = 4,
    window_width: int = 16,
    multi_factor: Optional[float] = None,
    entropy_factor: Optional[float] = None,
    frame_tokens: int = 1560,
    training_frames: int = 21,
    text_false_length: int = 0,
    **kwargs
) -> torch.Tensor:
    """
    SageAttention with UltraViCo decay support.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor. Shape [batch_size, num_heads, seq_len, head_dim] if HND.

    k : torch.Tensor
        Key tensor. Same shape constraints as q.

    v : torch.Tensor
        Value tensor. Same shape constraints as q.

    tensor_layout : str
        Tensor layout, must be "HND" (batch, heads, seq, dim).

    is_causal : bool
        Whether to apply causal masking. Not supported, must be False.

    sm_scale : Optional[float]
        Softmax scale. If None, uses 1/sqrt(head_dim).

    smooth_k : bool
        Whether to smooth keys by subtracting mean.

    multi_factor : Optional[float]
        UltraViCo decay factor (alpha). When not None, applies decay
        to attention scores for tokens beyond the training window.
        Recommended range: 0.85-0.95. Default: None (no decay).

    frame_tokens : int
        Number of tokens per latent frame. Resolution-dependent.
        For 720x1280: 3600, for 480x854: ~1605.

    training_frames : int
        Training window in latent frames. Default: 21 for Wan.

    Returns
    -------
    torch.Tensor
        Attention output with same shape as q.
    """
    assert tensor_layout == 'HND', f"Only HND layout supported, got {tensor_layout}"
    b, h = q.shape[0], q.shape[1]

    if flags is None:
        flags = torch.zeros([b, h], dtype=torch.long, device=q.device)

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16, torch.float32], \
        "Input tensors must be in dtype of torch.float16, torch.bfloat16, or torch.float32."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    headdim = q.size(-1)
    assert headdim in [64, 96, 128], f"headdim should be in [64, 96, 128], got {headdim}."

    # Assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, \
        "Last dim of qkv must be contiguous."

    seq_dim = 2  # For HND layout

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        k = k - km

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if headdim == 96:
        raise NotImplementedError("headdim=96 not implemented")

    # Quantize Q and K to INT8
    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q, k, sm_scale=sm_scale, tensor_layout=tensor_layout, BLKQ=128, BLKK=128
    )

    if is_causal:
        raise NotImplementedError("Causal attention not implemented")
    else:
        o = attn_forward(
            q_int8, k_int8, v, flags, block_bias, decay_mask, q_scale, k_scale,
            tensor_layout=tensor_layout,
            output_dtype=dtype,
            xpos_xi=xpos_xi,
            frame_tokens=frame_tokens,
            training_frames=training_frames,
            sigmoid_a=sigmoid_a,
            alpha_xpos_xi=alpha_xpos_xi,
            beta_xpos_xi=beta_xpos_xi,
            BLOCK_M=128,
            BLOCK_N=128,
            sink_width=sink_width,
            window_width=window_width,
            multi_factor=multi_factor,
            entropy_factor=entropy_factor,
        )

    return o
