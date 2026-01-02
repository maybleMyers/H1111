"""
UltraViCo-enabled INT8 attention kernel.

Modified from DiT-Extrapolation ultra-wan branch to support:
- Dynamic frame_tokens (resolution-dependent)
- Dynamic training_frames
- Configurable multi_factor (alpha decay)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, current_flag,
                    K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                    Block_bias_ptrs, stride_bbz, stride_bbh, stride_bm, stride_bn,
                    Decay_mask_ptrs, stride_dmz, stride_dmh, stride_dm, stride_dn,
                    start_m,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    xpos_xi: tl.constexpr = 0.9999934149894527,
                    window_th: tl.constexpr = 16380,  # frame_tokens * training_frames / 2 (dynamic)
                    sigmoid_a: tl.constexpr = 1.0,
                    alpha_xpos_xi: tl.constexpr = 0.9999967941742395,
                    beta_xpos_xi: tl.constexpr = 0.9999860536252945,
                    sink_width: tl.constexpr = 4,
                    window_width: tl.constexpr = 16,
                    multi_factor: tl.constexpr = None,
                    entropy_factor: tl.constexpr = None,
                    frame_tokens: tl.constexpr = 1560,
                    training_frames: tl.constexpr = 21,
                    suppress_harmonics: tl.constexpr = False,
                    harmonic_beta: tl.constexpr = 0.6,
                    harmonic_gamma: tl.constexpr = 4,
                    ):

    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_ptrs, mask = k_mask)
        k_scale = tl.load(K_scale_ptr)


        m = offs_m[:, None]
        n = start_n + offs_n

        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale

        # UltraViCo: Apply decay to tokens beyond training window (only if multi_factor is set)
        # window_th is computed dynamically as frame_tokens * training_frames / 2
        if multi_factor is not None:
            dist2 = tl.abs(m - n).to(tl.int32)
            dist_mask = dist2 <= window_th

            negative_mask = (qk < 0)

            # Apply decay factor to out-of-window tokens
            qk = tl.where(dist_mask | negative_mask, qk, qk * multi_factor)

            # Harmonic suppression: apply stronger decay (beta) at harmonic positions
            # Harmonics occur at multiples of training_frames * frame_tokens
            if suppress_harmonics:
                harmonic_period = training_frames * frame_tokens
                harmonic_gamma_tokens = harmonic_gamma * frame_tokens
                # Check distance to nearest harmonic (1x, 2x, 3x, 4x training length)
                # We check up to 4 harmonics which covers up to 4x extrapolation
                for harmonic_mult in range(1, 5):
                    harmonic_center = harmonic_mult * harmonic_period
                    near_harmonic = (dist2 >= harmonic_center - harmonic_gamma_tokens) & \
                                   (dist2 <= harmonic_center + harmonic_gamma_tokens)
                    # Apply beta (stronger decay) to harmonic risk positions that are out of window
                    harmonic_risk = near_harmonic & (~dist_mask) & (~negative_mask)
                    qk = tl.where(harmonic_risk, qk * (harmonic_beta / multi_factor), qk)

            # Additional masking for extreme positions (prevent attention to very distant future)
            window3 = (m <= frame_tokens) & (n > training_frames * frame_tokens)
            qk = tl.where(window3, -1e4, qk)


        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]

        v = tl.load(V_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
        p = p.to(tl.float16)

        acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i

@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out,
              Block_bias, Decay_mask,
              flags, stride_f_b, stride_f_h,
              stride_qz, stride_qh, stride_qn,
              stride_kz, stride_kh, stride_kn,
              stride_vz, stride_vh, stride_vn,
              stride_oz, stride_oh, stride_on,
              stride_bbz, stride_bbh, stride_bm, stride_bn,
              stride_dmz, stride_dmh, stride_dm, stride_dn,
              qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr,
              xpos_xi: tl.constexpr = 0.9999934149894527,
              window_th: tl.constexpr = 16380,  # Dynamic: frame_tokens * training_frames / 2
              sigmoid_a: tl.constexpr = 1.0,
              alpha_xpos_xi: tl.constexpr = 0.9999967941742395,
              beta_xpos_xi: tl.constexpr = 0.9999860536252945,
              sink_width: tl.constexpr = 4,
              window_width: tl.constexpr = 16,
              multi_factor: tl.constexpr = None,
              entropy_factor: tl.constexpr = None,
              frame_tokens: tl.constexpr = 1560,
              training_frames: tl.constexpr = 21,
              suppress_harmonics: tl.constexpr = False,
              harmonic_beta: tl.constexpr = 0.6,
              harmonic_gamma: tl.constexpr = 4,
              ):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)

    flag_ptr = flags + off_z * stride_f_b + off_h * stride_f_h
    current_flag = tl.load(flag_ptr)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]

    # Block bias pointers (unused in this simplified version)
    Block_bias_ptrs = Block_bias + off_z * stride_bbz + off_h * stride_bbh

    # Decay mask pointers (unused in this simplified version)
    Decay_mask_ptrs = Decay_mask + off_z * stride_dmz + off_h * stride_dmh

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, current_flag, K_ptrs, K_scale_ptr, V_ptrs,
                                stride_kn, stride_vn,
                                Block_bias_ptrs, stride_bbz, stride_bbh, stride_bm, stride_bn,
                                Decay_mask_ptrs, stride_dmz, stride_dmh, stride_dm, stride_dn,
                                start_m,
                                BLOCK_M, HEAD_DIM, BLOCK_N,
                                4 - STAGE, offs_m, offs_n,
                                xpos_xi=xpos_xi,
                                window_th=window_th,
                                sigmoid_a=sigmoid_a,
                                alpha_xpos_xi=alpha_xpos_xi,
                                beta_xpos_xi=beta_xpos_xi,
                                sink_width=sink_width,
                                window_width=window_width,
                                multi_factor=multi_factor,
                                entropy_factor=entropy_factor,
                                frame_tokens=frame_tokens,
                                training_frames=training_frames,
                                suppress_harmonics=suppress_harmonics,
                                harmonic_beta=harmonic_beta,
                                harmonic_gamma=harmonic_gamma,
                                )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

def forward(q, k, v, flags, block_bias, decay_mask, q_scale, k_scale, tensor_layout="HND", output_dtype=torch.float16,
              xpos_xi: float = 0.9999934149894527,
              frame_tokens: int = 1560,
              training_frames: int = 21,
              sigmoid_a: float = 1.0,
              alpha_xpos_xi: float = 0.9999967941742395,
              beta_xpos_xi: float = 0.9999860536252945,
              BLOCK_M: int = 32,  # Reduced for GPU shared memory compatibility (101KB limit)
              BLOCK_N: int = 32,  # Reduced for GPU shared memory compatibility (101KB limit)
              sink_width: int = 4,
              window_width: int = 16,
              multi_factor: float = None,
              entropy_factor: float = None,
              suppress_harmonics: bool = False,
              harmonic_beta: float = 0.6,
              harmonic_gamma: int = 4,
              ):
    """
    Forward pass for UltraViCo-enabled INT8 attention.

    Args:
        q, k, v: Query, Key, Value tensors (INT8 for q, k; FP16 for v)
        flags: Control flags tensor
        block_bias, decay_mask: Optional bias tensors (can be None)
        q_scale, k_scale: Quantization scales
        tensor_layout: "HND" (batch, heads, seq, dim) format
        output_dtype: Output tensor dtype
        frame_tokens: Tokens per latent frame (resolution-dependent)
        training_frames: Training window in latent frames (default: 21 for Wan)
        multi_factor: UltraViCo decay factor (alpha), e.g., 0.9
        suppress_harmonics: Whether to apply stronger decay at harmonic positions
        harmonic_beta: Decay factor for harmonic risk positions (stronger than alpha)
        harmonic_gamma: Number of frames around harmonic peaks to suppress

    Returns:
        Output attention tensor
    """
    stage = 1

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    b, h_qo, qo_len, head_dim = q.shape
    if block_bias is None:
        block_bias = torch.zeros((b, h_qo, (qo_len + BLOCK_M - 1) // BLOCK_M, (qo_len + BLOCK_N - 1) // BLOCK_N), dtype=torch.float16, device=q.device)

    if decay_mask is None:
        decay_mask = torch.zeros((b, h_qo, (qo_len + BLOCK_M - 1) // BLOCK_M, (qo_len + BLOCK_N - 1) // BLOCK_N), dtype=torch.bool, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
        stride_bbz, stride_bbh, stride_bm, stride_bn = block_bias.stride()
        stride_dmz, stride_dmh, stride_dm, stride_dn = decay_mask.stride()
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    stride_f_b, stride_f_h = flags.stride()

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    # Compute window threshold dynamically
    window_th = int(frame_tokens * training_frames / 2)

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o,
        block_bias, decay_mask,
        flags,
        stride_f_b, stride_f_h,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_v, stride_h_v, stride_seq_v,
        stride_bz_o, stride_h_o, stride_seq_o,
        stride_bbz, stride_bbh, stride_bm, stride_bn,
        stride_dmz, stride_dmh, stride_dm, stride_dn,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,
        STAGE=stage,
        num_warps=2,  # Reduced for shared memory compatibility (101KB limit)
        num_stages=1,  # Single stage to minimize shared memory usage
        xpos_xi=xpos_xi,
        window_th=window_th,
        sigmoid_a=sigmoid_a,
        alpha_xpos_xi=alpha_xpos_xi,
        beta_xpos_xi=beta_xpos_xi,
        sink_width=sink_width,
        window_width=window_width,
        multi_factor=multi_factor,
        entropy_factor=entropy_factor,
        frame_tokens=frame_tokens,
        training_frames=training_frames,
        suppress_harmonics=suppress_harmonics,
        harmonic_beta=harmonic_beta,
        harmonic_gamma=harmonic_gamma,
        )
    return o
