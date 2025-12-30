# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Adapted from HuMo repository for wan2_generate_video.py integration
#
# HuMo Model Architecture: WanModel with Audio Cross-Attention
# This is a 17B parameter model trained from Wan2.2 A14B architecture

import torch
from torch import nn
import torch.cuda.amp as amp
import math
from typing import List, Optional

from .attention import flash_attention
from .audio_proj import AudioProjModel
from modules.custom_offloading_utils import ModelOffloader
from utils.device_utils import clean_memory_on_device

__all__ = ["WanHuMoModel", "WanAttentionBlockHuMo"]


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    if position.dim() == 0:
        position = position.unsqueeze(0)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float32).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply_inplace(x, grid_sizes, freqs):
    """In-place rotary position embedding to avoid memory duplication during block swap."""
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples - modify x in-place
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs_split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding IN-PLACE (only modify first seq_len elements)
        x[i, :seq_len] = torch.view_as_real(x_i * freqs_i).flatten(2).to(x.dtype)

    return x


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, _attn_debug=False):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def _mem(msg):
            if _attn_debug and torch.cuda.is_available():
                torch.cuda.synchronize()
                print(f"      [SelfAttn] {msg}: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

        _mem("Start")

        # query, key, value function
        def qkv_fn(x):
            _mem("Before q projection")
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            _mem("After q projection")
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            _mem("After k projection")
            v = self.v(x).view(b, s, n, d)
            _mem("After v projection")
            return q, k, v

        q, k, v = qkv_fn(x)
        del x  # Free input tensor immediately

        _mem("Before rope_apply")
        rope_apply_inplace(q, grid_sizes, freqs)
        rope_apply_inplace(k, grid_sizes, freqs)
        qkv = [q, k, v]
        del q, k, v  # Free references after creating qkv list
        _mem("After rope_apply")
        x = flash_attention(qkv, k_lens=seq_lens, window_size=self.window_size)
        _mem("After flash_attention")

        # output
        x = x.flatten(2)
        x = self.o(x)
        _mem("After output projection")
        return x


class WanSelfAttentionSepKVDim(nn.Module):
    """Self-attention with separate key/value dimension for audio cross-attention."""

    def __init__(self,
                 kv_dim,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        del x  # Free input tensor immediately

        rope_apply_inplace(q, grid_sizes, freqs)
        rope_apply_inplace(k, grid_sizes, freqs)
        qkv = [q, k, v]
        del q, k, v  # Free references after creating qkv list
        x = flash_attention(qkv, k_lens=seq_lens, window_size=self.window_size)

        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        qkv = [q, k, v]
        del q, k, v  # Free tensors to prevent memory accumulation
        x = flash_attention(qkv, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttentionGather(WanSelfAttentionSepKVDim):
    """Audio cross-attention with spatial gathering for video-audio alignment."""

    def forward(self, x, context, context_lens, grid_sizes, freqs, audio_seq_len):
        """
        Args:
            x: Video tokens [B, L1, C]
            context: Audio tokens [B, frames*16, 1536]
            context_lens: Video sequence length
            grid_sizes: Video grid dimensions (F, H, W)
            freqs: RoPE frequencies
            audio_seq_len: Actual audio sequence length (frames * 16)
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # Handle video spatial structure
        hlen_wlen = int(grid_sizes[0][1] * grid_sizes[0][2])
        q = q.reshape(-1, hlen_wlen, n, d)

        # Handle audio temporal structure (16 tokens per frame)
        k = k.reshape(-1, 16, n, d)
        v = v.reshape(-1, 16, n, d)

        # Cross-attention
        qkv = [q, k, v]
        del q, k, v  # Free tensors to prevent memory accumulation
        x = flash_attention(qkv, k_lens=None)

        x = x.view(b, -1, n, d).flatten(2)
        x = self.o(x)
        return x


class AudioCrossAttentionWrapper(nn.Module):
    """Wrapper for audio cross-attention in each attention block."""

    def __init__(self, dim, kv_dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()

        self.audio_cross_attn = WanT2VCrossAttentionGather(
                kv_dim, dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm1_audio = WanLayerNorm(dim, eps, elementwise_affine=True)

    def forward(self, x, audio, seq_lens, grid_sizes, freqs, audio_seq_len):
        x = x + self.audio_cross_attn(
            self.norm1_audio(x), audio, seq_lens, grid_sizes, freqs, audio_seq_len)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

    def forward(self, x, context, context_lens):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        qkv = [q, k, v]
        del q, k, v  # Free tensors to prevent memory accumulation
        x = flash_attention(qkv, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlockHuMo(nn.Module):
    """HuMo Attention Block with Audio Cross-Attention."""

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 use_audio=True):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.use_audio = use_audio
        if use_audio:
            self.audio_cross_attn_wrapper = AudioCrossAttentionWrapper(dim, 1536, num_heads, qk_norm, eps)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        audio=None,
        audio_seq_len=None,
        ref_num_list=None,
        _block_debug=False,  # Debug flag
    ):
        # Debug memory tracking for block
        if _block_debug and torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"    [Block] Start: {torch.cuda.memory_allocated() / 1e9:.2f} GB, x device: {x.device}", flush=True)

        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        if _block_debug and torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"    [Block] After modulation: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
            # Check all parameter devices
            for name, param in self.named_parameters():
                if param.device.type != 'cuda':
                    print(f"    [Block] WARN: {name} is on {param.device}!", flush=True)
                    break

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs, _attn_debug=_block_debug)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)

            if self.use_audio and audio is not None:
                x = self.audio_cross_attn_wrapper(x, audio, seq_lens, grid_sizes, freqs, audio_seq_len)

            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)

        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanHuMoModel(nn.Module):
    """
    HuMo diffusion backbone - Wan2.2 architecture with audio cross-attention.
    17B parameter model supporting TIA (Text+Image+Audio) and TA (Text+Audio) modes.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlockHuMo']

    gradient_checkpointing = False

    def __init__(self,
                 model_type='i2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=36,  # HuMo uses 36 for i2v: mask(4) + latent(16) + ref_latent(16)
                 dim=5120,
                 ffn_dim=13824,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=40,
                 num_layers=40,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 audio_token_num=16,
                 insert_audio=True):
        """
        Initialize HuMo model.

        Args:
            model_type: 't2v' or 'i2v'
            patch_size: 3D patch dimensions (t, h, w)
            text_len: Maximum text token length
            in_dim: Input channels (36 for i2v with mask+latent+ref)
            dim: Hidden dimension (5120 for 17B model)
            ffn_dim: FFN intermediate dimension
            freq_dim: Sinusoidal time embedding dimension
            text_dim: Text embedding dimension (T5)
            out_dim: Output channels
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            window_size: Window attention size (-1, -1) for global
            qk_norm: Whether to apply QK normalization
            cross_attn_norm: Whether to apply cross-attention normalization
            eps: Epsilon for normalization
            audio_token_num: Number of audio tokens per frame (16)
            insert_audio: Whether to use audio cross-attention
        """
        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.insert_audio = insert_audio
        self.blocks = nn.ModuleList([
            WanAttentionBlockHuMo(cross_attn_type, dim, ffn_dim, num_heads,
                        window_size, qk_norm, cross_attn_norm,
                        eps, use_audio=self.insert_audio)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        if self.insert_audio:
            self.audio_proj = AudioProjModel(seq_len=8, blocks=5, channels=1280,
                intermediate_dim=512, output_dim=1536, context_tokens=audio_token_num)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        # block swap support
        self.blocks_to_swap = None
        self.offloader = None

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        context: List[torch.Tensor],
        seq_len: int,
        audio: Optional[List[torch.Tensor]] = None,
        y: Optional[List[torch.Tensor]] = None,
    ):
        """
        Forward pass through HuMo model.

        Args:
            x: List of input video tensors [C_in, F, H, W]
            t: Diffusion timesteps [B]
            context: List of text embeddings [L, C]
            seq_len: Maximum sequence length
            audio: List of windowed audio embeddings [iter, 8, 5, 1280]
            y: List of conditioning tensors (mask + latent) [C, F, H, W]

        Returns:
            List of denoised video tensors [C_out, F, H/8, W/8]
        """
        if self.model_type == 'i2v':
            assert y is not None

        # Debug: track memory usage
        _debug_mem = hasattr(self, '_debug_forward_mem') and self._debug_forward_mem
        import sys
        def _log_mem(msg):
            if _debug_mem and torch.cuda.is_available():
                torch.cuda.synchronize()
                print(f"[HuMo Forward] {msg}: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

        if _debug_mem:
            print(f"[HuMo Forward] blocks_to_swap={self.blocks_to_swap}, offloader={self.offloader is not None}", flush=True)

        _log_mem("Start")

        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        _log_mem("After concat x,y")

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

        _log_mem("After patch_embedding")

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        assert seq_lens.max() <= seq_len

        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        _log_mem("After x padding")

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float()).float()
            e0 = self.time_projection(e).unflatten(1, (6, self.dim)).float()
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        _log_mem("After time_embedding")

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        _log_mem("After text_embedding")

        # audio processing
        if self.insert_audio and audio is not None:
            audio = [self.audio_proj(au.unsqueeze(0)).permute(0, 3, 1, 2) for au in audio]

            audio_seq_len = torch.tensor(max([au.shape[2] for au in audio]) * audio[0].shape[3], device=device)
            audio = [au.flatten(2).transpose(1, 2) for au in audio]  # [1, t*16, 1536]
            audio = torch.cat([
                torch.cat([au, au.new_zeros(1, audio_seq_len - au.size(1), au.size(2))],
                        dim=1) for au in audio
            ])

        _log_mem("After audio processing")

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            audio=audio,
            audio_seq_len=audio_seq_len)

        # Clean memory before block loop to prevent accumulation during block swap
        if self.blocks_to_swap:
            clean_memory_on_device(torch.device('cuda'))

        for block_idx, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                if block_idx < 5:
                    _log_mem(f"Before wait_for_block({block_idx})")
                self.offloader.wait_for_block(block_idx)
                if block_idx < 5:
                    _log_mem(f"After wait_for_block({block_idx})")

            if block_idx < 5:
                _log_mem(f"Before block {block_idx}")
                # Check if block is on correct device
                first_param = next(block.parameters(), None)
                if first_param is not None:
                    print(f"  Block {block_idx} first param device: {first_param.device}, dtype: {first_param.dtype}", flush=True)
                else:
                    print(f"  Block {block_idx} has no parameters!", flush=True)

            # Pass debug flag for first few blocks
            if block_idx < 3 and _debug_mem:
                x = block(x, _block_debug=True, **kwargs)
            else:
                x = block(x, **kwargs)

            if block_idx < 5:
                _log_mem(f"After block {block_idx}")

            if self.blocks_to_swap:
                if _debug_mem and block_idx < 5:
                    print(f"[HuMo Forward] Before submit_swap({block_idx}): {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
                self.offloader.submit_move_blocks_forward(self.blocks, block_idx)
                if _debug_mem and block_idx < 5:
                    print(f"[HuMo Forward] After submit_swap({block_idx}): {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

            # Force memory cleanup to check for retention issues
            if _debug_mem and block_idx < 10:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"[HuMo Forward] After cleanup block {block_idx}: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def enable_block_swap(self, blocks_to_swap: int, device: torch.device, supports_backward: bool):
        self.blocks_to_swap = blocks_to_swap
        self.num_blocks = len(self.blocks)

        assert (
            self.blocks_to_swap <= self.num_blocks - 1
        ), f"Cannot swap more than {self.num_blocks - 1} blocks. Requested {self.blocks_to_swap} blocks to swap."

        self.offloader = ModelOffloader(
            "wan_humo_attn_block", self.blocks, self.num_blocks, self.blocks_to_swap, supports_backward, device
        )
        print(
            f"WanHuMoModel: Block swap enabled. Swapping {self.blocks_to_swap} blocks out of {self.num_blocks} blocks. Supports backward: {supports_backward}"
        )

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
        if self.blocks_to_swap:
            save_blocks = self.blocks
            self.blocks = None

        self.to(device)

        if self.blocks_to_swap:
            self.blocks = save_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.blocks)

        # Verify block placement
        num_resident = self.num_blocks - self.blocks_to_swap
        print(f"[HuMo] Verifying block placement after prepare:", flush=True)
        for i, block in enumerate(self.blocks):
            first_param = next(block.parameters(), None)
            if first_param is not None:
                expected = "GPU" if i < num_resident else "CPU"
                actual = "GPU" if first_param.device.type == "cuda" else "CPU"
                if expected != actual:
                    print(f"  Block {i}: MISMATCH! Expected {expected}, got {actual}", flush=True)
                elif i < 3 or i >= self.num_blocks - 2:  # Only log first 3 and last 2
                    print(f"  Block {i}: {actual} (correct)", flush=True)
