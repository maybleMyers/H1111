# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import re # <-- IMPORTED FOR PARSING LORA KEYS
from typing import Optional, Union, List, Dict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from accelerate import init_empty_weights

import logging

from utils.safetensors_utils import MemoryEfficientSafeOpen, load_safetensors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from utils.device_utils import clean_memory_on_device

from .attention import flash_attention
from utils.device_utils import clean_memory_on_device
from modules.custom_offloading_utils import ModelOffloader
from modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8

__all__ = ["WanModel"]


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# @amp.autocast(enabled=False)
# no autocast is needed for rope_apply, because it is already in float64
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    device_type = x.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        n, c = x.size(2), x.size(3) // 2

        # split freqs
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # loop over samples
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w

            # precompute multipliers
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
            freqs_i = torch.cat(
                [
                    freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(seq_len, 1, -1)

            # apply rotary embedding
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])

            # append to collection
            output.append(x_i)
        return torch.stack(output).float()


def calculate_freqs_i(fhw, c, freqs):
    f, h, w = fhw
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    freqs_i = torch.cat(
        [
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)
    return freqs_i


# inplace version of rope_apply
def rope_apply_inplace_cached(x, grid_sizes, freqs_list):
    # with torch.amp.autocast(device_type=device_type, enabled=False):
    rope_dtype = torch.float64  # float32 does not reduce memory usage significantly

    n, c = x.size(2), x.size(3) // 2

    # loop over samples
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(rope_dtype).reshape(seq_len, n, -1, 2))
        freqs_i = freqs_list[i]

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        # x_i = torch.cat([x_i, x[i, seq_len:]])

        # inplace update
        x[i, :seq_len] = x_i.to(x.dtype)

    return x


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # return self._norm(x.float()).type_as(x) * self.weight
        # support fp8
        return self._norm(x.float()).type_as(x) * self.weight.to(x.dtype)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    # def forward(self, x):
    #     r"""
    #     Args:
    #         x(Tensor): Shape [B, L, C]
    #     """
    #     # inplace version, also supports fp8 -> does not have significant performance improvement
    #     original_dtype = x.dtype
    #     x = x.float()
    #     y = x.pow(2).mean(dim=-1, keepdim=True)
    #     y.add_(self.eps)
    #     y.rsqrt_()
    #     x *= y
    #     x = x.to(original_dtype)
    #     x *= self.weight.to(original_dtype)
    #     return x


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, attn_mode="torch", split_attn=False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # # query, key, value function
        # def qkv_fn(x):
        #     q = self.norm_q(self.q(x)).view(b, s, n, d)
        #     k = self.norm_k(self.k(x)).view(b, s, n, d)
        #     v = self.v(x).view(b, s, n, d)
        #     return q, k, v
        # q, k, v = qkv_fn(x)
        # del x
        # query, key, value function

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        del x
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = q.view(b, s, n, d)
        k = k.view(b, s, n, d)
        v = v.view(b, s, n, d)

        rope_apply_inplace_cached(q, grid_sizes, freqs)
        rope_apply_inplace_cached(k, grid_sizes, freqs)
        qkv = [q, k, v]
        del q, k, v
        x = flash_attention(
            qkv, k_lens=seq_lens, window_size=self.window_size, attn_mode=self.attn_mode, split_attn=self.split_attn
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        # q = self.norm_q(self.q(x)).view(b, -1, n, d)
        # k = self.norm_k(self.k(context)).view(b, -1, n, d)
        # v = self.v(context).view(b, -1, n, d)
        q = self.q(x)
        del x
        k = self.k(context)
        v = self.v(context)
        del context
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = q.view(b, -1, n, d)
        k = k.view(b, -1, n, d)
        v = v.view(b, -1, n, d)

        # compute attention
        qkv = [q, k, v]
        del q, k, v
        x = flash_attention(qkv, k_lens=context_lens, attn_mode=self.attn_mode, split_attn=self.split_attn)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, attn_mode="torch", split_attn=False):
        super().__init__(dim, num_heads, window_size, qk_norm, eps, attn_mode, split_attn)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x)
        del x
        q = self.norm_q(q)
        q = q.view(b, -1, n, d)
        k = self.k(context)
        k = self.norm_k(k).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        del context

        # compute attention
        qkv = [q, k, v]
        del k, v
        x = flash_attention(qkv, k_lens=context_lens, attn_mode=self.attn_mode, split_attn=self.split_attn)

        # compute query, key, value
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        del context_img

        # compute attention
        qkv = [q, k_img, v_img]
        del q, k_img, v_img
        img_x = flash_attention(qkv, k_lens=None, attn_mode=self.attn_mode, split_attn=self.split_attn)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        if self.training:
            x = x + img_x  # avoid inplace
        else:
            x += img_x
        del img_x

        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode="torch",
        split_attn=False,
    ):
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
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, attn_mode, split_attn)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps, attn_mode, split_attn)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        #     e = (self.modulation + e).chunk(6, dim=1)
        # support fp8
        e = self.modulation.to(torch.float32) + e
        e = e.chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs)
        # with amp.autocast(dtype=torch.float32):
        #     x = x + y * e[2]
        x = x + y.to(torch.float32) * e[2]
        del y

        # cross-attention & ffn function
        # def cross_attn_ffn(x, context, context_lens, e):
        #     x += self.cross_attn(self.norm3(x), context, context_lens)
        #     y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        #     # with amp.autocast(dtype=torch.float32):
        #     #     x = x + y * e[5]
        #     x += y.to(torch.float32) * e[5]
        #     return x
        # x = cross_attn_ffn(x, context, context_lens, e)

        # x += self.cross_attn(self.norm3(x), context, context_lens) # backward error
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        del context
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        x = x + y.to(torch.float32) * e[5]
        del y
        return x

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, e, seq_lens, grid_sizes, freqs, context, context_lens, use_reentrant=False)
        return self._forward(x, e, seq_lens, grid_sizes, freqs, context, context_lens)


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
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        #     e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        #     x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        # support fp8
        e = (self.modulation.to(torch.float32) + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(nn.Module):  # ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanAttentionBlock"]

    # @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        attn_mode=None,
        split_attn=False,
        add_ref_conv=False, 
        in_dim_ref_conv=16,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v"]
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
        self.attn_mode = attn_mode if attn_mode is not None else "torch"
        self.split_attn = split_attn

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, attn_mode, split_attn
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1
        )
        self.freqs_fhw = {}

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        self.add_ref_conv = add_ref_conv # Store the flag
        if add_ref_conv:
            # Use spatial dimensions from patch_size for Conv2d
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
            logger.info(f"Initialized ref_conv layer with in_channels={in_dim_ref_conv}, out_channels={dim}")
        else:
            self.ref_conv = None            

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        # offloading
        self.blocks_to_swap = None
        self.offloader = None

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def fp8_optimization(
        self, state_dict: dict[str, torch.Tensor], device: torch.device, move_to_device: bool, use_scaled_mm: bool = False
    ) -> int:
        """
        Optimize the model state_dict with fp8.

        Args:
            state_dict (dict[str, torch.Tensor]):
                The state_dict of the model.
            device (torch.device):
                The device to calculate the weight.
            move_to_device (bool):
                Whether to move the weight to the device after optimization.
        """
        TARGET_KEYS = ["blocks"]
        EXCLUDE_KEYS = [
            "norm",
            "patch_embedding",
            "text_embedding",
            "time_embedding",
            "time_projection",
            "head",
            "modulation",
            "img_emb",
        ]

        # inplace optimization
        state_dict = optimize_state_dict_with_fp8(state_dict, device, TARGET_KEYS, EXCLUDE_KEYS, move_to_device=move_to_device)

        # apply monkey patching
        apply_fp8_monkey_patch(self, state_dict, use_scaled_mm=use_scaled_mm)

        return state_dict

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

        for block in self.blocks:
            block.enable_gradient_checkpointing()

        print(f"WanModel: Gradient checkpointing enabled.")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

        for block in self.blocks:
            block.disable_gradient_checkpointing()

        print(f"WanModel: Gradient checkpointing disabled.")

    def enable_block_swap(self, blocks_to_swap: int, device: torch.device, supports_backward: bool):
        self.blocks_to_swap = blocks_to_swap
        self.num_blocks = len(self.blocks)

        assert (
            self.blocks_to_swap <= self.num_blocks - 1
        ), f"Cannot swap more than {self.num_blocks - 1} blocks. Requested {self.blocks_to_swap} blocks to swap."

        self.offloader = ModelOffloader(
            "wan_attn_block", self.blocks, self.num_blocks, self.blocks_to_swap, supports_backward, device  # , debug=True
        )
        print(
            f"WanModel: Block swap enabled. Swapping {self.blocks_to_swap} blocks out of {self.num_blocks} blocks. Supports backward: {supports_backward}"
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print(f"WanModel: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print(f"WanModel: Block swap set to forward and backward.")

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

    def forward(self, x, t, context, seq_len, clip_fea=None, y=None, skip_block_indices=None, fun_ref=None):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # remove assertions to work with Fun-Control T2V
        # if self.model_type == "i2v":
        #     assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if isinstance(x, list) and len(x) > 0:
             _, F_orig, H_orig, W_orig = x[0].shape
        else:
             # Fallback or error handling if x is not as expected
             raise ValueError("Input x is not in the expected list format.")            

        if y is not None:
            #print('WanModel concat debug:')
            #for i, (u, v) in enumerate(zip(x, y)):
                #print(f"x[{i}]: {u.shape}, y[{i}]: {v.shape}, y[{i}].dim(): {v.dim()}")
            x = [
                torch.cat([u, v], dim=0)
                for u, v in zip(x, y)
            ]
            y = None
        

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

        # <<< START: Process fun_ref if applicable >>>
        F = F_orig # Use original frame count for RoPE calculation unless fun_ref modifies it
        if self.ref_conv is not None and fun_ref is not None:
            # fun_ref is expected to be the raw reference image latent [B, C_ref, H_ref, W_ref]
            # Ensure it's on the correct device
            fun_ref = fun_ref.to(device)
            logger.debug(f"Processing fun_ref with shape: {fun_ref.shape}")

            # Apply the 2D convolution
            # Note: fun_ref needs batch dim for Conv2d, add if missing
            if fun_ref.dim() == 3: fun_ref = fun_ref.unsqueeze(0)
            processed_ref = self.ref_conv(fun_ref) # Output: [B, C, H_out, W_out]
            logger.debug(f"Processed ref_conv output shape: {processed_ref.shape}")

            # Reshape to token sequence: [B, L_ref, C]
            processed_ref = processed_ref.flatten(2).transpose(1, 2)
            logger.debug(f"Reshaped processed_ref shape: {processed_ref.shape}")

            # Adjust grid_sizes, seq_len, and F to account for the prepended tokens
            # Assuming the reference adds effectively one "frame" worth of tokens spatially
            # Note: This might need adjustment depending on how seq_len is used later.
            # We increment the frame dimension 'F' in grid_sizes.
            grid_sizes = torch.stack([torch.tensor([gs[0] + 1, gs[1], gs[2]], dtype=torch.long) for gs in grid_sizes]).to(grid_sizes.device)
            seq_len += processed_ref.size(1) # Add number of reference tokens
            F = F_orig + 1 # Indicate one extra effective frame for RoPE/freq calculation
            logger.debug(f"Adjusted grid_sizes: {grid_sizes}, seq_len: {seq_len}, F for RoPE: {F}")

            # Prepend the reference tokens to each element in the list x
            x = [torch.cat([processed_ref, u.flatten(2).transpose(1, 2)], dim=1) for u in x] # x was already flattened+transposed below, do it here
            # x is now list of [B, L_new, C]
        else:
            # Original flattening if no fun_ref
            x = [u.flatten(2).transpose(1, 2) for u in x]     
        # <<< END: Process fun_ref if applicable >>>               

        freqs_list = []
        for fhw in grid_sizes: # Use the potentially updated grid_sizes
            fhw_tuple = tuple(fhw.tolist())
            if fhw_tuple not in self.freqs_fhw:
                c_rope = self.dim // self.num_heads // 2
                # Use the potentially updated frame count F from fhw[0]
                self.freqs_fhw[fhw_tuple] = calculate_freqs_i(fhw, c_rope, self.freqs)
            freqs_list.append(self.freqs_fhw[fhw_tuple])

        # ... (seq_len calculation and padding using potentially updated seq_len) ...
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if seq_lens.max() > seq_len:
             # This might happen if seq_len wasn't updated correctly or padding logic needs review
             logger.warning(f"Calculated seq_lens.max()={seq_lens.max()} > adjusted seq_len={seq_len}. Adjusting seq_len.")
             seq_len = seq_lens.max().item() # Use the actual max length required

        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings 
        # with amp.autocast(dtype=torch.float32):
        with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        if type(context) is list:
            context = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        context = self.text_embedding(context)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
            clip_fea = None
            context_clip = None

        # arguments
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs_list, context=context, context_lens=context_lens)

        if self.blocks_to_swap:
            clean_memory_on_device(device)

        # print(f"x: {x.shape}, e: {e0.shape}, context: {context.shape}, seq_lens: {seq_lens}")
        for block_idx, block in enumerate(self.blocks):
            is_block_skipped = skip_block_indices is not None and block_idx in skip_block_indices

            if self.blocks_to_swap and not is_block_skipped:
                self.offloader.wait_for_block(block_idx)

            if not is_block_skipped:
                x = block(x, **kwargs)

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.blocks, block_idx)

        if self.ref_conv is not None and fun_ref is not None:
            num_ref_tokens = processed_ref.size(1)
            logger.debug(f"Removing {num_ref_tokens} prepended reference tokens before head.")
            x = x[:, num_ref_tokens:, :]
            # Restore original grid_sizes F dimension for unpatchify
            grid_sizes = torch.stack([torch.tensor([gs[0] - 1, gs[1], gs[2]], dtype=torch.long) for gs in grid_sizes]).to(grid_sizes.device)                

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

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
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)


def detect_wan_sd_dtype(path: str) -> torch.dtype:
    # get dtype from model weights
    with MemoryEfficientSafeOpen(path) as f:
        keys = set(f.keys())
        key1 = "model.diffusion_model.blocks.0.cross_attn.k.weight"  # 1.3B
        key2 = "blocks.0.cross_attn.k.weight"  # 14B
        if key1 in keys:
            dit_dtype = f.get_tensor(key1).dtype
        elif key2 in keys:
            dit_dtype = f.get_tensor(key2).dtype
        else:
            raise ValueError(f"Could not find the dtype in the model weights: {path}")
    logger.info(f"Detected DiT dtype: {dit_dtype}")
    return dit_dtype

def load_wan_model(
    config: any,
    device: Union[str, torch.device],
    dit_path: Union[str, List[str]], # Can now be a list
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[List[float]] = None,
    use_scaled_mm: bool = False,
) -> WanModel:
    assert not fp8_scaled, "FP8 scaling is not compatible with this modified LoRA loader."

    device = torch.device(device)
    loading_device = torch.device(loading_device)
    
    # --- Step 1: Analyze LoRA to determine the final model architecture ---
    total_layers = config.num_layers
    merged_lora_sd = {}

    if lora_weights_list and len(lora_weights_list) > 0:
        logger.info("Pusa LoRA detected. Analyzing architecture before model creation...")
        max_block_index = -1
        
        # We assume one Pusa LoRA is provided per model, but handle multiple just in case
        for lora_sd in lora_weights_list: 
            for key in lora_sd.keys():
                # Find keys like "blocks.47.ffn.0.weight" to detect new blocks
                match = re.search(r"blocks\.(\d+)\.", key)
                if match:
                    block_idx = int(match.group(1))
                    if block_idx > max_block_index:
                        max_block_index = block_idx
            # Merge LoRA weights, new weights will overwrite existing ones
            merged_lora_sd.update(lora_sd)

        if max_block_index != -1 and (max_block_index + 1) > config.num_layers:
            total_layers = max_block_index + 1
            num_new_blocks = total_layers - config.num_layers
            logger.info(f"Pusa LoRA adds {num_new_blocks} new blocks.")
            logger.info(f"Base layers: {config.num_layers}, Total layers: {total_layers}")
        else:
            logger.info("LoRA does not seem to add new blocks. Treating as standard weight patch.")
            
    # --- Step 2: Load the base model state dict ---
    logger.info(f"Loading DiT base model state dict from {dit_path}")
    sd = load_safetensors(dit_path, loading_device, disable_mmap=True, dtype=dit_weight_dtype)

    # remove "model.diffusion_model." prefix if it exists
    sd_keys = list(sd.keys())
    for key in sd_keys:
        if key.startswith("model.diffusion_model."):
            sd[key[22:]] = sd.pop(key)

    # --- Step 3: Merge LoRA weights into the base state dict ---
    if merged_lora_sd:
        logger.info("Merging LoRA weights into base model state dict.")
        sd.update(merged_lora_sd)
        # Clean up memory
        del merged_lora_sd
        del lora_weights_list
        clean_memory_on_device(loading_device)


    # Check for ref_conv layer weights
    has_ref_conv = "ref_conv.weight" in sd
    in_dim_ref_conv = sd["ref_conv.weight"].shape[1] if has_ref_conv else 16
    if has_ref_conv:
        logger.info(f"Detected ref_conv layer in model weights. Input channels: {in_dim_ref_conv}")

    # --- Step 4: Create the correctly-sized empty model ---
    with init_empty_weights():
        logger.info(f"Creating WanModel with {total_layers} layers.")
        model = WanModel(
            model_type="i2v" if config.i2v else "t2v",
            dim=config.dim,
            eps=config.eps,
            ffn_dim=config.ffn_dim,
            freq_dim=config.freq_dim,
            in_dim=config.in_dim,
            num_heads=config.num_heads,
            num_layers=total_layers, # <-- CRITICAL CHANGE HERE
            out_dim=config.out_dim,
            text_len=config.text_len,
            attn_mode=attn_mode,
            split_attn=split_attn,
            add_ref_conv=has_ref_conv,
            in_dim_ref_conv=in_dim_ref_conv,
        )
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    # --- Step 5: Load the final merged state dict into the model ---
    info = model.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded merged state dict into model. Load info: {info}")
    if info.missing_keys:
        logger.warning(f"Missing keys during load: {info.missing_keys}")
    if info.unexpected_keys:
        logger.warning(f"Unexpected keys during load: {info.unexpected_keys}")

    return model
