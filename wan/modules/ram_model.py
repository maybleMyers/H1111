# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import re
from typing import Optional, Union, List, Dict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from accelerate import init_empty_weights

import logging

from utils.safetensors_utils import MemoryEfficientSafeOpen, load_safetensors

# 1. Import the new CPUBouncingLinear layer
from wan.modules.linear import CPUBouncingLinear

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

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, attn_mode="torch", split_attn=False, use_bouncing_linear=False, device="cuda"):
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

        # 2. Conditionally create Linear or CPUBouncingLinear layers
        if use_bouncing_linear:
            self.q = CPUBouncingLinear(dim, dim, device=device)
            self.k = CPUBouncingLinear(dim, dim, device=device)
            self.v = CPUBouncingLinear(dim, dim, device=device)
            self.o = CPUBouncingLinear(dim, dim, device=device)
        else:
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

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, attn_mode="torch", split_attn=False, use_bouncing_linear=False, device="cuda"):

        # 3. Pass new arguments to the super constructor
        super().__init__(dim, num_heads, window_size, qk_norm, eps, attn_mode, split_attn, use_bouncing_linear, device)

        # 4. Conditionally create layers for this specific subclass
        if use_bouncing_linear:
            self.k_img = CPUBouncingLinear(dim, dim, device=device)
            self.v_img = CPUBouncingLinear(dim, dim, device=device)
        else:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
        
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
        use_bouncing_linear=False,
        device="cuda",
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
        
        # 5. Pass arguments down to sub-module constructors
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, attn_mode, split_attn, use_bouncing_linear, device)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps, attn_mode, split_attn, use_bouncing_linear, device)
        self.norm2 = WanLayerNorm(dim, eps)

        # 6. Conditionally create the FFN sequential block
        if use_bouncing_linear:
            self.ffn = nn.Sequential(
                CPUBouncingLinear(dim, ffn_dim, device=device),
                nn.GELU(approximate="tanh"),
                CPUBouncingLinear(ffn_dim, dim, device=device)
            )
        else:
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
        e = self.modulation.to(torch.float32) + e
        e = e.chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs)
        x = x + y.to(torch.float32) * e[2]
        del y

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

    def __init__(self, dim, out_dim, patch_size, eps=1e-6, use_bouncing_linear=False, device="cuda"):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim_calc = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        
        # 7. Conditionally create the head layer
        if use_bouncing_linear:
            self.head = CPUBouncingLinear(dim, out_dim_calc, device=device)
        else:
            self.head = nn.Linear(dim, out_dim_calc)
        
        
        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        e = (self.modulation.to(torch.float32) + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, use_bouncing_linear=False, device="cuda"):
        super().__init__()

        
        # 8. Conditionally create the projection layers
        if use_bouncing_linear:
            self.proj = torch.nn.Sequential(
                torch.nn.LayerNorm(in_dim),
                CPUBouncingLinear(in_dim, in_dim, device=device),
                torch.nn.GELU(),
                CPUBouncingLinear(in_dim, out_dim, device=device),
                torch.nn.LayerNorm(out_dim),
            )
        else:
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


class WanModel(nn.Module):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanAttentionBlock"]

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
        use_bouncing_linear=False,
        device="cuda"
    ):
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
        
        
        # 9. Conditionally create all top-level Sequential blocks with Linear layers
        if use_bouncing_linear:
            self.text_embedding = nn.Sequential(
                CPUBouncingLinear(text_dim, dim, device=device),
                nn.GELU(approximate="tanh"),
                CPUBouncingLinear(dim, dim, device=device)
            )
            self.time_embedding = nn.Sequential(
                CPUBouncingLinear(freq_dim, dim, device=device),
                nn.SiLU(),
                CPUBouncingLinear(dim, dim, device=device)
            )
            self.time_projection = nn.Sequential(
                nn.SiLU(),
                CPUBouncingLinear(dim, dim * 6, device=device)
            )
        else:
            self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
            self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, attn_mode, split_attn,
                    # 10. Pass the new arguments down to the block constructor
                    use_bouncing_linear=use_bouncing_linear, device=device
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps, use_bouncing_linear=use_bouncing_linear, device=device)

        # buffers
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1
        )
        self.freqs_fhw = {}

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim, use_bouncing_linear=use_bouncing_linear, device=device)

        self.add_ref_conv = add_ref_conv
        if add_ref_conv:
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
        """
        TARGET_KEYS = ["blocks"]
        EXCLUDE_KEYS = [
            "norm", "patch_embedding", "text_embedding", "time_embedding",
            "time_projection", "head", "modulation", "img_emb",
        ]
        state_dict = optimize_state_dict_with_fp8(state_dict, device, TARGET_KEYS, EXCLUDE_KEYS, move_to_device=move_to_device)
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
            "wan_attn_block", self.blocks, self.num_blocks, self.blocks_to_swap, supports_backward, device
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
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if isinstance(x, list) and len(x) > 0:
             _, F_orig, H_orig, W_orig = x[0].shape
        else:
             raise ValueError("Input x is not in the expected list format.")            

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            y = None
        
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

        F = F_orig
        if self.ref_conv is not None and fun_ref is not None:
            fun_ref = fun_ref.to(device)
            logger.debug(f"Processing fun_ref with shape: {fun_ref.shape}")
            if fun_ref.dim() == 3: fun_ref = fun_ref.unsqueeze(0)
            processed_ref = self.ref_conv(fun_ref)
            logger.debug(f"Processed ref_conv output shape: {processed_ref.shape}")
            processed_ref = processed_ref.flatten(2).transpose(1, 2)
            logger.debug(f"Reshaped processed_ref shape: {processed_ref.shape}")
            grid_sizes = torch.stack([torch.tensor([gs[0] + 1, gs[1], gs[2]], dtype=torch.long) for gs in grid_sizes]).to(grid_sizes.device)
            seq_len += processed_ref.size(1)
            F = F_orig + 1
            logger.debug(f"Adjusted grid_sizes: {grid_sizes}, seq_len: {seq_len}, F for RoPE: {F}")
            x = [torch.cat([processed_ref, u.flatten(2).transpose(1, 2)], dim=1) for u in x]
        else:
            x = [u.flatten(2).transpose(1, 2) for u in x]     

        freqs_list = []
        for fhw in grid_sizes:
            fhw_tuple = tuple(fhw.tolist())
            if fhw_tuple not in self.freqs_fhw:
                c_rope = self.dim // self.num_heads // 2
                self.freqs_fhw[fhw_tuple] = calculate_freqs_i(fhw, c_rope, self.freqs)
            freqs_list.append(self.freqs_fhw[fhw_tuple])

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if seq_lens.max() > seq_len:
             logger.warning(f"Calculated seq_lens.max()={seq_lens.max()} > adjusted seq_len={seq_len}. Adjusting seq_len.")
             seq_len = seq_lens.max().item()

        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        context_lens = None
        if type(context) is list:
            context = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        context = self.text_embedding(context)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)
            clip_fea = None
            context_clip = None

        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs_list, context=context, context_lens=context_lens)

        if self.blocks_to_swap:
            clean_memory_on_device(device)

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
            grid_sizes = torch.stack([torch.tensor([gs[0] - 1, gs[1], gs[2]], dtype=torch.long) for gs in grid_sizes]).to(grid_sizes.device)                

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)


def detect_wan_sd_dtype(path: str) -> torch.dtype:
    with MemoryEfficientSafeOpen(path) as f:
        keys = set(f.keys())
        key1 = "model.diffusion_model.blocks.0.cross_attn.k.weight"
        key2 = "blocks.0.cross_attn.k.weight"
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
    dit_path: Union[str, List[str]],
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[List[float]] = None,
    use_scaled_mm: bool = False,
    use_bouncing_linear: bool = False, 
) -> WanModel:
    assert not fp8_scaled, "FP8 scaling is not compatible with this LoRA loader."

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    logger.info(f"Creating WanModel with {config.num_layers} base layers.")
    with init_empty_weights():
        model = WanModel(
            model_type="i2v" if config.i2v else "t2v",
            dim=config.dim, eps=config.eps, ffn_dim=config.ffn_dim, freq_dim=config.freq_dim,
            in_dim=config.in_dim, num_heads=config.num_heads, num_layers=config.num_layers,
            out_dim=config.out_dim, text_len=config.text_len, attn_mode=attn_mode,
            split_attn=split_attn, add_ref_conv=False, in_dim_ref_conv=16,
            
            # 12. Pass the new arguments to the WanModel constructor
            use_bouncing_linear=use_bouncing_linear,
            device=device,
            
        )

    dit_path_list = dit_path if isinstance(dit_path, list) else [dit_path]
    logger.info(f"Loading DiT base model state dict from: {dit_path_list}")
    sd = {}
    for path in dit_path_list:
        sd.update(load_safetensors(path, "cpu", disable_mmap=True, dtype=dit_weight_dtype))
    
    sd_keys = list(sd.keys())
    for key in sd_keys:
        if key.startswith("model.diffusion_model."):
            sd[key[22:]] = sd.pop(key)
    
    info = model.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded base model weights to CPU. Info: {info}")
    
    if lora_weights_list and lora_multipliers:
        logger.info(f"Applying {len(lora_weights_list)} LoRA(s) to the CPU model...")
        for lora_sd, multiplier in zip(lora_weights_list, lora_multipliers):
            lora_A_weights = {k: v for k, v in lora_sd.items() if 'lora_A' in k}

            for lora_A_key, lora_A_tensor in lora_A_weights.items():
                lora_B_key = lora_A_key.replace('lora_A', 'lora_B')
                if lora_B_key in lora_sd:
                    lora_B_tensor = lora_sd[lora_B_key]
                    
                    target_key = lora_A_key.replace('.lora_A.default.weight', '.weight')
                    if target_key.startswith("model.diffusion_model."):
                        target_key = target_key[22:]

                    try:
                        module_path, param_name = target_key.rsplit('.', 1)
                        parent_module = model.get_submodule(module_path)
                        original_weight = getattr(parent_module, param_name)
                    except (AttributeError, ValueError):
                        logger.warning(f"Could not find target parameter for LoRA key: {lora_A_key}")
                        continue
                        
                    rank = lora_A_tensor.shape[0]
                    scale = multiplier / rank

                    lora_A_tensor = lora_A_tensor.to(device=device, dtype=torch.float32)
                    lora_B_tensor = lora_B_tensor.to(device=device, dtype=torch.float32)
                    delta_W = (lora_B_tensor @ lora_A_tensor) * scale
                    
                    original_weight.data += delta_W.to(device=original_weight.device, dtype=original_weight.dtype)
        
        logger.info("Finished applying LoRA weights.")
        
    else:
        logger.info("No LoRA weights to apply.")

    return model