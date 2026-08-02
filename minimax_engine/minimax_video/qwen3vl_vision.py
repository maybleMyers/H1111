# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone port of transformers' Qwen3VLVisionModel (models/qwen3_vl/modeling_qwen3_vl.py,
# Apache-2.0) plus the Qwen2-VL image preprocessing math (smart_resize + patchify).
# Copied from cosmos_engine/cosmos_video/vision_encoder.py for the MiniMax-H3 conditioner;
# vendored because the environment pins transformers 4.46.x, which predates qwen3_vl.
# Attribute names match the upstream module exactly so the checkpoint's vision tower state
# dict loads without key rewriting. SDPA-only, inference-only.
#
# MiniMax-H3 uses this tower inside its Qwen3-VL conditioner: merged features replace the
# image/video pad tokens of the presentation and the deepstack features are injected into
# the first text decoder layers.

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_state):
        # config.hidden_act is gelu_pytorch_tanh for the Cosmos3 tower
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_state), approximate="tanh"))


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, temporal_patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, hidden_size: int, spatial_merge_size: int, out_hidden_size: int, use_postshuffle_norm=False):
        super().__init__()
        self.hidden_size = hidden_size * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.dim // num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # [1, heads, seq, head_dim]
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # SDPA per image chunk (cu_seqlens delimits images; attention never crosses images)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        splits = [torch.split(tensor, lengths, dim=2) for tensor in (query_states, key_states, value_states)]
        attn_outputs = [
            F.scaled_dot_product_attention(q, k, v, is_causal=False) for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=2)
        attn_output = attn_output.transpose(1, 2).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(hidden_size, num_heads)
        self.mlp = Qwen3VLVisionMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionModel(nn.Module):
    """Qwen3-VL vision tower: returns merged features plus deepstack feature list."""

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_heads: int = 16,
        depth: int = 27,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        in_channels: int = 3,
        out_hidden_size: int = 5120,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: list[int] | None = None,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size
        self.out_hidden_size = out_hidden_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(patch_size, temporal_patch_size, in_channels, hidden_size)
        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(num_position_embeddings**0.5)

        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen3VLVisionBlock(hidden_size, num_heads, intermediate_size) for _ in range(depth)]
        )
        self.merger = Qwen3VLVisionPatchMerger(
            hidden_size, spatial_merge_size, out_hidden_size, use_postshuffle_norm=False
        )
        self.deepstack_visual_indexes = list(deepstack_visual_indexes or [])
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(hidden_size, spatial_merge_size, out_hidden_size, use_postshuffle_norm=True)
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self):
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        return embeddings.flatten(1)

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        """pixel patches [N, C*tp*p*p] + grid [n_img, 3] -> (merged features [N/4, out], deepstack list)."""
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        return self.merger(hidden_states), deepstack_feature_lists


# ---------------------------------------------------------------------------
# Image preprocessing (Qwen2VLImageProcessor math; params from the checkpoint's
# preprocessor_config.json: mean/std 0.5, patch 16, temporal 2, merge 2)
# ---------------------------------------------------------------------------


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Snap (height, width) to multiples of ``factor`` inside the pixel budget, keeping aspect."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess_image(
    image: Image.Image,
    *,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PIL image -> (pixel_values [N_patches, C*tp*p*p] float32, grid_thw [1, 3] long)."""
    image = image.convert("RGB")
    factor = patch_size * merge_size
    resized_height, resized_width = smart_resize(image.height, image.width, factor, min_pixels, max_pixels)
    image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(image_mean, dtype=np.float32)) / np.asarray(image_std, dtype=np.float32)
    patches = arr.transpose(2, 0, 1)[np.newaxis]  # [1, C, H, W]
    if patches.shape[0] % temporal_patch_size != 0:
        repeats = np.repeat(patches[-1][np.newaxis], temporal_patch_size - (patches.shape[0] % temporal_patch_size), axis=0)
        patches = np.concatenate([patches, repeats], axis=0)
    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
    return torch.from_numpy(flatten_patches.copy()), grid_thw




# ---------------------------------------------------------------------------
# Constructor from a Qwen3-VL `vision_config` dict (text_encoder/config.json)
# ---------------------------------------------------------------------------


def build_vision_tower(vision_config: dict) -> Qwen3VLVisionModel:
    """Construct the tower from the `vision_config` block of the conditioner's config.json."""
    return Qwen3VLVisionModel(
        hidden_size=vision_config["hidden_size"],
        intermediate_size=vision_config["intermediate_size"],
        num_heads=vision_config["num_heads"],
        depth=vision_config["depth"],
        patch_size=vision_config["patch_size"],
        temporal_patch_size=vision_config["temporal_patch_size"],
        spatial_merge_size=vision_config["spatial_merge_size"],
        in_channels=vision_config.get("in_channels", 3),
        out_hidden_size=vision_config["out_hidden_size"],
        num_position_embeddings=vision_config["num_position_embeddings"],
        deepstack_visual_indexes=vision_config.get("deepstack_visual_indexes", []),
    )
