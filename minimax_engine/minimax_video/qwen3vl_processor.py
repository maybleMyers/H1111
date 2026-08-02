# Qwen3-VL processing glue for the MiniMax-H3 conditioner, vendored because the environment
# pins transformers 4.46.x (no Qwen3VLProcessor). Ports, faithfully:
#   * the image preprocessing (Qwen2VLImageProcessor math — smart_resize + patchify; the
#     resize itself lives in qwen3vl_vision.preprocess_image),
#   * the video preprocessing (Qwen3VLVideoProcessor._preprocess at transformers v4.57.0:
#     per-video smart_resize over the total pixel budget, bicubic+antialias resize,
#     temporal_patch_size stacking -> pixel_values_videos / video_grid_thw),
#   * `create_mm_token_type_ids` (runs of image-pad -> 1, video-pad -> 2, text -> 0),
#   * `get_rope_index` for a single unpadded sequence (text sequential, vision blocks 3D).

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
IMAGE_PAD_TOKEN = "<|image_pad|>"
VIDEO_PAD_TOKEN = "<|video_pad|>"


def load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image_preprocessor_config(processor_dir: str) -> dict:
    """Image preprocessing params from <ckpt>/processor/preprocessor_config.json (with Qwen3-VL defaults)."""
    defaults = {
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": (0.5, 0.5, 0.5),
        "image_std": (0.5, 0.5, 0.5),
        "min_pixels": 65536,
        "max_pixels": 16777216,
    }
    path = os.path.join(processor_dir, "preprocessor_config.json")
    if not os.path.exists(path):
        return defaults
    cfg = load_json_config(path)
    size = cfg.get("size") or {}
    return {
        "patch_size": cfg.get("patch_size", defaults["patch_size"]),
        "temporal_patch_size": cfg.get("temporal_patch_size", defaults["temporal_patch_size"]),
        "merge_size": cfg.get("merge_size", defaults["merge_size"]),
        "image_mean": tuple(cfg.get("image_mean", defaults["image_mean"])),
        "image_std": tuple(cfg.get("image_std", defaults["image_std"])),
        "min_pixels": size.get("shortest_edge", cfg.get("min_pixels", defaults["min_pixels"])),
        "max_pixels": size.get("longest_edge", cfg.get("max_pixels", defaults["max_pixels"])),
    }


def load_video_preprocessor_config(processor_dir: str) -> dict:
    """Video preprocessing params from <ckpt>/processor/video_preprocessor_config.json (Qwen3-VL defaults)."""
    defaults = {
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": (0.5, 0.5, 0.5),
        "image_std": (0.5, 0.5, 0.5),
        # Qwen3VLVideoProcessor.size: total-pixel budget over the whole clip.
        "min_pixels": 128 * 32 * 32,
        "max_pixels": 32 * 32 * 768,
    }
    path = os.path.join(processor_dir, "video_preprocessor_config.json")
    if not os.path.exists(path):
        return defaults
    cfg = load_json_config(path)
    size = cfg.get("size") or {}
    return {
        "patch_size": cfg.get("patch_size", defaults["patch_size"]),
        "temporal_patch_size": cfg.get("temporal_patch_size", defaults["temporal_patch_size"]),
        "merge_size": cfg.get("merge_size", defaults["merge_size"]),
        "image_mean": tuple(cfg.get("image_mean", defaults["image_mean"])),
        "image_std": tuple(cfg.get("image_std", defaults["image_std"])),
        "min_pixels": size.get("shortest_edge", defaults["min_pixels"]),
        "max_pixels": size.get("longest_edge", defaults["max_pixels"]),
    }


def video_smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Qwen3VLVideoProcessor smart_resize: the pixel budget covers the whole clip, not one frame."""
    import math

    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess_video(frames: np.ndarray, config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """`[T, H, W, C]` uint8 frames -> (`pixel_values_videos [N_patches, C*tp*p*p]`, `video_grid_thw [1, 3]`).

    Mirrors Qwen3VLVideoProcessor._preprocess with `do_sample_frames=False`: the caller has
    already picked the frames (MiniMax-H3 samples at 2 fps in `sample_reference_video_frames`).
    The resize is bicubic with antialias, matching the transformers fast video processor.
    """
    patch_size = config["patch_size"]
    temporal_patch_size = config["temporal_patch_size"]
    merge_size = config["merge_size"]
    factor = patch_size * merge_size

    video = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()  # [T, C, H, W]
    num_frames, _, height, width = video.shape
    resized_height, resized_width = video_smart_resize(
        num_frames=num_frames,
        height=height,
        width=width,
        temporal_factor=temporal_patch_size,
        factor=factor,
        min_pixels=config["min_pixels"],
        max_pixels=config["max_pixels"],
    )
    video = F.interpolate(video, size=(resized_height, resized_width), mode="bicubic", antialias=True)

    mean = torch.tensor(config["image_mean"]).view(1, -1, 1, 1)
    std = torch.tensor(config["image_std"]).view(1, -1, 1, 1)
    video = (video / 255.0 - mean) / std

    patches = video[None]  # [1, T, C, H, W]
    if patches.shape[1] % temporal_patch_size != 0:
        repeats = patches[:, -1:].repeat(1, temporal_patch_size - (patches.shape[1] % temporal_patch_size), 1, 1, 1)
        patches = torch.cat([patches, repeats], dim=1)
    _, total_frames, channel = patches.shape[:3]
    grid_t = total_frames // temporal_patch_size
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

    patches = patches.view(
        1,
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
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
    )
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
    return flatten_patches.contiguous(), grid_thw


def create_mm_token_type_ids(token_ids: list[int], image_pad_id: int, video_pad_id: int) -> list[int]:
    """Per-token modality type: `0` text, `1` image pad, `2` video pad (Qwen3VLProcessor parity)."""
    return [1 if t == image_pad_id else 2 if t == video_pad_id else 0 for t in token_ids]


def get_rope_index(
    token_ids: list[int],
    mm_token_type_ids: list[int],
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Qwen3-VL `get_rope_index` for one unpadded sequence -> `[3, 1, T]` long tensor.

    Text tokens advance all three axes together; every vision block lays a `(t, h, w)` grid
    starting one past the running maximum. Video grids are split per frame block with `t = 1`
    (Qwen3-VL encodes video time through the timestamp text between blocks, so `t_index` stays 0
    inside each block), matching the `repeat_interleave` split of the reference.
    """
    if image_grid_thw is None and video_grid_thw is None:
        pos = torch.arange(len(token_ids), dtype=torch.long)
        return pos.view(1, 1, -1).expand(3, 1, -1).contiguous()

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
        video_grid_thw[:, 0] = 1

    types = mm_token_type_ids
    chunks: list[torch.Tensor] = []
    image_index = video_index = 0
    st = 0
    seq_len = len(token_ids)
    while st < seq_len:
        if types[st] == 0:
            ed = st
            while ed < seq_len and types[ed] == 0:
                ed += 1
            st_idx = int(chunks[-1].max()) + 1 if chunks else 0
            chunks.append(torch.arange(ed - st).view(1, -1).expand(3, -1) + st_idx)
            st = ed
        else:
            if types[st] == 1:
                grid = image_grid_thw[image_index]
                image_index += 1
            else:
                grid = video_grid_thw[video_index]
                video_index += 1
            t = int(grid[0])
            h = int(grid[1]) // spatial_merge_size
            w = int(grid[2]) // spatial_merge_size
            num_pad = t * h * w
            if any(types[st + k] != types[st] for k in range(num_pad)):
                raise ValueError("A vision block's pad-token run is shorter than its grid declares.")
            st_idx = int(chunks[-1].max()) + 1 if chunks else 0
            t_index = torch.arange(t).view(-1, 1).expand(-1, h * w).flatten()
            h_index = torch.arange(h).view(1, -1, 1).expand(t, -1, w).flatten()
            w_index = torch.arange(w).view(1, 1, -1).expand(t, h, -1).flatten()
            chunks.append(torch.stack([t_index, h_index, w_index]) + st_idx)
            st += num_pad

    return torch.cat(chunks, dim=1).view(3, 1, -1).contiguous()
