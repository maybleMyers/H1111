#!/usr/bin/env python
# cosmos_generate_video.py — NVIDIA Cosmos 3 video/image generation for H1111.
#
# Native integration built on the vendored cosmos_video/ package (diffusers-main
# Cosmos3 port) plus H1111 shared infra: block swap, fp8, LoRA merge-at-load,
# latent previews, latent save / decode-only.
#
# Modes (inferred from inputs, or forced with --task):
#   t2i, t2v, i2v, v2v, transfer (edge/blur/depth/seg/wsm),
#   forward_dynamics, inverse_dynamics, policy — each ± --enable_sound
#   (sound requires a Super/Nano checkpoint).

import argparse
import glob
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

# Runnable from anywhere: the H1111 repo root provides utils/, modules/, and
# blissful_tuner/; this directory (cosmos_engine/) provides cosmos_video/.
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_here), _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.device_utils import clean_memory_on_device
from utils.model_utils import str_to_dtype
from utils.safetensors_utils import load_safetensors, mem_eff_save_file

from cosmos_video import configs as cfg
from cosmos_video.model_loader import (
    detect_variant,
    load_distilled_scheduler,
    load_scheduler,
    load_sound_tokenizer,
    load_text_tokenizer,
    load_transformer,
    load_vae,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _StopGeneration(Exception):
    """Raised from the step callback when a stop-signal file is found."""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosmos3 generation (H1111)")

    # model / paths
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Cosmos3 HF snapshot dir (diffusers layout)")
    parser.add_argument("--dit", type=str, default=None, help="override transformer dir or merged .safetensors")
    parser.add_argument("--vae", type=str, default=None, help="override vae dir")
    parser.add_argument("--task", type=str, default=None, choices=list(cfg.TASKS), help="force task (default: infer)")
    parser.add_argument("--variant", type=str, default=None, choices=["super", "nano", "edge"], help="override variant detection")
    parser.add_argument("--distilled", action="store_true", help="DMD2 4-step checkpoint (FlowMatchEuler + fixed sigmas)")

    # core sampling
    parser.add_argument("--prompt", type=str, default=None, help="prompt text, JSON caption string, or path to .json/.txt")
    parser.add_argument("--negative_prompt", type=str, default=None, help="negative prompt (same forms as --prompt)")
    parser.add_argument("--no_default_negative_prompt", action="store_true",
                        help="do not auto-load <ckpt_dir>/assets/negative_prompt.json when --negative_prompt is unset")
    parser.add_argument("--video_size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--resolution", type=str, default=None, choices=sorted(cfg.VIDEO_RES_SIZE_INFO))
    parser.add_argument("--aspect_ratio", type=str, default="16:9", choices=["1:1", "4:3", "3:4", "16:9", "9:16"])
    parser.add_argument("--video_length", type=int, default=None, help="frames (4k+1 cadence; 1 = image)")
    parser.add_argument("--fps", type=int, default=None, help="output fps (10/16/24/30 supported by the model)")
    parser.add_argument("--infer_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--flow_shift", type=float, default=None, help="default: per-resolution (256->3, 480->5, 720->10)")
    parser.add_argument("--sigma_max", type=float, default=None)
    parser.add_argument("--guidance_interval", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="apply CFG only when timestep in [LO,HI] (0-1000)")
    parser.add_argument("--normalize_cfg", action="store_true", help="rescale CFG velocity to cond norm")
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc"], help="kept for UI parity")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_outputs", type=int, default=1)
    parser.add_argument("--cpu_noise", action="store_true", help="draw initial noise on CPU for reproducibility")
    parser.add_argument("--use_system_prompt", action="store_true", help="prepend the model's system prompt")

    # prompt templates
    parser.add_argument("--no_resolution_template", action="store_true", help="do not inject resolution metadata into prompt")
    parser.add_argument("--no_duration_template", action="store_true", help="do not inject duration/fps metadata into prompt")

    # i2v / v2v
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--condition_frame_indexes", type=int, nargs="*", default=None,
                        help="latent frame indexes held clean for v2v (default 0 1)")
    parser.add_argument("--condition_video_keep", type=str, default="first", choices=["first", "last"])

    # transfer control
    for hint in cfg.TRANSFER_HINTS:
        parser.add_argument(f"--control_{hint}", type=str, default=None, help=f"{hint} control video path")
    parser.add_argument("--control_path", type=str, default=None, help="generic control video (with --control_type)")
    parser.add_argument("--control_type", type=str, default=None, choices=list(cfg.TRANSFER_HINTS))
    parser.add_argument("--control_weight", type=float, nargs="*", default=None, help="per-hint weights")
    parser.add_argument("--control_guidance", type=float, default=None, help="control-CFG scale (1.0 disables)")
    parser.add_argument("--control_guidance_interval", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="apply control-CFG only when timestep in (LO,HI) (0-1000)")
    parser.add_argument("--edge_threshold", type=str, default="medium", choices=sorted(cfg.EDGE_THRESHOLD_PRESETS))
    parser.add_argument("--blur_strength", type=str, default="medium", choices=sorted(cfg.BLUR_STRENGTH_PRESETS))
    parser.add_argument("--no_emphasize_control_in_prompt", action="store_true")
    parser.add_argument("--num_frames_per_chunk", type=int, default=None, help="chunked long-video transfer")
    parser.add_argument("--num_conditional_frames", type=int, default=None)
    parser.add_argument("--num_first_chunk_conditional_frames", type=int, default=None)

    # action modes
    parser.add_argument("--domain_name", type=str, default=None, choices=sorted(cfg.EMBODIMENT_TO_DOMAIN_ID))
    parser.add_argument("--action_path", type=str, default=None, help=".json or .npy action chunk [T, raw_dim]")
    parser.add_argument("--action_data", type=str, default=None, help="inline JSON action array")
    parser.add_argument("--action_chunk_size", type=int, default=None)
    parser.add_argument("--raw_action_dim", type=int, default=None, help="override the per-domain action dim")
    parser.add_argument("--view_point", type=str, default="ego_view")
    parser.add_argument("--policy", action="store_true", help="policy mode (with --image_path + prompt instruction)")
    parser.add_argument("--instruction", type=str, default=None, help="alias for --prompt in action modes")
    parser.add_argument("--action_output", type=str, default=None, help="path for predicted-action JSON")

    # sound
    parser.add_argument("--enable_sound", action="store_true")
    parser.add_argument("--audio_save_path", type=str, default=None, help="also keep the .wav here")

    # memory / performance
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--attn_mode", type=str, default="torch",
                        choices=["torch", "sdpa", "flash", "flashattn", "flash2", "flash3", "sageattn", "xformers"])
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--no_text_cache", action="store_true",
                        help="disable the text (und) pathway K/V cache (cache is bit-identical; this is a debug switch)")
    parser.add_argument("--fp8", action="store_true", help="cast transformer linear weights to e4m3")
    parser.add_argument("--fp8_scaled", action="store_true", help="scaled fp8 quantization with monkey patch")
    parser.add_argument("--fp8_fast", action="store_true", help="use scaled_mm fp8 matmul (with --fp8_scaled)")
    parser.add_argument("--dit_dtype", type=str, default="bfloat16")
    parser.add_argument("--vae_dtype", type=str, default="float32")
    parser.add_argument("--vae_tiling", action="store_true")
    parser.add_argument("--offload_text_encoder", action="store_true",
                        help="kept for UI parity (text conditioning is token ids; nothing to offload)")

    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", default=None)
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=None)
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None)
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None)

    # output
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--output_type", type=str, default="video", choices=["video", "images", "latent", "both"])
    parser.add_argument("--output_filename", type=str, default=None, help="explicit output file path (queue)")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="decode-only: latent .safetensors")
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--preview", type=int, default=None, metavar="N", help="write latent preview every N steps")
    parser.add_argument("--preview_suffix", type=str, default=None)

    args = parser.parse_args()

    if args.latent_path is None and args.ckpt_dir is None:
        parser.error("--ckpt_dir is required (except in --latent_path decode-only mode)")
    if args.latent_path is None and args.prompt is None and args.instruction is None:
        parser.error("--prompt (or --instruction) is required")
    if args.video_size is None and args.resolution is None and args.latent_path is None and args.domain_name is None:
        args.resolution = "720"
    return args


# ---------------------------------------------------------------------------
# task detection / defaults
# ---------------------------------------------------------------------------


def detect_task(args) -> str:
    if args.task is not None:
        return args.task
    if args.domain_name is not None:
        if args.policy:
            return "policy"
        if args.action_path or args.action_data:
            return "forward_dynamics"
        return "inverse_dynamics" if args.video_path else "policy"
    if any(getattr(args, f"control_{h}") for h in cfg.TRANSFER_HINTS) or args.control_path:
        return "transfer"
    if args.video_path:
        return "v2v"
    if args.image_path:
        return "i2v"
    if args.video_length == 1:
        return "t2i"
    return "t2v"


def active_hints(args) -> List[str]:
    hints = [h for h in cfg.TRANSFER_HINTS if getattr(args, f"control_{h}")]
    if args.control_path and args.control_type and args.control_type not in hints:
        hints.append(args.control_type)
    return [h for h in cfg.TRANSFER_HINTS if h in hints]  # deterministic upstream order


def setup_args(args, task: str):
    args.active_hints = active_hints(args)
    cfg.apply_task_defaults(args, task)

    if task == "transfer":
        # Chunked long-video transfer defaults (framework OmniSampleArgs transfer defaults).
        chunk_defaults = cfg.TRANSFER_CHUNK_DEFAULTS
        if args.num_frames_per_chunk is None:
            args.num_frames_per_chunk = chunk_defaults["num_frames_per_chunk"]
        if args.num_conditional_frames is None:
            args.num_conditional_frames = chunk_defaults["num_conditional_frames"]
        if args.num_first_chunk_conditional_frames is None:
            args.num_first_chunk_conditional_frames = chunk_defaults["num_first_chunk_conditional_frames"]
        args.max_frames = chunk_defaults["max_frames"]
        if args.control_guidance is None:
            args.control_guidance = 1.0

    if task in cfg.ACTION_TASKS:
        if args.domain_name is None:
            raise ValueError(f"{task} requires --domain_name")
        if args.action_chunk_size is None:
            args.action_chunk_size = cfg.ACTION_DEFAULTS["action_chunk_size"]
        if args.video_length is None:
            args.video_length = args.action_chunk_size + 1
        if args.fps is None:
            args.fps = 24
        if args.prompt is None:
            args.prompt = args.instruction or ""

    if args.video_size is not None:
        height, width = args.video_size
    else:
        resolution = args.resolution or "720"
        if task in cfg.ACTION_TASKS and args.resolution is None:
            resolution = "480"
        width, height = cfg.resolve_video_size(resolution, args.aspect_ratio)
    args.height, args.width = height, width

    # 0/negative means "unset" for these (a zero sigma_max corrupts the sigma
    # ramp and a degenerate guidance interval silently disables CFG)
    if args.flow_shift is not None and args.flow_shift <= 0:
        logger.warning("ignoring non-positive --flow_shift; using per-resolution default")
        args.flow_shift = None
    if args.sigma_max is not None and args.sigma_max <= 0:
        logger.warning("ignoring non-positive --sigma_max; using checkpoint default")
        args.sigma_max = None
    if args.guidance_interval is not None and args.guidance_interval[1] <= args.guidance_interval[0]:
        logger.warning("ignoring degenerate --guidance_interval; CFG applies at every step")
        args.guidance_interval = None
    if args.flow_shift is None:
        args.flow_shift = cfg.resolve_flow_shift(args.height, args.width)

    if args.video_length is None:
        args.video_length = 189
    rounded = cfg.round_num_frames(args.video_length)
    if rounded != args.video_length:
        logger.warning(f"video_length {args.video_length} is not 4k+1; rounding up to {rounded}")
        args.video_length = rounded
    if args.fps is None:
        args.fps = 24

    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
    return args


# ---------------------------------------------------------------------------
# input loading
# ---------------------------------------------------------------------------


def read_text_or_path(value: Optional[str]) -> Optional[str]:
    """Prompts can be plain text, an inline JSON object, or a file path."""
    if value is None:
        return None
    if os.path.exists(value) and value.lower().endswith((".json", ".txt")):
        with open(value, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if value.lower().endswith(".json"):
            return json.dumps(json.loads(content))
        return content
    return value


def resolve_negative_prompt(args) -> Optional[str]:
    """Resolve the negative prompt: explicit --negative_prompt (text or file) wins; otherwise,
    for non-distilled checkpoints, fall back to <ckpt_dir>/assets/negative_prompt.json when it
    exists (disable with --no_default_negative_prompt). Distilled checkpoints run without CFG,
    so the default negative prompt is never loaded for them."""
    negative_prompt = read_text_or_path(args.negative_prompt)
    if (
        negative_prompt is None
        and not getattr(args, "distilled", False)
        and not getattr(args, "no_default_negative_prompt", False)
        and args.ckpt_dir
    ):
        default_path = os.path.join(args.ckpt_dir, "assets", "negative_prompt.json")
        if os.path.exists(default_path):
            negative_prompt = read_text_or_path(default_path)
            logger.info(f"using default negative prompt from {default_path}")
    return negative_prompt


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_video_frames(path: str, max_frames: Optional[int] = None) -> List[Image.Image]:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return frames


def compute_control_frames(args, task_frames: List[Image.Image], hint: str) -> List[Image.Image]:
    """On-the-fly edge/blur hints from the source video (framework parity)."""
    import cv2

    out = []
    if hint == "edge":
        t_lower, t_upper = cfg.EDGE_THRESHOLD_PRESETS[args.edge_threshold]
        for frame in task_frames:
            arr = np.array(frame)
            edges = cv2.Canny(arr, t_lower, t_upper)
            out.append(Image.fromarray(np.stack([edges] * 3, axis=-1)))
    elif hint == "blur":
        downup, gauss_downup = cfg.BLUR_STRENGTH_PRESETS[args.blur_strength]
        for frame in task_frames:
            arr = np.array(frame)
            h, w = arr.shape[:2]
            small = cv2.resize(arr, (max(1, w // downup), max(1, h // downup)), interpolation=cv2.INTER_AREA)
            if gauss_downup > 1:
                small = cv2.GaussianBlur(small, (5, 5), 0)
            out.append(Image.fromarray(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)))
    else:
        raise ValueError(f"{hint} control cannot be computed on the fly; provide --control_{hint}")
    return out


# ---------------------------------------------------------------------------
# chunked long-video transfer (framework inference/transfer.py parity)
# ---------------------------------------------------------------------------


def get_num_chunks(total_frames: int, frames_per_chunk: int, conditional_frames: int):
    """Return ``(num_chunks, stride)`` for autoregressive chunking (transfer.py:_get_num_chunks)."""
    if frames_per_chunk <= 0:
        raise ValueError("frames_per_chunk must be positive")
    if total_frames <= frames_per_chunk:
        return 1, frames_per_chunk
    stride = frames_per_chunk - conditional_frames
    if stride <= 0:
        raise ValueError("num_conditional_frames must be smaller than num_frames_per_chunk")
    remaining = total_frames - frames_per_chunk
    extra_chunks = remaining // stride + (1 if remaining % stride else 0)
    return 1 + extra_chunks, stride


def compute_transfer_chunk_plan(total_frames: int, num_frames_per_chunk: int, num_conditional_frames: int):
    """Chunk layout for transfer: ``(chunk_frames, num_chunks, stride, [(start, end), ...])``."""
    chunk_frames = 1 if total_frames == 1 else cfg.round_num_frames(num_frames_per_chunk)
    num_chunks, stride = get_num_chunks(total_frames, chunk_frames, num_conditional_frames)
    spans = [
        (chunk_id * stride, min(chunk_id * stride + chunk_frames, total_frames)) for chunk_id in range(num_chunks)
    ]
    return chunk_frames, num_chunks, stride, spans


def run_transfer_chunks(
    control_frames_per_hint: dict,
    source_frames,
    total_frames: int,
    chunk_frames: int,
    stride: int,
    num_chunks: int,
    num_conditional_frames: int,
    num_first_chunk_conditional_frames: int,
    generate_chunk,
) -> np.ndarray:
    """Autoregressive chunk loop (transfer.py generate_transfer_sample chunk logic).

    ``generate_chunk(chunk_id, control_chunk_frames, cond_frames, condition_frame_indexes)`` must return decoded
    uint8 frames ``[T, H, W, C]`` for the chunk (or ``None`` to stop early with what has been accumulated). Chunk 0
    conditions on ``num_first_chunk_conditional_frames`` frames of the source video; later chunks prepend the last
    ``num_conditional_frames`` decoded frames of the previous chunk and drop them from the output before
    concatenation. The concatenated video is trimmed to ``total_frames``.
    """
    outputs = []
    previous: np.ndarray | None = None
    for chunk_id in range(num_chunks):
        start = chunk_id * stride
        end = min(start + chunk_frames, total_frames)
        control_chunk = {hint: frames[start:end] for hint, frames in control_frames_per_hint.items()}

        if chunk_id == 0:
            current_conditional_frames = 0
            cond_frames = None
            if num_first_chunk_conditional_frames > 0 and source_frames:
                current_conditional_frames = min(num_first_chunk_conditional_frames, len(source_frames))
                cond_frames = source_frames[:current_conditional_frames]
        else:
            current_conditional_frames = min(num_conditional_frames, previous.shape[0])
            cond_frames = [Image.fromarray(frame) for frame in previous[-current_conditional_frames:]]

        condition_frame_indexes = (
            list(range((current_conditional_frames - 1) // cfg.VAE_TEMPORAL_COMPRESSION + 1))
            if current_conditional_frames > 0
            else None
        )
        frames = generate_chunk(chunk_id, control_chunk, cond_frames, condition_frame_indexes)
        if frames is None:
            break
        outputs.append(frames if chunk_id == 0 else frames[current_conditional_frames:])
        previous = frames
    if not outputs:
        raise RuntimeError("transfer chunk loop produced no output")
    return np.concatenate(outputs, axis=0)[:total_frames]


def load_actions(args) -> np.ndarray:
    raw_dim = args.raw_action_dim or cfg.EMBODIMENT_TO_RAW_ACTION_DIM[args.domain_name]
    if args.action_data:
        actions = np.asarray(json.loads(args.action_data), dtype=np.float32)
    elif args.action_path:
        if args.action_path.endswith(".npy"):
            actions = np.load(args.action_path).astype(np.float32)
        else:
            with open(args.action_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("actions", data.get("action"))
            actions = np.asarray(data, dtype=np.float32)
    else:
        actions = np.zeros((args.action_chunk_size, raw_dim), dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must be [T, dim], got shape {actions.shape}")
    if actions.shape[1] != raw_dim:
        raise ValueError(f"action dim {actions.shape[1]} != expected {raw_dim} for domain {args.domain_name}")
    return actions


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def build_pipeline(args, device: torch.device, task: str):
    from cosmos_video.attention import set_attention_backend
    from cosmos_video.pipeline import Cosmos3OmniPipeline

    set_attention_backend(args.attn_mode)
    dit_dtype = str_to_dtype(args.dit_dtype)
    vae_dtype = str_to_dtype(args.vae_dtype)

    lora_weights_list, lora_multipliers = None, None
    if args.lora_weight:
        from utils.lora_utils import filter_lora_state_dict

        lora_weights_list, lora_multipliers = [], []
        multipliers = args.lora_multiplier or [1.0] * len(args.lora_weight)
        for i, path in enumerate(args.lora_weight):
            sd = load_safetensors(path, device="cpu")
            sd = filter_lora_state_dict(sd, args.include_patterns, args.exclude_patterns)
            lora_weights_list.append(sd)
            lora_multipliers.append(multipliers[i] if i < len(multipliers) else 1.0)
            logger.info(f"loaded LoRA {path} x{lora_multipliers[-1]} ({len(sd)} tensors)")

    logger.info("loading transformer...")
    transformer = load_transformer(
        args.ckpt_dir,
        device=device,
        dit_dtype=dit_dtype,
        fp8=args.fp8,
        fp8_scaled=args.fp8_scaled,
        fp8_fast=args.fp8_fast,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        dit_path=args.dit,
    )
    if args.blocks_to_swap > 0:
        logger.info(f"enabling block swap: {args.blocks_to_swap} blocks")
        transformer.enable_block_swap(args.blocks_to_swap, device, supports_backward=False)
        transformer.move_to_device_except_swap_blocks(device)
        transformer.prepare_block_swap_before_forward()
    else:
        transformer.to(device)

    logger.info("loading vae...")
    vae = load_vae(args.ckpt_dir, device=device, vae_dtype=vae_dtype, vae_path=args.vae)
    if args.vae_tiling:
        vae.enable_tiling()

    sound_tokenizer = None
    if args.enable_sound:
        sound_tokenizer = load_sound_tokenizer(args.ckpt_dir, device=device)
        if sound_tokenizer is None:
            raise ValueError(
                "--enable_sound requires a checkpoint with a sound_tokenizer/ component "
                "(Cosmos3-Super or Cosmos3-Nano; Cosmos3-Edge does not generate sound)"
            )
    if args.enable_sound and task in cfg.ACTION_TASKS:
        raise ValueError("--enable_sound is not supported for action modes")

    tokenizer, chat_template = load_text_tokenizer(args.ckpt_dir)

    if args.distilled:
        scheduler, distilled_sigmas = load_distilled_scheduler(args.ckpt_dir)
        if scheduler is None:
            raise ValueError("--distilled set but no distilled_sigmas found in modular_model_index.json")
        args.distilled_sigmas = distilled_sigmas
        if args.guidance_scale not in (None, 1.0):
            logger.warning("distilled checkpoints run without CFG; forcing guidance_scale=1.0")
        args.guidance_scale = 1.0
        args.infer_steps = len(distilled_sigmas)
    else:
        scheduler = load_scheduler(args.ckpt_dir, flow_shift=args.flow_shift, sigma_max=args.sigma_max)

    pipe = Cosmos3OmniPipeline(
        transformer=transformer,
        text_tokenizer=tokenizer,
        vae=vae,
        scheduler=scheduler,
        sound_tokenizer=sound_tokenizer,
        default_use_system_prompt=args.use_system_prompt,
        use_native_flow_schedule=True,
    )
    if chat_template is not None:
        pipe._chat_template = chat_template
    return pipe


def make_step_callback(args, previewer_holder: dict, total_steps: int):
    stop_base = args.output_filename

    def callback(pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs.get("latents")
        if args.preview and latents is not None and (step + 1) % args.preview == 0 and step + 1 < total_steps:
            try:
                # Preview the scheduler's clean-image (x0) estimate, not the noisy
                # sample x_t: UniPC runs in predict_x0 mode and keeps its converted
                # model outputs, so mid-denoise previews show content instead of noise.
                preview_latents = latents
                model_outputs = getattr(pipe.scheduler, "model_outputs", None)
                if model_outputs:
                    for m in reversed(model_outputs):
                        if m is not None:
                            preview_latents = m
                            break
                while preview_latents.ndim > 5:  # scheduler stores an extra batch dim
                    preview_latents = preview_latents.squeeze(0)
                if previewer_holder.get("previewer") is None:
                    from blissful_tuner.latent_preview import LatentPreviewer

                    previewer_holder["previewer"] = LatentPreviewer(
                        args, None, None, latents.device, torch.float32, model_type="cosmos"
                    )
                previewer_holder["previewer"].preview(preview_latents.float(), step, preview_suffix=args.preview_suffix)
            except Exception as e:  # previews must never kill a run
                logger.warning(f"preview failed at step {step}: {e}")
        if stop_base is not None and os.path.exists(stop_base + ".stop_decode"):
            try:
                os.remove(stop_base + ".stop_decode")
            except OSError:
                pass
            previewer_holder["partial_latents"] = latents
            raise _StopGeneration()
        return callback_kwargs

    return callback


def run_generation(args, pipe, task: str, device: torch.device, seed: int):
    from cosmos_video.pipeline import CosmosActionCondition

    prompt = read_text_or_path(args.prompt)
    negative_prompt = resolve_negative_prompt(args)

    image = load_image(args.image_path) if args.image_path else None
    video = None
    action = None
    hints = args.active_hints

    control_videos = None
    transfer_plan = None
    source = None

    if task == "v2v":
        video = load_video_frames(args.video_path)
    elif task == "transfer":
        frame_cap = min(args.video_length, getattr(args, "max_frames", args.video_length) or args.video_length)
        source = load_video_frames(args.video_path, max_frames=frame_cap) if args.video_path else None
        control_videos = {}
        for hint in hints:
            path = getattr(args, f"control_{hint}") or (
                args.control_path if args.control_type == hint else None
            )
            if path:
                control_videos[hint] = load_video_frames(path, max_frames=frame_cap)
            elif source is not None:
                control_videos[hint] = compute_control_frames(args, source, hint)
            else:
                raise ValueError(f"--control_{hint} requires a control video path (or --video_path for edge/blur)")
        if args.num_first_chunk_conditional_frames > 0 and source is None:
            raise ValueError("--num_first_chunk_conditional_frames > 0 requires --video_path")
        total_frames = len(next(iter(control_videos.values())))
        transfer_plan = compute_transfer_chunk_plan(
            total_frames, args.num_frames_per_chunk, args.num_conditional_frames
        )
        if not args.no_emphasize_control_in_prompt and prompt and not prompt.lstrip().startswith("{"):
            prompt = prompt + cfg.EMPHASIZE_CONTROL_PROMPT_SUFFIX.format(hints="+".join(hints))
    elif task in cfg.ACTION_TASKS:
        actions = load_actions(args)
        action_video = (
            load_video_frames(args.video_path, max_frames=args.action_chunk_size + 1) if args.video_path else None
        )
        action_kwargs = dict(
            mode=task,
            chunk_size=args.action_chunk_size,
            domain_name=args.domain_name,
            raw_actions=torch.from_numpy(actions),
            view_point=args.view_point,
        )
        # image and video are mutually exclusive on CosmosActionCondition
        if action_video is not None:
            action_kwargs["video"] = action_video
        else:
            action_kwargs["image"] = image
        if args.resolution is not None:
            action_kwargs["resolution_tier"] = int(args.resolution)
        action = CosmosActionCondition(**action_kwargs)
        image = None

    generator_device = "cpu" if args.cpu_noise else device
    generator = torch.Generator(device=generator_device).manual_seed(seed)

    previewer_holder = {}
    callback = make_step_callback(args, previewer_holder, args.infer_steps)

    call_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        video=video,
        num_frames=args.video_length,
        height=args.height,
        width=args.width,
        fps=float(args.fps),
        num_inference_steps=args.infer_steps,
        guidance_scale=args.guidance_scale,
        guidance_interval=tuple(args.guidance_interval) if args.guidance_interval else None,
        normalize_cfg=args.normalize_cfg,
        enable_sound=args.enable_sound,
        generator=generator,
        action=action,
        output_type="latent",
        callback_on_step_end=callback,
        add_resolution_template=not args.no_resolution_template,
        add_duration_template=not args.no_duration_template,
        use_und_cache=not args.no_text_cache,
    )
    if args.condition_frame_indexes is not None:
        call_kwargs["condition_frame_indexes_vision"] = tuple(args.condition_frame_indexes)
    call_kwargs["condition_video_keep"] = args.condition_video_keep

    if task == "transfer":
        weights = list(args.control_weight or [])
        weights = (weights + [1.0] * len(hints))[: len(hints)]  # pad with 1.0, one per active hint
        transfer_kwargs = dict(
            control_weights=weights,
            control_guidance=args.control_guidance if args.control_guidance is not None else 1.0,
            control_guidance_interval=(
                tuple(args.control_guidance_interval) if args.control_guidance_interval else None
            ),
        )
        chunk_frames, num_chunks, stride, chunk_spans = transfer_plan
        total_frames = chunk_spans[-1][1]

        if num_chunks == 1:
            # Single chunk goes through the normal path (latent saving preserved).
            call_kwargs.update(transfer_kwargs)
            call_kwargs["control_videos"] = control_videos
            call_kwargs["num_frames"] = cfg.round_num_frames(min(total_frames, chunk_frames))
            first_cond = 0
            if args.num_first_chunk_conditional_frames > 0 and source:
                first_cond = min(args.num_first_chunk_conditional_frames, len(source))
            if first_cond > 0:
                call_kwargs["video"] = source[:first_cond]
                call_kwargs["condition_frame_indexes_vision"] = tuple(
                    range((first_cond - 1) // cfg.VAE_TEMPORAL_COMPRESSION + 1)
                )
            else:
                call_kwargs["video"] = None
                call_kwargs.pop("condition_frame_indexes_vision", None)
        else:
            if args.output_type in ("latent", "both"):
                logger.warning("chunked transfer saves the concatenated decoded video only; skipping latent output")
            logger.info(
                f"chunked transfer: {total_frames} frames -> {num_chunks} chunks of {chunk_frames} "
                f"(stride {stride}, cond {args.num_conditional_frames})"
            )
            stop_state = {}

            def generate_chunk(chunk_id, control_chunk, cond_frames, condition_frame_indexes):
                if stop_state.get("stopped"):
                    return None
                chunk_kwargs = dict(call_kwargs)
                chunk_kwargs.update(transfer_kwargs)
                chunk_kwargs["control_videos"] = control_chunk
                chunk_kwargs["num_frames"] = chunk_frames
                chunk_kwargs["video"] = cond_frames
                if condition_frame_indexes is not None:
                    chunk_kwargs["condition_frame_indexes_vision"] = tuple(condition_frame_indexes)
                else:
                    chunk_kwargs.pop("condition_frame_indexes_vision", None)
                chunk_kwargs["generator"] = torch.Generator(device=generator_device).manual_seed(seed + chunk_id)
                try:
                    with torch.no_grad():
                        chunk_latents = pipe(**chunk_kwargs).video
                except _StopGeneration:
                    logger.info(f"stop signal received during chunk {chunk_id}; decoding current latents")
                    stop_state["stopped"] = True
                    chunk_latents = previewer_holder.get("partial_latents")
                    if chunk_latents is None:
                        return None
                return decode_latents(chunk_latents, pipe.vae)

            frames = run_transfer_chunks(
                control_videos,
                source,
                total_frames,
                chunk_frames,
                stride,
                num_chunks,
                args.num_conditional_frames,
                args.num_first_chunk_conditional_frames,
                generate_chunk,
            )
            return None, None, None, frames

    try:
        with torch.no_grad():
            result = pipe(**call_kwargs)
        latents, sound, action_out = result.video, result.sound, result.action
    except _StopGeneration:
        logger.info("stop signal received; decoding current latents")
        latents = previewer_holder.get("partial_latents")
        sound, action_out = None, None
        if latents is None:
            raise RuntimeError("stopped before any denoising step completed")
    return latents, sound, action_out, None


# ---------------------------------------------------------------------------
# decode + save
# ---------------------------------------------------------------------------


def decode_latents(latents: torch.Tensor, vae) -> np.ndarray:
    """Normalized Cosmos3 latents -> uint8 frames [T, H, W, C] (pipeline decode math)."""
    device = next(vae.parameters()).device
    dtype = vae.dtype
    latents = latents.to(device=device, dtype=dtype)
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    z_raw = latents / inv_std + mean
    with torch.no_grad():
        decoded = vae.decode(z_raw).sample
    video = ((decoded[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return video.permute(1, 2, 3, 0).cpu().numpy()  # C,T,H,W -> T,H,W,C


def save_video(frames: np.ndarray, path: str, fps: int):
    try:
        import av

        container = av.open(path, mode="w")
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16"}
        for frame in frames:
            av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    except ImportError:
        import imageio

        imageio.mimwrite(path, list(frames), fps=fps, quality=8)


def save_audio_wav(sound: torch.Tensor, path: str, sample_rate: int):
    import wave

    audio = sound.detach().float().cpu().clamp(-1, 1)
    if audio.ndim == 3:
        audio = audio[0]
    pcm = (audio.transpose(0, 1).numpy() * 32767.0).astype(np.int16)  # [N, ch]
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def mux_audio(video_path: str, audio_path: str, out_path: str):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-i", audio_path,
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg mux failed: {result.stderr.strip()}; keeping silent video")
        return video_path
    return out_path


def build_metadata(args, task: str, seed: int) -> dict:
    meta = {
        "model": "cosmos3",
        "task": task,
        "prompt": str(args.prompt),
        "negative_prompt": str(args.negative_prompt),
        "seed": str(seed),
        "height": str(args.height),
        "width": str(args.width),
        "video_length": str(args.video_length),
        "fps": str(args.fps),
        "infer_steps": str(args.infer_steps),
        "guidance_scale": str(args.guidance_scale),
        "flow_shift": str(args.flow_shift),
        "enable_sound": str(args.enable_sound),
    }
    if task in cfg.ACTION_TASKS:
        meta.update({"domain_name": str(args.domain_name), "action_chunk_size": str(args.action_chunk_size)})
    return meta


def save_output(
    args, task: str, seed: int, latents, sound, action_out, vae, sound_tokenizer=None, index: int = 0, frames=None
):
    os.makedirs(args.save_path, exist_ok=True)
    if args.output_filename:
        base = os.path.splitext(args.output_filename)[0]
        if index > 0:
            base = f"{base}_{index}"
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = os.path.join(args.save_path, f"cosmos_{task}_{stamp}_{seed}")

    metadata = None if args.no_metadata else build_metadata(args, task, seed)
    saved = []

    output_type = args.output_type
    if frames is not None and output_type == "latent":
        # Chunked transfer has no latents to save; fall back to the decoded video.
        output_type = "video"

    if output_type in ("latent", "both"):
        if latents is None:
            logger.warning("no latents available for this run (chunked transfer); skipping latent output")
        else:
            latent_path = base + "_latent.safetensors"
            mem_eff_save_file({"latent": latents.detach().cpu().contiguous()}, latent_path, metadata=metadata)
            saved.append(latent_path)

    if output_type in ("video", "images", "both"):
        if frames is None:
            frames = decode_latents(latents, vae)
        if task == "t2i" or output_type == "images" or frames.shape[0] == 1:
            if frames.shape[0] == 1 or task == "t2i":
                img_path = base + ".png"
                Image.fromarray(frames[0]).save(img_path)
                saved.append(img_path)
            else:
                img_dir = base + "_frames"
                os.makedirs(img_dir, exist_ok=True)
                for i, frame in enumerate(frames):
                    Image.fromarray(frame).save(os.path.join(img_dir, f"frame_{i:05d}.png"))
                saved.append(img_dir)
        if output_type in ("video", "both") and frames.shape[0] > 1:
            video_path = base + ".mp4"
            save_video(frames, video_path, args.fps)
            if sound is not None and sound_tokenizer is not None:
                wav_path = args.audio_save_path or (base + ".wav")
                sample_rate = int(getattr(sound_tokenizer.config, "sampling_rate", 48000))
                save_audio_wav(sound, wav_path, sample_rate)
                mux_path = base + "_audio.mp4"
                final = mux_audio(video_path, wav_path, mux_path)
                if final == mux_path and os.path.exists(mux_path):
                    os.replace(mux_path, video_path)
                if args.audio_save_path is None and os.path.exists(wav_path):
                    os.remove(wav_path)
            saved.append(video_path)

    if action_out is not None:
        action_path = args.action_output or (base + "_actions.json")
        actions = action_out[0] if isinstance(action_out, (list, tuple)) else action_out
        payload = {
            "domain": args.domain_name,
            "chunk_size": args.action_chunk_size,
            "actions": actions.float().cpu().numpy().tolist(),
        }
        with open(action_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        saved.append(action_path)

    for p in saved:
        logger.info(f"saved: {p}")
    return saved


# ---------------------------------------------------------------------------
# decode-only mode
# ---------------------------------------------------------------------------


def decode_only(args, device: torch.device):
    vae_dtype = str_to_dtype(args.vae_dtype)
    vae = load_vae(args.ckpt_dir, device=device, vae_dtype=vae_dtype, vae_path=args.vae)
    if args.vae_tiling:
        vae.enable_tiling()
    for latent_file in args.latent_path:
        import safetensors.torch

        with safetensors.torch.safe_open(latent_file, framework="pt") as f:
            latents = f.get_tensor("latent")
            meta = f.metadata() or {}
        args.height = int(meta.get("height", args.height or 720))
        args.width = int(meta.get("width", args.width or 1280))
        args.fps = int(meta.get("fps", args.fps or 24))
        args.video_length = int(meta.get("video_length", latents.shape[-3] if latents.ndim >= 4 else 1))
        seed = int(meta.get("seed", 0))
        task = meta.get("task", "t2v")
        args.output_type = "video" if args.output_type in ("latent",) else args.output_type
        for key in ("prompt", "negative_prompt", "guidance_scale", "infer_steps", "flow_shift"):
            if getattr(args, key, None) is None and key in meta:
                setattr(args, key, meta[key])
        args.enable_sound = False
        save_output(args, task, seed, latents, None, None, vae, index=0)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.latent_path:
        if args.ckpt_dir is None:
            raise ValueError("decode-only mode still needs --ckpt_dir (or --vae) for the VAE")
        decode_only(args, device)
        return

    task = detect_task(args)
    setup_args(args, task)
    variant = args.variant or detect_variant(args.ckpt_dir)
    logger.info(f"task={task} variant={variant} size={args.width}x{args.height} frames={args.video_length} "
                f"steps={args.infer_steps} cfg={args.guidance_scale} shift={args.flow_shift} seed={args.seed}")

    if args.enable_sound and variant == "edge":
        raise ValueError("Cosmos3-Edge does not support sound generation")

    start = time.time()
    pipe = build_pipeline(args, device, task)
    logger.info(f"models loaded in {time.time() - start:.1f}s")

    for i in range(args.num_outputs):
        seed = args.seed + i
        latents, sound, action_out, frames = run_generation(args, pipe, task, device, seed)
        save_output(args, task, seed, latents, sound, action_out, pipe.vae, pipe.sound_tokenizer, index=i,
                    frames=frames)
        clean_memory_on_device(device)

    logger.info(f"done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
