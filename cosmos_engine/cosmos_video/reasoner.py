# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Native Cosmos3 prompt upsampler: a causal-LM view over the loaded
# Cosmos3OmniTransformer's understanding (und) pathway, ported from
# NVIDIA/cosmos-framework@058c8c0 unified_mot._impl_generate_reasoner_text and
# omni_mot_model.upsample_captions. The MoT und pathway *is* the reasoner LM —
# same embed_tokens / per-layer und projections / final norm / lm_head weights —
# so no extra checkpoint is needed: prompts are upsampled by the generator
# model itself, exactly as the framework's native_prompt_upsampling flag does.
#
# Image-to-video upsampling additionally runs the checkpoint's bundled
# Qwen3-VL vision tower (vision_encoder/, see vision_encoder.py) and injects
# its deepstack features into the first LM layers, so the caption is grounded
# in the actual conditioning frame.
#
# Implementation notes:
# - Per-layer KV cache: prefill processes the whole prompt once; decode feeds
#   one token at a time attending to the cache (framework decode-loop parity).
# - The attention math mirrors Cosmos3AttnProcessor's causal (und) path
#   verbatim: to_q/to_k/to_v + norm_q/norm_k -> interleaved mrope -> causal
#   SDPA -> to_out, then mlp with input/post_attention layernorms.
# - mrope position ids follow Qwen3-VL get_rope_index (text sequential, image
#   tokens on a t/h/w grid); decode continues at max(position) + 1 per token.
# - Works under H1111 block swap (mirrors the joint forward's offloader
#   calls) and fp8_scaled (the monkey-patched Linears are called normally).

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import torch
from PIL import Image

from .attention import dispatch_attention_fn
from .upsampler_templates import build_messages, clean_response, is_upsampled_prompt  # noqa: F401 (re-export)

logger = logging.getLogger(__name__)

IMAGE_TOKEN = "<|image_pad|>"
VISION_START_TOKEN = "<|vision_start|>"

UPSAMPLE_TASKS = ("t2v", "t2i", "i2v")


def resolve_task(task: Optional[str], *, has_image: bool, video_length: int) -> str:
    """Resolve the UI/CLI task selector: auto -> i2v with an image, else t2i at length 1, else t2v."""
    normalized = (task or "auto").strip().lower()
    if normalized == "auto":
        if has_image:
            return "i2v"
        return "t2i" if int(video_length) == 1 else "t2v"
    if normalized not in UPSAMPLE_TASKS:
        raise ValueError(f"Unsupported upsample task {task!r}. Valid: auto, {', '.join(UPSAMPLE_TASKS)}.")
    return normalized


def strip_generation_weights(transformer) -> None:
    """Drop the generation-pathway modules from a loaded Cosmos3OmniTransformer.

    The upsampler only runs the und pathway (embed_tokens, per-layer und
    projections + und MLP, final norm, lm_head), so the gen-side experts —
    roughly half the parameters — can be freed before moving to the GPU.
    The transformer can no longer run its joint denoising forward afterwards.
    """
    import torch.nn as nn

    for layer in transformer.layers:
        attn = layer.self_attn
        for name in ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out", "norm_added_q", "norm_added_k"):
            setattr(attn, name, nn.Identity())
        layer.mlp_moe_gen = nn.Identity()
        layer.input_layernorm_moe_gen = nn.Identity()
        layer.post_attention_layernorm_moe_gen = nn.Identity()
    for name in (
        "proj_in",
        "proj_out",
        "time_embedder",
        "norm_moe_gen",
        "action_proj_in",
        "action_proj_out",
        "audio_proj_in",
        "audio_proj_out",
    ):
        if hasattr(transformer, name):
            setattr(transformer, name, nn.Identity())


def place_und_layers(transformer, device: torch.device, gpu_layers: int = -1) -> tuple[int, int]:
    """llama.cpp-style layer placement for the und-pathway LM (its ``-ngl``).

    Keeps the LAST ``gpu_layers`` transformer layers resident on ``device`` and
    leaves the earlier ones on the CPU, where their forward runs at decode time
    — weights are never streamed over PCIe per token, only the tiny hidden
    state crosses the split boundary once per pass. ``embed_tokens`` stays with
    the CPU group (a lookup is cheap anywhere); ``norm``/``lm_head`` go to
    ``device`` since they run every token and the lm_head matmul is large.

    ``gpu_layers < 0`` or ``>= num_layers`` means everything on ``device``
    (previous behavior). Returns (n_gpu_layers, n_cpu_layers).
    """
    n = len(transformer.layers)
    if gpu_layers < 0 or gpu_layers >= n:
        transformer.to(device)
        return n, 0
    cpu = torch.device("cpu")
    split = n - int(gpu_layers)
    for i, layer in enumerate(transformer.layers):
        layer.to(cpu if i < split else device)
    transformer.embed_tokens.to(cpu)
    transformer.norm.to(device)
    transformer.lm_head.to(device)
    transformer.rotary_emb.to(device)
    return int(gpu_layers), split


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _KVCache:
    """Per-layer post-RoPE K/V cache: tensors [seq, kv_heads, head_dim]."""

    def __init__(self, num_layers: int):
        self.k: list[Optional[torch.Tensor]] = [None] * num_layers
        self.v: list[Optional[torch.Tensor]] = [None] * num_layers

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k[layer_idx] is None:
            self.k[layer_idx], self.v[layer_idx] = k, v
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k], dim=0)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v], dim=0)
        return self.k[layer_idx], self.v[layer_idx]


class Cosmos3Reasoner:
    """Autoregressive text generation over a loaded Cosmos3OmniTransformer's und pathway."""

    def __init__(
        self,
        transformer,
        tokenizer,
        chat_template: Optional[str] = None,
        vision_encoder=None,
        preprocessor_config: Optional[dict] = None,
        image_token_id: Optional[int] = None,
        prefill_device: Optional[torch.device] = None,
    ):
        self.transformer = transformer
        self.tokenizer = tokenizer
        # With place_und_layers CPU offload: stream CPU-resident layers through
        # this device for the one-time prompt prefill (one PCIe pass beats
        # minutes of CPU matmuls over thousands of prompt tokens). Decode then
        # runs those layers on the CPU. None = compute where the weights live.
        self.prefill_device = torch.device(prefill_device) if prefill_device is not None else None
        if getattr(tokenizer, "chat_template", None) is None and chat_template is not None:
            tokenizer.chat_template = chat_template
        self.vision_encoder = vision_encoder
        self.preprocessor_config = preprocessor_config or {}

        self.image_token_id = (
            image_token_id
            if image_token_id is not None
            else tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        )
        self.vision_start_token_id = tokenizer.convert_tokens_to_ids(VISION_START_TOKEN)
        self.eos_token_ids = {tid for tid in (
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
        ) if tid is not None and tid >= 0}

    # -- prompt building ----------------------------------------------------

    def _render_chat(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _expand_image_tokens(self, text: str, grid_thw: torch.Tensor, merge_size: int) -> str:
        """Replace each single <|image_pad|> with one placeholder per merged patch (processor parity)."""
        for t, h, w in grid_thw.tolist():
            num = (t * h * w) // (merge_size**2)
            text = text.replace(IMAGE_TOKEN, "<|placeholder|>" * num, 1)
        return text.replace("<|placeholder|>", IMAGE_TOKEN)

    def _build_position_ids(self, ids: torch.Tensor, image_grid_thw: Optional[torch.Tensor]) -> torch.Tensor:
        """Qwen3-VL get_rope_index for a single unpadded sequence -> [3, T] long tensor."""
        if image_grid_thw is None or (ids == self.image_token_id).sum() == 0:
            pos = torch.arange(ids.shape[0], dtype=torch.long)
            return pos.view(1, -1).expand(3, -1).contiguous()

        merge_size = int(self.preprocessor_config.get("merge_size", 2))
        input_tokens = ids.tolist()
        pos_chunks: list[torch.Tensor] = []
        st = 0
        for image_index in range(image_grid_thw.shape[0]):
            ed = input_tokens.index(self.image_token_id, st)
            t, h, w = (int(x) for x in image_grid_thw[image_index])
            llm_grid_t, llm_grid_h, llm_grid_w = t, h // merge_size, w // merge_size
            text_len = ed - st
            st_idx = int(pos_chunks[-1].max()) + 1 if pos_chunks else 0
            pos_chunks.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            pos_chunks.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w
        if st < len(input_tokens):
            st_idx = int(pos_chunks[-1].max()) + 1 if pos_chunks else 0
            text_len = len(input_tokens) - st
            pos_chunks.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
        return torch.cat(pos_chunks, dim=1).contiguous()

    # -- vision -------------------------------------------------------------

    def _encode_image(self, image: Image.Image):
        """Image -> (embeds [N_merged, hidden], deepstack list, grid_thw [1,3])."""
        if self.vision_encoder is None:
            raise ValueError(
                "image-to-video prompt upsampling needs the checkpoint's vision_encoder/ component "
                "(not found or not loaded); use task t2v/t2i for text-only upsampling instead"
            )
        from .vision_encoder import preprocess_image

        cfg = self.preprocessor_config
        pixel_values, grid_thw = preprocess_image(
            image,
            patch_size=int(cfg.get("patch_size", 16)),
            temporal_patch_size=int(cfg.get("temporal_patch_size", 2)),
            merge_size=int(cfg.get("merge_size", 2)),
            image_mean=tuple(cfg.get("image_mean", (0.5, 0.5, 0.5))),
            image_std=tuple(cfg.get("image_std", (0.5, 0.5, 0.5))),
            min_pixels=int(cfg.get("min_pixels", 65536)),
            max_pixels=int(cfg.get("max_pixels", 16777216)),
        )
        device = self.vision_encoder.device
        pixel_values = pixel_values.to(device=device, dtype=self.vision_encoder.dtype)
        grid_thw_dev = grid_thw.to(device)
        embeds, deepstack = self.vision_encoder(pixel_values, grid_thw_dev)
        return embeds, deepstack, grid_thw

    # -- und-pathway causal LM ----------------------------------------------

    def _rotary(self, position_ids: torch.Tensor, device, dtype):
        """[3, T] -> (cos, sin) each [T, 1, head_dim], ready to broadcast over heads."""
        cos, sin = self.transformer.rotary_emb(position_ids.unsqueeze(1).to(device), device=device, dtype=dtype)
        return cos.squeeze(0).unsqueeze(1), sin.squeeze(0).unsqueeze(1)

    def _forward_pass(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: _KVCache,
        *,
        is_prefill: bool,
        image_mask: Optional[torch.Tensor] = None,
        deepstack_embeds: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """One pass over all layers (prompt prefill or a single decode token). Returns final hidden [T, H].

        Layers may live on different devices (see place_und_layers): the hidden
        state follows the layer, the KV cache lives on each layer's home
        device, and CPU layers force the portable SDPA attention backend. With
        ``self.prefill_device`` set, CPU-resident layers are moved there for
        the prefill pass only and their K/V is stored back on the CPU.
        """
        transformer = self.transformer
        blocks_to_swap = getattr(transformer, "blocks_to_swap", None)
        rope: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {cos.device: (cos, sin)}

        for layer_idx, layer in enumerate(transformer.layers):
            if blocks_to_swap:
                transformer.offloader.wait_for_block(layer_idx)

            home = layer.input_layernorm.weight.device
            stream = (
                is_prefill
                and self.prefill_device is not None
                and home.type == "cpu"
                and self.prefill_device.type != "cpu"
            )
            run_device = self.prefill_device if stream else home
            if stream:
                layer.to(run_device)
            if hidden.device != run_device:
                hidden = hidden.to(run_device)
            if run_device not in rope:
                rope[run_device] = (cos.to(run_device), sin.to(run_device))
            cos_d, sin_d = rope[run_device]

            attn = layer.self_attn
            h_norm = layer.input_layernorm(hidden)
            q = attn.norm_q(attn.to_q(h_norm).view(-1, attn.num_attention_heads, attn.head_dim))
            k = attn.norm_k(attn.to_k(h_norm).view(-1, attn.num_key_value_heads, attn.head_dim))
            v = attn.to_v(h_norm).view(-1, attn.num_key_value_heads, attn.head_dim)
            q = q * cos_d + _rotate_half(q) * sin_d
            k = k * cos_d + _rotate_half(k) * sin_d

            if stream:
                # Prefill runs on an empty cache, so this layer's K/V is exactly k/v:
                # attend with the on-device tensors, store the cache on the home device.
                cache.append(layer_idx, k.to(home), v.to(home))
                k_all, v_all = k, v
            else:
                k_all, v_all = cache.append(layer_idx, k, v)
            # Prefill self-attends causally; a single decode query attends the whole cache unmasked.
            out = dispatch_attention_fn(
                q.unsqueeze(0),
                k_all.unsqueeze(0),
                v_all.unsqueeze(0),
                is_causal=is_prefill,
                enable_gqa=True,
                backend="torch" if run_device.type == "cpu" else None,
            )
            out = out.squeeze(0).flatten(-2, -1)
            hidden = hidden + attn.to_out(out)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

            # Deepstack: add merged ViT features at image-token rows after the first
            # len(deepstack) layers (Qwen3VLTextModel._deepstack_process parity; prefill only).
            if is_prefill and deepstack_embeds is not None and layer_idx < len(deepstack_embeds):
                ds = deepstack_embeds[layer_idx].to(device=hidden.device, dtype=hidden.dtype)
                mask = image_mask.to(hidden.device)
                hidden = hidden.clone()
                hidden[mask] = hidden[mask] + ds

            if stream:
                layer.to(home)
            if blocks_to_swap:
                transformer.offloader.submit_move_blocks_forward(transformer.layers, layer_idx)

        return hidden

    def _logits(self, hidden_last: torch.Tensor) -> torch.Tensor:
        norm = self.transformer.norm
        normed = norm(hidden_last.to(norm.weight.device))
        return self.transformer.lm_head(normed).float()

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: Optional[float],
        top_k: Optional[int],
        top_p: Optional[float],
        generator: Optional[torch.Generator],
    ) -> int:
        logits = logits.view(-1)
        if not temperature or temperature <= 0:
            return int(torch.argmax(logits).item())
        logits = logits / temperature
        if top_k:
            kth = torch.topk(logits, min(int(top_k), logits.shape[-1])).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p and 0 < top_p < 1:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove = cumulative - probs > top_p  # keep every token needed to reach top_p mass
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator).item())

    @torch.no_grad()
    def generate(
        self,
        messages: list[dict],
        *,
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        progress_callback=None,
    ) -> str:
        """Chat messages (+ optional conditioning image) -> decoded assistant text."""
        transformer = self.transformer
        device = transformer.embed_tokens.weight.device
        dtype = transformer.embed_tokens.weight.dtype

        image_embeds = deepstack = grid_thw = None
        if image is not None:
            image_embeds, deepstack, grid_thw = self._encode_image(image)

        text = self._render_chat(messages)
        if image is not None:
            merge_size = int(self.preprocessor_config.get("merge_size", 2))
            if IMAGE_TOKEN not in text:
                raise ValueError("chat template produced no image placeholder for the multimodal message")
            text = self._expand_image_tokens(text, grid_thw, merge_size)
        ids = torch.tensor(self.tokenizer(text, add_special_tokens=False)["input_ids"], dtype=torch.long)

        position_ids = self._build_position_ids(ids, grid_thw)
        next_pos = int(position_ids.max()) + 1

        ids_dev = ids.to(device)
        hidden = transformer.embed_tokens(ids_dev)
        image_mask = None
        if image is not None:
            image_mask = ids_dev == self.image_token_id
            n_placeholders = int(image_mask.sum())
            if n_placeholders != image_embeds.shape[0]:
                raise ValueError(
                    f"image placeholder count ({n_placeholders}) != vision features ({image_embeds.shape[0]})"
                )
            hidden = hidden.clone()
            hidden[image_mask] = image_embeds.to(device=device, dtype=dtype)

        if getattr(transformer, "blocks_to_swap", None):
            transformer.prepare_block_swap_before_forward()

        cache = _KVCache(len(transformer.layers))
        cos, sin = self._rotary(position_ids, device, dtype)
        hidden = self._forward_pass(
            hidden, cos, sin, cache, is_prefill=True, image_mask=image_mask, deepstack_embeds=deepstack
        )
        logits = self._logits(hidden[-1:])

        generator = None
        if seed is not None:
            generator = torch.Generator(device=logits.device).manual_seed(seed)

        new_tokens: list[int] = []
        for step in range(max_new_tokens):
            token = self._sample(logits, temperature, top_k, top_p, generator)
            if token in self.eos_token_ids:
                break
            new_tokens.append(token)
            if progress_callback is not None and (step + 1) % 25 == 0:
                progress_callback(step + 1)

            token_hidden = transformer.embed_tokens(torch.tensor([token], device=device))
            pos = torch.full((3, 1), next_pos, dtype=torch.long)
            next_pos += 1
            cos, sin = self._rotary(pos, device, dtype)
            token_hidden = self._forward_pass(token_hidden, cos, sin, cache, is_prefill=False)
            logits = self._logits(token_hidden)

        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # -- high-level upsampling ------------------------------------------------

    @torch.no_grad()
    def upsample(
        self,
        prompt: str,
        *,
        task: str,
        image: Optional[Image.Image] = None,
        resolution_w: int,
        resolution_h: int,
        aspect_ratio: str,
        fps: Optional[int] = None,
        duration_secs: Optional[int] = None,
        max_new_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        progress_callback=None,
    ) -> str:
        """Upsample one caption with the canonical V4.2 template; returns the cleaned response.

        The output is the model's fenced ```json object (framework native
        parity: no post-hoc metadata pinning — the template constraints carry
        resolution/fps/duration/aspect and the SFT'd model copies them). Falls
        back to the original prompt if cleaning leaves an empty string.
        """
        if task not in UPSAMPLE_TASKS:
            raise ValueError(f"task must be one of {UPSAMPLE_TASKS}, got {task!r} (resolve 'auto' via resolve_task first)")
        if task == "i2v" and image is None:
            raise ValueError("task='i2v' requires a conditioning image")
        is_video = task in ("t2v", "i2v")
        messages = build_messages(
            task,
            prompt.strip(),
            aspect_ratio=aspect_ratio.replace(":", ","),
            resolution_w=resolution_w,
            resolution_h=resolution_h,
            fps=fps if is_video else None,
            duration_secs=duration_secs if is_video else None,
            with_image=task == "i2v",
        )
        raw = self.generate(
            messages,
            image=image if task == "i2v" else None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            progress_callback=progress_callback,
        )
        cleaned, info = clean_response(raw)
        if not info.get("was_clean", True):
            logger.info(f"upsampler response cleaned: {info}")
        if not cleaned.strip():
            logger.warning("upsampler returned an empty response after cleaning; keeping the original prompt")
            return prompt
        return cleaned


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a bare or ```json-fenced object (for verification/status display; may raise)."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("upsampler JSON must be an object")
    return parsed
