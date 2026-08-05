# MiniMax-H3 conditioner: a Qwen3-VL read at its 50th decoder layer, with the language-model
# head unused. Ported from huggingface/diffusers#14355 @ e1b518d (modular_pipelines/minimax_h3/
# encoders.py — MiniMaxH3TextEncoderStep.encode_prompt and MiniMaxH3Ref2VATextEncoderStep.
# encode_prompt) onto the vendored Qwen3-VL modules in qwen3vl_vision / qwen3vl_text /
# qwen3vl_processor, because the environment pins transformers 4.46.x (no Qwen3-VL).
#
# The conditioner builds MiniMax-H3's presentation of a request — no chat template, no special
# tokens — and returns the *unnormalized* hidden state after decoder layer 50 plus the per-row
# modality tags (vision blocks are tagged as video; that is what the transformer's AdaLN keys
# off).

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

from .packing import MINIMAX_H3_TEXT_ENCODER_LAYER, MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG
from .packing_ref2va import MiniMaxH3PreparedReference, build_ref2va_presentation, sample_reference_video_frames
from .qwen3vl_processor import (
    IMAGE_PAD_TOKEN,
    VIDEO_PAD_TOKEN,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
    create_mm_token_type_ids,
    get_rope_index,
    load_image_preprocessor_config,
    load_video_preprocessor_config,
    preprocess_video,
)
from .qwen3vl_text import Qwen3VLTruncatedTextModel
from .qwen3vl_vision import build_vision_tower, preprocess_image


def _strip_known_prefixes(state_dict: dict) -> tuple[dict, dict]:
    """Split a Qwen3VLForConditionalGeneration state dict into (text keys, vision keys).

    transformers versions differ in how they nest the two submodules
    (`model.language_model.` / `model.visual.` vs `language_model.` / `visual.` vs `model.`
    flat), so every known spelling is tried; `lm_head.` is dropped outright.
    """
    text, vision = {}, {}
    for key, value in state_dict.items():
        for prefix in ("model.language_model.", "language_model.", "model.text_model."):
            if key.startswith(prefix):
                text[key[len(prefix) :]] = value
                break
        else:
            for prefix in ("model.visual.", "visual."):
                if key.startswith(prefix):
                    vision[key[len(prefix) :]] = value
                    break
            else:
                if key.startswith("lm_head."):
                    continue
                if key.startswith("model."):
                    text[key[len("model.") :]] = value
    return text, vision


def _wanted_text_key(key: str, num_read_layers: int) -> bool:
    """Keep embed_tokens and layers[0..num_read_layers-1]; drop deeper layers and the final norm."""
    if key.startswith("embed_tokens."):
        return True
    if key.startswith("layers."):
        layer_idx = int(key.split(".")[1])
        return layer_idx < num_read_layers
    return False


class MiniMaxH3Conditioner:
    """Loads and runs the truncated Qwen3-VL conditioner from `<ckpt_dir>/text_encoder`."""

    def __init__(
        self,
        ckpt_dir: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        gpu_layers: int = -1,
        stream_device: torch.device | str | None = None,
        text_encoder_path: str | None = None,
        int8_use_int_mm: bool = False,
    ):
        """
        Args:
            ckpt_dir: HF snapshot dir holding `text_encoder/`, `tokenizer/`, `processor/`.
            device: home device for the small parts (embeddings, vision tower).
            dtype: parameter dtype (the released conditioner is bfloat16).
            gpu_layers: how many text decoder layers to keep resident on `device`
                (-1 = all, 0 = none). The rest stay on CPU.
            stream_device: when set, CPU-resident layers are streamed there one at a
                time during the forward.
            text_encoder_path: optional single-file weight override (e.g. an int8
                convrot export). Text-model weights (and the vision tower, when the
                file carries `visual.*` keys — the "ultra_p" export does) come from this
                file; config/tokenizer/processor still come from the snapshot. int8
                tensors keep their `weight_scale` and get the int8 monkey patch.
            int8_use_int_mm: run quantized Linears through torch._int_mm instead of
                dequantize-per-forward.
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.stream_device = torch.device(stream_device) if stream_device is not None else None

        encoder_dir = os.path.join(ckpt_dir, "text_encoder")
        config = json.load(open(os.path.join(encoder_dir, "config.json")))
        text_config = config.get("text_config", config)
        vision_config = config["vision_config"]

        num_layers = text_config["num_hidden_layers"]
        if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER:
            raise ValueError(
                f"MiniMax-H3 conditions on hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}] of its Qwen3-VL "
                f"conditioner, which needs more than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers, but the "
                f"config declares {num_layers}. The last hidden state of a stack truncated to exactly "
                f"{MINIMAX_H3_TEXT_ENCODER_LAYER} layers is post-norm and is not the conditioning MiniMax-H3 expects."
            )

        self.image_token_id = config.get("image_token_id", 151655)
        self.video_token_id = config.get("video_token_id", 151656)
        self.spatial_merge_size = vision_config["spatial_merge_size"]
        self.text_config = text_config
        self.vision_config = vision_config

        from transformers import AutoTokenizer

        tokenizer_dir = os.path.join(ckpt_dir, "tokenizer")
        if not os.path.isdir(tokenizer_dir):
            tokenizer_dir = ckpt_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        processor_dir = os.path.join(ckpt_dir, "processor")
        self.image_config = load_image_preprocessor_config(processor_dir)
        self.video_config = load_video_preprocessor_config(processor_dir)

        with torch.device("meta"):
            self.text_model = Qwen3VLTruncatedTextModel(text_config, MINIMAX_H3_TEXT_ENCODER_LAYER)
            self.vision_tower = build_vision_tower(vision_config)

        self._load_weights(
            encoder_dir, gpu_layers, text_encoder_path=text_encoder_path, int8_use_int_mm=int8_use_int_mm
        )

    # ------------------------------------------------------------------ loading

    def _shard_files(self, encoder_dir: str) -> list[str]:
        index_path = os.path.join(encoder_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            index = json.load(open(index_path))
            names = sorted(set(index["weight_map"].values()))
            return [os.path.join(encoder_dir, name) for name in names]
        shards = sorted(glob.glob(os.path.join(encoder_dir, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"no safetensors weights found under {encoder_dir}")
        return shards

    def _load_weights(
        self,
        encoder_dir: str,
        gpu_layers: int,
        text_encoder_path: str | None = None,
        int8_use_int_mm: bool = False,
    ) -> None:
        import safetensors.torch

        from .int8_quant import QUANT_MARKER_SUFFIX, WEIGHT_SCALE_SUFFIX, apply_int8_monkey_patch, collect_quant_markers

        def keep_dtype(value: torch.Tensor, key: str) -> torch.Tensor:
            # int8 weights and their float32 scales must survive the cast untouched
            if value.dtype == torch.int8 or key.endswith(WEIGHT_SCALE_SUFFIX) or key.endswith(QUANT_MARKER_SUFFIX):
                return value
            return value.to(self.dtype)

        num_read = MINIMAX_H3_TEXT_ENCODER_LAYER
        text_sd, vision_sd = {}, {}
        if text_encoder_path:
            # single-file override (int8 convrot export): text weights always, vision
            # tower too when the file ships it; otherwise the snapshot still provides vision
            from utils.safetensors_utils import MemoryEfficientSafeOpen

            with MemoryEfficientSafeOpen(text_encoder_path) as f:
                for key in f.keys():
                    value = f.get_tensor(key)
                    text_part, vision_part = _strip_known_prefixes({key: value})
                    for k, v in text_part.items():
                        # quant markers and weight_scales share their layer's prefix, so the
                        # same filter keeps exactly the wanted layers' quant tensors
                        if _wanted_text_key(k, num_read):
                            text_sd[k] = keep_dtype(v, k)
                    for k, v in vision_part.items():
                        vision_sd[k] = keep_dtype(v, k)
            if not vision_sd:
                for shard in self._shard_files(encoder_dir):
                    raw = safetensors.torch.load_file(shard, device="cpu")
                    _, vision_part = _strip_known_prefixes(raw)
                    for key, value in vision_part.items():
                        vision_sd[key] = value.to(self.dtype)
                    del raw
        else:
            for shard in self._shard_files(encoder_dir):
                raw = safetensors.torch.load_file(shard, device="cpu")
                text_part, vision_part = _strip_known_prefixes(raw)
                for key, value in text_part.items():
                    if _wanted_text_key(key, num_read):
                        text_sd[key] = value.to(self.dtype)
                for key, value in vision_part.items():
                    vision_sd[key] = value.to(self.dtype)
                del raw

        # int8 layers announce themselves via their per-layer quant markers: register the
        # scale buffers and bind the int8 forwards before the state dict lands
        text_markers = collect_quant_markers(text_sd)
        if text_markers:
            self.text_model.requires_grad_(False)  # int8 tensors cannot carry grads under assign=True
            apply_int8_monkey_patch(
                self.text_model,
                text_markers,
                use_int_mm=int8_use_int_mm,
                embedding_output_dtype=self.dtype,
                state_dict=text_sd,
            )
        vision_markers = collect_quant_markers(vision_sd)
        if vision_markers:
            self.vision_tower.requires_grad_(False)
            apply_int8_monkey_patch(
                self.vision_tower,
                vision_markers,
                use_int_mm=int8_use_int_mm,
                embedding_output_dtype=self.dtype,
                state_dict=vision_sd,
            )

        missing, unexpected = self.text_model.load_state_dict(text_sd, strict=False, assign=True)
        # `rotary_emb.inv_freq` is computed, and everything past layer 50 was skipped on purpose.
        missing = [k for k in missing if not k.startswith("rotary_emb.")]
        if missing:
            raise RuntimeError(f"text_encoder missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
        unexpected = [k for k in unexpected]
        if unexpected:
            raise RuntimeError(f"text_encoder unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")

        missing, unexpected = self.vision_tower.load_state_dict(vision_sd, strict=False, assign=True)
        missing = [k for k in missing if "rotary_pos_emb" not in k]
        if missing:
            raise RuntimeError(f"vision tower missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")

        # Non-persistent buffers (the two rope inv_freq tables) are not in the checkpoint, so
        # `assign=True` over a meta-initialized module leaves them on the meta device; rebuild
        # the rope modules for real from the configs.
        from .qwen3vl_text import Qwen3VLTextRotaryEmbedding
        from .qwen3vl_vision import Qwen3VLVisionRotaryEmbedding

        text_config = self.text_config
        head_dim = text_config.get("head_dim", text_config["hidden_size"] // text_config["num_attention_heads"])
        rope_scaling = text_config.get("rope_scaling") or {}
        self.text_model.rotary_emb = Qwen3VLTextRotaryEmbedding(
            head_dim=head_dim,
            rope_theta=text_config.get("rope_theta", 1000000.0),
            mrope_section=rope_scaling.get("mrope_section", [24, 20, 20]),
        )
        vision_head_dim = self.vision_config["hidden_size"] // self.vision_config["num_heads"]
        self.vision_tower.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(vision_head_dim // 2)

        for name, tensor in list(self.text_model.named_parameters()) + list(self.text_model.named_buffers()):
            if tensor.is_meta:
                raise RuntimeError(f"text_encoder parameter left on meta after load: {name}")
        for name, tensor in list(self.vision_tower.named_parameters()) + list(self.vision_tower.named_buffers()):
            if tensor.is_meta:
                raise RuntimeError(f"vision tower parameter left on meta after load: {name}")

        self.text_model.eval()
        self.vision_tower.eval()

        self.vision_tower.to(self.device)
        self.text_model.embed_tokens.to(self.device)
        if gpu_layers < 0:
            self.text_model.layers.to(self.device)
        elif gpu_layers > 0:
            for layer in list(self.text_model.layers)[:gpu_layers]:
                layer.to(self.device)

    # ------------------------------------------------------------------ helpers

    def _vision_ids(self, pad_token: str, num_tokens: int) -> list[int]:
        return (
            [self.tokenizer.convert_tokens_to_ids(VISION_START_TOKEN)]
            + [self.tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
            + [self.tokenizer.convert_tokens_to_ids(VISION_END_TOKEN)]
        )

    def _preprocess_images(self, images: list) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        pixel_chunks, grids, token_counts = [], [], []
        merge = self.image_config["merge_size"] ** 2
        for image in images:
            pixel_values, grid_thw = preprocess_image(
                image,
                patch_size=self.image_config["patch_size"],
                temporal_patch_size=self.image_config["temporal_patch_size"],
                merge_size=self.image_config["merge_size"],
                image_mean=self.image_config["image_mean"],
                image_std=self.image_config["image_std"],
                min_pixels=self.image_config["min_pixels"],
                max_pixels=self.image_config["max_pixels"],
            )
            pixel_chunks.append(pixel_values)
            grids.append(grid_thw)
            token_counts.append(int(grid_thw.prod()) // merge)
        return torch.cat(pixel_chunks), torch.cat(grids), token_counts

    @torch.no_grad()
    def _encode(
        self,
        token_ids: list[int],
        token_tags: list[int],
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        out_dtype: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        inputs_embeds = self.text_model.embed_tokens(input_ids)

        image_mask = input_ids == self.image_token_id
        video_mask = input_ids == self.video_token_id

        deepstack_image, deepstack_video = None, None
        if pixel_values is not None:
            image_embeds, deepstack_image = self.vision_tower(
                pixel_values.to(device=device, dtype=self.vision_tower.dtype), image_grid_thw.to(device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds.to(inputs_embeds.dtype)
            )
        if pixel_values_videos is not None:
            video_embeds, deepstack_video = self.vision_tower(
                pixel_values_videos.to(device=device, dtype=self.vision_tower.dtype), video_grid_thw.to(device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask.unsqueeze(-1).expand_as(inputs_embeds), video_embeds.to(inputs_embeds.dtype)
            )

        # Aggregate the deepstack features across modalities (Qwen3VLModel.forward parity).
        visual_pos_masks, deepstack_visual_embeds = None, None
        if deepstack_image is not None and deepstack_video is not None:
            visual_pos_masks = image_mask | video_mask
            image_joint = image_mask[visual_pos_masks]
            video_joint = video_mask[visual_pos_masks]
            deepstack_visual_embeds = []
            for img_embed, vid_embed in zip(deepstack_image, deepstack_video):
                joint = img_embed.new_zeros(int(visual_pos_masks.sum()), img_embed.shape[-1])
                joint[image_joint, :] = img_embed
                joint[video_joint, :] = vid_embed
                deepstack_visual_embeds.append(joint)
        elif deepstack_image is not None:
            visual_pos_masks, deepstack_visual_embeds = image_mask, deepstack_image
        elif deepstack_video is not None:
            visual_pos_masks, deepstack_visual_embeds = video_mask, deepstack_video

        mm_types = create_mm_token_type_ids(token_ids, self.image_token_id, self.video_token_id)
        position_ids = get_rope_index(token_ids, mm_types, image_grid_thw, video_grid_thw, self.spatial_merge_size)

        hidden = self.text_model(
            inputs_embeds,
            position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            stream_device=self.stream_device,
        )
        if out_dtype is not None:
            hidden = hidden.to(out_dtype)
        return hidden, torch.tensor(token_tags, dtype=torch.long)

    # ------------------------------------------------------------------ public API

    @torch.no_grad()
    def encode_prompt(
        self, prompt: str, keyframes: list | None = None, out_dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`t2va` / `fl2va` presentation: `"<Picture i>: "` + vision block per keyframe, then the prompt verbatim."""
        if not isinstance(prompt, str):
            raise ValueError(
                f"MiniMax-H3 packs one request into one sequence, so `prompt` must be a single string, "
                f"got {type(prompt)}."
            )
        token_ids, token_tags = [], []
        pixel_values = image_grid_thw = None
        if keyframes:
            pixel_values, image_grid_thw, token_counts = self._preprocess_images(keyframes)
            for index, num_tokens in enumerate(token_counts):
                label_ids = self.tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                vision_ids = self._vision_ids(IMAGE_PAD_TOKEN, num_tokens)
                token_ids += label_ids + vision_ids
                token_tags += [MINIMAX_H3_TEXT_TAG] * len(label_ids) + [MINIMAX_H3_VIDEO_TAG] * len(vision_ids)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)

        return self._encode(token_ids, token_tags, pixel_values, image_grid_thw, None, None, out_dtype)

    @torch.no_grad()
    def encode_prompt_ref2va(
        self,
        prompt: str,
        references: list[MiniMaxH3PreparedReference],
        out_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`ref2va` presentation: per-reference labels + timestamped vision blocks, then the prompt verbatim."""
        if not isinstance(prompt, str):
            raise ValueError(
                f"MiniMax-H3 packs one request into one sequence, so `prompt` must be a single string, "
                f"got {type(prompt)}."
            )
        merge = self.image_config["merge_size"] ** 2

        pixel_values = image_grid_thw = None
        image_token_counts: list[int] = []
        images = [reference.image for reference in references if reference.kind == "image"]
        if images:
            pixel_values, image_grid_thw, image_token_counts = self._preprocess_images(images)

        pixel_values_videos = video_grid_thw = None
        video_block_token_counts: list[int] = []
        videos = [reference for reference in references if reference.kind == "video"]
        if videos:
            pixel_chunks, grids = [], []
            for reference in videos:
                frames, block_timestamps = sample_reference_video_frames(reference.frames)
                reference.block_timestamps = block_timestamps
                chunk, grid = preprocess_video(np.stack(frames), self.video_config)
                if int(grid[0, 0]) != len(block_timestamps):
                    raise ValueError(
                        f"The processor merged a reference video into {int(grid[0, 0])} vision blocks, but "
                        f"MiniMax-H3 labels {len(block_timestamps)} of them."
                    )
                pixel_chunks.append(chunk)
                grids.append(grid)
            pixel_values_videos = torch.cat(pixel_chunks)
            video_grid_thw = torch.cat(grids)
            video_block_token_counts = [int(grid[1]) * int(grid[2]) // merge for grid in video_grid_thw]

        token_ids, token_tags = build_ref2va_presentation(
            self.tokenizer, prompt, references, image_token_counts, video_block_token_counts
        )
        return self._encode(
            token_ids, token_tags, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, out_dtype
        )
