# MiniMax-H3 pipeline for H1111 — the modular blocks of huggingface/diffusers#14355 @ e1b518d
# (before_encoder.py / encoders.py / before_denoise.py / denoise.py / decoders.py) flattened
# into one class with staged sub-calls, so the CLI can sequence component loads:
# the conditioner, the VAEs and the 33B transformer never have to coexist on the GPU.
#
# Task modes: `t2va` (text), `fl2va` (first/last keyframe), `ref2va` (ordered references,
# served by the checkpoint's separate `transformer_ref/` partition).
#
# Checkpoint contracts preserved from the PR (do not "fix"):
#   * keyframe/reference posteriors are *sampled* under an independent seed-42 generator and
#     rounded to float16 before normalization;
#   * the request generator draws condition noise first, then video noise, then audio noise;
#   * condition rows are pinned at max(t, 0.999) (video) / 1.0 (audio) in the row plan;
#   * the video decode runs under float16 autocast over float32 VAE weights;
#   * the video VAE consumes/produces ImageNet-normalized RGB over a [0, 1] base range.

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image, ImageOps

from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils.torch_utils import randn_tensor

from .packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MIN_DURATION,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MiniMaxH3PackedSequence,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    keyframe_condition_noise,
    patchify_video_latents,
    prepare_keyframe_image,
    resolve_canvas_size,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from .packing_ref2va import (
    MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MINIMAX_H3_MAX_REFERENCE_VIDEOS,
    MINIMAX_H3_MAX_REFERENCES,
    MiniMaxH3PreparedReference,
    MiniMaxH3Reference,
    build_ref2va_packed_sequence,
    prepare_reference_frames,
    prepare_reference_image,
    prepare_reference_waveform,
    reference_kind,
    reference_media_to_uint8,
    resample_reference_frames,
    resolve_reference_image_size,
    trim_reference_num_frames,
)

logger = logging.getLogger(__name__)


@dataclass
class MiniMaxH3Plan:
    """The resolved geometry and conditioning of one request, produced by `setup()`."""

    task: str  # t2va | fl2va | ref2va
    height: int
    width: int
    num_frames: int
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int
    keyframes: list = field(default_factory=list)
    keyframe_anchors: tuple = ()
    prepared_references: list = field(default_factory=list)


class MiniMaxH3Pipeline:
    """Flattened MiniMax-H3 pipeline. Components are attached by the caller and may be None
    during stages that do not use them."""

    def __init__(
        self,
        vae=None,
        audio_vae=None,
        scheduler=None,
        audio_scheduler=None,
        transformer=None,
        device: torch.device | str = "cuda",
        patch_size: tuple[int, int, int] = (1, 2, 2),
        vae_latent_channels: int = 24,
        vae_spatial_compression_ratio: int = 16,
        audio_latent_channels: int = 32,
        audio_sampling_rate: int = 32000,
    ):
        self.vae = vae
        self.audio_vae = audio_vae
        self.scheduler = scheduler
        self.audio_scheduler = audio_scheduler
        self.transformer = transformer
        self.device = torch.device(device)
        # The (t, h, w) patch belongs to the transformer, which attaches after the early stages
        # in the staged CLI flow, so it is taken as a constructor argument (default = released value).
        self.patch_size = tuple(patch_size)
        self.vae_latent_channels = vae.config.latent_channels if vae is not None else vae_latent_channels
        self.vae_spatial_compression_ratio = (
            vae.spatial_compression_ratio if vae is not None else vae_spatial_compression_ratio
        )
        self.audio_latent_channels = (
            audio_vae.config.latent_channels if audio_vae is not None else audio_latent_channels
        )
        self.audio_sampling_rate = audio_vae.config.sampling_rate if audio_vae is not None else audio_sampling_rate

    # ------------------------------------------------------------------ setup

    def _latent_geometry(self, height: int, width: int, num_frames: int) -> tuple[int, int, int, int]:
        ratio = self.vae_spatial_compression_ratio
        return video_latent_num_frames(num_frames), height // ratio, width // ratio, audio_latent_num_frames(num_frames)

    @staticmethod
    def _check_canvas(height, width):
        if (height is None) != (width is None):
            raise ValueError("`height` and `width` have to be passed together, or neither of them.")
        if height is not None and (height % MINIMAX_H3_CANVAS_MULTIPLE or width % MINIMAX_H3_CANVAS_MULTIPLE):
            raise ValueError(
                f"`height` and `width` must be multiples of {MINIMAX_H3_CANVAS_MULTIPLE}, got {height}x{width}."
            )

    @staticmethod
    def _check_duration(num_frames):
        aligned = align_num_frames(num_frames)
        duration = aligned / MINIMAX_H3_FPS
        if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
            raise ValueError(
                f"MiniMax-H3 generates between {MINIMAX_H3_MIN_DURATION} and {MINIMAX_H3_MAX_DURATION} seconds at "
                f"{MINIMAX_H3_FPS} fps, so `num_frames`, rounded up to the next `17 * n + 5` the video VAE can "
                f"encode, must be between {int(MINIMAX_H3_MIN_DURATION * MINIMAX_H3_FPS)} and "
                f"{int(MINIMAX_H3_MAX_DURATION * MINIMAX_H3_FPS)}, got {num_frames} (rounded up to {aligned})."
            )
        return aligned

    @torch.no_grad()
    def setup(
        self,
        task: str,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = 124,
        image: Image.Image | None = None,
        last_image: Image.Image | None = None,
        references: list[MiniMaxH3Reference] | None = None,
    ) -> MiniMaxH3Plan:
        """Port of MiniMaxH3SetupStep / MiniMaxH3Ref2VASetupStep."""
        self._check_canvas(height, width)

        if task == "ref2va":
            return self._setup_ref2va(height, width, num_frames, references)

        keyframes = [
            ImageOps.exif_transpose(keyframe).convert("RGB")
            for keyframe in (image, last_image)
            if keyframe is not None
        ]
        keyframe_anchors = tuple(
            anchor for anchor, keyframe in (("first", image), ("last", last_image)) if keyframe is not None
        )
        if height is None:
            height, width = resolve_canvas_size(*(keyframes[0].size if keyframes else (16, 9)))

        num_frames = self._check_duration(num_frames if num_frames else 124)
        num_latent_frames, latent_height, latent_width, num_audio_latents = self._latent_geometry(
            height, width, num_frames
        )
        keyframes = [
            prepare_keyframe_image(keyframe, height, width, stretch=index == 0)
            for index, keyframe in enumerate(keyframes)
        ]
        return MiniMaxH3Plan(
            task="fl2va" if keyframes else "t2va",
            height=height,
            width=width,
            num_frames=num_frames,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            keyframes=keyframes,
            keyframe_anchors=keyframe_anchors,
        )

    def _setup_ref2va(self, height, width, num_frames, references) -> MiniMaxH3Plan:
        if not references:
            raise ValueError("`ref2va` needs at least one reference.")
        kinds = [reference_kind(index, entry) for index, entry in enumerate(references)]
        for kind, limit in (
            ("image", MINIMAX_H3_MAX_REFERENCE_IMAGES),
            ("video", MINIMAX_H3_MAX_REFERENCE_VIDEOS),
            ("audio", MINIMAX_H3_MAX_REFERENCE_AUDIOS),
        ):
            if kinds.count(kind) > limit:
                raise ValueError(f"MiniMax-H3 accepts at most {limit} {kind} references, got {kinds.count(kind)}.")
        if len(kinds) > MINIMAX_H3_MAX_REFERENCES:
            raise ValueError(f"MiniMax-H3 accepts at most {MINIMAX_H3_MAX_REFERENCES} references, got {len(kinds)}.")
        if set(kinds) == {"audio"}:
            raise ValueError(
                "An audio reference has to be paired with at least one image or video reference and cannot be used "
                "on its own."
            )
        if num_frames is not None:
            self._check_duration(num_frames)

        if height is None:
            height, width = resolve_canvas_size(16, 9)

        prepared, num_frames = self._prepare_references(references, num_frames)
        num_latent_frames, latent_height, latent_width, num_audio_latents = self._latent_geometry(
            height, width, num_frames
        )
        return MiniMaxH3Plan(
            task="ref2va",
            height=height,
            width=width,
            num_frames=num_frames,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            prepared_references=prepared,
        )

    def _prepare_references(
        self, references: list[MiniMaxH3Reference], num_frames: int | None
    ) -> tuple[list[MiniMaxH3PreparedReference], int]:
        """Port of MiniMaxH3Ref2VASetupStep.prepare_references."""
        resolved = [
            MiniMaxH3PreparedReference(kind=reference_kind(index, entry), has_audio=entry.has_audio)
            for index, entry in enumerate(references)
        ]

        if num_frames is None:
            audio_bearing = [index for index, reference in enumerate(resolved) if reference.has_audio]
            if len(audio_bearing) != 1:
                raise ValueError(
                    "`num_frames` may only be left to the references when exactly one of them carries audio, got "
                    f"{len(audio_bearing)}."
                )
            index = audio_bearing[0]
            sample_rate = references[index].sample_rate or self.audio_sampling_rate
            duration = references[index].audio.shape[-1] / sample_rate
            if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
                raise ValueError(
                    f"`references[{index}]` is {duration:g} seconds long, outside the "
                    f"{MINIMAX_H3_MIN_DURATION} to {MINIMAX_H3_MAX_DURATION} seconds MiniMax-H3 generates."
                )
            num_frames = align_num_frames(round(duration * MINIMAX_H3_FPS))
            if num_frames / MINIMAX_H3_FPS > MINIMAX_H3_MAX_DURATION:
                raise ValueError(
                    f"`references[{index}]` is {duration:g} seconds long, which rounds up to {num_frames} frames "
                    f"(`17 * n + 5`), i.e. {num_frames / MINIMAX_H3_FPS:g} seconds — past the "
                    f"{MINIMAX_H3_MAX_DURATION} seconds MiniMax-H3 generates. Pass `num_frames` to generate a "
                    "shorter video from this soundtrack."
                )
        num_frames = align_num_frames(num_frames)

        for reference, entry in zip(resolved, references):
            if reference.kind == "image":
                image = entry.image
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(reference_media_to_uint8(image))
                image = ImageOps.exif_transpose(image).convert("RGB")
                ref_height, ref_width = resolve_reference_image_size(*image.size)
                reference.image = prepare_reference_image(image, ref_height, ref_width)
            elif reference.kind == "video":
                frames = resample_reference_frames(reference_media_to_uint8(entry.video), float(entry.fps))
                reference.frames = prepare_reference_frames(frames, num_frames)
            if reference.has_audio:
                reference.waveform = prepare_reference_waveform(
                    entry.audio,
                    entry.sample_rate or self.audio_sampling_rate,
                    self.audio_sampling_rate,
                    max_duration=num_frames / MINIMAX_H3_FPS,
                )
        return resolved, num_frames

    # ------------------------------------------------------------------ condition encoding

    def _pixel_stats(self, device):
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        return pixel_mean, pixel_std, latents_mean, latents_std

    def _sample_condition_latents(self, moments: torch.Tensor) -> torch.Tensor:
        posterior = DiagonalGaussianDistribution(moments)
        latents = posterior.sample(generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
        # The sampled latent is rounded to float16 before it is normalized: ~11 bits of every
        # conditioning latent, so the released model's conditioning cannot be reproduced without it.
        return latents.to(torch.float16).float().cpu()

    @torch.no_grad()
    def encode_conditions(
        self, plan: MiniMaxH3Plan, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Port of MiniMaxH3KeyframeVaeEncoderStep / MiniMaxH3Ref2VAReferenceEncoderStep.

        Draws the condition noise from the request generator — the FIRST draws of the request.
        Returns `(condition_latents, audio_condition_latents)`, either possibly None.
        """
        device = self.device
        if plan.task in ("t2va", "fl2va"):
            if not plan.keyframes:
                return None, None
            pixel_mean, pixel_std, latents_mean, latents_std = self._pixel_stats(device)
            rows = []
            for image in plan.keyframes:
                pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None, :, None]
                pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
                moments = self.vae._encode_clip(pixels)
                latents = self._sample_condition_latents(moments)
                rows.append(patchify_video_latents((latents - latents_mean) / latents_std, self.patch_size))
            condition_latents = torch.cat(rows)
            noise = keyframe_condition_noise(
                ((1, plan.latent_height, plan.latent_width),) * len(plan.keyframes),
                self.patch_size,
                self.vae_latent_channels,
                generator=generator,
                device=device,
            )
            condition_latents = self.scheduler.scale_noise(
                condition_latents.to(device), MINIMAX_H3_KEYFRAME_NOISE_AUG, noise
            )
            return condition_latents, None

        # ref2va
        pixel_mean, pixel_std, latents_mean, latents_std = self._pixel_stats(device)
        audio_latents_mean = torch.tensor(self.audio_vae.config.latents_mean).view(1, 1, -1)
        audio_latents_std = torch.tensor(self.audio_vae.config.latents_std).view(1, 1, -1)

        video_rows, audio_rows = [], []
        for reference in plan.prepared_references:
            if reference.kind != "audio":
                if reference.kind == "image":
                    pixels = torch.from_numpy(np.array(reference.image)).to(device).permute(2, 0, 1)[None, :, None]
                else:
                    frames = reference.frames[: trim_reference_num_frames(reference.frames.shape[0])]
                    pixels = torch.from_numpy(frames.copy()).to(device).permute(3, 0, 1, 2)[None]
                pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
                moments = self.vae._encode_clip(pixels) if reference.kind == "image" else self.vae._encode(pixels)
                latents = self._sample_condition_latents(moments)
                reference.num_latent_frames = latents.shape[2]
                reference.latent_height, reference.latent_width = latents.shape[3], latents.shape[4]
                video_rows.append(patchify_video_latents((latents - latents_mean) / latents_std, self.patch_size))

            if reference.has_audio:
                posterior = self.audio_vae.encode(reference.waveform.to(device)[:, None], return_dict=False)[0]
                # Channel-major rows: the two stereo channels are two batch items of the mono audio VAE.
                latents = posterior.mode().float().cpu().transpose(1, 2)
                reference.num_audio_latents = latents.shape[1]
                normalized = (latents - audio_latents_mean) / audio_latents_std
                audio_rows.append(normalized.reshape(-1, self.audio_latent_channels))

        condition_latents = torch.cat(video_rows) if video_rows else None
        audio_condition_latents = torch.cat(audio_rows) if audio_rows else None

        if condition_latents is not None:
            noise = keyframe_condition_noise(
                tuple(
                    (reference.num_latent_frames, reference.latent_height, reference.latent_width)
                    for reference in plan.prepared_references
                    if reference.kind != "audio"
                ),
                self.patch_size,
                self.vae_latent_channels,
                generator=generator,
                device=device,
            )
            condition_latents = self.scheduler.scale_noise(
                condition_latents.to(device), MINIMAX_H3_KEYFRAME_NOISE_AUG, noise
            )
        if audio_condition_latents is not None:
            audio_condition_latents = audio_condition_latents.to(device)
        return condition_latents, audio_condition_latents

    # ------------------------------------------------------------------ layout / latents / timesteps

    def build_layout(self, plan: MiniMaxH3Plan, text_token_tags: torch.Tensor) -> MiniMaxH3PackedSequence:
        if plan.task == "ref2va":
            return build_ref2va_packed_sequence(
                text_token_tags,
                plan.prepared_references,
                plan.num_latent_frames,
                plan.latent_height,
                plan.latent_width,
                plan.num_audio_latents,
                self.patch_size,
            )
        return build_packed_sequence(
            text_token_tags,
            plan.num_latent_frames,
            plan.latent_height,
            plan.latent_width,
            plan.num_audio_latents,
            self.patch_size,
            plan.keyframe_anchors,
        )

    @torch.no_grad()
    def prepare_latents(
        self,
        plan: MiniMaxH3Plan,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        condition_latents: torch.Tensor | None = None,
        audio_condition_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Port of MiniMaxH3PrepareLatentsStep: video noise drawn first, then audio noise —
        strictly after the condition draws of `encode_conditions`."""
        device = self.device
        if latents is None:
            latents = randn_tensor(
                (1, self.vae_latent_channels, plan.num_latent_frames, plan.latent_height, plan.latent_width),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        video_rows = patchify_video_latents(latents.to(torch.float32), self.patch_size)

        if audio_latents is None:
            audio_rows = randn_tensor(
                (plan.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS, self.audio_latent_channels),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            audio_rows = audio_latents.to(torch.float32).permute(0, 2, 1).reshape(-1, self.audio_latent_channels)
        video_rows, audio_rows = video_rows.to(device), audio_rows.to(device)

        if condition_latents is not None:
            video_rows = torch.cat([condition_latents.to(device), video_rows])
        if audio_condition_latents is not None:
            audio_rows = torch.cat([audio_condition_latents.to(device), audio_rows])
        return video_rows, audio_rows

    @torch.no_grad()
    def set_timesteps(self, num_inference_steps: int, layout: MiniMaxH3PackedSequence):
        """Port of MiniMaxH3SetTimestepsStep. Returns (timesteps, audio_timesteps, row_timestep_plan)."""
        device = self.device
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.audio_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        audio_timesteps = self.audio_scheduler.timesteps

        row_timestep_plan = [
            tuple(
                tensor.to(device)
                for tensor in build_row_timesteps(
                    layout,
                    float(timestep),
                    float(audio_timestep),
                    max(float(timestep), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                    1.0,
                )
            )
            for timestep, audio_timestep in zip(timesteps, audio_timesteps)
        ]
        return timesteps, audio_timesteps, row_timestep_plan

    # ------------------------------------------------------------------ denoise

    @torch.no_grad()
    def denoise(
        self,
        layout: MiniMaxH3PackedSequence,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
        row_timestep_plan: list,
        step_callback=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Port of MiniMaxH3DenoiseStep: one forward per step, two scheduler steps writing only
        the generated rows. `step_callback(step_index, total_steps, latents, audio_latents,
        noise_pred)` runs after each step; it may raise to abort (partial rows stay valid)."""
        device = self.device
        position_ids = layout.position_ids.to(device)
        token_tags = layout.token_tags.to(device)
        video_indices = layout.video_indices.to(device)
        audio_indices = layout.audio_indices.to(device)
        text_indices = layout.text_indices.to(device)
        num_condition_video_rows = layout.num_condition_video_rows
        num_condition_audio_rows = layout.num_condition_audio_rows
        prompt_embeds = prompt_embeds.to(device)

        total = len(timesteps)
        for i, t in enumerate(timesteps):
            unique_timesteps, timestep_indices = row_timestep_plan[i]
            noise_pred, audio_noise_pred = self.transformer(
                hidden_states=latents[None],
                audio_hidden_states=audio_latents[None],
                encoder_hidden_states=prompt_embeds,
                timestep=unique_timesteps,
                timestep_indices=timestep_indices,
                token_tags=token_tags,
                position_ids=position_ids,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
                return_dict=False,
            )
            latents[num_condition_video_rows:] = self.scheduler.step(
                noise_pred[0, num_condition_video_rows:].float(),
                t,
                latents[num_condition_video_rows:],
                return_dict=False,
            )[0]
            audio_latents[num_condition_audio_rows:] = self.audio_scheduler.step(
                audio_noise_pred[0, num_condition_audio_rows:].float(),
                audio_timesteps[i],
                audio_latents[num_condition_audio_rows:],
                return_dict=False,
            )[0]
            if step_callback is not None:
                step_callback(i, total, latents, audio_latents, noise_pred)
        return latents, audio_latents

    # ------------------------------------------------------------------ decode

    @torch.no_grad()
    def unpack_video_latents(self, latents: torch.Tensor, plan: MiniMaxH3Plan, layout=None) -> torch.Tensor:
        """Generated video rows -> denormalized latent tensor `(1, C, T, H, W)`."""
        num_condition_rows = layout.num_condition_video_rows if layout is not None else 0
        tensor = unpatchify_video_tokens(
            latents[num_condition_rows:],
            plan.num_latent_frames,
            plan.latent_height,
            plan.latent_width,
            self.vae_latent_channels,
            self.patch_size,
        )
        device = tensor.device
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        return tensor * latents_std + latents_mean

    @torch.no_grad()
    def unpack_audio_latents(self, audio_latents: torch.Tensor, plan: MiniMaxH3Plan, layout=None) -> torch.Tensor:
        """Generated audio rows -> denormalized latent tensor `(2, C, N)`."""
        num_condition_rows = layout.num_condition_audio_rows if layout is not None else 0
        tensor = unpack_audio_tokens(audio_latents[num_condition_rows:], plan.num_audio_latents)
        device = tensor.device
        audio_latents_mean = torch.tensor(self.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
        audio_latents_std = torch.tensor(self.audio_vae.config.latents_std, device=device).view(1, -1, 1)
        return tensor * audio_latents_std + audio_latents_mean

    @torch.no_grad()
    def decode_video(self, latents: torch.Tensor) -> np.ndarray:
        """Denormalized latents `(1, C, T, H, W)` -> uint8 frames `[T, H, W, C]`.

        The decode runs under float16 autocast over the float32 VAE weights — the PR's verified
        recipe — and the VAE emits ImageNet-normalized RGB that is reverted here.
        """
        device = latents.device
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            video = self.vae.decode(latents, return_dict=False)[0]
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
        video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
        frames = (video[0] * 255.0).round().to(torch.uint8)  # [C, T, H, W]
        return frames.permute(1, 2, 3, 0).cpu().numpy()

    @torch.no_grad()
    def decode_audio(self, audio_latents: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Denormalized latents `(2, C, N)` -> (waveform `(1, 2, num_samples)`, sample rate)."""
        audio = self.audio_vae.decode(audio_latents, return_dict=False)[0]
        return audio.float().permute(1, 0, 2), self.audio_sampling_rate
