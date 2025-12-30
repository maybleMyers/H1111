# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Adapted from HuMo repository for wan2_generate_video.py integration
#
# HuMo Audio Processing Utilities
# Provides audio feature extraction using Whisper and audio windowing for HuMo model

import os
import subprocess
import logging
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "HuMoAudioProcessor",
    "linear_interpolation_fps",
    "get_audio_emb_window",
    "audio_emb_enc",
]


def linear_interpolation_fps(features: torch.Tensor, input_fps: float, output_fps: float, output_len: Optional[int] = None) -> torch.Tensor:
    """
    Interpolate features from input_fps to output_fps.

    Args:
        features: Input features [batch, time, channels]
        input_fps: Input frames per second
        output_fps: Target frames per second
        output_len: Optional target length

    Returns:
        Interpolated features [batch, time', channels]
    """
    features = features.transpose(1, 2)  # [batch, channels, time]
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


def resample_audio(input_audio_file: str, output_audio_file: str, sample_rate: int) -> str:
    """
    Resample audio file to target sample rate using ffmpeg.

    Args:
        input_audio_file: Input audio path
        output_audio_file: Output audio path
        sample_rate: Target sample rate

    Returns:
        Output audio file path
    """
    p = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-i", input_audio_file, "-ar", str(sample_rate), output_audio_file
    ])
    ret = p.wait()
    assert ret == 0, "Resample audio failed!"
    return output_audio_file


def get_audio_emb_window(audio_emb: torch.Tensor, frame_num: int, frame0_idx: int = 0, audio_shift: int = 2) -> Tuple[torch.Tensor, int]:
    """
    Window audio embeddings for HuMo model input.

    Creates overlapping windows of audio features aligned with video latent frames.
    First window: 3 zero frames + 5 audio frames (8 total)
    Subsequent windows: 8 audio frames with audio_shift overlap

    Args:
        audio_emb: Audio embeddings [frames, blocks, channels] e.g., [T, 5, 1280]
        frame_num: Number of video frames
        frame0_idx: Starting frame index
        audio_shift: Overlap shift between windows (default 2)

    Returns:
        Tuple of (windowed_audio [iter, 8, blocks, channels], end_frame_idx)
    """
    zero_audio_embed = torch.zeros((audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
    zero_audio_embed_3 = torch.zeros((3, audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)

    # Number of latent iterations (VAE temporal stride is 4)
    iter_ = 1 + (frame_num - 1) // 4
    audio_emb_wind = []

    for lt_i in range(iter_):
        if lt_i == 0:
            # First latent frame: pad left with zeros
            st = frame0_idx + lt_i - 2
            ed = frame0_idx + lt_i + 3
            wind_feat = torch.stack([
                audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                for i in range(st, ed)
            ], dim=0)  # [5, blocks, channels]
            wind_feat = torch.cat((zero_audio_embed_3, wind_feat), dim=0)  # [8, blocks, channels]
        else:
            # Subsequent latent frames: sliding window
            st = frame0_idx + 1 + 4 * (lt_i - 1) - audio_shift
            ed = frame0_idx + 1 + 4 * lt_i + audio_shift
            wind_feat = torch.stack([
                audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                for i in range(st, ed)
            ], dim=0)  # [8, blocks, channels]
        audio_emb_wind.append(wind_feat)

    audio_emb_wind = torch.stack(audio_emb_wind, dim=0)  # [iter_, 8, blocks, channels]

    return audio_emb_wind, ed - audio_shift


def audio_emb_enc(audio_emb: torch.Tensor, wav_enc_type: str = "whisper") -> torch.Tensor:
    """
    Encode audio embeddings from Whisper hidden states.

    Takes Whisper encoder hidden states and averages them into 5 feature groups,
    then interpolates from 50fps to 25fps for video alignment.

    Args:
        audio_emb: Whisper hidden states [batch, time, layers, channels]
                   e.g., [1, T, 33, 1280]
        wav_enc_type: Encoder type ("whisper" or "wav2vec")

    Returns:
        Encoded features [time', 5, channels] e.g., [T/2, 5, 1280]
    """
    if wav_enc_type == "wav2vec":
        feat_merge = audio_emb
    elif wav_enc_type == "whisper":
        # Average hidden states into 5 groups and interpolate
        # [1, T, 33, 1280] -> 5 groups of 8 layers each (except last)
        feat0 = linear_interpolation_fps(audio_emb[:, :, 0: 8].mean(dim=2), 50, 25)
        feat1 = linear_interpolation_fps(audio_emb[:, :, 8: 16].mean(dim=2), 50, 25)
        feat2 = linear_interpolation_fps(audio_emb[:, :, 16: 24].mean(dim=2), 50, 25)
        feat3 = linear_interpolation_fps(audio_emb[:, :, 24: 32].mean(dim=2), 50, 25)
        feat4 = linear_interpolation_fps(audio_emb[:, :, 32], 50, 25)  # Last layer alone
        feat_merge = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]  # [T/2, 5, 1280]
    else:
        raise ValueError(f"Unsupported wav_enc_type: {wav_enc_type}")

    return feat_merge


class HuMoAudioProcessor:
    """
    Audio Processor for HuMo model.

    Uses Whisper-large-v3 for audio feature extraction. Supports optional
    vocal separation for cleaner audio input.

    Usage:
        processor = HuMoAudioProcessor(whisper_model_path, device="cuda")
        audio_emb, audio_length = processor.preprocess(audio_path)
        audio_windowed, _ = processor.get_audio_emb_window(audio_emb, frame_num)
    """

    def __init__(
        self,
        whisper_model_path: str,
        sample_rate: int = 16000,
        fps: int = 25,
        audio_separator_model_path: Optional[str] = None,
        audio_separator_model_name: Optional[str] = None,
        cache_dir: str = '',
        device: str = "cuda",
    ):
        """
        Initialize HuMo Audio Processor.

        Args:
            whisper_model_path: Path to Whisper model (e.g., openai/whisper-large-v3)
            sample_rate: Audio sample rate (default 16000)
            fps: Target frames per second (default 25)
            audio_separator_model_path: Optional path to vocal separator model
            audio_separator_model_name: Optional vocal separator model name
            cache_dir: Cache directory for intermediate files
            device: Device to run models on
        """
        self.sample_rate = sample_rate
        self.fps = fps
        self.device = device
        self.whisper = None
        self.feature_extractor = None
        self.audio_separator = None
        self.whisper_model_path = whisper_model_path

        # Lazy loading - models are loaded on first use
        self._whisper_loaded = False
        self._separator_loaded = False

        self.audio_separator_model_path = audio_separator_model_path
        self.audio_separator_model_name = audio_separator_model_name
        self.cache_dir = cache_dir

    def _load_whisper(self):
        """Lazy load Whisper model."""
        if self._whisper_loaded:
            return

        try:
            from transformers import WhisperModel, AutoFeatureExtractor
        except ImportError:
            raise ImportError("transformers package is required for HuMo audio processing. "
                            "Install with: pip install transformers")

        logger.info(f"Loading Whisper model from {self.whisper_model_path}")
        self.whisper = WhisperModel.from_pretrained(self.whisper_model_path).to(self.device).eval()
        self.whisper.requires_grad_(False)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.whisper_model_path)
        self._whisper_loaded = True
        logger.info("Whisper model loaded successfully")

    def _load_separator(self):
        """Lazy load audio separator model."""
        if self._separator_loaded or self.audio_separator_model_name is None:
            return

        try:
            from audio_separator.separator import Separator
        except ImportError:
            logger.warning("audio_separator package not available. Skipping vocal separation.")
            self._separator_loaded = True
            return

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError:
            logger.warning("Failed to create cache directory for audio separator")

        self.audio_separator = Separator(
            output_dir=self.cache_dir,
            output_single_stem="vocals",
            model_file_dir=self.audio_separator_model_path,
        )
        self.audio_separator.load_model(self.audio_separator_model_name)

        if self.audio_separator.model_instance is None:
            logger.warning("Failed to load audio separator model")
            self.audio_separator = None

        self._separator_loaded = True

    def get_audio_feature(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Extract audio features using Whisper feature extractor.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_features [1, 80, T], audio_length)
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa package is required for audio loading. "
                            "Install with: pip install librosa")

        self._load_whisper()

        audio_input, sampling_rate = librosa.load(audio_path, sr=16000)
        assert sampling_rate == 16000, f"Expected 16kHz audio, got {sampling_rate}Hz"

        # Process in windows to handle long audio
        audio_features = []
        window = 750 * 640  # ~30 seconds at 16kHz

        for i in range(0, len(audio_input), window):
            audio_feature = self.feature_extractor(
                audio_input[i:i+window],
                sampling_rate=sampling_rate,
                return_tensors="pt",
            ).input_features
            audio_features.append(audio_feature)

        audio_features = torch.cat(audio_features, dim=-1)
        return audio_features, len(audio_input) // 640

    def preprocess(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio file and extract HuMo-compatible features.

        Args:
            audio_path: Path to audio file (.wav)

        Returns:
            Tuple of (audio_emb [T, 5, 1280], audio_length)
        """
        self._load_whisper()

        audio_input, audio_len = self.get_audio_feature(audio_path)
        audio_feature = audio_input.to(self.whisper.device).float()

        # Process through Whisper encoder in windows
        window = 3000  # Whisper max input length
        audio_prompts = []

        for i in range(0, audio_feature.shape[-1], window):
            audio_prompt = self.whisper.encoder(
                audio_feature[:, :, i:i+window],
                output_hidden_states=True
            ).hidden_states
            audio_prompt = torch.stack(audio_prompt, dim=2)
            audio_prompts.append(audio_prompt)

        audio_prompts = torch.cat(audio_prompts, dim=1)
        audio_prompts = audio_prompts[:, :audio_len*2]  # Trim to actual length

        # Encode to HuMo format
        audio_emb = audio_emb_enc(audio_prompts, wav_enc_type="whisper")

        return audio_emb, audio_emb.shape[0]

    def preprocess_from_features(self, audio_feat_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load pre-extracted audio features from file.

        Args:
            audio_feat_path: Path to .pt file with audio features

        Returns:
            Tuple of (audio_emb [T, 5, 1280], audio_length)
        """
        audio_emb = torch.load(audio_feat_path)

        # Check if features need encoding
        if audio_emb.dim() == 4:
            # Raw Whisper hidden states [1, T, layers, channels]
            audio_emb = audio_emb_enc(audio_emb, wav_enc_type="whisper")
        elif audio_emb.dim() == 3 and audio_emb.shape[1] == 5:
            # Already encoded [T, 5, 1280]
            pass
        else:
            raise ValueError(f"Unexpected audio feature shape: {audio_emb.shape}")

        return audio_emb, audio_emb.shape[0]

    def get_audio_emb_window(self, audio_emb: torch.Tensor, frame_num: int,
                             frame0_idx: int = 0, audio_shift: int = 2) -> Tuple[torch.Tensor, int]:
        """
        Create windowed audio embeddings for HuMo model.

        Args:
            audio_emb: Audio embeddings [T, 5, 1280]
            frame_num: Number of video frames
            frame0_idx: Starting frame index
            audio_shift: Overlap shift (default 2)

        Returns:
            Tuple of (windowed_audio [iter, 8, 5, 1280], end_idx)
        """
        return get_audio_emb_window(audio_emb, frame_num, frame0_idx, audio_shift)

    def to_device(self, device: str):
        """Move models to specified device."""
        self.device = device
        if self.whisper is not None:
            self.whisper.to(device)

    def offload(self):
        """Offload models to CPU to free GPU memory."""
        if self.whisper is not None:
            self.whisper.cpu()
        torch.cuda.empty_cache()

    def close(self):
        """Clean up resources."""
        if self.whisper is not None:
            del self.whisper
            self.whisper = None
        if self.feature_extractor is not None:
            del self.feature_extractor
            self.feature_extractor = None
        if self.audio_separator is not None:
            del self.audio_separator
            self.audio_separator = None
        torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()


def load_zero_vae(path: str, target_frames: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Load precomputed zero VAE latents for HuMo conditioning.

    Args:
        path: Path to zero VAE cache file (.pt)
        target_frames: Number of target latent frames
        dtype: Target dtype
        device: Target device

    Returns:
        Zero VAE tensor [16, target_frames, H, W]
    """
    zero_vae = torch.load(path, map_location='cpu')
    return zero_vae[:, :target_frames].to(device=device, dtype=dtype)


def create_humo_conditioning(
    ref_latent: torch.Tensor,
    target_frames: int,
    zero_vae: torch.Tensor,
    lat_h: int,
    lat_w: int,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Create HuMo conditioning tensor (y) for inference.

    Constructs the conditioning tensor by combining:
    - Mask (4 channels): Indicates which frames are conditioned
    - Zero VAE + Reference latent (16 channels): Latent space conditioning

    Args:
        ref_latent: Reference image latent [16, ref_frames, H, W]
        target_frames: Total number of target latent frames
        zero_vae: Precomputed zero latents [16, max_frames, H, W]
        lat_h: Latent height
        lat_w: Latent width
        device: Target device
        dtype: Target dtype

    Returns:
        Conditioning tensor y [20, target_frames, H, W]
    """
    ref_frames = ref_latent.shape[1]

    # Create mask: 1 for reference frames, 0 for frames to generate
    msk = torch.ones(4, target_frames, lat_h, lat_w, device=device, dtype=dtype)
    msk[:, :-ref_frames] = 0  # Zero out frames to generate

    # Get zero VAE for non-reference frames
    zero_frames = target_frames - ref_frames
    zero_latent = zero_vae[:, :zero_frames].to(device=device, dtype=dtype)

    # Concatenate: [zero_frames | ref_frames]
    y_c = torch.cat([zero_latent, ref_latent.to(device=device, dtype=dtype)], dim=1)

    # Final conditioning: [mask(4) | latent(16)] = [20, F, H, W]
    y = torch.cat([msk, y_c], dim=0)

    return y
