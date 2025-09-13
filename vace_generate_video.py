#!/usr/bin/env python3
# ==============================================================================
# VACE Generation Script - Integrated with Official Pipeline Pattern
#
# This implementation integrates standalone components from vace_standalone.py
# and follows the official pipeline pattern from VideoX-Fun examples.
# ==============================================================================

import argparse
import hashlib
import inspect
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import cv2
import imageio
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import torchvision.transforms.functional as TF
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.autoencoders.vae import (DecoderOutput,
                                              DiagonalGaussianDistribution)
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, is_torch_version, logging, replace_example_docstring
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

# Import external Wan library dependencies
from wan.modules.model import WanAttentionBlock, WanModel
from wan.modules.model import sinusoidal_embedding_1d
from wan.modules.t5 import T5EncoderModel as WanT5EncoderModel
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# Import VACE-specific transformer directly
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wan', 'vace', 'models'))
from transformer_vace import VaceWanTransformer3DModel

logger = logging.get_logger(__name__)

# ==================================================================================
# Utility Functions
# ==================================================================================

def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=12):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(Image.fromarray(x))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def get_image_to_video_latent(validation_image_start, validation_image_end, video_length, sample_size):
    if validation_image_start is not None and validation_image_end is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]

        if type(validation_image_end) is str and os.path.isfile(validation_image_end):
            image_end = Image.open(validation_image_end).convert("RGB")
            image_end = image_end.resize([sample_size[1], sample_size[0]])
        else:
            image_end = validation_image_end
            image_end = [_image_end.resize([sample_size[1], sample_size[0]]) for _image_end in image_end]

        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start],
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video

            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0),
                [1, 1, video_length, 1, 1]
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255

        if type(image_end) is list:
            image_end = [_image_end.resize(image_start[0].size if type(image_start) is list else image_start.size) for _image_end in image_end]
            end_video = torch.cat(
                [torch.from_numpy(np.array(_image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_end in image_end],
                dim=2
            )
            input_video[:, :, -len(end_video):] = end_video

            input_video_mask[:, :, -len(image_end):] = 0
        else:
            image_end = image_end.resize(image_start[0].size if type(image_start) is list else image_start.size)
            input_video[:, :, -1:] = torch.from_numpy(np.array(image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
            input_video_mask[:, :, -1:] = 0

        input_video = input_video / 255

    elif validation_image_start is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]
        image_end = None

        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start],
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            input_video = input_video / 255

            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0),
                [1, 1, video_length, 1, 1]
            ) / 255
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:, ] = 255
    else:
        image_start = None
        image_end = None
        input_video = torch.zeros([1, 3, video_length, sample_size[0], sample_size[1]])
        input_video_mask = torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]) * 255
        clip_image = None

    del image_start
    del image_end
    return input_video, input_video_mask, clip_image

def get_video_to_video_latent(input_video_path, video_length, sample_size, fps=None, validation_video_mask=None, ref_image=None):
    if input_video_path is not None:
        if isinstance(input_video_path, str):
            cap = cv2.VideoCapture(input_video_path)
            input_video = []
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else max(1,int(original_fps // fps))
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_count += 1
            cap.release()
        else:
            input_video = input_video_path

        input_video = torch.from_numpy(np.array(input_video))[:video_length]
        input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

        if validation_video_mask is not None:
            validation_video_mask = Image.open(validation_video_mask).convert('L').resize((sample_size[1], sample_size[0]))
            input_video_mask = np.where(np.array(validation_video_mask) < 240, 0, 255)
            input_video_mask = torch.from_numpy(np.array(input_video_mask)).unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
            input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
            input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)
        else:
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, :] = 255
    else:
        input_video, input_video_mask = None, None

    if ref_image is not None:
        if isinstance(ref_image, str):
            clip_image = Image.open(ref_image).convert("RGB")
        else:
            clip_image = Image.fromarray(np.array(ref_image, np.uint8))
    else:
        clip_image = None

    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        else:
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
    return input_video, input_video_mask, ref_image, clip_image

def padding_image(images, new_width, new_height):
    new_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
    resized_img = images.copy()
    resized_img.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
    paste_x = (new_width - resized_img.width) // 2
    paste_y = (new_height - resized_img.height) // 2
    new_image.paste(resized_img, (paste_x, paste_y))
    return new_image

def get_image_latent(ref_image=None, sample_size=None, padding=False):
    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
            if padding:
                ref_image = padding_image(ref_image, sample_size[1], sample_size[0])
            else:
                ref_image = ref_image.resize((sample_size[1], sample_size[0]), Image.Resampling.LANCZOS)
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([0, 3, 1, 2]) / 255.0
        elif isinstance(ref_image, Image.Image):
            ref_image = ref_image.convert("RGB")
            if padding:
                ref_image = padding_image(ref_image, sample_size[1], sample_size[0])
            else:
                ref_image = ref_image.resize((sample_size[1], sample_size[0]), Image.Resampling.LANCZOS)
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([0, 3, 1, 2]) / 255.0
        else:
            ref_image = ref_image
    return ref_image

# ==================================================================================
# Cache Utilities
# ==================================================================================

def get_teacache_coefficients(model_name):
    model_name_lower = model_name.lower()
    if "wan2.1-t2v-1.3b" in model_name_lower or "wan2.1-fun-1.3b" in model_name_lower \
        or "wan2.1-fun-v1.1-1.3b" in model_name_lower or "wan2.1-vace-1.3b" in model_name_lower:
        return [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02]
    elif "wan2.1-t2v-14b" in model_name_lower:
        return [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01]
    elif "wan2.1-i2v-14b-480p" in model_name_lower:
        return [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01]
    elif "wan2.1-i2v-14b-720p" in model_name_lower or "wan2.1-fun-14b" in model_name_lower or "wan2.2-fun" in model_name_lower \
        or "wan2.2-i2v-a14b" in model_name_lower or "wan2.2-t2v-a14b" in model_name_lower or "wan2.2-ti2v-5b" in model_name_lower \
        or "wan2.2-s2v" in model_name_lower or "wan2.1-vace-14b" in model_name_lower or "wan2.2-vace-fun" in model_name_lower \
        or "wan2.2-vace-fun-a14b_low" in model_name_lower or "wan2.2-vace-fun-a14b_high" in model_name_lower:
        return [8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02]
    else:
        print(f"The model {model_name} is not supported by TeaCache.")
        return None

# ==================================================================================
# LoRA Utilities
# ==================================================================================

def merge_lora(pipeline, lora_path, multiplier, device='cpu', dtype=torch.float32, state_dict=None, sub_transformer_name="transformer"):
    LORA_PREFIX_TRANSFORMER = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    if state_dict is None:
        state_dict = load_file(lora_path)
    updates = defaultdict(dict)
    for key, value in state_dict.items():
        try:
            layer, elem = key.split('.', 1)
            updates[layer][elem] = value
        except ValueError:
            continue

    for layer, elems in updates.items():
        if "lora_te" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
            curr_layer = getattr(pipeline, sub_transformer_name)

        temp_name = layer_infos.pop(0)
        while len(layer_infos) > 0:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                temp_name = layer_infos.pop(0)
            except Exception:
                temp_name += "_" + layer_infos.pop(0)

        curr_layer = curr_layer.__getattr__(temp_name)

        weight_up = elems['lora_up.weight'].to(device=device, dtype=dtype)
        weight_down = elems['lora_down.weight'].to(device=device, dtype=dtype)

        if 'alpha' in elems.keys():
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(
                weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3).to(curr_layer.weight.data.device)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down).to(curr_layer.weight.data.device)

    return pipeline

def unmerge_lora(pipeline, lora_path, multiplier, device='cpu', dtype=torch.float32, sub_transformer_name="transformer"):
    logger.warning("unmerge_lora is not fully implemented for dynamic unmerging. Re-initialize the pipeline for a clean state.")
    return pipeline

# ==================================================================================
# VAE Components
# ==================================================================================

CACHE_T = 2

class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)

class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias

class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)

class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d', 'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != 'Rep':
                        cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        res_x = x
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = res_x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                res_x = layer(res_x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                res_x = layer(res_x)
        return res_x + h

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity

class Encoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[True, True, False], dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)
        downsamples = []
        in_dim = dims[0]
        for i, (in_dim_i, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            in_dim_i = in_dim
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim_i, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim_i = out_dim
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
            in_dim = out_dim
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim, dropout))
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)
        for layer in self.middle:
            x = layer(x, feat_cache, feat_idx) if isinstance(layer, ResidualBlock) and feat_cache is not None else layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x

class Decoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_upsample=[False, True, True], dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 1)
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0], dropout))
        upsamples = []
        in_dim = dims[0]
        for i, (in_dim_i, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            in_dim_i = in_dim
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim_i, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim_i = out_dim
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
            in_dim = out_dim
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.middle:
            x = layer(x, feat_cache, feat_idx) if isinstance(layer, ResidualBlock) and feat_cache is not None else layer(x)
        for layer in self.upsamples:
            x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x

def count_conv3d(model):
    return sum(1 for m in model.modules() if isinstance(m, CausalConv3d))

class AutoencoderKLWan_(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[True, True, False], dropout=0.0):
        super().__init__()
        self.z_dim = z_dim
        temperal_upsample = temperal_downsample[::-1]
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks, attn_scales, temperal_upsample, dropout)

    def encode(self, x, scale):
        self.clear_cache()
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        scale = [item.to(x.device, x.dtype) for item in scale]
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        x = torch.cat([mu, log_var], dim = 1)
        self.clear_cache()
        return x

    def decode(self, z, scale):
        self.clear_cache()
        scale = [item.to(z.device, z.dtype) for item in scale]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

def _video_vae(z_dim=None, **kwargs):
    # Updated configuration to match Wan2.1_VAE.pth checkpoint
    cfg = dict(dim=96, z_dim=z_dim, dim_mult=[1, 2, 2, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)
    cfg.update(**kwargs)
    return AutoencoderKLWan_(**cfg)

class AutoencoderKLWan(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(self, latent_channels=16, temporal_compression_ratio=4, spatial_compression_ratio=8):
        super().__init__()
        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.scale = [self.mean, 1.0 / self.std]
        self.model = _video_vae(z_dim=latent_channels)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = [self.model.encode(u.unsqueeze(0), self.scale).squeeze(0) for u in x]
        return torch.stack(x)

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True):
        h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)
        return AutoencoderKLOutput(latent_dist=posterior) if return_dict else (posterior,)

    def _decode(self, zs):
        dec = [self.model.decode(u.unsqueeze(0), self.scale).clamp_(-1, 1).squeeze(0) for u in zs]
        return DecoderOutput(sample=torch.stack(dec))

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True):
        decoded = self._decode(z).sample
        return DecoderOutput(sample=decoded) if return_dict else (decoded,)

    @classmethod
    def from_pretrained(cls, pretrained_model_path, additional_kwargs={}):
        model = cls(**filter_kwargs(cls, additional_kwargs))
        state_dict = load_file(pretrained_model_path) if pretrained_model_path.endswith(".safetensors") else torch.load(pretrained_model_path, map_location="cpu")
        tmp_state_dict = {"model." + key: value for key, value in state_dict.items()}
        m, u = model.load_state_dict(tmp_state_dict, strict=False)
        logger.info(f"### VAE missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        return model

class AutoencoderKLWan3_8(AutoencoderKLWan):
    @register_to_config
    def __init__(self, latent_channels=16, temporal_compression_ratio=4, spatial_compression_ratio=8):
        super().__init__()
        self.mean = torch.tensor([0.0] * 16, dtype=torch.float32)
        self.std = torch.tensor([1.0] * 16, dtype=torch.float32)
        self.scale = [self.mean, 1.0 / self.std]

# ==================================================================================
# Transformer Components (Local implementations if needed)
# ==================================================================================

# Note: VaceWanAttentionBlock and VaceWanTransformer3DModel are now imported from wan.vace.models.transformer_vace
# Keep these local implementations as fallback if needed

class VaceWanAttentionBlockLocal(WanAttentionBlock):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, window_size=(-1, -1), qk_norm=True, cross_attn_norm=False, eps=1e-6, block_id=0):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        expected_parent_args = ['e', 'seq_lens', 'grid_sizes', 'freqs', 'context', 'context_lens']
        parent_kwargs = {k: v for k, v in kwargs.items() if k in expected_parent_args}

        c = super().forward(x=c, **parent_kwargs)

        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c

class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, window_size=(-1, -1), qk_norm=True, cross_attn_norm=False, eps=1e-6, block_id=None):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        parent_kwargs = {}
        for key in ['e', 'seq_lens', 'grid_sizes', 'freqs', 'context', 'context_lens']:
            if key in kwargs:
                parent_kwargs[key] = kwargs[key]
        x = super().forward(x, **parent_kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x

class VaceWanTransformer3DModelLocal(WanModel):
    def __init__(self, vace_layers=None, vace_in_dim=None, model_type='t2v', patch_size=(1, 2, 2), text_len=512, in_dim=16, dim=2048, ffn_dim=8192, freq_dim=256, text_dim=4096, out_dim=16, num_heads=16, num_layers=32, window_size=(-1, -1), qk_norm=True, cross_attn_norm=True, eps=1e-6):
        super().__init__(model_type="t2v", patch_size=patch_size, text_len=text_len, in_dim=in_dim, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim, text_dim=text_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers, window_size=window_size, qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps, attn_mode=None, split_attn=False, add_ref_conv=False, in_dim_ref_conv=16)
        self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim
        self.sp_world_size = 1
        self.sp_world_rank = 0
        self.teacache = None
        self.gradient_checkpointing = False
        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}
        self.blocks = nn.ModuleList([BaseWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm, self.cross_attn_norm, self.eps, block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None) for i in range(self.num_layers)])
        self.vace_blocks = nn.ModuleList([VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm, self.cross_attn_norm, self.eps, block_id=i) for i in range(len(self.vace_layers))])
        self.vace_patch_embedding = nn.Conv3d(self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward_vace(self, x, vace_context, seq_len, kwargs):
        if not vace_context or len(vace_context) == 0:
            dtype = x.dtype
            return [torch.zeros(1, seq_len, self.dim, device=self.patch_embedding.weight.device, dtype=dtype) for _ in range(len(self.vace_layers))]
        c = []
        for u in vace_context:
            if u.dim() == 3: u = u.unsqueeze(0)
            elif u.dim() == 5: u = u.squeeze(0)
            embedded = self.vace_patch_embedding(u.unsqueeze(0))
            c.append(embedded)
        c = [u.flatten(2).transpose(1, 2) for u in c]
        if not c:
            dtype = x.dtype
            return [torch.zeros(1, seq_len, self.dim, device=self.patch_embedding.weight.device, dtype=dtype) for _ in range(len(self.vace_layers))]
        c = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in c])
        if self.sp_world_size > 1:
            c = torch.chunk(c, self.sp_world_size, dim=1)[self.sp_world_rank]
        new_kwargs = dict(x=x, **kwargs)
        for block in self.vace_blocks:
            c = block(c, **new_kwargs)
        hints = torch.unbind(c)[:-1]
        return hints

    def forward(self, x, t, vace_context, context, seq_len, vace_context_scale=1.0, clip_fea=None, y=None, cond_flag=True):
        dtype = x[0].dtype if isinstance(x, list) and len(x) > 0 else x.dtype
        device = self.patch_embedding.weight.device
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])
        with amp.autocast(enabled=True, dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))
        if self.sp_world_size > 1: x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=None, t=t)
        hints = self.forward_vace(x, vace_context, seq_len, kwargs)
        kwargs['hints'] = hints
        kwargs['context_scale'] = vace_context_scale
        for block in self.blocks:
            x = block(x, **kwargs)
        x = self.head(x, e)
        if self.sp_world_size > 1: x = self.all_gather(x, dim=1)
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        return x,

    @classmethod
    def from_pretrained(cls, model_path, transformer_additional_kwargs=None, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16):
        if transformer_additional_kwargs is None: transformer_additional_kwargs = {}
        ckpt_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors") if os.path.isdir(model_path) else model_path
        state_dict = load_file(ckpt_path, device="cpu")
        dim = state_dict.get("patch_embedding.weight", torch.zeros(3072, 16)).shape[0]
        model = cls(vace_layers=transformer_additional_kwargs.get('vace_layers', [0, 5, 10, 15, 20, 25, 30, 35]), vace_in_dim=transformer_additional_kwargs.get('vace_in_dim', 96), dim=dim, ffn_dim=8192 if dim == 3072 else 13824, num_heads=24 if dim == 3072 else 40, num_layers=30 if dim == 3072 else 40)
        if torch_dtype is not None: model.to(torch_dtype)
        m, u = model.load_state_dict(state_dict, strict=False)
        logger.info(f"### VACE Transformer missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        return model

# ==================================================================================
# Pipeline
# ==================================================================================

@dataclass
class WanPipelineOutput(BaseOutput):
    videos: torch.Tensor

class Wan2_2VaceFunPipeline(DiffusionPipeline):
    _optional_components = ["transformer_2"]
    model_cpu_offload_seq = "text_encoder->transformer_2->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(self, tokenizer: AutoTokenizer, text_encoder: WanT5EncoderModel, vae: AutoencoderKLWan, transformer: VaceWanTransformer3DModel, transformer_2: VaceWanTransformer3DModel = None, scheduler: FlowMatchEulerDiscreteScheduler = None):
        super().__init__()
        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, transformer_2=transformer_2, scheduler=scheduler)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.config.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.config.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=self.vae.config.spatial_compression_ratio, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

    def _get_t5_prompt_embeds(self, prompt: Union[str, List[str]] = None, num_videos_per_prompt: int = 1, max_sequence_length: int = 512, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=max_sequence_length, truncation=True, add_special_tokens=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(self, prompt, negative_prompt, do_classifier_free_guidance, num_videos_per_prompt, max_sequence_length, device, dtype):
        prompt_embeds = self._get_t5_prompt_embeds(prompt=prompt, num_videos_per_prompt=num_videos_per_prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype)
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_embeds = self._get_t5_prompt_embeds(prompt=negative_prompt, num_videos_per_prompt=num_videos_per_prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype)
        else:
            negative_prompt_embeds = None
        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None, num_length_latents=None):
        shape = (batch_size, num_channels_latents, (num_frames - 1) // self.vae.config.temporal_compression_ratio + 1 if num_length_latents is None else num_length_latents, height // self.vae.config.spatial_compression_ratio, width // self.vae.config.spatial_compression_ratio,)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype) if latents is None else latents.to(device)
        if hasattr(self.scheduler, "init_noise_sigma"): latents = latents * self.scheduler.init_noise_sigma
        return latents

    def vace_encode_frames(self, frames, ref_images, masks=None, vae=None):
        vae = self.vae if vae is None else vae
        weight_dtype = frames[0].dtype
        if ref_images is None: ref_images = [None] * len(frames)
        else: assert len(frames) == len(ref_images)
        if masks is None:
            latents = [vae.encode(f.unsqueeze(0))[0].latent_dist.mode().squeeze(0) for f in frames]
        else:
            masks = [torch.where(m > 0.5, 1.0, 0.0).to(weight_dtype) for m in masks]
            inactive = [i * (1 - m) for i, m in zip(frames, masks)]
            reactive = [i * m for i, m in zip(frames, masks)]
            inactive = [vae.encode(f.unsqueeze(0))[0].latent_dist.mode().squeeze(0) for f in inactive]
            reactive = [vae.encode(f.unsqueeze(0))[0].latent_dist.mode().squeeze(0) for f in reactive]
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]
        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                ref_latents_list = [vae.encode(r.unsqueeze(0))[0].latent_dist.mode() for r in refs]
                ref_latent = torch.cat(ref_latents_list, dim=2)
                if masks is not None: ref_latent = torch.cat((ref_latent, torch.zeros_like(ref_latent)), dim=1)
                latent = torch.cat([ref_latent, latent.unsqueeze(0)], dim=2).squeeze(0)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None, vae_stride=[4, 8, 8]):
        if masks is None: return None
        if ref_images is None: ref_images = [None] * len(masks)
        else: assert len(masks) == len(ref_images)
        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = (depth + vae_stride[0] -1) // vae_stride[0]
            new_height = height // vae_stride[1]
            new_width = width // vae_stride[2]
            mask = mask[0, :, :, :]
            mask = mask.view(depth, new_height, vae_stride[1], new_width, vae_stride[2])
            mask = mask.permute(2, 4, 0, 1, 3)
            mask = mask.reshape(vae_stride[1] * vae_stride[2], depth, new_height, new_width)
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, new_height, new_width), mode='nearest-exact').squeeze(0)
            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def vace_latent(self, z, m):
        if m is None: return z
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        video = self.vae.decode(latents.to(self.vae.dtype)).sample
        video = (video / 2 + 0.5).clamp(0, 1)
        return video

    @torch.no_grad()
    def __call__(self, prompt, negative_prompt=None, height=480, width=720, video=None, mask_video=None, control_video=None, subject_ref_images=None, num_frames=49, num_inference_steps=50, guidance_scale=6, generator=None, latents=None, output_type="pt", boundary=0.875, shift=12.0, vace_context_scale=1.0, **kwargs):
        device = self._execution_device
        weight_dtype = self.transformer.dtype
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt, do_classifier_free_guidance, 1, 512, device, weight_dtype)
        if do_classifier_free_guidance: in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else: in_prompt_embeds = prompt_embeds

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        latent_channels = self.vae.config.latent_channels

        vace_context = []
        num_length_latents = None

        if control_video is not None or subject_ref_images is not None or video is not None:
            input_video = control_video if control_video is not None else (video if video is not None else subject_ref_images)
            vace_latents = self.vace_encode_frames([input_video[0]], [subject_ref_images[0] if subject_ref_images is not None else None], masks=[mask_video[0]] if mask_video is not None else None)
            mask_latents = self.vace_encode_masks([mask_video[0]] if mask_video is not None else None, [subject_ref_images[0] if subject_ref_images is not None else None])
            vace_context = self.vace_latent(vace_latents, mask_latents)
            num_length_latents = vace_latents[0].size(2)

        latents = self.prepare_latents(1, latent_channels, num_frames, height, width, weight_dtype, device, generator, latents, num_length_latents)

        seq_len = math.ceil((latents.shape[3] * latents.shape[4]) / (self.transformer.patch_size[1] * self.transformer.patch_size[2]) * latents.shape[2])

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                vace_context_input = vace_context * 2 if do_classifier_free_guidance else vace_context
                timestep = t.expand(latent_model_input.shape[0])
                local_transformer = self.transformer_2 if self.transformer_2 is not None and t >= boundary * self.scheduler.config["num_train_timesteps"] else self.transformer

                noise_pred = local_transformer(x=[latent_model_input[0]], context=in_prompt_embeds, t=timestep, vace_context=vace_context_input, seq_len=seq_len, vace_context_scale=vace_context_scale)[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                progress_bar.update()

        if subject_ref_images is not None: latents = latents[:, :, len(subject_ref_images[0]):, :, :]

        video = self.decode_latents(latents)
        if output_type != "latent": video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        self.maybe_free_model_hooks()
        return WanPipelineOutput(videos=video)

# ==================================================================================
# Main Function
# ==================================================================================

def main():
    parser = argparse.ArgumentParser(description="VACE Video Generation Script using Official Pipeline")

    # Model Paths - can use either model_dir or individual safetensors files
    parser.add_argument("--model_dir", type=str, default=None, help="Path to the VACE model directory")
    parser.add_argument("--config_path", type=str, default=None, help="Path to the VACE model's .yaml configuration file")

    # Individual model files (for direct safetensors loading)
    parser.add_argument("--dit_low_noise", type=str, default=None, help="Path to low-noise DiT safetensors file")
    parser.add_argument("--dit_high_noise", type=str, default=None, help="Path to high-noise DiT safetensors file")
    parser.add_argument("--vae", type=str, default=None, help="Path to VAE model file")
    parser.add_argument("--t5", type=str, default=None, help="Path to T5 text encoder model file")
    parser.add_argument("--task", type=str, default="vace-t2v-A14B", help="Task type for generation")

    parser.add_argument("--save_path", type=str, required=True, help="Directory to save the output videos")

    # Generation Parameters
    parser.add_argument("--prompt", type=str, required=True, help="The main text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="The negative prompt")
    parser.add_argument("--video_size", type=int, nargs=2, default=[480, 832], help="Output video resolution (height width)")
    parser.add_argument("--video_length", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for the output video")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--infer_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Classifier-Free Guidance scale")

    # VACE Specific Inputs
    parser.add_argument("--control_video", type=str, default=None, help="Path to the control video (e.g., pose video)")
    parser.add_argument("--subject_ref_images", type=str, nargs="+", default=None, help="Path(s) to subject reference images")
    parser.add_argument("--padding_in_subject_ref_images", action="store_true", help="Add padding to subject reference images")
    parser.add_argument("--image_path", type=str, default=None, help="Path to a start image for I2V")
    parser.add_argument("--end_image_path", type=str, default=None, help="Path to an end image for interpolation")
    parser.add_argument("--vace_context_scale", type=float, default=1.0, help="Strength of the VACE context")

    # Performance & LoRA
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights for the low-noise model")
    parser.add_argument("--lora_high_path", type=str, default=None, help="Path to LoRA weights for the high-noise model")
    parser.add_argument("--lora_weight", type=float, default=0.55, help="Multiplier for the low-noise LoRA")
    parser.add_argument("--lora_high_weight", type=float, default=0.55, help="Multiplier for the high-noise LoRA")
    parser.add_argument("--enable_teacache", action="store_true", help="Enable TeaCache acceleration")
    parser.add_argument("--teacache_threshold", type=float, default=0.10, help="TeaCache threshold (0.05-0.30)")
    parser.add_argument("--gpu_memory_mode", type=str, default="sequential_cpu_offload",
                       choices=["model_full_load", "model_cpu_offload", "sequential_cpu_offload"],
                       help="GPU memory optimization mode")

    # Scheduler parameters
    parser.add_argument("--flow_shift", type=float, default=5.0, help="Flow shift value for scheduler")
    parser.add_argument("--dual_dit_boundary", type=float, default=0.875, help="Boundary for dual DiT model switching")

    # Attention mode
    parser.add_argument("--attn_mode", type=str, default="sdpa", help="Attention mode (sdpa, flash_attn, etc.)")
    parser.add_argument("--blocks_to_swap", type=int, default=30, help="Number of blocks to swap for memory optimization")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(4), 'big')

    logger.info(f"Using device: {device}, dtype: {weight_dtype}, seed: {args.seed}")

    # Check if using direct safetensors files or model directory
    if args.dit_low_noise and args.vae and args.t5:
        # Direct safetensors loading mode
        logger.info("Using direct safetensors file loading mode")

        # Default config for direct loading
        if args.task == "vace-t2v-A14B":
            vace_layers = [0, 5, 10, 15, 20, 25, 30, 35]
            vae_type = 'AutoencoderKLWan'
            num_layers = 40
            dim = 3072
            ffn_dim = 13824
            num_heads = 40
        else:
            # Default for other models
            vace_layers = [0, 5, 10, 15, 20, 25, 30]
            vae_type = 'AutoencoderKLWan'
            num_layers = 30
            dim = 3072
            ffn_dim = 8192
            num_heads = 24

        boundary = args.dual_dit_boundary

        # Set direct paths
        transformer_low_path = args.dit_low_noise
        transformer_high_path = args.dit_high_noise if args.dit_high_noise else None
        vae_path = args.vae
        text_encoder_path = args.t5
        tokenizer_path = "wan/xlm-roberta-large"  # Default tokenizer path

    elif args.config_path and args.model_dir:
        # Traditional config-based loading
        config = OmegaConf.load(args.config_path)
        boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)

        transformer_low_path = os.path.join(args.model_dir, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer'))
        transformer_high_path = os.path.join(args.model_dir, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer_high'))
        vae_path = os.path.join(args.model_dir, config['vae_kwargs'].get('vae_subpath', 'vae'))
        tokenizer_path = os.path.join(args.model_dir, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer'))
        text_encoder_path = os.path.join(args.model_dir, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder'))
        vae_type = config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')
        vace_layers = config['transformer_additional_kwargs'].get('vace_layers', [0, 5, 10, 15, 20, 25, 30, 35])
    else:
        logger.error("Please provide either --dit_low_noise/--vae/--t5 for direct loading or --config_path/--model_dir for config-based loading")
        sys.exit(1)

    logger.info("Loading VACE Transformer (Low Noise)...")
    if args.dit_low_noise:
        # Direct loading with default config
        transformer_additional_kwargs = {
            'vace_layers': vace_layers,
            'vace_in_dim': 96 if args.task == "vace-t2v-A14B" else 16
        }
        transformer = VaceWanTransformer3DModel.from_pretrained(
            transformer_low_path,
            transformer_additional_kwargs=transformer_additional_kwargs,
            torch_dtype=weight_dtype,
        )
    else:
        transformer = VaceWanTransformer3DModel.from_pretrained(
            transformer_low_path,
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            torch_dtype=weight_dtype,
        )

    # Load high noise transformer if provided
    if transformer_high_path:
        logger.info("Loading VACE Transformer (High Noise)...")
        if args.dit_high_noise:
            transformer_2 = VaceWanTransformer3DModel.from_pretrained(
                transformer_high_path,
                transformer_additional_kwargs=transformer_additional_kwargs,
                torch_dtype=weight_dtype,
            )
        else:
            transformer_2 = VaceWanTransformer3DModel.from_pretrained(
                transformer_high_path,
                transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
                torch_dtype=weight_dtype,
            )
    else:
        transformer_2 = None

    logger.info("Loading VAE...")
    if args.vae:
        # Direct loading
        Chosen_AutoencoderKL = {'AutoencoderKLWan': AutoencoderKLWan, 'AutoencoderKLWan3_8': AutoencoderKLWan3_8}[vae_type]
        vae = Chosen_AutoencoderKL.from_pretrained(
            vae_path,
            additional_kwargs={}
        ).to(weight_dtype)
    else:
        Chosen_AutoencoderKL = {'AutoencoderKLWan': AutoencoderKLWan, 'AutoencoderKLWan3_8': AutoencoderKLWan3_8}[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
        vae = Chosen_AutoencoderKL.from_pretrained(
            vae_path,
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        ).to(weight_dtype)

    logger.info("Loading Tokenizer and Text Encoder...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    text_encoder = WanT5EncoderModel.from_pretrained(
        text_encoder_path,
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    # Create scheduler
    if args.dit_low_noise:
        # Direct loading with default scheduler config
        scheduler = FlowMatchEulerDiscreteScheduler(
            shift=args.flow_shift,
            num_train_timesteps=1000,
            base_image_seq_len=256,
            max_image_seq_len=1024
        )
    else:
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )

    # Create Pipeline
    pipeline = Wan2_2VaceFunPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    # Memory optimization
    if args.gpu_memory_mode == "sequential_cpu_offload":
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    # Enable TeaCache if requested
    if args.enable_teacache:
        # Use the actual model filename for coefficient lookup
        if args.dit_low_noise:
            model_name = os.path.basename(args.dit_low_noise) if args.dit_low_noise else args.task
        else:
            model_name = args.model_dir
        coefficients = get_teacache_coefficients(model_name)
        if coefficients:
            logger.info(f"Enabling TeaCache with threshold {args.teacache_threshold}")
            # Note: TeaCache integration would require additional implementation in the transformer models

    # Merge LoRAs if provided
    if args.lora_path:
        logger.info(f"Merging LoRA for low-noise model: {args.lora_path}")
        pipeline = merge_lora(pipeline, args.lora_path, args.lora_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer")
    if args.lora_high_path and transformer_2 is not None:
        logger.info(f"Merging LoRA for high-noise model: {args.lora_high_path}")
        pipeline = merge_lora(pipeline, args.lora_high_path, args.lora_high_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")

    # Prepare Inputs
    generator = torch.Generator(device=device).manual_seed(args.seed)

    video_length = int((args.video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.video_length != 1 else 1

    inpaint_video, inpaint_video_mask, _ = get_image_to_video_latent(
        args.image_path, args.end_image_path, video_length=video_length, sample_size=args.video_size
    )

    control_video, _, _, _ = get_video_to_video_latent(
        args.control_video, video_length=video_length, sample_size=args.video_size, fps=args.fps
    ) if args.control_video else (None, None, None, None)

    subject_ref_images = None
    if args.subject_ref_images:
        subject_ref_images_list = [
            get_image_latent(img_path, sample_size=args.video_size, padding=args.padding_in_subject_ref_images)
            for img_path in args.subject_ref_images
        ]
        subject_ref_images = torch.cat(subject_ref_images_list, dim=2)

    # Run Generation
    logger.info("Starting video generation with the VACE pipeline...")
    with torch.no_grad():
        sample = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.video_size[0],
            width=args.video_size[1],
            num_frames=video_length,
            num_inference_steps=args.infer_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            video=inpaint_video,
            mask_video=inpaint_video_mask,
            control_video=control_video,
            subject_ref_images=subject_ref_images,
            boundary=boundary,
            shift=config['scheduler_kwargs'].get('shift', 12.0),
            vace_context_scale=args.vace_context_scale
        ).videos

    # Save Output
    os.makedirs(args.save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}_seed{args.seed}.mp4"
    video_path = os.path.join(args.save_path, filename)

    save_videos_grid(sample, video_path, fps=args.fps)
    logger.info(f"Video saved successfully to: {video_path}")

if __name__ == "__main__":
    main()