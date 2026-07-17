# Bernini-R (ByteDance) renderer inference utilities, ported from
# bernini_repo/bernini for use with the native WanModel implementation.
#
# Everything here mirrors the official implementation exactly:
#   - data_utils.py: resize/normalize preprocessing and smart frame sampling
#   - wan_diffusion.py: APG / normalized guidance and the FlowMatchScheduler
#   - cli.py / prompt_enhancer.py: default negative prompt and per-task system prompts
#   - gradio_demo.py: per-task default hyperparameters
#
# Original: Copyright (c) 2026 Bytedance Ltd. (Apache-2.0)

import html
import logging
import math
import re
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)

try:
    import ftfy

    _HAS_FTFY = True
except ImportError:
    _HAS_FTFY = False

try:
    import decord

    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False


# --------------------------------------------------------------------------- #
# Prompts / defaults (bernini/cli.py, prompt_enhancer.py, gradio_demo.py)
# --------------------------------------------------------------------------- #
DEFAULT_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

SYSTEM_PROMPTS = {
    "default": "You are a helpful assistant.",
    "t2i": "You are a helpful assistant specialized in text-to-image generation.",
    "t2v": "You are a helpful assistant specialized in text-to-video generation.",
    "i2i": "You are a helpful assistant specialized in image editing.",
    "r2i": "You are a helpful assistant specialized in subject-to-image generation.",
    "i2v": "You are a helpful assistant specialized in image-to-video generation.",
    "v2v": "You are a helpful assistant specialized in video editing.",
    "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
    "vi2v": "You are a helpful assistant specialized in video editing on content propagation.",
    "rv2v": "You are a helpful assistant specialized in video editing with reference.",
    "ads2v": "You are a helpful assistant specialized in ads insertion.",
    "vrc2v": (
        "You are a helpful assistant for editing. "
        "You may need to adjust the subject's action or position."
    ),
    "mv2v": (
        "You are a helpful assistant for editing. "
        "You might need to adjust the video's style, lighting, colors, "
        "textures, and the subject's pose or action."
    ),
}


def get_system_prompt_for_task(task_type: str) -> str:
    return SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["default"])


BERNINI_TASKS = ["t2i", "t2v", "i2v", "i2i", "v2v", "mv2v", "r2v", "rv2v", "ads2v"]
GUIDANCE_MODES = ["rv2v", "v2v", "v2v_chain", "t2v", "r2v_apg", "v2v_apg", "t2v_apg"]
IMAGE_TASKS = {"t2i", "i2i"}

GUIDANCE_MODE_BY_TASK = {
    "t2i": "t2v_apg",
    "t2v": "t2v_apg",
    "i2v": "r2v_apg",
    "i2i": "v2v",
    "v2v": "v2v_apg",
    "mv2v": "v2v_apg",
    "r2v": "r2v_apg",
    "rv2v": "rv2v",
    "ads2v": "v2v_apg",
}

# gradio_demo.py BASE_TASK_DEFAULTS (renderer-relevant subset)
BASE_DEFAULTS = {
    "max_image_size": 848,
    "infer_steps": 40,
    "video_length": 81,
    "flow_shift": 5.0,
    "fps": 16,
    "omega_vid": 1.25,
    "omega_img": 4.5,
    "omega_txt": 4.0,
    "omega_scale": 0.8,
    "eta": 0.5,
    "momentum": 0.0,
}

# gradio_demo.py RENDERER_TASK_DEFAULTS
TASK_DEFAULTS = {
    "t2i": {"video_length": 1},
    "i2i": {"video_length": 1},
    "t2v": {},
    "i2v": {},
    "v2v": {},
    "mv2v": {},
    "r2v": {},
    "rv2v": {},
    "ads2v": {},
}


def prompt_clean(text: str) -> str:
    """bernini/pipeline.py:_prompt_clean (ftfy + html unescape + whitespace)."""
    if _HAS_FTFY:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Preprocessing (bernini/data_utils.py)
# --------------------------------------------------------------------------- #
def make_divisible(value: int, stride: int) -> int:
    return max(stride, int(round(value / stride) * stride))


def _apply_scale(width, height, scale, stride):
    new_width = make_divisible(round(width * scale), stride)
    new_height = make_divisible(round(height * scale), stride)
    return new_width, new_height


class MaxLongEdgeMinShortEdgeResize(torch.nn.Module):
    """Resize so the long edge <= max_size and short edge >= min_size, snapped to `stride`."""

    def __init__(self, max_size, min_size, stride, interpolation=InterpolationMode.BICUBIC, antialias=True):
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = _apply_scale(width, height, scale, self.stride)
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = _apply_scale(new_width, new_height, scale, self.stride)

        if (new_width, new_height) == (width, height):
            return img

        if isinstance(img, torch.Tensor) and img.dtype == torch.uint8:
            resized = TF.resize(img.float(), (new_height, new_width), self.interpolation, antialias=self.antialias)
            return resized.clamp_(0, 255).round_().to(torch.uint8)
        return TF.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)


class VAEVideoTransform:
    """Resize -> ToTensor -> Normalize to [-1, 1] for VAE input."""

    def __init__(self, max_image_size, min_image_size=1, image_stride=16,
                 image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5)):
        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size, min_size=min_image_size, stride=image_stride
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=list(image_mean), std=list(image_std), inplace=True)

    def __call__(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert("RGB")
        else:
            img = img.convert("RGB")
        img = self.resize_transform(img)
        img = self.to_tensor_transform(img)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return self.normalize_transform(img)


def smart_video_nframes(
    total_frames: int,
    video_fps: Union[int, float],
    fps: int = 2.0,
    frame_factor: int = None,
    min_frames: int = None,
    max_frames: int = None,
    add_one: bool = False,
) -> List[int]:
    """Pick frame indices so the sampled clip matches the target `fps`/frame count."""
    nframes = total_frames / video_fps * fps
    if frame_factor is not None:
        nframes = math.floor(nframes / frame_factor) * frame_factor + int(add_one)
        nframes = max(nframes, frame_factor + int(add_one))
        if video_fps == fps:
            total_frames = math.floor(total_frames / frame_factor) * frame_factor + int(add_one)
    else:
        nframes = int(nframes + int(add_one))

    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()

    if min_frames is not None:
        if frame_factor is not None:
            min_frames = math.ceil(min_frames / frame_factor) * frame_factor
        nframes = max(min_frames + int(add_one), nframes)
    while len(idx) < int(nframes):
        idx.append(idx[-1])

    if max_frames is not None:
        if frame_factor is not None:
            max_frames = math.floor(max_frames / frame_factor) * frame_factor
        nframes = min(max_frames + int(add_one), nframes)
    if len(idx) > int(nframes):
        idx = idx[: int(nframes)]
    return idx


class VideoFrameReader:
    """Read RGB frames from a local video file by index (decord, cv2 fallback)."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        if _HAS_DECORD:
            self.vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0), fault_tol=1)
            self.vr.seek(0)
            self.fps = self.vr.get_avg_fps()
            self.length = len(self.vr)
        else:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"cannot open video: {video_path}")
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
            self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # decode the whole clip once; index-based access afterwards
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            self._frames = frames
            self.length = len(frames)
            logger.warning("decord not installed; reading %s with OpenCV (colors may differ slightly)", video_path)

    def sample(self, frame_indices: List[int]) -> List[Image.Image]:
        indices = [max(0, min(int(i), self.length - 1)) for i in frame_indices]
        if _HAS_DECORD:
            frames = self.vr.get_batch(indices).asnumpy()
            return [Image.fromarray(f).convert("RGB") for f in frames]
        return [Image.fromarray(self._frames[i]).convert("RGB") for i in indices]


def preprocess_video(
    video_path, fps=16, max_image_size=848, min_image_size=1, max_image_num=81,
    torch_dtype=torch.float32, device="cpu",
) -> torch.Tensor:
    """Read a video and return a normalized tensor `[1, C, T, H, W]`."""
    transform = VAEVideoTransform(max_image_size=max_image_size, min_image_size=min_image_size)
    reader = VideoFrameReader(video_path)
    idx = smart_video_nframes(
        total_frames=reader.length, video_fps=reader.fps, fps=fps,
        frame_factor=4, max_frames=max_image_num, add_one=True,
    )
    frames = reader.sample(idx)
    video_tensor = torch.stack([transform(f) for f in frames], dim=1).unsqueeze(0)
    return video_tensor.to(dtype=torch_dtype, device=device)


def preprocess_image(image, max_image_size=848, min_image_size=1,
                     torch_dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Return a normalized tensor `[1, C, 1, H, W]` for a single image."""
    transform = VAEVideoTransform(max_image_size=max_image_size, min_image_size=min_image_size)
    img = Image.open(image).convert("RGB") if isinstance(image, str) else image
    img_tensor = transform(img).unsqueeze(0).unsqueeze(2)
    return img_tensor.to(dtype=torch_dtype, device=device)


# --------------------------------------------------------------------------- #
# Guidance (bernini/models/wan_diffusion.py)
# --------------------------------------------------------------------------- #
class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        self.running_average = update_value + self.momentum * self.running_average


def _normalize_diff(diff, base_pred, momentum_buffer, eta, norm_threshold):
    """Project `diff` onto / off `base_pred` and recombine with weight `eta`.

    Operates on spatial `[B, C, T, H, W]` tensors; the norm is taken over
    (C, H, W) per frame — dims [-1, -2, -4], exactly as in the official code."""
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -4], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    v0, v1 = diff.double(), base_pred.double()
    v1 = F.normalize(v1, dim=[-1, -2, -4])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -4], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    diff_parallel, diff_orthogonal = v0_parallel.to(diff.dtype), v0_orthogonal.to(diff.dtype)
    return diff_orthogonal + eta * diff_parallel


def normalized_guidance(pred_cond, pred_uncond, guidance_scale, momentum_buffer=None, eta=1.0, norm_threshold=0.0):
    """Single-condition APG."""
    nd = _normalize_diff(pred_cond - pred_uncond, pred_cond, momentum_buffer, eta, norm_threshold)
    return pred_uncond + guidance_scale * nd


def normalized_guidance_chain(pred_uncond, preds, scales, momentum_buffers, eta, norm_thresholds):
    """Chained APG: each condition's diff is taken against the previous one."""
    bases = [pred_uncond] + list(preds)
    result = pred_uncond
    for i, cond in enumerate(preds):
        nd = _normalize_diff(cond - bases[i], cond, momentum_buffers[i], eta, norm_thresholds[i])
        result = result + scales[i] * nd
    return result


def make_source_ids(n: int, interpolate: bool = True, max_trained: int = 5) -> List[float]:
    """Source-id assignment for conditioning segments (target keeps 0)."""
    if n <= 0:
        return []
    if interpolate and n > max_trained:
        return torch.linspace(1.0, float(max_trained), n).tolist()
    return [float(i) for i in range(1, n + 1)]


# --------------------------------------------------------------------------- #
# Schedulers
# --------------------------------------------------------------------------- #
class FlowMatchScheduler:
    """bernini/models/scheduler.py — used with --sample_solver vanilla (no-UniPC parity)."""

    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, shift=None, device=None,
                      dtype=torch.bfloat16):
        if shift is not None:
            self.shift = shift
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1, device=device, dtype=dtype)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps, device=device, dtype=dtype)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_ - sigma)


def make_unipc_scheduler(flow_shift: float):
    """diffusers UniPCMultistepScheduler with the Wan2.2 flow-matching config,
    as loaded by GEN_Wanx22 (UniPCMultistepScheduler.from_pretrained(..., flow_shift=shift))."""
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    return UniPCMultistepScheduler(
        num_train_timesteps=1000,
        solver_order=2,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=flow_shift,
    )


def resolve_bernini_defaults(args) -> None:
    """Fill unset generation args with the official per-task defaults (in place)."""
    task = args.bernini_task
    merged = dict(BASE_DEFAULTS)
    merged.update(TASK_DEFAULTS.get(task, {}))

    if args.guidance_mode is None:
        if task == "i2v" and not getattr(args, "start_image_as_ref", True):
            # injection-only i2v: no reference segment, so image guidance has nothing to act on
            args.guidance_mode = "t2v_apg"
        else:
            args.guidance_mode = GUIDANCE_MODE_BY_TASK[task]
    if args.infer_steps is None:
        args.infer_steps = merged["infer_steps"]
    if args.flow_shift is None:
        args.flow_shift = merged["flow_shift"]
    if args.video_length is None:
        args.video_length = merged["video_length"]
    for name in ("omega_vid", "omega_img", "omega_txt", "omega_scale", "eta", "momentum"):
        if getattr(args, name) is None:
            setattr(args, name, merged[name])
    if args.max_image_size is None:
        args.max_image_size = merged["max_image_size"]
    if args.negative_prompt is None:
        args.negative_prompt = DEFAULT_NEG_PROMPT
    if args.bernini_system_prompt is None:
        args.bernini_system_prompt = get_system_prompt_for_task(task)
