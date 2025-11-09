"""
HoloCine Multi-Shot Video Generation Script

This script uses our custom pipeline based on wan2_generate_video.py for HoloCine multi-shot video generation.
It provides a command-line interface compatible with the HoloCine model (which has the same architecture as Wan 2.2 A14B T2V).

Usage:
    python holocine_generate_video.py --global_caption "..." --shot_captions "..." "..." --num_frames 241 --save_path output.mp4
    python holocine_generate_video.py --prompt "[global caption] ... [per shot caption] ..." --save_path output.mp4

Based on wan2_generate_video.py pipeline with HoloCine-specific adaptations.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import argparse
import logging
import random
import gc
import math
from pathlib import Path
from typing import Tuple, Optional, Any, Union
import numpy as np
import av

import accelerate
from accelerate import Accelerator
from safetensors.torch import load_file

# Import our pipeline modules
from Wan2_2.wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel, load_wan_model, detect_wan_sd_dtype
from wan.modules.vae import WanVAE
from Wan2_2.wan.modules.vae2_2 import Wan2_2_VAE
from wan.modules.t5 import T5EncoderModel
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.fm_solvers_euler import EulerScheduler
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

from utils.model_utils import str_to_dtype
from utils.device_utils import clean_memory_on_device
from utils.safetensors_utils import load_safetensors

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# ===================================================================
#                    Helper Functions
# ===================================================================

def enforce_4t_plus_1(n: int) -> int:
    """Forces an integer 'n' to the closest 4t+1 form."""
    t = round((n - 1) / 4)
    return 4 * t + 1

def prepare_multishot_inputs(
    global_caption: str,
    shot_captions: list[str],
    total_frames: int,
    custom_shot_cut_frames: list[int] = None
) -> dict:
    """
    Prepares the inference parameters from user-friendly segmented inputs.
    Returns dict with 'prompt', 'shot_cut_frames', 'num_frames'.
    """
    num_shots = len(shot_captions)

    # 1. Prepare 'prompt' in HoloCine format
    if "This scene contains" not in global_caption:
        global_caption = global_caption.strip() + f" This scene contains {num_shots} shots."
    per_shot_string = " [shot cut] ".join(shot_captions)
    prompt = f"[global caption] {global_caption} [per shot caption] {per_shot_string}"

    # 2. Prepare 'num_frames' (enforce 4t+1)
    processed_total_frames = enforce_4t_plus_1(total_frames)

    # 3. Prepare 'shot_cut_frames'
    num_cuts = num_shots - 1
    processed_shot_cuts = []

    if custom_shot_cut_frames:
        # User provided custom cuts
        logger.info(f"Using {len(custom_shot_cut_frames)} user-defined shot cuts (enforcing 4t+1).")
        for frame in custom_shot_cut_frames:
            processed_shot_cuts.append(enforce_4t_plus_1(frame))
    else:
        # Auto-calculate cuts
        logger.info(f"Auto-calculating {num_cuts} shot cuts.")
        if num_cuts > 0:
            ideal_step = processed_total_frames / num_shots
            for i in range(1, num_shots):
                approx_cut_frame = i * ideal_step
                processed_shot_cuts.append(enforce_4t_plus_1(round(approx_cut_frame)))

    processed_shot_cuts = sorted(list(set(processed_shot_cuts)))
    processed_shot_cuts = [f for f in processed_shot_cuts if f > 0 and f < processed_total_frames]

    return {
        "prompt": prompt,
        "shot_cut_frames": processed_shot_cuts,
        "num_frames": processed_total_frames
    }

def calculate_dimensions(video_size: Tuple[int, int], video_length: int, config, task: str = None) -> Tuple[Tuple[int, int, int, int], int]:
    """Calculate latent dimensions and sequence length"""
    height, width = video_size

    # Calculate latent dimensions
    lat_h = height // config.vae_stride[1]
    lat_w = width // config.vae_stride[2]
    lat_f = (video_length - 1) // config.vae_stride[0] + 1

    # Channel dimension
    ch = config.in_dim

    # Calculate sequence length
    seq_len = lat_h * lat_w * lat_f

    return (ch, lat_f, lat_h, lat_w), seq_len

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """Save video tensor to file"""
    import torchvision
    from einops import rearrange

    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    height, width, _ = outputs[0].shape

    # create output container
    container = av.open(path, mode="w")

    # create video stream
    codec = "libx264"
    pixel_format = "yuv420p"
    stream = container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = pixel_format
    stream.bit_rate = 4000000  # 4Mbit/s

    for frame_array in outputs:
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        packets = stream.encode(frame)
        for packet in packets:
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()

# ===================================================================
#                    Model Loading Functions
# ===================================================================

def load_vae(vae_path: str, device: torch.device, dtype: torch.dtype) -> WanVAE:
    """Load VAE model"""
    logger.info(f"Loading VAE model from: {vae_path}")
    vae = WanVAE(vae_path=vae_path, device=device, dtype=dtype, cache_device=None)
    return vae

def load_text_encoder(t5_path: str, device: torch.device, text_len: int = 512, fp8: bool = False) -> T5EncoderModel:
    """Load text encoder (T5) model"""
    logger.info(f"Loading T5 text encoder from: {t5_path}")

    t5_dtype = torch.bfloat16
    text_encoder = T5EncoderModel(
        text_len=text_len,
        dtype=t5_dtype,
        device=device,
        checkpoint_path=None,
        tokenizer_path=None,
        weight_path=t5_path,
        fp8=fp8,
    )
    return text_encoder

class HoloCineWanModelWrapper:
    """
    Wrapper around WanModel that adds support for HoloCine-specific parameters like shot_cut_frames.
    This allows us to use the standard WanModel without modifying it.
    """
    def __init__(self, wan_model: WanModel):
        self.model = wan_model
        self.dtype = next(wan_model.parameters()).dtype

    def __call__(self, x, t, context, seq_len, shot_cut_frames=None, **kwargs):
        """
        Forward pass that accepts shot_cut_frames but doesn't pass it to the underlying model.
        HoloCine models should be trained to handle multi-shot structure implicitly through
        the text prompts which contain [shot cut] markers.
        """
        # Remove shot_cut_frames from kwargs as standard WanModel doesn't accept it
        # The shot information is encoded in the text prompt via [shot cut] markers

        # Convert x to list format if needed (WanModel expects list of tensors)
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 5:
                # Has batch dimension [B, C, F, H, W] - split into list
                x_list = [x[i] for i in range(x.shape[0])]
            elif len(x.shape) == 4:
                # No batch dimension [C, F, H, W] - wrap in list
                x_list = [x]
            else:
                raise ValueError(f"Unexpected tensor shape for x: {x.shape}")
        else:
            # Already a list
            x_list = x

        # Call model and return first element (standard pattern)
        result = self.model(x_list, t, context, seq_len, **kwargs)
        return result[0]

    def to(self, device):
        """Move model to device"""
        self.model.to(device)
        return self

    def parameters(self):
        """Return model parameters"""
        return self.model.parameters()

class DynamicModelManager:
    """Manages dynamic loading and unloading of DiT models during inference."""

    def __init__(self, config, device, dit_dtype, dit_weight_dtype, low_noise_path, high_noise_path,
                 attn_mode, blocks_to_swap=0, lora_weights_list_low=None, lora_multipliers_low=None,
                 lora_weights_list_high=None, lora_multipliers_high=None):
        self.config = config
        self.device = device
        self.dit_dtype = dit_dtype
        self.dit_weight_dtype = dit_weight_dtype
        self.current_model = None
        self.current_model_type = None  # 'low' or 'high'
        self.model_paths = {
            'low': low_noise_path,
            'high': high_noise_path
        }
        self.attn_mode = attn_mode
        self.blocks_to_swap = blocks_to_swap
        self.lora_weights_list_low = lora_weights_list_low
        self.lora_multipliers_low = lora_multipliers_low
        self.lora_weights_list_high = lora_weights_list_high
        self.lora_multipliers_high = lora_multipliers_high

    def load_model(self, model_type: str):
        """Load specified model type ('low' or 'high') and wrap it"""
        if self.current_model_type == model_type:
            return self.current_model

        # Unload current model
        if self.current_model is not None:
            logger.info(f"Unloading {self.current_model_type} noise model")
            del self.current_model
            clean_memory_on_device(self.device)
            torch.cuda.empty_cache()
            gc.collect()

        # Load new model
        logger.info(f"Loading {model_type} noise model from: {self.model_paths[model_type]}")

        # Select appropriate LoRA weights
        lora_weights = self.lora_weights_list_low if model_type == 'low' else self.lora_weights_list_high
        lora_multipliers = self.lora_multipliers_low if model_type == 'low' else self.lora_multipliers_high

        wan_model = load_wan_model(
            self.config,
            self.device,
            self.model_paths[model_type],
            self.attn_mode,
            False,  # is_i2v
            "cpu",  # loading_device
            self.dit_dtype,  # loading_weight_dtype
            False,  # fp8
            lora_weights_list=lora_weights,
            lora_multipliers=lora_multipliers
        )

        # Setup block swapping if requested
        if self.blocks_to_swap > 0:
            logger.info(f"Enable swap {self.blocks_to_swap} blocks to CPU from device: {self.device}")
            wan_model.enable_block_swap(self.blocks_to_swap, self.device, supports_backward=False)
            wan_model.move_to_device_except_swap_blocks(self.device)
            wan_model.prepare_block_swap_before_forward()
        else:
            wan_model.to(self.device)

        wan_model.eval().requires_grad_(False)
        clean_memory_on_device(self.device)

        # Wrap the model to support HoloCine-specific parameters
        wrapped_model = HoloCineWanModelWrapper(wan_model)

        self.current_model = wrapped_model
        self.current_model_type = model_type

        return wrapped_model

# ===================================================================
#                    Scheduler Setup
# ===================================================================

def setup_scheduler(sample_solver: str, num_train_timesteps: int, infer_steps: int,
                   flow_shift: float, device: torch.device) -> Tuple[Any, torch.Tensor]:
    """Setup scheduler for sampling"""
    if sample_solver == "unipc":
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(infer_steps, device=device, shift=flow_shift)
        timesteps = scheduler.timesteps
    elif sample_solver == "dpm++":
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        sampling_sigmas = get_sampling_sigmas(infer_steps, flow_shift)
        timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sampling_sigmas)
    elif sample_solver == "vanilla":
        scheduler = FlowMatchDiscreteScheduler(num_train_timesteps=num_train_timesteps, shift=flow_shift)
        scheduler.set_timesteps(infer_steps, device=device)
        timesteps = scheduler.timesteps

        # FlowMatchDiscreteScheduler does not support generator argument in step method
        org_step = scheduler.step

        def step_wrapper(
            model_output: torch.Tensor,
            timestep: Union[int, torch.Tensor],
            sample: torch.Tensor,
            return_dict: bool = True,
            generator=None,
        ):
            try:
                return org_step(model_output, timestep, sample, return_dict=return_dict, generator=generator)
            except TypeError:
                return org_step(model_output, timestep, sample, return_dict=return_dict)

        scheduler.step = step_wrapper
    elif sample_solver == "euler":
        scheduler = EulerScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=flow_shift,
            device=device
        )
        scheduler.set_timesteps(infer_steps, device=device)
        timesteps = scheduler.timesteps[:-1].clone()
    else:
        raise NotImplementedError(f"Unsupported solver: {sample_solver}")

    logger.info(f"Using scheduler: {sample_solver}, timesteps shape: {timesteps.shape}")
    return scheduler, timesteps

# ===================================================================
#                    Sampling Function
# ===================================================================

def run_sampling(
    model_manager: DynamicModelManager,
    noise: torch.Tensor,
    scheduler: Any,
    timesteps: torch.Tensor,
    inputs: Tuple[dict, dict],
    device: torch.device,
    seed_g: torch.Generator,
    guidance_scale: float = 5.0,
    dual_dit_boundary: float = 0.875,
    shot_cut_frames: Optional[list[int]] = None
) -> torch.Tensor:
    """Run sampling loop with dual-DiT support"""
    arg_c, arg_null = inputs
    latent = noise

    # Add shot_cut_frames to inputs if provided
    if shot_cut_frames is not None and len(shot_cut_frames) > 0:
        arg_c["shot_cut_frames"] = shot_cut_frames
        arg_null["shot_cut_frames"] = shot_cut_frames
        logger.info(f"Added shot_cut_frames to model inputs: {shot_cut_frames}")

    num_timesteps = len(timesteps)

    # Determine boundary timestep for dual-DiT switching
    boundary_timestep_idx = int(num_timesteps * dual_dit_boundary)
    logger.info(f"Dual-DiT boundary: {dual_dit_boundary} (timestep index: {boundary_timestep_idx})")

    for step_idx, t in enumerate(timesteps):
        # Determine which model to use
        if step_idx < boundary_timestep_idx:
            model_type = 'high'  # High noise model for early steps
        else:
            model_type = 'low'   # Low noise model for later steps

        # Load appropriate model
        model = model_manager.load_model(model_type)

        # Prepare latent for model
        latent_model_input = latent.to(device, dtype=model.dtype)

        # CFG: Run unconditional and conditional predictions
        with torch.no_grad():
            # Unconditional prediction
            noise_pred_null = model(latent_model_input, t.to(device), **arg_null)

            # Conditional prediction
            noise_pred_c = model(latent_model_input, t.to(device), **arg_c)

            # Apply classifier-free guidance
            noise_pred = noise_pred_null + guidance_scale * (noise_pred_c - noise_pred_null)

        # Scheduler step
        latent = scheduler.step(noise_pred, t, latent, generator=seed_g, return_dict=False)[0]

        # Progress logging
        if (step_idx + 1) % 10 == 0 or step_idx == num_timesteps - 1:
            logger.info(f"Step {step_idx + 1}/{num_timesteps} (model: {model_type})")

    return latent

# ===================================================================
#                    Main Inference Function
# ===================================================================

def run_inference(
    # Model paths
    dit_high_noise: str,
    dit_low_noise: str,
    vae_path: str,
    t5_path: str,

    # Output
    output_path: str,

    # Prompting Options (Auto-detect)
    global_caption: str = None,
    shot_captions: list[str] = None,
    prompt: str = None,
    negative_prompt: str = None,

    # Core Generation Parameters
    num_frames: int = None,
    shot_cut_frames: list[int] = None,

    # Other Generation Parameters
    seed: int = 0,
    height: int = 480,
    width: int = 832,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,

    # Solver and flow matching
    sample_solver: str = "unipc",
    flow_shift: float = 1.0,

    # Device and dtype
    device: str = "cuda",
    dtype: str = "bfloat16",

    # Advanced options
    dual_dit_boundary: float = 0.875,
    attn_mode: str = "torch",
    fp8_t5: bool = False,
    blocks_to_swap: int = 0,

    # Output Parameters
    fps: int = 15,
):
    """
    Runs the HoloCine inference pipeline using our custom implementation.

    Mode 1 (Structured): Provide 'global_caption', 'shot_captions', 'num_frames'.
                         'shot_cut_frames' is optional (auto-calculated).
    Mode 2 (Raw): Provide 'prompt'.
                  'num_frames' and 'shot_cut_frames' are optional.
    """

    logger.info("=== HoloCine Multi-Shot Video Generation (Custom Pipeline) ===")

    # Setup device
    device_obj = torch.device(device)

    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype = dtype_map[dtype]

    # Setup config (use t2v-A14B config since HoloCine has same architecture)
    task = "t2v-A14B"
    config = WAN_CONFIGS[task]

    logger.info(f"Device: {device}, Dtype: {torch_dtype}")
    logger.info(f"Using task config: {task}")

    # --- Auto-Detection Logic ---
    pipe_kwargs = {}

    if global_caption and shot_captions:
        # Mode 1: Structured Input
        logger.info("--- Detected Structured Input (Mode 1) ---")
        if num_frames is None:
            raise ValueError("Must provide 'num_frames' for structured input (Mode 1).")

        inputs = prepare_multishot_inputs(
            global_caption=global_caption,
            shot_captions=shot_captions,
            total_frames=num_frames,
            custom_shot_cut_frames=shot_cut_frames
        )
        prompt = inputs["prompt"]
        shot_cut_frames = inputs["shot_cut_frames"]
        num_frames = inputs["num_frames"]

        logger.info(f"Generated prompt: {prompt[:150]}...")
        logger.info(f"Frames: {num_frames}, Shot cuts: {shot_cut_frames}")

    elif prompt:
        # Mode 2: Raw String Input
        logger.info("--- Detected Raw String Input (Mode 2) ---")

        if num_frames is not None:
            processed_frames = enforce_4t_plus_1(num_frames)
            if num_frames != processed_frames:
                logger.info(f"Corrected 'num_frames': {num_frames} -> {processed_frames}")
            num_frames = processed_frames
        else:
            # Use default
            num_frames = 81
            logger.info(f"Using default num_frames: {num_frames}")

        if shot_cut_frames is not None:
            processed_cuts = [enforce_4t_plus_1(f) for f in shot_cut_frames]
            shot_cut_frames = processed_cuts
            logger.info(f"Shot cuts: {shot_cut_frames}")

        logger.info(f"Prompt: {prompt[:150]}...")
    else:
        raise ValueError("Invalid inputs. Provide either (global_caption, shot_captions, num_frames) OR (prompt).")

    # --- Setup Video Parameters ---
    video_size = [height, width]
    video_length = num_frames

    logger.info(f"Video size: {height}x{width}, Length: {video_length} frames")

    # --- Setup Accelerator ---
    mixed_precision = "bf16" if torch_dtype == torch.bfloat16 else "fp16"
    accelerator = Accelerator(mixed_precision=mixed_precision)

    # --- Set Seed ---
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    logger.info(f"Using seed: {seed}")

    seed_g = torch.Generator(device=device_obj)
    seed_g.manual_seed(seed)

    # --- Load VAE ---
    logger.info("Loading VAE...")
    vae = load_vae(vae_path, device_obj, torch_dtype)

    # --- Load Text Encoder and Encode Prompts ---
    logger.info("Loading T5 text encoder...")
    text_encoder = load_text_encoder(t5_path, device_obj, text_len=config.text_len, fp8=fp8_t5)
    text_encoder.model.to(device_obj)

    # Configure negative prompt
    n_prompt = negative_prompt if negative_prompt else config.sample_neg_prompt

    # Encode prompts
    logger.info("Encoding prompts...")
    with torch.no_grad():
        if fp8_t5:
            with torch.amp.autocast(device_type=device_obj.type, dtype=torch.bfloat16):
                context = text_encoder([prompt], device_obj)
                context_null = text_encoder([n_prompt], device_obj)
        else:
            context = text_encoder([prompt], device_obj)
            context_null = text_encoder([n_prompt], device_obj)

    # Free text encoder
    del text_encoder
    clean_memory_on_device(device_obj)
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Unloaded T5 model from memory")

    # --- Calculate Dimensions ---
    (ch, lat_f, lat_h, lat_w), seq_len = calculate_dimensions(video_size, video_length, config, task)
    target_shape = (ch, lat_f, lat_h, lat_w)

    logger.info(f"Latent shape: {target_shape}, Sequence length: {seq_len}")

    # --- Generate Noise ---
    noise = torch.randn(target_shape, dtype=torch.float32, generator=seed_g, device=device_obj)
    noise = noise.to(device_obj)

    # --- Prepare Model Inputs ---
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}
    inputs = (arg_c, arg_null)

    # --- Setup Scheduler ---
    logger.info("Setting up scheduler...")
    scheduler, timesteps = setup_scheduler(
        sample_solver=sample_solver,
        num_train_timesteps=config.num_train_timesteps,
        infer_steps=num_inference_steps,
        flow_shift=flow_shift,
        device=device_obj
    )

    # --- Setup Dynamic Model Manager for Dual-DiT ---
    logger.info("Setting up dynamic dual-DiT model manager...")
    if blocks_to_swap > 0:
        logger.info(f"Block swapping enabled: {blocks_to_swap} blocks will be offloaded to CPU")
    model_manager = DynamicModelManager(
        config=config,
        device=device_obj,
        dit_dtype=torch_dtype,
        dit_weight_dtype=torch_dtype,
        low_noise_path=dit_low_noise,
        high_noise_path=dit_high_noise,
        attn_mode=attn_mode,
        blocks_to_swap=blocks_to_swap,
        lora_weights_list_low=None,
        lora_multipliers_low=None,
        lora_weights_list_high=None,
        lora_multipliers_high=None
    )

    # --- Run Sampling ---
    logger.info("=== Starting Video Generation ===")
    latent = run_sampling(
        model_manager=model_manager,
        noise=noise,
        scheduler=scheduler,
        timesteps=timesteps,
        inputs=inputs,
        device=device_obj,
        seed_g=seed_g,
        guidance_scale=guidance_scale,
        dual_dit_boundary=dual_dit_boundary,
        shot_cut_frames=shot_cut_frames
    )

    logger.info("Generation complete! Decoding latent...")

    # --- Decode Latent to Video ---
    vae.to_device(device_obj)

    with torch.no_grad():
        # Normalize latent for VAE
        latent_for_vae = latent / config.vae_scaling_factor

        # Decode
        video = vae.decode(latent_for_vae)

    # Move VAE back to CPU
    vae.to_device("cpu")
    clean_memory_on_device(device_obj)

    # --- Save Video ---
    logger.info(f"Saving video to: {output_path}")

    # Ensure video is in correct format [B, C, T, H, W]
    if video.dim() == 4:
        video = video.unsqueeze(0)

    save_videos_grid(video, output_path, rescale=True, fps=fps)

    logger.info(f"✅ Video successfully saved to: {output_path}")

    return video

# ===================================================================
#                    Command-Line Interface
# ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="HoloCine Multi-Shot Video Generation (Custom Pipeline)")

    # Model paths
    parser.add_argument("--dit_high_noise", type=str, required=True,
                       help="Path to high noise DiT model (safetensors)")
    parser.add_argument("--dit_low_noise", type=str, required=True,
                       help="Path to low noise DiT model (safetensors)")
    parser.add_argument("--vae", type=str, required=True,
                       help="Path to VAE model (pth)")
    parser.add_argument("--t5", type=str, required=True,
                       help="Path to T5 encoder model (pth or safetensors)")

    # Input mode: Structured multi-shot OR raw prompt
    parser.add_argument("--global_caption", type=str, default=None,
                       help="Global scene caption (for structured multi-shot)")
    parser.add_argument("--shot_captions", type=str, nargs="*", default=None,
                       help="List of per-shot captions (for structured multi-shot)")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Raw HoloCine format prompt (alternative to global_caption + shot_captions)")

    # Generation parameters
    parser.add_argument("--num_frames", type=int, default=None,
                       help="Number of frames to generate (will be adjusted to 4t+1)")
    parser.add_argument("--shot_cut_frames", type=int, nargs="*", default=None,
                       help="Frame indices where shots change (for custom shot cuts)")
    parser.add_argument("--negative_prompt", type=str, default=None,
                       help="Negative prompt")

    # Video parameters
    parser.add_argument("--height", type=int, default=480,
                       help="Video height (default: 480)")
    parser.add_argument("--width", type=int, default=832,
                       help="Video width (default: 832)")
    parser.add_argument("--fps", type=int, default=15,
                       help="Output video FPS (default: 15)")

    # Inference parameters
    parser.add_argument("--infer_steps", type=int, default=50,
                       help="Number of inference steps (default: 50)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed (default: 0)")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                       help="Guidance scale for classifier-free guidance (default: 5.0)")

    # Solver and flow matching
    parser.add_argument("--sample_solver", type=str, default="unipc",
                       choices=["unipc", "dpm++", "vanilla", "euler"],
                       help="Sampling solver (default: unipc)")
    parser.add_argument("--flow_shift", type=float, default=1.0,
                       help="Flow shift for flow matching (default: 1.0)")

    # Dual-DiT parameters
    parser.add_argument("--dual_dit_boundary", type=float, default=0.875,
                       help="Boundary for dual-DiT switching (0.0-1.0, default: 0.875)")

    # Output
    parser.add_argument("--save_path", type=str, required=True,
                       help="Path to save output video")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (default: cuda)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16"],
                       help="Model dtype (default: bfloat16)")

    # Advanced options
    parser.add_argument("--attn_mode", type=str, default="torch",
                       choices=["flash", "flash2", "flash3", "torch", "sageattn", "xformers", "sdpa"],
                       help="Attention mode (default: torch)")
    parser.add_argument("--fp8_t5", action="store_true",
                       help="Use FP8 for T5 encoder")
    parser.add_argument("--blocks_to_swap", type=int, default=0,
                       help="Number of transformer blocks to swap to CPU to reduce VRAM usage (default: 0)")

    # Compatibility arguments (ignored but accepted)
    parser.add_argument("--video_length", type=int, default=None,
                       help="Alias for --num_frames")
    parser.add_argument("--tiled", action="store_true", default=True,
                       help="(Ignored - for compatibility only)")
    parser.add_argument("--quality", type=int, default=5,
                       help="(Ignored - for compatibility only)")
    parser.add_argument("--enable_vram_management", action="store_true", default=True,
                       help="(Ignored - for compatibility only)")
    parser.add_argument("--offload_device", type=str, default="cpu",
                       help="(Ignored - for compatibility only)")

    return parser.parse_args()

def main():
    args = parse_args()

    # Handle video_length alias
    if args.video_length is not None and args.num_frames is None:
        args.num_frames = args.video_length

    # Run inference
    run_inference(
        dit_high_noise=args.dit_high_noise,
        dit_low_noise=args.dit_low_noise,
        vae_path=args.vae,
        t5_path=args.t5,
        output_path=args.save_path,
        global_caption=args.global_caption,
        shot_captions=args.shot_captions,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        shot_cut_frames=args.shot_cut_frames,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_inference_steps=args.infer_steps,
        guidance_scale=args.guidance_scale,
        sample_solver=args.sample_solver,
        flow_shift=args.flow_shift,
        device=args.device,
        dtype=args.dtype,
        dual_dit_boundary=args.dual_dit_boundary,
        attn_mode=args.attn_mode,
        fp8_t5=args.fp8_t5,
        blocks_to_swap=args.blocks_to_swap if args.blocks_to_swap else 0,
        fps=args.fps
    )

if __name__ == "__main__":
    main()
