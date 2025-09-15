# Fixed version of pusa_v2v.py with proper LoRA loading
# Based on wan22_14b_v2v_pusa_optimized_v3.py with corrected LoRA implementation

from PIL import Image
import torch
import os
import sys
import argparse
import datetime
import cv2
import gc
import random
import glob
import numpy as np
from tqdm import tqdm
from safetensors.torch import load_file
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- START: Ported components from wan2_generate_video.py ---

# Utility Functions
def clean_memory_on_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# Pusa Schedulers (essential for the V2V logic)
from wan.pusa.flow_match_pusa_v2v import FlowMatchSchedulerPusaV2V
from Wan2_2.wan.configs import WAN_CONFIGS

# Model Loading Components
from wan.modules.model import WanModel, load_wan_model, detect_wan_sd_dtype
from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel
from utils.safetensors_utils import load_safetensors

def apply_pusa_loras(
    model_state_dict: Dict[str, torch.Tensor],
    lora_weights_list: List[Dict[str, torch.Tensor]],
    lora_multipliers: List[float],
    device: torch.device = torch.device("cuda"),
    model_type: str = "unknown"
) -> Dict[str, torch.Tensor]:
    """
    Apply Pusa LoRA weights to model state dict using proper PEFT-style merging.

    This function handles LoRA weights in the format used by Pusa models:
    - Keys ending with .lora_A.weight (down projection)
    - Keys ending with .lora_B.weight (up projection)
    - Optional .alpha keys for scaling

    Args:
        model_state_dict: The base model's state dictionary
        lora_weights_list: List of LoRA weight dictionaries
        lora_multipliers: List of multipliers for each LoRA
        device: Device to perform calculations on
        model_type: Type of model (for logging)

    Returns:
        Modified state dict with LoRA weights merged
    """
    if not lora_weights_list or len(lora_weights_list) == 0:
        logger.info(f"No LoRA weights to apply for {model_type} model")
        return model_state_dict

    # Ensure multipliers match LoRA count
    if lora_multipliers is None:
        lora_multipliers = [1.0] * len(lora_weights_list)
    while len(lora_multipliers) < len(lora_weights_list):
        lora_multipliers.append(1.0)

    logger.info(f"Applying {len(lora_weights_list)} LoRA(s) to {model_type} model with multipliers: {lora_multipliers}")

    # Track applied weights
    total_updated = 0
    unused_lora_keys = set()

    # Process each LoRA
    for lora_idx, (lora_sd, multiplier) in enumerate(zip(lora_weights_list, lora_multipliers)):
        logger.info(f"Processing LoRA {lora_idx + 1}/{len(lora_weights_list)} with multiplier {multiplier}")

        # Build mapping of LoRA pairs
        lora_pairs = {}
        for key in lora_sd.keys():
            if ".lora_B.weight" in key or ".lora_B.default.weight" in key:
                # Found an up projection weight
                base_key = key.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")

                # Find corresponding down weight
                down_key_options = [
                    key.replace(".lora_B.", ".lora_A."),
                    base_key + ".lora_A.weight",
                    base_key + ".lora_A.default.weight"
                ]

                down_key = None
                for dk in down_key_options:
                    if dk in lora_sd:
                        down_key = dk
                        break

                if down_key:
                    # Map to model parameter name
                    # Remove diffusion_model prefix if present
                    target_key = base_key
                    if target_key.startswith("diffusion_model."):
                        target_key = target_key[len("diffusion_model."):]

                    # Remove .default if present
                    target_key = target_key.replace(".default", "")

                    # Add .weight suffix if not present
                    if not target_key.endswith(".weight"):
                        target_key = target_key + ".weight"

                    # Check for alpha
                    alpha_key = base_key + ".alpha"
                    if alpha_key not in lora_sd:
                        alpha_key = base_key + ".lora_alpha"

                    lora_pairs[target_key] = {
                        'up': key,
                        'down': down_key,
                        'alpha': alpha_key if alpha_key in lora_sd else None
                    }

        # Track unused keys
        used_keys = set()
        updated_count = 0

        # Apply LoRA weights to model parameters
        for param_name, param_value in model_state_dict.items():
            if param_name in lora_pairs:
                lora_info = lora_pairs[param_name]

                # Get LoRA weights
                up_weight = lora_sd[lora_info['up']]
                down_weight = lora_sd[lora_info['down']]

                # Get alpha value
                if lora_info['alpha'] and lora_info['alpha'] in lora_sd:
                    alpha = float(lora_sd[lora_info['alpha']])
                else:
                    # Default alpha is the rank
                    alpha = down_weight.shape[0]

                # Calculate scale
                rank = down_weight.shape[0]
                scale = (alpha / rank) * multiplier

                # Move to computation device
                param_compute = param_value.to(device, dtype=torch.float32)
                up_compute = up_weight.to(device, dtype=torch.float32)
                down_compute = down_weight.to(device, dtype=torch.float32)

                # Handle different tensor shapes
                if len(param_value.shape) == 2:
                    # Linear layer
                    if len(up_compute.shape) == 4:
                        # Conv weights being used as linear
                        up_compute = up_compute.squeeze(3).squeeze(2)
                        down_compute = down_compute.squeeze(3).squeeze(2)
                    delta = scale * torch.mm(up_compute, down_compute)
                elif len(param_value.shape) == 4:
                    # Convolutional layer
                    if down_compute.shape[2:4] == (1, 1):
                        # 1x1 convolution
                        delta = scale * (up_compute.squeeze(3).squeeze(2) @ down_compute.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    else:
                        # 3x3 convolution
                        delta = scale * torch.nn.functional.conv2d(
                            down_compute.permute(1, 0, 2, 3),
                            up_compute
                        ).permute(1, 0, 2, 3)
                else:
                    logger.warning(f"Unsupported parameter shape for {param_name}: {param_value.shape}")
                    continue

                # Apply the update
                param_merged = param_compute + delta
                model_state_dict[param_name] = param_merged.to(param_value.device, dtype=param_value.dtype)

                updated_count += 1
                used_keys.add(lora_info['up'])
                used_keys.add(lora_info['down'])
                if lora_info['alpha']:
                    used_keys.add(lora_info['alpha'])

        # Check for unused keys
        for key in lora_sd.keys():
            if key not in used_keys and key.endswith(('.weight', '.alpha')):
                unused_lora_keys.add(key)

        logger.info(f"LoRA {lora_idx + 1}: Updated {updated_count} parameters")
        total_updated += updated_count

    if unused_lora_keys:
        logger.warning(f"Unused LoRA keys ({len(unused_lora_keys)}): {list(unused_lora_keys)[:5]}...")

    logger.info(f"Total parameters updated with LoRA weights: {total_updated}")
    return model_state_dict


def load_wan_model_with_proper_loras(
    config: any,
    device: torch.device,
    dit_path: List[str],
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[List[float]] = None,
    dit_weight_dtype: torch.dtype = torch.bfloat16,
    model_type: str = "unknown"
) -> WanModel:
    """
    Load WAN model with proper LoRA application.

    This function loads the base model and applies LoRA weights correctly
    by merging them into the model state dict before loading.
    """
    # Load base model weights
    logger.info(f"Loading base {model_type} model from {dit_path}")
    loading_device = torch.device("cpu")

    # Load state dict - handle both single file and list of files
    if isinstance(dit_path, list) and len(dit_path) == 1:
        # Single file in a list
        state_dict = load_safetensors(dit_path[0], loading_device, disable_mmap=True, dtype=dit_weight_dtype)
    elif isinstance(dit_path, list):
        # Multiple files - need to merge them
        state_dict = {}
        for path in dit_path:
            partial_sd = load_safetensors(path, loading_device, disable_mmap=True, dtype=dit_weight_dtype)
            state_dict.update(partial_sd)
    else:
        # Single file path as string
        state_dict = load_safetensors(dit_path, loading_device, disable_mmap=True, dtype=dit_weight_dtype)

    # Remove model.diffusion_model prefix if present
    sd_keys = list(state_dict.keys())
    for key in sd_keys:
        if key.startswith("model.diffusion_model."):
            state_dict[key[22:]] = state_dict.pop(key)

    # Apply LoRA weights if provided
    if lora_weights_list and len(lora_weights_list) > 0:
        state_dict = apply_pusa_loras(
            state_dict,
            lora_weights_list,
            lora_multipliers,
            device=device,
            model_type=model_type
        )

    # Create model and load merged state dict
    # Use the standard load_wan_model but without LoRA parameters
    model = load_wan_model(
        config, device, dit_path,
        "torch", False, loading_device, dit_weight_dtype, False,
        lora_weights_list=None,  # Don't pass LoRAs here since we already merged them
        lora_multipliers=None
    )

    # Load our merged state dict
    model.load_state_dict(state_dict, strict=False)

    return model


class DynamicModelManager:
    """Manages dynamic loading and unloading of DiT models during inference."""
    def __init__(self, config, device, dit_dtype, args):
        self.config = config
        self.device = device
        self.dit_dtype = dit_dtype
        self.args = args
        self.current_model = None
        self.current_model_type = None  # 'low' or 'high'
        self.model_paths = {}
        self.lora_weights_list_low = None
        self.lora_multipliers_low = None
        self.lora_weights_list_high = None
        self.lora_multipliers_high = None

    def set_model_paths(self, low_path_list, high_path_list):
        self.model_paths['low'] = low_path_list
        self.model_paths['high'] = high_path_list

    def set_lora_weights(self, lora_weights_list_low, lora_multipliers_low,
                         lora_weights_list_high, lora_multipliers_high):
        self.lora_weights_list_low = lora_weights_list_low
        self.lora_multipliers_low = lora_multipliers_low
        self.lora_weights_list_high = lora_weights_list_high
        self.lora_multipliers_high = lora_multipliers_high

    def get_model(self, model_type: str) -> WanModel:
        if self.current_model_type == model_type:
            return self.current_model

        if self.current_model is not None:
            print(f"Unloading {self.current_model_type} noise model...")
            if hasattr(self.current_model, 'offloader') and self.current_model.offloader:
                if hasattr(self.current_model.offloader, 'thread_pool'):
                    self.current_model.offloader.thread_pool.shutdown(wait=True)
            del self.current_model
            self.current_model = None
            clean_memory_on_device(self.device)

        print(f"Loading {model_type} noise DiT model...")
        dit_weight_dtype = self.dit_dtype

        lora_weights_list = self.lora_weights_list_low if model_type == 'low' else self.lora_weights_list_high
        lora_multipliers = self.lora_multipliers_low if model_type == 'low' else self.lora_multipliers_high

        # Use the new loading function with proper LoRA support
        model = load_wan_model_with_proper_loras(
            self.config,
            self.device,
            self.model_paths[model_type],
            lora_weights_list=lora_weights_list,
            lora_multipliers=lora_multipliers,
            dit_weight_dtype=dit_weight_dtype,
            model_type=model_type
        )

        optimize_model(model, self.args, self.device, self.dit_dtype)
        self.current_model = model
        self.current_model_type = model_type
        return model

    def cleanup(self):
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            clean_memory_on_device(self.device)

def optimize_model(model: WanModel, args: argparse.Namespace, device: torch.device, dit_dtype: torch.dtype):
    # Match the robust logic from wan2_generate_video.py
    target_dtype = dit_dtype
    target_device = None

    if args.blocks_to_swap == 0:
        print(f"Move model to device: {device}")
        target_device = device

    # Move and cast at the same time for efficiency.
    model.to(target_device, target_dtype)

    if args.blocks_to_swap > 0:
        print(f"Enable swap {args.blocks_to_swap} blocks to CPU from device: {device}")
        model.enable_block_swap(args.blocks_to_swap, device, supports_backward=False)
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    else:
        # ensure all parameters are on the right device
        model.to(device)

    model.eval().requires_grad_(False)
    clean_memory_on_device(device)

def save_video(frames, path, fps=24, quality=5):
    import imageio
    writer = imageio.get_writer(path, fps=fps, quality=quality)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()

# --- END: Ported components ---

def process_video_frames(video_path, target_width=832, target_height=480):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_ratio = target_width / target_height
    original_ratio = width / height
    while True:
        ret, frame = cap.read()
        if not ret: break
        if original_ratio > target_ratio:
            new_width = int(height * target_ratio)
            start_x = (width - new_width) // 2
            frame = frame[:, start_x:start_x + new_width]
        else:
            new_height = int(width / target_ratio)
            start_y = (height - new_height) // 2
            frame = frame[start_y:start_y + new_height]
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames

def _get_model_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not files: raise FileNotFoundError(f"No .safetensors files in directory: {path}")
        return files
    else:
        raise FileNotFoundError(f"Model path not found: {path}")

def main():
    parser = argparse.ArgumentParser(description="Pusa V2V (Fixed): Video-to-Video with proper LoRA loading")
    parser.add_argument("--task", type=str, default="t2v-A14B", help="The task configuration to use from wan.configs.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the conditioning video.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--negative_prompt", type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", help="Negative text prompt.")
    parser.add_argument("--cond_position", type=str, default=None, help="Comma-separated list of frame indices for conditioning.")
    parser.add_argument("--extend_from_end", type=int, default=None, help="Number of frames from the end of `--video_path` to use for conditioning the start of the new video. Mutually exclusive with `--cond_position`.")
    parser.add_argument("--noise_multipliers", type=str, required=True, help="Comma-separated noise multipliers for conditioning frames.")
    parser.add_argument("--dit_high_noise_path", type=str, required=True, help="Path to the high noise DiT model (.safetensors file or directory).")
    parser.add_argument("--dit_low_noise_path", type=str, required=True, help="Path to the low noise DiT model (.safetensors file or directory).")
    parser.add_argument("--high_lora_path", type=str, required=True, help="Path(s) to Pusa LoRA for high noise model. Comma-separated.")
    parser.add_argument("--low_lora_path", type=str, required=True, help="Path(s) to Pusa LoRA for low noise model. Comma-separated.")
    parser.add_argument("--base_dir", type=str, default="wan", help="Directory of T5 and VAE models.")
    parser.add_argument("--high_lora_alpha", type=str, default="1.0", help="Alpha(s) for high noise LoRA. Comma-separated.")
    parser.add_argument("--low_lora_alpha", type=str, default="1.0", help="Alpha(s) for low noise LoRA. Comma-separated.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--switch_DiT_boundary", type=float, default=0.875, help="Boundary to switch DiT models.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the output video.")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale.")
    parser.add_argument("--width", type=int, default=832, help="Width of the output video.")
    parser.add_argument("--height", type=int, default=480, help="Height of the output video.")
    parser.add_argument("--fps", type=int, default=24, help="FPS for the saved video.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--blocks_to_swap", type=int, default=8, help="Number of DiT blocks to swap to CPU. Default: 8.")
    parser.add_argument("--concatenate", action="store_true", help="Automatically concatenate the original video with the generated video. Only works with `--extend_from_end`.")
    args = parser.parse_args()

    if args.extend_from_end is not None and args.cond_position is not None:
        raise ValueError("Cannot use both `--extend_from_end` and `--cond_position`. Please choose one.")
    if args.extend_from_end is None and args.cond_position is None:
        raise ValueError("Either `--extend_from_end` or `--cond_position` must be specified for conditioning.")
    if args.concatenate and args.extend_from_end is None:
        raise ValueError("`--concatenate` can only be used with `--extend_from_end` for video extension.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # --- Setup Models ---
    print("Loading models...")
    task_config = WAN_CONFIGS[args.task]
    high_model_files = _get_model_files(args.dit_high_noise_path)
    low_model_files = _get_model_files(args.dit_low_noise_path)

    vae = WanVAE(vae_path=os.path.join(args.base_dir, "Wan2.1_VAE.pth"), device=device, dtype=dtype, cache_device="cpu")
    t5 = T5EncoderModel(text_len=task_config.text_len, device=device, weight_path=os.path.join(args.base_dir, "models_t5_umt5-xxl-enc-bf16.pth"))

    model_manager = DynamicModelManager(task_config, device, dtype, args)
    model_manager.set_model_paths(low_model_files, high_model_files)

    # --- Load LoRAs ---
    def load_loras(lora_paths_str, lora_alphas_str):
        if not lora_paths_str or not lora_paths_str.strip():
            return None, None
        paths = [p.strip() for p in lora_paths_str.split(',')]
        alphas = [float(a.strip()) for a in lora_alphas_str.split(',')]
        if len(paths) != len(alphas):
            raise ValueError("Number of LoRA paths must match number of alphas.")

        logger.info(f"Loading {len(paths)} LoRA file(s)")
        weights = []
        for path in paths:
            logger.info(f"Loading LoRA from: {path}")
            lora_sd = load_file(path, device="cpu")
            weights.append(lora_sd)
        return weights, alphas

    low_loras, low_alphas = load_loras(args.low_lora_path, args.low_lora_alpha)
    high_loras, high_alphas = load_loras(args.high_lora_path, args.high_lora_alpha)
    model_manager.set_lora_weights(low_loras, low_alphas, high_loras, high_alphas)

    # --- Prepare Inputs ---
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed(seed)

    all_video_frames = process_video_frames(args.video_path, target_width=args.width, target_height=args.height)
    noise_mult_list = [float(x.strip()) for x in args.noise_multipliers.split(',')]

    if args.extend_from_end:
        print(f"Video extension mode: Using last {args.extend_from_end} frames for conditioning.")
        if args.extend_from_end > len(all_video_frames):
            raise ValueError(f"`--extend_from_end` ({args.extend_from_end}) is greater than video length ({len(all_video_frames)}).")
        if len(noise_mult_list) != args.extend_from_end:
            raise ValueError(f"Number of noise multipliers ({len(noise_mult_list)}) must match `--extend_from_end` ({args.extend_from_end}).")

        conditioning_video = all_video_frames[-args.extend_from_end:]
        cond_pos_list = list(range(args.extend_from_end))
    else:
        print(f"Standard conditioning mode: Using frames from specified positions.")
        cond_pos_list = [int(x.strip()) for x in args.cond_position.split(',')]
        if len(noise_mult_list) != len(cond_pos_list):
            raise ValueError(f"Number of noise multipliers ({len(noise_mult_list)}) must match conditioning positions ({len(cond_pos_list)}).")

        conditioning_video = [all_video_frames[i] for i in cond_pos_list if i < len(all_video_frames)]

    # Get latents for conditioning frames
    vae.to_device(device)
    cond_latents = {}
    with torch.no_grad():
        for i, source_frame in enumerate(conditioning_video):
            target_frame_idx = cond_pos_list[i]
            img_tensor = torch.from_numpy(np.array(source_frame)).permute(2, 0, 1).float() / 127.5 - 1.0
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(2).to(device, dtype) # B,C,F,H,W
            latent = vae.encode([img_tensor.squeeze(0)])[0] # C,F,H,W
            cond_latents[target_frame_idx] = latent.squeeze(1).cpu() # C,H,W
    vae.to_device("cpu")
    clean_memory_on_device(device)

    # Prepare context and initial noise
    t5.model.to(device)
    with torch.no_grad():
        context = t5([args.prompt], device)
        context_null = t5([args.negative_prompt], device)
    t5.model.cpu()
    clean_memory_on_device(device)

    num_frames = 81
    lat_f = (num_frames - 1) // task_config.vae_stride[0] + 1
    lat_h = args.height // task_config.vae_stride[1]
    lat_w = args.width // task_config.vae_stride[2]
    seq_len = int(np.ceil(lat_f * lat_h * lat_w / (task_config.patch_size[1] * task_config.patch_size[2])))
    noise = torch.randn(1, 16, lat_f, lat_h, lat_w, dtype=dtype, device="cpu")

    for frame_idx, latent in cond_latents.items():
        if frame_idx < lat_f:
            noise[:, :, frame_idx, :, :] = latent.unsqueeze(0).to(dtype)
        else:
            print(f"Warning: Conditioning frame index {frame_idx} is out of bounds for latent length {lat_f}. Skipping.")

    # --- Setup Pipeline and Run ---
    scheduler = FlowMatchSchedulerPusaV2V()
    scheduler.set_timesteps(num_inference_steps=args.num_inference_steps)
    timesteps = scheduler.timesteps.to(device)
    latent = noise.to(device)
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}
    noise_mapping = {pos: mult for pos, mult in zip(cond_pos_list, noise_mult_list)}

    print("Generating new video frames...")
    for i, t in enumerate(tqdm(timesteps)):
        boundary = args.switch_DiT_boundary * 1000
        model_type = 'high' if t.item() >= boundary else 'low'
        current_model = model_manager.get_model(model_type)

        timestep_2d = t.unsqueeze(0).unsqueeze(1).repeat(1, lat_f)
        for frame_idx in cond_pos_list:
            timestep_2d[:, frame_idx] = timestep_2d[:, frame_idx] * noise_mapping.get(frame_idx, 1.0)

        with torch.autocast(device_type=device.type, dtype=dtype), torch.no_grad():
            latent_model_input = [latent.squeeze(0)]
            timestep_1d = t.unsqueeze(0)

            pred_cond = current_model(latent_model_input, t=timestep_1d, **arg_c)[0]
            pred_uncond = current_model(latent_model_input, t=timestep_1d, **arg_null)[0]
            noise_pred = pred_uncond + args.cfg_scale * (pred_cond - pred_uncond)

        latent = scheduler.step(model_output=noise_pred,
                                timestep=timestep_2d,
                                sample=latent,
                                cond_frame_latent_indices=cond_pos_list,
                                noise_multipliers=noise_mapping)

    final_latent = latent.cpu()
    model_manager.cleanup()

    # --- Decode and Save ---
    print("Decoding final latent...")
    vae.to_device(device)
    with torch.no_grad():
        latent_to_decode = final_latent.to(device)
        decoded_frames_list = vae.decode([latent_to_decode.squeeze(0)])
    video_tensor = decoded_frames_list[0]
    video_tensor = (video_tensor.permute(1, 2, 3, 0) + 1) / 2
    video_tensor = (video_tensor.clamp(0, 1) * 255).byte().cpu().numpy()
    generated_frames_pil = [Image.fromarray(frame) for frame in video_tensor]

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename_base = os.path.basename(args.video_path).split('.')[0]

    if args.extend_from_end:
        mode_str = f"extend_{args.extend_from_end}f"
    else:
        mode_str = f"cond_{'_'.join(map(str, cond_pos_list))}"

    base_video_filename = f"wan22_v2v_{output_filename_base}_{timestamp}_{mode_str}"

    if args.concatenate and args.extend_from_end:
        print("Concatenating original video with the generated video...")
        final_video_frames = all_video_frames + generated_frames_pil
        final_filename_stem = base_video_filename.replace(mode_str, f"extended_total_{len(final_video_frames)}f")
        video_to_save = final_video_frames
        video_filename = os.path.join(args.output_dir, final_filename_stem + ".mp4")
    else:
        video_to_save = generated_frames_pil
        video_filename = os.path.join(args.output_dir, base_video_filename + ".mp4")

    print(f"Saving video to {video_filename}")
    save_video(video_to_save, video_filename, fps=args.fps)
    print("Video saved successfully.")


if __name__ == "__main__":
    main()