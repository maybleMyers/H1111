import gradio as gr
from gradio import update as gr_update
import subprocess
import threading
import time
import re
import os
import random
import tiktoken
import sys
import ffmpeg
from typing import List, Tuple, Optional, Generator, Dict, Any
import json
from gradio import themes
from gradio.themes.utils import colors
import subprocess
from PIL import Image
import math
import cv2
import glob
import shutil
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
from diffusers_helper.bucket_tools import find_nearest_bucket
import time
from gradio_image_annotation import image_annotator


# Add global stop event
stop_event = threading.Event()
skip_event = threading.Event()
logger = logging.getLogger(__name__)

UI_CONFIGS_DIR = "ui_configs"
FRAMEPROK_DEFAULTS_FILE = os.path.join(UI_CONFIGS_DIR, "framepack_defaults.json")

# Helper functions for model detection (moved to global scope)
def get_wan_of_dit_models(dit_folder: str, filter_name: str = "") -> List[str]:
    """Get filtered DiT models for Wan One Frame tab"""
    if not os.path.exists(dit_folder):
        return ["wan22_i2v_14B_low_noise_bf16.safetensors"]
    models = [f for f in os.listdir(dit_folder) if f.endswith('.safetensors')]
    if filter_name:
        models = [m for m in models if filter_name.lower() in m.lower()]
    models.sort(key=str.lower)
    return models if models else ["wan22_i2v_14B_low_noise_bf16.safetensors"]

def get_wan_of_low_noise_models(dit_folder: str) -> List[str]:
    """Get low noise DiT models"""
    if not os.path.exists(dit_folder):
        return ["wan22_i2v_14B_low_noise_bf16.safetensors"]
    models = [f for f in os.listdir(dit_folder) if f.endswith('.safetensors')]
    # Look for models with 'low' in the name (more flexible matching)
    low_models = [m for m in models if 'low' in m.lower()]
    low_models.sort(key=str.lower)
    return low_models if low_models else ["wan22_i2v_14B_low_noise_bf16.safetensors"]

def get_wan_of_high_noise_models(dit_folder: str) -> List[str]:
    """Get high noise DiT models"""
    if not os.path.exists(dit_folder):
        return ["wan22_i2v_14B_high_noise_bf16.safetensors"]
    models = [f for f in os.listdir(dit_folder) if f.endswith('.safetensors')]
    # Look for models with 'high' in the name (more flexible matching)
    high_models = [m for m in models if 'high' in m.lower()]
    high_models.sort(key=str.lower)
    return high_models if high_models else ["wan22_i2v_14B_high_noise_bf16.safetensors"]
    
def get_wan_of_clip_models(dit_folder: str) -> List[str]:
    """Get CLIP models"""
    if not os.path.exists(dit_folder):
        return ["models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]
    models = [f for f in os.listdir(dit_folder) if f.endswith('.pth') and 'clip' in f.lower()]
    models.sort(key=str.lower)
    return models if models else ["models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]

def get_wan_of_vae_models(dit_folder: str) -> List[str]:
    """Get VAE models"""
    if not os.path.exists(dit_folder):
        return ["Wan2.1_VAE.pth"]
    models = [f for f in os.listdir(dit_folder) if f.endswith('.pth') and 'vae' in f.lower()]
    models.sort(key=str.lower)
    return models if models else ["Wan2.1_VAE.pth"]

def get_wan_of_t5_models(dit_folder: str) -> List[str]:
    """Get T5 models"""
    if not os.path.exists(dit_folder):
        return ["models_t5_umt5-xxl-enc-bf16.pth"]
    models = [f for f in os.listdir(dit_folder) if (f.endswith('.pth') or f.endswith('.safetensors')) and 't5' in f.lower()]
    models.sort(key=str.lower)
    return models if models else ["models_t5_umt5-xxl-enc-bf16.pth"]

# Helper functions to get default (first available) model
def get_default_low_noise_model(dit_folder: str = "wan") -> str:
    """Get the first available low noise model as default"""
    models = get_wan_of_low_noise_models(dit_folder)
    return models[0] if models else "wan22_i2v_14B_low_noise_bf16.safetensors"
    
def get_default_high_noise_model(dit_folder: str = "wan") -> str:
    """Get the first available high noise model as default"""
    models = get_wan_of_high_noise_models(dit_folder)
    return models[0] if models else "wan22_i2v_14B_high_noise_bf16.safetensors"
    
def get_default_clip_model(dit_folder: str = "wan") -> str:
    """Get the first available CLIP model as default"""
    models = get_wan_of_clip_models(dit_folder)
    return models[0] if models else "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    
def get_default_vae_model(dit_folder: str = "wan") -> str:
    """Get the first available VAE model as default"""
    models = get_wan_of_vae_models(dit_folder)
    return models[0] if models else "Wan2.1_VAE.pth"
    
def get_default_t5_model(dit_folder: str = "wan") -> str:
    """Get the first available T5 model as default"""
    models = get_wan_of_t5_models(dit_folder)
    return models[0] if models else "models_t5_umt5-xxl-enc-bf16.pth"

# Task-specific model detection functions
def get_task_specific_low_noise_model(dit_folder: str = "wan", task_type: str = "i2v") -> str:
    """Get task-specific low noise model (prefers models matching the task type)"""
    if not os.path.exists(dit_folder):
        return get_default_low_noise_model(dit_folder)
    
    models = [f for f in os.listdir(dit_folder) if f.endswith('.safetensors')]
    low_models = [m for m in models if 'low' in m.lower()]
    
    # Prefer models that match the task type
    task_specific = [m for m in low_models if task_type.lower() in m.lower()]
    if task_specific:
        task_specific.sort(key=str.lower)
        return task_specific[0]
    
    # Fall back to any low noise model
    if low_models:
        low_models.sort(key=str.lower)
        return low_models[0]
    
    return get_default_low_noise_model(dit_folder)

def get_task_specific_high_noise_model(dit_folder: str = "wan", task_type: str = "i2v") -> str:
    """Get task-specific high noise model (prefers models matching the task type)"""
    if not os.path.exists(dit_folder):
        return get_default_high_noise_model(dit_folder)
    
    models = [f for f in os.listdir(dit_folder) if f.endswith('.safetensors')]
    high_models = [m for m in models if 'high' in m.lower()]
    
    # Prefer models that match the task type
    task_specific = [m for m in high_models if task_type.lower() in m.lower()]
    if task_specific:
        task_specific.sort(key=str.lower)
        return task_specific[0]
    
    # Fall back to any high noise model
    if high_models:
        high_models.sort(key=str.lower)
        return high_models[0]
    
    return get_default_high_noise_model(dit_folder)

# Wan One Frame Inference Handler
def wan_one_frame_handler(
    prompt: str,
    negative_prompt: str,
    image_path: str,
    task: str,
    conditioning_strength: float,
    target_index: int,
    control_index: str,
    inference_options: List[str],
    control_images: List,
    control_masks: List,
    width: int,
    height: int,
    frame_num: int,
    fps: int,
    seed: int,
    sample_steps: int,
    flow_shift: float,
    sample_guide_scale: float,
    timestep_boundary: float,
    sample_solver: str,
    batch_size: int,
    save_path: str,
    # Model Paths & Performance
    attn_mode: str,
    block_swap: int,
    fp8: bool,
    fp8_scaled: bool,
    fp8_t5: bool,
    mixed_dtype: bool,
    vae_fp32: bool,
    dit_low_noise_path: str,
    dit_high_noise_path: str,
    clip_path: str,
    vae_path: str,
    t5_path: str,
    # LoRAs
    lora_folder: str,
    lora1_str: str, lora2_str: str, lora3_str: str, lora4_str: str,
    lora1_mult: float, lora2_mult: float, lora3_mult: float, lora4_mult: float,
    lora1_apply_low: bool, lora2_apply_low: bool, lora3_apply_low: bool, lora4_apply_low: bool,
    lora1_apply_high: bool, lora2_apply_high: bool, lora3_apply_high: bool, lora4_apply_high: bool
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    global stop_event
    stop_event.clear()
    
    os.makedirs(save_path, exist_ok=True)
    
    all_generated_images = []
    
    for i in range(int(batch_size)):
        if stop_event.is_set():
            yield all_generated_images, "Generation stopped by user.", ""
            return

        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif int(batch_size) > 1:
            current_seed = seed + i

        status_text = f"Processing Item {i+1}/{batch_size} (Seed: {current_seed})"
        yield all_generated_images.copy(), status_text, "Starting item..."

        # Use explicit task selection
        has_main_image = image_path and image_path.strip()
        
        # For T2V task with image provided, use image as control for one-frame inference
        # For I2V task, use image as main input as usual

        # Construct the command
        cmd = [
            sys.executable, "base_wan_generate_video.py",
            "--task", task,  # Use explicit task selection
            "--prompt", prompt,
            "--video_size", str(int(height)), str(int(width)),
            "--video_length", str(int(frame_num)),
            "--fps", str(int(fps)),
            "--seed", str(current_seed),
            "--infer_steps", str(int(sample_steps)),
            "--flow_shift", str(flow_shift),
            "--guidance_scale", str(sample_guide_scale),
            "--timestep_boundary", str(timestep_boundary),
            "--sample_solver", sample_solver,
            "--output_type", "images",
            "--save_path", save_path,
            "--conditioning_strength", str(conditioning_strength),
            "--vae", os.path.join("wan", vae_path),
            "--t5", os.path.join("wan", t5_path),
            "--dit", os.path.join("wan", dit_low_noise_path),
            "--dit_high_noise", os.path.join("wan", dit_high_noise_path)
        ]
        
        # Only add CLIP for i2v tasks
        if task == "i2v-A14B":
            cmd.extend(["--clip", os.path.join("wan", clip_path)])
        
        if negative_prompt.strip():
            cmd.extend(["--negative_prompt", negative_prompt])
        
        # Handle image input based on task
        if has_main_image:
            if task == "i2v-A14B":
                # For I2V tasks, use image as main input
                cmd.extend(["--image_path", image_path])
            elif task == "t2v-A14B":
                # For T2V tasks, use image as control image for conditioning
                if not control_images:
                    control_images = [image_path]
                else:
                    # Prepend main image to control images list
                    control_images.insert(0, image_path)
            
        # Add control images if provided
        if control_images and len(control_images) > 0:
            control_paths = []
            for ctrl_img in control_images:
                if hasattr(ctrl_img, 'name') and ctrl_img.name:
                    control_paths.append(ctrl_img.name)
                elif isinstance(ctrl_img, str) and ctrl_img.strip():
                    control_paths.append(ctrl_img)
            if control_paths:
                cmd.extend(["--control_image_path"] + control_paths)
                
        # Add control masks if provided
        if control_masks and len(control_masks) > 0:
            mask_paths = []
            for mask in control_masks:
                if hasattr(mask, 'name') and mask.name:
                    mask_paths.append(mask.name)
                elif isinstance(mask, str) and mask.strip():
                    mask_paths.append(mask)
            if mask_paths:
                cmd.extend(["--control_image_mask_path"] + mask_paths)
        
        # Build one_frame_inference parameter
        of_params = []
        of_params.append(f"target_index={target_index}")
        of_params.append(f"control_index={control_index}")
        
        for option in inference_options:
            of_params.append(option)
        
        if of_params:
            cmd.extend(["--one_frame_inference", ",".join(of_params)])
        
        # Performance flags
        if fp8:
            cmd.append("--fp8")
        if fp8_scaled:
            cmd.append("--fp8_scaled") 
        if fp8_t5:
            cmd.append("--fp8_t5")
        if mixed_dtype:
            cmd.append("--mixed_dtype")
        if vae_fp32:
            cmd.extend(["--vae_dtype", "float32"])
        
        cmd.extend(["--blocks_to_swap", str(int(block_swap))])
        
        # LoRA Handling
        lora_weights_low = []
        lora_multipliers_low = []
        lora_weights_high = []
        lora_multipliers_high = []
        
        lora_inputs = [
            (lora1_str, lora1_mult, lora1_apply_low, lora1_apply_high),
            (lora2_str, lora2_mult, lora2_apply_low, lora2_apply_high),
            (lora3_str, lora3_mult, lora3_apply_low, lora3_apply_high),
            (lora4_str, lora4_mult, lora4_apply_low, lora4_apply_high)
        ]
        
        if lora_folder and os.path.exists(lora_folder):
            for name, mult, apply_low, apply_high in lora_inputs:
                if name and name != "None":
                    full_path = os.path.join(lora_folder, name)
                    if os.path.exists(full_path):
                        if apply_low:
                            lora_weights_low.append(full_path)
                            lora_multipliers_low.append(mult)
                        if apply_high:
                            lora_weights_high.append(full_path)
                            lora_multipliers_high.append(mult)
        
        if lora_weights_low:
            cmd.extend(["--lora_weight"] + lora_weights_low)
            cmd.extend(["--lora_multiplier"] + [str(m) for m in lora_multipliers_low])
        
        if lora_weights_high:
            cmd.extend(["--lora_weight_high_noise"] + lora_weights_high)
            cmd.extend(["--lora_multiplier_high_noise"] + [str(m) for m in lora_multipliers_high])
        
        # --- Execute Subprocess ---
        print(f"Running Wan One Frame Command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1
            )
            
            progress_text_update = "Subprocess started..."
            
            for line in iter(process.stdout.readline, ''):
                if stop_event.is_set():
                    try: 
                        process.terminate()
                        process.wait(timeout=5)
                    except: 
                        process.kill()
                        process.wait()
                    yield all_generated_images, "Generation stopped by user.", ""
                    return
                    
                line_strip = line.strip()
                if not line_strip: 
                    continue
                    
                print(f"WAN_OF_SUBPROCESS: {line_strip}")
                
                progress_text_update = line_strip
                yield all_generated_images, status_text, progress_text_update
                
                # Check for saved files
                if "Saved" in line_strip or "saved" in line_strip.lower():
                    # Try to find generated files
                    if os.path.exists(save_path):
                        for file in os.listdir(save_path):
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                full_path = os.path.join(save_path, file)
                                if (full_path, file) not in all_generated_images:
                                    all_generated_images.append((full_path, file))
                                    yield all_generated_images, status_text, f"Found generated image: {file}"
            
            process.wait()
            print(f"WAN_OF_SUBPROCESS: Process completed with return code {process.returncode}")
            
            # Final check for output files
            if os.path.exists(save_path):
                for file in sorted(os.listdir(save_path)):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(save_path, file)
                        if (full_path, file) not in all_generated_images:
                            all_generated_images.append((full_path, file))
            
            item_status = f"Item {i+1}/{batch_size} completed!" if process.returncode == 0 else f"Item {i+1}/{batch_size} failed with code {process.returncode}"
            yield all_generated_images, item_status, ""
            
        except Exception as e:
            error_msg = f"Error executing item {i+1}/{batch_size}: {str(e)}"
            print(f"WAN_OF_SUBPROCESS: {error_msg}")
            yield all_generated_images, error_msg, ""
            return
    
    yield all_generated_images, "All generations completed!", ""

#wan2.2
def wan22_batch_handler(
    prompt: str,
    negative_prompt: str,
    image_path: str,
    task: str,
    width: int,
    height: int,
    frame_num: int,
    fps: int,
    base_seed: int,
    sample_solver: str,
    sample_steps: int,
    flow_shift: float,
    sample_guide_scale: float,
    dual_dit_boundary: float,
    batch_size: int,
    save_path: str,
    # Model Paths & Performance
    attn_mode: str,
    mixed_dtype: bool,
    block_swap: int,
    fp8: bool,
    fp8_scaled: bool,
    fp8_t5: bool,
    dit_low_noise_path: str,
    dit_high_noise_path: str,
    clip_path: str,
    dit_path: str,
    vae_path: str,
    t5_path: str,
    # LoRAs
    lora_folder: str,
    lora1_str: str, lora2_str: str, lora3_str: str, lora4_str: str,
    lora5_str: str, lora6_str: str, lora7_str: str, lora8_str: str,
    lora1_mult: float, lora2_mult: float, lora3_mult: float, lora4_mult: float,
    lora5_mult: float, lora6_mult: float, lora7_mult: float, lora8_mult: float,
    lora1_apply_low: bool, lora2_apply_low: bool, lora3_apply_low: bool, lora4_apply_low: bool,
    lora5_apply_low: bool, lora6_apply_low: bool, lora7_apply_low: bool, lora8_apply_low: bool,
    lora1_apply_high: bool, lora2_apply_high: bool, lora3_apply_high: bool, lora4_apply_high: bool,
    lora5_apply_high: bool, lora6_apply_high: bool, lora7_apply_high: bool, lora8_apply_high: bool,
    # Previews
    enable_preview: bool,
    preview_steps: int,
    dynamic_model_loading: bool,
    unload_text_encoders: bool,
    vae_fp32: bool,
    enable_v2v: bool, input_video: str, v2v_strength: float, v2v_low_noise_only: bool, v2v_use_i2v: bool,  # V2V parameters
    enable_extension: bool, extend_frames: int, frames_to_check: int  # Extension parameters
) -> Generator[Tuple[List[Tuple[str, str]], Optional[str], str, str], None, None]:
    global stop_event
    stop_event.clear()

    # --- Initial Checks ---
    os.makedirs(save_path, exist_ok=True)

    all_generated_videos = []
    
    for i in range(int(batch_size)):
        if stop_event.is_set():
            yield all_generated_videos, [], "Generation stopped by user.", ""
            return

        current_seed = base_seed
        if base_seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif int(batch_size) > 1:
            current_seed = base_seed + i

        status_text = f"Processing Item {i+1}/{batch_size} (Seed: {current_seed})"
        yield all_generated_videos.copy(), [], status_text, "Starting item..."

        # --- Prepare command for a single generation ---
        run_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
        unique_preview_suffix = f"wan22_{run_id}"

        command = [
            sys.executable, "wan2_generate_video.py",
            "--task", str(task),
            "--prompt", str(prompt),
            "--video_size", str(height), str(width),
            "--video_length", str(frame_num),
            "--fps", str(fps),
            "--infer_steps", str(sample_steps),
            "--guidance_scale", str(sample_guide_scale),
            "--dual_dit_boundary", str(dual_dit_boundary),
            "--flow_shift", str(flow_shift),
            "--sample_solver", str(sample_solver),
            "--seed", str(current_seed),
            "--save_path", str(save_path),
            "--attn_mode", str(attn_mode),
            "--blocks_to_swap", str(block_swap),
            "--vae", os.path.join("wan", vae_path),
            "--t5", os.path.join("wan", t5_path),
        ]

        if negative_prompt:
            command.extend(["--negative_prompt", str(negative_prompt)])
        
        # --- Model Path Logic based on Task ---
        if "A14B" in task:
            command.extend(["--dit_low_noise", os.path.join("wan", dit_low_noise_path), "--dit_high_noise", os.path.join("wan", dit_high_noise_path)])
            if "i2v" in task:
                command.extend(["--clip", os.path.join("wan", clip_path)])
        elif "ti2v-5B" in task:
            command.extend(["--dit", os.path.join("wan", dit_path)])

        # Handle V2V vs Extension mode
        if enable_v2v and input_video:
            if enable_extension:
                # Use extension mode instead of regular V2V
                command.extend(["--extend_video", str(input_video)])
                command.extend(["--extend_frames", str(extend_frames)])
                command.extend(["--frames_to_check", str(frames_to_check)])
                
                # Force i2v-A14B task for extension
                if "i2v" not in task:
                    yield [], None, f"Warning: Extension mode requires i2v-A14B task, but got {task}. Please change task to i2v-A14B.", ""
                    return
            else:
                # Regular V2V mode
                command.extend(["--video_path", str(input_video)])
                command.extend(["--strength", str(v2v_strength)])
                if v2v_low_noise_only:
                    command.append("--v2v_low_noise_only")
                if v2v_use_i2v:
                    command.append("--v2v_use_i2v")
        elif "i2v" in task and image_path:
            command.extend(["--image_path", str(image_path)])

        if fp8: command.append("--fp8")
        if fp8_scaled: command.append("--fp8_scaled")
        if mixed_dtype: command.append("--mixed_dtype")
        if fp8_t5: command.append("--fp8_t5")
        # ADD THIS:
        if dynamic_model_loading and "A14B" in task:
            command.append("--dynamic_model_loading")
        if unload_text_encoders:
            command.append("--unload_text_encoders")
        if vae_fp32:
            command.extend(["--vae_dtype", "float32"])
        
        if enable_preview and preview_steps > 0:
            command.extend(["--preview", str(preview_steps)])
            command.extend(["--preview_suffix", unique_preview_suffix])

        # --- LoRA Handling ---
        lora_weights_paths = []
        lora_multipliers_values = []
        lora_weights_paths_high = []
        lora_multipliers_values_high = []
        
        lora_inputs = [
            (lora1_str, lora1_mult, lora1_apply_low, lora1_apply_high), 
            (lora2_str, lora2_mult, lora2_apply_low, lora2_apply_high),
            (lora3_str, lora3_mult, lora3_apply_low, lora3_apply_high), 
            (lora4_str, lora4_mult, lora4_apply_low, lora4_apply_high),
            (lora5_str, lora5_mult, lora5_apply_low, lora5_apply_high),
            (lora6_str, lora6_mult, lora6_apply_low, lora6_apply_high),
            (lora7_str, lora7_mult, lora7_apply_low, lora7_apply_high),
            (lora8_str, lora8_mult, lora8_apply_low, lora8_apply_high)
        ]
        
        if lora_folder and os.path.exists(lora_folder):
            for name, mult, apply_low, apply_high in lora_inputs:
                if name and name != "None":
                    path = os.path.join(lora_folder, name)
                    if os.path.exists(path):
                        # Apply to low noise model (default behavior)
                        if apply_low:
                            lora_weights_paths.append(path)
                            lora_multipliers_values.append(str(mult))
                        
                        # Apply to high noise model (for dual-dit models)
                        if apply_high and "A14B" in task:
                            lora_weights_paths_high.append(path)
                            lora_multipliers_values_high.append(str(mult))
                    else:
                        print(f"Warning: LoRA file not found: {path}")
        
        # Add low noise LoRA arguments
        if lora_weights_paths:
            command.extend(["--lora_weight"] + lora_weights_paths)
            command.extend(["--lora_multiplier"] + lora_multipliers_values)
        
        # Add high noise LoRA arguments (only for A14B models)
        if lora_weights_paths_high and "A14B" in task:
            command.extend(["--lora_weight_high"] + lora_weights_paths_high)
            command.extend(["--lora_multiplier_high"] + lora_multipliers_values_high)
        
        # --- Execute Subprocess ---
        print(f"Running Wan2.2 Command: {' '.join(command)}")
        
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1
        )

        current_preview_yield_list = []
        last_preview_mtime = 0
        preview_base_dir = os.path.join(save_path, "previews")
        preview_mp4_path = os.path.join(preview_base_dir, f"latent_preview_{unique_preview_suffix}.mp4")

        current_video_file_for_item = None
        progress_text_update = "Subprocess started..."
        for line in iter(process.stdout.readline, ''):
            if stop_event.is_set():
                try: process.terminate(); process.wait(timeout=5)
                except: process.kill(); process.wait()
                yield all_generated_videos, [], "Generation stopped by user.", ""
                return

            line_strip = line.strip()
            if not line_strip: continue
            print(f"WAN2.2_SUBPROCESS: {line_strip}")
            progress_text_update = line_strip

            tqdm_match = re.search(r'(\d+)\%\|.+\| (\d+/\d+) \[(\d{2}:\d{2})<(\d{2}:\d{2})', line_strip)
            video_saved_match = re.search(r"Video saved to:\s*(.*\.mp4)", line_strip)

            if video_saved_match:
                found_path = video_saved_match.group(1).strip()
                if os.path.exists(found_path):
                    current_video_file_for_item = found_path
                progress_text_update = f"Finalizing: {os.path.basename(found_path)}"
                status_text = f"Item {i+1}/{batch_size} (Seed: {current_seed}) - Saved"
            elif tqdm_match:
                percentage = tqdm_match.group(1)
                steps_iter = tqdm_match.group(2)
                time_elapsed = tqdm_match.group(3)
                time_remaining = tqdm_match.group(4)
                progress_text_update = f"Step {steps_iter} ({percentage}%) | ETA: {time_remaining}"
                status_text = f"Item {i+1}/{batch_size} (Seed: {current_seed}) - Denoising"

            if enable_preview:
                if os.path.exists(preview_mp4_path):
                    current_mtime = os.path.getmtime(preview_mp4_path)
                    if current_mtime > last_preview_mtime:
                        current_preview_yield_list = [preview_mp4_path]
                        last_preview_mtime = current_mtime

            yield all_generated_videos.copy(), current_preview_yield_list, status_text, progress_text_update

        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0 and current_video_file_for_item:
            params_for_meta = {
                "model_type": "Wan2.2", "prompt": prompt, "negative_prompt": negative_prompt,
                "image_path": os.path.basename(image_path) if image_path else None,
                "task": task, "width": width, "height": height, "frame_num": frame_num, "fps": fps,
                "dit_low_noise": dit_low_noise_path, "dit_high_noise": dit_high_noise_path,
                "dit": dit_path, "vae": vae_path, "t5": t5_path, "clip": clip_path,
                "seed": current_seed, "sample_solver": sample_solver, "sample_steps": sample_steps,
                "flow_shift": flow_shift, "sample_guide_scale": sample_guide_scale,
                "dual_dit_boundary": dual_dit_boundary,  # Add dual_dit_boundary to metadata
                "lora_weights": [lora1_str, lora2_str, lora3_str, lora4_str, lora5_str, lora6_str, lora7_str, lora8_str],
                "lora_multipliers": [lora1_mult, lora2_mult, lora3_mult, lora4_mult, lora5_mult, lora6_mult, lora7_mult, lora8_mult],
            }
            try:
                add_metadata_to_video(current_video_file_for_item, params_for_meta)
            except Exception as meta_err:
                print(f"Warning: Failed to add metadata to {current_video_file_for_item}: {meta_err}")
            
            all_generated_videos.append((current_video_file_for_item, f"Wan2.2 - Seed: {current_seed}"))
            status_text = f"Item {i+1}/{batch_size} (Seed: {current_seed}) - Completed"
            progress_text_update = f"Saved: {os.path.basename(current_video_file_for_item)}"
        else:
            status_text = f"Item {i+1}/{batch_size} (Seed: {current_seed}) - Failed (Code: {return_code})"
            progress_text_update = "Subprocess failed. Check console."
        
        yield all_generated_videos.copy(), [], status_text, progress_text_update
        
        clear_cuda_cache()
        time.sleep(0.2)
        
    yield all_generated_videos, [], "Wan2.2 Batch complete.", ""

### Multitalk
def multitalk_batch_handler(
    prompt: str,
    negative_prompt: str,
    cond_image_data: dict,
    audio_person1: Optional[str],
    audio_person2: Optional[str],
    batch_size: int,
    # Generation params
    size: str,
    mode: str,
    frame_num: int,
    motion_frame: int,
    sample_steps: int,
    sample_shift: float,
    text_guide_scale: float,
    audio_guide_scale: float,
    seed: int,
    # Advanced & Performance
    audio_type: str,
    num_persistent: float,
    use_teacache: bool,
    teacache_thresh: float,
    use_apg: bool,
    apg_momentum: float,
    apg_norm_thresh: float,
    use_full_video_preview: bool,
    # Paths
    ckpt_dir: str,
    wav2vec_dir: str,
    t5_tokenizer_path: str,
    save_path: str,
    # LoRAs
    lora_folder: str,
    lora1_str: str, lora2_str: str, lora3_str: str, lora4_str: str,
    lora1_mult: float, lora2_mult: float, lora3_mult: float, lora4_mult: float,
) -> Generator[Tuple[List[Tuple[str, str]], List[str], str, str], None, None]:
    global stop_event
    stop_event.clear()

    # --- Initial Checks ---
    if not cond_image_data or not cond_image_data.get('image'):
        yield [], None, "Error: Reference Image not provided.", ""
        return
    cond_image = cond_image_data['image']
    if not cond_image or not os.path.exists(cond_image):
        yield [], None, "Error: Reference Image not found.", ""
        return
    os.makedirs(save_path, exist_ok=True)

    all_generated_videos = []
    
    for i in range(int(batch_size)):
        if stop_event.is_set():
            yield all_generated_videos, None, "Generation stopped by user.", ""
            return

        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif int(batch_size) > 1:
            current_seed = seed + i

        status_text = f"Processing Item {i+1}/{batch_size} (Seed: {current_seed})"
        yield all_generated_videos.copy(), None, status_text, "Starting item..."

        # --- Prepare command for a single generation ---
        
        run_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
        unique_preview_suffix = f"multitalk_{run_id}"
        save_file_prefix = os.path.join(save_path, f"multitalk_{run_id}_s{current_seed}")
        num_persistent_backend = int(num_persistent * 1_000_000_000)

        command = [
            sys.executable,
            "generate_multitalk_video.py",
            "--ckpt_dir", str(ckpt_dir),
            "--wav2vec_dir", str(wav2vec_dir),
            "--t5_tokenizer_path", str(t5_tokenizer_path),
            "--prompt", str(prompt),
            "--n_prompt", str(negative_prompt),
            "--cond_image", str(cond_image),
            "--cond_audio_person1", str(audio_person1),
            "--base_seed", str(current_seed),
            "--save_file", save_file_prefix,
            "--size", str(size),
            "--mode", str(mode),
            "--frame_num", str(frame_num),
            "--motion_frame", str(motion_frame),
            "--sample_steps", str(sample_steps),
            "--sample_shift", str(sample_shift),
            "--sample_text_guide_scale", str(text_guide_scale),
            "--sample_audio_guide_scale", str(audio_guide_scale),
            "--audio_type", str(audio_type),
            "--num_persistent_param_in_dit", str(num_persistent_backend),
        ]
        
        if audio_person2 and os.path.exists(audio_person2):
            command.extend(["--cond_audio_person2", str(audio_person2)])

        boxes = cond_image_data.get('boxes', [])
        bbox1_str = None
        bbox2_str = None
        if boxes:
            for box_info in boxes:
                label = box_info.get('label', '').strip().lower()
                # The coordinates are directly in box_info, not in a nested 'box' dictionary.
                if all(k in box_info for k in ["xmin", "ymin", "xmax", "ymax"]):
                    coords = ",".join(map(str, [
                        box_info["xmin"], box_info["ymin"], 
                        box_info["xmax"], box_info["ymax"]
                    ]))
                    
                    if label == 'person 1' and not bbox1_str:
                        bbox1_str = coords
                    elif label == 'person 2' and not bbox2_str:
                        bbox2_str = coords
        
        if bbox1_str:
            command.extend(["--bbox_person1", bbox1_str])
        if bbox2_str:
            command.extend(["--bbox_person2", bbox2_str])   

        if use_teacache:
            command.append("--use_teacache")
            command.extend(["--teacache_thresh", str(teacache_thresh)])

        if use_apg:
            command.append("--use_apg")
            command.extend(["--apg_momentum", str(apg_momentum)])
            command.extend(["--apg_norm_threshold", str(apg_norm_thresh)])

        if use_full_video_preview:
            command.append("--full_preview")
            # The generation script now requires the suffix to be passed for the full preview
            command.extend(["--preview_suffix", unique_preview_suffix])
        # LoRA Handling
        lora_weights_paths = []
        lora_multipliers_values = []
        lora_inputs = [
            (lora1_str, lora1_mult), (lora2_str, lora2_mult),
            (lora3_str, lora3_mult), (lora4_str, lora4_mult)
        ]
        if lora_folder and os.path.exists(lora_folder):
            for name, mult in lora_inputs:
                if name and name != "None":
                    path = os.path.join(lora_folder, name)
                    if os.path.exists(path):
                        lora_weights_paths.append(path)
                        lora_multipliers_values.append(str(mult))
                    else:
                        print(f"Warning: LoRA file not found: {path}")
        if lora_weights_paths:
            command.extend(["--lora_weight"] + lora_weights_paths)
            command.extend(["--lora_multiplier"] + lora_multipliers_values)
            
        # --- Execute Subprocess ---
        print(f"Running MultiTalk Command: {' '.join(command)}")
        
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1
        )

        current_preview_yield_path = None
        last_preview_mtime = 0
        preview_base_dir = os.path.join(os.path.dirname(save_file_prefix), "previews")
        preview_mp4_gen_path = os.path.join(preview_base_dir, f"latent_preview_{unique_preview_suffix}.mp4")

        current_video_file_for_item = None
        progress_text_update = "Subprocess started..."

        for line in iter(process.stdout.readline, ''):
            if stop_event.is_set():
                try: process.terminate(); process.wait(timeout=5)
                except: process.kill(); process.wait()
                yield all_generated_videos, None, "Generation stopped by user.", ""
                return

            line_strip = line.strip()
            if not line_strip: continue
            print(f"MULTITALK_SUBPROCESS: {line_strip}")

            tqdm_match = re.search(r"(\d+)\s*%\|.*?\|\s*(\d+/\d+)\s*\[([^\]]+)\]", line_strip)
            clip_progress_match = re.search(r"Generating clip (\d+/\d+)", line_strip)
            saving_video_match = re.search(r'Saving video:.*', line_strip)
            final_save_match = re.search(r'Saving generated video to (.*\.mp4)', line_strip)          
            if final_save_match:
                found_path = final_save_match.group(1).strip()
                if os.path.exists(found_path):
                    current_video_file_for_item = found_path
                progress_text_update = f"Finalized: {os.path.basename(current_video_file_for_item or found_path)}"
                status_text = f"Item {i+1}/{batch_size} (Seed: {current_seed}) - Saved"
            elif saving_video_match:
                progress_text_update = "Saving final video..."
            elif clip_progress_match:
                status_text = f"Item {i+1}/{batch_size} (Seed: {current_seed}) - Clip {clip_progress_match.group(1)}"
                tqdm_match_on_clip_line = re.search(r"(\d+)\s*%\|.*?\|\s*(\d+/\d+)\s*\[([^\]]+)\]", line_strip)
                if tqdm_match_on_clip_line:
                    percentage = tqdm_match_on_clip_line.group(1)
                    steps_iter = tqdm_match_on_clip_line.group(2)
                    time_details = tqdm_match_on_clip_line.group(3)
                    time_elapsed = "??"
                    time_remaining = "??"
                    time_split = time_details.split('<')
                    if len(time_split) > 1:
                        time_elapsed = time_split[0].strip()
                        remaining_part = time_split[1]
                        time_remaining = remaining_part.split(',')[0].strip()
                    else:
                        time_elapsed = time_details.split(',')[0].strip().replace('?','')
                    progress_text_update = f"Step {steps_iter} ({percentage}%) | Elapsed: {time_elapsed} | ETA: {time_remaining}"
                else:
                    progress_text_update = "Starting new clip generation..."
            elif tqdm_match:
                percentage = tqdm_match.group(1)
                steps_iter = tqdm_match.group(2)
                time_details = tqdm_match.group(3) # e.g., "00:01<00:09, 1.05it/s" or "00:00<?, ?it/s"

                time_elapsed = "??"
                time_remaining = "??"
                
                time_split = time_details.split('<')
                if len(time_split) > 1:
                    time_elapsed = time_split[0].strip()
                    remaining_part = time_split[1]
                    time_remaining = remaining_part.split(',')[0].strip()
                else:
                    time_elapsed = time_details.split(',')[0].strip().replace('?','')
                
                if "Clip" not in status_text: # Update status if not already showing clip info
                     status_text = f"Item {i+1}/{batch_size} (Seed: {current_seed}) - Denoising"
                
                progress_text_update = f"Step {steps_iter} ({percentage}%) | Elapsed: {time_elapsed} | ETA: {time_remaining}"
            else:
                 progress_text_update = line_strip

            if use_full_video_preview:
                found_preview_path = None

                if os.path.exists(preview_mp4_gen_path):
                    current_mtime = os.path.getmtime(preview_mp4_gen_path)
                    if current_mtime > last_preview_mtime:
                        current_preview_yield_path = preview_mp4_gen_path
                        last_preview_mtime = current_mtime

            yield all_generated_videos.copy(), current_preview_yield_path, status_text, progress_text_update

        process.stdout.close()
        return_code = process.wait()
        
        # Check for the video file again after process completion
        final_video_path = save_file_prefix + ".mp4"
        if os.path.exists(final_video_path):
            current_video_file_for_item = final_video_path

        # After subprocess finishes for one item
        if return_code == 0 and current_video_file_for_item:
            # --- START METADATA SAVING ---
            params_for_meta = {
                "model_type": "MultiTalk",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "cond_image": os.path.basename(cond_image) if cond_image else None,
                "audio_person1": os.path.basename(audio_person1) if audio_person1 else None,
                "audio_person2": os.path.basename(audio_person2) if audio_person2 else None,
                "batch_size": batch_size,
                "seed": current_seed,
                "size": size,
                "mode": mode,
                "frame_num": frame_num,
                "motion_frame": motion_frame,
                "sample_steps": sample_steps,
                "sample_shift": sample_shift,
                "text_guide_scale": text_guide_scale,
                "audio_guide_scale": audio_guide_scale,
                "audio_type": audio_type,
                "num_persistent_param_in_dit": num_persistent,
                "use_teacache": use_teacache,
                "teacache_thresh": teacache_thresh,
                "use_apg": use_apg,
                "apg_momentum": apg_momentum,
                "apg_norm_thresh": apg_norm_thresh,
                "ckpt_dir": ckpt_dir,
                "wav2vec_dir": wav2vec_dir,
                "t5_tokenizer_path": t5_tokenizer_path,
                "save_path": save_path,
                "lora_folder": lora_folder,
                "lora_weights": [lora1_str, lora2_str, lora3_str, lora4_str],
                "lora_multipliers": [lora1_mult, lora2_mult, lora3_mult, lora4_mult],
            }
            try:
                add_metadata_to_video(current_video_file_for_item, params_for_meta)
                print(f"Added metadata to {current_video_file_for_item}")
            except Exception as meta_err:
                print(f"Warning: Failed to add metadata to {current_video_file_for_item}: {meta_err}")
            # --- END METADATA SAVING ---
            
            all_generated_videos.append((current_video_file_for_item, f"MultiTalk - Seed: {current_seed}"))
            status_text = f"Item {i+1}/{batch_size} (Seed: {current_seed}) - Completed"
            progress_text_update = f"Saved: {os.path.basename(current_video_file_for_item)}"
        else:
            status_text = f"Item {i+1}/{batch_size} (Seed: {current_seed}) - Failed (Code: {return_code})"
            progress_text_update = "Subprocess failed. Check console."
        
        yield all_generated_videos.copy(), None, status_text, progress_text_update
        
        clear_cuda_cache()
        time.sleep(0.2)
        
    yield all_generated_videos, None, "MultiTalk Batch complete.", ""

### InfiniteTalk
def infinitetalk_batch_handler(
    prompt: str,
    negative_prompt: str,
    cond_image: str,  # Changed from cond_video to image path
    audio_person1: Optional[str],
    audio_person2: Optional[str],
    batch_size: int,
    # Generation params
    size: str,
    mode: str,
    frame_num: int,
    motion_frame: int,
    sample_steps: int,
    sample_shift: float,
    text_guide_scale: float,
    audio_guide_scale: float,
    seed: int,
    # Advanced & Performance
    audio_type: str,
    num_persistent: float,
    use_teacache: bool,
    teacache_thresh: float,
    use_apg: bool,
    apg_momentum: float,
    apg_norm_thresh: float,
    use_full_video_preview: bool,
    # InfiniteTalk-specific params
    scene_seg: bool,
    dit_path: Optional[str],
    infinitetalk_dir: Optional[str],
    # Paths
    ckpt_dir: str,
    wav2vec_dir: str,
    t5_tokenizer_path: str,
    save_path: str,
    # LoRAs
    lora_folder: str,
    lora1_str: str, lora2_str: str, lora3_str: str, lora4_str: str,
    lora1_mult: float, lora2_mult: float, lora3_mult: float, lora4_mult: float,
) -> Generator[Tuple[List[Tuple[str, str]], List[str], str, str], None, None]:
    global stop_event
    stop_event.clear()

    # --- Initial Checks ---
    if not cond_image or not os.path.exists(cond_image):
        yield [], None, "Error: Reference Image not provided or not found.", ""
        return
    
    if not audio_person1:
        yield [], None, "Error: At least one audio file is required.", ""
        return
        
    os.makedirs(save_path, exist_ok=True)

    all_generated_videos = []
    
    for i in range(int(batch_size)):
        if stop_event.is_set():
            yield all_generated_videos, None, "Generation stopped by user.", ""
            return

        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif int(batch_size) > 1:
            current_seed = seed + i

        status_text = f"Processing Item {i+1}/{batch_size} (Seed: {current_seed})"
        yield all_generated_videos.copy(), None, status_text, "Starting item..."

        # --- Prepare command for a single generation ---
        
        run_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
        unique_preview_suffix = f"infinitetalk_{run_id}"
        save_file_prefix = os.path.join(save_path, f"infinitetalk_{run_id}_s{current_seed}")
        num_persistent_backend = int(num_persistent * 1_000_000_000)

        command = [
            sys.executable,
            "generate_infinitalk_video.py",
            "--ckpt_dir", str(ckpt_dir),
            "--wav2vec_dir", str(wav2vec_dir),
            "--t5_tokenizer_path", str(t5_tokenizer_path),
            "--prompt", str(prompt),
            "--n_prompt", str(negative_prompt),
            "--cond_image", str(cond_image),  # Changed from cond_video
            "--cond_audio_person1", str(audio_person1),
            "--base_seed", str(current_seed),
            "--save_file", save_file_prefix,
            "--size", str(size),
            "--mode", str(mode),
            "--frame_num", str(frame_num),
            "--motion_frame", str(motion_frame),
            "--sample_steps", str(sample_steps),
            "--sample_shift", str(sample_shift),
            "--sample_text_guide_scale", str(text_guide_scale),
            "--sample_audio_guide_scale", str(audio_guide_scale),
            "--audio_type", str(audio_type),
            "--num_persistent_param_in_dit", str(num_persistent_backend),
        ]
        
        if audio_person2 and os.path.exists(audio_person2):
            command.extend(["--cond_audio_person2", str(audio_person2)])

        # Add InfiniteTalk-specific parameters
        if scene_seg:
            command.append("--scene_seg")
            
        if dit_path and os.path.exists(dit_path):
            command.extend(["--dit_path", str(dit_path)])
            
        if infinitetalk_dir and os.path.exists(infinitetalk_dir):
            command.extend(["--infinitetalk_dir", str(infinitetalk_dir)])

        if use_teacache:
            command.append("--use_teacache")
            command.extend(["--teacache_thresh", str(teacache_thresh)])

        if use_apg:
            command.append("--use_apg")
            command.extend(["--apg_momentum", str(apg_momentum)])
            command.extend(["--apg_norm_threshold", str(apg_norm_thresh)])

        if use_full_video_preview:
            command.append("--full_preview")
            command.extend(["--preview_suffix", unique_preview_suffix])
            
        # LoRA Handling
        lora_weights_paths = []
        lora_multipliers_values = []
        lora_inputs = [
            (lora1_str, lora1_mult), (lora2_str, lora2_mult),
            (lora3_str, lora3_mult), (lora4_str, lora4_mult)
        ]
        if lora_folder and os.path.exists(lora_folder):
            for name, mult in lora_inputs:
                if name and name != "None":
                    path = os.path.join(lora_folder, name)
                    if os.path.exists(path):
                        lora_weights_paths.append(path)
                        lora_multipliers_values.append(str(mult))
                    else:
                        print(f"Warning: LoRA file not found: {path}")
        if lora_weights_paths:
            command.extend(["--lora_weight"] + lora_weights_paths)
            command.extend(["--lora_multiplier"] + lora_multipliers_values)
            
        # --- Execute Subprocess ---
        print(f"Running InfiniteTalk Command: {' '.join(command)}")
        
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1
        )

        current_preview_yield_path = None
        last_preview_mtime = 0
        preview_base_dir = os.path.join(os.path.dirname(save_file_prefix), "previews")
        preview_mp4_gen_path = os.path.join(preview_base_dir, f"latent_preview_{unique_preview_suffix}.mp4")
        # Also try the absolute path in case the working directory is different
        preview_mp4_gen_path_abs = os.path.abspath(preview_mp4_gen_path)

        current_video_file_for_item = None
        progress_text_update = "Subprocess started..."

        for line in iter(process.stdout.readline, ''):
            if stop_event.is_set():
                try: process.terminate(); process.wait(timeout=5)
                except: process.kill(); process.wait()
                yield all_generated_videos, None, "Generation stopped by user.", ""
                return

            line_strip = line.strip()
            if not line_strip: continue
            print(f"INFINITETALK_SUBPROCESS: {line_strip}")

            tqdm_match = re.search(r"(\d+)\s*%\|.*?\|\s*(\d+/\d+)\s*\[([^\]]+)\]", line_strip)
            clip_progress_match = re.search(r"Generating clip (\d+/\d+)", line_strip)
            scene_progress_match = re.search(r"Processing scene (\d+/\d+)", line_strip)
            saving_video_match = re.search(r'Saving video:.*', line_strip)
            final_save_match = re.search(r'Saving generated video to (.*\.mp4)', line_strip)
            
            if final_save_match:
                found_path = final_save_match.group(1).strip()
                print(f"DEBUG: Final save detected, found_path: {found_path}")
                # Try both relative and absolute paths
                if os.path.exists(found_path):
                    current_video_file_for_item = found_path
                    print(f"DEBUG: Found video file at: {found_path}")
                elif os.path.exists(os.path.abspath(found_path)):
                    current_video_file_for_item = os.path.abspath(found_path)
                    print(f"DEBUG: Found video file at absolute path: {os.path.abspath(found_path)}")
                else:
                    # Sometimes the path might be relative to the script location
                    current_video_file_for_item = found_path  # Store the path anyway for later verification
                    print(f"DEBUG: Video file not immediately found, storing path for later verification: {found_path}")
                progress_text_update = f"Finalized: {os.path.basename(current_video_file_for_item or found_path)}"
                status_text = f"Item {i+1}/{batch_size} (Seed: {current_seed}) - Saved"
            elif saving_video_match:
                progress_text_update = "Saving final video..."
            elif scene_progress_match:
                status_text = f"Item {i+1}/{batch_size} (Seed: {current_seed}) - Scene {scene_progress_match.group(1)}"
            elif clip_progress_match:
                status_text = f"Item {i+1}/{batch_size} (Seed: {current_seed}) - Clip {clip_progress_match.group(1)}"
                tqdm_match_on_clip_line = re.search(r"(\d+)\s*%\|.*?\|\s*(\d+/\d+)\s*\[([^\]]+)\]", line_strip)
                if tqdm_match_on_clip_line:
                    percentage = tqdm_match_on_clip_line.group(1)
                    steps_iter = tqdm_match_on_clip_line.group(2)
                    time_details = tqdm_match_on_clip_line.group(3)
                    progress_text_update = f"Clip: {percentage}% | {steps_iter} | {time_details}"
            elif tqdm_match:
                percentage = tqdm_match.group(1)
                steps_iter = tqdm_match.group(2)
                time_details = tqdm_match.group(3)
                progress_text_update = f"{percentage}% | {steps_iter} | {time_details}"

            # Check for preview updates
            if use_full_video_preview:
                # Try both relative and absolute paths for preview
                preview_found = False
                if os.path.exists(preview_mp4_gen_path):
                    if os.path.getmtime(preview_mp4_gen_path) > last_preview_mtime:
                        current_preview_yield_path = preview_mp4_gen_path
                        last_preview_mtime = os.path.getmtime(preview_mp4_gen_path)
                        preview_found = True
                        print(f"DEBUG: Preview found at: {preview_mp4_gen_path}")
                elif os.path.exists(preview_mp4_gen_path_abs):
                    if os.path.getmtime(preview_mp4_gen_path_abs) > last_preview_mtime:
                        current_preview_yield_path = preview_mp4_gen_path_abs
                        last_preview_mtime = os.path.getmtime(preview_mp4_gen_path_abs)
                        preview_found = True
                        print(f"DEBUG: Preview found at absolute path: {preview_mp4_gen_path_abs}")
                        
                # Also check if there's any preview message in the logs
                if "Saved full preview to" in line_strip:
                    preview_path_match = re.search(r'Saved full preview to (.+\.mp4)', line_strip)
                    if preview_path_match:
                        detected_preview_path = preview_path_match.group(1).strip()
                        print(f"DEBUG: Preview save detected in logs: {detected_preview_path}")
                        if os.path.exists(detected_preview_path):
                            current_preview_yield_path = detected_preview_path
                            last_preview_mtime = os.path.getmtime(detected_preview_path)
                            print(f"DEBUG: Using preview from log detection: {detected_preview_path}")
                        elif os.path.exists(os.path.abspath(detected_preview_path)):
                            abs_preview_path = os.path.abspath(detected_preview_path)
                            current_preview_yield_path = abs_preview_path
                            last_preview_mtime = os.path.getmtime(abs_preview_path)
                            print(f"DEBUG: Using preview from log detection (absolute): {abs_preview_path}")

            yield all_generated_videos.copy(), current_preview_yield_path, status_text, progress_text_update

        process.wait()

        # Final verification of output file if not detected during streaming
        if not current_video_file_for_item:
            print(f"DEBUG: No video file detected during streaming, attempting final verification...")
            # Try to find the expected output file based on save_file_prefix
            expected_output = f"{save_file_prefix}.mp4"
            print(f"DEBUG: Looking for expected output: {expected_output}")
            if os.path.exists(expected_output):
                current_video_file_for_item = expected_output
                print(f"DEBUG: Found expected output at: {expected_output}")
            elif os.path.exists(os.path.abspath(expected_output)):
                current_video_file_for_item = os.path.abspath(expected_output)
                print(f"DEBUG: Found expected output at absolute path: {os.path.abspath(expected_output)}")
            else:
                # Last resort: look for any .mp4 files in the output directory with matching timestamp
                output_dir = os.path.dirname(save_file_prefix)
                print(f"DEBUG: Searching output directory: {output_dir}")
                if os.path.exists(output_dir):
                    for f in os.listdir(output_dir):
                        if f.endswith('.mp4') and run_id in f:
                            potential_path = os.path.join(output_dir, f)
                            print(f"DEBUG: Found potential match: {potential_path}")
                            if os.path.exists(potential_path):
                                current_video_file_for_item = potential_path
                                print(f"DEBUG: Using fallback file: {potential_path}")
                                break
                else:
                    print(f"DEBUG: Output directory does not exist: {output_dir}")

        if current_video_file_for_item:
            # --- START METADATA SAVING ---
            params_for_meta = {
                "model_type": "InfiniteTalk",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "cond_image": cond_image,
                "audio_person1": audio_person1,
                "audio_person2": audio_person2,
                "size": size,
                "mode": mode,
                "frame_num": frame_num,
                "motion_frame": motion_frame,
                "sample_steps": sample_steps,
                "sample_shift": sample_shift,
                "text_guide_scale": text_guide_scale,
                "audio_guide_scale": audio_guide_scale,
                "seed": current_seed,
                "audio_type": audio_type,
                "scene_seg": scene_seg,
                "dit_path": dit_path,
                "infinitetalk_dir": infinitetalk_dir,
                "use_teacache": use_teacache,
                "teacache_thresh": teacache_thresh,
                "use_apg": use_apg,
                "apg_momentum": apg_momentum,
                "apg_norm_thresh": apg_norm_thresh,
            }
            
            if lora_weights_paths:
                params_for_meta["lora_weights"] = lora_weights_paths
                params_for_meta["lora_multipliers"] = lora_multipliers_values
            
            metadata_filename = current_video_file_for_item.replace('.mp4', '_params.json')
            try:
                with open(metadata_filename, 'w') as meta_file:
                    json.dump(params_for_meta, meta_file, indent=2)
            except Exception as e:
                print(f"Error saving metadata: {e}")
            # --- END METADATA SAVING ---
            
            all_generated_videos.append((current_video_file_for_item, f"InfiniteTalk - Seed: {current_seed}"))
            status_text = f"Item {i+1}/{batch_size} (Seed: {current_seed}) - Completed"
            progress_text_update = f"Saved: {os.path.basename(current_video_file_for_item)}"
            
            # Yield current generated videos and clear preview
            yield all_generated_videos.copy(), None, status_text, progress_text_update
        else:
            status_text = f"Item {i+1}/{batch_size} - No output detected"
            progress_text_update = "Check console for errors"
            yield all_generated_videos.copy(), None, status_text, progress_text_update

        clear_cuda_cache()
        time.sleep(0.2)
        
    yield all_generated_videos, None, "InfiniteTalk Batch complete.", ""

def save_framepack_defaults(*values):
    os.makedirs(UI_CONFIGS_DIR, exist_ok=True)
    settings_to_save = {}
    for i, key in enumerate(framepack_ui_default_keys):
        settings_to_save[key] = values[i]

    try:
        with open(FRAMEPROK_DEFAULTS_FILE, 'w') as f:
            json.dump(settings_to_save, f, indent=2)
        return "FramePack defaults saved successfully."
    except Exception as e:
        return f"Error saving FramePack defaults: {e}"

def load_framepack_defaults(request: gr.Request): # request can be used to see if it's initial load
    if not os.path.exists(FRAMEPROK_DEFAULTS_FILE):
        if request: # If it's not an initial load (i.e., button click)
            return [gr.update()] * len(framepack_ui_default_keys) + ["No defaults file found."]
        else: # Initial load, let Gradio use its defined values
            return [gr.update()] * len(framepack_ui_default_keys) + [""]


    try:
        with open(FRAMEPROK_DEFAULTS_FILE, 'r') as f:
            loaded_settings = json.load(f)
    except Exception as e:
        return [gr.update()] * len(framepack_ui_default_keys) + [f"Error loading defaults: {e}"]

    updates = []
    for i, key in enumerate(framepack_ui_default_keys):
        component = framepack_ui_default_components_ORDERED_LIST[i]
        default_value_from_component = None
        # Try to get the 'value' attribute from the component definition
        if hasattr(component, 'value'):
            default_value_from_component = component.value
        elif isinstance(component, gr.State): # gr.State has value in its init
             pass # For gr.State, default is often None or handled by its initial value

        # Use loaded setting if available, otherwise the component's initial value
        value_to_set = loaded_settings.get(key, default_value_from_component)

        # Special handling for LoRA dropdowns to ensure 'None' is a valid choice
        if "lora_weight" in key: # This covers framepack_lora_weight_1 to framepack_lora_weight_4
            lora_choices = get_lora_options(loaded_settings.get("framepack_lora_folder", "lora"))
            if value_to_set not in lora_choices:
                value_to_set = "None" # Default to "None" if saved value is invalid
            updates.append(gr.update(choices=lora_choices, value=value_to_set))
        else:
            updates.append(gr.update(value=value_to_set))

    return updates + ["FramePack defaults loaded successfully."]

def refresh_lora_dropdowns_simple(lora_folder: str) -> List[gr.update]:
    """Refreshes LoRA choices, always defaulting the selection to 'None'."""
    new_choices = get_lora_options(lora_folder)
    results = []
    print(f"Refreshing LoRA dropdowns. Found choices: {new_choices}") # Debug print
    for i in range(4): # Update all 4 slots
        results.extend([
            gr.update(choices=new_choices, value="None"), # Always reset value to None
            gr.update(value=1.0) # Reset multiplier
        ])
    return results
def _extract_path_from_gallery_item(item: Any) -> Optional[str]:
    """
    Helper to extract a string file path from various Gradio component item formats.
    Verifies that the extracted path exists.
    """
    path_to_check = None
    if isinstance(item, str):
        path_to_check = item
    elif isinstance(item, Path):
        path_to_check = str(item)
    elif isinstance(item, tuple) and len(item) > 0:
        path_obj_in_tuple = item[0]
        if isinstance(path_obj_in_tuple, str):
            path_to_check = path_obj_in_tuple
        elif isinstance(path_obj_in_tuple, Path):
            path_to_check = str(path_obj_in_tuple)
    elif hasattr(item, 'name'): 
        name_attr = getattr(item, 'name', None)
        if isinstance(name_attr, str):
            path_to_check = name_attr
        elif isinstance(name_attr, Path):
            path_to_check = str(name_attr)

    if path_to_check and os.path.exists(path_to_check):
        return os.path.abspath(path_to_check)
    else:
        print(f"DEBUG _extract_path_from_gallery_item: Could not get valid path from {repr(item)} or path '{path_to_check}' does not exist.")
    return None

def combine_image_sets(current_gallery_value: Optional[List[Any]],
                       newly_uploaded_paths_from_button: Optional[List[str]]) -> List[str]:
    all_valid_paths = []
    if current_gallery_value:
        for item in current_gallery_value:
            path = _extract_path_from_gallery_item(item)
            if path:
                all_valid_paths.append(path)
    if newly_uploaded_paths_from_button:
        for path_str in newly_uploaded_paths_from_button:
            if isinstance(path_str, str) and os.path.exists(path_str):
                all_valid_paths.append(os.path.abspath(path_str))

    unique_final_paths = []
    seen_paths = set()
    for path in all_valid_paths:
        if path not in seen_paths:
            unique_final_paths.append(path)
            seen_paths.add(path)
    return unique_final_paths

def phantom_generate_video(
    prompt_text: str,
    negative_prompt_text: str,
    input_images_list: List[str], # List of file paths
    width_val: int,
    height_val: int,
    video_length_frames: int,
    fps_val: int,
    infer_steps_val: int,
    flow_shift_val: float,
    # S2V specific guidance scales for Phantom
    guide_scale_img_val: float,
    guide_scale_text_val: float,
    # Main CFG scale (might not be used by S2V if its CFG is hardcoded or different)
    cfg_scale_main_val: float,
    seed_val: int,
    task_str: str, # Should be an S2V task
    dit_folder_str: str, # Folder for DiT
    phantom_dit_model_path: str, # Specific Phantom DiT path (--phantom_ckpt in script)
    vae_path_str: str,
    t5_path_str: str,
    # clip_path_str: str, # CLIP not directly used for S2V image features
    save_path_str: str,
    output_type_str: str,
    sample_solver_str: str,
    attn_mode_str: str,
    block_swap_val: int,
    fp8_val: bool,
    fp8_scaled_val: bool,
    fp8_t5_val: bool,
    lora_folder_str: str,
    slg_layers_str: str,
    slg_start_val: float,
    slg_end_val: float,
    lora1_str: str, lora2_str: str, lora3_str: str, lora4_str: str,
    lora1_mult: float, lora2_mult: float, lora3_mult: float, lora4_mult: float,
    enable_cfg_skip_val: bool,
    cfg_skip_mode_str: str,
    cfg_apply_ratio_val: float,
    enable_preview_val: bool,
    preview_steps_val: int
) -> Generator[Tuple[List[Tuple[str, str]], List[str], str, str], None, None]:
    global stop_event
    stop_event.clear()

    videos_gallery = []
    previews_gallery = []
    status_text_update = "Preparing Phantom generation..."
    progress_text_update = ""
    yield videos_gallery, previews_gallery, status_text_update, progress_text_update

    if not input_images_list:
        yield [], [], "Error: No input images provided for Phantom.", "Please select reference images."
        return

    ref_images_arg_str = ",".join(input_images_list) # Join paths with comma for --ref_images

    current_seed = seed_val
    if seed_val == -1:
        current_seed = random.randint(0, 2**32 - 1)

    # Construct path to the Phantom DiT model
    # The `phantom_dit_model_path` is already the full path or filename from the dropdown
    # If it's just a filename, `dit_folder_str` is used to prefix it.
    actual_phantom_dit_path = os.path.join(dit_folder_str, phantom_dit_model_path) \
        if not os.path.isabs(phantom_dit_model_path) and dit_folder_str \
        else phantom_dit_model_path

    if not os.path.exists(actual_phantom_dit_path):
        yield [], [], f"Error: Phantom DiT model not found at {actual_phantom_dit_path}", ""
        return


    # Prepare environment and command
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    clear_cuda_cache()

    run_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
    unique_preview_suffix = f"phantom_{run_id}"

    command = [
        sys.executable,
        "phantom_generate_video.py", # We're using the multi-purpose script
        "--task", str(task_str), # e.g., "s2v-14B"
        "--prompt", str(prompt_text),
        "--video_size", str(height_val), str(width_val),
        "--video_length", str(video_length_frames),
        "--fps", str(fps_val),
        "--infer_steps", str(infer_steps_val),
        "--save_path", str(save_path_str),
        "--seed", str(current_seed),
        "--flow_shift", str(flow_shift_val),
        # S2V uses specific guidance scales
        "--guide_scale_img", str(guide_scale_img_val),
        "--guide_scale_text", str(guide_scale_text_val),
        # Pass the main CFG scale (script might use it or S2V logic might override)
        "--guidance_scale", str(cfg_scale_main_val),
        "--output_type", str(output_type_str),
        "--sample_solver", str(sample_solver_str),
        "--attn_mode", str(attn_mode_str),
        "--blocks_to_swap", str(block_swap_val),
        # Model paths
        "--phantom_ckpt", str(actual_phantom_dit_path), # Pass Phantom DiT here
        "--vae", str(vae_path_str),
        "--t5", str(t5_path_str),
        # "--clip", str(clip_path_str), # CLIP not typically used for S2V image features
        # Reference images for S2V
        "--ref_images", ref_images_arg_str,
    ]

    if negative_prompt_text:
        command.extend(["--negative_prompt", str(negative_prompt_text)])
    if fp8_val: command.append("--fp8")
    if fp8_scaled_val: command.append("--fp8_scaled")
    if fp8_t5_val: command.append("--fp8_t5")

    if slg_layers_str and str(slg_layers_str).strip() and str(slg_layers_str).lower() != "none":
        command.extend(["--slg_layers", str(slg_layers_str).strip()])
        if slg_start_val is not None: command.extend(["--slg_start", str(slg_start_val)])
        if slg_end_val is not None: command.extend(["--slg_end", str(slg_end_val)])

    # LoRAs
    valid_loras_paths = []
    valid_loras_mults_str = []
    lora_inputs = [
        (lora1_str, lora1_mult), (lora2_str, lora2_mult),
        (lora3_str, lora3_mult), (lora4_str, lora4_mult)
    ]
    if lora_folder_str and os.path.exists(lora_folder_str):
        for name, mult in lora_inputs:
            if name and name != "None":
                path = os.path.join(lora_folder_str, name)
                if os.path.exists(path):
                    valid_loras_paths.append(path)
                    valid_loras_mults_str.append(str(mult))
                else:
                    print(f"Warning: LoRA file not found: {path}")
    if valid_loras_paths:
        command.extend(["--lora_weight"] + valid_loras_paths)
        command.extend(["--lora_multiplier"] + valid_loras_mults_str)

    if enable_cfg_skip_val and cfg_skip_mode_str != "none":
        command.extend(["--cfg_skip_mode", str(cfg_skip_mode_str)])
        command.extend(["--cfg_apply_ratio", str(cfg_apply_ratio_val)])

    if enable_preview_val and preview_steps_val > 0:
        command.extend(["--preview", str(preview_steps_val)])
        command.extend(["--preview_suffix", unique_preview_suffix])

    command_str_list = [str(item) for item in command]
    print(f"Running Phantom Command: {' '.join(command_str_list)}")

    process = subprocess.Popen(
        command_str_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True, encoding='utf-8', errors='replace', bufsize=1
    )

    # --- Preview file paths based on unique_preview_suffix ---
    preview_base_dir = os.path.join(save_path_str, "previews") # Script saves previews here
    preview_mp4_gen_path = os.path.join(preview_base_dir, f"latent_preview_{unique_preview_suffix}.mp4")
    preview_png_glob_pattern = os.path.join(preview_base_dir, f"latent_preview_step_*_{unique_preview_suffix}.png")

    last_preview_mtime = 0
    current_preview_yield_list = [] # Store paths of previews to yield

    current_video_file_for_item = None
    status_text_update = f"Processing Phantom (Seed: {current_seed})"

    for line in iter(process.stdout.readline, ''):
        if stop_event.is_set():
            try: process.terminate(); process.wait(timeout=5)
            except: process.kill(); process.wait()
            yield videos_gallery, [], "Generation stopped by user.", ""
            return

        line_strip = line.strip()
        if not line_strip: continue
        print(f"PHANTOM_SUBPROCESS: {line_strip}")
        progress_text_update = line_strip

        tqdm_match = re.search(r'(\d+)\%\|.+\| (\d+/\d+) \[(\d{2}:\d{2})<(\d{2}:\d{2})', line_strip)
        video_saved_match = re.search(r"Video saved to:\s*(.*\.mp4)", line_strip) # Common save message

        if tqdm_match:
            percentage = tqdm_match.group(1)
            steps_iter = tqdm_match.group(2)
            time_elapsed = tqdm_match.group(3)
            time_remaining = tqdm_match.group(4)
            progress_text_update = f"Step {steps_iter} ({percentage}%) | ETA: {time_remaining}"
            status_text_update = f"Phantom Denoising (Seed: {current_seed})"

        elif video_saved_match:
            found_path = video_saved_match.group(1).strip()
            if os.path.exists(found_path):
                current_video_file_for_item = found_path
                progress_text_update = f"Finalizing: {os.path.basename(found_path)}"
                status_text_update = f"Phantom Saved (Seed: {current_seed})"
            else:
                print(f"Warning: Video path from log not found: {found_path}")

        # Preview handling (similar to WanX-i2v)
        preview_updated_this_line = False
        if enable_preview_val:
            # Check MP4 (full video preview from script)
            if os.path.exists(preview_mp4_gen_path):
                current_mtime_mp4 = os.path.getmtime(preview_mp4_gen_path)
                if current_mtime_mp4 > last_preview_mtime:
                    current_preview_yield_list = [preview_mp4_gen_path] # Replace list with MP4
                    last_preview_mtime = current_mtime_mp4
                    preview_updated_this_line = True
                    print(f"DEBUG Phantom: MP4 Preview updated - {preview_mp4_gen_path}")

            # Check PNGs (step previews) - only if MP4 wasn't updated this line
            if not preview_updated_this_line:
                # This part is tricky because many PNGs can be generated.
                # We want to show the *latest* set of step previews.
                # A simpler approach for Gradio Gallery is to show the MP4 if available,
                # or a single latest PNG. Let's stick to the MP4 from wan_generate_video.py for now
                # as it's more straightforward. If the script only outputs PNGs, this needs rework.
                # Assuming `--preview` in wan_generate_video.py generates an MP4.
                # If it generates PNGs, we might need to glob and sort them.
                # For now, `current_preview_yield_list` is primarily for the MP4.
                pass

        yield videos_gallery, current_preview_yield_list, status_text_update, progress_text_update

    process.stdout.close()
    return_code = process.wait()
    clear_cuda_cache()

    if return_code == 0 and current_video_file_for_item and os.path.exists(current_video_file_for_item):
        videos_gallery.append((current_video_file_for_item, f"Phantom - Seed: {current_seed}"))
        status_text_update = f"Phantom (Seed: {current_seed}) - Completed"
        progress_text_update = f"Saved: {os.path.basename(current_video_file_for_item)}"
        # Metadata parameters
        params_for_meta = {
            "prompt": prompt_text, "negative_prompt": negative_prompt_text,
            "ref_images": [os.path.basename(p) for p in input_images_list],
            "width": width_val, "height": height_val, "video_length": video_length_frames, "fps": fps_val,
            "infer_steps": infer_steps_val, "flow_shift": flow_shift_val,
            "guide_scale_img": guide_scale_img_val, "guide_scale_text": guide_scale_text_val,
            "cfg_scale_main": cfg_scale_main_val,
            "seed": current_seed, "task": task_str, "dit_path": actual_phantom_dit_path,
            "vae_path": vae_path_str, "t5_path": t5_path_str,
            "save_path": save_path_str, "output_type": output_type_str, "sample_solver": sample_solver_str,
            "attn_mode": attn_mode_str, "block_swap": block_swap_val,
            "fp8": fp8_val, "fp8_scaled": fp8_scaled_val, "fp8_t5": fp8_t5_val,
            "lora_weights": [lora1_str, lora2_str, lora3_str, lora4_str],
            "lora_multipliers": [lora1_mult, lora2_mult, lora3_mult, lora4_mult],
            "slg_layers": slg_layers_str, "slg_start": slg_start_val, "slg_end": slg_end_val,
        }
        try:
            add_metadata_to_video(current_video_file_for_item, params_for_meta)
        except Exception as meta_err:
            print(f"Warning: Failed to add metadata to {current_video_file_for_item}: {meta_err}")

        yield videos_gallery.copy(), [], status_text_update, progress_text_update # Clear previews on success
    else:
        status_text_update = f"Phantom (Seed: {current_seed}) - Failed (Code: {return_code})"
        progress_text_update = "Subprocess failed. Check console."
        yield videos_gallery.copy(), [], status_text_update, progress_text_update

def phantom_batch_handler(
    prompt_text: str,
    negative_prompt_text: str,
    input_images_list_files: List[Any], # This is the value from phantom_input_images gallery
    width_val: int,
    height_val: int,
    video_length_frames: int,
    fps_val: int,
    infer_steps_val: int,
    flow_shift_val: float,
    guide_scale_img_val: float,
    guide_scale_text_val: float,
    cfg_scale_main_val: float,
    seed_val: int,
    batch_size_val: int,
    task_str: str,
    dit_folder_str: str,
    phantom_dit_model_path_str: str,
    vae_path_str: str,
    t5_path_str: str,
    save_path_str: str,
    output_type_str: str,
    sample_solver_str: str,
    attn_mode_str: str,
    block_swap_val: int,
    fp8_val: bool,
    fp8_scaled_val: bool,
    fp8_t5_val: bool,
    lora_folder_str: str,
    slg_layers_str: str,
    slg_start_val: float,
    slg_end_val: float,
    enable_cfg_skip_val: bool,
    cfg_skip_mode_str: str,
    cfg_apply_ratio_val: float,
    enable_preview_val: bool,
    preview_steps_val: int,
    lora1_str: str, lora2_str: str, lora3_str: str, lora4_str: str,
    lora1_mult: float, lora2_mult: float, lora3_mult: float, lora4_mult: float
):
    global stop_event
    stop_event.clear()

    actual_input_image_paths = []
    if input_images_list_files:
        # print(f"DEBUG phantom_batch_handler - Received input_images_list_files: {input_images_list_files}")
        for item_data in input_images_list_files:
            # Use the same robust extraction logic here
            path_str = _extract_path_from_gallery_item(item_data)
            if path_str:
                actual_input_image_paths.append(path_str)
            else:
                # This is where your warning was originating
                print(f"Warning: Could not resolve path for file data (in phantom_batch_handler): {item_data}")
    
    # print(f"DEBUG phantom_batch_handler - Extracted actual_input_image_paths: {actual_input_image_paths}")

    if not actual_input_image_paths:
        yield [], [], "Error: No valid input image paths provided for Phantom.", "Please ensure selected reference images exist on the server."
        return

    all_generated_videos = []
    all_previews = []
    status_text = "Preparing Phantom batch..."
    progress_text = ""
    yield all_generated_videos, all_previews, status_text, progress_text

    for i in range(int(batch_size_val)):
        if stop_event.is_set():
            yield all_generated_videos, all_previews, "Phantom Batch stopped by user.", ""
            return

        current_seed_for_item = seed_val
        if seed_val == -1:
            current_seed_for_item = random.randint(0, 2**32 - 1)
        elif int(batch_size_val) > 1:
            current_seed_for_item = seed_val + i

        status_text = f"Processing Phantom Item {i+1}/{batch_size_val} (Seed: {current_seed_for_item})"
        yield all_generated_videos.copy(), [], status_text, "Starting item..."

        item_videos = []
        item_previews_temp = []
        last_status_for_item = status_text
        last_progress_for_item = "Generating..."

        single_gen = phantom_generate_video(
            prompt_text, negative_prompt_text, actual_input_image_paths, # Use the processed list
            width_val, height_val, video_length_frames, fps_val, infer_steps_val,
            flow_shift_val, guide_scale_img_val, guide_scale_text_val, cfg_scale_main_val,
            current_seed_for_item, task_str, dit_folder_str, phantom_dit_model_path_str,
            vae_path_str, t5_path_str, save_path_str, output_type_str,
            sample_solver_str, attn_mode_str, block_swap_val,
            fp8_val, fp8_scaled_val, fp8_t5_val, lora_folder_str,
            slg_layers_str, slg_start_val, slg_end_val,
            lora1_str, lora2_str, lora3_str, lora4_str,
            lora1_mult, lora2_mult, lora3_mult, lora4_mult,
            enable_cfg_skip_val, cfg_skip_mode_str, cfg_apply_ratio_val,
            enable_preview_val, preview_steps_val
        )

        for videos_update, previews_update, status_upd, progress_upd in single_gen:
            item_videos = videos_update
            item_previews_temp = previews_update
            last_status_for_item = f"Item {i+1}/{batch_size_val}: {status_upd}"
            last_progress_for_item = progress_upd
            yield all_generated_videos + ([item_videos[-1]] if item_videos and item_videos[-1] not in all_generated_videos else []), \
                  item_previews_temp, last_status_for_item, last_progress_for_item

        if item_videos and item_videos[-1] not in all_generated_videos:
            all_generated_videos.append(item_videos[-1])

        clear_cuda_cache()
        time.sleep(0.2)

    yield all_generated_videos, [], "Phantom Batch complete.", ""

def process_framepack_extension_video(
    input_video: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    batch_count: int,
    fpe_use_normal_framepack: bool,
    fpe_end_frame: Optional[str],
    fpe_end_frame_weight: float,    
    resolution_max_dim: int,
    total_second_length: float,
    latent_window_size: int,
    steps: int,
    cfg_scale: float, # Maps to --cfg
    distilled_guidance_scale: float, # Maps to --gs
    # rs_scale: float, # --rs, usually 0.0, can be fixed or advanced option
    gpu_memory_preservation: float,
    use_teacache: bool,
    no_resize: bool,
    mp4_crf: int,
    num_clean_frames: int,
    vae_batch_size: int,
    save_path: str, # Maps to --output_dir
    # Model Paths
    fpe_transformer_path: str, # DiT
    fpe_vae_path: str,
    fpe_text_encoder_path: str, # TE1
    fpe_text_encoder_2_path: str, # TE2
    fpe_image_encoder_path: str,
    # Advanced performance
    fpe_attn_mode: str,
    fpe_fp8_llm: bool,
    fpe_vae_chunk_size: Optional[int],
    fpe_vae_spatial_tile_sample_min_size: Optional[int],
    # LoRAs
    fpe_lora_folder: str,
    fpe_lora_weight_1: str, fpe_lora_mult_1: float,
    fpe_lora_weight_2: str, fpe_lora_mult_2: float,
    fpe_lora_weight_3: str, fpe_lora_mult_3: float,
    fpe_lora_weight_4: str, fpe_lora_mult_4: float,
    # Preview
    fpe_enable_preview: bool,
    fpe_preview_interval: int, # This arg is not used by f1_video_cli_local.py
    fpe_extension_only: bool,
    fpe_start_guidance_image: Optional[str],
    fpe_start_guidance_image_clip_weight: float,
    fpe_use_guidance_image_as_first_latent: bool,
    *args: Any # For future expansion or unmapped params, not strictly needed here
) -> Generator[Tuple[List[Tuple[str, str]], Optional[str], str, str], None, None]:
    global stop_event, skip_event
    stop_event.clear()
    skip_event.clear() # Assuming skip_event might be used for batch items

    if not input_video or not os.path.exists(input_video):
        yield [], None, "Error: Input video for extension not found.", ""
        return

    if not save_path or not save_path.strip():
        save_path = "outputs/framepack_extensions" # Default save path for extensions
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare LoRA arguments
    lora_weights_paths = []
    lora_multipliers_values = []
    lora_params_ui = [
        (fpe_lora_weight_1, fpe_lora_mult_1), (fpe_lora_weight_2, fpe_lora_mult_2),
        (fpe_lora_weight_3, fpe_lora_mult_3), (fpe_lora_weight_4, fpe_lora_mult_4)
    ]
    if fpe_lora_folder and os.path.exists(fpe_lora_folder):
        for weight_name, mult_val in lora_params_ui:
            if weight_name and weight_name != "None":
                lora_path = os.path.join(fpe_lora_folder, weight_name)
                if os.path.exists(lora_path):
                    lora_weights_paths.append(lora_path)
                    lora_multipliers_values.append(str(mult_val))
                else:
                    print(f"Warning: LoRA file not found: {lora_path}")

    all_generated_videos = [] 
    script_to_use = "f_video_end_cli_local.py" if fpe_use_normal_framepack else "f1_video_cli_local.py"
    model_type_str = "Normal FramePack" if fpe_use_normal_framepack else "FramePack F1"
    print(f"Using {model_type_str} model for extension via script: {script_to_use}")    
    
    for i in range(batch_count):
        if stop_event.is_set():
            yield all_generated_videos, None, "Generation stopped by user.", ""
            return
        skip_event.clear() 

        current_seed_val = seed
        if seed == -1:
            current_seed_val = random.randint(0, 2**32 - 1)
        elif batch_count > 1:
            current_seed_val = seed + i
        
        # This run_id is not directly used for preview file naming by f1_video_cli_local.py
        # as it constructs its own job_id based filenames for section previews.
        # run_id = f"{int(time.time())}_{random.randint(1000, 9999)}_ext_s{current_seed_val}"
        
        current_preview_yield_path = None 
        last_preview_section_processed = -1 
        
        status_text = f"Processing Extension {i + 1}/{batch_count} (Seed: {current_seed_val})"
        progress_text = "Preparing extension subprocess..."
        yield all_generated_videos, current_preview_yield_path, status_text, progress_text

        command = [
            sys.executable, script_to_use,
            "--input_video", str(input_video),
            "--prompt", str(prompt),
            "--n_prompt", str(negative_prompt),
            "--seed", str(current_seed_val),
            "--resolution_max_dim", str(resolution_max_dim),
            "--total_second_length", str(total_second_length), # Script uses this for *additional* length
            "--latent_window_size", str(latent_window_size),
            "--steps", str(steps),
            "--cfg", str(cfg_scale),
            "--gs", str(distilled_guidance_scale),
            "--rs", "0.0", 
            "--gpu_memory_preservation", str(gpu_memory_preservation),
            "--mp4_crf", str(mp4_crf),
            "--num_clean_frames", str(num_clean_frames),
            "--vae_batch_size", str(vae_batch_size),
            "--output_dir", str(save_path), 
            "--dit", str(fpe_transformer_path), "--vae", str(fpe_vae_path),
            "--text_encoder1", str(fpe_text_encoder_path), "--text_encoder2", str(fpe_text_encoder_2_path),
            "--image_encoder", str(fpe_image_encoder_path),
            "--attn_mode", str(fpe_attn_mode),
        ]
        if use_teacache: command.append("--use_teacache")
        if no_resize: command.append("--no_resize")
        if fpe_fp8_llm: command.append("--fp8_llm") # Though F1 script might not use this
        if fpe_vae_chunk_size is not None and fpe_vae_chunk_size > 0:
            command.extend(["--vae_chunk_size", str(fpe_vae_chunk_size)])
        if fpe_vae_spatial_tile_sample_min_size is not None and fpe_vae_spatial_tile_sample_min_size > 0:
            command.extend(["--vae_spatial_tile_sample_min_size", str(fpe_vae_spatial_tile_sample_min_size)])

        if lora_weights_paths:
            command.extend(["--lora_weight"] + lora_weights_paths)
            command.extend(["--lora_multiplier"] + lora_multipliers_values)
        if fpe_extension_only:
            command.append("--extension_only")
        # Script-specific arguments
        if fpe_use_normal_framepack:
            if fpe_fp8_llm: # Normal FP script uses this
                 command.append("--fp8_llm")
            if fpe_end_frame and os.path.exists(fpe_end_frame):
                command.extend(["--end_frame", str(fpe_end_frame)])
                command.extend(["--end_frame_weight", str(fpe_end_frame_weight)])
        else: 
            if fpe_start_guidance_image and os.path.exists(fpe_start_guidance_image):
                command.extend(["--start_guidance_image", str(fpe_start_guidance_image)])
                command.extend(["--start_guidance_image_clip_weight", str(fpe_start_guidance_image_clip_weight)])
                if fpe_use_guidance_image_as_first_latent:
                    command.append("--use_guidance_image_as_first_latent")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        print(f"Running FramePack-Extension Command: {' '.join(command)}")

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1, universal_newlines=True)
        
        # Regex patterns based on script
        if fpe_use_normal_framepack: # This means f_video_end_cli_local.py was used
            final_video_path_regex = re.compile(r"Final (?:extended video saved:|extension-only video saved:) (.*\.mp4)")
            # Regex for "--- Generating Extension: ... Section X / Y (backward) ---"
            fpe_section_progress_regex = re.compile(r"--- Generating Extension: .*?: Section\s+(\d+)\s*/\s*(\d+)\s+\(backward\)")
            tqdm_cli_progress_regex = re.compile(r"Sampling Extension Section .*?:\s*(\d+)%\|.*?\|\s*(\d+/\d+)\s*\[([^<]+)<([^,]+),") 
        else: # F1 script (f1_video_cli_local.py) was used
            final_video_path_regex = re.compile(r"Final (?:extension-only )?video for seed \d+.*? saved as: (.*\.mp4)")
            fpe_section_progress_regex = re.compile(r"--- F1 Extension: .*?: Section (\d+)\s*/\s*(\d+) ---")
            tqdm_cli_progress_regex = re.compile(r"Sampling Extension Section .*?:\s*(\d+)%\|.*?\|\s*(\d+/\d+)\s*\[([^<]+)<([^,]+),")
        fpe_preview_saved_regex = re.compile(r"MP4 Preview for section (\d+) saved: (.*\.mp4)")
        
        current_video_file_for_item = None
        current_section_being_processed = 0 
        total_sections_from_log = 0

        for line in iter(process.stdout.readline, ''):
            if stop_event.is_set():
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                except Exception as e: print(f"Error terminating FPE subprocess: {e}")
                yield all_generated_videos, None, "Generation stopped by user.", ""
                return
            if skip_event.is_set() and batch_count > 1:
                print(f"Skip signal received for FPE batch item {i+1}. Terminating subprocess...")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                except Exception as e: print(f"Error terminating FPE subprocess during skip: {e}")
                skip_event.clear()
                yield all_generated_videos, current_preview_yield_path, f"Skipping FPE item {i+1}/{batch_count}...", ""
                break 

            line_strip = line.strip()
            if not line_strip:
                continue
            print(f"FPE_SUBPROCESS: {line_strip}")
            
            progress_text_update = line_strip 
            
            section_match = fpe_section_progress_regex.search(line_strip)
            tqdm_match_cli = tqdm_cli_progress_regex.search(line_strip)
            final_video_match = final_video_path_regex.search(line_strip)
            preview_saved_match = fpe_preview_saved_regex.search(line_strip)

            if preview_saved_match and fpe_enable_preview:
                saved_section_num = int(preview_saved_match.group(1))
                preview_mp4_path_from_log = preview_saved_match.group(2).strip()
                if os.path.exists(preview_mp4_path_from_log) and saved_section_num > last_preview_section_processed:
                    current_preview_yield_path = preview_mp4_path_from_log # Yield clean path
                    last_preview_section_processed = saved_section_num
                    print(f"DEBUG FPE: MP4 Preview updated from log - {current_preview_yield_path}")
                # This log usually comes *after* the section info, so status might already be updated

            if section_match:
                current_section_being_processed = int(section_match.group(1))
                total_sections_from_log = int(section_match.group(2))
                status_text = f"Extending Video {i + 1}/{batch_count} (Seed: {current_seed_val}) - Section {current_section_being_processed}/{total_sections_from_log}"
                progress_text_update = f"Starting Section {current_section_being_processed}..."
                # Fallback logic for preview (if enabled and explicit log was missed)
                # This is less likely to be needed if fpe_preview_saved_regex is robust
                if fpe_enable_preview and current_section_being_processed > 1:
                    section_to_check_for_preview = current_section_being_processed - 1
                    if section_to_check_for_preview > last_preview_section_processed:
                        # Construct the expected preview filename based on f1_video_cli_local.py's naming
                        # It uses a job_id that includes seed, resolution, etc. We don't know the exact job_id here.
                        # Relying on "MP4 Preview for section X saved:" log is more reliable.
                        # For a fallback, we could glob for *partX*.mp4, but that's risky.
                        # For now, this fallback is removed as the primary log line should be sufficient.
                        pass


            elif tqdm_match_cli:
                percentage = tqdm_match_cli.group(1)
                steps_iter_total = tqdm_match_cli.group(2)
                time_elapsed = tqdm_match_cli.group(3).strip()
                time_remaining = tqdm_match_cli.group(4).strip()
                # Ensure total_sections_from_log is not zero before using in f-string
                total_sections_display = total_sections_from_log if total_sections_from_log > 0 else "?"
                progress_text_update = f"Section {current_section_being_processed}/{total_sections_display} - Step {steps_iter_total} ({percentage}%) | ETA: {time_remaining}"
                status_text = f"Extending Video {i + 1}/{batch_count} (Seed: {current_seed_val}) - Sampling Section {current_section_being_processed}"
            
            elif final_video_match:
                found_video_path = final_video_match.group(1).strip()
                if os.path.exists(found_video_path):
                    current_video_file_for_item = found_video_path
                    progress_text_update = f"Finalizing: {os.path.basename(current_video_file_for_item)}"
                    status_text = f"Extension {i + 1}/{batch_count} (Seed: {current_seed_val}) - Saved"
                else:
                    print(f"Warning FPE: Final video path from log not found: {found_video_path}")
            
            yield all_generated_videos, current_preview_yield_path, status_text, progress_text_update
        
        process.stdout.close()
        return_code = process.wait()

        if return_code == 0 and current_video_file_for_item and os.path.exists(current_video_file_for_item):
            all_generated_videos.append((current_video_file_for_item, f"Extended - Seed: {current_seed_val}"))
            status_text = f"Extension {i + 1}/{batch_count} (Seed: {current_seed_val}) - Completed and Added"
            progress_text = f"Saved: {os.path.basename(current_video_file_for_item)}"
            yield all_generated_videos.copy(), None, status_text, progress_text # Clear preview after item completion
        elif return_code != 0:
            status_text = f"Extension {i + 1}/{batch_count} (Seed: {current_seed_val}) - Failed (Code: {return_code})"
            progress_text = f"Subprocess failed. Check console for errors from f1_video_cli_local.py"
            yield all_generated_videos.copy(), None, status_text, progress_text
        else: # rc == 0 but no video path
            status_text = f"Extension {i + 1}/{batch_count} (Seed: {current_seed_val}) - Finished, but no video file confirmed."
            progress_text = "Check console logs from f1_video_cli_local.py for the saved path."
            yield all_generated_videos.copy(), None, status_text, progress_text

        # The F1 script already cleans up its intermediate _partX files.
        # No need for unique_preview_suffix based cleanup here for FPE.
    
    yield all_generated_videos, None, "FramePack-Extension Batch complete.", ""

def set_random_seed():
    """Returns -1 to set the seed input to random."""
    return -1

def get_step_from_preview_path(path): # Helper function
    # Extracts step number from preview filenames like latent_preview_step_005.mp4
    # or for framepack: latent_preview_section_002.mp4 (assuming sections for framepack)
    # Let's adjust for potential FramePack naming convention (using 'section' instead of 'step')
    base = os.path.basename(path)
    match_step = re.search(r"step_(\d+)", base)
    if match_step:
        return int(match_step.group(1))
    match_section = re.search(r"section_(\d+)", base) # Check for FramePack section naming
    if match_section:
        # Maybe treat sections differently? Or just return the number? Let's return number.
        return int(match_section.group(1))
    return -1 # Default if no number found

def parse_time_to_seconds(time_str: str) -> Optional[float]:
    """Parses MM:SS or HH:MM:SS into seconds."""
    parts = time_str.strip().split(':')
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, IndexError):
        return None
    return None

def process_framepack_video(
    prompt: str,
    negative_prompt: str,
    input_image: str, # Start image path
    input_end_frame: Optional[str], # End image path
    end_frame_influence: str,
    end_frame_weight: float,
    transformer_path: str,
    vae_path: str,
    text_encoder_path: str,
    text_encoder_2_path: str,
    image_encoder_path: str,
    target_resolution: Optional[int],
    framepack_width: Optional[int],
    framepack_height: Optional[int],
    original_dims_str: str, # This comes from framepack_original_dims state
    total_second_length: float,
    framepack_video_sections: Optional[int],
    fps: int,
    seed: int,
    steps: int,
    distilled_guidance_scale: float,
    cfg: float,
    rs: float,
    sample_solver: str,
    latent_window_size: int,
    fp8: bool,
    fp8_scaled: bool,
    fp8_llm: bool,
    blocks_to_swap: int,
    bulk_decode: bool,
    attn_mode: str,
    vae_chunk_size: Optional[int],
    vae_spatial_tile_sample_min_size: Optional[int],
    device: Optional[str],
    use_teacache: bool,
    teacache_steps: int,
    teacache_thresh: float,
    batch_size: int,
    save_path: str,
    lora_folder: str,
    enable_preview: bool,
    preview_every_n_sections: int,
    use_full_video_preview: bool,
    is_f1: bool,
    use_random_folder: bool,
    input_folder_path: str,
    *args: Any
) -> Generator[Tuple[List[Tuple[str, str]], Optional[str], str, str], None, None]:
    """Generate video using fpack_generate_video.py"""
    global stop_event
    stop_event.clear()

    if not save_path or not save_path.strip():
        print("Warning: save_path was empty, defaulting to 'outputs'")
        save_path = "outputs"

    num_section_controls = 4
    num_loras = 4
    secs_end = num_section_controls
    prompts_end = secs_end + num_section_controls
    images_end = prompts_end + num_section_controls
    lora_weights_end = images_end + num_loras
    lora_mults_end = lora_weights_end + num_loras

    framepack_secs = args[0:secs_end]
    framepack_sec_prompts = args[secs_end:prompts_end]
    framepack_sec_images = args[prompts_end:images_end]
    lora_weights_list = list(args[images_end:lora_weights_end])
    lora_multipliers_list = list(args[lora_weights_end:lora_mults_end])

    if not use_random_folder and not input_image and not any(img for img in framepack_sec_images if img):
        yield [], None, "Error: Input start image or at least one section image override is required when not using folder mode.", ""
        return

    if use_random_folder and (not input_folder_path or not os.path.isdir(input_folder_path)):
        yield [], None, f"Error: Random image folder path '{input_folder_path}' is invalid or not a directory.", ""
        return

    section_prompts_parts = []
    section_images_parts = []
    index_pattern = re.compile(r"^\d+(-\d+)?$")

    for idx_str, sec_prompt, sec_image in zip(framepack_secs, framepack_sec_prompts, framepack_sec_images):
        if not idx_str or not isinstance(idx_str, str) or not index_pattern.match(idx_str.strip()):
             if idx_str and idx_str.strip():
                 print(f"Warning: Invalid section index/range format '{idx_str}'. Skipping.")
             continue
        current_idx_str = idx_str.strip()
        if sec_prompt and sec_prompt.strip():
            section_prompts_parts.append(f"{current_idx_str}:{sec_prompt.strip()}")
        if sec_image and os.path.exists(sec_image):
             section_images_parts.append(f"{current_idx_str}:{sec_image}")

    final_prompt_arg = prompt
    if section_prompts_parts:
        final_prompt_arg = ";;;".join(section_prompts_parts)
        print(f"Using section prompt overrides: {final_prompt_arg}")

    final_image_path_arg = None
    if section_images_parts:
        final_image_path_arg = ";;;".join(section_images_parts)
        print(f"Using section image overrides for --image_path: {final_image_path_arg}")
    elif input_image:
        final_image_path_arg = input_image
        print(f"Using base input image for --image_path: {final_image_path_arg}")

    # These are batch-wide defaults if not overridden by folder mode + target res per item.
    batch_wide_final_height, batch_wide_final_width = None, None

    if framepack_width is not None and framepack_width > 0 and framepack_height is not None and framepack_height > 0:
        if framepack_width % 8 != 0 or framepack_height % 8 != 0:
             yield [], None, "Error: Explicit Width and Height must be divisible by 8.", ""
             return
        batch_wide_final_height = int(framepack_height)
        batch_wide_final_width = int(framepack_width)
        print(f"Using explicit dimensions for all items: H={batch_wide_final_height}, W={batch_wide_final_width}")
    elif target_resolution is not None and target_resolution > 0 and not use_random_folder:
        # This case applies if:
        # 1. Target resolution is set.
        # 2. We are NOT in random folder mode (so aspect ratio from UI image is reliable).
        if not original_dims_str: # original_dims_str comes from the UI input image
            yield [], None, "Error: Target Resolution selected (not in folder mode), but no UI input image provided for aspect ratio.", ""
            return
        try:
            orig_w, orig_h = map(int, original_dims_str.split('x'))
            if orig_w <= 0 or orig_h <= 0:
                yield [], None, "Error: Invalid original dimensions stored from UI image.", ""
                return
            bucket_dims = find_nearest_bucket(orig_h, orig_w, resolution=target_resolution)
            if bucket_dims:
                batch_wide_final_height, batch_wide_final_width = bucket_dims
                print(f"Using Target Resolution {target_resolution} with UI image aspect. Batch-wide bucket: H={batch_wide_final_height}, W={batch_wide_final_width}")
            else:
                yield [], None, f"Error: Could not find bucket for Target Res {target_resolution} and UI image aspect.", ""
                return
        except Exception as e:
            yield [], None, f"Error calculating bucket dimensions from UI image: {e}", ""
            return
    elif use_random_folder and target_resolution is not None and target_resolution > 0:
        # Folder mode with target resolution: resolution will be determined per item.
        # batch_wide_final_height and batch_wide_final_width remain None.
        print(f"Folder mode with Target Resolution {target_resolution}. Resolution will be determined per item.")
    elif not (framepack_width is not None and framepack_width > 0 and framepack_height is not None and framepack_height > 0) and \
         not (target_resolution is not None and target_resolution > 0):
        # This is the fallback if no resolution strategy is active for the batch.
        yield [], None, "Error: Resolution required. Please provide Target Resolution OR valid Width and Height (divisible by 8).", ""
        return

    all_videos = []
    if framepack_video_sections is not None and framepack_video_sections > 0:
        total_sections_estimate = framepack_video_sections
        print(f"Using user-defined total sections for UI: {total_sections_estimate}")
    else:
        total_sections_estimate_float = (total_second_length * fps) / (latent_window_size * 4)
        total_sections_estimate = int(max(round(total_sections_estimate_float), 1))
        print(f"Calculated total sections for UI from duration: {total_sections_estimate}")
    progress_text = f"Starting FramePack generation batch ({total_sections_estimate} estimated sections per video)..."
    status_text = "Preparing batch..."
    yield all_videos, None, status_text, progress_text

    valid_loras_paths = []
    valid_loras_mults = []
    if lora_folder and os.path.exists(lora_folder):
        for weight_name, mult in zip(lora_weights_list, lora_multipliers_list):
            if weight_name and weight_name != "None":
                 if os.path.isabs(weight_name):
                     lora_path = weight_name
                 else:
                     lora_path = os.path.join(lora_folder, weight_name)
                 if os.path.exists(lora_path):
                     valid_loras_paths.append(lora_path)
                     valid_loras_mults.append(str(mult))
                 else:
                     print(f"Warning: LoRA file not found: {lora_path}")

    for i in range(batch_size): # <<< START OF THE BATCH LOOP >>>
        if stop_event.is_set():
            yield all_videos, None, "Generation stopped by user.", ""
            return
        skip_event.clear()
        item_start_time = time.time()
        last_preview_mtime = 0

        run_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
        unique_preview_suffix = f"fpack_{run_id}"
        preview_base_dir = os.path.join(save_path, "previews") # Corrected: Look in 'previews' subdir
        preview_mp4_path = os.path.join(preview_base_dir, f"latent_preview_{unique_preview_suffix}.mp4") # Corrected
        preview_png_path = os.path.join(preview_base_dir, f"latent_preview_{unique_preview_suffix}.png")

        current_seed = seed
        if seed == -1: current_seed = random.randint(0, 2**32 - 1)
        elif batch_size > 1: current_seed = seed + i

        status_text = f"Generating video {i + 1} of {batch_size} (Seed: {current_seed})"
        progress_text_update = f"Item {i+1}/{batch_size}: Preparing..." # Renamed progress_text to progress_text_update for clarity
        current_video_path = None
        current_preview_yield_path = None
        current_input_image_for_item = input_image
        current_original_dims_str_for_item = original_dims_str # Use batch-wide original_dims_str initially

        if use_random_folder:
            progress_text_update = f"Item {i+1}/{batch_size}: Selecting random image..."
            yield all_videos.copy(), current_preview_yield_path, status_text, progress_text_update

            random_image_path, random_status = get_random_image_from_folder(input_folder_path)
            if random_image_path is None:
                error_msg = f"Error for item {i+1}/{batch_size}: {random_status}. Skipping."
                print(error_msg)
                yield all_videos.copy(), None, status_text, error_msg
                continue

            current_input_image_for_item = random_image_path
            progress_text_update = f"Item {i+1}/{batch_size}: Using random image: {os.path.basename(random_image_path)}"
            print(progress_text_update)
            yield all_videos.copy(), current_preview_yield_path, status_text, progress_text_update

            # Derive original_dims_str_for_item from the random image if using target resolution
            # and explicit UI W/H were not provided.
            if target_resolution is not None and target_resolution > 0 and \
               not (framepack_width is not None and framepack_width > 0 and framepack_height is not None and framepack_height > 0):
                try:
                    img_for_dims = Image.open(random_image_path)
                    rand_w, rand_h = img_for_dims.size
                    current_original_dims_str_for_item = f"{rand_w}x{rand_h}"
                    print(f"Folder mode item {i+1}: Using random image dims {current_original_dims_str_for_item} for target resolution bucketing.")
                except Exception as e:
                    error_msg = f"Error getting dims for random image {random_image_path}: {e}. Skipping item {i+1}."
                    print(error_msg)
                    yield all_videos.copy(), None, status_text, error_msg
                    continue
        
        final_image_path_arg_for_item = None
        if section_images_parts:
            final_image_path_arg_for_item = ";;;".join(section_images_parts)
            if current_input_image_for_item:
                has_section_0_override = any(part.strip().startswith("0:") for part in section_images_parts)
                if not has_section_0_override:
                    final_image_path_arg_for_item = f"0:{current_input_image_for_item};;;{final_image_path_arg_for_item}"
            print(f"Using section image overrides (potentially with prepended base) for --image_path (item {i+1}): {final_image_path_arg_for_item}")
        elif current_input_image_for_item:
            final_image_path_arg_for_item = current_input_image_for_item
            print(f"Using {'random' if use_random_folder else 'base'} input image as the primary for --image_path (item {i+1}): {final_image_path_arg_for_item}")
        
        if final_image_path_arg_for_item is None:
            yield [], None, f"Error for item {i+1}: No valid start image could be determined. Ensure an image is provided.", ""
            continue

        final_height_for_item, final_width_for_item = None, None

        # 1. Use batch-wide dimensions if they were set (from explicit UI W/H or target_res + UI image)
        if batch_wide_final_height is not None and batch_wide_final_width is not None:
            final_height_for_item = batch_wide_final_height
            final_width_for_item = batch_wide_final_width
            print(f"Item {i+1}: Using batch-wide dimensions: H={final_height_for_item}, W={final_width_for_item}")
        # 2. Else, if using target resolution (this implies folder mode, as other cases were handled above)
        elif target_resolution is not None and target_resolution > 0:
             if not current_original_dims_str_for_item: # This should now be populated for folder mode
                  yield [], None, f"Error for item {i+1}: Target Resolution selected, but no original dimensions available for aspect ratio.", ""
                  continue
             try:
                 orig_w_item, orig_h_item = map(int, current_original_dims_str_for_item.split('x'))
                 if orig_w_item <= 0 or orig_h_item <= 0:
                     yield [], None, f"Error for item {i+1}: Invalid original dimensions '{current_original_dims_str_for_item}'.", ""
                     continue
                 bucket_dims_item = find_nearest_bucket(orig_h_item, orig_w_item, resolution=target_resolution)
                 if bucket_dims_item:
                     final_height_for_item, final_width_for_item = bucket_dims_item
                     print(f"Item {i+1}: Using Target Resolution {target_resolution} with item-specific aspect from '{current_original_dims_str_for_item}'. Bucket: H={final_height_for_item}, W={final_width_for_item}")
                 else:
                     yield [], None, f"Error for item {i+1}: Could not find bucket for Target Res {target_resolution} and aspect {current_original_dims_str_for_item}.", ""
                     continue
             except Exception as e_res:
                 yield [], None, f"Error calculating bucket dimensions for item {i+1} ({current_original_dims_str_for_item}): {e_res}", ""
                 continue
        else:
            # This case should ideally not be hit if the initial batch-wide resolution checks were thorough.
            # It implies no explicit W/H, no target_res, or some other unhandled state.
            yield [], None, f"Error for item {i+1}: Failed to determine resolution strategy for the item.", ""
            continue # Skip this item

        if final_height_for_item is None or final_width_for_item is None: # Final check for the item
            yield [], None, f"Error for item {i+1}: Final resolution could not be determined for this item.", ""
            continue

        # Update status text with the preparing subprocess message
        yield all_videos.copy(), current_preview_yield_path, status_text, progress_text_update # Use progress_text_update

        env = os.environ.copy()
        env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        clear_cuda_cache()

        command = [
            sys.executable, "fpack_generate_video.py",
            "--text_encoder1", text_encoder_path, "--text_encoder2", text_encoder_2_path,
            "--image_encoder", image_encoder_path,
            *(["--image_path", final_image_path_arg_for_item] if final_image_path_arg_for_item else []),
            "--save_path", save_path, "--prompt", final_prompt_arg,
            "--video_size", str(final_height_for_item), str(final_width_for_item),
            *(["--video_sections", str(framepack_video_sections)] if framepack_video_sections is not None and framepack_video_sections > 0 else ["--video_seconds", str(total_second_length)]),
            "--infer_steps", str(steps), "--seed", str(current_seed),
            "--embedded_cfg_scale", str(distilled_guidance_scale),
            "--guidance_scale", str(cfg), "--guidance_rescale", str(rs),
            "--latent_window_size", str(latent_window_size),
            "--sample_solver", sample_solver, "--output_type", "video", "--attn_mode", attn_mode
        ]
        if is_f1: command.append("--is_f1")
        if transformer_path and os.path.exists(transformer_path): command.extend(["--dit", transformer_path.strip()])
        if vae_path and os.path.exists(vae_path): command.extend(["--vae", vae_path.strip()])
        if negative_prompt and negative_prompt.strip(): command.extend(["--negative_prompt", negative_prompt.strip()])
        if input_end_frame and os.path.exists(input_end_frame): command.extend(["--end_image_path", input_end_frame])
        if fp8: command.append("--fp8")
        if fp8 and fp8_scaled: command.append("--fp8_scaled")
        if fp8_llm: command.append("--fp8_llm")
        if bulk_decode: command.append("--bulk_decode")
        if blocks_to_swap > 0: command.extend(["--blocks_to_swap", str(blocks_to_swap)])
        if vae_chunk_size is not None and vae_chunk_size > 0: command.extend(["--vae_chunk_size", str(vae_chunk_size)])
        if vae_spatial_tile_sample_min_size is not None and vae_spatial_tile_sample_min_size > 0: command.extend(["--vae_spatial_tile_sample_min_size", str(vae_spatial_tile_sample_min_size)])
        if device and device.strip(): command.extend(["--device", device.strip()])
        if valid_loras_paths:
            command.extend(["--lora_weight"] + valid_loras_paths)
            command.extend(["--lora_multiplier"] + valid_loras_mults)
        if enable_preview and preview_every_n_sections > 0:
            command.extend(["--preview_latent_every", str(preview_every_n_sections)])
            command.extend(["--preview_suffix", unique_preview_suffix])
            if use_full_video_preview: # Check if full preview is requested
                command.append("--full_preview")
                print(f"DEBUG: Enabling FULL VIDEO preview every {preview_every_n_sections} sections with suffix {unique_preview_suffix}.")
            else:
                print(f"DEBUG: Enabling latent preview every {preview_every_n_sections} sections with suffix {unique_preview_suffix}.")
        if use_teacache:
            command.append("--use_teacache")
            command.extend(["--teacache_steps", str(teacache_steps)])
            command.extend(["--teacache_thresh", str(teacache_thresh)])

        command_str = [str(c) for c in command]
        print(f"Running FramePack Command: {' '.join(command_str)}")

        p = subprocess.Popen(
            command_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, encoding='utf-8', errors='replace', bufsize=1
        )
        current_phase = "Preparing"
        actual_total_sections = None
        display_section_num = 1

        while True:
            if stop_event.is_set():
                try:
                    p.terminate()
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait()
                except Exception as e:
                    print(f"Error terminating subprocess: {e}")
                yield all_videos.copy(), None, "Generation stopped by user.", ""
                return
            if skip_event.is_set():
                print(f"Skip signal received for batch item {i+1}. Terminating subprocess...")
                try:
                    p.terminate()
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait()
                except Exception as e:
                    print(f"Error terminating subprocess during skip: {e}")
                skip_event.clear()
                yield all_videos.copy(), current_preview_yield_path, f"Skipping item {i+1}/{batch_size}...", ""
                break

            line = p.stdout.readline()
            if not line:
                if p.poll() is not None: break
                time.sleep(0.01); continue

            line = line.strip()
            if not line: continue
            print(f"SUBPROCESS: {line}")

            section_match = re.search(r"---.*?Section\s+(\d+)\s*/\s*(\d+)(?:\s+|$|\()", line)
            tqdm_match = re.search(r'(\d+)\%\|.+\| (\d+)/(\d+) \[(\d{2}:\d{2})<(\d{2}:\d{2})', line)
            phase_changed = False # Initialize phase_changed inside the loop
            
            # Default progress_text_update to the current line for general logging
            progress_text_update = line # This was defined outside the loop before, moved inside

            if section_match:
                current_section_num_display = int(section_match.group(1))
                total_sections_from_log = int(section_match.group(2))
                display_section_num = current_section_num_display
                if actual_total_sections != total_sections_from_log:
                    actual_total_sections = total_sections_from_log
                    print(f"Detected/Updated actual total sections: {actual_total_sections}")
                new_phase = f"Generating Section {display_section_num}"
                if current_phase != new_phase:
                    current_phase = new_phase
                    phase_changed = True
                progress_text_update = f"Item {i+1}/{batch_size} | Section {display_section_num}/{actual_total_sections} | Preparing..."
                status_text = f"Generating video {i + 1} of {batch_size} (Seed: {current_seed}) - {current_phase}"
            elif tqdm_match:
                percentage = int(tqdm_match.group(1))
                current_step = int(tqdm_match.group(2))
                total_steps = int(tqdm_match.group(3))
                time_elapsed_str = tqdm_match.group(4)
                time_remaining_str = tqdm_match.group(5)

                total_eta_str = time_remaining_str # Default to per-section ETA
                if actual_total_sections and actual_total_sections > 0:
                    remaining_sec_section = parse_time_to_seconds(time_remaining_str)
                    elapsed_sec_section = parse_time_to_seconds(time_elapsed_str)

                    if remaining_sec_section is not None and elapsed_sec_section is not None and (elapsed_sec_section + remaining_sec_section) > 0:
                        total_time_per_section = elapsed_sec_section + remaining_sec_section
                        remaining_full_sections = actual_total_sections - display_section_num
                        remaining_time_full_sections = remaining_full_sections * total_time_per_section
                        total_remaining_seconds = remaining_sec_section + remaining_time_full_sections

                        if total_remaining_seconds > 0:
                            eta_hours = int(total_remaining_seconds / 3600)
                            eta_minutes = int((total_remaining_seconds % 3600) / 60)
                            eta_seconds = int(total_remaining_seconds % 60)
                            if eta_hours > 0:
                                total_eta_str = f"{eta_hours:d}:{eta_minutes:02d}:{eta_seconds:02d}"
                            else:
                                total_eta_str = f"{eta_minutes:02d}:{eta_seconds:02d}"

                current_total_for_display = actual_total_sections if actual_total_sections is not None else total_sections_estimate
                section_str = f"Section {display_section_num}/{current_total_for_display}"
                progress_text_update = f"Item {i+1}/{batch_size} | {section_str} | Step {current_step}/{total_steps} ({percentage}%) | Elapsed: {time_elapsed_str}, Remaining: {total_eta_str}"
                denoising_phase = f"Denoising Section {display_section_num}"
                if current_phase != denoising_phase:
                    current_phase = denoising_phase
                    phase_changed = True
                status_text = f"Generating video {i + 1} of {batch_size} (Seed: {current_seed}) - {current_phase}"
            elif "Decoding video..." in line:
                 if current_phase != "Decoding Video":
                     current_phase = "Decoding Video"
                     phase_changed = True
                 progress_text_update = f"Item {i+1}/{batch_size} | {current_phase}..."
                 status_text = f"Generating video {i + 1} of {batch_size} (Seed: {current_seed}) - {current_phase}"
            elif "INFO:__main__:Video saved to:" in line:
                 match = re.search(r"Video saved to:\s*(.*\.mp4)", line)
                 if match:
                     found_video_path = match.group(1).strip()
                     if os.path.exists(found_video_path):
                         current_video_path = found_video_path
                         # Don't add to all_videos here, add after subprocess completion
                     else:
                          print(f"Warning: Parsed video path does not exist: {found_video_path}")
                     status_text = f"Video {i+1}/{batch_size} Saved (Seed: {current_seed})"
                     progress_text_update = f"Saved: {os.path.basename(found_video_path) if found_video_path else 'Unknown Path'}"
                     current_phase = "Saved"
                     phase_changed = True
                 else:
                     print(f"Warning: Could not parse video path from INFO line: {line}")
            elif "ERROR" in line.upper() or "TRACEBACK" in line.upper():
                 status_text = f"Item {i+1}/{batch_size}: Error Detected (Check Console)"
                 progress_text_update = line
                 if current_phase != "Error":
                    current_phase = "Error"
                    phase_changed = True
            elif phase_changed and current_phase not in ["Saved", "Error"]:
                 status_text = f"Generating video {i + 1} of {batch_size} (Seed: {current_seed}) - {current_phase}"

            preview_updated = False
            current_mtime_check = 0 # Renamed from current_mtime to avoid conflict
            found_preview_path_check = None # Renamed

            if enable_preview:
                if os.path.exists(preview_mp4_path):
                    current_mtime_check = os.path.getmtime(preview_mp4_path)
                    found_preview_path_check = preview_mp4_path
                elif os.path.exists(preview_png_path):
                    current_mtime_check = os.path.getmtime(preview_png_path)
                    found_preview_path_check = preview_png_path

                if found_preview_path_check and current_mtime_check > last_preview_mtime:
                    print(f"DEBUG: Preview file updated: {found_preview_path_check} (mtime: {current_mtime_check})")
                    current_preview_yield_path = found_preview_path_check
                    last_preview_mtime = current_mtime_check
                    preview_updated = True
            
            yield all_videos.copy(), current_preview_yield_path, status_text, progress_text_update

        p.stdout.close(); rc = p.wait()
        clear_cuda_cache(); time.sleep(0.1)

        if rc == 0 and current_video_path and os.path.exists(current_video_path):
            all_videos.append((current_video_path, f"Seed: {current_seed}")) # Add video here
            parameters = {
                "prompt": prompt, "negative_prompt": negative_prompt,
                "input_image": os.path.basename(current_input_image_for_item) if current_input_image_for_item else None,
                "section_controls": [
                     {"index": s, "prompt_override": p_override, "image_override": os.path.basename(img_override) if img_override else None}
                     for s, p_override, img_override in zip(framepack_secs, framepack_sec_prompts, framepack_sec_images)
                     if (p_override and p_override.strip()) or img_override
                 ],
                "final_prompt_arg": final_prompt_arg,
                "final_image_path_arg": final_image_path_arg_for_item, # Use item-specific image path
                "input_end_frame": os.path.basename(input_end_frame) if input_end_frame else None,
                "transformer_path": transformer_path, "vae_path": vae_path,
                "text_encoder_path": text_encoder_path, "text_encoder_2_path": text_encoder_2_path,
                "image_encoder_path": image_encoder_path,
                "video_width": final_width_for_item, "video_height": final_height_for_item,
                "video_seconds": total_second_length, "fps": fps, "seed": current_seed,
                "infer_steps": steps, "embedded_cfg_scale": distilled_guidance_scale,
                "guidance_scale": cfg, "guidance_rescale": rs, "sample_solver": sample_solver,
                "latent_window_size": latent_window_size,
                "fp8": fp8, "fp8_scaled": fp8_scaled, "fp8_llm": fp8_llm,
                "blocks_to_swap": blocks_to_swap, "bulk_decode": bulk_decode, "attn_mode": attn_mode,
                "vae_chunk_size": vae_chunk_size, "vae_spatial_tile_sample_min_size": vae_spatial_tile_sample_min_size,
                "device": device,
                "lora_weights": [os.path.basename(p) for p in valid_loras_paths],
                "lora_multipliers": [float(m) for m in valid_loras_mults],
                "original_dims_str": current_original_dims_str_for_item,
                "target_resolution": target_resolution,
                "is_f1": is_f1
            }
            try:
                add_metadata_to_video(current_video_path, parameters)
                print(f"Added metadata to {current_video_path}")
            except Exception as meta_err:
                print(f"Warning: Failed to add metadata to {current_video_path}: {meta_err}")
            status_text = f"Item {i+1}/{batch_size} Completed (Seed: {current_seed})"
            progress_text_update = f"Video saved: {os.path.basename(current_video_path)}"
            current_preview_yield_path = None # Clear preview for next item
            yield all_videos.copy(), current_preview_yield_path, status_text, progress_text_update
        elif rc != 0:
            status_text = f"Item {i+1}/{batch_size} Failed (Seed: {current_seed}, Code: {rc})"
            progress_text_update = f"Subprocess failed. Check console logs."
            current_preview_yield_path = None # Clear preview
            yield all_videos.copy(), current_preview_yield_path, status_text, progress_text_update
        else:
            status_text = f"Item {i+1}/{batch_size} Finished (Seed: {current_seed}), but no video file confirmed."
            progress_text_update = "Check console logs for the saved path."
            current_preview_yield_path = None # Clear preview
            yield all_videos.copy(), current_preview_yield_path, status_text, progress_text_update
        
        # Cleanup preview files for the completed item to avoid them being picked up by next item
        if enable_preview:
            for prev_file in [preview_mp4_path, preview_png_path]:
                if os.path.exists(prev_file):
                    try:
                        os.remove(prev_file)
                        print(f"Cleaned up preview file: {prev_file}")
                    except Exception as e_clean:
                        print(f"Warning: Could not remove preview file {prev_file}: {e_clean}")
        
        time.sleep(0.2)

    yield all_videos, None, "FramePack Batch complete", ""

def calculate_framepack_width(height, original_dims):
    """Calculate FramePack width based on height maintaining aspect ratio (divisible by 32)"""
    if not original_dims or height is None:
        return gr.update()
    try:
        # Ensure height is an integer and divisible by 32
        height = int(height)
        if height <= 0 : return gr.update()
        height = (height // 32) * 32 # <-- Use 32
        height = max(64, height) # Min height (64 is divisible by 32)

        orig_w, orig_h = map(int, original_dims.split('x'))
        if orig_h == 0: return gr.update()
        aspect_ratio = orig_w / orig_h
        # Calculate new width, rounding to the nearest multiple of 32
        new_width = round((height * aspect_ratio) / 32) * 32 # <-- Round and use 32
        return gr.update(value=max(64, new_width)) # Ensure minimum size (also divisible by 32)

    except Exception as e:
        print(f"Error calculating width: {e}")
        return gr.update()

def calculate_framepack_height(width, original_dims):
    """Calculate FramePack height based on width maintaining aspect ratio (divisible by 32)"""
    if not original_dims or width is None:
        return gr.update()
    try:
        # Ensure width is an integer and divisible by 32
        width = int(width)
        if width <= 0: return gr.update()
        width = (width // 32) * 32 # <-- Use 32
        width = max(64, width) # Min width (64 is divisible by 32)

        orig_w, orig_h = map(int, original_dims.split('x'))
        if orig_w == 0: return gr.update()
        aspect_ratio = orig_w / orig_h
        # Calculate new height, rounding to the nearest multiple of 32
        new_height = round((width / aspect_ratio) / 32) * 32 # <-- Round and use 32
        return gr.update(value=max(64, new_height)) # Ensure minimum size (also divisible by 32)
    except Exception as e:
        print(f"Error calculating height: {e}")
        return gr.update()

def update_framepack_from_scale(scale, original_dims):
    """Update FramePack dimensions based on scale percentage (divisible by 32)"""
    if not original_dims:
        return gr.update(), gr.update(), gr.update()
    try:
        scale = float(scale) if scale is not None else 100.0
        if scale <= 0: scale = 100.0

        orig_w, orig_h = map(int, original_dims.split('x'))
        scale_factor = scale / 100.0

        # Calculate and round to the nearest multiple of 32
        new_w = round((orig_w * scale_factor) / 32) * 32 # <-- Round and use 32
        new_h = round((orig_h * scale_factor) / 32) * 32 # <-- Round and use 32

        # Ensure minimum size (must be multiple of 32)
        new_w = max(64, new_w) # 64 is divisible by 32
        new_h = max(64, new_h)

        # Clear target resolution if using scale slider for explicit dims
        return gr.update(value=new_w), gr.update(value=new_h), gr.update(value=None)
    except Exception as e:
        print(f"Error updating from scale: {e}")
        return gr.update(), gr.update(), gr.update()

def process_i2v_single_video(
    prompt: str,
    image_path: str,
    width: int,
    height: int,
    batch_size: int,
    video_length: int,
    fps: int,
    infer_steps: int,
    seed: int,
    dit_folder: str,
    model: str,
    vae: str,
    te1: str,
    te2: str,
    clip_vision_path: str,
    save_path: str,
    flow_shift: float,
    cfg_scale: float, # embedded_cfg_scale
    guidance_scale: float, # main CFG
    output_type: str,
    attn_mode: str,
    block_swap: int,
    exclude_single_blocks: bool,
    use_split_attn: bool,
    lora_folder: str,
    vae_chunk_size: int,
    vae_spatial_tile_min: int,
    # --- Explicit LoRA args instead of *lora_params ---
    lora1: str = "None",
    lora2: str = "None",
    lora3: str = "None",
    lora4: str = "None",
    lora1_multiplier: float = 1.0,
    lora2_multiplier: float = 1.0,
    lora3_multiplier: float = 1.0,
    lora4_multiplier: float = 1.0,
    # --- End LoRA args ---
    negative_prompt: Optional[str] = None,
    use_fp8: bool = False,
    fp8_llm: bool = False
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """Generate a single video using hv_i2v_generate_video.py"""
    global stop_event

    # ... (Keep existing argument validation and env setup) ...
    if stop_event.is_set():
        yield [], "", ""
        return

    # Argument validation
    if not image_path or not os.path.exists(image_path):
         yield [], "Error: Input image not found", f"Cannot find image: {image_path}"
         return
    # Check clip vision path only if needed (Hunyuan-I2V, not SkyReels-I2V based on script name)
    is_hunyuan_i2v = "mp_rank_00_model_states_i2v" in model # Heuristic check
    if is_hunyuan_i2v and (not clip_vision_path or not os.path.exists(clip_vision_path)):
         yield [], "Error: CLIP Vision model not found", f"Cannot find file: {clip_vision_path}"
         return

    if os.path.isabs(model):
        model_path = model
    else:
        model_path = os.path.normpath(os.path.join(dit_folder, model))

    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    if seed == -1:
        current_seed = random.randint(0, 2**32 - 1)
    else:
        current_seed = seed

    clear_cuda_cache()

    command = [
        sys.executable,
        "hv_i2v_generate_video.py", # <<< Use the new script
        "--dit", model_path,
        "--vae", vae,
        "--text_encoder1", te1,
        "--text_encoder2", te2,
        # Add clip vision path only if it's likely the Hunyuan I2V model
        *(["--clip_vision_path", clip_vision_path] if is_hunyuan_i2v else []),
        "--prompt", prompt,
        "--video_size", str(height), str(width),
        "--video_length", str(video_length),
        "--fps", str(fps),
        "--infer_steps", str(infer_steps),
        "--save_path", save_path,
        "--seed", str(current_seed),
        "--flow_shift", str(flow_shift),
        "--embedded_cfg_scale", str(cfg_scale),
        "--guidance_scale", str(guidance_scale),
        "--output_type", output_type,
        "--attn_mode", attn_mode,
        "--blocks_to_swap", str(block_swap),
        "--image_path", image_path
    ]

    if negative_prompt:
        command.extend(["--negative_prompt", negative_prompt])

    if use_fp8:
        command.append("--fp8")
    if fp8_llm:
        command.append("--fp8_llm")

    if exclude_single_blocks:
        command.append("--exclude_single_blocks")
    if use_split_attn:
        command.append("--split_attn")

    if vae_chunk_size > 0:
        command.extend(["--vae_chunk_size", str(vae_chunk_size)])
    if vae_spatial_tile_min > 0:
        command.extend(["--vae_spatial_tile_sample_min_size", str(vae_spatial_tile_min)])

    # --- Updated LoRA handling using named arguments ---
    lora_weights_list = [lora1, lora2, lora3, lora4]
    lora_multipliers_list = [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier]
    valid_loras = []
    for weight, mult in zip(lora_weights_list, lora_multipliers_list):
        if weight and weight != "None":
            lora_file_path = os.path.join(lora_folder, weight)
            if os.path.exists(lora_file_path):
                 valid_loras.append((lora_file_path, mult))
            else:
                print(f"Warning: LoRA file not found: {lora_file_path}")

    if valid_loras:
        weights = [weight for weight, _ in valid_loras]
        multipliers = [str(mult) for _, mult in valid_loras]
        command.extend(["--lora_weight"] + weights)
        command.extend(["--lora_multiplier"] + multipliers)
    # --- End Updated LoRA handling ---

    # ... (Keep subprocess execution, output collection, and metadata saving logic) ...
    command_str = [str(c) for c in command] # Ensure all args are strings
    print(f"Running Command (I2V): {' '.join(command_str)}")

    p = subprocess.Popen(
        command_str, # Use stringified command
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )

    videos = []

    while True:
        if stop_event.is_set():
            p.terminate()
            p.wait()
            yield videos, current_previews, "Generation stopped by user.", ""
            return

        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                break
            continue

        print(line, end='') # Print progress to console
        if '|' in line and '%' in line and '[' in line and ']' in line:
            yield videos.copy(), f"Processing (seed: {current_seed})", line.strip()

    p.stdout.close()
    p.wait()

    clear_cuda_cache()
    time.sleep(0.5)

    # Collect generated video
    save_path_abs = os.path.abspath(save_path)
    generated_video_path = None
    if os.path.exists(save_path_abs):
        all_videos_files = sorted(
            [f for f in os.listdir(save_path_abs) if f.endswith('.mp4')],
            key=lambda x: os.path.getmtime(os.path.join(save_path_abs, x)),
            reverse=True
        )
        # Try to find the video matching the seed
        matching_videos = [v for v in all_videos_files if f"_{current_seed}" in v]
        if matching_videos:
            generated_video_path = os.path.join(save_path_abs, matching_videos[0])

    if generated_video_path:
         # Collect parameters for metadata (adjust as needed for i2v specifics)
        parameters = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "video_length": video_length,
            "fps": fps,
            "infer_steps": infer_steps,
            "seed": current_seed,
            "model": model,
            "vae": vae,
            "te1": te1,
            "te2": te2,
            "clip_vision_path": clip_vision_path,
            "save_path": save_path,
            "flow_shift": flow_shift,
            "embedded_cfg_scale": cfg_scale,
            "guidance_scale": guidance_scale,
            "output_type": output_type,
            "attn_mode": attn_mode,
            "block_swap": block_swap,
            "lora_weights": list(lora_weights_list), # Save the list
            "lora_multipliers": list(lora_multipliers_list), # Save the list
            "input_image": image_path,
            "negative_prompt": negative_prompt if negative_prompt else None,
            "vae_chunk_size": vae_chunk_size,
            "vae_spatial_tile_min": vae_spatial_tile_min,
            "use_fp8_dit": use_fp8,
            "use_fp8_llm": fp8_llm
        }
        add_metadata_to_video(generated_video_path, parameters)
        videos.append((str(generated_video_path), f"Seed: {current_seed}"))
        yield videos, f"Completed (seed: {current_seed})", ""
    else:
        yield [], f"Failed (seed: {current_seed})", "Could not find generated video file."


def process_i2v_batch(
    prompt: str,
    image_path: str,
    width: int,
    height: int,
    batch_size: int,
    video_length: int,
    fps: int,
    infer_steps: int,
    seed: int,
    dit_folder: str,
    model: str,
    vae: str,
    te1: str,
    te2: str,
    clip_vision_path: str, # Added
    save_path: str,
    flow_shift: float,
    cfg_scale: float, # embedded_cfg_scale
    guidance_scale: float, # main CFG
    output_type: str,
    attn_mode: str,
    block_swap: int,
    exclude_single_blocks: bool,
    use_split_attn: bool,
    lora_folder: str,
    vae_chunk_size: int, # Added
    vae_spatial_tile_min: int, # Added
    negative_prompt: Optional[str] = None, # Added
    use_fp8: bool = False, # Added
    fp8_llm: bool = False, # Added
    *lora_params # Captures LoRA weights and multipliers
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """Process a batch of videos using the new I2V script"""
    global stop_event
    stop_event.clear()

    all_videos = []
    progress_text = "Starting I2V generation..."
    yield [], "Preparing...", progress_text

    # Extract LoRA weights and multipliers once
    num_lora_weights = 4
    lora_weights_list = lora_params[:num_lora_weights]
    lora_multipliers_list = lora_params[num_lora_weights:num_lora_weights*2]

    for i in range(batch_size):
        if stop_event.is_set():
            yield all_videos, "Generation stopped by user.", ""
            return

        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif batch_size > 1:
            current_seed = seed + i

        batch_text = f"Generating video {i + 1} of {batch_size} (I2V)"
        yield all_videos.copy(), batch_text, progress_text

        # Call the single video processing function
        single_gen = process_i2v_single_video(
            prompt=prompt,
            image_path=image_path,
            width=width,
            height=height,
            batch_size=batch_size,
            video_length=video_length,
            fps=fps,
            infer_steps=infer_steps,
            seed=current_seed,
            dit_folder=dit_folder,
            model=model,
            vae=vae,
            te1=te1,
            te2=te2,
            clip_vision_path=clip_vision_path,
            save_path=save_path,
            flow_shift=flow_shift,
            cfg_scale=cfg_scale,
            guidance_scale=guidance_scale,
            output_type=output_type,
            attn_mode=attn_mode,
            block_swap=block_swap,
            exclude_single_blocks=exclude_single_blocks,
            use_split_attn=use_split_attn,
            lora_folder=lora_folder,
            vae_chunk_size=vae_chunk_size,
            vae_spatial_tile_min=vae_spatial_tile_min,
            # --- Pass LoRA params by keyword ---
            lora1=lora_weights_list[0],
            lora2=lora_weights_list[1],
            lora3=lora_weights_list[2],
            lora4=lora_weights_list[3],
            lora1_multiplier=lora_multipliers_list[0],
            lora2_multiplier=lora_multipliers_list[1],
            lora3_multiplier=lora_multipliers_list[2],
            lora4_multiplier=lora_multipliers_list[3],
            # --- End LoRA keyword args ---
            negative_prompt=negative_prompt,
            use_fp8=use_fp8,
            fp8_llm=fp8_llm
        )

        # Yield progress updates from the single generator
        try:
            for videos, status, progress in single_gen:
                if videos:
                    # Only add the latest video from this specific generation
                    new_video = videos[-1]
                    if new_video not in all_videos:
                         all_videos.append(new_video)
                yield all_videos.copy(), f"Batch {i+1}/{batch_size}: {status}", progress
        except Exception as e:
             yield all_videos.copy(), f"Error in batch {i+1}: {e}", ""
             print(f"Error during single I2V generation: {e}") # Log error

        # Optional small delay between batch items
        time.sleep(0.1)

    yield all_videos, "I2V Batch complete", ""


def wanx_extend_video_wrapper(
    prompt, negative_prompt, input_image, base_video_path,
    width, height, video_length, fps, infer_steps,
    flow_shift, guidance_scale, seed,
    task, dit_folder, dit_path, vae_path, t5_path, clip_path, # <--- Parameters received here
    save_path, output_type, sample_solver, exclude_single_blocks,
    attn_mode, block_swap, fp8, fp8_scaled, fp8_t5, lora_folder,
    slg_layers="", slg_start=0.0, slg_end=1.0,
    lora1="None", lora2="None", lora3="None", lora4="None",
    lora1_multiplier=1.0, lora2_multiplier=1.0, lora3_multiplier=1.0, lora4_multiplier=1.0,
    enable_cfg_skip=False, cfg_skip_mode="none", cfg_apply_ratio=0.7
):
    """Direct wrapper that bypasses the problematic wanx_generate_video function"""
    global stop_event

    # All videos generated
    all_videos = []

    # Debug prints to understand what we're getting
    print(f"DEBUG - Received parameters in wanx_extend_video_wrapper:")
    print(f"  task: {task}")
    print(f"  dit_folder: {dit_folder}") # <<< Should be the folder path ('wan')
    print(f"  dit_path: {dit_path}")     # <<< Should be the model filename
    print(f"  vae_path: {vae_path}")     # <<< Should be the VAE path
    print(f"  t5_path: {t5_path}")       # <<< Should be the T5 path
    print(f"  clip_path: {clip_path}")     # <<< Should be the CLIP path
    print(f"  output_type: {output_type}")
    print(f"  sample_solver: {sample_solver}")
    print(f"  attn_mode: {attn_mode}")
    print(f"  block_swap: {block_swap}")

    # Get current seed
    current_seed = seed
    if seed == -1:
        current_seed = random.randint(0, 2**32 - 1)

    # --- START CRITICAL FIX ---
    # Detect if parameters are swapped based on the pattern observed in the error log
    # Check if dit_path looks like a VAE path (contains "VAE" or ends with .pth)
    # AND dit_folder looks like a model filename (ends with .safetensors or .pt)
    params_swapped = False
    if dit_path and dit_folder and \
       (("VAE" in dit_path or dit_path.endswith(".pth")) and \
        (dit_folder.endswith(".safetensors") or dit_folder.endswith(".pt"))):
        params_swapped = True
        print("WARNING: Parameters appear to be swapped in extend workflow. Applying correction...")

        # Correct the parameters based on the observed swap
        actual_model_filename = dit_folder # Original dit_folder was the filename
        actual_vae_path = dit_path         # Original dit_path was the VAE path
        actual_t5_path = vae_path          # Original vae_path was the T5 path
        actual_clip_path = t5_path         # Original t5_path was the CLIP path

        # Assign corrected values back to expected variable names for the rest of the function
        dit_path = actual_model_filename
        vae_path = actual_vae_path
        t5_path = actual_t5_path
        clip_path = actual_clip_path
        dit_folder = "wan" # Assume default 'wan' folder if swapped

        print(f"  Corrected dit_folder: {dit_folder}")
        print(f"  Corrected dit_path (model filename): {dit_path}")
        print(f"  Corrected vae_path: {vae_path}")
        print(f"  Corrected t5_path: {t5_path}")
        print(f"  Corrected clip_path: {clip_path}")

    # Construct the full model path using the potentially corrected dit_folder and dit_path
    actual_model_path = os.path.join(dit_folder, dit_path) if not os.path.isabs(dit_path) else dit_path
    print(f"  Using actual_model_path for --dit: {actual_model_path}")
    # --- END CRITICAL FIX ---

    # Prepare environment
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    # Clear CUDA cache
    clear_cuda_cache()

    # Validate and fix parameters
    # Fix output_type - must be one of: video, images, latent, both
    valid_output_types = ["video", "images", "latent", "both"]
    actual_output_type = "video" if output_type not in valid_output_types else output_type

    # Fix sample_solver - must be one of: unipc, dpm++, vanilla
    valid_sample_solvers = ["unipc", "dpm++", "vanilla"]
    actual_sample_solver = "unipc" if sample_solver not in valid_sample_solvers else sample_solver

    # Fix attn_mode - must be one of: sdpa, flash, sageattn, xformers, torch
    valid_attn_modes = ["sdpa", "flash", "sageattn", "xformers", "torch"]
    actual_attn_mode = "sdpa" if attn_mode not in valid_attn_modes else attn_mode

    # Fix block_swap - must be an integer
    try:
        actual_block_swap = int(block_swap)
    except (ValueError, TypeError):
        actual_block_swap = 0

    # Build command array with explicit string conversions for EVERY parameter
    command = [
        sys.executable,
        "wan_generate_video.py",
        "--task", str(task),
        "--prompt", str(prompt),
        "--video_size", str(height), str(width),
        "--video_length", str(video_length),
        "--fps", str(fps),
        "--infer_steps", str(infer_steps),
        "--save_path", str(save_path),
        "--seed", str(current_seed),
        "--flow_shift", str(flow_shift),
        "--guidance_scale", str(guidance_scale),
        "--output_type", actual_output_type,
        "--sample_solver", actual_sample_solver,
        "--attn_mode", actual_attn_mode,
        "--blocks_to_swap", str(actual_block_swap),
        # Use the corrected model path and other paths
        "--dit", str(actual_model_path), # <<< Use corrected full model path
        "--vae", str(vae_path),          # <<< Use potentially corrected vae_path
        "--t5", str(t5_path)            # <<< Use potentially corrected t5_path
    ]

    # Add image path and clip model path if needed
    if input_image:
        command.extend(["--image_path", str(input_image)])
        # Use the potentially corrected clip_path
        if clip_path and clip_path != "outputs" and "output" not in clip_path:
            command.extend(["--clip", str(clip_path)]) # <<< Use potentially corrected clip_path

    # Add negative prompt
    if negative_prompt:
        command.extend(["--negative_prompt", str(negative_prompt)])

    # Handle boolean flags - keep original values
    if fp8:
        command.append("--fp8")

    if fp8_scaled:
        command.append("--fp8_scaled")

    if fp8_t5:
        command.append("--fp8_t5")

    # Add SLG parameters
    try:
        # Ensure slg_layers is treated as a string before splitting
        slg_layers_str = str(slg_layers) if slg_layers is not None else ""
        if slg_layers_str and slg_layers_str.strip() and slg_layers_str.lower() != "none":
            slg_list = []
            for layer in slg_layers_str.split(","):
                layer = layer.strip()
                if layer.isdigit():  # Only add if it's a valid integer
                    slg_list.append(int(layer))
            if slg_list:  # Only add if we have valid layers
                command.extend(["--slg_layers", ",".join(map(str, slg_list))])

                # Only add slg_start and slg_end if we have valid slg_layers
                if slg_start is not None:
                    try:
                         slg_start_float = float(slg_start)
                         if slg_start_float >= 0:
                             command.extend(["--slg_start", str(slg_start_float)])
                    except (ValueError, TypeError): pass # Ignore if conversion fails
                if slg_end is not None:
                     try:
                         slg_end_float = float(slg_end)
                         if slg_end_float <= 1.0:
                             command.extend(["--slg_end", str(slg_end_float)])
                     except (ValueError, TypeError): pass # Ignore if conversion fails
    except Exception as e: # Catch potential errors during processing
        print(f"Warning: Error processing SLG parameters: {e}")
        pass

    # Handle LoRA weights and multipliers
    valid_loras = []
    if lora_folder and isinstance(lora_folder, str):
        for weight, mult in zip([lora1, lora2, lora3, lora4],
                              [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier]):
            # Skip None or empty values
            if not weight or str(weight).lower() == "none":
                continue

            # Construct path and check existence
            full_path = os.path.join(str(lora_folder), str(weight))
            if not os.path.exists(full_path):
                print(f"LoRA file not found: {full_path}")
                continue

            # Add valid LoRA
            valid_loras.append((full_path, str(mult)))

    if valid_loras:
        weights = [w for w, _ in valid_loras]
        multipliers = [m for _, m in valid_loras]
        command.extend(["--lora_weight"] + weights)
        command.extend(["--lora_multiplier"] + multipliers)

    # Final conversion to ensure all elements are strings
    command_str = [str(item) for item in command]

    print(f"Running Command (wanx_extend_video_wrapper): {' '.join(command_str)}")

    # Process execution
    p = subprocess.Popen(
        command_str, # Use stringified command
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )

    videos = [] # Store the generated (non-extended) video first

    # Process stdout in real time
    while True:
        if stop_event.is_set():
            p.terminate()
            p.wait()
            yield [], "", "Generation stopped by user."
            return

        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                break
            continue

        print(line, end='')
        if '|' in line and '%' in line and '[' in line and ']' in line:
             # Yield empty list during processing, actual video is collected later
            yield [], f"Processing (seed: {current_seed})", line.strip()

    p.stdout.close()
    return_code = p.wait() # Get return code

    # Clean CUDA cache and wait
    clear_cuda_cache()
    time.sleep(0.5)

    # Check return code
    if return_code != 0:
        print(f"❌ Error: wan_generate_video.py exited with code {return_code}")
        yield [], f"Failed (seed: {current_seed})", f"Subprocess failed with code {return_code}"
        return

    # Find the *newly generated* video first
    generated_video_path = None
    save_path_abs = os.path.abspath(save_path)
    if os.path.exists(save_path_abs):
        # Find the most recent mp4 containing the seed
        all_mp4_files = glob.glob(os.path.join(save_path_abs, f"*_{current_seed}*.mp4"))
        if all_mp4_files:
            generated_video_path = max(all_mp4_files, key=os.path.getmtime)
            print(f"Found newly generated video: {generated_video_path}")

            # Add metadata to the generated video before potential concatenation
            parameters = {
                "prompt": prompt, "negative_prompt": negative_prompt, "input_image": input_image,
                "width": width, "height": height, "video_length": video_length, "fps": fps,
                "infer_steps": infer_steps, "flow_shift": flow_shift, "guidance_scale": guidance_scale,
                "seed": current_seed, "task": task, "dit_path": actual_model_path, # Store the actual path used
                "vae_path": vae_path, "t5_path": t5_path, "clip_path": clip_path,
                "save_path": save_path, "output_type": actual_output_type, "sample_solver": actual_sample_solver,
                "exclude_single_blocks": exclude_single_blocks, "attn_mode": actual_attn_mode,
                "block_swap": actual_block_swap, "fp8": fp8, "fp8_scaled": fp8_scaled, "fp8_t5": fp8_t5,
                "lora_weights": [lora1, lora2, lora3, lora4],
                "lora_multipliers": [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier],
                "slg_layers": slg_layers, "slg_start": slg_start, "slg_end": slg_end,
                "is_extension_source": True # Flag this as the source for an extension
            }
            add_metadata_to_video(generated_video_path, parameters)
            # videos.append((str(generated_video_path), f"Generated segment (Seed: {current_seed})")) # Optionally yield segment
        else:
             print(f"Could not find generated video segment for seed {current_seed} in {save_path_abs}")

    # Stop here if no new video segment was generated
    if not generated_video_path:
        yield [], f"Failed (seed: {current_seed})", "Could not find generated video segment."
        return

    # Now concatenate with base video if we have the new segment and a base_video_path
    if generated_video_path and base_video_path and os.path.exists(base_video_path):
        try:
            print(f"Extending base video: {base_video_path}")

            # Create unique output filename for the *extended* video
            timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
            output_filename = f"extended_{timestamp}_seed{current_seed}_{Path(base_video_path).stem}.mp4"
            output_path = os.path.join(save_path_abs, output_filename)

            # Create a temporary file list for ffmpeg concatenation
            list_file = os.path.join(save_path_abs, f"temp_concat_list_{current_seed}.txt")
            with open(list_file, "w") as f:
                f.write(f"file '{os.path.abspath(base_video_path)}'\n")
                f.write(f"file '{os.path.abspath(generated_video_path)}'\n") # Use the newly generated segment

            print(f"Concatenating: {base_video_path} + {generated_video_path} -> {output_path}")

            # Run ffmpeg concatenation command
            concat_command = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",       # Allow relative paths if needed, but we use absolute
                "-i", list_file,
                "-c", "copy",       # Fast concatenation without re-encoding
                "-y",               # Overwrite output if exists
                output_path
            ]

            # Convert all command parts to strings
            concat_command_str = [str(item) for item in concat_command]

            print(f"Running FFmpeg command: {' '.join(concat_command_str)}")
            concat_result = subprocess.run(concat_command_str, check=False, capture_output=True, text=True) # Don't check=True initially

            # Clean up temporary list file
            if os.path.exists(list_file):
                try:
                    os.remove(list_file)
                except OSError as e:
                    print(f"Warning: Could not remove temp list file {list_file}: {e}")


            # Check if concatenation was successful
            if concat_result.returncode == 0 and os.path.exists(output_path):
                # Optionally, add metadata to the *extended* video as well
                extended_parameters = parameters.copy()
                extended_parameters["is_extension_source"] = False
                extended_parameters["base_video"] = os.path.basename(base_video_path)
                add_metadata_to_video(output_path, extended_parameters)

                extended_video_gallery_item = [(output_path, f"Extended (Seed: {current_seed})")]
                print(f"✅ Successfully created extended video: {output_path}")
                yield extended_video_gallery_item, "Extended video created successfully", ""
                return # Success!
            else:
                print(f"❌ Failed to create extended video at {output_path}")
                print(f"FFmpeg stderr: {concat_result.stderr}")
                # Yield the generated segment if concatenation failed
                yield [(generated_video_path, f"Generated segment (Seed: {current_seed})")], "Generated segment (extension failed)", f"FFmpeg failed: {concat_result.stderr[:200]}..."
                return

        except Exception as e:
            print(f"❌ Error during concatenation: {str(e)}")
            # Yield the generated segment if concatenation failed
            yield [(generated_video_path, f"Generated segment (Seed: {current_seed})")], "Generated segment (extension error)", f"Error: {str(e)}"
            return

    # If we got here, base_video_path was likely None or didn't exist, but generation succeeded
    yield [(generated_video_path, f"Generated segment (Seed: {current_seed})")], "Generated segment (no base video provided)", ""

def wanx_v2v_generate_video(
    prompt, 
    negative_prompt,
    input_video,
    width,
    height,
    video_length,
    fps,
    infer_steps,
    flow_shift,
    guidance_scale,
    strength,
    seed,
    task,
    dit_folder,
    dit_path,
    vae_path,
    t5_path,
    save_path,
    output_type,
    sample_solver,
    exclude_single_blocks,
    attn_mode,
    block_swap,
    fp8,
    fp8_scaled,
    fp8_t5,
    lora_folder,
    slg_layers,
    slg_start,
    slg_end,
    lora1="None",
    lora2="None",
    lora3="None",
    lora4="None",
    lora1_multiplier=1.0,
    lora2_multiplier=1.0,
    lora3_multiplier=1.0,
    lora4_multiplier=1.0,
    enable_cfg_skip=False,
    cfg_skip_mode="none",
    cfg_apply_ratio=0.7,
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """Generate video with WanX model in video-to-video mode"""
    global stop_event
    
    # Convert values safely to float or None
    try:
        slg_start_float = float(slg_start) if slg_start is not None and str(slg_start).lower() != "none" else None
    except (ValueError, TypeError):
        slg_start_float = None
        print(f"Warning: Could not convert slg_start '{slg_start}' to float")
    
    try:
        slg_end_float = float(slg_end) if slg_end is not None and str(slg_end).lower() != "none" else None
    except (ValueError, TypeError):
        slg_end_float = None
        print(f"Warning: Could not convert slg_end '{slg_end}' to float")
    
    print(f"slg_start_float: {slg_start_float}, slg_end_float: {slg_end_float}")
    
    if stop_event.is_set():
        yield [], "", ""
        return

    # Check if we need input video (required for v2v)
    if not input_video:
        yield [], "Error: No input video provided", "Please provide an input video for video-to-video generation"
        return

    if seed == -1:
        current_seed = random.randint(0, 2**32 - 1)
    else:
        current_seed = seed

    # Prepare environment
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    
    clear_cuda_cache()

    # Construct full dit_path including folder - this is the fix
    full_dit_path = os.path.join(dit_folder, dit_path) if not os.path.isabs(dit_path) else dit_path

    command = [
        sys.executable,
        "wan_generate_video.py",
        "--task", task,
        "--prompt", prompt,
        "--video_size", str(height), str(width),
        "--video_length", str(video_length),
        "--fps", str(fps),
        "--infer_steps", str(infer_steps),
        "--save_path", save_path,
        "--seed", str(current_seed),
        "--flow_shift", str(flow_shift),
        "--guidance_scale", str(guidance_scale),
        "--output_type", output_type,
        "--attn_mode", attn_mode,
        "--blocks_to_swap", str(block_swap),
        "--dit", full_dit_path,  # Use full_dit_path instead of dit_path
        "--vae", vae_path,
        "--t5", t5_path,
        "--sample_solver", sample_solver,
        "--video_path", input_video,  # This is the key for v2v mode
        "--strength", str(strength)   # Strength parameter for v2v
    ]
    if enable_cfg_skip and cfg_skip_mode != "none":
        command.extend([
            "--cfg_skip_mode", cfg_skip_mode,
            "--cfg_apply_ratio", str(cfg_apply_ratio)
        ])
    # Handle SLG parameters
    if slg_layers and str(slg_layers).strip() and slg_layers.lower() != "none":
        try:
            # Parse SLG layers
            layer_list = [int(x) for x in str(slg_layers).split(",")]
            if layer_list:  # Only proceed if we have valid layer values
                command.extend(["--slg_layers", ",".join(map(str, layer_list))])
                
                # Only add slg_start and slg_end if we have valid slg_layers
                try:
                    if slg_start_float is not None and slg_start_float >= 0:
                        command.extend(["--slg_start", str(slg_start_float)])
                    if slg_end_float is not None and slg_end_float <= 1.0:
                        command.extend(["--slg_end", str(slg_end_float)])
                except ValueError as e:
                    print(f"Invalid SLG timing values: {str(e)}")
        except ValueError as e:
            print(f"Invalid SLG layers format: {slg_layers} - {str(e)}")
    
    if negative_prompt:
        command.extend(["--negative_prompt", negative_prompt])
    
    if fp8:
        command.append("--fp8")
    
    if fp8_scaled:
        command.append("--fp8_scaled")
    
    if fp8_t5:
        command.append("--fp8_t5")
        
    if exclude_single_blocks:
        command.append("--exclude_single_blocks")
    
    # Handle LoRA weights and multipliers
    lora_weights = [lora1, lora2, lora3, lora4]
    lora_multipliers = [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier]
    
    valid_loras = []
    for weight, mult in zip(lora_weights, lora_multipliers):
        if weight and weight != "None":
            full_path = os.path.join(lora_folder, weight)
            if not os.path.exists(full_path):
                print(f"LoRA file not found: {full_path}")
                continue
            valid_loras.append((full_path, mult))

    if valid_loras:
        weights = [w for w, _ in valid_loras]
        multipliers = [str(m) for _, m in valid_loras]
        command.extend(["--lora_weight"] + weights)
        command.extend(["--lora_multiplier"] + multipliers)
    
    print(f"Running: {' '.join(command)}")

    p = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )

    videos = []
    
    while True:
        if stop_event.is_set():
            p.terminate()
            p.wait()
            yield [], "", "Generation stopped by user."
            return

        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                break
            continue
            
        print(line, end='')
        if '|' in line and '%' in line and '[' in line and ']' in line:
            yield videos.copy(), f"Processing (seed: {current_seed})", line.strip()

    p.stdout.close()
    p.wait()

    clear_cuda_cache()
    time.sleep(0.5)

    # Collect generated video
    save_path_abs = os.path.abspath(save_path)
    if os.path.exists(save_path_abs):
        all_videos = sorted(
            [f for f in os.listdir(save_path_abs) if f.endswith('.mp4')],
            key=lambda x: os.path.getmtime(os.path.join(save_path_abs, x)),
            reverse=True
        )
        matching_videos = [v for v in all_videos if f"_{current_seed}" in v]
        if matching_videos:
            video_path = os.path.join(save_path_abs, matching_videos[0])
            
            # Collect parameters for metadata
            parameters = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "video_length": video_length,
                "fps": fps,
                "infer_steps": infer_steps,
                "seed": current_seed,
                "task": task,
                "flow_shift": flow_shift,
                "guidance_scale": guidance_scale,
                "output_type": output_type,
                "attn_mode": attn_mode,
                "block_swap": block_swap,
                "input_video": input_video,
                "strength": strength,
                "lora_weights": [lora1, lora2, lora3, lora4],
                "lora_multipliers": [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier],
                "dit_path": full_dit_path,  # Store the full path in metadata
                "vae_path": vae_path,
                "t5_path": t5_path,
                "negative_prompt": negative_prompt if negative_prompt else None,
                "sample_solver": sample_solver
            }
            
            add_metadata_to_video(video_path, parameters)
            videos.append((str(video_path), f"Seed: {current_seed}"))

    yield videos, f"Completed (seed: {current_seed})", ""

def wanx_v2v_batch_handler(
    prompt,
    negative_prompt,
    input_video,
    width,
    height,
    video_length,
    fps,
    infer_steps,
    flow_shift,
    guidance_scale,
    strength,
    seed,
    batch_size,
    task,
    dit_folder,  # folder path
    dit_path,    # model filename
    vae_path,
    t5_path,
    save_path,
    output_type,
    sample_solver,
    exclude_single_blocks,
    attn_mode,
    block_swap,
    fp8,
    fp8_scaled,
    fp8_t5,
    lora_folder,
    slg_layers: str,
    slg_start: Optional[str],
    slg_end: Optional[str],
    enable_cfg_skip: bool,
    cfg_skip_mode: str,
    cfg_apply_ratio: float,
    *lora_params
):
    """Handle batch generation for WanX v2v"""
    global stop_event
    stop_event.clear()
    
    # Extract LoRA parameters
    num_lora_weights = 4
    lora_weights = lora_params[:num_lora_weights]
    lora_multipliers = lora_params[num_lora_weights:num_lora_weights*2]
    
    all_videos = []
    progress_text = "Starting generation..."
    yield [], "Preparing...", progress_text
    
    # Process each item in the batch
    for i in range(batch_size):
        if stop_event.is_set():
            yield all_videos, "Generation stopped by user", ""
            return
            
        # Calculate seed for this batch item
        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif batch_size > 1:
            current_seed = seed + i
            
        batch_text = f"Generating video {i + 1} of {batch_size}"
        yield all_videos.copy(), batch_text, progress_text
        
        # Generate a single video
        for videos, status, progress in wanx_v2v_generate_video(
            prompt, 
            negative_prompt, 
            input_video, 
            width, 
            height, 
            video_length, 
            fps, 
            infer_steps, 
            flow_shift, 
            guidance_scale,
            strength,
            current_seed,
            task, 
            dit_folder,  # Pass folder path
            dit_path,    # Pass model filename
            vae_path, 
            t5_path, 
            save_path, 
            output_type, 
            sample_solver, 
            exclude_single_blocks,
            attn_mode, 
            block_swap, 
            fp8, 
            fp8_scaled, 
            fp8_t5,
            lora_folder,
            slg_layers,
            slg_start,
            slg_end,
            *lora_weights,
            *lora_multipliers,
            enable_cfg_skip,
            cfg_skip_mode,
            cfg_apply_ratio,
        ):
            if videos:
                all_videos.extend(videos)
            yield all_videos.copy(), f"Batch {i+1}/{batch_size}: {status}", progress
        
        # Clear CUDA cache between generations
        clear_cuda_cache()
        time.sleep(0.5)
    
    yield all_videos, "Batch complete", ""

def update_wanx_v2v_dimensions(video):
    """Update dimensions from uploaded video"""
    if video is None:
        return "", gr.update(value=832), gr.update(value=480)
    
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return "Error opening video", gr.update(), gr.update()
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Make dimensions divisible by 32
    w = (w // 32) * 32
    h = (h // 32) * 32
    
    return f"{w}x{h}", w, h

def send_wanx_v2v_to_hunyuan_v2v(
    gallery: list,
    prompt: str,
    selected_index: int,
    width: int,
    height: int,
    video_length: int,
    fps: int,
    infer_steps: int,
    seed: int,
    flow_shift: float,
    guidance_scale: float,
    negative_prompt: str
) -> Tuple:
    """Send the selected WanX v2v video to Hunyuan v2v tab"""
    if gallery is None or not gallery:
        return (None, "", width, height, video_length, fps, infer_steps, seed, 
                flow_shift, guidance_scale, negative_prompt)
    
    # If no selection made but we have videos, use the first one
    if selected_index is None and len(gallery) > 0:
        selected_index = 0
        
    if selected_index is None or selected_index >= len(gallery):
        return (None, "", width, height, video_length, fps, infer_steps, seed, 
                flow_shift, guidance_scale, negative_prompt)

    selected_item = gallery[selected_index]
    
    # Handle different gallery item formats
    if isinstance(selected_item, tuple):
        video_path = selected_item[0]
    elif isinstance(selected_item, dict):
        video_path = selected_item.get("name", selected_item.get("data", None))
    else:
        video_path = selected_item

    # Clean up path for Video component
    if isinstance(video_path, tuple):
        video_path = video_path[0]
        
    # Make sure it's a string
    video_path = str(video_path)

    return (video_path, prompt, width, height, video_length, fps, infer_steps, seed, 
            flow_shift, guidance_scale, negative_prompt)

def handle_wanx_v2v_gallery_select(evt: gr.SelectData) -> int:
    """Track selected index when gallery item is clicked"""
    return evt.index

def variance_of_laplacian(image):
    """
    Compute the variance of the Laplacian of the image.
    Higher variance indicates a sharper image.
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()

def extract_sharpest_frame(video_path, frames_to_check=30):
    """
    Extract the sharpest frame from the last N frames of the video.
    
    Args:
        video_path (str): Path to the video file
        frames_to_check (int): Number of frames from the end to check
        
    Returns:
        tuple: (temp_image_path, frame_number, sharpness_score)
    """
    print(f"\n=== Extracting sharpest frame from the last {frames_to_check} frames ===")
    print(f"Input video path: {video_path}")
    
    if not video_path or not os.path.exists(video_path):
        print("❌ Error: Video file does not exist")
        return None, None, None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error: Failed to open video file")
            return None, None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Total frames detected: {total_frames}, FPS: {fps:.2f}")
        
        if total_frames < 1:
            print("❌ Error: Video contains 0 frames")
            return None, None, None
        
        # Determine how many frames to check (the last N frames)
        if frames_to_check > total_frames:
            frames_to_check = total_frames
            start_frame = 0
        else:
            start_frame = total_frames - frames_to_check
        
        print(f"Checking frames {start_frame} to {total_frames-1}")
        
        # Find the sharpest frame
        sharpest_frame = None
        max_sharpness = -1
        sharpest_frame_number = -1
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames with a progress bar
        with tqdm(total=frames_to_check, desc="Finding sharpest frame") as pbar:
            frame_idx = start_frame
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale and calculate sharpness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = variance_of_laplacian(gray)
                
                # Update if this is the sharpest frame so far
                if sharpness > max_sharpness:
                    max_sharpness = sharpness
                    sharpest_frame = frame.copy()
                    sharpest_frame_number = frame_idx
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        if sharpest_frame is None:
            print("❌ Error: Failed to find a sharp frame")
            return None, None, None
        
        # Prepare output path
        temp_dir = os.path.abspath("temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"sharpest_frame_{os.path.basename(video_path)}.png")
        print(f"Saving frame to: {temp_path}")
        
        # Write and verify
        if not cv2.imwrite(temp_path, sharpest_frame):
            print("❌ Error: Failed to write frame to file")
            return None, None, None
            
        if not os.path.exists(temp_path):
            print("❌ Error: Output file not created")
            return None, None, None
        
        # Calculate frame time in seconds
        frame_time = sharpest_frame_number / fps
        
        print(f"✅ Extracted sharpest frame: {sharpest_frame_number} (at {frame_time:.2f}s) with sharpness {max_sharpness:.2f}")
        return temp_path, sharpest_frame_number, max_sharpness

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return None, None, None
    finally:
        if 'cap' in locals():
            cap.release()

def trim_video_to_frame(video_path, frame_number, output_dir="outputs"):
    """
    Trim video up to the specified frame and save as a new video.
    
    Args:
        video_path (str): Path to the video file
        frame_number (int): Frame number to trim to
        output_dir (str): Directory to save the trimmed video
        
    Returns:
        str: Path to the trimmed video file
    """
    print(f"\n=== Trimming video to frame {frame_number} ===")
    if not video_path or not os.path.exists(video_path):
        print("❌ Error: Video file does not exist")
        return None
    
    try:
        # Get video information
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error: Failed to open video file")
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Calculate time in seconds
        time_seconds = frame_number / fps
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        timestamp = f"{int(time_seconds)}s"
        base_name = Path(video_path).stem
        output_file = os.path.join(output_dir, f"{base_name}_trimmed_to_{timestamp}.mp4")
        
        # Use ffmpeg to trim the video
        (
            ffmpeg
            .input(video_path)
            .output(output_file, to=time_seconds, c="copy")
            .global_args('-y')  # Overwrite output files
            .run(quiet=True)
        )
        
        if not os.path.exists(output_file):
            print("❌ Error: Failed to create trimmed video")
            return None
            
        print(f"✅ Successfully trimmed video to {time_seconds:.2f}s: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Error trimming video: {str(e)}")
        return None

def send_sharpest_frame_handler(gallery, selected_idx, frames_to_check=30):
    """
    Extract the sharpest frame from the last N frames of the selected video
    
    Args:
        gallery: Gradio gallery component with videos
        selected_idx: Index of the selected video
        frames_to_check: Number of frames from the end to check
        
    Returns:
        tuple: (image_path, video_path, frame_number, sharpness)
    """
    if gallery is None or not gallery:
        return None, None, None, "No videos in gallery"
        
    if selected_idx is None and len(gallery) == 1:
        selected_idx = 0
        
    if selected_idx is None or selected_idx >= len(gallery):
        return None, None, None, "No video selected"
    
    # Get the video path
    item = gallery[selected_idx]
    if isinstance(item, tuple):
        video_path = item[0]
    elif isinstance(item, dict):
        video_path = item.get('name') or item.get('data')
    else:
        video_path = str(item)
    
    # Extract the sharpest frame
    image_path, frame_number, sharpness = extract_sharpest_frame(video_path, frames_to_check)
    
    if image_path is None:
        return None, None, None, "Failed to extract sharpest frame"
    
    return image_path, video_path, frame_number, f"Extracted frame {frame_number} with sharpness {sharpness:.2f}"

def trim_and_prepare_for_extension(video_path, frame_number, save_path="outputs"):
    """
    Trim the video to the specified frame and prepare for extension.
    
    Args:
        video_path: Path to the video file
        frame_number: Frame number to trim to
        save_path: Directory to save the trimmed video
        
    Returns:
        tuple: (trimmed_video_path, status_message)
    """
    if not video_path or not os.path.exists(video_path):
        return None, "No video selected or video file does not exist"
    
    if frame_number is None:
        return None, "No frame number provided, please extract sharpest frame first"
    
    # Trim the video
    trimmed_video = trim_video_to_frame(video_path, frame_number, save_path)
    
    if trimmed_video is None:
        return None, "Failed to trim video"
    
    return trimmed_video, f"Video trimmed to frame {frame_number} and ready for extension"

def send_last_frame_handler(gallery, selected_idx):
    """Handle sending last frame to input with better error handling"""
    if gallery is None or not gallery:
        return None, None
        
    if selected_idx is None and len(gallery) == 1:
        selected_idx = 0
        
    if selected_idx is None or selected_idx >= len(gallery):
        return None, None
        
    # Get the frame and video path
    frame = handle_last_frame_transfer(gallery, selected_idx)
    video_path = None
    
    if selected_idx < len(gallery):
        item = gallery[selected_idx]
        video_path = parse_video_path(item)
        
    return frame, video_path

def extract_last_frame(video_path: str) -> Optional[str]:
    """Extract last frame from video and return temporary image path with error handling"""
    print(f"\n=== Starting frame extraction ===")
    print(f"Input video path: {video_path}")
    
    if not video_path or not os.path.exists(video_path):
        print("❌ Error: Video file does not exist")
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error: Failed to open video file")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames detected: {total_frames}")
        
        if total_frames < 1:
            print("❌ Error: Video contains 0 frames")
            return None

        # Extract last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        success, frame = cap.read()
        
        if not success or frame is None:
            print("❌ Error: Failed to read last frame")
            return None

        # Prepare output path
        temp_dir = os.path.abspath("temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"last_frame_{os.path.basename(video_path)}.png")
        print(f"Saving frame to: {temp_path}")

        # Write and verify
        if not cv2.imwrite(temp_path, frame):
            print("❌ Error: Failed to write frame to file")
            return None
            
        if not os.path.exists(temp_path):
            print("❌ Error: Output file not created")
            return None

        print("✅ Frame extraction successful")
        return temp_path

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return None
    finally:
        if 'cap' in locals():
            cap.release()

def handle_last_frame_transfer(gallery: list, selected_idx: int) -> Optional[str]:
    """Improved frame transfer with video input validation"""
    try:
        if gallery is None or not gallery:
            raise ValueError("No videos generated yet")
            
        if selected_idx is None:
            # Auto-select last generated video if batch_size=1
            if len(gallery) == 1:
                selected_idx = 0
            else:
                raise ValueError("Please select a video first")
                
        if selected_idx >= len(gallery):
            raise ValueError("Invalid selection index")
            
        item = gallery[selected_idx]
        
        # Video file existence check
        video_path = parse_video_path(item)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file missing: {video_path}")
            
        return extract_last_frame(video_path)
        
    except Exception as e:
        print(f"Frame transfer failed: {str(e)}")
        return None

def parse_video_path(item) -> str:
    """Parse different gallery item formats"""
    if isinstance(item, tuple):
        return item[0]
    elif isinstance(item, dict):
        return item.get('name') or item.get('data')
    return str(item)

def get_random_image_from_folder(folder_path):
    """Get a random image from the specified folder"""
    if not os.path.isdir(folder_path):
        return None, f"Error: {folder_path} is not a valid directory"
    
    # Get all image files in the folder
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    for ext in ('*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.WEBP'):
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_files:
        return None, f"Error: No image files found in {folder_path}"
    
    # Select a random image
    random_image = random.choice(image_files)
    return random_image, f"Selected: {os.path.basename(random_image)}"

def resize_image_keeping_aspect_ratio(image_path, max_width, max_height):
    """Resize image keeping aspect ratio and ensuring dimensions are divisible by 16"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Calculate new dimensions while maintaining aspect ratio
        if width > height:
            new_width = min(max_width, width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(max_height, height)
            new_width = int(new_height * aspect_ratio)
        
        # Make dimensions divisible by 16
        new_width = math.floor(new_width / 16) * 16
        new_height = math.floor(new_height / 16) * 16
        
        # Ensure minimum size
        new_width = max(16, new_width)
        new_height = max(16, new_height)
        
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save to temporary file
        temp_path = f"temp_resized_{os.path.basename(image_path)}"
        resized_img.save(temp_path)
        
        return temp_path, (new_width, new_height)
    except Exception as e:
        return None, f"Error: {str(e)}"
# Function to process a batch of images from a folder
def batch_handler(
    use_random,
    prompt, negative_prompt, 
    width, height, 
    video_length, fps, infer_steps, 
    seed, flow_shift, guidance_scale, embedded_cfg_scale,
    batch_size, input_folder_path,
    dit_folder, model, vae, te1, te2, save_path, output_type, attn_mode, 
    block_swap, exclude_single_blocks, use_split_attn, use_fp8, split_uncond,
    lora_folder, *lora_params
):
    """Handle both folder-based batch processing and regular batch processing"""
    global stop_event
    
    # Check if this is a SkyReels model that needs special handling
    is_skyreels = "skyreels" in model.lower()
    is_skyreels_i2v = is_skyreels and "i2v" in model.lower()
    
    if use_random:
        # Random image from folder mode
        stop_event.clear()

        all_videos = []
        progress_text = "Starting generation..."
        yield [], "Preparing...", progress_text

        for i in range(batch_size):
            if stop_event.is_set():
                break

            batch_text = f"Generating video {i + 1} of {batch_size}"
            yield all_videos.copy(), batch_text, progress_text

            # Get random image from folder
            random_image, status = get_random_image_from_folder(input_folder_path)
            if random_image is None:
                yield all_videos, f"Error in batch {i+1}: {status}", ""
                continue

            # Resize image
            resized_image, size_info = resize_image_keeping_aspect_ratio(random_image, width, height)
            if resized_image is None:
                yield all_videos, f"Error resizing image in batch {i+1}: {size_info}", ""
                continue

            # If we have dimensions, update them
            local_width, local_height = width, height
            if isinstance(size_info, tuple):
                local_width, local_height = size_info
                progress_text = f"Using image: {os.path.basename(random_image)} - Resized to {local_width}x{local_height}"
            else:
                progress_text = f"Using image: {os.path.basename(random_image)}"
            
            yield all_videos.copy(), batch_text, progress_text

            # Calculate seed for this batch item
            current_seed = seed
            if seed == -1:
                current_seed = random.randint(0, 2**32 - 1)
            elif batch_size > 1:
                current_seed = seed + i

            # Process the image
            # For SkyReels models, we need to create a command with dit_in_channels=32
            if is_skyreels_i2v:
                env = os.environ.copy()
                env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
                env["PYTHONIOENCODING"] = "utf-8"
                
                model_path = os.path.join(dit_folder, model) if not os.path.isabs(model) else model
                
                # Extract parameters from lora_params
                num_lora_weights = 4
                lora_weights = lora_params[:num_lora_weights]
                lora_multipliers = lora_params[num_lora_weights:num_lora_weights*2]
                
                cmd = [
                    sys.executable,
                    "hv_generate_video.py",
                    "--dit", model_path,
                    "--vae", vae,
                    "--text_encoder1", te1,
                    "--text_encoder2", te2,
                    "--prompt", prompt,
                    "--video_size", str(local_height), str(local_width),
                    "--video_length", str(video_length),
                    "--fps", str(fps),
                    "--infer_steps", str(infer_steps),
                    "--save_path", save_path,
                    "--seed", str(current_seed),
                    "--flow_shift", str(flow_shift),
                    "--embedded_cfg_scale", str(embedded_cfg_scale),
                    "--output_type", output_type,
                    "--attn_mode", attn_mode,
                    "--blocks_to_swap", str(block_swap),
                    "--fp8_llm",
                    "--vae_chunk_size", "32",
                    "--vae_spatial_tile_sample_min_size", "128",
                    "--dit_in_channels", "32",  # This is crucial for SkyReels i2v
                    "--image_path", resized_image  # Pass the image directly
                ]
                
                if use_fp8:
                    cmd.append("--fp8")
                
                if split_uncond:
                    cmd.append("--split_uncond")
                
                if use_split_attn:
                    cmd.append("--split_attn")
                
                if exclude_single_blocks:
                    cmd.append("--exclude_single_blocks")
                
                if negative_prompt:
                    cmd.extend(["--negative_prompt", negative_prompt])
                    
                if guidance_scale is not None:
                    cmd.extend(["--guidance_scale", str(guidance_scale)])
                
                # Add LoRA weights and multipliers if provided
                valid_loras = []
                for weight, mult in zip(lora_weights, lora_multipliers):
                    if weight and weight != "None":
                        valid_loras.append((os.path.join(lora_folder, weight), mult))
                
                if valid_loras:
                    weights = [weight for weight, _ in valid_loras]
                    multipliers = [str(mult) for _, mult in valid_loras]
                    cmd.extend(["--lora_weight"] + weights)
                    cmd.extend(["--lora_multiplier"] + multipliers)
                
                print(f"Running command: {' '.join(cmd)}")
                
                # Run the process
                p = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1
                )
                
                while True:
                    if stop_event.is_set():
                        p.terminate()
                        p.wait()
                        yield all_videos, "Generation stopped by user.", ""
                        return
                        
                    line = p.stdout.readline()
                    if not line:
                        if p.poll() is not None:
                            break
                        continue
                        
                    print(line, end='')
                    if '|' in line and '%' in line and '[' in line and ']' in line:
                        yield all_videos.copy(), f"Processing video {i+1} (seed: {current_seed})", line.strip()
                
                p.stdout.close()
                p.wait()
                
                # Collect generated video
                save_path_abs = os.path.abspath(save_path)
                if os.path.exists(save_path_abs):
                    all_videos_files = sorted(
                        [f for f in os.listdir(save_path_abs) if f.endswith('.mp4')],
                        key=lambda x: os.path.getmtime(os.path.join(save_path_abs, x)),
                        reverse=True
                    )
                    matching_videos = [v for v in all_videos_files if f"_{current_seed}" in v]
                    if matching_videos:
                        video_path = os.path.join(save_path_abs, matching_videos[0])
                        all_videos.append((str(video_path), f"Seed: {current_seed}"))
            else:
                # For non-SkyReels models, use the regular process_single_video function
                num_lora_weights = 4
                lora_weights = lora_params[:num_lora_weights]
                lora_multipliers = lora_params[num_lora_weights:num_lora_weights*2]
                
                single_video_args = [
                    prompt, local_width, local_height, 1, video_length, fps, infer_steps,
                    current_seed, dit_folder, model, vae, te1, te2, save_path, flow_shift, embedded_cfg_scale,
                    output_type, attn_mode, block_swap, exclude_single_blocks, use_split_attn,
                    lora_folder
                ]
                single_video_args.extend(lora_weights)
                single_video_args.extend(lora_multipliers)
                single_video_args.extend([None, resized_image, None, negative_prompt, embedded_cfg_scale, split_uncond, guidance_scale, use_fp8])

                for videos, status, progress in process_single_video(*single_video_args):
                    if videos:
                        all_videos.extend(videos)
                    yield all_videos.copy(), f"Batch {i+1}/{batch_size}: {status}", progress

            # Clean up temporary file
            try:
                if os.path.exists(resized_image):
                    os.remove(resized_image)
            except:
                pass
            
            # Clear CUDA cache between generations
            clear_cuda_cache()
            time.sleep(0.5)

        yield all_videos, "Batch complete", ""
    else:
        # Regular image input - this is the part we need to fix
        # When a SkyReels I2V model is used, we need to use the direct command approach
        # with dit_in_channels=32 explicitly specified, just like in the folder processing branch
        if is_skyreels_i2v:
            stop_event.clear()
            
            all_videos = []
            progress_text = "Starting generation..."
            yield [], "Preparing...", progress_text
            
            # Extract lora parameters
            num_lora_weights = 4
            lora_weights = lora_params[:num_lora_weights]
            lora_multipliers = lora_params[num_lora_weights:num_lora_weights*2]
            extra_args = list(lora_params[num_lora_weights*2:]) if len(lora_params) > num_lora_weights*2 else []
            
            # Print extra_args for debugging
            print(f"Extra args: {extra_args}")
            
            # Get input image path from extra args - this is where we need to fix
            # In skyreels_generate_btn.click, we're passing skyreels_input which
            # should be the image path
            image_path = None
            if len(extra_args) > 0 and extra_args[0] is not None:
                image_path = extra_args[0]
                print(f"Image path found in extra_args[0]: {image_path}")
            
            # If we still don't have an image path, this is a problem
            if not image_path:
                # Let's try to debug what's happening - in the future, you can remove these
                # debug prints once everything works correctly
                print("No image path found in extra_args[0]")
                print(f"Full lora_params: {lora_params}")
                yield [], "Error: No input image provided", "An input image is required for SkyReels I2V models"
                return
            
            for i in range(batch_size):
                if stop_event.is_set():
                    yield all_videos, "Generation stopped by user", ""
                    return
                
                # Calculate seed for this batch item
                current_seed = seed
                if seed == -1:
                    current_seed = random.randint(0, 2**32 - 1)
                elif batch_size > 1:
                    current_seed = seed + i
                
                batch_text = f"Generating video {i + 1} of {batch_size}"
                yield all_videos.copy(), batch_text, progress_text
                
                # Set up environment
                env = os.environ.copy()
                env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
                env["PYTHONIOENCODING"] = "utf-8"
                
                model_path = os.path.join(dit_folder, model) if not os.path.isabs(model) else model
                
                # Build the command with dit_in_channels=32
                cmd = [
                    sys.executable,
                    "hv_generate_video.py",
                    "--dit", model_path,
                    "--vae", vae,
                    "--text_encoder1", te1,
                    "--text_encoder2", te2,
                    "--prompt", prompt,
                    "--video_size", str(height), str(width),
                    "--video_length", str(video_length),
                    "--fps", str(fps),
                    "--infer_steps", str(infer_steps),
                    "--save_path", save_path,
                    "--seed", str(current_seed),
                    "--flow_shift", str(flow_shift),
                    "--embedded_cfg_scale", str(embedded_cfg_scale),
                    "--output_type", output_type,
                    "--attn_mode", attn_mode,
                    "--blocks_to_swap", str(block_swap),
                    "--fp8_llm",
                    "--vae_chunk_size", "32",
                    "--vae_spatial_tile_sample_min_size", "128",
                    "--dit_in_channels", "32",  # This is crucial for SkyReels i2v
                    "--image_path", image_path
                ]
                
                if use_fp8:
                    cmd.append("--fp8")
                
                if split_uncond:
                    cmd.append("--split_uncond")
                
                if use_split_attn:
                    cmd.append("--split_attn")
                
                if exclude_single_blocks:
                    cmd.append("--exclude_single_blocks")
                
                if negative_prompt:
                    cmd.extend(["--negative_prompt", negative_prompt])
                    
                if guidance_scale is not None:
                    cmd.extend(["--guidance_scale", str(guidance_scale)])
                
                # Add LoRA weights and multipliers if provided
                valid_loras = []
                for weight, mult in zip(lora_weights, lora_multipliers):
                    if weight and weight != "None":
                        valid_loras.append((os.path.join(lora_folder, weight), mult))
                
                if valid_loras:
                    weights = [weight for weight, _ in valid_loras]
                    multipliers = [str(mult) for _, mult in valid_loras]
                    cmd.extend(["--lora_weight"] + weights)
                    cmd.extend(["--lora_multiplier"] + multipliers)
                
                print(f"Running command: {' '.join(cmd)}")
                
                # Run the process
                p = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1
                )
                
                while True:
                    if stop_event.is_set():
                        p.terminate()
                        p.wait()
                        yield all_videos, "Generation stopped by user.", ""
                        return
                        
                    line = p.stdout.readline()
                    if not line:
                        if p.poll() is not None:
                            break
                        continue
                        
                    print(line, end='')
                    if '|' in line and '%' in line and '[' in line and ']' in line:
                        yield all_videos.copy(), f"Processing (seed: {current_seed})", line.strip()
                
                p.stdout.close()
                p.wait()
                
                # Collect generated video
                save_path_abs = os.path.abspath(save_path)
                if os.path.exists(save_path_abs):
                    all_videos_files = sorted(
                        [f for f in os.listdir(save_path_abs) if f.endswith('.mp4')],
                        key=lambda x: os.path.getmtime(os.path.join(save_path_abs, x)),
                        reverse=True
                    )
                    matching_videos = [v for v in all_videos_files if f"_{current_seed}" in v]
                    if matching_videos:
                        video_path = os.path.join(save_path_abs, matching_videos[0])
                        all_videos.append((str(video_path), f"Seed: {current_seed}"))
                
                # Clear CUDA cache between generations
                clear_cuda_cache()
                time.sleep(0.5)
            
            yield all_videos, "Batch complete", ""
        else:
            # For regular non-SkyReels models, use the original process_batch function
            regular_args = [
                prompt, width, height, batch_size, video_length, fps, infer_steps,
                seed, dit_folder, model, vae, te1, te2, save_path, flow_shift, guidance_scale,
                output_type, attn_mode, block_swap, exclude_single_blocks, use_split_attn,
                lora_folder
            ]
            yield from process_batch(*(regular_args + list(lora_params)))

def get_dit_models(dit_folder: str) -> List[str]:
    """Get list of available DiT models in the specified folder"""
    if not os.path.exists(dit_folder):
        return ["mp_rank_00_model_states.pt"]
    models = [f for f in os.listdir(dit_folder) if f.endswith('.pt') or f.endswith('.safetensors')]
    models.sort(key=str.lower)
    return models if models else ["mp_rank_00_model_states.pt"]

def update_dit_dropdown(dit_folder: str) -> Dict:
    """Update DiT model dropdown choices."""
    models = get_dit_models(dit_folder)
    return gr.update(choices=models, value=models[0] if models else None)

def change_to_tab_two():
    return gr.Tabs(selected=2)

def update_dit_and_lora_dropdowns(dit_folder: str, lora_folder: str, *current_values) -> List[gr.update]:
    """Update both DiT and LoRA dropdowns"""
    # Get model lists
    dit_models = get_dit_models(dit_folder)
    lora_choices = get_lora_options(lora_folder)
    
    # Current values processing
    dit_value = current_values[0]
    if dit_value not in dit_models:
        dit_value = dit_models[0] if dit_models else None
        
    weights = current_values[1:5]
    multipliers = current_values[5:9]
    
    results = [gr.update(choices=dit_models, value=dit_value)]
    
    # Add LoRA updates
    for i in range(4):
        weight = weights[i] if i < len(weights) else "None"
        multiplier = multipliers[i] if i < len(multipliers) else 1.0
        if weight not in lora_choices:
            weight = "None"
        results.extend([
            gr.update(choices=lora_choices, value=weight),
            gr.update(value=multiplier)
        ])
    
    return results

def extract_video_metadata(video_path: str) -> Dict:
    """Extract metadata from video file using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        metadata = json.loads(result.stdout.decode('utf-8'))
        if 'format' in metadata and 'tags' in metadata['format']:
            comment = metadata['format']['tags'].get('comment', '{}')
            return json.loads(comment)
        return {}
    except Exception as e:
        print(f"Metadata extraction failed: {str(e)}")
        return {}

def create_parameter_transfer_map(metadata: Dict, target_tab: str) -> Dict:
    """Map metadata parameters to Gradio components for different tabs"""
    mapping = {
        'common': {
            'prompt': ('prompt', 'v2v_prompt', 'wanx_v2v_prompt'),  # Add WanX-v2v mapping
            'width': ('width', 'v2v_width', 'wanx_v2v_width'),
            'height': ('height', 'v2v_height', 'wanx_v2v_height'),
            'batch_size': ('batch_size', 'v2v_batch_size', 'wanx_v2v_batch_size'),
            'video_length': ('video_length', 'v2v_video_length', 'wanx_v2v_video_length'),
            'fps': ('fps', 'v2v_fps', 'wanx_v2v_fps'),
            'infer_steps': ('infer_steps', 'v2v_infer_steps', 'wanx_v2v_infer_steps'),
            'seed': ('seed', 'v2v_seed', 'wanx_v2v_seed'),
            'flow_shift': ('flow_shift', 'v2v_flow_shift', 'wanx_v2v_flow_shift'),
            'guidance_scale': ('cfg_scale', 'v2v_cfg_scale', 'wanx_v2v_guidance_scale'),
            'negative_prompt': ('negative_prompt', 'v2v_negative_prompt', 'wanx_v2v_negative_prompt'),
            'strength': ('strength', 'v2v_strength', 'wanx_v2v_strength')
        },
        'lora': {
            'lora_weights': [(f'lora{i+1}', f'v2v_lora_weights[{i}]', f'wanx_v2v_lora_weights[{i}]') for i in range(4)],
            'lora_multipliers': [(f'lora{i+1}_multiplier', f'v2v_lora_multipliers[{i}]', f'wanx_v2v_lora_multipliers[{i}]') for i in range(4)]
        }
    }
    
    results = {}
    for param, value in metadata.items():
        # Handle common parameters
        if param in mapping['common']:
            target_idx = 0 if target_tab == 't2v' else 1 if target_tab == 'v2v' else 2
            if target_idx < len(mapping['common'][param]):
                target = mapping['common'][param][target_idx]
                results[target] = value
        
        # Handle LoRA parameters
        if param == 'lora_weights':
            for i, weight in enumerate(value[:4]):
                target_idx = 0 if target_tab == 't2v' else 1 if target_tab == 'v2v' else 2
                if target_idx < len(mapping['lora']['lora_weights'][i]):
                    target = mapping['lora']['lora_weights'][i][target_idx]
                    results[target] = weight
                
        if param == 'lora_multipliers':
            for i, mult in enumerate(value[:4]):
                target_idx = 0 if target_tab == 't2v' else 1 if target_tab == 'v2v' else 2
                if target_idx < len(mapping['lora']['lora_multipliers'][i]):
                    target = mapping['lora']['lora_multipliers'][i][target_idx]
                    results[target] = float(mult)
                
    return results

def add_metadata_to_video(video_path: str, parameters: dict) -> None:
    """Add generation parameters to video metadata using ffmpeg."""
    import json
    import subprocess

    # Convert parameters to JSON string
    params_json = json.dumps(parameters, indent=2)
    
    # Temporary output path
    temp_path = video_path.replace(".mp4", "_temp.mp4")
    
    # Add Fun-Control information to metadata if applicable
    task = parameters.get("task", "")
    if task.endswith("-FC"):
        parameters["fun_control"] = True
        # Store the control path in metadata if available
        if "control_path" in parameters:
            parameters["control_video"] = os.path.basename(parameters["control_path"])
    
    # FFmpeg command to add metadata without re-encoding
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-metadata', f'comment={params_json}',
        '-codec', 'copy',
        temp_path
    ]
    
    try:
        # Execute FFmpeg command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Replace original file with the metadata-enhanced version
        os.replace(temp_path, video_path)
    except subprocess.CalledProcessError as e:
        print(f"Failed to add metadata: {e.stderr.decode()}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"Error: {str(e)}")

def count_prompt_tokens(prompt: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(prompt)
    return len(tokens)


def get_lora_options(lora_folder: str = "lora") -> List[str]:
    if not os.path.exists(lora_folder):
        return ["None"]
    lora_files = [f for f in os.listdir(lora_folder) if f.endswith('.safetensors') or f.endswith('.pt')]
    lora_files.sort(key=str.lower)
    return ["None"] + lora_files

def update_lora_dropdowns(lora_folder: str, *current_values) -> List[gr.update]:
    new_choices = get_lora_options(lora_folder)
    weights = current_values[:4]
    multipliers = current_values[4:8]
    
    results = []
    for i in range(4):
        weight = weights[i] if i < len(weights) else "None"
        multiplier = multipliers[i] if i < len(multipliers) else 1.0
        if weight not in new_choices:
            weight = "None"
        results.extend([
            gr.update(choices=new_choices, value=weight),
            gr.update(value=multiplier) 
        ])
    
    return results

def send_to_v2v(evt: gr.SelectData, gallery: list, prompt: str, selected_index: gr.State) -> Tuple[Optional[str], str, int]:
    """Transfer selected video and prompt to Video2Video tab"""
    if not gallery or evt.index >= len(gallery):
        return None, "", selected_index.value
    
    selected_item = gallery[evt.index]
    
    # Handle different gallery item formats
    if isinstance(selected_item, dict):
        video_path = selected_item.get("name", selected_item.get("data", None))
    elif isinstance(selected_item, (tuple, list)):
        video_path = selected_item[0]
    else:
        video_path = selected_item
    
    # Final cleanup for Gradio Video component
    if isinstance(video_path, tuple):
        video_path = video_path[0]
    
    # Update the selected index
    selected_index.value = evt.index
    
    return str(video_path), prompt, evt.index

def send_selected_to_v2v(gallery: list, prompt: str, selected_index: gr.State) -> Tuple[Optional[str], str]:
    """Send the currently selected video to V2V tab"""
    if not gallery or selected_index.value is None or selected_index.value >= len(gallery):
        return None, ""
    
    selected_item = gallery[selected_index.value]
    
    # Handle different gallery item formats
    if isinstance(selected_item, dict):
        video_path = selected_item.get("name", selected_item.get("data", None))
    elif isinstance(selected_item, (tuple, list)):
        video_path = selected_item[0]
    else:
        video_path = selected_item
    
    # Final cleanup for Gradio Video component
    if isinstance(video_path, tuple):
        video_path = video_path[0]
    
    return str(video_path), prompt

def clear_cuda_cache():
    """Clear CUDA cache if available"""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Optional: synchronize to ensure cache is cleared
        torch.cuda.synchronize()

def wanx_batch_handler(
    use_random,
    prompt, 
    negative_prompt,
    width,
    height,
    video_length,
    fps,
    infer_steps,
    flow_shift,
    guidance_scale,
    seed,
    batch_size,
    input_folder_path,
    wanx_input_end,
    task,
    dit_folder,
    dit_path,
    vae_path,
    t5_path,
    clip_path,
    save_path,
    output_type,
    sample_solver,
    exclude_single_blocks,
    attn_mode,
    block_swap,
    fp8,
    fp8_scaled,
    fp8_t5,
    lora_folder,
    slg_layers: str,
    slg_start: Optional[str],
    slg_end: Optional[str],
    enable_cfg_skip: bool,
    cfg_skip_mode: str,
    cfg_apply_ratio: float,
    enable_preview: bool,
    preview_steps: int,
    *lora_params,   # <-- DO NOT ADD NAMED ARGS AFTER THIS!
):
    """Handle both folder-based batch processing and regular processing for all WanX tabs"""
    global stop_event
    
    # Convert None strings to actual None
    slg_layers = None if slg_layers == "None" else slg_layers
    slg_start = None if slg_start == "None" else slg_start
    slg_end = None if slg_end == "None" else slg_end
    
    # Construct full dit_path including folder
    full_dit_path = os.path.join(dit_folder, dit_path) if not os.path.isabs(dit_path) else dit_path    
    # Clean up LoRA params to proper format
    clean_lora_params = []
    for param in lora_params:
        # Convert None strings to "None" for consistency
        if param is None or str(param).lower() == "none":
            clean_lora_params.append("None")
        else:
            clean_lora_params.append(str(param))
    
    # Extract LoRA weights and multipliers
    num_lora_weights = 4
    lora_weights = clean_lora_params[:num_lora_weights]
    lora_multipliers = []
    for mult in clean_lora_params[num_lora_weights:num_lora_weights*2]:
        try:
            lora_multipliers.append(float(mult))
        except (ValueError, TypeError):
            lora_multipliers.append(1.0)
    while len(lora_weights) < 4:
        lora_weights.append("None")
    while len(lora_multipliers) < 4:
        lora_multipliers.append(1.0)

    # Now extract trailing params: input_file, control_video, control_strength, control_start, control_end
    remaining_params = clean_lora_params[num_lora_weights*2:]
    input_file = remaining_params[0] if len(remaining_params) > 0 else None
    control_video = remaining_params[1] if len(remaining_params) > 1 else None
    try:
        control_strength = float(remaining_params[2]) if len(remaining_params) > 2 else 1.0
    except Exception:
        control_strength = 1.0
    try:
        control_start = float(remaining_params[3]) if len(remaining_params) > 3 else 0.0
    except Exception:
        control_start = 0.0
    try:
        control_end = float(remaining_params[4]) if len(remaining_params) > 4 else 1.0
    except Exception:
        control_end = 1.0

    yield [], [], "Preparing batch...", "" # Clear main and preview galleries

    if use_random:
        stop_event.clear()
        all_videos = []
        all_previews = [] # Keep track of previews from the last successful item? Or clear each time? Let's clear.
        progress_text = "Starting generation..."
        yield [], [], "Preparing...", progress_text # Clear galleries again just in case
        batch_size = int(batch_size)
        for i in range(batch_size):
            if stop_event.is_set():
                yield all_videos, [], "Generation stopped by user", "" # Yield empty previews on stop
                return

            # --- Clear previews for this item ---
            current_previews_for_item = []
            yield all_videos.copy(), current_previews_for_item, f"Generating video {i + 1} of {batch_size}", progress_text # Yield cleared previews

            # ... (Keep existing random image logic: get random, resize) ...
            random_image, status = get_random_image_from_folder(input_folder_path)
            if random_image is None:
                yield all_videos, current_previews_for_item, f"Error in batch {i+1}: {status}", ""
                continue # Skip to next batch item on error

            resized_image, size_info = resize_image_keeping_aspect_ratio(random_image, width, height)
            if resized_image is None:
                yield all_videos, current_previews_for_item, f"Error resizing image in batch {i+1}: {size_info}", ""
                # Clean up the random image if resize failed but image exists
                try:
                    if os.path.exists(random_image) and "temp_resized" not in random_image: # Avoid double delete if resize output existed
                       pass # Might not want to delete original random image here
                except: pass
                continue # Skip to next batch item on error

            local_width, local_height = width, height
            if isinstance(size_info, tuple): local_width, local_height = size_info
            progress_text = f"Using image: {os.path.basename(random_image)} - Resized to {local_width}x{local_height}"
            yield all_videos.copy(), current_previews_for_item, f"Generating video {i + 1} of {batch_size}", progress_text

            current_seed = seed
            if seed == -1: current_seed = random.randint(0, 2**32 - 1)
            elif batch_size > 1: current_seed = seed + i

            # --- Corrected call to wanx_generate_video with accumulation ---
            newly_generated_video = None # Track the video generated *in this iteration*
            last_status_for_item = f"Generating video {i+1}/{batch_size}" # Keep track of last status
            last_progress_for_item = progress_text # Keep track of last progress line

            # Inner loop iterates through the generator for ONE batch item
            for videos_update, previews_update, status, progress in wanx_generate_video(
                prompt, negative_prompt, resized_image, local_width, local_height,
                video_length, fps, infer_steps, flow_shift, guidance_scale, current_seed,
                wanx_input_end, # Pass the argument
                task, dit_folder, full_dit_path, vae_path, t5_path, clip_path, save_path,
                output_type, sample_solver, exclude_single_blocks, attn_mode, block_swap,
                fp8, fp8_scaled, fp8_t5, lora_folder,
                slg_layers, slg_start, slg_end,
                lora_weights[0], lora_weights[1], lora_weights[2], lora_weights[3],
                lora_multipliers[0], lora_multipliers[1], lora_multipliers[2], lora_multipliers[3],
                enable_cfg_skip, cfg_skip_mode, cfg_apply_ratio,
                None, 1.0, 0.0, 1.0, # Placeholders for control video args in random mode
                enable_preview=enable_preview,
                preview_steps=preview_steps
            ):
                # Store the latest video info from this *specific* generator run
                if videos_update:
                    # wanx_generate_video yields the *full* list it knows about,
                    # so we take the last item assuming it's the new one.
                    newly_generated_video = videos_update[-1]

                current_previews_for_item = previews_update # Update previews for *this* item
                last_status_for_item = f"Batch {i+1}/{batch_size}: {status}" # Store last status
                last_progress_for_item = progress # Store last progress line
                # Yield the *current cumulative* list during progress updates
                yield all_videos.copy(), current_previews_for_item, last_status_for_item, last_progress_for_item

            # --- After the inner loop finishes for item 'i' ---
            # Now, add the video generated in this iteration to the main list
            if newly_generated_video and newly_generated_video not in all_videos:
                all_videos.append(newly_generated_video)
                print(f"DEBUG: Appended video {newly_generated_video[1] if isinstance(newly_generated_video, tuple) else 'unknown'} to all_videos (Total: {len(all_videos)})")
                # Yield the updated cumulative list *immediately* after appending
                yield all_videos.copy(), current_previews_for_item, last_status_for_item, last_progress_for_item
            elif not newly_generated_video:
                 print(f"DEBUG: No new video generated or yielded by wanx_generate_video for batch item {i+1}.")


            # --- Cleanup for item 'i' (Correctly indented) ---
            try:
                # Only remove the temporary resized image
                if os.path.exists(resized_image) and "temp_resized" in resized_image:
                     os.remove(resized_image)
                     print(f"DEBUG: Removed temporary resized image: {resized_image}")
            except Exception as e:
                print(f"Warning: Could not remove temp image {resized_image}: {e}")
            clear_cuda_cache()
            time.sleep(0.5)
            # --- End Cleanup for item 'i' ---

        # --- After the outer loop (all batch items processed) ---
        yield all_videos, [], "Batch complete", "" # Yield empty previews at the end
    else:
        # ... (Keep existing checks for non-random mode: input file, control video) ...
        batch_size = int(batch_size)
        if not input_file and "i2v" in task:
            yield [], [], "Error: No input image provided", "An input image is required for I2V models"
            return
        if "-FC" in task and not control_video:
            yield [], [], "Error: No control video provided", "A control video is required for Fun-Control models"
            return

        if batch_size > 1:
            stop_event.clear()
            all_videos = []
            all_previews = [] # Clear previews at start of batch
            progress_text = "Starting generation..."
            yield [], [], "Preparing...", progress_text # Clear galleries

            for i in range(batch_size):
                if stop_event.is_set():
                    yield all_videos, [], "Generation stopped by user", "" # Yield empty previews
                    return

                # --- Clear previews for this item ---
                current_previews_for_item = []
                yield all_videos.copy(), current_previews_for_item, f"Generating video {i+1}/{batch_size}", progress_text

                current_seed = seed
                if seed == -1: current_seed = random.randint(0, 2**32 - 1)
                elif batch_size > 1: current_seed = seed + i
                batch_text = f"Generating video {i+1}/{batch_size} (seed: {current_seed})"
                yield all_videos.copy(), current_previews_for_item, batch_text, progress_text # Update status

                # --- Corrected call to wanx_generate_video with accumulation ---
                newly_generated_video = None # Track the video generated *in this iteration*
                last_status_for_item = f"Generating video {i+1}/{batch_size}" # Keep track of last status
                last_progress_for_item = progress_text # Keep track of last progress line

                # Inner loop iterates through the generator for ONE batch item
                for videos_update, previews_update, status, progress in wanx_generate_video(
                    prompt, negative_prompt, input_file, width, height,
                    video_length, fps, infer_steps, flow_shift, guidance_scale, current_seed,
                    wanx_input_end, # Pass the argument
                    task, dit_folder, full_dit_path, vae_path, t5_path, clip_path, save_path,
                    output_type, sample_solver, exclude_single_blocks, attn_mode, block_swap,
                    fp8, fp8_scaled, fp8_t5, lora_folder,
                    slg_layers, slg_start, slg_end,
                    lora_weights[0], lora_weights[1], lora_weights[2], lora_weights[3],
                    lora_multipliers[0], lora_multipliers[1], lora_multipliers[2], lora_multipliers[3],
                    enable_cfg_skip, cfg_skip_mode, cfg_apply_ratio,
                    control_video, control_strength, control_start, control_end,
                    # --- Pass preview args ---
                    enable_preview=enable_preview,
                    preview_steps=preview_steps
                ):
                     # Store the latest video info from this *specific* generator run
                    if videos_update:
                        # wanx_generate_video yields the *full* list it knows about,
                        # so we take the last item assuming it's the new one.
                        newly_generated_video = videos_update[-1]

                    current_previews_for_item = previews_update # Update previews for *this* item
                    last_status_for_item = f"Batch {i+1}/{batch_size}: {status}" # Store last status
                    last_progress_for_item = progress # Store last progress line
                    # Yield the *current cumulative* list during progress updates
                    yield all_videos.copy(), current_previews_for_item, last_status_for_item, last_progress_for_item

                # --- After the inner loop finishes for item 'i' ---
                # Now, add the video generated in this iteration to the main list
                if newly_generated_video and newly_generated_video not in all_videos:
                    all_videos.append(newly_generated_video)
                    print(f"DEBUG: Appended video {newly_generated_video[1] if isinstance(newly_generated_video, tuple) else 'unknown'} to all_videos (Total: {len(all_videos)})")
                    # Yield the updated cumulative list *immediately* after appending
                    yield all_videos.copy(), current_previews_for_item, last_status_for_item, last_progress_for_item
                elif not newly_generated_video:
                    print(f"DEBUG: No new video generated or yielded by wanx_generate_video for batch item {i+1}.")
                # --- End modified call ---

                clear_cuda_cache()
                time.sleep(0.5)
            yield all_videos, [], "Batch complete", "" # Yield empty previews at the end
        else: # Single generation (batch_size = 1)
            stop_event.clear()
            # --- Modified call to wanx_generate_video (yield from) ---
            # Add preview args directly
            yield from wanx_generate_video(
                prompt, negative_prompt, input_file, width, height,
                video_length, fps, infer_steps, flow_shift, guidance_scale, seed,
                wanx_input_end, # Pass the argument
                task, dit_folder, full_dit_path, vae_path, t5_path, clip_path, save_path,
                output_type, sample_solver, exclude_single_blocks, attn_mode, block_swap,
                fp8, fp8_scaled, fp8_t5, lora_folder,
                slg_layers, slg_start, slg_end,
                lora_weights[0], lora_weights[1], lora_weights[2], lora_weights[3],
                lora_multipliers[0], lora_multipliers[1], lora_multipliers[2], lora_multipliers[3],
                enable_cfg_skip, cfg_skip_mode, cfg_apply_ratio,
                control_video, control_strength, control_start, control_end,
                # --- Pass preview args ---
                enable_preview=enable_preview,
                preview_steps=preview_steps
            )

def process_single_video(
    prompt: str,
    width: int,
    height: int,
    batch_size: int,
    video_length: int,
    fps: int,
    infer_steps: int,
    seed: int,
    dit_folder: str,
    model: str,
    vae: str,
    te1: str,
    te2: str,
    save_path: str,
    flow_shift: float,
    cfg_scale: float,
    output_type: str,
    attn_mode: str,
    block_swap: int,
    exclude_single_blocks: bool,
    use_split_attn: bool,    
    lora_folder: str,
    lora1: str = "",
    lora2: str = "",
    lora3: str = "",
    lora4: str = "",
    lora1_multiplier: float = 1.0,
    lora2_multiplier: float = 1.0,
    lora3_multiplier: float = 1.0,
    lora4_multiplier: float = 1.0,
    video_path: Optional[str] = None,
    image_path: Optional[str] = None,
    strength: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    embedded_cfg_scale: Optional[float] = None,
    split_uncond: Optional[bool] = None,
    guidance_scale: Optional[float] = None,
    use_fp8: bool = True
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """Generate a single video with the given parameters"""
    global stop_event
    
    if stop_event.is_set():
        yield [], "", ""
        return

    # Determine if this is a SkyReels model and what type
    is_skyreels = "skyreels" in model.lower()
    is_skyreels_i2v = is_skyreels and "i2v" in model.lower()
    is_skyreels_t2v = is_skyreels and "t2v" in model.lower()
    
    if is_skyreels:
        # Force certain parameters for SkyReels
        if negative_prompt is None:
            negative_prompt = ""
        if embedded_cfg_scale is None:
            embedded_cfg_scale = 1.0  # Force to 1.0 for SkyReels
        if split_uncond is None:
            split_uncond = True
        if guidance_scale is None:
            guidance_scale = cfg_scale  # Use cfg_scale as guidance_scale if not provided
            
        # Determine the input channels based on model type
        if is_skyreels_i2v:
            dit_in_channels = 32  # SkyReels I2V uses 32 channels
        else:
            dit_in_channels = 16  # SkyReels T2V uses 16 channels (same as regular models)
    else:
        dit_in_channels = 16  # Regular Hunyuan models use 16 channels
        embedded_cfg_scale = cfg_scale 

    if os.path.isabs(model):
        model_path = model
    else:
        model_path = os.path.normpath(os.path.join(dit_folder, model))
    
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["BATCH_RUN_ID"] = f"{time.time()}"

    if seed == -1:
        current_seed = random.randint(0, 2**32 - 1)
    else:
        batch_id = int(env.get("BATCH_RUN_ID", "0").split('.')[-1])
        if batch_size > 1:  # Only modify seed for batch generation
            current_seed = (seed + batch_id * 100003) % (2**32)
        else:
            current_seed = seed

    clear_cuda_cache()

    command = [
        sys.executable,
        "hv_generate_video.py",
        "--dit", model_path,
        "--vae", vae,
        "--text_encoder1", te1,
        "--text_encoder2", te2,
        "--prompt", prompt,
        "--video_size", str(height), str(width),
        "--video_length", str(video_length),
        "--fps", str(fps),
        "--infer_steps", str(infer_steps),
        "--save_path", save_path,
        "--seed", str(current_seed),
        "--flow_shift", str(flow_shift),
        "--embedded_cfg_scale", str(cfg_scale),
        "--output_type", output_type,
        "--attn_mode", attn_mode,
        "--blocks_to_swap", str(block_swap),
        "--fp8_llm",
        "--vae_chunk_size", "32",
        "--vae_spatial_tile_sample_min_size", "128"
    ]
    
    if use_fp8:
        command.append("--fp8")

    # Add negative prompt and embedded cfg scale for SkyReels
    if is_skyreels:
        command.extend(["--dit_in_channels", str(dit_in_channels)])
        command.extend(["--guidance_scale", str(guidance_scale)])
        
        if negative_prompt:
            command.extend(["--negative_prompt", negative_prompt])
        if split_uncond:
            command.append("--split_uncond")

    # Add LoRA weights and multipliers if provided
    valid_loras = []
    for weight, mult in zip([lora1, lora2, lora3, lora4], 
                          [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier]):
        if weight and weight != "None":
            valid_loras.append((os.path.join(lora_folder, weight), mult))
    if valid_loras:
        weights = [weight for weight, _ in valid_loras]
        multipliers = [str(mult) for _, mult in valid_loras]
        command.extend(["--lora_weight"] + weights)
        command.extend(["--lora_multiplier"] + multipliers)

    if exclude_single_blocks:
        command.append("--exclude_single_blocks")
    if use_split_attn:
        command.append("--split_attn")

    # Handle input paths
    if video_path:
        command.extend(["--video_path", video_path])
        if strength is not None:
            command.extend(["--strength", str(strength)])
    elif image_path:
        command.extend(["--image_path", image_path])
        # Only add strength parameter for non-SkyReels I2V models
        # SkyReels I2V doesn't use strength parameter for image-to-video generation
        if strength is not None and not is_skyreels_i2v:
            command.extend(["--strength", str(strength)])
            
    print(f"{command}")

    p = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )

    videos = []
    
    while True:
        if stop_event.is_set():
            p.terminate()
            p.wait()
            yield [], "", "Generation stopped by user."
            return

        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                break
            continue
            
        print(line, end='')
        if '|' in line and '%' in line and '[' in line and ']' in line:
            yield videos.copy(), f"Processing (seed: {current_seed})", line.strip()

    p.stdout.close()
    p.wait()

    clear_cuda_cache()
    time.sleep(0.5)

    # Collect generated video
    save_path_abs = os.path.abspath(save_path)
    if os.path.exists(save_path_abs):
        all_videos = sorted(
            [f for f in os.listdir(save_path_abs) if f.endswith('.mp4')],
            key=lambda x: os.path.getmtime(os.path.join(save_path_abs, x)),
            reverse=True
        )
        matching_videos = [v for v in all_videos if f"_{current_seed}" in v]
        if matching_videos:
            video_path = os.path.join(save_path_abs, matching_videos[0])
            
            # Collect parameters for metadata
            parameters = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "video_length": video_length,
                "fps": fps,
                "infer_steps": infer_steps,
                "seed": current_seed,
                "model": model,
                "vae": vae,
                "te1": te1,
                "te2": te2,
                "save_path": save_path,
                "flow_shift": flow_shift,
                "cfg_scale": cfg_scale,
                "output_type": output_type,
                "attn_mode": attn_mode,
                "block_swap": block_swap,
                "lora_weights": [lora1, lora2, lora3, lora4],
                "lora_multipliers": [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier],
                "input_video": video_path if video_path else None,
                "input_image": image_path if image_path else None,
                "strength": strength,
                "negative_prompt": negative_prompt if is_skyreels else None,
                "embedded_cfg_scale": embedded_cfg_scale if is_skyreels else None
            }
            
            add_metadata_to_video(video_path, parameters)
            videos.append((str(video_path), f"Seed: {current_seed}"))

    yield videos, f"Completed (seed: {current_seed})", ""

# The issue is in the process_batch function, in the section that handles different input types
# Here's the corrected version of that section:

def process_batch(
    prompt: str,
    width: int,
    height: int,
    batch_size: int,
    video_length: int,
    fps: int,
    infer_steps: int,
    seed: int,
    dit_folder: str,
    model: str,
    vae: str,
    te1: str,
    te2: str,
    save_path: str,
    flow_shift: float,
    cfg_scale: float,
    output_type: str,
    attn_mode: str,
    block_swap: int,
    exclude_single_blocks: bool,
    use_split_attn: bool,
    lora_folder: str,
    *args
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """Process a batch of videos using Gradio's queue"""
    global stop_event
    stop_event.clear()

    all_videos = []
    progress_text = "Starting generation..."
    yield [], "Preparing...", progress_text

    # Extract additional arguments
    num_lora_weights = 4
    lora_weights = args[:num_lora_weights]
    lora_multipliers = args[num_lora_weights:num_lora_weights*2]
    extra_args = args[num_lora_weights*2:]

    # Determine if this is a SkyReels model and what type
    is_skyreels = "skyreels" in model.lower()
    is_skyreels_i2v = is_skyreels and "i2v" in model.lower()
    is_skyreels_t2v = is_skyreels and "t2v" in model.lower()

    # Handle input paths and additional parameters
    input_path = extra_args[0] if extra_args else None
    strength = float(extra_args[1]) if len(extra_args) > 1 else None
    
    # Get use_fp8 flag (it should be the last parameter)
    use_fp8 = bool(extra_args[-1]) if extra_args and len(extra_args) >= 3 else True
    
    # Get SkyReels specific parameters if applicable
    if is_skyreels:
        # Always set embedded_cfg_scale to 1.0 for SkyReels models
        embedded_cfg_scale = 1.0
        
        negative_prompt = str(extra_args[2]) if len(extra_args) > 2 and extra_args[2] is not None else ""
        # Use cfg_scale for guidance_scale parameter
        guidance_scale = float(extra_args[3]) if len(extra_args) > 3 and extra_args[3] is not None else cfg_scale
        split_uncond = True if len(extra_args) > 4 and extra_args[4] else False
    else:
        negative_prompt = str(extra_args[2]) if len(extra_args) > 2 and extra_args[2] is not None else None
        guidance_scale = cfg_scale
        embedded_cfg_scale = cfg_scale
        split_uncond = bool(extra_args[4]) if len(extra_args) > 4 else None

    for i in range(batch_size):
        if stop_event.is_set():
            break

        batch_text = f"Generating video {i + 1} of {batch_size}"
        yield all_videos.copy(), batch_text, progress_text

        # Handle different input types
        video_path = None
        image_path = None
        
        if input_path:
            # Check if it's an image file (common image extensions)
            is_image = False
            lower_path = input_path.lower()
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
            is_image = any(lower_path.endswith(ext) for ext in image_extensions)
            
            # Only use image_path for SkyReels I2V models and actual image files
            if is_skyreels_i2v and is_image:
                image_path = input_path
            else:
                video_path = input_path

        # Prepare arguments for process_single_video
        single_video_args = [
            prompt, width, height, batch_size, video_length, fps, infer_steps,
            seed, dit_folder, model, vae, te1, te2, save_path, flow_shift, cfg_scale,
            output_type, attn_mode, block_swap, exclude_single_blocks, use_split_attn,
            lora_folder
        ]
        single_video_args.extend(lora_weights)
        single_video_args.extend(lora_multipliers)
        single_video_args.extend([video_path, image_path, strength, negative_prompt, embedded_cfg_scale, split_uncond, guidance_scale, use_fp8])

        for videos, status, progress in process_single_video(*single_video_args):
            if videos:
                all_videos.extend(videos)
            yield all_videos.copy(), f"Batch {i+1}/{batch_size}: {status}", progress

    yield all_videos, "Batch complete", ""

def update_wanx_image_dimensions(image):
    """Update dimensions from uploaded image"""
    if image is None:
        return "", gr.update(value=832), gr.update(value=480)
    img = Image.open(image)
    w, h = img.size
    w = (w // 32) * 32
    h = (h // 32) * 32
    return f"{w}x{h}", w, h

def calculate_wanx_width(height, original_dims):
    """Calculate width based on height maintaining aspect ratio"""
    if not original_dims:
        return gr.update()
    orig_w, orig_h = map(int, original_dims.split('x'))
    aspect_ratio = orig_w / orig_h
    new_width = math.floor((height * aspect_ratio) / 32) * 32
    return gr.update(value=new_width)

def calculate_wanx_height(width, original_dims):
    """Calculate height based on width maintaining aspect ratio"""
    if not original_dims:
        return gr.update()
    orig_w, orig_h = map(int, original_dims.split('x'))
    aspect_ratio = orig_w / orig_h
    new_height = math.floor((width / aspect_ratio) / 32) * 32
    return gr.update(value=new_height)

def update_wanx_from_scale(scale, original_dims):
    """Update dimensions based on scale percentage"""
    if not original_dims:
        return gr.update(), gr.update()
    orig_w, orig_h = map(int, original_dims.split('x'))
    new_w = math.floor((orig_w * scale / 100) / 32) * 32
    new_h = math.floor((orig_h * scale / 100) / 32) * 32
    return gr.update(value=new_w), gr.update(value=new_h)

def recommend_wanx_flow_shift(width, height):
    """Get recommended flow shift value based on dimensions"""
    recommended_shift = 3.0 if (width == 832 and height == 480) or (width == 480 and height == 832) else 5.0
    return gr.update(value=recommended_shift)

def handle_wanx_gallery_select(evt: gr.SelectData, gallery) -> tuple:
    """Track selected index and video path when gallery item is clicked"""
    if gallery is None:
        return None, None
    
    if evt.index >= len(gallery):
        return None, None
    
    selected_item = gallery[evt.index]
    video_path = None
    
    # Extract the video path based on the item type
    if isinstance(selected_item, tuple):
        video_path = selected_item[0]
    elif isinstance(selected_item, dict):
        video_path = selected_item.get("name", selected_item.get("data", None))
    else:
        video_path = selected_item
    
    return evt.index, video_path

def get_step_from_preview_path(path):
    match = re.search(r"step_(\d+)_", os.path.basename(path))
    return int(match.group(1)) if match else -1

def wanx_generate_video(
    prompt, 
    negative_prompt,
    input_image,
    width,
    height,
    video_length,
    fps,
    infer_steps,
    flow_shift,
    guidance_scale,
    seed,
    wanx_input_end,
    task,
    dit_folder,
    dit_path,
    vae_path,
    t5_path,
    clip_path,
    save_path,
    output_type,
    sample_solver,
    exclude_single_blocks,
    attn_mode,
    block_swap,
    fp8,
    fp8_scaled,
    fp8_t5,
    lora_folder,
    slg_layers,
    slg_start,
    slg_end,
    lora1="None",
    lora2="None",
    lora3="None",
    lora4="None",
    lora1_multiplier=1.0,
    lora2_multiplier=1.0,
    lora3_multiplier=1.0,
    lora4_multiplier=1.0,
    enable_cfg_skip=False,
    cfg_skip_mode="none",
    cfg_apply_ratio=0.7,
    control_video=None,
    control_strength=1.0,
    control_start=0.0,
    control_end=1.0,
    enable_preview: bool = False,
    preview_steps: int = 5
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """Generate video with WanX model (supports both i2v, t2v and Fun-Control)"""
    global stop_event

    current_previews = []
    yield [], current_previews, "Preparing...", "" # Yield empty previews
    
    # Fix 1: Ensure lora_folder is a string
    lora_folder = str(lora_folder) if lora_folder else "lora"
    
    # Debug prints
    print(f"DEBUG - LoRA params: {lora1}, {lora2}, {lora3}, {lora4}")
    print(f"DEBUG - LoRA multipliers: {lora1_multiplier}, {lora2_multiplier}, {lora3_multiplier}, {lora4_multiplier}")
    print(f"DEBUG - LoRA folder: {lora_folder}")
    
    # Convert values safely to float or None
    try:
        slg_start_float = float(slg_start) if slg_start is not None and str(slg_start).lower() != "none" else None
    except (ValueError, TypeError):
        slg_start_float = None
        print(f"Warning: Could not convert slg_start '{slg_start}' to float")
    
    try:
        slg_end_float = float(slg_end) if slg_end is not None and str(slg_end).lower() != "none" else None
    except (ValueError, TypeError):
        slg_end_float = None
        print(f"Warning: Could not convert slg_end '{slg_end}' to float")
    
    print(f"slg_start_float: {slg_start_float}, slg_end_float: {slg_end_float}")
    
    if stop_event.is_set():
        yield [], [], "", "" # Yield empty previews
        return

    run_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
    unique_preview_suffix = f"wanx_{run_id}" # Add prefix for clarity
    preview_base_dir = os.path.join(save_path, "previews")
    preview_mp4_path = os.path.join(preview_base_dir, f"latent_preview_{unique_preview_suffix}.mp4")
    preview_png_path = os.path.join(preview_base_dir, f"latent_preview_{unique_preview_suffix}.png")

    # Check if this is a Fun-Control task
    is_fun_control = "-FC" in task and control_video is not None
    if is_fun_control:
        print(f"DEBUG - Using Fun-Control mode with control video: {control_video}")
        # Verify control video is provided
        if not control_video:
            yield [], "Error: No control video provided", "Fun-Control requires a control video"
            return
        
        # Verify needed files exist
        for path_name, path in [
            ("DIT", dit_path),
            ("VAE", vae_path),
            ("T5", t5_path),
            ("CLIP", clip_path)
        ]:
            if not os.path.exists(path):
                yield [], f"Error: {path_name} model not found", f"Model file doesn't exist: {path}"
                return
    
    # Get current seed or use provided seed
    current_seed = seed
    if seed == -1:
        current_seed = random.randint(0, 2**32 - 1)
        
    # Check if we need input image (required for i2v, not for t2v)
    if "i2v" in task and not input_image:
        yield [], "Error: No input image provided", "Please provide an input image for image-to-video generation"
        return
        
    # Check for Fun-Control requirements
    if is_fun_control and not control_video:
        yield [], "Error: No control video provided", "Please provide a control video for Fun-Control generation"
        return

    # Prepare environment
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    
    clear_cuda_cache()

    # Fix 2: Create command array with all string values
    command = [
        sys.executable,
        "wan_generate_video.py",
        "--task", str(task),
        "--prompt", str(prompt),
        "--video_size", str(height), str(width),
        "--video_length", str(video_length),
        "--fps", str(fps),
        "--infer_steps", str(infer_steps),
        "--save_path", str(save_path),
        "--seed", str(current_seed),
        "--flow_shift", str(flow_shift),
        "--guidance_scale", str(guidance_scale),
        "--output_type", str(output_type),
        "--attn_mode", str(attn_mode),
        "--blocks_to_swap", str(block_swap),
        "--dit", str(dit_path),
        "--vae", str(vae_path),
        "--t5", str(t5_path),
        "--sample_solver", str(sample_solver)
    ]
    
    if enable_preview and preview_steps > 0:
        command.extend(["--preview", str(preview_steps)])
        # --- ADDED: Pass the unique suffix ---
        command.extend(["--preview_suffix", unique_preview_suffix])
        # --- End Pass Suffix ---
        print(f"DEBUG - Enabling preview every {preview_steps} steps with suffix {unique_preview_suffix}.")

    if enable_cfg_skip and cfg_skip_mode != "none":
        command.extend([
            "--cfg_skip_mode", str(cfg_skip_mode),
            "--cfg_apply_ratio", str(cfg_apply_ratio)
        ])
        
    if wanx_input_end and wanx_input_end != "none" and os.path.exists(str(wanx_input_end)):
        command.extend(["--end_image_path", str(wanx_input_end)])
        command.extend(["--trim_tail_frames", "3"])
        
    # Handle Fun-Control (control video path)
    if is_fun_control and control_video:
        command.extend(["--control_path", str(control_video)])
        command.extend(["--control_weight", str(control_strength)])
        command.extend(["--control_start", str(control_start)])
        command.extend(["--control_end", str(control_end)])

    # Handle SLG parameters
    if slg_layers and str(slg_layers).strip() and str(slg_layers).lower() != "none":
        try:
            # Make sure slg_layers is parsed as a list of integers
            slg_list = []
            for layer in str(slg_layers).split(","):
                layer = layer.strip()
                if layer.isdigit():  # Only add if it's a valid integer
                    slg_list.append(int(layer))
            if slg_list:  # Only add if we have valid layers
                command.extend(["--slg_layers", ",".join(map(str, slg_list))])

                # Only add slg_start and slg_end if we have valid slg_layers
                try:
                    if slg_start_float is not None and slg_start_float >= 0:
                        command.extend(["--slg_start", str(slg_start_float)])
                    if slg_end_float is not None and slg_end_float <= 1.0:
                        command.extend(["--slg_end", str(slg_end_float)])
                except ValueError as e:
                    print(f"Invalid SLG timing values: {str(e)}")
        except ValueError as e:
            print(f"Invalid SLG layers format: {slg_layers} - {str(e)}")

    
    # Add image path only for i2v task and if input image is provided
    if "i2v" in task and input_image:
        command.extend(["--image_path", str(input_image)])
        command.extend(["--clip", str(clip_path)])  # CLIP is needed for i2v and Fun-Control
    
    # Add video path for v2v task
    if "v2v" in task and input_image:
        command.extend(["--video_path", str(input_image)])
        # Add strength parameter for video-to-video
        if isinstance(guidance_scale, (int, float)) and guidance_scale > 0:
            command.extend(["--strength", str(guidance_scale)])
    
    if negative_prompt:
        command.extend(["--negative_prompt", str(negative_prompt)])
    
    # Add boolean flags correctly
    if fp8:
        command.append("--fp8")
    
    if fp8_scaled:
        command.append("--fp8_scaled")
    
    if fp8_t5:
        command.append("--fp8_t5")
        
    if exclude_single_blocks:
        command.append("--exclude_single_blocks")
    
    # Handle LoRA weights and multipliers
    lora_weights = [lora1, lora2, lora3, lora4]
    lora_multipliers = [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier]
    
    valid_loras = []
    for weight, mult in zip(lora_weights, lora_multipliers):
        # Skip None, empty, or "None" values
        if weight is None or not str(weight) or str(weight).lower() == "none":
            continue
        
        # Ensure weight is a string
        weight_str = str(weight)

        # Construct full path and verify file exists
        full_path = os.path.join(lora_folder, weight_str)
        if not os.path.exists(full_path):
            print(f"LoRA file not found: {full_path}")
            continue

        # Add valid LoRA to the list
        valid_loras.append((full_path, mult))

    # Only add LoRA parameters if we have valid LoRAs
    if valid_loras:
        weights = [w for w, _ in valid_loras]
        multipliers = [str(m) for _, m in valid_loras]
        command.extend(["--lora_weight"] + weights)
        command.extend(["--lora_multiplier"] + multipliers)
    
    # Make sure every item in command is a string
    command = [str(item) for item in command]
    
    print(f"Running: {' '.join(command)}")

    p = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )

    videos = []
    processed_preview_files = set() # Keep track of previews already yielded - REMAINS THE SAME IN UI FUNCTION
    # --- Reset preview state for this run ---
    current_preview_yield_path = None
    last_preview_mtime = 0
    
    current_phase = "Preparing" # Add phase tracking like FramePack
    while True:
        if stop_event.is_set():
            try:
                p.terminate()
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            except Exception as e:
                print(f"Error terminating subprocess: {e}")
            yield [], [], "Generation stopped by user.", "" # Yield empty previews
            return

        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                break
            time.sleep(0.01); continue

        line = line.strip()
        if not line: continue
        print(f"WANX SUBPROCESS: {line}") # Log subprocess output

        # --- Adopt FramePack's Parsing Logic ---
        status_text = f"Processing (seed: {current_seed})" # Default status
        progress_text_update = line # Default progress

        # Check for TQDM progress using regex
        tqdm_match = re.search(r'(\d+)\%\|.+\| (\d+)/(\d+) \[(\d{2}:\d{2})<(\d{2}:\d{2})', line)

        if tqdm_match:
             percentage = int(tqdm_match.group(1))
             current_step = int(tqdm_match.group(2))
             total_steps = int(tqdm_match.group(3))
             time_elapsed = tqdm_match.group(4)
             time_remaining = tqdm_match.group(5)

             current_phase = f"Denoising Step {current_step}/{total_steps}" # Update phase

             # Format progress text like FramePack for JS compatibility
             progress_text_update = f"Step {current_step}/{total_steps} ({percentage}%) | Elapsed: {time_elapsed}, Remaining: {time_remaining}"
             status_text = f"Generating (seed: {current_seed}) - {current_phase}"

        elif "ERROR" in line.upper() or "TRACEBACK" in line.upper():
             status_text = f"Error (seed: {current_seed})"
             progress_text_update = line # Show error line
             current_phase = "Error"

        # Add more phases if needed (e.g., "Decoding", "Saving") by checking logs
        elif "Decoding video..." in line: # Placeholder check
             current_phase = "Decoding Video"
             status_text = f"Generating (seed: {current_seed}) - {current_phase}"
             progress_text_update = "Decoding video..."

        elif "Video saved to:" in line: # Placeholder check
             current_phase = "Saved"
             status_text = f"Completed (seed: {current_seed})"
             progress_text_update = line # Show the save line
        # Add any other status parsing if needed
        preview_updated = False
        current_mtime = 0
        found_preview_path = None

        if enable_preview:
            # --- MODIFIED: Check unique paths ---
            if os.path.exists(preview_mp4_path):
                current_mtime = os.path.getmtime(preview_mp4_path)
                found_preview_path = preview_mp4_path
            elif os.path.exists(preview_png_path):
                current_mtime = os.path.getmtime(preview_png_path)
                found_preview_path = preview_png_path
            # --- End Modified Check ---

            if found_preview_path and current_mtime > last_preview_mtime:
                print(f"DEBUG: Preview file updated: {found_preview_path} (mtime: {current_mtime})")
                # Yield the clean path (already unique)
                current_preview_yield_path = found_preview_path # No cache buster needed
                last_preview_mtime = current_mtime
                preview_updated = True
        # --- End Preview Check ---

        # --- YIELD ---
        # Yield progress and potentially updated unique preview path
            preview_list_for_yield = [current_preview_yield_path] if current_preview_yield_path else []
            # Yield progress and potentially updated unique preview path list
            yield videos.copy(), preview_list_for_yield, status_text, progress_text_update

    p.stdout.close()
    rc = p.wait() 

    clear_cuda_cache()
    time.sleep(0.5)

    # --- Collect final generated video ---
    generated_video_path = None
    if rc == 0: # Only look for video if process succeeded
        save_path_abs = os.path.abspath(save_path)
        if os.path.exists(save_path_abs):
            # Find the most recent mp4 containing the seed
            all_mp4_files = glob.glob(os.path.join(save_path_abs, f"*_{current_seed}*.mp4"))
            # Exclude files in the 'previews' subdirectory
            all_mp4_files = [f for f in all_mp4_files if "previews" not in os.path.dirname(f)]

            if all_mp4_files:
                # Find the *absolute* most recent one, as multiple might match seed in edge cases
                generated_video_path = max(all_mp4_files, key=os.path.getmtime)
                print(f"Found newly generated video: {generated_video_path}")

                # Add metadata (assuming add_metadata_to_video exists and works)
                parameters = {
                    "prompt": prompt, "negative_prompt": negative_prompt,
                    "input_image": input_image if "i2v" in task else None,
                    "width": width, "height": height, "video_length": video_length, "fps": fps,
                    "infer_steps": infer_steps, "flow_shift": flow_shift, "guidance_scale": guidance_scale,
                    "seed": current_seed, "task": task, "dit_path": dit_path,
                    "vae_path": vae_path, "t5_path": t5_path, "clip_path": clip_path if "i2v" in task or is_fun_control else None,
                    "save_path": save_path, "output_type": output_type, "sample_solver": sample_solver,
                    "exclude_single_blocks": exclude_single_blocks, "attn_mode": attn_mode,
                    "block_swap": block_swap, "fp8": fp8, "fp8_scaled": fp8_scaled, "fp8_t5": fp8_t5,
                    "lora_weights": [lora1, lora2, lora3, lora4],
                    "lora_multipliers": [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier],
                    "slg_layers": slg_layers, "slg_start": slg_start, "slg_end": slg_end,
                    "enable_cfg_skip": enable_cfg_skip, "cfg_skip_mode": cfg_skip_mode, "cfg_apply_ratio": cfg_apply_ratio,
                    "control_video": control_video if is_fun_control else None,
                    "control_strength": control_strength if is_fun_control else None,
                    "control_start": control_start if is_fun_control else None,
                    "control_end": control_end if is_fun_control else None,
                }
                try:
                     add_metadata_to_video(generated_video_path, parameters)
                except NameError:
                     print("Warning: add_metadata_to_video function not found. Skipping metadata.")
                except Exception as meta_err:
                     print(f"Warning: Failed to add metadata: {meta_err}")

                # Append to the final video list
                videos.append((str(generated_video_path), f"Seed: {current_seed}"))
            else:
                 print(f"Subprocess finished successfully (rc=0), but could not find generated video for seed {current_seed} in {save_path_abs}")

# --- Final Yield ---
    final_status = f"Completed (seed: {current_seed})" if rc == 0 and generated_video_path else f"Failed (seed: {current_seed}, rc={rc})"
    final_progress = f"Video saved: {os.path.basename(generated_video_path)}" if rc == 0 and generated_video_path else f"Subprocess failed with exit code {rc}"

    # Check for the preview file one last time for the final update (using unique path)
    # --- MODIFIED Final Preview Check and List Creation ---
    final_preview_path = None
    # --- Use the UNIQUE paths defined earlier in the function ---
    if os.path.exists(preview_mp4_path):
        final_preview_path = os.path.abspath(preview_mp4_path)
    elif os.path.exists(preview_png_path):
        final_preview_path = os.path.abspath(preview_png_path)
    # --- End path checking ---

    final_preview_list_for_yield = [final_preview_path] if final_preview_path else []
    # --- End Modified ---

    yield videos, final_preview_list_for_yield, final_status, final_progress

def send_wanx_to_v2v(
    gallery: list,
    prompt: str,
    selected_index: int,
    width: int,
    height: int,
    video_length: int,
    fps: int,
    infer_steps: int,
    seed: int,
    flow_shift: float,
    guidance_scale: float,
    negative_prompt: str
) -> Tuple:
    """Send the selected WanX video to Video2Video tab"""
    if gallery is None or not gallery:
        return (None, "", width, height, video_length, fps, infer_steps, seed, 
                flow_shift, guidance_scale, negative_prompt)
    
    # If no selection made but we have videos, use the first one
    if selected_index is None and len(gallery) > 0:
        selected_index = 0
        
    if selected_index is None or selected_index >= len(gallery):
        return (None, "", width, height, video_length, fps, infer_steps, seed, 
                flow_shift, guidance_scale, negative_prompt)

    selected_item = gallery[selected_index]
    
    # Handle different gallery item formats
    if isinstance(selected_item, tuple):
        video_path = selected_item[0]
    elif isinstance(selected_item, dict):
        video_path = selected_item.get("name", selected_item.get("data", None))
    else:
        video_path = selected_item

    # Clean up path for Video component
    if isinstance(video_path, tuple):
        video_path = video_path[0]
        
    # Make sure it's a string
    video_path = str(video_path)

    return (video_path, prompt, width, height, video_length, fps, infer_steps, seed, 
            flow_shift, guidance_scale, negative_prompt)

def wanx_generate_video_batch(
    prompt,
    negative_prompt,
    width,
    height,
    video_length,
    fps,
    infer_steps,
    flow_shift,
    guidance_scale,
    seed,
    task,
    dit_path,
    vae_path,
    t5_path,
    clip_path,
    save_path,
    output_type,
    sample_solver,
    exclude_single_blocks,
    attn_mode,
    block_swap,
    fp8,
    fp8_scaled,
    fp8_t5,
    lora_folder,
    slg_layers: int,
    slg_start: Optional[str],
    slg_end: Optional[str],
    lora1="None",
    lora2="None",
    lora3="None",
    lora4="None",
    lora1_multiplier=1.0,
    lora2_multiplier=1.0,
    lora3_multiplier=1.0,
    lora4_multiplier=1.0,
    batch_size=1,
    input_image=None  # Make input_image optional and place it at the end
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """Generate videos with WanX with support for batches"""
    slg_start = None if slg_start == 'None' or slg_start is None else slg_start
    slg_end = None if slg_end == 'None' or slg_end is None else slg_end

    # Now safely convert to float if not None
    slg_start_float = float(slg_start) if slg_start is not None and isinstance(slg_start, (str, int, float)) else None
    slg_end_float = float(slg_end) if slg_end is not None and isinstance(slg_end, (str, int, float)) else None
    print(f"slg_start_float: {slg_start_float}, slg_end_float: {slg_end_float}")
    global stop_event
    stop_event.clear()
    
    all_videos = []
    progress_text = "Starting generation..."
    yield [], "Preparing...", progress_text
    
    # Process each item in the batch
    for i in range(batch_size):
        if stop_event.is_set():
            yield all_videos, "Generation stopped by user", ""
            return
            
        # Calculate seed for this batch item
        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif batch_size > 1:
            current_seed = seed + i
            
        batch_text = f"Generating video {i + 1} of {batch_size}"
        yield all_videos.copy(), batch_text, progress_text
        
        # Generate a single video using the existing function
        for videos, status, progress in wanx_generate_video(
            prompt, 
            negative_prompt, 
            input_image, 
            width, 
            height, 
            video_length, 
            fps, 
            infer_steps, 
            flow_shift, 
            guidance_scale, 
            current_seed,
            task, 
            dit_path, 
            vae_path, 
            t5_path, 
            clip_path, 
            save_path, 
            output_type, 
            sample_solver, 
            exclude_single_blocks,
            attn_mode, 
            block_swap, 
            fp8, 
            fp8_scaled, 
            fp8_t5,
            lora_folder,
            slg_layers,
            slg_start,
            slg_end,
            lora1,
            lora2,
            lora3,
            lora4,
            lora1_multiplier,
            lora2_multiplier,
            lora3_multiplier,
            lora4_multiplier
        ):
            if videos:
                all_videos.extend(videos)
            yield all_videos.copy(), f"Batch {i+1}/{batch_size}: {status}", progress
    
    yield all_videos, "Batch complete", ""

def update_wanx_t2v_dimensions(size):
    """Update width and height based on selected size"""
    width, height = map(int, size.split('*'))
    return gr.update(value=width), gr.update(value=height)

def handle_wanx_t2v_gallery_select(evt: gr.SelectData) -> int:
    """Track selected index when gallery item is clicked"""
    return evt.index

def send_wanx_t2v_to_v2v(
    gallery, prompt, selected_index, width, height, video_length,
    fps, infer_steps, seed, flow_shift, guidance_scale, negative_prompt
) -> Tuple:
    """Send the selected WanX T2V video to Video2Video tab"""
    if not gallery or selected_index is None or selected_index >= len(gallery):
        return (None, "", width, height, video_length, fps, infer_steps, seed, 
                flow_shift, guidance_scale, negative_prompt)

    selected_item = gallery[selected_index]

    if isinstance(selected_item, dict):
        video_path = selected_item.get("name", selected_item.get("data", None))
    elif isinstance(selected_item, (tuple, list)):
        video_path = selected_item[0]
    else:
        video_path = selected_item

    if isinstance(video_path, tuple):
        video_path = video_path[0]

    return (str(video_path), prompt, width, height, video_length, fps, infer_steps, seed, 
            flow_shift, guidance_scale, negative_prompt)

def prepare_for_batch_extension(input_img, base_video, batch_size):
    """Prepare inputs for batch video extension"""
    if input_img is None:
        return None, None, batch_size, "No input image found", ""
        
    if base_video is None:
        return input_img, None, batch_size, "No base video selected for extension", ""
        
    return input_img, base_video, batch_size, "Preparing batch extension...", f"Will create {batch_size} variations of extended video"

def concat_batch_videos(base_video_path, generated_videos, save_path, original_video_path=None):
    """Concatenate multiple generated videos with the base video"""
    if not base_video_path:
        return [], "No base video provided"
            
    if not generated_videos or len(generated_videos) == 0:
        return [], "No new videos generated"
    
    # Create output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Track all extended videos
    extended_videos = []
    
    # For each generated video, create an extended version
    for i, video_item in enumerate(generated_videos):
        try:
            # Extract video path from gallery item
            if isinstance(video_item, tuple):
                new_video_path = video_item[0]
                seed_info = video_item[1] if len(video_item) > 1 else ""
            elif isinstance(video_item, dict):
                new_video_path = video_item.get("name", video_item.get("data", None))
                seed_info = ""
            else:
                new_video_path = video_item
                seed_info = ""
                
            if not new_video_path or not os.path.exists(new_video_path):
                print(f"Skipping missing video: {new_video_path}")
                continue
                
            # Create unique output filename
            timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
            # Extract seed from seed_info if available
            seed_match = re.search(r"Seed: (\d+)", seed_info)
            seed_part = f"_seed{seed_match.group(1)}" if seed_match else f"_{i}"
            
            output_filename = f"extended_{timestamp}{seed_part}_{Path(base_video_path).stem}.mp4"
            output_path = os.path.join(save_path, output_filename)
            
            # Create a temporary file list for ffmpeg
            list_file = os.path.join(save_path, f"temp_list_{i}.txt")
            with open(list_file, "w") as f:
                f.write(f"file '{os.path.abspath(base_video_path)}'\n")
                f.write(f"file '{os.path.abspath(new_video_path)}'\n")
            
            # Run ffmpeg concatenation
            command = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c", "copy",
                output_path
            ]
            
            subprocess.run(command, check=True, capture_output=True)
            
            # Clean up temporary file
            if os.path.exists(list_file):
                os.remove(list_file)
                
            # Add to extended videos list if successful
            if os.path.exists(output_path):
                seed_display = f"Extended {seed_info}" if seed_info else f"Extended video #{i+1}"
                extended_videos.append((output_path, seed_display))
            
        except Exception as e:
            print(f"Error processing video {i}: {str(e)}")
    
    if not extended_videos:
        return [], "Failed to create any extended videos"
        
    return extended_videos, f"Successfully created {len(extended_videos)} extended videos"

def wanx_extend_single_video(
    prompt, negative_prompt, input_image, base_video_path,
    width, height, video_length, fps, infer_steps, 
    flow_shift, guidance_scale, seed, 
    task, dit_path, vae_path, t5_path, clip_path,
    save_path, output_type, sample_solver, exclude_single_blocks,
    attn_mode, block_swap, fp8, fp8_scaled, fp8_t5, lora_folder,
    slg_layers="", slg_start=0.0, slg_end=1.0,
    lora1="None", lora2="None", lora3="None", lora4="None",
    lora1_multiplier=1.0, lora2_multiplier=1.0, lora3_multiplier=1.0, lora4_multiplier=1.0
):
    """Generate a single video and concatenate with base video"""
    # First, generate the video with proper parameter handling
    all_videos = []
    
    # Sanitize lora parameters
    lora_weights = [str(lora1) if lora1 is not None else "None", 
                   str(lora2) if lora2 is not None else "None", 
                   str(lora3) if lora3 is not None else "None", 
                   str(lora4) if lora4 is not None else "None"]
    
    # Convert multipliers to float
    try:
        lora_multipliers = [float(lora1_multiplier), float(lora2_multiplier), 
                           float(lora3_multiplier), float(lora4_multiplier)]
    except (ValueError, TypeError):
        # Fallback to defaults if conversion fails
        lora_multipliers = [1.0, 1.0, 1.0, 1.0]
    
    # Debug print
    print(f"Sanitized LoRA weights: {lora_weights}")
    print(f"Sanitized LoRA multipliers: {lora_multipliers}")
    
    # Generate video
    for videos, status, progress in wanx_generate_video(
        prompt, negative_prompt, input_image, width, height, 
        video_length, fps, infer_steps, flow_shift, guidance_scale, 
        seed, task, dit_path, vae_path, t5_path, clip_path, 
        save_path, output_type, sample_solver, exclude_single_blocks,
        attn_mode, block_swap, fp8, fp8_scaled, fp8_t5, lora_folder,
        slg_layers, slg_start, slg_end,
        lora_weights[0], lora_weights[1], lora_weights[2], lora_weights[3],
        lora_multipliers[0], lora_multipliers[1], lora_multipliers[2], lora_multipliers[3],
        enable_cfg_skip=False,
        cfg_skip_mode="none",
        cfg_apply_ratio=0.7
    ):
        
        # Keep track of generated videos
        if videos:
            all_videos = videos
        
        # Forward progress updates
        yield all_videos, status, progress
    
    # Now concatenate with base video if we have something
    if all_videos and base_video_path and os.path.exists(base_video_path):
        try:
            print(f"Extending base video: {base_video_path}")
            
            # Create unique output filename
            timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
            output_filename = f"extended_{timestamp}_seed{seed}_{Path(base_video_path).stem}.mp4"
            output_path = os.path.join(save_path, output_filename)
            
            # Extract the path from the gallery item
            new_video_path = all_videos[0][0] if isinstance(all_videos[0], tuple) else all_videos[0]
            
            # Create a temporary file list for ffmpeg
            list_file = os.path.join(save_path, f"temp_list_{seed}.txt")
            with open(list_file, "w") as f:
                f.write(f"file '{os.path.abspath(base_video_path)}'\n")
                f.write(f"file '{os.path.abspath(new_video_path)}'\n")
            
            print(f"Concatenating: {base_video_path} + {new_video_path}")
            
            # Run ffmpeg concatenation
            command = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c", "copy",
                "-y",
                output_path
            ]
            
            subprocess.run(command, check=True, capture_output=True)
            
            # Clean up temporary file
            if os.path.exists(list_file):
                os.remove(list_file)
                
            # Return the extended video if successful
            if os.path.exists(output_path):
                extended_video = [(output_path, f"Extended (Seed: {seed})")]
                print(f"Successfully created extended video: {output_path}")
                yield extended_video, "Extended video created successfully", ""
                return
            else:
                print(f"Failed to create extended video at {output_path}")
        except Exception as e:
            print(f"Error creating extended video: {str(e)}")
    
    # If we got here, something went wrong with the concatenation
    yield all_videos, "Generated video (extension failed)", ""

def process_batch_extension(
    prompt, negative_prompt, input_image, base_video,
    width, height, video_length, fps, infer_steps,
    flow_shift, guidance_scale, seed, batch_size,
    task, dit_folder, dit_path, vae_path, t5_path, clip_path, # <<< Added dit_folder
    save_path, output_type, sample_solver, exclude_single_blocks,
    attn_mode, block_swap, fp8, fp8_scaled, fp8_t5, lora_folder,
    slg_layers, slg_start, slg_end,
    lora1="None", lora2="None", lora3="None", lora4="None",
    lora1_multiplier=1.0, lora2_multiplier=1.0, lora3_multiplier=1.0, lora4_multiplier=1.0
):
    """Process a batch of video extensions one at a time"""
    global stop_event
    stop_event.clear()

    all_extended_videos = [] # Store successfully extended videos
    progress_text = "Starting video extension batch..."
    yield [], progress_text, "" # Initial yield

    try:
        # Ensure batch_size is treated as an integer
        batch_size = int(batch_size)
    except (ValueError, TypeError):
        batch_size = 1
        print("Warning: Invalid batch_size, defaulting to 1.")

    # Ensure base_video exists
    if not base_video or not os.path.exists(base_video):
        yield [], "Error: Base video not found", f"Cannot find video at {base_video}"
        return

    # Process each batch item independently
    for i in range(batch_size):
        if stop_event.is_set():
            yield all_extended_videos, "Extension stopped by user", ""
            return

        # Calculate seed for this batch item
        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif batch_size > 1:
            current_seed = seed + i

        batch_text = f"Processing extension {i+1}/{batch_size} (seed: {current_seed})"
        yield all_extended_videos, batch_text, progress_text # Update progress

        # Use the direct wrapper with correct parameter order, including dit_folder
        generation_iterator = wanx_extend_video_wrapper(
            prompt=prompt, negative_prompt=negative_prompt, input_image=input_image, base_video_path=base_video,
            width=width, height=height, video_length=video_length, fps=fps, infer_steps=infer_steps,
            flow_shift=flow_shift, guidance_scale=guidance_scale, seed=current_seed,
            task=task,
            dit_folder=dit_folder, # <<< Pass the folder path
            dit_path=dit_path,     # <<< Pass the model filename
            vae_path=vae_path,
            t5_path=t5_path,
            clip_path=clip_path,
            save_path=save_path, output_type=output_type, sample_solver=sample_solver,
            exclude_single_blocks=exclude_single_blocks, attn_mode=attn_mode, block_swap=block_swap,
            fp8=fp8, fp8_scaled=fp8_scaled, fp8_t5=fp8_t5, lora_folder=lora_folder,
            slg_layers=slg_layers, slg_start=slg_start, slg_end=slg_end,
            lora1=lora1, lora2=lora2, lora3=lora3, lora4=lora4,
            lora1_multiplier=lora1_multiplier, lora2_multiplier=lora2_multiplier,
            lora3_multiplier=lora3_multiplier, lora4_multiplier=lora4_multiplier
        )

        # Iterate through the generator for this single extension
        final_videos_for_item = []
        final_status_for_item = "Unknown status"
        final_progress_for_item = ""
        try:
            for videos, status, progress in generation_iterator:
                # Forward progress information immediately
                yield all_extended_videos, f"Batch {i+1}/{batch_size}: {status}", progress

                # Store the latest state for this item
                final_videos_for_item = videos
                final_status_for_item = status
                final_progress_for_item = progress

            # After the loop for one item finishes, check the result
            if final_videos_for_item:
                 # Check if the video is actually an extended one
                is_extended = any("Extended" in (v[1] if isinstance(v, tuple) else "") for v in final_videos_for_item)
                if is_extended:
                    all_extended_videos.extend(final_videos_for_item)
                    print(f"Added extended video to collection (total: {len(all_extended_videos)})")
                else:
                    # It was just the generated segment, maybe log this?
                    print(f"Video segment generated for batch {i+1} but extension failed or wasn't performed.")
            else:
                print(f"No video returned for batch item {i+1}.")


        except Exception as e:
            print(f"Error during single extension processing (batch {i+1}): {e}")
            yield all_extended_videos, f"Error in batch {i+1}: {e}", ""


        # Clean CUDA cache between generations
        clear_cuda_cache()
        time.sleep(0.5)

    # Final yield after the loop
    yield all_extended_videos, "Batch extension complete", ""

def handle_extend_generation(base_video_path: str, new_videos: list, save_path: str, current_gallery: list) -> tuple:
    """Combine generated video with base video and update gallery"""
    if not base_video_path:
        return current_gallery, "Extend failed: No base video provided"
        
    if not new_videos:
        return current_gallery, "Extend failed: No new video generated"
    
    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)
    
    # Get the first video from new_videos (gallery item)
    new_video_path = new_videos[0][0] if isinstance(new_videos[0], tuple) else new_videos[0]
    
    # Create a unique output filename
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    output_filename = f"extended_{timestamp}_{Path(base_video_path).stem}.mp4"
    output_path = str(Path(save_path) / output_filename)
    
    try:
        # Concatenate the videos using ffmpeg
        (
            ffmpeg
            .input(base_video_path)
            .concat(
                ffmpeg.input(new_video_path)
            )
            .output(output_path)
            .run(overwrite_output=True, quiet=True)
        )
        
        # Create a new gallery entry with the combined video
        updated_gallery = [(output_path, f"Extended video: {Path(output_path).stem}")]
        
        return updated_gallery, f"Successfully extended video to {Path(output_path).name}"
    except Exception as e:
        print(f"Error extending video: {str(e)}")
        return current_gallery, f"Failed to extend video: {str(e)}"

# UI setup
with gr.Blocks(
    theme=themes.Default(
        primary_hue=colors.Color(
            name="custom",
            c50="#E6F0FF",
            c100="#CCE0FF",
            c200="#99C1FF",
            c300="#66A3FF",
            c400="#3384FF",
            c500="#0060df",  # This is your main color
            c600="#0052C2",
            c700="#003D91",
            c800="#002961",
            c900="#001430",
            c950="#000A18"
        )
    ),
    css="""
    .gallery-item:first-child { border: 2px solid #4CAF50 !important; }
    .gallery-item:first-child:hover { border-color: #45a049 !important; }
    .green-btn {
        background: linear-gradient(to bottom right, #2ecc71, #27ae60) !important;
        color: white !important;
        border: none !important;
    }
    .green-btn:hover {
        background: linear-gradient(to bottom right, #27ae60, #219651) !important;
    }
    .refresh-btn {
        max-width: 40px !important;
        min-width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .light-blue-btn {
        background: linear-gradient(to bottom right, #AEC6CF, #9AB8C4) !important; /* Light blue gradient */
        color: #333 !important; /* Darker text for readability */
        border: 1px solid #9AB8C4 !important; /* Subtle border */
    }
    .light-blue-btn:hover {
        background: linear-gradient(to bottom right, #9AB8C4, #8AA9B5) !important; /* Slightly darker on hover */
        border-color: #8AA9B5 !important;
    }    
    """, 

) as demo:
    # Add state for tracking selected video indices in both tabs
    selected_index = gr.State(value=None)  # For Text to Video
    v2v_selected_index = gr.State(value=None)  # For Video to Video
    params_state = gr.State() #New addition
    i2v_selected_index = gr.State(value=None) 
    skyreels_selected_index = gr.State(value=None)
    wanx_i2v_selected_index = gr.State(value=None)
    extended_videos = gr.State(value=[])
    wanx_base_video = gr.State(value=None)
    wanx_sharpest_frame_number = gr.State(value=None)  
    wanx_sharpest_frame_path = gr.State(value=None)   
    wanx_trimmed_video_path = gr.State(value=None) 
    wanx_v2v_selected_index = gr.State(value=None)
    wanx_t2v_selected_index = gr.State(value=None)
    wan22_selected_index = gr.State(value=None)
    framepack_selected_index = gr.State(value=None)
    framepack_original_dims = gr.State(value="")
    fpe_selected_index = gr.State(value=None)
    phantom_selected_index = gr.State(value=None)
    multitalk_annotations_state = gr.State(value=None)
    demo.load(None, None, None, js=r"""
        () => {
            document.title = 'H1111';

            function updateTitle(text) {
                if (text && text.trim()) {
                    // This single regex handles both raw TQDM and custom formatted progress strings.
                    // It looks for a percentage, then finds a time string (HH:MM:SS) after it.
                    // Group 1: Percentage from custom format like "(XX%)"
                    // Group 2: Time from custom format like "ETA: HH:MM:SS"
                    // Group 3: Percentage from raw TQDM format like "XX%|"
                    // Group 4: Time from raw TQDM format like "<HH:MM:SS"
                    const pattern = /(?:.*?\((\d+)%\).*?(?:ETA|Remaining):\s*([\d:]+))|(?:(\d+)%\|.*\[.*<([\d:?]+))/;
                    const match = text.match(pattern);

                    if (match) {
                        const percentage = match[1] || match[3];
                        const time = match[2] || match[4];
                        if (percentage && time) {
                             document.title = `[${percentage}% ETA: ${time}] - H1111`;
                        }
                    }
                }
            }

            setTimeout(() => {
                const progressElements = document.querySelectorAll('textarea.scroll-hide');
                progressElements.forEach(element => {
                    if (element) {
                        new MutationObserver(() => {
                            updateTitle(element.value);
                        }).observe(element, {
                            attributes: true,
                            childList: true,
                            characterData: true,
                            subtree: true
                        });
                    }
                });
            }, 1000);
        }
        """)
        
    with gr.Tabs() as tabs:

        #FRAME PACK TAB
        with gr.Tab(id=10, label="FramePack") as framepack_tab:
            
            with gr.Row():
                with gr.Column(scale=4):
                    framepack_prompt = gr.Textbox(
                        scale=3, label="Prompt (Supports sections: index:prompt;;;index:prompt)",
                        value="cinematic video of a cat wizard casting a spell", lines=3,
                        info="Use '0:prompt;;;-1:prompt' or '0-2:prompt;;;3:prompt'. Index total sections -1 is last section."
                    )
                    framepack_negative_prompt = gr.Textbox(scale=3, label="Negative Prompt", value="", lines=3)
                with gr.Column(scale=1):
                    framepack_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    framepack_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)
                    framepack_is_f1 = gr.Checkbox(label="️ Use F1 Model", value=False,
                                                  info="Switches to the F1 model (different DiT path and logic).")                    
                with gr.Column(scale=2):
                    framepack_batch_progress = gr.Textbox(label="Status", interactive=False, value="")
                    framepack_progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text")
                    with gr.Row():
                        framepack_save_defaults_btn = gr.Button("Save Defaults")
                        framepack_load_defaults_btn = gr.Button("Load Defaults")
                        framepack_defaults_status = gr.Textbox(label="Defaults Status", interactive=False, visible=False)                    
            with gr.Row():
                framepack_generate_btn = gr.Button("Generate FramePack Video", elem_classes="green-btn")
                framepack_stop_btn = gr.Button("Stop Generation", variant="stop")

            # Main Content
            with gr.Row():
                # --- Left Column ---
                with gr.Column():
                    framepack_input_image = gr.Image(label="Input Image (Video Start)", type="filepath")
                    with gr.Row():
                        framepack_use_random_folder = gr.Checkbox(label="Use Random Images from Folder", value=False,
                                                                  info="If checked, 'Input Image (Video Start)' is hidden. Each batch item uses a random image from the folder.")
                        framepack_input_folder_path = gr.Textbox(
                            label="Image Folder Path", 
                            placeholder="Path to folder containing images for batch processing",
                            visible=False # Initially hidden
                        )
                    with gr.Row(visible=False) as framepack_folder_options_row: # Parent Row for folder options
                        framepack_validate_folder_btn = gr.Button("Validate Folder")
                        framepack_folder_status_text = gr.Textbox(
                            label="Folder Status", 
                            placeholder="Validation status will appear here",
                            interactive=False
                        )                    
                    with gr.Accordion("Optional End Frame Control (normal model only)", open=False):
                        framepack_input_end_frame = gr.Image(label="End Frame Image (Video End)", type="filepath", scale=1)
                        framepack_end_frame_influence = gr.Dropdown(
                            label="End Frame Influence Mode",
                            choices=["last", "half", "progressive", "bookend"],
                            value="last",
                            info="How the end frame affects generation (if provided)",
                            visible=False
                        )
                        framepack_end_frame_weight = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05, value=0.5, # Default changed from 0.3
                            label="End Frame Weight",
                            info="Influence strength of the end frame (if provided)",
                            visible=False
                        )

                    gr.Markdown("### Resolution Options (Choose One)")
                    framepack_target_resolution = gr.Number(
                        label="Option 1: Target Resolution (Uses Buckets)",
                        value=640, minimum=0, maximum=1280, step=32,
                        info="Target bucket size (e.g., 640 for 640x640). Uses input image aspect ratio. Final size divisible by 32.",
                        interactive=True
                    )
                    with gr.Accordion("Option 2: Explicit Resolution (Overrides Option 1)", open=False):
                         framepack_scale_slider = gr.Slider(
                             minimum=1, maximum=200, value=100, step=1, label="Scale % (UI Only)"
                         )
                         with gr.Row():
                             framepack_width = gr.Number(
                                 label="Width", value=None, minimum=0, step=32, 
                                 info="Must be divisible by 32.", interactive=True
                             )
                             framepack_calc_height_btn = gr.Button("→")
                             framepack_calc_width_btn = gr.Button("←")
                             framepack_height = gr.Number(
                                 label="Height", value=None, minimum=0, step=32,
                                 info="Must be divisible by 32.", interactive=True
                             )
                    framepack_total_second_length = gr.Slider(minimum=1.0, maximum=120.0, step=0.5, label="Total Video Length (seconds)", value=5.0)
                    framepack_video_sections = gr.Number(
                        label="Total Video Sections (Overrides seconds if > 0)",
                        value=None, step=1,
                        info="Specify exact number of sections. If set, 'Total Video Length (seconds)' is ignored by the backend."
                    )                    
                    framepack_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Output FPS", value=30)
                    with gr.Row():
                        framepack_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                        framepack_random_seed =gr.Button("🎲️")
                    framepack_steps = gr.Slider(minimum=1, maximum=100, step=1, label="Steps", value=25, interactive=True) # Moved here

                # --- Right Column ---
                with gr.Column():
                    framepack_output = gr.Gallery(
                        label="Generated Videos (Click to select)",
                        columns=[2], rows=[1],
                        object_fit="contain", height="auto", show_label=True,
                        elem_id="gallery_framepack", allow_preview=True, preview=True
                    )
                    with gr.Accordion("Latent Preview (During Generation)", open=True):
                        with gr.Row():
                            framepack_enable_preview = gr.Checkbox(label="Enable Latent Preview", value=True)
                            framepack_use_full_video_preview = gr.Checkbox(label="Use Full Video Previews (slower)", value=False)
                        with gr.Row():
                            framepack_preview_every_n_sections = gr.Slider(
                                minimum=1, maximum=50, step=1, value=1,
                                label="Preview Every N Sections",
                                info="Generates previews during the sampling loop."
                            )
                        framepack_preview_output = gr.Video( # Changed from Gallery to Video
                             label="Latest Preview", height=300,
                             interactive=False, # Not interactive for display
                             elem_id="framepack_preview_video"
                        )
                        framepack_skip_btn = gr.Button("Skip Batch Item", elem_classes="light-blue-btn")
                    with gr.Group():
                        with gr.Row():
                            framepack_refresh_lora_btn = gr.Button("🔄 LoRA", elem_classes="refresh-btn") # Specific LoRA refresh
                            framepack_lora_folder = gr.Textbox(label="LoRa Folder", value="lora", scale=4)
                        framepack_lora_weights = []
                        framepack_lora_multipliers = []
                        for i in range(4): # Assuming max 4 LoRAs like other tabs
                            with gr.Row():
                                framepack_lora_weights.append(gr.Dropdown(
                                    label=f"LoRA {i+1}", choices=get_lora_options("lora"),
                                    value="None", allow_custom_value=False, interactive=True, scale=2
                                ))
                                framepack_lora_multipliers.append(gr.Slider(
                                    label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1, interactive=True
                                ))
                    # Fixed Generation Parameters Section
                    with gr.Accordion("Generation Parameters", open=True):
                        with gr.Row():
                            framepack_distilled_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Distilled Guidance Scale (embedded_cfg_scale)", value=10.0, interactive=True)
                            framepack_guidance_scale = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, label="Guidance Scale (CFG)", value=1.0, interactive=True, info="Default 1.0 (no CFG), backend recommends not changing.")
                        with gr.Row():
                            framepack_guidance_rescale = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="CFG Rescale (rs)", value=0.0, interactive=True, info="Default 0.0, backend recommends not changing.")
                            framepack_latent_window_size = gr.Number(label="Latent Window Size", value=9, interactive=True, info="Default 9")
                            framepack_sample_solver = gr.Dropdown(label="Sample Solver", choices=["unipc", "dpm++", "vanilla"], value="unipc", interactive=True)

            with gr.Accordion("Advanced Section Control (Optional)", open=False):
                gr.Markdown(
                    "Define specific prompts and starting images for different sections of the video. "
                    "For the index you can input a range or a single index. A 5 second default video has 4 sections. The first section is 0 and the last is 3"
                )
                # --- Define section controls explicitly ---
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("**--- Control Slot 1 ---**")
                            with gr.Row():
                                 
                                framepack_sec_1 = gr.Textbox(label="Index/Range", value="0", placeholder="e.g., 0 or 0-1", interactive=True)
                            framepack_sec_prompt_1 = gr.Textbox(label="Prompt Override", lines=2, placeholder="Overrides base prompt for these sections")
                            framepack_sec_image_1 = gr.Image(label="Start Image Override", type="filepath", scale=1)
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("**--- Control Slot 2 ---**")
                            with gr.Row():
                                 
                                framepack_sec_2 = gr.Textbox(label="Index/Range", value="1", placeholder="e.g., 2 or 2-3", interactive=True)
                            framepack_sec_prompt_2 = gr.Textbox(label="Prompt Override", lines=2)
                            framepack_sec_image_2 = gr.Image(label="Start Image Override", type="filepath", scale=1)
                with gr.Row():
                    with gr.Column(scale=1):
                         with gr.Group():
                            gr.Markdown("**--- Control Slot 3 ---**")
                            with gr.Row():
                                
                                framepack_sec_3 = gr.Textbox(label="Index/Range", value="2", placeholder="e.g., 4 or 4-5", interactive=True)
                            framepack_sec_prompt_3 = gr.Textbox(label="Prompt Override", lines=2)
                            framepack_sec_image_3 = gr.Image(label="Start Image Override", type="filepath", scale=1)
                    with gr.Column(scale=1):
                         with gr.Group():
                            gr.Markdown("**--- Control Slot 4 ---**")
                            with gr.Row():
                                framepack_sec_4 = gr.Textbox(label="Index/Range", value="3", placeholder="e.g., 6 or 6-7", interactive=True)
                            framepack_sec_prompt_4 = gr.Textbox(label="Prompt Override", lines=2)
                            framepack_sec_image_4 = gr.Image(label="Start Image Override", type="filepath", scale=1)

                # Group section control components for easier passing to functions (remains the same)
                framepack_secs = [framepack_sec_1, framepack_sec_2, framepack_sec_3, framepack_sec_4]
                framepack_sec_prompts = [framepack_sec_prompt_1, framepack_sec_prompt_2, framepack_sec_prompt_3, framepack_sec_prompt_4]
                framepack_sec_images = [framepack_sec_image_1, framepack_sec_image_2, framepack_sec_image_3, framepack_sec_image_4]                            

            # Performance/Memory Accordion - Updated
            with gr.Accordion("Performance / Memory", open=True):
                with gr.Row():
                    framepack_fp8 = gr.Checkbox(label="Use FP8 DiT", value=False, info="Enable FP8 precision for the main Transformer model.")
                    framepack_fp8_llm = gr.Checkbox(label="Use FP8 LLM (Text Encoder 1)", value=False, info="Enable FP8 for the Llama text encoder.", visible=False)
                    framepack_fp8_scaled = gr.Checkbox(label="Use Scaled FP8 DiT", value=False, info="Requires FP8 DiT. Use scaled math (potential quality improvement).")
                    framepack_blocks_to_swap = gr.Slider(minimum=0, maximum=39, step=1, label="Blocks to Swap (to Save VRAM, 0=disable)", value=26,
                                    info="Higher values = less VRAM usage but slower generation")
                    framepack_bulk_decode = gr.Checkbox(label="Bulk Decode Frames (Faster Decode, Higher VRAM)", value=False, info="Decode all frames at once instead of section by section.")
                with gr.Row():
                    framepack_attn_mode = gr.Dropdown(
                        label="Attention Mode",
                        choices=["torch", "sdpa", "flash", "xformers", "sageattn"], # Added choices from script
                        value="sdpa", # Defaulting to sdpa
                        interactive=True
                    )
                    framepack_vae_chunk_size = gr.Number(label="VAE Chunk Size (CausalConv3d)", value=32, step=1, minimum=0, info="0 or None=disable (Default: None)")
                    framepack_vae_spatial_tile_sample_min_size = gr.Number(label="VAE Spatial Tile Min Size", value=128, step=16, minimum=0, info="0 or None=disable (Default: None)")
                    framepack_device = gr.Textbox(label="Device Override (optional)", placeholder="e.g., cuda:0, cpu")
                with gr.Row():
                    framepack_use_teacache = gr.Checkbox(label="Use TeaCache", value=False, info="Enable TeaCache for faster generation (shits hands).")
                    framepack_teacache_steps = gr.Number(label="TeaCache Init Steps", value=25, step=1, minimum=1, info="Steps for TeaCache init (match Inference Steps)")
                    framepack_teacache_thresh = gr.Slider(label="TeaCache Threshold", minimum=0.0, maximum=1.0, step=0.01, value=0.15, info="Relative L1 distance threshold for skipping.")                    

            with gr.Accordion("Model Paths / Advanced", open=False):
                 with gr.Row():
                    framepack_transformer_path = gr.Textbox(label="Transformer Path (DiT)", value="hunyuan/FramePackI2V_HY_bf16.safetensors", interactive=True)
                    framepack_vae_path = gr.Textbox(label="VAE Path", value="hunyuan/pytorch_model.pt")
                 with gr.Row():
                    framepack_text_encoder_path = gr.Textbox(label="Text Encoder 1 (Llama) Path *Required*", value="hunyuan/llava_llama3_fp16.safetensors")
                    framepack_text_encoder_2_path = gr.Textbox(label="Text Encoder 2 (CLIP) Path *Required*", value="hunyuan/clip_l.safetensors")
                 with gr.Row():
                    framepack_image_encoder_path = gr.Textbox(label="Image Encoder (SigLIP) Path *Required*", value="hunyuan/model.safetensors")
                    framepack_save_path = gr.Textbox(label="Save Path *Required*", value="outputs")
### FRAMEPACK EXTENSION
        with gr.Tab(id=11, label="FramePack-Extension") as framepack_extension_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    fpe_prompt = gr.Textbox(
                        scale=3, label="Prompt",
                        value="cinematic video of a cat wizard casting a spell, epic action scene", lines=3
                    )
                    fpe_negative_prompt = gr.Textbox(scale=3, label="Negative Prompt", value="", lines=3)
                with gr.Column(scale=1):
                    fpe_use_normal_framepack = gr.Checkbox(label="Use Normal FramePack Model", value=False, info="Uses og model supports end frame. Default is F1 model.")                    
                    fpe_batch_count = gr.Number(label="Batch Count", value=1, minimum=1, step=1)
                with gr.Column(scale=2):
                    fpe_batch_progress = gr.Textbox(label="Status", interactive=False, value="")
                    fpe_progress_text = gr.Textbox(label="Progress", interactive=False, lines=1, elem_id="fpe_progress_text") # Unique elem_id

            with gr.Row():
                fpe_generate_btn = gr.Button("Generate Extended Video", elem_classes="green-btn")
                fpe_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column(): # Left column for inputs
                    fpe_input_video = gr.Video(label="Input Video for Extension", sources=['upload'], height=300)
                    with gr.Accordion("Optional End Frame (for Normal FramePack Model)", open=False, visible=False) as fpe_end_frame_accordion:
                        fpe_end_frame = gr.Image(label="End Frame for Extension", type="filepath")
                        fpe_end_frame_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=1.0, label="End Frame Weight")                    
                    with gr.Accordion("Optional Start Guidance Image (for F1 Model Extension)", open=False, visible=True) as fpe_start_guidance_accordion: # Initially hidden
                        fpe_start_guidance_image = gr.Image(label="Start Guidance Image for Extension", type="filepath")
                        fpe_start_guidance_image_clip_weight = gr.Slider(
                            minimum=0.0, maximum=2.0, step=0.05, value=0.75, 
                            label="Start Guidance Image CLIP Weight",
                            info="Blend weight for the guidance image's CLIP embedding with input video's first frame CLIP."
                        )
                        fpe_use_guidance_image_as_first_latent = gr.Checkbox(
                            label="Use Guidance Image as First Latent", value=False,
                            info="If checked, the VAE latent of the guidance image will be used as the initial conditioning for the first generated segment. Turn down context frames when using this"
                        )                    
                    gr.Markdown("### Core Generation Parameters")
                    with gr.Row():
                        fpe_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                        # fpe_random_seed_btn = gr.Button("🎲️") # Optional: Add random seed button
                    
                    fpe_resolution_max_dim = gr.Number(label="Resolution (Max Dimension)", value=640, step=32, info="Target max width/height for bucket.")
                    fpe_total_second_length = gr.Slider(minimum=1.0, maximum=120.0, step=0.5, label="Additional Video Length (seconds)", value=5.0)
                    fpe_latent_window_size = gr.Slider(minimum=9, maximum=33, step=1, label="Latent Window Size", value=9, info="Default 9 for F1 model.")
                    fpe_steps = gr.Slider(minimum=1, maximum=100, step=1, label="Inference Steps", value=25)

                    with gr.Row():
                        fpe_cfg_scale = gr.Slider(minimum=1.0, maximum=32.0, step=0.1, label="CFG Scale", value=1.0, info="Usually 1.0 for F1 (no external CFG).")
                        fpe_distilled_guidance_scale = gr.Slider(minimum=1.0, maximum=32.0, step=0.1, label="Distilled Guidance (GS)", value=3.0)
                    # fpe_rs_scale = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="CFG Rescale (RS)", value=0.0, visible=False)
                    with gr.Row():
                        with gr.Accordion("Advanced & Performance", open=True):
                            fpe_gpu_memory_preservation = gr.Slider(label="GPU Memory Preserve (GB)", minimum=1.0, maximum=16.0, value=6.0, step=0.1)
                            fpe_use_teacache = gr.Checkbox(label="Use TeaCache", value=False)
                            fpe_no_resize = gr.Checkbox(label="Force Original Video Resolution (No Resize)", value=False)
                            fpe_extension_only = gr.Checkbox(label="Save Extension Only", value=False, info="If checked, only the newly generated extension part of the video will be saved.")
                            fpe_mp4_crf = gr.Slider(label="MP4 CRF (Quality)", minimum=0, maximum=51, value=1, step=1, info="Lower is better quality, larger file.")
                            fpe_num_clean_frames = gr.Slider(label="Context Frames (1x from Input)", minimum=1, maximum=10, value=5, step=1)
                            fpe_vae_batch_size = gr.Slider(label="VAE Batch Size (Input Video Encoding)", minimum=4, maximum=128, value=72, step=4)

                            fpe_attn_mode = gr.Dropdown(label="Attention Mode (DiT)", choices=["torch", "sdpa", "flash", "xformers", "sageattn"], value="torch")
                            fpe_fp8_llm = gr.Checkbox(label="Use FP8 LLM (Text Encoder 1)", value=False, visible=False)
                            fpe_vae_chunk_size = gr.Number(label="VAE Chunk Size (CausalConv3d)", value=32, step=1, minimum=0, info="0 or None=disable")
                            fpe_vae_spatial_tile_sample_min_size = gr.Number(label="VAE Spatial Tile Min Size", value=128, step=16, minimum=0, info="0 or None=disable")

                
                with gr.Column(): # Right column for outputs and advanced settings
                    fpe_output_gallery = gr.Gallery(
                        label="Generated Extended Videos", columns=[1], rows=[1], # Show one main video at a time
                        object_fit="contain", height=480, show_label=True,
                        elem_id="gallery_framepack_extension", allow_preview=True, preview=True
                    )
                    with gr.Accordion("Live Preview (During Generation)", open=True):
                        with gr.Row():
                            fpe_enable_preview = gr.Checkbox(label="Enable Live Preview", value=True, visible=False)
                            fpe_preview_interval = gr.Slider(
                                minimum=1, maximum=50, step=1, value=5,
                                label="Preview Every N Steps",
                                info="Saves a PNG preview during sampling.",
                                visible=False
                            )
                        fpe_preview_output_component = gr.Video( # Changed to Video for MP4 previews
                             label="Latest Section Preview", height=300,
                             interactive=False, elem_id="fpe_preview_video"
                        )
                        # fpe_skip_btn = gr.Button("Skip Batch Item", elem_classes="light-blue-btn") # Optional
                    gr.Markdown("### LoRA Configuration")
                    with gr.Row():
                        fpe_refresh_lora_btn = gr.Button("🔄 LoRA", elem_classes="refresh-btn")
                        fpe_lora_folder = gr.Textbox(label="LoRA Folder", value="lora", scale=4)
                    fpe_lora_weights_ui = []
                    fpe_lora_multipliers_ui = []
                    for i in range(4):
                        with gr.Row():
                            fpe_lora_weights_ui.append(gr.Dropdown(
                                label=f"LoRA {i+1}", choices=get_lora_options("lora"),
                                value="None", allow_custom_value=False, interactive=True, scale=2
                            ))
                            fpe_lora_multipliers_ui.append(gr.Slider(
                                label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1, interactive=True
                            ))
            with gr.Row():        
                with gr.Accordion("Model Paths (FramePack-Extension)", open=False):
                    fpe_transformer_path = gr.Textbox(label="DiT Path (F1 Model)", value="hunyuan/FramePack_F1_I2V_HY_20250503.safetensors") # Default to F1
                    fpe_vae_path = gr.Textbox(label="VAE Path", value="hunyuan/pytorch_model.pt")
                    fpe_text_encoder_path = gr.Textbox(label="Text Encoder 1 (Llama)", value="hunyuan/llava_llama3_fp16.safetensors")
                    fpe_text_encoder_2_path = gr.Textbox(label="Text Encoder 2 (CLIP)", value="hunyuan/clip_l.safetensors")
                    fpe_image_encoder_path = gr.Textbox(label="Image Encoder (SigLIP)", value="hunyuan/model.safetensors")
                    fpe_save_path = gr.Textbox(label="Save Path (Output Directory)", value="outputs/framepack_extensions")


### MultiTalk Tab
        with gr.Tab(id=8, label="MultiTalk") as multitalk_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    multitalk_prompt = gr.Textbox(
                        label="Prompt",
                        value="A conversation between two female bunny anchors in a studio in Ghangzhou.",
                        lines=5,
                    )
                    multitalk_negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        lines=3,
                        value="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards."
                    )                    
                with gr.Column(scale=1):
                    multitalk_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)
                with gr.Column(scale=2):
                    multitalk_status = gr.Textbox(label="Status", interactive=False, value="")
                    multitalk_progress = gr.Textbox(label="Progress", interactive=False, value="", elem_id="multitalk_progress_text")

            with gr.Row():
                multitalk_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                multitalk_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                # Left Column for inputs and core settings
                with gr.Column():
                    multitalk_cond_image = image_annotator(
                        label="Reference Image & Bounding Boxes",
                        image_type="filepath",
                        label_list=["Person 1", "Person 2"],
                        label_colors=[(255, 0, 0), (0, 255, 0)], # Red for P1, Green for P2
                    )
                    multitalk_bbox_json_output = gr.JSON(label="Current BBox Data (for Debugging)", visible=False)
                    with gr.Row():
                        multitalk_audio_person1 = gr.Audio(label="Audio for Person 1 (or single person)", type="filepath")
                        multitalk_audio_person2 = gr.Audio(label="Audio for Person 2 (optional)", type="filepath")
                    
                    gr.Markdown("### Generation Parameters")
                    with gr.Row():
                        multitalk_size = gr.Dropdown(label="Resolution", choices=["multitalk-480", "multitalk-720"], value="multitalk-480")
                    with gr.Row():
                        multitalk_mode = gr.Dropdown(label="Mode", choices=["clip", "streaming"], value="streaming", info="'streaming' for long video, 'clip' for a single chunk.")
                        multitalk_frame_num = gr.Slider(label="Frames per Chunk", minimum=41, maximum=201, value=81, step=4, info="Must be 4n+1. Default is 81.")
                        multitalk_motion_frame = gr.Slider(label="Motion Frame (for streaming)", minimum=1, maximum=40, value=25, step=1, info="Frames from previous chunk to condition the next.")
                    
                    with gr.Row():
                        multitalk_sample_steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=100, value=40, step=1)
                        multitalk_sample_shift = gr.Slider(label="Sample Shift", minimum=1.0, maximum=20.0, value=7.0, step=0.1)
                    
                    with gr.Row():
                        multitalk_text_guide_scale = gr.Slider(label="Text Guidance Scale", minimum=1.0, maximum=10.0, value=5.0, step=0.1)
                        multitalk_audio_guide_scale = gr.Slider(label="Audio Guidance Scale", minimum=1.0, maximum=10.0, value=4.0, step=0.1)

                    with gr.Row():
                        multitalk_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                        multitalk_random_seed_btn = gr.Button("🎲️")
                    with gr.Accordion("Advanced & Performance", open=True):
                        multitalk_audio_type = gr.Radio(label="Audio Mixing Type", choices=["para", "add"], value="para", info="'para' for parallel talking, 'add' for sequential.")
                        multitalk_num_persistent = gr.Slider(
                            minimum=0, 
                            maximum=26, 
                            step=0.01, 
                            value=5.0, 
                            label="Low VRAM (Persistent Params in Billions)", 
                            info="Slider value in billions. 0=very low VRAM (slower), 5=default(24gb vram)."
                        )
                        with gr.Row():
                            multitalk_use_teacache = gr.Checkbox(label="Use TeaCache (Acceleration)", value=False)
                            multitalk_teacache_thresh = gr.Slider(label="TeaCache Threshold", minimum=0.1, maximum=1.0, value=0.2, step=0.05, info="Higher is faster but may reduce quality.")
                        with gr.Row():
                            multitalk_use_apg = gr.Checkbox(label="Use APG (Reduces Color Shift)", value=False)
                            multitalk_apg_momentum = gr.Slider(label="APG Momentum", minimum=-1.0, maximum=1.0, value=-0.75, step=0.05)
                            multitalk_apg_norm_thresh = gr.Slider(label="APG Norm Threshold", minimum=10, maximum=100, value=55, step=1)

                # Right column for outputs and advanced settings
                with gr.Column():
                    multitalk_output = gr.Gallery(
                        label="Generated Videos",
                        columns=[1], rows=[1], object_fit="contain", height=480,
                        allow_preview=True, preview=True
                    )
                    with gr.Accordion("Live Preview (During Generation)", open=True):
                        multitalk_use_full_video_preview = gr.Checkbox(label="Use Full Video Previews Every Section", value=True)
                        multitalk_preview_output = gr.Video(
                            label="Latest Preview", height=300,
                            interactive=False, elem_id="multitalk_preview_video"
                        )
                    
                    with gr.Accordion("LoRA", open=True):                        
                        with gr.Row():
                            multitalk_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                            multitalk_lora_refresh_btn = gr.Button("🔄 LoRA", elem_classes="refresh-btn")
                        
                        multitalk_lora_weights_ui = []
                        multitalk_lora_multipliers_ui = []
                        for i in range(4):
                            with gr.Row():
                                multitalk_lora_weights_ui.append(gr.Dropdown(
                                    label=f"LoRA {i+1}", choices=get_lora_options("lora"),
                                    value="None", allow_custom_value=False, interactive=True, scale=2
                                ))
                                multitalk_lora_multipliers_ui.append(gr.Slider(
                                    label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1, interactive=True
                                ))
            with gr.Row():
                multitalk_ckpt_dir = gr.Textbox(label="Base Model Directory", value="wan")
                multitalk_wav2vec_dir = gr.Textbox(label="Wav2Vec Directory", value="wan/chinese-wav2vec2-base")
                multitalk_t5_tokenizer_path = gr.Textbox(label="T5 Tokenizer Override (optional)", value="wan/google/umt5-xxl")
                multitalk_save_path = gr.Textbox(label="Save Path", value="outputs/multitalk")

### InfiniteTalk Tab
        with gr.Tab(id=14, label="InfiniteTalk") as infinitetalk_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    infinitetalk_prompt = gr.Textbox(
                        label="Prompt",
                        value="A conversation between two people in a studio.",
                        lines=5,
                    )
                    infinitetalk_negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        lines=3,
                        value="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards."
                    )                    
                with gr.Column(scale=1):
                    infinitetalk_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)
                with gr.Column(scale=2):
                    infinitetalk_status = gr.Textbox(label="Status", interactive=False, value="")
                    infinitetalk_progress = gr.Textbox(label="Progress", interactive=False, value="", elem_id="infinitetalk_progress_text")

            with gr.Row():
                infinitetalk_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                infinitetalk_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                # Left Column for inputs and core settings
                with gr.Column():
                    infinitetalk_cond_image = gr.Image(
                        label="Reference Image",
                        sources=["upload"],
                        type="filepath",
                        height=300
                    )
                    with gr.Row():
                        infinitetalk_audio_person1 = gr.Audio(label="Audio for Person 1 (or single person)", type="filepath")
                        infinitetalk_audio_person2 = gr.Audio(label="Audio for Person 2 (optional)", type="filepath")
                    
                    gr.Markdown("### Generation Parameters")
                    with gr.Row():
                        infinitetalk_size = gr.Dropdown(label="Resolution", choices=["infinitetalk-480", "infinitetalk-720"], value="infinitetalk-480")
                    with gr.Row():
                        infinitetalk_mode = gr.Dropdown(label="Mode", choices=["clip", "streaming"], value="streaming", info="'streaming' for long video, 'clip' for a single chunk.")
                        infinitetalk_frame_num = gr.Slider(label="Frames per Chunk", minimum=41, maximum=201, value=81, step=4, info="Must be 4n+1. Default is 81.")
                        infinitetalk_motion_frame = gr.Slider(label="Motion Frame (for streaming)", minimum=1, maximum=40, value=9, step=1, info="Frames from previous chunk to condition the next. Default: 9 for InfiniteTalk.")
                    
                    with gr.Row():
                        infinitetalk_sample_steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=100, value=40, step=1)
                        infinitetalk_sample_shift = gr.Slider(label="Sample Shift", minimum=1.0, maximum=20.0, value=7.0, step=0.1)
                    
                    with gr.Row():
                        infinitetalk_text_guide_scale = gr.Slider(label="Text Guidance Scale", minimum=1.0, maximum=10.0, value=5.0, step=0.1)
                        infinitetalk_audio_guide_scale = gr.Slider(label="Audio Guidance Scale", minimum=1.0, maximum=10.0, value=4.0, step=0.1)

                    with gr.Row():
                        infinitetalk_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                        infinitetalk_random_seed_btn = gr.Button("🎲️")
                    
                    with gr.Accordion("Advanced & Performance", open=True):
                        infinitetalk_audio_type = gr.Radio(label="Audio Mixing Type", choices=["para", "add"], value="para", info="'para' for parallel talking, 'add' for sequential.")
                        infinitetalk_scene_seg = gr.Checkbox(label="Enable Scene Segmentation", value=False, info="Detect and process scenes separately (requires scenedetect)")
                        infinitetalk_num_persistent = gr.Slider(
                            minimum=0, 
                            maximum=26, 
                            step=0.01, 
                            value=5.0, 
                            label="Low VRAM (Persistent Params in Billions)", 
                            info="Slider value in billions. 0=very low VRAM (slower), 5=default(24gb vram)."
                        )
                        with gr.Row():
                            infinitetalk_use_teacache = gr.Checkbox(label="Use TeaCache (Acceleration)", value=False)
                            infinitetalk_teacache_thresh = gr.Slider(label="TeaCache Threshold", minimum=0.1, maximum=1.0, value=0.2, step=0.05, info="Higher is faster but may reduce quality.")
                        with gr.Row():
                            infinitetalk_use_apg = gr.Checkbox(label="Use APG (Reduces Color Shift)", value=False)
                            infinitetalk_apg_momentum = gr.Slider(label="APG Momentum", minimum=-1.0, maximum=1.0, value=-0.75, step=0.05)
                            infinitetalk_apg_norm_thresh = gr.Slider(label="APG Norm Threshold", minimum=10, maximum=100, value=55, step=1)

                # Right column for outputs and advanced settings
                with gr.Column():
                    infinitetalk_output = gr.Gallery(
                        label="Generated Videos",
                        columns=[1], rows=[1], object_fit="contain", height=480,
                        allow_preview=True, preview=True
                    )
                    with gr.Accordion("Live Preview (During Generation)", open=True):
                        infinitetalk_use_full_video_preview = gr.Checkbox(label="Use Full Video Previews Every Section", value=True)
                        infinitetalk_preview_output = gr.Video(
                            label="Latest Preview", height=300,
                            interactive=False, elem_id="infinitetalk_preview_video"
                        )
                    
                    with gr.Accordion("LoRA", open=True):                        
                        with gr.Row():
                            infinitetalk_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                            infinitetalk_lora_refresh_btn = gr.Button("🔄 LoRA", elem_classes="refresh-btn")
                        
                        infinitetalk_lora_weights_ui = []
                        infinitetalk_lora_multipliers_ui = []
                        for i in range(4):
                            with gr.Row():
                                infinitetalk_lora_weights_ui.append(gr.Dropdown(
                                    label=f"LoRA {i+1}", choices=get_lora_options("lora"),
                                    value="None", allow_custom_value=False, interactive=True, scale=2
                                ))
                                infinitetalk_lora_multipliers_ui.append(gr.Slider(
                                    label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1, interactive=True
                                ))
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### InfiniteTalk-Specific Paths")
                    infinitetalk_dit_path = gr.Textbox(label="DiT Model Path (optional)", value="", placeholder="Path to DiT model checkpoint")
                    infinitetalk_infinitetalk_dir = gr.Textbox(label="InfiniteTalk Model", value="", placeholder="Directory or safetensors file (e.g., infinitetalk_single.safetensors or infinitetalk_multi.safetensors)")
            
            with gr.Row():
                infinitetalk_ckpt_dir = gr.Textbox(label="Base Wan Model", value="wan", placeholder="Directory or safetensors file (e.g., wan2.1_i2v_480p_14B_fp16.safetensors)")
                infinitetalk_wav2vec_dir = gr.Textbox(label="Wav2Vec Directory", value="wan/chinese-wav2vec2-base")
                infinitetalk_t5_tokenizer_path = gr.Textbox(label="T5 Tokenizer Override (optional)", value="wan/google/umt5-xxl")
                infinitetalk_save_path = gr.Textbox(label="Save Path", value="outputs/infinitetalk")

        # Text to Video Tab
        with gr.Tab(id=1, label="Hunyuan-t2v"):
            with gr.Row():
                with gr.Column(scale=4):
                    prompt = gr.Textbox(scale=3, label="Enter your prompt", value="POV video of a cat chasing a frob.", lines=5)

                with gr.Column(scale=1):
                    token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    batch_progress = gr.Textbox(label="", visible=True, elem_id="batch_progress")
                    progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text")

            with gr.Row():
                generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column():
                    
                    t2v_width = gr.Slider(minimum=64, maximum=1536, step=16, value=544, label="Video Width")
                    t2v_height = gr.Slider(minimum=64, maximum=1536, step=16, value=544, label="Video Height")
                    video_length = gr.Slider(minimum=1, maximum=201, step=1, label="Video Length in Frames", value=25, elem_id="my_special_slider")
                    fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=24, elem_id="my_special_slider")
                    infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=30, elem_id="my_special_slider")
                    flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=11.0, elem_id="my_special_slider")
                    cfg_scale = gr.Slider(minimum=0.0, maximum=14.0, step=0.1, label="cfg Scale", value=7.0, elem_id="my_special_slider")
            
                with gr.Column():

                    with gr.Row():
                        video_output = gr.Gallery(
                            label="Generated Videos (Click to select)",
                            columns=[2],
                            rows=[2],
                            object_fit="contain",
                            height="auto",
                            show_label=True,
                            elem_id="gallery",
                            allow_preview=True,
                            preview=True
                        )
                    with gr.Row():send_t2v_to_v2v_btn = gr.Button("Send Selected to Video2Video")
            
            with gr.Row():
                    refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
                    lora_weights = []
                    lora_multipliers = []
                    for i in range(4):
                        with gr.Column():
                            lora_weights.append(gr.Dropdown(
                                label=f"LoRA {i+1}", 
                                choices=get_lora_options(), 
                                value="None", 
                                allow_custom_value=True,
                                interactive=True
                            ))
                            lora_multipliers.append(gr.Slider(
                                label=f"Multiplier", 
                                minimum=0.0, 
                                maximum=2.0, 
                                step=0.05, 
                                value=1.0
                            ))            
            with gr.Row():
                exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)
                seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                dit_folder = gr.Textbox(label="DiT Model Folder", value="hunyuan")
                model = gr.Dropdown(
                    label="DiT Model",
                    choices=get_dit_models("hunyuan"),
                    value="mp_rank_00_model_states.pt",
                    allow_custom_value=True,
                    interactive=True
                )
                vae = gr.Textbox(label="vae", value="hunyuan/pytorch_model.pt")
                te1 = gr.Textbox(label="te1", value="hunyuan/llava_llama3_fp16.safetensors")
                te2 = gr.Textbox(label="te2", value="hunyuan/clip_l.safetensors")
                save_path = gr.Textbox(label="Save Path", value="outputs")
            with gr.Row():
                lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                use_split_attn = gr.Checkbox(label="Use Split Attention", value=False)
                use_fp8 = gr.Checkbox(label="Use FP8 (faster but lower precision)", value=True)
                attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                block_swap = gr.Slider(minimum=0, maximum=36, step=1, label="Block Swap to Save Vram", value=0)

        #Image to Video Tab
        with gr.Tab(label="Hunyuan-i2v", visible=False) as i2v_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    i2v_prompt = gr.Textbox(scale=3, label="Enter your prompt", value="POV video of a cat chasing a frob.", lines=5)

                with gr.Column(scale=1):
                    i2v_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    i2v_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    i2v_batch_progress = gr.Textbox(label="", visible=True, elem_id="batch_progress_i2v") # Unique elem_id
                    i2v_progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text_i2v") # Unique elem_id

            with gr.Row():
                i2v_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                i2v_stop_btn = gr.Button("Stop Generation", variant="stop")


            with gr.Row():
                with gr.Column():
                    i2v_input = gr.Image(label="Input Image", type="filepath")
                    # REMOVED i2v_strength slider, as hv_i2v_generate_video.py doesn't seem to use it based on the sample command
                    # i2v_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.75, label="Denoise Strength")
                    scale_slider = gr.Slider(minimum=1, maximum=200, value=100, step=1, label="Scale % (UI Only - affects W/H)") # Clarified UI only
                    original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=True)
                    # Width and height inputs
                    with gr.Row():
                        # Renamed width/height to avoid potential conflicts if they weren't already prefixed
                        i2v_width = gr.Number(label="New Width", value=720, step=16) # Default from sample
                        calc_height_btn = gr.Button("→")
                        calc_width_btn = gr.Button("←")
                        i2v_height = gr.Number(label="New Height", value=720, step=16) # Default from sample
                    i2v_video_length = gr.Slider(minimum=1, maximum=201, step=1, label="Video Length in Frames", value=49) # Default from sample
                    i2v_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=24) # Default from sample
                    i2v_infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=30) # Default from sample
                    i2v_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=17.0) # Default from sample
                    i2v_cfg_scale = gr.Slider(minimum=0.0, maximum=14.0, step=0.1, label="Embedded CFG Scale", value=7.0) # Default from sample
                    i2v_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale (CFG)", value=1.0) # Default from sample (usually 1.0 for no CFG)

                with gr.Column():
                    i2v_output = gr.Gallery(
                        label="Generated Videos (Click to select)",
                        columns=[2],
                        rows=[2],
                        object_fit="contain",
                        height="auto",
                        show_label=True,
                        elem_id="gallery_i2v", # Unique elem_id
                        allow_preview=True,
                        preview=True
                    )
                    i2v_send_to_v2v_btn = gr.Button("Send Selected to Hunyuan-v2v") # Keep sending to original V2V

                    # Add LoRA section for Image2Video
                    i2v_refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
                    i2v_lora_weights = []
                    i2v_lora_multipliers = []
                    for i in range(4):
                        with gr.Column():
                            i2v_lora_weights.append(gr.Dropdown(
                                label=f"LoRA {i+1}",
                                choices=get_lora_options(),
                                value="None",
                                allow_custom_value=True,
                                interactive=True
                            ))
                            i2v_lora_multipliers.append(gr.Slider(
                                label=f"Multiplier",
                                minimum=0.0,
                                maximum=2.0,
                                step=0.05,
                                value=1.0
                            ))

            with gr.Row():
                i2v_exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)
                i2v_seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                i2v_dit_folder = gr.Textbox(label="DiT Model Folder", value="hunyuan")
                i2v_model = gr.Dropdown(
                    label="DiT Model",
                    choices=get_dit_models("hunyuan"),
                    value="mp_rank_00_model_states_i2v.pt", # Default from sample
                    allow_custom_value=True,
                    interactive=True
                )
                i2v_vae = gr.Textbox(label="VAE Path", value="hunyuan/pytorch_model.pt") # Default from sample
                i2v_te1 = gr.Textbox(label="Text Encoder 1 Path", value="hunyuan/llava_llama3_fp16.safetensors") # Default from sample
                i2v_te2 = gr.Textbox(label="Text Encoder 2 Path", value="hunyuan/clip_l.safetensors") # Default from sample
                i2v_clip_vision_path = gr.Textbox(label="CLIP Vision Path", value="hunyuan/llava_llama3_vision.safetensors") # Default from sample
                i2v_save_path = gr.Textbox(label="Save Path", value="outputs") # Default from sample
            with gr.Row():
                i2v_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                i2v_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video") # Default from sample
                i2v_use_split_attn = gr.Checkbox(label="Use Split Attention", value=False) # Not in sample, keep default False
                i2v_use_fp8 = gr.Checkbox(label="Use FP8 DiT", value=False) # Not in sample, keep default False
                i2v_fp8_llm = gr.Checkbox(label="Use FP8 LLM", value=False) # Not in sample, keep default False
                i2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa") # Default from sample
                i2v_block_swap = gr.Slider(minimum=0, maximum=36, step=1, label="Block Swap to Save Vram", value=30) # Default from sample
                # Add VAE tiling options like sample command
                i2v_vae_chunk_size = gr.Number(label="VAE Chunk Size", value=32, step=1, info="For CausalConv3d, set 0 to disable")
                i2v_vae_spatial_tile_min = gr.Number(label="VAE Spatial Tile Min Size", value=128, step=16, info="Set 0 to disable spatial tiling")

        # Video to Video Tab
        with gr.Tab(id=2, label="Hunyuan v2v", visible=False) as v2v_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    v2v_prompt = gr.Textbox(scale=3, label="Enter your prompt", value="POV video of a cat chasing a frob.", lines=5)
                    v2v_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt (for SkyReels models)",
                        value="Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
                        lines=3
                    )

                with gr.Column(scale=1):
                    v2v_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    v2v_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    v2v_batch_progress = gr.Textbox(label="", visible=True, elem_id="batch_progress")
                    v2v_progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text")

            with gr.Row():
                v2v_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                v2v_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column():
                    v2v_input = gr.Video(label="Input Video", format="mp4")
                    v2v_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.75, label="Denoise Strength")
                    v2v_scale_slider = gr.Slider(minimum=1, maximum=200, value=100, step=1, label="Scale %")
                    v2v_original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=True)

                    # Width and Height Inputs
                    with gr.Row():
                        v2v_width = gr.Number(label="New Width", value=544, step=16)
                        v2v_calc_height_btn = gr.Button("→")
                        v2v_calc_width_btn = gr.Button("←")
                        v2v_height = gr.Number(label="New Height", value=544, step=16)
                    v2v_video_length = gr.Slider(minimum=1, maximum=201, step=1, label="Video Length in Frames", value=25)
                    v2v_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=24)
                    v2v_infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=30)
                    v2v_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=11.0)
                    v2v_cfg_scale = gr.Slider(minimum=0.0, maximum=14.0, step=0.1, label="cfg scale", value=7.0)
                with gr.Column():
                    v2v_output = gr.Gallery(
                        label="Generated Videos",
                        columns=[1],
                        rows=[1],
                        object_fit="contain",
                        height="auto"
                    )
                    v2v_send_to_input_btn = gr.Button("Send Selected to Input")  # New button
                    v2v_refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
                    v2v_lora_weights = []
                    v2v_lora_multipliers = []
                    for i in range(4):
                        with gr.Column():
                            v2v_lora_weights.append(gr.Dropdown(
                                label=f"LoRA {i+1}", 
                                choices=get_lora_options(), 
                                value="None", 
                                allow_custom_value=True,
                                interactive=True
                            ))
                            v2v_lora_multipliers.append(gr.Slider(
                                label=f"Multiplier", 
                                minimum=0.0, 
                                maximum=2.0, 
                                step=0.05, 
                                value=1.0
                            ))

            with gr.Row():
                v2v_exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)                
                v2v_seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                v2v_dit_folder = gr.Textbox(label="DiT Model Folder", value="hunyuan")
                v2v_model = gr.Dropdown(
                    label="DiT Model",
                    choices=get_dit_models("hunyuan"),
                    value="mp_rank_00_model_states.pt",
                    allow_custom_value=True,
                    interactive=True
                )
                v2v_vae = gr.Textbox(label="vae", value="hunyuan/pytorch_model.pt")
                v2v_te1 = gr.Textbox(label="te1", value="hunyuan/llava_llama3_fp16.safetensors")
                v2v_te2 = gr.Textbox(label="te2", value="hunyuan/clip_l.safetensors")
                v2v_save_path = gr.Textbox(label="Save Path", value="outputs")
            with gr.Row():
                v2v_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                v2v_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                v2v_use_split_attn = gr.Checkbox(label="Use Split Attention", value=False)
                v2v_use_fp8 = gr.Checkbox(label="Use FP8 (faster but lower precision)", value=True)
                v2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                v2v_block_swap = gr.Slider(minimum=0, maximum=36, step=1, label="Block Swap to Save Vram", value=0)
                v2v_split_uncond = gr.Checkbox(label="Split Unconditional (for SkyReels)", value=True)

### SKYREELS

        with gr.Tab(label="SkyReels-i2v", visible=False) as skyreels_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    skyreels_prompt = gr.Textbox(
                        scale=3, 
                        label="Enter your prompt", 
                        value="A person walking on a beach at sunset", 
                        lines=5
                    )
                    skyreels_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt",
                        value="Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
                        lines=3
                    )

                with gr.Column(scale=1):
                    skyreels_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    skyreels_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    skyreels_batch_progress = gr.Textbox(label="", visible=True, elem_id="batch_progress")
                    skyreels_progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text")

            with gr.Row():
                skyreels_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                skyreels_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column():
                    skyreels_input = gr.Image(label="Input Image (optional)", type="filepath")
                    with gr.Row():
                        skyreels_use_random_folder = gr.Checkbox(label="Use Random Images from Folder", value=False)
                        skyreels_input_folder = gr.Textbox(
                            label="Image Folder Path", 
                            placeholder="Path to folder containing images",
                            visible=False
                        )
                        skyreels_folder_status = gr.Textbox(
                            label="Folder Status", 
                            placeholder="Status will appear here",
                            interactive=False,
                            visible=False
                        )
                        skyreels_validate_folder_btn = gr.Button("Validate Folder", visible=False)
                    skyreels_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.75, label="Denoise Strength")

                    # Scale slider as percentage 
                    skyreels_scale_slider = gr.Slider(minimum=1, maximum=200, value=100, step=1, label="Scale %")
                    skyreels_original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=True)

                    # Width and height inputs
                    with gr.Row():
                        skyreels_width = gr.Number(label="New Width", value=544, step=16)
                        skyreels_calc_height_btn = gr.Button("→")
                        skyreels_calc_width_btn = gr.Button("←")
                        skyreels_height = gr.Number(label="New Height", value=544, step=16)

                    skyreels_video_length = gr.Slider(minimum=1, maximum=201, step=1, label="Video Length in Frames", value=25)
                    skyreels_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=24)
                    skyreels_infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=30)
                    skyreels_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=11.0)
                    skyreels_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale", value=6.0)
                    skyreels_embedded_cfg_scale = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, label="Embedded CFG Scale", value=1.0)

                with gr.Column():
                    skyreels_output = gr.Gallery(
                        label="Generated Videos (Click to select)",
                        columns=[2],
                        rows=[2],
                        object_fit="contain",
                        height="auto",
                        show_label=True,
                        elem_id="gallery",
                        allow_preview=True,
                        preview=True
                    )
                    skyreels_send_to_v2v_btn = gr.Button("Send Selected to Video2Video")

                    # Add LoRA section for SKYREELS
                    skyreels_refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
                    skyreels_lora_weights = []
                    skyreels_lora_multipliers = []
                    for i in range(4):
                        with gr.Column():
                            skyreels_lora_weights.append(gr.Dropdown(
                                label=f"LoRA {i+1}", 
                                choices=get_lora_options(), 
                                value="None", 
                                allow_custom_value=True,
                                interactive=True
                            ))
                            skyreels_lora_multipliers.append(gr.Slider(
                                label=f"Multiplier", 
                                minimum=0.0, 
                                maximum=2.0, 
                                step=0.05, 
                                value=1.0
                            ))
            with gr.Row():
                skyreels_exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)                
                skyreels_seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                skyreels_dit_folder = gr.Textbox(label="DiT Model Folder", value="hunyuan")
                skyreels_model = gr.Dropdown(
                    label="DiT Model",
                    choices=get_dit_models("skyreels"),
                    value="skyreels_hunyuan_i2v_bf16.safetensors",
                    allow_custom_value=True,
                    interactive=True
                )
                skyreels_vae = gr.Textbox(label="vae", value="hunyuan/pytorch_model.pt")
                skyreels_te1 = gr.Textbox(label="te1", value="hunyuan/llava_llama3_fp16.safetensors")
                skyreels_te2 = gr.Textbox(label="te2", value="hunyuan/clip_l.safetensors")
                skyreels_save_path = gr.Textbox(label="Save Path", value="outputs")

            with gr.Row():
                skyreels_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                skyreels_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                skyreels_use_split_attn = gr.Checkbox(label="Use Split Attention", value=False)
                skyreels_use_fp8 = gr.Checkbox(label="Use FP8 (faster but lower precision)", value=True)
                skyreels_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                skyreels_block_swap = gr.Slider(minimum=0, maximum=36, step=1, label="Block Swap to Save Vram", value=0)
                skyreels_split_uncond = gr.Checkbox(label="Split Unconditional", value=True)

        # Wan2.2 Tab
        with gr.Tab(id=12, label="Wan2.2") as wan22_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    wan22_prompt = gr.Textbox(
                        scale=3, 
                        label="Enter your prompt", 
                        value="A cat wearing a chef hat, cooking pizza.", 
                        lines=5
                    )
                    wan22_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt",
                        value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, distorted view.",
                        lines=3,
                    )
                with gr.Column(scale=1):
                    wan22_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    wan22_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)
                with gr.Column(scale=2):
                    wan22_batch_progress = gr.Textbox(label="Status", interactive=False, value="")
                    wan22_progress_text = gr.Textbox(label="Progress", interactive=False, value="", elem_id="wan22_progress_text")

            with gr.Row():
                wan22_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                wan22_stop_btn = gr.Button("Stop Generation", variant="stop")
            
            with gr.Row():
                with gr.Column():
                    wan22_input_image = gr.Image(label="Input Image (for i2v tasks)", type="filepath")
                    wan22_original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=False)
                    
                    # V2V controls
                    wan22_enable_v2v = gr.Checkbox(
                        label="Enable Video-to-Video (V2V) Mode", 
                        value=False,
                        info="Use input video for video-to-video generation"
                    )
                    with gr.Group(visible=False) as wan22_v2v_controls:
                        wan22_input_video = gr.Video(label="Input Video", format="mp4")
                        wan22_v2v_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="V2V Strength", value=0.5, info="How much to modify the input video (0=keep original, 1=full rewrite)")
                        wan22_v2v_low_noise_only = gr.Checkbox(label="Use Low Noise Model Only", value=False, info="For V2V with dual-dit models, use only the low noise model")
                        wan22_v2v_use_i2v = gr.Checkbox(label="Use I2V Model for V2V", value=False, info="Extract first frame for CLIP conditioning. Recommended for i2v-A14B.")
                        
                        # Clean I2V-based video extension
                        wan22_enable_extension = gr.Checkbox(
                            label="Enable Video Extension", 
                            value=False,
                            info="Extend input video using clean i2v-based generation with smooth blending"
                        )
                        with gr.Row(visible=False) as wan22_extension_controls:
                            wan22_extend_frames = gr.Number(
                                label="Total Frames to Generate", 
                                value=200, 
                                minimum=81, 
                                maximum=500,
                                step=1,
                                info="Total number of frames in the extended video (81-500 frames)"
                            )
                            wan22_frames_to_check = gr.Number(
                                label="Frames to Check from End",
                                value=30,
                                minimum=1,
                                maximum=100,
                                step=1,
                                info="Number of frames from the end to analyze for the best transition point"
                            )
                        with gr.Row(visible=False) as wan22_extension_info:
                            wan22_extension_summary = gr.HTML(
                                value="<p><i>Extension will intelligently find the best transition frame and generate smooth video chunks.</i></p>",
                                label="Extension Info"
                            )
                    
                    gr.Markdown("### Generation Parameters")
                    wan22_task = gr.Dropdown(
                        label="Task", 
                        choices=["t2v-A14B", "i2v-A14B", "ti2v-5B"], 
                        value="i2v-A14B",
                        info="Selects the model architecture and configuration to use."
                    )
                    # Width and height inputs
                    with gr.Row():
                        wan22_width = gr.Number(label="Width", value=832, interactive=True)
                        wan22_calc_height_btn = gr.Button("→")
                        wan22_calc_width_btn = gr.Button("←")
                        wan22_height = gr.Number(label="Height", value=480, interactive=True)
                    wan22_frame_num = gr.Slider(minimum=9, maximum=201, step=4, label="Frame Count", value=81, info="Must be 4n+1")
                    wan22_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=16)
                    wan22_sample_steps = gr.Slider(minimum=4, maximum=100, step=1, label="Sampling Steps", value=40)
                    wan22_flow_shift = gr.Slider(minimum=0.0, maximum=20.0, step=0.1, label="Flow Shift", value=5.0)
                    wan22_sample_guide_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale", value=3.5)
                    wan22_dual_dit_boundary = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Dual-DiT Boundary", value=0.875, visible=True, info="Low noise model used after this threshold (0.875 = 87.5%). Only for A14B models")
                    wan22_sample_solver = gr.Radio(choices=["unipc", "dpm++", "vanilla"], label="Sample Solver", value="unipc")
                    with gr.Row():
                        wan22_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                        wan22_random_seed_btn = gr.Button("🎲")

                with gr.Column():
                    wan22_output = gr.Gallery(
                        label="Generated Videos (Click to select)",
                        columns=[2], rows=[2], object_fit="contain", height="auto",
                        show_label=True, elem_id="gallery_wan22", allow_preview=True, preview=True
                    )
                    with gr.Accordion("Latent Preview (During Generation)", open=True):
                        wan22_enable_preview = gr.Checkbox(label="Enable Latent Preview", value=True)
                        wan22_preview_steps = gr.Slider(minimum=1, maximum=50, step=1, value=5,
                                                       label="Preview Every N Steps")
                        wan22_preview_output = gr.Gallery(
                            label="Latent Previews", columns=4, rows=2, object_fit="contain", height=300,
                            allow_preview=True, preview=True, show_label=True, elem_id="wan22_preview_gallery"
                        )
                    with gr.Accordion("LoRA", open=True):
                        with gr.Row():
                            wan22_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                            wan22_lora_refresh_btn = gr.Button("🔄 LoRA", elem_classes="refresh-btn")
                        wan22_lora_weights = []
                        wan22_lora_multipliers = []
                        wan22_lora_apply_low = []
                        wan22_lora_apply_high = []
                        for i in range(4):
                            with gr.Row():
                                wan22_lora_weights.append(gr.Dropdown(
                                    label=f"LoRA {i+1}", choices=get_lora_options("lora"),
                                    value="None", allow_custom_value=False, interactive=True, scale=2
                                ))
                                wan22_lora_multipliers.append(gr.Slider(
                                    label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1, interactive=True
                                ))
                            with gr.Row():
                                wan22_lora_apply_low.append(gr.Checkbox(
                                    label="Apply to Low Noise", value=True, scale=1
                                ))
                                wan22_lora_apply_high.append(gr.Checkbox(
                                    label="Apply to High Noise", value=False, scale=1
                                ))
                    with gr.Accordion("Additional LoRAs (5-8)", open=False):
                        for i in range(4, 8):
                            with gr.Row():
                                wan22_lora_weights.append(gr.Dropdown(
                                    label=f"LoRA {i+1}", choices=get_lora_options("lora"),
                                    value="None", allow_custom_value=False, interactive=True, scale=2
                                ))
                                wan22_lora_multipliers.append(gr.Slider(
                                    label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1, interactive=True
                                ))
                            with gr.Row():
                                wan22_lora_apply_low.append(gr.Checkbox(
                                    label="Apply to Low Noise", value=True, scale=1
                                ))
                                wan22_lora_apply_high.append(gr.Checkbox(
                                    label="Apply to High Noise", value=False, scale=1
                                ))                                
            
            with gr.Accordion("Model Paths & Performance", open=True):
                with gr.Row():
                    wan22_attn_mode = gr.Radio(choices=["sdpa", "flash", "torch", "xformers"], label="Attention Mode", value="sdpa")
                    wan22_block_swap = gr.Slider(minimum=0, maximum=39, step=1, label="Block Swap to Save VRAM", value=30)
                with gr.Row():
                    wan22_fp8 = gr.Checkbox(label="Use FP8 (DiT)", value=False)
                    wan22_fp8_scaled = gr.Checkbox(label="Use Scaled FP8 (DiT)", value=False)
                    wan22_fp8_t5 = gr.Checkbox(label="Use FP8 for T5", value=False)
                    wan22_dynamic_model_loading = gr.Checkbox(
                        label="Dynamic Model Loading (A14B models only to lower RAM usages)", 
                        value=False, visible=False
                    )
                    wan22_unload_text_encoders = gr.Checkbox(
                        label="Unload Text Encoders after use (T5/CLIP) to save RAM",
                        value=False, visible=False
                    )
                    wan22_mixed_dtype = gr.Checkbox(label="Mixed Dtype (preserve fp32 weights)", value=False)
                    wan22_vae_fp32 = gr.Checkbox(
                        label="Use FP32 VAE (higher quality, more VRAM)",
                        value=True,
                    )
                with gr.Row():
                    wan22_model_folder = gr.Textbox(label="Model Folder", value="wan")
                    wan22_refresh_models_btn = gr.Button("🔄 Models", elem_classes="refresh-btn")
                with gr.Row():
                    with gr.Group(visible=True) as wan22_a14b_paths:
                        wan22_dit_low_noise_path = gr.Dropdown(
                            label="DiT Low Noise Model (.safetensors)",
                            choices=get_wan_of_low_noise_models("wan"),
                            value=get_default_low_noise_model("wan"),
                            allow_custom_value=True,
                            interactive=True
                        )
                        wan22_dit_high_noise_path = gr.Dropdown(
                            label="DiT High Noise Model (.safetensors)",
                            choices=get_wan_of_high_noise_models("wan"),
                            value=get_default_high_noise_model("wan"),
                            allow_custom_value=True,
                            interactive=True
                        )
                        wan22_clip_path = gr.Dropdown(
                            label="CLIP Model (.pth, for i2v)",
                            choices=get_wan_of_clip_models("wan"),
                            value=get_default_clip_model("wan"),
                            allow_custom_value=True,
                            interactive=True,
                            visible=True
                        )
                    with gr.Group(visible=False) as wan22_ti2v5b_paths:
                        wan22_dit_path = gr.Dropdown(
                            label="DiT Model (.safetensors, for ti2v-5B)",
                            choices=get_wan_of_dit_models("wan"),
                            value="Wan2.2-TI2V-5B_fp16.safetensors",
                            allow_custom_value=True,
                            interactive=True
                        )
                with gr.Row():
                    wan22_vae_path = gr.Dropdown(
                        label="VAE Model (.pth)",
                        choices=get_wan_of_vae_models("wan"),
                        value=get_default_vae_model("wan"),
                        allow_custom_value=True,
                        interactive=True
                    )
                    wan22_t5_path = gr.Dropdown(
                        label="T5 Model (.pth/.safetensors)",
                        choices=get_wan_of_t5_models("wan"),
                        value=get_default_t5_model("wan"),
                        allow_custom_value=True,
                        interactive=True
                    )
                wan22_save_path = gr.Textbox(label="Save Path", value="outputs")
        
# Phantom Tab (Subject-to-Video style)
        with gr.Tab(id=7, label="Phantom") as phantom_tab: # Assign a unique ID
            with gr.Row():
                with gr.Column(scale=4):
                    phantom_prompt = gr.Textbox(
                        scale=3,
                        label="Enter your prompt",
                        value="A cute kitty is preparing a spicy breakfast meal for her brother in law.",
                        lines=5,
                        info="Describe the scene and subject. Use a unique token for your subject if fine-tuned."
                    )
                    phantom_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt",
                        value="low quality, blurry, watermark, text, signature, ugly, deformed",
                        lines=3,
                    )

                with gr.Column(scale=1):
                    phantom_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    phantom_batch_size = gr.Number(label="Batch Count (Seed Variations)", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    phantom_batch_progress = gr.Textbox(label="Status", interactive=False, value="", elem_id="phantom_batch_progress")
                    phantom_progress_text = gr.Textbox(label="Progress", interactive=False, value="", elem_id="phantom_progress_text")

            with gr.Row():
                phantom_generate_btn = gr.Button("Generate Phantom Video", elem_classes="green-btn")
                phantom_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column():
                    phantom_input_images = gr.Gallery(
                        label="Reference Images",
                        type="filepath",  # CHANGED from "pil" to "filepath"
                        show_label=True,
                        columns=3,  # Adjusted for potentially more images
                        rows="auto", # Let rows adjust
                        object_fit="contain",
                        height="auto",
                        selected_index=0,
                        interactive=True
                    )
                    phantom_add_more_images_btn = gr.UploadButton(
                        "Upload/Add More Reference Images",
                        file_types=["image"],
                        file_count="multiple",
                        type="filepath" # Ensures it provides file paths
                    )
                    # Simplified dimension inputs, no automatic aspect ratio from single image
                    with gr.Row():
                        phantom_width = gr.Number(label="Width", value=832, interactive=True, step=32, info="Should be divisible by 32")
                        phantom_height = gr.Number(label="Height", value=480, interactive=True, step=32, info="Should be divisible by 32")
                        phantom_recommend_flow_btn = gr.Button("Recommend Flow Shift", size="sm")

                    phantom_video_length = gr.Slider(minimum=1, maximum=201, step=4, label="Video Length in Frames", value=81)
                    phantom_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=16)
                    phantom_infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=50) # S2V default is 50
                    phantom_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=5.0,
                                            info="Recommended: 3.0 for 480p, 5.0 for others. S2V default often 5.0 or 3.0.")
                    # S2V has separate guidance scales for image and text
                    phantom_guide_scale_img = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, label="Image Guidance Scale", value=5.0)
                    phantom_guide_scale_text = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, label="Text Guidance Scale", value=7.5)
                    # General guidance_scale (used by non-S2V CFG, but S2V has its own way)
                    # For clarity, let's call the script's main --guidance_scale `phantom_cfg_scale_main`
                    # S2V's internal CFG is driven by guide_scale_img and guide_scale_text
                    phantom_cfg_scale_main = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, label="Main CFG Scale (Script Default)", value=5.0, visible=False, info="Usually not directly used by S2V logic if task is S2V.")


                with gr.Column():
                    phantom_output = gr.Gallery(
                        label="Generated Videos (Click to select)",
                        columns=[2], rows=[2], object_fit="contain", height="auto",
                        show_label=True, elem_id="gallery_phantom", allow_preview=True, preview=True
                    )
                    with gr.Accordion("Latent Preview (During Generation)", open=True):
                        phantom_enable_preview = gr.Checkbox(label="Enable Latent Preview", value=True)
                        phantom_preview_steps = gr.Slider(minimum=1, maximum=50, step=1, value=5,
                                                       label="Preview Every N Steps", info="Generates previews during the sampling loop.")
                        phantom_preview_output = gr.Gallery(
                            label="Latent Previews", columns=4, rows=2, object_fit="contain", height=300,
                            allow_preview=True, preview=True, show_label=True, elem_id="phantom_preview_gallery"
                        )
                    phantom_send_to_hunyuan_v2v_btn = gr.Button("Send Selected to Hunyuan-v2v")
                    phantom_send_to_wanx_v2v_btn = gr.Button("Send Selected to WanX-v2v")

                    phantom_refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
                    phantom_lora_weights_ui = []
                    phantom_lora_multipliers_ui = []
                    for i in range(4):
                        with gr.Column():
                            phantom_lora_weights_ui.append(gr.Dropdown(
                                label=f"LoRA {i+1}", choices=get_lora_options(), value="None",
                                allow_custom_value=True, interactive=True
                            ))
                            phantom_lora_multipliers_ui.append(gr.Slider(
                                label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0
                            ))

            with gr.Row():
                phantom_seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                phantom_task = gr.Dropdown(
                    label="Task (S2V variant)",
                    choices=["s2v-14B-phantom", "s2v-1.3B-phantom"], # Example S2V tasks
                    value="s2v-14B-phantom", # Default to an S2V task
                    info="Select S2V model type. This task determines internal model configuration."
                )
                phantom_dit_folder = gr.Textbox(label="DiT Model Folder", value="wan")
                phantom_dit_path = gr.Dropdown( # This will be the --phantom_ckpt path
                    label="Phantom DiT Model (.pth or .safetensors)",
                    choices=get_dit_models("wan"), # User should place Phantom model here
                    value="phantom14B.safetensors", # Example name
                    allow_custom_value=True, interactive=True
                )
                phantom_vae_path = gr.Textbox(label="VAE Path", value="wan/Wan2.1_VAE.pth")
                phantom_t5_path = gr.Textbox(label="T5 Path", value="wan/models_t5_umt5-xxl-enc-bf16.pth")
                # CLIP path is not typically used for S2V's image features, as VAE features are used.
                # However, the WanModel `model_type="i2v"` might expect CLIP features if not careful.
                # The `prepare_s2v_inputs` function in `wan_generate_video.py` does not use CLIP.
                phantom_clip_path = gr.Textbox(label="CLIP Path (for text or non-S2V I2V)", value="wan/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", visible=False)
                phantom_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                phantom_save_path = gr.Textbox(label="Save Path", value="outputs/phantom")

            with gr.Row():
                phantom_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                phantom_sample_solver = gr.Radio(choices=["unipc", "dpm++", "vanilla"], label="Sample Solver", value="unipc")
                phantom_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                phantom_block_swap = gr.Slider(minimum=0, maximum=39, step=1, label="Block Swap to Save VRAM", value=26)

                with gr.Column():
                    phantom_fp8 = gr.Checkbox(label="Use FP8 (DiT)", value=False)
                    phantom_fp8_scaled = gr.Checkbox(label="Use Scaled FP8 (DiT)", value=False, info="For mixing fp16/bf16 and fp8 weights")
                    phantom_fp8_t5 = gr.Checkbox(label="Use FP8 for T5", value=False)
            # SLG options can be kept as they are model-agnostic ways to modify CFG
            with gr.Row():
                phantom_slg_layers = gr.Textbox(label="SLG Layers", value="", placeholder="Comma-separated, e.g. 1,5,10")
                phantom_slg_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SLG Start", value=0.0)
                phantom_slg_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SLG End", value=1.0)

            with gr.Row():
                phantom_enable_cfg_skip = gr.Checkbox(label="Enable CFG Skip", value=False)
                with gr.Column(visible=False) as phantom_cfg_skip_options:
                    phantom_cfg_skip_mode = gr.Radio(
                        choices=["early", "late", "middle", "early_late", "alternate", "none"],
                        label="CFG Skip Mode", value="none"
                    )
                    phantom_cfg_apply_ratio = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.05, value=0.7, label="CFG Apply Ratio"
                    )

        # WanX Image to Video Tab
        with gr.Tab(id=4, label="WanX-i2v") as wanx_i2v_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    wanx_prompt = gr.Textbox(
                        scale=3, 
                        label="Enter your prompt", 
                        value="A person walking on a beach at sunset", 
                        lines=5
                    )
                    wanx_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt",
                        value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        lines=3,
                    )

                with gr.Column(scale=1):
                    wanx_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    wanx_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    wanx_batch_progress = gr.Textbox(label="", visible=True, elem_id="batch_progress")
                    wanx_progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text")

            with gr.Row():
                wanx_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                wanx_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column():
                    wanx_input = gr.Image(label="Input Image", type="filepath")
                    with gr.Row():
                        wanx_use_random_folder = gr.Checkbox(label="Use Random Images from Folder", value=False)
                        wanx_input_folder = gr.Textbox(
                            label="Image Folder Path", 
                            placeholder="Path to folder containing images",
                            visible=False
                        )
                        wanx_folder_status = gr.Textbox(
                            label="Folder Status", 
                            placeholder="Status will appear here",
                            interactive=False,
                            visible=False
                        )
                        wanx_validate_folder_btn = gr.Button("Validate Folder", visible=False)
                    with gr.Row():
                        wanx_use_end_image = gr.Checkbox(label="use ending image", value=False)
                        wanx_input_end = gr.Image(label="End Image", type="filepath", visible=False)
                        wanx_trim_frames = gr.Checkbox(label="trim last 3 frames", value=True, visible=False, interactive=True)

                    with gr.Row():
                        wanx_use_fun_control = gr.Checkbox(label="Use Fun-Control Model", value=False)
                        wanx_control_video = gr.Video(label="Control Video for Fun-Control", visible=False, format="mp4")
                        wanx_control_strength = gr.Slider(minimum=0.1, maximum=2.0, step=0.05, value=1.0, 
                            label="Control Strength", visible=False,
                            info="Adjust influence of control video (1.0 = normal)")
                        wanx_control_start = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.0,
                            label="Control Start (Fun-Control fade-in)",
                            visible=False,
                            info="When (0-1) in the timeline control influence is full after fade-in"
                        )
                        wanx_control_end = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=1.0,
                            label="Control End (Fun-Control fade-out start)",
                            visible=False,
                            info="When (0-1) in the timeline control starts to fade out"
                        )
                    wanx_scale_slider = gr.Slider(minimum=1, maximum=200, value=100, step=1, label="Scale %")
                    wanx_original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=True)
        
                    # Width and height display
                    with gr.Row():
                        wanx_width = gr.Number(label="Width", value=832, interactive=True)
                        wanx_calc_height_btn = gr.Button("→")
                        wanx_calc_width_btn = gr.Button("←")
                        wanx_height = gr.Number(label="Height", value=480, interactive=True)
                        wanx_recommend_flow_btn = gr.Button("Recommend Flow Shift", size="sm")

                    wanx_video_length = gr.Slider(minimum=1, maximum=401, step=4, label="Video Length in Frames", value=81)
                    wanx_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=16)
                    wanx_infer_steps = gr.Slider(minimum=1, maximum=100, step=1, label="Inference Steps", value=20)
                    wanx_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=3.0, 
                                            info="Recommended: 3.0 for 480p, 5.0 for others")
                    wanx_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, label="Guidance Scale", value=5.0)

                with gr.Column():
                    wanx_output = gr.Gallery(
                        label="Generated Videos (Click to select)",
                        columns=[2],
                        rows=[2],
                        object_fit="contain",
                        height="auto",
                        show_label=True,
                        elem_id="gallery",
                        allow_preview=True,
                        preview=True
                    )
                    with gr.Accordion("Latent Preview (During Generation)", open=True):
                        wanx_enable_preview = gr.Checkbox(label="Enable Latent Preview", value=True)
                        wanx_preview_steps = gr.Slider(minimum=1, maximum=50, step=1, value=5,
                                                       label="Preview Every N Steps", info="Generates previews during the sampling loop.")
                        wanx_preview_output = gr.Gallery(
                            label="Latent Previews", columns=4, rows=2, object_fit="contain", height=300,
                            allow_preview=True, preview=True, show_label=True, elem_id="wanx_preview_gallery"
                        )                    
                    wanx_send_to_v2v_btn = gr.Button("Send Selected to Hunyuan-v2v")
                    wanx_i2v_send_to_wanx_v2v_btn = gr.Button("Send Selected to WanX-v2v")
                    wanx_send_last_frame_btn = gr.Button("Send Last Frame to Input")
                    wanx_extend_btn = gr.Button("Extend Video")
                    wanx_frames_to_check = gr.Slider(minimum=1, maximum=100, step=1, value=30, 
                                                   label="Frames to Check from End", 
                                                   info="Number of frames from the end to check for sharpness")
                    wanx_send_sharpest_frame_btn = gr.Button("Extract Sharpest Frame")
                    wanx_trim_and_extend_btn = gr.Button("Trim Video & Prepare for Extension")
                    wanx_sharpest_frame_status = gr.Textbox(label="Status", interactive=False)

                # Add a new button for directly extending with the trimmed video
                    wanx_extend_with_trimmed_btn = gr.Button("Extend with Trimmed Video")

                    # Add LoRA section for WanX-i2v similar to other tabs
                    wanx_refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
                    wanx_lora_weights = []
                    wanx_lora_multipliers = []
                    for i in range(4):
                        with gr.Column():
                            wanx_lora_weights.append(gr.Dropdown(
                                label=f"LoRA {i+1}", 
                                choices=get_lora_options(), 
                                value="None", 
                                allow_custom_value=True,
                                interactive=True
                            ))
                            wanx_lora_multipliers.append(gr.Slider(
                                label=f"Multiplier", 
                                minimum=0.0, 
                                maximum=2.0, 
                                step=0.05, 
                                value=1.0
                            ))

            with gr.Row():
                wanx_seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                # Update the wanx_task dropdown choices to include Fun-Control options
                wanx_task = gr.Dropdown(
                    label="Task",
                    choices=["i2v-14B", "i2v-14B-FC", "i2v-14B-FC-1.1", "t2v-14B", "t2v-1.3B", "t2v-14B-FC", "t2v-1.3B-FC", "i2v-1.3B-new"],
                    value="i2v-14B",
                    info="Select model type. *-FC options enable Fun-Control features"
                )
                wanx_dit_folder = gr.Textbox(label="DiT Model Folder", value="wan")
                wanx_dit_path = gr.Dropdown(
                    label="DiT Model",
                    choices=get_dit_models("wan"),  # Use the existing function to get available models
                    value="wan2.1_i2v_720p_14B_fp16.safetensors",
                    allow_custom_value=True,
                    interactive=True
                )
                wanx_vae_path = gr.Textbox(label="VAE Path", value="wan/Wan2.1_VAE.pth")
                wanx_t5_path = gr.Textbox(label="T5 Path", value="wan/models_t5_umt5-xxl-enc-bf16.pth")
                wanx_clip_path = gr.Textbox(label="CLIP Path", value="wan/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
                wanx_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                wanx_save_path = gr.Textbox(label="Save Path", value="outputs")

            with gr.Row():
                wanx_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                wanx_sample_solver = gr.Radio(choices=["unipc", "dpm++", "vanilla"], label="Sample Solver", value="unipc")
                wanx_exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)
                wanx_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                wanx_block_swap = gr.Slider(minimum=0, maximum=39, step=1, label="Block Swap to Save VRAM", value=0)
                
                with gr.Column():
                    wanx_fp8 = gr.Checkbox(label="Use FP8", value=True)
                    wanx_fp8_scaled = gr.Checkbox(label="Use Scaled FP8", value=False, info="For mixing fp16/bf16 and fp8 weights")
                    wanx_fp8_t5 = gr.Checkbox(label="Use FP8 for T5", value=False)

            # Add new row for Skip Layer Guidance options
            with gr.Row():
                wanx_slg_layers = gr.Textbox(label="SLG Layers", value="", placeholder="Comma-separated layer indices, e.g. 1,5,10", info="Layers to skip for guidance")
                wanx_slg_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SLG Start", value=0.0, info="When to start skipping layers (% of total steps)")
                wanx_slg_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SLG End", value=1.0, info="When to stop skipping layers (% of total steps)")    
            
            with gr.Row():
                wanx_enable_cfg_skip = gr.Checkbox(label="Enable CFG Skip (similar to teacache)", value=False)
                with gr.Column(visible=False) as wanx_cfg_skip_options:
                    wanx_cfg_skip_mode = gr.Radio(
                        choices=["early", "late", "middle", "early_late", "alternate", "none"],
                        label="CFG Skip Mode",
                        value="none",
                        info="Controls which steps to apply CFG on"
                    )
                    wanx_cfg_apply_ratio = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                        label="CFG Apply Ratio", 
                        info="Ratio of steps to apply CFG (0.0-1.0). Lower values = faster, but less accurate"
                    )

        #WanX-t2v Tab

        # WanX Text to Video Tab
        with gr.Tab(id=5, label="WanX-t2v") as wanx_t2v_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    wanx_t2v_prompt = gr.Textbox(
                        scale=3, 
                        label="Enter your prompt", 
                        value="A person walking on a beach at sunset", 
                        lines=5
                    )
                    wanx_t2v_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt",
                        value="",
                        lines=3,
                        info="Leave empty to use default negative prompt"
                    )

                with gr.Column(scale=1):
                    wanx_t2v_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    wanx_t2v_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    wanx_t2v_batch_progress = gr.Textbox(label="", visible=True, elem_id="batch_progress")
                    wanx_t2v_progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text")

            with gr.Row():
                wanx_t2v_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                wanx_t2v_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        wanx_t2v_width = gr.Number(label="Width", value=832, interactive=True, info="Should be divisible by 32")
                        wanx_t2v_height = gr.Number(label="Height", value=480, interactive=True, info="Should be divisible by 32")
                        wanx_t2v_recommend_flow_btn = gr.Button("Recommend Flow Shift", size="sm")

                    wanx_t2v_video_length = gr.Slider(minimum=1, maximum=201, step=4, label="Video Length in Frames", value=81)
                    wanx_t2v_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=16)
                    wanx_t2v_infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=20)
                    wanx_t2v_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=5.0, 
                                             info="Recommended: 3.0 for I2V with 480p, 5.0 for others")
                    wanx_t2v_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale", value=5.0)

                with gr.Column():
                    wanx_t2v_output = gr.Gallery(
                        label="Generated Videos (Click to select)",
                        columns=[2],
                        rows=[2],
                        object_fit="contain",
                        height="auto",
                        show_label=True,
                        elem_id="gallery",
                        allow_preview=True,
                        preview=True
                    )
                    with gr.Accordion("Latent Preview (During Generation)", open=False):
                        wanx_t2v_enable_preview = gr.Checkbox(label="Enable Latent Preview", value=False)
                        wanx_t2v_preview_steps = gr.Slider(minimum=1, maximum=50, step=1, value=5,
                                                        label="Preview Every N Steps", info="Generates previews during the sampling loop.")
                        wanx_t2v_preview_output = gr.Gallery(
                            label="Latent Previews", columns=4, rows=2, object_fit="contain", height=300,
                            allow_preview=True, preview=True, show_label=True, elem_id="wanx_t2v_preview_gallery"
                        )                    
                    wanx_t2v_send_to_v2v_btn = gr.Button("Send Selected to Hunyuan v2v")
                    wanx_t2v_send_to_wanx_v2v_btn = gr.Button("Send Selected to WanX-v2v")

                    # Add LoRA section for WanX-t2v
                    wanx_t2v_refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
                    wanx_t2v_lora_weights = []
                    wanx_t2v_lora_multipliers = []
                    for i in range(4):
                        with gr.Column():
                            wanx_t2v_lora_weights.append(gr.Dropdown(
                                label=f"LoRA {i+1}", 
                                choices=get_lora_options(), 
                                value="None", 
                                allow_custom_value=True,
                                interactive=True
                            ))
                            wanx_t2v_lora_multipliers.append(gr.Slider(
                                label=f"Multiplier", 
                                minimum=0.0, 
                                maximum=2.0, 
                                step=0.05, 
                                value=1.0
                            ))

            with gr.Row():
                wanx_t2v_seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                wanx_t2v_task = gr.Dropdown(
                    label="Task",
                    choices=["t2v-1.3B", "t2v-14B", "t2i-14B"],
                    value="t2v-14B",
                    info="Select model size: t2v-1.3B is faster, t2v-14B has higher quality"
                )
                wanx_t2v_dit_path = gr.Dropdown(
                    label="DiT Model",
                    choices=get_dit_models("wan"),
                    value="wan2.1_t2v_14B_fp16.safetensors",
                    allow_custom_value=True,
                    interactive=True
                )
                wanx_t2v_vae_path = gr.Textbox(label="VAE Path", value="wan/Wan2.1_VAE.pth")
                wanx_t2v_t5_path = gr.Textbox(label="T5 Path", value="wan/models_t5_umt5-xxl-enc-bf16.pth")
                wanx_t2v_clip_path = gr.Textbox(label="CLIP Path", visible=False, value="")
                wanx_t2v_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                wanx_t2v_save_path = gr.Textbox(label="Save Path", value="outputs")

            with gr.Row():
                wanx_t2v_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                wanx_t2v_sample_solver = gr.Radio(choices=["unipc", "dpm++", "vanilla"], label="Sample Solver", value="unipc")
                wanx_t2v_exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)
                wanx_t2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                wanx_t2v_block_swap = gr.Slider(minimum=0, maximum=39, step=1, label="Block Swap to Save VRAM", value=0, 
                                         info="Max 39 for 14B model, 29 for 1.3B model")
                
                with gr.Column():
                    wanx_t2v_fp8 = gr.Checkbox(label="Use FP8", value=True)
                    wanx_t2v_fp8_scaled = gr.Checkbox(label="Use Scaled FP8", value=False,
                                                info="For mixing fp16/bf16 and fp8 weights")
                    wanx_t2v_fp8_t5 = gr.Checkbox(label="Use FP8 for T5", value=False)
            
            # Add new row for Skip Layer Guidance options
            with gr.Row():
                wanx_t2v_slg_layers = gr.Textbox(label="SLG Layers", value="", placeholder="Comma-separated layer indices, e.g. 1,5,10", info="Layers to skip for guidance")
                wanx_t2v_slg_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SLG Start", value=0.0, info="When to start skipping layers (% of total steps)")
                wanx_t2v_slg_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SLG End", value=1.0, info="When to stop skipping layers (% of total steps)")
                wanx_t2v_use_random_folder = gr.Checkbox(visible=False, value=False, label="Use Random Images")
                wanx_t2v_input_folder = gr.Textbox(visible=False, value="", label="Image Folder")
                wanx_t2v_input_end = gr.Textbox(visible=False, value="none", label="End Frame")
            
            with gr.Row():
                wanx_t2v_enable_cfg_skip = gr.Checkbox(label="Enable CFG Skip (similar to teacache)", value=False)
                with gr.Column(visible=False) as wanx_t2v_cfg_skip_options:
                    wanx_t2v_cfg_skip_mode = gr.Radio(
                        choices=["early", "late", "middle", "early_late", "alternate", "none"],
                        label="CFG Skip Mode",
                        value="none",
                        info="Controls which steps to apply CFG on"
                    )
                    wanx_t2v_cfg_apply_ratio = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                        label="CFG Apply Ratio", 
                        info="Ratio of steps to apply CFG (0.0-1.0). Lower values = faster, but less accurate"
                    )

        #WanX-v2v Tab
        with gr.Tab(id=6, label="WanX-v2v", visible=False) as wanx_v2v_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    wanx_v2v_prompt = gr.Textbox(
                        scale=3, 
                        label="Enter your prompt", 
                        value="A person walking on a beach at sunset", 
                        lines=5
                    )
                    wanx_v2v_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt",
                        value="",
                        lines=3,
                        info="Leave empty to use default negative prompt"
                    )

                with gr.Column(scale=1):
                    wanx_v2v_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    wanx_v2v_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    wanx_v2v_batch_progress = gr.Textbox(label="", visible=True, elem_id="batch_progress")
                    wanx_v2v_progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text")

            with gr.Row():
                wanx_v2v_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                wanx_v2v_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column():
                    wanx_v2v_input = gr.Video(label="Input Video", format="mp4")
                    wanx_v2v_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.75, label="Denoise Strength", 
                                                info="0 = keep original, 1 = full generation")
                    wanx_v2v_scale_slider = gr.Slider(minimum=1, maximum=200, value=100, step=1, label="Scale %")
                    wanx_v2v_original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=True)

                    # Width and Height Inputs
                    with gr.Row():
                        wanx_v2v_width = gr.Number(label="New Width", value=832, step=32)
                        wanx_v2v_calc_height_btn = gr.Button("→")
                        wanx_v2v_calc_width_btn = gr.Button("←")
                        wanx_v2v_height = gr.Number(label="New Height", value=480, step=32)
                        wanx_v2v_recommend_flow_btn = gr.Button("Recommend Flow Shift", size="sm")

                    wanx_v2v_video_length = gr.Slider(minimum=1, maximum=201, step=4, label="Video Length in Frames", value=81)
                    wanx_v2v_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=16)
                    wanx_v2v_infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=40)
                    wanx_v2v_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=5.0,
                                               info="Recommended: 3.0 for 480p, 5.0 for others")
                    wanx_v2v_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale", value=5.0)

                with gr.Column():
                    wanx_v2v_output = gr.Gallery(
                        label="Generated Videos (Click to select)",
                        columns=[2],
                        rows=[2],
                        object_fit="contain",
                        height="auto",
                        show_label=True,
                        elem_id="gallery",
                        allow_preview=True,
                        preview=True
                    )
                    wanx_v2v_send_to_v2v_btn = gr.Button("Send Selected to Hunyuan-v2v")

                    # Add LoRA section for WanX-v2v
                    wanx_v2v_refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
                    wanx_v2v_lora_weights = []
                    wanx_v2v_lora_multipliers = []
                    for i in range(4):
                        with gr.Column():
                            wanx_v2v_lora_weights.append(gr.Dropdown(
                                label=f"LoRA {i+1}", 
                                choices=get_lora_options(), 
                                value="None", 
                                allow_custom_value=True,
                                interactive=True
                            ))
                            wanx_v2v_lora_multipliers.append(gr.Slider(
                                label=f"Multiplier", 
                                minimum=0.0, 
                                maximum=2.0, 
                                step=0.05, 
                                value=1.0
                            ))

            with gr.Row():
                wanx_v2v_seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                wanx_v2v_task = gr.Dropdown(
                    label="Task",
                    choices=["t2v-14B", "t2v-1.3B"],
                    value="t2v-14B",
                    info="Model size: t2v-1.3B is faster, t2v-14B has higher quality"
                )
                wanx_v2v_dit_folder = gr.Textbox(label="DiT Model Folder", value="wan")
                wanx_v2v_dit_path = gr.Dropdown(
                    label="DiT Model",
                    choices=get_dit_models("wan"),
                    value="wan2.1_t2v_14B_fp16.safetensors",
                    allow_custom_value=True,
                    interactive=True
                )
                wanx_v2v_vae_path = gr.Textbox(label="VAE Path", value="wan/Wan2.1_VAE.pth")
                wanx_v2v_t5_path = gr.Textbox(label="T5 Path", value="wan/models_t5_umt5-xxl-enc-bf16.pth")
                wanx_v2v_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                wanx_v2v_save_path = gr.Textbox(label="Save Path", value="outputs")

            with gr.Row():
                wanx_v2v_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                wanx_v2v_sample_solver = gr.Radio(choices=["unipc", "dpm++", "vanilla"], label="Sample Solver", value="unipc")
                wanx_v2v_exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)
                wanx_v2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                wanx_v2v_block_swap = gr.Slider(minimum=0, maximum=39, step=1, label="Block Swap to Save VRAM", value=0,
                                           info="Max 39 for 14B model, 29 for 1.3B model")

                with gr.Column():
                    wanx_v2v_fp8 = gr.Checkbox(label="Use FP8", value=True)
                    wanx_v2v_fp8_scaled = gr.Checkbox(label="Use Scaled FP8", value=False,
                                                  info="For mixing fp16/bf16 and fp8 weights")
                    wanx_v2v_fp8_t5 = gr.Checkbox(label="Use FP8 for T5", value=False)

            # Add Skip Layer Guidance options
            with gr.Row():
                wanx_v2v_slg_layers = gr.Textbox(label="SLG Layers", value="", placeholder="Comma-separated layer indices, e.g. 1,5,10", info="Layers to skip for guidance")
                wanx_v2v_slg_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SLG Start", value=0.0, info="When to start skipping layers (% of total steps)")
                wanx_v2v_slg_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SLG End", value=1.0, info="When to stop skipping layers (% of total steps)")

            with gr.Row():
                wanx_v2v_enable_cfg_skip = gr.Checkbox(label="Enable CFG Skip (similar to teacache)", value=False)
                with gr.Column(visible=False) as wanx_v2v_cfg_skip_options:
                    wanx_v2v_cfg_skip_mode = gr.Radio(
                        choices=["early", "late", "middle", "early_late", "alternate", "none"],
                        label="CFG Skip Mode",
                        value="none",
                        info="Controls which steps to apply CFG on"
                    )
                    wanx_v2v_cfg_apply_ratio = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                        label="CFG Apply Ratio", 
                        info="Ratio of steps to apply CFG (0.0-1.0). Lower values = faster, but less accurate"
                    )

        #Video Info Tab
        with gr.Tab("Video Info") as video_info_tab:
            with gr.Row():
                video_input = gr.Video(label="Upload Video", interactive=True)
                metadata_output = gr.JSON(label="Generation Parameters")

            with gr.Row():
                send_to_fpe_btn = gr.Button("Send to FramePack-Extension", variant="primary")                
                send_to_t2v_btn = gr.Button("Send to Text2Video", variant="primary")
                send_to_v2v_btn = gr.Button("Send to Video2Video", variant="primary")
                send_to_multitalk_btn = gr.Button("Send to MultiTalk", variant="primary")
            with gr.Row():
                send_to_framepack_btn = gr.Button("Send to FramePack", variant="primary")
                send_to_wan22_btn = gr.Button("Send to Wan2.2", variant="primary")
                send_to_wanx_i2v_btn = gr.Button("Send to WanX-i2v", variant="primary")
                send_to_wanx_t2v_btn = gr.Button("Send to WanX-t2v", variant="primary")
                send_to_wanx_v2v_btn = gr.Button("Send to WanX-v2v", variant="primary")
                


            with gr.Row():
                status = gr.Textbox(label="Status", interactive=False)

        #Convert lora tab        
        with gr.Tab("Convert LoRA") as convert_lora_tab:
            def suggest_output_name(file_obj) -> str:
                """Generate suggested output name from input file"""
                if not file_obj:
                    return ""
                # Get input filename without extension and add MUSUBI
                base_name = os.path.splitext(os.path.basename(file_obj.name))[0]
                return f"{base_name}_MUSUBI"

            def convert_lora(input_file, output_name: str, target_format: str) -> str:
                """Convert LoRA file to specified format"""
                try:
                    if input_file is None:
                        return "Error: No input file selected"

                    # Ensure output directory exists
                    os.makedirs("lora", exist_ok=True)

                    # Construct output path
                    output_path = os.path.join("lora", f"{output_name}.safetensors")

                    # Determine which script to use based on target_format
                    if target_format == "Hunyuan to FramePack":
                        script_name = "convert_hunyuan_to_framepack.py"
                        cmd = [
                            sys.executable,
                            script_name,
                            "--input", input_file.name,
                            "--output", output_path
                        ]
                        print(f"Using '{script_name}' to convert {input_file.name} to {output_path} for FramePack.")
                    else: # Existing logic for "default" and "other"
                        script_name = "convert_lora.py"
                        cmd = [
                            sys.executable,
                            script_name,
                            "--input", input_file.name,
                            "--output", output_path,
                            "--target", target_format.lower()
                        ]

                    print(f"Running conversion command: {' '.join(cmd)}")

                    # Check if the selected script file exists
                    if not os.path.exists(script_name):
                         return f"Error: Conversion script '{script_name}' not found. Please ensure it's in the same directory as h1111.py."

                    # Execute conversion
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True
                    )

                    console_output = result.stdout if result.stdout else ""
                    if result.stderr:
                        console_output += f"\n--- Script STDERR ---\n{result.stderr}"
                    if not console_output.strip():
                        console_output = "Conversion script completed with no output."
                        if os.path.exists(output_path):
                            console_output += f"\n[UI Info] Output file confirmed by h1111.py at: {output_path}"
                        else:
                            console_output += f"\n[UI Warning] Output file NOT found by h1111.py at expected location: {output_path}"               
                    return console_output.strip()
                except subprocess.CalledProcessError as e:
                    error_message = f"Conversion Script Error (Exit Code: {e.returncode}):\n"
                    if e.stdout and e.stdout.strip():
                        error_message += f"--- Script STDOUT ---\n{e.stdout.strip()}\n"
                    if e.stderr and e.stderr.strip():
                        error_message += f"--- Script STDERR ---\n{e.stderr.strip()}\n"
                    if not (e.stdout and e.stdout.strip()) and not (e.stderr and e.stderr.strip()):
                        error_message += "Script produced no output on STDOUT or STDERR."
                    
                    print(f"Subprocess error details logged to console. UI will show combined script output.") # Log for server console
                    return error_message.strip()


            with gr.Row():
                input_file = gr.File(label="Input LoRA File", file_types=[".safetensors"])
                output_name = gr.Textbox(label="Output Name", placeholder="Output filename (without extension)")
                format_radio = gr.Radio(
                    choices=["default", "other", "Hunyuan to FramePack"], # <-- Added new choice here
                    value="default",
                    label="Target Format",
                    info="Choose 'default' for H1111/MUSUBI format, 'other' for diffusion pipe format, or 'Hunyuan to FramePack' for FramePack compatibility."
                )

            with gr.Row():
                convert_btn = gr.Button("Convert LoRA", variant="primary")
                status_output = gr.Textbox(label="Status", interactive=False)

            # Automatically update output name when file is selected
            input_file.change(
                fn=suggest_output_name,
                inputs=[input_file],
                outputs=[output_name]
            )

            # Handle conversion
            convert_btn.click(
                fn=convert_lora,
                inputs=[input_file, output_name, format_radio],
                outputs=status_output
            )
        with gr.Tab("Model Merging") as model_merge_tab:
            with gr.Row():
                with gr.Column():
                    # Model selection
                    dit_model = gr.Dropdown(
                        label="Base DiT Model",
                        choices=["mp_rank_00_model_states.pt"],
                        value="mp_rank_00_model_states.pt",
                        allow_custom_value=True,
                        interactive=True
                    )
                    merge_refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
            with gr.Row():
                with gr.Column():
                    # Output model name
                    output_model = gr.Textbox(label="Output Model Name", value="merged_model.safetensors")
                    exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)
                    merge_btn = gr.Button("Merge Models", variant="primary")
                    merge_status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                # LoRA selection section (similar to Text2Video)
                merge_lora_weights = []
                merge_lora_multipliers = []
                for i in range(4):
                    with gr.Column():
                        merge_lora_weights.append(gr.Dropdown(
                            label=f"LoRA {i+1}",
                            choices=get_lora_options(),
                            value="None",
                            allow_custom_value=True,
                            interactive=True
                        ))
                        merge_lora_multipliers.append(gr.Slider(
                            label=f"Multiplier",
                            minimum=0.0,
                            maximum=2.0,
                            step=0.05,
                            value=1.0
                        ))
                with gr.Row():
                    merge_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                    dit_folder = gr.Textbox(label="DiT Model Folder", value="hunyuan")


        # Wan One Frame Inference Tab
        with gr.Tab(id=13, label="Wan One Frame") as wan_one_frame_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    wan_of_prompt = gr.Textbox(
                        scale=3, 
                        label="Enter your prompt", 
                        value="A cat wearing a chef hat, cooking pizza.", 
                        lines=5
                    )
                    wan_of_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt",
                        value="low quality, blurry, watermark, text, signature, ugly, deformed",
                        lines=3,
                    )
                with gr.Column(scale=1):
                    wan_of_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    wan_of_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)
                with gr.Column(scale=2):
                    wan_of_batch_progress = gr.Textbox(label="Status", interactive=False, value="")
                    wan_of_progress_text = gr.Textbox(label="Progress", interactive=False, value="", elem_id="wan_of_progress_text")

            with gr.Row():
                wan_of_generate_btn = gr.Button("Generate One Frame", elem_classes="green-btn")
                wan_of_stop_btn = gr.Button("Stop Generation", variant="stop")
            
            with gr.Row():
                with gr.Column():
                    wan_of_input_image = gr.Image(label="Control Image (Optional)", type="filepath")
                    wan_of_original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=False)
                    
                    # One Frame Inference Options
                    with gr.Group():
                        gr.Markdown("### One Frame Inference Options")
                        with gr.Row():
                            wan_of_target_index = gr.Number(
                                label="Target Index",
                                value=1,
                                minimum=0,
                                maximum=15,
                                step=1,
                                info="Target frame index for generation"
                            )
                            wan_of_control_index = gr.Textbox(
                                label="Control Index",
                                value="0",
                                info="Control frame indices (e.g., '0' or '0;2')"
                            )
                        wan_of_inference_options = gr.CheckboxGroup(
                            label="One Frame Options",
                            choices=["no_post", "no_2x", "no_4x", "default"],
                            value=["default"],
                            info="One frame inference options"
                        )
                        
                        # Control Images for One Frame Inference
                        with gr.Row():
                            wan_of_control_images = gr.File(
                                label="Control Images (Optional)",
                                file_types=["image"],
                                file_count="multiple"
                            )
                        with gr.Row():
                            wan_of_control_masks = gr.File(
                                label="Control Image Masks (Optional)", 
                                file_types=["image"],
                                file_count="multiple"
                            )
                    
                    gr.Markdown("### Generation Parameters")
                    wan_of_task = gr.Dropdown(
                        label="Task", 
                        choices=["i2v-A14B", "t2v-A14B", "ti2v-5B"], 
                        value="i2v-A14B",
                        info="Selects the model architecture and configuration to use."
                    )
                    wan_of_conditioning_strength = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, 
                        label="Image Conditioning Strength", 
                        value=0.3,
                        info="For T2V with image: 0.0 = preserve original image, 1.0 = ignore image (full generation)",
                        visible=False
                    )
                    # Width and height inputs
                    with gr.Row():
                        wan_of_width = gr.Number(label="Width", value=832, interactive=True)
                        wan_of_calc_height_btn = gr.Button("→")
                        wan_of_calc_width_btn = gr.Button("←")
                        wan_of_height = gr.Number(label="Height", value=480, interactive=True)
                    wan_of_frame_num = gr.Slider(minimum=9, maximum=201, step=4, label="Frame Count", value=81, info="Must be 4n+1")
                    wan_of_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=16)
                    wan_of_sample_steps = gr.Slider(minimum=4, maximum=100, step=1, label="Sampling Steps", value=40)
                    wan_of_flow_shift = gr.Slider(minimum=0.0, maximum=20.0, step=0.1, label="Flow Shift", value=5.0)
                    wan_of_sample_guide_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale", value=3.5)
                    wan_of_timestep_boundary = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Timestep Boundary", value=0.875, visible=True, info="Low noise model used after this threshold (0.875 = 87.5%)")
                    wan_of_sample_solver = gr.Radio(choices=["unipc", "dpm++", "vanilla"], label="Sample Solver", value="unipc")
                    with gr.Row():
                        wan_of_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                        wan_of_random_seed_btn = gr.Button("🎲")

                with gr.Column():
                    wan_of_output = gr.Gallery(
                        label="Generated Images (Click to select)",
                        columns=[2], rows=[2], object_fit="contain", height="auto",
                        show_label=True, elem_id="gallery_wan_of", allow_preview=True, preview=True
                    )
                    with gr.Accordion("LoRA", open=True):
                        with gr.Row():
                            wan_of_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                            wan_of_lora_refresh_btn = gr.Button("🔄 LoRA", elem_classes="refresh-btn")
                        wan_of_lora_weights = []
                        wan_of_lora_multipliers = []
                        wan_of_lora_apply_low = []
                        wan_of_lora_apply_high = []
                        for i in range(4):
                            with gr.Row():
                                wan_of_lora_weights.append(gr.Dropdown(
                                    label=f"LoRA {i+1}", choices=get_lora_options("lora"),
                                    value="None", allow_custom_value=False, interactive=True, scale=2
                                ))
                                wan_of_lora_multipliers.append(gr.Slider(
                                    label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1, interactive=True
                                ))
                            with gr.Row():
                                wan_of_lora_apply_low.append(gr.Checkbox(
                                    label="Apply to Low Noise", value=True, scale=1
                                ))
                                wan_of_lora_apply_high.append(gr.Checkbox(
                                    label="Apply to High Noise", value=False, scale=1
                                ))
            
            with gr.Accordion("Model Paths & Performance", open=True):
                with gr.Row():
                    wan_of_attn_mode = gr.Radio(choices=["sdpa", "flash", "torch", "xformers"], label="Attention Mode", value="sdpa")
                    wan_of_block_swap = gr.Slider(minimum=0, maximum=39, step=1, label="Block Swap to Save VRAM", value=30)
                with gr.Row():
                    wan_of_fp8 = gr.Checkbox(label="Use FP8 (DiT)", value=False)
                    wan_of_fp8_scaled = gr.Checkbox(label="Use Scaled FP8 (DiT)", value=False)
                    wan_of_fp8_t5 = gr.Checkbox(label="Use FP8 for T5", value=False)
                    wan_of_mixed_dtype = gr.Checkbox(label="Mixed Dtype (preserve fp32 weights)", value=False)
                    wan_of_vae_fp32 = gr.Checkbox(
                        label="Use FP32 VAE (higher quality, more VRAM)",
                        value=True,
                    )
                with gr.Row():
                    wan_of_model_folder = gr.Textbox(label="Model Folder", value="wan")
                    wan_of_refresh_models_btn = gr.Button("🔄 Models", elem_classes="refresh-btn")
                with gr.Row():
                    wan_of_dit_low_noise_path = gr.Dropdown(
                        label="DiT Low Noise Model (.safetensors)",
                        choices=get_wan_of_low_noise_models("wan"),
                        value=get_default_low_noise_model("wan"),
                        allow_custom_value=True,
                        interactive=True
                    )
                    wan_of_dit_high_noise_path = gr.Dropdown(
                        label="DiT High Noise Model (.safetensors)",
                        choices=get_wan_of_high_noise_models("wan"),
                        value=get_default_high_noise_model("wan"),
                        allow_custom_value=True,
                        interactive=True
                    )
                with gr.Row():
                    wan_of_clip_path = gr.Dropdown(
                        label="CLIP Model (.pth)",
                        choices=get_wan_of_clip_models("wan"),
                        value=get_default_clip_model("wan"),
                        allow_custom_value=True,
                        interactive=True
                    )
                    wan_of_vae_path = gr.Dropdown(
                        label="VAE Model (.pth)",
                        choices=get_wan_of_vae_models("wan"),
                        value=get_default_vae_model("wan"),
                        allow_custom_value=True,
                        interactive=True
                    )
                with gr.Row():
                    wan_of_t5_path = gr.Dropdown(
                        label="T5 Model (.pth)",
                        choices=get_wan_of_t5_models("wan"),
                        value=get_default_t5_model("wan"),
                        allow_custom_value=True,
                        interactive=True
                    )
                    wan_of_save_path = gr.Textbox(label="Save Path", value="outputs/wan_one_frame")

    #Event handlers etc
    
    # Wan One Frame event handlers
    wan_of_stop_btn.click(fn=lambda: stop_event.set(), queue=False)
    wan_of_random_seed_btn.click(fn=set_random_seed, inputs=None, outputs=[wan_of_seed])
    
    wan_of_generate_btn.click(
        fn=wan_one_frame_handler,
        inputs=[
            wan_of_prompt,
            wan_of_negative_prompt,
            wan_of_input_image,
            wan_of_task,
            wan_of_conditioning_strength,
            wan_of_target_index,
            wan_of_control_index,
            wan_of_inference_options,
            wan_of_control_images,
            wan_of_control_masks,
            wan_of_width,
            wan_of_height,
            wan_of_frame_num,
            wan_of_fps,
            wan_of_seed,
            wan_of_sample_steps,
            wan_of_flow_shift,
            wan_of_sample_guide_scale,
            wan_of_timestep_boundary,
            wan_of_sample_solver,
            wan_of_batch_size,
            wan_of_save_path,
            # Model Paths & Performance
            wan_of_attn_mode,
            wan_of_block_swap,
            wan_of_fp8,
            wan_of_fp8_scaled,
            wan_of_fp8_t5,
            wan_of_mixed_dtype,
            wan_of_vae_fp32,
            wan_of_dit_low_noise_path,
            wan_of_dit_high_noise_path,
            wan_of_clip_path,
            wan_of_vae_path,
            wan_of_t5_path,
            # LoRAs
            wan_of_lora_folder,
            *wan_of_lora_weights,
            *wan_of_lora_multipliers,
            *wan_of_lora_apply_low,
            *wan_of_lora_apply_high
        ],
        outputs=[wan_of_output, wan_of_batch_progress, wan_of_progress_text],
        queue=True
    )

    # LoRA refresh functionality for Wan One Frame tab
    def update_wan_of_lora_dropdowns(lora_folder: str, *current_values) -> List[gr.update]:
        new_choices = get_lora_options(lora_folder)
        weights = current_values[:4]
        multipliers = current_values[4:8]

        results = []
        for i in range(4):
            weight = weights[i] if i < len(weights) else "None"
            multiplier = multipliers[i] if i < len(multipliers) else 1.0
            if weight not in new_choices:
                weight = "None"
            results.extend([
                gr.update(choices=new_choices, value=weight),
                gr.update(value=multiplier)
            ])

        return results

    wan_of_lora_refresh_outputs_list = []
    for i in range(len(wan_of_lora_weights)):
        wan_of_lora_refresh_outputs_list.extend([wan_of_lora_weights[i], wan_of_lora_multipliers[i]])

    wan_of_lora_refresh_btn.click(
        fn=update_wan_of_lora_dropdowns,
        inputs=[wan_of_lora_folder] + wan_of_lora_weights + wan_of_lora_multipliers,
        outputs=wan_of_lora_refresh_outputs_list
    )

    # Model refresh functionality for Wan One Frame tab
    def update_wan_of_model_dropdowns(model_folder: str):
        """Update all model dropdowns based on folder contents"""
        return [
            gr.update(choices=get_wan_of_low_noise_models(model_folder)),
            gr.update(choices=get_wan_of_high_noise_models(model_folder)),
            gr.update(choices=get_wan_of_clip_models(model_folder)),
            gr.update(choices=get_wan_of_vae_models(model_folder)),
            gr.update(choices=get_wan_of_t5_models(model_folder))
        ]

    wan_of_refresh_models_btn.click(
        fn=update_wan_of_model_dropdowns,
        inputs=[wan_of_model_folder],
        outputs=[wan_of_dit_low_noise_path, wan_of_dit_high_noise_path, wan_of_clip_path, wan_of_vae_path, wan_of_t5_path]
    )

    # Model refresh functionality for Wan2.2 tab
    def update_wan22_model_dropdowns(model_folder: str):
        """Update all Wan2.2 model dropdowns based on folder contents"""
        return [
            gr.update(choices=get_wan_of_low_noise_models(model_folder)),
            gr.update(choices=get_wan_of_high_noise_models(model_folder)),
            gr.update(choices=get_wan_of_clip_models(model_folder)),
            gr.update(choices=get_wan_of_dit_models(model_folder)),
            gr.update(choices=get_wan_of_vae_models(model_folder)),
            gr.update(choices=get_wan_of_t5_models(model_folder))
        ]

    wan22_refresh_models_btn.click(
        fn=update_wan22_model_dropdowns,
        inputs=[wan22_model_folder],
        outputs=[wan22_dit_low_noise_path, wan22_dit_high_noise_path, wan22_clip_path, wan22_dit_path, wan22_vae_path, wan22_t5_path]
    )

    # Task-based model path switching for Wan One Frame tab
    def update_wan_of_model_paths_and_settings(task):
        """Update model paths and settings based on selected task"""
        is_a14b = "A14B" in task
        is_i2v_a14b = "i2v-A14B" in task
        is_t2v_a14b = "t2v-A14B" in task
        is_ti2v5b = "ti2v-5B" in task
        
        # Set model paths based on task using task-specific model detection
        if is_i2v_a14b:
            dit_low = get_task_specific_low_noise_model("wan", "i2v")
            dit_high = get_task_specific_high_noise_model("wan", "i2v")
            clip_model = get_default_clip_model("wan")
            clip_visible = True
        elif is_t2v_a14b:
            dit_low = get_task_specific_low_noise_model("wan", "t2v")
            dit_high = get_task_specific_high_noise_model("wan", "t2v")
            clip_model = get_default_clip_model("wan")
            clip_visible = False  # T2V doesn't need CLIP
        elif is_ti2v5b:
            dit_low = "Wan2.2-TI2V-5B_fp16.safetensors"  # Keep specific for ti2v-5B
            dit_high = "Wan2.2-TI2V-5B_fp16.safetensors"  # Same model for both
            clip_model = get_default_clip_model("wan")
            clip_visible = False
        else:
            # Default to i2v
            dit_low = get_task_specific_low_noise_model("wan", "i2v")
            dit_high = get_task_specific_high_noise_model("wan", "i2v")
            clip_model = get_default_clip_model("wan")
            clip_visible = True

        # Show conditioning strength slider for T2V tasks only
        strength_visible = is_t2v_a14b
        
        return [
            gr.update(value=dit_low),
            gr.update(value=dit_high),
            gr.update(value=clip_model, visible=clip_visible),
            gr.update(visible=strength_visible)
        ]

    wan_of_task.change(
        fn=update_wan_of_model_paths_and_settings,
        inputs=[wan_of_task],
        outputs=[wan_of_dit_low_noise_path, wan_of_dit_high_noise_path, wan_of_clip_path, wan_of_conditioning_strength]
    )

#multitalk event handlers
    multitalk_stop_btn.click(fn=lambda: stop_event.set(), queue=False)
    multitalk_random_seed_btn.click(fn=set_random_seed, inputs=None, outputs=[multitalk_seed])
    multitalk_cond_image.change(
        fn=lambda data: (data, data), # Pass the data through to both outputs
        inputs=[multitalk_cond_image],
        outputs=[multitalk_annotations_state, multitalk_bbox_json_output]
    )
    multitalk_generate_btn.click(
        fn=multitalk_batch_handler,
        inputs=[
            multitalk_prompt,
            multitalk_negative_prompt,
            multitalk_annotations_state,
            multitalk_audio_person1,
            multitalk_audio_person2,
            multitalk_batch_size,
            multitalk_size,
            multitalk_mode,
            multitalk_frame_num,
            multitalk_motion_frame,
            multitalk_sample_steps,
            multitalk_sample_shift,
            multitalk_text_guide_scale,
            multitalk_audio_guide_scale,
            multitalk_seed,
            multitalk_audio_type,
            multitalk_num_persistent,
            multitalk_use_teacache,
            multitalk_teacache_thresh,
            multitalk_use_apg,
            multitalk_apg_momentum,
            multitalk_apg_norm_thresh,
            multitalk_use_full_video_preview,
            multitalk_ckpt_dir,
            multitalk_wav2vec_dir,
            multitalk_t5_tokenizer_path,
            multitalk_save_path,
            multitalk_lora_folder,
            *multitalk_lora_weights_ui,
            *multitalk_lora_multipliers_ui
        ],
        outputs=[multitalk_output, multitalk_preview_output, multitalk_status, multitalk_progress],
        queue=True
    )

    multitalk_lora_refresh_outputs_list = []
    for i in range(len(multitalk_lora_weights_ui)):
        multitalk_lora_refresh_outputs_list.extend([multitalk_lora_weights_ui[i], multitalk_lora_multipliers_ui[i]])
    
    multitalk_lora_refresh_btn.click(
        fn=refresh_lora_dropdowns_simple,
        inputs=[multitalk_lora_folder],
        outputs=multitalk_lora_refresh_outputs_list
    )
    
    # InfiniteTalk connections
    infinitetalk_generate_btn.click(
        fn=infinitetalk_batch_handler,
        inputs=[
            infinitetalk_prompt,
            infinitetalk_negative_prompt,
            infinitetalk_cond_image,  # Direct image path instead of video
            infinitetalk_audio_person1,
            infinitetalk_audio_person2,
            infinitetalk_batch_size,
            infinitetalk_size,
            infinitetalk_mode,
            infinitetalk_frame_num,
            infinitetalk_motion_frame,
            infinitetalk_sample_steps,
            infinitetalk_sample_shift,
            infinitetalk_text_guide_scale,
            infinitetalk_audio_guide_scale,
            infinitetalk_seed,
            infinitetalk_audio_type,
            infinitetalk_num_persistent,
            infinitetalk_use_teacache,
            infinitetalk_teacache_thresh,
            infinitetalk_use_apg,
            infinitetalk_apg_momentum,
            infinitetalk_apg_norm_thresh,
            infinitetalk_use_full_video_preview,
            infinitetalk_scene_seg,
            infinitetalk_dit_path,
            infinitetalk_infinitetalk_dir,
            infinitetalk_ckpt_dir,
            infinitetalk_wav2vec_dir,
            infinitetalk_t5_tokenizer_path,
            infinitetalk_save_path,
            infinitetalk_lora_folder,
            *infinitetalk_lora_weights_ui,
            *infinitetalk_lora_multipliers_ui
        ],
        outputs=[infinitetalk_output, infinitetalk_preview_output, infinitetalk_status, infinitetalk_progress],
        queue=True
    )
    
    infinitetalk_stop_btn.click(fn=lambda: stop_event.set(), queue=False)
    
    infinitetalk_random_seed_btn.click(
        fn=set_random_seed,
        inputs=None,
        outputs=[infinitetalk_seed]
    )
    
    infinitetalk_lora_refresh_outputs_list = []
    for i in range(len(infinitetalk_lora_weights_ui)):
        infinitetalk_lora_refresh_outputs_list.extend([infinitetalk_lora_weights_ui[i], infinitetalk_lora_multipliers_ui[i]])
    
    infinitetalk_lora_refresh_btn.click(
        fn=refresh_lora_dropdowns_simple,
        inputs=[infinitetalk_lora_folder],
        outputs=infinitetalk_lora_refresh_outputs_list
    )
# Toggle visibility of End Frame controls and DiT path based on fpe_use_normal_framepack
    def toggle_fpe_normal_framepack_options(use_normal_fp):
        f1_dit_path = "hunyuan/FramePack_F1_I2V_HY_20250503.safetensors"
        normal_fp_dit_path = "hunyuan/FramePackI2V_HY_bf16.safetensors" 
        
        updated_dit_path = normal_fp_dit_path if use_normal_fp else f1_dit_path
        
        # Check if the target path exists and fallback if necessary
        if not os.path.exists(updated_dit_path):
            fallback_path = f1_dit_path if use_normal_fp and os.path.exists(f1_dit_path) else normal_fp_dit_path if not use_normal_fp and os.path.exists(normal_fp_dit_path) else None
            if fallback_path and os.path.exists(fallback_path):
                print(f"Warning: DiT path '{updated_dit_path}' not found. Falling back to '{fallback_path}'.")
                updated_dit_path = fallback_path
            else: # If preferred and fallback are missing, stick to the intended one and let later checks handle it.
                print(f"Warning: DiT path '{updated_dit_path}' not found. No fallback available or fallback also missing.")

        return (
            gr.update(visible=use_normal_fp),  # fpe_end_frame_accordion
            gr.update(visible=not use_normal_fp), # fpe_start_guidance_accordion (NEW)
            gr.update(value=updated_dit_path), # fpe_transformer_path
            gr.update(visible=use_normal_fp)   # fpe_fp8_llm
        )

    fpe_use_normal_framepack.change(
        fn=toggle_fpe_normal_framepack_options,
        inputs=[fpe_use_normal_framepack],
        outputs=[
            fpe_end_frame_accordion, 
            fpe_start_guidance_accordion, # NEW output
            fpe_transformer_path, 
            fpe_fp8_llm
        ]
    )

    fpe_generate_btn.click(
        fn=process_framepack_extension_video,
        inputs=[
            fpe_input_video, fpe_prompt, fpe_negative_prompt, fpe_seed, fpe_batch_count,
            fpe_use_normal_framepack, fpe_end_frame, fpe_end_frame_weight,
            fpe_resolution_max_dim, fpe_total_second_length, fpe_latent_window_size,
            fpe_steps, fpe_cfg_scale, fpe_distilled_guidance_scale,
            fpe_gpu_memory_preservation, fpe_use_teacache, fpe_no_resize, fpe_mp4_crf,
            fpe_num_clean_frames, fpe_vae_batch_size, fpe_save_path,
            # Model Paths
            fpe_transformer_path, fpe_vae_path, fpe_text_encoder_path,
            fpe_text_encoder_2_path, fpe_image_encoder_path,
            # Advanced
            fpe_attn_mode, fpe_fp8_llm, fpe_vae_chunk_size, fpe_vae_spatial_tile_sample_min_size,
            # LoRAs
            fpe_lora_folder,
            fpe_lora_weights_ui[0], fpe_lora_multipliers_ui[0],
            fpe_lora_weights_ui[1], fpe_lora_multipliers_ui[1],
            fpe_lora_weights_ui[2], fpe_lora_multipliers_ui[2],
            fpe_lora_weights_ui[3], fpe_lora_multipliers_ui[3],
            # Preview (UI state, not directly passed to scripts)
            fpe_enable_preview, fpe_preview_interval, 
            fpe_extension_only,
            fpe_start_guidance_image,
            fpe_start_guidance_image_clip_weight,
            fpe_use_guidance_image_as_first_latent,            
        ],
        outputs=[
            fpe_output_gallery,
            fpe_preview_output_component, 
            fpe_batch_progress,
            fpe_progress_text
        ],
        queue=True
    )

    fpe_stop_btn.click(fn=lambda: stop_event.set(), queue=False)
    
    def handle_fpe_gallery_select(evt: gr.SelectData) -> int:
        return evt.index
    fpe_output_gallery.select(fn=handle_fpe_gallery_select, outputs=fpe_selected_index)

    fpe_lora_refresh_outputs_list = []
    for i in range(len(fpe_lora_weights_ui)):
        fpe_lora_refresh_outputs_list.extend([fpe_lora_weights_ui[i], fpe_lora_multipliers_ui[i]])

    fpe_refresh_lora_btn.click(
        fn=refresh_lora_dropdowns_simple,
        inputs=[fpe_lora_folder],
        outputs=fpe_lora_refresh_outputs_list
    )

    def change_to_framepack_tab():
        return gr.Tabs(selected=10) # FramePack tab has id=10

    def handle_send_to_framepack_tab(metadata: dict) -> Tuple[str, dict, str]: # Added str return type for state value
        """Prepare parameters specifically for the FramePack tab."""
        if not metadata:
            # Return default/empty values for status, params, and original_dims state
            return "No parameters to send", {}, ""

        # Extract the value intended for the state here
        original_dims_value = metadata.get("original_dims_str", "")

        # Return status message, the full metadata for params_state, and the specific value for framepack_original_dims state
        return "Parameters ready for FramePack", metadata, original_dims_value

    send_to_framepack_btn.click(
            fn=handle_send_to_framepack_tab,
            inputs=[metadata_output],
            outputs=[status, params_state, framepack_original_dims] # Add framepack_original_dims here
        ).then(
            # This lambda now prepares updates for UI components (32 items)
            lambda params: (
                # Prepare the full list of 32 update values first
                (
                    # Fetch LoRA lists from params, default to empty lists if not found
                    (weights_from_meta := params.get("lora_weights", [])),
                    (mults_from_meta := params.get("lora_multipliers", [])),
                    # Create explicitly padded lists ensuring 4 elements
                    (padded_weights := (weights_from_meta + ["None"] * 4)[:4]),
                    (padded_mults := ([float(m) for m in mults_from_meta] + [1.0] * 4)[:4]), # Ensure multipliers are floats
    
                    # Build the list of update values
                    [
                        params.get("prompt", "cinematic video of a cat wizard casting a spell"),
                        params.get("negative_prompt", ""),
                        # Handle resolution: Prioritize explicit W/H if valid (divisible by 8), else use target_res, else default
                        gr_update(value=int(params["video_width"])) if params.get("video_width") and int(params.get("video_width", 0)) > 0 and int(params.get("video_width", 0)) % 8 == 0 else gr_update(value=None),
                        gr_update(value=int(params["video_height"])) if params.get("video_height") and int(params.get("video_height", 0)) > 0 and int(params.get("video_height", 0)) % 8 == 0 else gr_update(value=None),
                        # Use target resolution only if explicit width/height are *not* validly provided from metadata
                        gr_update(value=int(params.get("target_resolution"))) if not (params.get("video_width") and int(params.get("video_width", 0)) > 0 and int(params.get("video_width", 0)) % 8 == 0) and params.get("target_resolution") else gr_update(value=640),
                        params.get("video_seconds", 5.0),
                        params.get("fps", 30),
                        params.get("seed", -1),
                        params.get("infer_steps", 25),
                        params.get("embedded_cfg_scale", 10.0), # Distilled Guidance
                        params.get("guidance_scale", 1.0),      # CFG
                        params.get("guidance_rescale", 0.0),    # RS
                        params.get("sample_solver", "unipc"),
                        # Unpack the *padded* lists
                        *padded_weights, # 4 items
                        *padded_mults,   # 4 items
                        # Performance/Memory
                        params.get("fp8", False),
                        params.get("fp8_scaled", False),
                        params.get("fp8_llm", False),
                        params.get("blocks_to_swap", 26),
                        params.get("bulk_decode", False),
                        params.get("attn_mode", "sdpa"),
                        params.get("vae_chunk_size", 32),
                        params.get("vae_spatial_tile_sample_min_size", 128),
                        params.get("device", ""),
                        # End Frame Blending Params - Use UI defaults
                        params.get("end_frame_influence", "last"),
                        params.get("end_frame_weight", 0.5),
                        params.get("is_f1", False)
                    ]
                )[-1] # Return the list of values we just built
            ) if params else [gr.update()] * 32, 
            inputs=params_state, # Read parameters from state
            outputs=[
                # Map to FramePack components (UI only - 32 components)
                framepack_prompt,
                framepack_negative_prompt,
                framepack_width, # Will be updated or set to None
                framepack_height, # Will be updated or set to None
                framepack_target_resolution, # Will be updated or set to None/default
                framepack_total_second_length,
                framepack_fps,
                framepack_seed,
                framepack_steps,
                framepack_distilled_guidance_scale,
                framepack_guidance_scale,
                framepack_guidance_rescale,
                framepack_sample_solver,
                # LoRAs (unpacking the lists - 8 components total)
                *framepack_lora_weights, # 4 components
                *framepack_lora_multipliers, # 4 components
                 # Performance/Memory
                framepack_fp8,
                framepack_fp8_scaled,
                framepack_fp8_llm,
                framepack_blocks_to_swap,
                framepack_bulk_decode,
                framepack_attn_mode,
                framepack_vae_chunk_size,
                framepack_vae_spatial_tile_sample_min_size,
                framepack_device,
                # Map to new UI components
                framepack_end_frame_influence,
                framepack_end_frame_weight,
                framepack_is_f1
            ]
        ).then(
            fn=change_to_framepack_tab, # Switch to the FramePack tab
            inputs=None,
            outputs=[tabs]
        )
    # Connect FramePack Generate button
    def update_framepack_image_dimensions(image):
        """Update FramePack dimensions from uploaded image, store raw dims, set default target res"""
        if image is None:
            return "", gr.update(value=None), gr.update(value=None), gr.update(value=640) # Reset W/H, default target res
        try:
            img = Image.open(image)
            w, h = img.size
            original_dims_str = f"{w}x{h}" # Store raw WxH
            target_res_default = 640
            # Return original dims string, clear explicit W/H, set default target res
            return original_dims_str, gr.update(value=None), gr.update(value=None), gr.update(value=target_res_default)
        except Exception as e:
            print(f"Error reading image dimensions: {e}")
            return "", gr.update(value=None), gr.update(value=None), gr.update(value=640) # Fallback
        
    framepack_input_image.change(
        fn=update_framepack_image_dimensions,
        inputs=[framepack_input_image],
        outputs=[framepack_original_dims, framepack_width, framepack_height, framepack_target_resolution]
    )
        
    framepack_prompt.change(fn=count_prompt_tokens, inputs=framepack_prompt, outputs=framepack_token_counter)
    # If explicit width/height is set (and valid), clear target resolution
    def clear_target_res_on_explicit_change(val):
        return gr.update(value=None) if val is not None and val > 0 else gr.update()

    framepack_scale_slider.change(
        fn=update_framepack_from_scale,
        inputs=[framepack_scale_slider, framepack_original_dims],
        outputs=[framepack_width, framepack_height, framepack_target_resolution] # Also clears target res
    )

    framepack_calc_width_btn.click(
        fn=calculate_framepack_width,
        inputs=[framepack_height, framepack_original_dims],
        outputs=[framepack_width]
    ).then(
        fn=clear_target_res_on_explicit_change, # Clear target res if width is manually set
        inputs=[framepack_width],
        outputs=[framepack_target_resolution]
    )

    framepack_calc_height_btn.click(
        fn=calculate_framepack_height,
        inputs=[framepack_width, framepack_original_dims],
        outputs=[framepack_height]
    ).then(
        fn=clear_target_res_on_explicit_change, # Clear target res if height is manually set
        inputs=[framepack_height],
        outputs=[framepack_target_resolution]
    )

    framepack_width.change(
         fn=clear_target_res_on_explicit_change,
         inputs=[framepack_width],
         outputs=[framepack_target_resolution]
    )
    framepack_height.change(
         fn=clear_target_res_on_explicit_change,
         inputs=[framepack_height],
         outputs=[framepack_target_resolution]
    )

    # If target resolution is set (and valid), clear explicit width/height
    def clear_explicit_res_on_target_change(target_res):
        return (gr.update(value=None), gr.update(value=None)) if target_res is not None and target_res > 0 else (gr.update(), gr.update())

    framepack_target_resolution.change(
        fn=clear_explicit_res_on_target_change,
        inputs=[framepack_target_resolution],
        outputs=[framepack_width, framepack_height]
    )
    framepack_use_random_folder.change(
        fn=lambda use_folder_mode: (
            gr.update(visible=use_folder_mode),  # framepack_input_folder_path
            gr.update(visible=use_folder_mode),  # framepack_folder_options_row (which contains validate button and status)
            gr.update(visible=not use_folder_mode)   # framepack_input_image
        ),
        inputs=[framepack_use_random_folder],
        outputs=[framepack_input_folder_path, framepack_folder_options_row, framepack_input_image]
    )

    # Validate folder button handler
    framepack_validate_folder_btn.click(
        fn=lambda folder: get_random_image_from_folder(folder)[1], # Reuse existing helper
        inputs=[framepack_input_folder_path],
        outputs=[framepack_folder_status_text]
    )
    def toggle_f1_model_path(is_f1):
        f1_path = "hunyuan/FramePack_F1_I2V_HY_20250503.safetensors"
        standard_path = "hunyuan/FramePackI2V_HY_bf16.safetensors"
        target_path = f1_path if is_f1 else standard_path

        # Check if the target path exists
        if not os.path.exists(target_path):
             print(f"Warning: F1 model path '{target_path}' not found. Falling back to standard path.")
             # Optionally fall back or just update with the non-existent path
             # Let's fall back to standard if F1 is missing, but keep standard if standard is missing (error handled later)
             if is_f1 and os.path.exists(standard_path):
                 print(f"Falling back to standard path: {standard_path}")
                 return gr.update(value=standard_path)
             elif is_f1:
                 print(f"F1 path missing and standard path also missing. Cannot automatically switch.")
                 # Return the intended (missing) path, error will be caught later
                 return gr.update(value=target_path)
             else: # Standard path is missing
                  print(f"Warning: Standard path '{standard_path}' not found.")
                  return gr.update(value=target_path) # Return the missing standard path

        print(f"Switching DiT path to: {target_path}")
        return gr.update(value=target_path)

    framepack_is_f1.change(
        fn=toggle_f1_model_path,
        inputs=[framepack_is_f1],
        outputs=[framepack_transformer_path]
    )    

    framepack_ui_default_components_ORDERED_LIST = [
        # Top section
        framepack_prompt, framepack_negative_prompt, framepack_batch_size, framepack_is_f1,
        # Input Image / Folder
        framepack_input_image, # Path to image
        framepack_use_random_folder, framepack_input_folder_path,
        # End Frame
        framepack_input_end_frame, # Path to image
        framepack_end_frame_influence, framepack_end_frame_weight,
        # Resolution
        framepack_target_resolution, framepack_scale_slider, framepack_width, framepack_height,
        framepack_original_dims, # This is a gr.State
        # Video params
        framepack_total_second_length, framepack_video_sections, framepack_fps, framepack_seed, framepack_steps,
        # Generation params
        framepack_distilled_guidance_scale, framepack_guidance_scale, framepack_guidance_rescale,
        framepack_latent_window_size, framepack_sample_solver,
        # LoRAs
        framepack_lora_folder,
        *framepack_lora_weights,  # This unpacks the list of 4 Dropdown components
        *framepack_lora_multipliers, # This unpacks the list of 4 Slider components
        # Advanced Section Control (Section definitions themselves - 4 sets)
        *framepack_secs, # 4 Textbox components
        *framepack_sec_prompts, # 4 Textbox components
        *framepack_sec_images, # 4 Image components (paths)
        # Performance / Memory
        framepack_fp8, framepack_fp8_llm, framepack_fp8_scaled, framepack_blocks_to_swap,
        framepack_bulk_decode, framepack_attn_mode, framepack_vae_chunk_size,
        framepack_vae_spatial_tile_sample_min_size, framepack_device,
        framepack_use_teacache, framepack_teacache_steps, framepack_teacache_thresh,
        # Model Paths
        framepack_transformer_path, framepack_vae_path, framepack_text_encoder_path,
        framepack_text_encoder_2_path, framepack_image_encoder_path, framepack_save_path,
        # Preview (only settings, not the output component itself)
        framepack_enable_preview, framepack_use_full_video_preview, framepack_preview_every_n_sections,
    ]
    framepack_ui_default_keys = [
        "framepack_prompt", "framepack_negative_prompt", "framepack_batch_size", "framepack_is_f1",
        "framepack_input_image",
        "framepack_use_random_folder", "framepack_input_folder_path",
        "framepack_input_end_frame",
        "framepack_end_frame_influence", "framepack_end_frame_weight",
        "framepack_target_resolution", "framepack_scale_slider", "framepack_width", "framepack_height",
        "framepack_original_dims_state", # Key for the gr.State
        "framepack_total_second_length", "framepack_video_sections", "framepack_fps", "framepack_seed", "framepack_steps",
        "framepack_distilled_guidance_scale", "framepack_guidance_scale", "framepack_guidance_rescale",
        "framepack_latent_window_size", "framepack_sample_solver",
        "framepack_lora_folder",
        *[f"framepack_lora_weight_{i+1}" for i in range(4)],
        *[f"framepack_lora_multiplier_{i+1}" for i in range(4)],
        *[f"framepack_sec_{i+1}" for i in range(4)],
        *[f"framepack_sec_prompt_{i+1}" for i in range(4)],
        *[f"framepack_sec_image_{i+1}" for i in range(4)],
        "framepack_fp8", "framepack_fp8_llm", "framepack_fp8_scaled", "framepack_blocks_to_swap",
        "framepack_bulk_decode", "framepack_attn_mode", "framepack_vae_chunk_size",
        "framepack_vae_spatial_tile_sample_min_size", "framepack_device",
        "framepack_use_teacache", "framepack_teacache_steps", "framepack_teacache_thresh",
        "framepack_transformer_path", "framepack_vae_path", "framepack_text_encoder_path",
        "framepack_text_encoder_2_path", "framepack_image_encoder_path", "framepack_save_path",
        "framepack_enable_preview", "framepack_use_full_video_preview", "framepack_preview_every_n_sections",
    ]
    # Ensure the lengths match, critical for zipping
    if len(framepack_ui_default_components_ORDERED_LIST) != len(framepack_ui_default_keys):
        raise ValueError(f"FramePack Mismatch: {len(framepack_ui_default_components_ORDERED_LIST)} components, {len(framepack_ui_default_keys)} keys. This is a developer error.")

    def save_framepack_defaults(*values):
        os.makedirs(UI_CONFIGS_DIR, exist_ok=True)
        settings_to_save = {}
        for i, key in enumerate(framepack_ui_default_keys):
            settings_to_save[key] = values[i]

        try:
            with open(FRAMEPROK_DEFAULTS_FILE, 'w') as f:
                json.dump(settings_to_save, f, indent=2)
            return "FramePack defaults saved successfully."
        except Exception as e:
            return f"Error saving FramePack defaults: {e}"

    def load_framepack_defaults(request: gr.Request): # request can be used to see if it's initial load
        if not os.path.exists(FRAMEPROK_DEFAULTS_FILE):
            if request: # If it's not an initial load (i.e., button click)
                return [gr.update()] * len(framepack_ui_default_keys) + ["No defaults file found."]
            else: # Initial load, let Gradio use its defined values
                return [gr.update()] * len(framepack_ui_default_keys) + [""]


        try:
            with open(FRAMEPROK_DEFAULTS_FILE, 'r') as f:
                loaded_settings = json.load(f)
        except Exception as e:
            return [gr.update()] * len(framepack_ui_default_keys) + [f"Error loading defaults: {e}"]

        updates = []
        for i, key in enumerate(framepack_ui_default_keys):
            component = framepack_ui_default_components_ORDERED_LIST[i]
            default_value_from_component = None
            # Try to get the 'value' attribute from the component definition
            if hasattr(component, 'value'):
                default_value_from_component = component.value
            elif isinstance(component, gr.State): # gr.State has value in its init
                 pass # For gr.State, default is often None or handled by its initial value

            # Use loaded setting if available, otherwise the component's initial value
            value_to_set = loaded_settings.get(key, default_value_from_component)

            # Special handling for LoRA dropdowns to ensure 'None' is a valid choice
            if "lora_weight" in key: # This covers framepack_lora_weight_1 to framepack_lora_weight_4
                lora_choices = get_lora_options(loaded_settings.get("framepack_lora_folder", "lora"))
                if value_to_set not in lora_choices:
                    value_to_set = "None" # Default to "None" if saved value is invalid
                updates.append(gr.update(choices=lora_choices, value=value_to_set))
            else:
                updates.append(gr.update(value=value_to_set))

        return updates + ["FramePack defaults loaded successfully."]
    framepack_save_defaults_btn.click(
        fn=save_framepack_defaults,
        inputs=framepack_ui_default_components_ORDERED_LIST, # Pass all components
        outputs=[framepack_defaults_status]
    )
    framepack_load_defaults_btn.click(
        fn=load_framepack_defaults,
        inputs=None, # No direct inputs, reads from file
        outputs=framepack_ui_default_components_ORDERED_LIST + [framepack_defaults_status]
    )
    def initial_load_framepack_defaults():
        # Call load_framepack_defaults without the request, or handle None request inside
        results_and_status = load_framepack_defaults(None) # Pass None for request
        return results_and_status[:-1]
    demo.load(
        fn=initial_load_framepack_defaults,
        inputs=None,
        outputs=framepack_ui_default_components_ORDERED_LIST # Outputs for demo.load
    )
    framepack_generate_btn.click(
        fn=process_framepack_video,
        inputs=[
            framepack_prompt, framepack_negative_prompt, framepack_input_image,
            framepack_input_end_frame, framepack_end_frame_influence, framepack_end_frame_weight,
            framepack_transformer_path, framepack_vae_path, framepack_text_encoder_path,
            framepack_text_encoder_2_path, framepack_image_encoder_path,
            framepack_target_resolution, framepack_width, framepack_height, framepack_original_dims,
            framepack_total_second_length, framepack_video_sections, framepack_fps, framepack_seed, framepack_steps,
            framepack_distilled_guidance_scale, framepack_guidance_scale, framepack_guidance_rescale,
            framepack_sample_solver, framepack_latent_window_size,
            framepack_fp8, framepack_fp8_scaled, framepack_fp8_llm,
            framepack_blocks_to_swap, framepack_bulk_decode, framepack_attn_mode,
            framepack_vae_chunk_size, framepack_vae_spatial_tile_sample_min_size,
            framepack_device,
            framepack_use_teacache,
            framepack_teacache_steps,
            framepack_teacache_thresh,
            framepack_batch_size, framepack_save_path,
            framepack_lora_folder,
            framepack_enable_preview,
            framepack_preview_every_n_sections,
            framepack_use_full_video_preview,
            framepack_is_f1,
            framepack_use_random_folder,
            framepack_input_folder_path,
            *framepack_secs, *framepack_sec_prompts, *framepack_sec_images,
            *framepack_lora_weights, *framepack_lora_multipliers
        ],
        outputs=[
            framepack_output,           # Main gallery
            framepack_preview_output,   # Preview video player
            framepack_batch_progress,   # Status text
            framepack_progress_text     # Progress text
        ],
        queue=True
    )

    framepack_random_seed.click(
        fn=set_random_seed,
        inputs=None, 
        outputs=[framepack_seed] 
    )
    # Connect FramePack Stop button
    framepack_stop_btn.click(fn=lambda: stop_event.set(), queue=False)

    # Connect FramePack Gallery selection
    def handle_framepack_gallery_select(evt: gr.SelectData) -> int:
        return evt.index

    framepack_output.select(
        fn=handle_framepack_gallery_select,
        outputs=framepack_selected_index
    )
    
    # FramePack LoRA Refresh Button Handler
    framepack_lora_refresh_outputs = []
    for i in range(len(framepack_lora_weights)):
        framepack_lora_refresh_outputs.extend([framepack_lora_weights[i], framepack_lora_multipliers[i]])

    framepack_refresh_lora_btn.click(
        fn=refresh_lora_dropdowns_simple, # Use the new simplified function
        inputs=[framepack_lora_folder],   # Only needs the folder path as input
        outputs=framepack_lora_refresh_outputs # Still outputs updates to all 8 components
    )
    def trigger_skip():
        """Sets the skip event and returns a status message."""
        print("FramePack Skip button clicked, setting skip_event.")
        skip_event.set()
        return "Skip signal sent..."

    framepack_skip_btn.click(
        fn=trigger_skip,
        inputs=None,
        outputs=[framepack_batch_progress], # Update status text
        queue=False # Send signal immediately
    )    

    def toggle_fun_control(use_fun_control):
        """Toggle control video visibility and update task suffix"""
        # Only update visibility, don't try to set paths
        return gr.update(visible=use_fun_control)

    def update_task_for_funcontrol(use_fun_control, current_task):
        """Add or remove -FC suffix from task based on checkbox"""
        if use_fun_control:
            if not current_task.endswith("-FC"):
                if "i2v" in current_task:
                    return "i2v-14B-FC"
                elif "t2v" in current_task:
                    return "t2v-14B-FC"
            return current_task
        else:
            if current_task.endswith("-FC"):
                return current_task.replace("-FC", "")
            return current_task

    wanx_use_fun_control.change(
        fn=lambda x: (gr.update(visible=x), gr.update(visible=x), gr.update(visible=x), gr.update(visible=x)),
        inputs=[wanx_use_fun_control],
        outputs=[wanx_control_video, wanx_control_strength, wanx_control_start, wanx_control_end]
    )

    # Make task change update checkbox state
    def update_from_task(task):
        """Update Fun-Control checkbox and control video visibility based on task"""
        is_fun_control = "-FC" in task
        return gr.update(value=is_fun_control), gr.update(visible=is_fun_control)

    wanx_task.change(
        fn=update_from_task,
        inputs=[wanx_task],
        outputs=[wanx_use_fun_control, wanx_control_video]
    )
    wanx_enable_cfg_skip.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[wanx_enable_cfg_skip],
        outputs=[wanx_cfg_skip_options]
    )

    wanx_t2v_enable_cfg_skip.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[wanx_t2v_enable_cfg_skip],
        outputs=[wanx_t2v_cfg_skip_options]
    )

    wanx_v2v_enable_cfg_skip.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[wanx_v2v_enable_cfg_skip],
        outputs=[wanx_v2v_cfg_skip_options]
    )

    #WanX-v2v tab functions
    wanx_v2v_prompt.change(fn=count_prompt_tokens, inputs=wanx_v2v_prompt, outputs=wanx_v2v_token_counter)

    # Stop button handler
    wanx_v2v_stop_btn.click(fn=lambda: stop_event.set(), queue=False)

    # Video input handling
    wanx_v2v_input.change(
        fn=update_wanx_v2v_dimensions,
        inputs=[wanx_v2v_input],
        outputs=[wanx_v2v_original_dims, wanx_v2v_width, wanx_v2v_height]
    )

    # Flow shift recommendation button
    wanx_v2v_recommend_flow_btn.click(
        fn=recommend_wanx_flow_shift,
        inputs=[wanx_v2v_width, wanx_v2v_height],
        outputs=[wanx_v2v_flow_shift]
    )

    # Width/height calculation buttons
    wanx_v2v_calc_width_btn.click(
        fn=calculate_wanx_width,  # Reuse function from WanX tabs
        inputs=[wanx_v2v_height, wanx_v2v_original_dims],
        outputs=[wanx_v2v_width]
    )

    wanx_v2v_calc_height_btn.click(
        fn=calculate_wanx_height,  # Reuse function from WanX tabs
        inputs=[wanx_v2v_width, wanx_v2v_original_dims],
        outputs=[wanx_v2v_height]
    )

    # Scale slider handling for adjusting dimensions
    wanx_v2v_scale_slider.change(
        fn=update_wanx_from_scale,  # Reuse function from WanX tabs
        inputs=[wanx_v2v_scale_slider, wanx_v2v_original_dims],
        outputs=[wanx_v2v_width, wanx_v2v_height]
    )

    def change_to_wanx_v2v_tab():
        return gr.Tabs(selected=6) 

    def send_wanx_t2v_to_v2v_input(gallery, selected_index):
        """Send the selected WanX-t2v video to WanX-v2v input"""
        if gallery is None or not gallery:
            return None, None

        if selected_index is None and len(gallery) == 1:
            selected_index = 0

        if selected_index is None or selected_index >= len(gallery):
            return None, None

        # Get the video path
        item = gallery[selected_index]
        video_path = parse_video_path(item)

        return video_path, "Video sent from WanX-t2v tab"
    
    wanx_t2v_send_to_wanx_v2v_btn.click(
        fn=send_wanx_t2v_to_v2v_input,
        inputs=[wanx_t2v_output, wanx_t2v_selected_index],
        outputs=[wanx_v2v_input, wanx_v2v_batch_progress]
    ).then(
        fn=lambda prompt: prompt,
        inputs=[wanx_t2v_prompt],
        outputs=[wanx_v2v_prompt]
    ).then(
        fn=change_to_wanx_v2v_tab,
        inputs=None,
        outputs=[tabs]
    )

    # Send video from WanX-i2v to WanX-v2v
    wanx_i2v_send_to_wanx_v2v_btn.click(
        fn=send_wanx_t2v_to_v2v_input,  # Reuse the same function
        inputs=[wanx_output, wanx_i2v_selected_index],
        outputs=[wanx_v2v_input, wanx_v2v_batch_progress]
    ).then(
        fn=lambda prompt: prompt,
        inputs=[wanx_prompt],
        outputs=[wanx_v2v_prompt]
    ).then(
        fn=change_to_wanx_v2v_tab,
        inputs=None,
        outputs=[tabs]
    )

    # Update model paths when task changes
    def update_model_paths_for_task(task):
        if "1.3B" in task:
            return gr.update(value="wan/wan2.1_t2v_1.3B_fp16.safetensors")
        else:
            return gr.update(value="wan/wan2.1_t2v_14B_fp16.safetensors")

    wanx_v2v_task.change(
        fn=update_model_paths_for_task,
        inputs=[wanx_v2v_task],
        outputs=[wanx_v2v_dit_path]
    )

    # Generate button handler
    wanx_v2v_generate_btn.click(
        fn=wanx_v2v_batch_handler,
        inputs=[
            wanx_v2v_prompt,
            wanx_v2v_negative_prompt,
            wanx_v2v_input,
            wanx_v2v_width,
            wanx_v2v_height,
            wanx_v2v_video_length,
            wanx_v2v_fps,
            wanx_v2v_infer_steps,
            wanx_v2v_flow_shift,
            wanx_v2v_guidance_scale,
            wanx_v2v_strength,
            wanx_v2v_seed,
            wanx_v2v_batch_size,
            wanx_v2v_task,
            wanx_v2v_dit_folder,
            wanx_v2v_dit_path,
            wanx_v2v_vae_path,
            wanx_v2v_t5_path,
            wanx_v2v_save_path,
            wanx_v2v_output_type,
            wanx_v2v_sample_solver,
            wanx_v2v_exclude_single_blocks,
            wanx_v2v_attn_mode,
            wanx_v2v_block_swap,
            wanx_v2v_fp8,
            wanx_v2v_fp8_scaled,
            wanx_v2v_fp8_t5,
            wanx_v2v_lora_folder,
            wanx_v2v_slg_layers,
            wanx_v2v_slg_start,
            wanx_v2v_slg_end,
            wanx_v2v_enable_cfg_skip,
            wanx_v2v_cfg_skip_mode,
            wanx_v2v_cfg_apply_ratio,
            *wanx_v2v_lora_weights,
            *wanx_v2v_lora_multipliers
        ],
        outputs=[wanx_v2v_output, wanx_v2v_batch_progress, wanx_v2v_progress_text],
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[wanx_v2v_batch_size],
        outputs=wanx_v2v_selected_index
    )

    # Gallery selection handling
    wanx_v2v_output.select(
        fn=handle_wanx_v2v_gallery_select,
        outputs=wanx_v2v_selected_index
    )
    def change_to_tab_two():
        return gr.Tabs(selected=2)

    # Send to Hunyuan v2v tab
    wanx_v2v_send_to_v2v_btn.click(
        fn=send_wanx_v2v_to_hunyuan_v2v,
        inputs=[
            wanx_v2v_output,
            wanx_v2v_prompt,
            wanx_v2v_selected_index,
            wanx_v2v_width,
            wanx_v2v_height,
            wanx_v2v_video_length,
            wanx_v2v_fps,
            wanx_v2v_infer_steps,
            wanx_v2v_seed,
            wanx_v2v_flow_shift,
            wanx_v2v_guidance_scale,
            wanx_v2v_negative_prompt
        ],
        outputs=[
            v2v_input,
            v2v_prompt,
            v2v_width,
            v2v_height,
            v2v_video_length,
            v2v_fps,
            v2v_infer_steps,
            v2v_seed,
            v2v_flow_shift,
            v2v_cfg_scale,
            v2v_negative_prompt
        ]
    ).then(
        fn=change_to_tab_two,
        inputs=None,
        outputs=[tabs]
    )

    # Add refresh button handler for WanX-v2v tab
    wanx_v2v_refresh_outputs = [wanx_v2v_dit_path]  # This is one output
    for i in range(4):
        wanx_v2v_refresh_outputs.extend([wanx_v2v_lora_weights[i], wanx_v2v_lora_multipliers[i]])  # This adds 8 more outputs

    wanx_v2v_refresh_btn.click(
        fn=update_dit_and_lora_dropdowns,  # We need to use this function instead
        inputs=[wanx_v2v_dit_folder, wanx_v2v_lora_folder, wanx_v2v_dit_path] + wanx_v2v_lora_weights + wanx_v2v_lora_multipliers,
        outputs=wanx_v2v_refresh_outputs
    )

    # Add function to send videos from Video Info tab to WanX-v2v
    def send_to_wanx_v2v(metadata: dict, video_path: str) -> Tuple[str, Dict, str]:
        """Handle both parameters and video transfer from Video Info to WanX-v2v tab with debugging"""
        if not video_path:
            return "No video selected", {}, None

        # Print debug information
        print(f"VIDEO INFO TO WANX-V2V TRANSFER:")
        print(f"Original metadata: {metadata}")
        print(f"Video path: {video_path}")

        # Special handling for WanX-v2v prompt fields
        # Create a copy of metadata with explicit prompt fields
        enhanced_metadata = metadata.copy()
        if "prompt" in metadata:
            enhanced_metadata["wanx_v2v_prompt"] = metadata["prompt"]
        if "negative_prompt" in metadata:
            enhanced_metadata["wanx_v2v_negative_prompt"] = metadata["negative_prompt"]

        print(f"Enhanced metadata: {enhanced_metadata}")

        status_msg, params = send_parameters_to_tab(enhanced_metadata, "wanx_v2v")
        print(f"Mapped parameters: {params}")

        return f"Parameters ready for WanX-v2v (DEBUG INFO IN CONSOLE)", enhanced_metadata, video_path

    # Then, implement a proper handler to change to the WanX-v2v tab
    def change_to_wanx_v2v_tab():
        return gr.Tabs(selected=6)  # WanX-v2v is tab index 6

    # Next, connect the button to the functions with proper parameter mapping
    send_to_wanx_v2v_btn.click(
        fn=lambda m, v: handle_send_to_wanx_tab(m, 'wanx_v2v', v),
        inputs=[metadata_output, video_input],
        outputs=[status, params_state, wanx_v2v_input]
    ).then(
        lambda params: [
            params.get("prompt", ""),
            params.get("width", 832),
            params.get("height", 480),
            params.get("video_length", 81),
            params.get("fps", 16),
            params.get("infer_steps", 40),
            params.get("seed", -1),
            params.get("flow_shift", 5.0),
            params.get("guidance_scale", 5.0),
            params.get("attn_mode", "sdpa"),
            params.get("block_swap", 0),
            params.get("negative_prompt", ""),
            params.get("strength", 0.75),
            *[params.get("lora_weights", ["None"]*4)[i] if isinstance(params.get("lora_weights", []), list) and i < len(params.get("lora_weights", [])) else "None" for i in range(4)],
            *[params.get("lora_multipliers", [1.0]*4)[i] if isinstance(params.get("lora_multipliers", []), list) and i < len(params.get("lora_multipliers", [])) else 1.0 for i in range(4)]
        ] if params else [gr.update()]*21,
        inputs=params_state,
        outputs=[
            wanx_v2v_prompt,
            wanx_v2v_width,
            wanx_v2v_height,
            wanx_v2v_video_length,
            wanx_v2v_fps,
            wanx_v2v_infer_steps,
            wanx_v2v_seed,
            wanx_v2v_flow_shift,
            wanx_v2v_guidance_scale,
            wanx_v2v_attn_mode,
            wanx_v2v_block_swap,
            wanx_v2v_negative_prompt,
            wanx_v2v_strength,
            *wanx_v2v_lora_weights,
            *wanx_v2v_lora_multipliers
        ]
    ).then(
        fn=change_to_wanx_v2v_tab, inputs=None, outputs=[tabs]
    )


    def change_to_multitalk_tab():
        return gr.Tabs(selected=8) # MultiTalk tab has id=8

    send_to_multitalk_btn.click(
        fn=lambda m: ("Parameters ready for MultiTalk", m),
        inputs=[metadata_output],
        outputs=[status, params_state]
    ).then(
        lambda params: (
            (
                # LoRA handling
                (weights_from_meta := params.get("lora_weights", [])),
                (mults_from_meta := params.get("lora_multipliers", [])),
                (padded_weights := (weights_from_meta + ["None"] * 4)[:4]),
                (padded_mults := ([float(m) if isinstance(m, (int, float, str)) and str(m).replace('.', '', 1).isdigit() else 1.0 for m in mults_from_meta] + [1.0] * 4)[:4]),

                [
                    params.get("prompt", "A conversation between two female bunny anchors in a studio in Ghangzhou."),
                    params.get("negative_prompt", "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards."),
                    *padded_weights,
                    *padded_mults,
                ]
            )[-1] # Return the list of values
        ) if params else [gr.update()] * 10, # 2 prompts + 4 loras + 4 mults
        inputs=params_state,
        outputs=[
            multitalk_prompt,
            multitalk_negative_prompt,
            *multitalk_lora_weights_ui,
            *multitalk_lora_multipliers_ui
        ]
    ).then(
        fn=change_to_multitalk_tab,
        inputs=None,
        outputs=[tabs]
    )

    #Video Extension
    wanx_send_last_frame_btn.click(
        fn=send_last_frame_handler,
        inputs=[wanx_output, wanx_i2v_selected_index],
        outputs=[wanx_input, wanx_base_video]
    )

    wanx_extend_btn.click(
        fn=prepare_for_batch_extension,
        inputs=[wanx_input, wanx_base_video, wanx_batch_size],
        outputs=[wanx_input, wanx_base_video, wanx_batch_size, wanx_batch_progress, wanx_progress_text]
    ).then(
        fn=lambda batch_size, base_video:
            "Starting batch extension..." if base_video and batch_size > 0 else
            "Error: Missing base video or invalid batch size",
        inputs=[wanx_batch_size, wanx_base_video],
        outputs=[wanx_batch_progress]
    ).then(
        # Process batch extension one at a time
        fn=process_batch_extension,
        inputs=[
            wanx_prompt,
            wanx_negative_prompt,
            wanx_input,               # Input image (last frame)
            wanx_base_video,          # Base video to extend
            wanx_width,
            wanx_height,
            wanx_video_length,
            wanx_fps,
            wanx_infer_steps,
            wanx_flow_shift,
            wanx_guidance_scale,
            wanx_seed,
            wanx_batch_size,
            wanx_task,
            wanx_dit_folder,          # <<< Pass the folder path
            wanx_dit_path,            # <<< Pass the model filename
            wanx_vae_path,
            wanx_t5_path,
            wanx_clip_path,
            wanx_save_path,
            wanx_output_type,
            wanx_sample_solver,
            wanx_exclude_single_blocks,
            wanx_attn_mode,
            wanx_block_swap,
            wanx_fp8,
            wanx_fp8_scaled,
            wanx_fp8_t5,
            wanx_lora_folder,
            wanx_slg_layers,
            wanx_slg_start,
            wanx_slg_end,
            # Pass LoRA weights and multipliers individually
            wanx_lora_weights[0],
            wanx_lora_weights[1],
            wanx_lora_weights[2],
            wanx_lora_weights[3],
            wanx_lora_multipliers[0],
            wanx_lora_multipliers[1],
            wanx_lora_multipliers[2],
            wanx_lora_multipliers[3]
        ],
        outputs=[wanx_output, wanx_batch_progress, wanx_progress_text]
    )

    # Extract and send sharpest frame to input
    wanx_send_sharpest_frame_btn.click(
        fn=send_sharpest_frame_handler,
        inputs=[wanx_output, wanx_i2v_selected_index, wanx_frames_to_check],
        outputs=[wanx_input, wanx_base_video, wanx_sharpest_frame_number, wanx_sharpest_frame_status]
    )

    # Trim video to sharpest frame and prepare for extension
    wanx_trim_and_extend_btn.click(
        fn=trim_and_prepare_for_extension,
        inputs=[wanx_base_video, wanx_sharpest_frame_number, wanx_save_path],
        outputs=[wanx_trimmed_video_path, wanx_sharpest_frame_status]
    ).then(
        fn=lambda path, status: (path, status if "Failed" in status else "Video trimmed successfully and ready for extension"),
        inputs=[wanx_trimmed_video_path, wanx_sharpest_frame_status],
        outputs=[wanx_base_video, wanx_sharpest_frame_status]
    )

    wanx_extend_with_trimmed_btn.click(
        # Prepare step: Sets the base video to the trimmed video path
        fn=prepare_for_batch_extension,
        inputs=[wanx_input, wanx_trimmed_video_path, wanx_batch_size], # Use trimmed video path here
        outputs=[wanx_input, wanx_base_video, wanx_batch_size, wanx_batch_progress, wanx_progress_text] # Update base_video state
    ).then(
        # Actual extension processing step
        fn=process_batch_extension,
        inputs=[
            wanx_prompt,
            wanx_negative_prompt,
            wanx_input,               # Input image (sharpest frame)
            wanx_trimmed_video_path,  # Base video to extend (the trimmed one)
            wanx_width,
            wanx_height,
            wanx_video_length,
            wanx_fps,
            wanx_infer_steps,
            wanx_flow_shift,
            wanx_guidance_scale,
            wanx_seed,
            wanx_batch_size,
            wanx_task,
            wanx_dit_folder,          # <<< Pass the folder path
            wanx_dit_path,            # <<< Pass the model filename
            wanx_vae_path,
            wanx_t5_path,
            wanx_clip_path,
            wanx_save_path,
            wanx_output_type,
            wanx_sample_solver,
            wanx_exclude_single_blocks,
            wanx_attn_mode,
            wanx_block_swap,
            wanx_fp8,
            wanx_fp8_scaled,
            wanx_fp8_t5,
            wanx_lora_folder,
            wanx_slg_layers,
            wanx_slg_start,
            wanx_slg_end,
            # Pass LoRA weights and multipliers individually
            wanx_lora_weights[0],
            wanx_lora_weights[1],
            wanx_lora_weights[2],
            wanx_lora_weights[3],
            wanx_lora_multipliers[0],
            wanx_lora_multipliers[1],
            wanx_lora_multipliers[2],
            wanx_lora_multipliers[3]
        ],
        outputs=[wanx_output, wanx_batch_progress, wanx_progress_text]
    )

    #Video Info
    def handle_send_to_wanx_tab(metadata, target_tab, video_path=None):
        """Common handler for sending video parameters to WanX tabs"""
        if not metadata:
            return "No parameters to send", {}, None  # Return three values
    
        # Tab names for clearer messages
        tab_names = {
            'wanx_i2v': 'WanX-i2v',
            'wanx_t2v': 'WanX-t2v',
            'wanx_v2v': 'WanX-v2v'
        }
    
        # Just pass through all parameters - we'll use them in the .then() function
        return f"Parameters ready for {tab_names.get(target_tab, target_tab)}", metadata, video_path

    def change_to_wanx_i2v_tab():
        return gr.Tabs(selected=4)  # WanX-i2v tab index

    def change_to_wanx_t2v_tab():
        return gr.Tabs(selected=5)  # WanX-t2v tab index
    
    def change_to_wan22_tab():
        return gr.Tabs(selected=12)  # Wan2.2 tab index


    send_to_wanx_i2v_btn.click(
        fn=lambda m: ("Parameters ready for WanX-i2v", m),
        inputs=[metadata_output],
        outputs=[status, params_state]
    ).then(
        # Reusing the same pattern as other tab transfers with LoRA handling
        lambda params: [
            params.get("prompt", ""),
            params.get("width", 832),
            params.get("height", 480),
            params.get("video_length", 81),
            params.get("fps", 16),
            params.get("infer_steps", 40),
            params.get("seed", -1),
            params.get("flow_shift", 3.0),
            params.get("guidance_scale", 5.0),
            params.get("attn_mode", "sdpa"),
            params.get("block_swap", 0),
            params.get("task", "i2v-14B"),
            params.get("negative_prompt", ""),
            *[params.get("lora_weights", ["None"]*4)[i] if isinstance(params.get("lora_weights", []), list) and i < len(params.get("lora_weights", [])) else "None" for i in range(4)],
            *[params.get("lora_multipliers", [1.0]*4)[i] if isinstance(params.get("lora_multipliers", []), list) and i < len(params.get("lora_multipliers", [])) else 1.0 for i in range(4)]
        ] if params else [gr.update()]*20,
        inputs=params_state,
        outputs=[
            wanx_prompt, wanx_width, wanx_height, wanx_video_length, 
            wanx_fps, wanx_infer_steps, wanx_seed, wanx_flow_shift, 
            wanx_guidance_scale, wanx_attn_mode, wanx_block_swap,
            wanx_task, wanx_negative_prompt, 
            *wanx_lora_weights,
            *wanx_lora_multipliers
        ]
    ).then(
        fn=change_to_wanx_i2v_tab, 
        inputs=None, 
        outputs=[tabs]
    )

    # 3. Update the WanX-t2v button handler
    send_to_wanx_t2v_btn.click(
        fn=lambda m: handle_send_to_wanx_tab(m, 'wanx_t2v'),
        inputs=[metadata_output],
        outputs=[status, params_state]
    ).then(
        lambda params: [
            params.get("prompt", ""),
            params.get("width", 832),
            params.get("height", 480),
            params.get("video_length", 81),
            params.get("fps", 16),
            params.get("infer_steps", 50),
            params.get("seed", -1),
            params.get("flow_shift", 5.0),
            params.get("guidance_scale", 5.0),
            params.get("attn_mode", "sdpa"),
            params.get("block_swap", 0),
            params.get("negative_prompt", ""),
            *[params.get("lora_weights", ["None"]*4)[i] if isinstance(params.get("lora_weights", []), list) and i < len(params.get("lora_weights", [])) else "None" for i in range(4)],
            *[params.get("lora_multipliers", [1.0]*4)[i] if isinstance(params.get("lora_multipliers", []), list) and i < len(params.get("lora_multipliers", [])) else 1.0 for i in range(4)]
        ] if params else [gr.update()]*20,
        inputs=params_state,
        outputs=[
            wanx_t2v_prompt,
            wanx_t2v_width,
            wanx_t2v_height,
            wanx_t2v_video_length,
            wanx_t2v_fps,
            wanx_t2v_infer_steps,
            wanx_t2v_seed,
            wanx_t2v_flow_shift,
            wanx_t2v_guidance_scale,
            wanx_t2v_attn_mode,
            wanx_t2v_block_swap,
            wanx_t2v_negative_prompt,
            *wanx_t2v_lora_weights,
            *wanx_t2v_lora_multipliers
        ]
    ).then(
        fn=change_to_wanx_t2v_tab, inputs=None, outputs=[tabs]
    )
    
    # Add a function to handle video transfer to wan22 tab
    def handle_send_to_wan22_tab(metadata: dict, video_path: str) -> Tuple[str, Dict, str]:
        """Handle both parameters and video transfer from Video Info to Wan2.2 tab"""
        if not metadata:
            metadata = {}
        
        # If we have a video, enable V2V mode automatically
        if video_path:
            metadata["enable_v2v"] = True
            
        return f"Parameters ready for Wan2.2", metadata, video_path

# Wan2.2 send-to logic
    send_to_wan22_btn.click(
        fn=handle_send_to_wan22_tab,
        inputs=[metadata_output, video_input],
        outputs=[status, params_state, wan22_input_video]
    ).then(
        # This lambda function is updated to return values for all 8 LoRAs and other new controls.
        lambda params, video_path: (
            (
                # Helper to safely get and pad LoRA lists from metadata
                (weights_from_meta := params.get("lora_weights", [])),
                (mults_from_meta := params.get("lora_multipliers", [])),
                (padded_weights := (weights_from_meta + ["None"] * 8)[:8]),
                (padded_mults := ([float(m) if isinstance(m, (int, float, str)) and str(m).replace('.', '', 1).isdigit() else 1.0 for m in mults_from_meta] + [1.0] * 8)[:8]),
                
                # Create the full list of return values
                [
                    params.get("prompt", ""),
                    params.get("negative_prompt", ""),
                    None,  # image_path
                    params.get("task", "i2v-A14B"),
                    params.get('width', 832),
                    params.get('height', 480),
                    params.get("frame_num", 81),
                    params.get("fps", 16),
                    params.get("seed", -1),
                    params.get("sample_solver", "unipc"),
                    params.get("sample_steps", 40),
                    params.get("flow_shift", 5.0),
                    params.get("sample_guide_scale", 3.5),
                    params.get("dual_dit_boundary", 0.875),
                    1,  # batch_size
                    "outputs",  # save_path
                    params.get("attn_mode", "sdpa"),
                    params.get("mixed_dtype", False),
                    params.get("block_swap", 30),
                    params.get("fp8", False),
                    params.get("fp8_scaled", False),
                    params.get("fp8_t5", False),
                    # Model paths - use dynamic defaults, don't transfer from metadata
                    get_default_low_noise_model('wan'),
                    get_default_high_noise_model('wan'),
                    get_default_clip_model('wan'),
                    "Wan2.2-TI2V-5B_fp16.safetensors",
                    get_default_vae_model('wan'),
                    get_default_t5_model('wan'),
                    # LoRAs
                    "lora",  # lora_folder
                    *padded_weights,          # Unpack 8 LoRA weights
                    *padded_mults,            # FIX: Corrected variable name from padded_multipliers
                    *[True] * 8,              # Defaults for 8 "apply low" checkboxes
                    *[False] * 8,             # Defaults for 8 "apply high" checkboxes
                    # Previews & Performance
                    True,  # enable_preview
                    5,  # preview_steps
                    params.get("dynamic_model_loading", False),
                    params.get("unload_text_encoders", False),
                    params.get("vae_fp32", True),
                    # V2V controls
                    params.get("enable_v2v", False)
                ]
            )[-1] # Return the created list
        ),
        inputs=[params_state, wan22_input_video],
        outputs=[
            wan22_prompt, wan22_negative_prompt, wan22_input_image, wan22_task, wan22_width, wan22_height,
            wan22_frame_num, wan22_fps, wan22_seed, wan22_sample_solver, wan22_sample_steps,
            wan22_flow_shift, wan22_sample_guide_scale, wan22_dual_dit_boundary, wan22_batch_size,
            wan22_save_path, wan22_attn_mode, wan22_mixed_dtype, wan22_block_swap, wan22_fp8, wan22_fp8_scaled, wan22_fp8_t5,
            wan22_dit_low_noise_path, wan22_dit_high_noise_path, wan22_clip_path, wan22_dit_path,
            wan22_vae_path, wan22_t5_path, wan22_lora_folder,
            *wan22_lora_weights,           # Unpack all 8 weight dropdowns
            *wan22_lora_multipliers,        # Unpack all 8 multiplier sliders
            *wan22_lora_apply_low,          # Unpack all 8 "apply low" checkboxes
            *wan22_lora_apply_high,         # Unpack all 8 "apply high" checkboxes
            wan22_enable_preview, wan22_preview_steps,
            wan22_dynamic_model_loading, wan22_unload_text_encoders, wan22_vae_fp32,
            wan22_enable_v2v
        ]
    ).then(
        fn=change_to_wan22_tab, inputs=None, outputs=[tabs]
    )
    
    # FramePack-Extension send-to logic
    def handle_send_to_fpe_tab(metadata: dict, video_path: str) -> Tuple[str, Dict, str]:
        """Prepare parameters and video path for the FramePack-Extension tab."""
        if not video_path:
            return "No video selected to send to FramePack-Extension", {}, None
        
        # If metadata is empty, provide a message but still allow video transfer
        status_msg = "Parameters ready for FramePack-Extension."
        if not metadata:
            status_msg = "Video sent to FramePack-Extension (no parameters found in metadata)."
            metadata = {} # Ensure metadata is a dict

        return status_msg, metadata, video_path

    def change_to_fpe_tab():
        return gr.Tabs(selected=11) # FramePack-Extension tab has id=11

    send_to_fpe_btn.click(
        fn=handle_send_to_fpe_tab,
        inputs=[metadata_output, video_input],
        outputs=[status, params_state, fpe_input_video] # status, state for params, and video input for FPE
    ).then(
        lambda params: (
            (
                (is_f1_from_meta := params.get("is_f1", True)), # Default to F1 if not specified
                (use_normal_fp_val := not is_f1_from_meta), # fpe_use_normal_framepack is opposite of is_f1
                
                # Determine resolution_max_dim
                (target_res_meta := params.get("target_resolution")),
                (video_w_meta := params.get("video_width")),
                (video_h_meta := params.get("video_height")),
                (
                    res_max_dim_val := int(target_res_meta) if target_res_meta and int(target_res_meta) > 0
                    else max(int(video_w_meta), int(video_h_meta)) if video_w_meta and video_h_meta and int(video_w_meta) > 0 and int(video_h_meta) > 0
                    else 640 # Default
                ),
                 # LoRA handling
                (weights_from_meta := params.get("lora_weights", [])),
                (mults_from_meta := params.get("lora_multipliers", [])),
                (padded_weights := (weights_from_meta + ["None"] * 4)[:4]),
                (padded_mults := ([float(m) if isinstance(m, (int, float, str)) and str(m).replace('.', '', 1).isdigit() else 1.0 for m in mults_from_meta] + [1.0] * 4)[:4]),

                [
                    params.get("prompt", "cinematic video of a cat wizard casting a spell"),
                    params.get("negative_prompt", ""),
                    params.get("seed", -1),
                    use_normal_fp_val,
                    # fpe_end_frame and fpe_end_frame_weight are typically not in generic metadata, use defaults
                    gr_update(value=None), # fpe_end_frame (Image)
                    gr_update(value=1.0),  # fpe_end_frame_weight
                    res_max_dim_val,
                    params.get("video_seconds", params.get("total_second_length", 5.0)), # Map from FramePack's video_seconds
                    params.get("latent_window_size", 9),
                    params.get("infer_steps", params.get("steps", 25)), # Map from FramePack's infer_steps
                    params.get("guidance_scale", params.get("cfg_scale", 1.0)), # Map from FramePack's guidance_scale to fpe_cfg_scale
                    params.get("embedded_cfg_scale", params.get("distilled_guidance_scale", 3.0)), # Map from FramePack's embedded_cfg_scale
                    # Model Paths - use FPE defaults or specific paths from metadata if available
                    # The DiT path is now primarily handled by the fpe_use_normal_framepack.change event
                    params.get("transformer_path", "hunyuan/FramePack_F1_I2V_HY_20250503.safetensors"), # Placeholder, will be overridden
                    params.get("vae_path", "hunyuan/pytorch_model.pt"),
                    params.get("text_encoder_path", "hunyuan/llava_llama3_fp16.safetensors"),
                    params.get("text_encoder_2_path", "hunyuan/clip_l.safetensors"),
                    params.get("image_encoder_path", "hunyuan/model.safetensors"),
                    # Advanced performance
                    params.get("attn_mode", "torch"),
                    params.get("fp8_llm", False), # This will be correctly set by fpe_use_normal_framepack.change
                    params.get("vae_chunk_size", 32),
                    params.get("vae_spatial_tile_sample_min_size", 128),
                    # LoRAs
                    *padded_weights,
                    *padded_mults,
                ]
            )[-1] # Return the list of values
        ) if params else [gr.update()] * (18 + 8), # 18 direct params + 4 lora weights + 4 lora mults
        inputs=params_state,
        outputs=[
            fpe_prompt, fpe_negative_prompt, fpe_seed,
            fpe_use_normal_framepack, # This will trigger its own .change event
            fpe_end_frame, fpe_end_frame_weight, # These are UI only if fpe_use_normal_framepack is True
            fpe_resolution_max_dim, fpe_total_second_length, fpe_latent_window_size,
            fpe_steps, fpe_cfg_scale, fpe_distilled_guidance_scale,
            # Model Paths
            fpe_transformer_path, # Will be set by fpe_use_normal_framepack.change
            fpe_vae_path, fpe_text_encoder_path, fpe_text_encoder_2_path, fpe_image_encoder_path,
            # Advanced
            fpe_attn_mode, fpe_fp8_llm, # fpe_fp8_llm also set by fpe_use_normal_framepack.change
            fpe_vae_chunk_size, fpe_vae_spatial_tile_sample_min_size,
            # LoRAs
            *fpe_lora_weights_ui, *fpe_lora_multipliers_ui,
        ]
    ).then(
        fn=change_to_fpe_tab,
        inputs=None,
        outputs=[tabs]
    )
    #text to video
    def change_to_tab_one():
        return gr.Tabs(selected=1) #This will navigate
    #video to video

    def change_to_skyreels_tab():
        return gr.Tabs(selected=3) 
    
    #SKYREELS TAB!!!
    # Add state management for dimensions
    def sync_skyreels_dimensions(width, height):
        return gr.update(value=width), gr.update(value=height)

    # Add this function to update the LoRA dropdowns in the SKYREELS tab
    def update_skyreels_lora_dropdowns(lora_folder: str, *current_values) -> List[gr.update]:
        new_choices = get_lora_options(lora_folder)
        weights = current_values[:4]
        multipliers = current_values[4:8]

        results = []
        for i in range(4):
            weight = weights[i] if i < len(weights) else "None"
            multiplier = multipliers[i] if i < len(multipliers) else 1.0
            if weight not in new_choices:
                weight = "None"
            results.extend([
                gr.update(choices=new_choices, value=weight),
                gr.update(value=multiplier) 
            ])

        return results

    # Add this function to update the models dropdown in the SKYREELS tab
    def update_skyreels_model_dropdown(dit_folder: str) -> Dict:
        models = get_dit_models(dit_folder)
        return gr.update(choices=models, value=models[0] if models else None)

    # Add event handler for model dropdown refresh
    skyreels_dit_folder.change(
        fn=update_skyreels_model_dropdown,
        inputs=[skyreels_dit_folder],
        outputs=[skyreels_model]
    )

    # Add handlers for the refresh button
    skyreels_refresh_btn.click(
        fn=update_skyreels_lora_dropdowns,
        inputs=[skyreels_lora_folder] + skyreels_lora_weights + skyreels_lora_multipliers,
        outputs=[drop for _ in range(4) for drop in [skyreels_lora_weights[_], skyreels_lora_multipliers[_]]]
    )      
    # Skyreels dimension handling
    def calculate_skyreels_width(height, original_dims):
        if not original_dims:
            return gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        aspect_ratio = orig_w / orig_h
        new_width = math.floor((height * aspect_ratio) / 16) * 16
        return gr.update(value=new_width)

    def calculate_skyreels_height(width, original_dims):
        if not original_dims:
            return gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        aspect_ratio = orig_w / orig_h
        new_height = math.floor((width / aspect_ratio) / 16) * 16
        return gr.update(value=new_height)

    def update_skyreels_from_scale(scale, original_dims):
        if not original_dims:
            return gr.update(), gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        new_w = math.floor((orig_w * scale / 100) / 16) * 16
        new_h = math.floor((orig_h * scale / 100) / 16) * 16
        return gr.update(value=new_w), gr.update(value=new_h)

    def update_skyreels_dimensions(image):
        if image is None:
            return "", gr.update(value=544), gr.update(value=544)
        img = Image.open(image)
        w, h = img.size
        w = (w // 16) * 16
        h = (h // 16) * 16
        return f"{w}x{h}", w, h

    def handle_skyreels_gallery_select(evt: gr.SelectData) -> int:
        return evt.index

    def send_skyreels_to_v2v(
        gallery: list,
        prompt: str,
        selected_index: int,
        width: int,
        height: int,
        video_length: int,
        fps: int,
        infer_steps: int,
        seed: int,
        flow_shift: float,
        cfg_scale: float,
        lora1: str,
        lora2: str,
        lora3: str,
        lora4: str,
        lora1_multiplier: float,
        lora2_multiplier: float,
        lora3_multiplier: float,
        lora4_multiplier: float,
        negative_prompt: str = ""  # Add this parameter
    ) -> Tuple:
        if not gallery or selected_index is None or selected_index >= len(gallery):
            return (None, "", width, height, video_length, fps, infer_steps, seed, 
                    flow_shift, cfg_scale, lora1, lora2, lora3, lora4,
                    lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier,
                    negative_prompt)  # Add negative_prompt to return

        selected_item = gallery[selected_index]

        if isinstance(selected_item, dict):
            video_path = selected_item.get("name", selected_item.get("data", None))
        elif isinstance(selected_item, (tuple, list)):
            video_path = selected_item[0]
        else:
            video_path = selected_item

        if isinstance(video_path, tuple):
            video_path = video_path[0]

        return (str(video_path), prompt, width, height, video_length, fps, infer_steps, seed, 
                flow_shift, cfg_scale, lora1, lora2, lora3, lora4,
                lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier,
                negative_prompt)  # Add negative_prompt to return

    # Add event handlers for the SKYREELS tab
    skyreels_prompt.change(fn=count_prompt_tokens, inputs=skyreels_prompt, outputs=skyreels_token_counter)
    skyreels_stop_btn.click(fn=lambda: stop_event.set(), queue=False)

    # Image input handling
    skyreels_input.change(
        fn=update_skyreels_dimensions,
        inputs=[skyreels_input],
        outputs=[skyreels_original_dims, skyreels_width, skyreels_height]
    )

    skyreels_scale_slider.change(
        fn=update_skyreels_from_scale,
        inputs=[skyreels_scale_slider, skyreels_original_dims],
        outputs=[skyreels_width, skyreels_height]
    )

    skyreels_calc_width_btn.click(
        fn=calculate_skyreels_width,
        inputs=[skyreels_height, skyreels_original_dims],
        outputs=[skyreels_width]
    )

    skyreels_calc_height_btn.click(
        fn=calculate_skyreels_height,
        inputs=[skyreels_width, skyreels_original_dims],
        outputs=[skyreels_height]
    )

    # Handle checkbox visibility toggling
    skyreels_use_random_folder.change(
        fn=lambda x: (gr.update(visible=x), gr.update(visible=x), gr.update(visible=not x)),
        inputs=[skyreels_use_random_folder],
        outputs=[skyreels_input_folder, skyreels_folder_status, skyreels_input]
    )

    # Validate folder button click handler
    skyreels_validate_folder_btn.click(
        fn=lambda folder: get_random_image_from_folder(folder)[1],
        inputs=[skyreels_input_folder],
        outputs=[skyreels_folder_status]
    )

    skyreels_use_random_folder.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[skyreels_use_random_folder],
        outputs=[skyreels_validate_folder_btn]
    )

    # Modify the skyreels_generate_btn.click event handler to use process_random_image_batch when folder mode is on
    skyreels_generate_btn.click(
        fn=batch_handler,
        inputs=[
            skyreels_use_random_folder,
            # Rest of the arguments
            skyreels_prompt,
            skyreels_negative_prompt,
            skyreels_width,
            skyreels_height,
            skyreels_video_length,
            skyreels_fps,
            skyreels_infer_steps,
            skyreels_seed,
            skyreels_flow_shift,
            skyreels_guidance_scale,
            skyreels_embedded_cfg_scale,
            skyreels_batch_size,
            skyreels_input_folder,
            skyreels_dit_folder,
            skyreels_model,
            skyreels_vae,
            skyreels_te1,
            skyreels_te2,
            skyreels_save_path,
            skyreels_output_type,
            skyreels_attn_mode,
            skyreels_block_swap,
            skyreels_exclude_single_blocks,
            skyreels_use_split_attn,
            skyreels_use_fp8,
            skyreels_split_uncond,
            skyreels_lora_folder,
            *skyreels_lora_weights,
            *skyreels_lora_multipliers,
            skyreels_input  # Add the input image path
        ],
        outputs=[skyreels_output, skyreels_batch_progress, skyreels_progress_text],
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[skyreels_batch_size],
        outputs=skyreels_selected_index
    )

    # Gallery selection handling
    skyreels_output.select(
        fn=handle_skyreels_gallery_select,
        outputs=skyreels_selected_index
    )

    # Send to Video2Video handler
    skyreels_send_to_v2v_btn.click(
        fn=send_skyreels_to_v2v,
        inputs=[
            skyreels_output, skyreels_prompt, skyreels_selected_index,
            skyreels_width, skyreels_height, skyreels_video_length,
            skyreels_fps, skyreels_infer_steps, skyreels_seed,
            skyreels_flow_shift, skyreels_guidance_scale
        ] + skyreels_lora_weights + skyreels_lora_multipliers + [skyreels_negative_prompt],  # This is ok because skyreels_negative_prompt is a Gradio component
        outputs=[
            v2v_input, v2v_prompt, v2v_width, v2v_height,
            v2v_video_length, v2v_fps, v2v_infer_steps,
            v2v_seed, v2v_flow_shift, v2v_cfg_scale
        ] + v2v_lora_weights + v2v_lora_multipliers + [v2v_negative_prompt]
    ).then(
        fn=change_to_tab_two,
        inputs=None,
        outputs=[tabs]
    )

    # Refresh button handler
    skyreels_refresh_outputs = [skyreels_model]
    for i in range(4):
        skyreels_refresh_outputs.extend([skyreels_lora_weights[i], skyreels_lora_multipliers[i]])

    skyreels_refresh_btn.click(
        fn=update_dit_and_lora_dropdowns,
        inputs=[skyreels_dit_folder, skyreels_lora_folder, skyreels_model] + skyreels_lora_weights + skyreels_lora_multipliers,
        outputs=skyreels_refresh_outputs
    )
      
    def calculate_v2v_width(height, original_dims):
        if not original_dims:
            return gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        aspect_ratio = orig_w / orig_h
        new_width = math.floor((height * aspect_ratio) / 16) * 16  # Ensure divisible by 16
        return gr.update(value=new_width)

    def calculate_v2v_height(width, original_dims):
        if not original_dims:
            return gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        aspect_ratio = orig_w / orig_h
        new_height = math.floor((width / aspect_ratio) / 16) * 16  # Ensure divisible by 16
        return gr.update(value=new_height)

    def update_v2v_from_scale(scale, original_dims):
        if not original_dims:
            return gr.update(), gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        new_w = math.floor((orig_w * scale / 100) / 16) * 16  # Ensure divisible by 16
        new_h = math.floor((orig_h * scale / 100) / 16) * 16  # Ensure divisible by 16
        return gr.update(value=new_w), gr.update(value=new_h)

    def update_v2v_dimensions(video):
        if video is None:
            return "", gr.update(value=544), gr.update(value=544)
        cap = cv2.VideoCapture(video)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        # Make dimensions divisible by 16
        w = (w // 16) * 16
        h = (h // 16) * 16
        return f"{w}x{h}", w, h
    
    # Event Handlers for Video to Video Tab
    v2v_input.change(
        fn=update_v2v_dimensions,
        inputs=[v2v_input],
        outputs=[v2v_original_dims, v2v_width, v2v_height]
    )

    v2v_scale_slider.change(
        fn=update_v2v_from_scale,
        inputs=[v2v_scale_slider, v2v_original_dims],
        outputs=[v2v_width, v2v_height]
    )

    v2v_calc_width_btn.click(
        fn=calculate_v2v_width,
        inputs=[v2v_height, v2v_original_dims],
        outputs=[v2v_width]
    )

    v2v_calc_height_btn.click(
        fn=calculate_v2v_height,
        inputs=[v2v_width, v2v_original_dims],
        outputs=[v2v_height]
    )

    ##Image 2 video dimension logic
    def calculate_width(height, original_dims):
        if not original_dims:
            return gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        aspect_ratio = orig_w / orig_h
        new_width = math.floor((height * aspect_ratio) / 16) * 16  # Changed from 8 to 16
        return gr.update(value=new_width)

    def calculate_height(width, original_dims):
        if not original_dims:
            return gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        aspect_ratio = orig_w / orig_h
        new_height = math.floor((width / aspect_ratio) / 16) * 16  # Changed from 8 to 16
        return gr.update(value=new_height)

    def update_from_scale(scale, original_dims):
        if not original_dims:
            return gr.update(), gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        new_w = math.floor((orig_w * scale / 100) / 16) * 16  # Changed from 8 to 16
        new_h = math.floor((orig_h * scale / 100) / 16) * 16  # Changed from 8 to 16
        return gr.update(value=new_w), gr.update(value=new_h)

    def update_dimensions(image):
        if image is None:
            return "", gr.update(value=544), gr.update(value=544)
        img = Image.open(image)
        w, h = img.size
        # Make dimensions divisible by 16
        w = (w // 16) * 16  # Changed from 8 to 16
        h = (h // 16) * 16  # Changed from 8 to 16
        return f"{w}x{h}", w, h
    i2v_input.change(
        fn=update_dimensions,
        inputs=[i2v_input],
        outputs=[original_dims, i2v_width, i2v_height] # Update correct components
    )

    scale_slider.change(
        fn=update_from_scale,
        inputs=[scale_slider, original_dims],
        outputs=[i2v_width, i2v_height] # Update correct components
    )

    calc_width_btn.click(
        fn=calculate_width,
        inputs=[i2v_height, original_dims], # Update correct components
        outputs=[i2v_width]
    )

    calc_height_btn.click(
        fn=calculate_height,
        inputs=[i2v_width, original_dims], # Update correct components
        outputs=[i2v_height]    
    )

    # Function to get available DiT models
    def get_dit_models(dit_folder: str) -> List[str]:
        if not os.path.exists(dit_folder):
            return ["mp_rank_00_model_states.pt"]
        models = [f for f in os.listdir(dit_folder) if f.endswith('.pt') or f.endswith('.safetensors')]
        models.sort(key=str.lower)
        return models if models else ["mp_rank_00_model_states.pt"]

    # Function to perform model merging
    def merge_models(
        dit_folder: str,
        dit_model: str,
        output_model: str,
        exclude_single_blocks: bool,
        merge_lora_folder: str,
        *lora_params  # Will contain both weights and multipliers
    ) -> str:
        try:
            # Separate weights and multipliers
            num_loras = len(lora_params) // 2
            weights = list(lora_params[:num_loras])
            multipliers = list(lora_params[num_loras:])

            # Filter out "None" selections
            valid_loras = []
            for weight, mult in zip(weights, multipliers):
                if weight and weight != "None":
                    valid_loras.append((os.path.join(merge_lora_folder, weight), mult))

            if not valid_loras:
                return "No LoRA models selected for merging"

            # Create output path in the dit folder
            os.makedirs(dit_folder, exist_ok=True)
            output_path = os.path.join(dit_folder, output_model)
            
            # Prepare command
            cmd = [
                sys.executable,
                "merge_lora.py",
                "--dit", os.path.join(dit_folder, dit_model),
                "--save_merged_model", output_path
            ]

            # Add LoRA weights and multipliers
            weights = [weight for weight, _ in valid_loras]
            multipliers = [str(mult) for _, mult in valid_loras]
            cmd.extend(["--lora_weight"] + weights)
            cmd.extend(["--lora_multiplier"] + multipliers)

            if exclude_single_blocks:
                cmd.append("--exclude_single_blocks")

            # Execute merge operation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            if os.path.exists(output_path):
                return f"Successfully merged model and saved to {output_path}"
            else:
                return "Error: Output file not created"

        except subprocess.CalledProcessError as e:
            return f"Error during merging: {e.stderr}"
        except Exception as e:
            return f"Error: {str(e)}"

    # Connect events
    merge_btn.click(
        fn=merge_models,
        inputs=[
            dit_folder,
            dit_model,
            output_model,
            exclude_single_blocks,
            merge_lora_folder,
            *merge_lora_weights,
            *merge_lora_multipliers
        ],
        outputs=merge_status
    )

    # Refresh buttons for both DiT and LoRA dropdowns
    merge_refresh_btn.click(
        fn=lambda f: update_dit_dropdown(f),
        inputs=[dit_folder],
        outputs=[dit_model]
    )

    # LoRA refresh handling
    merge_refresh_outputs = []
    for i in range(4):
        merge_refresh_outputs.extend([merge_lora_weights[i], merge_lora_multipliers[i]])

    merge_refresh_btn.click(
        fn=update_lora_dropdowns,
        inputs=[merge_lora_folder] + merge_lora_weights + merge_lora_multipliers,
        outputs=merge_refresh_outputs
    )
    # Event handlers
    prompt.change(fn=count_prompt_tokens, inputs=prompt, outputs=token_counter)
    v2v_prompt.change(fn=count_prompt_tokens, inputs=v2v_prompt, outputs=v2v_token_counter)
    stop_btn.click(fn=lambda: stop_event.set(), queue=False)
    v2v_stop_btn.click(fn=lambda: stop_event.set(), queue=False)

    #Image_to_Video
    def image_to_video(image_path, output_path, width, height, frames=240):  # Add width, height parameters
        img = Image.open(image_path)

        # Resize to the specified dimensions
        img_resized = img.resize((width, height), Image.LANCZOS)
        temp_image_path = os.path.join(os.path.dirname(output_path), "temp_resized_image.png")
        img_resized.save(temp_image_path)

        # Rest of function remains the same
        frame_rate = 24
        duration = frames / frame_rate
        command = [
            "ffmpeg", "-loop", "1", "-i", temp_image_path, "-c:v", "libx264",
            "-t", str(duration), "-pix_fmt", "yuv420p",
            "-vf", f"fps={frame_rate}", output_path
        ]
        
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Video saved to {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while creating the video: {e}")
            return False
        finally:
            # Clean up the temporary image file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            img.close()  # Make sure to close the image file explicitly

    def generate_from_image(
        image_path, 
        prompt, width, height, video_length, fps, infer_steps,
        seed, model, vae, te1, te2, save_path, flow_shift, cfg_scale, 
        output_type, attn_mode, block_swap, exclude_single_blocks, use_split_attn,
        lora_folder, strength, batch_size, *lora_params
    ):
        """Generate video from input image with progressive updates"""
        global stop_event
        stop_event.clear()
    
        # Create temporary video path
        temp_video_path = os.path.join(save_path, f"temp_{os.path.basename(image_path)}.mp4")
    
        try:
            # Convert image to video
            if not image_to_video(image_path, temp_video_path, width, height, frames=video_length):
                yield [], "Failed to create temporary video", "Error in video creation"
                return
    
            # Ensure video is fully written before proceeding
            time.sleep(1)
            if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
                yield [], "Failed to create temporary video", "Temporary video file is empty or missing"
                return
    
            # Get video dimensions
            try:
                probe = ffmpeg.probe(temp_video_path)
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                if video_stream is None:
                    raise ValueError("No video stream found")
                width = int(video_stream['width'])
                height = int(video_stream['height'])
            except Exception as e:
                yield [], f"Error reading video dimensions: {str(e)}", "Video processing error"
                return
    
            # Generate the video using the temporary file
            try:
                generator = process_single_video(
                    prompt, width, height, batch_size, video_length, fps, infer_steps,
                    seed, model, vae, te1, te2, save_path, flow_shift, cfg_scale,
                    output_type, attn_mode, block_swap, exclude_single_blocks, use_split_attn,
                    lora_folder, *lora_params, video_path=temp_video_path, strength=strength
                )
    
                # Forward all generator updates
                for videos, batch_text, progress_text in generator:
                    yield videos, batch_text, progress_text
    
            except Exception as e:
                yield [], f"Error in video generation: {str(e)}", "Generation error"
                return
    
        except Exception as e:
            yield [], f"Unexpected error: {str(e)}", "Error occurred"
            return
    
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
            except Exception:
                pass  # Ignore cleanup errors


    # Add event handlers
    i2v_prompt.change(fn=count_prompt_tokens, inputs=i2v_prompt, outputs=i2v_token_counter)
    i2v_stop_btn.click(fn=lambda: stop_event.set(), queue=False)

    def handle_i2v_gallery_select(evt: gr.SelectData) -> int:
        """Track selected index when I2V gallery item is clicked"""
        return evt.index

    def send_i2v_to_v2v(
        gallery: list, 
        prompt: str, 
        selected_index: int,
        width: int,
        height: int,
        video_length: int,
        fps: int,
        infer_steps: int,
        seed: int,
        flow_shift: float,
        cfg_scale: float,
        lora1: str,
        lora2: str,
        lora3: str,
        lora4: str,
        lora1_multiplier: float,
        lora2_multiplier: float,
        lora3_multiplier: float,
        lora4_multiplier: float
    ) -> Tuple[Optional[str], str, int, int, int, int, int, int, float, float, str, str, str, str, float, float, float, float]:
        """Send the selected video and parameters from Image2Video tab to Video2Video tab"""
        if not gallery or selected_index is None or selected_index >= len(gallery):
            return None, "", width, height, video_length, fps, infer_steps, seed, flow_shift, cfg_scale, \
                   lora1, lora2, lora3, lora4, lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier

        selected_item = gallery[selected_index]

        # Handle different gallery item formats
        if isinstance(selected_item, dict):
            video_path = selected_item.get("name", selected_item.get("data", None))
        elif isinstance(selected_item, (tuple, list)):
            video_path = selected_item[0]
        else:
            video_path = selected_item

        # Final cleanup for Gradio Video component
        if isinstance(video_path, tuple):
            video_path = video_path[0]

        # Use the original width and height without doubling
        return (str(video_path), prompt, width, height, video_length, fps, infer_steps, seed, 
                flow_shift, cfg_scale, lora1, lora2, lora3, lora4, 
                lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier)

    # Generate button handler for h-basic-i2v
    i2v_generate_btn.click(
        fn=process_i2v_batch, # <<< Use the new batch function
        inputs=[
            i2v_prompt,
            i2v_input, # Image path
            i2v_width,
            i2v_height,
            i2v_batch_size,
            i2v_video_length,
            i2v_fps,
            i2v_infer_steps,
            i2v_seed,
            i2v_dit_folder,
            i2v_model,
            i2v_vae,
            i2v_te1,
            i2v_te2,
            i2v_clip_vision_path,
            i2v_save_path,
            i2v_flow_shift,
            i2v_cfg_scale, # embedded_cfg_scale
            i2v_guidance_scale, # main CFG scale
            i2v_output_type,
            i2v_attn_mode,
            i2v_block_swap,
            i2v_exclude_single_blocks,
            i2v_use_split_attn,
            i2v_lora_folder,
            i2v_vae_chunk_size,
            i2v_vae_spatial_tile_min,
            # --- Add negative prompt component if you have one ---
            # i2v_negative_prompt, # Uncomment if you added this textbox
            # --- If no negative prompt textbox, pass None or "": ---
            gr.Textbox(value="", visible=False), # Placeholder if no UI element
            # --- End negative prompt handling ---
            i2v_use_fp8,
            i2v_fp8_llm,
            *i2v_lora_weights, # Pass LoRA weights components
            *i2v_lora_multipliers # Pass LoRA multipliers components
        ],
        outputs=[i2v_output, i2v_batch_progress, i2v_progress_text],
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[i2v_batch_size],
        outputs=i2v_selected_index
    )
    # Send to Video2Video
    i2v_output.select(
        fn=handle_i2v_gallery_select,
        outputs=i2v_selected_index
    )

    i2v_send_to_v2v_btn.click(
        fn=send_i2v_to_v2v, # Function definition needs careful review/update if args changed
        inputs=[
            i2v_output, i2v_prompt, i2v_selected_index,
            i2v_width, i2v_height, # <<< Use i2v width/height
            i2v_video_length, i2v_fps, i2v_infer_steps,
            i2v_seed, i2v_flow_shift, i2v_cfg_scale # <<< Use i2v cfg_scale (embedded)
        ] + i2v_lora_weights + i2v_lora_multipliers, # <<< Use i2v LoRAs
        outputs=[
            v2v_input, v2v_prompt,
            v2v_width, v2v_height, # Target V2V components
            v2v_video_length, v2v_fps, v2v_infer_steps,
            v2v_seed, v2v_flow_shift, v2v_cfg_scale # Target V2V components
        ] + v2v_lora_weights + v2v_lora_multipliers # Target V2V LoRAs
    ).then(
        fn=change_to_tab_two, inputs=None, outputs=[tabs]
    )

    # Wan2.2 Tab Event Handlers
    wan22_prompt.change(fn=count_prompt_tokens, inputs=wan22_prompt, outputs=wan22_token_counter)
    wan22_stop_btn.click(fn=lambda: stop_event.set(), queue=False)
    wan22_random_seed_btn.click(fn=set_random_seed, inputs=None, outputs=[wan22_seed])
    
    # V2V visibility toggle
    wan22_enable_v2v.change(
        fn=lambda enabled: gr.update(visible=enabled),
        inputs=[wan22_enable_v2v],
        outputs=[wan22_v2v_controls]
    )
    
    # Extension visibility toggle
    wan22_enable_extension.change(
        fn=lambda enabled: [gr.update(visible=enabled), gr.update(visible=enabled), gr.update(visible=enabled)],
        inputs=[wan22_enable_extension],
        outputs=[wan22_extension_controls, wan22_extension_model_controls, wan22_extension_injection_controls]
    )
    
    # Mutual exclusivity for extension model selection
    def handle_force_high_noise_change(high_noise_checked):
        if high_noise_checked:
            return gr.update(value=False)  # Uncheck force_low_noise
        return gr.update()
    
    def handle_force_low_noise_change(low_noise_checked):
        if low_noise_checked:
            return gr.update(value=False)  # Uncheck force_high_noise
        return gr.update()
    
    wan22_force_high_noise.change(
        fn=handle_force_high_noise_change,
        inputs=[wan22_force_high_noise],
        outputs=[wan22_force_low_noise]
    )
    
    wan22_force_low_noise.change(
        fn=handle_force_low_noise_change,
        inputs=[wan22_force_low_noise],
        outputs=[wan22_force_high_noise]
    )
    
    # Image input handling for wan22
    wan22_input_image.change(
        fn=update_wanx_image_dimensions,  # Reuse the same function
        inputs=[wan22_input_image],
        outputs=[wan22_original_dims, wan22_width, wan22_height]
    )
    
    # Width/height calculation buttons
    wan22_calc_width_btn.click(
        fn=calculate_wanx_width,  # Reuse function from WanX
        inputs=[wan22_height, wan22_original_dims],
        outputs=[wan22_width]
    )
    
    wan22_calc_height_btn.click(
        fn=calculate_wanx_height,  # Reuse function from WanX
        inputs=[wan22_width, wan22_original_dims],
        outputs=[wan22_height]
    )
    
    # Video input handling for wan22 V2V
    def update_wan22_video_dimensions(video_path):
        """Extract video dimensions and update UI"""
        if not video_path:
            return gr.update(), gr.update(), gr.update(), gr.update()
        
        info = get_video_info(video_path)  # This function already exists
        if info:
            return (
                gr.update(value=info['width']),
                gr.update(value=info['height']), 
                gr.update(value=min(info['total_frames'], 201)),  # Cap at 201 frames
                gr.update(value=info['fps'])
            )
        return gr.update(), gr.update(), gr.update(), gr.update()

    wan22_input_video.change(
        fn=update_wan22_video_dimensions,
        inputs=[wan22_input_video],
        outputs=[wan22_width, wan22_height, wan22_frame_num, wan22_fps]
    )

    def update_wan22_model_paths_and_settings(task):
        is_a14b = "A14B" in task
        is_i2v_a14b = "i2v-A14B" in task
        is_t2v_a14b = "t2v-A14B" in task
        is_ti2v5b = "ti2v-5B" in task
        
        # Set model paths based on task using task-specific model detection
        if is_t2v_a14b:
            dit_low_path = get_task_specific_low_noise_model('wan', 't2v')
            dit_high_path = get_task_specific_high_noise_model('wan', 't2v')
            boundary_value = 0.875  # Default for t2v-A14B
            boundary_visible = True
        elif is_i2v_a14b:
            dit_low_path = get_task_specific_low_noise_model('wan', 'i2v')
            dit_high_path = get_task_specific_high_noise_model('wan', 'i2v')
            boundary_value = 0.900  # Default for i2v-A14B
            boundary_visible = True
        elif is_ti2v5b:
            dit_low_path = get_task_specific_low_noise_model('wan', 'i2v')  # Use i2v models for ti2v-5B
            dit_high_path = get_task_specific_high_noise_model('wan', 'i2v')  # Use i2v models for ti2v-5B
            boundary_value = 0.875  # Default value
            boundary_visible = False  # Hide for ti2v-5B as it's not dual-dit
        else:
            dit_low_path = get_task_specific_low_noise_model('wan', 'i2v')  # Default to i2v
            dit_high_path = get_task_specific_high_noise_model('wan', 'i2v')  # Default to i2v
            boundary_value = 0.875
            boundary_visible = False
        
        return (
            gr.update(visible=is_a14b),      # A14B model paths group
            gr.update(visible=is_ti2v5b),    # ti2v-5B model path group
            gr.update(visible=is_i2v_a14b),  # CLIP path textbox inside A14B group
            gr.update(value=dit_low_path),   # Low noise model path
            gr.update(value=dit_high_path),  # High noise model path
            gr.update(value=boundary_value, visible=boundary_visible),  # Boundary slider
            # ADD THIS:
            gr.update(visible=is_a14b)  # Dynamic model loading checkbox
        )

    wan22_task.change(
        fn=update_wan22_model_paths_and_settings,
        inputs=[wan22_task],
        outputs=[wan22_a14b_paths, wan22_ti2v5b_paths, wan22_clip_path, 
                wan22_dit_low_noise_path, wan22_dit_high_noise_path, wan22_dual_dit_boundary,
                wan22_dynamic_model_loading]  # ADD THIS
    )
    
    # Add visibility control for high noise LoRA checkboxes based on task
    def update_high_noise_lora_visibility(task):
        """Show/hide high noise LoRA checkboxes based on task"""
        is_dual_dit = "A14B" in task  # A14B models use dual-dit
        updates = []
        for i in range(8):  # FIX: Changed from 4 to 8
            updates.append(gr.update(visible=is_dual_dit))  # High noise checkbox visibility
        return updates
    
    # Connect the task dropdown to update visibility of high noise checkboxes
    wan22_task.change(
        fn=update_high_noise_lora_visibility,
        inputs=[wan22_task],
        outputs=wan22_lora_apply_high
    )

    wan22_generate_btn.click(
        fn=wan22_batch_handler,
        inputs=[
            wan22_prompt,
            wan22_negative_prompt,
            wan22_input_image,
            wan22_task,
            wan22_width,
            wan22_height,
            wan22_frame_num,
            wan22_fps,
            wan22_seed,
            wan22_sample_solver,
            wan22_sample_steps,
            wan22_flow_shift,
            wan22_sample_guide_scale,
            wan22_dual_dit_boundary,
            wan22_batch_size,
            wan22_save_path,
            # Performance & Model Paths
            wan22_attn_mode,
            wan22_mixed_dtype,
            wan22_block_swap,
            wan22_fp8,
            wan22_fp8_scaled,
            wan22_fp8_t5,
            wan22_dit_low_noise_path,
            wan22_dit_high_noise_path,
            wan22_clip_path,
            wan22_dit_path,
            wan22_vae_path,
            wan22_t5_path,
            # LoRAs
            wan22_lora_folder,
            *wan22_lora_weights,
            *wan22_lora_multipliers,
            *wan22_lora_apply_low,
            *wan22_lora_apply_high,
            # Previews
            wan22_enable_preview,
            wan22_preview_steps,
            # ADD THIS:
            wan22_dynamic_model_loading,
            wan22_unload_text_encoders,
            wan22_vae_fp32,
            # V2V arguments
            wan22_enable_v2v,
            wan22_input_video,
            wan22_v2v_strength,
            wan22_v2v_low_noise_only,
            wan22_v2v_use_i2v,
            # Extension arguments
            wan22_enable_extension,
            wan22_extend_frames,
            wan22_frames_to_check,
        ],
        outputs=[wan22_output, wan22_preview_output, wan22_batch_progress, wan22_progress_text],
        queue=True
    )

    def handle_wan22_gallery_select(evt: gr.SelectData) -> int:
        return evt.index

    wan22_output.select(fn=handle_wan22_gallery_select, outputs=wan22_selected_index)

    wan22_lora_refresh_outputs_list = []
    for i in range(len(wan22_lora_weights)):
        wan22_lora_refresh_outputs_list.extend([wan22_lora_weights[i], wan22_lora_multipliers[i]])
    
    def refresh_8_loras(folder: str) -> List[gr.update]:
        """Helper to refresh 8 LoRA dropdowns and reset multipliers."""
        choices = get_lora_options(folder)
        updates = []
        for _ in range(8):
            updates.extend([gr.update(choices=choices, value="None"), gr.update(value=1.0)])
        return updates

    wan22_lora_refresh_btn.click(
        fn=refresh_8_loras,
        inputs=[wan22_lora_folder],
        outputs=wan22_lora_refresh_outputs_list
    )

    #Video Info
    def clean_video_path(video_path) -> str:
        """Extract clean video path from Gradio's various return formats"""
        print(f"Input video_path: {video_path}, type: {type(video_path)}")
        if isinstance(video_path, dict):
            path = video_path.get("name", "")
        elif isinstance(video_path, (tuple, list)):
            path = video_path[0]
        elif isinstance(video_path, str):
            path = video_path
        else:
            path = ""
        print(f"Cleaned path: {path}")
        return path
    def handle_video_upload(video_path: str) -> Dict:
        """Handle video upload and metadata extraction"""
        if not video_path:
            return {}, "No video uploaded"

        metadata = extract_video_metadata(video_path)
        if not metadata:
            return {}, "No metadata found in video"

        return metadata, "Metadata extracted successfully"
    
    def get_video_info(video_path: str) -> dict:
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
            
            width = int(video_info['width'])
            height = int(video_info['height'])
            fps = eval(video_info['r_frame_rate'])  # This converts '30/1' to 30.0
            
            # Calculate total frames
            duration = float(probe['format']['duration'])
            total_frames = int(duration * fps)
            
            # Ensure video length does not exceed 201 frames
            if total_frames > 201:
                total_frames = 201
                duration = total_frames / fps  # Adjust duration accordingly
    
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration  # Might be useful in some contexts
            }
        except Exception as e:
            print(f"Error extracting video info: {e}")
            return {}
        
    def extract_video_details(video_path: str) -> Tuple[dict, str]:
        metadata = extract_video_metadata(video_path)
        video_details = get_video_info(video_path)

        # Combine metadata with video details
        for key, value in video_details.items():
            if key not in metadata:
                metadata[key] = value

        # Ensure video length does not exceed 201 frames
        if 'video_length' in metadata:
            metadata['video_length'] = min(metadata['video_length'], 201)
        else:
            metadata['video_length'] = min(video_details.get('total_frames', 0), 201)

        # Return both the updated metadata and a status message
        return metadata, "Video details extracted successfully"

    def send_parameters_to_tab(metadata: Dict, target_tab: str) -> Tuple[str, Dict]:
        """Create parameter mapping for target tab"""
        if not metadata:
            return "No parameters to send", {}

        tab_name = "Text2Video" if target_tab == "t2v" else "Video2Video"
        try:
            mapping = create_parameter_transfer_map(metadata, target_tab)
            return f"Parameters ready for {tab_name}", mapping
        except Exception as e:
            return f"Error: {str(e)}", {}
        
    video_input.upload(
        fn=extract_video_details,
        inputs=video_input,
        outputs=[metadata_output, status]
    )

    send_to_t2v_btn.click(
        fn=lambda m: send_parameters_to_tab(m, "t2v"),
        inputs=metadata_output,
        outputs=[status, params_state]
    ).then(
        fn=change_to_tab_one, inputs=None, outputs=[tabs]
    ).then(
        lambda params: [
            params.get("prompt", ""),
            params.get("width", 544),              # Parameter mapping is fine here
            params.get("height", 544),             # Parameter mapping is fine here
            params.get("batch_size", 1),
            params.get("video_length", 25),
            params.get("fps", 24),
            params.get("infer_steps", 30),
            params.get("seed", -1),
            params.get("model", "hunyuan/mp_rank_00_model_states.pt"),
            params.get("vae", "hunyuan/pytorch_model.pt"),
            params.get("te1", "hunyuan/llava_llama3_fp16.safetensors"),
            params.get("te2", "hunyuan/clip_l.safetensors"),
            params.get("save_path", "outputs"),
            params.get("flow_shift", 11.0),
            params.get("cfg_scale", 7.0),
            params.get("output_type", "video"),
            params.get("attn_mode", "sdpa"),
            params.get("block_swap", "0"),
            *[params.get(f"lora{i+1}", "") for i in range(4)],
            *[params.get(f"lora{i+1}_multiplier", 1.0) for i in range(4)]
        ] if params else [gr.update()]*26, # This lambda returns values based on param keys
        inputs=params_state,
        outputs=[prompt, t2v_width, t2v_height, batch_size, video_length, fps, infer_steps, seed, # <<< CORRECTED HERE: use t2v_width, t2v_height
                 model, vae, te1, te2, save_path, flow_shift, cfg_scale,
                 output_type, attn_mode, block_swap] + lora_weights + lora_multipliers
    )
    # Text to Video generation
    generate_btn.click(
        fn=process_batch,
        inputs=[
            prompt, t2v_width, t2v_height, batch_size, video_length, fps, infer_steps,
            seed, dit_folder, model, vae, te1, te2, save_path, flow_shift, cfg_scale,
            output_type, attn_mode, block_swap, exclude_single_blocks, use_split_attn,
            lora_folder, *lora_weights, *lora_multipliers, gr.Textbox(visible=False), gr.Number(visible=False), use_fp8
        ],
        outputs=[video_output, batch_progress, progress_text],
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[batch_size],
        outputs=selected_index
    )    

    # Update gallery selection handling
    def handle_gallery_select(evt: gr.SelectData) -> int:
        return evt.index

    # Track selected index when gallery item is clicked
    video_output.select(
        fn=handle_gallery_select,
        outputs=selected_index
    )

    # Track selected index when Video2Video gallery item is clicked
    def handle_v2v_gallery_select(evt: gr.SelectData) -> int:
        """Handle gallery selection without automatically updating the input"""
        return evt.index

    # Update the gallery selection event
    v2v_output.select(
        fn=handle_v2v_gallery_select,
        outputs=v2v_selected_index
    )
    
    # Send button handler with gallery selection
    def handle_send_button(
        gallery: list, 
        prompt: str, 
        idx: int, 
        width: int,
        height: int,
        batch_size: int, 
        video_length: int, 
        fps: int, 
        infer_steps: int, 
        seed: int, 
        flow_shift: float, 
        cfg_scale: float,
        lora1: str,
        lora2: str,
        lora3: str,
        lora4: str,
        lora1_multiplier: float,
        lora2_multiplier: float,
        lora3_multiplier: float,
        lora4_multiplier: float
    ) -> tuple:
        if not gallery or idx is None or idx >= len(gallery):
            return (None, "", width, height, batch_size, video_length, fps, infer_steps, 
                    seed, flow_shift, cfg_scale, 
                    lora1, lora2, lora3, lora4,
                    lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier,
                    "")  # Add empty string for negative_prompt in the return values

        # Auto-select first item if only one exists and no selection made
        if idx is None and len(gallery) == 1:
            idx = 0

        selected_item = gallery[idx]

        # Handle different gallery item formats
        if isinstance(selected_item, dict):
            video_path = selected_item.get("name", selected_item.get("data", None))
        elif isinstance(selected_item, (tuple, list)):
            video_path = selected_item[0]
        else:
            video_path = selected_item

        # Final cleanup for Gradio Video component
        if isinstance(video_path, tuple):
            video_path = video_path[0]

        return (
            str(video_path), 
            prompt,
            width,
            height, 
            batch_size, 
            video_length, 
            fps, 
            infer_steps, 
            seed, 
            flow_shift, 
            cfg_scale,
            lora1,
            lora2,
            lora3,
            lora4,
            lora1_multiplier,
            lora2_multiplier,
            lora3_multiplier,
            lora4_multiplier,
            ""  # Add empty string for negative_prompt
        )
    
    send_t2v_to_v2v_btn.click(
        fn=handle_send_button,
        inputs=[
            video_output, prompt, selected_index,
            t2v_width, t2v_height, batch_size, video_length,
            fps, infer_steps, seed, flow_shift, cfg_scale
        ] + lora_weights + lora_multipliers,  # Remove the string here
        outputs=[
            v2v_input, 
            v2v_prompt,
            v2v_width,
            v2v_height,
            v2v_batch_size,
            v2v_video_length,
            v2v_fps,
            v2v_infer_steps,
            v2v_seed,
            v2v_flow_shift,
            v2v_cfg_scale
        ] + v2v_lora_weights + v2v_lora_multipliers + [v2v_negative_prompt]
    ).then(
        fn=change_to_tab_two, inputs=None, outputs=[tabs]
    )

    def handle_send_to_v2v(metadata: dict, video_path: str) -> Tuple[str, dict, str]:
        """Handle both parameters and video transfer"""
        status_msg, params = send_parameters_to_tab(metadata, "v2v")
        return status_msg, params, video_path
    
    def handle_info_to_v2v(metadata: dict, video_path: str) -> Tuple[str, Dict, str]:
        """Handle both parameters and video transfer from Video Info to V2V tab"""
        if not video_path:
            return "No video selected", {}, None

        status_msg, params = send_parameters_to_tab(metadata, "v2v")
        # Just return the path directly
        return status_msg, params, video_path

    # Send button click handler
    send_to_v2v_btn.click(
        fn=handle_info_to_v2v,
        inputs=[metadata_output, video_input],
        outputs=[status, params_state, v2v_input]
    ).then(
        lambda params: [
            params.get("v2v_prompt", ""),
            params.get("v2v_width", 544),
            params.get("v2v_height", 544),
            params.get("v2v_batch_size", 1),
            params.get("v2v_video_length", 25),
            params.get("v2v_fps", 24),
            params.get("v2v_infer_steps", 30),
            params.get("v2v_seed", -1),
            params.get("v2v_model", "hunyuan/mp_rank_00_model_states.pt"),
            params.get("v2v_vae", "hunyuan/pytorch_model.pt"),
            params.get("v2v_te1", "hunyuan/llava_llama3_fp16.safetensors"),
            params.get("v2v_te2", "hunyuan/clip_l.safetensors"),
            params.get("v2v_save_path", "outputs"),
            params.get("v2v_flow_shift", 11.0),
            params.get("v2v_cfg_scale", 7.0),
            params.get("v2v_output_type", "video"),
            params.get("v2v_attn_mode", "sdpa"),
            params.get("v2v_block_swap", "0"),
            *[params.get(f"v2v_lora_weights[{i}]", "") for i in range(4)],
            *[params.get(f"v2v_lora_multipliers[{i}]", 1.0) for i in range(4)]
        ] if params else [gr.update()] * 26,
        inputs=params_state,
        outputs=[
            v2v_prompt, v2v_width, v2v_height, v2v_batch_size, v2v_video_length,
            v2v_fps, v2v_infer_steps, v2v_seed, v2v_model, v2v_vae, v2v_te1,
            v2v_te2, v2v_save_path, v2v_flow_shift, v2v_cfg_scale, v2v_output_type,
            v2v_attn_mode, v2v_block_swap
        ] + v2v_lora_weights + v2v_lora_multipliers
    ).then(
        lambda: print(f"Tabs object: {tabs}"),  # Debug print
        outputs=None
    ).then(
        fn=change_to_tab_two, inputs=None, outputs=[tabs]
    )

    # Handler for sending selected video from Video2Video gallery to input
    def handle_v2v_send_button(gallery: list, prompt: str, idx: int) -> Tuple[Optional[str], str]:
        """Send the currently selected video in V2V gallery to V2V input"""
        if not gallery or idx is None or idx >= len(gallery):
            return None, ""

        selected_item = gallery[idx]
        video_path = None

        # Handle different gallery item formats
        if isinstance(selected_item, tuple):
            video_path = selected_item[0]  # Gallery returns (path, caption)
        elif isinstance(selected_item, dict):
            video_path = selected_item.get("name", selected_item.get("data", None))
        elif isinstance(selected_item, str):
            video_path = selected_item

        if not video_path:
            return None, ""

        # Check if the file exists and is accessible
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found at {video_path}")
            return None, ""

        return video_path, prompt

    v2v_send_to_input_btn.click(
        fn=handle_v2v_send_button,
        inputs=[v2v_output, v2v_prompt, v2v_selected_index],
        outputs=[v2v_input, v2v_prompt]
    ).then(
        lambda: gr.update(visible=True),  # Ensure the video input is visible
        outputs=v2v_input
    )

    # Video to Video generation
    v2v_generate_btn.click(
        fn=process_batch,
        inputs=[
            v2v_prompt, v2v_width, v2v_height, v2v_batch_size, v2v_video_length, 
            v2v_fps, v2v_infer_steps, v2v_seed, v2v_dit_folder, v2v_model, v2v_vae, v2v_te1, v2v_te2,
            v2v_save_path, v2v_flow_shift, v2v_cfg_scale, v2v_output_type, v2v_attn_mode, 
            v2v_block_swap, v2v_exclude_single_blocks, v2v_use_split_attn, v2v_lora_folder, 
            *v2v_lora_weights, *v2v_lora_multipliers, v2v_input, v2v_strength,
            v2v_negative_prompt, v2v_cfg_scale, v2v_split_uncond, v2v_use_fp8
        ],
        outputs=[v2v_output, v2v_batch_progress, v2v_progress_text],
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[v2v_batch_size],
        outputs=v2v_selected_index
    )
    refresh_outputs = [model]  # Add model dropdown to outputs
    for i in range(4):
        refresh_outputs.extend([lora_weights[i], lora_multipliers[i]])
    
    refresh_btn.click(
        fn=update_dit_and_lora_dropdowns,
        inputs=[dit_folder, lora_folder, model] + lora_weights + lora_multipliers,
        outputs=refresh_outputs
    )
    # Image2Video refresh
    i2v_refresh_outputs = [i2v_model]  # Add model dropdown to outputs
    for i in range(4):
        i2v_refresh_outputs.extend([i2v_lora_weights[i], i2v_lora_multipliers[i]])
    
    i2v_refresh_btn.click(
        fn=update_dit_and_lora_dropdowns,
        inputs=[i2v_dit_folder, i2v_lora_folder, i2v_model] + i2v_lora_weights + i2v_lora_multipliers,
        outputs=i2v_refresh_outputs
    )
    
    # Video2Video refresh
    v2v_refresh_outputs = [v2v_model]  # Add model dropdown to outputs
    for i in range(4):
        v2v_refresh_outputs.extend([v2v_lora_weights[i], v2v_lora_multipliers[i]])
    
    v2v_refresh_btn.click(
        fn=update_dit_and_lora_dropdowns,
        inputs=[v2v_dit_folder, v2v_lora_folder, v2v_model] + v2v_lora_weights + v2v_lora_multipliers,
        outputs=v2v_refresh_outputs
    )

    # WanX-i2v tab connections
    wanx_prompt.change(fn=count_prompt_tokens, inputs=wanx_prompt, outputs=wanx_token_counter)
    wanx_stop_btn.click(fn=lambda: stop_event.set(), queue=False)
    
    # Image input handling for WanX-i2v
    wanx_input.change(
        fn=update_wanx_image_dimensions,
        inputs=[wanx_input],
        outputs=[wanx_original_dims, wanx_width, wanx_height]
    )

    # Scale slider handling for WanX-i2v
    wanx_scale_slider.change(
        fn=update_wanx_from_scale,
        inputs=[wanx_scale_slider, wanx_original_dims],
        outputs=[wanx_width, wanx_height]
    )

    # Width/height calculation buttons for WanX-i2v
    wanx_calc_width_btn.click(
        fn=calculate_wanx_width,
        inputs=[wanx_height, wanx_original_dims],
        outputs=[wanx_width]
    )

    wanx_calc_height_btn.click(
        fn=calculate_wanx_height,
        inputs=[wanx_width, wanx_original_dims],
        outputs=[wanx_height]
    )
    # Add visibility toggle for the folder input components
    wanx_use_random_folder.change(
        fn=lambda x: (gr.update(visible=x), gr.update(visible=x), gr.update(visible=x), gr.update(visible=not x)),
        inputs=[wanx_use_random_folder],
        outputs=[wanx_input_folder, wanx_folder_status, wanx_validate_folder_btn, wanx_input]
    )
    def toggle_end_image(use_end_image):
        return (
            gr.update(visible=use_end_image, interactive=use_end_image),  # wanx_input_end
            gr.update(visible=False)  # wanx_trim_frames
        )
    wanx_use_end_image.change(
        fn=toggle_end_image,
        inputs=[wanx_use_end_image],
        outputs=[wanx_input_end, wanx_trim_frames]
    )
    # Validate folder button handler
    wanx_validate_folder_btn.click(
        fn=lambda folder: get_random_image_from_folder(folder)[1],
        inputs=[wanx_input_folder],
        outputs=[wanx_folder_status]
    )

    # Flow shift recommendation buttons
    wanx_recommend_flow_btn.click(
        fn=recommend_wanx_flow_shift,
        inputs=[wanx_width, wanx_height],
        outputs=[wanx_flow_shift]
    )

    wanx_t2v_recommend_flow_btn.click(
        fn=recommend_wanx_flow_shift,
        inputs=[wanx_t2v_width, wanx_t2v_height],
        outputs=[wanx_t2v_flow_shift]
    )
    
    # Generate button handler
    wanx_generate_btn.click(
        fn=wanx_batch_handler,
        inputs=[
            wanx_use_random_folder,
            wanx_prompt,
            wanx_negative_prompt,
            wanx_width,
            wanx_height,
            wanx_video_length,
            wanx_fps,
            wanx_infer_steps,
            wanx_flow_shift,
            wanx_guidance_scale,
            wanx_seed,
            wanx_batch_size,
            wanx_input_folder,
            wanx_input_end, # Make sure this is passed
            wanx_task,
            wanx_dit_folder,
            wanx_dit_path,
            wanx_vae_path,
            wanx_t5_path,
            wanx_clip_path,
            wanx_save_path,
            wanx_output_type,
            wanx_sample_solver,
            wanx_exclude_single_blocks,
            wanx_attn_mode,
            wanx_block_swap,
            wanx_fp8,
            wanx_fp8_scaled,
            wanx_fp8_t5,
            wanx_lora_folder,
            wanx_slg_layers,
            wanx_slg_start,
            wanx_slg_end,
            wanx_enable_cfg_skip,
            wanx_cfg_skip_mode,
            wanx_cfg_apply_ratio,
            # --- ADDED PREVIEW INPUTS ---
            wanx_enable_preview,
            wanx_preview_steps,
            # --- END ADDED ---
            *wanx_lora_weights,
            *wanx_lora_multipliers,
            wanx_input,              # Input image (used as input_file in handler)
            wanx_control_video,      # Control video
            wanx_control_strength,
            wanx_control_start,
            wanx_control_end,
        ],
        outputs=[
            wanx_output,          # Main video gallery
            wanx_preview_output,  # ADDED: Preview gallery
            wanx_batch_progress,  # Status text
            wanx_progress_text    # Progress text
        ], # Now 4 outputs
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[wanx_batch_size],
        outputs=wanx_i2v_selected_index
    )
    
    # Add refresh button handler for WanX-i2v tab
    wanx_refresh_outputs = [wanx_dit_path]  # Add model dropdown to outputs
    for i in range(4):
        wanx_refresh_outputs.extend([wanx_lora_weights[i], wanx_lora_multipliers[i]])

    wanx_refresh_btn.click(
        fn=update_dit_and_lora_dropdowns,  # This function already exists and handles both updates
        inputs=[wanx_dit_folder, wanx_lora_folder, wanx_dit_path] + wanx_lora_weights + wanx_lora_multipliers,
        outputs=wanx_refresh_outputs
    )
    wanx_dit_folder.change(
        fn=update_dit_dropdown,
        inputs=[wanx_dit_folder],
        outputs=[wanx_dit_path]
    )

    wanx_dit_folder.change(
        fn=update_dit_dropdown,
        inputs=[wanx_dit_folder],
        outputs=[wanx_t2v_dit_path]
    )

    wanx_dit_folder.change(
        fn=update_dit_dropdown,
        inputs=[wanx_dit_folder],
        outputs=[wanx_v2v_dit_path]
    )
    
    # Gallery selection handling
    wanx_output.select(
        fn=handle_wanx_gallery_select,
        inputs=[wanx_output],
        outputs=[wanx_i2v_selected_index, wanx_base_video]
    )
    
    # Send to Video2Video handler
    wanx_send_to_v2v_btn.click(
        fn=send_wanx_to_v2v,
        inputs=[
            wanx_output,  # Gallery with videos
            wanx_prompt,  # Prompt text
            wanx_i2v_selected_index,  # Use the correct selected index state
            wanx_width, 
            wanx_height, 
            wanx_video_length,
            wanx_fps, 
            wanx_infer_steps, 
            wanx_seed,
            wanx_flow_shift, 
            wanx_guidance_scale,
            wanx_negative_prompt
        ],
        outputs=[
            v2v_input,  # Video input in V2V tab
            v2v_prompt,  # Prompt in V2V tab
            v2v_width, 
            v2v_height,
            v2v_video_length, 
            v2v_fps, 
            v2v_infer_steps,
            v2v_seed, 
            v2v_flow_shift, 
            v2v_cfg_scale,
            v2v_negative_prompt
        ]
    ).then(
        fn=change_to_tab_two,  # Function to switch to Video2Video tab
        inputs=None,
        outputs=[tabs]
    )
    # Connect prompt token counter
    wanx_t2v_prompt.change(fn=count_prompt_tokens, inputs=wanx_t2v_prompt, outputs=wanx_t2v_token_counter)

    # Stop button handler
    wanx_t2v_stop_btn.click(fn=lambda: stop_event.set(), queue=False)

    # Flow shift recommendation button
    wanx_t2v_recommend_flow_btn.click(
        fn=recommend_wanx_flow_shift,
        inputs=[wanx_t2v_width, wanx_t2v_height],
        outputs=[wanx_t2v_flow_shift]
    )

    # Task change handler to update CLIP visibility and path
    def update_clip_visibility(task):
        is_i2v = "i2v" in task
        return gr.update(visible=is_i2v)

    wanx_t2v_task.change(
        fn=update_clip_visibility,
        inputs=[wanx_t2v_task],
        outputs=[wanx_t2v_clip_path]
    )

        # Generate button handler for T2V
    wanx_t2v_generate_btn.click(
        fn=wanx_batch_handler,
        inputs=[
            wanx_t2v_use_random_folder, # use_random
            wanx_t2v_prompt,            # prompt
            wanx_t2v_negative_prompt,   # negative_prompt
            wanx_t2v_width,             # width
            wanx_t2v_height,            # height
            wanx_t2v_video_length,      # video_length
            wanx_t2v_fps,               # fps
            wanx_t2v_infer_steps,       # infer_steps
            wanx_t2v_flow_shift,        # flow_shift
            wanx_t2v_guidance_scale,    # guidance_scale
            wanx_t2v_seed,              # seed
            wanx_t2v_batch_size,        # batch_size
            wanx_t2v_input_folder,      # input_folder_path
            wanx_t2v_input_end,         # wanx_input_end
            wanx_t2v_task,              # task
            wanx_dit_folder,            # dit_folder (shared)
            wanx_t2v_dit_path,          # dit_path
            wanx_t2v_vae_path,          # vae_path
            wanx_t2v_t5_path,           # t5_path
            wanx_t2v_clip_path,         # clip_path (often None for t2v)
            wanx_t2v_save_path,         # save_path
            wanx_t2v_output_type,       # output_type
            wanx_t2v_sample_solver,     # sample_solver
            wanx_t2v_exclude_single_blocks, # exclude_single_blocks
            wanx_t2v_attn_mode,         # attn_mode
            wanx_t2v_block_swap,        # block_swap
            wanx_t2v_fp8,               # fp8
            wanx_t2v_fp8_scaled,        # fp8_scaled
            wanx_t2v_fp8_t5,            # fp8_t5
            wanx_t2v_lora_folder,       # lora_folder
            wanx_t2v_slg_layers,        # slg_layers
            wanx_t2v_slg_start,         # slg_start
            wanx_t2v_slg_end,           # slg_end
            wanx_t2v_enable_cfg_skip,   # enable_cfg_skip
            wanx_t2v_cfg_skip_mode,     # cfg_skip_mode
            wanx_t2v_cfg_apply_ratio,   # cfg_apply_ratio
            # --- ADDED PREVIEW INPUTS ---
            wanx_t2v_enable_preview,
            wanx_t2v_preview_steps,
            # --- END ADDED ---
            *wanx_t2v_lora_weights,     # *lora_params (weights)
            *wanx_t2v_lora_multipliers, # *lora_params (multipliers)
            # --- ADDED Placeholders for trailing args expected by wanx_batch_handler ---
            gr.File(value=None, visible=False), # Placeholder for input_file (None for T2V)
            gr.Video(value=None, visible=False), # Placeholder for control_video (None for T2V)
            gr.Number(value=1.0, visible=False), # Placeholder for control_strength
            gr.Number(value=0.0, visible=False), # Placeholder for control_start
            gr.Number(value=1.0, visible=False), # Placeholder for control_end
            # --- END Placeholders ---
        ],
        outputs=[
            wanx_t2v_output,         # Main video gallery
            wanx_t2v_preview_output, # ADDED: Preview gallery
            wanx_t2v_batch_progress, # Status text
            wanx_t2v_progress_text   # Progress text
        ], # Now 4 outputs
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[wanx_t2v_batch_size],
        outputs=wanx_t2v_selected_index
    )
    
    # Add refresh button handler for WanX-t2v tab
    wanx_t2v_refresh_outputs = [wanx_t2v_dit_path]  # This is one output
    for i in range(4):
        wanx_t2v_refresh_outputs.extend([wanx_t2v_lora_weights[i], wanx_t2v_lora_multipliers[i]])  # This adds 8 more outputs

    wanx_t2v_refresh_btn.click(
        fn=update_dit_and_lora_dropdowns,  # Change to this function instead
        inputs=[wanx_dit_folder, wanx_t2v_lora_folder, wanx_t2v_dit_path] + wanx_t2v_lora_weights + wanx_t2v_lora_multipliers,
        outputs=wanx_t2v_refresh_outputs
    )

    # Gallery selection handling
    wanx_t2v_output.select(
        fn=handle_wanx_t2v_gallery_select,
        outputs=wanx_t2v_selected_index
    )

    # Send to Video2Video handler
    wanx_t2v_send_to_v2v_btn.click(
        fn=send_wanx_t2v_to_v2v,
        inputs=[
            wanx_t2v_output, 
            wanx_t2v_prompt, 
            wanx_t2v_selected_index,
            wanx_t2v_width, 
            wanx_t2v_height, 
            wanx_t2v_video_length,
            wanx_t2v_fps, 
            wanx_t2v_infer_steps, 
            wanx_t2v_seed,
            wanx_t2v_flow_shift, 
            wanx_t2v_guidance_scale,
            wanx_t2v_negative_prompt
        ],
        outputs=[
            v2v_input, 
            v2v_prompt, 
            v2v_width, 
            v2v_height,
            v2v_video_length, 
            v2v_fps, 
            v2v_infer_steps,
            v2v_seed, 
            v2v_flow_shift, 
            v2v_cfg_scale,
            v2v_negative_prompt
        ]
    ).then(
        fn=change_to_tab_two,
        inputs=None,
        outputs=[tabs]
    )
    # Phantom Tab Event Handlers
    phantom_prompt.change(fn=count_prompt_tokens, inputs=phantom_prompt, outputs=phantom_token_counter)
    phantom_stop_btn.click(fn=lambda: stop_event.set(), queue=False)

    phantom_recommend_flow_btn.click(
        fn=recommend_wanx_flow_shift, # Reusing WanX logic as it's dimension-based
        inputs=[phantom_width, phantom_height],
        outputs=[phantom_flow_shift]
    )

    phantom_generate_btn.click(
        fn=phantom_batch_handler,
        inputs=[
            phantom_prompt, phantom_negative_prompt, phantom_input_images,
            phantom_width, phantom_height, phantom_video_length, phantom_fps, phantom_infer_steps,
            phantom_flow_shift, phantom_guide_scale_img, phantom_guide_scale_text, phantom_cfg_scale_main,
            phantom_seed, phantom_batch_size, phantom_task, phantom_dit_folder, phantom_dit_path,
            phantom_vae_path, phantom_t5_path, phantom_save_path, phantom_output_type,
            phantom_sample_solver, phantom_attn_mode, phantom_block_swap,
            phantom_fp8, phantom_fp8_scaled, phantom_fp8_t5, phantom_lora_folder,
            phantom_slg_layers, phantom_slg_start, phantom_slg_end,
            phantom_enable_cfg_skip, phantom_cfg_skip_mode, phantom_cfg_apply_ratio,
            phantom_enable_preview, phantom_preview_steps,
            *phantom_lora_weights_ui, *phantom_lora_multipliers_ui
        ],
        outputs=[phantom_output, phantom_preview_output, phantom_batch_progress, phantom_progress_text],
        queue=True
    ).then(
        fn=lambda b_size: 0 if int(b_size) == 1 else None,
        inputs=[phantom_batch_size],
        outputs=phantom_selected_index
    )

    phantom_refresh_outputs_list = [phantom_dit_path]
    for i in range(len(phantom_lora_weights_ui)):
        phantom_refresh_outputs_list.extend([phantom_lora_weights_ui[i], phantom_lora_multipliers_ui[i]])

    phantom_refresh_btn.click(
        fn=update_dit_and_lora_dropdowns,
        inputs=[phantom_dit_folder, phantom_lora_folder, phantom_dit_path] + phantom_lora_weights_ui + phantom_lora_multipliers_ui,
        outputs=phantom_refresh_outputs_list
    )
    phantom_dit_folder.change(fn=update_dit_dropdown, inputs=[phantom_dit_folder], outputs=[phantom_dit_path])


    def handle_phantom_gallery_select(evt: gr.SelectData) -> int:
        return evt.index
    phantom_output.select(fn=handle_phantom_gallery_select, outputs=phantom_selected_index)

    phantom_send_to_hunyuan_v2v_btn.click(
        fn=send_wanx_to_v2v, # Re-use existing send function logic for parameter transfer
        inputs=[
            phantom_output, phantom_prompt, phantom_selected_index,
            phantom_width, phantom_height, phantom_video_length, phantom_fps, phantom_infer_steps,
            phantom_seed, phantom_flow_shift, phantom_guide_scale_text, # Use text guide scale as primary for V2V prompt
            phantom_negative_prompt
        ],
        outputs=[
            v2v_input, v2v_prompt, v2v_width, v2v_height, v2v_video_length,
            v2v_fps, v2v_infer_steps, v2v_seed, v2v_flow_shift, v2v_cfg_scale,
            v2v_negative_prompt
        ]
    ).then(fn=change_to_tab_two, inputs=None, outputs=[tabs])

    phantom_send_to_wanx_v2v_btn.click(
        fn=send_wanx_v2v_to_hunyuan_v2v, # Re-use existing send function logic
        inputs=[
            phantom_output, phantom_prompt, phantom_selected_index,
            phantom_width, phantom_height, phantom_video_length, phantom_fps, phantom_infer_steps,
            phantom_seed, phantom_flow_shift, phantom_guide_scale_text, # Use text guide scale
            phantom_negative_prompt
        ],
        outputs=[
            wanx_v2v_input, wanx_v2v_prompt, wanx_v2v_width, wanx_v2v_height,
            wanx_v2v_video_length, wanx_v2v_fps, wanx_v2v_infer_steps,
            wanx_v2v_seed, wanx_v2v_flow_shift, wanx_v2v_guidance_scale, # wanx_v2v uses 'guidance_scale'
            wanx_v2v_negative_prompt
        ]
    ).then(fn=change_to_wanx_v2v_tab, inputs=None, outputs=[tabs])

    phantom_enable_cfg_skip.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[phantom_enable_cfg_skip],
        outputs=[phantom_cfg_skip_options]
    )
    phantom_add_more_images_btn.upload(
        fn=combine_image_sets,
        inputs=[phantom_input_images, phantom_add_more_images_btn],
        outputs=[phantom_input_images],
        show_progress="hidden" # Can be "full" or "minimal" if you want progress for upload
    )

if __name__ == "__main__":
    # Make sure 'outputs' directory exists
    os.makedirs("outputs", exist_ok=True)
    # Optional: Clean temp_frames directory on startup
    #if os.path.exists("temp_frames"):
    #    try: shutil.rmtree("temp_frames")
    #    except OSError as e: print(f"Error removing temp_frames: {e}")
    os.makedirs("temp_frames", exist_ok=True)

demo.queue().launch(server_name="0.0.0.0", share=False)