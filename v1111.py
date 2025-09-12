import gradio as gr
from gradio import update as gr_update
import subprocess
import threading
import time
import re
import os
import random
import sys
import ffmpeg
from typing import List, Tuple, Optional, Generator, Dict, Any
import json
from gradio import themes
from gradio.themes.utils import colors
from PIL import Image
import math
import cv2
import glob
import shutil
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import torch

# Add global stop event
stop_event = threading.Event()
skip_event = threading.Event()
logger = logging.getLogger(__name__)

# VACE UI Configuration
UI_CONFIGS_DIR = "ui_configs"
VACE_PRESETS_FILE = os.path.join(UI_CONFIGS_DIR, "vace_presets.json")

# Helper functions for VACE model detection (matching h1111.py style)
def get_vace_dit_models(dit_folder: str, filter_name: str = "") -> List[str]:
    """Get filtered DiT models for VACE"""
    if not os.path.exists(dit_folder):
        return ["vace_14b_low_noise_bf16.safetensors"]
    models = [f for f in os.listdir(dit_folder) if f.endswith('.safetensors')]
    if filter_name:
        models = [m for m in models if filter_name.lower() in m.lower()]
    models.sort(key=str.lower)
    return models if models else ["vace_14b_low_noise_bf16.safetensors"]

def get_vace_low_noise_models(dit_folder: str) -> List[str]:
    """Get low noise DiT models for VACE"""
    if not os.path.exists(dit_folder):
        return ["wan22_i2v_14B_low_noise_bf16.safetensors"]
    models = [f for f in os.listdir(dit_folder) if f.endswith('.safetensors')]
    # Look for models with 'low' in the name
    low_models = [m for m in models if 'low' in m.lower()]
    low_models.sort(key=str.lower)
    return low_models if low_models else ["wan22_i2v_14B_low_noise_bf16.safetensors"]

def get_vace_high_noise_models(dit_folder: str) -> List[str]:
    """Get high noise DiT models for VACE"""
    if not os.path.exists(dit_folder):
        return ["wan22_i2v_14B_high_noise_bf16.safetensors"]
    models = [f for f in os.listdir(dit_folder) if f.endswith('.safetensors')]
    # Look for models with 'high' in the name
    high_models = [m for m in models if 'high' in m.lower()]
    high_models.sort(key=str.lower)
    return high_models if high_models else ["wan22_i2v_14B_high_noise_bf16.safetensors"]

def get_vace_clip_models(dit_folder: str) -> List[str]:
    """Get CLIP models for VACE I2V"""
    if not os.path.exists(dit_folder):
        return ["models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]
    models = [f for f in os.listdir(dit_folder) 
              if (f.endswith('.pth') or f.endswith('.safetensors')) and 'clip' in f.lower()]
    models.sort(key=str.lower)
    return models if models else ["models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]

def get_vace_vae_models(dit_folder: str) -> List[str]:
    """Get VAE models for VACE"""
    if not os.path.exists(dit_folder):
        return ["Wan2.1_VAE.pth"]
    models = [f for f in os.listdir(dit_folder) if f.endswith('.pth') and 'vae' in f.lower()]
    models.sort(key=str.lower)
    return models if models else ["Wan2.1_VAE.pth"]

def get_vace_t5_models(dit_folder: str) -> List[str]:
    """Get T5 text encoder models for VACE"""
    if not os.path.exists(dit_folder):
        return ["models_t5_umt5-xxl-enc-bf16.pth"]
    models = [f for f in os.listdir(dit_folder) 
              if (f.endswith('.pth') or f.endswith('.safetensors')) and 't5' in f.lower()]
    models.sort(key=str.lower)
    return models if models else ["models_t5_umt5-xxl-enc-bf16.pth"]

# Helper functions to get default models
def get_default_low_noise_model(dit_folder: str = "wan") -> str:
    """Get the first available low noise model as default"""
    models = get_vace_low_noise_models(dit_folder)
    return models[0] if models else "wan22_i2v_14B_low_noise_bf16.safetensors"

def get_default_high_noise_model(dit_folder: str = "wan") -> str:
    """Get the first available high noise model as default"""
    models = get_vace_high_noise_models(dit_folder)
    return models[0] if models else "wan22_i2v_14B_high_noise_bf16.safetensors"

def get_default_clip_model(dit_folder: str = "wan") -> str:
    """Get the first available CLIP model as default"""
    models = get_vace_clip_models(dit_folder)
    return models[0] if models else "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"

def get_default_vae_model(dit_folder: str = "wan") -> str:
    """Get the first available VAE model as default"""
    models = get_vace_vae_models(dit_folder)
    return models[0] if models else "Wan2.1_VAE.pth"

def get_default_t5_model(dit_folder: str = "wan") -> str:
    """Get the first available T5 model as default"""
    models = get_vace_t5_models(dit_folder)
    return models[0] if models else "models_t5_umt5-xxl-enc-bf16.pth"

def get_lora_files(lora_folder: str = "lora") -> List[str]:
    """Get available LoRA files"""
    if not os.path.exists(lora_folder):
        return ["None"]
    
    loras = ["None"]  # Always include None option
    loras.extend([f for f in os.listdir(lora_folder) 
                  if f.endswith('.safetensors') or f.endswith('.pth')])
    loras[1:] = sorted(loras[1:], key=str.lower)  # Sort all except "None"
    return loras

# Video processing utilities
def get_video_info(video_path: str) -> Dict[str, Any]:
    """Extract video information and metadata"""
    if not video_path or not os.path.exists(video_path):
        return {}
    
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        if video_stream:
            return {
                "width": int(video_stream['width']),
                "height": int(video_stream['height']),
                "fps": eval(video_stream['r_frame_rate']),
                "duration": float(video_stream.get('duration', 0)),
                "frames": int(video_stream.get('nb_frames', 0)),
                "codec": video_stream.get('codec_name', 'unknown')
            }
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
    
    return {}

def update_image_dimensions(image):
    """Update dimensions from uploaded image"""
    if image is None:
        return "", gr.update(value=832), gr.update(value=480)
    
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image
        
    w, h = img.size
    w = (w // 32) * 32
    h = (h // 32) * 32
    return f"{w}x{h}", w, h

def calculate_width_from_height(height, original_dims):
    """Calculate width based on height maintaining aspect ratio"""
    if not original_dims:
        return gr.update()
    orig_w, orig_h = map(int, original_dims.split('x'))
    aspect_ratio = orig_w / orig_h
    new_width = math.floor((height * aspect_ratio) / 32) * 32
    return gr.update(value=new_width)

def calculate_height_from_width(width, original_dims):
    """Calculate height based on width maintaining aspect ratio"""
    if not original_dims:
        return gr.update()
    orig_w, orig_h = map(int, original_dims.split('x'))
    aspect_ratio = orig_w / orig_h
    new_height = math.floor((width / aspect_ratio) / 32) * 32
    return gr.update(value=new_height)

def count_prompt_tokens(prompt):
    """Estimate token count (simplified)"""
    return len(prompt.split()) * 1.3

# Main VACE generation function updated for vace_generate_video.py
def run_vace_generation(
    mode: str,
    *args
) -> Generator[Tuple[List, str], None, None]:
    """
    Run VACE video generation with progress updates using vace_generate_video.py
    """
    # Unpack args based on mode
    idx = 0
    prompt = args[idx]; idx += 1
    negative_prompt = args[idx]; idx += 1
    width = args[idx]; idx += 1
    height = args[idx]; idx += 1
    video_length = args[idx]; idx += 1
    fps = args[idx]; idx += 1
    seed = args[idx]; idx += 1
    infer_steps = args[idx]; idx += 1
    guidance_scale = args[idx]; idx += 1
    flow_shift = args[idx]; idx += 1
    
    # VACE optimization parameters
    enable_teacache = args[idx]; idx += 1
    teacache_threshold = args[idx]; idx += 1
    cfg_skip_ratio = args[idx]; idx += 1
    enable_riflex = args[idx]; idx += 1
    riflex_k = args[idx]; idx += 1
    
    # Model paths
    dit_low_noise = args[idx]; idx += 1
    dit_high_noise = args[idx]; idx += 1
    vae_model = args[idx]; idx += 1
    t5_model = args[idx]; idx += 1
    
    # LoRA arrays (8 each)
    lora_weights = args[idx:idx+8]; idx += 8
    lora_multipliers = args[idx:idx+8]; idx += 8
    lora_apply_low = args[idx:idx+8]; idx += 8
    lora_apply_high = args[idx:idx+8]; idx += 8
    
    batch_size = args[idx]; idx += 1
    dual_dit_boundary = args[idx]; idx += 1
    attn_mode = args[idx]; idx += 1
    block_swap = args[idx]; idx += 1
    fp8 = args[idx]; idx += 1
    fp8_scaled = args[idx]; idx += 1
    fp8_t5 = args[idx]; idx += 1
    vae_fp32 = args[idx]; idx += 1
    model_folder = args[idx]; idx += 1
    lora_folder = args[idx]; idx += 1
    save_path = args[idx]; idx += 1
    
    # Optional mode-specific parameters
    start_image = args[idx] if idx < len(args) else None; idx += 1
    end_image = args[idx] if idx < len(args) else None; idx += 1
    clip_model = args[idx] if idx < len(args) else None; idx += 1
    subject_ref_images = args[idx] if idx < len(args) else None; idx += 1
    control_video = args[idx] if idx < len(args) else None; idx += 1
    padding_ref_images = args[idx] if idx < len(args) else True; idx += 1
    control_strength = args[idx] if idx < len(args) else 0.7; idx += 1
    vace_context_scale = args[idx] if idx < len(args) else 1.0
    
    global stop_event
    stop_event.clear()
    all_generated_videos = []
    
    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("temp_frames", exist_ok=True)
    
    for i in range(int(batch_size)):
        if stop_event.is_set():
            yield all_generated_videos.copy(), "Generation stopped by user"
            return
            
        current_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        if batch_size > 1 and seed >= 0:
            current_seed = seed + i
        
        status_text = f"Processing item {i+1}/{int(batch_size)} (Seed: {current_seed})"
        yield all_generated_videos.copy(), status_text
        
        # Construct command for vace_generate_video.py
        run_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
        unique_suffix = f"vace_{run_id}"
        
        # Map mode to task - use VACE tasks when VACE models are detected
        # Check if we're using VACE models (they have "VACE" in the name)
        is_vace = (dit_low_noise and "vace" in dit_low_noise.lower()) or \
                  (dit_high_noise and "vace" in dit_high_noise.lower())
        
        if is_vace:
            task_map = {
                "t2v": "vace-t2v-A14B",
                "i2v": "vace-i2v-A14B",
                "s2v": "vace-s2v-A14B",
                "v2v": "vace-v2v-A14B"
            }
        else:
            task_map = {
                "t2v": "t2v-A14B",
                "i2v": "i2v-A14B",
                "s2v": "i2v-A14B",  # Use i2v for s2v in non-VACE
                "v2v": "i2v-A14B"  # V2V uses i2v with video input
            }
        
        cmd = [
            sys.executable, "vace_generate_video.py",
            "--task", task_map.get(mode, "t2v-A14B"),
            "--prompt", str(prompt),
            "--negative_prompt", str(negative_prompt) if negative_prompt else "",
            "--video_size", str(int(height)), str(int(width)),
            "--video_length", str(int(video_length)),
            "--fps", str(int(fps)),
            "--seed", str(current_seed),
            "--infer_steps", str(int(infer_steps)),
            "--guidance_scale", str(guidance_scale),
            "--flow_shift", str(flow_shift),
            "--dual_dit_boundary", str(dual_dit_boundary),
            "--attn_mode", attn_mode,
            "--blocks_to_swap", str(int(block_swap)),
            "--save_path", save_path,
        ]
        
        # Add dual DiT models with full paths
        if dit_low_noise and dit_low_noise != "None":
            cmd.extend(["--dit_low_noise", os.path.join(model_folder, dit_low_noise)])
        if dit_high_noise and dit_high_noise != "None":
            cmd.extend(["--dit_high_noise", os.path.join(model_folder, dit_high_noise)])
        
        # Add VAE and T5 models
        if vae_model and vae_model != "None":
            cmd.extend(["--vae", os.path.join(model_folder, vae_model)])
        if t5_model and t5_model != "None":
            cmd.extend(["--t5", os.path.join(model_folder, t5_model)])
        
        # Add CLIP for I2V modes
        if mode in ["i2v", "s2v"] and clip_model and clip_model != "None":
            cmd.extend(["--clip", os.path.join(model_folder, clip_model)])
        
        # Add performance options
        if fp8:
            cmd.append("--fp8")
        if fp8_scaled:
            cmd.append("--fp8_scaled")
        if fp8_t5:
            cmd.append("--fp8_t5")
        if not vae_fp32:
            cmd.extend(["--vae_dtype", "bfloat16"])
        
        # Add VACE optimization parameters
        if enable_teacache:
            cmd.extend(["--enable_teacache", "--teacache_threshold", str(teacache_threshold)])
        
        if cfg_skip_ratio > 0:
            cmd.extend(["--cfg_skip_ratio", str(cfg_skip_ratio)])
        
        if enable_riflex:
            cmd.extend(["--enable_riflex", "--riflex_k", str(int(riflex_k))])
        
        # Handle LoRA weights with dual model support
        if lora_weights and lora_multipliers and lora_apply_low and lora_apply_high:
            lora_low_list = []
            lora_high_list = []
            lora_low_mult = []
            lora_high_mult = []
            
            for idx, (lora_file, mult, apply_low, apply_high) in enumerate(
                zip(lora_weights, lora_multipliers, lora_apply_low, lora_apply_high)
            ):
                if lora_file and lora_file != "None":
                    lora_path = os.path.join(lora_folder, lora_file)
                    if apply_low:
                        lora_low_list.append(lora_path)
                        lora_low_mult.append(str(mult))
                    if apply_high:
                        lora_high_list.append(lora_path)
                        lora_high_mult.append(str(mult))
            
            if lora_low_list:
                cmd.extend(["--lora_weight"] + lora_low_list)
                cmd.extend(["--lora_multiplier"] + lora_low_mult)
            if lora_high_list:
                cmd.extend(["--lora_weight_high"] + lora_high_list)
                cmd.extend(["--lora_multiplier_high"] + lora_high_mult)
        
        # Add mode-specific inputs
        if mode == "i2v" and start_image:
            cmd.extend(["--image_path", start_image])
            
            # Check for FLF mode (first-last frame)
            if end_image:
                cmd.extend(["--end_image_path", end_image])
        
        elif mode == "s2v" and subject_ref_images:
            # Handle subject reference images
            if subject_ref_images:
                for ref_img in subject_ref_images:
                    if ref_img:
                        cmd.extend(["--subject_ref_images", ref_img])
                if padding_ref_images:
                    cmd.append("--padding_subject_ref_images")
        
        elif mode == "v2v":
            # Handle control video (using video_path parameter)
            if control_video:
                cmd.extend(["--control_video", control_video])
            if control_strength is not None:
                cmd.extend(["--strength", str(control_strength)])
            
            # Add VACE-specific V2V parameters
            if vace_context_scale is not None and vace_context_scale != 1.0:
                cmd.extend(["--vace_context_scale", str(vace_context_scale)])
            
            # Add start/end images if provided (V2V can use i2v-style conditioning)
            if start_image:
                cmd.extend(["--image_path", start_image])
            if end_image:
                cmd.extend(["--end_image_path", end_image])
            
            # Add subject reference images if provided
            if subject_ref_images:
                for ref_img in subject_ref_images:
                    if ref_img:
                        cmd.extend(["--subject_ref_images", ref_img])
                if padding_ref_images:
                    cmd.append("--padding_in_subject_ref_images")
        
        # Run generation with progress monitoring
        try:
            print(f"Running command: {' '.join(cmd)}")  # Full command debug print
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor progress
            for line in process.stdout:
                line = line.strip()
                if line:
                    print(line)  # Debug output
                    
                # Check for progress updates
                if "Step" in line or "step" in line or "%" in line:
                    # Try to extract step info
                    step_match = re.search(r'(\d+)/(\d+)', line)
                    if step_match:
                        current_step = int(step_match.group(1))
                        total_steps = int(step_match.group(2))
                        progress_text = f"Step {current_step}/{total_steps}"
                        status_text = f"Item {i+1}/{int(batch_size)}: {progress_text}"
                        yield all_generated_videos.copy(), status_text
                
                if stop_event.is_set():
                    process.terminate()
                    process.wait(timeout=5)
                    yield all_generated_videos.copy(), "Generation stopped by user"
                    return
            
            process.wait()
            
            if process.returncode == 0:
                # Find generated video
                all_videos = glob.glob(os.path.join(save_path, "*.mp4"))
                if all_videos:
                    # Get the newest video
                    latest_video = max(all_videos, key=os.path.getmtime)
                    all_generated_videos.append(latest_video)
                    status_text = f"Completed item {i+1}/{int(batch_size)}"
                    print(f"Generated video: {latest_video}")
                else:
                    status_text = f"Item {i+1} completed but no output found"
                    print(f"Warning: No output video found")
            else:
                status_text = f"Item {i+1} failed with return code {process.returncode}"
                print(f"Error: Process failed with return code {process.returncode}")
                
        except Exception as e:
            status_text = f"Error in item {i+1}: {str(e)}"
            logger.error(f"Generation error: {e}")
            print(f"Exception: {e}")
        
        yield all_generated_videos.copy(), status_text
        
        # Clean up temp files
        try:
            for temp_file in glob.glob(f"temp_frames/*{run_id}*"):
                os.remove(temp_file)
        except:
            pass
    
    final_status = f"Batch complete: {len(all_generated_videos)} videos generated"
    yield all_generated_videos, final_status

# Main UI Application with h1111.py structure
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
        # VACE T2V Tab - Text to Video
        with gr.Tab("VACE T2V", id=0):
            with gr.Row():
                with gr.Column(scale=4):
                    t2v_prompt = gr.Textbox(
                        scale=3,
                        label="Enter your prompt",
                        value="A magical forest with glowing butterflies dancing in the moonlight.",
                        lines=5
                    )
                    t2v_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt",
                        value="low quality, blurry, static, distorted, deformed",
                        lines=3
                    )
                
                with gr.Column(scale=1):
                    t2v_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    t2v_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)
                
                with gr.Column(scale=2):
                    t2v_batch_progress = gr.Textbox(label="Status", interactive=False, value="")
                    t2v_progress_text = gr.Textbox(label="Progress", interactive=False, value="")
            
            with gr.Row():
                t2v_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                t2v_stop_btn = gr.Button("Stop Generation", variant="stop")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Generation Parameters")
                    with gr.Row():
                        t2v_width = gr.Number(label="Width", value=832, step=32)
                        t2v_calc_height_btn = gr.Button("→")
                        t2v_calc_width_btn = gr.Button("←")
                        t2v_height = gr.Number(label="Height", value=480, step=32)
                    
                    t2v_original_dims = gr.Textbox(label="Original Dimensions", visible=False)
                    t2v_video_length = gr.Slider(minimum=9, maximum=161, step=4, label="Frame Count", value=81, info="Must be 4n+1")
                    t2v_fps = gr.Slider(minimum=8, maximum=30, step=1, label="Frames Per Second", value=16)
                    t2v_infer_steps = gr.Slider(minimum=20, maximum=100, step=1, label="Sampling Steps", value=50)
                    t2v_flow_shift = gr.Slider(minimum=0.0, maximum=20.0, step=0.1, label="Flow Shift", value=5.0)
                    t2v_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale", value=7.0)
                    t2v_dual_dit_boundary = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Dual-DiT Boundary", value=0.875, info="Low noise model used after this threshold")
                    
                    with gr.Row():
                        t2v_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                        t2v_random_seed_btn = gr.Button("🎲")
                    
                    # VACE Optimizations Accordion
                    with gr.Accordion("VACE Optimizations", open=False):
                        with gr.Row():
                            t2v_enable_teacache = gr.Checkbox(value=True, label="Enable TeaCache")
                            t2v_teacache_threshold = gr.Slider(0.05, 0.30, value=0.10, step=0.01, label="TeaCache Threshold")
                        
                        t2v_cfg_skip_ratio = gr.Slider(0.0, 0.25, value=0.0, step=0.05, label="CFG Skip Ratio (0=disabled)")
                        
                        with gr.Row():
                            t2v_enable_riflex = gr.Checkbox(value=False, label="Enable RiFlex")
                            t2v_riflex_k = gr.Slider(1, 10, value=6, step=1, label="RiFlex K", visible=False)
                
                with gr.Column():
                    t2v_output = gr.Gallery(
                        label="Generated Videos (Click to select)",
                        columns=[2], rows=[2], object_fit="contain", height="auto",
                        show_label=True, elem_id="gallery_vace_t2v", allow_preview=True, preview=True
                    )
                    
                    # LoRA Configuration
                    with gr.Accordion("LoRA Configuration", open=True):
                        with gr.Row():
                            t2v_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                            t2v_lora_refresh_btn = gr.Button("🔄 LoRA", elem_classes="refresh-btn")
                        
                        t2v_lora_weights = []
                        t2v_lora_multipliers = []
                        t2v_lora_apply_low = []
                        t2v_lora_apply_high = []
                        
                        for i in range(4):
                            with gr.Row():
                                t2v_lora_weights.append(gr.Dropdown(
                                    label=f"LoRA {i+1}", choices=get_lora_files("lora"),
                                    value="None", allow_custom_value=False, interactive=True, scale=2
                                ))
                                t2v_lora_multipliers.append(gr.Slider(
                                    label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1
                                ))
                            with gr.Row():
                                t2v_lora_apply_low.append(gr.Checkbox(label="Apply to Low Noise", value=True, scale=1))
                                t2v_lora_apply_high.append(gr.Checkbox(label="Apply to High Noise", value=False, scale=1))
                        
                        with gr.Accordion("Additional LoRAs (5-8)", open=False):
                            for i in range(4, 8):
                                with gr.Row():
                                    t2v_lora_weights.append(gr.Dropdown(
                                        label=f"LoRA {i+1}", choices=get_lora_files("lora"),
                                        value="None", allow_custom_value=False, interactive=True, scale=2
                                    ))
                                    t2v_lora_multipliers.append(gr.Slider(
                                        label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1
                                    ))
                                with gr.Row():
                                    t2v_lora_apply_low.append(gr.Checkbox(label="Apply to Low Noise", value=True, scale=1))
                                    t2v_lora_apply_high.append(gr.Checkbox(label="Apply to High Noise", value=False, scale=1))
            
            # Model Paths & Performance section
            with gr.Accordion("Model Paths & Performance", open=True):
                with gr.Row():
                    t2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "torch", "xformers"], label="Attention Mode", value="sdpa")
                    t2v_block_swap = gr.Slider(minimum=0, maximum=39, step=1, label="Block Swap to Save VRAM", value=30)
                with gr.Row():
                    t2v_fp8 = gr.Checkbox(label="Use FP8 (DiT)", value=False)
                    t2v_fp8_scaled = gr.Checkbox(label="Use Scaled FP8 (DiT)", value=False)
                    t2v_fp8_t5 = gr.Checkbox(label="Use FP8 for T5", value=False)
                    t2v_vae_fp32 = gr.Checkbox(label="Use FP32 VAE (higher quality, more VRAM)", value=True)
                with gr.Row():
                    t2v_model_folder = gr.Textbox(label="Model Folder", value="wan")
                    t2v_refresh_models_btn = gr.Button("🔄 Models", elem_classes="refresh-btn")
                with gr.Row():
                    t2v_dit_low_noise = gr.Dropdown(
                        label="DiT Low Noise Model (.safetensors)",
                        choices=get_vace_low_noise_models("wan"),
                        value=get_default_low_noise_model("wan"),
                        allow_custom_value=True,
                        interactive=True
                    )
                    t2v_dit_high_noise = gr.Dropdown(
                        label="DiT High Noise Model (.safetensors)",
                        choices=get_vace_high_noise_models("wan"),
                        value=get_default_high_noise_model("wan"),
                        allow_custom_value=True,
                        interactive=True
                    )
                with gr.Row():
                    t2v_vae_model = gr.Dropdown(
                        label="VAE Model (.pth)",
                        choices=get_vace_vae_models("wan"),
                        value=get_default_vae_model("wan"),
                        allow_custom_value=True,
                        interactive=True
                    )
                    t2v_t5_model = gr.Dropdown(
                        label="T5 Model (.pth/.safetensors)",
                        choices=get_vace_t5_models("wan"),
                        value=get_default_t5_model("wan"),
                        allow_custom_value=True,
                        interactive=True
                    )
                t2v_save_path = gr.Textbox(label="Save Path", value="outputs")
        
        # VACE I2V Tab - Image to Video
        with gr.Tab("VACE I2V", id=1):
            with gr.Row():
                with gr.Column(scale=4):
                    i2v_prompt = gr.Textbox(
                        scale=3,
                        label="Enter your prompt",
                        value="The image comes to life with natural motion and dynamic camera movement.",
                        lines=5
                    )
                    i2v_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt",
                        value="low quality, blurry, static, distorted, deformed",
                        lines=3
                    )
                
                with gr.Column(scale=1):
                    i2v_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    i2v_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)
                
                with gr.Column(scale=2):
                    i2v_batch_progress = gr.Textbox(label="Status", interactive=False, value="")
                    i2v_progress_text = gr.Textbox(label="Progress", interactive=False, value="")
            
            with gr.Row():
                i2v_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                i2v_stop_btn = gr.Button("Stop Generation", variant="stop")
            
            with gr.Row():
                with gr.Column():
                    i2v_start_image = gr.Image(label="Start Image", type="filepath")
                    i2v_end_image = gr.Image(label="End Image (optional for FLF mode)", type="filepath")
                    i2v_auto_flf = gr.Checkbox(value=True, label="Auto-detect FLF mode when both images provided")
                    
                    i2v_original_dims = gr.Textbox(label="Original Dimensions", interactive=False)
                    i2v_auto_size = gr.Checkbox(value=True, label="Auto-detect size from image")
                    
                    with gr.Row():
                        i2v_width = gr.Number(label="Width", value=832, step=32)
                        i2v_calc_height_btn = gr.Button("→")
                        i2v_calc_width_btn = gr.Button("←")
                        i2v_height = gr.Number(label="Height", value=480, step=32)
                    
                    i2v_video_length = gr.Slider(minimum=9, maximum=161, step=4, label="Frame Count", value=81, info="Must be 4n+1")
                    i2v_fps = gr.Slider(minimum=8, maximum=30, step=1, label="Frames Per Second", value=16)
                    i2v_infer_steps = gr.Slider(minimum=20, maximum=100, step=1, label="Sampling Steps", value=50)
                    i2v_flow_shift = gr.Slider(minimum=0.0, maximum=20.0, step=0.1, label="Flow Shift", value=5.0)
                    i2v_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale", value=7.0)
                    i2v_dual_dit_boundary = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Dual-DiT Boundary", value=0.875)
                    
                    with gr.Row():
                        i2v_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                        i2v_random_seed_btn = gr.Button("🎲")
                    
                    # VACE Optimizations
                    with gr.Accordion("VACE Optimizations", open=False):
                        with gr.Row():
                            i2v_enable_teacache = gr.Checkbox(value=True, label="Enable TeaCache")
                            i2v_teacache_threshold = gr.Slider(0.05, 0.30, value=0.10, step=0.01, label="TeaCache Threshold")
                        
                        i2v_cfg_skip_ratio = gr.Slider(0.0, 0.25, value=0.0, step=0.05, label="CFG Skip Ratio")
                        
                        with gr.Row():
                            i2v_enable_riflex = gr.Checkbox(value=False, label="Enable RiFlex")
                            i2v_riflex_k = gr.Slider(1, 10, value=6, step=1, label="RiFlex K", visible=False)
                
                with gr.Column():
                    i2v_output = gr.Gallery(
                        label="Generated Videos",
                        columns=[2], rows=[2], object_fit="contain", height="auto",
                        show_label=True, elem_id="gallery_vace_i2v", allow_preview=True, preview=True
                    )
                    
                    # LoRA Configuration
                    with gr.Accordion("I2V LoRA Configuration", open=True):
                        with gr.Row():
                            i2v_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                            i2v_lora_refresh_btn = gr.Button("🔄 LoRA", elem_classes="refresh-btn")
                        
                        i2v_lora_weights = []
                        i2v_lora_multipliers = []
                        i2v_lora_apply_low = []
                        i2v_lora_apply_high = []
                        
                        for i in range(4):
                            with gr.Row():
                                i2v_lora_weights.append(gr.Dropdown(
                                    label=f"LoRA {i+1}", choices=get_lora_files("lora"),
                                    value="None", allow_custom_value=False, interactive=True, scale=2
                                ))
                                i2v_lora_multipliers.append(gr.Slider(
                                    label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1
                                ))
                            with gr.Row():
                                i2v_lora_apply_low.append(gr.Checkbox(label="Apply to Low Noise", value=True, scale=1))
                                i2v_lora_apply_high.append(gr.Checkbox(label="Apply to High Noise", value=False, scale=1))
                        
                        with gr.Accordion("Additional LoRAs (5-8)", open=False):
                            for i in range(4, 8):
                                with gr.Row():
                                    i2v_lora_weights.append(gr.Dropdown(
                                        label=f"LoRA {i+1}", choices=get_lora_files("lora"),
                                        value="None", allow_custom_value=False, interactive=True, scale=2
                                    ))
                                    i2v_lora_multipliers.append(gr.Slider(
                                        label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1
                                    ))
                                with gr.Row():
                                    i2v_lora_apply_low.append(gr.Checkbox(label="Apply to Low Noise", value=True, scale=1))
                                    i2v_lora_apply_high.append(gr.Checkbox(label="Apply to High Noise", value=False, scale=1))
            
            # Model Paths & Performance with CLIP
            with gr.Accordion("I2V Model Paths & Performance", open=True):
                with gr.Row():
                    i2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "torch", "xformers"], label="Attention Mode", value="sdpa")
                    i2v_block_swap = gr.Slider(minimum=0, maximum=39, step=1, label="Block Swap to Save VRAM", value=30)
                with gr.Row():
                    i2v_fp8 = gr.Checkbox(label="Use FP8 (DiT)", value=False)
                    i2v_fp8_scaled = gr.Checkbox(label="Use Scaled FP8 (DiT)", value=False)
                    i2v_fp8_t5 = gr.Checkbox(label="Use FP8 for T5", value=False)
                    i2v_vae_fp32 = gr.Checkbox(label="Use FP32 VAE", value=True)
                with gr.Row():
                    i2v_model_folder = gr.Textbox(label="Model Folder", value="wan")
                    i2v_refresh_models_btn = gr.Button("🔄 Models", elem_classes="refresh-btn")
                with gr.Row():
                    i2v_dit_low_noise = gr.Dropdown(
                        label="DiT Low Noise Model",
                        choices=get_vace_low_noise_models("wan"),
                        value=get_default_low_noise_model("wan"),
                        allow_custom_value=True
                    )
                    i2v_dit_high_noise = gr.Dropdown(
                        label="DiT High Noise Model",
                        choices=get_vace_high_noise_models("wan"),
                        value=get_default_high_noise_model("wan"),
                        allow_custom_value=True
                    )
                    i2v_clip_model = gr.Dropdown(
                        label="CLIP Model (.pth, for i2v)",
                        choices=get_vace_clip_models("wan"),
                        value=get_default_clip_model("wan"),
                        allow_custom_value=True,
                        interactive=True
                    )
                with gr.Row():
                    i2v_vae_model = gr.Dropdown(
                        label="VAE Model",
                        choices=get_vace_vae_models("wan"),
                        value=get_default_vae_model("wan"),
                        allow_custom_value=True
                    )
                    i2v_t5_model = gr.Dropdown(
                        label="T5 Model",
                        choices=get_vace_t5_models("wan"),
                        value=get_default_t5_model("wan"),
                        allow_custom_value=True
                    )
                i2v_save_path = gr.Textbox(label="Save Path", value="outputs")
        
        # VACE S2V Tab - Subject to Video
        with gr.Tab("VACE S2V", id=2):
            with gr.Row():
                with gr.Column(scale=4):
                    s2v_prompt = gr.Textbox(
                        scale=3,
                        label="Enter your prompt (describe action/scene, not appearance)",
                        value="The subject is walking through a beautiful garden, turning to look at flowers.",
                        lines=5
                    )
                    s2v_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt",
                        value="low quality, blurry, static, distorted, deformed, inconsistent identity",
                        lines=3
                    )
                
                with gr.Column(scale=1):
                    s2v_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    s2v_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)
                
                with gr.Column(scale=2):
                    s2v_batch_progress = gr.Textbox(label="Status", interactive=False, value="")
                    s2v_progress_text = gr.Textbox(label="Progress", interactive=False, value="")
            
            with gr.Row():
                s2v_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                s2v_stop_btn = gr.Button("Stop Generation", variant="stop")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Subject Reference Images")
                    s2v_ref_images = gr.File(
                        label="Upload Reference Images (1-4)",
                        file_count="multiple",
                        file_types=["image"]
                    )
                    s2v_ref_gallery = gr.Gallery(
                        label="Reference Preview",
                        columns=2,
                        height=200,
                        object_fit="contain"
                    )
                    s2v_padding_ref = gr.Checkbox(value=True, label="Add padding to match aspect ratio")
                    
                    gr.Markdown("### Generation Parameters")
                    
                    with gr.Row():
                        s2v_width = gr.Number(label="Width", value=832, step=32)
                        s2v_calc_height_btn = gr.Button("→")
                        s2v_calc_width_btn = gr.Button("←")
                        s2v_height = gr.Number(label="Height", value=480, step=32)
                    
                    s2v_original_dims = gr.Textbox(label="Original Dimensions", visible=False)
                    
                    s2v_video_length = gr.Slider(minimum=9, maximum=161, step=4, label="Frame Count", value=81, info="Must be 4n+1")
                    s2v_fps = gr.Slider(minimum=8, maximum=30, step=1, label="Frames Per Second", value=16)
                    s2v_infer_steps = gr.Slider(minimum=20, maximum=100, step=1, label="Sampling Steps", value=50)
                    s2v_flow_shift = gr.Slider(minimum=0.0, maximum=20.0, step=0.1, label="Flow Shift", value=5.0)
                    s2v_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale", value=7.0)
                    s2v_dual_dit_boundary = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Dual-DiT Boundary", value=0.875)
                    
                    with gr.Row():
                        s2v_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                        s2v_random_seed_btn = gr.Button("🎲")
                    
                    # VACE Optimizations
                    with gr.Accordion("VACE Optimizations", open=False):
                        with gr.Row():
                            s2v_enable_teacache = gr.Checkbox(value=True, label="Enable TeaCache")
                            s2v_teacache_threshold = gr.Slider(0.05, 0.30, value=0.10, step=0.01, label="TeaCache Threshold")
                        
                        s2v_cfg_skip_ratio = gr.Slider(0.0, 0.25, value=0.0, step=0.05, label="CFG Skip Ratio")
                        
                        with gr.Row():
                            s2v_enable_riflex = gr.Checkbox(value=False, label="Enable RiFlex")
                            s2v_riflex_k = gr.Slider(1, 10, value=6, step=1, label="RiFlex K", visible=False)
                
                with gr.Column():
                    s2v_output = gr.Gallery(
                        label="Generated Videos",
                        columns=[2], rows=[2], object_fit="contain", height="auto",
                        show_label=True, elem_id="gallery_vace_s2v", allow_preview=True, preview=True
                    )
                    
                    # LoRA Configuration
                    with gr.Accordion("S2V LoRA Configuration", open=True):
                        with gr.Row():
                            s2v_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                            s2v_lora_refresh_btn = gr.Button("🔄 LoRA", elem_classes="refresh-btn")
                        
                        s2v_lora_weights = []
                        s2v_lora_multipliers = []
                        s2v_lora_apply_low = []
                        s2v_lora_apply_high = []
                        
                        for i in range(4):
                            with gr.Row():
                                s2v_lora_weights.append(gr.Dropdown(
                                    label=f"LoRA {i+1}", choices=get_lora_files("lora"),
                                    value="None", allow_custom_value=False, interactive=True, scale=2
                                ))
                                s2v_lora_multipliers.append(gr.Slider(
                                    label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1
                                ))
                            with gr.Row():
                                s2v_lora_apply_low.append(gr.Checkbox(label="Apply to Low Noise", value=True, scale=1))
                                s2v_lora_apply_high.append(gr.Checkbox(label="Apply to High Noise", value=False, scale=1))
                        
                        with gr.Accordion("Additional LoRAs (5-8)", open=False):
                            for i in range(4, 8):
                                with gr.Row():
                                    s2v_lora_weights.append(gr.Dropdown(
                                        label=f"LoRA {i+1}", choices=get_lora_files("lora"),
                                        value="None", allow_custom_value=False, interactive=True, scale=2
                                    ))
                                    s2v_lora_multipliers.append(gr.Slider(
                                        label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1
                                    ))
                                with gr.Row():
                                    s2v_lora_apply_low.append(gr.Checkbox(label="Apply to Low Noise", value=True, scale=1))
                                    s2v_lora_apply_high.append(gr.Checkbox(label="Apply to High Noise", value=False, scale=1))
            
            # Model Paths & Performance
            with gr.Accordion("S2V Model Paths & Performance", open=True):
                with gr.Row():
                    s2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "torch", "xformers"], label="Attention Mode", value="sdpa")
                    s2v_block_swap = gr.Slider(minimum=0, maximum=39, step=1, label="Block Swap to Save VRAM", value=30)
                with gr.Row():
                    s2v_fp8 = gr.Checkbox(label="Use FP8 (DiT)", value=False)
                    s2v_fp8_scaled = gr.Checkbox(label="Use Scaled FP8 (DiT)", value=False)
                    s2v_fp8_t5 = gr.Checkbox(label="Use FP8 for T5", value=False)
                    s2v_vae_fp32 = gr.Checkbox(label="Use FP32 VAE", value=True)
                with gr.Row():
                    s2v_model_folder = gr.Textbox(label="Model Folder", value="wan")
                    s2v_refresh_models_btn = gr.Button("🔄 Models", elem_classes="refresh-btn")
                with gr.Row():
                    s2v_dit_low_noise = gr.Dropdown(
                        label="DiT Low Noise Model",
                        choices=get_vace_low_noise_models("wan"),
                        value=get_default_low_noise_model("wan"),
                        allow_custom_value=True
                    )
                    s2v_dit_high_noise = gr.Dropdown(
                        label="DiT High Noise Model",
                        choices=get_vace_high_noise_models("wan"),
                        value=get_default_high_noise_model("wan"),
                        allow_custom_value=True
                    )
                    s2v_clip_model = gr.Dropdown(
                        label="CLIP Model (.pth)",
                        choices=get_vace_clip_models("wan"),
                        value=get_default_clip_model("wan"),
                        allow_custom_value=True
                    )
                with gr.Row():
                    s2v_vae_model = gr.Dropdown(
                        label="VAE Model",
                        choices=get_vace_vae_models("wan"),
                        value=get_default_vae_model("wan"),
                        allow_custom_value=True
                    )
                    s2v_t5_model = gr.Dropdown(
                        label="T5 Model",
                        choices=get_vace_t5_models("wan"),
                        value=get_default_t5_model("wan"),
                        allow_custom_value=True
                    )
                s2v_save_path = gr.Textbox(label="Save Path", value="outputs")
        
        # VACE V2V Tab - Video to Video
        with gr.Tab("VACE V2V", id=3):
            with gr.Row():
                with gr.Column(scale=4):
                    v2v_prompt = gr.Textbox(
                        scale=3,
                        label="Enter your prompt (style/content changes)",
                        value="Transform the video into an anime style with vibrant colors and dynamic effects.",
                        lines=5
                    )
                    v2v_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt",
                        value="low quality, blurry, static, distorted, deformed, flickering",
                        lines=3
                    )
                
                with gr.Column(scale=1):
                    v2v_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    v2v_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)
                
                with gr.Column(scale=2):
                    v2v_batch_progress = gr.Textbox(label="Status", interactive=False, value="")
                    v2v_progress_text = gr.Textbox(label="Progress", interactive=False, value="")
            
            with gr.Row():
                v2v_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                v2v_stop_btn = gr.Button("Stop Generation", variant="stop")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Input Sources")
                    v2v_control_video = gr.Video(label="Control Video", format="mp4")
                    v2v_video_info = gr.JSON(label="Video Info", visible=True)
                    
                    with gr.Accordion("Advanced VACE Inputs", open=False):
                        v2v_start_image = gr.Image(label="Start Image (optional)", type="filepath")
                        v2v_end_image = gr.Image(label="End Image (optional for FLF mode)", type="filepath")
                        
                        gr.Markdown("#### Subject Reference Images")
                        v2v_subject_ref_images = gr.File(
                            label="Upload Subject Reference Images (1-4)",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        v2v_ref_gallery = gr.Gallery(
                            label="Reference Preview",
                            columns=2,
                            height=150,
                            object_fit="contain"
                        )
                        v2v_padding_ref = gr.Checkbox(value=True, label="Add padding to match aspect ratio")
                    
                    gr.Markdown("### Control Settings")
                    v2v_control_strength = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.05,
                        label="Control Strength",
                        value=0.7,
                        info="How closely to follow the input video (0=ignore, 1=exact)"
                    )
                    
                    v2v_vace_context_scale = gr.Slider(
                        minimum=0.0, maximum=2.0, step=0.05,
                        label="VACE Context Scale",
                        value=1.0,
                        info="Strength of VACE context influence (1.0=normal)"
                    )
                    
                    v2v_auto_settings = gr.Checkbox(
                        value=True,
                        label="Auto-detect settings from video"
                    )
                    
                    gr.Markdown("### Generation Parameters")
                    
                    with gr.Row():
                        v2v_width = gr.Number(label="Width", value=832, step=32)
                        v2v_calc_height_btn = gr.Button("→")
                        v2v_calc_width_btn = gr.Button("←")
                        v2v_height = gr.Number(label="Height", value=480, step=32)
                    
                    v2v_original_dims = gr.Textbox(label="Original Dimensions", visible=False)
                    
                    v2v_video_length = gr.Number(label="Frame Count", value=81, step=4)
                    v2v_fps = gr.Number(label="FPS", value=16, step=1)
                    
                    v2v_infer_steps = gr.Slider(minimum=20, maximum=100, step=1, label="Sampling Steps", value=50)
                    v2v_flow_shift = gr.Slider(minimum=0.0, maximum=20.0, step=0.1, label="Flow Shift", value=5.0)
                    v2v_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale", value=7.0)
                    v2v_dual_dit_boundary = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Dual-DiT Boundary", value=0.875)
                    
                    with gr.Row():
                        v2v_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                        v2v_random_seed_btn = gr.Button("🎲")
                    
                    # VACE Optimizations
                    with gr.Accordion("VACE Optimizations", open=False):
                        with gr.Row():
                            v2v_enable_teacache = gr.Checkbox(value=True, label="Enable TeaCache")
                            v2v_teacache_threshold = gr.Slider(0.05, 0.30, value=0.10, step=0.01, label="TeaCache Threshold")
                        
                        v2v_cfg_skip_ratio = gr.Slider(0.0, 0.25, value=0.0, step=0.05, label="CFG Skip Ratio")
                        
                        with gr.Row():
                            v2v_enable_riflex = gr.Checkbox(value=False, label="Enable RiFlex")
                            v2v_riflex_k = gr.Slider(1, 10, value=6, step=1, label="RiFlex K", visible=False)
                
                with gr.Column():
                    v2v_output = gr.Gallery(
                        label="Original vs Generated",
                        columns=[2], rows=[2], object_fit="contain", height="auto",
                        show_label=True, elem_id="gallery_vace_v2v", allow_preview=True, preview=True
                    )
                    
                    # LoRA Configuration
                    with gr.Accordion("V2V LoRA Configuration", open=True):
                        with gr.Row():
                            v2v_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                            v2v_lora_refresh_btn = gr.Button("🔄 LoRA", elem_classes="refresh-btn")
                        
                        v2v_lora_weights = []
                        v2v_lora_multipliers = []
                        v2v_lora_apply_low = []
                        v2v_lora_apply_high = []
                        
                        for i in range(4):
                            with gr.Row():
                                v2v_lora_weights.append(gr.Dropdown(
                                    label=f"LoRA {i+1}", choices=get_lora_files("lora"),
                                    value="None", allow_custom_value=False, interactive=True, scale=2
                                ))
                                v2v_lora_multipliers.append(gr.Slider(
                                    label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1
                                ))
                            with gr.Row():
                                v2v_lora_apply_low.append(gr.Checkbox(label="Apply to Low Noise", value=True, scale=1))
                                v2v_lora_apply_high.append(gr.Checkbox(label="Apply to High Noise", value=False, scale=1))
                        
                        with gr.Accordion("Additional LoRAs (5-8)", open=False):
                            for i in range(4, 8):
                                with gr.Row():
                                    v2v_lora_weights.append(gr.Dropdown(
                                        label=f"LoRA {i+1}", choices=get_lora_files("lora"),
                                        value="None", allow_custom_value=False, interactive=True, scale=2
                                    ))
                                    v2v_lora_multipliers.append(gr.Slider(
                                        label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1
                                    ))
                                with gr.Row():
                                    v2v_lora_apply_low.append(gr.Checkbox(label="Apply to Low Noise", value=True, scale=1))
                                    v2v_lora_apply_high.append(gr.Checkbox(label="Apply to High Noise", value=False, scale=1))
            
            # Model Paths & Performance
            with gr.Accordion("V2V Model Paths & Performance", open=True):
                with gr.Row():
                    v2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "torch", "xformers"], label="Attention Mode", value="sdpa")
                    v2v_block_swap = gr.Slider(minimum=0, maximum=39, step=1, label="Block Swap to Save VRAM", value=30)
                with gr.Row():
                    v2v_fp8 = gr.Checkbox(label="Use FP8 (DiT)", value=False)
                    v2v_fp8_scaled = gr.Checkbox(label="Use Scaled FP8 (DiT)", value=False)
                    v2v_fp8_t5 = gr.Checkbox(label="Use FP8 for T5", value=False)
                    v2v_vae_fp32 = gr.Checkbox(label="Use FP32 VAE", value=True)
                with gr.Row():
                    v2v_model_folder = gr.Textbox(label="Model Folder", value="wan")
                    v2v_refresh_models_btn = gr.Button("🔄 Models", elem_classes="refresh-btn")
                with gr.Row():
                    v2v_dit_low_noise = gr.Dropdown(
                        label="DiT Low Noise Model",
                        choices=get_vace_low_noise_models("wan"),
                        value=get_default_low_noise_model("wan"),
                        allow_custom_value=True
                    )
                    v2v_dit_high_noise = gr.Dropdown(
                        label="DiT High Noise Model",
                        choices=get_vace_high_noise_models("wan"),
                        value=get_default_high_noise_model("wan"),
                        allow_custom_value=True
                    )
                    v2v_clip_model = gr.Dropdown(
                        label="CLIP Model (.pth, for image conditioning)",
                        choices=get_vace_clip_models("wan"),
                        value=get_default_clip_model("wan"),
                        allow_custom_value=True,
                        interactive=True
                    )
                with gr.Row():
                    v2v_vae_model = gr.Dropdown(
                        label="VAE Model",
                        choices=get_vace_vae_models("wan"),
                        value=get_default_vae_model("wan"),
                        allow_custom_value=True
                    )
                    v2v_t5_model = gr.Dropdown(
                        label="T5 Model",
                        choices=get_vace_t5_models("wan"),
                        value=get_default_t5_model("wan"),
                        allow_custom_value=True
                    )
                v2v_save_path = gr.Textbox(label="Save Path", value="outputs")
        
        # Video Info Tab
        with gr.Tab("Video Info", id=4):
            with gr.Column():
                info_video_input = gr.Video(label="Upload Video", interactive=True)
                
                with gr.Row():
                    info_json = gr.JSON(label="Video Information")
                    info_metadata_json = gr.JSON(label="Generation Metadata (if available)")
                
                info_frame_gallery = gr.Gallery(label="Extracted Frames", columns=4)
                info_status = gr.Textbox(label="Status", interactive=False)
    
    # Footer
    gr.Markdown(
        """
        ---
        💡 **Tips**: 
        - TeaCache provides 2-5x speedup with minimal quality impact
        - Use CFG skip ratio 0.15-0.20 for faster generation
        - Enable RiFlex for better temporal consistency
        - Dual DiT models automatically switch at the boundary threshold
        """
    )
    
    # Event Handlers
    
    # T2V Tab Event Handlers
    t2v_prompt.change(fn=count_prompt_tokens, inputs=t2v_prompt, outputs=t2v_token_counter)
    t2v_stop_btn.click(fn=lambda: stop_event.set())
    t2v_random_seed_btn.click(fn=lambda: random.randint(0, 2**32 - 1), outputs=[t2v_seed])
    
    t2v_enable_teacache.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[t2v_enable_teacache],
        outputs=[t2v_teacache_threshold]
    )
    
    t2v_enable_riflex.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[t2v_enable_riflex],
        outputs=[t2v_riflex_k]
    )
    
    def refresh_t2v_models(folder):
        return (
            gr.update(choices=get_vace_low_noise_models(folder)),
            gr.update(choices=get_vace_high_noise_models(folder)),
            gr.update(choices=get_vace_vae_models(folder)),
            gr.update(choices=get_vace_t5_models(folder))
        )
    
    t2v_refresh_models_btn.click(
        fn=refresh_t2v_models,
        inputs=[t2v_model_folder],
        outputs=[t2v_dit_low_noise, t2v_dit_high_noise, t2v_vae_model, t2v_t5_model]
    )
    
    def refresh_t2v_loras(folder):
        choices = get_lora_files(folder)
        return [gr.update(choices=choices) for _ in range(8)]
    
    t2v_lora_refresh_btn.click(
        fn=refresh_t2v_loras,
        inputs=[t2v_lora_folder],
        outputs=t2v_lora_weights
    )
    
    def t2v_generate_wrapper(*args):
        generator = run_vace_generation("t2v", *args)
        for videos, status in generator:
            yield videos, status
    
    t2v_generate_btn.click(
        fn=t2v_generate_wrapper,
        inputs=[
            t2v_prompt, t2v_negative_prompt, t2v_width, t2v_height, t2v_video_length, t2v_fps,
            t2v_seed, t2v_infer_steps, t2v_guidance_scale, t2v_flow_shift,
            t2v_enable_teacache, t2v_teacache_threshold, t2v_cfg_skip_ratio,
            t2v_enable_riflex, t2v_riflex_k, t2v_dit_low_noise, t2v_dit_high_noise,
            t2v_vae_model, t2v_t5_model
        ] + t2v_lora_weights + t2v_lora_multipliers + t2v_lora_apply_low + t2v_lora_apply_high + [
            t2v_batch_size, t2v_dual_dit_boundary, t2v_attn_mode, t2v_block_swap, 
            t2v_fp8, t2v_fp8_scaled, t2v_fp8_t5, t2v_vae_fp32, t2v_model_folder, 
            t2v_lora_folder, t2v_save_path
        ],
        outputs=[t2v_output, t2v_batch_progress]
    )
    
    # I2V Tab Event Handlers
    i2v_prompt.change(fn=count_prompt_tokens, inputs=i2v_prompt, outputs=i2v_token_counter)
    i2v_stop_btn.click(fn=lambda: stop_event.set())
    i2v_random_seed_btn.click(fn=lambda: random.randint(0, 2**32 - 1), outputs=[i2v_seed])
    
    i2v_start_image.change(
        fn=update_image_dimensions,
        inputs=[i2v_start_image],
        outputs=[i2v_original_dims, i2v_width, i2v_height]
    )
    
    i2v_calc_height_btn.click(
        fn=calculate_height_from_width,
        inputs=[i2v_width, i2v_original_dims],
        outputs=[i2v_height]
    )
    
    i2v_calc_width_btn.click(
        fn=calculate_width_from_height,
        inputs=[i2v_height, i2v_original_dims],
        outputs=[i2v_width]
    )
    
    i2v_enable_teacache.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[i2v_enable_teacache],
        outputs=[i2v_teacache_threshold]
    )
    
    i2v_enable_riflex.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[i2v_enable_riflex],
        outputs=[i2v_riflex_k]
    )
    
    def refresh_i2v_models(folder):
        return (
            gr.update(choices=get_vace_low_noise_models(folder)),
            gr.update(choices=get_vace_high_noise_models(folder)),
            gr.update(choices=get_vace_vae_models(folder)),
            gr.update(choices=get_vace_t5_models(folder)),
            gr.update(choices=get_vace_clip_models(folder))
        )
    
    i2v_refresh_models_btn.click(
        fn=refresh_i2v_models,
        inputs=[i2v_model_folder],
        outputs=[i2v_dit_low_noise, i2v_dit_high_noise, i2v_vae_model, i2v_t5_model, i2v_clip_model]
    )
    
    def refresh_i2v_loras(folder):
        choices = get_lora_files(folder)
        return [gr.update(choices=choices) for _ in range(8)]
    
    i2v_lora_refresh_btn.click(
        fn=refresh_i2v_loras,
        inputs=[i2v_lora_folder],
        outputs=i2v_lora_weights
    )
    
    def i2v_generate_wrapper(*args):
        generator = run_vace_generation("i2v", *args)
        for videos, status in generator:
            yield videos, status
    
    i2v_generate_btn.click(
        fn=i2v_generate_wrapper,
        inputs=[
            i2v_prompt, i2v_negative_prompt, i2v_width, i2v_height, i2v_video_length, i2v_fps,
            i2v_seed, i2v_infer_steps, i2v_guidance_scale, i2v_flow_shift,
            i2v_enable_teacache, i2v_teacache_threshold, i2v_cfg_skip_ratio,
            i2v_enable_riflex, i2v_riflex_k, i2v_dit_low_noise, i2v_dit_high_noise,
            i2v_vae_model, i2v_t5_model
        ] + i2v_lora_weights + i2v_lora_multipliers + i2v_lora_apply_low + i2v_lora_apply_high + [
            i2v_batch_size, i2v_dual_dit_boundary, i2v_attn_mode, i2v_block_swap,
            i2v_fp8, i2v_fp8_scaled, i2v_fp8_t5, i2v_vae_fp32, i2v_model_folder,
            i2v_lora_folder, i2v_save_path, i2v_start_image, i2v_end_image, i2v_clip_model
        ],
        outputs=[i2v_output, i2v_batch_progress]
    )
    
    # S2V Tab Event Handlers  
    s2v_prompt.change(fn=count_prompt_tokens, inputs=s2v_prompt, outputs=s2v_token_counter)
    s2v_stop_btn.click(fn=lambda: stop_event.set())
    s2v_random_seed_btn.click(fn=lambda: random.randint(0, 2**32 - 1), outputs=[s2v_seed])
    
    def update_s2v_ref_gallery(files):
        if not files:
            return []
        images = []
        for f in files[:4]:  # Limit to 4 images
            try:
                images.append(Image.open(f.name))
            except:
                pass
        return images
    
    s2v_ref_images.change(
        fn=update_s2v_ref_gallery,
        inputs=[s2v_ref_images],
        outputs=[s2v_ref_gallery]
    )
    
    s2v_enable_teacache.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[s2v_enable_teacache],
        outputs=[s2v_teacache_threshold]
    )
    
    s2v_enable_riflex.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[s2v_enable_riflex],
        outputs=[s2v_riflex_k]
    )
    
    def refresh_s2v_models(folder):
        return (
            gr.update(choices=get_vace_low_noise_models(folder)),
            gr.update(choices=get_vace_high_noise_models(folder)),
            gr.update(choices=get_vace_vae_models(folder)),
            gr.update(choices=get_vace_t5_models(folder)),
            gr.update(choices=get_vace_clip_models(folder))
        )
    
    s2v_refresh_models_btn.click(
        fn=refresh_s2v_models,
        inputs=[s2v_model_folder],
        outputs=[s2v_dit_low_noise, s2v_dit_high_noise, s2v_vae_model, s2v_t5_model, s2v_clip_model]
    )
    
    def refresh_s2v_loras(folder):
        choices = get_lora_files(folder)
        return [gr.update(choices=choices) for _ in range(8)]
    
    s2v_lora_refresh_btn.click(
        fn=refresh_s2v_loras,
        inputs=[s2v_lora_folder],
        outputs=s2v_lora_weights
    )
    
    def generate_s2v(ref_files, *other_args):
        # Convert file objects to path list for subject_ref_images
        ref_imgs = []
        if ref_files:
            for f in ref_files[:4]:
                try:
                    ref_imgs.append(f.name)  # Pass file paths
                except:
                    pass
        
        # Build full args list with ref_imgs in the right position
        full_args = list(other_args[:35])  # First 35 args up to save_path
        full_args.extend([None, None, other_args[35]])  # start_image, end_image, clip_model
        full_args.append(ref_imgs)  # subject_ref_images
        full_args.append(None)  # control_video
        full_args.append(other_args[36])  # padding_ref_images
        
        generator = run_vace_generation("s2v", *full_args)
        for videos, status in generator:
            yield videos, status
    
    s2v_generate_btn.click(
        fn=generate_s2v,
        inputs=[
            s2v_ref_images,  # This will be handled specially in generate_s2v
            s2v_prompt, s2v_negative_prompt, s2v_width, s2v_height, s2v_video_length, s2v_fps,
            s2v_seed, s2v_infer_steps, s2v_guidance_scale, s2v_flow_shift,
            s2v_enable_teacache, s2v_teacache_threshold, s2v_cfg_skip_ratio,
            s2v_enable_riflex, s2v_riflex_k, s2v_dit_low_noise, s2v_dit_high_noise,
            s2v_vae_model, s2v_t5_model
        ] + s2v_lora_weights + s2v_lora_multipliers + s2v_lora_apply_low + s2v_lora_apply_high + [
            s2v_batch_size, s2v_dual_dit_boundary, s2v_attn_mode, s2v_block_swap,
            s2v_fp8, s2v_fp8_scaled, s2v_fp8_t5, s2v_vae_fp32, s2v_model_folder,
            s2v_lora_folder, s2v_save_path, s2v_clip_model, s2v_padding_ref
        ],
        outputs=[s2v_output, s2v_batch_progress]
    )
    
    # V2V Tab Event Handlers
    v2v_prompt.change(fn=count_prompt_tokens, inputs=v2v_prompt, outputs=v2v_token_counter)
    v2v_stop_btn.click(fn=lambda: stop_event.set())
    v2v_random_seed_btn.click(fn=lambda: random.randint(0, 2**32 - 1), outputs=[v2v_seed])
    
    def update_v2v_from_video(video_path, auto):
        if not video_path or not auto:
            return gr.update(), gr.update(), gr.update(), gr.update(), {}
        
        info = get_video_info(video_path)
        if info:
            return (
                info.get('width', 832),
                info.get('height', 480),
                info.get('frames', 81),
                info.get('fps', 16),
                info
            )
        return gr.update(), gr.update(), gr.update(), gr.update(), {}
    
    v2v_control_video.change(
        fn=update_v2v_from_video,
        inputs=[v2v_control_video, v2v_auto_settings],
        outputs=[v2v_width, v2v_height, v2v_video_length, v2v_fps, v2v_video_info]
    )
    
    def update_v2v_ref_gallery(files):
        if not files:
            return []
        images = []
        for f in files[:4]:  # Limit to 4 images
            try:
                images.append(Image.open(f.name))
            except:
                pass
        return images
    
    v2v_subject_ref_images.change(
        fn=update_v2v_ref_gallery,
        inputs=[v2v_subject_ref_images],
        outputs=[v2v_ref_gallery]
    )
    
    v2v_enable_teacache.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[v2v_enable_teacache],
        outputs=[v2v_teacache_threshold]
    )
    
    v2v_enable_riflex.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[v2v_enable_riflex],
        outputs=[v2v_riflex_k]
    )
    
    def refresh_v2v_models(folder):
        return (
            gr.update(choices=get_vace_low_noise_models(folder)),
            gr.update(choices=get_vace_high_noise_models(folder)),
            gr.update(choices=get_vace_vae_models(folder)),
            gr.update(choices=get_vace_t5_models(folder)),
            gr.update(choices=get_vace_clip_models(folder))
        )
    
    v2v_refresh_models_btn.click(
        fn=refresh_v2v_models,
        inputs=[v2v_model_folder],
        outputs=[v2v_dit_low_noise, v2v_dit_high_noise, v2v_vae_model, v2v_t5_model, v2v_clip_model]
    )
    
    def refresh_v2v_loras(folder):
        choices = get_lora_files(folder)
        return [gr.update(choices=choices) for _ in range(8)]
    
    v2v_lora_refresh_btn.click(
        fn=refresh_v2v_loras,
        inputs=[v2v_lora_folder],
        outputs=v2v_lora_weights
    )
    
    def generate_v2v(video_path, control_str, vace_context_scale, start_img, end_img, ref_files, padding_ref, *other_args):
        # Convert file objects to path list for subject_ref_images
        ref_imgs = []
        if ref_files:
            for f in ref_files[:4]:
                try:
                    ref_imgs.append(f.name)
                except:
                    pass
        
        # Count the arguments correctly
        # other_args contains: prompt, negative_prompt, width, height, video_length, fps,
        # seed, infer_steps, guidance_scale, flow_shift,
        # enable_teacache, teacache_threshold, cfg_skip_ratio,
        # enable_riflex, riflex_k, dit_low_noise, dit_high_noise,
        # vae_model, t5_model
        # + 8 lora_weights + 8 lora_multipliers + 8 lora_apply_low + 8 lora_apply_high (32 total)
        # + batch_size, dual_dit_boundary, attn_mode, block_swap,
        # fp8, fp8_scaled, fp8_t5, vae_fp32, model_folder,
        # lora_folder, save_path, clip_model (12 more)
        # Total = 19 + 32 + 12 = 63
        
        # Build full args list - run_vace_generation expects args in specific order
        full_args = []
        full_args.extend(other_args[:19])  # Basic parameters through t5_model
        full_args.extend(other_args[19:51])  # All LoRA parameters (32 total)
        full_args.extend(other_args[51:62])  # batch_size through save_path (11 items)
        
        # Add mode-specific parameters
        full_args.append(start_img)  # start_image
        full_args.append(end_img)  # end_image
        full_args.append(other_args[62] if len(other_args) > 62 else None)  # clip_model
        full_args.append(ref_imgs)  # subject_ref_images
        full_args.append(video_path)  # control_video
        full_args.append(padding_ref)  # padding_ref_images
        full_args.append(control_str)  # control_strength
        full_args.append(vace_context_scale)  # vace_context_scale
        
        generator = run_vace_generation("v2v", *full_args)
        for videos, status in generator:
            yield videos, status
    
    v2v_generate_btn.click(
        fn=generate_v2v,
        inputs=[
            v2v_control_video, v2v_control_strength, v2v_vace_context_scale,
            v2v_start_image, v2v_end_image, v2v_subject_ref_images, v2v_padding_ref,
            v2v_prompt, v2v_negative_prompt, v2v_width, v2v_height, v2v_video_length, v2v_fps,
            v2v_seed, v2v_infer_steps, v2v_guidance_scale, v2v_flow_shift,
            v2v_enable_teacache, v2v_teacache_threshold, v2v_cfg_skip_ratio,
            v2v_enable_riflex, v2v_riflex_k, v2v_dit_low_noise, v2v_dit_high_noise,
            v2v_vae_model, v2v_t5_model
        ] + v2v_lora_weights + v2v_lora_multipliers + v2v_lora_apply_low + v2v_lora_apply_high + [
            v2v_batch_size, v2v_dual_dit_boundary, v2v_attn_mode, v2v_block_swap,
            v2v_fp8, v2v_fp8_scaled, v2v_fp8_t5, v2v_vae_fp32, v2v_model_folder,
            v2v_lora_folder, v2v_save_path, v2v_clip_model
        ],
        outputs=[v2v_output, v2v_batch_progress]
    )
    
    # Video Info Tab Event Handlers
    def analyze_video(video_path):
        if not video_path:
            return {}, {}, "No video uploaded"
        
        info = get_video_info(video_path)
        metadata = {}
        
        return info, metadata, "Video analyzed successfully"
    
    info_video_input.change(
        fn=analyze_video,
        inputs=[info_video_input],
        outputs=[info_json, info_metadata_json, info_status]
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