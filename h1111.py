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
from typing import List, Tuple, Optional, Generator, Dict
import json
from gradio import themes
from gradio.themes.utils import colors
import subprocess
from PIL import Image
import math
import cv2
import glob


# Add global stop event
stop_event = threading.Event()

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
            'prompt': ('prompt', 'v2v_prompt'),
            'width': ('width', 'v2v_width'),
            'height': ('height', 'v2v_height'),
            'batch_size': ('batch_size', 'v2v_batch_size'),
            'video_length': ('video_length', 'v2v_video_length'),
            'fps': ('fps', 'v2v_fps'),
            'infer_steps': ('infer_steps', 'v2v_infer_steps'),
            'seed': ('seed', 'v2v_seed'),
            'model': ('model', 'v2v_model'),
            'vae': ('vae', 'v2v_vae'),
            'te1': ('te1', 'v2v_te1'),
            'te2': ('te2', 'v2v_te2'),
            'save_path': ('save_path', 'v2v_save_path'),
            'flow_shift': ('flow_shift', 'v2v_flow_shift'),
            'cfg_scale': ('cfg_scale', 'v2v_cfg_scale'),
            'output_type': ('output_type', 'v2v_output_type'),
            'attn_mode': ('attn_mode', 'v2v_attn_mode'),
            'block_swap': ('block_swap', 'v2v_block_swap')
        },
        'lora': {
            'lora_weights': [(f'lora{i+1}', f'v2v_lora_weights[{i}]') for i in range(4)],
            'lora_multipliers': [(f'lora{i+1}_multiplier', f'v2v_lora_multipliers[{i}]') for i in range(4)]
        }
    }
    
    results = {}
    for param, value in metadata.items():
        # Handle common parameters
        if param in mapping['common']:
            target = mapping['common'][param][0 if target_tab == 't2v' else 1]
            results[target] = value
        
        # Handle LoRA parameters
        if param == 'lora_weights':
            for i, weight in enumerate(value[:4]):
                target = mapping['lora']['lora_weights'][i][1 if target_tab == 'v2v' else 0]
                results[target] = weight
                
        if param == 'lora_multipliers':
            for i, mult in enumerate(value[:4]):
                target = mapping['lora']['lora_multipliers'][i][1 if target_tab == 'v2v' else 0]
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
    fp8_t5,
    lora_folder,
    *lora_params
):
    """Handle both folder-based batch processing and regular processing for WanX"""
    global stop_event
    
    if use_random:
        # Random image from folder mode
        stop_event.clear()

        all_videos = []
        progress_text = "Starting generation..."
        yield [], "Preparing...", progress_text

        # Ensure batch_size is treated as an integer
        batch_size = int(batch_size)
        
        # Process each item in the batch separately
        for i in range(batch_size):
            if stop_event.is_set():
                yield all_videos, "Generation stopped by user", ""
                return

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

            # Use the dimensions returned from the resize function
            local_width, local_height = width, height  # Default fallback
            if isinstance(size_info, tuple):
                local_width, local_height = size_info
                progress_text = f"Using image: {os.path.basename(random_image)} - Resized to {local_width}x{local_height} (maintaining aspect ratio)"
            else:
                progress_text = f"Using image: {os.path.basename(random_image)}"
            
            yield all_videos.copy(), batch_text, progress_text

            # Calculate seed for this batch item
            current_seed = seed
            if seed == -1:
                current_seed = random.randint(0, 2**32 - 1)
            elif batch_size > 1:
                current_seed = seed + i

            # Extract LoRA weights and multipliers
            num_lora_weights = 4
            lora_weights = lora_params[:num_lora_weights]
            lora_multipliers = lora_params[num_lora_weights:num_lora_weights*2]

            # Generate video for this image - one at a time
            for videos, status, progress in wanx_generate_video(
                prompt, 
                negative_prompt, 
                resized_image, 
                local_width,
                local_height,
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
                fp8_t5,
                lora_folder,
                *lora_weights,
                *lora_multipliers
            ):
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
        # For non-random mode, if batch_size > 1, we need to process multiple times
        # with the same input image but different seeds
        if int(batch_size) > 1:
            stop_event.clear()
            
            all_videos = []
            progress_text = "Starting generation..."
            yield [], "Preparing...", progress_text
            
            # Extract LoRA weights and multipliers and input image
            num_lora_weights = 4
            lora_weights = lora_params[:num_lora_weights]
            lora_multipliers = lora_params[num_lora_weights:num_lora_weights*2]
            input_image = lora_params[num_lora_weights*2] if len(lora_params) > num_lora_weights*2 else None
            
            # Process each batch item
            for i in range(int(batch_size)):
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
                
                # Generate a single video with the current seed
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
                    fp8_t5,
                    lora_folder,
                    *lora_weights,
                    *lora_multipliers
                ):
                    if videos:
                        all_videos.extend(videos)
                    yield all_videos.copy(), f"Batch {i+1}/{batch_size}: {status}", progress
                
                # Clear CUDA cache between generations
                clear_cuda_cache()
                time.sleep(0.5)
            
            yield all_videos, "Batch complete", ""
        else:
            # Single image, single generation - use existing function
            num_lora_weights = 4
            lora_weights = lora_params[:num_lora_weights]
            lora_multipliers = lora_params[num_lora_weights:num_lora_weights*2]
            input_image = lora_params[num_lora_weights*2] if len(lora_params) > num_lora_weights*2 else None
            
            yield from wanx_generate_video(
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
                fp8_t5, 
                lora_folder,
                *lora_weights,
                *lora_multipliers
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

def handle_wanx_gallery_select(evt: gr.SelectData) -> int:
    """Track selected index when gallery item is clicked"""
    return evt.index

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
    fp8_t5,
    lora_folder,
    lora1="None",
    lora2="None",
    lora3="None",
    lora4="None",
    lora1_multiplier=1.0,
    lora2_multiplier=1.0,
    lora3_multiplier=1.0,
    lora4_multiplier=1.0
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """Generate video with WanX model (supports both i2v and t2v)"""
    global stop_event
    
    if stop_event.is_set():
        yield [], "", ""
        return

    if seed == -1:
        current_seed = random.randint(0, 2**32 - 1)
    else:
        current_seed = seed
        
    # Check if we need input image (required for i2v, not for t2v)
    if "i2v" in task and not input_image:
        yield [], "Error: No input image provided", "Please provide an input image for image-to-video generation"
        return

    # Prepare environment
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    
    clear_cuda_cache()

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
        "--dit", dit_path,
        "--vae", vae_path,
        "--t5", t5_path,
        "--sample_solver", sample_solver
    ]
    
    # Add image path only for i2v task and if input image is provided
    if "i2v" in task and input_image:
        command.extend(["--image_path", input_image])
        command.extend(["--clip", clip_path])  # CLIP is only needed for i2v
    
    if negative_prompt:
        command.extend(["--negative_prompt", negative_prompt])
    
    if fp8:
        command.append("--fp8")
    
    if fp8_t5:
        command.append("--fp8_t5")
        
    if exclude_single_blocks:
        command.append("--exclude_single_blocks")
    
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
                "input_image": input_image if "i2v" in task else None
            }
            
            add_metadata_to_video(video_path, parameters)
            videos.append((str(video_path), f"Seed: {current_seed}"))

    yield videos, f"Completed (seed: {current_seed})", ""

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
    fp8_t5, 
    lora_folder,
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
            prompt, negative_prompt, input_image, width, height, 
            video_length, fps, infer_steps, flow_shift, guidance_scale, 
            current_seed, task, dit_path, vae_path, t5_path, clip_path, 
            save_path, output_type, sample_solver, exclude_single_blocks,
            attn_mode, block_swap, fp8, fp8_t5,
            lora_folder,
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
    """, 

) as demo:
    # Add state for tracking selected video indices in both tabs
    selected_index = gr.State(value=None)  # For Text to Video
    v2v_selected_index = gr.State(value=None)  # For Video to Video
    params_state = gr.State() #New addition
    i2v_selected_index = gr.State(value=None) 
    skyreels_selected_index = gr.State(value=None)
    demo.load(None, None, None, js="""
    () => {
        document.title = 'H1111';

        function updateTitle(text) {
            if (text && text.trim()) {
                const progressMatch = text.match(/(\d+)%.*\[.*<(\d+:\d+),/);
                if (progressMatch) {
                    const percentage = progressMatch[1];
                    const timeRemaining = progressMatch[2];
                    document.title = `[${percentage}% ETA: ${timeRemaining}] - H1111`;
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
                        characterData: true
                    });
                }
            });
        }, 1000);
    }
    """)
        
    with gr.Tabs() as tabs:
        # Text to Video Tab
        with gr.Tab(id=1, label="Text to Video"):
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
                    refresh_btn = gr.Button("", elem_classes="refresh-btn")
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
        with gr.Tab(label="Image to Video") as i2v_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    i2v_prompt = gr.Textbox(scale=3, label="Enter your prompt", value="POV video of a cat chasing a frob.", lines=5)

                with gr.Column(scale=1):
                    i2v_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    i2v_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    i2v_batch_progress = gr.Textbox(label="", visible=True, elem_id="batch_progress")
                    i2v_progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text")

            with gr.Row():
                i2v_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                i2v_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column():
                    i2v_input = gr.Image(label="Input Image", type="filepath")
                    i2v_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.75, label="Denoise Strength")
                    # Scale slider as percentage 
                    scale_slider = gr.Slider(minimum=1, maximum=200, value=100, step=1, label="Scale %")
                    original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=True)
                    # Width and height inputs
                    with gr.Row():
                        width = gr.Number(label="New Width", value=544, step=16)
                        calc_height_btn = gr.Button("→")
                        calc_width_btn = gr.Button("←")
                        height = gr.Number(label="New Height", value=544, step=16)
                    i2v_video_length = gr.Slider(minimum=1, maximum=201, step=1, label="Video Length in Frames", value=25)
                    i2v_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=24)
                    i2v_infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=30)
                    i2v_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=11.0)
                    i2v_cfg_scale = gr.Slider(minimum=0.0, maximum=14.0, step=0.1, label="cfg scale", value=7.0)
                with gr.Column():
                    i2v_output = gr.Gallery(
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
                    i2v_send_to_v2v_btn = gr.Button("Send Selected to Video2Video")

                    # Add LoRA section for Image2Video
                    i2v_refresh_btn = gr.Button("", elem_classes="refresh-btn")
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
                    value="mp_rank_00_model_states.pt",
                    allow_custom_value=True,
                    interactive=True
                )

                i2v_vae = gr.Textbox(label="vae", value="hunyuan/pytorch_model.pt")
                i2v_te1 = gr.Textbox(label="te1", value="hunyuan/llava_llama3_fp16.safetensors")
                i2v_te2 = gr.Textbox(label="te2", value="hunyuan/clip_l.safetensors")
                i2v_save_path = gr.Textbox(label="Save Path", value="outputs")
            with gr.Row():
                i2v_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                i2v_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                i2v_use_split_attn = gr.Checkbox(label="Use Split Attention", value=False)
                i2v_use_fp8 = gr.Checkbox(label="Use FP8 (faster but lower precision)", value=True)
                i2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                i2v_block_swap = gr.Slider(minimum=0, maximum=36, step=1, label="Block Swap to Save Vram", value=0)

        # Video to Video Tab
        with gr.Tab(id=2, label="Video to Video") as v2v_tab:
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
                    v2v_refresh_btn = gr.Button("", elem_classes="refresh-btn")
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

        with gr.Tab(label="SkyReels-i2v") as skyreels_tab:
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
                    skyreels_refresh_btn = gr.Button("", elem_classes="refresh-btn")
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
                        value="",
                        lines=3,
                        info="Leave empty to use default negative prompt"
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
                    wanx_scale_slider = gr.Slider(minimum=1, maximum=200, value=100, step=1, label="Scale %")
                    wanx_original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=True)
        
                    # Width and height display
                    with gr.Row():
                        wanx_width = gr.Number(label="Width", value=832, interactive=True)
                        wanx_calc_height_btn = gr.Button("→")
                        wanx_calc_width_btn = gr.Button("←")
                        wanx_height = gr.Number(label="Height", value=480, interactive=True)
                        wanx_recommend_flow_btn = gr.Button("Recommend Flow Shift", size="sm")

                    wanx_video_length = gr.Slider(minimum=1, maximum=201, step=4, label="Video Length in Frames", value=81)
                    wanx_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=16)
                    wanx_infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=20)
                    wanx_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=3.0, 
                                            info="Recommended: 3.0 for 480p, 5.0 for others")
                    wanx_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale", value=5.0)

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
                    wanx_send_to_v2v_btn = gr.Button("Send Selected to Video2Video")

                    # Add LoRA section for WanX-i2v similar to other tabs
                    wanx_refresh_btn = gr.Button("", elem_classes="refresh-btn")
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
                wanx_task = gr.Dropdown(
                    label="Task",
                    choices=["i2v-14B"],
                    value="i2v-14B",
                    info="Currently only i2v-14B is supported"
                )
                wanx_dit_path = gr.Textbox(label="DiT Model Path", value="wan/wan2.1_i2v_480p_14B_bf16.safetensors")
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
                wanx_fp8 = gr.Checkbox(label="Use FP8", value=True)
                wanx_fp8_t5 = gr.Checkbox(label="Use FP8 for T5", value=False)                

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
                    wanx_t2v_send_to_v2v_btn = gr.Button("Send Selected to Video2Video")

                    # Add LoRA section for WanX-t2v
                    wanx_t2v_refresh_btn = gr.Button("", elem_classes="refresh-btn")
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
                wanx_t2v_dit_path = gr.Textbox(label="DiT Model Path", value="wan/wan2.1_t2v_14B_bf16.safetensors")
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
                wanx_t2v_fp8 = gr.Checkbox(label="Use FP8", value=True)
                wanx_t2v_fp8_t5 = gr.Checkbox(label="Use FP8 for T5", value=False)

        #Video Info Tab
        with gr.Tab("Video Info") as video_info_tab:
            with gr.Row():
                video_input = gr.Video(label="Upload Video", interactive=True)
                metadata_output = gr.JSON(label="Generation Parameters")

            with gr.Row():
                send_to_t2v_btn = gr.Button("Send to Text2Video", variant="primary")
                send_to_v2v_btn = gr.Button("Send to Video2Video", variant="primary")
                send_to_wanx_i2v_btn = gr.Button("Send to WanX-i2v", variant="primary")
                send_to_wanx_t2v_btn = gr.Button("Send to WanX-t2v", variant="primary")

            with gr.Row():
                status = gr.Textbox(label="Status", interactive=False)

        #Merge Model's tab        
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
                    if not input_file:
                        return "Error: No input file selected"

                    # Ensure output directory exists
                    os.makedirs("lora", exist_ok=True)

                    # Construct output path
                    output_path = os.path.join("lora", f"{output_name}.safetensors")

                    # Build command
                    cmd = [
                        sys.executable,
                        "convert_lora.py",
                        "--input", input_file.name,
                        "--output", output_path,
                        "--target", target_format
                    ]

                    print(f"Converting {input_file.name} to {output_path}")

                    # Execute conversion
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True
                    )

                    if os.path.exists(output_path):
                        return f"Successfully converted LoRA to {output_path}"
                    else:
                        return "Error: Output file not created"

                except subprocess.CalledProcessError as e:
                    return f"Error during conversion: {e.stderr}"
                except Exception as e:
                    return f"Error: {str(e)}"

            with gr.Row():
                input_file = gr.File(label="Input LoRA File", file_types=[".safetensors"])
                output_name = gr.Textbox(label="Output Name", placeholder="Output filename (without extension)")
                format_radio = gr.Radio(
                    choices=["default", "other"],
                    value="default",
                    label="Target Format",
                    info="Choose 'default' for H1111/MUSUBI format or 'other' for diffusion pipe format"
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
                    merge_refresh_btn = gr.Button("", elem_classes="refresh-btn")
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

    ### Video Info

    # 2. Add a simplified function to handle all tab transfers
    def handle_send_to_wanx_tab(metadata, target_tab):
        """Common handler for sending video parameters to WanX tabs"""
        if not metadata:
            return "No parameters to send", {}

        # Tab names for clearer messages
        tab_names = {
            'wanx_i2v': 'WanX-i2v',
            'wanx_t2v': 'WanX-t2v'
        }

        # Just pass through all parameters - we'll use them in the .then() function
        return f"Parameters ready for {tab_names.get(target_tab, target_tab)}", metadata

    def change_to_wanx_i2v_tab():
        return gr.Tabs(selected=4)  # WanX-i2v tab index

    def change_to_wanx_t2v_tab():
        return gr.Tabs(selected=5)  # WanX-t2v tab index

    send_to_wanx_i2v_btn.click(
        fn=lambda m: handle_send_to_wanx_tab(m, 'wanx_i2v'),
        inputs=[metadata_output],
        outputs=[status, params_state]
    ).then(
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
            params.get("task", "i2v-14B")
        ] if params else [gr.update()]*12,
        inputs=params_state,
        outputs=[
            wanx_prompt, 
            wanx_width, 
            wanx_height, 
            wanx_video_length, 
            wanx_fps, 
            wanx_infer_steps,
            wanx_seed,
            wanx_flow_shift, 
            wanx_guidance_scale,
            wanx_attn_mode,
            wanx_block_swap,
            wanx_task
        ]
    ).then(
        fn=change_to_wanx_i2v_tab, inputs=None, outputs=[tabs]
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
            params.get("block_swap", 0)
        ] if params else [gr.update()]*11,
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
            wanx_t2v_block_swap
        ]
    ).then(
        fn=change_to_wanx_t2v_tab, inputs=None, outputs=[tabs]
    )

    #text to video
    def change_to_tab_one():
        return gr.Tabs(selected=1) #This will navigate
    #video to video
    def change_to_tab_two():
        return gr.Tabs(selected=2) #This will navigate
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

    # Add skyreels_selected_index to the initial states at the beginning of the script
    skyreels_selected_index = gr.State(value=None)  # Add this with other state declarations    
    
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
        outputs=[original_dims, width, height]
    )

    scale_slider.change(
        fn=update_from_scale,
        inputs=[scale_slider, original_dims],
        outputs=[width, height]
    )

    calc_width_btn.click(
        fn=calculate_width,
        inputs=[height, original_dims],
        outputs=[width]
    )

    calc_height_btn.click(
        fn=calculate_height,
        inputs=[width, original_dims],
        outputs=[height]
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

    # Update DiT model dropdown
    def update_dit_dropdown(dit_folder: str) -> Dict:
        models = get_dit_models(dit_folder)
        return gr.update(choices=models, value=models[0] if models else None)

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

    # Generate button handler
    i2v_generate_btn.click(
        fn=process_batch,
        inputs=[
            i2v_prompt, width, height,
            i2v_batch_size, i2v_video_length, 
            i2v_fps, i2v_infer_steps, i2v_seed, i2v_dit_folder, i2v_model, i2v_vae, i2v_te1, i2v_te2,
            i2v_save_path, i2v_flow_shift, i2v_cfg_scale, i2v_output_type, i2v_attn_mode, 
            i2v_block_swap, i2v_exclude_single_blocks, i2v_use_split_attn, i2v_lora_folder, 
            *i2v_lora_weights, *i2v_lora_multipliers, i2v_input, i2v_strength, i2v_use_fp8
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
        fn=send_i2v_to_v2v,
        inputs=[
            i2v_output, i2v_prompt, i2v_selected_index,
            width, height,
            i2v_video_length, i2v_fps, i2v_infer_steps,
            i2v_seed, i2v_flow_shift, i2v_cfg_scale
        ] + i2v_lora_weights + i2v_lora_multipliers,
        outputs=[
            v2v_input, v2v_prompt,
            v2v_width, v2v_height,
            v2v_video_length, v2v_fps, v2v_infer_steps,
            v2v_seed, v2v_flow_shift, v2v_cfg_scale
        ] + v2v_lora_weights + v2v_lora_multipliers
    ).then(
        fn=change_to_tab_two, inputs=None, outputs=[tabs]
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
            params.get("width", 544),
            params.get("height", 544),
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
        ] if params else [gr.update()]*26,
        inputs=params_state,
        outputs=[prompt, width, height, batch_size, video_length, fps, infer_steps, seed, 
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
            wanx_task,
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
            wanx_fp8_t5,
            wanx_lora_folder,
            *wanx_lora_weights,
            *wanx_lora_multipliers,
            wanx_input  # Include input image path for non-batch mode
        ],
        outputs=[wanx_output, wanx_batch_progress, wanx_progress_text],
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[wanx_batch_size],
        outputs=skyreels_selected_index  # Reusing existing state variable
    )
    
    # Add refresh button handler for WanX-i2v tab
    wanx_refresh_outputs = []
    for i in range(4):
        wanx_refresh_outputs.extend([wanx_lora_weights[i], wanx_lora_multipliers[i]])
    
    wanx_refresh_btn.click(
        fn=update_lora_dropdowns,
        inputs=[wanx_lora_folder] + wanx_lora_weights + wanx_lora_multipliers,
        outputs=wanx_refresh_outputs
    )
    
    # Gallery selection handling
    wanx_output.select(
        fn=handle_wanx_gallery_select,
        outputs=skyreels_selected_index  # Reuse the skyreels_selected_index
    )
    
    # Send to Video2Video handler
    wanx_send_to_v2v_btn.click(
        fn=send_wanx_to_v2v,
        inputs=[
            wanx_output, 
            wanx_prompt, 
            skyreels_selected_index,  # Reuse the skyreels_selected_index
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

        # Add state for T2V tab selected index
    wanx_t2v_selected_index = gr.State(value=None)

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
        fn=wanx_generate_video_batch,
        inputs=[
            wanx_t2v_prompt, 
            wanx_t2v_negative_prompt,
            wanx_t2v_width,
            wanx_t2v_height,
            wanx_t2v_video_length,
            wanx_t2v_fps,
            wanx_t2v_infer_steps,
            wanx_t2v_flow_shift,
            wanx_t2v_guidance_scale,
            wanx_t2v_seed,
            wanx_t2v_task,
            wanx_t2v_dit_path,
            wanx_t2v_vae_path,
            wanx_t2v_t5_path,
            wanx_t2v_clip_path,
            wanx_t2v_save_path,
            wanx_t2v_output_type,
            wanx_t2v_sample_solver,
            wanx_t2v_exclude_single_blocks,
            wanx_t2v_attn_mode,
            wanx_t2v_block_swap,
            wanx_t2v_fp8,
            wanx_t2v_fp8_t5, 
            wanx_t2v_lora_folder,
            *wanx_t2v_lora_weights,
            *wanx_t2v_lora_multipliers,
            wanx_t2v_batch_size,
            # input_image is now optional and not included here
        ],
        outputs=[wanx_t2v_output, wanx_t2v_batch_progress, wanx_t2v_progress_text],
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[wanx_t2v_batch_size],
        outputs=wanx_t2v_selected_index
    )
    
    # Add refresh button handler for WanX-t2v tab
    wanx_t2v_refresh_outputs = []
    for i in range(4):
        wanx_t2v_refresh_outputs.extend([wanx_t2v_lora_weights[i], wanx_t2v_lora_multipliers[i]])
    
    wanx_t2v_refresh_btn.click(
        fn=update_lora_dropdowns,
        inputs=[wanx_t2v_lora_folder] + wanx_t2v_lora_weights + wanx_t2v_lora_multipliers,
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

demo.queue().launch(server_name="0.0.0.0", share=False)