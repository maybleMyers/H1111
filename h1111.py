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

# Add global stop event
stop_event = threading.Event()

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

def generate_video(
    prompt: str,
    width: int,  # Changed from video_size
    height: int,
    batch_size: int,
    video_length: int,
    fps: int,
    infer_steps: int,
    seed: int,
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
    strength: Optional[float] = None
) -> Generator[List[Tuple[str, str]], None, None]:  
    global stop_event
    stop_event.clear()  # Reset stop event at start

    progress_text = "Starting generation..."
    batch_text = "Preparing..."
    yield [], batch_text, progress_text

    generated_videos = []
    seeds = []

    for batch_idx in range(batch_size):
        if stop_event.is_set():
            break  # Exit early if stopped

        current_seed = random.randint(0, 2**32 - 1) if (batch_size > 1 or seed == -1) else seed
        seeds.append(current_seed)
        
        batch_text = f"Generating video {batch_idx + 1} of {batch_size} (seed: {current_seed})"
        print(f"\n--- {batch_text} ---\n")
        yield generated_videos.copy(), batch_text, progress_text
        
        command = [
            sys.executable,
            "hv_generate_video.py",
            "--dit", model,
            "--vae", vae,
            "--text_encoder1", te1,
            "--text_encoder2", te2,
            "--prompt", prompt,
            "--video_size", str(height), str(width), 
            "--video_length", str(video_length),
            "--fps", str(fps),
            "--infer_steps", str(infer_steps),
            "--save_path", save_path,
            "--seed", str(current_seed),  # Ensure seed is converted to string
            "--fp8",
            "--flow_shift", str(flow_shift),
            "--embedded_cfg_scale", str(cfg_scale),
            "--output_type", output_type,
            "--attn_mode", attn_mode,
            "--blocks_to_swap", str(block_swap),
            "--fp8_llm",
            "--vae_chunk_size", "32",
            "--vae_spatial_tile_sample_min_size", "128"            
        ]

        # Add LoRA weights and multipliers if provided
        valid_loras = []
        for weight, mult in zip([lora1, lora2, lora3, lora4], [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier]):
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

        # Add video2video parameters if provided
        if video_path:
            command.extend(["--video_path", video_path])
            if strength is not None:
                command.extend(["--strength", str(strength)])
        
        # Get current environment variables
        env = os.environ.copy()
        env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
        # Explicitly set PYTHONIOENCODING to ensure proper UTF-8 handling
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Modified subprocess setup with stdout capture
        p = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,  # Use text mode instead of universal_newlines
            encoding='utf-8',  # Explicitly specify UTF-8 encoding
            errors='replace',  # Replace invalid characters
            bufsize=1
        )

        # Read output line by line and print to console
        while True:
            line = p.stdout.readline()
            if not line:
                if p.poll() is not None:
                    break
                continue
                
            # Print to console immediately
            print(line)
            
            # Check for progress percentage
            if '|' in line and '%' in line and '[' in line and ']' in line:
                # This captures lines that look like: 76%|██████████████████████████▉    | 38/50 [00:19<00:06,  1.92it/s]
                progress_text = line
                yield generated_videos.copy(), batch_text, progress_text

            if stop_event.is_set():
                p.terminate()
                p.wait()
                print("Generation stopped by user.")
                return

        p.stdout.close()
        p.wait()

        # Collect generated videos after each batch
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
                    "batch_size": batch_size,
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
                    "input_video": video_path,
                    "strength": strength
                }
                
                # Add metadata to the video file
                add_metadata_to_video(video_path, parameters)
                
                # Append to generated_videos
                generated_videos.append((str(video_path), f"Seed: {current_seed}"))

                yield generated_videos.copy(), batch_text, ""

    yield generated_videos, "", ""

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
    demo.load(None, None, None, js="""
    () => {
        document.title = 'H1111';
        
        function updateTitle(text) {
            if (text && text.trim()) {
                const percentMatch = text.match(/(\d+)%/);
                if (percentMatch) {
                    document.title = `${percentMatch[1]}% - H1111`;
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
        with gr.Tab("Text to Video"):
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
                    
                    width = gr.Slider(minimum=64, maximum=1536, step=8, value=544, label="Video Width", elem_id="my_special_slider")
                    height = gr.Slider(minimum=64, maximum=1536, step=8, value=544, label="Video Height", elem_id="my_special_slider")
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
                model = gr.Textbox(label="Enter dit location", value="hunyuan/mp_rank_00_model_states.pt")
                vae = gr.Textbox(label="vae", value="hunyuan/pytorch_model.pt")
                te1 = gr.Textbox(label="te1", value="hunyuan/llava_llama3_fp16.safetensors")
                te2 = gr.Textbox(label="te2", value="hunyuan/clip_l.safetensors")
                save_path = gr.Textbox(label="Save Path", value="outputs")
            with gr.Row():
                lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                use_split_attn = gr.Checkbox(label="Use Split Attention", value=False)
                attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                block_swap = gr.Slider(minimum=0, maximum=36, step=1, label="Block Swap to Save Vram", value=0)

        #Image to Video Tab
        with gr.Tab("Image to Video") as i2v_tab:
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
                    i2v_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.75, label="Denoise Strength")
                    i2v_max_res = gr.Number(label="Max Resolution (Width or Height)", value=544, step=8, minimum=64, maximum=1536)
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
                i2v_model = gr.Textbox(label="Enter dit location", value="hunyuan/mp_rank_00_model_states.pt")
                i2v_vae = gr.Textbox(label="vae", value="hunyuan/pytorch_model.pt")
                i2v_te1 = gr.Textbox(label="te1", value="hunyuan/llava_llama3_fp16.safetensors")
                i2v_te2 = gr.Textbox(label="te2", value="hunyuan/clip_l.safetensors")
                i2v_save_path = gr.Textbox(label="Save Path", value="outputs")
            with gr.Row():
                i2v_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                i2v_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                i2v_use_split_attn = gr.Checkbox(label="Use Split Attention", value=False)
                i2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                i2v_block_swap = gr.Slider(minimum=0, maximum=36, step=1, label="Block Swap to Save Vram", value=0)

        # Video to Video Tab
        with gr.Tab("Video to Video") as v2v_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    v2v_prompt = gr.Textbox(scale=3, label="Enter your prompt", value="POV video of a cat chasing a frob.", lines=5)

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
                    v2v_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.75, label="Denoise Strength")
                    v2v_width = gr.Slider(minimum=64, maximum=1536, step=8, value=544, label="Video Width")
                    v2v_height = gr.Slider(minimum=64, maximum=1536, step=8, value=544, label="Video Height")
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
                v2v_model = gr.Textbox(label="Enter dit location", value="hunyuan/mp_rank_00_model_states.pt")
                v2v_vae = gr.Textbox(label="vae", value="hunyuan/pytorch_model.pt")
                v2v_te1 = gr.Textbox(label="te1", value="hunyuan/llava_llama3_fp16.safetensors")
                v2v_te2 = gr.Textbox(label="te2", value="hunyuan/clip_l.safetensors")
                v2v_save_path = gr.Textbox(label="Save Path", value="outputs")
            with gr.Row():
                v2v_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                v2v_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                v2v_use_split_attn = gr.Checkbox(label="Use Split Attention", value=False)
                v2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                v2v_block_swap = gr.Slider(minimum=0, maximum=36, step=1, label="Block Swap to Save Vram", value=0)
        with gr.Tab("Video Info") as video_info_tab:
            with gr.Row():
                video_input = gr.Video(label="Upload Video", interactive=True)
                metadata_output = gr.JSON(label="Generation Parameters")

            with gr.Row():
                send_to_t2v_btn = gr.Button("Send to Text2Video", variant="primary")
                send_to_v2v_btn = gr.Button("Send to Video2Video", variant="primary")

            with gr.Row():
                status = gr.Textbox(label="Status", interactive=False)
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

    # Event handlers
    prompt.change(fn=count_prompt_tokens, inputs=prompt, outputs=token_counter)
    v2v_prompt.change(fn=count_prompt_tokens, inputs=v2v_prompt, outputs=v2v_token_counter)
    stop_btn.click(fn=lambda: stop_event.set(), queue=False)
    v2v_stop_btn.click(fn=lambda: stop_event.set(), queue=False)

    #Image_to_Video
    def image_to_video(image_path, output_path, frames=240, max_res=None):
        img = Image.open(image_path)
        width, height = img.size

        # Determine new dimensions
        if max_res:
            if width > height:
                aspect_ratio = width / height
                new_width = min(width, max_res)
                new_height = int(new_width / aspect_ratio)
            else:
                aspect_ratio = height / width
                new_height = min(height, max_res)
                new_width = int(new_height / aspect_ratio)
        else:
            new_width, new_height = width, height

        # Ensure dimensions are divisible by 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8

        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        temp_image_path = os.path.join(os.path.dirname(output_path), "temp_resized_image.png")
        img_resized.save(temp_image_path)
    
        # Use FFmpeg to create the video
        frame_rate = 24
        duration = frames / frame_rate  # Convert frames to duration in seconds
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
        prompt, max_res, video_length, fps, infer_steps,
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
            if not image_to_video(image_path, temp_video_path, frames=video_length, max_res=max_res):
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
                generator = generate_video(
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

    def send_i2v_to_v2v(gallery: list, prompt: str, selected_index: int) -> Tuple[Optional[str], str]:
        """Send the selected video from Image2Video tab to Video2Video tab"""
        if not gallery or selected_index is None or selected_index >= len(gallery):
            return None, ""

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

        return str(video_path), prompt

    # Generate button handler
    i2v_generate_btn.click(
        fn=generate_from_image,
        inputs=[
            i2v_input, i2v_prompt, i2v_max_res, i2v_video_length, i2v_fps, i2v_infer_steps,
            i2v_seed, i2v_model, i2v_vae, i2v_te1, i2v_te2, i2v_save_path, i2v_flow_shift, i2v_cfg_scale, 
            i2v_output_type, i2v_attn_mode, i2v_block_swap, i2v_exclude_single_blocks, i2v_use_split_attn,
            i2v_lora_folder, i2v_strength, i2v_batch_size
        ] + i2v_lora_weights + i2v_lora_multipliers,
        outputs=[i2v_output, i2v_batch_progress, i2v_progress_text]
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
        inputs=[i2v_output, i2v_prompt, i2v_selected_index],
        outputs=[v2v_input, v2v_prompt]
    ).then(
        lambda: gr.update(selected="Video to Video"),
        outputs=tabs
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
        lambda: gr.Tabs(selected="Text to Video"),
        outputs=tabs
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
        fn=generate_video,
        inputs=[
            prompt, width, height, batch_size, video_length, fps, infer_steps,  # Updated inputs
            seed, model, vae, te1, te2, save_path, flow_shift, cfg_scale, 
            output_type, attn_mode, block_swap, exclude_single_blocks, use_split_attn, lora_folder
        ] + lora_weights + lora_multipliers,
        outputs=[video_output, batch_progress, progress_text]
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
        return evt.index

    v2v_output.select(
        fn=handle_v2v_gallery_select,
        outputs=v2v_selected_index
    )
    
    # Send button handler with gallery selection
    def handle_send_button(
        gallery: list, 
        prompt: str, 
        idx: int, 
        width: int,  # Changed from video_size
        height: int,  # Added height
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
                    lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier)

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
            width,  # Changed
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
            lora4_multiplier
        )
    
    send_t2v_to_v2v_btn.click(
        fn=handle_send_button,
        inputs=[
            video_output, prompt, selected_index,
            width, height, batch_size, video_length,
            fps, infer_steps, seed, flow_shift, cfg_scale
        ] + lora_weights + lora_multipliers,
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
        ] + v2v_lora_weights + v2v_lora_multipliers
    ).then(
        lambda: gr_update(selected="Video to Video"),
        outputs=tabs
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
        lambda: gr.Tabs(selected="Video to Video"),
        outputs=tabs
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
    )

    # Handler for sending selected video from Video2Video gallery to input
    def handle_v2v_send_button(gallery: list, prompt: str, idx: int) -> Tuple[Optional[str], str]:
        """Send the currently selected video in V2V gallery to V2V input"""
        if not gallery or idx is None or idx >= len(gallery):
            return None, ""

        selected_item = gallery[idx]

        # Handle different gallery item formats (same as Text to Video)
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

    v2v_send_to_input_btn.click(
        fn=handle_v2v_send_button,
        inputs=[v2v_output, v2v_prompt, v2v_selected_index],
        outputs=[v2v_input, v2v_prompt]
    )

    # Video to Video generation
    v2v_generate_btn.click(
        fn=generate_video,
        inputs=[
            v2v_prompt, v2v_width, v2v_height, v2v_batch_size, v2v_video_length,  # Updated inputs
            v2v_fps, v2v_infer_steps, v2v_seed, v2v_model, v2v_vae, 
            v2v_te1, v2v_te2, v2v_save_path, v2v_flow_shift, v2v_cfg_scale, 
            v2v_output_type, v2v_attn_mode, v2v_block_swap, v2v_exclude_single_blocks, v2v_use_split_attn, v2v_lora_folder
        ] + v2v_lora_weights + v2v_lora_multipliers + [v2v_input, v2v_strength],
        outputs=[v2v_output, v2v_batch_progress, v2v_progress_text]
    )
    refresh_outputs = []
    for i in range(4):
        refresh_outputs.extend([lora_weights[i], lora_multipliers[i]])
    

    refresh_btn.click(
        fn=update_lora_dropdowns,
        inputs=[lora_folder] + lora_weights + lora_multipliers,  # Include both weights and multipliers
        outputs=refresh_outputs
    )

    i2v_refresh_outputs = []
    for i in range(4):
        i2v_refresh_outputs.extend([i2v_lora_weights[i], i2v_lora_multipliers[i]])

    i2v_refresh_btn.click(
        fn=update_lora_dropdowns,
        inputs=[i2v_lora_folder] + i2v_lora_weights + i2v_lora_multipliers,
        outputs=i2v_refresh_outputs
    )
    
    v2v_refresh_outputs = []
    for i in range(4):
        v2v_refresh_outputs.extend([v2v_lora_weights[i], v2v_lora_multipliers[i]])
        
    v2v_refresh_btn.click(
        fn=update_lora_dropdowns,
        inputs=[v2v_lora_folder] + v2v_lora_weights + v2v_lora_multipliers,  # Include both weights and multipliers
        outputs=v2v_refresh_outputs
    )


demo.launch(server_name="0.0.0.0", share=False) 