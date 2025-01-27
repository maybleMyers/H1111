import gradio as gr
import subprocess
import threading
import time
import re
import os
import random
import tiktoken
import sys
from typing import List, Tuple, Optional, Generator

# Add global stop event
stop_event = threading.Event()

def count_prompt_tokens(prompt: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(prompt)
    return len(tokens)

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
    video_size: str,
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
    block_swap: str,
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
) -> List[Tuple[str, str]]:  
    global stop_event
    stop_event.clear()  # Reset stop event at start

    # Validate video dimensions
    match = re.match(r'(\d+)\s+(\d+)', video_size)
    if not match:
        print("Invalid video size format. Use 'Width Height'.")
        return []
    
    width, height = int(match.group(1)), int(match.group(2))
    if height % 8 != 0 or width % 8 != 0:
        print(f"Video dimensions must be divisible by 8. Current dimensions are {height}x{width}.")
        return []

    generated_videos: List[Tuple[str, str]] = []
    seeds: List[int] = []

    for batch_idx in range(batch_size):
        if stop_event.is_set():
            break  # Exit early if stopped

    # Process one video at a time
    for batch_idx in range(batch_size):
        current_seed = random.randint(0, 2**32 - 1) if (batch_size > 1 or seed == -1) else seed
        seeds.append(current_seed)
        
        batch_message = f"\n--- Generating video {batch_idx + 1} of {batch_size} (seed: {current_seed}) ---\n"
        print(batch_message)
        
        command = [
            sys.executable,
            "hv_generate_video.py",
            "--dit", model,
            "--vae", vae,
            "--text_encoder1", te1,
            "--text_encoder2", te2,
            "--prompt", prompt,
            "--video_size", str(width), str(height),
            "--video_length", str(video_length),
            "--fps", str(fps),
            "--infer_steps", str(infer_steps),
            "--save_path", save_path,
            "--seed", str(current_seed),
            "--fp8",
            "--flow_shift", str(flow_shift),
            "--embedded_cfg_scale", str(cfg_scale),
            "--output_type", output_type,
            "--attn_mode", attn_mode,
            "--blocks_to_swap", block_swap,
            "--split_attn",
            "--fp8_llm",
            "--vae_chunk_size", "32",
            "--vae_spatial_tile_sample_min_size", "128"            

        ]

        # Add LoRA weights and multipliers if provided
        lora_weights = [lora1, lora2, lora3, lora4]
        lora_multipliers = [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier]
        
        # Filter out empty LoRA paths
        valid_loras = [(weight, mult) for weight, mult in zip(lora_weights, lora_multipliers) if weight.strip()]
        
        if valid_loras:
            command.extend(["--lora_weight"] + [weight for weight, _ in valid_loras])
            command.extend(["--lora_multiplier"] + [str(mult) for _, mult in valid_loras])

        # Add video2video parameters if provided
        if video_path:
            command.extend(["--video_path", video_path])
            if strength is not None:
                command.extend(["--strength", str(strength)])
        
        # Get current environment variables
        env = os.environ.copy()
        env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
        
        # Run single process with direct console output
        p = subprocess.Popen(
            command,
            stdout=None,
            stderr=None,
            env=env,
            executable=sys.executable
        )

        # Modified wait with stop check
        while True:
            retcode = p.poll()
            if retcode is not None:
                break
            if stop_event.is_set():
                p.terminate()
                p.wait()
                print("Generation stopped by user.")
                return generated_videos  # Return videos generated so far
            time.sleep(0.5)
            
        # Find the most recently generated video after each batch
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
                caption = f"Seed: {current_seed}"
                generated_videos.append((str(video_path), caption))

    return generated_videos

# UI setup
with gr.Blocks(css="""
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
""") as demo:
    # Add state for tracking selected video indices in both tabs
    selected_index = gr.State(value=None)  # For Text to Video
    v2v_selected_index = gr.State(value=None)  # For Video to Video
    
    with gr.Tabs() as tabs:
        # Text to Video Tab
        with gr.Tab("Text to Video"):
            with gr.Row():
                prompt = gr.Textbox(label="Enter your prompt", value="POV video of a cat chasing a frob.", scale=2)
                token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)

            with gr.Row():
                lora_weights = []
                lora_multipliers = []
                for i in range(4):
                    with gr.Column():
                        lora_weights.append(gr.Textbox(
                            label=f"LoRA {i+1}", 
                            value="", 
                            placeholder="Path to LoRA weight file"
                        ))
                        lora_multipliers.append(gr.Slider(
                            label=f"Multiplier", 
                            minimum=0.0, 
                            maximum=2.0, 
                            step=0.05, 
                            value=1.0
                        ))
            
            with gr.Row():
                video_size = gr.Textbox(label="Video Size (Width Height)", value="544 544", info="Space-separated values, must be divisible by 8")
                batch_size = gr.Number(label="Batch Size", value=1, minimum=1, step=1)
                video_length = gr.Slider(minimum=1, maximum=201, step=1, label="Video Length in Frames", value=25)
                fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=24)
                infer_steps = gr.Slider(minimum=10, maximum=100, step=10, label="Inference Steps", value=30)
                seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=11.0)
                cfg_scale = gr.Slider(minimum=0.0, maximum=14.0, step=0.1, label="cfg Scale", value=7.0)

            with gr.Row():
                generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                stop_btn = gr.Button("Stop Generation", variant="stop")

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
            
            send_to_v2v_btn = gr.Button("Send Selected to Video2Video")
            
            with gr.Row():
                model = gr.Textbox(label="Enter dit location", value="hunyuan/mp_rank_00_model_states.pt")
                vae = gr.Textbox(label="vae", value="hunyuan/pytorch_model.pt")
                te1 = gr.Textbox(label="te1", value="hunyuan/llava_llama3_fp16.safetensors")
                te2 = gr.Textbox(label="te2", value="hunyuan/clip_l.safetensors")
                save_path = gr.Textbox(label="Save Path", value="outputs")
                output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                block_swap = gr.Textbox(label="Blocks to Swap to save vram", value="0")

        # Video to Video Tab
        with gr.Tab("Video to Video") as v2v_tab:
            with gr.Row():
                v2v_prompt = gr.Textbox(label="Enter your prompt", value="POV video of a cat chasing a frob.", scale=2)
                v2v_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)

            with gr.Row():
                v2v_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                v2v_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column():
                    v2v_input = gr.Video(label="Input Video", format="mp4")
                    v2v_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.75, label="Denoise Strength")
                with gr.Column():
                    v2v_output = gr.Gallery(
                        label="Generated Videos",
                        columns=[1],
                        rows=[1],
                        object_fit="contain",
                        height="auto"
                    )
                    v2v_send_to_input_btn = gr.Button("Send Selected to Input")  # New button
            
            with gr.Row():
                v2v_video_size = gr.Textbox(label="Video Size (Width Height)", value="544 544", info="Space-separated values, must be divisible by 8")
                # Add these as actual components instead of invisible values
                v2v_batch_size = gr.Number(label="Batch Size", value=1, minimum=1, step=1)
                v2v_video_length = gr.Slider(minimum=1, maximum=201, step=1, label="Video Length in Frames", value=25)
                v2v_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=24)
                v2v_infer_steps = gr.Slider(minimum=10, maximum=100, step=10, label="Inference Steps", value=30)
                v2v_seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                v2v_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=11.0)
                v2v_cfg_scale = gr.Slider(minimum=0.0, maximum=14.0, step=0.1, label="cfg scale", value=7.0)
            
            # Add LoRA inputs for Video2Video tab
            with gr.Row():
                v2v_lora_weights = []
                v2v_lora_multipliers = []
                for i in range(4):
                    with gr.Column():
                        v2v_lora_weights.append(gr.Textbox(
                            label=f"LoRA {i+1}", 
                            value="", 
                            placeholder="Path to LoRA weight file"
                        ))
                        v2v_lora_multipliers.append(gr.Slider(
                            label=f"Multiplier", 
                            minimum=0.0, 
                            maximum=2.0, 
                            step=0.05, 
                            value=1.0
                        ))

            with gr.Row():
                v2v_model = gr.Textbox(label="Enter dit location", value="hunyuan/mp_rank_00_model_states.pt")
                v2v_vae = gr.Textbox(label="vae", value="hunyuan/pytorch_model.pt")
                v2v_te1 = gr.Textbox(label="te1", value="hunyuan/llava_llama3_fp16.safetensors")
                v2v_te2 = gr.Textbox(label="te2", value="hunyuan/clip_l.safetensors")
                v2v_save_path = gr.Textbox(label="Save Path", value="outputs")
                v2v_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                v2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                v2v_block_swap = gr.Textbox(label="Blocks to Swap to save vram", value="0")

    # Event handlers
    prompt.change(fn=count_prompt_tokens, inputs=prompt, outputs=token_counter)
    v2v_prompt.change(fn=count_prompt_tokens, inputs=v2v_prompt, outputs=v2v_token_counter)
    stop_btn.click(fn=lambda: stop_event.set(), queue=False)
    v2v_stop_btn.click(fn=lambda: stop_event.set(), queue=False)

    # Text to Video generation
    generate_btn.click(
        fn=generate_video,
        inputs=[
            prompt, video_size, batch_size, video_length, fps, infer_steps, 
            seed, model, vae, te1, te2, save_path, flow_shift, cfg_scale, output_type, attn_mode, block_swap
        ] + lora_weights + lora_multipliers,
        outputs=video_output
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
    def handle_send_button(gallery: list, prompt: str, idx: int) -> Tuple[Optional[str], str]:
        if not gallery or idx is None or idx >= len(gallery):
            return None, ""
        
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
        
        return str(video_path), prompt
    
            # Send button click handler
    send_to_v2v_btn.click(
        fn=handle_send_button,
        inputs=[video_output, prompt, selected_index],
        outputs=[v2v_input, v2v_prompt]
    ).then(lambda: gr.Tabs(selected="Video to Video"), outputs=tabs)

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


    # Connect the new button's click event
    v2v_send_to_input_btn.click(
        fn=handle_v2v_send_button,
        inputs=[v2v_output, v2v_prompt, v2v_selected_index],
        outputs=[v2v_input, v2v_prompt]
    )


    # Video to Video generation
    v2v_generate_btn.click(
        fn=generate_video,
        inputs=[
            v2v_prompt, v2v_video_size, v2v_batch_size, v2v_video_length, 
            v2v_fps, v2v_infer_steps, v2v_seed, v2v_model, v2v_vae, 
            v2v_te1, v2v_te2, v2v_save_path, v2v_flow_shift, v2v_cfg_scale, v2v_output_type, v2v_attn_mode, v2v_block_swap
        ] + v2v_lora_weights + v2v_lora_multipliers + [v2v_input, v2v_strength],
        outputs=v2v_output
    )

demo.launch(server_name="0.0.0.0", share=False) 