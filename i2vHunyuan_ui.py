import os
import time
import random
import gradio as gr
import torch
import accelerate
import threading
import sys
import math
from pathlib import Path
from datetime import datetime
from PIL import Image

# Import functions from our i2vHunyuan.py script
from i2vHunyuan import (
    generate_video, 
    batch_process_folder, 
    resize_image_keeping_aspect_ratio,
    get_random_image_from_folder,
    clean_memory_on_device,
    stop_event
)

# Global stop event
stop_event = threading.Event()

def update_dimensions(image):
    """Update dimensions from uploaded image"""
    if image is None:
        return "", gr.update(value=544), gr.update(value=544)
    try:
        img = Image.open(image)
        w, h = img.size
        w = (w // 16) * 16
        h = (h // 16) * 16
        return f"{w}x{h}", w, h
    except Exception as e:
        print(f"Error loading image: {e}")
        return "", gr.update(value=544), gr.update(value=544)

def calculate_width(height, original_dims):
    """Calculate width based on height maintaining aspect ratio"""
    if not original_dims:
        return gr.update()
    try:
        orig_w, orig_h = map(int, original_dims.split('x'))
        aspect_ratio = orig_w / orig_h
        new_width = math.floor((height * aspect_ratio) / 16) * 16
        return gr.update(value=new_width)
    except:
        return gr.update()

def calculate_height(width, original_dims):
    """Calculate height based on width maintaining aspect ratio"""
    if not original_dims:
        return gr.update()
    try:
        orig_w, orig_h = map(int, original_dims.split('x'))
        aspect_ratio = orig_w / orig_h
        new_height = math.floor((width / aspect_ratio) / 16) * 16
        return gr.update(value=new_height)
    except:
        return gr.update()

def update_from_scale(scale, original_dims):
    """Update dimensions based on scale percentage"""
    if not original_dims:
        return gr.update(), gr.update()
    try:
        orig_w, orig_h = map(int, original_dims.split('x'))
        new_w = math.floor((orig_w * scale / 100) / 16) * 16
        new_h = math.floor((orig_h * scale / 100) / 16) * 16
        return gr.update(value=new_w), gr.update(value=new_h)
    except:
        return gr.update(), gr.update()

def recommend_flow_shift(width, height):
    """Get recommended flow shift value based on dimensions"""
    if (width == 832 and height == 480) or (width == 480 and height == 832):
        return gr.update(value=3.0)
    if width <= 512 and height <= 512:
        return gr.update(value=5.0)
    return gr.update(value=11.0)

def handle_gallery_select(evt: gr.SelectData) -> int:
    """Track selected index when gallery item is clicked"""
    return evt.index

class Args:
    """Simple class to hold arguments for the model"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def create_args(
    prompt, negative_prompt, image_path, input_folder, save_path,
    dit, vae, text_encoder1, text_encoder2, 
    i2v_resolution, prompt_template_video, prompt_template,
    video_length, fps, infer_steps, guidance_scale, embedded_cfg_scale, flow_shift, seed,
    batch_size, max_width, max_height,
    fp8, fp8_llm, no_metadata, device, attn_mode, split_attn, split_uncond,
    blocks_to_swap, img_in_txt_in_offloading, vae_chunk_size, vae_spatial_tile_sample_min_size
):
    """Create Args object from UI parameters"""
    return Args(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_path=image_path,
        input_folder=input_folder,
        save_path=save_path,
        dit=dit,
        vae=vae,
        text_encoder1=text_encoder1,
        text_encoder2=text_encoder2,
        i2v_resolution=i2v_resolution,
        prompt_template_video=prompt_template_video,
        prompt_template=prompt_template,
        video_length=video_length,
        fps=fps,
        infer_steps=infer_steps,
        guidance_scale=guidance_scale,
        embedded_cfg_scale=embedded_cfg_scale,
        flow_shift=flow_shift,
        seed=seed,
        batch_size=batch_size,
        max_width=max_width,
        max_height=max_height,
        fp8=fp8,
        fp8_llm=fp8_llm,
        no_metadata=no_metadata,
        device=device,
        attn_mode=attn_mode,
        split_attn=split_attn,
        split_uncond=split_uncond,
        blocks_to_swap=blocks_to_swap,
        img_in_txt_in_offloading=img_in_txt_in_offloading,
        vae_chunk_size=vae_chunk_size,
        vae_spatial_tile_sample_min_size=vae_spatial_tile_sample_min_size
    )

def generate_from_ui(
    prompt, negative_prompt, input_image, width, height, video_length,
    fps, infer_steps, flow_shift, guidance_scale, embedded_cfg_scale, seed,
    use_random_folder, input_folder, batch_size,
    dit_path, vae_path, te1_path, te2_path, save_path, i2v_resolution,
    attn_mode, block_swap, split_attn, split_uncond, fp8, fp8_llm,
    add_metadata, img_in_txt_in_offloading, vae_chunk_size, vae_spatial_tile_sample_min_size
):
    """Generate video from UI inputs"""
    # Reset stop event
    stop_event.clear()
    
    # Check inputs
    if not use_random_folder and input_image is None:
        return [], "Error: No input image provided", "Please upload an input image or enable random folder selection"
    
    if use_random_folder and (not input_folder or not os.path.isdir(input_folder)):
        return [], "Error: Invalid input folder", "Please provide a valid input folder path"
    
    # Create results list
    all_videos = []
    
    # Prepare device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create accelerator
    mixed_precision = "bf16"
    accelerator = accelerate.Accelerator(mixed_precision=mixed_precision)
    
    # Prepare common args
    args = create_args(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_path=None if use_random_folder else input_image,
        input_folder=input_folder if use_random_folder else None,
        save_path=save_path,
        dit=dit_path,
        vae=vae_path,
        text_encoder1=te1_path,
        text_encoder2=te2_path,
        i2v_resolution=i2v_resolution,
        prompt_template_video="dit-llm-encode-video-i2v",
        prompt_template="dit-llm-encode-i2v",
        video_length=video_length,
        fps=fps,
        infer_steps=infer_steps,
        guidance_scale=guidance_scale,
        embedded_cfg_scale=embedded_cfg_scale,
        flow_shift=flow_shift,
        seed=-1 if seed == -1 else seed,
        batch_size=batch_size,
        max_width=width,
        max_height=height,
        fp8=fp8,
        fp8_llm=fp8_llm,
        no_metadata=not add_metadata,
        device=device,
        attn_mode=attn_mode,
        split_attn=split_attn,
        split_uncond=split_uncond,
        blocks_to_swap=block_swap,
        img_in_txt_in_offloading=img_in_txt_in_offloading,
        vae_chunk_size=vae_chunk_size,
        vae_spatial_tile_sample_min_size=vae_spatial_tile_sample_min_size
    )
    
    try:
        # Track progress
        progress_text = "Starting generation..."
        yield [], "Preparing...", progress_text

        # Process based on mode
        if use_random_folder:
            # Generate from folder
            for i in range(batch_size):
                if stop_event.is_set():
                    yield all_videos, "Generation stopped by user", ""
                    return
                
                batch_text = f"Generating video {i + 1} of {batch_size}"
                yield all_videos.copy(), batch_text, progress_text
                
                # Get random image
                random_image, status = get_random_image_from_folder(input_folder)
                if random_image is None:
                    yield all_videos, f"Error in batch {i+1}: {status}", ""
                    continue
                
                # Resize image
                resized_image, size_info = resize_image_keeping_aspect_ratio(random_image, width, height)
                if resized_image is None:
                    yield all_videos, f"Error resizing image in batch {i+1}: {size_info}", ""
                    continue
                
                # Use dimensions from resized image
                if isinstance(size_info, tuple):
                    local_width, local_height = size_info
                    progress_text = f"Using image: {os.path.basename(random_image)} - Resized to {local_width}x{local_height}"
                else:
                    local_width, local_height = width, height
                    progress_text = f"Using image: {os.path.basename(random_image)}"
                
                yield all_videos.copy(), batch_text, progress_text
                
                # Generate videos
                args.image_path = resized_image
                
                # Calculate seed for this batch item
                current_seed = seed
                if seed == -1:
                    current_seed = random.randint(0, 2**32 - 1)
                elif batch_size > 1:
                    current_seed = seed + i
                
                # Generate single video
                def progress_callback(step, total, batch_idx=i, total_batches=batch_size):
                    percent = int((step / total) * 100)
                    progress_msg = f"Batch {batch_idx+1}/{total_batches}: Generating [{percent}%] Step {step}/{total}"
                    return progress_msg
                
                for step in range(infer_steps):
                    if stop_event.is_set():
                        break
                    progress_msg = progress_callback(step+1, infer_steps)
                    yield all_videos.copy(), batch_text, progress_msg
                
                videos = generate_video(
                    args=args,
                    device=torch.device(device),
                    accelerator=accelerator,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image_path=resized_image,
                    batch_size=1,
                    seed=current_seed,
                    height=local_height,
                    width=local_width,
                    callback=None
                )
                
                # Add to results
                for video in videos:
                    all_videos.append((video, f"Seed: {current_seed}"))
                
                # Clean up temporary file
                try:
                    if os.path.exists(resized_image):
                        os.remove(resized_image)
                except:
                    pass
                
                # Clear CUDA cache between generations
                clean_memory_on_device(torch.device(device))
                time.sleep(0.5)
        else:
            # Single image mode
            batch_text = f"Generating {batch_size} videos"
            yield all_videos.copy(), batch_text, progress_text
            
            # Generate videos from the single image
            for i in range(batch_size):
                if stop_event.is_set():
                    yield all_videos, "Generation stopped by user", ""
                    return
                
                # Calculate seed for this batch
                current_seed = seed
                if seed == -1:
                    current_seed = random.randint(0, 2**32 - 1)
                elif batch_size > 1:
                    current_seed = seed + i
                
                # Progress callback
                def progress_callback(step, total, batch_idx=i, total_batches=batch_size):
                    percent = int((step / total) * 100)
                    progress_msg = f"Batch {batch_idx+1}/{total_batches}: Generating [{percent}%] Step {step}/{total}"
                    return progress_msg
                
                for step in range(infer_steps):
                    if stop_event.is_set():
                        break
                    progress_msg = progress_callback(step+1, infer_steps)
                    yield all_videos.copy(), batch_text, progress_msg
                
                # Generate video
                args.seed = current_seed
                
                videos = generate_video(
                    args=args,
                    device=torch.device(device),
                    accelerator=accelerator,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image_path=input_image,
                    batch_size=1,
                    seed=current_seed,
                    height=height,
                    width=width,
                    callback=None
                )
                
                # Add to results
                for video in videos:
                    all_videos.append((video, f"Seed: {current_seed}"))
                
                # Clear CUDA cache between generations
                clean_memory_on_device(torch.device(device))
                time.sleep(0.5)
        
        # Final result
        yield all_videos, "Generation complete", ""
        
    except Exception as e:
        import traceback
        error_msg = f"Error during generation: {str(e)}\n{traceback.format_exc()}"
        yield all_videos, "Error", error_msg
        
def create_ui():
    """Create Gradio UI"""
    with gr.Blocks(title="Hunyuan Image-to-Video") as demo:
        gr.HTML("<h1 style='text-align: center'>Hunyuan Image-to-Video</h1>")
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=4):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    value="A person walking on a beach at sunset",
                    lines=3
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt", 
                    placeholder="Enter negative prompt (optional)",
                    value="Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
                    lines=2
                )
            
            with gr.Column(scale=2):
                batch_progress = gr.Textbox(label="Status", interactive=False)
                progress_text = gr.Textbox(label="Progress", interactive=False)
        
        with gr.Row():
            with gr.Column():
                # Image input
                with gr.Row():
                    input_image = gr.Image(label="Input Image", type="filepath")
                    
                with gr.Row():
                    use_random_folder = gr.Checkbox(label="Use Random Images from Folder", value=False)
                    input_folder = gr.Textbox(
                        label="Image Folder Path",
                        placeholder="Path to folder with images",
                        visible=False
                    )
                
                # Dimensions
                original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=True)
                
                with gr.Row():
                    width = gr.Number(label="Width", value=544, precision=0)
                    calc_height_btn = gr.Button("→", elem_classes=["small-button"])
                    calc_width_btn = gr.Button("←", elem_classes=["small-button"])
                    height = gr.Number(label="Height", value=544, precision=0)
                
                scale_slider = gr.Slider(
                    minimum=10, maximum=200, value=100, step=1, 
                    label="Scale %"
                )
                
                with gr.Row():
                    recommend_flow_btn = gr.Button("Recommend Flow Shift")
                
                # Generation parameters
                with gr.Row():
                    video_length = gr.Slider(
                        minimum=1, maximum=201, value=25, step=1,
                        label="Video Length (frames)"
                    )
                    fps = gr.Slider(
                        minimum=1, maximum=60, value=24, step=1,
                        label="FPS"
                    )
                
                with gr.Row():
                    infer_steps = gr.Slider(
                        minimum=10, maximum=100, value=30, step=1,
                        label="Inference Steps"
                    )
                    flow_shift = gr.Slider(
                        minimum=0.0, maximum=28.0, value=11.0, step=0.5,
                        label="Flow Shift"
                    )
                
                with gr.Row():
                    guidance_scale = gr.Slider(
                        minimum=1.0, maximum=15.0, value=7.0, step=0.1, 
                        label="Guidance Scale"
                    )
                    embedded_cfg_scale = gr.Slider(
                        minimum=0.0, maximum=10.0, value=1.0, step=0.1,
                        label="Embedded CFG Scale"
                    )
                
                with gr.Row():
                    batch_size = gr.Slider(
                        minimum=1, maximum=10, value=1, step=1,
                        label="Batch Size"
                    )
                    seed = gr.Number(
                        label="Seed (-1 for random)", value=-1, precision=0
                    )
                
                with gr.Row():
                    i2v_resolution = gr.Dropdown(
                        label="Resolution Quality",
                        choices=["360p", "540p", "720p"],
                        value="720p"
                    )
                
            with gr.Column():
                # Output gallery
                output_gallery = gr.Gallery(
                    label="Generated Videos",
                    columns=[2],
                    rows=[2],
                    height="auto",
                    object_fit="contain"
                )
                
                with gr.Row():
                    generate_btn = gr.Button("Generate Video", variant="primary", interactive=True)
                    stop_btn = gr.Button("Stop Generation", variant="stop")
                
                # Model settings
                with gr.Accordion("Model Settings", open=False):
                    with gr.Row():
                        dit_path = gr.Textbox(
                            label="DiT Model Path",
                            value="hunyuan/mp_rank_00_model_states.pt"
                        )
                        vae_path = gr.Textbox(
                            label="VAE Path",
                            value="hunyuan/pytorch_model.pt"
                        )
                    
                    with gr.Row():
                        te1_path = gr.Textbox(
                            label="Text Encoder 1 Path",
                            value="hunyuan/llava_llama3_fp16.safetensors"
                        )
                        te2_path = gr.Textbox(
                            label="Text Encoder 2 Path",
                            value="hunyuan/clip_l.safetensors"
                        )
                    
                    save_path = gr.Textbox(
                        label="Save Path",
                        value="outputs"
                    )
                
                # Advanced settings
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        attn_mode = gr.Dropdown(
                            label="Attention Mode",
                            choices=["sdpa", "flash", "torch", "xformers"],
                            value="sdpa"
                        )
                        block_swap = gr.Slider(
                            minimum=0, maximum=36, value=0, step=1,
                            label="Block Swap (save VRAM)"
                        )
                    
                    with gr.Row():
                        fp8 = gr.Checkbox(label="Use FP8", value=True)
                        fp8_llm = gr.Checkbox(label="Use FP8 for LLM", value=False)
                        add_metadata = gr.Checkbox(label="Add Metadata", value=True)
                    
                    with gr.Row():
                        split_attn = gr.Checkbox(label="Split Attention", value=False)
                        split_uncond = gr.Checkbox(label="Split Unconditional", value=False)
                        img_in_txt_in_offloading = gr.Checkbox(label="Offload Embedders", value=False)
                    
                    with gr.Row():
                        vae_chunk_size = gr.Slider(
                            minimum=8, maximum=64, value=32, step=8,
                            label="VAE Chunk Size"
                        )
                        vae_spatial_tile_sample_min_size = gr.Slider(
                            minimum=64, maximum=256, value=128, step=16,
                            label="VAE Tile Sample Min Size"
                        )
        
        # Connect events
        # Input image handling
        input_image.change(
            fn=update_dimensions,
            inputs=[input_image],
            outputs=[original_dims, width, height]
        )
        
        # Scale slider
        scale_slider.change(
            fn=update_from_scale,
            inputs=[scale_slider, original_dims],
            outputs=[width, height]
        )
        
        # Width/height calculation buttons
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
        
        # Flow shift recommendation
        recommend_flow_btn.click(
            fn=recommend_flow_shift,
            inputs=[width, height],
            outputs=[flow_shift]
        )
        
        # Random folder checkbox
        use_random_folder.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_random_folder],
            outputs=[input_folder]
        )
        
        # Generation button
        generate_btn.click(
            fn=generate_from_ui,
            inputs=[
                prompt, negative_prompt, input_image, width, height, video_length,
                fps, infer_steps, flow_shift, guidance_scale, embedded_cfg_scale, seed,
                use_random_folder, input_folder, batch_size,
                dit_path, vae_path, te1_path, te2_path, save_path, i2v_resolution,
                attn_mode, block_swap, split_attn, split_uncond, fp8, fp8_llm,
                add_metadata, img_in_txt_in_offloading, vae_chunk_size, vae_spatial_tile_sample_min_size
            ],
            outputs=[output_gallery, batch_progress, progress_text]
        )
        
        # Stop button
        stop_btn.click(fn=lambda: stop_event.set(), queue=False)
        
        # CSS for small buttons
        gr.Markdown("""
        <style>
        .small-button {
            min-width: 40px !important;
            max-width: 40px !important;
        }
        </style>
        """)
        
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.queue().launch()