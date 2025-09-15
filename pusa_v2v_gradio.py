import gradio as gr
import subprocess
import threading
import os
import sys
import time
import re
from pathlib import Path
from typing import List, Tuple, Optional, Generator
import shutil
import tempfile
from datetime import datetime

# Global stop event for interrupting generation
stop_event = threading.Event()

# CSS styling similar to h1111.py
custom_css = """
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
    }
    .stop-btn {
        background: linear-gradient(to bottom right, #e74c3c, #c0392b) !important;
        color: white !important;
        border: none !important;
    }
    .stop-btn:hover {
        background: linear-gradient(to bottom right, #c0392b, #a93226) !important;
    }
"""

def get_lora_files(lora_folder: str = "model_zoo/PusaV1/Wan2.2-Models") -> List[str]:
    """Get available LoRA files from the specified folder"""
    if not os.path.exists(lora_folder):
        return []

    lora_files = []
    for file in os.listdir(lora_folder):
        if file.endswith('.safetensors'):
            lora_files.append(os.path.join(lora_folder, file))

    return sorted(lora_files)

def refresh_lora_list(lora_folder: str):
    """Refresh the LoRA dropdown lists"""
    lora_files = ["None"] + get_lora_files(lora_folder)
    # Return updates for all LoRA dropdowns
    return [gr.update(choices=lora_files) for _ in range(8)]

def parse_lora_inputs(lora_paths: List[str], lora_alphas: List[float]) -> Tuple[str, str]:
    """Parse multiple LoRA paths and alphas into comma-separated strings"""
    # Filter out None values and empty strings
    valid_paths = []
    valid_alphas = []

    for path, alpha in zip(lora_paths, lora_alphas):
        # Check if path is valid (not None, not "None", and not empty)
        if path and isinstance(path, str) and path != "None" and path.strip():
            valid_paths.append(path.strip())
            valid_alphas.append(str(alpha))

    if not valid_paths:
        # Return empty strings to indicate no LoRAs selected
        return "", ""

    return ",".join(valid_paths), ",".join(valid_alphas)

def run_generation(
    video_path: str,
    prompt: str,
    negative_prompt: str,
    # Conditioning parameters
    use_extend_from_end: bool,
    extend_from_end: int,
    cond_position: str,
    noise_multipliers: str,
    # Generation parameters
    width: int,
    height: int,
    fps: int,
    num_inference_steps: int,
    cfg_scale: float,
    # Model directories
    dit_high_noise_path: str,
    dit_low_noise_path: str,
    base_dir: str,
    # High LoRAs
    high_lora_1: str, high_alpha_1: float,
    high_lora_2: str, high_alpha_2: float,
    high_lora_3: str, high_alpha_3: float,
    high_lora_4: str, high_alpha_4: float,
    # Low LoRAs
    low_lora_1: str, low_alpha_1: float,
    low_lora_2: str, low_alpha_2: float,
    low_lora_3: str, low_alpha_3: float,
    low_lora_4: str, low_alpha_4: float,
    # Other options
    switch_boundary: float,
    blocks_to_swap: int,
    concatenate: bool,
    output_dir: str,
    progress=gr.Progress()
) -> Generator[Tuple[str, Optional[str]], None, None]:
    """Run the V2V generation script"""

    # Reset stop event
    stop_event.clear()

    # Validate inputs
    if not video_path:
        yield "Error: No video file provided", None
        return

    if not os.path.exists(video_path):
        yield f"Error: Video file not found: {video_path}", None
        return

    # Parse LoRA inputs
    high_lora_paths = [high_lora_1, high_lora_2, high_lora_3, high_lora_4]
    high_lora_alphas = [high_alpha_1, high_alpha_2, high_alpha_3, high_alpha_4]
    low_lora_paths = [low_lora_1, low_lora_2, low_lora_3, low_lora_4]
    low_lora_alphas = [low_alpha_1, low_alpha_2, low_alpha_3, low_alpha_4]

    high_paths_str, high_alphas_str = parse_lora_inputs(high_lora_paths, high_lora_alphas)
    low_paths_str, low_alphas_str = parse_lora_inputs(low_lora_paths, low_lora_alphas)

    # Debug print
    print(f"[DEBUG] High LoRA paths: {high_paths_str}")
    print(f"[DEBUG] High LoRA alphas: {high_alphas_str}")
    print(f"[DEBUG] Low LoRA paths: {low_paths_str}")
    print(f"[DEBUG] Low LoRA alphas: {low_alphas_str}")

    # Build command
    cmd = [
        sys.executable,
        "pusa_v2v.py",  # Use the updated script name
        "--video_path", video_path,
        "--prompt", prompt,
        "--negative_prompt", negative_prompt,
        "--noise_multipliers", noise_multipliers,
        "--num_inference_steps", str(num_inference_steps),
        "--dit_high_noise_path", dit_high_noise_path, # Updated argument
        "--dit_low_noise_path", dit_low_noise_path,   # Updated argument
        "--base_dir", base_dir,
        "--high_lora_path", high_paths_str if high_paths_str else "''",
        "--high_lora_alpha", high_alphas_str if high_alphas_str else "''",
        "--low_lora_path", low_paths_str if low_paths_str else "''",
        "--low_lora_alpha", low_alphas_str if low_alphas_str else "''",
        "--switch_DiT_boundary", str(switch_boundary),
        "--cfg_scale", str(cfg_scale),
        "--width", str(width),
        "--height", str(height),
        "--fps", str(fps),
        "--output_dir", output_dir,
        "--blocks_to_swap", str(blocks_to_swap) # Added argument
    ]

    # Add conditioning mode
    if use_extend_from_end:
        cmd.extend(["--extend_from_end", str(extend_from_end)])
        if concatenate:
            cmd.append("--concatenate")
    else:
        if cond_position:
            cmd.extend(["--cond_position", cond_position])
        else:
            yield "Error: Either extend_from_end or cond_position must be specified", None
            return

    # Print command for debugging
    print("\n" + "="*80)
    print("Executing command:")
    print(" ".join(cmd))
    print("="*80 + "\n")

    # Run the command
    try:
        yield "Starting generation...", None
        progress(0, desc="Initializing...")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        output_lines = []
        last_video_path = None

        # Create threads to read stdout and stderr
        import queue
        output_queue = queue.Queue()

        def read_output(pipe, pipe_name):
            if pipe is None:
                return
            for line in pipe:
                output_queue.put((pipe_name, line))
            pipe.close()

        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, "stdout"))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, "stderr"))

        stdout_thread.start()
        stderr_thread.start()

        # Process output from both streams
        while True:
            # Check if process is still running
            poll_status = process.poll()

            # Read from queue with timeout
            try:
                pipe_name, line = output_queue.get(timeout=0.1)
            except queue.Empty:
                # Check if process finished and threads are done
                if poll_status is not None and not stdout_thread.is_alive() and not stderr_thread.is_alive():
                    break
                continue

            if stop_event.is_set():
                process.terminate()
                print("\n[STOPPED] Generation stopped by user")
                yield "Generation stopped by user", last_video_path
                return

            # Print to console with proper prefix
            if pipe_name == "stderr":
                print(f"[ERROR] {line.strip()}")
            else:
                print(f"[OUTPUT] {line.strip()}")

            output_lines.append(line.strip())

            # Parse progress
            if "Loading models" in line:
                progress(0.1, desc="Loading models...")
            elif "Enabling swap" in line:
                progress(0.15, desc=line.strip())
            elif "Generating new video frames" in line:
                progress(0.2, desc="Generating frames...")
            elif "Decoding final latent..." in line:
                progress(0.9, desc="Decoding and saving video...")
            elif "Saving video to" in line:
                # Extract output path
                match = re.search(r"Saving video to (.+\.mp4)", line)
                if match:
                    last_video_path = match.group(1)
            elif "Video saved successfully" in line:
                progress(1.0, desc="Complete!")

            # Update status
            status = "\n".join(output_lines[-20:])  # Show last 20 lines
            yield status, last_video_path

        # Wait for threads to complete
        stdout_thread.join()
        stderr_thread.join()

        process.wait()

        print(f"\n[COMPLETED] Process exited with code: {process.returncode}")

        if process.returncode == 0:
            yield "Generation completed successfully!", last_video_path
        else:
            error_msg = f"Generation failed with code {process.returncode}\n\nLast output:\n" + "\n".join(output_lines[-30:])
            print(f"[FAILED] {error_msg}")
            yield error_msg, last_video_path

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{type(e).__name__}"
        print(f"\n[EXCEPTION] {error_msg}")
        import traceback
        traceback.print_exc()
        yield error_msg, None

def stop_generation():
    """Stop the current generation"""
    stop_event.set()
    return "Stopping generation..."

def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(css=custom_css, title="Pusa V2V Interface") as interface:
        gr.Markdown("# 🎬 Pusa Video-to-Video Generation")
        gr.Markdown("Generate new videos using the Pusa V2V pipeline with multiple LoRA support")

        with gr.Row():
            with gr.Column(scale=3):
                # Input section
                with gr.Group():
                    gr.Markdown("### 📹 Input Video")
                    video_input = gr.Video(
                        label="Input Video",
                        format="mp4"
                    )

                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe what you want to generate...",
                        lines=3,
                        value="A fast action video featuring a cute tabby cat wearing a pink hat, eating a blueberry and cucumber sandwich."
                    )

                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        lines=2,
                        value="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality"
                    )

                # Conditioning section
                with gr.Group():
                    gr.Markdown("### 🎯 Conditioning Settings")

                    with gr.Row():
                        use_extend_from_end = gr.Checkbox(
                            label="Use Video Extension Mode",
                            value=True,
                            info="Use last N frames from input video to condition the start of new video"
                        )

                    with gr.Row():
                        extend_from_end = gr.Number(
                            label="Frames from End",
                            value=6,
                            minimum=1,
                            maximum=20,
                            step=1,
                            visible=True,
                            info="Number of frames from the end to use for conditioning"
                        )

                        cond_position = gr.Textbox(
                            label="Conditioning Positions",
                            placeholder="0,10,20,30",
                            visible=False,
                            info="Comma-separated frame indices for conditioning (alternative to extension mode)"
                        )

                    noise_multipliers = gr.Textbox(
                        label="Noise Multipliers",
                        value="0.1,0.1,0.1,0.1,0.1,0.1",
                        info="Comma-separated noise values (one per conditioning frame)"
                    )

                    concatenate = gr.Checkbox(
                        label="Concatenate with Original",
                        value=True,
                        visible=True,
                        info="Automatically join original video with generated video (extension mode only)"
                    )

                # Generation parameters
                with gr.Group():
                    gr.Markdown("### ⚙️ Generation Parameters")

                    with gr.Row():
                        width = gr.Number(
                            label="Width",
                            value=832,
                            minimum=128,
                            maximum=2048,
                            step=8
                        )
                        height = gr.Number(
                            label="Height",
                            value=480,
                            minimum=128,
                            maximum=2048,
                            step=8
                        )
                        fps = gr.Number(
                            label="FPS",
                            value=24,
                            minimum=1,
                            maximum=60,
                            step=1
                        )

                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=10,
                            maximum=100,
                            value=30,
                            step=1
                        )
                        cfg_scale = gr.Slider(
                            label="CFG Scale",
                            minimum=1.0,
                            maximum=20.0,
                            value=3.0,
                            step=0.1
                        )

                    with gr.Row():
                        switch_boundary = gr.Slider(
                            label="DiT Switch Boundary",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.875,
                            step=0.001,
                            info="Switch from high to low noise model at this threshold"
                        )
                        blocks_to_swap = gr.Slider(
                            label="Blocks to Swap",
                            minimum=0,
                            maximum=40,
                            value=8,
                            step=1,
                            info="Number of DiT blocks to swap to CPU for VRAM management (0 to disable)"
                        )

            with gr.Column(scale=2):
                # LoRA configuration
                with gr.Group():
                    gr.Markdown("### 🎨 LoRA Configuration")

                    with gr.Row():
                        lora_folder = gr.Textbox(
                            label="LoRA Folder",
                            value="lora",
                        )
                        refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")

                    with gr.Accordion("High Noise LoRAs", open=True):
                        high_loras = []
                        high_alphas = []
                        for i in range(4):
                            with gr.Row():
                                lora = gr.Dropdown(
                                    label=f"LoRA {i+1}",
                                    choices=["None"] + get_lora_files(),
                                    value="None" if i > 0 else "model_zoo/PusaV1/Wan2.2-Models/high_noise_pusa.safetensors"
                                )
                                alpha = gr.Slider(
                                    label=f"Alpha {i+1}",
                                    minimum=0.0,
                                    maximum=3.0,
                                    value=1.5 if i == 0 else 1.0,
                                    step=0.1
                                )
                                high_loras.append(lora)
                                high_alphas.append(alpha)

                    with gr.Accordion("Low Noise LoRAs", open=True):
                        low_loras = []
                        low_alphas = []
                        for i in range(4):
                            with gr.Row():
                                lora = gr.Dropdown(
                                    label=f"LoRA {i+1}",
                                    choices=["None"] + get_lora_files(),
                                    value="None" if i > 0 else "model_zoo/PusaV1/Wan2.2-Models/low_noise_pusa.safetensors"
                                )
                                alpha = gr.Slider(
                                    label=f"Alpha {i+1}",
                                    minimum=0.0,
                                    maximum=3.0,
                                    value=1.4 if i == 0 else 1.0,
                                    step=0.1
                                )
                                low_loras.append(lora)
                                low_alphas.append(alpha)

                # Model directories
                with gr.Group():
                    gr.Markdown("### 📁 Model Directories")

                    dit_high_noise_path = gr.Textbox(
                        label="High Noise DiT Path (File or Dir)",
                        value="wan/wan22_t2v_14B_high_noise_bf16.safetensors"
                    )

                    dit_low_noise_path = gr.Textbox(
                        label="Low Noise DiT Path (File or Dir)",
                        value="wan/wan22_t2v_14B_low_noise_bf16.safetensors"
                    )

                    base_dir = gr.Textbox(
                        label="Base Model Directory (T5, VAE)",
                        value="wan"
                    )

                    output_dir = gr.Textbox(
                        label="Output Directory",
                        value="./outputs"
                    )

        # Generation controls
        with gr.Row():
            generate_btn = gr.Button(
                "🎬 Generate Video",
                elem_classes="green-btn",
                scale=2
            )
            stop_btn = gr.Button(
                "⏹️ Stop",
                elem_classes="stop-btn",
                scale=1
            )

        # Output section
        with gr.Row():
            with gr.Column():
                status_output = gr.Textbox(
                    label="Status",
                    lines=10,
                    max_lines=15,
                    interactive=False
                )
            with gr.Column():
                video_output = gr.Video(
                    label="Generated Video",
                    format="mp4"
                )

        # Event handlers
        def toggle_conditioning_mode(use_extend):
            return (
                gr.update(visible=use_extend),  # extend_from_end
                gr.update(visible=not use_extend),  # cond_position
                gr.update(visible=use_extend)  # concatenate
            )

        use_extend_from_end.change(
            toggle_conditioning_mode,
            inputs=[use_extend_from_end],
            outputs=[extend_from_end, cond_position, concatenate]
        )

        refresh_btn.click(
            refresh_lora_list,
            inputs=[lora_folder],
            outputs=high_loras + low_loras
        )

        # FIX: Correctly interleave LoRA and Alpha components to match the function signature
        interleaved_high_loras_alphas = []
        for lora, alpha in zip(high_loras, high_alphas):
            interleaved_high_loras_alphas.extend([lora, alpha])

        interleaved_low_loras_alphas = []
        for lora, alpha in zip(low_loras, low_alphas):
            interleaved_low_loras_alphas.extend([lora, alpha])

        # Prepare all inputs for generation in the correct order
        generation_inputs = [
            video_input, prompt, negative_prompt,
            use_extend_from_end, extend_from_end, cond_position, noise_multipliers,
            width, height, fps, num_inference_steps, cfg_scale,
            dit_high_noise_path, dit_low_noise_path, base_dir
        ] + interleaved_high_loras_alphas + interleaved_low_loras_alphas + [
            switch_boundary, blocks_to_swap, concatenate, output_dir
        ]

        generate_btn.click(
            run_generation,
            inputs=generation_inputs,
            outputs=[status_output, video_output]
        )

        stop_btn.click(
            stop_generation,
            outputs=[status_output]
        )


    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7863,
        show_error=True
    )