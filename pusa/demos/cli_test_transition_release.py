#! /usr/bin/env python
import json
import os
import time

import click
import numpy as np
import torch

from genmo.lib.progress import progress_bar
from genmo.lib.utils import save_video
from genmo.mochi_preview.pipelines_v2v_release import (
    DecoderModelFactory,
    EncoderModelFactory,
    DitModelFactory,
    MochiMultiGPUPipeline,
    MochiSingleGPUPipeline,
    T5ModelFactory,
    linear_quadratic_schedule,
)


import torch
from torch.utils.data import Dataset, DataLoader
import random
import string
from lightning.pytorch  import LightningDataModule
from genmo.mochi_preview.vae.models import Encoder, add_fourier_features
from genmo.mochi_preview.vae.latent_dist import LatentDistribution
import torchvision
from einops import rearrange
from safetensors.torch import load_file
from genmo.mochi_preview.pipelines import DecoderModelFactory, decode_latents_tiled_spatial, decode_latents, decode_latents_tiled_full
from genmo.mochi_preview.vae.vae_stats import dit_latents_to_vae_latents


pipeline = None
model_dir_path = None
num_gpus = torch.cuda.device_count()
cpu_offload = False
dit_path = None

def configure_model(model_dir_path_, dit_path_, cpu_offload_):
    global model_dir_path, dit_path, cpu_offload
    model_dir_path = model_dir_path_
    dit_path = dit_path_
    cpu_offload = cpu_offload_


def load_model():
    global num_gpus, pipeline, model_dir_path, dit_path
    if pipeline is None:
        MOCHI_DIR = model_dir_path
        print(f"Launching with {num_gpus} GPUs. If you want to force single GPU mode use CUDA_VISIBLE_DEVICES=0.")
        klass = MochiSingleGPUPipeline if num_gpus == 1 else MochiMultiGPUPipeline
        kwargs = dict(
            text_encoder_factory=T5ModelFactory(),
            dit_factory=DitModelFactory(
                model_path=dit_path,
                model_dtype="bf16"
            ),
            decoder_factory=DecoderModelFactory(
                model_path=f"{MOCHI_DIR}/decoder.safetensors",
            ),
            encoder_factory=EncoderModelFactory(
                model_path=f"{MOCHI_DIR}/encoder.safetensors",
            ),
        )
        if num_gpus > 1:
            assert not cpu_offload, "CPU offload not supported in multi-GPU mode"
            kwargs["world_size"] = num_gpus
        else:
            kwargs["cpu_offload"] = cpu_offload
            # kwargs["decode_type"] = "tiled_full"
            kwargs["decode_type"] = "tiled_spatial"
        pipeline = klass(**kwargs)


def generate_video(
    prompt,
    negative_prompt,
    width,
    height,
    num_frames,
    seed,
    cfg_scale,
    num_inference_steps,
    data_path,
    input_image=None,
):
    load_model()
    global dit_path

    # sigma_schedule should be a list of floats of length (num_inference_steps + 1),
    # such that sigma_schedule[0] == 1.0 and sigma_schedule[-1] == 0.0 and monotonically decreasing.
    sigma_schedule = linear_quadratic_schedule(num_inference_steps, 0.025)

    # cfg_schedule should be a list of floats of length num_inference_steps.
    # For simplicity, we just use the same cfg scale at all timesteps,
    # but more optimal schedules may use varying cfg, e.g:
    # [5.0] * (num_inference_steps // 2) + [4.5] * (num_inference_steps // 2)
    cfg_schedule = [cfg_scale] * num_inference_steps

    args = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "sigma_schedule": sigma_schedule,
        "cfg_schedule": cfg_schedule,
        "num_inference_steps": num_inference_steps,
        # We *need* flash attention to batch cfg
        # and it's only worth doing in a high-memory regime (assume multiple GPUs)
        "batch_cfg": False,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "data_path": data_path,
    }
    
    # Handle different input types
    if input_image is not None:
        # if "tensor" in input_image:
        # Check if this is an image tensor (for image conditioning) or latent tensor
        # if len(input_image["tensor"].shape) == 4:  # [B, C, H, W] - image tensor
        # This is an image tensor, prepare it for conditioning
        # cond_position = input_image.get("cond_position", 0)
        args["condition_image"] = input_image["tensor"]
        args["condition_frame_idx"] = input_image["cond_position"]
        args["noise_multiplier"] = input_image["noise_multiplier"]
        # else:  # Latent tensor
        #     args["input_image"] = input_image["tensor"]

    # print(args)
    with progress_bar(type="tqdm"):
        final_frames = pipeline(**args)

        final_frames = final_frames[0]

        assert isinstance(final_frames, np.ndarray)
        assert final_frames.dtype == np.float32

        # Create a results directory based on model name and timestamp
        model_name = os.path.basename(dit_path.split('/')[-2])
        checkpoint_name = dit_path.split('/')[-1].split('train_loss')[0]
        # Use datetime format for timestamp_dir
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        cond_position = input_image["cond_position"]
        results_base_dir = "./video_test_demos_results"
        results_dir = os.path.join(results_base_dir, f"{model_name}_{checkpoint_name}_dawn_{cond_position}pos_{num_inference_steps}steps_0sigma")
        os.makedirs(results_dir, exist_ok=True)
        
        # Extract filename from input_image if available
        filename_prefix = ""
        if isinstance(input_image, dict) and "filename" in input_image:
            filename_prefix = f"{os.path.basename(input_image['filename']).split('.')[0]}_"
            
        output_path = os.path.join(
            results_dir,
            f"{filename_prefix}{timestamp_str}.mp4"
        )
        save_video(final_frames, output_path)
        json_path = os.path.splitext(output_path)[0] + ".json"
        
        # Save args to JSON but remove input_image tensor and convert non-serializable objects
        json_args = args.copy()
        
        # Handle input_image for JSON serialization
        if "input_image" in json_args:
            json_args["input_image"] = None
        
        # Handle condition_image for JSON serialization
        if "condition_image" in json_args:
            json_args["condition_image"] = "Image tensor (removed for JSON)"
            
        if isinstance(input_image, dict):
            json_args["input_filename"] = input_image.get("filename", None)
            if "cond_position" in input_image:
                json_args["condition_frame_idx"] = input_image["cond_position"]
        
        # Convert sigma_schedule and cfg_schedule from tensors to lists if needed
        if isinstance(json_args["sigma_schedule"], torch.Tensor):
            json_args["sigma_schedule"] = json_args["sigma_schedule"].tolist()
        if isinstance(json_args["cfg_schedule"], torch.Tensor):
            json_args["cfg_schedule"] = json_args["cfg_schedule"].tolist()
            
        # Handle prompt if it's a tensor or other non-serializable object
        if not isinstance(json_args["prompt"], (str, type(None))):
            if hasattr(json_args["prompt"], "tolist"):
                json_args["prompt"] = "Tensor prompt (converted to string for JSON)"
            else:
                json_args["prompt"] = str(json_args["prompt"])
                
        # Handle negative_prompt if it's a tensor
        if not isinstance(json_args["negative_prompt"], (str, type(None))):
            if hasattr(json_args["negative_prompt"], "tolist"):
                json_args["negative_prompt"] = "Tensor negative prompt (converted to string for JSON)"
            else:
                json_args["negative_prompt"] = str(json_args["negative_prompt"])
                
        json.dump(json_args, open(json_path, "w"), indent=4)

        return output_path

from textwrap import dedent


@click.command()
@click.option("--prompt", default="A man is playing the basketball", help="Prompt for video generation.")
@click.option("--negative_prompt", default="", help="Negative prompt for video generation.")
@click.option("--width", default=848, type=int, help="Width of the video.")
@click.option("--height", default=480, type=int, help="Height of the video.")
@click.option("--num_frames", default=163, type=int, help="Number of frames.")
@click.option("--seed", default=1710977262, type=int, help="Random seed.")
@click.option("--cfg_scale", default=4.5, type=float, help="CFG Scale.")
@click.option("--num_steps", default=64, type=int, help="Number of inference steps.")
@click.option("--model_dir", required=True, help="Path to the model directory.")
@click.option("--dit_path", required=True, help="Path to the dit model directory.")
@click.option("--cpu_offload", is_flag=True, help="Whether to offload model to CPU")
@click.option("--data_path", required=True, default="/home/dyvm6xra/dyvm6xrauser02/data/vidgen1m", help="Path to the data directory.")
@click.option("--video_start_dir", default=None, help="Path to the start conditioning video.")
@click.option("--video_end_dir", default=None, help="Path to the end conditioning video.")
@click.option("--prompt_dir", default=None, help="Path to directory containing prompt text files.")
@click.option("--cond_position_start", default="[0,1,2]", type=str, help="Frame positions list to place the start conditioning video, position from 0 to 27.")
@click.option("--cond_position_end", default="[-3,-2,-1]", type=str, help="Frame positions list to place the end conditioning video, position from 0 to 27.")
@click.option("--noise_multiplier", default="[0.1,0.3,0.3,0.3,0.3,0.3]", type=str, help="Noise multiplier for noise on the conditioning positions, length must match len(cond_position_start) + len(cond_position_end).")

def generate_cli(
    prompt, negative_prompt, width, height, num_frames, seed, cfg_scale, num_steps, model_dir, 
    dit_path, cpu_offload, data_path, video_start_dir, video_end_dir, prompt_dir, cond_position_start, cond_position_end, noise_multiplier
):
    configure_model(model_dir, dit_path, cpu_offload)
    config = dict(
            prune_bottlenecks=[False, False, False, False, False],
            has_attentions=[False, True, True, True, True],
            affine=True,
            bias=True,
            input_is_conv_1x1=True,
            padding_mode="replicate",
        )
    # Create VAE encoder
    encoder = Encoder(
        in_channels=15,
        base_channels=64,
        channel_multipliers=[1, 2, 4, 6],
        num_res_blocks=[3, 3, 4, 6, 3],
        latent_dim=12,
        temporal_reductions=[1, 2, 3],
        spatial_reductions=[2, 2, 2],
        **config,
    )
    device = torch.device("cuda:0")
    encoder = encoder.to(device, memory_format=torch.channels_last_3d)
    encoder.load_state_dict(load_file(f"{model_dir}/encoder.safetensors"))
    encoder.eval()

    # Import required libraries
    import cv2
    import torchvision.transforms as transforms
    from PIL import Image
    
    def process_video(video_path, width, height):
        """Process a video file and return a tensor of normalized frames"""
        if not os.path.isfile(video_path):
            click.echo(f"Video file not found: {video_path}")
            return None
            
        click.echo(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Read frames from video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
                
        cap.release()
        
        if not frames:
            click.echo(f"Error: Could not read frames from video {video_path}")
            return None
            
        print(f"Loaded {len(frames)} frames from video {os.path.basename(video_path)}")
        
        # Process frames - crop and resize
        processed_frames = []
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        target_ratio = width / height
        
        for frame in frames:
            # Convert to PIL for easier processing
            pil_frame = Image.fromarray(frame)
            
            # Calculate crop dimensions to maintain aspect ratio
            current_ratio = pil_frame.width / pil_frame.height
            
            if current_ratio > target_ratio:
                # Frame is wider than target ratio - crop width
                new_width = int(pil_frame.height * target_ratio)
                x1 = (pil_frame.width - new_width) // 2
                pil_frame = pil_frame.crop((x1, 0, x1 + new_width, pil_frame.height))
            else:
                # Frame is taller than target ratio - crop height
                new_height = int(pil_frame.width / target_ratio)
                y1 = (pil_frame.height - new_height) // 2
                pil_frame = pil_frame.crop((0, y1, pil_frame.width, y1 + new_height))
            
            # Resize the cropped frame
            pil_frame = pil_frame.resize((width, height), Image.LANCZOS)
            
            # Convert to tensor
            frame_tensor = transform(pil_frame)
            processed_frames.append(frame_tensor)
        
        # Stack frames into a single tensor [T, C, H, W]
        video_tensor = torch.stack(processed_frames)
        # Normalize to [-1, 1]
        video_tensor = video_tensor * 2 - 1
        # Add batch dimension [1, T, C, H, W]
        video_tensor = video_tensor.unsqueeze(0)
        
        return video_tensor, os.path.basename(video_path)

    # Process start and end videos if provided
    start_tensor = None
    end_tensor = None
    filename_parts = []
    
    if video_start_dir and os.path.isfile(video_start_dir):
        start_result = process_video(video_start_dir, width, height)
        if start_result:
            start_tensor, start_filename = start_result
            filename_parts.append(os.path.splitext(start_filename)[0])
            
    
    if video_end_dir and os.path.isfile(video_end_dir):
        end_result = process_video(video_end_dir, width, height)
        if end_result:
            end_tensor, end_filename = end_result
            filename_parts.append(os.path.splitext(end_filename)[0])
    
    # Concatenate tensors if both are available
    if start_tensor is not None and end_tensor is not None:
        # Ensure both tensors have the same number of frames
        min_frames = min(start_tensor.shape[1], end_tensor.shape[1], 82)
        start_tensor = start_tensor[:, :min_frames]
        print(f"Start video tensor shape: {start_tensor.shape}")
        # import ipdb;ipdb.set_trace()
        end_tensor = end_tensor[:, :min_frames-1]
        print(f"End video tensor shape: {end_tensor.shape}")

        # Rearrange tensors to [B, C, T, H, W] format for temporal concatenation
        start_tensor_rearranged = start_tensor.permute(0, 2, 1, 3, 4)  # [1, 3, 82, 480, 848]
        end_tensor_rearranged = end_tensor.permute(0, 2, 1, 3, 4)      # [1, 3, 81, 480, 848]
        
        # Concatenate along dimension 2 (temporal dimension in the rearranged format)
        combined_tensor = torch.cat([start_tensor_rearranged, end_tensor_rearranged], dim=2)
        print(f"Combined tensor shape after temporal concatenation: {combined_tensor.shape}")
        
        # Add Fourier features and encode to latent
        combined_tensor = add_fourier_features(combined_tensor.to(device))
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t0 = time.time()
                encoder = encoder.to(device)
                ldist = encoder(combined_tensor)
                image_tensor = ldist.sample()
                print(f"Encoding took {time.time() - t0:.2f} seconds")
        
        # Move encoder to CPU to free GPU memory
        torch.cuda.empty_cache()
        encoder = encoder.to("cpu")
        del ldist

        # Create combined filename
        combined_filename = "_and_".join(filename_parts)
        
        # Parse string representations of position lists to actual lists
        cond_position_start_list = eval(cond_position_start)
        cond_position_end_list = eval(cond_position_end)
        cond_position = cond_position_start_list + cond_position_end_list
        
        # Package input for generate_video
        input_image = {
            "tensor": image_tensor,
            "filename": combined_filename,
            "cond_position": cond_position,
            "noise_multiplier": noise_multiplier
        }
        
        with torch.inference_mode():
            output = generate_video(
                prompt,
                negative_prompt,
                width,
                height,
                num_frames,
                seed,
                cfg_scale,
                num_steps,
                data_path,
                input_image,
            )
            click.echo(f"Video generated at: {output}")
        return

if __name__ == "__main__":
    generate_cli()
