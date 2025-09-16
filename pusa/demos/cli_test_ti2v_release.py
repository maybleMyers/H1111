#! /usr/bin/env python
import json
import os
import time

import click
import numpy as np
import torch

from genmo.lib.progress import progress_bar
from genmo.lib.utils import save_video
from genmo.mochi_preview.pipelines_ti2v_release import (
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
    noise_multiplier=0,
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
        "noise_multiplier": noise_multiplier,
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
        
        if input_image is not None:
            cond_position = input_image["cond_position"]
        else:
            cond_position = ""
        results_base_dir = "/home/dyvm6xra/dyvm6xrauser02/raphael/mochi-1-preview/models/video_test_demos_results"
        results_dir = os.path.join(results_base_dir, f"{model_name}_{checkpoint_name}_github_user_demo_{cond_position}pos_{num_inference_steps}steps_crop_{noise_multiplier}sigma")
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
@click.option("--image_dir", default=None, help="Path to image or directory of images for conditioning.")
@click.option("--prompt_dir", default=None, help="Path to directory containing prompt text files.")
@click.option("--cond_position", default=0, type=int, help="Frame position to place the conditioning image, from 0 to 27.")
@click.option("--noise_multiplier", default=0, type=float, help="Noise multiplier for noise on the conditioning image.")

def generate_cli(
    prompt, negative_prompt, width, height, num_frames, seed, cfg_scale, num_steps, model_dir, 
    dit_path, cpu_offload, data_path, image_dir, prompt_dir, cond_position, noise_multiplier
):
    configure_model(model_dir, dit_path, cpu_offload)

    # Case 1: Text to video generation
    if image_dir is None:
        
        # Check if we have a prompt directory to process multiple prompts
        if prompt_dir is not None and os.path.isdir(prompt_dir):
            prompt_files = [f for f in os.listdir(prompt_dir) if f.lower().endswith('.txt')]

            prompt_files = prompt_files[210:] # TODO: Remove this

            if not prompt_files:
                click.echo(f"No prompt files found in {prompt_dir}")
                return
                
            click.echo(f"Found {len(prompt_files)} prompt files to process")
            
            for i, prompt_file in enumerate(prompt_files):
                file_path = os.path.join(prompt_dir, prompt_file)
                click.echo(f"Processing prompt file {i+1}/{len(prompt_files)}: {file_path}")
                
                # Read prompt from file
                with open(file_path, 'r') as f:
                    file_prompt = f.read().strip()
                
                click.echo(f"Using prompt: {file_prompt}")
                
                with torch.inference_mode():
                    output = generate_video(
                        file_prompt,
                        negative_prompt,
                        width,
                        height,
                        num_frames,
                        seed,
                        cfg_scale,
                        num_steps,
                        data_path,
                        input_image=None,
                        noise_multiplier=noise_multiplier,
                    )
                    click.echo(f"Video generated at: {output}")
        else:
            # Process single prompt as before
            click.echo("Running text-to-video generation with provided prompt")
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
                    input_image=None,
                )
                click.echo(f"Video generated at: {output}")
        return

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

    # Case 2: Image-to-video, image_dir is a single file
    if image_dir is not None and os.path.isfile(image_dir) and image_dir.lower().endswith(('.jpg', '.jpeg', '.png')):
        click.echo(f"Processing single image: {image_dir}")
        
        # Load the image
        from PIL import Image
        import torchvision.transforms as transforms
        
        image = Image.open(image_dir)
        
        # Crop and resize the image to the target dimensions rather than directly resize
        # Calculate crop dimensions to maintain aspect ratio
        target_ratio = width / height
        current_ratio = image.width / image.height
        
        if current_ratio > target_ratio:
            # Image is wider than target ratio - crop width
            new_width = int(image.height * target_ratio)
            x1 = (image.width - new_width) // 2
            image = image.crop((x1, 0, x1 + new_width, image.height))
        else:
            # Image is taller than target ratio - crop height
            new_height = int(image.width / target_ratio)
            y1 = (image.height - new_height) // 2
            image = image.crop((0, y1, image.width, y1 + new_height))
        
        # Now resize the cropped image
        transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ])
        
        image_tensor = (transform(image)* 2 - 1).unsqueeze(1).unsqueeze(0)
        print("image_tensor.shape", image_tensor.shape)
        image_tensor = add_fourier_features(image_tensor.to(device))
        # Encode image to latent
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t0 = time.time()
                encoder = encoder.to(device)
                ldist = encoder(image_tensor)
                image_tensor = ldist.sample()
        
        torch.cuda.empty_cache()
        encoder = encoder.to("cpu")
        del ldist
        
        # Package input for generate_video
        input_image = {
            "tensor": image_tensor,
            "filename": os.path.basename(image_dir),
            "cond_position": cond_position
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
                noise_multiplier=noise_multiplier,
            )
            click.echo(f"Video generated at: {output}")
        return
    

    # Case 3: image_dir is a directory of images
    if image_dir is not None and os.path.isdir(image_dir):
        # Get all image files in the directory
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            click.echo(f"No image files found in {image_dir}")
            return
        
        click.echo(f"Found {len(image_files)} image files to process")
        
        from PIL import Image
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        with torch.inference_mode():
            for i, image_file in enumerate(image_files):
                file_path = os.path.join(image_dir, image_file)
                click.echo(f"Processing file {i+1}/{len(image_files)}: {file_path}")
                
                # Load image
                image = Image.open(file_path)
                
                # Calculate crop dimensions to maintain aspect ratio
                target_ratio = width / height
                current_ratio = image.width / image.height
                
                if current_ratio > target_ratio:
                    # Image is wider than target ratio - crop width
                    new_width = int(image.height * target_ratio)
                    x1 = (image.width - new_width) // 2
                    image = image.crop((x1, 0, x1 + new_width, image.height))
                else:
                    # Image is taller than target ratio - crop height
                    new_height = int(image.width / target_ratio)
                    y1 = (image.height - new_height) // 2
                    image = image.crop((0, y1, image.width, y1 + new_height))
                
                # Now resize the cropped image
                image = image.resize((width, height))
                
                image_tensor = (transform(image)* 2 - 1).unsqueeze(1).unsqueeze(0)
                print("image_tensor.shape", image_tensor.shape)
                image_tensor = add_fourier_features(image_tensor.to(device))
                # Encode image to latent
                with torch.inference_mode():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        t0 = time.time()
                        encoder = encoder.to(device)
                        ldist = encoder(image_tensor)
                        image_tensor = ldist.sample()
                torch.cuda.empty_cache()
                encoder = encoder.to("cpu")
                del ldist

                # Get corresponding prompt
                img_basename = os.path.basename(file_path).split('.')[0]
                prompt_file = os.path.join(prompt_dir, f"{img_basename}.txt")
                
                if os.path.exists(prompt_file):
                    with open(prompt_file, 'r') as f:
                        file_prompt = f.read().strip()
                    click.echo(f"Using prompt from file: {file_prompt}")
                else:
                    click.echo(f"Warning: Prompt file not found for {file_path}. Using default prompt.")
            
                # Package input for generate_video
                input_image = {
                    "tensor": image_tensor,
                    "filename": os.path.basename(file_path),
                    "cond_position": cond_position
                }
                
                output = generate_video(
                    file_prompt,
                    negative_prompt,
                    width,
                    height,
                    num_frames,
                    seed,
                    cfg_scale,
                    num_steps,
                    data_path,
                    input_image,
                    noise_multiplier=noise_multiplier,
                )
                click.echo(f"Video generated at: {output}")
        return


if __name__ == "__main__":
    generate_cli()
