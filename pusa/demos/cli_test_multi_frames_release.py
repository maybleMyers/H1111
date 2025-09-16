#! /usr/bin/env python
import json
import os
import time

import click
import numpy as np
import torch

from genmo.lib.progress import progress_bar
from genmo.lib.utils import save_video
from genmo.mochi_preview.pipelines_multi_frames_release import (
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
    multi_cond=None,
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
        "condition_image": multi_cond["tensors"],
        "condition_frame_idx": multi_cond["positions"],
        "noise_multiplier": multi_cond["noise_multipliers"]
    }


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
        
        # Generate descriptive prefix for the result filename
        positions_str =  multi_cond["positions"]
        cond_position = f"multi_{positions_str}"
        noise_multiplier = multi_cond["noise_multipliers"]
            
        results_base_dir = "./video_test_demos_results"
        results_dir = os.path.join(results_base_dir, f"{model_name}_{checkpoint_name}_github_user_demo_{cond_position}pos_{num_inference_steps}steps_crop_{noise_multiplier}sigma")
        os.makedirs(results_dir, exist_ok=True)
            
        output_path = os.path.join(
            results_dir,
            f"{timestamp_str}.mp4"
        )
        save_video(final_frames, output_path)        

        return output_path

from textwrap import dedent


@click.command()
@click.option("--prompt", default=None, type=str, help="Prompt for generation.")
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
@click.option("--multi_cond", default=None, help="JSON string with multiple condition inputs in format: {pos: [img_dir, prompt_dir, noise_mult]}.")


def generate_cli(
    prompt, negative_prompt, width, height, num_frames, seed, cfg_scale, num_steps, model_dir, 
    dit_path, cpu_offload, data_path, multi_cond
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

    # Process multi-conditional inputs

    # Parse JSON input for multiple conditioning
    import json
    conditions = json.loads(multi_cond)
    
    # Create structures to store tensors, and noise multipliers
    latent_tensors = []
    noise_multipliers = []
    positions = []
    
    # Process each conditioning position
    for pos, cond_info in conditions.items():
        img_dir, noise_mult = cond_info
        pos = int(pos)
        positions.append(pos)
        
        # Load image and encode
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Load the image
        image = Image.open(img_dir)
        
        # Crop and resize the image
        target_ratio = width / height
        current_ratio = image.width / image.height
        
        if current_ratio > target_ratio:
            new_width = int(image.height * target_ratio)
            x1 = (image.width - new_width) // 2
            image = image.crop((x1, 0, x1 + new_width, image.height))
        else:
            new_height = int(image.width / target_ratio)
            y1 = (image.height - new_height) // 2
            image = image.crop((0, y1, image.width, y1 + new_height))
        
        # Resize the cropped image
        transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ])
        
        image_tensor = (transform(image) * 2 - 1).unsqueeze(1).unsqueeze(0)
        image_tensor = add_fourier_features(image_tensor.to(device))
        
        # Encode image to latent
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoder = encoder.to(device)
                ldist = encoder(image_tensor)
                image_latent = ldist.sample()
        
        # Store the individual latent tensor for this position
        latent_tensors.append(image_latent[:, :, 0, :, :])
        
        # Store noise multiplier
        noise_multipliers.append(float(noise_mult) if noise_mult else 0.3)
        
        # Clean up to save memory
        del ldist, image_tensor
        torch.cuda.empty_cache()
    
    # Build multi-condition data structure
    multi_cond_data = {
        "tensors": latent_tensors,       # Dict of position -> tensor
        "positions": positions,  # Dict of position -> noise multiplier
        "noise_multipliers": noise_multipliers,  # Dict of position -> noise multiplier
    }
    


    prompt = prompt
    
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
            multi_cond=multi_cond_data,
        )
        click.echo(f"Video generated at: {output}")
    return


if __name__ == "__main__":
    generate_cli()
