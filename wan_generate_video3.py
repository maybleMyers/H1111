import argparse
from datetime import datetime
import random
import os
import time
import math
import numpy as np
import inspect

import torch
import accelerate
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from PIL import Image

from networks import lora_wan
from utils.safetensors_utils import mem_eff_save_file
from wan.configs import WAN_CONFIGS, SUPPORTED_SIZES
import wan
from wan.modules.vae import WanVAE

# Import the local flow match scheduler
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

try:
    from lycoris.kohya import create_network_from_weights
except:
    pass

from utils.model_utils import str_to_dtype
from utils.device_utils import clean_memory_on_device
from hv_generate_video import save_images_grid, save_videos_grid, synchronize_device, load_video, glob_images, resize_image_to_bucket

import logging
import cv2
from typing import Union, Callable
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def extend_video_frames(video: torch.Tensor, target_frames: int) -> torch.Tensor:
    current_frames = video.shape[2]
    if current_frames >= target_frames:
        return video
    
    base_repeats = target_frames // current_frames
    extra = target_frames % current_frames  # Remaining repeats to distribute
    
    # Create repeat tensor with partial repetition for even distribution
    repeats = torch.full((current_frames,), base_repeats, 
                        dtype=torch.int64, device=video.device)
    repeats[:extra] += 1  # Distribute extra repeats to early frames
    
    # Create interleaved index pattern (e.g., 001122.. instead of 012012)
    indices = torch.arange(current_frames, device=video.device)
    indices = indices.repeat_interleave(repeats)
    
    extended_video = torch.index_select(video, 2, indices)
    return extended_video

def load_and_extend_video(video_path, size, video_length):
    """
    Load video and extend it if needed to match target length.
    """
    width, height = size
    
    if os.path.isfile(video_path):
        logger.info(f"Loading video from file: {video_path}")
        # Use OpenCV to load the video directly
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to target dimensions
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame)
        
        cap.release()
    else:
        logger.info(f"Loading images from directory: {video_path}")
        # Load from directory of images
        image_files = glob_images(video_path)
        if len(image_files) == 0:
            raise ValueError(f"No image files found in {video_path}")
            
        image_files.sort()
        frames = []
        for image_file in image_files:
            image = Image.open(image_file)
            image = resize_image_to_bucket(image, (width, height))
            frames.append(np.array(image))
    
    logger.info(f"Loaded {len(frames)} frames from video")
    
    if len(frames) == 0:
        raise ValueError(f"No frames could be read from video: {video_path}")
    
    if len(frames) < video_length:
        logger.info(f"Video length ({len(frames)}) is less than target length ({video_length}). Extending video...")
        
        # Convert list of frames to tensor
        video_tensor = torch.from_numpy(np.stack(frames, axis=0))  # [F, H, W, C]
        video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, F, H, W]
        
        # Extend the video
        extended_tensor = extend_video_frames(video_tensor, video_length)
        
        # Convert back to list of frames
        extended_tensor = extended_tensor.squeeze(0).permute(1, 2, 3, 0)  # [F, H, W, C]
        frames = [frame.numpy() for frame in extended_tensor]
        
        logger.info(f"Extended to {len(frames)} frames")
    elif len(frames) > video_length:
        # If we have too many frames, evenly sample the required number
        indices = np.linspace(0, len(frames) - 1, video_length, dtype=int)
        frames = [frames[i] for i in indices]
        logger.info(f"Sampled down to {len(frames)} frames")
    
    # Stack frames and convert to tensor format
    video = np.stack(frames, axis=0)  # F, H, W, C
    video = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0).float()  # 1, C, F, H, W
    video = video / 255.0
    
    return video

def parse_args():
    parser = argparse.ArgumentParser(description="Wan 2.1 inference script")

    # WAN arguments
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory (Wan 2.1 official).")
    parser.add_argument("--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run.")
    parser.add_argument(
        "--sample_solver", type=str, default="vanilla", choices=["unipc", "dpm++", "vanilla"], help="The solver used to sample."
    )

    parser.add_argument("--dit", type=str, default=None, help="DiT checkpoint path")
    parser.add_argument("--vae", type=str, default=None, help="VAE checkpoint path")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default is bfloat16")
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
    parser.add_argument("--t5", type=str, default=None, help="text encoder (T5) checkpoint path")
    parser.add_argument("--clip", type=str, default=None, help="text encoder (CLIP) checkpoint path")
    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument("--exclude_single_blocks", action="store_true", help="Exclude single blocks when loading LoRA weights")
    parser.add_argument(
        "--save_merged_model",
        type=str,
        default=None,
        help="Save merged model to path. If specified, no inference will be performed.",
    )

    # inference
    parser.add_argument("--prompt", type=str, required=True, help="prompt for generation")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="negative prompt for generation, use default negative prompt if not specified",
    )
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256], help="video size, height and width")
    parser.add_argument("--video_length", type=int, default=None, help="video length, Default is 81 for video inference")
    parser.add_argument("--fps", type=int, default=16, help="video fps, Default is 16")
    parser.add_argument("--infer_steps", type=int, default=None, help="number of inference steps")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for classifier free guidance. Default is 5.0.",
    )
    parser.add_argument("--video_path", type=str, default=None, help="path to video for video2video inference")
    parser.add_argument("--strength", type=float, default=0.8, help="strength for video2video inference (0.0-1.0)")
    parser.add_argument("--image_path", type=str, default=None, help="path to image for image2video inference")

    # Flow Matching
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=None,
        help="Shift factor for flow matching schedulers. Default is 3.0 for I2V with 832*480, 5.0 for others.",
    )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "flash2", "flash3", "torch", "sageattn", "xformers", "sdpa"],
        help="attention mode",
    )
    parser.add_argument("--blocks_to_swap", type=int, default=None, help="number of blocks to swap in the model")
    parser.add_argument(
        "--output_type", type=str, default="video", choices=["video", "images", "latent", "both"], help="output type"
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")
    parser.add_argument("--lycoris", action="store_true", help="use lycoris for inference")

    args = parser.parse_args()

    assert (args.latent_path is None or len(args.latent_path) == 0) or (
        args.output_type == "images" or args.output_type == "video"
    ), "latent_path is only supported for images or video output"

    return args

def check_inputs(args):
    height = args.video_size[0]
    width = args.video_size[1]
    size = f"{width}*{height}"

    if size not in SUPPORTED_SIZES[args.task]:
        logger.warning(f"Size {size} is not supported for task {args.task}. Supported sizes are {SUPPORTED_SIZES[args.task]}.")
    video_length = args.video_length

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
    return height, width, video_length

def monkey_patch_generate(
    original_generate: Callable, 
    precomputed_latents: torch.Tensor,
    reduced_steps: int,
    total_steps: int
) -> Callable:
    """
    Create a monkey-patched version of the generate method that uses precomputed latents
    and adjusts the number of steps.
    """
    def patched_generate(self, *args, **kwargs):
        logger.info(f"patched_generate args: {args}")
        logger.info(f"patched_generate kwargs: {kwargs}")
        sampling_steps = kwargs.get('sampling_steps', 50)
        precomputed_latents = kwargs.pop('precomputed_latents', None)
        if 'sampling_steps' in kwargs:
            original_sampling_steps = kwargs['sampling_steps']
            kwargs['sampling_steps'] = reduced_steps
        result = original_generate(*args, **kwargs)
        if 'sampling_steps' in kwargs:
            kwargs['sampling_steps'] = original_sampling_steps
        return result
    
    return patched_generate

def main():
    args = parse_args()

    # validate args
    if args.video_length is None:
        args.video_length = 1 if "t2i" in args.task else 81
    if "t2i" in args.task:
        assert args.video_length == 1, f"video_length should be 1 for task {args.task}"

    latents_mode = args.latent_path is not None and len(args.latent_path) > 0
    if not latents_mode:
        # check inputs: may be height, width, video_length etc will be changed for each generation in future
        height, width, video_length = check_inputs(args)
        size = (width, height)
    else:
        height, width, video_length = None, None, None
        size = None

    if args.infer_steps is None:
        args.infer_steps = 40 if "i2v" in args.task else 50
    if args.flow_shift is None:
        args.flow_shift = 3.0 if "i2v" in args.task and (width == 832 and height == 480 or width == 480 and height == 832) else 5.0

    print(
        f"video size: {height}x{width}@{video_length} (HxW@F), fps: {args.fps}, infer_steps: {args.infer_steps}, flow_shift: {args.flow_shift}"
    )

    cfg = WAN_CONFIGS[args.task]

    # prepare device and dtype
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dit_dtype = torch.bfloat16
    dit_weight_dtype = torch.float8_e4m3fn if args.fp8 else dit_dtype
    vae_dtype = str_to_dtype(args.vae_dtype) if args.vae_dtype is not None else dit_dtype
    logger.info(
        f"Using device: {device}, DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}, VAE precision: {vae_dtype}"
    )

    def load_vae():
        vae_path = args.vae if args.vae is not None else os.path.join(args.ckpt_dir, cfg.vae_checkpoint)

        logger.info(f"Loading VAE model from {vae_path}")
        cache_device = torch.device("cpu") if args.vae_cache_cpu else None
        vae = WanVAE(vae_path=vae_path, device=device, dtype=vae_dtype, cache_device=cache_device)
        return vae

    vae = None

    original_base_names = None
    if latents_mode:
        original_base_names = []
        latents_list = []
        seeds = []
        assert len(args.latent_path) == 1, "Only one latent path is supported for now"
        for latent_path in args.latent_path:
            original_base_names.append(os.path.splitext(os.path.basename(latent_path))[0])
            seed = 0

            if os.path.splitext(latent_path)[1] != ".safetensors":
                latents = torch.load(latent_path, map_location="cpu")
            else:
                latents = load_file(latent_path)["latent"]
                with safe_open(latent_path, framework="pt") as f:
                    metadata = f.metadata()
                if metadata is None:
                    metadata = {}
                logger.info(f"Loaded metadata: {metadata}")

                if "seeds" in metadata:
                    seed = int(metadata["seeds"])

            seeds.append(seed)
            latents_list.append(latents)

            logger.info(f"Loaded latent from {latent_path}. Shape: {latents.shape}")
        latents = torch.stack(latents_list, dim=0)  # [N, ...]
    else:
        # prepare accelerator
        mixed_precision = "bf16" if dit_dtype == torch.bfloat16 else "fp16"
        accelerator = accelerate.Accelerator(mixed_precision=mixed_precision)

        # load prompt
        prompt = args.prompt
        assert prompt is not None, "prompt is required"

        seed = args.seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0

        # load LoRA weights
        merge_lora = None
        if args.lora_weight is not None and len(args.lora_weight) > 0:

            def merge_lora(transformer):
                for i, lora_weight in enumerate(args.lora_weight):
                    if args.lora_multiplier is not None and len(args.lora_multiplier) > i:
                        lora_multiplier = args.lora_multiplier[i]
                    else:
                        lora_multiplier = 1.0

                    logger.info(f"Loading LoRA weights from {lora_weight} with multiplier {lora_multiplier}")
                    weights_sd = load_file(lora_weight)
                    # Filter to exclude keys that are part of single_blocks
                    if args.exclude_single_blocks:
                        filtered_weights = {k: v for k, v in weights_sd.items() if "single_blocks" not in k}
                        weights_sd = filtered_weights
                    if args.lycoris:
                        lycoris_net, _ = create_network_from_weights(
                            multiplier=lora_multiplier,
                            file=None,
                            weights_sd=weights_sd,
                            unet=transformer,
                            text_encoder=None,
                            vae=None,
                            for_inference=True,
                        )
                    else:
                        network = lora_wan.create_arch_network_from_weights(
                            lora_multiplier, weights_sd, unet=transformer, for_inference=True
                        )
                    logger.info("Merging LoRA weights to DiT model")

                    if args.lycoris:
                        lycoris_net.merge_to(None, transformer, weights_sd, dtype=None, device=device)
                    else:
                        network.merge_to(None, transformer, weights_sd, device=device, non_blocking=True)

                    synchronize_device(device)

                    logger.info("LoRA weights loaded")

                # save model here before casting to dit_weight_dtype
                if args.save_merged_model:
                    logger.info(f"Saving merged model to {args.save_merged_model}")
                    mem_eff_save_file(transformer.state_dict(), args.save_merged_model)
                    logger.info("Merged model saved")
                    return

        # Video latents processing for video2video
        video_latents = None
        if args.video_path is not None:
            # v2v inference
            logger.info(f"Video2Video inference: {args.video_path}")
            
            try:
                # Load the input video
                video = load_and_extend_video(args.video_path, (width, height), video_length)
                
                # Load VAE for encoding
                vae = load_vae()
                
                # Encode video to latents
                logger.info(f"Encoding video to latents")
                
                # Process the video according to WanVAE's expectations
                video = video.to(device=device)
                video = video * 2 - 1  # 0, 1 -> -1, 1
                
                with torch.no_grad():
                    # Convert single batch tensor to list for WanVAE
                    video_list = [video.squeeze(0)]  # Shape becomes [C, F, H, W]
                    encoded_latents = vae.encode(video_list)
                    # Take the first element from the list
                    video_latents = encoded_latents[0]
                    
                    # Add batch dimension back if needed for compatibility
                    if video_latents.ndim == 4:  # [C, F, H, W]
                        video_latents = video_latents.unsqueeze(0)  # [1, C, F, H, W]
                    
                    video_latents = video_latents.to(device=device, dtype=dit_dtype)
                
                logger.info(f"Video encoded to latents: {video_latents.shape}")
                clean_memory_on_device(device)
                
            except Exception as e:
                logger.error(f"Error processing input video: {e}")
                logger.info("Continuing with regular text-to-video generation")
                video_latents = None

        # Create T2V model instance
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device=device,
            dtype=dit_weight_dtype,
            dit_path=args.dit,
            dit_attn_mode=args.attn_mode,
            t5_path=args.t5,
            t5_fp8=args.fp8_t5,
        )

        # Handle video2video generation with strength parameter
        if video_latents is not None and args.strength < 1.0:
            logger.info(f"Setting up v2v with strength {args.strength}")
            generator = torch.Generator(device=device).manual_seed(seed)

            # Get VAE scale params matching latent channels
            scale_shift, scale_scale = vae.scale  # Now [8], [8]

            # Reshape for [1,1,8,1,1] broadcasting
            scale_shift = scale_shift.view(1, 1, -1, 1, 1).to(device=video_latents.device, dtype=video_latents.dtype)
            scale_scale = scale_scale.view(1, 1, -1, 1, 1).to(device=video_latents.device, dtype=video_latents.dtype)

            # Apply inverse VAE transform
            scaled_video_latents = (video_latents - scale_shift) / scale_scale

            # Configure scheduler
            fm_scheduler = FlowMatchDiscreteScheduler(shift=args.flow_shift, reverse=True)
            fm_scheduler.set_timesteps(args.infer_steps, device=device)

            # Calculate starting step
            reduced_steps = max(1, int(args.infer_steps * args.strength))
            start_step_idx = max(0, len(fm_scheduler.timesteps) - reduced_steps)
            start_time = fm_scheduler.timesteps[start_step_idx]

            # Generate noise
            base_noise = torch.randn_like(scaled_video_latents, generator=generator)
            noisy_latents = fm_scheduler.add_noise(scaled_video_latents, base_noise, start_time)

            # Patch scheduler initialization
            original_step = fm_scheduler.step
            fm_scheduler.step = lambda *args, **kw: (noisy_latents, None)

            try:
                latents = wan_t2v.generate(
                    accelerator=accelerator,
                    merge_lora=None,
                    dit_loading_dtype=None,
                    input_prompt=prompt,
                    size=size,
                    frame_num=video_length,
                    shift=args.flow_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=reduced_steps,
                    guide_scale=args.guidance_scale,
                    n_prompt=args.negative_prompt if args.negative_prompt else "",
                    seed=seed,
                    blocks_to_swap=blocks_to_swap
                )
                
            finally:
                fm_scheduler.step = original_step

        elif "t2v" in args.task or "t2i" in args.task:
            # Regular generation from scratch
            logging.info(f"Generating {'image' if 't2i' in args.task else 'video'} from scratch...")
            latents = wan_t2v.generate(
                accelerator,
                merge_lora,
                torch.bfloat16 if merge_lora is not None else None,
                prompt,
                size=size,
                frame_num=video_length,
                shift=args.flow_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.infer_steps,
                guide_scale=args.guidance_scale,
                n_prompt=args.negative_prompt if args.negative_prompt else "",
                seed=seed,
                blocks_to_swap=blocks_to_swap,
            )
        
        elif "i2v" in args.task:
            wan_i2v = wan.WanI2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device=device,
                dtype=dit_weight_dtype,
                dit_path=args.dit,
                dit_attn_mode=args.attn_mode,
                t5_path=args.t5,
                clip_path=args.clip,
                t5_fp8=args.fp8_t5,
            )

            # i2v inference
            logger.info(f"Image2Video inference: {args.image_path}")
            image = Image.open(args.image_path).convert("RGB")

            vae = load_vae()

            logging.info(f"Generating video ...")
            latents = wan_i2v.generate(
                accelerator,
                merge_lora,
                torch.bfloat16 if merge_lora is not None else None,
                prompt,
                img=image,
                size=size,
                frame_num=video_length,
                shift=args.flow_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.infer_steps,
                guide_scale=args.guidance_scale,
                n_prompt=args.negative_prompt if args.negative_prompt else "",
                seed=seed,
                blocks_to_swap=blocks_to_swap,
                vae=vae,
            )
            del wan_i2v
            
        # Ensure latents is properly formatted
        if latents.ndim == 4:  # [C, F, H, W]
            latents = latents.unsqueeze(0)  # [1, C, F, H, W]
        
        # Clean up
        del wan_t2v
        logger.info(f"Wait for 5s to clean memory")
        time.sleep(5.0)
        clean_memory_on_device(device)

    # prepare for decode
    output_type = args.output_type

    def decode_latents(x0):
        nonlocal vae
        if vae is None:
            vae = load_vae()
        vae.to_device(device)

        logger.info(f"Decoding video from latents: {x0.shape}")
        x0 = x0.to(device)
        with torch.autocast(device_type=device.type, dtype=vae_dtype), torch.no_grad():
            videos = vae.decode([x0[0]])  # WanVAE expects a list of latents
        logger.info(f"Decoding complete")
        video = videos[0]
        del videos
        video = video.to(torch.float32).cpu()
        return video

    # Save samples
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    if output_type == "latent" or output_type == "both":
        # save latent
        latent_path = f"{save_path}/{time_flag}_{seed}_latent.safetensors"

        if args.no_metadata:
            metadata = None
        else:
            metadata = {
                "seeds": f"{seed}",
                "prompt": f"{args.prompt}",
                "height": f"{height}",
                "width": f"{width}",
                "video_length": f"{video_length}",
                "infer_steps": f"{args.infer_steps}",
                "guidance_scale": f"{args.guidance_scale}",
            }
            if args.negative_prompt is not None:
                metadata["negative_prompt"] = f"{args.negative_prompt}"
            if args.video_path is not None:
                metadata["strength"] = f"{args.strength}"
                
        sd = {"latent": latents[0]}
        save_file(sd, latent_path, metadata=metadata)

        logger.info(f"Latent saved to: {latent_path}")
        
    if output_type == "video" or output_type == "both":
        # save video
        sample = decode_latents(latents)
        original_name = "" if original_base_names is None else f"_{original_base_names[0]}"
        sample = sample.unsqueeze(0)
        video_path = f"{save_path}/{time_flag}_{seed}{original_name}.mp4"
        save_videos_grid(sample, video_path, fps=args.fps, rescale=True)
        logger.info(f"Sample saved to: {video_path}")
        
    elif output_type == "images":
        # save images
        sample = decode_latents(latents)
        original_name = "" if original_base_names is None else f"_{original_base_names[0]}"
        sample = sample.unsqueeze(0)
        image_name = f"{time_flag}_{seed}{original_name}"
        save_images_grid(sample, save_path, image_name, rescale=True)
        logger.info(f"Sample images saved to: {save_path}/{image_name}")

    logger.info("Done!")

if __name__ == "__main__":
    main()