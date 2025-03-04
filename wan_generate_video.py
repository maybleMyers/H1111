import argparse
from datetime import datetime
import random
import os
import time

import torch
import accelerate
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from PIL import Image

from wan.configs import WAN_CONFIGS, SUPPORTED_SIZES
import wan
from wan.modules.vae import WanVAE
from networks import lora

try:
    from lycoris.kohya import create_network_from_weights
except ImportError:
    pass

from utils.model_utils import str_to_dtype
from utils.device_utils import clean_memory_on_device, synchronize_device
from utils.safetensors_utils import mem_eff_save_file
from hv_generate_video import save_images_grid, save_videos_grid

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Wan 2.1 inference script")

    # WAN arguments
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory (Wan 2.1 official).")
    parser.add_argument("--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run.")
    # parser.add_argument("--use_prompt_extend", action="store_true", default=False, help="Whether to use prompt extend.")
    # prompt extend is not supported
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"], help="The solver used to sample.")

    parser.add_argument("--dit", type=str, default=None, help="DiT checkpoint path")
    parser.add_argument("--vae", type=str, default=None, help="VAE checkpoint path")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default is bfloat16")
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
    parser.add_argument("--t5", type=str, default=None, help="text encoder (T5) checkpoint path")
    parser.add_argument("--clip", type=str, default=None, help="text encoder (CLIP) checkpoint path")
    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument(
        "--save_merged_model",
        type=str,
        default=None,
        help="Save merged model to path. If specified, no inference will be performed.",
    )
    parser.add_argument("--exclude_single_blocks", action="store_true", help="Exclude single blocks when loading LoRA weights")
    parser.add_argument("--lycoris", action="store_true", help="use lycoris for inference")

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
    parser.add_argument("--image_path", type=str, default=None, help="path to image for image2video inference")
    # parser.add_argument(
    #     "--split_uncond",
    #     action="store_true",
    #     help="split unconditional call for classifier free guidance, slower but less memory usage",
    # )
    # parser.add_argument("--strength", type=float, default=0.8, help="strength for video2video inference")

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
    # parser.add_argument(
    #     "--split_attn", action="store_true", help="use split attention, default is False. if True, --split_uncond becomes True"
    # )
    # parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    # parser.add_argument(
    #     "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    # )
    parser.add_argument("--blocks_to_swap", type=int, default=None, help="number of blocks to swap in the model")
    # parser.add_argument("--img_in_txt_in_offloading", action="store_true", help="offload img_in and txt_in to cpu")
    parser.add_argument(
        "--output_type", type=str, default="video", choices=["video", "images", "latent", "both"], help="output type"
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")

    args = parser.parse_args()

    assert (args.latent_path is None or len(args.latent_path) == 0) or (
        args.output_type == "images" or args.output_type == "video"
    ), "latent_path is only supported for images or video output"

    return args


def check_inputs(args):
    height = args.video_size[0]
    width = args.video_size[1]
    size = f"{width}*{height}"

    # assert (
    #     size in SUPPORTED_SIZES[args.task]
    # ), f"Size {size} is not supported for task {args.task}. Supported sizes are {SUPPORTED_SIZES[args.task]}."
    if size not in SUPPORTED_SIZES[args.task]:
        logger.warning(f"Size {size} is not supported for task {args.task}. Supported sizes are {SUPPORTED_SIZES[args.task]}.")
    video_length = args.video_length

    return height, width, video_length


def load_transformer_with_lora(
    transformer, lora_weights, lora_multipliers, device, dit_weight_dtype, exclude_single_blocks=False, use_lycoris=False
):
    """Load transformer model and apply LoRA weights"""
    if lora_weights is None or len(lora_weights) == 0:
        return transformer
    
    logger.info(f"Loading {len(lora_weights)} LoRA weight(s) for Wan model")

    for i, lora_weight in enumerate(lora_weights):
        if lora_multipliers is None or not isinstance(lora_multipliers, list) or len(lora_multipliers) <= i:
            lora_multiplier = 1.0
        else:
            lora_multiplier = lora_multipliers[i]

        logger.info(f"Loading LoRA weights from {lora_weight} with multiplier {lora_multiplier}")
        weights_sd = load_file(lora_weight)
        
        # Filter to exclude keys that are part of single_blocks
        if exclude_single_blocks:
            filtered_weights = {k: v for k, v in weights_sd.items() if "single_blocks" not in k}
            weights_sd = filtered_weights
            
        if use_lycoris:
            from lycoris.kohya import create_network_from_weights
            lycoris_net, _ = create_network_from_weights(
                multiplier=lora_multiplier,
                file=None,
                weights_sd=weights_sd,
                unet=transformer,
                text_encoder=None,
                vae=None,
                for_inference=True,
            )
            logger.info("Merging LyCORIS weights to transformer model")
            lycoris_net.merge_to(None, transformer, weights_sd, dtype=None, device=device)
        else:
            network = lora.create_network_from_weights_wan(
                lora_multiplier, weights_sd, transformer=transformer, for_inference=True
            )
            logger.info("Merging LoRA weights to transformer model")
            network.merge_to(None, transformer, weights_sd, device=device, non_blocking=True)

        synchronize_device(device)
        logger.info(f"LoRA weights from {lora_weight} loaded and merged")

    return transformer


def apply_lora_to_wan_model(model_obj, lora_weight, lora_multiplier, device, dtype, exclude_single_blocks=False, use_lycoris=False):
    """
    Apply LoRA weights to Wan model by accessing the transformer correctly
    
    Args:
        model_obj: The Wan model instance (WanI2V or WanT2V)
        lora_weight: List of LoRA weight paths to apply
        lora_multiplier: List of multipliers for each LoRA
        device: The device to use
        dtype: The data type to use
        exclude_single_blocks: Whether to exclude single blocks
        use_lycoris: Whether to use LyCORIS
    
    Returns:
        The modified model object
    """
    # Debug the model structure
    logger.info(f"Inspecting model structure for LoRA application...")
    logger.info(f"Model object type: {type(model_obj)}")
    
    # List the key attributes of the model object to find the transformer
    transformer = None
    
    # Try different possible attribute names
    if hasattr(model_obj, 'model'):
        transformer = model_obj.model
        logger.info("Found transformer at model_obj.model")
    elif hasattr(model_obj, 'transformer'):
        transformer = model_obj.transformer
        logger.info("Found transformer at model_obj.transformer")
    elif hasattr(model_obj, 'diffusion_model'):
        transformer = model_obj.diffusion_model
        logger.info("Found transformer at model_obj.diffusion_model")
    elif hasattr(model_obj, 'pipeline') and 'model' in model_obj.pipeline:
        transformer = model_obj.pipeline['model']
        logger.info("Found transformer at model_obj.pipeline['model']")
    elif hasattr(model_obj, 'hybrid_seq_parallel_attn'):
        transformer = model_obj  # The model_obj itself might be the transformer
        logger.info("Model object appears to be the transformer itself")
    else:
        # As a last resort, check all attributes for a likely transformer
        for attr_name in dir(model_obj):
            if not attr_name.startswith('_'):  # Skip private attributes
                attr = getattr(model_obj, attr_name)
                if isinstance(attr, torch.nn.Module) and hasattr(attr, 'blocks'):
                    transformer = attr
                    logger.info(f"Found possible transformer at model_obj.{attr_name}")
                    break
    
    if transformer is None:
        logger.warning("Could not identify transformer model. LoRA weights not applied.")
        logger.info("Available attributes:")
        for attr_name in dir(model_obj):
            if not attr_name.startswith('_'):  # Skip private attributes
                logger.info(f"  {attr_name}: {type(getattr(model_obj, attr_name))}")
        return model_obj
    
    # Apply LoRA weights to the transformer model
    transformer = load_transformer_with_lora(
        transformer, 
        lora_weight, 
        lora_multiplier if isinstance(lora_multiplier, list) else [lora_multiplier],
        device, 
        dtype,
        exclude_single_blocks,
        use_lycoris
    )
    
    # Update the model with the modified transformer
    # Match how we found the transformer to how we update it
    if hasattr(model_obj, 'model'):
        model_obj.model = transformer
    elif hasattr(model_obj, 'transformer'):
        model_obj.transformer = transformer
    elif hasattr(model_obj, 'diffusion_model'):
        model_obj.diffusion_model = transformer
    elif hasattr(model_obj, 'pipeline') and 'model' in model_obj.pipeline:
        model_obj.pipeline['model'] = transformer
    elif hasattr(model_obj, 'hybrid_seq_parallel_attn'):
        # If the model_obj itself is the transformer, we don't need to update anything
        pass
    else:
        for attr_name in dir(model_obj):
            if not attr_name.startswith('_'):
                attr = getattr(model_obj, attr_name)
                if attr is transformer:
                    setattr(model_obj, attr_name, transformer)
                    break
    
    return model_obj


def debug_model_structure(model_obj, logger):
    """
    Debug the model structure to help identify issues with LoRA application
    
    Args:
        model_obj: The model object to debug
        logger: Logger instance for output
    """
    logger.info(f"Model object type: {type(model_obj)}")
    
    # Check if model attribute exists
    if hasattr(model_obj, 'model'):
        logger.info(f"Model attribute type: {type(model_obj.model)}")
        
        # Check if the model has blocks attribute (common in transformer models)
        if hasattr(model_obj.model, 'blocks'):
            logger.info(f"Model has {len(model_obj.model.blocks)} blocks")
            
            # Check the type of the first block
            if len(model_obj.model.blocks) > 0:
                logger.info(f"First block type: {type(model_obj.model.blocks[0])}")
    else:
        logger.warning("Model does not have a 'model' attribute")
    
    # List all attributes of the model object
    logger.info("Model attributes:")
    for attr_name in dir(model_obj):
        if not attr_name.startswith('_'):  # Skip private attributes
            logger.info(f"  {attr_name}: {type(getattr(model_obj, attr_name))}")


def debug_lora_weights(lora_weights, logger):
    """
    Debug the LoRA weights structure
    
    Args:
        lora_weights: The LoRA weights to debug
        logger: Logger instance for output
    """
    # Print some stats about the lora weights
    if isinstance(lora_weights, dict):
        logger.info(f"LoRA weights has {len(lora_weights)} keys")
        
        # Show some example keys
        sample_keys = list(lora_weights.keys())[:5]
        logger.info(f"Sample keys: {sample_keys}")
        
        # Check key formats
        key_prefixes = set()
        for key in lora_weights.keys():
            if '.' in key:
                prefix = key.split('.')[0]
                key_prefixes.add(prefix)
        
        logger.info(f"Key prefixes found: {key_prefixes}")
    else:
        logger.info(f"LoRA weights is not a dict: {type(lora_weights)}")


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
        prompt = args.prompt  # TODO load prompts from file
        assert prompt is not None, "prompt is required"

        seed = args.seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0

        # create pipeline
        if "t2v" in args.task or "t2i" in args.task:
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
            
            # Apply LoRA weights if provided - using the updated function
            if args.lora_weight and len(args.lora_weight) > 0:
                logger.info(f"Applying LoRA weights to WanT2V model: {args.lora_weight}")
                wan_t2v = apply_lora_to_wan_model(
                    wan_t2v,
                    args.lora_weight,
                    args.lora_multiplier if isinstance(args.lora_multiplier, list) else [args.lora_multiplier],
                    device,
                    dit_weight_dtype,
                    args.exclude_single_blocks,
                    args.lycoris
                )
                
                # If we found a transformer and save_merged_model is specified, save the model
                if args.save_merged_model:
                    # We need to search for the transformer again
                    transformer_found = False
                    for attr_name in dir(wan_t2v):
                        if not attr_name.startswith('_'):  # Skip private attributes
                            attr = getattr(wan_t2v, attr_name)
                            if isinstance(attr, torch.nn.Module) and hasattr(attr, 'blocks'):
                                logger.info(f"Saving merged model transformer from {attr_name} to {args.save_merged_model}")
                                mem_eff_save_file(attr.state_dict(), args.save_merged_model)
                                transformer_found = True
                                logger.info("Merged model saved")
                                return
                    
                    if not transformer_found:
                        logger.warning("No suitable transformer found to save as merged model.")

            logging.info(f"Generating {'image' if 't2i' in args.task else 'video'} ...")
            latents = wan_t2v.generate(
                accelerator,
                prompt,
                size=size,
                frame_num=video_length,
                shift=args.flow_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.infer_steps,
                guide_scale=args.guidance_scale,
                seed=seed,
                blocks_to_swap=blocks_to_swap,
            )
            latents = latents.unsqueeze(0)
            del wan_t2v
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
            
            # Apply LoRA weights if provided - using the updated function
            if args.lora_weight and len(args.lora_weight) > 0:
                logger.info(f"Applying LoRA weights to WanI2V model: {args.lora_weight}")
                wan_i2v = apply_lora_to_wan_model(
                    wan_i2v,
                    args.lora_weight,
                    args.lora_multiplier if isinstance(args.lora_multiplier, list) else [args.lora_multiplier],
                    device,
                    dit_weight_dtype,
                    args.exclude_single_blocks,
                    args.lycoris
                )
                
                # If we found a transformer and save_merged_model is specified, save the model
                if args.save_merged_model:
                    # We need to search for the transformer again
                    transformer_found = False
                    for attr_name in dir(wan_i2v):
                        if not attr_name.startswith('_'):  # Skip private attributes
                            attr = getattr(wan_i2v, attr_name)
                            if isinstance(attr, torch.nn.Module) and hasattr(attr, 'blocks'):
                                logger.info(f"Saving merged model transformer from {attr_name} to {args.save_merged_model}")
                                mem_eff_save_file(attr.state_dict(), args.save_merged_model)
                                transformer_found = True
                                logger.info("Merged model saved")
                                return
                    
                    if not transformer_found:
                        logger.warning("No suitable transformer found to save as merged model.")

            # i2v inference
            logger.info(f"Image2Video inference: {args.image_path}")
            image = Image.open(args.image_path).convert("RGB")

            vae = load_vae()

            logging.info(f"Generating video ...")
            latents = wan_i2v.generate(
                accelerator,
                prompt,
                img=image,
                size=size,
                frame_num=video_length,
                shift=args.flow_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.infer_steps,
                guide_scale=args.guidance_scale,
                seed=seed,
                blocks_to_swap=blocks_to_swap,
                vae=vae,
            )
            del wan_i2v
            latents = latents.unsqueeze(0)

    clean_memory_on_device(device)

    # prepare accelerator for decode
    output_type = args.output_type

    def decode_latents(x0):
        nonlocal vae
        if vae is None:
            vae = load_vae()
        vae.to_device(device)

        logger.info(f"Decoding video from latents: {x0.shape}")
        x0 = x0.to(device)  # , dtype=vae_dtype)
        # with accelerator.autocast(), torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=vae_dtype), torch.no_grad():
            videos = vae.decode(x0)
        logger.info(f"Decoding complete")
        video = videos[0]
        del videos
        video = video.to(torch.float32).cpu()
        return video

    # Save samples
    save_path = args.save_path  # if args.save_path_suffix == "" else f"{args.save_path}_{args.save_path_suffix}"
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
                # "embedded_cfg_scale": f"{args.embedded_cfg_scale}",
            }
            if args.negative_prompt is not None:
                metadata["negative_prompt"] = f"{args.negative_prompt}"
            # Add LoRA information to metadata
            if args.lora_weight and len(args.lora_weight) > 0:
                metadata["lora_weights"] = ",".join(args.lora_weight)
                if isinstance(args.lora_multiplier, list):
                    metadata["lora_multipliers"] = ",".join(map(str, args.lora_multiplier))
                else:
                    metadata["lora_multipliers"] = str(args.lora_multiplier)
                    
        sd = {"latent": latents[0]}
        save_file(sd, latent_path, metadata=metadata)

        logger.info(f"Latent save to: {latent_path}")
    if output_type == "video" or output_type == "both":
        # save video
        sample = decode_latents(latents)
        original_name = "" if original_base_names is None else f"_{original_base_names[0]}"
        sample = sample.unsqueeze(0)
        video_path = f"{save_path}/{time_flag}_{seed}{original_name}.mp4"
        save_videos_grid(sample, video_path, fps=args.fps, rescale=True)
        logger.info(f"Sample save to: {video_path}")
    elif output_type == "images":
        # save images
        sample = decode_latents(latents)
        original_name = "" if original_base_names is None else f"_{original_base_names[0]}"
        sample = sample.unsqueeze(0)
        image_name = f"{time_flag}_{seed}{original_name}"
        save_images_grid(sample, save_path, image_name, rescale=True)
        logger.info(f"Sample images save to: {save_path}/{image_name}")

    logger.info("Done!")


if __name__ == "__main__":
    main()