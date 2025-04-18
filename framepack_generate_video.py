import argparse
import os
import sys
import time
import random
import traceback
from datetime import datetime
from pathlib import Path

import einops
import numpy as np
import safetensors.torch as sf
import torch
import av # For saving video
from PIL import Image
from tqdm import tqdm
import cv2
import subprocess
import torchvision
import tempfile
import shutil


# --- Dependencies from diffusers_helper ---
# Ensure this library is installed or in the PYTHONPATH
try:
    # from diffusers_helper.hf_login import login # Not strictly needed for inference if models public/cached
    from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode #, vae_decode_fake # vae_decode_fake not used here
    from diffusers_helper.utils import (save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw,
                                        resize_and_center_crop, generate_timestamp)
    from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
    from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
    from diffusers_helper.memory import (cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation,
                                         offload_model_from_device_for_memory_preservation, fake_diffusers_current_device,
                                         DynamicSwapInstaller, unload_complete_models, load_model_as_complete)
    from diffusers_helper.clip_vision import hf_clip_vision_encode
    from diffusers_helper.bucket_tools import find_nearest_bucket#, bucket_options # bucket_options no longer needed here
except ImportError:
    print("Error: Could not import modules from 'diffusers_helper'.")
    print("Please ensure the 'diffusers_helper' library is installed and accessible.")
    print("You might need to clone the repository and add it to your PYTHONPATH.")
    sys.exit(1)
# --- End Dependencies ---

from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from transformers import SiglipImageProcessor, SiglipVisionModel

# --- Constants ---
DIMENSION_MULTIPLE = 16 # VAE and model constraints often require divisibility by 8 or 16. 16 is safer.

def parse_args():
    parser = argparse.ArgumentParser(description="FramePack HunyuanVideo inference script (CLI version)")

    # --- Model Paths ---
    parser.add_argument('--transformer_path', type=str, default='lllyasviel/FramePackI2V_HY', help="Path to the FramePack Transformer model")
    parser.add_argument('--vae_path', type=str, default='hunyuanvideo-community/HunyuanVideo', help="Path to the VAE model directory")
    parser.add_argument('--text_encoder_path', type=str, default='hunyuanvideo-community/HunyuanVideo', help="Path to the Llama text encoder directory")
    parser.add_argument('--text_encoder_2_path', type=str, default='hunyuanvideo-community/HunyuanVideo', help="Path to the CLIP text encoder directory")
    parser.add_argument('--image_encoder_path', type=str, default='lllyasviel/flux_redux_bfl', help="Path to the SigLIP image encoder directory")
    parser.add_argument('--hf_home', type=str, default='./hf_download', help="Directory to download/cache Hugging Face models")

    # --- Input ---
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt for generation")

    # --- Output Resolution (Choose ONE method) ---
    parser.add_argument("--target_resolution", type=int, default=None, help=f"Target resolution for the longer side for automatic aspect ratio calculation (bucketing). Used if --width and --height are not specified. Must be positive and ideally divisible by {DIMENSION_MULTIPLE}.")
    parser.add_argument("--width", type=int, default=None, help=f"Explicit target width for the output video. Overrides --target_resolution. Must be positive and ideally divisible by {DIMENSION_MULTIPLE}.")
    parser.add_argument("--height", type=int, default=None, help=f"Explicit target height for the output video. Overrides --target_resolution. Must be positive and ideally divisible by {DIMENSION_MULTIPLE}.")

    # --- Output ---
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save the generated video")

    # --- Generation Parameters (Matching Gradio Demo Defaults where applicable) ---
    parser.add_argument("--seed", type=int, default=None, help="Seed for generation. Random if not set.")
    parser.add_argument("--total_second_length", type=float, default=5.0, help="Total desired video length in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps (changing not recommended)")
    parser.add_argument("--distilled_guidance_scale", "--gs", type=float, default=10.0, help="Distilled CFG Scale (gs)")
    parser.add_argument("--cfg", type=float, default=1.0, help="Classifier-Free Guidance Scale (fixed at 1.0 for FramePack usually)")
    parser.add_argument("--rs", type=float, default=0.0, help="CFG Rescale (fixed at 0.0 for FramePack usually)")
    parser.add_argument("--latent_window_size", type=int, default=9, help="Latent window size (changing not recommended)")

    # --- Performance / Memory ---
    parser.add_argument('--high_vram', action='store_true', help="Force high VRAM mode (loads all models to GPU)")
    parser.add_argument('--low_vram', action='store_true', help="Force low VRAM mode (uses dynamic swapping)")
    parser.add_argument("--gpu_memory_preservation", type=float, default=6.0, help="GPU memory (GB) to preserve when offloading (low VRAM mode)")
    parser.add_argument('--use_teacache', action='store_true', default=True, help="Use TeaCache optimization (default: True)")
    parser.add_argument('--no_teacache', action='store_false', dest='use_teacache', help="Disable TeaCache optimization")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu'). Auto-detects if None.")

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
        print(f"Generated random seed: {args.seed}")

    # Resolution validation
    if args.width is not None and args.height is not None:
        if args.width <= 0 or args.height <= 0:
            print(f"Error: Explicit --width ({args.width}) and --height ({args.height}) must be positive.")
            sys.exit(1)
        if args.target_resolution is not None:
            print("Warning: Both --width/--height and --target_resolution specified. Using explicit --width and --height.")
            args.target_resolution = None # Ignore target_resolution
    elif args.target_resolution is not None:
        if args.target_resolution <= 0:
            print(f"Error: --target_resolution ({args.target_resolution}) must be positive.")
            sys.exit(1)
        if args.width is not None or args.height is not None:
            print("Error: Cannot specify --target_resolution with only one of --width or --height. Provide both or neither.")
            sys.exit(1)
    else:
        # Neither explicit nor target resolution provided
        print(f"Error: You must specify the target resolution using either --target_resolution OR both --width and --height.")
        sys.exit(1)

    # Check divisibility later after calculation/rounding, but warn if initial inputs aren't ideal
    if args.width is not None and args.width % DIMENSION_MULTIPLE != 0:
         print(f"Warning: Specified --width ({args.width}) is not divisible by {DIMENSION_MULTIPLE}. It will be rounded down.")
    if args.height is not None and args.height % DIMENSION_MULTIPLE != 0:
         print(f"Warning: Specified --height ({args.height}) is not divisible by {DIMENSION_MULTIPLE}. It will be rounded down.")
    if args.target_resolution is not None and args.target_resolution % DIMENSION_MULTIPLE != 0:
         print(f"Warning: Specified --target_resolution ({args.target_resolution}) is not divisible by {DIMENSION_MULTIPLE}. The calculated dimensions will be rounded down.")


    # Set HF_HOME environment variable
    os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(args.hf_home))
    os.makedirs(os.environ['HF_HOME'], exist_ok=True)

    return args


def load_models(args):
    """Loads all necessary models."""
    print("Loading models...")
    # Login if needed
    # login() # Uncomment if you need to login for gated models

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(gpu if torch.cuda.is_available() else cpu)
    print(f"Using device: {device}")

    # Load models to CPU first
    print("  Loading Text Encoder 1 (Llama)...")
    text_encoder = LlamaModel.from_pretrained(args.text_encoder_path, subfolder='text_encoder', torch_dtype=torch.float16).cpu()
    print("  Loading Text Encoder 2 (CLIP)...")
    text_encoder_2 = CLIPTextModel.from_pretrained(args.text_encoder_2_path, subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
    print("  Loading Tokenizer 1 (Llama)...")
    tokenizer = LlamaTokenizerFast.from_pretrained(args.text_encoder_path, subfolder='tokenizer')
    print("  Loading Tokenizer 2 (CLIP)...")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.text_encoder_2_path, subfolder='tokenizer_2')
    print("  Loading VAE...")
    vae = AutoencoderKLHunyuanVideo.from_pretrained(args.vae_path, subfolder='vae', torch_dtype=torch.float16).cpu()
    print("  Loading Image Feature Extractor (SigLIP)...")
    feature_extractor = SiglipImageProcessor.from_pretrained(args.image_encoder_path, subfolder='feature_extractor')
    print("  Loading Image Encoder (SigLIP)...")
    image_encoder = SiglipVisionModel.from_pretrained(args.image_encoder_path, subfolder='image_encoder', torch_dtype=torch.float16).cpu()
    print("  Loading Transformer (FramePack)...")
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(args.transformer_path, torch_dtype=torch.bfloat16).cpu()

    # Set to evaluation mode
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()

    # Apply settings
    transformer.high_quality_fp32_output_for_inference = True
    print('transformer.high_quality_fp32_output_for_inference = True')

    # Convert to appropriate dtypes (will be moved to device later)
    transformer.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)

    # Disable gradients
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    print("Models loaded.")
    return {
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "vae": vae,
        "feature_extractor": feature_extractor,
        "image_encoder": image_encoder,
        "transformer": transformer,
        "device": device
    }

def save_video_with_fallback(video_tensor, output_path, fps=30):
    """
    Save a video tensor to a file with multiple fallback options to handle platform differences.
    
    Args:
        video_tensor: A tensor of shape [B, C, T, H, W]
        output_path: Output file path
        fps: Frames per second
    """
    try:
        # First attempt: Try torchvision's writer directly
        import torchvision
        video_tensor = video_tensor.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        video_tensor = video_tensor.squeeze(0)  # [T, H, W, C]
        video_tensor = (video_tensor * 255).to(torch.uint8)
        try:
            torchvision.io.write_video(output_path, video_tensor, fps=fps, video_codec='h264', options={'crf': '0'})
            print(f"Successfully saved video using torchvision.io.write_video to {output_path}")
            return True
        except TypeError as e:
            print(f"Torchvision writer failed with TypeError: {e}, trying alternative method...")
            
        # Second attempt: Try using PIL and moviepy
        try:

            # Create a temporary directory for storing frames
            temp_dir = tempfile.mkdtemp()
            try:
                # Save each frame as a PNG file
                frames = []
                for i in range(video_tensor.shape[0]):
                    frame = video_tensor[i].cpu().numpy()
                    img = Image.fromarray(frame)
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    img.save(frame_path)
                    frames.append(frame_path)
                
                # Create a video from the frames
                clip = ImageSequenceClip(frames, fps=fps)
                clip.write_videofile(output_path, codec="libx264", fps=fps)
                print(f"Successfully saved video using moviepy to {output_path}")
                return True
            finally:
                # Clean up the temporary directory
                shutil.rmtree(temp_dir)
                
        except ImportError:
            print("Moviepy not available, trying next method...")
        
        # Third attempt: Try using OpenCV
        try:
            import cv2
            import numpy as np
            
            video_tensor = video_tensor.cpu().numpy()
            height, width = video_tensor.shape[1], video_tensor.shape[2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'avc1' or other codecs
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for i in range(video_tensor.shape[0]):
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(video_tensor[i], cv2.COLOR_RGB2BGR)
                out.write(frame)
            
            out.release()
            print(f"Successfully saved video using OpenCV to {output_path}")
            return True
            
        except ImportError:
            print("OpenCV not available, no more fallback methods...")
            
        return False
        
    except Exception as e:
        print(f"All video saving methods failed with error: {e}")
        return False

def adjust_to_multiple(value, multiple):
    """Rounds down value to the nearest multiple."""
    return (value // multiple) * multiple

@torch.no_grad()
def generate_video(args, models):
    """Generates the video using the loaded models and arguments."""

    # Unpack models
    text_encoder = models["text_encoder"]
    text_encoder_2 = models["text_encoder_2"]
    tokenizer = models["tokenizer"]
    tokenizer_2 = models["tokenizer_2"]
    vae = models["vae"]
    feature_extractor = models["feature_extractor"]
    image_encoder = models["image_encoder"]
    transformer = models["transformer"]
    device = models["device"] # Use the device determined during loading

    # --- Determine Memory Mode ---
    if args.high_vram and args.low_vram:
        print("Warning: Both --high_vram and --low_vram specified. Defaulting to auto-detection.")
        force_high_vram = force_low_vram = False
    else:
        force_high_vram = args.high_vram
        force_low_vram = args.low_vram

    if force_high_vram:
        high_vram = True
    elif force_low_vram:
        high_vram = False
    else:
        # Use gpu directly if device is cuda, otherwise assume high_vram=False (cpu case)
        free_mem_gb = get_cuda_free_memory_gb(device) if device.type == 'cuda' else 0
        high_vram = free_mem_gb > 60 # Heuristic, adjust as needed
        print(f'Auto-detected Free VRAM {free_mem_gb:.2f} GB')

    print(f'High-VRAM Mode: {high_vram}')

    # --- Configure Models based on VRAM mode ---
    if not high_vram:
        print("Configuring for Low VRAM mode...")
        vae.enable_slicing()
        vae.enable_tiling()
        # DynamicSwapInstaller is like huggingface's enable_sequential_offload but faster
        print("  Installing DynamicSwap for Transformer...")
        DynamicSwapInstaller.install_model(transformer, device=device)
        print("  Installing DynamicSwap for Text Encoder 1...")
        DynamicSwapInstaller.install_model(text_encoder, device=device)
        # Smaller models might still be loaded fully or swapped as needed by other functions
    else:
        print("Configuring for High VRAM mode (moving models to GPU)...")
        text_encoder.to(device)
        text_encoder_2.to(device)
        image_encoder.to(device)
        vae.to(device)
        transformer.to(device)
        print("  Models moved to GPU.")

    # --- Prepare Inputs ---
    print("Preparing inputs...")
    prompt = args.prompt
    n_prompt = args.negative_prompt
    seed = args.seed
    total_second_length = args.total_second_length
    latent_window_size = args.latent_window_size
    steps = args.steps
    # Use args.cfg and args.gs directly
    cfg = args.cfg
    gs = args.distilled_guidance_scale
    rs = args.rs
    gpu_memory_preservation = args.gpu_memory_preservation
    use_teacache = args.use_teacache
    fps = args.fps

    # Calculate total latent sections (Mimics Gradio Demo)
    # Assumes internal 30fps for calculation, output fps is separate
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    print(f"Calculated total latent sections: {total_latent_sections}")

    job_id = generate_timestamp() + f"_seed{seed}"
    output_dir = Path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_video_path = None # Track the path of the final complete video

    try:
        # --- Initial Cleanup (Low VRAM) ---
        if not high_vram:
            print("Unloading models from GPU (Low VRAM setup)...")
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # --- Text Encoding ---
        print("Encoding text prompts...")
        if not high_vram:
            fake_diffusers_current_device(text_encoder, device)
            load_model_as_complete(text_encoder_2, target_device=device)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        # Handle negative prompt based on CFG scale (as in Gradio demo)
        if cfg == 1.0:
             print("  CFG scale is 1.0, using zero negative embeddings.")
             llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
             print(f"  Encoding negative prompt: '{n_prompt}'")
             llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        # Pad / Crop Embeddings
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        print("  Text encoded and processed.")

        if not high_vram: unload_complete_models(text_encoder_2) # Offload TE2 after use

        # --- Input Image Processing ---
        print("Processing input image and determining dimensions...")
        try:
            input_image = Image.open(args.input_image).convert('RGB')
            input_image_np = np.array(input_image) # Keep as numpy HWC for later
        except Exception as e:
            print(f"Error loading input image '{args.input_image}': {e}")
            raise
        H_orig, W_orig, _ = input_image_np.shape
        print(f"  Input image size: {W_orig}x{H_orig}")

        # --- Determine Target Dimensions ---
        if args.width is not None and args.height is not None:
            # User specified explicit dimensions
            target_w = args.width
            target_h = args.height
            print(f"  Using explicit target dimensions: {target_w}x{target_h}")
        elif args.target_resolution is not None:
            # User specified target for longer side, calculate based on aspect ratio
            print(f"  Calculating dimensions based on target resolution for longer side: {args.target_resolution}")
            # find_nearest_bucket calculates H, W based on target_resolution for the longer side
            target_h, target_w = find_nearest_bucket(H_orig, W_orig, resolution=args.target_resolution)
            print(f"  Calculated dimensions (before adjustment): {target_w}x{target_h}")
        else:
            # This should have been caught by parse_args, but as a failsafe:
            raise ValueError("Internal Error: Resolution determination failed. No target specified.")

        # --- Adjust Dimensions to be Divisible by DIMENSION_MULTIPLE ---
        final_w = adjust_to_multiple(target_w, DIMENSION_MULTIPLE)
        final_h = adjust_to_multiple(target_h, DIMENSION_MULTIPLE)

        if final_w <= 0 or final_h <= 0:
            print(f"Error: Calculated dimensions ({target_w}x{target_h}) resulted in non-positive dimensions after adjusting to be divisible by {DIMENSION_MULTIPLE} ({final_w}x{final_h}).")
            print(f"       Please check input image aspect ratio or target resolution/dimensions.")
            raise ValueError("Adjusted dimensions are invalid.")

        if final_w != target_w or final_h != target_h:
            print(f"Warning: Adjusted dimensions from {target_w}x{target_h} to {final_w}x{final_h} to be divisible by {DIMENSION_MULTIPLE}.")
        else:
            print(f"  Final dimensions confirmed: {final_w}x{final_h}")

        # Use final_w and final_h from now on
        width, height = final_w, final_h

        # Memory warning for large resolutions
        if width * height > 1024 * 1024: # Arbitrary threshold (e.g., > 1 megapixel)
             print(f"Warning: Target resolution {width}x{height} is large. Ensure you have sufficient VRAM, especially in high_vram mode.")

        # Resize input image
        input_image_resized_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)

        # Save input image for reference
        Image.fromarray(input_image_resized_np).save(output_dir / f'{job_id}_input_resized_{width}x{height}.png')

        # Prepare PT tensor for VAE: B, C, T, H, W, normalized
        input_image_pt = torch.from_numpy(input_image_resized_np).float() / 127.5 - 1.0
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        print(f"  Input image processed to tensor shape: {input_image_pt.shape}")


        # --- VAE Encoding (Initial Frame) ---
        print("VAE encoding initial frame...")
        if not high_vram:
            load_model_as_complete(vae, target_device=device)

        # Move input image tensor to the correct device and dtype for VAE
        input_image_pt = input_image_pt.to(device=vae.device, dtype=vae.dtype)
        start_latent = vae_encode(input_image_pt, vae) # B, C, 1, H/8, W/8
        print(f"  Initial latent shape: {start_latent.shape}")

        if not high_vram: unload_complete_models(vae)

        # --- CLIP Vision Encoding ---
        print("CLIP Vision encoding image...")
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=device)

        # Pass the *resized* numpy HWC array
        image_encoder_output = hf_clip_vision_encode(input_image_resized_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state # B, SeqLen, Dim
        print(f"  Image embedding shape: {image_encoder_last_hidden_state.shape}")

        if not high_vram: unload_complete_models(image_encoder)

        # --- Prepare Embeddings for Transformer ---
        print("Preparing embeddings for Transformer...")
        target_dtype = transformer.dtype
        # Ensure embeddings are on CPU before potential .to() calls if models aren't loaded yet
        llama_vec = llama_vec.cpu().to(target_dtype)
        llama_vec_n = llama_vec_n.cpu().to(target_dtype)
        clip_l_pooler = clip_l_pooler.cpu().to(target_dtype)
        clip_l_pooler_n = clip_l_pooler_n.cpu().to(target_dtype)
        # Image embeddings are already on CPU from the encoder output
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.cpu().to(target_dtype)
        print("  Embeddings converted to Transformer dtype on CPU.")


        # --- Sampling Setup ---
        print("Setting up sampling...")
        rnd = torch.Generator(cpu).manual_seed(seed) # Use CPU generator for reproducibility
        num_frames = latent_window_size * 4 - 3 # Latent frames to generate per sampling call (matches Gradio)
        print(f"  Latent frames per sampling step (num_frames input): {num_frames}")

        # History latents shape: B, C, T_hist_ctx + T_gen, H, W
        # T_hist_ctx includes context frames: 1 (initial) + 2 (2x down) + 16 (4x down) = 19
        # Initialize on CPU with float32 for accumulation (as in Gradio)
        latent_c, latent_h, latent_w = start_latent.shape[1], start_latent.shape[3], start_latent.shape[4]
        history_latents = torch.zeros(size=(1, latent_c, 1 + 2 + 16, latent_h, latent_w), dtype=torch.float32).cpu()
        history_pixels = None # Will store the BCTHW tensor of decoded pixels on CPU
        total_generated_latent_frames = 0 # Tracks length of the *generated* part in history_latents

        # Determine padding sequence (controls how the video is extended, mimics Gradio)
        latent_paddings = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
             # This trick from Gradio demo seems important for longer videos
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            print(f"  Using adjusted padding sequence for >4 sections: {latent_paddings}")
        else:
            print(f"  Using standard padding sequence: {latent_paddings}")


        # --- Main Sampling Loop (Mirrors Gradio `worker` loop) ---
        start_time = time.time()

        for i, latent_padding in enumerate(latent_paddings):
            section_start_time = time.time()
            is_last_section = latent_padding == 0
            # latent_padding_size is number of blank latent frames before the window
            latent_padding_size = latent_padding * latent_window_size

            print(f"\n--- Starting Section {i+1}/{len(latent_paddings)} (Padding: {latent_padding}, Last: {is_last_section}) ---")
            print(f'  Padding size (latent frames): {latent_padding_size}, Window size (latent frames): {latent_window_size}')

            # Define indices for sample_hunyuan (determines which latents are condition, which are generated)
            # Structure: [clean_pre(1), blank(pad), latent(win), clean_post(1), clean_2x(2), clean_4x(16)]
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = \
                indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            # Prepare conditioning latents from history (on CPU, will be moved to device later)
            # Uses the single initial frame latent
            clean_latents_pre = start_latent.cpu().to(history_latents.dtype)
            # Takes the *first* 1+2+16 entries from history (which are the context from the *previous* step)
            clean_latents_post, clean_latents_2x, clean_latents_4x = \
                history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            print(f"  Conditioning latent shapes (on CPU): clean={clean_latents.shape}, 2x={clean_latents_2x.shape}, 4x={clean_latents_4x.shape}")


            # Load Transformer (Low VRAM)
            if not high_vram:
                print("  Moving Transformer to GPU...")
                unload_complete_models() # Clear space just in case
                move_model_to_device_with_memory_preservation(transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation)

            # Configure TeaCache
            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                print("  TeaCache enabled.")
            else:
                transformer.initialize_teacache(enable_teacache=False)
                print("  TeaCache disabled.")

            # --- Run Sampling ---
            print(f"  Starting sampling ({steps} steps) for {num_frames} latent frames...")
            sampling_step_start_time = time.time()

            # Define a simple callback for TQDM progress
            pbar = tqdm(total=steps, desc=f"  Section {i+1} Sampling", leave=False)
            def callback(d):
                pbar.update(1)
                # preview = d['denoised'] # Could add fake VAE decode here for visual preview
                return

            # Ensure all inputs are on the correct device and dtype for the transformer
            current_device = transformer.device # Device transformer is actually on (could be meta if swapped)
            current_dtype = transformer.dtype

            # Move necessary tensors to the sampling device just before the call
            _prompt_embeds = llama_vec.to(current_device, current_dtype)
            _prompt_embeds_mask = llama_attention_mask.to(current_device)
            _prompt_poolers = clip_l_pooler.to(current_device, current_dtype)
            _negative_prompt_embeds = llama_vec_n.to(current_device, current_dtype)
            _negative_prompt_embeds_mask = llama_attention_mask_n.to(current_device)
            _negative_prompt_poolers = clip_l_pooler_n.to(current_device, current_dtype)
            _image_embeddings = image_encoder_last_hidden_state.to(current_device, current_dtype)
            _latent_indices = latent_indices.to(current_device)
            _clean_latents = clean_latents.to(current_device, current_dtype)
            _clean_latent_indices = clean_latent_indices.to(current_device)
            _clean_latents_2x = clean_latents_2x.to(current_device, current_dtype)
            _clean_latent_2x_indices = clean_latent_2x_indices.to(current_device)
            _clean_latents_4x = clean_latents_4x.to(current_device, current_dtype)
            _clean_latent_4x_indices = clean_latent_4x_indices.to(current_device)


            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width, # Use the final adjusted width
                height=height, # Use the final adjusted height
                frames=num_frames, # Target number of latent frames for this call
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=_prompt_embeds,
                prompt_embeds_mask=_prompt_embeds_mask,
                prompt_poolers=_prompt_poolers,
                negative_prompt_embeds=_negative_prompt_embeds,
                negative_prompt_embeds_mask=_negative_prompt_embeds_mask,
                negative_prompt_poolers=_negative_prompt_poolers,
                device=current_device, # Explicitly pass transformer's device
                dtype=current_dtype,  # Explicitly pass transformer's dtype
                image_embeddings=_image_embeddings,
                latent_indices=_latent_indices,
                clean_latents=_clean_latents,
                clean_latent_indices=_clean_latent_indices,
                clean_latents_2x=_clean_latents_2x,
                clean_latent_2x_indices=_clean_latent_2x_indices,
                clean_latents_4x=_clean_latents_4x,
                clean_latent_4x_indices=_clean_latent_4x_indices,
                callback=callback,
                # No callback_steps here, sample_hunyuan handles it internally
            )
            pbar.close()
            sampling_step_end_time = time.time()
            print(f"  Sampling finished in {sampling_step_end_time - sampling_step_start_time:.2f} seconds.")
            print(f"  Raw generated latent shape for this section: {generated_latents.shape}") # B, C, T=num_frames, H, W

            # --- History Update (Crucial Step) ---
            # Move generated latents to CPU/float32 for history storage
            generated_latents_cpu = generated_latents.cpu().float()

            # Prepend start latent ONLY for the very last section's output before adding to history
            if is_last_section:
                print("  Prepending initial frame latent to the final generated section.")
                generated_latents_cpu = torch.cat([start_latent.cpu().float(), generated_latents_cpu], dim=2)

            # Get the number of *new* frames added in this step (accounts for prepended start_latent)
            new_latent_frames_count = generated_latents_cpu.shape[2]

            # Prepend the newly generated (and potentially start_latent) frames to the history buffer
            # history_latents always stores newest -> oldest
            history_latents = torch.cat([generated_latents_cpu, history_latents], dim=2)

            # Update the total count of *purely generated* latent frames (excluding start_latent and context)
            # Let's use the Gradio approach: total_generated_latent_frames tracks the useful length in history_latents
            total_generated_latent_frames += int(generated_latents.shape[2]) # Add count *before* potential start_latent prepend
            print(f"  Latent history buffer updated. Total generated latent frames tracked: {total_generated_latent_frames}")
            print(f"  Current full history_latents shape (on CPU): {history_latents.shape}") # Includes context


            # --- VAE Decoding & Merging (Crucial Step - Mirrors Gradio) ---
            print("  Decoding generated latents and merging video...")
            decode_start_time = time.time()

            # Load VAE (Low VRAM)
            if not high_vram:
                print("    Moving VAE to GPU...")
                # Offload transformer first (important!)
                offload_model_from_device_for_memory_preservation(transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation if high_vram else 8.0)
                load_model_as_complete(vae, target_device=device)

            # Select the relevant portion of the *accumulated* history for decoding.
            # `history_latents` has newest frames first. total_generated_latent_frames tracks the length of the generated part.
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
            print(f"    Decoding based on real_history_latents shape: {real_history_latents.shape}")

            # Move the relevant slice to the VAE device
            real_history_latents_device = real_history_latents.to(device=vae.device, dtype=vae.dtype)

            if history_pixels is None:
                 # First section: Decode the whole relevant history generated so far
                 print(f"    First section, decoding all {real_history_latents_device.shape[2]} latent frames.")
                 history_pixels = vae_decode(real_history_latents_device, vae).cpu() # Decode and move to CPU
                 print(f"    Initial decoded pixel shape: {history_pixels.shape}")
            else:
                 # Subsequent sections: Decode only the newly relevant part and soft-append
                 # Calculate how many latent frames to decode for the current append operation (matches Gradio)
                 section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                 # Calculate the pixel overlap expected by soft_append (matches Gradio)
                 overlapped_frames = latent_window_size * 4 - 3

                 print(f"    Appending section. Decoding {section_latent_frames} latent frames from history start.")
                 print(f"    Using pixel overlap: {overlapped_frames}")

                 # Decode only the required slice from the *start* of real_history_latents
                 current_latents_to_decode = real_history_latents_device[:, :, :section_latent_frames, :, :]
                 print(f"    Decoding current section latents of shape: {current_latents_to_decode.shape}")
                 current_pixels = vae_decode(current_latents_to_decode, vae).cpu() # Decode and move to CPU
                 print(f"    Current decoded pixel section shape: {current_pixels.shape}")

                 # Append the new pixels smoothly onto the beginning of the existing video (on CPU)
                 # `soft_append_bcthw` expects (new_chunk, old_video, overlap)
                 history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlap=overlapped_frames)
                 print(f"    Appended. New total pixel shape: {history_pixels.shape}")


            # Unload VAE (Low VRAM)
            if not high_vram:
                unload_complete_models(vae)

            decode_end_time = time.time()
            print(f"  Decoding and merging finished in {decode_end_time - decode_start_time:.2f} seconds.")

            # --- Save Intermediate Video ---
            current_num_pixel_frames = history_pixels.shape[2]
            # Keep the filename generation that includes width/height
            output_filename = output_dir / f'{job_id}_section{i+1}_frames{current_num_pixel_frames}_{width}x{height}.mp4'
            print(f"  Saving intermediate video ({current_num_pixel_frames} frames) to: {output_filename}")
            
            try:
                # Use the original saving function
                save_bcthw_as_mp4(history_pixels, str(output_filename), fps=fps)
                print(f"  Saved video using save_bcthw_as_mp4")
            except Exception as e:
                print(f"  Error saving video using save_bcthw_as_mp4: {e}")
                # Optional: Add minimal fallback like saving the first frame if needed
                try:
                    first_frame = history_pixels[0, :, 0].permute(1, 2, 0).cpu().numpy()
                    first_frame = (first_frame * 0.5 + 0.5) * 255 # Denormalize
                    first_frame = first_frame.astype(np.uint8)
                    frame_path = str(output_filename).replace('.mp4', '_first_frame.png')
                    Image.fromarray(first_frame).save(frame_path)
                    print(f"  Saved first frame as image to {frame_path} due to video saving error.")
                except Exception as frame_err:
                    print(f"  Could not save first frame either: {frame_err}")
            
            final_video_path = str(output_filename) # Update the path to the latest complete video

            section_end_time = time.time()
            print(f"--- Section {i+1} finished in {section_end_time - section_start_time:.2f} seconds ---")

            # Exit loop after the last section is processed
            if is_last_section:
                print("\nLast section completed.")
                break

        # --- Final Report ---
        end_time = time.time()
        print(f"\nVideo generation finished in {end_time - start_time:.2f} seconds.")
        if final_video_path and os.path.exists(final_video_path):
            # Optionally rename the last saved file to something simpler
            final_output_name = output_dir / f"{job_id}_final_{width}x{height}.mp4"
            try:
                os.rename(final_video_path, final_output_name)
                print(f"Renamed final video to: {final_output_name}")
                print(f"ACTUAL_FINAL_PATH:{final_output_name}") # Distinct print for potential parsing
                return str(final_output_name)
            except OSError as e:
                print(f"Could not rename final video: {e}")
                # If rename fails, still report the intermediate path as the final one
                print(f"ACTUAL_FINAL_PATH:{final_video_path}") # Distinct print for potential parsing
                return final_video_path # Return original name if rename fails
        else:
            print("Error: No final video file was found.")
            return None

    except Exception as e:
        print("\n--- ERROR DURING GENERATION ---")
        traceback.print_exc()
        print("-----------------------------")
        return None # Indicate failure
    finally:
        # Final cleanup regardless of success/failure
        print("Performing final model cleanup...")
        unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        # Clean CUDA cache if possible
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")


def main():
    args = parse_args()
    models = load_models(args)
    final_path = generate_video(args, models)
    if final_path:
        print(f"\nSuccessfully generated: {final_path}")
    else:
        print("\nVideo generation failed.")


if __name__ == "__main__":
    main()
