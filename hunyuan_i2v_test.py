#!/usr/bin/env python3
import os
import sys
import logging
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_paths():
    """Set up paths to necessary model files"""
    base_dir = os.path.expanduser("~/diffusion/H1111/hunyuan")
    return {
        'dit_model': os.path.join(base_dir, 'I2Vmp_rank_00_model_states.pt'),
        'vae_model': os.path.join(base_dir, 'I2Vpytorch_model.pt'),
        'text_encoder1': os.path.join(base_dir, 'llava_llama3_fp16.safetensors'),
        'text_encoder2': os.path.join(base_dir, 'clip_l.safetensors')
    }

def preprocess_image(image_path, target_resolution=(960, 960)):
    """Preprocess image for I2V model"""
    try:
        # Open image
        image = Image.open(image_path).convert('RGB')
        logger.info(f"Original Image Size: {image.size}")

        # Resize and center crop
        transform = transforms.Compose([
            transforms.Resize(target_resolution),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # Process image
        processed_image = transform(image)
        
        # Add batch and temporal dimensions
        processed_image = processed_image.unsqueeze(0).unsqueeze(2)
        
        logger.info(f"Processed Image Shape: {processed_image.shape}")
        return processed_image
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise

def import_hunyuan_modules():
    """Import required Hunyuan modules"""
    try:
        from hunyuan_model.text_encoder import TextEncoder
        from hunyuan_model.vae import load_vae
        from hunyuan_model.models import load_transformer
        from hunyuan_model.modules import load_model
        from hunyuan_model.diffusion.schedulers import FlowMatchDiscreteScheduler
        from hunyuan_model.diffusion.pipelines import HunyuanVideoPipeline
        return locals()
    except ImportError as e:
        logger.error(f"Failed to import Hunyuan modules: {e}")
        raise

def encode_prompt(prompt, text_encoder, device):
    """Encode text prompt"""
    try:
        # Prepare inputs
        text_inputs = text_encoder.text2tokens(prompt, data_type="video")
        
        # Encode prompt
        with torch.no_grad():
            prompt_outputs = text_encoder.encode(text_inputs, data_type="video", device=device)
        
        return prompt_outputs.hidden_state
    except Exception as e:
        logger.error(f"Prompt encoding error: {e}")
        raise

def generate_i2v(image_path, prompt, model_paths):
    """Main Image-to-Video generation function"""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Import required modules
        modules = import_hunyuan_modules()
        
        # Preprocessed image
        image_tensor = preprocess_image(image_path)
        
        # Load VAE
        logger.info("Loading VAE...")
        vae, _, s_ratio, t_ratio = modules['load_vae'](
            vae_dtype=torch.float16, 
            device=device, 
            vae_path=model_paths['vae_model']
        )
        vae.eval()
        
        # Encode image to latents
        logger.info("Encoding image to latents...")
        image_tensor = image_tensor.to(device=device, dtype=torch.float16) * 2 - 1
        with torch.no_grad():
            img_latents = vae.encode(image_tensor).latent_dist.mode()
            img_latents = img_latents * vae.config.scaling_factor
        
        # Load text encoder
        logger.info("Loading text encoder...")
        text_encoder = modules['TextEncoder'](
            text_encoder_type="llm",
            max_length=256,
            text_encoder_dtype=torch.float16,
            text_encoder_path=model_paths['text_encoder1'],
            tokenizer_type="llm",
            i2v_mode=True
        )
        text_encoder.eval().to(device)
        
        # Encode prompt
        logger.info("Encoding prompt...")
        prompt_embeds = encode_prompt(prompt, text_encoder, device)
        
        # Load transformer model
        logger.info("Loading transformer model...")
        transformer = modules['load_transformer'](
            dit_path=model_paths['dit_model'], 
            attn_mode="sdpa", 
            split_attn=False, 
            device=device, 
            dtype=torch.bfloat16, 
            in_channels=33,  # Specific for I2V
            i2v_mode=True
        )
        transformer.eval()
        
        # Create scheduler
        logger.info("Setting up scheduler...")
        scheduler = modules['FlowMatchDiscreteScheduler'](
            shift=5.0,  # Typical flow shift for I2V
            reverse=True,
            solver="euler"
        )
        scheduler.set_timesteps(50, device=device)
        
        # Create pipeline
        logger.info("Creating video pipeline...")
        pipeline = modules['HunyuanVideoPipeline'](
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=None,  # Use None for I2V
            transformer=transformer,
            scheduler=scheduler
        )
        
        # Run generation
        logger.info("Generating video...")
        with torch.no_grad():
            output = pipeline(
                prompt=[prompt],
                height=960,
                width=960,
                video_length=25,
                num_inference_steps=50,
                guidance_scale=7.0,
                negative_prompt=["low quality, blurry"],
                i2v_mode=True,
                img_latents=img_latents
            )
        
        # Save video
        logger.info("Saving video...")
        from hunyuan_model.utils.file_utils import save_videos_grid
        save_videos_grid(output[0], "test_i2v_output.mp4", fps=24)
        
        logger.info("Video generation completed successfully!")
        return output[0]
    
    except Exception as e:
        logger.error(f"Video generation failed: {e}", exc_info=True)
        raise

def main():
    # Paths and configuration
    model_paths = setup_paths()
    
    # Example image and prompt
    image_path = input("Enter path to input image: ")
    prompt = input("Enter video generation prompt: ")
    
    # Generate video
    generate_i2v(image_path, prompt, model_paths)

if __name__ == "__main__":
    main()