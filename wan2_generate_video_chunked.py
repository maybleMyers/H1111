"""
Extended wan2_generate_video.py with support for chunked video generation and APG.
This module adds MultiTalk-style long video generation capabilities to Wan 2.2.
"""

import torch
import logging
from typing import Optional, List, Tuple, Dict, Any
from tqdm import tqdm
import gc

# Import APG utilities
from wan.utils.apg_utils import (
    MomentumBuffer,
    ChunkedVideoGenerator,
    apply_apg_to_cfg,
    adaptive_projected_guidance
)

logger = logging.getLogger(__name__)


def run_sampling_chunked(
    scheduler,
    timesteps,
    latent: torch.Tensor,
    model,
    arg_c: dict,
    arg_null: dict,
    guidance_scale: float,
    device: torch.device,
    latent_storage_device: torch.device,
    accelerator,
    seed_g=None,
    previewer=None,
    preview_suffix="",
    is_ti2v: bool = False,
    model_manager=None,
    args=None,
    # Chunked generation parameters
    frames_per_chunk: int = 25,
    motion_frames: int = 5,
    use_apg: bool = False,
    apg_momentum: float = -0.75,
    apg_norm_threshold: float = 55.0,
    total_frames: Optional[int] = None,
    progress_callback=None
) -> torch.Tensor:
    """
    Run sampling with support for chunked video generation and APG.
    
    Args:
        Standard sampling arguments from wan2_generate_video.py
        frames_per_chunk: Number of frames to generate per chunk
        motion_frames: Number of overlapping frames between chunks
        use_apg: Whether to use Adaptive Projected Guidance
        apg_momentum: Momentum value for APG
        apg_norm_threshold: Norm threshold for APG
        total_frames: Total number of frames to generate (if None, uses latent shape)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Final denoised latent tensor
    """
    
    # Initialize chunked generator
    chunk_gen = ChunkedVideoGenerator(
        frames_per_chunk=frames_per_chunk,
        motion_frames=motion_frames,
        blend_overlap=True
    )
    
    # Get total frames from latent if not specified
    if total_frames is None:
        # Latent shape is [B, C, F, H, W] or [C, F, H, W]
        if len(latent.shape) == 5:
            total_frames = latent.shape[2]
        else:
            total_frames = latent.shape[1]
    
    # Get chunk parameters
    chunks = chunk_gen.get_chunk_params(total_frames)
    logger.info(f"Generating {total_frames} frames in {len(chunks)} chunks")
    
    # Initialize APG momentum buffers if needed
    momentum_buffer = None
    if use_apg:
        momentum_buffer = MomentumBuffer(momentum=apg_momentum)
        logger.info(f"APG enabled with momentum={apg_momentum}, norm_threshold={apg_norm_threshold}")
    
    all_generated_frames = []
    
    for chunk_idx, (start_frame, end_frame, cond_frames) in enumerate(chunks):
        logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)}: frames {start_frame}-{end_frame}")
        
        # Extract chunk from initial noise
        if len(latent.shape) == 5:
            chunk_latent = latent[:, :, start_frame:end_frame].clone()
        else:
            chunk_latent = latent[:, start_frame:end_frame].clone()
        
        # Apply conditioning from previous chunk if not first chunk
        if chunk_idx > 0 and cond_frames > 0:
            conditioning_frames = chunk_gen.get_conditioning_frames()
            if conditioning_frames is not None:
                # Add noise to conditioning frames for next chunk
                # This follows MultiTalk's approach
                noise_level = timesteps[0]  # Use first timestep noise level
                cond_noise = torch.randn_like(conditioning_frames).to(device)
                noisy_cond = scheduler.add_noise(conditioning_frames, cond_noise, noise_level)
                
                # Replace first motion_frames with noisy conditioning
                if len(chunk_latent.shape) == 5:
                    chunk_latent[:, :, :cond_frames] = noisy_cond.unsqueeze(0)
                else:
                    chunk_latent[:, :cond_frames] = noisy_cond
        
        # Run denoising for this chunk
        chunk_result = run_sampling_chunk_internal(
            scheduler=scheduler,
            timesteps=timesteps,
            latent=chunk_latent,
            model=model,
            arg_c=arg_c,
            arg_null=arg_null,
            guidance_scale=guidance_scale,
            device=device,
            latent_storage_device=latent_storage_device,
            accelerator=accelerator,
            seed_g=seed_g,
            is_ti2v=is_ti2v,
            model_manager=model_manager,
            args=args,
            momentum_buffer=momentum_buffer,
            use_apg=use_apg,
            apg_norm_threshold=apg_norm_threshold,
            chunk_idx=chunk_idx,
            total_chunks=len(chunks)
        )
        
        # Add chunk to generator
        if len(chunk_result.shape) == 5:
            chunk_gen.add_chunk(chunk_result.squeeze(0), is_first=(chunk_idx == 0))
        else:
            chunk_gen.add_chunk(chunk_result, is_first=(chunk_idx == 0))
        
        # Clear GPU cache between chunks
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Progress callback with detailed info
        if progress_callback:
            chunk_progress = {
                'current_chunk': chunk_idx + 1,
                'total_chunks': len(chunks),
                'frames_processed': end_frame,
                'total_frames': total_frames,
                'chunk_range': f"{start_frame}-{end_frame}"
            }
            progress_callback(chunk_progress)
    
    # Get final combined video
    final_latent = chunk_gen.generated_frames
    
    # Add batch dimension back if needed
    if len(latent.shape) == 5 and len(final_latent.shape) == 4:
        final_latent = final_latent.unsqueeze(0)
    
    return final_latent.to(latent_storage_device)


def run_sampling_chunk_internal(
    scheduler,
    timesteps,
    latent: torch.Tensor,
    model,
    arg_c: dict,
    arg_null: dict,
    guidance_scale: float,
    device: torch.device,
    latent_storage_device: torch.device,
    accelerator,
    seed_g=None,
    is_ti2v: bool = False,
    model_manager=None,
    args=None,
    momentum_buffer: Optional[MomentumBuffer] = None,
    use_apg: bool = False,
    apg_norm_threshold: float = 55.0,
    chunk_idx: int = 0,
    total_chunks: int = 1
) -> torch.Tensor:
    """
    Internal function to run sampling for a single chunk.
    Based on the original run_sampling function but with APG support.
    """
    
    num_timesteps = len(timesteps)
    
    # Setup CFG skip array (simplified, can be extended with args.cfg_skip_mode support)
    apply_cfg_array = [True] * num_timesteps
    
    logger.info(f"Chunk {chunk_idx + 1}/{total_chunks}: Starting sampling loop for {num_timesteps} steps")
    
    for i, t in enumerate(tqdm(timesteps, desc=f"Chunk {chunk_idx + 1}")):
        # Prepare input for the model
        latent_on_device = latent.to(device)
        
        # Fix dimensions if needed
        if len(latent_on_device.shape) > 5:
            while len(latent_on_device.shape) > 5:
                latent_on_device = latent_on_device.squeeze(0)
        
        # Prepare model input list
        if len(latent_on_device.shape) == 5:
            latent_model_input_list = [latent_on_device[i] for i in range(latent_on_device.shape[0])]
        elif len(latent_on_device.shape) == 4:
            latent_model_input_list = [latent_on_device]
        else:
            raise ValueError(f"Unexpected latent shape: {latent_on_device.shape}")
        
        # Prepare timestep
        timestep = torch.stack([t]).to(device)
        
        with accelerator.autocast(), torch.no_grad():
            # Select model for dual-dit if applicable
            if model_manager is not None:
                cfg = args.task if hasattr(args, 'task') else None
                if hasattr(args, 'dual_dit_boundary') and args.dual_dit_boundary is not None:
                    boundary = args.dual_dit_boundary * 1000
                else:
                    boundary = 875  # Default boundary
                
                if t.item() >= boundary:
                    current_model = model_manager.get_model('high')
                else:
                    current_model = model_manager.get_model('low')
            else:
                current_model = model
            
            # Filter model arguments
            model_arg_c = {k: v for k, v in arg_c.items() if not k.startswith('_')}
            model_arg_null = {k: v for k, v in arg_null.items() if not k.startswith('_')}
            
            # Predict conditional noise
            noise_pred_cond = current_model(latent_model_input_list, t=timestep, **model_arg_c)[0]
            noise_pred_cond = noise_pred_cond.to(latent_storage_device)
            
            # Apply CFG with optional APG
            apply_cfg = apply_cfg_array[i]
            if apply_cfg:
                # Predict unconditional noise
                noise_pred_uncond = current_model(latent_model_input_list, t=timestep, **model_arg_null)[0]
                noise_pred_uncond = noise_pred_uncond.to(latent_storage_device)
                
                if use_apg:
                    # Apply APG-modified CFG
                    noise_pred = apply_apg_to_cfg(
                        noise_pred_cond=noise_pred_cond,
                        noise_pred_uncond=noise_pred_uncond,
                        guidance_scale=guidance_scale,
                        momentum_buffer=momentum_buffer,
                        norm_threshold=apg_norm_threshold,
                        eta=0.0,
                        verbose=(i % 10 == 0)  # Print debug info every 10 steps
                    )
                else:
                    # Standard CFG
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond
            
            # Fix dimensions for scheduler
            if len(noise_pred.shape) < len(latent_on_device.shape):
                noise_pred = noise_pred.unsqueeze(0)
            
            # Compute previous sample
            scheduler_output = scheduler.step(
                noise_pred.to(device),
                t,
                latent_on_device,
                return_dict=False,
                generator=seed_g
            )
            prev_latent = scheduler_output[0]
            
            # Update latent state
            latent = prev_latent.to(latent_storage_device)
            
            # Apply TI2V conditioning if needed
            if "_image_latent" in arg_c and "_ti2v_mask2" in arg_c:
                image_latent = arg_c["_image_latent"].to(latent_storage_device)
                ti2v_mask2 = arg_c["_ti2v_mask2"]
                latent = (1. - ti2v_mask2[0]) * image_latent + ti2v_mask2[0] * latent
    
    logger.info(f"Chunk {chunk_idx + 1}/{total_chunks}: Sampling complete")
    return latent


def generate_with_chunks(
    args,
    model,
    vae,
    text_encoder,
    clip=None,
    initial_latent=None,
    scheduler=None,
    timesteps=None,
    arg_c=None,
    arg_null=None,
    device=None,
    accelerator=None,
    model_manager=None,
    # Chunking parameters
    enable_chunking: bool = False,
    frames_per_chunk: int = 25,
    motion_frames: int = 5,
    use_apg: bool = False,
    apg_momentum: float = -0.75,
    apg_norm_threshold: float = 55.0,
    progress_callback=None
) -> torch.Tensor:
    """
    Wrapper function to enable chunked generation in the existing pipeline.
    
    This function can be called from wan2_generate_video.py's generate() function
    when chunking is enabled.
    """
    
    if not enable_chunking:
        # Fall back to original sampling if chunking disabled
        logger.info("Chunking disabled, using standard sampling")
        from wan2_generate_video import run_sampling
        return run_sampling(
            scheduler=scheduler,
            timesteps=timesteps,
            latent=initial_latent,
            model=model,
            arg_c=arg_c,
            arg_null=arg_null,
            guidance_scale=args.guidance_scale,
            device=device,
            latent_storage_device=device,
            accelerator=accelerator,
            model_manager=model_manager,
            args=args
        )
    
    # Use chunked sampling
    logger.info(f"Using chunked sampling: frames_per_chunk={frames_per_chunk}, motion_frames={motion_frames}")
    
    return run_sampling_chunked(
        scheduler=scheduler,
        timesteps=timesteps,
        latent=initial_latent,
        model=model,
        arg_c=arg_c,
        arg_null=arg_null,
        guidance_scale=args.guidance_scale,
        device=device,
        latent_storage_device=device,
        accelerator=accelerator,
        seed_g=None,
        previewer=None,
        preview_suffix="",
        is_ti2v=hasattr(args, 'task') and 'ti2v' in args.task,
        model_manager=model_manager,
        args=args,
        frames_per_chunk=frames_per_chunk,
        motion_frames=motion_frames,
        use_apg=use_apg,
        apg_momentum=apg_momentum,
        apg_norm_threshold=apg_norm_threshold,
        progress_callback=progress_callback
    )


# Export functions for use in wan2_generate_video.py
__all__ = [
    'run_sampling_chunked',
    'generate_with_chunks'
]