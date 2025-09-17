# Pusa Utilities for Multi-Frame Conditioning and Video Processing
# Copyright 2024-2025 Implementation for wan2_generate_video.py

import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import logging


def process_conditioning_images(
    image_paths: List[str],
    target_width: int = 1280,
    target_height: int = 720
) -> List[Image.Image]:
    """
    Process conditioning images with proper aspect ratio handling.
    
    Args:
        image_paths: List of paths to conditioning images
        target_width: Target width for output images
        target_height: Target height for output images
        
    Returns:
        List of processed PIL Images
    """
    processed_images = []
    
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Conditioning image not found: {path}")
            
        # Load and convert to RGB
        img = Image.open(path).convert("RGB")
        original_w, original_h = img.size
        
        # Calculate aspect ratio preserving resize
        ratio = min(target_width / original_w, target_height / original_h)
        new_w = int(original_w * ratio)
        new_h = int(original_h * ratio)
        
        # Resize maintaining aspect ratio
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Create black background and center the image
        background = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        paste_x = (target_width - new_w) // 2
        paste_y = (target_height - new_h) // 2
        background.paste(img_resized, (paste_x, paste_y))
        
        processed_images.append(background)
        logging.info(f"Processed conditioning image: {path} -> {target_width}x{target_height}")
    
    return processed_images


def process_conditioning_video(
    video_path: str,
    target_width: int = 1280,
    target_height: int = 720
) -> List[Image.Image]:
    """
    Process conditioning video frames with proper aspect ratio handling.
    
    Args:
        video_path: Path to conditioning video
        target_width: Target width for output frames
        target_height: Target height for output frames
        
    Returns:
        List of processed PIL Images (frames)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Conditioning video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Get original video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate scaling and cropping parameters
    target_ratio = target_width / target_height
    original_ratio = width / height
    
    logging.info(f"Processing video: {video_path} ({width}x{height}, {frame_count} frames)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize maintaining aspect ratio with center crop
        if original_ratio > target_ratio:
            # Video is wider than target - crop width
            new_width = int(height * target_ratio)
            start_x = (width - new_width) // 2
            frame = frame[:, start_x:start_x + new_width]
        else:
            # Video is taller than target - crop height
            new_height = int(width / target_ratio)
            start_y = (height - new_height) // 2
            frame = frame[start_y:start_y + new_height]
        
        # Resize to target dimensions
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    
    cap.release()
    logging.info(f"Processed {len(frames)} video frames to {target_width}x{target_height}")
    return frames


def parse_conditioning_positions(cond_position_str: str) -> List[int]:
    """
    Parse conditioning position string to list of integers.
    
    Args:
        cond_position_str: Comma-separated position indices (e.g., "0,20,40")
        
    Returns:
        List of position indices
    """
    if not cond_position_str:
        return []
        
    try:
        positions = [int(x.strip()) for x in cond_position_str.split(',')]
        return positions
    except ValueError as e:
        raise ValueError(f"Invalid conditioning position format: {cond_position_str}. Expected comma-separated integers.") from e


def parse_noise_multipliers(noise_mult_str: str) -> List[float]:
    """
    Parse noise multipliers string to list of floats.
    
    Args:
        noise_mult_str: Comma-separated multiplier values (e.g., "0.0,0.4,0.6")
        
    Returns:
        List of noise multiplier values
    """
    if not noise_mult_str:
        return []
        
    try:
        multipliers = [float(x.strip()) for x in noise_mult_str.split(',')]
        return multipliers
    except ValueError as e:
        raise ValueError(f"Invalid noise multipliers format: {noise_mult_str}. Expected comma-separated floats.") from e


def create_conditioning_dict(
    positions: List[int],
    images: List[Image.Image],
    noise_multipliers: List[float]
) -> Dict[int, Tuple[Image.Image, float]]:
    """
    Create conditioning dictionary mapping positions to (image, noise_multiplier) pairs.
    
    Args:
        positions: List of frame positions for conditioning
        images: List of conditioning images
        noise_multipliers: List of noise multiplier values
        
    Returns:
        Dict mapping position to (image, noise_multiplier) tuple
    """
    if len(positions) != len(images):
        raise ValueError(f"Mismatch: {len(positions)} positions but {len(images)} images")
    if len(positions) != len(noise_multipliers):
        raise ValueError(f"Mismatch: {len(positions)} positions but {len(noise_multipliers)} noise multipliers")
    
    conditioning_dict = {}
    for pos, img, noise_mult in zip(positions, images, noise_multipliers):
        conditioning_dict[pos] = (img, noise_mult)
        logging.info(f"Conditioning frame {pos}: noise_multiplier={noise_mult}")
    
    return conditioning_dict


def validate_conditioning_parameters(
    num_frames: int,
    cond_positions: List[int],
    noise_multipliers: List[float],
    images: List[Image.Image]
) -> None:
    """
    Validate conditioning parameters for consistency.
    
    Args:
        num_frames: Total number of frames in the video
        cond_positions: List of conditioning frame positions
        noise_multipliers: List of noise multiplier values  
        images: List of conditioning images
        
    Raises:
        ValueError: If parameters are inconsistent
    """
    # Check position bounds
    for pos in cond_positions:
        if pos < 0 or pos >= num_frames:
            raise ValueError(f"Conditioning position {pos} is out of bounds for {num_frames} frames")
    
    # Check length consistency
    if len(cond_positions) != len(noise_multipliers):
        raise ValueError(f"Length mismatch: {len(cond_positions)} positions, {len(noise_multipliers)} multipliers")
    
    if len(cond_positions) != len(images):
        raise ValueError(f"Length mismatch: {len(cond_positions)} positions, {len(images)} images")
    
    # Check noise multiplier ranges (warn if outside typical range)
    for i, mult in enumerate(noise_multipliers):
        if mult < 0 or mult > 10:
            logging.warning(f"Noise multiplier {mult} at position {cond_positions[i]} is outside typical range [0, 10]")


def create_frame_noise_mapping(
    cond_positions: List[int],
    noise_multipliers: List[float]
) -> Dict[int, float]:
    """
    Create mapping from frame index to noise multiplier.
    
    Args:
        cond_positions: List of conditioning frame positions
        noise_multipliers: List of noise multiplier values
        
    Returns:
        Dict mapping frame index to noise multiplier
    """
    return dict(zip(cond_positions, noise_multipliers))


def log_conditioning_info(
    task_type: str,
    cond_positions: List[int],
    noise_multipliers: List[float],
    num_images: int,
    video_path: Optional[str] = None
) -> None:
    """
    Log conditioning configuration information.
    
    Args:
        task_type: Type of conditioning task (i2v, v2v, multi-frame)
        cond_positions: List of conditioning positions
        noise_multipliers: List of noise multipliers
        num_images: Number of conditioning images
        video_path: Path to conditioning video (if applicable)
    """
    logging.info(f"=== Pusa Conditioning Configuration ===")
    logging.info(f"Task Type: {task_type}")
    logging.info(f"Conditioning Positions: {cond_positions}")
    logging.info(f"Noise Multipliers: {noise_multipliers}")
    logging.info(f"Number of Images: {num_images}")
    if video_path:
        logging.info(f"Video Path: {video_path}")
    logging.info(f"=======================================")