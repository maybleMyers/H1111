import os
import numpy as np
from PIL import Image
import ffmpeg
import argparse

def add_noise(image: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Add Gaussian noise to an image with increasing intensity.
    
    Args:
        image: Input image as numpy array (0-255)
        noise_level: Amount of noise to add (0-1)
    """
    # Convert to float for calculations
    img_float = image.astype(float)
    
    # Generate noise
    noise = np.random.normal(0, 255 * noise_level, image.shape)
    
    # Add noise to image
    noisy_image = img_float + noise
    
    # Clip values to valid range
    noisy_image = np.clip(noisy_image, 0, 255)
    
    return noisy_image.astype(np.uint8)

def create_noise_sequence(input_image_path: str, output_folder: str, num_frames: int = 201):
    """
    Create a sequence of increasingly noisy images.
    
    Args:
        input_image_path: Path to the input image
        output_folder: Folder to save the sequence
        num_frames: Number of frames to generate
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the image
    image = Image.open(input_image_path)
    image_array = np.array(image)
    
    # Calculate noise levels
    # First 5 frames are clean, then noise increases to 0.25
    noise_levels = np.zeros(num_frames)
    if num_frames > 5:
        noise_levels[5:] = np.linspace(0, 0.50, num_frames - 5)
    
    # Generate and save frames
    for i, noise_level in enumerate(noise_levels):
        # First 5 frames are the original image
        if i < 5:
            noisy_image = image_array
        else:
            noisy_image = add_noise(image_array, noise_level)
            
        # Save frame
        output_path = os.path.join(output_folder, f"frame_{i:03d}.png")
        Image.fromarray(noisy_image).save(output_path)
        
        # Print progress
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{num_frames} frames")

def create_video(image_folder: str, output_path: str, fps: int = 24):
    """
    Create a video from a sequence of images using ffmpeg command line.
    
    Args:
        image_folder: Folder containing the image sequence
        output_path: Path for the output video
        fps: Frames per second for the video
    """
    input_pattern = os.path.join(image_folder, 'frame_%03d.png')
    
    # Construct ffmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-framerate', str(fps),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        '-crf', '23',
        output_path
    ]
    
    # Run ffmpeg command
    import subprocess
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e.stderr.decode()}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate a sequence of increasingly noisy images and create a video")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("--output_folder", default="noise_sequence", help="Output folder for image sequence")
    parser.add_argument("--output_video", default="noise_sequence.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the video")
    parser.add_argument("--frames", type=int, default=201, help="Number of frames to generate")
    
    args = parser.parse_args()
    
    print("Generating noise sequence...")
    create_noise_sequence(args.input_image, args.output_folder, args.frames)
    
    print("Creating video...")
    create_video(args.output_folder, args.output_video, args.fps)
    
    print(f"Video saved to {args.output_video}")

if __name__ == "__main__":
    main()