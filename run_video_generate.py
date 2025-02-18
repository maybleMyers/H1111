#!/usr/bin/env python3

import sys
import subprocess
import random

def main():
    """
    Usage:
      python run_hv_generate_video.py "Your prompt text here"
    """
    if len(sys.argv) < 2:
        print("Error: No prompt provided.")
        print("Usage: python run_hv_generate_video.py \"<prompt>\"")
        #sys.exit(1)

    # Capture the prompt from command-line arguments
    #prompt = sys.argv[1]
    SkyReelsModel = "Skywork/SkyReels-V1-Hunyuan-I2V"
        # Generate a random seed
    random_seed = random.randint(0, 2**32 - 1)
 # Construct the command
    cmd = [
        # quant: Enable FP8 weight-only quantization
        # offload: Enable offload model
        # high_cpu_memory: Enable pinned memory to reduce the overhead of model offloading.
        # parameters_level: Further reduce GPU VRAM usage.
        "python3", "video_generate.py",
            "--model_id", SkyReelsModel,
            "--guidance_scale", "6.0",
            "--height", "720",
            "--width", "720",
            "--num_frames", "97",
            "--prompt", "FPS-24, In a serene scene along a detailed oceanfront, a feral female alicorn Twilight Sparkle from My Little Pony stands alone. The waves crash against the shore, splashing her face with salty water, as she gazes out at the vast, indifferent sea. ",
            "--embedded_guidance_scale", "1.0",
            "--quant",
            "--offload",
            "--high_cpu_memory",
            "--parameters_level",
            "--image", "img/ocean.webp",
            "--seed", str(random_seed),
            "--task_type", "i2v"
    ]
 # Print the exact command (for debugging/logging)
    print("Executing command with random seed:", random_seed)
    print(" ".join(cmd))

    # Run the command
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
