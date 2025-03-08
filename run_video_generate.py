import sys
import subprocess
import random
import os

def main():
    # Hardcoded default values for Hunyuan I2V
    cmd = [
        sys.executable,
        "newhun_gen.py",
        "--i2v_mode",
        "--i2v_image_path", "boy.png",
        "--prompt", "A detailed, cinematic scene showcasing a vibrant landscape",
        "--dit", "hunyuan/I2Vmp_rank_00_model_states.pt",
        "--vae", "hunyuan/I2Vpytorch_model.pt",
        "--text_encoder1", "hunyuan/llava_llama3_fp16.safetensors",
        "--text_encoder2", "hunyuan/clip_l.safetensors",
        #"--video_size", "544", "544",
        "--video_length", "977",
        "--fps", "24",
        "--infer_steps", "30",
        "--save_path", "outputs",
        "--seed", str(random.randint(0, 2**32 - 1)),
        "--flow_shift", "5.0",
        "--guidance_scale", "7.0",
        "--embedded_cfg_scale", "1.0",
        "--output_type", "video",
        "--i2v_resolution", "360p",
        "--attn_mode", "sdpa",
        "--blocks_to_swap", "0",
        "--fp8"
    ]

    # Print the command for logging
    print("Executing video generation command:")
    print(" ".join(cmd))

    # Run the command
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
