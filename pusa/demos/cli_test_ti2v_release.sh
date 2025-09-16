#!/bin/bash
# Model directory
MODEL_DIR="<path_to_Pusa-V0.5_dir>"

# Checkpoint paths
CHECKPOINTS=(
    "<path_to_pusa_dit_checkpoint>"
)

# I2V - default conditioned on the first frame
image_dir="<path_to_input_image_dir>"
# prompt_dir=None
prompt="<your_prompt_here>"
# Strongly suggest to try different con_position here and also you can modify the level of noise add to the condition image, you may find some surprise
cond_position=0
num_steps=30
noise_multiplier=0.4

# Generate with 1 GPU, generated video will be splitted into 4 parts for decoding
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CUDA_VISIBLE_DEVICES=6 python3 ./demos/cli_test_ti2v_release.py --model_dir "$MODEL_DIR" --dit_path "$CHECKPOINT" --image_dir "$image_dir" --prompt "$prompt" --prompt_dir "$prompt_dir" --cond_position "$cond_position" --num_steps "$num_steps" --noise_multiplier "$noise_multiplier"
done

