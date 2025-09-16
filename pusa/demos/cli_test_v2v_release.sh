#!/bin/bash

# Model directory
MODEL_DIR="<path_to_Pusa-V0.5_dir>"

# Checkpoint paths
CHECKPOINTS=(
    "<path_to_pusa_dit_checkpoint>"
)


prompt="A cinematic shot captures a fluffy Cockapoo, perched atop a vibrant pink flamingo float, in a sun-drenched Los Angeles swimming pool. "
video_dir="./demos/example1.mp4"
cond_position="[0,1,2,3]"
noise_multiplier="[0.3,0.6,0.6,0.6]"
num_steps=10

# Loop through each checkpoint and run the command
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 ./demos/cli_test_extension_release.py \
        --model_dir "$MODEL_DIR" \
        --dit_path "$CHECKPOINT" \
        --prompt "$prompt" \
        --video_dir "$video_dir" \
        --cond_position "$cond_position" \
        --num_steps "$num_steps" \
        --noise_multiplier "$noise_multiplier" \
done
