#!/bin/bash
# Model directory
MODEL_DIR="<path_to_Pusa-V0.5_dir>"

# Checkpoint paths
CHECKPOINTS=(
    "<path_to_pusa_dit_checkpoint>"
)

# For example
prompt="A fluffy Cockapoo, perched atop a vibrant pink flamingo jumps into a crystal-clear pool."
video_start_dir="./demos/example1.mp4"
video_end_dir="./demos/example2.mp4"
cond_position_start="[0]"
cond_position_end="[-3,-2,-1]"
noise_multiplier="[0.1,0.8,0.8,0.8]"
num_steps=30

# Loop through each checkpoint and run the command
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 ./demos/cli_test_transition_release.py --model_dir "$MODEL_DIR" --dit_path "$CHECKPOINT" --prompt "$prompt" --prompt_dir "$prompt_dir" --video_start_dir "$video_start_dir" --video_end_dir "$video_end_dir" --cond_position_start "$cond_position_start" --cond_position_end "$cond_position_end" --noise_multiplier "$noise_multiplier" --num_steps "$num_steps"
done


