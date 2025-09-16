#!/bin/bash
# Model directory
MODEL_DIR="<path_to_Pusa-V0.5_dir>"

# Checkpoint paths
CHECKPOINTS=(
    "<path_to_pusa_dit_checkpoint>"
)

num_steps=30
# Define multiple conditioning inputs
MULTI_COND='<your_multi_cond_here>' # Format: {position: [image_path, prompt_path, noise_multiplier], ...}
# e.g., Start&End Frames Generation: MULTI_COND='{
#   "0": ["./demos/example3.jpg", 0.3],
#   "20": ["./demos/example4.jpg", 0.7]
# }'
prompt="<your_prompt_here>" 

# Loop through each checkpoint and run the command with multiple conditioning
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 ./demos/cli_test_multi_frames_release.py \
    --model_dir "$MODEL_DIR" \
    --prompt "$prompt" \
    --dit_path "$CHECKPOINT" \
    --multi_cond "$MULTI_COND" \
    --num_steps "$num_steps" \
done
