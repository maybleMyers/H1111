#!/bin/bash

# Launch script for dual GPU sequence parallelism using torchrun
# This script runs the wan2_generate_video.py with distributed PyTorch

# Check if required arguments are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <prompt> [additional_args...]"
    echo "Example: $0 'A beautiful sunset over mountains' --output output.mp4"
    exit 1
fi

# Extract the prompt (first argument)
PROMPT="$1"
shift  # Remove the first argument so $@ contains only additional args

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Optional: Set NCCL debug level for troubleshooting
# export NCCL_DEBUG=INFO

echo "Launching with sequence parallelism on GPUs 0 and 1..."
echo "Prompt: $PROMPT"
echo "Additional arguments: $@"

# Launch with torchrun for 2 processes (one per GPU)
torchrun \
    --nproc_per_node=2 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    wan2_generate_video.py \
    --use_sequence_parallel \
    --prompt "$PROMPT" \
    $@

echo "Generation complete!"