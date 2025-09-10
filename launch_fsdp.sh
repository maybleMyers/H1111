#!/bin/bash
# Launch script for FSDP multi-GPU video generation
# This script runs wan2_generate_video.py with FSDP enabled for reduced VRAM usage

# Get the prompt from command line (default if not provided)
PROMPT="${1:-A beautiful sunset over mountains}"

# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29501
export NCCL_DEBUG=INFO

echo "========================================"
echo "Launching FSDP Video Generation"
echo "========================================"
echo "Prompt: $PROMPT"
echo "Using FSDP for memory-efficient multi-GPU inference"
echo "========================================"

# Run with torchrun for multi-GPU FSDP
torchrun --nproc_per_node=2 \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         wan2_generate_video.py \
         --dit_fsdp \
         --t5_fsdp \
         --fsdp_sharding_strategy FULL_SHARD \
         --fsdp_mixed_precision bf16 \
         --prompt "$PROMPT" \
         --output fsdp_output.mp4 \
         --num_frames 49 \
         --cfg_scale 3.0 \
         --seed 42 "${@:2}"

echo "========================================"
echo "FSDP generation complete!"
echo "Output saved to: fsdp_output.mp4"
echo "========================================"