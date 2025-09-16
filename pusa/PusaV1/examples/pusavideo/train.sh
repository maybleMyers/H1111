#!/bin/bash
# This is an example script for training on a single node with multiple GPUs.
# You may need to modify this script to fit your environment.

# Define paths and parameters
MODEL_DIR="./model_zoo/PusaV1/Wan2.1-T2V-14B"
DATASET_PATH="./dataset/PusaV1_training" # TODO: Or change this to your own dataset
OUTPUT_PATH="./outputs/my_lora_model" # TODO: Change this to your desired output path

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python examples/pusavideo/train_wan_pusa.py \
  --task train \
  --lora_rank 512 \
  --lora_alpha 512 \
  --train_architecture lora \
  --dataset_path "${DATASET_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --dit_path "${MODEL_DIR}/diffusion_pytorch_model-00001-of-00006.safetensors,${MODEL_DIR}/diffusion_pytorch_model-00002-of-00006.safetensors,${MODEL_DIR}/diffusion_pytorch_model-00003-of-00006.safetensors,${MODEL_DIR}/diffusion_pytorch_model-00004-of-00006.safetensors,${MODEL_DIR}/diffusion_pytorch_model-00005-of-00006.safetensors,${MODEL_DIR}/diffusion_pytorch_model-00006-of-00006.safetensors" \
  --steps_per_epoch 1200 \
  --max_epochs 10 \
  --learning_rate 1e-5 \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --training_strategy "deepspeed_stage_2" \