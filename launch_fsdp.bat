@echo off
REM Launch script for FSDP multi-GPU video generation
REM This script runs wan2_generate_video.py with FSDP enabled for reduced VRAM usage

REM Get the prompt from command line (default if not provided)
if "%~1"=="" (
    set PROMPT="A beautiful sunset over mountains"
) else (
    set PROMPT=%1
)

REM Set environment variables for distributed training
set MASTER_ADDR=localhost
set MASTER_PORT=29501
set NCCL_DEBUG=INFO

echo ========================================
echo Launching FSDP Video Generation
echo ========================================
echo Prompt: %PROMPT%
echo Using FSDP for memory-efficient multi-GPU inference
echo ========================================

REM Run with torchrun for multi-GPU FSDP
torchrun --nproc_per_node=2 ^
         --master_addr=%MASTER_ADDR% ^
         --master_port=%MASTER_PORT% ^
         wan2_generate_video.py ^
         --dit_fsdp ^
         --t5_fsdp ^
         --fsdp_sharding_strategy FULL_SHARD ^
         --fsdp_mixed_precision bf16 ^
         --prompt %PROMPT% ^
         --output fsdp_output.mp4 ^
         --num_frames 49 ^
         --cfg_scale 3.0 ^
         --seed 42 %2 %3 %4 %5 %6 %7 %8 %9

echo ========================================
echo FSDP generation complete!
echo Output saved to: fsdp_output.mp4
echo ========================================