@echo off
REM Launch script for dual GPU sequence parallelism using torchrun on Windows
REM This script runs the wan2_generate_video.py with distributed PyTorch

REM Check if required arguments are provided
if "%~1"=="" (
    echo Usage: %0 "prompt" [additional_args...]
    echo Example: %0 "A beautiful sunset over mountains" --output output.mp4
    exit /b 1
)

REM Extract the prompt (first argument)
set PROMPT=%~1
shift

REM Build additional arguments
set ADDITIONAL_ARGS=
:build_args
if "%~1"=="" goto :done_args
set ADDITIONAL_ARGS=%ADDITIONAL_ARGS% %1
shift
goto :build_args
:done_args

REM Set environment variables for distributed training
set CUDA_VISIBLE_DEVICES=0,1
set MASTER_ADDR=localhost
set MASTER_PORT=29500

REM Optional: Set NCCL debug level for troubleshooting
REM set NCCL_DEBUG=INFO

echo Launching with sequence parallelism on GPUs 0 and 1...
echo Prompt: %PROMPT%
echo Additional arguments: %ADDITIONAL_ARGS%

REM Launch with torchrun for 2 processes (one per GPU)
python -m torch.distributed.launch ^
    --nproc_per_node=2 ^
    --master_addr=%MASTER_ADDR% ^
    --master_port=%MASTER_PORT% ^
    wan2_generate_video.py ^
    --use_sequence_parallel ^
    --prompt "%PROMPT%" ^
    %ADDITIONAL_ARGS%

echo Generation complete!