# SeedVR2 7B Video Super-Resolution GUI
# Standalone Gradio interface for video upscaling/restoration

import os
import sys
import gc
import datetime
import random
import traceback
from typing import Optional, List, Generator, Tuple

import torch
import gradio as gr
from gradio import themes
from gradio.themes.utils import colors
from PIL import Image
import numpy as np
from einops import rearrange
from tqdm import tqdm

# Add SeedVR module path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "modules", "SeedVR"))

from omegaconf import OmegaConf
from torchvision.transforms import Compose, Lambda, Normalize
from torchvision.io.video import read_video
from torchvision.io import read_image
import mediapy

# SeedVR imports - but NOT init_torch (it requires distributed)
from common.config import load_config
from common.seed import set_seed
from projects.video_diffusion_sr.infer import VideoDiffusionInfer
from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize
from data.video.transforms.rearrange import Rearrange

# Block swapping utilities
from modules.custom_offloading_utils import ModelOffloader

# ============================================================================
# Memory Debugging Utilities
# ============================================================================

def get_memory_stats() -> dict:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {"available": False}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - reserved

    return {
        "available": True,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
        "total_gb": total,
        "free_gb": free,
    }

def log_memory(stage: str, verbose: bool = True):
    """Log memory usage at a specific stage."""
    stats = get_memory_stats()
    if not stats["available"]:
        return

    msg = f"[MEMORY] {stage}: Allocated={stats['allocated_gb']:.2f}GB, Reserved={stats['reserved_gb']:.2f}GB, Free={stats['free_gb']:.2f}GB"
    print(msg)
    if verbose:
        print(f"         Max allocated so far: {stats['max_allocated_gb']:.2f}GB")
    return stats

def get_model_memory_footprint(model: torch.nn.Module, name: str = "Model") -> float:
    """Calculate memory footprint of a model in GB."""
    total_bytes = 0
    device_breakdown = {}

    for param in model.parameters():
        param_bytes = param.numel() * param.element_size()
        total_bytes += param_bytes
        device = str(param.device)
        device_breakdown[device] = device_breakdown.get(device, 0) + param_bytes

    for buffer in model.buffers():
        buffer_bytes = buffer.numel() * buffer.element_size()
        total_bytes += buffer_bytes
        device = str(buffer.device)
        device_breakdown[device] = device_breakdown.get(device, 0) + buffer_bytes

    print(f"[MODEL] {name} total: {total_bytes / 1e9:.2f}GB")
    for device, bytes_on_device in device_breakdown.items():
        print(f"         {device}: {bytes_on_device / 1e9:.2f}GB")

    return total_bytes / 1e9

def get_dit_block_distribution(dit_model) -> dict:
    """Get the device distribution of DiT blocks."""
    if not hasattr(dit_model, 'blocks'):
        return {"error": "No blocks attribute"}

    distribution = {"cuda": 0, "cpu": 0, "meta": 0, "other": 0}
    block_devices = []

    for i, block in enumerate(dit_model.blocks):
        # Check first parameter's device
        first_param = next(block.parameters(), None)
        if first_param is not None:
            device_type = first_param.device.type
            block_devices.append((i, device_type))
            if device_type in distribution:
                distribution[device_type] += 1
            else:
                distribution["other"] += 1

    print(f"[DiT BLOCKS] Distribution: {distribution}")
    print(f"         First 5 blocks: {[f'B{i}:{d}' for i, d in block_devices[:5]]}")
    print(f"         Last 5 blocks: {[f'B{i}:{d}' for i, d in block_devices[-5:]]}")

    return distribution

# Optional color fix
try:
    from projects.video_diffusion_sr.color_fix import wavelet_reconstruction
    USE_COLORFIX = True
except ImportError:
    USE_COLORFIX = False
    print("Note: Color fix (wavelet reconstruction) is not available")


def init_torch_simple():
    """Simple torch initialization for single-GPU inference."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # Initialize a single-process distributed group (required by SeedVR decorators)
    if not torch.distributed.is_initialized():
        # Use gloo backend for CPU-only init, then use CUDA for actual work
        torch.distributed.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )


def get_device() -> torch.device:
    """Get the current device."""
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")

# ============================================================================
# Theme and CSS (from h1111.py)
# ============================================================================

CUSTOM_THEME = themes.Default(
    primary_hue=colors.Color(
        name="custom",
        c50="#E6F0FF",
        c100="#CCE0FF",
        c200="#99C1FF",
        c300="#66A3FF",
        c400="#3384FF",
        c500="#0060df",
        c600="#0052C2",
        c700="#003D91",
        c800="#002961",
        c900="#001430",
        c950="#000A18"
    )
)

CUSTOM_CSS = """
.green-btn {
    background: linear-gradient(to bottom right, #2ecc71, #27ae60) !important;
    color: white !important;
    border: none !important;
}
.green-btn:hover {
    background: linear-gradient(to bottom right, #27ae60, #219651) !important;
}
.refresh-btn {
    max-width: 40px !important;
    min-width: 40px !important;
    height: 40px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.light-blue-btn {
    background: linear-gradient(to bottom right, #AEC6CF, #9AB8C4) !important;
    color: #333 !important;
    border: 1px solid #9AB8C4 !important;
}
.light-blue-btn:hover {
    background: linear-gradient(to bottom right, #9AB8C4, #8AA9B5) !important;
    border-color: #8AA9B5 !important;
}
"""

# ============================================================================
# Global State
# ============================================================================

class GlobalState:
    runner = None
    is_loaded = False
    stop_generation = False
    blocks_to_swap = 0

global_state = GlobalState()

# ============================================================================
# Model Loading with Block Swap Support
# ============================================================================

def enable_block_swap_for_nadit(model, blocks_to_swap: int, device: torch.device):
    """Enable block swapping for NaDiT model."""
    if blocks_to_swap <= 0:
        return

    num_blocks = len(model.blocks)
    blocks_to_swap = min(blocks_to_swap, num_blocks - 1)

    model.blocks_to_swap = blocks_to_swap
    model.num_blocks = num_blocks

    model.offloader = ModelOffloader(
        "nadit_block",
        model.blocks,
        num_blocks,
        blocks_to_swap,
        supports_backward=False,
        device=device,
    )

    print(f"NaDiT: Block swap enabled. Swapping {blocks_to_swap} of {num_blocks} blocks.")

def move_to_device_except_swap_blocks(model, device: torch.device):
    """Move model to device except swap blocks."""
    if hasattr(model, 'blocks_to_swap') and model.blocks_to_swap:
        save_blocks = model.blocks
        model.blocks = None
        model.to(device)
        model.blocks = save_blocks
    else:
        model.to(device)

def prepare_block_swap_before_forward(model):
    """Prepare block devices before forward pass."""
    if hasattr(model, 'blocks_to_swap') and model.blocks_to_swap:
        model.offloader.prepare_block_devices_before_forward(model.blocks)

def patch_nadit_forward_for_block_swap(model):
    """Patch NaDiT forward to support block swapping during inference."""
    original_forward = model.forward

    def forward_with_block_swap(vid, txt, vid_shape, txt_shape, timestep, disable_cache=True):
        from common.cache import Cache
        from common.distributed.ops import slice_inputs
        from models.dit import na

        # Text input
        if txt_shape.size(-1) == 1 and model.need_txt_repeat:
            txt, txt_shape = na.repeat(txt, txt_shape, "l c -> t l c", t=vid_shape[:, 0])
        txt = slice_inputs(txt, dim=0)
        txt = model.txt_in(txt)

        # Video input
        vid, vid_shape = model.vid_in(vid, vid_shape)

        # Embedding input
        emb = model.emb_in(timestep, device=vid.device, dtype=vid.dtype)

        # Body - with block swapping
        cache = Cache(disable=disable_cache)

        for i, block in enumerate(model.blocks):
            # Wait for block to be on GPU
            if hasattr(model, 'offloader') and model.offloader is not None:
                model.offloader.wait_for_block(i)

            vid, txt, vid_shape, txt_shape = block(
                vid=vid,
                txt=txt,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                emb=emb,
                cache=cache,
            )

            # Schedule next block swap
            if hasattr(model, 'offloader') and model.offloader is not None:
                model.offloader.submit_move_blocks_forward(model.blocks, i)

        vid, vid_shape = model.vid_out(vid, vid_shape, cache)

        from models.dit.nadit import NaDiTOutput
        return NaDiTOutput(vid_sample=vid)

    model.forward = forward_with_block_swap
    model._original_forward = original_forward

def load_model(
    config_path: str,
    checkpoint_path: str,
    vae_path: str,
    blocks_to_swap: int,
    progress=gr.Progress()
) -> Tuple[str, bool]:
    """Load SeedVR2 model with block swap support."""
    global global_state

    print(f"\n{'='*60}")
    print(f"LOADING MODEL")
    print(f"Blocks to swap: {blocks_to_swap}")
    print(f"{'='*60}\n")

    try:
        progress(0, desc="Initializing...")
        log_memory("Load: Initial state")

        # Clean up existing model
        if global_state.runner is not None:
            print("[LOAD] Cleaning up existing model...")
            del global_state.runner
            global_state.runner = None
            gc.collect()
            torch.cuda.empty_cache()
            log_memory("Load: After cleanup")

        progress(0.1, desc="Loading config...")

        # Initialize torch (simple, non-distributed)
        init_torch_simple()

        # Load config - need to be in SeedVR directory for relative paths
        seedvr_dir = os.path.join(SCRIPT_DIR, "modules", "SeedVR")
        original_cwd = os.getcwd()
        os.chdir(seedvr_dir)
        config = load_config(os.path.join(SCRIPT_DIR, config_path))
        os.chdir(original_cwd)

        runner = VideoDiffusionInfer(config)
        OmegaConf.set_readonly(runner.config, False)

        # Update VAE checkpoint path if provided (make absolute)
        if vae_path:
            runner.config.vae.checkpoint = os.path.join(SCRIPT_DIR, vae_path)

        progress(0.2, desc="Loading DiT model (this takes a while)...")
        print("\n[LOAD] Loading DiT model to CPU...")
        log_memory("Load: Before DiT load")

        # Load DiT to CPU first (make checkpoint path absolute)
        checkpoint_abs = os.path.join(SCRIPT_DIR, checkpoint_path)
        runner.configure_dit_model(device="cpu", checkpoint=checkpoint_abs)

        log_memory("Load: After DiT load to CPU")
        get_model_memory_footprint(runner.dit, "DiT (on CPU)")

        device = get_device()
        global_state.blocks_to_swap = blocks_to_swap

        progress(0.5, desc="Setting up block swap...")

        if blocks_to_swap > 0:
            print(f"\n[LOAD] Setting up block swap with {blocks_to_swap} blocks...")

            # Enable block swapping
            enable_block_swap_for_nadit(runner.dit, blocks_to_swap, device)
            log_memory("Load: After enable_block_swap_for_nadit")

            # Patch forward for block swap
            patch_nadit_forward_for_block_swap(runner.dit)
            print("[LOAD] Patched NaDiT forward for block swap")

            # Move to device except swap blocks
            print("[LOAD] Moving non-swap parts to GPU...")
            move_to_device_except_swap_blocks(runner.dit, device)
            log_memory("Load: After move_to_device_except_swap_blocks")
            get_dit_block_distribution(runner.dit)
        else:
            print("\n[LOAD] Moving entire DiT to GPU (no block swap)...")
            runner.dit.to(device)
            log_memory("Load: After DiT to GPU")

        progress(0.7, desc="Loading VAE...")
        print("\n[LOAD] Loading VAE...")
        log_memory("Load: Before VAE load")

        # Load VAE
        runner.configure_vae_model()
        if hasattr(runner.vae, "set_memory_limit"):
            runner.vae.set_memory_limit(**runner.config.vae.memory_limit)

        log_memory("Load: After VAE load (on GPU by default)")
        get_model_memory_footprint(runner.vae, "VAE")

        # Move VAE to CPU initially
        print("[LOAD] Moving VAE to CPU...")
        runner.vae.to("cpu")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        log_memory("Load: After VAE to CPU")

        progress(0.9, desc="Loading text embeddings...")

        # Load pre-computed embeddings (in SeedVR root)
        pos_emb_path = os.path.join(SCRIPT_DIR, "modules", "SeedVR", "pos_emb.pt")
        neg_emb_path = os.path.join(SCRIPT_DIR, "modules", "SeedVR", "neg_emb.pt")

        runner.pos_emb = torch.load(pos_emb_path, map_location="cpu")
        runner.neg_emb = torch.load(neg_emb_path, map_location="cpu")
        print(f"[LOAD] Text embeddings loaded: pos={runner.pos_emb.shape}, neg={runner.neg_emb.shape}")

        global_state.runner = runner
        global_state.is_loaded = True

        progress(1.0, desc="Done!")

        num_params = sum(p.numel() for p in runner.dit.parameters()) / 1e9
        print(f"\n{'='*60}")
        print(f"MODEL LOADED SUCCESSFULLY")
        print(f"DiT: {num_params:.1f}B params, Block swap: {blocks_to_swap} blocks")
        log_memory("Load: Final state")
        get_dit_block_distribution(runner.dit)
        print(f"{'='*60}\n")

        return f"Model loaded successfully! DiT: {num_params:.1f}B params, Block swap: {blocks_to_swap} blocks", True

    except Exception as e:
        # Print full error to console
        print(f"\n{'='*60}")
        print(f"ERROR loading model:")
        print(f"{'='*60}")
        log_memory("Load: Error state")
        traceback.print_exc()
        print(f"{'='*60}\n")

        gc.collect()
        torch.cuda.empty_cache()
        return f"Failed to load model. Check console for details.", False

# ============================================================================
# Video Processing Utilities
# ============================================================================

def is_image_file(filename: str) -> bool:
    """Check if file is an image."""
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return os.path.splitext(filename.lower())[1] in image_exts

def build_video_transform(res_h: int, res_w: int):
    """Build video preprocessing transform."""
    return Compose([
        NaResize(
            resolution=(res_h * res_w) ** 0.5,
            mode="area",
            downsample_only=False,
        ),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Rearrange("t c h w -> c t h w"),
    ])

def cut_videos_for_sp(videos: torch.Tensor, sp_size: int = 1) -> torch.Tensor:
    """Pad video frames for sequence parallel."""
    t = videos.size(1)
    if t == 1:
        return videos
    if t <= 4 * sp_size:
        padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
        padding = torch.cat(padding, dim=1)
        videos = torch.cat([videos, padding], dim=1)
        return videos
    if (t - 1) % (4 * sp_size) == 0:
        return videos
    else:
        padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - ((t - 1) % (4 * sp_size)))
        padding = torch.cat(padding, dim=1)
        videos = torch.cat([videos, padding], dim=1)
        return videos

# ============================================================================
# Inference Pipeline
# ============================================================================

def run_inference_custom(
    runner,
    noises: List[torch.Tensor],
    conditions: List[torch.Tensor],
    texts_pos: List[torch.Tensor],
    texts_neg: List[torch.Tensor],
    blocks_to_swap: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Custom inference that handles block swap properly.

    This bypasses the upstream runner.inference() method which has
    a problematic self.dit.to(get_device()) call at the end that
    ignores block swap and causes OOM.
    """
    from models.dit_v2 import na
    from common.diffusion import classifier_free_guidance_dispatcher

    batch_size = len(noises)
    cfg_scale = runner.config.diffusion.cfg.scale

    print(f"\n{'='*60}")
    print(f"[CUSTOM INFERENCE] Starting with {batch_size} samples")
    print(f"[CUSTOM INFERENCE] Blocks to swap: {blocks_to_swap}")
    log_memory("Before inference setup")
    get_dit_block_distribution(runner.dit)
    print(f"{'='*60}\n")

    # Text embeddings - flatten
    text_pos_embeds, text_pos_shapes = na.flatten(texts_pos)
    text_neg_embeds, text_neg_shapes = na.flatten(texts_neg)

    # Flatten latents
    latents, latents_shapes = na.flatten(noises)
    latents_cond, _ = na.flatten(conditions)

    log_memory("After flattening tensors")

    # Enter eval mode
    was_training = runner.dit.training
    runner.dit.eval()

    # Sampling with detailed logging
    print(f"\n[CUSTOM INFERENCE] Starting sampler...")
    log_memory("Before sampling")

    def forward_with_logging(args):
        """Wrapper that logs memory during forward pass."""
        log_memory(f"CFG forward step {args.i}", verbose=False)

        result = classifier_free_guidance_dispatcher(
            pos=lambda: runner.dit(
                vid=torch.cat([args.x_t, latents_cond], dim=-1),
                txt=text_pos_embeds,
                vid_shape=latents_shapes,
                txt_shape=text_pos_shapes,
                timestep=args.t.repeat(batch_size),
            ).vid_sample,
            neg=lambda: runner.dit(
                vid=torch.cat([args.x_t, latents_cond], dim=-1),
                txt=text_neg_embeds,
                vid_shape=latents_shapes,
                txt_shape=text_neg_shapes,
                timestep=args.t.repeat(batch_size),
            ).vid_sample,
            scale=(
                cfg_scale
                if (args.i + 1) / len(runner.sampler.timesteps)
                <= runner.config.diffusion.cfg.get("partial", 1)
                else 1.0
            ),
            rescale=runner.config.diffusion.cfg.rescale,
        )

        log_memory(f"After CFG forward step {args.i}", verbose=False)
        return result

    latents = runner.sampler.sample(x=latents, f=forward_with_logging)

    log_memory("After sampling complete")

    # Exit eval mode
    runner.dit.train(was_training)

    # Unflatten
    latents = na.unflatten(latents, latents_shapes)

    print(f"\n[CUSTOM INFERENCE] Sampling complete!")
    log_memory("After unflatten")
    get_dit_block_distribution(runner.dit)

    return latents


def run_inference(
    input_path: str,
    res_h: int,
    res_w: int,
    seed: int,
    cfg_scale: float,
    cfg_rescale: float,
    sample_steps: int,
    out_fps: float,
    use_colorfix: bool,
    output_dir: str,
    progress=gr.Progress()
) -> Generator[Tuple[str, Optional[str]], None, None]:
    """Run SeedVR2 inference pipeline with comprehensive memory debugging."""
    global global_state

    if not global_state.is_loaded or global_state.runner is None:
        yield "Error: Model not loaded. Please load the model first.", None
        return

    runner = global_state.runner
    device = get_device()

    # Reset max memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"\n{'='*60}")
    print(f"STARTING INFERENCE PIPELINE")
    print(f"Resolution: {res_w}x{res_h}, Blocks to swap: {global_state.blocks_to_swap}")
    print(f"{'='*60}\n")

    try:
        # Set seed
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        set_seed(seed, same_across_ranks=True)

        progress(0.05, desc="Configuring diffusion...")
        log_memory("Step 1: Configure diffusion")

        # Configure diffusion parameters
        runner.config.diffusion.cfg.scale = cfg_scale
        runner.config.diffusion.cfg.rescale = cfg_rescale
        runner.config.diffusion.timesteps.sampling.steps = sample_steps
        runner.configure_diffusion()

        progress(0.1, desc="Loading input...")
        yield "Loading input video/image...", None
        log_memory("Step 2: Loading input")

        # Build transform
        transform = build_video_transform(res_h, res_w)

        # Load input
        is_image = is_image_file(input_path)
        if is_image:
            video = read_image(input_path).unsqueeze(0) / 255.0
            input_fps = 30.0  # Default for images
        else:
            video, _, info = read_video(input_path, output_format="TCHW")
            video = video / 255.0
            input_fps = info["video_fps"]

        ori_length = video.size(0)
        log_memory(f"Step 2b: Loaded {ori_length} frames")

        input_video = transform(video.to(device))
        cond_video = cut_videos_for_sp(input_video, sp_size=1)

        print(f"[INPUT] Original: {ori_length} frames, {video.shape[-2]}x{video.shape[-1]}")
        print(f"[INPUT] Transformed: {cond_video.shape}")
        log_memory("Step 2c: After transform")

        yield f"Input: {ori_length} frames, {video.shape[-2]}x{video.shape[-1]}", None

        # ============== VAE ENCODING ==============
        progress(0.2, desc="VAE encoding...")
        yield "Encoding with VAE...", None

        print(f"\n{'='*60}")
        print(f"VAE ENCODING PHASE")
        print(f"{'='*60}")
        log_memory("Step 3: Before VAE encode setup")
        get_dit_block_distribution(runner.dit)

        # Move DiT to CPU for VAE encoding
        print("\n[OFFLOAD] Moving DiT to CPU for VAE encoding...")

        if global_state.blocks_to_swap > 0:
            # Wait for any pending block swaps
            if hasattr(runner.dit, 'offloader') and runner.dit.offloader is not None:
                print("[OFFLOAD] Waiting for pending block swaps...")
                for idx in range(len(runner.dit.blocks)):
                    runner.dit.offloader.wait_for_block(idx)

        # Move entire DiT to CPU (including blocks)
        runner.dit.to("cpu")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        log_memory("Step 3b: After DiT to CPU")
        get_dit_block_distribution(runner.dit)

        # Move VAE to GPU
        print("\n[OFFLOAD] Moving VAE to GPU...")
        runner.vae.to(device)
        log_memory("Step 3c: After VAE to GPU")
        get_model_memory_footprint(runner.vae, "VAE")

        # Encode
        print("\n[VAE] Encoding...")
        cond_latents = runner.vae_encode([cond_video])
        print(f"[VAE] Encoded latent shape: {cond_latents[0].shape}")
        log_memory("Step 3d: After VAE encode")

        # Move VAE back to CPU
        print("\n[OFFLOAD] Moving VAE to CPU...")
        runner.vae.to("cpu")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        log_memory("Step 3e: After VAE to CPU, before DiT setup")

        # ============== DiT INFERENCE ==============
        progress(0.4, desc="Preparing DiT...")
        yield "Preparing DiT inference...", None

        print(f"\n{'='*60}")
        print(f"DiT INFERENCE PHASE")
        print(f"{'='*60}")

        # Prepare DiT for inference with block swap
        if global_state.blocks_to_swap > 0:
            print(f"\n[DiT] Setting up block swap ({global_state.blocks_to_swap} blocks)...")
            move_to_device_except_swap_blocks(runner.dit, device)
            log_memory("Step 4a: After move_to_device_except_swap_blocks")
            get_dit_block_distribution(runner.dit)

            prepare_block_swap_before_forward(runner.dit)
            log_memory("Step 4b: After prepare_block_swap_before_forward")
            get_dit_block_distribution(runner.dit)
        else:
            print("\n[DiT] Moving entire model to GPU (no block swap)...")
            runner.dit.to(device)
            log_memory("Step 4: After DiT to GPU (no block swap)")

        # Prepare text embeddings
        print("\n[DiT] Preparing text embeddings...")
        texts_pos = [runner.pos_emb.to(device)]
        texts_neg = [runner.neg_emb.to(device)]
        log_memory("Step 4c: After text embeddings to GPU")

        progress(0.5, desc="Running DiT inference...")
        yield "Running diffusion sampling...", None

        # Generation step
        print("\n[DiT] Creating noise tensors...")
        noises = [torch.randn_like(lat) for lat in cond_latents]
        noises = [n.to(device) for n in noises]
        cond_latents_gpu = [lat.to(device) for lat in cond_latents]
        log_memory("Step 5: After noise tensors created")

        # Build conditions (no noise for SR)
        print("\n[DiT] Building conditions...")
        conditions = [
            runner.get_condition(noise, task="sr", latent_blur=lat)
            for noise, lat in zip(noises, cond_latents_gpu)
        ]
        log_memory("Step 5b: After conditions built")

        # Run custom inference (bypasses problematic upstream dit.to() call)
        print("\n[DiT] Running custom inference...")
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
            latent_outputs = run_inference_custom(
                runner=runner,
                noises=noises,
                conditions=conditions,
                texts_pos=texts_pos,
                texts_neg=texts_neg,
                blocks_to_swap=global_state.blocks_to_swap,
                device=device,
            )

        log_memory("Step 6: After DiT inference complete")

        # ============== VAE DECODING ==============
        progress(0.7, desc="VAE decoding...")
        yield "Decoding with VAE...", None

        print(f"\n{'='*60}")
        print(f"VAE DECODING PHASE")
        print(f"{'='*60}")

        # Move DiT to CPU before VAE decode
        print("\n[OFFLOAD] Moving DiT to CPU for VAE decode...")
        runner.dit.to("cpu")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        log_memory("Step 7: After DiT to CPU")

        # Move VAE to GPU
        print("\n[OFFLOAD] Moving VAE to GPU...")
        runner.vae.to(device)
        log_memory("Step 7b: After VAE to GPU")

        # Decode
        print("\n[VAE] Decoding...")
        samples = runner.vae_decode(latent_outputs)
        log_memory("Step 7c: After VAE decode")

        # Move VAE back to CPU
        runner.vae.to("cpu")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        log_memory("Step 7d: After VAE to CPU")

        # Rearrange output
        print("\n[OUTPUT] Rearranging samples...")
        samples = [
            rearrange(v[:, None], "c t h w -> t c h w") if v.ndim == 3
            else rearrange(v, "c t h w -> t c h w")
            for v in samples
        ]

        # Cleanup intermediate tensors
        del latent_outputs, noises, conditions, cond_latents_gpu, cond_latents
        gc.collect()
        torch.cuda.empty_cache()
        log_memory("Step 8: After cleanup")

        # Trim to original length
        sample = samples[0]
        if ori_length < sample.shape[0]:
            sample = sample[:ori_length]

        progress(0.8, desc="Post-processing...")

        # Apply color fix if enabled
        if use_colorfix and USE_COLORFIX:
            yield "Applying color correction...", None
            print("\n[POST] Applying wavelet color fix...")
            input_for_colorfix = rearrange(input_video, "c t h w -> t c h w")
            sample = wavelet_reconstruction(
                sample.to("cpu"),
                input_for_colorfix[:sample.size(0)].to("cpu")
            )
        else:
            sample = sample.to("cpu")

        log_memory("Step 9: After post-processing")

        progress(0.9, desc="Saving output...")
        yield "Saving output...", None

        # Save output
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare sample for saving
        sample = rearrange(sample, "t c h w -> t h w c")
        sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round()
        sample = sample.to(torch.uint8).numpy()

        save_fps = out_fps if out_fps > 0 else input_fps

        if sample.shape[0] == 1:
            output_path = os.path.join(output_dir, f"seedvr2_{timestamp}_{seed}.png")
            mediapy.write_image(output_path, sample.squeeze(0))
        else:
            output_path = os.path.join(output_dir, f"seedvr2_{timestamp}_{seed}.mp4")
            mediapy.write_video(output_path, sample, fps=save_fps)

        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print(f"INFERENCE COMPLETE")
        log_memory("Final state")
        print(f"Output saved to: {output_path}")
        print(f"{'='*60}\n")

        progress(1.0, desc="Done!")
        yield f"Done! Saved to {output_path}", output_path

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n{'='*60}")
        print(f"CUDA OUT OF MEMORY:")
        print(f"{'='*60}")
        log_memory("OOM state")
        get_dit_block_distribution(runner.dit)
        traceback.print_exc()
        print(f"{'='*60}\n")
        gc.collect()
        torch.cuda.empty_cache()
        yield "CUDA out of memory! Try increasing 'Blocks to Swap' or reducing resolution. Check console for memory details.", None
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR during inference:")
        print(f"{'='*60}")
        log_memory("Error state")
        traceback.print_exc()
        print(f"{'='*60}\n")
        gc.collect()
        torch.cuda.empty_cache()
        yield "Error during inference. Check console for details.", None

# ============================================================================
# Gradio UI
# ============================================================================

def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    """Get video dimensions, frame count, and FPS."""
    if video_path is None:
        return 0, 0, 0, 0.0
    try:
        if is_image_file(video_path):
            img = read_image(video_path)
            return img.shape[2], img.shape[1], 1, 30.0  # W, H, frames, fps
        else:
            video, _, info = read_video(video_path, output_format="TCHW", pts_unit="sec")
            return video.shape[3], video.shape[2], video.shape[0], info.get("video_fps", 30.0)
    except Exception as e:
        print(f"Error reading video info: {e}")
        return 0, 0, 0, 0.0


def create_ui():
    """Create the Gradio interface."""

    with gr.Blocks(theme=CUSTOM_THEME, css=CUSTOM_CSS, title="SeedVR2 7B Video SR") as demo:
        gr.Markdown("# SeedVR2 7B Video Super-Resolution")
        gr.Markdown("One-step diffusion model for video upscaling and restoration. The 7B model (33GB FP32) requires block swapping for memory management.")

        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                input_video = gr.Video(
                    label="Input Video",
                    sources=["upload"],
                    height=250
                )
                input_image = gr.Image(
                    label="Or Input Image",
                    type="filepath",
                    height=200
                )

                # Input info display
                input_info = gr.Textbox(
                    label="Input Info",
                    value="No input loaded",
                    interactive=False
                )

                gr.Markdown("### Output Resolution")

                # Hidden state for input dimensions
                input_width = gr.State(value=0)
                input_height = gr.State(value=0)

                scale_slider = gr.Slider(
                    label="Scale Factor",
                    minimum=0.5, maximum=4.0, step=0.25, value=1.0,
                    info="Output size = Input size × Scale"
                )

                with gr.Row():
                    res_w = gr.Number(label="Output Width", value=1280, step=16, minimum=256, maximum=4096)
                    res_h = gr.Number(label="Output Height", value=720, step=16, minimum=256, maximum=2160)

                with gr.Accordion("Generation Parameters", open=True):
                    with gr.Row():
                        seed = gr.Number(label="Seed (-1 = random)", value=-1)
                        random_seed_btn = gr.Button("🎲", elem_classes="refresh-btn")

                    cfg_scale = gr.Slider(
                        label="CFG Scale",
                        minimum=1.0, maximum=15.0, step=0.5, value=7.5,
                        info="Classifier-free guidance scale"
                    )
                    cfg_rescale = gr.Slider(
                        label="CFG Rescale",
                        minimum=0.0, maximum=1.0, step=0.05, value=0.0,
                        info="CFG rescaling factor"
                    )
                    sample_steps = gr.Slider(
                        label="Sample Steps",
                        minimum=1, maximum=10, step=1, value=1,
                        info="SeedVR2 uses 1 step by design"
                    )
                    out_fps = gr.Slider(
                        label="Output FPS (0 = match input)",
                        minimum=0, maximum=60, step=1, value=0
                    )
                    use_colorfix = gr.Checkbox(
                        label="Apply Color Fix (Wavelet Reconstruction)",
                        value=USE_COLORFIX,
                        interactive=USE_COLORFIX
                    )

            # Right column - Output and Settings
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                output_video = gr.Video(
                    label="Enhanced Video",
                    height=300,
                    interactive=False
                )

                progress_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )

                with gr.Accordion("Performance / Memory", open=True):
                    blocks_to_swap = gr.Slider(
                        label="Blocks to Swap (VRAM Saving)",
                        minimum=0, maximum=35, step=1, value=20,
                        info="Higher = less VRAM, slower. 36 total blocks. Recommend: 24GB→20, 16GB→28, 12GB→32"
                    )
                    vae_on_cpu = gr.Checkbox(
                        label="VAE on CPU (slower but saves ~4GB VRAM)",
                        value=False
                    )

                with gr.Accordion("Model Paths", open=False):
                    config_path = gr.Textbox(
                        label="Config Path",
                        value="modules/SeedVR/configs_7b/main.yaml"
                    )
                    checkpoint_path = gr.Textbox(
                        label="DiT Checkpoint",
                        value="wan/SeedVR2-7B/seedvr2_ema_7b.pth"
                    )
                    checkpoint_variant = gr.Dropdown(
                        label="Checkpoint Variant",
                        choices=["seedvr2_ema_7b.pth", "seedvr2_ema_7b_sharp.pth"],
                        value="seedvr2_ema_7b.pth"
                    )
                    vae_path = gr.Textbox(
                        label="VAE Checkpoint",
                        value="wan/SeedVR2-7B/ema_vae.pth"
                    )
                    output_dir = gr.Textbox(
                        label="Output Directory",
                        value="outputs/seedvr2"
                    )

        # Buttons
        with gr.Row():
            load_model_btn = gr.Button("Load Model", elem_classes="light-blue-btn")
            generate_btn = gr.Button("Generate Enhanced Video", elem_classes="green-btn", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop")

        # State
        model_loaded = gr.State(value=False)

        # ============== Event Handlers ==============

        # Video/Image upload detection
        def on_video_upload(video_path):
            if video_path is None:
                return "No input loaded", 0, 0, 1280, 720
            w, h, frames, fps = get_video_info(video_path)
            if w == 0 or h == 0:
                return "Failed to read video", 0, 0, 1280, 720
            info = f"{w}x{h}, {frames} frames, {fps:.1f} fps"
            return info, w, h, w, h

        def on_image_upload(image_path):
            if image_path is None:
                return "No input loaded", 0, 0, 1280, 720
            w, h, frames, fps = get_video_info(image_path)
            if w == 0 or h == 0:
                return "Failed to read image", 0, 0, 1280, 720
            info = f"{w}x{h} (image)"
            return info, w, h, w, h

        input_video.change(
            fn=on_video_upload,
            inputs=[input_video],
            outputs=[input_info, input_width, input_height, res_w, res_h]
        )

        input_image.change(
            fn=on_image_upload,
            inputs=[input_image],
            outputs=[input_info, input_width, input_height, res_w, res_h]
        )

        # Scale slider
        def update_resolution_from_scale(scale, in_w, in_h):
            if in_w == 0 or in_h == 0:
                return gr.update(), gr.update()
            # Make divisible by 16
            out_w = int((in_w * scale) // 16) * 16
            out_h = int((in_h * scale) // 16) * 16
            out_w = max(256, min(4096, out_w))
            out_h = max(256, min(2160, out_h))
            return out_w, out_h

        scale_slider.change(
            fn=update_resolution_from_scale,
            inputs=[scale_slider, input_width, input_height],
            outputs=[res_w, res_h]
        )

        # Random seed
        random_seed_btn.click(
            fn=lambda: random.randint(0, 2**32 - 1),
            outputs=[seed]
        )

        # Checkpoint variant update
        def update_checkpoint_path(variant):
            return f"wan/SeedVR2-7B/{variant}"

        checkpoint_variant.change(
            fn=update_checkpoint_path,
            inputs=[checkpoint_variant],
            outputs=[checkpoint_path]
        )

        # Load model
        load_model_btn.click(
            fn=load_model,
            inputs=[config_path, checkpoint_path, vae_path, blocks_to_swap],
            outputs=[progress_text, model_loaded]
        )

        # Generate
        def generate_wrapper(
            input_video_path, input_image_path,
            res_w, res_h, seed, cfg_scale, cfg_rescale,
            sample_steps, out_fps, use_colorfix, output_dir,
            model_loaded, progress=gr.Progress()
        ):
            if not model_loaded:
                yield "Error: Model not loaded. Click 'Load Model' first.", None
                return

            # Determine input path
            input_path = None
            if input_video_path:
                input_path = input_video_path
            elif input_image_path:
                input_path = input_image_path

            if not input_path:
                yield "Error: Please provide an input video or image.", None
                return

            for status, output_path in run_inference(
                input_path=input_path,
                res_h=int(res_h),
                res_w=int(res_w),
                seed=int(seed),
                cfg_scale=float(cfg_scale),
                cfg_rescale=float(cfg_rescale),
                sample_steps=int(sample_steps),
                out_fps=float(out_fps),
                use_colorfix=use_colorfix,
                output_dir=output_dir,
                progress=progress
            ):
                yield status, output_path

        generate_btn.click(
            fn=generate_wrapper,
            inputs=[
                input_video, input_image,
                res_w, res_h, seed, cfg_scale, cfg_rescale,
                sample_steps, out_fps, use_colorfix, output_dir,
                model_loaded
            ],
            outputs=[progress_text, output_video]
        )

        # Stop generation
        def stop_generation():
            global global_state
            global_state.stop_generation = True
            return "Stopping..."

        stop_btn.click(fn=stop_generation, outputs=[progress_text])

    return demo

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    demo = create_ui()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()
