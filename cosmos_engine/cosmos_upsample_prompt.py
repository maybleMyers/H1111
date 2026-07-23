#!/usr/bin/env python
# cosmos_upsample_prompt.py — native Cosmos3 prompt upsampling for H1111.
#
# Runs the checkpoint's OWN understanding-pathway LM (plus its bundled Qwen3-VL
# vision tower for i2v) to expand a terse prompt into the JSON-structured
# caption Cosmos3 was trained on. No external API is ever contacted; the model
# named by --ckpt_dir does the upsampling, exactly like cosmos-framework's
# native_prompt_upsampling. Runs once and exits, freeing VRAM for generation.
#
# The generation-pathway weights (diffusion experts, projections, heads) are
# stripped after load — roughly half the transformer — so the LM fits in far
# less memory than a generation run of the same checkpoint.

import argparse
import json
import logging
import os
import sys
import time

import torch
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_here), _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.model_utils import str_to_dtype

from cosmos_video import configs as cfg
from cosmos_video.model_loader import load_text_tokenizer, load_transformer
from cosmos_video.reasoner import (
    Cosmos3Reasoner,
    extract_json_object,
    is_upsampled_prompt,
    place_und_layers,
    resolve_task,
    strip_generation_weights,
)
from cosmos_video.vision_encoder import load_preprocessor_config, load_vision_encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosmos3 native prompt upsampling (H1111)")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Cosmos3 HF snapshot dir (diffusers layout)")
    parser.add_argument("--dit", type=str, default=None, help="override transformer dir or merged .safetensors")
    parser.add_argument("--prompt", type=str, required=True, help="terse prompt text (or path to a .txt file)")
    parser.add_argument("--image_path", type=str, default=None, help="conditioning image for i2v upsampling")
    parser.add_argument("--task", type=str, default="auto", choices=["auto", "t2v", "t2i", "i2v"])

    # generation geometry (baked into the V4.2 template constraints)
    parser.add_argument("--resolution", type=str, default="720", choices=sorted(cfg.VIDEO_RES_SIZE_INFO))
    parser.add_argument("--aspect_ratio", type=str, default="16:9", choices=["1:1", "4:3", "3:4", "16:9", "9:16"])
    parser.add_argument("--video_length", type=int, default=189)
    parser.add_argument("--fps", type=int, default=24)

    # decoding
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (deterministic, recommended)")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)

    # memory / performance (same knobs as generation)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--attn_mode", type=str, default="torch",
                        choices=["torch", "sdpa", "flash", "flashattn", "flash2", "flash3", "sageattn", "xformers"])
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--gpu_layers", type=int, default=-1,
                        help="llama.cpp-style CPU offload (its -ngl): keep only the last N transformer layers "
                             "on the GPU and compute the rest on the CPU where their weights live (-1 = all on "
                             "GPU). Far faster than --blocks_to_swap for token-by-token decoding, which would "
                             "re-stream the swapped weights over PCIe for every generated token.")
    parser.add_argument("--no_prefill_stream", action="store_true",
                        help="with --gpu_layers: do not stream the CPU-resident layers through the GPU for the "
                             "one-time prompt prefill (saves a little VRAM, costs prefill speed)")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8_scaled", action="store_true")
    parser.add_argument("--fp8_fast", action="store_true")
    parser.add_argument("--dit_dtype", type=str, default="bfloat16")
    parser.add_argument("--keep_gen_weights", action="store_true",
                        help="do not strip the generation-pathway weights (debugging only; doubles memory)")

    parser.add_argument("--output", type=str, default=None, help="write the result JSON record here (else stdout)")
    return parser.parse_args()


def read_prompt(value: str) -> str:
    if os.path.exists(value) and value.lower().endswith(".txt"):
        with open(value, "r", encoding="utf-8") as f:
            return f.read().strip()
    return value


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt = read_prompt(args.prompt)
    if not prompt.strip():
        raise ValueError("--prompt is empty")
    if is_upsampled_prompt(prompt):
        raise ValueError("prompt already looks like upsampled JSON; refusing to double-upsample")

    task = resolve_task(args.task, has_image=bool(args.image_path), video_length=args.video_length)
    if task == "i2v" and not args.image_path:
        raise ValueError("task=i2v requires --image_path")

    from cosmos_video.attention import set_attention_backend

    set_attention_backend(args.attn_mode)
    dit_dtype = str_to_dtype(args.dit_dtype)

    width, height = cfg.resolve_video_size(args.resolution, args.aspect_ratio)
    duration_secs = max(1, int(args.video_length / args.fps)) if task != "t2i" else None

    start = time.time()
    logger.info(f"loading transformer (und pathway) from {args.ckpt_dir} ...")
    transformer = load_transformer(
        args.ckpt_dir,
        device=device,
        dit_dtype=dit_dtype,
        fp8=args.fp8,
        fp8_scaled=args.fp8_scaled,
        fp8_fast=args.fp8_fast,
        dit_path=args.dit,
    )
    if not args.keep_gen_weights:
        strip_generation_weights(transformer)
        logger.info("generation-pathway weights stripped (LM-only upsampler load)")
    if args.gpu_layers >= 0 and args.blocks_to_swap > 0:
        raise ValueError("--gpu_layers and --blocks_to_swap are mutually exclusive; for the upsampler's "
                         "autoregressive decode prefer --gpu_layers (block swap re-streams weights every token)")
    prefill_device = None
    if args.blocks_to_swap > 0:
        logger.info(f"enabling block swap: {args.blocks_to_swap} blocks")
        transformer.enable_block_swap(args.blocks_to_swap, device, supports_backward=False)
        transformer.move_to_device_except_swap_blocks(device)
        transformer.prepare_block_swap_before_forward()
    else:
        n_gpu, n_cpu = place_und_layers(transformer, device, args.gpu_layers)
        if n_cpu:
            if args.fp8_fast:
                raise ValueError("--fp8_fast uses torch._scaled_mm (CUDA-only) and cannot run the CPU-resident "
                                 "layers of --gpu_layers; drop --fp8_fast (plain --fp8_scaled works on CPU)")
            if device.type != "cpu" and not args.no_prefill_stream:
                prefill_device = device
            logger.info(f"llm-style CPU offload: {n_gpu} layers on {device}, {n_cpu} layers on cpu "
                        f"(norm/lm_head on {device}; prefill {'streamed via ' + str(device) if prefill_device else 'on cpu'})")

    tokenizer, chat_template = load_text_tokenizer(args.ckpt_dir)

    vision_encoder = None
    image = None
    if task == "i2v":
        vision_encoder = load_vision_encoder(args.ckpt_dir, device=device, dtype=dit_dtype)
        if vision_encoder is None:
            raise ValueError(
                f"{args.ckpt_dir} has no vision_encoder/ component; i2v prompt upsampling needs it "
                "(re-download the checkpoint including vision_encoder/, or use --task t2v)"
            )
        image = Image.open(args.image_path).convert("RGB")

    reasoner = Cosmos3Reasoner(
        transformer,
        tokenizer,
        chat_template=chat_template,
        vision_encoder=vision_encoder,
        preprocessor_config=load_preprocessor_config(args.ckpt_dir),
        prefill_device=prefill_device,
    )
    logger.info(f"models loaded in {time.time() - start:.1f}s; upsampling task={task} "
                f"{width}x{height} fps={args.fps} duration={duration_secs}s")

    def progress(n_tokens):
        logger.info(f"decoded {n_tokens} tokens...")

    t0 = time.time()
    upsampled = reasoner.upsample(
        prompt,
        task=task,
        image=image,
        resolution_w=width,
        resolution_h=height,
        aspect_ratio=args.aspect_ratio,
        fps=args.fps if task != "t2i" else None,
        duration_secs=duration_secs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature if args.temperature > 0 else None,
        top_k=args.top_k if args.temperature > 0 else None,
        top_p=args.top_p if args.temperature > 0 else None,
        seed=args.seed,
        progress_callback=progress,
    )
    logger.info(f"upsampled in {time.time() - t0:.1f}s")

    record = {"prompt": upsampled, "task": task, "original_prompt": prompt}
    try:
        parsed = extract_json_object(upsampled)
        record["parsed_keys"] = sorted(parsed.keys())
    except Exception as e:  # output stays usable even if the model deviated from the schema
        logger.warning(f"upsampled output did not parse as a JSON object: {e}")
        record["parsed_keys"] = None

    payload = json.dumps(record, ensure_ascii=False, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        logger.info(f"saved: {args.output}")
    print(payload)


if __name__ == "__main__":
    main()
