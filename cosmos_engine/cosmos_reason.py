#!/usr/bin/env python
# cosmos_reason.py — Cosmos3 understanding / Q&A for H1111.
#
# Runs the checkpoint's understanding-pathway LM (plus its bundled Qwen3-VL
# vision tower when an image or video frame is provided) as a general
# reasoner: describe an image, answer a question about it, caption a video
# frame, or do text-only Q&A. Same loading path as cosmos_upsample_prompt.py
# (generation weights stripped, block swap / gpu_layers CPU offload, fp8),
# one-shot: runs once, prints the answer, exits and frees VRAM.

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

from cosmos_video.model_loader import load_text_tokenizer, load_transformer
from cosmos_video.reasoner import Cosmos3Reasoner, place_und_layers, strip_generation_weights
from cosmos_video.upsampler_templates import SYSTEM_MESSAGE
from cosmos_video.vision_encoder import load_preprocessor_config, load_vision_encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosmos3 understanding / Q&A (H1111)")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Cosmos3 HF snapshot dir (diffusers layout)")
    parser.add_argument("--dit", type=str, default=None, help="override transformer dir or merged .safetensors")
    parser.add_argument("--question", type=str, required=True, help="question / instruction (or path to a .txt file)")
    parser.add_argument("--image_path", type=str, default=None, help="image to ground the answer on")
    parser.add_argument("--video_path", type=str, default=None,
                        help="video to ground the answer on (the middle frame is used)")
    parser.add_argument("--system_prompt", type=str, default=None, help="override the system message")

    # decoding
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (deterministic)")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)

    # memory / performance (same knobs as generation and the upsampler)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--attn_mode", type=str, default="torch",
                        choices=["torch", "sdpa", "flash", "flashattn", "flash2", "flash3", "sageattn", "xformers"])
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--gpu_layers", type=int, default=-1,
                        help="llama.cpp-style CPU offload: keep only the last N transformer layers on the GPU "
                             "and compute the rest on the CPU (-1 = all on GPU)")
    parser.add_argument("--no_prefill_stream", action="store_true",
                        help="with --gpu_layers: do not stream CPU-resident layers through the GPU for prefill")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8_scaled", action="store_true")
    parser.add_argument("--fp8_fast", action="store_true")
    parser.add_argument("--dit_dtype", type=str, default="bfloat16")

    parser.add_argument("--output", type=str, default=None, help="write the result JSON record here (else stdout only)")
    return parser.parse_args()


def read_text_arg(value: str) -> str:
    if os.path.exists(value) and value.lower().endswith(".txt"):
        with open(value, "r", encoding="utf-8") as f:
            return f.read().strip()
    return value


def load_grounding_image(image_path: str | None, video_path: str | None) -> Image.Image | None:
    """The conditioning image, or the middle frame of the video, or None for text-only Q&A."""
    if image_path:
        return Image.open(image_path).convert("RGB")
    if video_path:
        import imageio.v3 as iio

        frames = iio.imread(video_path, plugin="pyav")
        frame = frames[len(frames) // 2]
        logger.info(f"using middle frame {len(frames) // 2}/{len(frames)} of {video_path}")
        return Image.fromarray(frame).convert("RGB")
    return None


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    question = read_text_arg(args.question)
    if not question.strip():
        raise ValueError("--question is empty")
    if args.gpu_layers >= 0 and args.blocks_to_swap > 0:
        logger.info(f"--gpu_layers {args.gpu_layers} set: ignoring --blocks_to_swap {args.blocks_to_swap}")
        args.blocks_to_swap = 0

    from cosmos_video.attention import set_attention_backend

    set_attention_backend(args.attn_mode)
    dit_dtype = str_to_dtype(args.dit_dtype)

    image = load_grounding_image(args.image_path, args.video_path)

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
    strip_generation_weights(transformer)
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
                raise ValueError("--fp8_fast is CUDA-only and cannot run the CPU-resident layers of --gpu_layers")
            if device.type != "cpu" and not args.no_prefill_stream:
                prefill_device = device
            logger.info(f"llm-style CPU offload: {n_gpu} layers on {device}, {n_cpu} layers on cpu")

    tokenizer, chat_template = load_text_tokenizer(args.ckpt_dir)

    vision_encoder = None
    if image is not None:
        vision_encoder = load_vision_encoder(args.ckpt_dir, device=device, dtype=dit_dtype)
        if vision_encoder is None:
            raise ValueError(
                f"{args.ckpt_dir} has no vision_encoder/ component; image/video-grounded Q&A needs it"
            )

    reasoner = Cosmos3Reasoner(
        transformer,
        tokenizer,
        chat_template=chat_template,
        vision_encoder=vision_encoder,
        preprocessor_config=load_preprocessor_config(args.ckpt_dir),
        prefill_device=prefill_device,
    )
    logger.info(f"models loaded in {time.time() - start:.1f}s; asking ...")

    content = [{"type": "image"}, {"type": "text", "text": question}] if image is not None else question
    messages = [
        {"role": "system", "content": args.system_prompt or SYSTEM_MESSAGE},
        {"role": "user", "content": content},
    ]

    def progress(n_tokens):
        logger.info(f"decoded {n_tokens} tokens...")

    t0 = time.time()
    answer = reasoner.generate(
        messages,
        image=image,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature if args.temperature > 0 else None,
        top_k=args.top_k if args.temperature > 0 else None,
        top_p=args.top_p if args.temperature > 0 else None,
        seed=args.seed,
        progress_callback=progress,
    )
    logger.info(f"answered in {time.time() - t0:.1f}s")

    record = {
        "question": question,
        "answer": answer,
        "image_path": args.image_path,
        "video_path": args.video_path,
    }
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        logger.info(f"saved: {args.output}")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
