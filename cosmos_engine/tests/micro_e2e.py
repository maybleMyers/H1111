# Micro end-to-end Cosmos3 pipeline test: tiny random-weight components + real
# tokenizer/chat template, CPU only. Exercises tokenize_prompt, sequence packing,
# the denoise loop with CFG, v2v conditioning, action conditioning, and decode.
import json
import sys

import numpy as np
import torch
from PIL import Image


from transformers import AutoTokenizer

import os as _os, sys as _sys
_ENGINE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.dirname(_ENGINE), _ENGINE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from cosmos_video.pipeline import Cosmos3OmniPipeline, CosmosActionCondition
from cosmos_video.scheduler_unipc import UniPCMultistepScheduler
from cosmos_video.sound_tokenizer import Cosmos3AVAEAudioTokenizer  # noqa: F401 (import check)
from cosmos_video.transformer import Cosmos3OmniTransformer
from cosmos_video.vae import AutoencoderKLWan

torch.manual_seed(0)

IDX = _os.path.join(_ENGINE, "cosmos_hf_indexes", "Cosmos3-Nano")

tokenizer = AutoTokenizer.from_pretrained(IDX + "/text_tokenizer")
chat_template = getattr(tokenizer, "chat_template", None)
if chat_template is None:
    with open(IDX + "/text_tokenizer/chat_template.jinja") as f:
        tokenizer.chat_template = f.read()
print("tokenizer loaded; has_template:", tokenizer.chat_template is not None)

transformer = Cosmos3OmniTransformer(
    hidden_size=64,
    intermediate_size=128,
    head_dim=16,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    latent_channel=48,
    latent_patch_size=2,
    patch_latent_dim=192,
    vocab_size=len(tokenizer),
    action_gen=True,
    action_dim=64,
    sound_gen=False,
    rope_scaling={"mrope_interleaved": True, "mrope_section": [8, 4, 4], "rope_type": "default"},
).float().eval()
print("tiny transformer params:", sum(p.numel() for p in transformer.parameters()) / 1e6, "M")

vae_cfg = json.load(open(IDX + "/vae/config.json"))
vae_cfg.pop("_class_name", None); vae_cfg.pop("_diffusers_version", None); vae_cfg.pop("_name_or_path", None); vae_cfg.pop("clip_output", None)
vae_cfg["base_dim"] = 16
vae_cfg["decoder_base_dim"] = 16
vae_cfg["num_res_blocks"] = 1
vae = AutoencoderKLWan(**vae_cfg).float().eval()
print("tiny vae params:", sum(p.numel() for p in vae.parameters()) / 1e6, "M")

sched_cfg = json.load(open(IDX + "/scheduler/scheduler_config.json"))
sched_cfg.pop("_class_name", None); sched_cfg.pop("_diffusers_version", None)
scheduler = UniPCMultistepScheduler(**sched_cfg)

pipe = Cosmos3OmniPipeline(
    transformer=transformer,
    text_tokenizer=tokenizer,
    vae=vae,
    scheduler=scheduler,
    sound_tokenizer=None,
    default_use_system_prompt=False,
    use_native_flow_schedule=True,
)

gen = torch.Generator().manual_seed(42)
common = dict(num_inference_steps=2, guidance_scale=6.0, generator=gen, return_dict=True)

steps_seen = []


def cb(pipe_, step, t, kw):
    steps_seen.append(step)
    assert "latents" in kw and torch.isfinite(kw["latents"]).all()
    return kw


# t2v (latent output)
r = pipe(prompt="a red ball", negative_prompt="blurry", num_frames=5, height=64, width=64,
         fps=24.0, output_type="latent", callback_on_step_end=cb, **common)
print("t2v latents:", tuple(r.video.shape), "steps:", steps_seen)
assert torch.isfinite(r.video).all()

# t2i (decoded)
r = pipe(prompt="a cat", num_frames=1, height=64, width=64, fps=24.0, output_type="pil", **common)
print("t2i output:", type(r.video[0]).__name__, r.video[0].size if hasattr(r.video[0], "size") else None)

# i2v
img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
r = pipe(prompt="move", image=img, num_frames=5, height=64, width=64, output_type="latent", **common)
print("i2v latents:", tuple(r.video.shape))

# v2v with condition frames
frames = [Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)) for _ in range(9)]
r = pipe(prompt="continue", video=frames, condition_frame_indexes_vision=(0, 1), num_frames=9,
         height=64, width=64, output_type="latent", **common)
print("v2v latents:", tuple(r.video.shape))

# forward_dynamics action
action = CosmosActionCondition(
    mode="forward_dynamics",
    chunk_size=4,
    domain_name="droid_lerobot",
    raw_actions=torch.randn(4, 10),
    image=img,
    view_point="ego_view",
    resolution_tier=256,
)
r = pipe(prompt="", action=action, num_inference_steps=2, guidance_scale=1.0,
         generator=gen, output_type="latent", return_dict=True)
print("fd latents:", tuple(r.video.shape), "action out:", r.action)

# inverse_dynamics returns predicted actions
action = CosmosActionCondition(
    mode="inverse_dynamics",
    chunk_size=4,
    domain_name="av",
    video=frames[:5],
    view_point="ego_view",
    resolution_tier=256,
)
r = pipe(prompt="", action=action, num_inference_steps=2, guidance_scale=1.0,
         generator=gen, output_type="latent", return_dict=True)
a = r.action[0]
print("id action shape:", tuple(a.shape))
assert a.shape[-1] == 9, "av raw dim should be 9"

print("MICRO-E2E: ALL PASS")
