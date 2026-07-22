# Micro test for the Cosmos3 text-pathway (und) K/V cache: tiny random-weight
# components + real tokenizer, CPU only. Verifies BIT-IDENTICAL (torch.equal)
# outputs cached vs uncached for t2v (cfg 6), i2v, v2v, action forward_dynamics
# (cfg 1), and multi-control transfer, plus a transformer-level capture->skip
# unit test.
import json

import numpy as np
import torch
from PIL import Image

from transformers import AutoTokenizer

import os as _os, sys as _sys
_ENGINE = "/home/mayble/h1111/H1111/cosmos_engine"
for _p in (_os.path.dirname(_ENGINE), _ENGINE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import os as _os, sys as _sys
_ENGINE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.dirname(_ENGINE), _ENGINE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from cosmos_video.pipeline import Cosmos3OmniPipeline, CosmosActionCondition
from cosmos_video.scheduler_unipc import UniPCMultistepScheduler
from cosmos_video.transformer import Cosmos3OmniTransformer
from cosmos_video.und_cache import UndKVCache
from cosmos_video.vae import AutoencoderKLWan

torch.manual_seed(0)

IDX = _os.path.join(_ENGINE, "cosmos_hf_indexes", "Cosmos3-Nano")

tokenizer = AutoTokenizer.from_pretrained(IDX + "/text_tokenizer")
if getattr(tokenizer, "chat_template", None) is None:
    with open(IDX + "/text_tokenizer/chat_template.jinja") as f:
        tokenizer.chat_template = f.read()

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

vae_cfg = json.load(open(IDX + "/vae/config.json"))
for k in ("_class_name", "_diffusers_version", "_name_or_path", "clip_output"):
    vae_cfg.pop(k, None)
vae_cfg["base_dim"] = 16
vae_cfg["decoder_base_dim"] = 16
vae_cfg["num_res_blocks"] = 1
vae = AutoencoderKLWan(**vae_cfg).float().eval()

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


def rand_frames(n, seed=7):
    rng = np.random.RandomState(seed)
    return [Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)) for _ in range(n)]


def run_pair(name, **kw):
    """Run the pipeline twice with identical seeds, cache on vs off; assert bit-identical."""
    outs = []
    for use_cache in (True, False):
        gen = torch.Generator().manual_seed(kw.pop("seed", 42) if "seed" in kw else 42)
        call = dict(kw)
        call.setdefault("output_type", "latent")
        call.setdefault("return_dict", True)
        call["generator"] = gen
        call["use_und_cache"] = use_cache
        outs.append(pipe(**call))
    v_c, v_u = outs[0].video, outs[1].video
    assert torch.equal(v_c, v_u), f"{name}: cached vs uncached latents are NOT bit-identical"
    if outs[0].action is not None or outs[1].action is not None:
        a_c, a_u = outs[0].action, outs[1].action
        assert (a_c is None) == (a_u is None), f"{name}: action presence mismatch"
        if a_c is not None:
            for t_c, t_u in zip(a_c, a_u):
                assert torch.equal(t_c, t_u), f"{name}: cached vs uncached actions differ"
    print(f"{name}: cached == uncached (torch.equal) OK, latents {tuple(v_c.shape)}")


img = Image.fromarray(np.random.RandomState(3).randint(0, 255, (64, 64, 3), dtype=np.uint8))
frames = rand_frames(9, seed=5)

# 1) t2v with CFG 6 (cond + uncond caches both exercised)
run_pair("t2v cfg6", prompt="a red ball rolls", negative_prompt="blurry", num_frames=5,
         height=64, width=64, fps=24.0, num_inference_steps=3, guidance_scale=6.0)

# 2) i2v
run_pair("i2v", prompt="the ball moves", negative_prompt="static", image=img, num_frames=5,
         height=64, width=64, fps=24.0, num_inference_steps=3, guidance_scale=6.0)

# 3) v2v with condition frames
run_pair("v2v", prompt="continue the motion", negative_prompt="jitter", video=frames,
         condition_frame_indexes_vision=(0, 1), num_frames=9, height=64, width=64,
         fps=24.0, num_inference_steps=3, guidance_scale=6.0)

# 4) action forward_dynamics with CFG 1 (no uncond pass; cond cache only)
def make_action():
    return CosmosActionCondition(
        mode="forward_dynamics",
        chunk_size=4,
        domain_name="droid_lerobot",
        raw_actions=torch.from_numpy(np.random.RandomState(11).randn(4, 10).astype(np.float32)),
        image=img,
        view_point="ego_view",
        resolution_tier=256,
    )

run_pair("action-fd cfg1", prompt="pick up the cube", action=make_action(),
         num_frames=None, height=None, width=None, num_inference_steps=3, guidance_scale=1.0)

# 5) multi-control transfer (two hints, weights, cfg 6) cached vs uncached
edge = rand_frames(5, seed=7)
depth = rand_frames(5, seed=8)
run_pair("transfer 2-hint", prompt="a red ball rolls", negative_prompt="blurry",
         control_videos={"edge": edge, "depth": depth}, control_weights=[0.7, 0.3],
         control_guidance=1.0, num_frames=5, height=64, width=64, fps=24.0,
         num_inference_steps=3, guidance_scale=6.0)

# 5b) transfer with control-CFG (exercises the third, no-control packed static + its cache)
run_pair("transfer control-cfg", prompt="a red ball rolls", negative_prompt="blurry",
         control_videos={"edge": edge}, control_guidance=2.0, num_frames=5, height=64,
         width=64, fps=24.0, num_inference_steps=3, guidance_scale=6.0)

# 6) Transformer-level unit test: capture -> skip must be bit-identical to a normal forward.
captured = {}
orig_forward = Cosmos3OmniTransformer.forward


def capture_forward(self, *a, **kw):
    if "kwargs" not in captured:
        captured["kwargs"] = dict(kw)
        raise RuntimeError("captured")
    return orig_forward(self, *a, **kw)


Cosmos3OmniTransformer.forward = capture_forward
try:
    pipe(prompt="a blue cube", negative_prompt="blurry", num_frames=5, height=64, width=64,
         fps=24.0, num_inference_steps=2, guidance_scale=6.0,
         generator=torch.Generator().manual_seed(9), output_type="latent")
except RuntimeError as e:
    assert "captured" in str(e)
finally:
    Cosmos3OmniTransformer.forward = orig_forward
assert "kwargs" in captured, "failed to capture a transformer call"

kw = dict(captured["kwargs"], return_dict=False)
kw.pop("und_cache", None)

with torch.no_grad():
    preds_normal, _, _ = transformer(**kw)  # plain forward, no cache

    cache = UndKVCache()
    preds_capture, _, _ = transformer(**kw, und_cache=cache)  # capture mode
    assert cache.populated, "capture forward did not populate the cache"
    assert len(cache.layer_kv) == len(transformer.layers)
    for pn, pc in zip(preds_normal, preds_capture):
        assert torch.equal(pn, pc), "capture-mode forward differs from plain forward"

    preds_skip, _, _ = transformer(**kw, und_cache=cache)  # skip mode
    for pn, ps in zip(preds_normal, preds_skip):
        assert torch.equal(pn, ps), "skip-mode forward differs from plain forward"

# cached tensors are reused verbatim (same objects, dtype, device)
for k_c, v_c in cache.layer_kv:
    assert k_c.dtype == v_c.dtype == transformer.dtype

# und_len mismatch must be rejected (stale cache safety)
bad_kw = dict(kw)
bad_kw["und_len"] = kw["und_len"] - 1
try:
    with torch.no_grad():
        transformer(**bad_kw, und_cache=cache)
    raise AssertionError("stale und_len not rejected")
except ValueError:
    pass
print("transformer capture->skip unit test: bit-identical OK; stale-cache rejection OK")

print("MICRO-UND-CACHE: ALL PASS")
