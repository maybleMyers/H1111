# Micro end-to-end Cosmos3 TRANSFER test: tiny random-weight components + real
# tokenizer, CPU only. Exercises multi-control packing, multi-control two-way
# attention, control weights, control-CFG, the N=1 fast-path equivalence, and
# the script-level chunked transfer loop.
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
from cosmos_video.transformer import Cosmos3OmniTransformer
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


def run(seed=42, **kw):
    gen = torch.Generator().manual_seed(seed)
    common = dict(prompt="a red ball rolls", negative_prompt="blurry", num_frames=5, height=64, width=64,
                  fps=24.0, num_inference_steps=2, guidance_scale=6.0, generator=gen,
                  output_type="latent", return_dict=True)
    common.update(kw)
    return pipe(**common)


edge = rand_frames(5, seed=7)
depth = rand_frames(5, seed=8)

# (a) single-hint transfer, control_guidance=1.0
r_a = run(control_videos={"edge": edge}, control_guidance=1.0)
print("(a) single-hint latents:", tuple(r_a.video.shape))
assert torch.isfinite(r_a.video).all()
assert r_a.video.shape == (1, 48, 2, 4, 4)

# transfer changes the output vs plain t2v (controls actually attended)
r_t2v = run()
assert not torch.allclose(r_a.video, r_t2v.video), "transfer output identical to t2v — controls ignored?"

# (b) two hints with weights [0.7, 0.3]
r_b = run(control_videos={"edge": edge, "depth": depth}, control_weights=[0.7, 0.3], control_guidance=1.0)
print("(b) two-hint latents:", tuple(r_b.video.shape))
assert torch.isfinite(r_b.video).all()
# weights matter: [0.3, 0.7] must differ from [0.7, 0.3]
r_b2 = run(control_videos={"edge": edge, "depth": depth}, control_weights=[0.3, 0.7], control_guidance=1.0)
assert not torch.allclose(r_b.video, r_b2.video), "control weights had no effect"

# (c) control_guidance=2.0 runs and differs from 1.0 at the same seed
r_c = run(control_videos={"edge": edge}, control_guidance=2.0)
print("(c) control-CFG latents:", tuple(r_c.video.shape))
assert torch.isfinite(r_c.video).all()
assert not torch.allclose(r_a.video, r_c.video), "control_guidance=2.0 did not change the output"

# control-CFG interval that excludes every timestep must reduce to control_guidance=1.0
r_c_gated = run(control_videos={"edge": edge}, control_guidance=2.0, control_guidance_interval=(0.0, 0.5))
assert torch.equal(r_c_gated.video, r_a.video), "fully-gated control-CFG should equal control_guidance=1.0"

# share_vision_temporal_positions=False also runs (per-item sequential temporal offsets)
r_ns = run(control_videos={"edge": edge}, share_vision_temporal_positions=False)
assert torch.isfinite(r_ns.video).all()
assert not torch.allclose(r_ns.video, r_a.video), "temporal-position regime had no effect"

# (d) N=1 equivalence: the multi-control attention path (vision_item_spans set) must
# reduce EXACTLY to the two-way fast path (spans=None) on the identical packed inputs.
captured = {}
orig_forward = Cosmos3OmniTransformer.forward


def capture_forward(self, *a, **kw):
    if "kwargs" not in captured and kw.get("vision_item_spans") is not None:
        captured["kwargs"] = {k: v for k, v in kw.items()}
        raise RuntimeError("captured")
    return orig_forward(self, *a, **kw)


Cosmos3OmniTransformer.forward = capture_forward
try:
    run(control_videos={"edge": edge}, control_weights=[1.0], control_guidance=1.0)
except RuntimeError as e:
    assert "captured" in str(e)
finally:
    Cosmos3OmniTransformer.forward = orig_forward
assert "kwargs" in captured, "failed to capture a transfer transformer call"

kw_multi = dict(captured["kwargs"], return_dict=False)
with torch.no_grad():
    preds_multi, _, _ = transformer(**kw_multi)
kw_fast = dict(kw_multi)
kw_fast["vision_item_spans"] = None
kw_fast["control_weights"] = None
with torch.no_grad():
    preds_fast, _, _ = transformer(**kw_fast)
diff = (preds_multi[-1] - preds_fast[-1]).abs().max().item()
print("(d) N=1 multi-control vs fast-path max |diff|:", diff)
assert torch.allclose(preds_multi[-1], preds_fast[-1], atol=1e-5), "N=1 multi-control does not reduce to two-way"

# transfer composes with v2v-style conditioning (chunk conditioning path)
cond_clip = rand_frames(1, seed=9)
r_cond = run(control_videos={"edge": edge}, video=cond_clip, condition_frame_indexes_vision=(0,))
assert torch.isfinite(r_cond.video).all()

# rejects transfer+sound and transfer+action
try:
    run(control_videos={"edge": edge}, enable_sound=True)
    raise AssertionError("transfer+sound not rejected")
except ValueError:
    pass
try:
    action = CosmosActionCondition(mode="policy", chunk_size=4, domain_name="av",
                                   image=rand_frames(1)[0], resolution_tier=256)
    run(control_videos={"edge": edge}, action=action, num_frames=None, height=None, width=None)
    raise AssertionError("transfer+action not rejected")
except ValueError:
    pass
try:
    run(control_videos={"bogus": edge})
    raise AssertionError("unknown hint not rejected")
except ValueError:
    pass
print("rejection checks: OK")

# (e) chunked transfer: 12-frame control, num_frames_per_chunk=9, num_conditional_frames=1
from cosmos_generate_video import compute_transfer_chunk_plan, decode_latents, run_transfer_chunks

total = 12
chunk_frames, num_chunks, stride, spans = compute_transfer_chunk_plan(total, 9, 1)
assert (chunk_frames, num_chunks, stride) == (9, 2, 8) and spans == [(0, 9), (8, 12)]
control_12 = rand_frames(total, seed=11)
chunk_calls = []


def generate_chunk(chunk_id, control_chunk, cond_frames, condition_frame_indexes):
    chunk_calls.append((chunk_id, len(control_chunk["edge"]),
                        None if cond_frames is None else len(cond_frames), condition_frame_indexes))
    kw = dict(control_videos=control_chunk, control_guidance=1.0, num_frames=chunk_frames)
    if cond_frames is not None:
        kw["video"] = cond_frames
        kw["condition_frame_indexes_vision"] = tuple(condition_frame_indexes)
    latents = run(seed=42 + chunk_id, **kw).video
    return decode_latents(latents, vae)


out = run_transfer_chunks({"edge": control_12}, None, total, chunk_frames, stride, num_chunks, 1, 0, generate_chunk)
print("(e) chunked output frames:", out.shape, "calls:", chunk_calls)
assert out.shape == (12, 64, 64, 3), out.shape
assert chunk_calls == [(0, 9, None, None), (1, 4, 1, [0])], chunk_calls

print("MICRO-TRANSFER: ALL PASS")
