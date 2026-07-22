# Sound-path micro test: joint vision+sound denoise + audio decode on CPU with
# tiny random weights and the real tokenizer/chat template. Exercises the sound
# latent packing, the per-modality sound scheduler, decode_sound, and the
# script's wav writer + shapes — everything short of real-weight audio quality.
import json
import sys

import numpy as np
import torch


from transformers import AutoTokenizer

import os as _os, sys as _sys
_ENGINE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.dirname(_ENGINE), _ENGINE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from cosmos_video.pipeline import Cosmos3OmniPipeline
from cosmos_video.scheduler_unipc import UniPCMultistepScheduler
from cosmos_video.sound_tokenizer import Cosmos3AVAEAudioTokenizer
from cosmos_video.transformer import Cosmos3OmniTransformer
from cosmos_video.vae import AutoencoderKLWan

torch.manual_seed(0)
IDX = _os.path.join(_ENGINE, "cosmos_hf_indexes", "Cosmos3-Nano")

tokenizer = AutoTokenizer.from_pretrained(IDX + "/text_tokenizer")
if getattr(tokenizer, "chat_template", None) is None:
    with open(IDX + "/text_tokenizer/chat_template.jinja") as f:
        tokenizer.chat_template = f.read()

# tiny sound tokenizer: real structure (oobleck/spec_convnext/vae/snakebeta), shrunk dims
sound_cfg = json.load(open(IDX + "/sound_tokenizer/config.json"))
sound_cfg.update(
    enc_dim=16, enc_intermediate_dim=32, enc_num_layers=2, enc_num_blocks=1,
    dec_dim=16, dec_c_mults=[1, 1, 2, 2, 4],
)
sound_tok = Cosmos3AVAEAudioTokenizer(**sound_cfg).float().eval()
print("tiny sound tokenizer params:", sum(p.numel() for p in sound_tok.parameters()) / 1e6, "M")
print("hop size:", sound_tok._hop_size if hasattr(sound_tok, "_hop_size") else sound_cfg["hop_size"])

transformer = Cosmos3OmniTransformer(
    hidden_size=64, intermediate_size=128, head_dim=16,
    num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
    latent_channel=48, latent_patch_size=2, patch_latent_dim=192,
    vocab_size=len(tokenizer),
    sound_gen=True, sound_dim=64, sound_latent_fps=25.0,
    rope_scaling={"mrope_interleaved": True, "mrope_section": [8, 4, 4], "rope_type": "default"},
).float().eval()

vae_cfg = json.load(open(IDX + "/vae/config.json"))
for k in ("_class_name", "_diffusers_version", "_name_or_path", "clip_output"):
    vae_cfg.pop(k, None)
vae_cfg.update(base_dim=16, decoder_base_dim=16, num_res_blocks=1)
vae = AutoencoderKLWan(**vae_cfg).float().eval()

sched_cfg = json.load(open(IDX + "/scheduler/scheduler_config.json"))
sched_cfg.pop("_class_name", None); sched_cfg.pop("_diffusers_version", None)

pipe = Cosmos3OmniPipeline(
    transformer=transformer,
    text_tokenizer=tokenizer,
    vae=vae,
    scheduler=UniPCMultistepScheduler(**sched_cfg),
    sound_tokenizer=sound_tok,
    default_use_system_prompt=False,
    use_native_flow_schedule=True,
)

gen = torch.Generator().manual_seed(7)
r = pipe(prompt="a bell ringing", negative_prompt="silence", num_frames=5, height=64, width=64,
         fps=24.0, num_inference_steps=2, guidance_scale=6.0, enable_sound=True,
         generator=gen, output_type="latent", return_dict=True)

print("video latents:", tuple(r.video.shape))
assert torch.isfinite(r.video).all()
assert r.sound is not None, "sound output missing"
sound = r.sound
print("decoded sound:", tuple(sound.shape), "dtype", sound.dtype)
assert torch.isfinite(sound).all()

# expected duration: 5 frames @ 24fps = 0.2083s -> ~10000 samples @ 48kHz;
# decode upsamples latents by hop_size, so allow one hop of slack
expected = int(5 / 24.0 * sound_cfg["sampling_rate"])
n_samples = sound.shape[-1]
assert abs(n_samples - expected) <= sound_cfg["hop_size"], (n_samples, expected)
ch = sound.shape[-2] if sound.ndim >= 2 else 1
assert ch == 2, f"expected stereo, got {ch} channels"
print(f"duration check: {n_samples} samples (expected ~{expected}), stereo OK")

# exercise the script's wav writer + mux input shape handling
from cosmos_generate_video import save_audio_wav

wav_path = _os.path.join(__import__("tempfile").gettempdir(), "cosmos_micro_sound.wav")
save_audio_wav(sound, wav_path, sound_cfg["sampling_rate"])
import wave

with wave.open(wav_path, "rb") as w:
    assert w.getnchannels() == 2 and w.getframerate() == 48000
    assert w.getnframes() == n_samples
print("wav writer: OK")

# latent round trip through decode_sound directly
sl = torch.randn(64, 6)
decoded = pipe.decode_sound(sl)
assert torch.isfinite(decoded).all()
print("decode_sound direct:", tuple(decoded.shape))

print("MICRO-SOUND: ALL PASS")
