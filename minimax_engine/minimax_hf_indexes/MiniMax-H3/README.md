# MiniMax-H3 checked-in component configs (no weights)

Component configs for `MiniMaxAI/MiniMax-H3`, derived from the diffusers PR
huggingface/diffusers#14355 @ e1b518d (`scripts/convert_minimax_h3_to_diffusers.py`).
They let the loader and the dummy-weight tests run before any checkpoint download.

**Must be regenerated from the published HF repo once weights are available** (Phase 5):

- `vae/config.json` `latents_mean` / `latents_std` are identity placeholders — the real
  per-channel values live in the original `video_vae/config.json` and are required for
  correct generation.
- `audio_vae/config.json` `latents_mean` / `latents_std` are `null` placeholders — the real
  values live in the original `audio_vae/config.json`.
- `*.safetensors.index.json` shard indexes and the `text_encoder/`, `tokenizer/`,
  `processor/` configs are not checked in yet; copy them from the HF snapshot.
- `transformer_ref/` shares the architecture config with `transformer/`; only the weights
  differ.
