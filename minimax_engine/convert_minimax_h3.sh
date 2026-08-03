#!/usr/bin/env bash
# One-shot MiniMax-H3 release -> diffusers-layout conversion + checkpoint assembly.
#
# Converts both partitions of the original MiniMax release (lossless: key renames and the
# reference's own QKV de-interleave only; dtypes untouched) and assembles the single
# checkpoint dir the H1111 MiniMax engine loads from:
#
#   OUT/
#     transformer/       (from FL2VA, sharded safetensors + index)
#     transformer_ref/   (from Ref2VA)
#     vae/  audio_vae/   (shared, converted once; real latents_mean/std in the configs)
#     scheduler/  audio_scheduler/
#     text_encoder/  tokenizer/  processor/   (stock Qwen3-VL, linked or copied as-is)
#
# Usage:
#   minimax_engine/convert_minimax_h3.sh /path/to/MiniMax-H3 /path/to/MiniMax-H3-diffusers [--copy-shared]
#
#   SRC must contain FL2VA/ and Ref2VA/. Needs ~135 GB free at OUT (plus ~63 GB more with
#   --copy-shared, which copies the Qwen3-VL components instead of symlinking them — use it
#   when OUT will be moved to another filesystem afterwards).
#
# Environment: run from the H1111 repo (this script resolves it from its own location).
#   PYTHON=/path/to/python overrides the interpreter (default: repo env/bin/python, then python3).
set -euo pipefail

SRC=${1:?usage: convert_minimax_h3.sh SRC_ROOT OUT_DIR [--copy-shared]}
OUT=${2:?usage: convert_minimax_h3.sh SRC_ROOT OUT_DIR [--copy-shared]}
COPY_SHARED=${3:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$REPO_ROOT/env/bin/python" ]]; then PYTHON="$REPO_ROOT/env/bin/python"; else PYTHON=python3; fi
fi
CONVERT="$SCRIPT_DIR/convert_checkpoint.py"

for part in FL2VA Ref2VA; do
    [[ -d "$SRC/$part/transformer" ]] || { echo "ERROR: $SRC/$part/transformer not found — SRC must be the release root holding FL2VA/ and Ref2VA/" >&2; exit 1; }
done
[[ -e "$OUT/transformer" ]] && { echo "ERROR: $OUT/transformer already exists — refusing to overwrite" >&2; exit 1; }

echo "== [1/5] dry-run key-mapping check (both partitions) =="
"$PYTHON" "$CONVERT" --checkpoint_path "$SRC/FL2VA" --output_path "$OUT" --dry_run | tail -8
"$PYTHON" "$CONVERT" --checkpoint_path "$SRC/Ref2VA" --output_path "$OUT" --transformer_only --dry_run | tail -8

echo "== [2/5] converting FL2VA (transformer + vae + audio_vae + schedulers) =="
"$PYTHON" "$CONVERT" --checkpoint_path "$SRC/FL2VA" --output_path "$OUT" --modular_repo_id MiniMaxAI/MiniMax-H3

echo "== [3/5] converting Ref2VA transformer -> transformer_ref =="
REF_TMP="$OUT/.ref2va_tmp"
"$PYTHON" "$CONVERT" --checkpoint_path "$SRC/Ref2VA" --output_path "$REF_TMP" --transformer_only
mv "$REF_TMP/transformer" "$OUT/transformer_ref"
rmdir "$REF_TMP"

echo "== [4/5] attaching the shared Qwen3-VL components =="
for component in text_encoder tokenizer processor; do
    if [[ "$COPY_SHARED" == "--copy-shared" ]]; then
        cp -r "$SRC/FL2VA/$component" "$OUT/$component"
    else
        ln -sfn "$(cd "$SRC/FL2VA/$component" && pwd)" "$OUT/$component"
    fi
done

echo "== [5/5] verifying the assembled checkpoint =="
"$PYTHON" - "$OUT" <<'EOF'
import glob, json, os, sys

out = sys.argv[1]
problems = []

for component, needs_shards in (
    ("transformer", True), ("transformer_ref", True), ("vae", True), ("audio_vae", True),
    ("text_encoder", True), ("tokenizer", False), ("processor", False),
    ("scheduler", False), ("audio_scheduler", False),
):
    path = os.path.join(out, component)
    if not os.path.isdir(path):
        problems.append(f"missing component dir: {component}/")
        continue
    if needs_shards and not glob.glob(os.path.join(path, "*.safetensors")):
        problems.append(f"no safetensors shards in {component}/")

vae_config = json.load(open(os.path.join(out, "vae", "config.json")))
if all(v == 0.0 for v in vae_config.get("latents_mean", [0.0])):
    problems.append("vae/config.json latents_mean is all-zero (placeholder, not the released values)")
audio_config = json.load(open(os.path.join(out, "audio_vae", "config.json")))
if not audio_config.get("latents_mean"):
    problems.append("audio_vae/config.json has no latents_mean")
te_config = json.load(open(os.path.join(out, "text_encoder", "config.json")))
layers = te_config.get("text_config", te_config).get("num_hidden_layers", 0)
if layers <= 50:
    problems.append(f"text_encoder has {layers} layers; MiniMax-H3 reads hidden_states[50] and needs > 50")
if not os.path.exists(os.path.join(out, "processor", "video_preprocessor_config.json")):
    problems.append("processor/video_preprocessor_config.json missing (ref2va video references need it)")

if problems:
    print("VERIFICATION FAILED:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)

shards = {c: len(glob.glob(os.path.join(out, c, '*.safetensors'))) for c in ("transformer", "transformer_ref", "vae", "text_encoder")}
print(f"OK: all components present ({shards['transformer']} + {shards['transformer_ref']} transformer shards, "
      f"{shards['vae']} vae shard(s), {shards['text_encoder']} text_encoder shards); "
      f"latents_mean/std populated ({len(vae_config['latents_mean'])} video / {len(audio_config['latents_mean'])} audio channels).")
print(f"\nPoint the MiniMax tab's Checkpoint Dir (or --ckpt_dir) at: {out}")
EOF

echo "== done =="
