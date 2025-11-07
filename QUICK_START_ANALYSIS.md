# Quick Start: Model Analysis

## Prerequisites

```bash
pip install torch safetensors
```

## For Your Setup

### 1. Analyze Wan2.1 Base Model (single bf16 safetensors)

```bash
python analyze_model_weights.py \
  /path/to/wan2.1_model.safetensors \
  --export wan21_keys.txt \
  --names "Wan2.1 Base (bf16)"
```

### 2. Analyze LongCat DiT (sharded fp32 diffusers format)

```bash
# Point to the dit directory containing the sharded files
python analyze_model_weights.py \
  /path/to/longcat/checkpoint/dit \
  --export longcat_keys.txt \
  --names "LongCat DiT (fp32)"
```

The script will automatically detect:
- `model.safetensors.index.json` (diffusers index)
- All `model-00001-of-XXXXX.safetensors` shard files
- Total model size across all shards

### 3. Compare Wan2.1 and LongCat

```bash
python analyze_model_weights.py \
  /path/to/wan2.1_model.safetensors \
  /path/to/longcat/checkpoint/dit \
  --compare \
  --mapping wan2_to_longcat_mapping.json \
  --names "Wan2.1 (bf16)" "LongCat (fp32)"
```

This will show you:
- Whether keys match (indicating compatibility)
- Naming convention differences
- Attention architecture (QKV vs separate Q/K/V)
- Potential key mappings for LoRA conversion

## What to Look For

### Expected: Models are architecturally similar
```
Model 1 keys: ~1500
Model 2 keys: ~1500
Common keys: ~1400
Overlap: 93%
```

**If you see this:** LoRAs should be compatible with minor key renaming!

### Expected: LongCat has extra layers
```
Model 1 keys: 1500
Model 2 keys: 1543
Keys only in Model 2: 43 (refinement/BSA-specific layers)
```

**If you see this:** Normal - LongCat has Block Sparse Attention layers that Wan2.1 doesn't have.

### Red Flag: Completely different structure
```
Model 1: Separate Q/K/V (960 keys)
Model 2: Combined QKV (320 keys)
Overlap: 15%
```

**If you see this:** LoRAs will need complex conversion or retraining.

## Understanding the Output

### Key Sections

1. **Loading**: Shows file size and loading method
   - Single file: "Loading as safetensors..."
   - Sharded: "Loading sharded model using index..."

2. **Parameter Types**: Count of weights, biases, norms

3. **Naming Patterns**: How attention/FFN layers are named

4. **Block Analysis**: Number of transformer blocks (should be same for both)

5. **Attention Structure**:
   - ✅ "Combined QKV" in both = Good
   - ✅ "Separate Q/K/V" in both = Good
   - ❌ Different patterns = Problem

6. **Sample Keys**: First and last keys (check naming conventions)

## Next Steps Based on Results

### Case 1: High Compatibility (>90% overlap)
```bash
# Keys match well - test direct LoRA loading
python longcat_generate_video.py \
  --task longcat-A14B \
  --lora_weight /path/to/wan2_lora.safetensors \
  --lora_multiplier 1.0 \
  ...
```

### Case 2: Need Key Mapping (50-90% overlap)
```bash
# Review the mapping file
cat wan2_to_longcat_mapping.json

# Create conversion script based on mappings
python create_lora_converter.py wan2_to_longcat_mapping.json
```

### Case 3: Low Compatibility (<50% overlap)
Focus on LongCat-native LoRAs:
```bash
# Use LongCat's official LoRAs
python longcat_generate_video.py \
  --task longcat-A14B \
  --ckpt_dir /path/to/longcat/checkpoint \
  --mode continuation \
  --refinement_lora_path lora/refinement_lora.safetensors
```

## Common Model Locations

Typical structure:
```
wan/
  ├── Wan_2.1_A14B_dit_bf16.safetensors  ← Single file
  └── ...

longcat_checkpoint/
  ├── dit/
  │   ├── config.json
  │   ├── model.safetensors.index.json  ← Index file
  │   ├── model-00001-of-00005.safetensors  ← Shards
  │   ├── model-00002-of-00005.safetensors
  │   └── ...
  ├── lora/
  │   ├── refinement_lora.safetensors
  │   └── cfg_step_lora.safetensors
  ├── vae/
  └── text_encoder/
```

## Troubleshooting

### "ERROR: No model files found in directory"
Make sure you're pointing to the `dit` subdirectory, not the root checkpoint directory:
```bash
# ❌ Wrong
python analyze_model_weights.py /path/to/checkpoint

# ✅ Correct
python analyze_model_weights.py /path/to/checkpoint/dit
```

### "File size: 27000 MB" - Script is slow
For very large models (>20GB), loading takes time. The script shows progress:
```
Loading shard 1/5...
Loading shard 2/5...
...
```

Just wait - it's reading from disk, not downloading.

### Memory issues
The script only loads metadata (keys and shapes), not full weights, so it should work even on low-memory systems. If you still have issues:
```bash
# Export keys without loading full shapes
python -c "
from safetensors import safe_open
with safe_open('model.safetensors', framework='pt', device='cpu') as f:
    for key in f.keys():
        print(key)
"
```

## Example Complete Workflow

```bash
#!/bin/bash

echo "Step 1: Analyzing Wan2.1..."
python analyze_model_weights.py \
  wan/Wan_2.1_A14B_dit_bf16.safetensors \
  --export analysis/wan21_keys.txt \
  --names "Wan2.1"

echo "Step 2: Analyzing LongCat..."
python analyze_model_weights.py \
  longcat_checkpoint/dit \
  --export analysis/longcat_keys.txt \
  --names "LongCat"

echo "Step 3: Comparing models..."
python analyze_model_weights.py \
  wan/Wan_2.1_A14B_dit_bf16.safetensors \
  longcat_checkpoint/dit \
  --compare \
  --mapping analysis/mapping.json \
  --names "Wan2.1" "LongCat" \
  | tee analysis/comparison.log

echo "Step 4: Review results..."
echo "Files generated:"
ls -lh analysis/

echo "Step 5: Check key overlap..."
grep "Overlap:" analysis/comparison.log

echo "Done! Review the files to determine LoRA compatibility."
```

Save this as `run_analysis.sh`, make it executable, and run it:
```bash
chmod +x run_analysis.sh
./run_analysis.sh
```
