# Model Analysis Guide

This guide explains how to use the analysis scripts to understand model compatibility and LoRA conversion requirements.

## Scripts Overview

### 1. `analyze_model_weights.py`
Comprehensive model checkpoint analysis tool.

**Features:**
- Load and analyze model checkpoints (.safetensors, .pth, .bin, .ckpt)
- Display weight naming conventions and structure
- Identify transformer block organization
- Detect attention mechanism architecture (QKV vs Q/K/V separate)
- Compare two models and find differences
- Generate potential key mappings between models

### 2. `analyze_lora_keys.py`
LoRA-specific analysis tool.

**Features:**
- Analyze LoRA safetensors files
- Display key naming patterns
- Compare two LoRA files
- Show parameter statistics

## Usage Examples

### Analyze a Single Model

```bash
# Analyze Wan2.1 base model
python analyze_model_weights.py /path/to/wan2.1_model.safetensors

# Analyze LongCat model
python analyze_model_weights.py /path/to/longcat/dit/model.safetensors

# With custom name
python analyze_model_weights.py /path/to/model.safetensors --names "Wan2.1 Base"
```

### Compare Two Models

```bash
# Compare Wan2.1 and LongCat
python analyze_model_weights.py \
  /path/to/wan2.1_model.safetensors \
  /path/to/longcat/dit/model.safetensors \
  --compare \
  --names "Wan2.1" "LongCat"

# Generate mapping file
python analyze_model_weights.py \
  /path/to/wan2.1_model.safetensors \
  /path/to/longcat/dit/model.safetensors \
  --compare \
  --mapping wan2_to_longcat_mapping.json
```

### Export Key Lists

```bash
# Export single model keys
python analyze_model_weights.py /path/to/model.safetensors --export model_keys.txt

# Export and compare multiple models
python analyze_model_weights.py \
  model1.safetensors model2.safetensors \
  --compare \
  --export keys_export.txt
```

### Analyze LoRA Files

```bash
# Analyze single LoRA
python analyze_lora_keys.py lora/my_lora.safetensors

# Compare Wan2 LoRA vs LongCat LoRA
python analyze_lora_keys.py \
  lora/wan2_lora.safetensors \
  /path/to/longcat/lora/refinement_lora.safetensors \
  --compare \
  --names "Wan2 LoRA" "LongCat LoRA"
```

## Understanding the Output

### Model Structure Analysis

The script will show:

1. **Total Parameters**: Number of weight tensors
2. **Parameter Types**: Weights, biases, normalization layers
3. **Naming Patterns**: How many blocks use attention, feedforward, etc.
4. **Block Structure**: Number of transformer blocks and their organization
5. **Attention Architecture**: Whether it uses:
   - Combined QKV (single matrix for Q, K, V)
   - Separate Q/K/V (individual matrices)
   - Q + combined KV (Q separate, K and V together)

### Comparison Output

When comparing two models:

1. **Key Statistics**:
   - Total keys in each model
   - Common keys (exact matches)
   - Keys only in model 1 or 2
   - Overlap percentage

2. **Shape Comparison**:
   - Which keys have matching shapes
   - Which have different dimensions

3. **Key Mapping Attempts**:
   - Potential mappings based on shape and name similarity
   - Confidence levels (high/medium/low)

4. **Naming Conventions**:
   - Separator styles (dots, underscores, etc.)
   - Case conventions

## Interpreting Results for LoRA Compatibility

### Scenario 1: Models Have Identical Keys
```
✓ EXACT KEY MATCH - Models have identical structure!
Matching shapes: 1500/1500
```
**Result**: LoRAs trained on one model will work directly on the other.

### Scenario 2: Models Have Different Keys But Similar Structure
```
Model 1 keys: 1500
Model 2 keys: 1543
Common keys: 150
Overlap: 10%
```
**Result**: LoRAs need conversion. Check the mapping file to see if automatic conversion is feasible.

### Scenario 3: Models Have Different Architectures
```
Model 1: Separate Q/K/V (960 keys)
Model 2: Combined QKV (240 keys)
```
**Result**: LoRAs are **not directly compatible**. Would require:
- Architecture-specific training
- Complex conversion involving reshaping and merging/splitting

## What to Look For

### For Wan2.1 → LongCat Compatibility

Compare these aspects:

1. **Key Naming Convention**:
   - Wan2: `blocks.0.attn.q_proj.weight`
   - LongCat: `blocks.0.attn.qkv.weight` or similar

2. **Attention Structure**:
   - Check if both use same Q/K/V organization
   - Look for combined vs separate projections

3. **Block Count**:
   - Number of transformer blocks should match
   - Layer names should have similar structure

4. **Weight Shapes**:
   - Attention head dimension
   - Hidden dimension
   - FFN intermediate dimension

### Red Flags for Incompatibility

- ❌ Different number of blocks/layers
- ❌ Different attention mechanisms (QKV fused vs separate)
- ❌ Different hidden dimensions
- ❌ Completely different naming schemes with no pattern

### Green Flags for Compatibility

- ✅ Same number of blocks
- ✅ Similar key structure (even if prefixes differ)
- ✅ Matching weight shapes
- ✅ Same attention architecture

## Next Steps Based on Results

### If Compatible (>80% overlap):
1. Document the key mapping
2. Test loading a LoRA directly
3. Verify outputs are reasonable

### If Partially Compatible (30-80% overlap):
1. Review the generated mapping file
2. Create a conversion script using the mapping
3. Test converted LoRAs carefully

### If Incompatible (<30% overlap):
1. Train new LoRAs specifically for LongCat
2. Or use LongCat's native LoRA loading (already works)
3. Focus on LongCat-native LoRAs (refinement_lora, cfg_step_lora)

## Example Workflow

```bash
# Step 1: Analyze both base models
echo "=== Analyzing Wan2.1 Base Model ==="
python analyze_model_weights.py \
  /path/to/wan2.1/dit.safetensors \
  --export wan21_keys.txt \
  --names "Wan2.1"

echo "=== Analyzing LongCat Base Model ==="
python analyze_model_weights.py \
  /path/to/longcat/dit/model.safetensors \
  --export longcat_keys.txt \
  --names "LongCat"

# Step 2: Compare the models
echo "=== Comparing Models ==="
python analyze_model_weights.py \
  /path/to/wan2.1/dit.safetensors \
  /path/to/longcat/dit/model.safetensors \
  --compare \
  --mapping wan2_to_longcat_mapping.json \
  --names "Wan2.1" "LongCat"

# Step 3: Review outputs
echo "=== Review these files ==="
ls -lh wan21_keys.txt longcat_keys.txt wan2_to_longcat_mapping.json

# Step 4: Based on results, decide on LoRA strategy
```

## Troubleshooting

### "ERROR: torch is required"
```bash
pip install torch
```

### "WARNING: safetensors not available"
```bash
pip install safetensors
```

### "Failed to load as PyTorch checkpoint"
The file might be in a different format. Try:
```bash
# Check file type
file /path/to/model.safetensors

# Try different loading with PyTorch directly
python -c "import torch; print(torch.load('model.pth', map_location='cpu').keys())"
```

### Script is slow for large models
Large models (>10GB) may take time to load. The script loads to CPU to avoid GPU memory issues. Be patient or:
- Use a machine with more RAM
- Analyze LoRAs instead (much smaller)
- Use `--export` to save keys, then analyze the text file

## Additional Resources

- LongCat official repo: Check `LongCat-Video/` for reference implementations
- Wan2 model docs: See model architecture specifications
- LoRA paper: Understanding rank-adaptation methods
