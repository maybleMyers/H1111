# Weight Value Analysis Guide

## Quick Start

Since LongCat was trained from Wan2.1, the actual weight VALUES should show similarity even though the key NAMES don't match.

### Run the Analysis

```bash
# This will compare actual weight values between the models
python analyze_weight_values.py \
  wan/wan2.1_t2v_14B_bf16.safetensors \
  /home/mayble/diffusion/LongCat-Video/dit \
  --names "Wan2.1" "LongCat" \
  --sample-size 20 \
  --output weight_similarity_report.json
```

### Memory Constraints

If you run out of memory loading 27GB + 50GB of weights:

```bash
# Load only first N weights for analysis
python analyze_weight_values.py \
  wan/wan2.1_t2v_14B_bf16.safetensors \
  /home/mayble/diffusion/LongCat-Video/dit \
  --names "Wan2.1" "LongCat" \
  --max-weights 200 \
  --sample-size 10 \
  --output weight_similarity_report.json
```

## What It Does

### 1. Shape Matching
Finds weights with identical shapes between models:
```
Shape: (5120, 5120)
  Model 1: 160 weights (Q/K/V projections)
  Model 2: 96 weights (combined QKV)
```

### 2. Value Comparison
For each matching shape, compares actual values:
- **Cosine similarity**: How aligned are the weight vectors? (1.0 = identical)
- **Correlation**: Statistical correlation (1.0 = perfect)
- **Relative L2 difference**: How different are the magnitudes?

### 3. Block Analysis
Compares transformer blocks layer by layer:
```
Block 0:
  Model 1: 27 weights
  Model 2: 21 weights

  Potential matches:
    blocks.0.self_attn.q.weight → blocks.0.attn.qkv.weight
    Cosine: 0.9876
```

## Expected Results

### Scenario 1: Weights Were Directly Copied
```
Mean cosine similarity: 0.998
High similarity (>0.95): 150/150 matches
✓ MATCH: blocks.0.cross_attn.q.weight → blocks.0.cross_attn.q_linear.weight
  Cosine similarity: 0.999999
```
**Interpretation**: Models share the same trained weights with just renaming. LoRA conversion is straightforward!

### Scenario 2: Weights Were Transformed
```
Mean cosine similarity: 0.850
Medium similarity (0.8-0.95): 120/150 matches

blocks.0.self_attn.q.weight (5120, 5120)
  → blocks.0.attn.qkv.weight[0:5120] (12288, 4096)
  Cosine: 0.887 (after reshape)
```
**Interpretation**: Weights were reorganized (e.g., Q/K/V merged into QKV). LoRA conversion requires reshaping.

### Scenario 3: Weights Were Retrained
```
Mean cosine similarity: 0.234
Low similarity (<0.8): 145/150 matches

No clear weight correspondence found.
```
**Interpretation**: LongCat was fine-tuned significantly or trained differently. Need new LoRAs.

## Key Questions This Answers

### Q1: Are the base weights the same?
**Look at**: Mean cosine similarity
- \>0.95 = Essentially identical
- 0.80-0.95 = Similar but transformed
- <0.80 = Different training

### Q2: Can we map Q/K/V to QKV?
**Look at**: Shape analysis
```
# If Wan2.1 has:
blocks.0.self_attn.q.weight: (5120, 5120)
blocks.0.self_attn.k.weight: (5120, 5120)
blocks.0.self_attn.v.weight: (5120, 5120)

# And LongCat has:
blocks.0.attn.qkv.weight: (15360, 4096)  # 3*5120 = 15360

# Then Q/K/V were likely CONCATENATED into QKV
```

### Q3: Can LoRAs be converted?
**Look at**: Block correspondence
- If weights match with just renaming → Simple key remapping
- If weights are reshaped → Need reshape logic
- If weights are different → Train new LoRAs

## Understanding the Output

### High Similarity (Cosine > 0.95)
```
✓ MATCH
  blocks.5.cross_attn.o.weight
  → blocks.5.cross_attn.proj.weight
  Cosine similarity: 0.998765
  Correlation: 0.998821
  Relative L2 diff: 0.002134
```
**Meaning**: These are essentially the same weights! Just renamed `o` → `proj`.

### Medium Similarity (Cosine 0.80-0.95)
```
~ PARTIAL MATCH
  blocks.3.ffn.0.weight
  → blocks.3.ffn.w1.weight
  Cosine similarity: 0.876543
```
**Meaning**: Weights are related but not identical. Possibly:
- Underwent additional training
- Were partially merged/split
- Precision conversion affected values

### Low Similarity (Cosine < 0.80)
```
✗ Different
  blocks.10.modulation
  → blocks.10.adaLN_modulation.1.weight
  Cosine similarity: 0.234567
```
**Meaning**: These are different weights. Either:
- Wrong correspondence (not actually the same layer)
- LongCat added new functionality here
- Significantly retrained

## Next Steps Based on Results

### Case 1: High Similarity Found (>0.95)
```bash
# Create LoRA key mapping
python create_lora_converter.py \
  --weight-analysis weight_similarity_report.json \
  --output wan2_to_longcat_lora_converter.py
```

### Case 2: Medium Similarity (0.80-0.95)
- Review the shape transformations needed
- Check if LoRAs need value adjustments beyond renaming
- Test conversion with a small LoRA first

### Case 3: Low Similarity (<0.80)
- Focus on LongCat-native LoRAs
- Train new LoRAs on LongCat base
- Use existing implementation (already complete!)

## Memory Usage Notes

- Full analysis: ~30GB RAM (loading both 27GB + 50GB models)
- With `--max-weights 200`: ~5GB RAM (sufficient for analysis)
- With `--max-weights 50`: ~1GB RAM (quick check)

The script loads weights incrementally and can stop at any limit.

## Technical Details

### Weight Comparison Metrics

1. **Cosine Similarity**:
   - Range: -1 to 1 (1 = identical direction)
   - Formula: `dot(w1, w2) / (|w1| * |w2|)`
   - Best for: Detecting same weights with different scales

2. **Correlation**:
   - Range: -1 to 1 (1 = perfect correlation)
   - Formula: Pearson correlation coefficient
   - Best for: Detecting linear relationships

3. **Relative L2 Difference**:
   - Range: 0 to ∞ (0 = identical)
   - Formula: `|w1 - w2| / |w1|`
   - Best for: Measuring exact differences

### Shape Matching Strategy

The script:
1. Groups all weights by shape
2. Finds common shapes between models
3. Compares values for weights with matching shapes
4. Uses multiple metrics to determine similarity
5. Identifies best matches per weight

## Example Full Workflow

```bash
#!/bin/bash

echo "Step 1: Analyze weight values..."
python analyze_weight_values.py \
  wan/wan2.1_t2v_14B_bf16.safetensors \
  /home/mayble/diffusion/LongCat-Video/dit \
  --names "Wan2.1" "LongCat" \
  --sample-size 20 \
  --output weight_similarity_report.json

echo "Step 2: Review report..."
python -c "
import json
with open('weight_similarity_report.json') as f:
    report = json.load(f)
    print('Mean similarity:', report['summary']['mean_cosine_similarity'])
    print('High matches:', report['summary']['high_similarity_count'])
"

echo "Step 3: Based on similarity, decide next action..."
# If high similarity: Create converter
# If medium: Investigate transformations
# If low: Use LongCat-native LoRAs
```

## Troubleshooting

### "Out of memory"
Use `--max-weights` to limit memory usage:
```bash
python analyze_weight_values.py ... --max-weights 100
```

### Analysis is slow
- Normal for large models (27GB + 50GB)
- Each weight comparison does matrix operations
- Consider using `--max-weights` and `--sample-size` to speed up

### Can't interpret results
Look at the JSON report:
```bash
cat weight_similarity_report.json | jq '.summary'
```

Key field: `mean_cosine_similarity`
- Close to 1.0? Weights are same!
- Around 0.8-0.9? Transformed but related
- Below 0.7? Different models
