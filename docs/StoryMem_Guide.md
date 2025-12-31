# StoryMem LoRA & Mode Guide

## Overview

StoryMem is a multi-shot video storytelling system that maintains character identity consistency across scenes using a **Memory Bank** and **Dual-DiT** architecture.

---

## Additional Dependencies

StoryMem requires these additional packages beyond the base H1111 installation:

### Required for Keyframe Extraction

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| **hpsv3** | HPSv3 quality scoring for keyframe selection | `pip install hpsv3` |
| **CLIP** | Similarity measurement between keyframes | `pip install git+https://github.com/openai/CLIP.git` |

### Required for Story Processing

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| **json5** | Parse story script JSON files | `pip install json5` |
| **decord** | Video frame extraction | `pip install decord` |

### Full Requirements (from source repository)

```
pip install hpsv3
pip install git+https://github.com/openai/CLIP.git
pip install json5 decord easydict
```

### Optional (already in base install)

These are likely already installed with H1111:
- `opencv-python>=4.9.0.80`
- `diffusers==0.32.2`
- `transformers==4.45.2`
- `peft==0.14.0`
- `accelerate>=1.1.1`
- `imageio[ffmpeg]`

---

## LoRA Files

There are **4 LoRA files** in total, organized into 2 mode-specific folders:

```
StoryMem_loras/
├── Wan2.2-MI2V-A14B/
│   ├── backbone_low_noise.safetensors   (~2.3GB)
│   └── backbone_high_noise.safetensors  (~2.3GB)
└── Wan2.2-MM2V-A14B/
    ├── backbone_low_noise.safetensors   (~2.3GB)
    └── backbone_high_noise.safetensors  (~2.3GB)
```

### LoRA Format

The original LoRAs are in **PEFT format** and must be converted before use:

| Original Key Format | Converted Key Format |
|---------------------|---------------------|
| `base_model.model.blocks.0.cross_attn.k.lora_A.weight` | `lora_unet_blocks_0_cross_attn_k.lora_down.weight` |
| `base_model.model.blocks.0.cross_attn.k.lora_B.weight` | `lora_unet_blocks_0_cross_attn_k.lora_up.weight` |

**To convert:** Use the Convert LoRA tab → select "peft to default"

### LoRA Configuration

| Property | Value |
|----------|-------|
| Rank | 128 |
| Alpha | 128 |
| Target Modules | self_attn (q,k,v,o), cross_attn (q,k,v,o), ffn (0,2) |
| Use RSLoRA | True |

---

## Generation Modes

### First Shot Generation

| Mode | Flag | Description |
|------|------|-------------|
| **T2V First Shot** | `--t2v_first_shot` | Generate first shot with Text-to-Video model. No input required. |
| **M2V First Shot** | `--m2v_first_shot` | Generate first shot using reference images as initial memory. Requires pre-placed keyframe images. |

**For M2V First Shot**, place reference images in output directory as:
```
00_00_keyframe0.jpg
00_00_keyframe1.jpg
00_00_keyframe2.jpg
...
```

### Transition Modes (for subsequent shots)

| Mode | Flag | LoRA Folder | Description |
|------|------|-------------|-------------|
| **MI2V** | `--mi2v` | `Wan2.2-MI2V-A14B/` | Memory + **last frame** of previous shot |
| **MM2V** | `--mm2v` | `Wan2.2-MM2V-A14B/` | Memory + **5 motion frames** from previous shot |

**Choose ONE transition mode** - they are mutually exclusive.

- **MI2V**: Simpler, uses 1 frame for transition
- **MM2V**: Smoother motion continuity, uses 5 frames

---

## Dual-DiT Architecture

StoryMem uses two DiT models that switch during the denoising process:

| Model | When Used | LoRA |
|-------|-----------|------|
| **High Noise Model** | timestep ≥ 0.9 (90%) | `backbone_high_noise.safetensors` |
| **Low Noise Model** | timestep < 0.9 (90%) | `backbone_low_noise.safetensors` |

The **boundary** (default 0.9) determines when to switch from high noise to low noise model during denoising.

### Denoising Flow
```
Start (t=1.0)
    │
    │  High Noise Model + High Noise LoRA
    │  (Early denoising: coarse structure)
    │
    ▼
Boundary (t=0.9)
    │
    │  Low Noise Model + Low Noise LoRA
    │  (Late denoising: fine details)
    │
    ▼
End (t=0.0)
```

---

## How to Configure LoRAs in H1111 GUI

### For MI2V Mode:

1. Convert LoRAs (if not already done):
   - Convert LoRA tab → Input: `Wan2.2-MI2V-A14B/backbone_low_noise.safetensors` → Target: "peft to default"
   - Convert LoRA tab → Input: `Wan2.2-MI2V-A14B/backbone_high_noise.safetensors` → Target: "peft to default"

2. In StoryMem tab:
   - Check "MI2V Transitions" ✓
   - Uncheck "MM2V Transitions"
   - LoRA 1: Select converted low noise LoRA → "Apply to Low Noise" ✓
   - LoRA 2: Select converted high noise LoRA → "Apply to High Noise" ✓

### For MM2V Mode:

1. Convert LoRAs:
   - Convert `Wan2.2-MM2V-A14B/backbone_low_noise.safetensors`
   - Convert `Wan2.2-MM2V-A14B/backbone_high_noise.safetensors`

2. In StoryMem tab:
   - Check "MM2V Transitions" ✓
   - Uncheck "MI2V Transitions"
   - LoRA 1: Select converted low noise LoRA → "Apply to Low Noise" ✓
   - LoRA 2: Select converted high noise LoRA → "Apply to High Noise" ✓

---

## Memory Bank

The memory bank stores keyframes from generated videos to maintain identity consistency.

| Setting | Default | Description |
|---------|---------|-------------|
| Max Memory Size | 8 | Maximum keyframes kept in memory |
| Fixed Keyframes | 3 | Number of initial keyframes always kept |
| Keyframes per Video | 3 | Max keyframes extracted from each shot |
| Similarity Threshold | 0.9 | CLIP similarity threshold for keyframe selection |
| Quality Threshold | 3.0 | HPSv3 quality threshold |

### Memory Bank Strategy (Sliding Window)

```
If memory_bank size > max_memory_size:
    memory_bank = first N fixed keyframes + most recent keyframes
```

This ensures the initial character identity is preserved while also using recent context.

---

## Complete Example

### Example 1: T2V First Shot + MI2V Transitions

```
1. First Shot: T2V generates video from text prompt
       ↓
   Extract keyframes → Add to Memory Bank
       ↓
2. Shot 2: M2V + MI2V (memory + last frame from Shot 1)
       ↓
   Extract keyframes → Update Memory Bank
       ↓
3. Shot 3: M2V + MI2V (memory + last frame from Shot 2)
       ↓
   ... continue for all shots ...
       ↓
4. Concatenate all shots → Final video
```

**Scene Cuts (`cut: true`)**: No transition frames used, only memory bank
**Smooth Transitions (`cut: false`)**: Memory bank + last frame (MI2V) or 5 motion frames (MM2V)

### Example 2: M2V First Shot (Reference Images)

```
1. Upload reference images → Saved as 00_00_keyframe0.jpg, etc.
       ↓
2. First Shot: M2V generates video using reference images as memory
       ↓
   Extract keyframes → Add to Memory Bank
       ↓
3. Continue with MI2V or MM2V transitions...
```

---

## Story JSON Format

```json
{
  "story_name": "My Story",
  "story_overview": "Brief description of the story...",
  "scenes": [
    {
      "scene_num": 1,
      "video_prompts": [
        "Shot 1 description...",
        "Shot 2 description..."
      ],
      "first_frame_prompt": [
        "First frame of shot 1...",
        "First frame of shot 2..."
      ],
      "cut": [true, false]
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `scene_num` | Scene number (1-indexed) |
| `video_prompts` | Array of video descriptions for each shot |
| `first_frame_prompt` | Array of first frame descriptions (for I2V conditioning) |
| `cut` | Array of booleans - `true` = scene cut, `false` = smooth transition |

---

## Sample Solvers

StoryMem only supports:
- `unipc` (default, recommended)
- `dpm++`

Other solvers (vanilla, euler, step_distill) are **not supported**.

---

## Quick Reference

| If you want... | Use Mode | LoRA Folder |
|----------------|----------|-------------|
| Simple transitions, less VRAM | MI2V | `Wan2.2-MI2V-A14B/` |
| Smoother motion continuity | MM2V | `Wan2.2-MM2V-A14B/` |
| Auto-generate first shot | T2V First Shot | N/A (uses T2V model) |
| Control initial character appearance | M2V First Shot + Reference Images | MI2V or MM2V folder |
