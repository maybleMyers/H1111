#!/usr/bin/env python3
"""
Analyze LoRA and Model key mismatch to understand why LoRA is not being applied
"""

import os
import sys
import argparse
from pathlib import Path
from safetensors import safe_open
import torch
import re

def load_model_keys(model_path):
    """Load model keys from a safetensors file"""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return []

    keys = []
    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    return keys

def load_lora_keys(lora_path):
    """Load LoRA keys from a safetensors file"""
    if not os.path.exists(lora_path):
        print(f"Error: LoRA file not found: {lora_path}")
        return {}

    state_dict = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict

def convert_lora_key_to_model_key(lora_key):
    """
    Convert LoRA key to expected model key format
    Based on the lora_utils.py conversion logic
    """
    # Handle _img suffix by stripping it
    if '_img' in lora_key:
        lora_key = lora_key.replace('_img', '')

    # Handle MUSUBI format (lora_unet_ prefix)
    if lora_key.startswith('lora_unet_'):
        key = lora_key[len('lora_unet_'):]

        # Parse blocks.N pattern specially
        if key.startswith('blocks_'):
            # Split only first two underscores for block number
            parts = key.split('_', 2)  # ['blocks', '0', 'cross_attn_k_diff_b']
            if len(parts) >= 3:
                block_part = f"blocks.{parts[1]}"
                rest = parts[2]

                # Determine parameter type
                if rest.endswith('_diff_b'):
                    param = rest[:-7]  # Remove _diff_b
                    return f"{block_part}.{param}.bias"
                elif rest.endswith('_diff_m'):
                    param = rest[:-7]  # Remove _diff_m
                    return f"{block_part}.{param}.modulation"
                elif rest.endswith('_diff'):
                    param = rest[:-5]  # Remove _diff
                    return f"{block_part}.{param}.weight"
        else:
            # Handle non-block keys (like embedding, time_in, etc.)
            if key.endswith('_diff_b'):
                param = key[:-7]  # Remove _diff_b
                return f"{param}.bias"
            elif key.endswith('_diff_m'):
                param = key[:-7]  # Remove _diff_m
                return f"{param}.modulation"
            elif key.endswith('_diff'):
                param = key[:-5]  # Remove _diff
                return f"{param}.weight"

    # Handle original lightx2v format (diffusion_model. prefix)
    elif lora_key.startswith('diffusion_model.'):
        key = lora_key[len('diffusion_model.'):]

        if key.endswith('.diff_b'):
            return key[:-7] + '.bias'
        elif key.endswith('.diff_m'):
            return key[:-7] + '.modulation'
        elif key.endswith('.diff'):
            return key[:-5] + '.weight'

    return None

def convert_model_key_to_lora_format(model_key):
    """
    Convert model key to expected LoRA key formats
    This is the reverse of what lora_utils.py expects
    """
    formats = []

    # Remove .weight/.bias/.modulation suffix
    base_key = model_key
    suffix_type = None
    if model_key.endswith('.weight'):
        base_key = model_key[:-7]
        suffix_type = 'weight'
    elif model_key.endswith('.bias'):
        base_key = model_key[:-5]
        suffix_type = 'bias'
    elif model_key.endswith('.modulation'):
        base_key = model_key[:-11]
        suffix_type = 'modulation'

    # For standard LoRA format (what the LoRA file actually uses)
    # Just use the base key directly for lora_down/lora_up
    if suffix_type == 'weight':
        # Direct format (what the LoRA actually has)
        formats.append(f"{base_key}.lora_down.weight")
        formats.append(f"{base_key}.lora_up.weight")
        formats.append(f"{base_key}.alpha")

        # MUSUBI format with lora_unet_ prefix
        lora_base = "lora_unet_" + base_key.replace(".", "_")
        formats.append(f"{lora_base}.lora_down.weight")
        formats.append(f"{lora_base}.lora_up.weight")
        formats.append(f"{lora_base}.alpha")

        # Underscores variant
        formats.append(f"{lora_base}_lora_down_weight")
        formats.append(f"{lora_base}_lora_up_weight")

        # _img variant
        formats.append(f"{lora_base}_img_lora_down_weight")
        formats.append(f"{lora_base}_img_lora_up_weight")

    return formats

def analyze_mismatch(lora_path, model_path):
    """Analyze why LoRA keys don't match model keys"""
    print(f"\n{'='*80}")
    print(f"Analyzing LoRA-Model Key Mismatch")
    print(f"{'='*80}\n")
    print(f"LoRA: {lora_path}")
    print(f"Model: {model_path}")
    print()

    # Load keys
    lora_state_dict = load_lora_keys(lora_path)
    model_keys = load_model_keys(model_path)

    if not lora_state_dict or not model_keys:
        print("Error: Could not load files")
        return

    lora_keys = list(lora_state_dict.keys())

    print(f"LoRA keys: {len(lora_keys)}")
    print(f"Model keys: {len(model_keys)}")
    print()

    # Analyze LoRA key format
    print("LoRA Key Format Analysis:")
    print("-" * 40)

    # Check what format the LoRA uses
    sample_lora_keys = lora_keys[:5]
    print("Sample LoRA keys:")
    for key in sample_lora_keys:
        print(f"  {key}")
    print()

    # Check model key format
    print("Model Key Format Analysis:")
    print("-" * 40)

    # Find model keys that should have LoRA
    model_weight_keys = [k for k in model_keys if k.endswith('.weight')]
    sample_model_keys = model_weight_keys[:5]
    print("Sample model weight keys:")
    for key in sample_model_keys:
        print(f"  {key}")
    print()

    # Try to match LoRA keys to model keys
    print("Key Matching Analysis:")
    print("-" * 40)

    # Track matches
    matched_lora_keys = set()
    unmatched_lora_keys = set()

    # For each model weight key, check if corresponding LoRA exists
    potential_matches = 0
    actual_matches = 0

    for model_key in model_weight_keys:
        if not model_key.endswith('.weight'):
            continue

        # Get expected LoRA key formats
        expected_formats = convert_model_key_to_lora_format(model_key)

        # Check if any expected format exists in LoRA
        found_match = False
        for expected in expected_formats:
            if expected in lora_keys:
                found_match = True
                matched_lora_keys.add(expected)
                break

        if 'blocks.' in model_key and ('cross_attn' in model_key or 'self_attn' in model_key or 'ffn' in model_key):
            potential_matches += 1
            if found_match:
                actual_matches += 1

    print(f"Potential model keys for LoRA: {potential_matches}")
    print(f"Actually matched LoRA keys: {actual_matches}")
    print()

    # Check for the actual format mismatch
    print("Format Mismatch Detection:")
    print("-" * 40)

    # The LoRA uses direct model key format (blocks.0.cross_attn.k.lora_down.weight)
    # But lora_utils.py expects lora_unet_ prefix format

    direct_format_count = 0
    musubi_format_count = 0

    for key in lora_keys:
        if key.startswith('blocks.') and '.lora_' in key:
            direct_format_count += 1
        elif key.startswith('lora_unet_'):
            musubi_format_count += 1

    print(f"LoRA keys in direct format (blocks.X.layer.lora_*): {direct_format_count}")
    print(f"LoRA keys in MUSUBI format (lora_unet_*): {musubi_format_count}")

    if direct_format_count > 0 and musubi_format_count == 0:
        print("\n⚠️  ISSUE IDENTIFIED: LoRA uses direct model key format but lora_utils.py expects lora_unet_ prefix!")
        print("   The LoRA file has keys like: blocks.0.cross_attn.k.lora_down.weight")
        print("   But the code expects: lora_unet_blocks_0_cross_attn_k.lora_down.weight")

        print("\n📝 SOLUTION: The lora_utils.py weight_hook_func needs to be updated to handle direct format.")
        print("   Need to add a check for keys that start with 'blocks.' directly.")

    # Show unmatched LoRA keys
    unmatched_lora_keys = set(lora_keys) - matched_lora_keys
    if unmatched_lora_keys:
        print(f"\nUnmatched LoRA keys: {len(unmatched_lora_keys)}")
        print("Sample unmatched LoRA keys:")
        for key in list(unmatched_lora_keys)[:5]:
            print(f"  {key}")

def main():
    parser = argparse.ArgumentParser(description='Analyze LoRA-Model key mismatch')
    parser.add_argument('lora_path', type=str, help='Path to LoRA file')
    parser.add_argument('--model', type=str, default='wan/wan22_i2v_14B_high_noise_bf16.safetensors',
                       help='Path to model file (default: wan/wan22_i2v_14B_high_noise_bf16.safetensors)')

    args = parser.parse_args()

    analyze_mismatch(args.lora_path, args.model)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()