#!/usr/bin/env python3
"""
Analyze LoRA usage to understand why certain keys aren't being applied
"""

import os
import sys
import argparse
from pathlib import Path
from safetensors import safe_open
import torch
from collections import defaultdict

def load_model_keys(model_path):
    """Load all keys from a model file"""
    model_keys = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            model_keys[key] = tensor.shape
    return model_keys

def load_lora_keys(lora_path):
    """Load all keys from a LoRA file"""
    lora_keys = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            lora_keys[key] = tensor.shape
    return lora_keys

def convert_lora_key_to_model_key(lora_key):
    """
    Convert LoRA key to model key format - exact copy of logic from utils/lora_utils.py
    """
    # Handle _img suffix by stripping it
    original_key = lora_key
    if '_img' in lora_key:
        lora_key = lora_key.replace('_img', '')

    # Handle MUSUBI format (lora_unet_ prefix)
    if lora_key.startswith('lora_unet_'):
        key = lora_key[len('lora_unet_'):]

        # Parse blocks.N pattern specially
        if key.startswith('blocks_'):
            parts = key.split('_')

            if len(parts) >= 4:  # blocks, N, module, param, ...
                block_num = parts[1]

                # Find the suffix and extract the middle part
                if '_diff_b' in key:
                    suffix_start = key.rfind('_diff_b')
                    # Get everything between blocks_N_ and _diff_b
                    param_part = key[len(f'blocks_{block_num}_'):suffix_start]

                    # Handle module boundaries (cross_attn, self_attn, ffn need dots after them)
                    if 'cross_attn_' in param_part:
                        param_part = param_part.replace('cross_attn_', 'cross_attn.')
                    if 'self_attn_' in param_part:
                        param_part = param_part.replace('self_attn_', 'self_attn.')
                    if 'ffn_' in param_part:
                        param_part = param_part.replace('ffn_', 'ffn.')

                    return f"blocks.{block_num}.{param_part}.bias"

                elif '_diff_m' in key:
                    suffix_start = key.rfind('_diff_m')
                    param_part = key[len(f'blocks_{block_num}_'):suffix_start]

                    if 'cross_attn_' in param_part:
                        param_part = param_part.replace('cross_attn_', 'cross_attn.')
                    if 'self_attn_' in param_part:
                        param_part = param_part.replace('self_attn_', 'self_attn.')
                    if 'ffn_' in param_part:
                        param_part = param_part.replace('ffn_', 'ffn.')

                    return f"blocks.{block_num}.{param_part}.modulation"

                elif '_diff' in key:
                    suffix_start = key.rfind('_diff')
                    param_part = key[len(f'blocks_{block_num}_'):suffix_start]

                    if 'cross_attn_' in param_part:
                        param_part = param_part.replace('cross_attn_', 'cross_attn.')
                    if 'self_attn_' in param_part:
                        param_part = param_part.replace('self_attn_', 'self_attn.')
                    if 'ffn_' in param_part:
                        param_part = param_part.replace('ffn_', 'ffn.')

                    return f"blocks.{block_num}.{param_part}.weight"
        else:
            # Handle non-block keys (like embedding, time_in, etc.)
            if key.endswith('_diff_b'):
                param = key[:-7]  # Remove _diff_b
                # Replace underscores with dots for proper module hierarchy
                param = param.replace('_', '.')
                return f"{param}.bias"
            elif key.endswith('_diff_m'):
                param = key[:-7]  # Remove _diff_m
                param = param.replace('_', '.')
                return f"{param}.modulation"
            elif key.endswith('_diff'):
                param = key[:-5]  # Remove _diff
                param = param.replace('_', '.')
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

def simulate_lora_matching(model_keys, lora_keys):
    """Simulate the exact LoRA matching logic from utils/lora_utils.py"""

    matched_keys = defaultdict(list)  # model_key -> [lora_keys]
    unmatched_lora = set(lora_keys.keys())

    print("\n" + "="*80)
    print("Simulating LoRA Matching Logic")
    print("="*80)

    # Track different types of keys
    diff_keys = [k for k in lora_keys if 'diff' in k and 'diff_b' not in k and 'diff_m' not in k and 'lora_down' not in k]
    diff_b_keys = [k for k in lora_keys if 'diff_b' in k]
    diff_m_keys = [k for k in lora_keys if 'diff_m' in k]
    lora_down_keys = [k for k in lora_keys if 'lora_down' in k or 'lora_down_weight' in k]
    lora_up_keys = [k for k in lora_keys if 'lora_up' in k or 'lora_up_weight' in k]
    alpha_keys = [k for k in lora_keys if '.alpha' in k]

    print(f"\nLoRA key types:")
    print(f"  diff keys: {len(diff_keys)}")
    print(f"  diff_b keys: {len(diff_b_keys)}")
    print(f"  diff_m keys: {len(diff_m_keys)}")
    print(f"  lora_down keys: {len(lora_down_keys)}")
    print(f"  lora_up keys: {len(lora_up_keys)}")
    print(f"  alpha keys: {len(alpha_keys)}")

    # Simulate diff/diff_b matching
    print("\n--- Testing diff/diff_b matching ---")
    for lora_key in diff_keys + diff_b_keys + diff_m_keys:
        converted = convert_lora_key_to_model_key(lora_key)
        if converted and converted in model_keys:
            matched_keys[converted].append(lora_key)
            if lora_key in unmatched_lora:
                unmatched_lora.remove(lora_key)

    # Simulate standard LoRA matching
    print("\n--- Testing standard LoRA matching ---")
    for model_key in model_keys:
        if not model_key.endswith('.weight'):
            continue

        base_key = model_key.rsplit(".", 1)[0]
        if base_key.startswith("diffusion_model."):
            base_key = base_key[len("diffusion_model."):]

        lora_base_key = "lora_unet_" + base_key.replace(".", "_")

        # MUSUBI format
        down_key = lora_base_key + "_lora_down_weight"
        up_key = lora_base_key + "_lora_up_weight"
        alpha_key = down_key + ".alpha"

        # Also check _img variants
        down_key_img = lora_base_key + "_img_lora_down_weight"
        up_key_img = lora_base_key + "_img_lora_up_weight"
        alpha_key_img = down_key_img + ".alpha"

        # Check regular pair
        if down_key in lora_keys and up_key in lora_keys:
            matched_keys[model_key].extend([down_key, up_key])
            if down_key in unmatched_lora:
                unmatched_lora.remove(down_key)
            if up_key in unmatched_lora:
                unmatched_lora.remove(up_key)
            if alpha_key in lora_keys and alpha_key in unmatched_lora:
                unmatched_lora.remove(alpha_key)
                matched_keys[model_key].append(alpha_key)

        # Check _img variant pair
        elif down_key_img in lora_keys and up_key_img in lora_keys:
            matched_keys[model_key].extend([down_key_img, up_key_img])
            if down_key_img in unmatched_lora:
                unmatched_lora.remove(down_key_img)
            if up_key_img in unmatched_lora:
                unmatched_lora.remove(up_key_img)
            if alpha_key_img in lora_keys and alpha_key_img in unmatched_lora:
                unmatched_lora.remove(alpha_key_img)
                matched_keys[model_key].append(alpha_key_img)

    return matched_keys, unmatched_lora

def analyze_unmatched_keys(unmatched_keys):
    """Analyze patterns in unmatched keys"""
    print("\n" + "="*80)
    print("Analyzing Unmatched Keys")
    print("="*80)

    # Group by pattern
    patterns = defaultdict(list)
    for key in unmatched_keys:
        # Identify pattern
        if '.alpha' in key:
            if '_img' in key:
                patterns['alpha_img'].append(key)
            else:
                patterns['alpha_regular'].append(key)
        elif 'lora_down' in key:
            if '_img' in key:
                patterns['lora_down_img'].append(key)
            else:
                patterns['lora_down_regular'].append(key)
        elif 'lora_up' in key:
            if '_img' in key:
                patterns['lora_up_img'].append(key)
            else:
                patterns['lora_up_regular'].append(key)
        elif 'diff_b' in key:
            patterns['diff_b'].append(key)
        elif 'diff_m' in key:
            patterns['diff_m'].append(key)
        elif 'diff' in key:
            patterns['diff'].append(key)
        else:
            patterns['other'].append(key)

    print(f"\nUnmatched keys by pattern:")
    for pattern, keys in sorted(patterns.items()):
        if keys:
            print(f"\n{pattern}: {len(keys)} keys")
            # Show first 5 examples
            for key in keys[:5]:
                print(f"  - {key}")
            if len(keys) > 5:
                print(f"  ... and {len(keys)-5} more")

    # Special analysis for alpha keys
    alpha_img_keys = patterns.get('alpha_img', [])
    if alpha_img_keys:
        print("\n" + "-"*40)
        print("Special Analysis: Unmatched _img alpha keys")
        print("-"*40)

        # Extract the base keys these alphas belong to
        base_keys = set()
        for alpha_key in alpha_img_keys:
            # Remove .alpha suffix
            base = alpha_key.replace('.alpha', '')
            base_keys.add(base)

        print(f"These alpha keys belong to {len(base_keys)} unique LoRA weights")
        print("\nChecking if their corresponding down/up weights exist:")

        for base in list(base_keys)[:5]:  # Check first 5
            print(f"\n  Base: {base}")
            # The base IS the down weight key
            # Up weight would be base.replace('_lora_down_weight', '_lora_up_weight')
            up_key = base.replace('_lora_down_weight', '_lora_up_weight')
            print(f"    Expected up key: {up_key}")

    return patterns

def main():
    parser = argparse.ArgumentParser(description='Analyze LoRA usage and matching')
    parser.add_argument('model_path', type=str, help='Path to model file')
    parser.add_argument('lora_path', type=str, help='Path to LoRA file')
    parser.add_argument('--verbose', action='store_true', help='Show detailed matching')
    parser.add_argument('--test-keys', action='store_true', help='Test specific key conversions')

    args = parser.parse_args()

    # Load files
    print(f"Loading model: {args.model_path}")
    model_keys = load_model_keys(args.model_path)
    print(f"  Found {len(model_keys)} model keys")

    print(f"\nLoading LoRA: {args.lora_path}")
    lora_keys = load_lora_keys(args.lora_path)
    print(f"  Found {len(lora_keys)} LoRA keys")

    # Test specific keys if requested
    if args.test_keys:
        print("\n" + "="*80)
        print("Testing Specific Key Conversions")
        print("="*80)

        # Test problem keys from the unmatched list
        test_lora_keys = [
            'lora_unet_blocks_21_cross_attn_o_diff_b',
            'lora_unet_blocks_14_self_attn_q_diff_b',
            'lora_unet_blocks_8_self_attn_norm_q_diff',
            'lora_unet_blocks_7_cross_attn_norm_k_diff',
        ]

        print("\nTesting LoRA key conversions:")
        for lora_key in test_lora_keys:
            converted = convert_lora_key_to_model_key(lora_key)
            exists = converted in model_keys if converted else False
            print(f"\nLoRA: {lora_key}")
            print(f"  → Model: {converted}")
            print(f"  → Exists in model: {exists}")

        # Check what similar keys DO exist in the model
        print("\n" + "-"*40)
        print("Checking what blocks.21.cross_attn keys exist in model:")
        found_21 = False
        for key in sorted(model_keys.keys()):
            if 'blocks.21' in key and 'cross_attn' in key:
                print(f"  {key}")
                found_21 = True
        if not found_21:
            print("  NONE - Block 21 doesn't exist in model!")

        print("\nChecking what blocks.8.self_attn keys exist in model:")
        found_8 = False
        for key in sorted(model_keys.keys()):
            if 'blocks.8' in key and 'self_attn' in key:
                print(f"  {key}")
                found_8 = True
        if not found_8:
            print("  NONE - Block 8 doesn't exist in model!")

        # Find the maximum block number in model
        print("\n" + "-"*40)
        print("Finding maximum block number in model:")
        max_block = -1
        for key in model_keys.keys():
            if 'blocks.' in key:
                # Extract block number
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part == 'blocks' and i+1 < len(parts):
                        try:
                            block_num = int(parts[i+1])
                            max_block = max(max_block, block_num)
                        except ValueError:
                            pass
        print(f"  Maximum block number in model: {max_block}")

        # Find block numbers in LoRA
        print("\nFinding block numbers in LoRA:")
        lora_blocks = set()
        for key in lora_keys.keys():
            if 'blocks_' in key:
                # Extract block number from LoRA key
                parts = key.split('_')
                for i, part in enumerate(parts):
                    if part == 'blocks' and i+1 < len(parts):
                        try:
                            block_num = int(parts[i+1])
                            lora_blocks.add(block_num)
                        except ValueError:
                            pass
        print(f"  LoRA has modifications for blocks: {sorted(lora_blocks)}")
        print(f"  LoRA max block: {max(lora_blocks) if lora_blocks else -1}")
        print(f"  LoRA blocks beyond model: {sorted([b for b in lora_blocks if b > max_block])}")

        return  # Exit after test

    # Simulate matching
    matched, unmatched = simulate_lora_matching(model_keys, lora_keys)

    # Report results
    print("\n" + "="*80)
    print("Matching Results")
    print("="*80)

    total_lora_matched = sum(len(v) for v in matched.values())
    print(f"\nModel keys with LoRA matches: {len(matched)}/{len(model_keys)}")
    print(f"LoRA keys matched: {total_lora_matched}/{len(lora_keys)}")
    print(f"LoRA keys unmatched: {len(unmatched)}")

    if args.verbose and matched:
        print("\n--- Sample Matches ---")
        for model_key, lora_list in list(matched.items())[:5]:
            print(f"\nModel: {model_key}")
            for lora_key in lora_list:
                print(f"  ← {lora_key}")

    # Analyze unmatched
    if unmatched:
        patterns = analyze_unmatched_keys(unmatched)

        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"\nTotal unmatched LoRA keys: {len(unmatched)}")

        # Identify the main issue
        if patterns.get('alpha_img'):
            print(f"\n⚠️  Main issue: {len(patterns['alpha_img'])} _img alpha keys are not being matched")
            print("   These are scaling factors for image-specific LoRA weights")

        if patterns.get('lora_down_img') or patterns.get('lora_up_img'):
            down_count = len(patterns.get('lora_down_img', []))
            up_count = len(patterns.get('lora_up_img', []))
            print(f"\n⚠️  Also unmatched: {down_count} _img lora_down and {up_count} _img lora_up keys")
            print("   These image-specific LoRA weights aren't finding model matches")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()