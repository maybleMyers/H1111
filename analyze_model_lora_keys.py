#!/usr/bin/env python3
"""
Analyze model and LoRA key structures to debug matching issues
"""

import os
import sys
import argparse
from pathlib import Path
from safetensors import safe_open
import torch
from collections import defaultdict

def analyze_model_structure(model_path):
    """Analyze the structure of a model file"""
    print(f"\n{'='*80}")
    print(f"Analyzing Model: {model_path}")
    print(f"{'='*80}\n")

    if not os.path.exists(model_path):
        print(f"Error: File not found: {model_path}")
        return {}

    # Load model keys
    model_keys = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            model_keys[key] = tensor.shape

    print(f"Total keys: {len(model_keys)}\n")

    # Categorize by type
    weight_keys = []
    bias_keys = []
    other_keys = []

    for key in sorted(model_keys.keys()):
        if key.endswith('.weight'):
            weight_keys.append(key)
        elif key.endswith('.bias'):
            bias_keys.append(key)
        else:
            other_keys.append(key)

    print(f"Weight parameters: {len(weight_keys)}")
    print(f"Bias parameters: {len(bias_keys)}")
    print(f"Other parameters: {len(other_keys)}")

    # Show sample keys
    print("\nSample weight keys (first 10):")
    for key in weight_keys[:10]:
        print(f"  {key}")

    print("\nSample bias keys (first 10):")
    for key in bias_keys[:10]:
        print(f"  {key}")

    return model_keys

def analyze_lora_structure(lora_path):
    """Analyze the structure of a LoRA file"""
    print(f"\n{'='*80}")
    print(f"Analyzing LoRA: {lora_path}")
    print(f"{'='*80}\n")

    if not os.path.exists(lora_path):
        print(f"Error: File not found: {lora_path}")
        return {}

    # Load LoRA keys
    lora_keys = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            lora_keys[key] = tensor.shape

    print(f"Total keys: {len(lora_keys)}\n")

    # Categorize by type
    categories = defaultdict(list)

    for key in sorted(lora_keys.keys()):
        if 'lora_down' in key:
            categories['lora_down'].append(key)
        elif 'lora_up' in key:
            categories['lora_up'].append(key)
        elif 'alpha' in key:
            categories['alpha'].append(key)
        elif 'diff_b' in key:
            categories['diff_b'].append(key)
        elif 'diff' in key:
            categories['diff'].append(key)
        else:
            categories['other'].append(key)

    print("Key categories:")
    for cat, keys in categories.items():
        print(f"  {cat}: {len(keys)} keys")

    # Show sample keys from each category
    for cat, keys in categories.items():
        if keys:
            print(f"\nSample {cat} keys (first 5):")
            for key in keys[:5]:
                print(f"    {key}")

    return lora_keys

def find_matching_patterns(model_keys, lora_keys):
    """Find potential matching patterns between model and LoRA keys"""
    print(f"\n{'='*80}")
    print("Analyzing Key Matching Patterns")
    print(f"{'='*80}\n")

    # Extract base names from model keys (removing .weight/.bias)
    model_bases = set()
    model_weights = set()
    model_biases = set()

    for key in model_keys:
        if key.endswith('.weight'):
            base = key.rsplit('.', 1)[0]
            model_bases.add(base)
            model_weights.add(key)
        elif key.endswith('.bias'):
            base = key.rsplit('.', 1)[0]
            model_bases.add(base)
            model_biases.add(key)

    print(f"Unique model base keys: {len(model_bases)}")
    print(f"Model weight keys: {len(model_weights)}")
    print(f"Model bias keys: {len(model_biases)}")

    print("\nSample model base keys (first 10):")
    for base in sorted(model_bases)[:10]:
        print(f"  {base}")

    # Try to match LoRA keys to model keys
    print("\n" + "="*60)
    print("Checking LoRA key mapping patterns:")
    print("="*60)

    # Check diff keys
    diff_keys = [k for k in lora_keys if 'diff' in k and 'diff_b' not in k and 'diff_m' not in k]
    diff_b_keys = [k for k in lora_keys if 'diff_b' in k]

    # Statistics
    matched_diff = 0
    unmatched_diff = []
    matched_diff_b = 0
    unmatched_diff_b = []

    if diff_keys:
        print(f"\n{len(diff_keys)} diff keys found")
        print("Checking mapping for diff keys:")

        for diff_key in diff_keys:
            found_match = False

            # Try to find corresponding model key
            # Pattern 1: diffusion_model.X.diff -> X.weight
            if diff_key.startswith('diffusion_model.') and diff_key.endswith('.diff'):
                potential_model_key = diff_key[len('diffusion_model.'):-len('.diff')] + '.weight'
                if potential_model_key in model_keys:
                    matched_diff += 1
                    found_match = True
                    if matched_diff <= 5:  # Show first 5 matches
                        print(f"\n  LoRA key: {diff_key}")
                        print(f"    ✓ Matches model key: {potential_model_key}")

            # Pattern 2: lora_unet_X_diff -> X.weight (with underscores to dots)
            elif 'lora_unet_' in diff_key and '_diff' in diff_key:
                # Handle both .diff and _diff endings
                if diff_key.endswith('.diff'):
                    base = diff_key[len('lora_unet_'):-len('.diff')]
                elif diff_key.endswith('_diff'):
                    base = diff_key[len('lora_unet_'):-len('_diff')]
                else:
                    base = None

                if base:
                    potential_model_key = base.replace('_', '.') + '.weight'
                    if potential_model_key in model_keys:
                        matched_diff += 1
                        found_match = True
                        if matched_diff <= 5:  # Show first 5 matches
                            print(f"\n  LoRA key: {diff_key}")
                            print(f"    ✓ Matches model key: {potential_model_key}")

            if not found_match:
                unmatched_diff.append(diff_key)

        print(f"\n  Summary: {matched_diff}/{len(diff_keys)} diff keys matched")
        if unmatched_diff:
            print(f"  First 5 unmatched diff keys:")
            for key in unmatched_diff[:5]:
                print(f"    - {key}")

    if diff_b_keys:
        print(f"\n{len(diff_b_keys)} diff_b keys found")
        print("Checking mapping for diff_b keys:")

        for diff_b_key in diff_b_keys:
            found_match = False

            # Try to find corresponding model key
            # Pattern 1: diffusion_model.X.diff_b -> X.bias
            if diff_b_key.startswith('diffusion_model.') and diff_b_key.endswith('.diff_b'):
                potential_model_key = diff_b_key[len('diffusion_model.'):-len('.diff_b')] + '.bias'
                if potential_model_key in model_keys:
                    matched_diff_b += 1
                    found_match = True
                    if matched_diff_b <= 5:  # Show first 5 matches
                        print(f"\n  LoRA key: {diff_b_key}")
                        print(f"    ✓ Matches model key: {potential_model_key}")

            # Pattern 2: lora_unet_X_diff_b -> X.bias (with underscores to dots)
            elif 'lora_unet_' in diff_b_key and 'diff_b' in diff_b_key:
                # Handle both .diff_b and _diff_b endings
                if diff_b_key.endswith('.diff_b'):
                    base = diff_b_key[len('lora_unet_'):-len('.diff_b')]
                elif diff_b_key.endswith('_diff_b'):
                    base = diff_b_key[len('lora_unet_'):-len('_diff_b')]
                else:
                    base = None

                if base:
                    potential_model_key = base.replace('_', '.') + '.bias'
                    if potential_model_key in model_keys:
                        matched_diff_b += 1
                        found_match = True
                        if matched_diff_b <= 5:  # Show first 5 matches
                            print(f"\n  LoRA key: {diff_b_key}")
                            print(f"    ✓ Matches model key: {potential_model_key}")

            if not found_match:
                unmatched_diff_b.append(diff_b_key)

        print(f"\n  Summary: {matched_diff_b}/{len(diff_b_keys)} diff_b keys matched")
        if unmatched_diff_b:
            print(f"  First 5 unmatched diff_b keys:")
            for key in unmatched_diff_b[:5]:
                print(f"    - {key}")

    # Check standard LoRA patterns
    lora_down_keys = [k for k in lora_keys if 'lora_down' in k]
    if lora_down_keys:
        print(f"\n{len(lora_down_keys)} lora_down keys found")
        print("Checking mapping for standard LoRA keys:")

        for lora_key in lora_down_keys[:5]:
            print(f"\n  LoRA key: {lora_key}")

            # Pattern 1: diffusion_model.X.lora_down.weight -> X.weight
            if lora_key.startswith('diffusion_model.') and '.lora_down.weight' in lora_key:
                potential_model_key = lora_key.replace('diffusion_model.', '').replace('.lora_down.weight', '.weight')
                if potential_model_key in model_keys:
                    print(f"    ✓ Matches model key: {potential_model_key}")
                else:
                    print(f"    ✗ Expected model key not found: {potential_model_key}")

            # Pattern 2: lora_unet_X.lora_down.weight -> X.weight
            elif lora_key.startswith('lora_unet_') and '.lora_down' in lora_key:
                base = lora_key[len('lora_unet_'):].replace('.lora_down.weight', '').replace('.lora_down_weight', '')
                potential_model_key = base.replace('_', '.') + '.weight'
                if potential_model_key in model_keys:
                    print(f"    ✓ Matches model key: {potential_model_key}")
                else:
                    print(f"    ✗ Expected model key not found: {potential_model_key}")

def main():
    parser = argparse.ArgumentParser(description='Analyze model and LoRA key structures')
    parser.add_argument('model_path', type=str, help='Path to model file')
    parser.add_argument('lora_path', type=str, help='Path to LoRA file')
    parser.add_argument('--verbose', action='store_true', help='Show all keys')

    args = parser.parse_args()

    # Analyze model
    model_keys = analyze_model_structure(args.model_path)

    # Analyze LoRA
    lora_keys = analyze_lora_structure(args.lora_path)

    # Find matching patterns
    if model_keys and lora_keys:
        find_matching_patterns(model_keys, lora_keys)

    # Special analysis for _img keys
    print("\n" + "="*80)
    print("Special Analysis: _img Keys")
    print("="*80)

    img_keys = [k for k in lora_keys if '_img' in k]
    if img_keys:
        print(f"\nFound {len(img_keys)} keys with '_img' suffix")
        print("These are image-specific keys for I2V models.")
        print("Sample _img keys:")
        for key in img_keys[:5]:
            print(f"  - {key}")

        # Check if model has corresponding non-img keys
        print("\nChecking if model has non-_img versions:")
        for img_key in img_keys[:5]:
            # Try to construct non-img version
            if 'diffusion_model.' in img_key:
                non_img_key = img_key.replace('_img', '')
                non_img_model_key = non_img_key.replace('diffusion_model.', '')
                if non_img_model_key in model_keys:
                    print(f"  ✓ {img_key} -> {non_img_model_key}")
                else:
                    # Try as weight/bias
                    if '.diff_b' in non_img_key:
                        test_key = non_img_key.replace('diffusion_model.', '').replace('.diff_b', '.bias')
                    elif '.diff' in non_img_key:
                        test_key = non_img_key.replace('diffusion_model.', '').replace('.diff', '.weight')
                    elif '.lora_down.weight' in non_img_key:
                        test_key = non_img_key.replace('diffusion_model.', '').replace('.lora_down.weight', '.weight')
                    else:
                        test_key = None

                    if test_key and test_key in model_keys:
                        print(f"  ✓ {img_key} -> {test_key} (via non-img)")
                    else:
                        print(f"  ✗ {img_key} -> No match found")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()