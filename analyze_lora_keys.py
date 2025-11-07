#!/usr/bin/env python3
"""
Analysis script to compare LoRA keys between Wan2 and LongCat models.

This script helps understand the naming differences between LoRA keys
trained on Wan2 models vs LongCat models to enable proper LoRA loading.
"""

import argparse
import os
from safetensors import safe_open
from collections import defaultdict


def analyze_lora_keys(lora_path: str, model_name: str = ""):
    """
    Analyze LoRA keys from a safetensors file.

    Args:
        lora_path: Path to LoRA safetensors file
        model_name: Optional name for display
    """
    print(f"\n{'=' * 80}")
    print(f"Analyzing LoRA: {model_name or os.path.basename(lora_path)}")
    print(f"Path: {lora_path}")
    print(f"{'=' * 80}\n")

    if not os.path.exists(lora_path):
        print(f"ERROR: File not found: {lora_path}")
        return None

    # Load LoRA keys
    keys = []
    key_stats = defaultdict(int)

    with safe_open(lora_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

    print(f"Total keys: {len(keys)}")
    print(f"\nFirst 20 keys:")
    for i, key in enumerate(keys[:20]):
        print(f"  {i+1:3d}. {key}")

    if len(keys) > 20:
        print(f"\n... and {len(keys) - 20} more keys")

    print(f"\nLast 10 keys:")
    for i, key in enumerate(keys[-10:]):
        print(f"  {len(keys)-9+i:3d}. {key}")

    # Analyze key patterns
    print(f"\n{'=' * 80}")
    print("Key Pattern Analysis:")
    print(f"{'=' * 80}\n")

    # Extract module types
    for key in keys:
        parts = key.split('.')
        if 'lora' in key.lower():
            # Identify lora_up, lora_down, alpha
            if 'lora_up' in key:
                key_stats['lora_up'] += 1
            elif 'lora_down' in key:
                key_stats['lora_down'] += 1
            elif 'alpha' in key:
                key_stats['alpha'] += 1

        # Identify module types
        if 'blocks' in key or 'block' in key:
            key_stats['blocks'] += 1
        if 'attn' in key or 'attention' in key:
            key_stats['attention'] += 1
        if 'mlp' in key or 'ff' in key:
            key_stats['mlp/feedforward'] += 1
        if 'proj' in key:
            key_stats['projection'] += 1
        if 'qkv' in key:
            key_stats['qkv'] += 1
        if 'to_q' in key or 'to_k' in key or 'to_v' in key:
            key_stats['to_qkv'] += 1
        if 'norm' in key or 'ln' in key:
            key_stats['norm'] += 1

    print("Statistics:")
    for pattern, count in sorted(key_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern:20s}: {count:4d}")

    # Identify unique prefixes
    prefixes = set()
    for key in keys:
        parts = key.split('.')
        if len(parts) >= 2:
            prefixes.add('.'.join(parts[:2]))

    print(f"\nUnique prefixes (first 2 levels):")
    for prefix in sorted(prefixes)[:30]:
        count = sum(1 for k in keys if k.startswith(prefix))
        print(f"  {prefix:40s}: {count:4d} keys")

    if len(prefixes) > 30:
        print(f"  ... and {len(prefixes) - 30} more prefixes")

    return {
        'keys': keys,
        'stats': dict(key_stats),
        'prefixes': sorted(prefixes)
    }


def compare_loras(lora1_path: str, lora2_path: str, name1: str = "LoRA 1", name2: str = "LoRA 2"):
    """
    Compare two LoRA files and identify key differences.

    Args:
        lora1_path: Path to first LoRA file
        lora2_path: Path to second LoRA file
        name1: Name for first LoRA
        name2: Name for second LoRA
    """
    print(f"\n{'=' * 80}")
    print(f"Comparing LoRAs")
    print(f"{'=' * 80}\n")

    result1 = analyze_lora_keys(lora1_path, name1)
    result2 = analyze_lora_keys(lora2_path, name2)

    if not result1 or not result2:
        return

    keys1 = set(result1['keys'])
    keys2 = set(result2['keys'])

    print(f"\n{'=' * 80}")
    print("Comparison Summary:")
    print(f"{'=' * 80}\n")

    print(f"{name1} keys: {len(keys1)}")
    print(f"{name2} keys: {len(keys2)}")
    print(f"Common keys: {len(keys1 & keys2)}")
    print(f"Only in {name1}: {len(keys1 - keys2)}")
    print(f"Only in {name2}: {len(keys2 - keys1)}")

    # Show key differences
    if keys1 - keys2:
        print(f"\nSample keys only in {name1} (first 10):")
        for i, key in enumerate(sorted(keys1 - keys2)[:10]):
            print(f"  {i+1:3d}. {key}")

    if keys2 - keys1:
        print(f"\nSample keys only in {name2} (first 10):")
        for i, key in enumerate(sorted(keys2 - keys1)[:10]):
            print(f"  {i+1:3d}. {key}")

    # Compare prefixes
    prefixes1 = set(result1['prefixes'])
    prefixes2 = set(result2['prefixes'])

    if prefixes1 - prefixes2:
        print(f"\nPrefixes only in {name1}:")
        for prefix in sorted(prefixes1 - prefixes2)[:20]:
            print(f"  - {prefix}")

    if prefixes2 - prefixes1:
        print(f"\nPrefixes only in {name2}:")
        for prefix in sorted(prefixes2 - prefixes1)[:20]:
            print(f"  - {prefix}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare LoRA safetensors files"
    )
    parser.add_argument(
        "lora_paths",
        nargs="+",
        help="Path(s) to LoRA safetensors file(s)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare two LoRA files (requires exactly 2 paths)"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="Names for the LoRA files (optional)"
    )

    args = parser.parse_args()

    if args.compare:
        if len(args.lora_paths) != 2:
            print("ERROR: --compare requires exactly 2 LoRA paths")
            return 1

        names = args.names if args.names and len(args.names) == 2 else ["LoRA 1", "LoRA 2"]
        compare_loras(args.lora_paths[0], args.lora_paths[1], names[0], names[1])
    else:
        for i, lora_path in enumerate(args.lora_paths):
            name = args.names[i] if args.names and i < len(args.names) else ""
            analyze_lora_keys(lora_path, name)

    return 0


if __name__ == "__main__":
    exit(main())
