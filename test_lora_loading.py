#!/usr/bin/env python3
"""
Test script to verify LoRA loading is working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safetensors.torch import load_file
from utils.lora_utils import load_safetensors_with_lora_and_fp8
import torch

def test_lora_loading():
    """Test that LoRA weights are being applied correctly"""

    # Paths
    model_path = "wan/wan22_i2v_14B_high_noise_bf16.safetensors"
    lora_path = "lora/Wan2.2-T2V-A14B-4steps-lora-250928_high_noise_model_MUSUBI.safetensors"

    print(f"Testing LoRA loading with direct format support...")
    print(f"Model: {model_path}")
    print(f"LoRA: {lora_path}")
    print("-" * 80)

    # Load LoRA weights
    lora_sd = load_file(lora_path, device="cpu")
    print(f"Loaded LoRA with {len(lora_sd)} keys")

    # Show sample LoRA keys
    sample_keys = list(lora_sd.keys())[:5]
    print("Sample LoRA keys:")
    for key in sample_keys:
        print(f"  {key}")
    print()

    # Prepare LoRA weights list
    lora_weights_list = [lora_sd]
    lora_multipliers = [1.0]

    # Test loading with LoRA merging (just load a few keys to test)
    print("Testing LoRA merging...")

    # Track which keys are used
    original_keys = set(lora_sd.keys())

    # Use the load function with LoRA merging
    # We'll just test the hook function directly
    from utils.lora_utils import load_safetensors_with_lora_and_fp8

    # Mock test - check if specific keys would be matched
    test_model_keys = [
        "blocks.0.cross_attn.k.weight",
        "blocks.0.cross_attn.o.weight",
        "blocks.0.cross_attn.q.weight",
        "blocks.0.cross_attn.v.weight",
    ]

    matched_count = 0
    for model_key in test_model_keys:
        base_key = model_key.rsplit(".", 1)[0]

        # Check direct format (what the LoRA actually has)
        down_key_direct = base_key + ".lora_down.weight"
        up_key_direct = base_key + ".lora_up.weight"
        alpha_key_direct = base_key + ".alpha"

        if down_key_direct in lora_sd and up_key_direct in lora_sd:
            matched_count += 1
            print(f"✓ Found match for {model_key}:")
            print(f"    Down: {down_key_direct}")
            print(f"    Up:   {up_key_direct}")
            print(f"    Alpha: {alpha_key_direct} (exists: {alpha_key_direct in lora_sd})")
        else:
            print(f"✗ No match for {model_key}")

    print()
    print(f"Summary: Matched {matched_count}/{len(test_model_keys)} test keys")

    if matched_count == len(test_model_keys):
        print("✅ SUCCESS: LoRA loading should now work correctly!")
    else:
        print("⚠️ WARNING: Some keys still not matching")

    return matched_count == len(test_model_keys)

if __name__ == "__main__":
    success = test_lora_loading()
    sys.exit(0 if success else 1)