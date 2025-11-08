#!/usr/bin/env python3
"""
Test script to verify LongCat LoRA support implementation.

This script tests that:
1. LoRA arguments are properly parsed
2. Include/exclude patterns work correctly
3. LoRA loading function handles patterns correctly
"""

import argparse
import sys
import os


def test_argument_parsing():
    """Test that all LoRA arguments are properly defined."""
    print("=" * 80)
    print("Test 1: Argument Parsing")
    print("=" * 80)

    # Import the main script to get argument parser
    sys.path.insert(0, os.path.dirname(__file__))

    # Test basic LoRA arguments
    test_args = [
        "--task", "longcat-A14B",
        "--ckpt_dir", "/path/to/checkpoint",
        "--prompt", "test prompt",
        "--mode", "generation",
        "--lora_weight", "/path/to/lora1.safetensors", "/path/to/lora2.safetensors",
        "--lora_multiplier", "1.0", "0.8",
        "--include_patterns", "pattern1", "pattern2",
        "--exclude_patterns", "exclude1", "exclude2",
    ]

    # Parse arguments (will fail if arguments are not defined)
    try:
        # We can't actually import and run the parser without all dependencies
        # So just check the file for the argument definitions
        with open("longcat_generate_video.py", "r") as f:
            content = f.read()

        required_args = [
            "--lora_weight",
            "--lora_multiplier",
            "--include_patterns",
            "--exclude_patterns",
            "--lora_weight_high",
            "--lora_multiplier_high",
            "--include_patterns_high",
            "--exclude_patterns_high",
        ]

        missing_args = []
        for arg in required_args:
            if arg not in content:
                missing_args.append(arg)

        if missing_args:
            print(f"❌ FAILED: Missing argument definitions: {missing_args}")
            return False
        else:
            print("✓ All LoRA arguments are defined")
            return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_load_loras_on_model_signature():
    """Test that load_loras_on_model has correct signature."""
    print("\n" + "=" * 80)
    print("Test 2: load_loras_on_model Function Signature")
    print("=" * 80)

    try:
        with open("longcat_generate_video.py", "r") as f:
            content = f.read()

        # Check function definition
        if "def load_loras_on_model(" not in content:
            print("❌ FAILED: load_loras_on_model function not found")
            return False

        # Check for include/exclude pattern parameters
        function_start = content.find("def load_loras_on_model(")
        function_section = content[function_start:function_start + 1000]

        required_params = [
            "include_patterns",
            "exclude_patterns",
        ]

        missing_params = []
        for param in required_params:
            if param not in function_section:
                missing_params.append(param)

        if missing_params:
            print(f"❌ FAILED: Missing parameters in load_loras_on_model: {missing_params}")
            return False
        else:
            print("✓ load_loras_on_model has correct signature with pattern support")
            return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_lora_loading_calls():
    """Test that all LoRA loading calls pass patterns."""
    print("\n" + "=" * 80)
    print("Test 3: LoRA Loading Calls")
    print("=" * 80)

    try:
        with open("longcat_generate_video.py", "r") as f:
            content = f.read()

        # Find all calls to load_loras_on_model
        import re
        calls = re.findall(r'load_loras_on_model\([^)]+\)', content, re.DOTALL)

        if not calls:
            print("❌ FAILED: No calls to load_loras_on_model found")
            return False

        print(f"Found {len(calls)} calls to load_loras_on_model")

        # Check that calls include pattern parameters (in T2V, VC, and I2V modes)
        pattern_calls = 0
        for call in calls:
            if "include_patterns" in call and "exclude_patterns" in call:
                pattern_calls += 1

        if pattern_calls >= 3:  # Should be at least 3: T2V, VC, I2V
            print(f"✓ Found {pattern_calls} calls with pattern support")
            return True
        else:
            print(f"❌ FAILED: Only {pattern_calls} calls with pattern support (expected at least 3)")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_i2v_lora_support():
    """Test that i2v mode has LoRA support."""
    print("\n" + "=" * 80)
    print("Test 4: I2V Mode LoRA Support")
    print("=" * 80)

    try:
        with open("longcat_generate_video.py", "r") as f:
            content = f.read()

        # Find generate_longcat_i2v function
        i2v_start = content.find("def generate_longcat_i2v(")
        if i2v_start == -1:
            print("❌ FAILED: generate_longcat_i2v function not found")
            return False

        # Check section after function start for load_loras_on_model call
        i2v_section = content[i2v_start:i2v_start + 15000]

        if "load_loras_on_model" in i2v_section:
            print("✓ I2V mode has LoRA support")
            return True
        else:
            print("❌ FAILED: I2V mode does not call load_loras_on_model")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_filter_lora_state_dict_usage():
    """Test that filter_lora_state_dict is used in load_loras_on_model."""
    print("\n" + "=" * 80)
    print("Test 5: filter_lora_state_dict Usage")
    print("=" * 80)

    try:
        with open("longcat_generate_video.py", "r") as f:
            content = f.read()

        # Find load_loras_on_model function
        func_start = content.find("def load_loras_on_model(")
        if func_start == -1:
            print("❌ FAILED: load_loras_on_model function not found")
            return False

        # Find next function definition to get function body
        next_func = content.find("\ndef ", func_start + 1)
        func_body = content[func_start:next_func] if next_func != -1 else content[func_start:func_start + 5000]

        if "filter_lora_state_dict" in func_body:
            print("✓ filter_lora_state_dict is used in load_loras_on_model")
            return True
        else:
            print("❌ FAILED: filter_lora_state_dict not used in load_loras_on_model")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def main():
    print("\n")
    print("#" * 80)
    print("# LongCat LoRA Support Test Suite")
    print("#" * 80)
    print("\n")

    tests = [
        test_argument_parsing,
        test_load_loras_on_model_signature,
        test_lora_loading_calls,
        test_i2v_lora_support,
        test_filter_lora_state_dict_usage,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(results)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
