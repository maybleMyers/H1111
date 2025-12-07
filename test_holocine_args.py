#!/usr/bin/env python3
"""
Test script for holocine_generate_video.py argument parsing.

This script validates that the argument parser works correctly without
requiring torch or other heavy dependencies.
"""

import sys
import argparse

# Mock the imports that require external dependencies
class MockTorch:
    class Tensor:
        pass

    @staticmethod
    def cuda_is_available():
        return False

    class device:
        def __init__(self, *args, **kwargs):
            pass

sys.modules['torch'] = MockTorch()
sys.modules['accelerate'] = type(sys)('accelerate')
sys.modules['safetensors'] = type(sys)('safetensors')
sys.modules['safetensors.torch'] = type(sys)('safetensors.torch')
sys.modules['PIL'] = type(sys)('PIL')
sys.modules['PIL.Image'] = type(sys)('PIL.Image')
sys.modules['cv2'] = type(sys)('cv2')
sys.modules['numpy'] = type(sys)('numpy')
sys.modules['torchvision'] = type(sys)('torchvision')
sys.modules['torchvision.transforms'] = type(sys)('torchvision.transforms')
sys.modules['torchvision.transforms.functional'] = type(sys)('torchvision.transforms.functional')
sys.modules['tqdm'] = type(sys)('tqdm')
sys.modules['networks'] = type(sys)('networks')
sys.modules['networks.lora_wan'] = type(sys)('networks.lora_wan')
sys.modules['utils'] = type(sys)('utils')
sys.modules['utils.safetensors_utils'] = type(sys)('utils.safetensors_utils')
sys.modules['utils.lora_utils'] = type(sys)('utils.lora_utils')
sys.modules['utils.model_utils'] = type(sys)('utils.model_utils')
sys.modules['utils.device_utils'] = type(sys)('utils.device_utils')
sys.modules['Wan2_2'] = type(sys)('Wan2_2')
sys.modules['Wan2_2.wan'] = type(sys)('Wan2_2.wan')
sys.modules['Wan2_2.wan.configs'] = type(sys)('Wan2_2.wan.configs')
sys.modules['Wan2_2.wan.modules'] = type(sys)('Wan2_2.wan.modules')
sys.modules['Wan2_2.wan.modules.vae2_2'] = type(sys)('Wan2_2.wan.modules.vae2_2')
sys.modules['Wan2_2.context_windows'] = type(sys)('Wan2_2.context_windows')
sys.modules['wan'] = type(sys)('wan')
sys.modules['wan.modules'] = type(sys)('wan.modules')
sys.modules['wan.modules.model'] = type(sys)('wan.modules.model')
sys.modules['wan.modules.vae'] = type(sys)('wan.modules.vae')
sys.modules['wan.modules.t5'] = type(sys)('wan.modules.t5')
sys.modules['wan.modules.clip'] = type(sys)('wan.modules.clip')
sys.modules['wan.utils'] = type(sys)('wan.utils')
sys.modules['wan.utils.fm_solvers'] = type(sys)('wan.utils.fm_solvers')
sys.modules['wan.utils.fm_solvers_unipc'] = type(sys)('wan.utils.fm_solvers_unipc')
sys.modules['wan.utils.fm_solvers_euler'] = type(sys)('wan.utils.fm_solvers_euler')
sys.modules['wan.utils.step_distill_scheduler'] = type(sys)('wan.utils.step_distill_scheduler')
sys.modules['modules'] = type(sys)('modules')
sys.modules['modules.scheduling_flow_match_discrete'] = type(sys)('modules.scheduling_flow_match_discrete')
sys.modules['blissful_tuner'] = type(sys)('blissful_tuner')
sys.modules['blissful_tuner.latent_preview'] = type(sys)('blissful_tuner.latent_preview')
sys.modules['lycoris'] = type(sys)('lycoris')
sys.modules['lycoris.kohya'] = type(sys)('lycoris.kohya')
sys.modules['av'] = type(sys)('av')
sys.modules['einops'] = type(sys)('einops')
sys.modules['wan2_generate_video'] = type(sys)('wan2_generate_video')

# Add mock WAN_CONFIGS
class MockConfig:
    pass

sys.modules['Wan2_2.wan.configs'].WAN_CONFIGS = {
    't2v-A14B': MockConfig(),
    'i2v-A14B': MockConfig(),
    'ti2v-5B': MockConfig(),
    'longcat-t2v-13.6B': MockConfig(),
}
sys.modules['Wan2_2.wan.configs'].SUPPORTED_SIZES = []

# Now import the holocine script
from holocine_generate_video import parse_args, prepare_multishot_inputs, enforce_4t_plus_1

def test_mode1_structured_input():
    """Test Mode 1: Structured multi-shot input"""
    print("\n=== Test 1: Mode 1 - Structured Multi-Shot Input ===")

    test_args = [
        '--global_caption', 'A 1920s masquerade party. This scene contains 5 shots.',
        '--shot_captions',
            'Medium shot of woman by pillar',
            'Close-up of gentleman watching',
            'Medium shot as he approaches',
            'Close-up of her eyes',
            'Two-shot of them conversing',
        '--video_length', '241',
        '--save_path', 'outputholo.mp4',
        '--dit_low_noise', 'wan/full/full_low_noise.safetensors',
        '--dit_high_noise', 'wan/full/full_high_noise.safetensors',
        '--blocks_to_swap', '20'
    ]

    sys.argv = ['test'] + test_args

    try:
        args = parse_args()
        print(f"✅ Argument parsing successful!")
        print(f"  - Global caption: {args.global_caption[:50]}...")
        print(f"  - Number of shots: {len(args.shot_captions)}")
        print(f"  - Video length: {args.video_length}")
        print(f"  - Blocks to swap: {args.blocks_to_swap}")
        print(f"  - Prompt (before processing): {args.prompt}")

        # Test shot processing
        shot_data = prepare_multishot_inputs(
            args.global_caption,
            args.shot_captions,
            args.video_length,
            args.shot_cut_frames
        )

        print(f"\n✅ Shot processing successful!")
        print(f"  - Generated prompt length: {len(shot_data['prompt'])} chars")
        print(f"  - Shot cuts: {shot_data['shot_cut_frames']}")
        print(f"  - Adjusted frames: {shot_data['num_frames']}")

        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mode2_raw_prompt():
    """Test Mode 2: Raw HoloCine format prompt"""
    print("\n=== Test 2: Mode 2 - Raw HoloCine Format ===")

    test_args = [
        '--prompt', '[global caption] A painter. This scene contains 6 shots. [per shot caption] Standing back [shot cut] Hand with brush',
        '--shot_cut_frames', '37', '73', '113', '169', '205',
        '--video_length', '241',
        '--save_path', 'output.mp4',
        '--dit_low_noise', 'wan/full/full_low_noise.safetensors',
        '--dit_high_noise', 'wan/full/full_high_noise.safetensors',
    ]

    sys.argv = ['test'] + test_args

    try:
        args = parse_args()
        print(f"✅ Argument parsing successful!")
        print(f"  - Prompt: {args.prompt[:50]}...")
        print(f"  - Shot cuts (before 4t+1): {args.shot_cut_frames}")
        print(f"  - Video length: {args.video_length}")

        # Test 4t+1 enforcement
        adjusted_cuts = [enforce_4t_plus_1(f) for f in args.shot_cut_frames]
        print(f"  - Shot cuts (after 4t+1): {adjusted_cuts}")

        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mode3_standard_t2v():
    """Test Mode 3: Standard T2V"""
    print("\n=== Test 3: Mode 3 - Standard T2V ===")

    test_args = [
        '--prompt', 'A beautiful sunset over mountains',
        '--video_length', '81',
        '--save_path', 'output.mp4',
        '--dit_low_noise', 'wan/full/full_low_noise.safetensors',
    ]

    sys.argv = ['test'] + test_args

    try:
        args = parse_args()
        print(f"✅ Argument parsing successful!")
        print(f"  - Prompt: {args.prompt}")
        print(f"  - Video length: {args.video_length}")

        # Test 4t+1 enforcement
        adjusted = enforce_4t_plus_1(args.video_length)
        print(f"  - Video length (after 4t+1): {adjusted}")

        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_missing_prompt_error():
    """Test that missing both --prompt and structured input raises error"""
    print("\n=== Test 4: Missing Prompt Error ===")

    test_args = [
        '--video_length', '81',
        '--save_path', 'output.mp4',
        '--dit_low_noise', 'wan/full/full_low_noise.safetensors',
    ]

    sys.argv = ['test'] + test_args

    try:
        args = parse_args()
        print(f"❌ Test failed: Should have raised error for missing prompt")
        return False
    except SystemExit:
        print(f"✅ Correctly raised error for missing prompt")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_4t_plus_1_enforcement():
    """Test 4t+1 enforcement function"""
    print("\n=== Test 5: 4t+1 Enforcement ===")

    test_cases = [
        (240, 241),
        (241, 241),
        (81, 81),
        (80, 81),
        (100, 101),
        (1, 1),
        (5, 5),
    ]

    all_passed = True
    for input_val, expected in test_cases:
        result = enforce_4t_plus_1(input_val)
        if result == expected:
            print(f"✅ {input_val} -> {result} (expected {expected})")
        else:
            print(f"❌ {input_val} -> {result} (expected {expected})")
            all_passed = False

    return all_passed


def main():
    print("=" * 70)
    print("HoloCine Generate Video - Argument Parsing Tests")
    print("=" * 70)

    tests = [
        ("Mode 1: Structured Multi-Shot", test_mode1_structured_input),
        ("Mode 2: Raw HoloCine Format", test_mode2_raw_prompt),
        ("Mode 3: Standard T2V", test_mode3_standard_t2v),
        ("Missing Prompt Error", test_missing_prompt_error),
        ("4t+1 Enforcement", test_4t_plus_1_enforcement),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    total = len(results)
    passed = sum(1 for _, r in results if r)

    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
