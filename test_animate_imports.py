#!/usr/bin/env python3
"""Test script to identify which Animate module imports are failing"""

import sys
import traceback

def test_import(module_path, description):
    """Test importing a module and report success/failure"""
    try:
        exec(f"import {module_path}")
        print(f"✓ {description}: SUCCESS")
        return True
    except Exception as e:
        print(f"✗ {description}: FAILED")
        print(f"  Error: {str(e)}")
        return False

def test_from_import(module_path, items, description):
    """Test importing specific items from a module"""
    try:
        exec(f"from {module_path} import {', '.join(items)}")
        print(f"✓ {description}: SUCCESS")
        return True
    except Exception as e:
        print(f"✗ {description}: FAILED")
        print(f"  Error: {str(e)}")
        print(f"  Traceback:")
        traceback.print_exc()
        return False

print("="*60)
print("Testing Animate Module Dependencies")
print("="*60)

# Test basic dependencies
print("\n1. Testing basic dependencies:")
test_import("torch", "PyTorch")
test_import("peft", "PEFT (LoRA support)")
test_import("decord", "Decord (video reading)")
test_import("imageio", "ImageIO")
test_import("cv2", "OpenCV")
test_import("loguru", "Loguru (logging)")
test_import("pandas", "Pandas")
test_import("matplotlib", "Matplotlib")
test_import("onnxruntime", "ONNX Runtime")

# Test Wan2_2 configs
print("\n2. Testing Wan2_2 configs:")
test_from_import("Wan2_2.wan.configs", ["WAN_CONFIGS"], "WAN_CONFIGS")
test_from_import("Wan2_2.wan.configs.wan_animate_14B", ["animate_14B"], "Animate config")

# Test Animate module components
print("\n3. Testing Animate module components:")

# Main animate imports
test_from_import("Wan2_2.wan.modules.animate",
                 ["WanAnimateModel", "CLIPModel"],
                 "Main Animate models")

# Animate utils
test_from_import("Wan2_2.wan.modules.animate.animate_utils",
                 ["TensorList", "get_loraconfig"],
                 "Animate utils")

# Preprocessing pipeline
try:
    print("\n4. Testing preprocessing components:")
    test_from_import("Wan2_2.wan.modules.animate.preprocess.process_pipepline",
                     ["ProcessPipeline"],
                     "Process Pipeline")
except:
    # Try alternative spelling
    test_from_import("Wan2_2.wan.modules.animate.preprocess.process_pipeline",
                     ["ProcessPipeline"],
                     "Process Pipeline (alt spelling)")

# Preprocessing utils
test_from_import("Wan2_2.wan.modules.animate.preprocess.utils",
                 ["resize_by_area", "get_frame_indices", "padding_resize",
                  "get_face_bboxes", "get_aug_mask", "get_mask_body_img"],
                 "Preprocessing utils")

# Pose detection
test_from_import("Wan2_2.wan.modules.animate.preprocess.pose2d",
                 ["Pose2d"],
                 "Pose2D")

test_from_import("Wan2_2.wan.modules.animate.preprocess.pose2d_utils",
                 ["AAPoseMeta"],
                 "Pose2D utils")

# Visualization
test_from_import("Wan2_2.wan.modules.animate.preprocess.human_visualization",
                 ["draw_aapose_by_meta_new"],
                 "Human visualization")

# Retargeting
test_from_import("Wan2_2.wan.modules.animate.preprocess.retarget_pose",
                 ["get_retarget_pose"],
                 "Pose retargeting")

# Face components
test_from_import("Wan2_2.wan.modules.animate.face_blocks",
                 ["FaceEncoder", "FaceAdapter"],
                 "Face blocks")

# Motion encoder
test_from_import("Wan2_2.wan.modules.animate.motion_encoder",
                 ["Generator"],
                 "Motion encoder")

# PEFT specific
test_from_import("peft",
                 ["set_peft_model_state_dict"],
                 "PEFT state dict functions")

# SAM2 specific tests
print("\n5. Testing SAM2 components:")

# Test basic SAM2 import
test_import("sam2", "SAM2 base package")
test_from_import("sam2.sam2_video_predictor", ["SAM2VideoPredictor"], "SAM2VideoPredictor")
test_from_import("sam2.utils.misc", ["AsyncVideoFrameLoader", "_load_img_as_tensor"], "SAM2 utils")
test_from_import("sam2.build_sam", ["_load_checkpoint"], "SAM2 checkpoint loader")

# Test Hydra components needed for SAM2
print("\n6. Testing Hydra components for SAM2:")
test_import("hydra", "Hydra")
test_from_import("hydra", ["compose"], "Hydra compose")
test_from_import("hydra.utils", ["instantiate"], "Hydra instantiate")
test_import("omegaconf", "OmegaConf")

# Test the local video_predictor wrapper
print("\n7. Testing local video_predictor wrapper:")
test_from_import("Wan2_2.wan.modules.animate.preprocess.video_predictor",
                 ["SAM2VideoPredictor"],
                 "Local SAM2VideoPredictor wrapper")

# Test SAM2 utils wrapper
test_from_import("Wan2_2.wan.modules.animate.preprocess.sam_utils",
                 ["build_sam2_video_predictor"],
                 "SAM2 predictor builder")

# Test different module path approaches for Hydra
print("\n8. Testing Hydra module path resolution:")
import os
import sys

# Test 1: Try importing video_predictor directly (this should fail)
print("  Testing direct import 'video_predictor':")
try:
    import video_predictor
    print("    ✓ Direct import works")
except ImportError as e:
    print(f"    ✗ Direct import failed: {e}")

# Test 2: Add Wan2_2 path and try again
print("  Testing with Wan2_2 path added:")
wan2_path = os.path.join(os.path.dirname(__file__), "Wan2_2", "wan", "modules", "animate", "preprocess")
if wan2_path not in sys.path:
    sys.path.insert(0, wan2_path)
    print(f"    Added path: {wan2_path}")
try:
    import video_predictor
    print("    ✓ Import works with path added")
    # Test if we can access the class
    from video_predictor import SAM2VideoPredictor as LocalSAM2VP
    print("    ✓ Can access SAM2VideoPredictor class")
except ImportError as e:
    print(f"    ✗ Import still fails: {e}")
finally:
    # Remove the path we added
    if wan2_path in sys.path:
        sys.path.remove(wan2_path)

# Test 3: Test Hydra instantiation with different paths
print("\n9. Testing Hydra instantiation paths:")
try:
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    # Test different target paths
    test_targets = [
        "video_predictor.SAM2VideoPredictor",
        "Wan2_2.wan.modules.animate.preprocess.video_predictor.SAM2VideoPredictor",
        "sam2.sam2_video_predictor.SAM2VideoPredictor"
    ]

    for target in test_targets:
        print(f"  Testing target: {target}")
        try:
            # Create a simple config to test instantiation
            cfg = OmegaConf.create({
                "_target_": target,
                "model": None  # We'll pass None for testing
            })
            # We can't actually instantiate without the full model, but we can check if the target resolves
            from hydra._internal.utils import _locate
            cls = _locate(target)
            if cls:
                print(f"    ✓ Target resolved to: {cls}")
            else:
                print(f"    ✗ Target could not be resolved")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

except Exception as e:
    print(f"  ✗ Hydra tests failed: {e}")

# Test SAM2 checkpoint existence
print("\n10. Testing SAM2 checkpoint files:")
sam_checkpoint_paths = [
    "wan/animate/process_checkpoint/sam2/sam2_hiera_large.pt",
    "wan/animate/process_checkpoint/sam2/sam2_hiera_base_plus.pt",
    "wan/animate/process_checkpoint/sam2/sam2_hiera_small.pt",
    "wan/animate/process_checkpoint/sam2/sam2_hiera_tiny.pt"
]

for path in sam_checkpoint_paths:
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
        print(f"  ✓ Found: {path} ({size:.1f} MB)")
    else:
        print(f"  ✗ Not found: {path}")

print("\n" + "="*60)
print("Testing complete!")
print("="*60)

# Now test the actual import block from wan2_generate_video.py
print("\n11. Testing exact import block from wan2_generate_video.py:")
try:
    from Wan2_2.wan.modules.animate import WanAnimateModel, CLIPModel as AnimateCLIPModel
    from Wan2_2.wan.modules.animate.animate_utils import TensorList, get_loraconfig
    from Wan2_2.wan.modules.animate.preprocess.process_pipepline import ProcessPipeline
    from Wan2_2.wan.modules.animate.preprocess.utils import (
        resize_by_area, get_frame_indices, padding_resize,
        get_face_bboxes, get_aug_mask, get_mask_body_img
    )
    from Wan2_2.wan.modules.animate.preprocess.pose2d import Pose2d
    from Wan2_2.wan.modules.animate.preprocess.pose2d_utils import AAPoseMeta
    from Wan2_2.wan.modules.animate.preprocess.human_visualization import draw_aapose_by_meta_new
    from Wan2_2.wan.modules.animate.preprocess.retarget_pose import get_retarget_pose
    from Wan2_2.wan.modules.animate.face_blocks import FaceEncoder, FaceAdapter
    from Wan2_2.wan.modules.animate.motion_encoder import Generator
    from decord import VideoReader
    from peft import set_peft_model_state_dict
    print("✓ All imports successful! ANIMATE_AVAILABLE should be True")
except Exception as e:
    print(f"✗ Import block failed: {e}")
    print("\nDetailed traceback:")
    traceback.print_exc()