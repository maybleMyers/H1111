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

print("\n" + "="*60)
print("Testing complete!")
print("="*60)

# Now test the actual import block from wan2_generate_video.py
print("\n5. Testing exact import block from wan2_generate_video.py:")
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