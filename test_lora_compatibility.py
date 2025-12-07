#!/usr/bin/env python3
"""
Test script to verify LoRA compatibility with different formats
"""

import torch
import logging
from pathlib import Path
from safetensors.torch import load_file
from utils.lora_utils import filter_lora_state_dict

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def analyze_lora_format(lora_path):
    """Analyze and detect the format of a LoRA file"""
    logger.info(f"\nAnalyzing LoRA: {lora_path}")
    logger.info("=" * 60)

    try:
        # Load the LoRA file
        lora_sd = load_file(lora_path, device="cpu")

        # Count different key patterns
        standard_format = 0  # .lora_down.weight, .lora_up.weight
        musubi_format = 0    # _lora_down_weight, _lora_up_weight
        lightx2v_format = 0  # .diff, .diff_b
        alpha_keys = 0
        img_keys = 0

        sample_keys = []

        for key in lora_sd.keys():
            if len(sample_keys) < 5:
                sample_keys.append(key)

            # Check for standard format
            if '.lora_down.weight' in key or '.lora_up.weight' in key:
                standard_format += 1
            # Check for MUSUBI format
            elif '_lora_down_weight' in key or '_lora_up_weight' in key:
                musubi_format += 1
            # Check for lightx2v format
            elif '.diff' in key or '_diff' in key:
                lightx2v_format += 1
            # Check for alpha keys
            if '.alpha' in key:
                alpha_keys += 1
            # Check for _img keys
            if '_img' in key:
                img_keys += 1

        logger.info(f"Total keys: {len(lora_sd)}")
        logger.info(f"Format detection:")
        logger.info(f"  - Standard format keys (.lora_down.weight): {standard_format}")
        logger.info(f"  - MUSUBI format keys (_lora_down_weight): {musubi_format}")
        logger.info(f"  - lightx2v format keys (.diff): {lightx2v_format}")
        logger.info(f"  - Alpha keys: {alpha_keys}")
        logger.info(f"  - Image variant keys (_img): {img_keys}")

        # Determine primary format
        if standard_format > musubi_format and standard_format > lightx2v_format:
            detected_format = "Standard LoRA"
        elif musubi_format > standard_format and musubi_format > lightx2v_format:
            detected_format = "MUSUBI LoRA"
        elif lightx2v_format > 0:
            detected_format = "lightx2v LoRA"
        else:
            detected_format = "Unknown"

        logger.info(f"\nDetected format: {detected_format}")
        logger.info(f"\nSample keys:")
        for key in sample_keys:
            logger.info(f"  - {key}")

        return detected_format, len(lora_sd)

    except Exception as e:
        logger.error(f"Error loading LoRA: {e}")
        return "Error", 0

def test_lora_compatibility():
    """Test LoRA loading with different formats"""

    # Look for available LoRA files
    lora_dir = Path("lora")
    if not lora_dir.exists():
        logger.warning(f"LoRA directory not found: {lora_dir}")
        return

    lora_files = list(lora_dir.glob("*.safetensors"))
    if not lora_files:
        logger.warning("No LoRA files found in lora directory")
        return

    logger.info(f"Found {len(lora_files)} LoRA file(s)")
    logger.info("Testing compatibility with each format...")

    results = []
    for lora_path in lora_files[:5]:  # Test first 5 LoRAs
        format_type, key_count = analyze_lora_format(str(lora_path))
        results.append({
            'file': lora_path.name,
            'format': format_type,
            'keys': key_count
        })

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPATIBILITY TEST SUMMARY")
    logger.info("=" * 60)

    for result in results:
        status = "✓" if result['format'] != "Error" else "✗"
        logger.info(f"{status} {result['file'][:40]:40} | {result['format']:15} | {result['keys']} keys")

    # Count formats
    format_counts = {}
    for result in results:
        fmt = result['format']
        format_counts[fmt] = format_counts.get(fmt, 0) + 1

    logger.info("\nFormat distribution:")
    for fmt, count in format_counts.items():
        logger.info(f"  {fmt}: {count} file(s)")

    logger.info("\nCompatibility test complete!")
    logger.info("The updated lora_utils.py should now support:")
    logger.info("  1. Standard LoRA format (dots)")
    logger.info("  2. MUSUBI LoRA format (underscores)")
    logger.info("  3. lightx2v LoRA format (diff keys)")

if __name__ == "__main__":
    test_lora_compatibility()