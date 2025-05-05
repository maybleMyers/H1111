# convert_lora.py
import argparse
import os
import re
import torch
from safetensors.torch import load_file, save_file
import logging

# Configure logging similar to the utility file
logger = logging.getLogger(__name__)
# Avoid re-configuring if basicConfig was already called by the imported module
if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO)

# Assuming framepack_lora_inf_utils.py is in the same directory
try:
    from framepack_lora_inf_utils import (
        convert_hunyuan_to_framepack,
        convert_from_diffusion_pipe_or_something,
    )
except ImportError:
    logger.error("Error: Could not import conversion functions from framepack_lora_inf_utils.")
    logger.error("Please make sure framepack_lora_inf_utils.py is in the same directory as this script.")
    exit(1)

def main():
    """
    Main function to parse arguments and perform the LoRA conversion,
    detecting the input format (Hunyuan or Diffusion Pipe-like).
    """
    parser = argparse.ArgumentParser(description="Convert various LoRA formats to FramePack format.")
    parser.add_argument(
        "--input_lora",
        type=str,
        required=True,
        help="Path to the input LoRA .safetensors file (Hunyuan, Diffusion Pipe-like, or Musubi).",
    )
    parser.add_argument(
        "--output_lora",
        type=str,
        required=True,
        help="Path to save the converted FramePack LoRA .safetensors file.",
    )

    args = parser.parse_args()

    input_file = args.input_lora
    output_file = args.output_lora

    # Validate input file
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        exit(1)
    if not input_file.lower().endswith(".safetensors"):
         logger.warning(f"Input file '{input_file}' does not end with .safetensors. Proceeding anyway.")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logger.error(f"Error creating output directory {output_dir}: {e}")
            exit(1)

    # Ensure output file ends with .safetensors
    if not output_file.lower().endswith(".safetensors"):
        output_file += ".safetensors"
        logger.warning(f"Output file name did not end with .safetensors. Appended: {output_file}")

    logger.info(f"Loading input LoRA file: {input_file}")
    loaded_lora_sd = None
    try:
        # Load the state dictionary from the input .safetensors file
        loaded_lora_sd = load_file(input_file)
        logger.info(f"Input LoRA loaded successfully. Found {len(loaded_lora_sd)} keys.")
    except Exception as e:
        logger.error(f"Error loading input LoRA file {input_file}: {e}")
        exit(1)

    # --- Determine LoRA format and apply conversion(s) ---
    # Following the logic flow from merge_lora_to_state_dict

    processed_lora_sd = None # This will hold the SD after potential conversions
    lora_keys = list(loaded_lora_sd.keys()) if loaded_lora_sd else []

    if not lora_keys:
        logger.error("Input LoRA file was empty or failed to load keys.")
        exit(1)

    # 1. Check if it's Musubi Tuner format (first key starts with "lora_unet_")
    is_musubi = lora_keys[0].startswith("lora_unet_")
    if is_musubi:
        logger.info("Detected Musubi Tuner format based on first key.")
        # Keep the original SD for now, as Musubi format might still contain Hunyuan patterns
        current_lora_sd_to_check = loaded_lora_sd
    else:
        # 2. If not Musubi (based on first key), check for Diffusion Pipe format
        diffusion_pipe_pattern_found = False
        transformer_prefixes = ["diffusion_model", "transformer"]
        lora_suffix_A_or_B_found = False

        # Find the first key with .lora_A or .lora_B and check its prefix
        for key in lora_keys:
             if ".lora_A." in key or ".lora_B." in key:
                 lora_suffix_A_or_B_found = True
                 pfx = key.split(".")[0]
                 if pfx in transformer_prefixes:
                     diffusion_pipe_pattern_found = True
                     break # Found the required pattern

        if diffusion_pipe_pattern_found:
            logger.info(f"Detected Diffusion Pipe (?) format based on keys like '{pfx}.*.lora_A/B.'. Attempting conversion...")
            target_prefix_for_diffusers_conversion = "lora_unet_"
            try:
                # Apply the Diffusion Pipe conversion
                current_lora_sd_to_check = convert_from_diffusion_pipe_or_something(loaded_lora_sd, target_prefix_for_diffusers_conversion)
                logger.info("Converted from Diffusion Pipe format.")
            except Exception as e:
                logger.error(f"Error during Diffusion Pipe conversion: {e}", exc_info=True) # Log traceback
                current_lora_sd_to_check = None # Conversion failed, treat as unprocessable
        else:
            # If not Musubi and not Diffusion Pipe, the format is unrecognized initially
            logger.warning(f"LoRA file format not recognized based on common patterns (Musubi, Diffusion Pipe-like). Checking for Hunyuan anyway...")
            current_lora_sd_to_check = loaded_lora_sd # Keep the original SD to check for Hunyuan keys next


    # 3. Check for Hunyuan pattern (double_blocks/single_blocks) in the *current* state dict
    if current_lora_sd_to_check is not None:
        keys_for_hunyuan_check = list(current_lora_sd_to_check.keys())
        is_hunyuan_pattern_found = any("double_blocks" in key or "single_blocks" in key for key in keys_for_hunyuan_check)

        if is_hunyuan_pattern_found:
            logger.info("Detected HunyuanVideo format based on keys (double_blocks/single_blocks). Attempting conversion...")
            try:
                # Apply the Hunyuan conversion
                processed_lora_sd = convert_hunyuan_to_framepack(current_lora_sd_to_check)
                logger.info("Converted from HunyuanVideo format.")
            except Exception as e:
                logger.error(f"Error during HunyuanVideo conversion: {e}", exc_info=True) # Log traceback
                processed_lora_sd = None # Conversion failed, treat as unprocessable
        else:
            # If Hunyuan pattern is not found, the current_lora_sd_to_check is the final state
            # (either the original Musubi SD, or the SD converted from Diffusion Pipe).
            processed_lora_sd = current_lora_sd_to_check
            if not is_musubi and not diffusion_pipe_pattern_found:
                 # If we reached here and neither Musubi nor Diffusion Pipe patterns were initially found,
                 # and no Hunyuan pattern was found either, then the format is truly unrecognized.
                 logger.warning("Input LoRA does not match Musubi, Diffusion Pipe-like, or Hunyuan patterns.")
                 # Log keys to help debugging unrecognized formats
                 logger.info(f"Input LoRA keys start with: {lora_keys[:20]}...") # Show first few keys
                 processed_lora_sd = None # Mark as unprocessable


    # --- Final check and saving ---
    if processed_lora_sd is None or not processed_lora_sd:
        logger.error("Could not convert the input LoRA file to a recognized FramePack format.")
        logger.error("Input file format not recognized or conversion failed.")
        # Log keys if conversion didn't happen due to format not matching
        if loaded_lora_sd is not None:
             logger.info(f"Input LoRA keys start with: {lora_keys[:20]}...") # Show first few keys
        exit(1) # Exit if conversion failed or no data resulted

    logger.info(f"Conversion complete. Converted state dictionary contains {len(processed_lora_sd)} keys.")
    logger.info(f"Saving converted LoRA file to: {output_file}")

    # --- WORKAROUND for older safetensors versions that don't support allow_shared=True ---
    # The conversion functions might create shared tensors in the dictionary.
    # Older safetensors versions require tensors to be distinct objects for save_file.
    # We create a deep copy of tensors to satisfy this requirement.
    # The recommended fix is to upgrade safetensors (pip install --upgrade safetensors)
    # and use allow_shared=True in save_file.
    logger.info("Checking for shared tensors and creating copies for saving (workaround for older safetensors)...")
    processed_lora_sd_copy = {}
    for key, tensor in processed_lora_sd.items():
        if isinstance(tensor, torch.Tensor):
             # Create a new tensor with copied data, detached from any computation graph
             processed_lora_sd_copy[key] = tensor.clone().detach()
        else:
             # Keep non-tensor items (like alpha which might be a scalar number) as is
             processed_lora_sd_copy[key] = tensor
    logger.info("Deep copy complete.")
    # --- END OF WORKAROUND ---


    try:
        # Save using the deep-copied dictionary.
        # This works with older safetensors versions (pre-0.3.0)
        # If you upgraded safetensors to 0.3.0 or later, you could use:
        # save_file(processed_lora_sd, output_file, allow_shared=True)
        save_file(processed_lora_sd_copy, output_file)

        logger.info(f"Successfully saved converted LoRA to {output_file}")
    except Exception as e:
        # Note: If you still get a shared memory error here, it implies the deep copy
        # workaround didn't fully resolve it for your specific setup, OR the error
        # is coming from a different source. Upgrading safetensors is then highly recommended.
        logger.error(f"Error saving converted LoRA file {output_file}: {e}", exc_info=True) # Log traceback
        exit(1)

if __name__ == "__main__":
    main()