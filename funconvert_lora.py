# convert_lora_i2v_to_fc.py
import torch
import safetensors.torch
import safetensors # Need this for safe_open
import argparse
import os
import re # Regular expressions might be useful for more complex key parsing if needed

# !!! IMPORTANT: Updated based on the output of analyze_wan_models.py !!!
# The base layer name identified with shape mismatch.
# Check your LoRA file's keys if they use a different prefix (e.g., 'transformer.')
# Assuming the base name identified in LoRA keys matches this.
BASE_LAYERS_TO_SKIP_LORA = {
    "patch_embedding", # The layer name from the analysis output
    # Add other layers here ONLY if the analysis revealed more mismatches
}
# !!! END IMPORTANT SECTION !!!

def get_base_layer_name(lora_key: str, prefixes = ["lora_transformer_", "lora_unet_"]):
    """
    Attempts to extract the base model layer name from a LoRA key.
    Handles common prefixes and suffixes. Adjust prefixes if needed.

    Example: "lora_transformer_patch_embedding_down.weight" -> "patch_embedding"
             "lora_transformer_blocks_0_attn_qkv.alpha" -> "blocks.0.attn.qkv"

    Args:
        lora_key (str): The key from the LoRA state dictionary.
        prefixes (list[str]): A list of potential prefixes used in LoRA keys.

    Returns:
        str: The inferred base model layer name.
    """
    cleaned_key = lora_key

    # Remove known prefixes
    for prefix in prefixes:
        if cleaned_key.startswith(prefix):
            cleaned_key = cleaned_key[len(prefix):]
            break # Assume only one prefix matches

    # Remove known suffixes
    # Order matters slightly if one suffix is part of another; list longer ones first if needed
    known_suffixes = [
        ".lora_up.weight",
        ".lora_down.weight",
        "_lora_up.weight",   # Include underscore variants just in case
        "_lora_down.weight",
        ".alpha"
    ]
    for suffix in known_suffixes:
        if cleaned_key.endswith(suffix):
            cleaned_key = cleaned_key[:-len(suffix)]
            break

    # Replace underscores used by some training scripts with periods for consistency
    # if the original model uses periods (like typical PyTorch modules).
    # Adjust this logic if the base model itself uses underscores extensively.
    cleaned_key = cleaned_key.replace("_", ".")

    # Specific fix for the target layer if prefix/suffix removal was incomplete or ambiguous
    # This is somewhat heuristic and might need adjustment based on exact LoRA key naming.
    if cleaned_key.startswith("patch.embedding"): # Handle case where prefix removal was incomplete
         # Map potential variants back to the canonical name found in analysis
         cleaned_key = "patch_embedding"
    elif cleaned_key == "patch.embedding.weight": # If suffix removal left .weight attached somehow
         cleaned_key = "patch_embedding"
    # Add elif clauses here if other specific key mappings are needed


    return cleaned_key


def convert_lora(source_lora_path: str, target_lora_path: str):
    """
    Converts an i2v_14B LoRA to be compatible with i2v_14B_FC by
    removing LoRA weights associated with layers that have incompatible shapes.

    Args:
        source_lora_path (str): Path to the input LoRA file (.safetensors).
        target_lora_path (str): Path to save the converted LoRA file (.safetensors).
    """
    print(f"Loading source LoRA from: {source_lora_path}")
    if not os.path.exists(source_lora_path):
        print(f"Error: Source file not found: {source_lora_path}")
        return

    try:
        # Load tensors and metadata using safe_open for better handling
        source_lora_state_dict = {}
        metadata = {}
        with safetensors.safe_open(source_lora_path, framework="pt", device="cpu") as f:
            metadata = f.metadata() # Get metadata if it exists
            if metadata is None: # Ensure metadata is a dict even if empty
                metadata = {}
            for key in f.keys():
                source_lora_state_dict[key] = f.get_tensor(key) # Load tensors

        print(f"Successfully loaded {len(source_lora_state_dict)} tensors.")
        if metadata:
            print(f"Found metadata: {metadata}")
        else:
            print("No metadata found.")

    except Exception as e:
        print(f"Error loading LoRA file: {e}")
        import traceback
        traceback.print_exc()
        return

    target_lora_state_dict = {}
    skipped_keys = []
    kept_keys = []
    base_name_map = {} # Store mapping for reporting

    print(f"\nConverting LoRA weights...")
    print(f"Will skip LoRA weights targeting these base layers: {BASE_LAYERS_TO_SKIP_LORA}")

    # Iterate through the loaded tensors
    for key, tensor in source_lora_state_dict.items():
        # Use the helper function to extract the base layer name
        base_layer_name = get_base_layer_name(key)
        base_name_map[key] = base_layer_name # Store for reporting purposes

        # Check if the identified base layer name should be skipped
        if base_layer_name in BASE_LAYERS_TO_SKIP_LORA:
            skipped_keys.append(key)
        else:
            # Keep the tensor if its base layer is not in the skip list
            target_lora_state_dict[key] = tensor
            kept_keys.append(key)

    # --- Reporting ---
    print(f"\nConversion Summary:")
    print(f"  - Total Tensors in Source: {len(source_lora_state_dict)}")
    print(f"  - Kept {len(kept_keys)} LoRA weight tensors.")
    print(f"  - Skipped {len(skipped_keys)} LoRA weight tensors (due to incompatible base layer shape):")

    if skipped_keys:
        max_print = 15 # Show a few more skipped keys if desired
        skipped_sorted = sorted(skipped_keys) # Sort for consistent output order
        for i, key in enumerate(skipped_sorted):
             base_name = base_name_map.get(key, "N/A") # Get the identified base name
             print(f"    - {key} (Base Layer Identified: {base_name})")
             if i >= max_print -1 and len(skipped_keys) > max_print:
                 print(f"    ... and {len(skipped_keys) - max_print} more.")
                 break
    else:
        print("      None")

    # --- Saving ---
    print(f"\nSaving converted LoRA ({len(target_lora_state_dict)} tensors) to: {target_lora_path}")
    try:
        # Save the filtered state dictionary with the original metadata
        safetensors.torch.save_file(target_lora_state_dict, target_lora_path, metadata=metadata)
        print("Conversion successful!")
    except Exception as e:
        print(f"Error saving converted LoRA file: {e}")


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Convert Wan i2v_14B LoRA to i2v_14B_FC LoRA by removing incompatible patch_embedding weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("source_lora", type=str, help="Path to the source i2v_14B LoRA file (.safetensors).")
    parser.add_argument("target_lora", type=str, help="Path to save the converted i2v_14B_FC LoRA file (.safetensors).")

    # Parse arguments
    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.exists(args.source_lora):
         print(f"Error: Source LoRA file not found at '{args.source_lora}'")
    elif not args.source_lora.lower().endswith(".safetensors"):
         print(f"Warning: Source file '{args.source_lora}' does not have a .safetensors extension.")
    elif args.source_lora == args.target_lora:
         print(f"Error: Source and target paths cannot be the same ('{args.source_lora}'). Choose a different target path.")
    elif os.path.exists(args.target_lora):
         print(f"Warning: Target file '{args.target_lora}' already exists and will be overwritten.")
         # Optionally add a --force flag or prompt user here
         convert_lora(args.source_lora, args.target_lora)
    else:
        # Run the conversion if basic checks pass
        convert_lora(args.source_lora, args.target_lora)