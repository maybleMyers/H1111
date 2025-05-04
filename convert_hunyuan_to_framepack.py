import argparse
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

import os
import re
from collections import defaultdict
import traceback # Added for better error reporting

# Define the mapping from source key parts to target model's internal layer names and logic
# Key: Source module name (from original non-working LoRA format, e.g., 'img_attn_qkv')
# Value: Dictionary describing target(s)
#    'target': List of target model layer names (from working LoRA/model index, dots included)
#    'split_B': Number of ways to split source B tensor along dim 0 (if > 1, for fused QKV)
LAYER_MAP_SOURCE_TO_MODEL = {
    # Mappings observed between the first non-working LoRA and the working LoRA's structure
    # These map SOURCE modules to TARGET modules.
    # double_blocks (Source) -> transformer_blocks (Target)
    'img_attn_qkv': {'target': ['attn.to_k', 'attn.to_q', 'attn.to_v'], 'split_B': 3}, # Maps fused img QKV to split img K, Q, V
    'img_attn_proj': {'target': ['attn.to_out.0']}, # Maps img attention output projection
    'img_mlp.fc1': {'target': ['ff.net.0.proj']}, # Maps img MLP fc1
    'img_mlp.fc2': {'target': ['ff.net.2']}, # Maps img MLP fc2

    'txt_attn_qkv': {'target': ['attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj'], 'split_B': 3}, # Maps fused txt QKV to split txt K, Q, V
    'txt_attn_proj': {'target': ['attn.to_add_out']}, # Maps txt attention output projection
    'txt_mlp.fc1': {'target': ['ff_context.net.0.proj']}, # Maps txt MLP fc1
    'txt_mlp.fc2': {'target': ['ff_context.net.2']}, # Maps txt MLP fc2

    # single_blocks (Source) -> single_transformer_blocks (Target)
    # Note: single_blocks.linear1 (Source) does not have a compatible output dimension
    # with single_transformer_blocks.proj_mlp (Target), so it is not mapped.
    'linear2': {'target': ['proj_out']}, # Maps single_blocks linear2 to single_transformer_blocks proj_out
}

# block_type_map: Maps source block type prefix to target block internal name prefix
BLOCK_TYPE_MAP_SOURCE_TO_MODEL = {
    'double_blocks': 'transformer_blocks',
    'single_blocks': 'single_transformer_blocks',
}

# Target default rank expected by the studio.py system (based on working LoRA analysis)
TARGET_DEFAULT_RANK = 32

# Expected output dimensions for target layers (based on working file analysis and model index)
# These are the *output* dimensions of the base layer, which correspond to the
# first dimension of the LoRA B/up matrix *before* splitting for fused QKV.
# The LoRA up matrix shape is (OutDim, Rank).
# Keys here should match the underscored target layer names used in the script's lookup.
TARGET_OUT_DIMS = {
    'single_transformer_blocks': {
        'proj_out': 3072,
        # These are targets of split QKV from SOURCE double_blocks, mapped to TARGET single_transformer_blocks.
        # Add them here to validate split QKV target dimensions if needed, although they are
        # already covered by the mapping targets being in 'transformer_blocks'.
        # This section mostly covers layers exclusive to single_transformer_blocks in the target.
        'proj_mlp': 12288
    },
    'transformer_blocks': {
        'attn_add_k_proj': 3072,
        'attn_add_q_proj': 3072,
        'attn_add_v_proj': 3072,
        'attn_to_add_out': 3072,
        'ff_net_0_proj': 12288,
        'ff_net_2': 3072,
        'ff_context_net_0_proj': 12288,
        'ff_context_net_2': 3072,
        # --- Missing Image Attention Layers (added) ---
        'attn_to_k': 3072,
        'attn_to_q': 3072,
        'attn_to_v': 3072,
        'attn_to_out_0': 3072
    }
}

# Max index for target transformer_blocks (corrected based on working file analysis / model index)
MAX_TRANSFORMER_BLOCK_INDEX = 19
# Max index for target single_transformer_blocks (from working file analysis / model index)
MAX_SINGLE_TRANSFORMER_BLOCK_INDEX = 39


def convert_lora(input_file, output_file):
    """
    Converts a LoRA from the original non-working format
    to the format expected by studio.py.
    Attempts conversion even if ranks mismatch, reporting the issue.
    """
    # --- Load Source LoRA ---
    print(f"\nLoading source LoRA: {input_file}")
    try:
        source_sd = load_file(input_file)
        with safe_open(input_file, framework="pt") as f:
             source_metadata = f.metadata() or {}
    except Exception as e:
        print(f"Error loading source file {input_file}: {e}")
        print("Conversion aborted.")
        return

    print(f"Found {len(source_sd)} keys in source.")
    # Filter for lora_A keys ending in .weight and sort them for consistent processing order
    source_lora_a_keys = sorted([k for k in source_sd.keys() if k.endswith('.lora_A.weight')])
    print(f"Found {len(source_lora_a_keys)} lora_A keys in source.")


    # --- Check Global Rank (for info/warning) ---
    source_global_rank = None
    if 'ss_network_dim' in source_metadata:
        try:
            source_global_rank = int(source_metadata['ss_network_dim'])
        except ValueError:
            pass # Metadata is not a valid integer

    if source_global_rank is not None:
        print(f"Source LoRA Global Rank (from metadata): {source_global_rank}")
    else:
        print("Could not determine source global rank from metadata.")

    print(f"Target System Default Rank: {TARGET_DEFAULT_RANK}")

    if source_global_rank is not None and source_global_rank != TARGET_DEFAULT_RANK:
        print("\n" + "="*50)
        print(" POTENTIAL INCOMPATIBILITY: Global LoRA Ranks Do Not Match! ".center(50))
        print(f"Source Metadata Rank: {source_global_rank}, Target Expected Rank: {TARGET_DEFAULT_RANK}".center(50))
        print("The converted file format will match the target, but functional")
        print("compatibility is NOT guaranteed due to different training ranks.")
        print("="*50 + "\n")


    new_sd = {}
    processed_modules = set() # Keep track of target module bases to add alpha later
    skipped_keys = []

    print("\nStarting key conversion...")
    for key_a in source_lora_a_keys:
        # Use regex to parse key: (diffusion_model|transformer).{block_type_source}.{index}.{module_name_source}.lora_A.weight
        match = re.match(r'(diffusion_model|transformer)\.(.*?)\.(\d+)\.(.*?)\.lora_A\.weight', key_a)
        if not match:
            print(f"Skipping key with unparseable format: {key_a}")
            skipped_keys.append(key_a)
            continue

        source_prefix, block_type_source, index_str, module_name_source = match.groups()
        index = int(index_str)

        # --- Apply Mapping & Process ---
        if block_type_source not in BLOCK_TYPE_MAP_SOURCE_TO_MODEL:
            # print(f"Skipping module due to unknown source block type: {key_a}") # Skip silently
            skipped_keys.append(key_a)
            continue

        target_block_model_name = BLOCK_TYPE_MAP_SOURCE_TO_MODEL[block_type_source] # e.g., 'transformer_blocks', 'single_transformer_blocks'

        if module_name_source not in LAYER_MAP_SOURCE_TO_MODEL:
            # print(f"Skipping module due to unmapped source module name: {key_a}") # Skip silently
            skipped_keys.append(key_a)
            continue

        mapping_info = LAYER_MAP_SOURCE_TO_MODEL[module_name_source]
        target_model_layer_names = mapping_info['target'] # These are like 'attn.to_k', 'ff.net.0.proj'
        split_B = mapping_info.get('split_B', 1)

        # Check index boundary based on target architecture
        max_target_index = -1
        if target_block_model_name == 'transformer_blocks':
             max_target_index = MAX_TRANSFORMER_BLOCK_INDEX
        elif target_block_model_name == 'single_transformer_blocks':
             max_target_index = MAX_SINGLE_TRANSFORMER_BLOCK_INDEX

        if index > max_target_index:
             print(f"Skipping {block_type_source} key with index {index} > max target index {max_target_index} for block type '{target_block_model_name}'. Key: {key_a}")
             skipped_keys.append(key_a)
             continue


        # Find the corresponding lora_B tensor
        key_b = key_a.replace('.lora_A.weight', '.lora_B.weight')
        if key_b not in source_sd:
             print(f"Warning: Found lora_A key '{key_a}', but no corresponding lora_B key '{key_b}'. Skipping module.")
             skipped_keys.append(key_a) # Skip the A key if B is missing
             continue

        source_A = source_sd[key_a].to(torch.bfloat16) # Ensure bfloat16 dtype
        source_B = source_sd[key_b].to(torch.bfloat16) # Ensure bfloat16 dtype
        source_layer_rank = source_A.shape[0]

        # Validate shapes and add to new_sd
        try:
            # Check source rank against target rank (per layer)
            # While we convert ranks that don't match globally, per-layer mismatch
            # is still a warning sign, but we will proceed with conversion.
            # Note: This check is comparing the source layer's rank to the *target system's*
            # default rank, not the rank the source LoRA was trained at if different
            # from its ss_network_dim metadata (unlikely but possible).
            # The primary shape validation is for the output dimension.
            if source_layer_rank != TARGET_DEFAULT_RANK:
                 print(f"Warning: Layer rank mismatch for '{key_a}': Source rank {source_layer_rank}, Target expected rank {TARGET_DEFAULT_RANK}.")
                 # We *still* proceed if output dim matches, this is just informational.


            if split_B > 1:
                # Handle QKV split case
                if not target_model_layer_names or len(target_model_layer_names) != split_B:
                     print(f"Warning: Split count {split_B} does not match target layer count {len(target_model_layer_names)} for mapping source module '{module_name_source}'. Skipping module: {key_a}")
                     skipped_keys.extend([key_a, key_b])
                     continue

                # Get the expected output dimension for one split part using the first target layer name
                # The lookup key should be the target layer name with dots replaced by underscores.
                first_target_layer_lookup_key = target_model_layer_names[0].replace('.', '_')

                expected_out_dim_per_split = TARGET_OUT_DIMS.get(target_block_model_name, {}).get(first_target_layer_lookup_key)

                if expected_out_dim_per_split is None:
                     print(f"Warning: Could not find expected output dimension for target layer '{target_model_layer_names[0]}' (lookup key '{first_target_layer_lookup_key}') in block '{target_block_model_name}' in TARGET_OUT_DIMS. Skipping module: {key_a}")
                     skipped_keys.extend([key_a, key_b])
                     continue

                expected_total_out_dim = expected_out_dim_per_split * split_B

                # Check output dimension consistency for B
                if source_B.shape[0] != expected_total_out_dim:
                    print(f"Warning: Output dimension mismatch for '{key_a}': Expected lora_B output dim {expected_total_out_dim} (based on target architecture lookup for '{target_model_layer_names[0]}'), but got {source_B.shape[0]}. Skipping module.")
                    skipped_keys.extend([key_a, key_b])
                    continue # Skip module if output dimension doesn't match


                # Split B tensor
                split_size = source_B.shape[0] // split_B
                b_tensors_split = torch.split(source_B, split_size, dim=0)

                # Map A and split B parts to target layers
                for i, target_model_layer_name in enumerate(target_model_layer_names):
                    target_layer_name_underscores = target_model_layer_name.replace('.', '_')
                    target_module_base = f"{target_block_model_name}_{index}_{target_layer_name_underscores}"
                    # Clone source_A when assigning to multiple target keys derived from one source A
                    new_sd[f"lora_unet_{target_module_base}.lora_down.weight"] = source_A.clone()
                    new_sd[f"lora_unet_{target_module_base}.lora_up.weight"] = b_tensors_split[i]
                    processed_modules.add(f"lora_unet_{target_module_base}") # Add the full lora_unet_... base

            else:
                # Simple A/B mapping
                if len(target_model_layer_names) != 1:
                     print(f"Warning: Simple mapping defined for source module '{module_name_source}', but target count is {len(target_model_layer_names)} (expected 1). Skipping module: {key_a}")
                     skipped_keys.extend([key_a, key_b])
                     continue

                target_model_layer_name = target_model_layer_names[0] # e.g., 'proj_out', 'ff.net.2'
                target_layer_name_underscores = target_model_layer_name.replace('.', '_')
                target_module_base = f"{target_block_model_name}_{index}_{target_layer_name_underscores}"

                # Get the expected output dimension for the target layer
                expected_out_dim = TARGET_OUT_DIMS.get(target_block_model_name, {}).get(target_layer_name_underscores)

                if expected_out_dim is None:
                     print(f"Warning: Could not find expected output dimension for target layer '{target_model_layer_name}' (lookup key '{target_layer_name_underscores}') in block '{target_block_model_name}' in TARGET_OUT_DIMS. Skipping module: {key_a}")
                     skipped_keys.extend([key_a, key_b])
                     continue

                # Check output dimension consistency for B
                if source_B.shape[0] != expected_out_dim:
                    print(f"Warning: Output dimension mismatch for '{key_a}': Expected lora_B output dim {expected_out_dim} (based on target architecture lookup for '{target_model_layer_name}'), but got {source_B.shape[0]}. Skipping module.")
                    skipped_keys.extend([key_a, key_b])
                    continue # Skip module if output dimension doesn't match


                # If shape check passed, add to new_sd
                new_sd[f"lora_unet_{target_module_base}.lora_down.weight"] = source_A
                new_sd[f"lora_unet_{target_module_base}.lora_up.weight"] = source_B
                processed_modules.add(f"lora_unet_{target_module_base}") # Add the full lora_unet_... base

        except Exception as e:
            print(f"Error processing module {key_a}: {e}. Skipping.")
            traceback.print_exc() # Print full traceback for unexpected errors
            skipped_keys.extend([key_a, key_b]) # Mark both A and B as skipped
            continue

    # --- Add Alpha tensors for processed modules ---
    # Use the alpha value from the working file analysis metadata
    alpha_value = 1.0
    # Use the dtype from the working file alpha tensors
    alpha_tensor_dtype = torch.bfloat16

    print(f"\nAdding alpha tensors (value={alpha_value}, dtype={alpha_tensor_dtype}) for {len(processed_modules)} mapped modules.")

    for module_base in sorted(list(processed_modules)): # Sort for consistent output order
         # Ensure alpha tensor is created fresh for each module base and has correct dtype
         new_sd[f"{module_base}.alpha"] = torch.tensor(alpha_value, dtype=alpha_tensor_dtype)


    # --- Add Metadata ---
    metadata = {}
    # Report the target expected rank, as that's what the system expects
    metadata['ss_network_dim'] = str(TARGET_DEFAULT_RANK)
    metadata['ss_network_alpha'] = str(alpha_value) # Store the *value* used for alpha
    metadata['format'] = 'pt'
    metadata['source_file'] = os.path.basename(input_file)
    metadata['source_rank'] = str(source_global_rank) if source_global_rank is not None else 'Unknown'
    metadata['target_expected_rank'] = str(TARGET_DEFAULT_RANK)
    metadata['conversion_script'] = os.path.basename(__file__)
    metadata['mapped_modules_count'] = str(len(processed_modules))
    metadata['source_lora_key_count'] = str(len(source_sd))
    metadata['converted_lora_key_count'] = str(len(new_sd))

    # Add a warning to metadata if ranks didn't match
    if source_global_rank is not None and source_global_rank != TARGET_DEFAULT_RANK:
        metadata['conversion_warning'] = f"Rank mismatch: Source={source_global_rank}, Target Expected={TARGET_DEFAULT_RANK}. Functional compatibility is not guaranteed."

    # Add list of skipped keys (up to a limit)
    if skipped_keys:
        metadata['skipped_source_keys'] = f"{len(skipped_keys)} keys skipped. Examples: {skipped_keys[:10]}{'...' if len(skipped_keys) > 10 else ''}"
        print(f"\nSkipped {len(skipped_keys)} source keys (check metadata for details).")


    print(f"\nSaving converted LoRA to: {output_file}")
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        save_file(new_sd, output_file, metadata=metadata)
        print("\nConversion complete.")
        print(f"Converted {len(new_sd)} keys ({len(processed_modules)} modules) from {len(source_sd)} source keys.")
        # Original modules calculation assumes A/B pairs
        original_modules_count = len([k for k in source_sd.keys() if k.endswith('.lora_A.weight')])
        print(f"Original LoRA had {original_modules_count} modules (lora_A keys).")
        print(f"Skipped {len(skipped_keys)} keys during conversion.")

        if source_global_rank is not None and source_global_rank != TARGET_DEFAULT_RANK:
             print("\nRemember the rank mismatch warning above! The converted file format is correct, but it may not work functionally.")

    except Exception as e:
        print(f"Error saving converted file {output_file}: {e}")
        print("Conversion failed during save.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a specific HunyuanVideo LoRA format (diffusion_model.double/single_blocks) to the format expected by studio.py (lora_unet_transformer/single_transformer_blocks). Attempts conversion even if ranks mismatch, reporting the issue."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the source LoRA Safetensors file (the one you want to convert)."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for the output converted LoRA Safetensors file."
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found at '{args.input}'")
        exit(1)

    convert_lora(args.input, args.output)

if __name__ == "__main__":
    main()