import argparse
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

import os
import re # Import regex for key parsing
from collections import defaultdict # To easily build the shape dict

# Define the mapping from source key parts to target model's internal layer names and logic
# Source format: transformer.{block_type}.{index}.{module_name}.lora_A/B.weight
# Target internal layer name format (from index.json): {block_name}.{index}.{layer_name}
# Final LoRA key format: lora_unet_{block_name}_{index}_{layer_name_underscores}.lora_down/up.weight (.alpha)

# block_type_map: Maps source block type to target block internal name prefix
BLOCK_TYPE_MAP_SOURCE_TO_MODEL = {
    'double_blocks': 'transformer_blocks',
    'single_blocks': 'single_transformer_blocks',
}

# layer_map: Maps source module name to target model layer name(s) and required processing
# Key: Source module name (from original non-working LoRA)
# Value: Dictionary describing target(s)
#    'target': List of target model layer names (from index.json, dots included)
#    'split_B': Number of ways to split source B tensor along dim 0 (if > 1, for fused QKV)
LAYER_MAP_SOURCE_TO_MODEL = {
    # --- Mappings for source double_blocks -> target transformer_blocks ---
    # Source img_attn_qkv (A: (32, 3072), B: (9216, 32)) -> Target attn.to_k/q/v (down: (32, 3072), up: (3072, 32) each)
    'img_attn_qkv': {'target': ['attn.to_k', 'attn.to_q', 'attn.to_v'], 'split_B': 3},
    # Source img_attn_proj (A: (32, 3072), B: (3072, 32)) -> Target attn.to_out.0 (down: (32, 3072), up: (3072, 32))
    'img_attn_proj': {'target': ['attn.to_out.0']},
    # Source img_mlp.fc1 (A: (32, 3072), B: (12288, 32)) -> Target ff.net.0.proj (down: (32, 3072), up: (12288, 32))
    'img_mlp.fc1': {'target': ['ff.net.0.proj']},
    # Source img_mlp.fc2 (A: (32, 12288), B: (3072, 32)) -> Target ff.net.2 (down: (32, 12288), up: (3072, 32))
    'img_mlp.fc2': {'target': ['ff.net.2']},

    # Source txt_attn_qkv (A: (32, 3072), B: (9216, 32)) -> Target attn.add_k/q/v_proj (down: (32, 3072), up: (3072, 32) each)
    'txt_attn_qkv': {'target': ['attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj'], 'split_B': 3},
    # Source txt_attn_proj (A: (32, 3072), B: (3072, 32)) -> Target attn.to_add_out (down: (32, 3072), up: (3072, 32))
    'txt_attn_proj': {'target': ['attn.to_add_out']},
    # Source txt_mlp.fc1 (A: (32, 3072), B: (12288, 32)) -> Target ff_context.net.0.proj (down: (32, 3072), up: (12288, 32))
    'txt_mlp.fc1': {'target': ['ff_context.net.0.proj']},
    # Source txt_mlp.fc2 (A: (32, 12288), B: (3072, 32)) -> Target ff_context.net.2 (down: (32, 12288), up: (3072, 32))
    'txt_mlp.fc2': {'target': ['ff_context.net.2']},

    # --- Mappings for source single_blocks -> target single_transformer_blocks ---
    # Source linear1 (A: (32, 3072), B: (21504, 32)) -> Target proj_mlp (down: (32, 3072), up: (12288, 32)) - DIMENSION MISMATCH ON B, SKIPPED
    # 'linear1': {'target': ['proj_mlp']}, # Skip due to dim mismatch
    # Source linear2 (A: (32, 15360), B: (3072, 32)) -> Target proj_out (down: (32, 15360), up: (3072, 32))
    'linear2': {'target': ['proj_out']},
    # Source modulation.linear (A: (32, 3072), B: (9216, 32)) - No obvious mapping with compatible dimensions in target
    # 'modulation.linear': {'target': [...]}, # Skip due to no clear mapping
}

# Max index for target transformer_blocks (from working file analysis/model index)
MAX_TRANSFORMER_BLOCK_INDEX = 11
# Max index for target single_transformer_blocks (from working file analysis/model index)
MAX_SINGLE_TRANSFORMER_BLOCK_INDEX = 39

# --- DYNAMICALLY BUILD TARGET_OUT_DIMS FROM WORKING FILE ---
# This helps ensure we use the correct expected output dimensions based on the actual model architecture
# represented by the working LoRA keys.
def build_target_out_dims_from_working_file(working_file_path):
    """
    Analyzes the working LoRA file to determine the expected output dimensions
    for each target layer based on the shape of its .lora_up.weight tensor.
    """
    print(f"Analyzing working file {working_file_path} to build target output dimensions map...")
    working_sd = load_file(working_file_path)
    target_out_dims = defaultdict(dict) # Nested dict for block_name -> {layer_name_underscores -> out_dim}

    # Regex to parse working LoRA keys: lora_unet_{block_name}_{index}_{layer_name_underscores}.lora_down/up.weight
    # We need to extract block_name and layer_name_underscores
    # block_name examples: single_transformer_blocks, transformer_blocks
    # layer_name_underscores examples: attn_to_k, proj_mlp, ff_context_net_0_proj

    for key, tensor in working_sd.items():
        if not key.startswith('lora_unet_') or not key.endswith('.lora_up.weight'):
            continue # Only process target format lora_up weights

        # Remove prefix and suffix
        core_name = key[len('lora_unet_'):-len('.lora_up.weight')] # e.g., 'single_transformer_blocks_0_attn_to_k'

        # Find the index part by looking for '_<number>_'
        index_match = re.search(r'_(\d+)_', core_name)
        if not index_match:
             print(f"Warning: Could not parse index in working key: {key}. Skipping.")
             continue

        index_start, index_end = index_match.span() # Get start and end index of the matched '_<number>_'
        block_name = core_name[:index_start] # Part before the index, e.g., 'single_transformer_blocks' or 'transformer_blocks'
        layer_name_underscores = core_name[index_end:] # Part after the index, e.g., 'attn_to_k' or 'ff_context_net_0_proj'

        if not block_name or not layer_name_underscores:
             print(f"Warning: Could not split block and layer name from working key: {key}. Skipping.")
             continue

        # The out_dim is the first dimension of the .lora_up.weight
        out_dim = tensor.shape[0]

        target_out_dims[block_name][layer_name_underscores] = out_dim

    print(f"Built target output dimensions map from {working_file_path}.")
    # print("Mapped dimensions:", target_out_dims) # Debug print the built map
    return dict(target_out_dims) # Convert back to regular dict


def convert_lora(input_file, output_file, working_file_for_dims=None):
    """
    Converts a LoRA from the original non-working format
    to the working format expected by studio.py.
    Optionally takes a working file path to dynamically get target dimensions.
    """
    print(f"Loading source LoRA: {input_file}")
    source_sd = load_file(input_file)
    print(f"Found {len(source_sd)} keys in source.")

    target_out_dims_map = {}
    if working_file_for_dims and os.path.exists(working_file_for_dims):
        target_out_dims_map = build_target_out_dims_from_working_file(working_file_for_dims)
        if not target_out_dims_map:
             print("Error: Could not build target dimensions map from the working file. Please check the working file structure.")
             return # Stop if we couldn't build the map
    else:
        print("Error: Working file path is required to determine target dimensions.")
        print("Please provide the path to a working LoRA using the --working_lora argument.")
        return # Stop if working file not provided/found


    new_sd = {}
    processed_modules = set() # Keep track of target module bases to add alpha later

    # Collect lora_A keys to iterate over
    lora_a_keys = sorted([k for k in source_sd.keys() if k.endswith('.lora_A.weight')])

    print(f"Found {len(lora_a_keys)} lora_A keys to process.")

    for key_a in lora_a_keys:
        # Use regex to parse key: transformer.{block_type_source}.{index}.{module_name_source}.lora_A.weight
        match = re.match(r'transformer\.(.*?)\.(\d+)\.(.*?)\.lora_A\.weight', key_a)
        if not match:
            print(f"Skipping key with unparseable format: {key_a}")
            continue

        block_type_source, index_str, module_name_source = match.groups()
        index = int(index_str)

        # --- Apply Mapping & Process ---
        if block_type_source not in BLOCK_TYPE_MAP_SOURCE_TO_MODEL:
            # print(f"Skipping module due to unknown block type: {key_a}") # Skip silently
            continue

        target_block_model_name = BLOCK_TYPE_MAP_SOURCE_TO_MODEL[block_type_source] # e.g., 'transformer_blocks', 'single_transformer_blocks'

        if module_name_source not in LAYER_MAP_SOURCE_TO_MODEL:
            # print(f"Skipping module due to unmapped module name: {key_a}") # Skip silently
            continue

        mapping_info = LAYER_MAP_SOURCE_TO_MODEL[module_name_source]
        target_model_layer_names = mapping_info['target'] # These are like 'attn.to_k', 'ff.net.0.proj'
        split_B = mapping_info.get('split_B', 1)

        # Check index boundary based on target architecture
        if target_block_model_name == 'transformer_blocks' and index > MAX_TRANSFORMER_BLOCK_INDEX:
             # print(f"Skipping {block_type_source} key with index {index} > max target index {MAX_TRANSFORMER_BLOCK_INDEX}. Key: {key_a}") # Skip silently
             continue
        if target_block_model_name == 'single_transformer_blocks' and index > MAX_SINGLE_TRANSFORMER_BLOCK_INDEX:
             # print(f"Skipping {block_type_source} key with index {index} > max target index {MAX_SINGLE_TRANSFORMER_BLOCK_INDEX}. Key: {key_a}") # Skip silently
             continue

        # Find the corresponding lora_B tensor
        key_b = key_a.replace('.lora_A.weight', '.lora_B.weight')
        if key_b not in source_sd:
             print(f"Warning: Found lora_A for {module_name_source}, but no corresponding lora_B. Skipping module: {key_a}")
             continue

        source_A = source_sd[key_a].to(torch.bfloat16) # Ensure bfloat16 dtype
        source_B = source_sd[key_b].to(torch.bfloat16) # Ensure bfloat16 dtype
        rank = source_A.shape[0]


        # Validate shapes and add to new_sd
        try:
            if split_B > 1:
                # Handle QKV split case
                if not target_model_layer_names:
                     print(f"Warning: Split mapping defined for {module_name_source}, but no target layer names provided. Skipping module: {key_a}")
                     continue

                # Get the expected output dimension for one split part using the first target layer name
                first_target_layer_name = target_model_layer_names[0] # e.g., 'attn.to_k'
                # Lookup in TARGET_OUT_DIMS uses underscore format
                target_layer_key_for_lookup = first_target_layer_name.replace('.', '_')

                expected_out_dim_per_split = target_out_dims_map.get(target_block_model_name, {}).get(target_layer_key_for_lookup)

                if expected_out_dim_per_split is None:
                     print(f"Warning: Could not find expected output dimension for target layer '{first_target_layer_name}' (lookup key '{target_layer_key_for_lookup}') in block '{target_block_model_name}'. Skipping module: {key_a}")
                     # Debug Info: Print available keys if lookup fails
                     # print(f"  Debug Info: Available keys in target_out_dims_map['{target_block_model_name}'] are: {list(target_out_dims_map.get(target_block_model_name, {}).keys())}")
                     continue

                expected_total_out_dim = expected_out_dim_per_split * split_B
                if source_B.shape[0] != expected_total_out_dim:
                    print(f"Warning: Shape mismatch for {key_a}: Expected lora_B output dim {expected_total_out_dim}, but got {source_B.shape[0]}. Skipping module.")
                    continue

                if len(target_model_layer_names) != split_B:
                     print(f"Warning: Split count {split_B} does not match target layer count {len(target_model_layer_names)} for {key_a}. Skipping module.")
                     continue

                # Split B tensor
                split_size = source_B.shape[0] // split_B
                b_tensors_split = torch.split(source_B, split_size, dim=0)

                # Map A and split B parts to target layers
                for i, target_model_layer_name in enumerate(target_model_layer_names):
                    # Construct the final LoRA key using underscores for the model layer name part
                    target_layer_name_underscores = target_model_layer_name.replace('.', '_')
                    target_module_base = f"{target_block_model_name}_{index}_{target_layer_name_underscores}"
                    # *** FIX: Clone source_A when assigning to multiple target keys ***
                    new_sd[f"lora_unet_{target_module_base}.lora_down.weight"] = source_A.clone()
                    new_sd[f"lora_unet_{target_module_base}.lora_up.weight"] = b_tensors_split[i]
                    processed_modules.add(f"lora_unet_{target_module_base}") # Add the full lora_unet_... base

            else:
                # Simple A/B mapping
                if len(target_model_layer_names) != 1:
                     print(f"Warning: Simple mapping defined for {module_name_source}, but target count is {len(target_model_layer_names)} (expected 1). Skipping module: {key_a}")
                     continue
                target_model_layer_name = target_model_layer_names[0] # e.g., 'proj_out', 'ff.net.2'
                target_layer_name_underscores = target_model_layer_name.replace('.', '_')
                target_module_base = f"{target_block_model_name}_{index}_{target_layer_name_underscores}"


                # Get the expected output dimension for the target layer
                # Lookup in TARGET_OUT_DIMS uses underscore format
                expected_out_dim = target_out_dims_map.get(target_block_model_name, {}).get(target_layer_name_underscores)

                if expected_out_dim is None:
                     print(f"Warning: Could not find expected output dimension for target layer '{target_model_layer_name}' (lookup key '{target_layer_name_underscores}') in block '{target_block_model_name}'. Skipping module: {key_a}")
                     # Debug Info: Print available keys if lookup fails
                     # print(f"  Debug Info: Available keys in target_out_dims_map['{target_block_model_name}'] are: {list(target_out_dims_map.get(target_block_model_name, {}).keys())}")
                     continue


                if source_B.shape[0] != expected_out_dim:
                    print(f"Warning: Shape mismatch for {key_a}: Expected lora_B output dim {expected_out_dim}, but got {source_B.shape[0]}. Skipping module.")
                    continue

                # If shape check passed, add to new_sd
                new_sd[f"lora_unet_{target_module_base}.lora_down.weight"] = source_A
                new_sd[f"lora_unet_{target_module_base}.lora_up.weight"] = source_B
                processed_modules.add(f"lora_unet_{target_module_base}") # Add the full lora_unet_... base

        except Exception as e:
            print(f"Error processing module {key_a}: {e}. Skipping.")
            import traceback
            traceback.print_exc() # Print full traceback for unexpected errors
            continue


    # --- Add Alpha tensors for processed modules ---
    rank = 32 # Based on analysis
    alpha_tensor_value = 1.0 # Value from working file analysis
    alpha_tensor_dtype = torch.bfloat16 # Dtype from working file analysis

    print(f"\nAdding alpha tensors (value={alpha_tensor_value}, dtype={alpha_tensor_dtype}) for {len(processed_modules)} mapped modules.")

    for module_base in sorted(list(processed_modules)): # Sort for consistent output order
         # Ensure alpha tensor is also created fresh for each module base
         new_sd[f"{module_base}.alpha"] = torch.tensor(alpha_tensor_value, dtype=alpha_tensor_dtype)


    # --- Add Minimal Metadata ---
    # Based on studio.py convert_to_diffusers logic and working file metadata:
    metadata = {}
    metadata['ss_network_dim'] = str(rank)
    metadata['ss_network_alpha'] = str(alpha_tensor_value) # Store the *value* used for alpha
    metadata['format'] = 'pt' # Indicate framework used for tensors
    metadata['source_file'] = os.path.basename(input_file)
    metadata['conversion_script'] = os.path.basename(__file__)
    metadata['mapped_modules_count'] = str(len(processed_modules))


    print(f"Saving converted LoRA to: {output_file}")
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    save_file(new_sd, output_file, metadata=metadata)
    print("Conversion complete.")
    print(f"Converted {len(new_sd)} keys ({len(processed_modules)} modules).")
    print(f"Original LoRA had {len(source_sd)//2} modules (lora_A/B pairs).")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a specific HunyuanVideo LoRA format (transformer.double/single_blocks) to the format expected by studio.py (lora_unet_transformer/single_transformer_blocks)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the source LoRA Safetensors file (e.g., the original non-working one)."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for the output converted LoRA Safetensors file."
    )
    parser.add_argument(
        "--working_lora",
        type=str,
        required=True, # Make working lora required to dynamically build target dims
        help="Path to a known working LoRA Safetensors file (used to determine target dimensions)."
    )


    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found at '{args.input}'")
        exit(1)
    if not os.path.exists(args.working_lora):
        print(f"Error: Working LoRA file not found at '{args.working_lora}'")
        exit(1)

    # Perform the conversion
    convert_lora(args.input, args.output, working_file_for_dims=args.working_lora)

if __name__ == "__main__":
    main()