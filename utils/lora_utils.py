import os
import re
import sys
from typing import Dict, List, Optional, Union
import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from utils.safetensors_utils import MemoryEfficientSafeOpen, load_safetensors


def filter_lora_state_dict(
    weights_sd: Dict[str, torch.Tensor],
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Filter LoRA state dict based on include/exclude patterns.
    
    Args:
        weights_sd: Dictionary of LoRA weights
        include_pattern: Regex pattern to include keys
        exclude_pattern: Regex pattern to exclude keys
        
    Returns:
        Filtered dictionary of LoRA weights
    """
    # Apply include/exclude patterns
    original_key_count = len(weights_sd.keys())
    
    if include_pattern is not None:
        regex_include = re.compile(include_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
        logger.info(f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}")

    if exclude_pattern is not None:
        original_key_count_ex = len(weights_sd.keys())
        regex_exclude = re.compile(exclude_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if not regex_exclude.search(k)}
        logger.info(f"Filtered keys with exclude pattern {exclude_pattern}: {original_key_count_ex} -> {len(weights_sd.keys())}")

    if len(weights_sd) != original_key_count:
        remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys()]))
        remaining_keys.sort()
        logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
        if len(weights_sd) == 0:
            logger.warning(f"No keys left after filtering.")

    return weights_sd


def load_safetensors_with_lora_and_fp8(
    model_files: Union[str, List[str]],
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]],
    lora_multipliers: Optional[List[float]],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model with fp8 optimization if needed.
    This function loads model weights and merges LoRA weights on-the-fly to save memory.

    Args:
        model_files: Path to the model file or list of paths
        lora_weights_list: List of LoRA weight dictionaries to merge
        lora_multipliers: List of multipliers for LoRA weights
        fp8_optimization: Whether to apply FP8 optimization (not used in this version)
        calc_device: Device to perform calculations on
        move_to_device: Whether to move tensors to the calculation device after loading
        target_keys: Keys to target for optimization (not used in this version)
        exclude_keys: Keys to exclude from optimization (not used in this version)
        
    Returns:
        Merged state dictionary
    """
    # Ensure model_files is a list
    if isinstance(model_files, str):
        model_files = [model_files]

    # Handle file patterns like "00001-of-00004"
    extended_model_files = []
    for model_file in model_files:
        basename = os.path.basename(model_file)
        match = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", basename)
        if match:
            prefix = basename[: match.start(2)]
            count = int(match.group(3))
            for i in range(count):
                filename = f"{prefix}{i+1:05d}-of-{count:05d}.safetensors"
                filepath = os.path.join(os.path.dirname(model_file), filename)
                if os.path.exists(filepath):
                    extended_model_files.append(filepath)
                else:
                    raise FileNotFoundError(f"File {filepath} not found")
        else:
            extended_model_files.append(model_file)

    # Deduplicate while preserving order: when a caller passes every shard of a
    # sharded checkpoint, the pattern expansion above would otherwise repeat the
    # full shard set once per input file (N^2 loads).
    model_files = list(dict.fromkeys(extended_model_files))
    logger.info(f"Loading model files: {model_files}")

    # Prepare LoRA weights
    if lora_weights_list is None or len(lora_weights_list) == 0:
        # No LoRA weights, just load the model normally
        state_dict = {}
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file) as f:
                for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", leave=False, miniters=100, file=sys.stdout, dynamic_ncols=False):
                    value = f.get_tensor(key)
                    if move_to_device:
                        value = value.to(calc_device)
                    state_dict[key] = value
        return state_dict

    # Prepare LoRA weight lookups for efficient access
    list_of_lora_weight_keys = []
    for lora_sd in lora_weights_list:
        lora_weight_keys = set(lora_sd.keys())
        list_of_lora_weight_keys.append(lora_weight_keys)

    # Prepare multipliers
    if lora_multipliers is None:
        lora_multipliers = [1.0] * len(lora_weights_list)
    while len(lora_multipliers) < len(lora_weights_list):
        lora_multipliers.append(1.0)
    if len(lora_multipliers) > len(lora_weights_list):
        lora_multipliers = lora_multipliers[: len(lora_weights_list)]

    logger.info(f"Merging LoRA weights into state dict. Multipliers: {lora_multipliers}")

    # Helper function to convert LoRA key to model key format
    def convert_lora_key_to_model_key(lora_key):
        """
        Convert LoRA key to model key format, handling _img suffix and MUSUBI format
        Examples:
        - lora_unet_blocks_0_cross_attn_k_diff_b → blocks.0.cross_attn.k.bias
        - lora_unet_blocks_0_cross_attn_k_img_diff_b → blocks.0.cross_attn.k.bias
        - diffusion_model.blocks.0.cross_attn.k.diff → blocks.0.cross_attn.k.weight
        """
        # Handle _img suffix by stripping it
        original_key = lora_key
        if '_img' in lora_key:
            lora_key = lora_key.replace('_img', '')

        # Handle MUSUBI format (lora_unet_ prefix)
        if lora_key.startswith('lora_unet_'):
            key = lora_key[len('lora_unet_'):]

            # Parse blocks.N pattern specially
            if key.startswith('blocks_'):
                # Split only first two underscores for block number
                parts = key.split('_', 2)  # ['blocks', '0', 'cross_attn_k_diff_b']
                if len(parts) >= 3:
                    block_part = f"blocks.{parts[1]}"
                    rest = parts[2]

                    # Determine parameter type
                    if rest.endswith('_diff_b'):
                        param = rest[:-7]  # Remove _diff_b
                        return f"{block_part}.{param}.bias"
                    elif rest.endswith('_diff_m'):
                        param = rest[:-7]  # Remove _diff_m
                        return f"{block_part}.{param}.modulation"
                    elif rest.endswith('_diff'):
                        param = rest[:-5]  # Remove _diff
                        return f"{block_part}.{param}.weight"
            else:
                # Handle non-block keys (like embedding, time_in, etc.)
                if key.endswith('_diff_b'):
                    param = key[:-7]  # Remove _diff_b
                    return f"{param}.bias"
                elif key.endswith('_diff_m'):
                    param = key[:-7]  # Remove _diff_m
                    return f"{param}.modulation"
                elif key.endswith('_diff'):
                    param = key[:-5]  # Remove _diff
                    return f"{param}.weight"

        # Handle original lightx2v format (diffusion_model. prefix)
        elif lora_key.startswith('diffusion_model.'):
            key = lora_key[len('diffusion_model.'):]

            if key.endswith('.diff_b'):
                return key[:-7] + '.bias'
            elif key.endswith('.diff_m'):
                return key[:-7] + '.modulation'
            elif key.endswith('.diff'):
                return key[:-5] + '.weight'

        return None

    # Precompute diff-key lookup (model key -> LoRA keys) once per LoRA, instead of
    # rescanning and re-converting every LoRA key for every model tensor in the load loop
    list_of_diff_lookups = []
    for lora_sd in lora_weights_list:
        diff_lookup = {}
        for lora_key in lora_sd.keys():
            if 'diff' in lora_key and 'lora_down' not in lora_key and 'lora_up' not in lora_key:
                converted_model_key = convert_lora_key_to_model_key(lora_key)
                if converted_model_key is not None:
                    diff_lookup.setdefault(converted_model_key, []).append(lora_key)
        list_of_diff_lookups.append(diff_lookup)

    # Helper to resolve lora_down/lora_up key pair for a model weight key
    def find_lora_pair(model_weight_key, lora_weight_keys):
        """Return (down_key, up_key, alpha_key) for this model key, or None if no match"""
        # Remove .weight suffix and handle different prefixes
        base_key = model_weight_key.rsplit(".", 1)[0]

        # Remove diffusion_model prefix if present (for 1.3B models)
        base_key_no_prefix = base_key
        if base_key.startswith("diffusion_model."):
            base_key_no_prefix = base_key[len("diffusion_model."):]

        # Convert dots to underscores and add lora_unet prefix for LoRA key format
        # "blocks.0.cross_attn.k" -> "lora_unet_blocks_0_cross_attn_k"
        lora_base_key = "lora_unet_" + base_key_no_prefix.replace(".", "_")

        # Direct format: blocks.0.cross_attn.k.lora_down.weight (no prefix conversion)
        down_key_direct = base_key_no_prefix + ".lora_down.weight"
        up_key_direct = base_key_no_prefix + ".lora_up.weight"
        alpha_key_direct = base_key_no_prefix + ".alpha"

        # MUSUBI format: lora_unet_blocks_0_cross_attn_k_lora_down_weight
        down_key_underscores = lora_base_key + "_lora_down_weight"
        up_key_underscores = lora_base_key + "_lora_up_weight"
        alpha_key_underscores = down_key_underscores + ".alpha"

        # MUSUBI _img variants
        down_key_img = lora_base_key + "_img_lora_down_weight"
        up_key_img = lora_base_key + "_img_lora_up_weight"
        alpha_key_img = down_key_img + ".alpha"

        # Standard format: lora_unet_blocks_0_cross_attn_k.lora_down.weight
        down_key_dots = lora_base_key + ".lora_down.weight"
        up_key_dots = lora_base_key + ".lora_up.weight"
        alpha_key_dots = lora_base_key + ".alpha"

        # LightX2V format: diffusion_model.blocks.0.cross_attn.k.lora_down.weight
        down_key_lightx2v = "diffusion_model." + base_key_no_prefix + ".lora_down.weight"
        up_key_lightx2v = "diffusion_model." + base_key_no_prefix + ".lora_up.weight"
        alpha_key_lightx2v = "diffusion_model." + base_key_no_prefix + ".alpha"

        # Try direct format first (for new MUSUBI LoRAs)
        if down_key_direct in lora_weight_keys and up_key_direct in lora_weight_keys:
            return down_key_direct, up_key_direct, alpha_key_direct
        # Try MUSUBI format with underscores
        elif down_key_underscores in lora_weight_keys and up_key_underscores in lora_weight_keys:
            return down_key_underscores, up_key_underscores, alpha_key_underscores
        # Try _img variant keys (MUSUBI format)
        elif down_key_img in lora_weight_keys and up_key_img in lora_weight_keys:
            return down_key_img, up_key_img, alpha_key_img
        # Try standard format with lora_unet prefix and dots
        elif down_key_dots in lora_weight_keys and up_key_dots in lora_weight_keys:
            return down_key_dots, up_key_dots, alpha_key_dots
        # Try lightx2v format last (has diffusion_model. prefix)
        elif down_key_lightx2v in lora_weight_keys and up_key_lightx2v in lora_weight_keys:
            return down_key_lightx2v, up_key_lightx2v, alpha_key_lightx2v
        return None

    # Define the weight hook function for on-the-fly merging
    def weight_hook_func(model_weight_key, model_weight):
        """Hook function to merge LoRA weights as model weights are loaded"""

        # Handle .weight, .bias, and .modulation parameters
        is_weight = model_weight_key.endswith(".weight")
        is_bias = model_weight_key.endswith(".bias")
        is_modulation = model_weight_key.endswith(".modulation")

        if not (is_weight or is_bias or is_modulation):
            return model_weight

        # Resolve which LoRA tensors touch this model weight BEFORE any device moves,
        # so unmatched tensors skip the CPU->GPU->CPU round trip entirely
        planned_merges = []
        for lora_weight_keys, lora_sd, multiplier, diff_lookup in zip(
            list_of_lora_weight_keys, lora_weights_list, lora_multipliers, list_of_diff_lookups
        ):
            diff_keys = [k for k in diff_lookup.get(model_weight_key, ()) if k in lora_weight_keys]
            lora_pair = find_lora_pair(model_weight_key, lora_weight_keys) if is_weight else None
            if diff_keys or lora_pair is not None:
                planned_merges.append((lora_weight_keys, lora_sd, multiplier, diff_keys, lora_pair))

        if not planned_merges:
            return model_weight

        original_device = model_weight.device
        if original_device != calc_device:
            model_weight = model_weight.to(calc_device)  # Move to calc device for faster computation

        for lora_weight_keys, lora_sd, multiplier, diff_keys, lora_pair in planned_merges:
            # Apply lightx2v format (diff/diff_b keys) first
            for lora_key in diff_keys:
                lora_weight = lora_sd[lora_key].to(calc_device)
                model_weight = model_weight + multiplier * lora_weight
                lora_weight_keys.remove(lora_key)

            # Then apply standard LoRA format (lora_down/lora_up)
            if lora_pair is not None:
                down_key, up_key, alpha_key = lora_pair

                # Get LoRA weights
                down_weight = lora_sd[down_key]
                up_weight = lora_sd[up_key]

                dim = down_weight.size()[0]
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim

                # Move to calc device
                down_weight = down_weight.to(calc_device)
                up_weight = up_weight.to(calc_device)

                # Merge LoRA weights: W <- W + U * D * scale * multiplier
                if len(model_weight.size()) == 2:
                    # Linear layer
                    if len(up_weight.size()) == 4:  # Conv2d weights used as linear
                        up_weight = up_weight.squeeze(3).squeeze(2)
                        down_weight = down_weight.squeeze(3).squeeze(2)
                    model_weight = model_weight + multiplier * (up_weight @ down_weight) * scale
                elif down_weight.size()[2:4] == (1, 1):
                    # Conv2d 1x1
                    model_weight = (
                        model_weight
                        + multiplier
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                else:
                    # Conv2d 3x3
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    model_weight = model_weight + multiplier * conved * scale

                # Remove used LoRA keys from tracking set
                lora_weight_keys.remove(down_key)
                lora_weight_keys.remove(up_key)
                if alpha_key in lora_weight_keys:
                    lora_weight_keys.remove(alpha_key)

        # Move back to original device (skip when the caller will move it to calc_device anyway)
        if not move_to_device:
            model_weight = model_weight.to(original_device)
        return model_weight
    # Load model with LoRA merging hook
    state_dict = {}
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)} with LoRA merge", leave=False, miniters=100, file=sys.stdout, dynamic_ncols=False):
                value = f.get_tensor(key)
                
                # Apply the hook to merge LoRA weights
                value = weight_hook_func(key, value)
                
                if move_to_device:
                    value = value.to(calc_device)
                state_dict[key] = value

    # Check for unused LoRA keys
    for i, lora_weight_keys in enumerate(list_of_lora_weight_keys):
        if len(lora_weight_keys) > 0:
            # Filter out non-weight keys (like alpha, diff, etc.)
            unused_weight_keys = [k for k in lora_weight_keys if k.endswith(('.weight', '.alpha', '.diff', '.diff_b'))]
            if unused_weight_keys:
                logger.warning(f"Warning: {len(unused_weight_keys)} LoRA keys were not used from set {i}: {', '.join(list(unused_weight_keys)[:5])}...")

    return state_dict