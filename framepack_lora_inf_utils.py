import os
import re
from typing import Optional
import torch
from safetensors.torch import load_file
from tqdm import tqdm

import logging

from utils.safetensors_utils import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from modules.fp8_optimization_utils import optimize_state_dict_with_fp8_on_the_fly


def merge_lora_to_state_dict(
    model_file: str,
    lora_files: Optional[list[str]],
    multipliers: Optional[list[float]],
    fp8_optimization: bool,
    device: torch.device,
    move_to_device: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model.
    """

    # if the file name ends with 00001-of-00004 etc, we need to load the files with the same prefix
    basename = os.path.basename(model_file)
    match = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", basename)
    if match:
        prefix = basename[: match.start(2)]
        count = int(match.group(3))
        model_files = [os.path.normpath(model_file)]
        for i in range(count):
            file_name = f"{prefix}{i+1:05d}-of-{count:05d}.safetensors"
            file_path = os.path.join(os.path.dirname(model_file), file_name)
            file_path = os.path.normpath(file_path)
            if os.path.exists(file_path) and file_path not in model_files:
                model_files.append(file_path)
        logger.info(f"Loading split weights: {model_files}")
    else:
        model_files = [os.path.normpath(model_file)]

    list_of_lora_sd = []
    if lora_files is not None:
        for lora_file in lora_files:
            # Load LoRA safetensors file
            lora_sd = load_file(lora_file)

            # Check the format of the LoRA file
            keys = list(lora_sd.keys())
            if keys[0].startswith("lora_unet_"):
                logging.info(f"Musubi Tuner LoRA detected")

            else:
                transformer_prefixes = ["diffusion_model", "transformer"]  # to ignore Text Encoder modules
                lora_suffix = None
                prefix = None
                for key in keys:
                    if lora_suffix is None and "lora_A" in key:
                        lora_suffix = "lora_A"
                    if prefix is None:
                        pfx = key.split(".")[0]
                        if pfx in transformer_prefixes:
                            prefix = pfx
                    if lora_suffix is not None and prefix is not None:
                        break

                if lora_suffix == "lora_A" and prefix is not None:
                    logging.info(f"Diffusion-pipe (?) LoRA detected")
                    lora_sd = convert_from_diffusion_pipe_or_something(lora_sd, "lora_unet_")

                else:
                    logging.info(f"LoRA file format not recognized: {os.path.basename(lora_file)}")
                    lora_sd = None

            if lora_sd is not None:
                # Check LoRA is for FramePack or for HunyuanVideo
                is_hunyuan = False
                for key in lora_sd.keys():
                    if "double_blocks" in key or "single_blocks" in key:
                        is_hunyuan = True
                        break
                if is_hunyuan:
                    logging.info("HunyuanVideo LoRA detected, converting to FramePack format")
                    lora_sd = convert_hunyuan_to_framepack(lora_sd)

            if lora_sd is not None:
                list_of_lora_sd.append(lora_sd)

    if len(list_of_lora_sd) == 0:
        # no LoRA files found, just load the model
        return load_safetensors_with_fp8_optimization(model_files, fp8_optimization, device, move_to_device, weight_hook=None)

    return load_safetensors_with_lora_and_fp8(model_files, list_of_lora_sd, multipliers, fp8_optimization, device, move_to_device)


def convert_from_diffusion_pipe_or_something(lora_sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """
    Convert LoRA weights to the format used by the diffusion pipeline to Musubi Tuner.
    Copy from Musubi Tuner repo.
    """
    # convert from diffusers(?) to default LoRA
    # Diffusers format: {"diffusion_model.module.name.lora_A.weight": weight, "diffusion_model.module.name.lora_B.weight": weight, ...}
    # default LoRA format: {"prefix_module_name.lora_down.weight": weight, "prefix_module_name.lora_up.weight": weight, ...}

    # note: Diffusers has no alpha, so alpha is set to rank
    new_weights_sd = {}
    lora_dims = {}
    for key, weight in lora_sd.items():
        diffusers_prefix, key_body = key.split(".", 1)
        if diffusers_prefix != "diffusion_model" and diffusers_prefix != "transformer":
            print(f"unexpected key: {key} in diffusers format")
            continue

        new_key = f"{prefix}{key_body}".replace(".", "_").replace("_lora_A_", ".lora_down.").replace("_lora_B_", ".lora_up.")
        new_weights_sd[new_key] = weight

        lora_name = new_key.split(".")[0]  # before first dot
        if lora_name not in lora_dims and "lora_down" in new_key:
            lora_dims[lora_name] = weight.shape[0]

    # add alpha with rank
    for lora_name, dim in lora_dims.items():
        new_weights_sd[f"{lora_name}.alpha"] = torch.tensor(dim)

    return new_weights_sd


def load_safetensors_with_lora_and_fp8(
    model_files: list[str],
    list_of_lora_sd: list[dict[str, torch.Tensor]],
    multipliers: Optional[list[float]],
    fp8_optimization: bool,
    device: torch.device,
    move_to_device: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model with fp8 optimization if needed.
    """
    if multipliers is None:
        multipliers = [1.0] * len(list_of_lora_sd)
    if len(multipliers) > len(list_of_lora_sd):
        multipliers = multipliers[: len(list_of_lora_sd)]
    if len(multipliers) < len(list_of_lora_sd):
        multipliers += [1.0] * (len(list_of_lora_sd) - len(multipliers))
    multipliers = [float(m) for m in multipliers]

    list_of_lora_weight_keys = []
    for lora_sd in list_of_lora_sd:
        lora_weight_keys = set(lora_sd.keys())
        list_of_lora_weight_keys.append(lora_weight_keys)

    # Merge LoRA weights into the state dict
    print(f"Merging LoRA weights into state dict on the fly. multipliers: {multipliers}")

    # make hook for LoRA merging
    def weight_hook(model_weight_key, model_weight):
        nonlocal list_of_lora_weight_keys, list_of_lora_sd, multipliers

        if not model_weight_key.endswith(".weight"):
            return model_weight

        original_device = model_weight.device
        if original_device != device:
            model_weight = model_weight.to(device)  # to make calculation faster

        for lora_weight_keys, lora_sd, multiplier in zip(list_of_lora_weight_keys, list_of_lora_sd, multipliers):
            # check if this weight has LoRA weights
            lora_name = model_weight_key.rsplit(".", 1)[0]  # remove trailing ".weight"
            lora_name = "lora_unet_" + lora_name.replace(".", "_")
            down_key = lora_name + ".lora_down.weight"
            up_key = lora_name + ".lora_up.weight"
            alpha_key = lora_name + ".alpha"
            if down_key not in lora_weight_keys or up_key not in lora_weight_keys:
                return model_weight

            # get LoRA weights
            down_weight = lora_sd[down_key]
            up_weight = lora_sd[up_key]

            dim = down_weight.size()[0]
            alpha = lora_sd.get(alpha_key, dim)
            scale = alpha / dim

            down_weight = down_weight.to(device)
            up_weight = up_weight.to(device)

            # W <- W + U * D
            if len(model_weight.size()) == 2:
                # linear
                if len(up_weight.size()) == 4:  # use linear projection mismatch
                    up_weight = up_weight.squeeze(3).squeeze(2)
                    down_weight = down_weight.squeeze(3).squeeze(2)
                model_weight = model_weight + multiplier * (up_weight @ down_weight) * scale
            elif down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                model_weight = (
                    model_weight
                    + multiplier
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * scale
                )
            else:
                # conv2d 3x3
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                # logger.info(conved.size(), weight.size(), module.stride, module.padding)
                model_weight = model_weight + multiplier * conved * scale

            # remove LoRA keys from set
            lora_weight_keys.remove(down_key)
            lora_weight_keys.remove(up_key)
            if alpha_key in lora_weight_keys:
                lora_weight_keys.remove(alpha_key)

        model_weight = model_weight.to(original_device)  # move back to original device
        return model_weight

    state_dict = load_safetensors_with_fp8_optimization(
        model_files, fp8_optimization, device, move_to_device, weight_hook=weight_hook
    )

    for lora_weight_keys in list_of_lora_weight_keys:
        if len(lora_weight_keys) > 0:
            # if there are still LoRA keys left, it means they are not used in the model
            # this is a warning, not an error
            logger.warning(f"Warning: {len(lora_weight_keys)} LoRA keys not used in the model: {lora_weight_keys}")

    return state_dict


def convert_hunyuan_to_framepack(lora_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert HunyuanVideo LoRA weights to FramePack format.
    """
    new_lora_sd = {}
    for key, weight in lora_sd.items():
        if "double_blocks" in key:
            key = key.replace("double_blocks", "transformer_blocks")
            key = key.replace("img_mod_linear", "norm1_linear")
            key = key.replace("img_attn_qkv", "attn_to_QKV")  # split later
            key = key.replace("img_attn_proj", "attn_to_out_0")
            key = key.replace("img_mlp_fc1", "ff_net_0_proj")
            key = key.replace("img_mlp_fc2", "ff_net_2")
            key = key.replace("txt_mod_linear", "norm1_context_linear")
            key = key.replace("txt_attn_qkv", "attn_add_QKV_proj")  # split later
            key = key.replace("txt_attn_proj", "attn_to_add_out")
            key = key.replace("txt_mlp_fc1", "ff_context_net_0_proj")
            key = key.replace("txt_mlp_fc2", "ff_context_net_2")
        elif "single_blocks" in key:
            key = key.replace("single_blocks", "single_transformer_blocks")
            key = key.replace("linear1", "attn_to_QKVM")  # split later
            key = key.replace("linear2", "proj_out")
            key = key.replace("modulation_linear", "norm_linear")
        else:
            print(f"Unsupported module name: {key}, only double_blocks and single_blocks are supported")
            continue

        if "QKVM" in key:
            # split QKVM into Q, K, V, M
            key_q = key.replace("QKVM", "q")
            key_k = key.replace("QKVM", "k")
            key_v = key.replace("QKVM", "v")
            key_m = key.replace("attn_to_QKVM", "proj_mlp")
            if "_down" in key or "alpha" in key:
                # copy QKVM weight or alpha to Q, K, V, M
                assert "alpha" in key or weight.size(1) == 3072, f"QKVM weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
                new_lora_sd[key_m] = weight
            elif "_up" in key:
                # split QKVM weight into Q, K, V, M
                assert weight.size(0) == 21504, f"QKVM weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 : 3072 * 3]
                new_lora_sd[key_m] = weight[3072 * 3 :]  # 21504 - 3072 * 3 = 12288
            else:
                print(f"Unsupported module name: {key}")
                continue
        elif "QKV" in key:
            # split QKV into Q, K, V
            key_q = key.replace("QKV", "q")
            key_k = key.replace("QKV", "k")
            key_v = key.replace("QKV", "v")
            if "_down" in key or "alpha" in key:
                # copy QKV weight or alpha to Q, K, V
                assert "alpha" in key or weight.size(1) == 3072, f"QKV weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
            elif "_up" in key:
                # split QKV weight into Q, K, V
                assert weight.size(0) == 3072 * 3, f"QKV weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 :]
            else:
                print(f"Unsupported module name: {key}")
                continue
        else:
            # no split needed
            new_lora_sd[key] = weight

    return new_lora_sd


def load_safetensors_with_fp8_optimization(
    model_files: list[str], fp8_optimization: bool, device: torch.device, move_to_device: bool, weight_hook: callable = None
) -> dict[str, torch.Tensor]:
    """
    Load state dict from safetensors files and merge LoRA weights into the state dict with fp8 optimization if needed.
    """
    if fp8_optimization:
        TARGET_KEYS = ["transformer_blocks", "single_transformer_blocks"]
        EXCLUDE_KEYS = ["norm"]  # Exclude norm layers (e.g., LayerNorm, RMSNorm) from FP8
        state_dict = optimize_state_dict_with_fp8_on_the_fly(
            model_files, device, TARGET_KEYS, EXCLUDE_KEYS, move_to_device=move_to_device, weight_hook=weight_hook
        )
    else:
        state_dict = {}
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file) as f:
                for key in tqdm(f.keys(), desc=f"Loading {model_file}", leave=False):
                    value = f.get_tensor(key)
                    if weight_hook is not None:
                        value = weight_hook(key, value)
                    if move_to_device:
                        value = value.to(device)
                    state_dict[key] = value

    return state_dict
