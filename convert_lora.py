import argparse

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from musubi_tuner.utils import model_utils  # Using the updated import path

import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def convert_from_diffusers(prefix, weights_sd):
    # convert from diffusers(?) to default LoRA
    # Diffusers format: {"diffusion_model.module.name.lora_A.weight": weight, "diffusion_model.module.name.lora_B.weight": weight, ...}
    # default LoRA format: {"prefix_module_name.lora_down.weight": weight, "prefix_module_name.lora_up.weight": weight, ...}

    new_weights_sd = {}
    lora_dims = {}
    unconverted_keys = []

    # Check if the input file is empty
    if not weights_sd:
        logger.warning("Input file is empty. Nothing to convert.")
        return {}

    for key, weight in weights_sd.items():
        try:
            diffusers_prefix, key_body = key.split(".", 1)
        except ValueError:
            # This happens if the key does not contain a '.', which is not a diffusers format.
            unconverted_keys.append(key)
            continue

        if diffusers_prefix not in ["diffusion_model", "transformer"]:
            unconverted_keys.append(key)
            continue

        # If we reach here, the key is in the expected diffusers format.
        new_key = f"{prefix}{key_body}".replace(".", "_").replace("_lora_A_", ".lora_down.").replace("_lora_B_", ".lora_up.")
        new_weights_sd[new_key] = weight

        lora_name = new_key.split(".")[0]  # before first dot
        if lora_name not in lora_dims and "lora_down" in new_key:
            lora_dims[lora_name] = weight.shape[0]

    # After checking all keys, decide what to do.
    if not new_weights_sd:
        # If new_weights_sd is empty, it means no keys were converted.
        # This implies the file was already in the target 'default' format.
        logger.info("Input file appears to be already in the 'default' format. Copying weights directly.")
        return weights_sd

    # If some keys were converted but others were not, it's a mixed/malformed file.
    if unconverted_keys:
        logger.warning("Some keys were not in the expected Diffusers format and were skipped:")
        for key in unconverted_keys:
            logger.warning(f"  - Skipped key: {key}")

    # Add alpha with rank, as diffusers format doesn't have it.
    for lora_name, dim in lora_dims.items():
        new_weights_sd[f"{lora_name}.alpha"] = torch.tensor(dim)

    return new_weights_sd


def convert_to_diffusers(prefix, weights_sd):
    # convert from default LoRA to diffusers

    # get alphas
    lora_alphas = {}
    for key, weight in weights_sd.items():
        if key.startswith(prefix):
            lora_name = key.split(".", 1)[0]  # before first dot
            if lora_name not in lora_alphas and "alpha" in key:
                lora_alphas[lora_name] = weight

    new_weights_sd = {}
    for key, weight in weights_sd.items():
        if key.startswith(prefix):
            if "alpha" in key:
                continue

            lora_name = key.split(".", 1)[0]  # before first dot

            module_name = lora_name[len(prefix) :]  # remove "lora_unet_"
            module_name = module_name.replace("_", ".")  # replace "_" with "."
            if ".cross.attn." in module_name or ".self.attn." in module_name:
                # Wan2.1 lora name to module name: ugly but works
                module_name = module_name.replace("cross.attn", "cross_attn")  # fix cross attn
                module_name = module_name.replace("self.attn", "self_attn")  # fix self attn
                module_name = module_name.replace("k.img", "k_img")  # fix k img (from new file)
                module_name = module_name.replace("v.img", "v_img")  # fix v img (from new file)
            else:
                # HunyuanVideo lora name to module name: ugly but works
                module_name = module_name.replace("double.blocks.", "double_blocks.")  # fix double blocks
                module_name = module_name.replace("single.blocks.", "single_blocks.")  # fix single blocks
                module_name = module_name.replace("img.", "img_")  # fix img
                module_name = module_name.replace("txt.", "txt_")  # fix txt
                module_name = module_name.replace("attn.", "attn_")  # fix attn

            diffusers_prefix = "diffusion_model"
            if "lora_down" in key:
                new_key = f"{diffusers_prefix}.{module_name}.lora_A.weight"
                dim = weight.shape[0]
            elif "lora_up" in key:
                new_key = f"{diffusers_prefix}.{module_name}.lora_B.weight"
                dim = weight.shape[1]
            else:
                logger.warning(f"unexpected key: {key} in default LoRA format")
                continue

            # scale weight by alpha
            if lora_name in lora_alphas:
                # we scale both down and up, so scale is sqrt
                scale = lora_alphas[lora_name] / dim
                scale = scale.sqrt()
                weight = weight * scale
            else:
                logger.warning(f"missing alpha for {lora_name}")

            new_weights_sd[new_key] = weight

    return new_weights_sd


def convert(input_file, output_file, target_format):
    logger.info(f"loading {input_file}")
    weights_sd = load_file(input_file)
    with safe_open(input_file, framework="pt") as f:
        metadata = f.metadata()

    logger.info(f"converting to {target_format}")
    prefix = "lora_unet_"
    if target_format == "default":
        new_weights_sd = convert_from_diffusers(prefix, weights_sd)
        metadata = metadata or {}
        # Only recalculate hashes if a conversion actually happened.
        # If we just copied the weights, the hashes are already valid.
        if new_weights_sd is not weights_sd:
             model_utils.precalculate_safetensors_hashes(new_weights_sd, metadata)
    elif target_format == "other":
        new_weights_sd = convert_to_diffusers(prefix, weights_sd)
    else:
        raise ValueError(f"unknown target format: {target_format}")

    if not new_weights_sd:
        logger.error("Conversion failed, no weights to save. Output file will not be created.")
        return

    logger.info(f"saving to {output_file}")
    save_file(new_weights_sd, output_file, metadata=metadata)

    logger.info("done")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LoRA weights between default and other formats")
    parser.add_argument("--input", type=str, required=True, help="input model file")
    parser.add_argument("--output", type=str, required=True, help="output model file")
    parser.add_argument("--target", type=str, required=True, choices=["other", "default"], help="target format")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    convert(args.input, args.output, args.target)


if __name__ == "__main__":
    main()