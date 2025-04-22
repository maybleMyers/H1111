from pathlib import Path
from typing import Optional
from diffusers.loaders.lora_pipeline import _fetch_state_dict
from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers

def load_lora(transformer, lora_path: Path, weight_name: Optional[str] = "pytorch_lora_weights.safetensors", diffuser_lora: bool = False):
    """
    Load LoRA weights into the transformer model.

    Args:
        transformer: The transformer model to which LoRA weights will be applied.
        lora_path (Path): Path to the LoRA weights file.
        weight_name (Optional[str]): Name of the weight to load.

    """
    
    state_dict = _fetch_state_dict(
    lora_path,
    weight_name,
    True,
    True,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None)


    if not diffuser_lora:
        print("Not a diffusers lora, assuming Hunyuan.")
        state_dict = _convert_hunyuan_video_lora_to_diffusers(state_dict)
    
    transformer.load_lora_adapter(state_dict, network_alphas=None)
    print("LoRA weights loaded successfully.")
    return transformer
    