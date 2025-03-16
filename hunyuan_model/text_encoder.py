from dataclasses import dataclass
import json
import os
from typing import Optional, Tuple, Union
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    AutoTokenizer,
    AutoModel,
    CLIPConfig,
    LlamaForCausalLM,
    LlamaConfig,
    LlavaConfig,
    LlavaProcessor,
    CLIPImageProcessor,
)
from transformers.utils import ModelOutput
from transformers.models.llama import LlamaModel
from transformers.models.llava import LlavaForConditionalGeneration
from safetensors.torch import load_file
from accelerate import init_empty_weights

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CLIP_L_HUGGINGFACE_MODEL_ID = "openai/clip-vit-large-patch14"
LLAVA_HUGGINGFACE_MODEL_ID = "xtuner/llava-llama-3-8b-v1_1-transformers"

CLIP_CONFIG = {
    "_name_or_path": "clip-vit-large-patch14/",
    "architectures": ["CLIPModel"],
    "initializer_factor": 1.0,
    "logit_scale_init_value": 2.6592,
    "model_type": "clip",
    "projection_dim": 768,
    #   "text_config": {
    "_name_or_path": "",
    "add_cross_attention": False,
    "architectures": None,
    "attention_dropout": 0.0,
    "bad_words_ids": None,
    "bos_token_id": 0,
    "chunk_size_feed_forward": 0,
    "cross_attention_hidden_size": None,
    "decoder_start_token_id": None,
    "diversity_penalty": 0.0,
    "do_sample": False,
    "dropout": 0.0,
    "early_stopping": False,
    "encoder_no_repeat_ngram_size": 0,
    "eos_token_id": 2,
    "finetuning_task": None,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
    "initializer_factor": 1.0,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "is_decoder": False,
    "is_encoder_decoder": False,
    "label2id": {"LABEL_0": 0, "LABEL_1": 1},
    "layer_norm_eps": 1e-05,
    "length_penalty": 1.0,
    "max_length": 20,
    "max_position_embeddings": 77,
    "min_length": 0,
    "model_type": "clip_text_model",
    "no_repeat_ngram_size": 0,
    "num_attention_heads": 12,
    "num_beam_groups": 1,
    "num_beams": 1,
    "num_hidden_layers": 12,
    "num_return_sequences": 1,
    "output_attentions": False,
    "output_hidden_states": False,
    "output_scores": False,
    "pad_token_id": 1,
    "prefix": None,
    "problem_type": None,
    "projection_dim": 768,
    "pruned_heads": {},
    "remove_invalid_values": False,
    "repetition_penalty": 1.0,
    "return_dict": True,
    "return_dict_in_generate": False,
    "sep_token_id": None,
    "task_specific_params": None,
    "temperature": 1.0,
    "tie_encoder_decoder": False,
    "tie_word_embeddings": True,
    "tokenizer_class": None,
    "top_k": 50,
    "top_p": 1.0,
    "torch_dtype": None,
    "torchscript": False,
    "transformers_version": "4.16.0.dev0",
    "use_bfloat16": False,
    "vocab_size": 49408,
    #   },
    #   "text_config_dict": {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "projection_dim": 768,
    #   },
    #   "torch_dtype": "float32",
    #   "transformers_version": null
}

LLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "max_position_embeddings": 8192,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
    "transformers_version": "4.46.3",
    "use_cache": True,
    "vocab_size": 128320,
}
LLAVA_CONFIG_JSON = json.loads(
    """
{
  "architectures": [
    "LlavaForConditionalGeneration"
  ],
  "ignore_index": -100,
  "image_token_index": 128257,
  "model_type": "llava",
  "pad_token_id": 128258,
  "projector_hidden_act": "gelu",
  "text_config": {
    "architectures": [
      "LlamaForCausalLM"
    ],
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "intermediate_size": 14336,
    "max_position_embeddings": 8192,
    "model_type": "llama",
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "torch_dtype": "float16",
    "vocab_size": 128320
  },
  "torch_dtype": "float16",
  "transformers_version": "4.40.1",
  "vision_config": {
    "architectures": [
      "CLIPVisionModel"
    ],
    "dropout": 0.0,
    "hidden_size": 1024,
    "image_size": 336,
    "intermediate_size": 4096,
    "model_type": "clip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768,
    "torch_dtype": "float32"
  },
  "vision_feature_layer": -2,
  "vision_feature_select_strategy": "default"
}"""
)

LLAVA_PROCESSOR_CONFIG = json.loads(
    """{
  "image_token": "<image>",
  "num_additional_image_tokens": 1,
  "patch_size": 14,
  "processor_class": "LlavaNextProcessor",
  "vision_feature_select_strategy": "default"
}"""
)

# When using decoder-only models, we must provide a prompt template to instruct the text encoder
# on how to generate the text.
# --------------------------------------------------------------------
PROMPT_TEMPLATE_ENCODE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)
PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)

PROMPT_TEMPLATE_ENCODE_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

PROMPT_TEMPLATE_ENCODE_VIDEO_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

NEGATIVE_PROMPT = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
NEGATIVE_PROMPT_I2V = "deformation, a poor composition and deformed video, bad teeth, bad eyes, bad limbs"


PROMPT_TEMPLATE = {
    "dit-llm-encode": {
        "template": PROMPT_TEMPLATE_ENCODE,
        "crop_start": 36,
    },
    "dit-llm-encode-video": {
        "template": PROMPT_TEMPLATE_ENCODE_VIDEO,
        "crop_start": 95,
    },
    "dit-llm-encode-i2v": {
        "template": PROMPT_TEMPLATE_ENCODE_I2V,
        "crop_start": 36,
        "image_emb_start": 5,
        "image_emb_end": 581,
        "image_emb_len": 576,
        "double_return_token_id": 271,
    },
    "dit-llm-encode-video-i2v": {
        "template": PROMPT_TEMPLATE_ENCODE_VIDEO_I2V,
        "crop_start": 103,
        "image_emb_start": 5,
        "image_emb_end": 581,
        "image_emb_len": 576,
        "double_return_token_id": 271,
    },
}


def use_default(value, default):
    return value if value is not None else default


def load_clip_l(text_encoder_path: str, dtype: Optional[Union[str, torch.dtype]] = None):
    if os.path.isdir(text_encoder_path):
        # load from directory, configs are in the directory
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=dtype)
    else:
        # load from file, we create the model with the appropriate config
        config = CLIPConfig(**CLIP_CONFIG)
        with init_empty_weights():
            text_encoder = CLIPTextModel._from_config(config, torch_dtype=dtype)

        state_dict = load_file(text_encoder_path)

        text_encoder.load_state_dict(state_dict, strict=True, assign=True)
    # if dtype is not None:
    #     text_encoder.to(dtype=dtype)

    return text_encoder


def load_clip_l_tokenizer(tokenizer_path: str):
    if os.path.isdir(tokenizer_path):
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=77)
    else:
        # load from Hugging Face
        logger.info(f"Loading tokenizer from Hugging Face: {CLIP_L_HUGGINGFACE_MODEL_ID}")
        tokenizer = CLIPTokenizer.from_pretrained(CLIP_L_HUGGINGFACE_MODEL_ID, max_length=77)

    return tokenizer


def load_llm(text_encoder_path: str, dtype: Optional[Union[str, torch.dtype]] = None):
    if os.path.isdir(text_encoder_path):
        # load from directory, configs are in the directory
        text_encoder = AutoModel.from_pretrained(text_encoder_path, low_cpu_mem_usage=True, torch_dtype=dtype)
    else:
        # load from file, we create the model with the appropriate config
        config = LlamaConfig(**LLAMA_CONFIG)
        with init_empty_weights():
            text_encoder = LlamaForCausalLM._from_config(config, torch_dtype=dtype)

        state_dict = load_file(text_encoder_path)

        # support weights from ComfyUI
        if "tokenizer" in state_dict:
            state_dict.pop("tokenizer")

        text_encoder.load_state_dict(state_dict, strict=True, assign=True)

    return text_encoder


def load_llm_i2v(text_encoder_path: str, clip_vision_path: str, dtype: Optional[Union[str, torch.dtype]] = None):
    if os.path.isdir(text_encoder_path):
        # load from directory, configs are in the directory
        text_encoder = LlavaForConditionalGeneration.from_pretrained(text_encoder_path, low_cpu_mem_usage=True)
    else:
        # load from file, we create the model with the appropriate config
        config = LlavaConfig(**LLAVA_CONFIG_JSON)
        with init_empty_weights():
            text_encoder = LlavaForConditionalGeneration._from_config(config, torch_dtype=dtype)

        state_dict = load_file(text_encoder_path)

        # support weights from ComfyUI
        if "tokenizer" in state_dict:
            state_dict.pop("tokenizer")

        state_dict = {"language_model." + k: v for k, v in state_dict.items()}

        state_dict_vision = load_file(clip_vision_path)
        state_dict_vision = {
            ("vision_tower." if "multi_modal_projector." not in k else "") + k: v for k, v in state_dict_vision.items()
        }
        state_dict.update(state_dict_vision)

        text_encoder.load_state_dict(state_dict, strict=True, assign=True)

    return text_encoder


def load_llm_tokenizer(tokenizer_path: str, padding_side="right"):
    if os.path.isdir(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        # load from Hugging Face
        logger.info(f"Loading tokenizer from Hugging Face: {LLAVA_HUGGINGFACE_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(LLAVA_HUGGINGFACE_MODEL_ID, padding_side=padding_side)

    return tokenizer


def load_text_encoder(
    text_encoder_type: str,
    text_encoder_path: str,
    text_encoder_dtype: Optional[Union[str, torch.dtype]] = None,
    clip_vision_path: Optional[str] = None,
):
    logger.info(f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}")

    # reduce peak memory usage by specifying the dtype of the model
    dtype = text_encoder_dtype
    processor = None
    if text_encoder_type == "clipL":
        text_encoder = load_clip_l(text_encoder_path, dtype=dtype)
        text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm
    elif text_encoder_type == "llm":
        text_encoder = load_llm(text_encoder_path, dtype=dtype)
        if hasattr(text_encoder, "norm"):
            text_encoder.final_layer_norm = text_encoder.norm  # by from_pretrained
        else:
            text_encoder.final_layer_norm = text_encoder.model.norm  # by _from_config
    elif text_encoder_type == "llm-i2v":
        text_encoder = load_llm_i2v(text_encoder_path, clip_vision_path, dtype=dtype)
        text_encoder.final_layer_norm = text_encoder.language_model.model.norm
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
    # from_pretrained will ensure that the model is in eval mode.

    if dtype is not None:
        text_encoder = text_encoder.to(dtype=dtype)

    text_encoder.requires_grad_(False)

    logger.info(f"Text encoder to dtype: {text_encoder.dtype}")
    return text_encoder, processor, text_encoder_path


def load_tokenizer(tokenizer_type, tokenizer_path=None, padding_side="right"):
    logger.info(f"Loading tokenizer ({tokenizer_type}) from: {tokenizer_path}")

    if tokenizer_type == "clipL":
        tokenizer = load_clip_l_tokenizer(tokenizer_path)
    elif tokenizer_type == "llm" or tokenizer_type == "llm-i2v":
        tokenizer = load_llm_tokenizer(tokenizer_path, padding_side=padding_side)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    return tokenizer, tokenizer_path


@dataclass
class TextEncoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
        hidden_states_list (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        text_outputs (`list`, *optional*, returned when `return_texts=True` is passed):
            List of decoded texts.
    """

    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    hidden_states_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_outputs: Optional[list] = None


class TextEncoder(nn.Module):
    def __init__(
        self,
        text_encoder_type: str,
        max_length: int,
        text_encoder_dtype: Optional[Union[str, torch.dtype]] = None,
        text_encoder_path: Optional[str] = None,
        clip_vision_path: Optional[str] = None,
        tokenizer_type: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        i2v_mode: bool = False,
        output_key: Optional[str] = None,
        use_attention_mask: bool = True,
        input_max_length: Optional[int] = None,
        prompt_template: Optional[dict] = None,
        prompt_template_video: Optional[dict] = None,
        hidden_state_skip_layer: Optional[int] = None,
        apply_final_norm: bool = False,
        reproduce: bool = False,
        image_embed_interleave: int = None,
    ):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.max_length = max_length
        # self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.tokenizer_type = tokenizer_type if tokenizer_type is not None else text_encoder_type
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else text_encoder_path
        self.i2v_mode = i2v_mode
        self.use_attention_mask = use_attention_mask
        if prompt_template_video is not None:
            assert use_attention_mask is True, "Attention mask is True required when training videos."
        self.input_max_length = input_max_length if input_max_length is not None else max_length
        self.prompt_template = prompt_template
        self.prompt_template_video = prompt_template_video
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.reproduce = reproduce
        self.image_embed_interleave = image_embed_interleave

        self.use_template = self.prompt_template is not None
        if self.use_template:
            assert (
                isinstance(self.prompt_template, dict) and "template" in self.prompt_template
            ), f"`prompt_template` must be a dictionary with a key 'template', got {self.prompt_template}"
            assert "{}" in str(self.prompt_template["template"]), (
                "`prompt_template['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template['template']}"
            )

        self.use_video_template = self.prompt_template_video is not None
        if self.use_video_template:
            if self.prompt_template_video is not None:
                assert (
                    isinstance(self.prompt_template_video, dict) and "template" in self.prompt_template_video
                ), f"`prompt_template_video` must be a dictionary with a key 'template', got {self.prompt_template_video}"
            assert "{}" in str(self.prompt_template_video["template"]), (
                "`prompt_template_video['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template_video['template']}"
            )

        if "t5" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        elif "clip" in text_encoder_type:
            self.output_key = output_key or "pooler_output"
        elif "llm" in text_encoder_type or "glm" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        else:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

        self.model, self.processor, self.model_path = load_text_encoder(
            text_encoder_type=self.text_encoder_type,
            text_encoder_path=self.model_path,
            text_encoder_dtype=text_encoder_dtype,
            clip_vision_path=clip_vision_path,
        )
        self.dtype = self.model.dtype

        self.tokenizer, self.tokenizer_path = load_tokenizer(
            tokenizer_type=self.tokenizer_type, tokenizer_path=self.tokenizer_path, padding_side="right"
        )

        if text_encoder_type == "llm-i2v":
            clip_processor = CLIPImageProcessor.from_pretrained(LLAVA_HUGGINGFACE_MODEL_ID)
            self.processor = LlavaProcessor.from_args_and_dict(
                args=[clip_processor, self.tokenizer], processor_dict=LLAVA_PROCESSOR_CONFIG
            )
            # print(f"patch size: {self.processor.patch_size}, vision strategy: {self.processor.vision_feature_select_strategy}")
        else:
            self.processor = None

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    @property
    def device(self):
        return self.model.device

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        """
        Apply text to template.

        Args:
            text (str): Input text.
            template (str or list): Template string or list of chat conversation.
            prevent_empty_text (bool): If Ture, we will prevent the user text from being empty
                by adding a space. Defaults to True.
        """
        if isinstance(template, str):
            # Will send string to tokenizer. Used for llm
            return template.format(text)
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def text2tokens(self, text, data_type="image", semantic_images=None):
        """
        Tokenize the input text.

        Args:
            text (str or list): Input text.
        """
        tokenize_input_type = "str"
        if self.use_template:
            if data_type == "image":
                prompt_template = self.prompt_template["template"]
            elif data_type == "video":
                prompt_template = self.prompt_template_video["template"]
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            if isinstance(text, (list, tuple)):
                text = [self.apply_text_to_template(one_text, prompt_template) for one_text in text]
                if isinstance(text[0], list):
                    tokenize_input_type = "list"
            elif isinstance(text, str):
                text = self.apply_text_to_template(text, prompt_template)
                if isinstance(text, list):
                    tokenize_input_type = "list"
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")
        else:
            if isinstance(text, (list, tuple)):
                tokenize_input_type = "list"
            elif isinstance(text, str):
                tokenize_input_type = "str"
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")

        kwargs = dict(
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        if tokenize_input_type == "str":
            if self.text_encoder_type != "llm-i2v":
                return self.tokenizer(
                    text,
                    return_length=False,
                    return_overflowing_tokens=False,
                    return_attention_mask=True,
                    **kwargs,
                )
            else:
                # support transformers >= 4.47
                assert semantic_images is not None, "semantic_images is required for i2v mode tokenization."
                kwargs["max_length"] += 575  # image feature length-1
                return self.processor(
                    semantic_images,
                    text,
                    return_length=False,
                    return_overflowing_tokens=False,
                    return_attention_mask=True,
                    **kwargs,
                )

        elif tokenize_input_type == "list":
            if self.use_template:
                # this block is not tested yet
                return self.tokenizer(
                    text,
                    return_length=False,
                    return_overflowing_tokens=False,
                    return_attention_mask=True,
                    **kwargs,
                )
            else:
                return self.tokenizer.apply_chat_template(
                    text,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    **kwargs,
                )
        else:
            raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")

    def encode(
        self,
        batch_encoding,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=None,
        hidden_state_skip_layer=None,
        return_texts=False,
        data_type="image",
        semantic_images=None,
        device=None,
    ):
        """
        Args:
            batch_encoding (dict): Batch encoding from tokenizer.
            use_attention_mask (bool): Whether to use attention mask. If None, use self.use_attention_mask.
                Defaults to None.
            output_hidden_states (bool): Whether to output hidden states. If False, return the value of
                self.output_key. If True, return the entire output. If set self.hidden_state_skip_layer,
                output_hidden_states will be set True. Defaults to False.
            do_sample (bool): Whether to sample from the model. Used for Decoder-Only LLMs. Defaults to None.
                When self.produce is False, do_sample is set to True by default.
            hidden_state_skip_layer (int): Number of hidden states to hidden_state_skip_layer. 0 means the last layer.
                If None, self.output_key will be used. Defaults to None.
            return_texts (bool): Whether to return the decoded texts. Defaults to False.
        """
        device = self.model.device if device is None else device
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(hidden_state_skip_layer, self.hidden_state_skip_layer)
        do_sample = use_default(do_sample, not self.reproduce)

        if not self.i2v_mode:
            attention_mask = batch_encoding["attention_mask"].to(device) if use_attention_mask else None
            outputs = self.model(
                input_ids=batch_encoding["input_ids"].to(device),
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states or hidden_state_skip_layer is not None,
            )
            if hidden_state_skip_layer is not None:
                last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
                # Real last hidden state already has layer norm applied. So here we only apply it
                # for intermediate layers.
                if hidden_state_skip_layer > 0 and self.apply_final_norm:
                    last_hidden_state = self.model.final_layer_norm(last_hidden_state)
            else:
                last_hidden_state = outputs[self.output_key]

            # Remove hidden states of instruction tokens, only keep prompt tokens.
            if self.use_template:
                if data_type == "image":
                    crop_start = self.prompt_template.get("crop_start", -1)
                elif data_type == "video":
                    crop_start = self.prompt_template_video.get("crop_start", -1)
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")
                if crop_start > 0:
                    last_hidden_state = last_hidden_state[:, crop_start:]
                    attention_mask = attention_mask[:, crop_start:] if use_attention_mask else None

            if output_hidden_states:
                return TextEncoderModelOutput(last_hidden_state, attention_mask, outputs.hidden_states)
            return TextEncoderModelOutput(last_hidden_state, attention_mask)
        else:
            # I2V mode
            """
            # original code from HunyuanVideo
            image_outputs = self.processor(semantic_images, return_tensors="pt")["pixel_values"].to(device)
            attention_mask = batch_encoding["attention_mask"].to(device) if use_attention_mask else None
            outputs = self.model(
                input_ids=batch_encoding["input_ids"].to(device),
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states or hidden_state_skip_layer is not None,
                pixel_values=image_outputs,
            )

            if hidden_state_skip_layer is not None:
                last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
                # Real last hidden state already has layer norm applied. So here we only apply it
                # for intermediate layers.
                if hidden_state_skip_layer > 0 and self.apply_final_norm:
                    last_hidden_state = self.model.final_layer_norm(last_hidden_state)
            else:
                last_hidden_state = outputs[self.output_key]

            if self.use_template:
                if data_type == "video":
                    crop_start = self.prompt_template_video.get("crop_start", -1)
                    text_crop_start = crop_start - 1 + self.prompt_template_video.get("image_emb_len", 576)
                    image_crop_start = self.prompt_template_video.get("image_emb_start", 5)
                    image_crop_end = self.prompt_template_video.get("image_emb_end", 581)
                    batch_indices, last_double_return_token_indices = torch.where(
                        batch_encoding["input_ids"] == self.prompt_template_video.get("double_return_token_id", 271)
                    )

                    if last_double_return_token_indices.shape[0] == 3:
                        # in case the prompt is too long
                        last_double_return_token_indices = torch.cat(
                            (last_double_return_token_indices, torch.tensor([batch_encoding["input_ids"].shape[-1]]))
                        )
                        batch_indices = torch.cat((batch_indices, torch.tensor([0])))

                    last_double_return_token_indices = last_double_return_token_indices.reshape(
                        batch_encoding["input_ids"].shape[0], -1
                    )[:, -1]
                    batch_indices = batch_indices.reshape(batch_encoding["input_ids"].shape[0], -1)[:, -1]
                    assistant_crop_start = (
                        last_double_return_token_indices - 1 + self.prompt_template_video.get("image_emb_len", 576) - 4
                    )
                    assistant_crop_end = last_double_return_token_indices - 1 + self.prompt_template_video.get("image_emb_len", 576)
                    attention_mask_assistant_crop_start = last_double_return_token_indices - 4
                    attention_mask_assistant_crop_end = last_double_return_token_indices
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")
            """
            # modified code for i2v mode, support transformers >= 4.47
            assert use_attention_mask is True, "Attention mask is True required for backward compatibility."
            batch_encoding = batch_encoding.to(device)
            attention_mask = batch_encoding["attention_mask"]
            outputs = self.model(**batch_encoding, output_hidden_states=True)

            if hidden_state_skip_layer is not None:
                last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
                # Real last hidden state already has layer norm applied. So here we only apply it
                # for intermediate layers.
                if hidden_state_skip_layer > 0 and self.apply_final_norm:
                    last_hidden_state = self.model.final_layer_norm(last_hidden_state)
            else:
                last_hidden_state = outputs[self.output_key]

            if self.use_template:
                if data_type == "video":
                    crop_start = self.prompt_template_video.get("crop_start", -1)
                    text_crop_start = crop_start - 1 + self.prompt_template_video.get("image_emb_len", 576)
                    image_crop_start = self.prompt_template_video.get("image_emb_start", 5)
                    image_crop_end = self.prompt_template_video.get("image_emb_end", 581)
                    batch_indices, last_double_return_token_indices = torch.where(
                        batch_encoding["input_ids"] == self.prompt_template_video.get("double_return_token_id", 271)
                    )

                    if last_double_return_token_indices.shape[0] == 3:
                        # in case the prompt is too long
                        last_double_return_token_indices = torch.cat(
                            (last_double_return_token_indices, torch.tensor([batch_encoding["input_ids"].shape[-1]]))
                        )
                        batch_indices = torch.cat((batch_indices, torch.tensor([0])))

                    last_double_return_token_indices = last_double_return_token_indices.reshape(
                        batch_encoding["input_ids"].shape[0], -1
                    )[:, -1]
                    batch_indices = batch_indices.reshape(batch_encoding["input_ids"].shape[0], -1)[:, -1]

                    # with transformers >= 4.47, <image> token in input_ids is already expanded to image embed size.
                    # so we don't need to add image_emb_len to the last_double_return_token_indices.
                    assistant_crop_start = last_double_return_token_indices - 4
                    assistant_crop_end = last_double_return_token_indices
                    # attention mask is also expanded to image embed size, so the same as hidden state.
                    attention_mask_assistant_crop_start = last_double_return_token_indices - 4
                    attention_mask_assistant_crop_end = last_double_return_token_indices
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")

                text_last_hidden_state = []
                text_attention_mask = []
                image_last_hidden_state = []
                image_attention_mask = []
                for i in range(batch_encoding["input_ids"].shape[0]):
                    text_last_hidden_state.append(
                        torch.cat(
                            [
                                last_hidden_state[i, text_crop_start : assistant_crop_start[i].item()],
                                last_hidden_state[i, assistant_crop_end[i].item() :],
                            ]
                        )
                    )
                    text_attention_mask.append(
                        torch.cat(
                            [
                                attention_mask[
                                    i,
                                    text_crop_start : attention_mask_assistant_crop_start[i].item(),  # this line is modified
                                ],
                                attention_mask[i, attention_mask_assistant_crop_end[i].item() :],
                            ]
                        )
                        if use_attention_mask
                        else None
                    )
                    image_last_hidden_state.append(last_hidden_state[i, image_crop_start:image_crop_end])
                    image_attention_mask.append(
                        torch.ones(image_last_hidden_state[-1].shape[0]).to(last_hidden_state.device).to(attention_mask.dtype)
                        if use_attention_mask
                        else None
                    )

                text_last_hidden_state = torch.stack(text_last_hidden_state)
                text_attention_mask = torch.stack(text_attention_mask)
                image_last_hidden_state = torch.stack(image_last_hidden_state)
                image_attention_mask = torch.stack(image_attention_mask)

                if semantic_images is not None and 0 < self.image_embed_interleave < 6:
                    image_last_hidden_state = image_last_hidden_state[:, :: self.image_embed_interleave, :]
                    image_attention_mask = image_attention_mask[:, :: self.image_embed_interleave]

                assert (
                    text_last_hidden_state.shape[0] == text_attention_mask.shape[0]
                    and image_last_hidden_state.shape[0] == image_attention_mask.shape[0]
                )

                last_hidden_state = torch.cat([image_last_hidden_state, text_last_hidden_state], dim=1)
                attention_mask = torch.cat([image_attention_mask, text_attention_mask], dim=1)

            if output_hidden_states:
                return TextEncoderModelOutput(
                    last_hidden_state,
                    attention_mask,
                    hidden_states_list=outputs.hidden_states,
                )
            return TextEncoderModelOutput(last_hidden_state, attention_mask)

    def forward(
        self,
        text,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=False,
        hidden_state_skip_layer=None,
        return_texts=False,
    ):
        batch_encoding = self.text2tokens(text)
        return self.encode(
            batch_encoding,
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            do_sample=do_sample,
            hidden_state_skip_layer=hidden_state_skip_layer,
            return_texts=return_texts,
        )


# region HunyanVideo architecture


def load_text_encoder_1(
    text_encoder_dir: str,
    device: torch.device,
    fp8_llm: bool,
    dtype: Optional[Union[str, torch.dtype]] = None,
    i2v_mode: bool = False,
    image_embed_interleave: int = None,
    clip_vision_path: Optional[str] = None,
) -> TextEncoder:
    """
    clip_vision_path is required for i2v mode with .safetensors file.
    """
    text_encoder_dtype = dtype or torch.float16
    text_encoder_type = "llm" if not i2v_mode else "llm-i2v"
    text_len = 256
    hidden_state_skip_layer = 2
    apply_final_norm = False
    reproduce = False

    prompt_template = "dit-llm-encode" if not i2v_mode else "dit-llm-encode-i2v"
    prompt_template = PROMPT_TEMPLATE[prompt_template]
    prompt_template_video = "dit-llm-encode-video" if not i2v_mode else "dit-llm-encode-video-i2v"
    prompt_template_video = PROMPT_TEMPLATE[prompt_template_video]

    crop_start = prompt_template_video["crop_start"]  # .get("crop_start", 0)
    max_length = text_len + crop_start

    text_encoder_1 = TextEncoder(
        text_encoder_type=text_encoder_type,
        max_length=max_length,
        text_encoder_dtype=text_encoder_dtype,
        text_encoder_path=text_encoder_dir,
        clip_vision_path=clip_vision_path,
        tokenizer_type=text_encoder_type,
        i2v_mode=i2v_mode,
        prompt_template=prompt_template,
        prompt_template_video=prompt_template_video,
        hidden_state_skip_layer=hidden_state_skip_layer,
        apply_final_norm=apply_final_norm,
        reproduce=reproduce,
        image_embed_interleave=image_embed_interleave,
    )
    text_encoder_1.eval()

    if fp8_llm:
        org_dtype = text_encoder_1.dtype
        logger.info(f"Moving and casting text encoder to {device} and torch.float8_e4m3fn")
        text_encoder_1.to(device=device, dtype=torch.float8_e4m3fn)

        # prepare LLM for fp8
        def prepare_fp8(llama_model: LlamaModel, target_dtype):
            def forward_hook(module):
                def forward(hidden_states):
                    input_dtype = hidden_states.dtype
                    hidden_states = hidden_states.to(torch.float32)
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(variance + module.variance_epsilon)
                    return module.weight.to(input_dtype) * hidden_states.to(input_dtype)

                return forward

            for module in llama_model.modules():
                if module.__class__.__name__ in ["Embedding"]:
                    # print("set", module.__class__.__name__, "to", target_dtype)
                    module.to(target_dtype)
                if module.__class__.__name__ in ["LlamaRMSNorm"]:
                    # print("set", module.__class__.__name__, "hooks")
                    module.forward = forward_hook(module)

        prepare_fp8(text_encoder_1.model, org_dtype)
    else:
        text_encoder_1.to(device=device)

    return text_encoder_1


def load_text_encoder_2(
    text_encoder_dir: str, device: torch.device, dtype: Optional[Union[str, torch.dtype]] = None
) -> TextEncoder:
    text_encoder_dtype = dtype or torch.float16
    reproduce = False

    text_encoder_2_type = "clipL"
    text_len_2 = 77

    text_encoder_2 = TextEncoder(
        text_encoder_type=text_encoder_2_type,
        max_length=text_len_2,
        text_encoder_dtype=text_encoder_dtype,
        text_encoder_path=text_encoder_dir,
        tokenizer_type=text_encoder_2_type,
        reproduce=reproduce,
    )
    text_encoder_2.eval()

    text_encoder_2.to(device=device)

    return text_encoder_2


# endregion


if __name__ == "__main__":
    # Test the text encoder
    import argparse
    from utils.model_utils import str_to_dtype

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if False:
        # This is a test script to check if the text encoder is loaded correctly and the outputs are the same.
        # Compare two directories or files of text encoders: Offcial ckpt and single file ckpt.
        parser = argparse.ArgumentParser()
        parser.add_argument("type", type=str, help="Text Encoder type")
        parser.add_argument("path1", type=str, help="Text Encoder directory or file 1")
        parser.add_argument("path2", type=str, help="Text Encoder directory or file 2")
        parser.add_argument("--clip_vision_path1", type=str, default=None, help="Vision Encoder directory or file 1")
        parser.add_argument("--clip_vision_path2", type=str, default=None, help="Vision Encoder directory or file 2")
        parser.add_argument("--image_path", type=str, default=None, help="Image path, if set, use i2v mode")
        parser.add_argument("--image_embed_interleave", type=int, default=None, help="Image embed interleave")
        parser.add_argument("--dtype", type=str, default=None, help="Data type for Text Encoder")
        args = parser.parse_args()

        dtype = str_to_dtype(args.dtype) if args.dtype is not None else torch.float16

        i2v_mode = args.image_path is not None
        if i2v_mode:
            from PIL import Image

            image = Image.open(args.image_path).convert("RGB")
            semantic_images = [image]
        else:
            semantic_images = None

        if args.type == "clipL":
            text_encoder_1 = load_text_encoder_2(args.path1, device, dtype)
            text_encoder_2nd = load_text_encoder_2(args.path2, "cpu", dtype)
        elif args.type == "llm" or args.type == "llm-i2v":
            print("loading text encoder 1st")
            text_encoder_1 = load_text_encoder_1(
                args.path1, device, False, dtype, i2v_mode, args.image_embed_interleave, args.clip_vision_path1
            )
            print("loading text encoder 2nd")
            text_encoder_2nd = load_text_encoder_1(
                args.path2, "cpu", False, dtype, i2v_mode, args.image_embed_interleave, args.clip_vision_path2
            )
        print(f"1st Text Encoder dtype: {text_encoder_1.dtype}")
        print(f"2nd Text Encoder dtype: {text_encoder_2nd.dtype}")

        prompt = "A cat sitting on a table"
        data_type = "video"  # video only, image is not supported
        text_inputs_new = text_encoder_1.text2tokens(prompt, data_type=data_type)
        text_inputs_2nd = text_encoder_2nd.text2tokens(prompt, data_type=data_type)
        print(text_inputs_new)
        assert torch.allclose(text_inputs_new["input_ids"], text_inputs_2nd["input_ids"])

        with torch.no_grad():
            print("Encoding with 1st text encoder")
            prompt_outputs_new = text_encoder_1.encode(text_inputs_new, data_type=data_type, semantic_images=semantic_images)
        del text_encoder_1
        text_encoder_2nd.to(device=device)
        with torch.no_grad():
            prompt_outputs_2nd = text_encoder_2nd.encode(text_inputs_new, data_type=data_type, semantic_images=semantic_images)

        # prompt_outputs.hidden_state, prompt_outputs.attention_mask
        assert torch.allclose(prompt_outputs_new.hidden_state, prompt_outputs_2nd.hidden_state)
        print("Hidden states are the same.")
        assert torch.allclose(prompt_outputs_new.attention_mask, prompt_outputs_2nd.attention_mask)
        print("Attention masks are the same.")
        print("All outputs are the same.")

    if True:
        # Test Llava with image in old transformers and new transformers
        # works only transformers < 4.47 (supports new behavior and legacy behavior)
        parser = argparse.ArgumentParser()
        parser.add_argument("path1", type=str, help="Text Encoder directory or file 1")
        parser.add_argument("--clip_vision_path1", type=str, default=None, help="Vision Encoder directory or file 1")
        parser.add_argument("--image_path", type=str, default=None, help="Image path, if set, use i2v mode")
        parser.add_argument("--dtype", type=str, default=None, help="Data type for Text Encoder")
        args = parser.parse_args()

        dtype = str_to_dtype(args.dtype) if args.dtype is not None else torch.float16

        from PIL import Image

        image = Image.open(args.image_path).convert("RGB")
        semantic_images = [image]

        text_encoder_1 = load_text_encoder_1(args.path1, device, False, dtype, True, 4, args.clip_vision_path1)

        prompt = "A short animated video of a girl standing in a classroom. The girl is wearing a sailor uniform. The upper body of the girl is shown, and the girl is talking to the camera with a rich expression and using gestures. The girl has a short black bob hairstyle, and the inner color of her hair is blue. She has red eyes and wears silver-framed glasses. High quality animated video, studio quality."
        # prompt = (
        #     "A short animated video of a girl standing in a classroom. The girl is wearing a sailor uniform. The upper body of the girl is shown, and the girl is talking to the camera with a rich expression and using gestures. The girl has a short black bob hairstyle, and the inner color of her hair is blue. She has red eyes and wears silver-framed glasses. High quality animated video, studio quality. "
        #     "A short animated video of a girl standing in a classroom. The girl is wearing a sailor uniform. The upper body of the girl is shown, and the girl is talking to the camera with a rich expression and using gestures. The girl has a short black bob hairstyle, and the inner color of her hair is blue. She has red eyes and wears silver-framed glasses. High quality animated video, studio quality. "
        #     "A short animated video of a girl standing in a classroom. The girl is wearing a sailor uniform. The upper body of the girl is shown, and the girl is talking to the camera with a rich expression and using gestures. The girl has a short black bob hairstyle, and the inner color of her hair is blue. She has red eyes and wears silver-framed glasses. High quality animated video, studio quality. "
        #     "A short animated video of a girl standing in a classroom. The girl is wearing a sailor uniform. The upper body of the girl is shown, and the girl is talking to the camera with a rich expression and using gestures. The girl has a short black bob hairstyle, and the inner color of her hair is blue. She has red eyes and wears silver-framed glasses. High quality animated video, studio quality. "
        # )
        data_type = "video"  # video only, image is not supported

        ### Test the new behavior of text encoder

        print("Encoding with text encoder, new behavior")
        text_inputs_new = text_encoder_1.text2tokens(prompt, data_type=data_type, semantic_images=semantic_images).to(device)
        print(f"text_inputs_new keys: {text_inputs_new.keys()}")
        print(f"input_ids shape: {text_inputs_new['input_ids'].shape}")
        print(f"attention_mask shape: {text_inputs_new['attention_mask'].shape}")

        with torch.no_grad():
            prompt_outputs_new = text_encoder_1.model(**text_inputs_new, output_hidden_states=True)

        ### Test the old behavior of text encoder

        print("Encoding with text encoder, old behavior")
        text_encoder_1.text_encoder_type = "llm"  # force old behavior, call tokenizer instead of processor
        text_inputs_old = text_encoder_1.text2tokens(prompt, data_type=data_type).to(device)
        print(f"text_inputs_old keys: {text_inputs_old.keys()}")
        print(f"input_ids shape: {text_inputs_old['input_ids'].shape}")
        print(f"attention_mask shape: {text_inputs_old['attention_mask'].shape}")

        with torch.no_grad():
            # original code from HunyuanVideo
            clip_processor = CLIPImageProcessor.from_pretrained(LLAVA_HUGGINGFACE_MODEL_ID)

            image_outputs = clip_processor(semantic_images, return_tensors="pt")["pixel_values"].to(device)
            attention_mask = text_inputs_old["attention_mask"].to(device)  # if use_attention_mask else None
            prompt_outputs_old = text_encoder_1.model(
                input_ids=text_inputs_old["input_ids"].to(device),
                attention_mask=attention_mask,
                output_hidden_states=True,
                pixel_values=image_outputs,
            )

        ### calc crop position

        crop_start = text_encoder_1.prompt_template_video.get("crop_start", -1)
        text_crop_start = crop_start - 1 + text_encoder_1.prompt_template_video.get("image_emb_len", 576)
        image_crop_start = text_encoder_1.prompt_template_video.get("image_emb_start", 5)
        image_crop_end = text_encoder_1.prompt_template_video.get("image_emb_end", 581)
        print(f"crop_start: {crop_start}")
        print(f"text_crop_start: {text_crop_start}, image_crop_start: {image_crop_start}, image_crop_end: {image_crop_end}")

        # we test with a single prompt, so the batch_indices will be 0
        def get_batch_and_last_double_return_token_indices(batch_encoding):
            batch_indices, last_double_return_token_indices = torch.where(
                batch_encoding["input_ids"] == text_encoder_1.prompt_template_video.get("double_return_token_id", 271)
            )
            if last_double_return_token_indices.shape[0] == 3:
                # in case the prompt is too long
                last_double_return_token_indices = torch.cat(
                    (last_double_return_token_indices, torch.tensor([batch_encoding["input_ids"].shape[-1]], device=device))
                )
                batch_indices = torch.cat((batch_indices, torch.tensor([0], device=device)))
            return batch_indices, last_double_return_token_indices

        batch_indices_new, last_double_return_token_indices_new = get_batch_and_last_double_return_token_indices(text_inputs_new)
        batch_indices_old, last_double_return_token_indices_old = get_batch_and_last_double_return_token_indices(text_inputs_old)
        print(
            f"batch_indices_new: {batch_indices_new}, last_double_return_token_indices_new: {last_double_return_token_indices_new}"
        )
        print(
            f"batch_indices_old: {batch_indices_old}, last_double_return_token_indices_old: {last_double_return_token_indices_old}"
        )

        def calc_attn_crop_new(batch_encoding, batch_indices, last_double_return_token_indices):
            last_double_return_token_indices = last_double_return_token_indices.reshape(batch_encoding["input_ids"].shape[0], -1)[
                :, -1
            ]
            print(f"new last_double_return_token_indices: {last_double_return_token_indices}")
            batch_indices = batch_indices.reshape(batch_encoding["input_ids"].shape[0], -1)[:, -1]
            assistant_crop_start = last_double_return_token_indices - 4
            assistant_crop_end = last_double_return_token_indices
            attention_mask_assistant_crop_start = last_double_return_token_indices - 4
            attention_mask_assistant_crop_end = last_double_return_token_indices
            return assistant_crop_start, assistant_crop_end, attention_mask_assistant_crop_start, attention_mask_assistant_crop_end

        def calc_attn_crop(batch_encoding, batch_indices, last_double_return_token_indices):
            last_double_return_token_indices = last_double_return_token_indices.reshape(batch_encoding["input_ids"].shape[0], -1)[
                :, -1
            ]
            print(f"old last_double_return_token_indices: {last_double_return_token_indices}")
            batch_indices = batch_indices.reshape(batch_encoding["input_ids"].shape[0], -1)[:, -1]
            assistant_crop_start = (
                last_double_return_token_indices - 1 + text_encoder_1.prompt_template_video.get("image_emb_len", 576) - 4
            )
            assistant_crop_end = (
                last_double_return_token_indices - 1 + text_encoder_1.prompt_template_video.get("image_emb_len", 576)
            )
            attention_mask_assistant_crop_start = last_double_return_token_indices - 4
            attention_mask_assistant_crop_end = last_double_return_token_indices
            return assistant_crop_start, assistant_crop_end, attention_mask_assistant_crop_start, attention_mask_assistant_crop_end

        (
            assistant_crop_start_new,
            assistant_crop_end_new,
            attention_mask_assistant_crop_start_new,
            attention_mask_assistant_crop_end_new,
        ) = calc_attn_crop_new(text_inputs_new, batch_indices_new, last_double_return_token_indices_new)
        (
            assistant_crop_start_old,
            assistant_crop_end_old,
            attention_mask_assistant_crop_start_old,
            attention_mask_assistant_crop_end_old,
        ) = calc_attn_crop(text_inputs_old, batch_indices_old, last_double_return_token_indices_old)

        print("Assistant crop start and end:")
        print(
            "new",
            assistant_crop_start_new,
            assistant_crop_end_new,
            attention_mask_assistant_crop_start_new,
            attention_mask_assistant_crop_end_new,
        )
        print(
            "old",
            assistant_crop_start_old,
            assistant_crop_end_old,
            attention_mask_assistant_crop_start_old,
            attention_mask_assistant_crop_end_old,
        )

        ### Compare the outputs of the two models

        hidden_state_new = prompt_outputs_new.hidden_states[-(2 + 1)]
        hidden_state_old = prompt_outputs_old.hidden_states[-(2 + 1)]

        def crop_hidden_state_and_attn_mask(
            hidden_state,
            attention_mask,
            text_crop_start,
            crop_start,
            assistant_crop_start,
            assistant_crop_end,
            attention_mask_assistant_crop_start,
            attention_mask_assistant_crop_end,
        ):
            hidden_state = torch.cat([hidden_state[0, text_crop_start:assistant_crop_start], hidden_state[0, assistant_crop_end:]])
            print(f"cropping attention mask: {attention_mask.shape}, {crop_start}, {assistant_crop_start}, {assistant_crop_end}")
            attention_mask = torch.cat(
                [
                    attention_mask[0, crop_start:attention_mask_assistant_crop_start],
                    attention_mask[0, attention_mask_assistant_crop_end:],
                ]
            )
            return hidden_state, attention_mask

        with torch.no_grad():
            hidden_state_new = text_encoder_1.model.final_layer_norm(hidden_state_new)
            hidden_state_old = text_encoder_1.model.final_layer_norm(hidden_state_old)

        hidden_state_new, attention_mask_new = crop_hidden_state_and_attn_mask(
            hidden_state_new,
            text_inputs_new["attention_mask"],
            text_crop_start,
            text_crop_start,
            assistant_crop_start_new,
            assistant_crop_end_new,
            attention_mask_assistant_crop_start_new,
            attention_mask_assistant_crop_end_new,
        )
        hidden_state_old, attention_mask_old = crop_hidden_state_and_attn_mask(
            hidden_state_old,
            text_inputs_old["attention_mask"],
            text_crop_start,
            crop_start,
            assistant_crop_start_old,
            assistant_crop_end_old,
            attention_mask_assistant_crop_start_old,
            attention_mask_assistant_crop_end_old,
        )

        assert (
            hidden_state_new.shape == hidden_state_old.shape
        ), f"hidden state shape is not the same: {hidden_state_new.shape} vs {hidden_state_old.shape}"
        assert (
            hidden_state_new.dtype == hidden_state_old.dtype
        ), f"hidden state dtype is not the same: {hidden_state_new.dtype} vs {hidden_state_old.dtype}"
        print(f"hidden state shape: {hidden_state_new.shape}")

        diff = (hidden_state_new - hidden_state_old).abs()
        print(f"hidden state diff: {diff.max()}, {diff.mean()}, {diff.std()}")
        print(hidden_state_new[-20:, 0])
        print(hidden_state_old[-20:, 0])
        print(diff[-20:, 0])

        assert (
            attention_mask_new.shape == attention_mask_old.shape
        ), f"attention mask shape is not the same: {attention_mask_new.shape} vs {attention_mask_old.shape}"
        assert (
            attention_mask_new.dtype == attention_mask_old.dtype
        ), f"attention mask dtype is not the same: {attention_mask_new.dtype} vs {attention_mask_old.dtype}"
        print(f"attention mask shape: {attention_mask_new.shape}")
        assert torch.allclose(
            attention_mask_new, attention_mask_old
        ), f"attention mask is not the same. diff: {(attention_mask_new - attention_mask_old).abs().max()}"

        print(f"final attention mask: {attention_mask_new}")

        # assert torch.allclose(hidden_state_new, hidden_state_old), f"hidden state is not the same. diff: {diff}"

        import numpy as np

        diff = diff.float().cpu().numpy()  # (934, 4096)
        diff = diff.mean(axis=1)

        attn_mask_np = attention_mask_new.float().cpu().numpy()
        diff = diff * attn_mask_np

        assert diff.max() < 1e-3, f"hidden state diff is too large: {diff.max()}"

        # # show as bar plot
        # import matplotlib.pyplot as plt

        # plt.bar(range(diff.shape[0]), diff)
        # plt.title("Hidden state diff")
        # plt.xlabel("Hidden state index")
        # plt.ylabel("Diff")
        # plt.show()
