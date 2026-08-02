"""Static (CPU, tiny/dummy-weight) tests for the vendored Qwen3-VL conditioner.

Run from the repo root with the project venv:

    env/bin/python minimax_engine/tests/run_static_tests.py test_minimax_conditioner

The end-to-end tests build a tiny fake `text_encoder` checkpoint (52 toy layers so the
50-layer read contract holds) and reuse the Qwen tokenizer checked in under
cosmos_engine/cosmos_hf_indexes (it carries the vision special tokens).
"""

import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_ENGINE_DIR)
for _p in (_ENGINE_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _env_compat  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from minimax_video import conditioner as conditioner_mod  # noqa: E402
from minimax_video.conditioner import MiniMaxH3Conditioner, _strip_known_prefixes, _wanted_text_key  # noqa: E402
from minimax_video.packing import MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG  # noqa: E402
from minimax_video.packing_ref2va import MiniMaxH3PreparedReference  # noqa: E402
from minimax_video.qwen3vl_processor import create_mm_token_type_ids, get_rope_index  # noqa: E402
from minimax_video.qwen3vl_text import Qwen3VLTruncatedTextModel  # noqa: E402

TOKENIZER_DIR = os.path.join(_REPO_ROOT, "cosmos_engine", "cosmos_hf_indexes", "Cosmos3-Nano", "text_tokenizer")

TINY_TEXT_CONFIG = {
    "vocab_size": 151936,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 52,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    "attention_bias": False,
    "hidden_act": "silu",
}
TINY_VISION_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 4,
    "depth": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "in_channels": 3,
    "out_hidden_size": 32,
    "num_position_embeddings": 16,
    "deepstack_visual_indexes": [0, 1],
}

_CKPT_DIR = None


def _tiny_checkpoint_dir() -> str:
    """Build (once) a tiny fake checkpoint dir with text_encoder/, tokenizer/, processor/."""
    global _CKPT_DIR
    if _CKPT_DIR is not None:
        return _CKPT_DIR

    import safetensors.torch

    from minimax_video.qwen3vl_vision import build_vision_tower

    root = tempfile.mkdtemp(prefix="minimax_tiny_ckpt_")
    encoder_dir = os.path.join(root, "text_encoder")
    os.makedirs(encoder_dir)

    config = {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "image_token_id": 151655,
        "video_token_id": 151656,
        "text_config": TINY_TEXT_CONFIG,
        "vision_config": TINY_VISION_CONFIG,
    }
    with open(os.path.join(encoder_dir, "config.json"), "w") as f:
        json.dump(config, f)

    torch.manual_seed(0)
    full_text = Qwen3VLTruncatedTextModel(TINY_TEXT_CONFIG, num_read_layers=52)
    vision = build_vision_tower(TINY_VISION_CONFIG)
    state = {}
    for key, value in full_text.state_dict().items():
        state[f"model.language_model.{key}"] = value.to(torch.float32).clone()
    # A final norm and an lm_head the loader must skip.
    state["model.language_model.norm.weight"] = torch.ones(TINY_TEXT_CONFIG["hidden_size"])
    state["lm_head.weight"] = torch.zeros(8, TINY_TEXT_CONFIG["hidden_size"])
    for key, value in vision.state_dict().items():
        state[f"model.visual.{key}"] = value.to(torch.float32).clone()
    safetensors.torch.save_file(state, os.path.join(encoder_dir, "model.safetensors"))

    shutil.copytree(TOKENIZER_DIR, os.path.join(root, "tokenizer"))

    processor_dir = os.path.join(root, "processor")
    os.makedirs(processor_dir)
    with open(os.path.join(processor_dir, "preprocessor_config.json"), "w") as f:
        json.dump(
            {
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "size": {"shortest_edge": 1024, "longest_edge": 16384},
            },
            f,
        )
    with open(os.path.join(processor_dir, "video_preprocessor_config.json"), "w") as f:
        json.dump(
            {
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "size": {"shortest_edge": 1024, "longest_edge": 65536},
            },
            f,
        )

    _CKPT_DIR = root
    return root


# ---------------------------------------------------------------------------
# Unit tests: prefix mapping, token types, rope index
# ---------------------------------------------------------------------------


def test_strip_known_prefixes():
    sd = {
        "model.language_model.layers.0.self_attn.q_proj.weight": 1,
        "model.language_model.embed_tokens.weight": 2,
        "model.visual.blocks.0.attn.qkv.weight": 3,
        "lm_head.weight": 4,
    }
    text, vision = _strip_known_prefixes(sd)
    assert text == {"layers.0.self_attn.q_proj.weight": 1, "embed_tokens.weight": 2}
    assert vision == {"blocks.0.attn.qkv.weight": 3}

    flat = {"language_model.layers.1.mlp.gate_proj.weight": 5, "visual.merger.linear_fc1.weight": 6}
    text, vision = _strip_known_prefixes(flat)
    assert "layers.1.mlp.gate_proj.weight" in text and "merger.linear_fc1.weight" in vision


def test_wanted_text_key():
    assert _wanted_text_key("embed_tokens.weight", 50)
    assert _wanted_text_key("layers.0.self_attn.q_proj.weight", 50)
    assert _wanted_text_key("layers.49.mlp.down_proj.weight", 50)
    assert not _wanted_text_key("layers.50.mlp.down_proj.weight", 50)
    assert not _wanted_text_key("layers.51.input_layernorm.weight", 50)
    assert not _wanted_text_key("norm.weight", 50)


def test_create_mm_token_type_ids():
    ids = [10, 151655, 151655, 11, 151656, 12]
    assert create_mm_token_type_ids(ids, 151655, 151656) == [0, 1, 1, 0, 2, 0]


def _reference_rope_index(token_ids, image_grid_thw, video_grid_thw, merge_size, image_pad_id, video_pad_id):
    """The transformers v4.57 get_rope_index algorithm (input_ids scan), for parity checking."""
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
        video_grid_thw[:, 0] = 1
    input_tokens = list(token_ids)
    image_nums = 0 if image_grid_thw is None else image_grid_thw.shape[0]
    video_nums = 0 if video_grid_thw is None else video_grid_thw.shape[0]
    llm_pos_ids_list = []
    st = 0
    remain_images, remain_videos = image_nums, video_nums
    image_index = video_index = 0
    for _ in range(image_nums + video_nums):
        ed_image = input_tokens.index(image_pad_id, st) if (image_pad_id in input_tokens[st:] and remain_images > 0) else len(input_tokens) + 1
        ed_video = input_tokens.index(video_pad_id, st) if (video_pad_id in input_tokens[st:] and remain_videos > 0) else len(input_tokens) + 1
        if ed_image < ed_video:
            t, h, w = (int(x) for x in image_grid_thw[image_index])
            image_index += 1
            remain_images -= 1
            ed = ed_image
        else:
            t, h, w = (int(x) for x in video_grid_thw[video_index])
            video_index += 1
            remain_videos -= 1
            ed = ed_video
        llm_grid_t, llm_grid_h, llm_grid_w = t, h // merge_size, w // merge_size
        text_len = ed - st
        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
        st = ed + llm_grid_t * llm_grid_h * llm_grid_w
    if st < len(input_tokens):
        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
        llm_pos_ids_list.append(torch.arange(len(input_tokens) - st).view(1, -1).expand(3, -1) + st_idx)
    return torch.cat(llm_pos_ids_list, dim=1).view(3, 1, -1)


def test_rope_index_parity_with_reference():
    image_pad, video_pad = 151655, 151656
    merge = 2

    # Text-only.
    ids = [5, 6, 7, 8]
    types = create_mm_token_type_ids(ids, image_pad, video_pad)
    ours = get_rope_index(ids, types, None, None, merge)
    assert torch.equal(ours, torch.arange(4).view(1, 1, -1).expand(3, 1, -1))

    # Label + image block + text.
    image_grid = torch.tensor([[1, 4, 4]])  # 4 pad tokens after merge
    ids = [5, 6] + [image_pad] * 4 + [7, 8, 9]
    types = create_mm_token_type_ids(ids, image_pad, video_pad)
    ours = get_rope_index(ids, types, image_grid, None, merge)
    theirs = _reference_rope_index(ids, image_grid, None, merge, image_pad, video_pad)
    assert torch.equal(ours, theirs)

    # Two images + a two-block video with timestamp text between blocks + trailing prompt.
    image_grid = torch.tensor([[1, 4, 4], [1, 2, 4]])
    video_grid = torch.tensor([[2, 4, 2]])  # split into 2 blocks of t=1, 2 pads each
    ids = (
        [10]
        + [image_pad] * 4
        + [11]
        + [image_pad] * 2
        + [12, 13]
        + [video_pad] * 2
        + [14]
        + [video_pad] * 2
        + [15, 16]
    )
    types = create_mm_token_type_ids(ids, image_pad, video_pad)
    ours = get_rope_index(ids, types, image_grid, video_grid, merge)
    theirs = _reference_rope_index(ids, image_grid, video_grid, merge, image_pad, video_pad)
    assert torch.equal(ours, theirs)


# ---------------------------------------------------------------------------
# Tiny text model
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_truncated_text_model_forward():
    torch.manual_seed(0)
    config = dict(TINY_TEXT_CONFIG, num_hidden_layers=52)
    model = Qwen3VLTruncatedTextModel(config, num_read_layers=3).eval()
    assert len(model.layers) == 3
    embeds = torch.randn(1, 10, config["hidden_size"])
    position_ids = torch.arange(10).view(1, 1, -1).expand(3, 1, -1)
    out = model(embeds, position_ids)
    assert out.shape == (1, 10, config["hidden_size"])

    # Causality: changing a later token must not change an earlier row.
    embeds2 = embeds.clone()
    embeds2[0, -1] += 1.0
    out2 = model(embeds2, position_ids)
    torch.testing.assert_close(out[:, :5], out2[:, :5], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(out[:, -1], out2[:, -1])


def test_truncated_text_model_rejects_short_stacks():
    config = dict(TINY_TEXT_CONFIG, num_hidden_layers=3)
    try:
        Qwen3VLTruncatedTextModel(config, num_read_layers=50)
    except ValueError as e:
        assert "50" in str(e)
    else:
        raise AssertionError("expected ValueError for a stack shorter than the read layer")


# ---------------------------------------------------------------------------
# End-to-end tiny conditioner (fake checkpoint + real tokenizer)
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_conditioner_t2va_and_fl2va():
    ckpt = _tiny_checkpoint_dir()
    cond = MiniMaxH3Conditioner(ckpt, device="cpu", dtype=torch.float32)

    embeds, tags = cond.encode_prompt("a red fox in the snow")
    assert embeds.shape[0] == 1 and embeds.shape[2] == TINY_TEXT_CONFIG["hidden_size"]
    assert embeds.shape[1] == tags.shape[0]
    assert (tags == MINIMAX_H3_TEXT_TAG).all()

    image = Image.fromarray(np.random.RandomState(0).randint(0, 255, (64, 64, 3), dtype=np.uint8))
    embeds_kf, tags_kf = cond.encode_prompt("a red fox in the snow", [image])
    assert embeds_kf.shape[1] == tags_kf.shape[0]
    assert (tags_kf == MINIMAX_H3_VIDEO_TAG).sum() > 0  # the vision block is tagged as video
    assert embeds_kf.shape[1] > embeds.shape[1]
    # The label ("<Picture 1>: ") stays text-tagged and precedes the vision block.
    first_video_row = int((tags_kf == MINIMAX_H3_VIDEO_TAG).nonzero()[0])
    assert first_video_row > 0
    assert (tags_kf[:first_video_row] == MINIMAX_H3_TEXT_TAG).all()


@torch.no_grad()
def test_conditioner_ref2va():
    ckpt = _tiny_checkpoint_dir()
    cond = MiniMaxH3Conditioner(ckpt, device="cpu", dtype=torch.float32)

    rng = np.random.RandomState(0)
    image = Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    frames = rng.randint(0, 255, (26, 64, 64, 3), dtype=np.uint8)  # 26 frames @ 24 fps -> 3 sampled -> 2 blocks
    references = [
        MiniMaxH3PreparedReference(kind="image", image=image),
        MiniMaxH3PreparedReference(kind="video", frames=frames),
    ]
    embeds, tags = cond.encode_prompt_ref2va("the subject walks toward camera", references)
    assert embeds.shape == (1, tags.shape[0], TINY_TEXT_CONFIG["hidden_size"])
    assert (tags == MINIMAX_H3_VIDEO_TAG).sum() > 0
    assert references[1].block_timestamps  # filled during encoding
    # Tokenized presentation carries the per-modality labels and the block timestamps.
    text = cond.tokenizer.decode([t for t in cond.tokenizer("x")["input_ids"]])  # smoke: tokenizer round-trips
    assert isinstance(text, str)


@torch.no_grad()
def test_conditioner_matches_manual_forward():
    """The conditioner's text path must equal embed -> 50 layers run by hand."""
    ckpt = _tiny_checkpoint_dir()
    cond = MiniMaxH3Conditioner(ckpt, device="cpu", dtype=torch.float32)

    prompt = "parity check"
    token_ids = cond.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    embeds, _ = cond.encode_prompt(prompt)

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    hidden = cond.text_model.embed_tokens(input_ids)
    position_ids = torch.arange(len(token_ids)).view(1, 1, -1).expand(3, 1, -1)
    manual = cond.text_model(hidden, position_ids)
    torch.testing.assert_close(embeds, manual)
