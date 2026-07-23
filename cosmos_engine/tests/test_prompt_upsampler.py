"""CPU tests for the native Cosmos3 prompt upsampler (no GPU, no real weights).

Covers the V4.2 template builder, response cleaner, mrope position ids,
image preprocessing, the vendored vision tower, and — with a tiny
random-weight transformer plus the real tokenizer — the KV-cached decode
loop, including a prefill-vs-incremental logits parity check.

Run from the repo root or cosmos_engine/:

    python3 cosmos_engine/tests/test_prompt_upsampler.py
"""

import os
import sys
import unittest

import numpy as np
import torch
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.dirname(_here)
for _p in (os.path.dirname(_ENGINE), _ENGINE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cosmos_video import upsampler_templates as templates
from cosmos_video.reasoner import (
    Cosmos3Reasoner,
    _KVCache,
    place_und_layers,
    resolve_task,
    strip_generation_weights,
)
from cosmos_video.vision_encoder import Qwen3VLVisionModel, preprocess_image, smart_resize

# The transformer pulls in diffusers; skip the decode-loop tests (not the
# template/rope/vision ones) on machines whose diffusers install is broken.
try:
    from cosmos_video.transformer import Cosmos3OmniTransformer

    _TRANSFORMER_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    Cosmos3OmniTransformer = None
    _TRANSFORMER_IMPORT_ERROR = str(e)

torch.manual_seed(0)

TOKENIZER_DIR = os.path.join(_ENGINE, "Cosmos3-Super-Image2Video-skeleton", "text_tokenizer")

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        if getattr(_tokenizer, "chat_template", None) is None:
            with open(os.path.join(TOKENIZER_DIR, "chat_template.jinja"), "r", encoding="utf-8") as f:
                _tokenizer.chat_template = f.read()
    return _tokenizer


def tiny_transformer(vocab_size: int) -> Cosmos3OmniTransformer:
    return Cosmos3OmniTransformer(
        hidden_size=64,
        intermediate_size=128,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        latent_channel=48,
        latent_patch_size=2,
        patch_latent_dim=192,
        vocab_size=vocab_size,
        rope_scaling={"mrope_interleaved": True, "mrope_section": [4, 2, 2], "rope_type": "default"},
    ).float().eval()


def tiny_vision_encoder() -> Qwen3VLVisionModel:
    return Qwen3VLVisionModel(
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        depth=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,  # must match the tiny LM hidden size
        num_position_embeddings=2304,
        deepstack_visual_indexes=[0, 1],
    ).float().eval()


TINY_PREPROC = {
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": (0.5, 0.5, 0.5),
    "image_std": (0.5, 0.5, 0.5),
    "min_pixels": 1024,
    "max_pixels": 16384,
}


def make_reasoner(with_vision=False):
    tokenizer = get_tokenizer()
    transformer = tiny_transformer(len(tokenizer))
    vision = tiny_vision_encoder() if with_vision else None
    return Cosmos3Reasoner(transformer, tokenizer, vision_encoder=vision, preprocessor_config=TINY_PREPROC)


class TestTemplates(unittest.TestCase):
    def test_substitution_all_tasks(self):
        for task in ("t2v", "i2v"):
            txt = templates.build_user_text(
                task, "a red cube spins", aspect_ratio="16,9", resolution_w=832, resolution_h=480,
                fps=24, duration_secs=7,
            )
            self.assertIn("a red cube spins", txt)
            self.assertIn("832", txt)
            self.assertIn("0:07", txt)
            self.assertNotIn("{description}", txt)
            self.assertNotIn("{aspect_ratio}", txt)
        txt = templates.build_user_text("t2i", "a red cube", aspect_ratio="1,1", resolution_w=960, resolution_h=960)
        self.assertIn("a red cube", txt)

    def test_video_tasks_require_fps_duration(self):
        with self.assertRaises(ValueError):
            templates.build_user_text("t2v", "x", aspect_ratio="16,9", resolution_w=832, resolution_h=480)
        with self.assertRaises(KeyError):
            templates.build_user_text("transfer", "x", aspect_ratio="16,9", resolution_w=832, resolution_h=480)

    def test_messages_shape(self):
        m = templates.build_messages(
            "i2v", "x", aspect_ratio="16,9", resolution_w=832, resolution_h=480, fps=24, duration_secs=7,
            with_image=True,
        )
        self.assertEqual(m[0]["role"], "system")
        self.assertEqual(m[1]["content"][0], {"type": "image"})
        self.assertEqual(m[1]["content"][1]["type"], "text")
        m = templates.build_messages(
            "t2v", "x", aspect_ratio="16,9", resolution_w=832, resolution_h=480, fps=24, duration_secs=7,
        )
        self.assertIsInstance(m[1]["content"], str)


class TestCleanResponse(unittest.TestCase):
    def test_clean_passthrough_and_strip(self):
        clean_in = '```json\n{"a": 1}\n```'
        out, info = templates.clean_response(clean_in)
        self.assertEqual(out, clean_in)
        self.assertTrue(info["was_clean"])
        out, info = templates.clean_response('<think>hmm</think>preamble\n```json\n{"a": 1}\n```')
        self.assertTrue(out.startswith("```json"))
        self.assertFalse(info["was_clean"])

    def test_is_upsampled_prompt(self):
        self.assertTrue(templates.is_upsampled_prompt('{"temporal_caption": "x"}'))
        self.assertTrue(templates.is_upsampled_prompt('```json\n{"a": 1}\n```'))
        self.assertFalse(templates.is_upsampled_prompt("a cat cooking pizza"))
        self.assertFalse(templates.is_upsampled_prompt("{not json"))
        self.assertFalse(templates.is_upsampled_prompt(None))


class TestResolveTask(unittest.TestCase):
    def test_auto(self):
        self.assertEqual(resolve_task("auto", has_image=True, video_length=189), "i2v")
        self.assertEqual(resolve_task("auto", has_image=False, video_length=189), "t2v")
        self.assertEqual(resolve_task("auto", has_image=False, video_length=1), "t2i")
        self.assertEqual(resolve_task("t2v", has_image=True, video_length=189), "t2v")
        with self.assertRaises(ValueError):
            resolve_task("bogus", has_image=False, video_length=189)


def make_tokenizer_only_reasoner():
    """Reasoner with no transformer — enough for tokenization/position-id tests."""
    return Cosmos3Reasoner(None, get_tokenizer(), preprocessor_config=TINY_PREPROC)


class TestPositionIds(unittest.TestCase):
    def test_text_only_sequential(self):
        r = make_tokenizer_only_reasoner()
        ids = torch.tensor([5, 6, 7, 8])
        pos = r._build_position_ids(ids, None)
        self.assertEqual(pos.shape, (3, 4))
        self.assertTrue(torch.equal(pos[0], torch.arange(4)))
        self.assertTrue(torch.equal(pos[0], pos[1]))

    def test_image_grid_positions(self):
        r = make_reasoner()
        img_id = r.image_token_id
        # [3 text tokens][4 image tokens (grid 1x4x4, merge 2)][2 text tokens]
        ids = torch.tensor([10, 11, 12] + [img_id] * 4 + [20, 21])
        grid = torch.tensor([[1, 4, 4]])
        pos = r._build_position_ids(ids, grid)
        self.assertEqual(pos.shape, (3, 9))
        # text prefix: 0..2 on all axes
        self.assertTrue(torch.equal(pos[:, :3], torch.arange(3).view(1, -1).expand(3, -1)))
        # image block starts at 3: t=3 flat, h=[3,3,4,4], w=[3,4,3,4]
        self.assertTrue(torch.equal(pos[0, 3:7], torch.tensor([3, 3, 3, 3])))
        self.assertTrue(torch.equal(pos[1, 3:7], torch.tensor([3, 3, 4, 4])))
        self.assertTrue(torch.equal(pos[2, 3:7], torch.tensor([3, 4, 3, 4])))
        # trailing text resumes at max+1 = 5
        self.assertTrue(torch.equal(pos[:, 7:], torch.tensor([[5, 6]] * 3)))


class TestImagePreprocessing(unittest.TestCase):
    def test_smart_resize_multiples(self):
        h, w = smart_resize(100, 60, factor=32, min_pixels=1024, max_pixels=16384)
        self.assertEqual(h % 32, 0)
        self.assertEqual(w % 32, 0)
        self.assertLessEqual(h * w, 16384)

    def test_preprocess_shapes(self):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        pixel_values, grid = preprocess_image(img, **TINY_PREPROC)
        t, gh, gw = grid[0].tolist()
        self.assertEqual(t, 1)
        self.assertEqual(pixel_values.shape, (t * gh * gw, 3 * 2 * 16 * 16))


class TestVisionEncoder(unittest.TestCase):
    def test_forward_shapes(self):
        model = tiny_vision_encoder()
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        pixel_values, grid = preprocess_image(img, **TINY_PREPROC)
        embeds, deepstack = model(pixel_values.float(), grid)
        n_merged = int(grid.prod()) // 4
        self.assertEqual(embeds.shape, (n_merged, 64))
        self.assertEqual(len(deepstack), 2)
        self.assertEqual(deepstack[0].shape, (n_merged, 64))


@unittest.skipIf(Cosmos3OmniTransformer is None, f"transformer import failed: {_TRANSFORMER_IMPORT_ERROR}")
class TestDecodeLoop(unittest.TestCase):
    def test_kv_cache_parity(self):
        """Incremental decode over the cache must match a full prefill recompute."""
        r = make_reasoner()
        ids = torch.tensor(r.tokenizer("The robot picks up the cube")["input_ids"], dtype=torch.long)
        tfm = r.transformer
        dtype = tfm.embed_tokens.weight.dtype

        def prefill(seq_ids):
            cache = _KVCache(len(tfm.layers))
            pos = r._build_position_ids(seq_ids, None)
            cos, sin = r._rotary(pos, torch.device("cpu"), dtype)
            hidden = tfm.embed_tokens(seq_ids)
            hidden = r._forward_pass(hidden, cos, sin, cache, is_prefill=True)
            return cache, r._logits(hidden[-1:])

        with torch.no_grad():
            _, logits_full = prefill(ids)

            cache, _ = prefill(ids[:-1])
            pos = torch.full((3, 1), int(ids.shape[0]) - 1, dtype=torch.long)
            cos, sin = r._rotary(pos, torch.device("cpu"), dtype)
            hidden = tfm.embed_tokens(ids[-1:])
            hidden = r._forward_pass(hidden, cos, sin, cache, is_prefill=False)
            logits_step = r._logits(hidden)

        self.assertTrue(
            torch.allclose(logits_full, logits_step, atol=1e-4),
            f"max diff {(logits_full - logits_step).abs().max().item()}",
        )

    def test_generate_deterministic_greedy(self):
        r = make_reasoner()
        messages = templates.build_messages(
            "t2v", "a cat", aspect_ratio="16,9", resolution_w=832, resolution_h=480, fps=24, duration_secs=7,
        )
        with torch.no_grad():
            out1 = r.generate(messages, max_new_tokens=4)
            out2 = r.generate(messages, max_new_tokens=4)
        self.assertIsInstance(out1, str)
        self.assertEqual(out1, out2)

    def test_generate_i2v_with_vision(self):
        r = make_reasoner(with_vision=True)
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        with torch.no_grad():
            out = r.upsample(
                "the cube melts",
                task="i2v",
                image=img,
                resolution_w=832,
                resolution_h=480,
                aspect_ratio="16:9",
                fps=24,
                duration_secs=7,
                max_new_tokens=4,
            )
        self.assertIsInstance(out, str)
        self.assertTrue(out)  # empty-output fallback returns the original prompt

    def test_i2v_without_vision_encoder_raises(self):
        r = make_reasoner(with_vision=False)
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            r.upsample(
                "x", task="i2v", image=img, resolution_w=832, resolution_h=480,
                aspect_ratio="16:9", fps=24, duration_secs=7, max_new_tokens=2,
            )

    def test_place_und_layers_split_parity(self):
        """llama.cpp-style split placement must not change greedy output.

        CPU-only, so both groups land on cpu, but the split-aware forward
        (per-device rope cache, forced torch attention backend, _logits device
        move) is the code that runs on a real GPU split too.
        """
        r = make_reasoner()
        messages = templates.build_messages(
            "t2v", "a cat", aspect_ratio="16,9", resolution_w=832, resolution_h=480, fps=24, duration_secs=7,
        )
        with torch.no_grad():
            baseline = r.generate(messages, max_new_tokens=4)
            n_gpu, n_cpu = place_und_layers(r.transformer, torch.device("cpu"), gpu_layers=1)
            self.assertEqual((n_gpu, n_cpu), (1, len(r.transformer.layers) - 1))
            split_out = r.generate(messages, max_new_tokens=4)
            # prefill_device=None on a cpu "gpu" group: stream branch must stay off
            r.prefill_device = torch.device("cpu")
            stream_off = r.generate(messages, max_new_tokens=4)
        self.assertEqual(baseline, split_out)
        self.assertEqual(baseline, stream_off)

    def test_place_und_layers_all_on_device(self):
        r = make_reasoner()
        n_gpu, n_cpu = place_und_layers(r.transformer, torch.device("cpu"), gpu_layers=-1)
        self.assertEqual((n_gpu, n_cpu), (len(r.transformer.layers), 0))

    def test_strip_generation_weights(self):
        r = make_reasoner()
        n_before = sum(p.numel() for p in r.transformer.parameters())
        strip_generation_weights(r.transformer)
        n_after = sum(p.numel() for p in r.transformer.parameters())
        self.assertLess(n_after, n_before)
        self.assertIsInstance(r.transformer.layers[0].mlp_moe_gen, torch.nn.Identity)
        messages = templates.build_messages(
            "t2v", "a cat", aspect_ratio="16,9", resolution_w=832, resolution_h=480, fps=24, duration_secs=7,
        )
        with torch.no_grad():
            out = r.generate(messages, max_new_tokens=2)
        self.assertIsInstance(out, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
