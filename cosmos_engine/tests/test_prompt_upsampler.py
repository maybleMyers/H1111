"""Offline tests for cosmos_video.prompt_upsampler (no network, no torch).

Run from the repo root or cosmos_engine/:

    python3 cosmos_engine/tests/test_prompt_upsampler.py
"""

import json
import os
import sys
import tempfile
import unittest

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))  # cosmos_engine/

from cosmos_video import prompt_upsampler as pu


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    """Records requests and replays canned chat-completion responses."""

    def __init__(self, contents):
        # contents: list of assistant message strings, one per call
        self.contents = list(contents)
        self.requests = []

    def request(self, method, url, json=None, headers=None, timeout=None):
        self.requests.append({"method": method, "url": url, "payload": json, "headers": headers})
        content = self.contents.pop(0)
        return FakeResponse({"choices": [{"message": {"content": content}}]})


def make_client(contents, **config_kwargs):
    config = pu.PromptUpsamplerConfig(
        endpoint_url="http://localhost:8000", model="test-model", retry_base_delay_s=0.0, **config_kwargs
    )
    session = FakeSession(contents)
    client = pu.PromptUpsamplerClient(config, session=session, sleep=lambda s: None)
    return client, session


POSTTRAIN_RESPONSE = (
    "<final_prompt>\n"
    '{"temporal_caption": "A dense caption.", "duration": "placeholder", "fps": "placeholder",'
    ' "resolution": {"H": "placeholder", "W": "placeholder"}, "aspect_ratio": "placeholder"}\n'
    "</final_prompt>\n"
    "<negative_prompt>blurry, washed out colors</negative_prompt>"
)


class TestUrlNormalization(unittest.TestCase):
    def test_variants(self):
        f = pu._normalize_openai_base_url
        self.assertEqual(f("http://localhost:8000"), "http://localhost:8000/v1")
        self.assertEqual(f("https://api.anthropic.com/v1/"), "https://api.anthropic.com/v1")
        self.assertEqual(f("https://x.test/v1/chat/completions"), "https://x.test/v1")
        self.assertEqual(f("api.example.com"), "https://api.example.com/v1")


class TestPromptDetection(unittest.TestCase):
    def test_is_upsampled_prompt(self):
        self.assertTrue(pu.is_upsampled_prompt('{"temporal_caption": "x"}'))
        self.assertTrue(pu.is_upsampled_prompt('```json\n{"a": 1}\n```'))
        self.assertFalse(pu.is_upsampled_prompt("a cat cooking pizza"))
        self.assertFalse(pu.is_upsampled_prompt("{not valid json"))
        self.assertFalse(pu.is_upsampled_prompt("[1, 2]"))
        self.assertFalse(pu.is_upsampled_prompt(""))
        self.assertFalse(pu.is_upsampled_prompt(None))


class TestResolveMode(unittest.TestCase):
    def test_auto(self):
        self.assertEqual(pu.resolve_mode("auto", has_image=True, video_length=189), "posttrain_image2video")
        self.assertEqual(pu.resolve_mode("auto", has_image=False, video_length=189), "text2video")
        self.assertEqual(pu.resolve_mode("auto", has_image=False, video_length=1), "text2image")

    def test_explicit_and_invalid(self):
        self.assertEqual(pu.resolve_mode("image2video", has_image=True, video_length=189), "image2video")
        with self.assertRaises(ValueError):
            pu.resolve_mode("bogus", has_image=False, video_length=189)


class TestTemplates(unittest.TestCase):
    def test_all_templates_render(self):
        t2i = pu.build_t2i_prompt_text("a red cube", resolution="720", aspect_ratio="16,9")
        self.assertIn("a red cube", t2i)
        self.assertIn("resolution 720", t2i)

        t2v = pu.build_t2v_prompt_text("a red cube spins", resolution="480", aspect_ratio="16,9", duration="7s", fps=24)
        self.assertIn("a red cube spins", t2v)
        self.assertIn("duration 7s", t2v)
        self.assertNotIn("IMAGE INPUT", t2v)

        i2v = pu.build_t2v_prompt_text(
            "the cube melts", resolution="480", aspect_ratio="16,9", duration="7s", fps=24, image_conditioned=True
        )
        self.assertIn("IMAGE INPUT", i2v)

        posttrain = pu.build_posttrain_i2v_prompt_text("the cube melts")
        self.assertIn("the cube melts", posttrain)
        self.assertIn("temporal_caption", posttrain)

    def test_i2v_messages_carry_image(self):
        messages = pu.build_i2v_messages(
            "x", image_url="data:image/png;base64,AAAA", resolution="480", aspect_ratio="16,9", duration="7s", fps=24
        )
        self.assertEqual(messages[1]["content"][0]["type"], "image_url")

    def test_posttrain_messages_have_no_system(self):
        messages = pu.build_posttrain_i2v_messages("x", image_url="data:image/png;base64,AAAA")
        self.assertEqual([m["role"] for m in messages], ["user"])


class TestClient(unittest.TestCase):
    def test_t2v_pins_output_parameters(self):
        raw = '```json\n{"temporal_caption": "cap", "fps": 999, "duration": "bad"}\n```'
        client, session = make_client([raw])
        record = client.upsample_t2v("a cat", resolution="480", aspect_ratio="16,9", duration="7s", fps=24)
        data = json.loads(record["prompt"])
        self.assertEqual(data["fps"], 24)
        self.assertEqual(data["duration"], "7s")
        self.assertEqual(data["resolution"], {"H": 480, "W": 832})
        self.assertEqual(data["aspect_ratio"], "16,9")
        # top_k/top_p omitted by default so plain OpenAI-compatible gateways accept the payload
        payload = session.requests[0]["payload"]
        self.assertNotIn("top_k", payload)
        self.assertNotIn("top_p", payload)
        self.assertEqual(payload["model"], "test-model")

    def test_posttrain_contract(self):
        client, _ = make_client([POSTTRAIN_RESPONSE])
        record = client.upsample_posttrain_i2v(
            "melt it", image_url="data:image/png;base64,AAAA", resolution="480", aspect_ratio="16,9",
            duration="7s", fps=24,
        )
        data = json.loads(record["prompt"])
        self.assertEqual(data["temporal_caption"], "A dense caption.")
        self.assertEqual(data["resolution"], {"H": 480, "W": 832})
        self.assertEqual(record["negative_prompt"], "blurry, washed out colors")

    def test_posttrain_missing_final_prompt_retries_then_fails(self):
        client, session = make_client(["no tags here"] * 3)
        with self.assertRaises(RuntimeError):
            client.upsample_posttrain_i2v(
                "x", image_url="d", resolution="480", aspect_ratio="16,9", duration="7s", fps=24
            )
        self.assertEqual(len(session.requests), 3)

    def test_retry_recovers(self):
        good = '{"temporal_caption": "cap"}'
        client, session = make_client(["not json at all", good])
        record = client.upsample_t2v("a cat", resolution="480", aspect_ratio="16,9", duration="7s", fps=24)
        self.assertIn("temporal_caption", record["prompt"])
        self.assertEqual(len(session.requests), 2)

    def test_auth_header(self):
        client, session = make_client(['{"a": 1}'], api_token="sk-test")
        client.upsample_t2i("x", resolution="720", aspect_ratio="1,1")
        self.assertEqual(session.requests[0]["headers"]["Authorization"], "Bearer sk-test")


class TestHelpers(unittest.TestCase):
    def test_duration_label(self):
        self.assertEqual(pu.derive_duration_label(189, 24), "7s")
        self.assertEqual(pu.derive_duration_label(121, 30), "4s")
        with self.assertRaises(ValueError):
            pu.derive_duration_label(189, 0)

    def test_resolution_tier_704_maps_to_720(self):
        self.assertEqual(pu._normalize_resolution_tier("704"), "720")
        self.assertEqual(pu._normalize_resolution_tier("480"), "480")

    def test_image_data_url(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n")
            path = f.name
        try:
            url = pu.image_path_to_data_url(path)
            self.assertTrue(url.startswith("data:image/png;base64,"))
        finally:
            os.unlink(path)

    def test_upsample_prompt_validations(self):
        with self.assertRaises(ValueError):
            pu.upsample_prompt("", endpoint_url="http://x")
        with self.assertRaises(ValueError):
            pu.upsample_prompt("a cat", endpoint_url="http://x", mode="image2video", image_path=None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
