"""Dummy-weight end-to-end tests: the full CLI against a tiny fake checkpoint (CPU).

Covers all three tasks, the audio mux (2 streams via ffprobe), the latent save and the
decode-only mode. Slow-ish (~a few minutes on CPU); run explicitly:

    env/bin/python minimax_engine/tests/run_static_tests.py test_minimax_cli_e2e
"""

import json
import os
import subprocess
import sys
import tempfile
import wave

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_ENGINE_DIR)
for _p in (_ENGINE_DIR, _REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _env_compat  # noqa: F401,E402

import numpy as np  # noqa: E402

_STATE = {}


def _tiny_ckpt() -> str:
    if "ckpt" not in _STATE:
        from make_tiny_checkpoint import build_tiny_checkpoint

        _STATE["ckpt"] = build_tiny_checkpoint(tempfile.mkdtemp(prefix="minimax_e2e_ckpt_"))
    return _STATE["ckpt"]


def _out_dir() -> str:
    if "out" not in _STATE:
        _STATE["out"] = tempfile.mkdtemp(prefix="minimax_e2e_out_")
    return _STATE["out"]


def _run_cli(extra_args, name):
    out_file = os.path.join(_out_dir(), f"{name}.mp4")
    cmd = [
        sys.executable,
        os.path.join(_ENGINE_DIR, "minimax_generate_video.py"),
        "--ckpt_dir", _tiny_ckpt(),
        "--save_path", _out_dir(),
        "--output_filename", out_file,
        "--infer_steps", "3",
        "--video_size", "64", "64",
        "--video_length", "124",
        "--seed", "7",
        "--attn_mode", "sdpa",
        "--device", "cpu",
        *extra_args,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = _HERE + os.pathsep + env.get("PYTHONPATH", "")
    env["MINIMAX_TEST_ENV_COMPAT"] = "1"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    if result.returncode != 0:
        raise AssertionError(f"CLI failed ({name}):\nSTDOUT:\n{result.stdout[-3000:]}\nSTDERR:\n{result.stderr[-3000:]}")
    return out_file, result


def _stream_kinds(path):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", path],
        capture_output=True,
        text=True,
    )
    return sorted(s["codec_type"] for s in json.loads(probe.stdout)["streams"])


def test_cli_t2va():
    out_file, result = _run_cli(["--prompt", "a red fox in the snow", "--output_type", "both"], "t2va")
    assert os.path.exists(out_file), "mp4 missing"
    assert _stream_kinds(out_file) == ["audio", "video"], "expected muxed audio+video streams"
    latent_file = os.path.join(_out_dir(), "t2va_latent.safetensors")
    assert os.path.exists(latent_file), "latent save missing"
    assert "Video saved to:" in result.stdout
    _STATE["latent_file"] = latent_file


def test_cli_fl2va():
    from PIL import Image

    first = os.path.join(_out_dir(), "first.png")
    last = os.path.join(_out_dir(), "last.png")
    rng = np.random.RandomState(0)
    Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(first)
    Image.fromarray(rng.randint(0, 255, (48, 80, 3), dtype=np.uint8)).save(last)

    out_file, _ = _run_cli(
        ["--prompt", "keyframe interpolation", "--image_path", first, "--last_image_path", last], "fl2va"
    )
    assert os.path.exists(out_file)
    assert _stream_kinds(out_file) == ["audio", "video"]


def test_cli_ref2va():
    from PIL import Image

    subject = os.path.join(_out_dir(), "subject.png")
    # Small reference image: resolve_reference_image_size upscales the short edge to 2048,
    # so keep the source modest for CPU (the tiny VAE still sees a 2048px encode).
    Image.fromarray(np.random.RandomState(1).randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(subject)

    voice = os.path.join(_out_dir(), "voice.wav")
    sample_rate = 32000  # audio VAE's own rate: no torchaudio resample needed
    t = np.arange(int(sample_rate * 6.0)) / sample_rate
    pcm = (np.sin(2 * np.pi * 440.0 * t) * 0.3 * 32767).astype(np.int16)
    with wave.open(voice, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())

    # num_frames left to the audio reference: --video_length 0 derives 6 s -> 141 frames? (snap 17n+5)
    out_file, result = _run_cli(
        [
            "--prompt", "the subject speaks in time with the reference recording",
            "--reference", subject,
            "--reference", voice,
            "--video_length", "0",
        ],
        "ref2va",
    )
    assert os.path.exists(out_file)
    assert _stream_kinds(out_file) == ["audio", "video"]
    assert "task=ref2va" in result.stderr + result.stdout


def test_cli_z_decode_only():
    latent_file = _STATE.get("latent_file")
    if latent_file is None or not os.path.exists(latent_file):
        import pytest

        pytest.skip("t2va latent output not available")
    out_file = os.path.join(_out_dir(), "decoded.mp4")
    cmd = [
        sys.executable,
        os.path.join(_ENGINE_DIR, "minimax_generate_video.py"),
        "--ckpt_dir", _tiny_ckpt(),
        "--save_path", _out_dir(),
        "--output_filename", out_file,
        "--latent_path", latent_file,
        "--device", "cpu",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = _HERE + os.pathsep + env.get("PYTHONPATH", "")
    env["MINIMAX_TEST_ENV_COMPAT"] = "1"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if result.returncode != 0:
        raise AssertionError(f"decode-only failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    assert os.path.exists(out_file)
    assert _stream_kinds(out_file) == ["audio", "video"]
