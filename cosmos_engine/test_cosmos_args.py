# Static tests for cosmos_generate_video.py: mode inference, defaults,
# action loading round-trip, and control preprocessors. CPU-only, no weights.
# Run: python test_cosmos_args.py
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cosmos_generate_video import (
    compute_control_frames,
    compute_transfer_chunk_plan,
    detect_task,
    get_num_chunks,
    load_actions,
    parse_args,
    resolve_negative_prompt,
    run_transfer_chunks,
    setup_args,
)
from cosmos_video import configs as cfg


def make_args(*argv):
    old = sys.argv
    sys.argv = ["cosmos_generate_video.py", "--save_path", "/tmp/out", "--ckpt_dir", "/tmp/ckpt", "--prompt", "p"] + list(argv)
    try:
        return parse_args()
    finally:
        sys.argv = old


def test_mode_inference():
    cases = [
        ([], "t2v"),
        (["--video_length", "1"], "t2i"),
        (["--image_path", "x.png"], "i2v"),
        (["--video_path", "x.mp4"], "v2v"),
        (["--control_edge", "e.mp4"], "transfer"),
        (["--control_path", "d.mp4", "--control_type", "depth"], "transfer"),
        (["--domain_name", "droid_lerobot", "--action_path", "a.json", "--image_path", "x.png"], "forward_dynamics"),
        (["--domain_name", "av", "--video_path", "x.mp4"], "inverse_dynamics"),
        (["--domain_name", "droid_lerobot", "--policy", "--image_path", "x.png"], "policy"),
        (["--domain_name", "umi", "--image_path", "x.png"], "policy"),
        (["--task", "t2v", "--image_path", "x.png"], "t2v"),  # explicit task wins
    ]
    for argv, expected in cases:
        got = detect_task(make_args(*argv))
        assert got == expected, f"{argv}: expected {expected}, got {got}"
    print(f"mode inference: {len(cases)} cases OK")


def test_defaults():
    args = make_args()
    setup_args(args, "t2v")
    assert (args.width, args.height) == (1280, 720)
    assert args.video_length == 189 and args.fps == 24
    assert args.infer_steps == 35 and args.guidance_scale == 6.0
    assert args.flow_shift == 10.0  # 720 tier

    args = make_args("--resolution", "480")
    setup_args(args, "t2v")
    assert (args.width, args.height) == (832, 480)
    assert args.flow_shift == 5.0

    args = make_args("--resolution", "256", "--aspect_ratio", "9:16")
    setup_args(args, "t2v")
    assert (args.width, args.height) == (192, 320)
    assert args.flow_shift == 3.0

    args = make_args("--video_length", "100")
    setup_args(args, "t2v")
    assert args.video_length == 101, args.video_length  # 4k+1 rounding

    args = make_args("--control_wsm", "w.mp4")
    setup_args(args, "transfer")
    assert args.guidance_scale == 1.0 and args.control_guidance == 3.0  # wsm tuning
    assert args.video_length == 101 and args.fps == 10

    args = make_args("--control_edge", "e.mp4")
    setup_args(args, "transfer")
    assert args.guidance_scale == 3.0 and args.control_guidance == 1.5

    args = make_args("--domain_name", "agibotworld", "--policy", "--image_path", "x.png")
    setup_args(args, "policy")
    assert args.video_length == 17  # chunk 16 + 1
    assert args.infer_steps == 30 and args.guidance_scale == 1.0
    print("defaults: OK")


def test_action_roundtrip():
    for domain, dim in cfg.EMBODIMENT_TO_RAW_ACTION_DIM.items():
        actions = np.random.RandomState(0).randn(16, dim).astype(np.float32)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"actions": actions.tolist()}, f)
            path = f.name
        try:
            args = make_args("--domain_name", domain, "--action_path", path, "--image_path", "x.png")
            setup_args(args, "forward_dynamics")
            loaded = load_actions(args)
            assert loaded.shape == (16, dim)
            assert np.allclose(loaded, actions, atol=1e-6)
        finally:
            os.unlink(path)
    # dim mismatch must raise
    args = make_args("--domain_name", "av", "--action_data", json.dumps(np.zeros((16, 5)).tolist()), "--image_path", "x")
    setup_args(args, "forward_dynamics")
    try:
        load_actions(args)
        raise AssertionError("dim mismatch not caught")
    except ValueError:
        pass
    print(f"action round-trip: {len(cfg.EMBODIMENT_TO_RAW_ACTION_DIM)} domains OK")


def test_transfer_args():
    # Multi-hint parsing: hints resolve in the fixed upstream order regardless of CLI order.
    args = make_args("--control_seg", "s.mp4", "--control_edge", "e.mp4", "--control_depth", "d.mp4",
                     "--control_weight", "0.7", "0.3", "--control_guidance", "2.0",
                     "--control_guidance_interval", "100", "900")
    assert detect_task(args) == "transfer"
    setup_args(args, "transfer")
    assert args.active_hints == ["edge", "depth", "seg"]
    assert args.control_weight == [0.7, 0.3]
    assert args.control_guidance == 2.0
    assert args.control_guidance_interval == [100.0, 900.0]
    # Multi-hint: no per-hint tuning, base transfer defaults apply.
    assert args.guidance_scale == 3.0 and args.infer_steps == 35

    # Chunk defaults filled from TRANSFER_CHUNK_DEFAULTS for transfer.
    args = make_args("--control_edge", "e.mp4")
    setup_args(args, "transfer")
    assert args.num_frames_per_chunk == cfg.TRANSFER_CHUNK_DEFAULTS["num_frames_per_chunk"]
    assert args.num_conditional_frames == cfg.TRANSFER_CHUNK_DEFAULTS["num_conditional_frames"]
    assert args.num_first_chunk_conditional_frames == cfg.TRANSFER_CHUNK_DEFAULTS["num_first_chunk_conditional_frames"]
    assert args.max_frames == cfg.TRANSFER_CHUNK_DEFAULTS["max_frames"]
    assert args.control_guidance_interval is None

    # User overrides win over chunk defaults.
    args = make_args("--control_edge", "e.mp4", "--num_frames_per_chunk", "49", "--num_conditional_frames", "5")
    setup_args(args, "transfer")
    assert args.num_frames_per_chunk == 49 and args.num_conditional_frames == 5

    # wsm single-hint tuning includes its own chunk length.
    args = make_args("--control_wsm", "w.mp4")
    setup_args(args, "transfer")
    assert args.num_frames_per_chunk == 101
    print("transfer args: OK")


def test_transfer_chunk_math():
    # Framework _get_num_chunks hand-computed values.
    assert get_num_chunks(93, 93, 1) == (1, 93)      # exactly one chunk
    assert get_num_chunks(50, 93, 1) == (1, 93)      # short video, single chunk
    assert get_num_chunks(200, 93, 1) == (3, 92)     # 200 frames, stride 93-1
    assert get_num_chunks(1, 1, 0) == (1, 1)         # single frame
    assert get_num_chunks(185, 93, 1) == (2, 92)     # remaining 92 exactly one stride

    # 200 frames / chunk 93 / cond 1 -> spans [0..93), [92..185), [184..200)
    chunk_frames, num_chunks, stride, spans = compute_transfer_chunk_plan(200, 93, 1)
    assert chunk_frames == 93 and num_chunks == 3 and stride == 92
    assert spans == [(0, 93), (92, 185), (184, 200)], spans

    # chunk length is rounded up to the VAE 4k+1 cadence
    chunk_frames, _, stride, _ = compute_transfer_chunk_plan(200, 90, 1)
    assert chunk_frames == 93 and stride == 92

    # 12 frames / chunk 9 / cond 1 -> 2 chunks, stride 8, spans [0..9), [8..12)
    chunk_frames, num_chunks, stride, spans = compute_transfer_chunk_plan(12, 9, 1)
    assert (chunk_frames, num_chunks, stride) == (9, 2, 8)
    assert spans == [(0, 9), (8, 12)], spans
    print("transfer chunk math: OK")


def test_transfer_chunk_loop():
    # Synthetic chunk loop with a fake generator: verify conditioning frames, control slicing,
    # drop-leading-conditional-frames concat behavior, and the final trim.
    total, chunk_frames, cond = 12, 9, 1
    chunk_frames, num_chunks, stride, spans = compute_transfer_chunk_plan(total, chunk_frames, cond)
    control = {"edge": [np.full((4, 4, 3), t, dtype=np.uint8) for t in range(total)]}
    calls = []

    def fake_generate(chunk_id, control_chunk, cond_frames, condition_frame_indexes):
        calls.append((chunk_id, len(control_chunk["edge"]), cond_frames is not None, condition_frame_indexes))
        start = chunk_id * stride
        # Return frames labeled by their absolute source index (padded chunks repeat the last label).
        labels = [min(start + i, total - 1) for i in range(chunk_frames)]
        return np.stack([np.full((4, 4, 3), lab, dtype=np.uint8) for lab in labels])

    out = run_transfer_chunks(control, None, total, chunk_frames, stride, num_chunks, cond, 0, fake_generate)
    assert out.shape[0] == total, out.shape
    assert calls[0] == (0, 9, False, None)
    assert calls[1] == (1, 4, True, [0])  # sliced control chunk [8:12), 1 decoded cond frame
    # chunk 0 contributes 9 frames, chunk 1 drops its leading conditional frame -> 8 more, trimmed to 12
    assert [int(out[t, 0, 0, 0]) for t in range(total)] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    print("transfer chunk loop: OK")


def test_text_cache_flag():
    args = make_args()
    assert args.no_text_cache is False  # cache is ON by default
    args = make_args("--no_text_cache")
    assert args.no_text_cache is True
    print("text cache flag: OK")


def test_default_negative_prompt():
    neg_content = {"negative_prompt": "blurry, low quality, distorted"}
    with tempfile.TemporaryDirectory() as ckpt:
        os.makedirs(os.path.join(ckpt, "assets"))
        with open(os.path.join(ckpt, "assets", "negative_prompt.json"), "w", encoding="utf-8") as f:
            json.dump(neg_content, f)

        # unset negative prompt + assets file present -> auto-loaded (json.dumps of parsed content)
        args = make_args()
        assert args.no_default_negative_prompt is False
        args.ckpt_dir = ckpt
        assert resolve_negative_prompt(args) == json.dumps(neg_content)

        # explicit --negative_prompt wins over the default
        args = make_args("--negative_prompt", "explicit neg")
        args.ckpt_dir = ckpt
        assert resolve_negative_prompt(args) == "explicit neg"

        # --no_default_negative_prompt disables the auto-load
        args = make_args("--no_default_negative_prompt")
        args.ckpt_dir = ckpt
        assert resolve_negative_prompt(args) is None

        # distilled checkpoints never load the default (no CFG)
        args = make_args("--distilled")
        args.ckpt_dir = ckpt
        assert resolve_negative_prompt(args) is None

    # no assets file -> None (pipeline falls back to empty string)
    args = make_args()
    assert resolve_negative_prompt(args) is None
    print("default negative prompt: OK")


def test_control_preprocessors():
    from PIL import Image

    gradient = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
    frames = [Image.fromarray(np.stack([gradient] * 3, axis=-1))] * 3

    args = make_args("--edge_threshold", "medium")
    edges = compute_control_frames(args, frames, "edge")
    assert len(edges) == 3 and edges[0].size == (128, 128)

    args = make_args("--blur_strength", "high")
    blurred = compute_control_frames(args, frames, "blur")
    assert len(blurred) == 3 and blurred[0].size == (128, 128)

    try:
        compute_control_frames(args, frames, "depth")
        raise AssertionError("depth on-the-fly should raise")
    except ValueError:
        pass
    print("control preprocessors: OK")


if __name__ == "__main__":
    test_mode_inference()
    test_defaults()
    test_action_roundtrip()
    test_transfer_args()
    test_transfer_chunk_math()
    test_transfer_chunk_loop()
    test_text_cache_flag()
    test_default_negative_prompt()
    test_control_preprocessors()
    print("ALL TESTS PASSED")
