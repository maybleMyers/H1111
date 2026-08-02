"""Static (CPU, dummy-weight) tests for the vendored MiniMax-H3 model code.

Run from the repo root with the project venv:

    env/bin/python -m pytest minimax_engine/tests/test_minimax_static.py -q

The parity tests additionally compare the vendored packing modules bit-for-bit against the
original PR sources when the PR checkout is available (MINIMAX_PR_DIR below or the
`MINIMAX_PR_DIR` environment variable); they are skipped otherwise.
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_ENGINE_DIR)
for _p in (_ENGINE_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _env_compat  # noqa: F401,E402  (works around the local bitsandbytes/triton breakage)

import pytest  # noqa: E402
import torch  # noqa: E402

from minimax_video import packing  # noqa: E402
from minimax_video.scheduler import MiniMaxH3Scheduler  # noqa: E402
from minimax_video.transformer import MiniMaxH3Transformer3DModel, MiniMaxH3TransformerOutput  # noqa: E402

MINIMAX_PR_DIR = os.environ.get(
    "MINIMAX_PR_DIR",
    "/tmp/claude-1000/-home-mayble-h1111-H1111/2e7c9119-cec4-4d1d-a752-591607e42cbd/scratchpad/minimax-pr",
)

# Mirrors the PR's own tiny test config (tests/models/transformers/test_models_transformer_minimax_h3.py):
# heads * head_dim (32) deliberately differs from hidden_size (24), and 2 * 3 * rope_freq_dim (12) is
# smaller than head_dim so the partial-rotary path is exercised.
TINY_CONFIG = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "hidden_size": 24,
    "num_layers": 2,
    "num_refiner_layers": 2,
    "ffn_dim": 32,
    "in_channels": 4,
    "audio_in_channels": 6,
    "patch_size": (1, 2, 2),
    "text_dim": 8,
    "freq_dim": 8,
    "time_embed_hidden_dim": 24,
    "time_embed_dim": 16,
    "rope_freq_dim": 2,
}

NUM_TEXT_TOKENS = 4
NUM_AUDIO_TOKENS = 6
NUM_VIDEO_TOKENS = 8


def _packed_layout():
    sequence_length = NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS + NUM_VIDEO_TOKENS
    text_indices = torch.arange(NUM_TEXT_TOKENS)
    audio_indices = torch.arange(NUM_TEXT_TOKENS, NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS)
    video_indices = torch.arange(NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS, sequence_length)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0

    timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
    timestep_indices[audio_indices] = 1

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32)
    position_ids[:, 0] = torch.arange(sequence_length, dtype=torch.float32)
    position_ids[video_indices, 1] = torch.arange(NUM_VIDEO_TOKENS, dtype=torch.float32) % 4
    position_ids[video_indices, 2] = torch.arange(NUM_VIDEO_TOKENS, dtype=torch.float32) % 2

    return {
        "timestep": torch.tensor([0.7, 0.3]),
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": position_ids,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
    }


def _dummy_inputs():
    generator = torch.Generator("cpu").manual_seed(0)
    patch = TINY_CONFIG["patch_size"]
    video_patch_dim = TINY_CONFIG["in_channels"] * patch[0] * patch[1] * patch[2]
    batch_size = 2
    return {
        "hidden_states": torch.randn(batch_size, NUM_VIDEO_TOKENS, video_patch_dim, generator=generator),
        "audio_hidden_states": torch.randn(
            batch_size, NUM_AUDIO_TOKENS, TINY_CONFIG["audio_in_channels"], generator=generator
        ),
        "encoder_hidden_states": torch.randn(batch_size, NUM_TEXT_TOKENS, TINY_CONFIG["text_dim"], generator=generator),
        **_packed_layout(),
    }


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_transformer_forward_shapes_and_output_format():
    torch.manual_seed(0)
    model = MiniMaxH3Transformer3DModel(**TINY_CONFIG).eval()
    inputs = _dummy_inputs()

    output = model(**inputs)
    output_tuple = model(**inputs, return_dict=False)

    patch = TINY_CONFIG["patch_size"]
    video_patch_dim = TINY_CONFIG["in_channels"] * patch[0] * patch[1] * patch[2]
    assert isinstance(output, MiniMaxH3TransformerOutput)
    assert output.sample.shape == (2, NUM_VIDEO_TOKENS, video_patch_dim)
    assert output.audio_sample.shape == (2, NUM_AUDIO_TOKENS, TINY_CONFIG["audio_in_channels"])
    torch.testing.assert_close(output.sample, output_tuple[0])
    torch.testing.assert_close(output.audio_sample, output_tuple[1])


@torch.no_grad()
def test_padding_rows_form_their_own_attention_document():
    """A padding tail (tag -1) must leave the live rows' predictions untouched.

    This exercises the boolean-mask path of the local attention dispatch, mirroring the
    reference's `cu_seqlens = [0, used, S]` split.
    """
    torch.manual_seed(0)
    model = MiniMaxH3Transformer3DModel(**TINY_CONFIG).eval()
    inputs = _dummy_inputs()
    padless = model(**inputs, return_dict=False)

    num_padding_rows = 3
    sequence_length = inputs["position_ids"].shape[0]
    padded = dict(inputs)
    padded["token_tags"] = torch.cat([inputs["token_tags"], torch.full((num_padding_rows,), -1, dtype=torch.long)])
    padded["timestep_indices"] = torch.cat(
        [inputs["timestep_indices"], torch.zeros(num_padding_rows, dtype=torch.long)]
    )
    padded["position_ids"] = torch.cat(
        [
            inputs["position_ids"],
            torch.arange(sequence_length, sequence_length + num_padding_rows, dtype=torch.float32)[:, None].repeat(
                1, 3
            ),
        ]
    )
    padded_out = model(**padded, return_dict=False)

    for a, b in zip(padless, padded_out):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_block_swap_methods_exist():
    model = MiniMaxH3Transformer3DModel(**TINY_CONFIG)
    assert callable(model.enable_block_swap)
    assert callable(model.move_to_device_except_swap_blocks)
    assert callable(model.prepare_block_swap_before_forward)
    model.prepare_block_swap_before_forward()  # no-op without enable_block_swap
    assert len(model.transformer_blocks) == TINY_CONFIG["num_layers"]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_scheduler_grid_matches_reference_formula():
    shift = 12.0
    sched = MiniMaxH3Scheduler(shift=shift)
    sched.set_timesteps(10)

    base = torch.linspace(1.0, 0.0, 10, dtype=torch.float32)
    expected = torch.unique_consecutive(shift * base / (1 + (shift - 1) * base))
    torch.testing.assert_close(sched.sigmas, expected)
    torch.testing.assert_close(sched.timesteps, 1.0 - expected[:-1])
    assert sched.num_inference_steps == expected.numel() - 1
    assert sched.sigmas[-1].item() == 0.0


def test_scheduler_step_data_ward_velocity():
    """One Euler step to sigma_next = 0 with the exact velocity must land on x0 (data-ward sign)."""
    sched = MiniMaxH3Scheduler(shift=1.0)  # shift 1 keeps the grid linear
    sched.set_timesteps(2)  # sigmas = [1.0, 0.0], one evaluation at t = 0
    x0 = torch.randn(2, 3, generator=torch.Generator().manual_seed(1))
    noise = torch.randn(2, 3, generator=torch.Generator().manual_seed(2))
    sigma = 1.0
    x_t = sched.scale_noise(x0, 1.0 - sigma, noise)  # t = 0 -> pure noise
    velocity = x0 - noise  # data-ward: x0 = x_t + sigma * v
    out = sched.step(velocity, sched.timesteps[0], x_t).prev_sample
    torch.testing.assert_close(out, x0, atol=1e-6, rtol=1e-6)


def test_scheduler_scale_noise_identity_at_t1():
    sched = MiniMaxH3Scheduler()
    x = torch.randn(4, 4, generator=torch.Generator().manual_seed(0))
    noise = torch.randn(4, 4, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(sched.scale_noise(x, 1.0, noise), x)
    torch.testing.assert_close(sched.scale_noise(x, 0.0, noise), noise)


# ---------------------------------------------------------------------------
# Packing geometry
# ---------------------------------------------------------------------------


def test_align_num_frames_and_latent_counts():
    assert packing.align_num_frames(120) == 124  # 17 * 7 + 5
    assert packing.align_num_frames(124) == 124
    assert packing.align_num_frames(125) == 141
    assert packing.video_latent_num_frames(124) == 37  # 5 * 7 + 2
    assert packing.audio_latent_num_frames(124) == round(124 / 24 * 40)


def test_resolve_canvas_size_contracts():
    h, w = packing.resolve_canvas_size(16, 9)
    assert h % 32 == 0 and w % 32 == 0
    assert min(h, w) == 768
    h2, w2 = packing.resolve_canvas_size(9, 16)
    assert (h2, w2) == (w, h)
    with pytest.raises(ValueError):
        packing.resolve_canvas_size(5, 1)  # aspect ratio above 4:1


def test_build_packed_sequence_structure():
    text_tags = torch.ones(7, dtype=torch.long)
    layout = packing.build_packed_sequence(
        text_token_tags=text_tags,
        num_latent_frames=4,
        latent_height=6,
        latent_width=8,
        num_audio_latents=5,
        patch_size=(1, 2, 2),
        keyframe_anchors=("first",),
    )
    rows_per_frame = (6 // 2) * (8 // 2)
    assert layout.sequence_length == 7 + rows_per_frame + 5 * 2 + 4 * rows_per_frame
    assert layout.num_condition_video_rows == rows_per_frame
    assert layout.num_condition_audio_rows == 0
    assert layout.position_ids.dtype == torch.float64
    # Row partition is exact and disjoint.
    all_indices = torch.cat([layout.text_indices, layout.video_indices, layout.audio_indices])
    assert sorted(all_indices.tolist()) == list(range(layout.sequence_length))
    # Tags match the index sets.
    assert (layout.token_tags[layout.audio_indices] == packing.MINIMAX_H3_AUDIO_TAG).all()
    assert (layout.token_tags[layout.video_indices] == packing.MINIMAX_H3_VIDEO_TAG).all()


def test_build_row_timesteps():
    text_tags = torch.ones(3, dtype=torch.long)
    layout = packing.build_packed_sequence(
        text_token_tags=text_tags,
        num_latent_frames=2,
        latent_height=2,
        latent_width=2,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
        keyframe_anchors=("first",),
    )
    timesteps, indices = packing.build_row_timesteps(
        layout, video_timestep=0.3, audio_timestep=0.5, condition_video_timestep=0.999, condition_audio_timestep=1.0
    )
    assert indices.shape == (layout.sequence_length,)
    reconstructed = timesteps[indices]
    assert reconstructed[layout.video_indices[: layout.num_condition_video_rows]].eq(0.999).all()
    assert reconstructed[layout.video_indices[layout.num_condition_video_rows :]].eq(0.3).all()
    assert reconstructed[layout.audio_indices].eq(0.5).all()
    assert reconstructed[layout.text_indices].eq(0.3).all()


def test_patchify_unpatchify_roundtrip():
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(1, 4, 3, 6, 8, generator=generator)
    rows = packing.patchify_video_latents(latents, (1, 2, 2))
    restored = packing.unpatchify_video_tokens(
        rows, num_latent_frames=3, latent_height=6, latent_width=8, channels=4, patch_size=(1, 2, 2)
    )
    torch.testing.assert_close(restored, latents)


# ---------------------------------------------------------------------------
# VAEs (tiny configs from the PR conversion script)
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_video_vae_tiny_encode_decode_shapes():
    from minimax_video.vae_video import AutoencoderKLMiniMaxH3

    torch.manual_seed(0)
    vae = AutoencoderKLMiniMaxH3(
        block_out_channels=(32, 64),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2),
        temporal_downsample_factors=(2, 2),
        decoder_num_layers=4,
        decoder_num_attention_heads=4,
        decoder_attention_head_dim=32,
    ).eval()
    assert vae.spatial_compression_ratio == 4
    # 17n + 5 frames -> 5n + 2 latents: one chunk (n = 1) is 22 frames -> 7 latents.
    frames = 22
    x = torch.randn(1, 3, frames, 32, 32)
    posterior = vae.encode(x).latent_dist
    latents = posterior.mode()
    assert latents.shape[:2] == (1, vae.config.latent_channels)
    assert latents.shape[2] == packing.video_latent_num_frames(frames)
    decoded = vae.decode(latents).sample
    assert decoded.shape == (1, 3, frames, 32, 32)


@torch.no_grad()
def test_audio_vae_tiny_encode_decode_shapes():
    from minimax_video.vae_audio import AutoencoderKLMiniMaxH3Audio

    torch.manual_seed(0)
    # The PR's own tiny config (tests/models/autoencoders/test_models_autoencoder_kl_minimax_h3_audio.py).
    vae = AutoencoderKLMiniMaxH3Audio(
        encoder_dim=4,
        encoder_rates=(2, 2),
        latent_dim=32,
        latent_channels=8,
        num_attention_heads=2,
        decoder_dim=16,
        decoder_rates=(2, 2),
        decoder_kernel_sizes=(4, 4),
        resblock_kernel_sizes=(3, 7),
        resblock_dilation_sizes=((1, 3), (1, 3)),
        sampling_rate=32000,
        latents_mean=[0.0] * 8,
        latents_std=[1.0] * 8,
    ).eval()
    hop = vae.hop_length
    waveform = torch.randn(1, 1, hop * 10)
    latents = vae.encode(waveform).latent_dist.mode()
    assert latents.shape[:2] == (1, vae.config.latent_channels)
    decoded = vae.decode(latents).sample
    assert decoded.shape[0] == 1 and decoded.shape[1] == 1


# ---------------------------------------------------------------------------
# Bit-for-bit parity with the PR sources (skipped when the checkout is absent)
# ---------------------------------------------------------------------------


def _load_pr_packing():
    path = os.path.join(MINIMAX_PR_DIR, "src/diffusers/modular_pipelines/minimax_h3/packing.py")
    if not os.path.isfile(path):
        pytest.skip(f"PR checkout not found at {MINIMAX_PR_DIR}")
    source = open(path).read().replace(
        "from ...utils.torch_utils import randn_tensor",
        "from diffusers.utils.torch_utils import randn_tensor",
    )
    spec = importlib.util.spec_from_loader("pr_packing", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, path, "exec"), module.__dict__)
    return module


def test_packing_parity_with_pr():
    pr = _load_pr_packing()

    for aspect in ((16, 9), (9, 16), (1, 1), (4, 3), (21, 9), (2.35, 1)):
        assert packing.resolve_canvas_size(*aspect) == pr.resolve_canvas_size(*aspect)
    for frames in (1, 5, 22, 120, 124, 361, 362):
        assert packing.align_num_frames(frames) == pr.align_num_frames(frames)

    text_tags = torch.tensor([1, 1, 0, 0, 1, 1], dtype=torch.long)
    for anchors in ((), ("first",), ("first", "last")):
        ours = packing.build_packed_sequence(text_tags, 7, 24, 40, 20, (1, 2, 2), anchors)
        theirs = pr.build_packed_sequence(text_tags, 7, 24, 40, 20, (1, 2, 2), anchors)
        assert ours.sequence_length == theirs.sequence_length
        assert torch.equal(ours.position_ids, theirs.position_ids)
        assert torch.equal(ours.token_tags, theirs.token_tags)
        assert torch.equal(ours.video_indices, theirs.video_indices)
        assert torch.equal(ours.audio_indices, theirs.audio_indices)
        assert torch.equal(ours.text_indices, theirs.text_indices)
        assert ours.num_condition_video_rows == theirs.num_condition_video_rows

    layout_ours = packing.build_packed_sequence(text_tags, 7, 24, 40, 20, (1, 2, 2), ("first",))
    layout_theirs = pr.build_packed_sequence(text_tags, 7, 24, 40, 20, (1, 2, 2), ("first",))
    ts_ours = packing.build_row_timesteps(layout_ours, 0.25, 0.75, 0.999, 1.0)
    ts_theirs = pr.build_row_timesteps(layout_theirs, 0.25, 0.75, 0.999, 1.0)
    assert torch.equal(ts_ours[0], ts_theirs[0]) and torch.equal(ts_ours[1], ts_theirs[1])

    noise_ours = packing.keyframe_condition_noise(
        ((1, 24, 40),), (1, 2, 2), 24, generator=torch.Generator().manual_seed(7)
    )
    noise_theirs = pr.keyframe_condition_noise(
        ((1, 24, 40),), (1, 2, 2), 24, generator=torch.Generator().manual_seed(7)
    )
    assert torch.equal(noise_ours, noise_theirs)


def test_scheduler_parity_with_pr():
    path = os.path.join(MINIMAX_PR_DIR, "src/diffusers/schedulers/scheduling_minimax_h3.py")
    if not os.path.isfile(path):
        pytest.skip(f"PR checkout not found at {MINIMAX_PR_DIR}")
    source = open(path).read()
    source = source.replace("from ..configuration_utils import", "from diffusers.configuration_utils import")
    source = source.replace("from ..utils import", "from diffusers.utils import")
    source = source.replace("from .scheduling_utils import", "from diffusers.schedulers.scheduling_utils import")
    spec = importlib.util.spec_from_loader("pr_scheduler", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, path, "exec"), module.__dict__)

    for shift, steps in ((12.0, 50), (3.0, 50), (12.0, 10), (5.0, 2)):
        ours = MiniMaxH3Scheduler(shift=shift)
        theirs = module.MiniMaxH3Scheduler(shift=shift)
        ours.set_timesteps(steps)
        theirs.set_timesteps(steps)
        assert torch.equal(ours.sigmas, theirs.sigmas)
        assert torch.equal(ours.timesteps, theirs.timesteps)

        x = torch.randn(2, 8, generator=torch.Generator().manual_seed(3))
        v = torch.randn(2, 8, generator=torch.Generator().manual_seed(4))
        a = ours.step(v, ours.timesteps[0], x).prev_sample
        b = theirs.step(v, theirs.timesteps[0], x).prev_sample
        assert torch.equal(a, b)
