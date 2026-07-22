# Cosmos3 task configs and constant tables for H1111.
# Values ported from NVIDIA/cosmos-framework@058c8c0 (Apache-2.0):
#   cosmos_framework/data/generator/utils.py (VIDEO_RES_SIZE_INFO)
#   cosmos_framework/inference/args.py (defaults, presets, transfer tuning)
#   cosmos_framework/data/generator/action/domain_utils.py (embodiment tables)

# (width, height) per resolution tier and aspect ratio.
VIDEO_RES_SIZE_INFO = {
    "256": {"1:1": (256, 256), "4:3": (320, 256), "3:4": (256, 320), "16:9": (320, 192), "9:16": (192, 320)},
    "480": {"1:1": (640, 640), "4:3": (736, 544), "3:4": (544, 736), "16:9": (832, 480), "9:16": (480, 832)},
    "704": {"1:1": (960, 960), "4:3": (1088, 832), "3:4": (832, 1088), "16:9": (1280, 704), "9:16": (704, 1280)},
    "720": {"1:1": (960, 960), "4:3": (1104, 832), "3:4": (832, 1104), "16:9": (1280, 720), "9:16": (720, 1280)},
    "768": {"1:1": (1024, 1024), "4:3": (1184, 880), "3:4": (880, 1184), "16:9": (1360, 768), "9:16": (768, 1360)},
    "1080": {"1:1": (1440, 1440), "4:3": (1664, 1248), "3:4": (1248, 1664), "16:9": (1920, 1080), "9:16": (1080, 1920)},
}
IMAGE_ONLY_RESOLUTIONS = {"1080"}
MIN_NUM_FRAMES = 24
MAX_NUM_FRAMES = {"256": 400, "480": 300, "704": 200, "720": 200, "768": 200}

VAE_SPATIAL_COMPRESSION = 16
VAE_TEMPORAL_COMPRESSION = 4

# Per-resolution flow-shift training regime (cosmos_framework/inference/args.py
# _RESOLUTION_SHIFT_DEFAULTS).
RESOLUTION_SHIFT_DEFAULTS = {"256": 3.0, "480": 5.0, "704": 10.0, "720": 10.0, "768": 10.0}

TASKS = (
    "t2i",
    "t2v",
    "i2v",
    "v2v",
    "transfer",
    "forward_dynamics",
    "inverse_dynamics",
    "policy",
)
ACTION_TASKS = ("forward_dynamics", "inverse_dynamics", "policy")

# Baseline sampling defaults (cosmos_framework/inference/defaults/*/sample_args.json).
# flow_shift is intentionally absent for the plain video tasks: when unset it
# resolves per resolution tier via RESOLUTION_SHIFT_DEFAULTS (256->3, 480->5, 720->10).
TASK_DEFAULTS = {
    "t2i":              {"infer_steps": 35, "guidance_scale": 6.0, "video_length": 1},
    "t2v":              {"infer_steps": 35, "guidance_scale": 6.0, "video_length": 189, "fps": 24},
    "i2v":              {"infer_steps": 35, "guidance_scale": 6.0, "video_length": 189, "fps": 24},
    "v2v":              {"infer_steps": 35, "guidance_scale": 6.0, "video_length": 189, "fps": 24},
    "transfer":         {"infer_steps": 35, "guidance_scale": 3.0, "video_length": 189, "fps": 24},
    "forward_dynamics": {"infer_steps": 30, "guidance_scale": 1.0, "flow_shift": 10.0},
    "inverse_dynamics": {"infer_steps": 30, "guidance_scale": 1.0, "flow_shift": 10.0},
    "policy":           {"infer_steps": 30, "guidance_scale": 1.0, "flow_shift": 10.0},
}


def resolve_flow_shift(height, width):
    """Per-resolution-tier training shift when the user did not set one."""
    short_side = min(height, width)
    if short_side >= 704:
        return 10.0
    if short_side >= 480:
        return 5.0
    return 3.0

# Transfer control hints, in the deterministic sequence order used upstream.
TRANSFER_HINTS = ("edge", "blur", "depth", "seg", "wsm")
# Per-hint tuned defaults, applied only for single-hint runs when the user did
# not override (cosmos_framework/inference/args.py _TRANSFER_DEFAULTS).
TRANSFER_HINT_DEFAULTS = {
    "edge": {"guidance_scale": 3.0, "control_guidance": 1.5, "flow_shift": 10.0},
    "blur": {"guidance_scale": 3.0, "control_guidance": 1.5, "flow_shift": 10.0},
    "depth": {"guidance_scale": 3.0, "control_guidance": 1.5, "flow_shift": 10.0},
    "seg": {"guidance_scale": 3.0, "control_guidance": 2.0, "flow_shift": 10.0},
    "wsm": {"guidance_scale": 1.0, "control_guidance": 3.0, "flow_shift": 10.0, "video_length": 101, "fps": 10,
            "num_frames_per_chunk": 101},
}
TRANSFER_CHUNK_DEFAULTS = {
    "num_frames_per_chunk": 93,
    "num_conditional_frames": 1,
    "num_first_chunk_conditional_frames": 0,
    "max_frames": 5000,
    "share_vision_temporal_positions": True,
    "emphasize_control_in_prompt": True,
}
EMPHASIZE_CONTROL_PROMPT_SUFFIX = (
    " Follow the {hints} control video precisely: shape, contour, silhouette, position, and motion of "
    "every visible structure must align with the {hints} signal at every frame."
)

# Canny thresholds (t_lower, t_upper) per preset
# (data/generator/augmentors/transfer_control_input/control_input.py).
EDGE_THRESHOLD_PRESETS = {
    "very_low": (20, 50),
    "low": (50, 100),
    "medium": (100, 200),
    "high": (200, 300),
    "very_high": (300, 400),
}
# Blur strength presets: (downup_factor, gaussian_downup_factor).
BLUR_STRENGTH_PRESETS = {
    "none": (1, 1),
    "very_low": (4, 1),
    "low": (4, 4),
    "medium": (10, 2),
    "high": (16, 1),
    "very_high": (16, 4),
}

# Embodiment tables (data/generator/action/domain_utils.py).
EMBODIMENT_TO_DOMAIN_ID = {
    "av": 0,
    "camera_pose": 1,
    "hand_pose": 2,
    "pusht": 3,
    "umi": 4,
    "bridge_orig_lerobot": 5,
    "droid_lerobot": 6,
    "robomind-franka": 7,
    "robomind-franka-dual": 8,
    "robomind-ur": 9,
    "embodiment_b": 10,
    "agibotworld": 11,
    "embodiment_c_gripper": 12,
    "embodiment_c_gripper_ext": 13,
    "xdof_yam": 14,
    "molmoact2_yam": 15,
    "fractal": 16,
}
EMBODIMENT_TO_RAW_ACTION_DIM = {
    "av": 9,
    "camera_pose": 9,
    "hand_pose": 57,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "embodiment_b": 30,
    "agibotworld": 29,
    "embodiment_c_gripper": 29,
    "embodiment_c_gripper_ext": 29,
    "xdof_yam": 20,
    "molmoact2_yam": 20,
    "fractal": 10,
}
MAX_ACTION_DIM = 64
ACTION_DEFAULTS = {"image_size": 256, "action_chunk_size": 16, "view_point": "ego_view"}

# Reasoner prompt-upsampling sampler defaults (cosmos/README.md).
PROMPT_UPSAMPLER_DEFAULTS = {
    "max_tokens": 20000,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
    "seed": 3407,
}


def resolve_video_size(resolution, aspect_ratio):
    """(width, height) for a resolution tier + aspect ratio."""
    try:
        return VIDEO_RES_SIZE_INFO[str(resolution)][aspect_ratio]
    except KeyError:
        raise ValueError(
            f"unsupported resolution/aspect combination: {resolution} / {aspect_ratio}; "
            f"resolutions: {sorted(VIDEO_RES_SIZE_INFO)}, aspects: 1:1, 4:3, 3:4, 16:9, 9:16"
        )


def round_num_frames(num_frames, temporal_compression=VAE_TEMPORAL_COMPRESSION):
    """Round frame count up to the VAE-legal 4k+1 cadence (args.py:531)."""
    if num_frames <= 1:
        return 1
    import math

    return math.ceil((num_frames - 1) / temporal_compression) * temporal_compression + 1


def apply_task_defaults(args, task):
    """Fill unset args (None) from TASK_DEFAULTS / transfer hint tuning."""
    defaults = dict(TASK_DEFAULTS.get(task, {}))
    if task == "transfer" and getattr(args, "active_hints", None) and len(args.active_hints) == 1:
        defaults.update(TRANSFER_HINT_DEFAULTS.get(args.active_hints[0], {}))
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args
