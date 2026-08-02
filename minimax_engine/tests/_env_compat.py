"""Test-only import shim for the local venv.

The local venv ships bitsandbytes 0.45.0 next to triton 3.4.0; triton 3.x removed `triton.ops`,
so `import diffusers.models.modeling_utils` dies inside `bitsandbytes.nn.triton_based_modules`.
This affects every vendored engine (cosmos included) and is unrelated to the MiniMax port —
the real fix is upgrading bitsandbytes (>= 0.47 no longer imports `triton.ops`).

Importing this module before diffusers installs a minimal fake `triton.ops` /
`triton.ops.matmul_perf_model` so the static (CPU, dummy-weight) tests can run. It is never
imported by the runtime code under `minimax_video/`.
"""

import sys
import types


def install() -> None:
    try:
        import triton  # noqa: F401
    except ImportError:
        return
    try:
        import triton.ops  # noqa: F401
        return  # triton 2.x — nothing to shim
    except ImportError:
        pass

    ops = types.ModuleType("triton.ops")
    perf = types.ModuleType("triton.ops.matmul_perf_model")
    perf.early_config_prune = lambda *args, **kwargs: []
    perf.estimate_matmul_time = lambda *args, **kwargs: 0.0
    ops.matmul_perf_model = perf
    sys.modules["triton.ops"] = ops
    sys.modules["triton.ops.matmul_perf_model"] = perf


install()
