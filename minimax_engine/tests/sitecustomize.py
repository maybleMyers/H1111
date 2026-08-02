"""Auto-loaded (via PYTHONPATH) when the e2e tests spawn the CLI as a subprocess: installs
the test-only triton.ops shim before anything imports diffusers. See _env_compat.py."""

import os
import sys

if os.environ.get("MINIMAX_TEST_ENV_COMPAT") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import _env_compat  # noqa: F401
    except Exception:
        pass
