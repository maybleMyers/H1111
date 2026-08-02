"""Minimal test runner for the minimax_engine static tests (the project venv has no pytest).

    env/bin/python minimax_engine/tests/run_static_tests.py [module ...]

Runs every `test_*` function of the given test modules (default: all test_minimax_* modules
in this directory). Provides a tiny `pytest` stand-in with `skip` and `raises`.
"""

import importlib
import importlib.machinery
import os
import sys
import traceback
import types


def _install_pytest_shim() -> type:
    pt = types.ModuleType("pytest")
    pt.__spec__ = importlib.machinery.ModuleSpec("pytest", None)

    class Skip(Exception):
        pass

    def skip(msg=""):
        raise Skip(msg)

    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, tb):
            if exc_type is None or not issubclass(exc_type, self.exc):
                raise AssertionError(f"expected {self.exc.__name__} to be raised")
            return True

    pt.skip = skip
    pt.raises = lambda exc: _Raises(exc)
    sys.modules.setdefault("pytest", pt)
    return Skip


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    Skip = _install_pytest_shim()

    modules = sys.argv[1:] or sorted(
        name[:-3] for name in os.listdir(here) if name.startswith("test_minimax") and name.endswith(".py")
    )
    passed = failed = skipped = 0
    for module_name in modules:
        module = importlib.import_module(module_name.removesuffix(".py"))
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(module, name)()
                passed += 1
                print(f"PASS {module_name}::{name}")
            except Skip as e:
                skipped += 1
                print(f"SKIP {module_name}::{name} ({e})")
            except Exception:
                failed += 1
                print(f"FAIL {module_name}::{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
