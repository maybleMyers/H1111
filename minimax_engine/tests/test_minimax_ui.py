"""Static checks for the h1111.py MiniMax tab: helper behavior and wiring consistency.

h1111.py launches the Gradio server at import time, so these tests parse the file and exec
only the MiniMax helper functions, and verify the wiring counts via the AST.
"""

import ast
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
H1111_PATH = os.path.join(_REPO_ROOT, "h1111.py")

_TREE = None


def _tree() -> ast.Module:
    global _TREE
    if _TREE is None:
        _TREE = ast.parse(open(H1111_PATH, encoding="utf-8").read())
    return _TREE


def _get_function(name: str) -> ast.FunctionDef:
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in h1111.py")


def _exec_function(name: str):
    """Exec one top-level function from h1111.py with stub globals; returns the callable."""
    node = _get_function(name)
    module = ast.Module(body=[node], type_ignores=[])

    class _GrError(Exception):
        pass

    gr_stub = types.SimpleNamespace(Error=_GrError)
    namespace = {"gr": gr_stub, "os": os, "sys": sys}
    exec(compile(module, H1111_PATH, "exec"), namespace)
    return namespace[name], _GrError


def test_align_num_frames_helper():
    fn, _ = _exec_function("minimax_align_num_frames")
    assert fn(120) == 124
    assert fn(124) == 124
    assert fn(125) == 141
    assert fn(1) == 5


def test_order_references_helper():
    fn, GrError = _exec_function("minimax_order_references")
    files = ["/a.png", "/b.mp4", "/c.wav"]
    assert fn(files, "") == files
    assert fn(files, "3 1 2") == ["/c.wav", "/a.png", "/b.mp4"]
    assert fn(files, "3, 1, 2") == ["/c.wav", "/a.png", "/b.mp4"]
    assert fn(None, "") == []
    try:
        fn(files, "1 1 2")
    except GrError:
        pass
    else:
        raise AssertionError("duplicate order indexes must raise")
    try:
        fn(files, "1 2")
    except GrError:
        pass
    else:
        raise AssertionError("incomplete order must raise")


def _click_input_count(func_name: str) -> int:
    """Count the expanded inputs list of the .click(fn=<func_name>, inputs=[...]) call."""
    for node in ast.walk(_tree()):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "click"):
            continue
        fn_kw = next((kw for kw in node.keywords if kw.arg == "fn"), None)
        if fn_kw is None or not isinstance(fn_kw.value, ast.Name) or fn_kw.value.id != func_name:
            continue
        inputs_kw = next(kw for kw in node.keywords if kw.arg == "inputs")
        count = 0
        for element in inputs_kw.value.elts:
            if isinstance(element, ast.Starred):
                # *minimax_lora_weights / *minimax_lora_multipliers are built in a range(4) loop.
                count += 4
            else:
                count += 1
        return count
    raise AssertionError(f".click(fn={func_name}) not found")


def test_click_inputs_match_submit_signature():
    submit = _get_function("minimax_submit_to_queue")
    expected = len(submit.args.args)
    assert expected == _click_input_count("minimax_generate_via_queue"), (
        "minimax_generate_btn.click inputs do not match minimax_submit_to_queue's parameter count"
    )


def test_defaults_lists_are_consistent():
    """The ORDERED_LIST and keys list must have equal length (keys list is explicit +4 lora
    weights +4 multipliers; components list appends the two 4-element lists)."""
    source = open(H1111_PATH, encoding="utf-8").read()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "minimax_ui_default_components_ORDERED_LIST" for t in node.targets
        ):
            components_expr = node.value
            break
    else:
        raise AssertionError("minimax_ui_default_components_ORDERED_LIST not found")
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "minimax_ui_default_keys" for t in node.targets
        ):
            keys_expr = node.value
            break
    else:
        raise AssertionError("minimax_ui_default_keys not found")

    def literal_count(expr) -> int:
        # A + B (+ C): lists of literals plus the two 4-element lora lists / comprehensions.
        if isinstance(expr, ast.BinOp):
            return literal_count(expr.left) + literal_count(expr.right)
        if isinstance(expr, ast.List):
            return len(expr.elts)
        if isinstance(expr, ast.ListComp):
            # every comprehension here iterates range(4)
            return 4
        if isinstance(expr, ast.Name):
            # minimax_lora_weights / minimax_lora_multipliers: built in a range(4) loop
            return 4
        raise AssertionError(f"unhandled expr node {type(expr).__name__}")

    assert literal_count(components_expr) == literal_count(keys_expr), (
        "minimax defaults components list and keys list are different lengths"
    )
    assert "MINIMAX_DEFAULTS_FILE" in source


def test_command_builder_flags_exist_in_cli():
    """Every --flag the submit handler emits must be a real CLI argument."""
    submit_src = ast.get_source_segment(open(H1111_PATH, encoding="utf-8").read(), _get_function("minimax_submit_to_queue"))
    cli_src = open(os.path.join(_REPO_ROOT, "minimax_engine", "minimax_generate_video.py"), encoding="utf-8").read()
    import re

    emitted = set(re.findall(r'"(--[a-z_]+)"', submit_src))
    # \s* tolerates multi-line add_argument( calls, where the flag sits on the next line
    declared = set(re.findall(r'add_argument\(\s*"(--[a-z_]+)"', cli_src))
    missing = emitted - declared
    assert not missing, f"submit handler emits flags the CLI does not declare: {sorted(missing)}"
