# Global flag to control torch.compile() usage for the MiniMax-H3 model code.
# Mirrors wan/modules/compile_config.py: set USE_TORCH_COMPILE before importing the
# model modules (transformer.py / attention.py) so the @maybe_compile decorators
# see the flag at import time.
USE_TORCH_COMPILE = False  # Default to False, enabled via --compile flag


def maybe_compile(fn=None, **compile_kwargs):
    """
    Decorator that conditionally applies torch.compile() based on USE_TORCH_COMPILE flag.

    Follows the wan2.2 pattern for torch.compile integration:
    - Dynamic shapes support (dynamic=True by default)
    - No CUDA graphs (max-autotune-no-cudagraphs mode)
    - Function-level compilation instead of block-level

    Usage:
        @maybe_compile()
        def my_function(x):
            ...

        @maybe_compile(mode="max-autotune-no-cudagraphs", dynamic=True)
        def my_other_function(x):
            ...
    """
    import torch

    # Set sensible defaults for video diffusion models
    if 'mode' not in compile_kwargs:
        compile_kwargs['mode'] = "max-autotune-no-cudagraphs"
    if 'dynamic' not in compile_kwargs:
        compile_kwargs['dynamic'] = True

    def decorator(func):
        if USE_TORCH_COMPILE:
            return torch.compile(func, **compile_kwargs)
        return func

    if fn is not None:
        # Called without parentheses: @maybe_compile
        return decorator(fn)

    # Called with parentheses: @maybe_compile() or @maybe_compile(mode=...)
    return decorator


def set_compile_enabled(enabled: bool):
    """Set the global compile flag. Must be called before importing model modules."""
    global USE_TORCH_COMPILE
    USE_TORCH_COMPILE = enabled
