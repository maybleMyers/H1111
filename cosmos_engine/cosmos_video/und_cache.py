# H1111 Cosmos3: per-prompt cache for the understanding (text) pathway.
#
# The und tokens of the joint Cosmos3 MoT sequence are text-only: they receive no
# timestep embedding and never attend to gen tokens, so for a fixed prompt every
# layer's und hidden states — and therefore the post-RoPE k_und_for_gen / v_und
# consumed by the gen pathway's cross attention — are constant across all
# denoising steps. This cache captures those tensors on the first forward of a
# denoising loop and lets every later forward skip the und pathway entirely
# (projections, norms, causal attention, MLP) while remaining bit-identical to
# the uncached computation: the very same tensors are reused (no re-cast, no
# clone) and the gen-side op order is untouched.

import torch


class UndKVCache:
    """Holds per-layer post-RoPE ``k_und_for_gen`` / ``v_und`` tensors plus the
    final normed und hidden states for one packed prompt.

    Lifecycle: an empty cache passed to ``Cosmos3OmniTransformer.forward`` puts
    the forward in *capture* mode (compute normally, store); a populated cache
    puts it in *skip* mode (und pathway not computed, cached tensors consumed).
    """

    def __init__(self) -> None:
        # One (k_und_for_gen, v_und) tuple per decoder layer, both post-RoPE,
        # exactly as they enter the gen pathway's KV concat.
        self.layer_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        # Final `self.norm(und_seq)` output; only used to rebuild the joint
        # last_hidden_state with the exact same concat as the uncached path.
        self.und_out: torch.Tensor | None = None
        # Length of the und prefix this cache was captured for (safety check).
        self.und_len: int | None = None

    @property
    def populated(self) -> bool:
        return self.und_out is not None

    def clear(self) -> None:
        self.layer_kv = []
        self.und_out = None
        self.und_len = None

    def begin_capture(self, und_len: int) -> None:
        self.clear()
        self.und_len = und_len

    def store_layer(self, k_und_for_gen: torch.Tensor, v_und: torch.Tensor) -> None:
        self.layer_kv.append((k_und_for_gen.detach(), v_und.detach()))

    def finish_capture(self, und_out: torch.Tensor) -> None:
        self.und_out = und_out.detach()

    def validate(self, und_len: int, num_layers: int) -> None:
        if self.und_len != und_len:
            raise ValueError(
                f"UndKVCache was captured for und_len={self.und_len} but this forward has und_len={und_len}; "
                "the cache must be invalidated when the packed text changes."
            )
        if len(self.layer_kv) != num_layers:
            raise ValueError(
                f"UndKVCache holds {len(self.layer_kv)} layer entries but the model has {num_layers} layers."
            )
