#!/usr/bin/env python
"""Run the diffusers PR 14355 conversion script against the vendored minimax_video classes.

The upstream `scripts/convert_minimax_h3_to_diffusers.py` (vendored below as
`_convert_minimax_h3_upstream.py`) imports two things from PR-branch diffusers:
`AutoencoderKLMiniMaxH3Audio` (key validation only) and `SAFE_WEIGHTS_INDEX_NAME` (present
in 0.33.1). This wrapper aliases the missing module path onto the vendored class and then
defers to the upstream script unchanged.

Usage (original release layout -> diffusers layout, sharded):

    python minimax_engine/convert_checkpoint.py \
        --checkpoint_path /media/mayble/External/MiniMax-H3/FL2VA \
        --output_path /media/mayble/External/MiniMax-H3-diffusers [--dry_run]

Convert the Ref2VA partition the same way with `--output_path <tmp>` and move its
`transformer/` into the main output as `transformer_ref/` (all other components are shared).
"""

import os
import sys
import types

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_here), _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Alias the PR-branch module path onto the vendored implementation before the upstream
# script imports it.
from minimax_video.vae_audio import AutoencoderKLMiniMaxH3Audio  # noqa: E402

_alias = types.ModuleType("diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio")
_alias.AutoencoderKLMiniMaxH3Audio = AutoencoderKLMiniMaxH3Audio
sys.modules["diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio"] = _alias

from _convert_minimax_h3_upstream import get_args, main  # noqa: E402

if __name__ == "__main__":
    # --transformer_only: convert just the transformer (used for the Ref2VA partition, whose
    # VAEs/scheduler are identical to FL2VA's and need no second conversion).
    transformer_only = "--transformer_only" in sys.argv
    if transformer_only:
        sys.argv.remove("--transformer_only")
    args = get_args()
    if not transformer_only:
        main(args)
    else:
        import _convert_minimax_h3_upstream as upstream
        from diffusers import __version__ as diffusers_version

        config = (
            upstream.MINIMAX_H3_TEST_TRANSFORMER_CONFIG
            if args.version == "test"
            else upstream.MINIMAX_H3_TRANSFORMER_CONFIG
        )
        if args.dry_run:
            upstream.dry_run(args.checkpoint_path, config)
        else:
            transformer_path = os.path.join(args.output_path, "transformer")
            upstream.convert_transformer(args.checkpoint_path, transformer_path, config, args.max_shard_size)
            upstream.write_transformer_config(transformer_path, config, diffusers_version)
