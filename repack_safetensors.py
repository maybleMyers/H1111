"""
Repack split safetensors shards into a single safetensors file.

Usage:
    python repack_safetensors.py <input_dir> <output_file>

Example:
    python repack_safetensors.py ./weightsHumo/HuMo-17B ./humo_17b.safetensors
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

from safetensors import safe_open
from safetensors.torch import save_file


def repack_safetensors(input_dir: str, output_file: str):
    input_path = Path(input_dir)
    output_path = Path(output_file)

    # Find and load the index file
    index_files = list(input_path.glob("*.safetensors.index.json"))
    if not index_files:
        raise FileNotFoundError(f"No .safetensors.index.json file found in {input_dir}")

    index_file = index_files[0]
    print(f"Loading index from: {index_file}")

    with open(index_file, "r") as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    total_size = index.get("metadata", {}).get("total_size", "unknown")
    print(f"Total model size: {total_size / 1e9:.2f} GB" if isinstance(total_size, int) else f"Total model size: {total_size}")

    # Group tensors by their shard file
    shard_to_tensors = defaultdict(list)
    for tensor_name, shard_file in weight_map.items():
        shard_to_tensors[shard_file].append(tensor_name)

    print(f"Found {len(weight_map)} tensors across {len(shard_to_tensors)} shards")

    # Load all tensors from each shard
    all_tensors = {}
    for i, (shard_file, tensor_names) in enumerate(sorted(shard_to_tensors.items()), 1):
        shard_path = input_path / shard_file
        print(f"[{i}/{len(shard_to_tensors)}] Loading {shard_file} ({len(tensor_names)} tensors)...")

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for tensor_name in tensor_names:
                all_tensors[tensor_name] = f.get_tensor(tensor_name)

    print(f"\nLoaded {len(all_tensors)} tensors total")
    print(f"Saving to: {output_path}")

    # Save as single file
    save_file(all_tensors, output_path)

    output_size = output_path.stat().st_size
    print(f"Done! Output file size: {output_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Repack split safetensors shards into a single file"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing the split safetensors shards and index.json"
    )
    parser.add_argument(
        "output_file",
        help="Output path for the merged safetensors file"
    )

    args = parser.parse_args()
    repack_safetensors(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
