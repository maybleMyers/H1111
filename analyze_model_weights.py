#!/usr/bin/env python3
"""
Comprehensive model weight analysis and comparison tool.

This script analyzes model checkpoints to understand:
1. Weight naming conventions
2. Layer structures and dimensions
3. Architecture differences
4. Potential LoRA compatibility

Supports: .safetensors, .pth, .bin, .ckpt
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict, OrderedDict
import json

try:
    from safetensors import safe_open
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("WARNING: safetensors not available, will try torch only")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: torch is required")
    sys.exit(1)


class ModelAnalyzer:
    """Analyzes model checkpoint structure and weights."""

    def __init__(self, model_path: str, name: str = None):
        self.model_path = model_path
        self.name = name or os.path.basename(model_path)
        self.keys = []
        self.shapes = {}
        self.key_patterns = defaultdict(int)
        self.layer_info = defaultdict(dict)

    def load_checkpoint(self):
        """Load checkpoint and extract keys and shapes."""
        print(f"\n{'=' * 80}")
        print(f"Loading: {self.name}")
        print(f"Path: {self.model_path}")
        print(f"{'=' * 80}\n")

        # Check if path is a directory (diffusers sharded format)
        if os.path.isdir(self.model_path):
            return self._load_sharded_model(self.model_path)

        if not os.path.exists(self.model_path):
            print(f"ERROR: File not found: {self.model_path}")
            return False

        file_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")

        # Try loading as safetensors first
        if self.model_path.endswith('.safetensors') and SAFETENSORS_AVAILABLE:
            try:
                print("Loading as safetensors...")
                with safe_open(self.model_path, framework="pt", device="cpu") as f:
                    self.keys = list(f.keys())
                    for key in self.keys:
                        tensor = f.get_tensor(key)
                        self.shapes[key] = tuple(tensor.shape)
                print(f"✓ Loaded {len(self.keys)} keys from safetensors")
                return True
            except Exception as e:
                print(f"Failed to load as safetensors: {e}")

        # Try loading as PyTorch checkpoint
        try:
            print("Loading as PyTorch checkpoint...")
            checkpoint = torch.load(self.model_path, map_location='cpu')

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    # Assume the dict itself is the state dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            self.keys = list(state_dict.keys())
            for key, tensor in state_dict.items():
                if hasattr(tensor, 'shape'):
                    self.shapes[key] = tuple(tensor.shape)
                else:
                    self.shapes[key] = None

            print(f"✓ Loaded {len(self.keys)} keys from PyTorch checkpoint")
            return True

        except Exception as e:
            print(f"Failed to load as PyTorch checkpoint: {e}")
            return False

    def _load_sharded_model(self, model_dir: str):
        """Load sharded diffusers model from directory."""
        print(f"Detected directory - attempting to load sharded model...")

        # Look for model index file
        index_file = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(index_file):
            # Try alternative locations
            index_file = os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")

        if os.path.exists(index_file):
            return self._load_sharded_safetensors(model_dir, index_file)

        # Try loading all .safetensors files in directory
        shard_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.safetensors')])
        if shard_files:
            print(f"Found {len(shard_files)} shard files (no index)")
            return self._load_all_shards(model_dir, shard_files)

        print(f"ERROR: No model files found in directory: {model_dir}")
        return False

    def _load_sharded_safetensors(self, model_dir: str, index_file: str):
        """Load sharded safetensors using index file."""
        print(f"Loading sharded model using index: {os.path.basename(index_file)}")

        try:
            with open(index_file, 'r') as f:
                index = json.load(f)

            # Get weight map
            weight_map = index.get('weight_map', {})
            if not weight_map:
                print("ERROR: No weight_map in index file")
                return False

            # Get unique shard files
            shard_files = sorted(set(weight_map.values()))
            print(f"Found {len(shard_files)} shards: {shard_files[:3]}{'...' if len(shard_files) > 3 else ''}")

            # Calculate total size
            total_size = sum(
                os.path.getsize(os.path.join(model_dir, shard))
                for shard in shard_files
                if os.path.exists(os.path.join(model_dir, shard))
            )
            print(f"Total model size: {total_size / (1024**3):.2f} GB")

            # Load keys and shapes from each shard
            all_keys = []
            for shard_name in shard_files:
                shard_path = os.path.join(model_dir, shard_name)
                if not os.path.exists(shard_path):
                    print(f"WARNING: Shard not found: {shard_name}")
                    continue

                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                    for key in keys:
                        tensor = f.get_tensor(key)
                        self.shapes[key] = tuple(tensor.shape)
                        all_keys.append(key)

            self.keys = all_keys
            print(f"✓ Loaded {len(self.keys)} keys from {len(shard_files)} shards")
            return True

        except Exception as e:
            print(f"ERROR loading sharded model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_all_shards(self, model_dir: str, shard_files: list):
        """Load all shard files without index."""
        print(f"Loading {len(shard_files)} shards without index...")

        try:
            all_keys = []
            for shard_name in shard_files:
                shard_path = os.path.join(model_dir, shard_name)
                print(f"  Loading {shard_name}...")

                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                    for key in keys:
                        tensor = f.get_tensor(key)
                        self.shapes[key] = tuple(tensor.shape)
                        all_keys.append(key)

            self.keys = all_keys
            print(f"✓ Loaded {len(self.keys)} keys from {len(shard_files)} shards")
            return True

        except Exception as e:
            print(f"ERROR loading shards: {e}")
            return False

    def analyze_structure(self):
        """Analyze model structure and naming patterns."""
        print(f"\n{'=' * 80}")
        print(f"Analyzing Structure: {self.name}")
        print(f"{'=' * 80}\n")

        print(f"Total parameters: {len(self.keys)}")

        # Count parameter types
        weight_count = sum(1 for k in self.keys if 'weight' in k)
        bias_count = sum(1 for k in self.keys if 'bias' in k)
        norm_count = sum(1 for k in self.keys if 'norm' in k or 'ln' in k)

        print(f"\nParameter types:")
        print(f"  Weights: {weight_count}")
        print(f"  Biases: {bias_count}")
        print(f"  Normalization: {norm_count}")

        # Analyze naming patterns
        print(f"\nNaming patterns:")
        patterns = {
            'blocks': sum(1 for k in self.keys if 'block' in k.lower()),
            'layers': sum(1 for k in self.keys if 'layer' in k.lower()),
            'attention': sum(1 for k in self.keys if 'attn' in k.lower()),
            'cross_attention': sum(1 for k in self.keys if 'cross_attn' in k.lower()),
            'self_attention': sum(1 for k in self.keys if 'self_attn' in k.lower()),
            'qkv': sum(1 for k in self.keys if 'qkv' in k.lower()),
            'q_proj/linear': sum(1 for k in self.keys if 'q_proj' in k.lower() or 'q_linear' in k.lower() or '.q.' in k.lower()),
            'k_proj/linear': sum(1 for k in self.keys if 'k_proj' in k.lower() or 'k_linear' in k.lower() or '.k.' in k.lower()),
            'v_proj/linear': sum(1 for k in self.keys if 'v_proj' in k.lower() or 'v_linear' in k.lower() or '.v.' in k.lower()),
            'feedforward/mlp': sum(1 for k in self.keys if 'mlp' in k.lower() or 'ffn' in k.lower() or 'ff' in k.lower()),
            'projection': sum(1 for k in self.keys if 'proj' in k.lower()),
        }

        for pattern, count in patterns.items():
            if count > 0:
                print(f"  {pattern:20s}: {count:4d}")

        # Find block/layer structure
        self._analyze_block_structure()

        # Sample keys
        print(f"\nFirst 30 keys:")
        for i, key in enumerate(self.keys[:30]):
            shape_str = str(self.shapes.get(key, 'N/A'))
            print(f"  {i+1:3d}. {key:80s} {shape_str}")

        if len(self.keys) > 30:
            print(f"\n... and {len(self.keys) - 30} more keys")

        print(f"\nLast 20 keys:")
        for i, key in enumerate(self.keys[-20:]):
            shape_str = str(self.shapes.get(key, 'N/A'))
            print(f"  {len(self.keys)-19+i:3d}. {key:80s} {shape_str}")

    def _analyze_block_structure(self):
        """Analyze transformer block structure."""
        print(f"\nBlock/Layer Analysis:")

        # Find unique block indices
        import re
        block_numbers = set()

        # Try different patterns
        patterns = [
            r'blocks?[._](\d+)',
            r'layers?[._](\d+)',
            r'h[._](\d+)',
        ]

        for key in self.keys:
            for pattern in patterns:
                matches = re.findall(pattern, key, re.IGNORECASE)
                block_numbers.update(int(m) for m in matches)

        if block_numbers:
            min_block = min(block_numbers)
            max_block = max(block_numbers)
            num_blocks = len(block_numbers)
            print(f"  Found {num_blocks} blocks/layers")
            print(f"  Range: {min_block} to {max_block}")

            # Analyze first block structure
            first_block_keys = [k for k in self.keys if f'block.{min_block}' in k or f'blocks.{min_block}' in k or f'block_{min_block}' in k or f'blocks_{min_block}' in k]
            if not first_block_keys:
                first_block_keys = [k for k in self.keys if f'layer.{min_block}' in k or f'layers.{min_block}' in k or f'layer_{min_block}' in k or f'layers_{min_block}' in k]

            if first_block_keys:
                print(f"\n  First block (index {min_block}) structure ({len(first_block_keys)} keys):")
                for key in sorted(first_block_keys)[:20]:
                    shape_str = str(self.shapes.get(key, 'N/A'))
                    # Remove block number for cleaner display
                    clean_key = re.sub(r'blocks?[._]\d+[._]?', 'block.N.', key, flags=re.IGNORECASE)
                    clean_key = re.sub(r'layers?[._]\d+[._]?', 'layer.N.', clean_key, flags=re.IGNORECASE)
                    print(f"    {clean_key:60s} {shape_str}")
                if len(first_block_keys) > 20:
                    print(f"    ... and {len(first_block_keys) - 20} more keys")
        else:
            print("  No clear block/layer structure detected")

    def find_attention_structure(self):
        """Identify attention mechanism structure."""
        print(f"\n{'=' * 80}")
        print(f"Attention Structure Analysis: {self.name}")
        print(f"{'=' * 80}\n")

        # Find different attention patterns
        attention_patterns = {
            'Combined QKV': [],
            'Separate Q/K/V': [],
            'Q and combined KV': [],
            'Other attention': [],
        }

        for key in self.keys:
            key_lower = key.lower()
            if 'attn' in key_lower or 'attention' in key_lower:
                if 'qkv' in key_lower:
                    attention_patterns['Combined QKV'].append(key)
                elif any(x in key_lower for x in ['.q.', 'q_proj', 'q_linear', 'to_q']):
                    attention_patterns['Separate Q/K/V'].append(key)
                elif 'kv_linear' in key_lower or 'kv_proj' in key_lower:
                    attention_patterns['Q and combined KV'].append(key)
                else:
                    attention_patterns['Other attention'].append(key)

        for pattern_name, keys in attention_patterns.items():
            if keys:
                print(f"{pattern_name}: {len(keys)} keys")
                print(f"  Sample keys:")
                for key in keys[:5]:
                    shape_str = str(self.shapes.get(key, 'N/A'))
                    print(f"    {key:80s} {shape_str}")
                if len(keys) > 5:
                    print(f"    ... and {len(keys) - 5} more")
                print()

    def export_key_list(self, output_path: str):
        """Export full key list with shapes to file."""
        print(f"\nExporting key list to: {output_path}")

        with open(output_path, 'w') as f:
            f.write(f"Model: {self.name}\n")
            f.write(f"Path: {self.model_path}\n")
            f.write(f"Total keys: {len(self.keys)}\n")
            f.write("=" * 100 + "\n\n")

            for i, key in enumerate(self.keys):
                shape_str = str(self.shapes.get(key, 'N/A'))
                f.write(f"{i+1:5d}. {key:100s} {shape_str}\n")

        print(f"✓ Exported {len(self.keys)} keys")


class ModelComparator:
    """Compare two model structures."""

    def __init__(self, analyzer1: ModelAnalyzer, analyzer2: ModelAnalyzer):
        self.analyzer1 = analyzer1
        self.analyzer2 = analyzer2

    def compare(self):
        """Compare the two models."""
        print(f"\n{'#' * 80}")
        print(f"# COMPARING MODELS")
        print(f"# Model 1: {self.analyzer1.name}")
        print(f"# Model 2: {self.analyzer2.name}")
        print(f"{'#' * 80}\n")

        keys1 = set(self.analyzer1.keys)
        keys2 = set(self.analyzer2.keys)

        print(f"Model 1 keys: {len(keys1)}")
        print(f"Model 2 keys: {len(keys2)}")
        print(f"Common keys: {len(keys1 & keys2)}")
        print(f"Only in Model 1: {len(keys1 - keys2)}")
        print(f"Only in Model 2: {len(keys2 - keys1)}")

        if len(keys1 & keys2) > 0:
            overlap_pct = 100 * len(keys1 & keys2) / max(len(keys1), len(keys2))
            print(f"Overlap: {overlap_pct:.1f}%")

        # Exact match analysis
        if keys1 == keys2:
            print("\n✓ EXACT KEY MATCH - Models have identical structure!")
            self._compare_shapes()
        else:
            print("\n✗ Keys do NOT match exactly")
            self._analyze_key_differences(keys1, keys2)
            self._attempt_key_mapping()

    def _compare_shapes(self):
        """Compare tensor shapes for matching keys."""
        print(f"\n{'=' * 80}")
        print("Shape Comparison (for matching keys)")
        print(f"{'=' * 80}\n")

        matching_shapes = 0
        different_shapes = []

        for key in self.analyzer1.keys:
            if key in self.analyzer2.keys:
                shape1 = self.analyzer1.shapes.get(key)
                shape2 = self.analyzer2.shapes.get(key)

                if shape1 == shape2:
                    matching_shapes += 1
                else:
                    different_shapes.append((key, shape1, shape2))

        print(f"Matching shapes: {matching_shapes}/{len(self.analyzer1.keys)}")

        if different_shapes:
            print(f"\nDifferent shapes: {len(different_shapes)}")
            print("Sample differences:")
            for key, shape1, shape2 in different_shapes[:10]:
                print(f"  {key}")
                print(f"    Model 1: {shape1}")
                print(f"    Model 2: {shape2}")

    def _analyze_key_differences(self, keys1: set, keys2: set):
        """Analyze what's different between key sets."""
        print(f"\n{'=' * 80}")
        print("Key Differences Analysis")
        print(f"{'=' * 80}\n")

        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1

        if only_in_1:
            print(f"Keys only in {self.analyzer1.name} (first 20):")
            for i, key in enumerate(sorted(only_in_1)[:20]):
                shape_str = str(self.analyzer1.shapes.get(key, 'N/A'))
                print(f"  {i+1:3d}. {key:80s} {shape_str}")
            if len(only_in_1) > 20:
                print(f"  ... and {len(only_in_1) - 20} more")

        if only_in_2:
            print(f"\nKeys only in {self.analyzer2.name} (first 20):")
            for i, key in enumerate(sorted(only_in_2)[:20]):
                shape_str = str(self.analyzer2.shapes.get(key, 'N/A'))
                print(f"  {i+1:3d}. {key:80s} {shape_str}")
            if len(only_in_2) > 20:
                print(f"  ... and {len(only_in_2) - 20} more")

    def _attempt_key_mapping(self):
        """Attempt to find key mappings between models."""
        print(f"\n{'=' * 80}")
        print("Attempting Key Mapping")
        print(f"{'=' * 80}\n")

        # Try to find patterns in key naming
        keys1 = self.analyzer1.keys
        keys2 = self.analyzer2.keys

        # Strategy: Look for similar shapes and positions
        print("Strategy 1: Matching by shape and position")
        shape_matches = defaultdict(list)

        for key1 in keys1[:100]:  # Sample first 100 keys
            shape1 = self.analyzer1.shapes.get(key1)
            if shape1:
                for key2 in keys2:
                    shape2 = self.analyzer2.shapes.get(key2)
                    if shape1 == shape2:
                        # Check if key names have similar structure
                        similarity = self._compute_key_similarity(key1, key2)
                        if similarity > 0.3:  # Threshold
                            shape_matches[key1].append((key2, similarity, shape1))

        if shape_matches:
            print(f"Found {len(shape_matches)} potential mappings (sample):")
            for i, (key1, matches) in enumerate(list(shape_matches.items())[:10]):
                matches.sort(key=lambda x: x[1], reverse=True)
                best_match = matches[0]
                print(f"\n  {i+1}. {key1}")
                print(f"     → {best_match[0]}")
                print(f"     Similarity: {best_match[1]:.2f}, Shape: {best_match[2]}")
        else:
            print("No obvious key mappings found")

        # Strategy 2: Identify naming convention differences
        print("\n\nStrategy 2: Naming convention analysis")
        self._analyze_naming_conventions()

    def _compute_key_similarity(self, key1: str, key2: str) -> float:
        """Compute similarity between two key names."""
        # Normalize keys
        k1_parts = set(key1.lower().replace('_', '.').split('.'))
        k2_parts = set(key2.lower().replace('_', '.').split('.'))

        # Remove numeric parts
        k1_parts = {p for p in k1_parts if not p.isdigit()}
        k2_parts = {p for p in k2_parts if not p.isdigit()}

        if not k1_parts or not k2_parts:
            return 0.0

        # Jaccard similarity
        intersection = len(k1_parts & k2_parts)
        union = len(k1_parts | k2_parts)

        return intersection / union if union > 0 else 0.0

    def _analyze_naming_conventions(self):
        """Analyze naming convention patterns."""
        print("\nNaming conventions:")

        conventions1 = self._detect_conventions(self.analyzer1.keys)
        conventions2 = self._detect_conventions(self.analyzer2.keys)

        print(f"\n{self.analyzer1.name}:")
        for conv, count in conventions1.items():
            print(f"  {conv:30s}: {count:4d} keys")

        print(f"\n{self.analyzer2.name}:")
        for conv, count in conventions2.items():
            print(f"  {conv:30s}: {count:4d} keys")

    def _detect_conventions(self, keys: list) -> dict:
        """Detect naming conventions in keys."""
        conventions = {
            'dot_separated': sum(1 for k in keys if '.' in k),
            'underscore_separated': sum(1 for k in keys if '_' in k),
            'mixed_separation': sum(1 for k in keys if '.' in k and '_' in k),
            'camelCase': sum(1 for k in keys if any(c.isupper() for c in k[1:])),
            'has_numbers': sum(1 for k in keys if any(c.isdigit() for c in k)),
        }
        return conventions

    def generate_mapping_file(self, output_path: str):
        """Generate a potential mapping file between models."""
        print(f"\n{'=' * 80}")
        print(f"Generating Mapping File: {output_path}")
        print(f"{'=' * 80}\n")

        mappings = []

        # Try to find shape-based mappings
        for key1 in self.analyzer1.keys:
            shape1 = self.analyzer1.shapes.get(key1)
            if shape1:
                candidates = []
                for key2 in self.analyzer2.keys:
                    shape2 = self.analyzer2.shapes.get(key2)
                    if shape1 == shape2:
                        similarity = self._compute_key_similarity(key1, key2)
                        candidates.append((key2, similarity))

                if candidates:
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    best_match = candidates[0]
                    mappings.append({
                        'source': key1,
                        'target': best_match[0],
                        'similarity': best_match[1],
                        'shape': str(shape1),
                        'confidence': 'high' if best_match[1] > 0.7 else 'medium' if best_match[1] > 0.4 else 'low'
                    })

        # Write to file
        with open(output_path, 'w') as f:
            json.dump({
                'source_model': self.analyzer1.name,
                'target_model': self.analyzer2.name,
                'total_mappings': len(mappings),
                'mappings': mappings
            }, f, indent=2)

        print(f"✓ Generated {len(mappings)} potential mappings")

        # Summary
        high_conf = sum(1 for m in mappings if m['confidence'] == 'high')
        med_conf = sum(1 for m in mappings if m['confidence'] == 'medium')
        low_conf = sum(1 for m in mappings if m['confidence'] == 'low')

        print(f"\nConfidence breakdown:")
        print(f"  High confidence: {high_conf}")
        print(f"  Medium confidence: {med_conf}")
        print(f"  Low confidence: {low_conf}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare model checkpoint structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single model
  python analyze_model_weights.py model.safetensors

  # Compare two models
  python analyze_model_weights.py model1.safetensors model2.safetensors --compare

  # Export key lists
  python analyze_model_weights.py model.safetensors --export model_keys.txt

  # Generate mapping file
  python analyze_model_weights.py model1.safetensors model2.safetensors --compare --mapping mapping.json
        """
    )

    parser.add_argument(
        "model_paths",
        nargs="+",
        help="Path(s) to model checkpoint file(s)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare two models (requires exactly 2 model paths)"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export key list(s) to file(s)"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        help="Generate mapping file between models (requires --compare)"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="Custom names for models (optional)"
    )

    args = parser.parse_args()

    if args.compare and len(args.model_paths) != 2:
        print("ERROR: --compare requires exactly 2 model paths")
        return 1

    if args.mapping and not args.compare:
        print("ERROR: --mapping requires --compare")
        return 1

    # Analyze models
    analyzers = []
    for i, model_path in enumerate(args.model_paths):
        name = args.names[i] if args.names and i < len(args.names) else None
        analyzer = ModelAnalyzer(model_path, name)

        if not analyzer.load_checkpoint():
            return 1

        analyzer.analyze_structure()
        analyzer.find_attention_structure()

        analyzers.append(analyzer)

        # Export if requested
        if args.export:
            if len(args.model_paths) == 1:
                export_path = args.export
            else:
                base, ext = os.path.splitext(args.export)
                export_path = f"{base}_{i+1}{ext}"
            analyzer.export_key_list(export_path)

    # Compare if requested
    if args.compare and len(analyzers) == 2:
        comparator = ModelComparator(analyzers[0], analyzers[1])
        comparator.compare()

        if args.mapping:
            comparator.generate_mapping_file(args.mapping)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
