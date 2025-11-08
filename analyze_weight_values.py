#!/usr/bin/env python3
"""
Deep weight analysis - Compare actual weight values between models.

This script goes beyond key names to compare actual tensor values,
useful when models have been converted/renamed but share the same base weights.
"""

import argparse
import os
import sys
import numpy as np
from collections import defaultdict
import json

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("ERROR: safetensors required")
    sys.exit(1)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: torch required")
    sys.exit(1)


def load_model_weights(model_path, max_weights=None):
    """Load model weights into memory."""
    weights = {}

    if os.path.isdir(model_path):
        # Load sharded model
        index_file = os.path.join(model_path, "diffusion_pytorch_model.safetensors.index.json")
        if not os.path.exists(index_file):
            index_file = os.path.join(model_path, "model.safetensors.index.json")

        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index = json.load(f)
            weight_map = index.get('weight_map', {})
            shard_files = sorted(set(weight_map.values()))

            count = 0
            for shard_name in shard_files:
                shard_path = os.path.join(model_path, shard_name)
                print(f"  Loading {shard_name}...")

                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if max_weights and count >= max_weights:
                            print(f"  Stopping at {max_weights} weights (limit reached)")
                            return weights

                        tensor = f.get_tensor(key)
                        weights[key] = tensor.numpy()
                        count += 1
        else:
            print(f"ERROR: No index file found in {model_path}")
            return None
    else:
        # Single file
        print(f"  Loading {os.path.basename(model_path)}...")
        with safe_open(model_path, framework="pt", device="cpu") as f:
            count = 0
            for key in f.keys():
                if max_weights and count >= max_weights:
                    print(f"  Stopping at {max_weights} weights (limit reached)")
                    break

                tensor = f.get_tensor(key)
                weights[key] = tensor.numpy()
                count += 1

    return weights


def compute_weight_stats(weight_array):
    """Compute statistical summary of a weight tensor."""
    return {
        'shape': weight_array.shape,
        'mean': float(np.mean(weight_array)),
        'std': float(np.std(weight_array)),
        'min': float(np.min(weight_array)),
        'max': float(np.max(weight_array)),
        'l2_norm': float(np.linalg.norm(weight_array.flatten())),
        'num_params': int(np.prod(weight_array.shape)),
    }


def compare_weight_tensors(w1, w2, threshold=0.01):
    """Compare two weight tensors and return similarity metrics."""
    # Must have same shape
    if w1.shape != w2.shape:
        return None

    # Compute various similarity metrics
    diff = w1 - w2

    # Relative difference (L2 norm)
    l2_diff = np.linalg.norm(diff.flatten())
    l2_w1 = np.linalg.norm(w1.flatten())
    relative_l2_diff = l2_diff / (l2_w1 + 1e-10)

    # Mean absolute difference
    mae = np.mean(np.abs(diff))

    # Correlation
    w1_flat = w1.flatten()
    w2_flat = w2.flatten()
    correlation = np.corrcoef(w1_flat, w2_flat)[0, 1]

    # Cosine similarity
    cosine_sim = np.dot(w1_flat, w2_flat) / (np.linalg.norm(w1_flat) * np.linalg.norm(w2_flat) + 1e-10)

    return {
        'l2_diff': float(l2_diff),
        'relative_l2_diff': float(relative_l2_diff),
        'mae': float(mae),
        'correlation': float(correlation),
        'cosine_similarity': float(cosine_sim),
        'is_similar': relative_l2_diff < threshold and correlation > 0.95
    }


def find_matching_weights_by_shape(weights1, weights2):
    """Find potential weight matches based on shape."""
    print("\n" + "="*80)
    print("Finding Potential Weight Matches by Shape")
    print("="*80 + "\n")

    # Group weights by shape
    shapes1 = defaultdict(list)
    shapes2 = defaultdict(list)

    for key, weight in weights1.items():
        shapes1[weight.shape].append(key)

    for key, weight in weights2.items():
        shapes2[weight.shape].append(key)

    # Find common shapes
    common_shapes = set(shapes1.keys()) & set(shapes2.keys())

    print(f"Found {len(common_shapes)} common shapes")
    print(f"Unique shapes in model 1: {len(shapes1)}")
    print(f"Unique shapes in model 2: {len(shapes2)}")

    # Analyze common shapes
    shape_analysis = []
    for shape in sorted(common_shapes, key=lambda s: np.prod(s), reverse=True):
        count1 = len(shapes1[shape])
        count2 = len(shapes2[shape])

        shape_analysis.append({
            'shape': shape,
            'count_model1': count1,
            'count_model2': count2,
            'num_params': int(np.prod(shape)),
            'keys_model1': shapes1[shape][:5],  # Sample
            'keys_model2': shapes2[shape][:5],  # Sample
        })

    return shape_analysis, shapes1, shapes2


def analyze_weight_similarity(weights1, weights2, shape_analysis, shapes1, shapes2, sample_size=10):
    """Analyze actual weight value similarity for matching shapes."""
    print("\n" + "="*80)
    print("Analyzing Weight Value Similarity")
    print("="*80 + "\n")

    similarity_results = []

    for shape_info in shape_analysis[:sample_size]:  # Limit to top N shapes
        shape = shape_info['shape']
        keys1 = shapes1[shape]
        keys2 = shapes2[shape]

        print(f"\nShape: {shape} ({shape_info['num_params']:,} params)")
        print(f"  Model 1: {len(keys1)} weights")
        print(f"  Model 2: {len(keys2)} weights")

        # Compare each weight in model1 with all weights of same shape in model2
        for key1 in keys1[:3]:  # Sample first 3 weights
            w1 = weights1[key1]

            best_match = None
            best_similarity = -1

            for key2 in keys2[:min(5, len(keys2))]:  # Compare with first 5 in model2
                w2 = weights2[key2]

                similarity = compare_weight_tensors(w1, w2)
                if similarity and similarity['cosine_similarity'] > best_similarity:
                    best_similarity = similarity['cosine_similarity']
                    best_match = {
                        'key1': key1,
                        'key2': key2,
                        'similarity': similarity
                    }

            if best_match:
                sim = best_match['similarity']
                similarity_results.append(best_match)

                is_match = "✓ MATCH" if sim['is_similar'] else "✗ Different"
                print(f"\n  {is_match}")
                print(f"    {key1}")
                print(f"    → {best_match['key2']}")
                print(f"    Cosine similarity: {sim['cosine_similarity']:.6f}")
                print(f"    Correlation: {sim['correlation']:.6f}")
                print(f"    Relative L2 diff: {sim['relative_l2_diff']:.6f}")

    return similarity_results


def analyze_block_correspondence(weights1, weights2):
    """Try to find block-to-block correspondence."""
    print("\n" + "="*80)
    print("Analyzing Block Correspondence")
    print("="*80 + "\n")

    import re

    # Extract block numbers and weights
    blocks1 = defaultdict(dict)
    blocks2 = defaultdict(dict)

    for key, weight in weights1.items():
        match = re.search(r'blocks?[._](\d+)', key)
        if match:
            block_num = int(match.group(1))
            blocks1[block_num][key] = weight

    for key, weight in weights2.items():
        match = re.search(r'blocks?[._](\d+)', key)
        if match:
            block_num = int(match.group(1))
            blocks2[block_num][key] = weight

    print(f"Model 1: {len(blocks1)} blocks")
    print(f"Model 2: {len(blocks2)} blocks")

    # Compare first few blocks
    print("\nComparing first 3 blocks from each model:")

    for i in range(min(3, len(blocks1))):
        if i not in blocks1 or i not in blocks2:
            continue

        block1_keys = list(blocks1[i].keys())
        block2_keys = list(blocks2[i].keys())

        print(f"\n  Block {i}:")
        print(f"    Model 1: {len(block1_keys)} weights")
        print(f"    Model 2: {len(block2_keys)} weights")

        # Sample a few weights from each block
        sample_keys1 = block1_keys[:3]
        sample_keys2 = block2_keys[:3]

        print(f"    Sample keys from Model 1:")
        for key in sample_keys1:
            clean_key = key.replace(f'blocks.{i}.', 'block.N.')
            print(f"      {clean_key:60s} {weights1[key].shape}")

        print(f"    Sample keys from Model 2:")
        for key in sample_keys2:
            clean_key = key.replace(f'blocks.{i}.', 'block.N.')
            print(f"      {clean_key:60s} {weights2[key].shape}")

        # Try to find matches within this block
        matches = []
        for key1 in sample_keys1:
            w1 = weights1[key1]
            for key2 in sample_keys2:
                w2 = weights2[key2]
                if w1.shape == w2.shape:
                    sim = compare_weight_tensors(w1, w2)
                    if sim and sim['cosine_similarity'] > 0.95:
                        matches.append((key1, key2, sim))

        if matches:
            print(f"    Found {len(matches)} potential matches:")
            for key1, key2, sim in matches[:2]:
                print(f"      {key1.split('.')[-2:]} → {key2.split('.')[-2:]}")
                print(f"        Cosine: {sim['cosine_similarity']:.6f}")


def generate_detailed_report(weights1, weights2, shape_analysis, similarity_results, output_path):
    """Generate detailed analysis report."""
    print(f"\n{'='*80}")
    print(f"Generating Detailed Report: {output_path}")
    print(f"{'='*80}\n")

    report = {
        'model1_stats': {
            'total_weights': len(weights1),
            'total_params': sum(np.prod(w.shape) for w in weights1.values()),
        },
        'model2_stats': {
            'total_weights': len(weights2),
            'total_params': sum(np.prod(w.shape) for w in weights2.values()),
        },
        'shape_analysis': shape_analysis,
        'similarity_results': similarity_results,
    }

    # Compute summary statistics
    if similarity_results:
        cosine_sims = [r['similarity']['cosine_similarity'] for r in similarity_results]
        correlations = [r['similarity']['correlation'] for r in similarity_results]

        report['summary'] = {
            'matches_analyzed': len(similarity_results),
            'high_similarity_count': sum(1 for s in cosine_sims if s > 0.95),
            'medium_similarity_count': sum(1 for s in cosine_sims if 0.8 < s <= 0.95),
            'low_similarity_count': sum(1 for s in cosine_sims if s <= 0.8),
            'mean_cosine_similarity': float(np.mean(cosine_sims)),
            'mean_correlation': float(np.mean(correlations)),
        }

        print("Summary Statistics:")
        print(f"  Matches analyzed: {report['summary']['matches_analyzed']}")
        print(f"  High similarity (>0.95): {report['summary']['high_similarity_count']}")
        print(f"  Medium similarity (0.8-0.95): {report['summary']['medium_similarity_count']}")
        print(f"  Low similarity (<0.8): {report['summary']['low_similarity_count']}")
        print(f"  Mean cosine similarity: {report['summary']['mean_cosine_similarity']:.6f}")
        print(f"  Mean correlation: {report['summary']['mean_correlation']:.6f}")

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Deep weight analysis - compare actual weight values between models"
    )
    parser.add_argument("model1", help="Path to first model")
    parser.add_argument("model2", help="Path to second model")
    parser.add_argument("--names", nargs=2, help="Names for the models")
    parser.add_argument("--max-weights", type=int, default=None,
                       help="Maximum weights to load (for memory constraints)")
    parser.add_argument("--sample-size", type=int, default=10,
                       help="Number of shapes to analyze in detail")
    parser.add_argument("--output", type=str, default="weight_analysis_report.json",
                       help="Output report path")

    args = parser.parse_args()

    name1 = args.names[0] if args.names else "Model 1"
    name2 = args.names[1] if args.names else "Model 2"

    print("="*80)
    print("Deep Weight Value Analysis")
    print("="*80)
    print(f"\nModel 1: {name1}")
    print(f"Path: {args.model1}")
    print(f"\nModel 2: {name2}")
    print(f"Path: {args.model2}")

    if args.max_weights:
        print(f"\nMemory limit: Loading max {args.max_weights} weights per model")

    print("\n" + "="*80)
    print("Loading Model Weights")
    print("="*80 + "\n")

    print(f"Loading {name1}...")
    weights1 = load_model_weights(args.model1, args.max_weights)
    if weights1 is None:
        return 1
    print(f"✓ Loaded {len(weights1)} weights")

    print(f"\nLoading {name2}...")
    weights2 = load_model_weights(args.model2, args.max_weights)
    if weights2 is None:
        return 1
    print(f"✓ Loaded {len(weights2)} weights")

    # Analyze shapes
    shape_analysis, shapes1, shapes2 = find_matching_weights_by_shape(weights1, weights2)

    # Analyze weight similarity
    similarity_results = analyze_weight_similarity(
        weights1, weights2, shape_analysis, shapes1, shapes2,
        sample_size=args.sample_size
    )

    # Analyze block correspondence
    analyze_block_correspondence(weights1, weights2)

    # Generate report
    generate_detailed_report(weights1, weights2, shape_analysis, similarity_results, args.output)

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
