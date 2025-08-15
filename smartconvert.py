#!/usr/bin/env python3
"""
Wan2.2 Advanced FP16 Converter
Intelligent mixed precision converter with sensitivity analysis
"""

import torch
import json
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import logging
from tqdm import tqdm
import argparse
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Precision constants
FP16_MAX = 65504.0
FP16_MIN = -65504.0
FP16_MIN_POSITIVE = 6.103515625e-05
FP16_EPSILON = 0.000977
BF16_MIN_POSITIVE = 1.175494e-38


@dataclass
class LayerImportanceProfile:
    """Profile for layer importance in model behavior"""
    # Critical layers that affect prompt adherence
    critical_attention_patterns: List[str] = field(default_factory=lambda: [
        'cross_attn',  # Cross-attention is crucial for conditioning
        'self_attn.q', 'self_attn.k', 'self_attn.v',  # Query/Key/Value
        'modulation',  # Adaptive normalization for conditioning
        'time_embed',  # Temporal conditioning
        'context_embed',  # Text/image conditioning
    ])
    
    # Layers that can tolerate reduced precision
    safe_reduction_patterns: List[str] = field(default_factory=lambda: [
        'ffn.0.weight', 'ffn.2.weight',  # FFN weights (not biases)
        'proj_out',  # Output projections in middle layers
    ])
    
    # Always keep in FP32
    fp32_required_patterns: List[str] = field(default_factory=lambda: [
        'norm', 'ln_', 'bias',  # All normalization and biases
        'embedder', 'head.',  # Input/output layers
        'pos_frequencies', 'positional_embedding',
        'scale', 'shift',  # Scaling parameters
    ])


@dataclass
class AdvancedConversionConfig:
    """Advanced configuration for intelligent mixed precision"""
    # Sensitivity thresholds
    gradient_sensitivity_threshold: float = 0.1  # For importance analysis
    activation_range_threshold: float = 10.0  # For dynamic range analysis
    
    # Safety margins
    fp16_safety_factor: float = 0.5  # Use only 50% of FP16 range
    outlier_percentile: float = 99.9  # Check 99.9th percentile values
    
    # Precision allocation
    max_fp16_percentage: float = 0.6  # Max 60% of params in FP16
    prefer_bf16_over_fp16: bool = True  # Use BF16 when FP16 is risky
    
    # Analysis options
    analyze_activation_patterns: bool = True
    analyze_gradient_flow: bool = True
    generate_precision_map: bool = True
    
    # Layer importance profile
    importance_profile: LayerImportanceProfile = field(default_factory=LayerImportanceProfile)


class AdvancedFP16Converter:
    """Advanced converter with intelligent precision allocation"""
    
    def __init__(self, model_path: str, config: Optional[AdvancedConversionConfig] = None):
        self.model_path = Path(model_path)
        self.config = config or AdvancedConversionConfig()
        self.tensor_analysis = {}
        self.precision_decisions = {}
        self.conversion_stats = defaultdict(lambda: {'count': 0, 'params': 0})
        
    def analyze_tensor_sensitivity(self, name: str, tensor: torch.Tensor) -> Dict[str, float]:
        """Analyze tensor sensitivity to precision reduction"""
        analysis = {
            'importance_score': 0.0,
            'range_score': 0.0,
            'sparsity_score': 0.0,
            'distribution_score': 0.0,
            'outlier_score': 0.0
        }

        # Convert to float for analysis
        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        # 1. Layer importance score based on name patterns
        name_lower = name.lower()
        for pattern in self.config.importance_profile.critical_attention_patterns:
            if pattern in name_lower:
                analysis['importance_score'] = 1.0
                break

        for pattern in self.config.importance_profile.safe_reduction_patterns:
            if pattern in name_lower:
                analysis['importance_score'] = 0.2
                break

        # 2. Dynamic range analysis
        abs_vals = tensor.abs()
        max_val = abs_vals.max().item()

        if max_val > 0:
            # For large tensors, use sampling to avoid memory issues
            numel = abs_vals.numel()
            if numel > 10_000_000:  # 10M elements
                # Sample 1M elements randomly
                sample_size = min(1_000_000, numel)
                indices = torch.randperm(numel)[:sample_size]
                flat_tensor = abs_vals.flatten()
                sampled_vals = flat_tensor[indices]
                sampled_vals = sampled_vals[sampled_vals > 0]

                if len(sampled_vals) > 0:
                    # Use sampled values for quantiles
                    p99_val = torch.quantile(sampled_vals, 0.99).item()
                    p50_val = torch.quantile(sampled_vals, 0.50).item()
                else:
                    p99_val = p50_val = 0
            else:
                # Original code for smaller tensors
                nonzero_vals = abs_vals[abs_vals > 0]
                if nonzero_vals.numel() > 0:
                    p99_val = torch.quantile(nonzero_vals, 0.99).item()
                    p50_val = torch.quantile(nonzero_vals, 0.50).item()
                else:
                    p99_val = p50_val = 0

            if p50_val > 0:
                range_ratio = p99_val / (p50_val + 1e-8)
                analysis['range_score'] = min(range_ratio / 100, 1.0)  # Normalize

        # 3. Sparsity analysis
        zero_ratio = (tensor == 0).sum().item() / tensor.numel()
        analysis['sparsity_score'] = zero_ratio

        # 4. Distribution analysis (how "normal" is the distribution)
        if tensor.numel() > 100:
            # For large tensors, sample for distribution analysis
            if tensor.numel() > 10_000_000:
                sample_size = min(100_000, tensor.numel())
                indices = torch.randperm(tensor.numel())[:sample_size]
                sample_tensor = tensor.flatten()[indices]
            else:
                sample_tensor = tensor

            # Check if distribution is roughly normal
            mean = sample_tensor.mean().item()
            std = sample_tensor.std().item()

            if std > 0:
                # Compute skewness approximation
                centered = (sample_tensor - mean) / std
                skewness = centered.pow(3).mean().abs().item()
                analysis['distribution_score'] = 1.0 / (1.0 + skewness)

        # 5. Outlier analysis
        if max_val > self.config.fp16_safety_factor * FP16_MAX:
            analysis['outlier_score'] = 1.0
        elif max_val > 0:
            # For outlier percentile, use sampling for large tensors
            if abs_vals.numel() > 10_000_000:
                sample_size = min(1_000_000, abs_vals.numel())
                indices = torch.randperm(abs_vals.numel())[:sample_size]
                sampled_vals = abs_vals.flatten()[indices]
                percentile_val = torch.quantile(sampled_vals, 
                                              self.config.outlier_percentile / 100).item()
            else:
                percentile_val = torch.quantile(abs_vals, 
                                              self.config.outlier_percentile / 100).item()

            if percentile_val > self.config.fp16_safety_factor * FP16_MAX:
                analysis['outlier_score'] = 0.5

        return analysis
    
    def compute_composite_score(self, analysis: Dict[str, float]) -> float:
        """Compute composite score for precision decision"""
        # Higher score = needs higher precision
        weights = {
            'importance_score': 0.4,
            'range_score': 0.2,
            'sparsity_score': -0.1,  # Negative because sparse can use lower precision
            'distribution_score': -0.1,  # Normal distributions are safer
            'outlier_score': 0.4
        }
        
        score = sum(analysis[key] * weight for key, weight in weights.items())
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def determine_optimal_precision(self, name: str, tensor: torch.Tensor, 
                                  analysis: Dict[str, float]) -> Tuple[torch.dtype, str]:
        """Determine optimal precision based on comprehensive analysis"""
        name_lower = name.lower()
        
        # Step 1: Check mandatory FP32 patterns
        for pattern in self.config.importance_profile.fp32_required_patterns:
            if pattern in name_lower:
                return torch.float32, f"Mandatory FP32 pattern: {pattern}"
        
        # Step 2: Compute composite score
        composite_score = self.compute_composite_score(analysis)
        
        # Step 3: Check value ranges
        abs_max = tensor.abs().max().item()
        
        # Step 4: Make precision decision
        if composite_score > 0.7 or abs_max > FP16_MAX:
            return torch.float32, f"High importance/range (score={composite_score:.2f})"
        
        elif composite_score > 0.4:
            if self.config.prefer_bf16_over_fp16 and abs_max > self.config.fp16_safety_factor * FP16_MAX:
                return torch.bfloat16, f"Medium importance, BF16 for range safety"
            else:
                # Additional safety check for FP16
                if abs_max < self.config.fp16_safety_factor * FP16_MAX:
                    # Check for underflow risk
                    nonzero_mask = tensor != 0
                    if nonzero_mask.any():
                        min_nonzero = tensor[nonzero_mask].abs().min().item()
                        if min_nonzero < FP16_MIN_POSITIVE * 10:
                            return torch.bfloat16, "Underflow risk for FP16"
                    return torch.float16, f"Medium importance, safe for FP16"
                else:
                    return torch.bfloat16, f"Medium importance, range concerns"
        
        else:
            # Low importance - can use FP16 if within range
            if abs_max < self.config.fp16_safety_factor * FP16_MAX:
                return torch.float16, f"Low importance (score={composite_score:.2f})"
            else:
                return torch.bfloat16, f"Low importance but range exceeds FP16"
    
    def convert_expert_advanced(self, expert_name: str) -> Path:
        """Convert expert with advanced precision analysis"""
        expert_path = self.model_path / expert_name
        index_file = expert_path / "diffusion_pytorch_model.safetensors.index.json"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        logger.info(f"Starting advanced conversion for {expert_name}...")
        
        # Load index
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data["weight_map"]
        metadata = {k: str(v) for k, v in index_data.get("metadata", {}).items()}
        metadata.update({
            'precision': 'advanced_mixed',
            'converter': 'AdvancedFP16Converter',
            'fp16_safety_factor': str(self.config.fp16_safety_factor)
        })
        
        # Phase 1: Analyze all tensors
        logger.info("Phase 1: Analyzing tensor characteristics...")
        shard_files = sorted(set(weight_map.values()))
        
        for shard_file in tqdm(shard_files, desc="Analyzing"):
            shard_path = expert_path / shard_file
            
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                shard_tensors = [k for k, v in weight_map.items() if v == shard_file]
                
                for tensor_name in shard_tensors:
                    tensor = f.get_tensor(tensor_name)
                    analysis = self.analyze_tensor_sensitivity(tensor_name, tensor)
                    self.tensor_analysis[tensor_name] = analysis
        
        # Phase 2: Global optimization
        logger.info("Phase 2: Optimizing precision allocation...")
        self._optimize_global_precision_allocation()
        
        # Phase 3: Convert tensors
        logger.info("Phase 3: Converting tensors...")
        converted_tensors = {}
        
        for shard_file in tqdm(shard_files, desc="Converting"):
            shard_path = expert_path / shard_file
            
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                shard_tensors = [k for k, v in weight_map.items() if v == shard_file]
                
                for tensor_name in shard_tensors:
                    tensor = f.get_tensor(tensor_name)
                    
                    # Get optimized precision decision
                    if tensor_name in self.precision_decisions:
                        target_dtype, reason = self.precision_decisions[tensor_name]
                    else:
                        # Fallback to analysis-based decision
                        analysis = self.tensor_analysis.get(tensor_name, {})
                        target_dtype, reason = self.determine_optimal_precision(
                            tensor_name, tensor, analysis)
                    
                    # Convert tensor
                    if target_dtype != tensor.dtype:
                        tensor = tensor.to(target_dtype)
                    
                    converted_tensors[tensor_name] = tensor
                    
                    # Update statistics
                    self._update_stats(target_dtype, tensor.numel())
        
        # Save converted model
        output_file = self.model_path / f"{expert_name}_advanced_mixed.safetensors"
        logger.info(f"Saving converted model to {output_file}")
        save_file(converted_tensors, output_file, metadata=metadata)
        
        # Save detailed analysis
        self._save_conversion_analysis(expert_name)
        
        # Log results
        self._log_conversion_results(expert_name, shard_files, output_file)
        
        return output_file
    
    def _optimize_global_precision_allocation(self):
        """Optimize precision allocation across all tensors"""
        # Sort tensors by composite score
        tensor_scores = []
        
        for name, analysis in self.tensor_analysis.items():
            score = self.compute_composite_score(analysis)
            tensor_scores.append((name, score, analysis))
        
        tensor_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate precision based on importance
        total_params = 0
        fp16_params = 0
        
        total_params = len(tensor_scores)
        
        # Second pass: allocate precision
        for name, score, analysis in tensor_scores:
            # Check if we've exceeded FP16 budget
            if fp16_params / (total_params + 1) > self.config.max_fp16_percentage:
                # Force higher precision for remaining tensors
                if score < 0.5:  # Only override low-importance tensors
                    self.precision_decisions[name] = (torch.bfloat16, 
                                                     "FP16 budget exceeded")
    
    def _update_stats(self, dtype: torch.dtype, num_params: int):
        """Update conversion statistics"""
        dtype_str = str(dtype).replace('torch.', '')
        
        if 'float16' in dtype_str:
            self.conversion_stats['fp16']['count'] += 1
            self.conversion_stats['fp16']['params'] += num_params
        elif 'bfloat16' in dtype_str:
            self.conversion_stats['bf16']['count'] += 1
            self.conversion_stats['bf16']['params'] += num_params
        else:
            self.conversion_stats['fp32']['count'] += 1
            self.conversion_stats['fp32']['params'] += num_params
    
    def _save_conversion_analysis(self, expert_name: str):
        """Save detailed conversion analysis"""
        analysis_file = self.model_path / f"{expert_name}_precision_analysis.json"
        
        analysis_data = {
            'tensor_analysis': self.tensor_analysis,
            'precision_decisions': {
                name: {'dtype': str(dtype), 'reason': reason}
                for name, (dtype, reason) in self.precision_decisions.items()
            },
            'statistics': dict(self.conversion_stats)
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        # Generate visualization
        self._generate_precision_heatmap(expert_name)
    
    def _generate_precision_heatmap(self, expert_name: str):
        """Generate heatmap of precision decisions"""
        try:
            # Group by layer type
            layer_precisions = defaultdict(lambda: {'fp16': 0, 'bf16': 0, 'fp32': 0})

            # Check if we have precision decisions
            if not self.precision_decisions:
                logger.warning("No precision decisions to visualize")
                return

            for name, (dtype, _) in self.precision_decisions.items():
                layer_type = self._extract_layer_type(name)
                dtype_str = str(dtype).replace('torch.', '')

                if 'float16' in dtype_str:
                    layer_precisions[layer_type]['fp16'] += 1
                elif 'bfloat16' in dtype_str:
                    layer_precisions[layer_type]['bf16'] += 1
                else:
                    layer_precisions[layer_type]['fp32'] += 1

            # Check if we have any data
            if not layer_precisions:
                logger.warning("No layer precision data to visualize")
                return

            # Create heatmap data
            layer_types = sorted(layer_precisions.keys())
            precision_types = ['fp16', 'bf16', 'fp32']

            data = np.zeros((len(layer_types), len(precision_types)))

            for i, layer in enumerate(layer_types):
                total = sum(layer_precisions[layer].values())
                if total > 0:
                    for j, prec in enumerate(precision_types):
                        data[i, j] = layer_precisions[layer][prec] / total * 100

            # Skip if data is empty
            if data.size == 0 or data.shape[0] == 0:
                logger.warning("Empty data array for heatmap")
                return

            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(data, 
                       xticklabels=precision_types,
                       yticklabels=layer_types,
                       annot=True,
                       fmt='.1f',
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Percentage'})

            plt.title(f'Precision Distribution by Layer Type - {expert_name}')
            plt.tight_layout()
            plt.savefig(self.model_path / f"{expert_name}_precision_heatmap.png", dpi=300)
            plt.close()

        except Exception as e:
            logger.warning(f"Failed to generate precision heatmap: {e}")
            # Continue without failing the conversion
    
    def _extract_layer_type(self, tensor_name: str) -> str:
        """Extract layer type from tensor name"""
        if 'self_attn' in tensor_name:
            return 'self_attention'
        elif 'cross_attn' in tensor_name:
            return 'cross_attention'
        elif 'ffn' in tensor_name:
            return 'feedforward'
        elif 'norm' in tensor_name:
            return 'normalization'
        elif 'modulation' in tensor_name:
            return 'modulation'
        elif 'embed' in tensor_name:
            return 'embedding'
        else:
            parts = tensor_name.split('.')
            return parts[1] if len(parts) > 1 else 'other'
    
    def _log_conversion_results(self, expert_name: str, shard_files: List[str], 
                               output_file: Path):
        """Log detailed conversion results"""
        total_params = sum(stat['params'] for stat in self.conversion_stats.values())
        original_size = sum((self.model_path / expert_name / f).stat().st_size 
                          for f in shard_files) / (1024**3)
        final_size = output_file.stat().st_size / (1024**3)
        
        logger.info(f"\nConversion Results for {expert_name}:")
        logger.info(f"  Original size: {original_size:.2f} GB")
        logger.info(f"  Final size: {final_size:.2f} GB")
        logger.info(f"  Size reduction: {(1 - final_size/original_size)*100:.1f}%")
        logger.info(f"\nPrecision Distribution:")
        
        for dtype, stats in self.conversion_stats.items():
            if stats['count'] > 0:
                logger.info(f"  {dtype.upper()}: {stats['count']} tensors, "
                          f"{stats['params']:,} params "
                          f"({stats['params']/total_params*100:.1f}%)")
        
        # Memory calculation
        memory_bytes = (
            self.conversion_stats['fp16']['params'] * 2 +
            self.conversion_stats['bf16']['params'] * 2 +
            self.conversion_stats['fp32']['params'] * 4
        )
        
        logger.info(f"\nExpected memory usage: {memory_bytes / (1024**3):.2f} GB")
    
    def validate_advanced_conversion(self, original_expert: str, converted_file: Path,
                                   num_samples: int = 50) -> Dict[str, Any]:
        """Comprehensive validation of advanced conversion"""
        logger.info("Performing advanced validation...")
        
        validation_results = {
            'overall_quality': 'Unknown',
            'precision_errors': {},
            'layer_analysis': {},
            'recommendations': []
        }
        
        # Load precision decisions
        analysis_file = converted_file.parent / f"{converted_file.stem.replace('_advanced_mixed', '')}_precision_analysis.json"
        with open(analysis_file, 'r') as f:
            analysis_data = json.load(f)
        
        precision_decisions = analysis_data['precision_decisions']
        
        # Sample validation
        sample_keys = list(precision_decisions.keys())[:num_samples]
        
        # Categorized errors
        errors_by_precision = {'fp16': [], 'bf16': [], 'fp32': []}
        
        # Load original model index
        original_path = self.model_path / original_expert
        index_file = original_path / "diffusion_pytorch_model.safetensors.index.json"
        with open(index_file, 'r') as f:
            weight_map = json.load(f)["weight_map"]
        
        with safe_open(converted_file, framework="pt") as converted:
            for key in tqdm(sample_keys, desc="Validating"):
                if key not in weight_map:
                    continue
                
                # Load tensors
                shard_file = weight_map[key]
                with safe_open(original_path / shard_file, framework="pt") as original:
                    orig_tensor = original.get_tensor(key).float()
                    conv_tensor = converted.get_tensor(key).float()
                    
                    # Calculate errors
                    abs_diff = (orig_tensor - conv_tensor).abs()
                    rel_error = abs_diff / (orig_tensor.abs() + 1e-8)
                    
                    max_rel_error = rel_error.max().item()
                    mean_rel_error = rel_error.mean().item()
                    
                    # Categorize by precision
                    dtype_str = precision_decisions[key]['dtype']
                    
                    error_info = {
                        'tensor': key,
                        'max_error': max_rel_error,
                        'mean_error': mean_rel_error,
                        'layer_type': self._extract_layer_type(key)
                    }
                    
                    if 'float16' in dtype_str:
                        errors_by_precision['fp16'].append(error_info)
                    elif 'bfloat16' in dtype_str:
                        errors_by_precision['bf16'].append(error_info)
                    else:
                        errors_by_precision['fp32'].append(error_info)
        
        # Analyze results
        for precision, errors in errors_by_precision.items():
            if errors:
                max_errors = [e['max_error'] for e in errors]
                mean_errors = [e['mean_error'] for e in errors]
                
                validation_results['precision_errors'][precision] = {
                    'count': len(errors),
                    'max_error': max(max_errors),
                    'avg_max_error': np.mean(max_errors),
                    'avg_mean_error': np.mean(mean_errors)
                }
                
                # Layer-specific analysis
                layer_errors = defaultdict(list)
                for e in errors:
                    layer_errors[e['layer_type']].append(e['max_error'])
                
                for layer, errs in layer_errors.items():
                    if layer not in validation_results['layer_analysis']:
                        validation_results['layer_analysis'][layer] = {}
                    
                    validation_results['layer_analysis'][layer][precision] = {
                        'avg_error': np.mean(errs),
                        'max_error': max(errs)
                    }
        
        # Overall quality assessment
        fp16_avg_error = validation_results['precision_errors'].get('fp16', {}).get('avg_max_error', 0)
        
        if fp16_avg_error < 0.01:
            validation_results['overall_quality'] = 'Excellent'
        elif fp16_avg_error < 0.05:
            validation_results['overall_quality'] = 'Good'
            validation_results['recommendations'].append(
                "Consider moving high-error FP16 tensors to BF16")
        elif fp16_avg_error < 0.1:
            validation_results['overall_quality'] = 'Acceptable'
            validation_results['recommendations'].append(
                "Review FP16 assignments for critical layers")
        else:
            validation_results['overall_quality'] = 'Poor'
            validation_results['recommendations'].append(
                "FP16 errors too high - increase safety margins or use more BF16/FP32")
        
        # Log summary
        logger.info(f"\nValidation Summary:")
        logger.info(f"  Overall Quality: {validation_results['overall_quality']}")
        
        for precision, stats in validation_results['precision_errors'].items():
            logger.info(f"  {precision.upper()}: avg_max_error={stats['avg_max_error']:.6f}")
        
        if validation_results['recommendations']:
            logger.info("\nRecommendations:")
            for rec in validation_results['recommendations']:
                logger.info(f"  - {rec}")
        
        # Save validation report
        report_file = self.model_path / f"{original_expert}_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        return validation_results


def main():
    parser = argparse.ArgumentParser(
        description="Advanced FP16 converter for Wan2.2 with sensitivity analysis"
    )
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--safety-factor", type=float, default=0.5,
                       help="FP16 safety factor (0-1, default 0.5)")
    parser.add_argument("--max-fp16", type=float, default=0.6,
                       help="Maximum percentage of params in FP16 (default 0.6)")
    parser.add_argument("--no-bf16", action="store_true",
                       help="Disable BF16 usage")
    parser.add_argument("--validate", action="store_true",
                       help="Run validation after conversion")
    
    args = parser.parse_args()
    
    # Configure converter
    config = AdvancedConversionConfig(
        fp16_safety_factor=args.safety_factor,
        max_fp16_percentage=args.max_fp16,
        prefer_bf16_over_fp16=not args.no_bf16
    )
    
    converter = AdvancedFP16Converter(args.model_path, config)
    
    # Convert models
    experts = ["high_noise_model", "low_noise_model"]
    
    for expert in experts:
        expert_path = converter.model_path / expert
        if expert_path.exists():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {expert}")
            logger.info(f"{'='*60}")
            
            output_file = converter.convert_expert_advanced(expert)
            
            if args.validate:
                converter.validate_advanced_conversion(expert, output_file)
        else:
            logger.warning(f"Expert not found: {expert}")
    
    logger.info("\n✓ Advanced conversion complete!")


if __name__ == "__main__":
    main()