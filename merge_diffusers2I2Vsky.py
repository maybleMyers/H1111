import os
import json
import torch
from safetensors.torch import load_file, save_file
import logging
import shutil
from typing import Dict, Any, Set
import re

logger = logging.getLogger("PeftMerger") 
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def normalize_key(key: str) -> str:
    """Normalize key format to match base model"""
    key = key.replace("transformer.double_blocks", "transformer_blocks")
    key = key.replace("transformer.single_blocks", "single_transformer_blocks")
    key = re.sub(r'\.+', '.', key)  # Remove double dots
    if key.endswith('.'):
        key = key[:-1]
    return key

def merge_lora_weights(base_weights: Dict[str, torch.Tensor],
                      lora_weights: Dict[str, torch.Tensor],
                      alpha: float = 1.0) -> Dict[str, torch.Tensor]:
    """Merge LoRA weights into base model weights"""
    merged = base_weights.copy()
    
    # Print first few keys for debugging
    logger.info(f"Base model keys (first 5): {list(base_weights.keys())[:5]}")
    logger.info(f"LoRA keys (first 5): {list(lora_weights.keys())[:5]}")
    
    # Process LoRA keys
    for key in lora_weights.keys():
        if '.lora_A.weight' not in key:
            continue
            
        logger.info(f"Processing LoRA key: {key}")
        base_key = key.replace('.lora_A.weight', '')
        lora_a = lora_weights[key]
        lora_b = lora_weights[base_key + '.lora_B.weight']
        
        # Normalize after getting both A and B weights
        normalized_key = normalize_key(base_key)
        logger.info(f"Normalized key: {normalized_key}")
        
        # Map double blocks
        if 'img_attn_qkv' in base_key:
            weights = torch.matmul(lora_b, lora_a)
            q, k, v = torch.chunk(weights, 3, dim=0)
            block_match = re.search(r'transformer_blocks\.(\d+)', normalized_key)
            block_num = block_match.group(1)
            
            q_key = f'transformer_blocks.{block_num}.attn.to_q.weight'
            k_key = f'transformer_blocks.{block_num}.attn.to_k.weight'
            v_key = f'transformer_blocks.{block_num}.attn.to_v.weight'
            
            if all(k in merged for k in [q_key, k_key, v_key]):
                merged[q_key] = merged[q_key] + alpha * q
                merged[k_key] = merged[k_key] + alpha * k
                merged[v_key] = merged[v_key] + alpha * v
                logger.info(f"Updated keys: {q_key}, {k_key}, {v_key}")
            else:
                logger.warning(f"Missing some keys: {[k for k in [q_key, k_key, v_key] if k not in merged]}")
            
        elif 'txt_attn_qkv' in base_key:
            weights = torch.matmul(lora_b, lora_a)
            q, k, v = torch.chunk(weights, 3, dim=0)
            block_match = re.search(r'transformer_blocks\.(\d+)', normalized_key)
            block_num = block_match.group(1)
            
            q_key = f'transformer_blocks.{block_num}.attn.add_q_proj.weight'
            k_key = f'transformer_blocks.{block_num}.attn.add_k_proj.weight'
            v_key = f'transformer_blocks.{block_num}.attn.add_v_proj.weight'
            
            if all(k in merged for k in [q_key, k_key, v_key]):
                merged[q_key] = merged[q_key] + alpha * q
                merged[k_key] = merged[k_key] + alpha * k
                merged[v_key] = merged[v_key] + alpha * v
                logger.info(f"Updated keys: {q_key}, {k_key}, {v_key}")
            else:
                logger.warning(f"Missing some keys: {[k for k in [q_key, k_key, v_key] if k not in merged]}")
            
        elif 'img_attn_proj' in base_key:
            block_match = re.search(r'transformer_blocks\.(\d+)', normalized_key)
            block_num = block_match.group(1)
            model_key = f'transformer_blocks.{block_num}.attn.to_out.0.weight'
            
            if model_key in merged:
                weights = torch.matmul(lora_b, lora_a)
                merged[model_key] = merged[model_key] + alpha * weights
                logger.info(f"Updated key: {model_key}")
            else:
                logger.warning(f"Missing key: {model_key}")
            
        elif 'txt_attn_proj' in base_key:
            block_match = re.search(r'transformer_blocks\.(\d+)', normalized_key)
            block_num = block_match.group(1)
            model_key = f'transformer_blocks.{block_num}.attn.to_add_out.weight'
            
            if model_key in merged:
                weights = torch.matmul(lora_b, lora_a)
                merged[model_key] = merged[model_key] + alpha * weights
                logger.info(f"Updated key: {model_key}")
            else:
                logger.warning(f"Missing key: {model_key}")
            
        elif 'img_mlp.fc1' in base_key:
            block_match = re.search(r'transformer_blocks\.(\d+)', normalized_key)
            block_num = block_match.group(1)
            model_key = f'transformer_blocks.{block_num}.ff.net.0.proj.weight'
            
            if model_key in merged:
                weights = torch.matmul(lora_b, lora_a)
                merged[model_key] = merged[model_key] + alpha * weights
                logger.info(f"Updated key: {model_key}")
            else:
                logger.warning(f"Missing key: {model_key}")
            
        elif 'img_mlp.fc2' in base_key:
            block_match = re.search(r'transformer_blocks\.(\d+)', normalized_key)
            block_num = block_match.group(1)
            model_key = f'transformer_blocks.{block_num}.ff.net.2.weight'
            
            if model_key in merged:
                weights = torch.matmul(lora_b, lora_a)
                merged[model_key] = merged[model_key] + alpha * weights
                logger.info(f"Updated key: {model_key}")
            else:
                logger.warning(f"Missing key: {model_key}")
            
        elif 'txt_mlp.fc1' in base_key:
            block_match = re.search(r'transformer_blocks\.(\d+)', normalized_key)
            block_num = block_match.group(1)
            model_key = f'transformer_blocks.{block_num}.ff_context.net.0.proj.weight'
            
            if model_key in merged:
                weights = torch.matmul(lora_b, lora_a)
                merged[model_key] = merged[model_key] + alpha * weights
                logger.info(f"Updated key: {model_key}")
            else:
                logger.warning(f"Missing key: {model_key}")
            
        elif 'txt_mlp.fc2' in base_key:
            block_match = re.search(r'transformer_blocks\.(\d+)', normalized_key)
            block_num = block_match.group(1)
            model_key = f'transformer_blocks.{block_num}.ff_context.net.2.weight'
            
            if model_key in merged:
                weights = torch.matmul(lora_b, lora_a)
                merged[model_key] = merged[model_key] + alpha * weights
                logger.info(f"Updated key: {model_key}")
            else:
                logger.warning(f"Missing key: {model_key}")
    
    return merged

def save_sharded_model(weights: Dict[str, torch.Tensor], 
                      index_data: dict, 
                      output_dir: str,
                      base_model_path: str):
    """Save merged weights in same sharded format as original"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy all non-safetensor files from original directory
    index_dir = os.path.dirname(os.path.abspath(base_model_path))
    for file in os.listdir(index_dir):
        if not file.endswith('.safetensors'):
            src = os.path.join(index_dir, file)
            dst = os.path.join(output_dir, file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                shutil.copytree(src, dst)
    
    # Group weights by shard
    weight_map = index_data['weight_map']
    shard_weights = {}
    
    for key, shard in weight_map.items():
        if shard not in shard_weights:
            shard_weights[shard] = {}
        if key in weights:
            shard_weights[shard][key] = weights[key]
    
    # Save each shard
    for shard, shard_dict in shard_weights.items():
        if not shard_dict:  # Skip empty shards
            continue
        shard_path = os.path.join(output_dir, shard)
        logger.info(f"Saving shard {shard} with {len(shard_dict)} tensors")
        save_file(shard_dict, shard_path)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    
    # Load base model index
    logger.info("Loading base model index...")
    with open(args.base_model, 'r') as f:
        index_data = json.load(f)
    weight_map = index_data['weight_map']
    
    # Load base weights
    logger.info("Loading base model weights...")
    base_dir = os.path.dirname(args.base_model)
    base_weights = {}
    for part_file in set(weight_map.values()):
        part_path = os.path.join(base_dir, part_file)
        logger.info(f"Loading from {part_path}")
        weights = load_file(part_path)
        base_weights.update(weights)
        
    # Load LoRA
    logger.info("Loading LoRA weights...")
    lora_weights = load_file(args.adapter)
    
    # Merge
    logger.info(f"Merging with alpha={args.alpha}")
    merged_weights = merge_lora_weights(base_weights, lora_weights, args.alpha)
    
    # Save in sharded format
    logger.info(f"Saving merged model to {args.output}")
    save_sharded_model(merged_weights, index_data, args.output, args.base_model)
    logger.info("Done!")

if __name__ == '__main__':
    main()