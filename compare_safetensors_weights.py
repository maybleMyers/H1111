import argparse
import logging
import gc
import math
from pathlib import Path
from typing import Dict, Set, Tuple, List, Any

import torch
from safetensors import safe_open
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Use MPS if available on Mac, otherwise CUDA or CPU
if torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
elif torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
else:
    DEFAULT_DEVICE = "cpu"

def get_tensor_keys(filepath: Path) -> Set[str]:
    """Gets all tensor keys from a safetensors file without loading tensors."""
    keys = set()
    try:
        with safe_open(filepath, framework="pt", device="cpu") as f:
            keys = set(f.keys())
        logging.debug(f"Found {len(keys)} keys in {filepath.name}")
        return keys
    except Exception as e:
        logging.error(f"Error opening or reading keys from {filepath}: {e}")
        raise

def compare_tensors(
    key: str, file1: Path, file2: Path, device: torch.device, atol: float
) -> Tuple[bool, float, float, float]:
    """
    Loads and compares a single tensor from two files.

    Args:
        key: The tensor key to compare.
        file1: Path to the first safetensors file.
        file2: Path to the second safetensors file.
        device: The torch device to use for comparison.
        atol: Absolute tolerance for torch.allclose check.

    Returns:
        Tuple containing:
        - is_close: Boolean indicating if tensors are close within tolerance.
        - mean_abs_diff: Mean absolute difference.
        - max_abs_diff: Maximum absolute difference.
        - cosine_sim: Cosine similarity (-2.0 if not applicable/error).
    """
    # Initialize variables to handle potential early returns
    t1, t2, diff = None, None, None
    mean_abs_diff = float('nan')
    max_abs_diff = float('nan')
    cosine_sim = -2.0  # Use -2.0 to indicate not computed or error
    is_close = False

    try:
        # Use safe_open for lazy loading
        with safe_open(file1, framework="pt", device="cpu") as f1, \
             safe_open(file2, framework="pt", device="cpu") as f2:

            if key not in f1.keys():
                 logging.warning(f"Key '{key}' missing in Model 1 ({file1.name}). Skipping comparison for this key.")
                 # No need to return here, let finally block handle cleanup if t2 was loaded
            elif key not in f2.keys():
                 logging.warning(f"Key '{key}' missing in Model 2 ({file2.name}). Skipping comparison for this key.")
                 # Load t1 to ensure it's deleted in finally if needed
                 t1 = f1.get_tensor(key)
            else:
                # Both keys exist, proceed with loading
                t1 = f1.get_tensor(key)
                t2 = f2.get_tensor(key)

                # --- Basic Checks ---
                if t1.shape != t2.shape:
                    logging.warning(
                        f"Shape mismatch for key '{key}': {t1.shape} vs {t2.shape}. Cannot compare."
                    )
                    # Return values indicating mismatch; t1/t2 will be cleaned up by finally
                    return False, float('nan'), float('nan'), -2.0 # Use NaN/special value for mismatch

                if t1.dtype != t2.dtype:
                    logging.warning(
                        f"Dtype mismatch for key '{key}': {t1.dtype} vs {t2.dtype}. Will attempt cast for comparison."
                    )
                    # Attempt comparison anyway, might fail or give less meaningful results
                    try:
                        t2 = t2.to(t1.dtype)
                    except Exception as cast_e:
                        logging.error(f"Could not cast tensor '{key}' for comparison: {cast_e}")
                        # Return values indicating error; t1/t2 will be cleaned up by finally
                        return False, float('nan'), float('nan'), -2.0

                # --- Move to device for computation ---
                try:
                    # Move original tensors (or casted t2)
                    t1_dev = t1.to(device)
                    t2_dev = t2.to(device)
                except Exception as move_e:
                    logging.error(f"Could not move tensor '{key}' to device '{device}': {move_e}. Trying CPU.")
                    device = torch.device('cpu')
                    t1_dev = t1.to(device)
                    t2_dev = t2.to(device)


                # --- Comparison Metrics ---
                with torch.no_grad():
                    # Use float32 for difference calculation stability
                    diff = torch.abs(t1_dev.float() - t2_dev.float()) # Assign diff here
                    mean_abs_diff = torch.mean(diff).item()
                    max_abs_diff = torch.max(diff).item()

                    # torch.allclose check
                    is_close = torch.allclose(t1_dev, t2_dev, atol=atol, rtol=0) # rtol=0 for FP16 comparison mostly depends on atol

                    # Cosine Similarity (avoid for scalars, ensure vectors are flat)
                    if t1_dev.numel() > 1:
                        try:
                            # Ensure tensors are flat and float for cosine sim
                            cos_sim_val = torch.nn.functional.cosine_similarity(
                                t1_dev.flatten().float(), t2_dev.flatten().float(), dim=0
                            ).item()
                            # Handle potential NaN/Inf from zero vectors etc.
                            cosine_sim = cos_sim_val if math.isfinite(cos_sim_val) else -1.0
                        except Exception as cs_err:
                            logging.warning(f"Could not compute cosine similarity for '{key}': {cs_err}")
                            cosine_sim = -1.0 # Indicate computation error
                    elif t1_dev.numel() == 1:
                        cosine_sim = 1.0 if torch.allclose(t1_dev, t2_dev) else 0.0 # Define for scalars

                    # Clean up device tensors explicitly after use
                    del t1_dev, t2_dev


    except Exception as e:
        logging.error(f"Unhandled error comparing tensor '{key}': {e}", exc_info=True)
        # Return default failure values
        return False, float('nan'), float('nan'), -2.0
    finally:
        # --- Modified Finally Block ---
        # Clear potential tensor references
        if t1 is not None:
            del t1
        if t2 is not None:
            del t2
        if diff is not None: # Now 'diff' might be defined or not
            del diff

        # Aggressive garbage collection and cache clearing
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
             try: # Newer pytorch versions have empty_cache for mps
                  torch.mps.empty_cache()
             except AttributeError:
                  pass # Ignore if not available

    # Return the calculated values if comparison was successful
    return is_close, mean_abs_diff, max_abs_diff, cosine_sim


def compare_models(file1_path: Path, file2_path: Path, device_str: str, atol: float, top_n_diff: int):
    """
    Compares two safetensors models weight by weight.

    Args:
        file1_path: Path to the first model file.
        file2_path: Path to the second model file.
        device_str: Device string ('cpu', 'cuda', 'mps').
        atol: Absolute tolerance for closeness check.
        top_n_diff: Number of most different tensors to report.
    """
    if not file1_path.is_file():
        logging.error(f"File not found: {file1_path}")
        return
    if not file2_path.is_file():
        logging.error(f"File not found: {file2_path}")
        return

    try:
        device = torch.device(device_str)
        logging.info(f"Using device: {device}")
    except Exception as e:
        logging.warning(f"Could not select device '{device_str}', falling back to CPU. Error: {e}")
        device = torch.device("cpu")

    logging.info(f"Comparing Model 1: {file1_path.name}")
    logging.info(f"          Model 2: {file2_path.name}")
    logging.info(f"Absolute tolerance (atol) for closeness: {atol}")

    try:
        keys1 = get_tensor_keys(file1_path)
        keys2 = get_tensor_keys(file2_path)
    except Exception:
        return # Error already logged by get_tensor_keys

    common_keys = sorted(list(keys1.intersection(keys2)))
    unique_keys1 = sorted(list(keys1 - keys2))
    unique_keys2 = sorted(list(keys2 - keys1))

    logging.info(f"Found {len(common_keys)} common tensor keys.")
    if unique_keys1:
        logging.warning(f"{len(unique_keys1)} keys unique to Model 1 ({file1_path.name}): {unique_keys1[:10]}{'...' if len(unique_keys1) > 10 else ''}")
    if unique_keys2:
        logging.warning(f"{len(unique_keys2)} keys unique to Model 2 ({file2_path.name}): {unique_keys2[:10]}{'...' if len(unique_keys2) > 10 else ''}")

    if not common_keys:
        logging.error("No common keys found between models. Cannot compare.")
        return

    results: List[Dict[str, Any]] = []
    close_count = 0
    compared_count = 0 # Track how many comparisons were attempted
    valid_comparisons = 0 # Track successful comparisons with numerical results
    mismatched_shape_keys = []
    comparison_error_keys = []

    all_mean_abs_diffs = []
    all_max_abs_diffs = []
    all_cosine_sims = []

    logging.info("Starting tensor comparison...")
    for key in tqdm(common_keys, desc="Comparing Tensors"):
        compared_count += 1
        is_close, mean_ad, max_ad, cos_sim = compare_tensors(
            key, file1_path, file2_path, device, atol
        )

        # Check for comparison failure (NaN or -2)
        if math.isnan(mean_ad) or math.isnan(max_ad) or cos_sim == -2.0:
            # Check if it was specifically a shape mismatch (common case)
            # Re-check shapes briefly - less efficient but simple for logging
            try:
                 with safe_open(file1_path, framework="pt", device="cpu") as f1, \
                      safe_open(file2_path, framework="pt", device="cpu") as f2:
                     t1_shape = f1.get_shape(key)
                     t2_shape = f2.get_shape(key)
                     if t1_shape != t2_shape:
                          mismatched_shape_keys.append(key)
                     else:
                          comparison_error_keys.append(key) # Other error
            except Exception:
                 comparison_error_keys.append(key) # Error getting shapes or other issue

            logging.debug(f"Skipping results aggregation for key '{key}' due to comparison errors/mismatch.")
            continue # Skip adding results for this key

        # If we reach here, comparison was numerically successful
        valid_comparisons += 1
        if is_close:
            close_count += 1
        all_mean_abs_diffs.append(mean_ad)
        all_max_abs_diffs.append(max_ad)
        # Store cosine similarity if validly computed (-1 means computation issue like 0 vector)
        if cos_sim >= -1.0: # Allow -1 (error during calc) but not -2 (no calc attempted/major error)
            all_cosine_sims.append(cos_sim)

        results.append({
            "key": key,
            "is_close": is_close,
            "mean_abs_diff": mean_ad,
            "max_abs_diff": max_ad,
            "cosine_sim": cos_sim
        })


    # --- Summary ---
    logging.info("\n--- Comparison Summary ---")
    logging.info(f"Attempted comparison for {compared_count} common keys.")
    if mismatched_shape_keys:
        logging.warning(f"Found {len(mismatched_shape_keys)} keys with mismatched shapes (skipped): {mismatched_shape_keys[:5]}{'...' if len(mismatched_shape_keys) > 5 else ''}")
    if comparison_error_keys:
         logging.error(f"Encountered errors during comparison for {len(comparison_error_keys)} keys (skipped): {comparison_error_keys[:5]}{'...' if len(comparison_error_keys) > 5 else ''}")


    if valid_comparisons == 0:
         logging.error("No tensors could be validly compared numerically (check for shape mismatches or errors).")
         return

    logging.info(f"Successfully compared {valid_comparisons} tensors numerically.")
    logging.info(f"Tensors within tolerance (atol={atol}): {close_count} / {valid_comparisons} ({close_count/valid_comparisons:.2%})")

    # Calculate overall stats only on valid comparisons
    avg_mean_ad = np.mean(all_mean_abs_diffs) if all_mean_abs_diffs else float('nan')
    avg_max_ad = np.mean(all_max_abs_diffs) if all_max_abs_diffs else float('nan')
    overall_max_ad = np.max(all_max_abs_diffs) if all_max_abs_diffs else float('nan')
    overall_max_ad_key = max(results, key=lambda x: x.get('max_abs_diff', -float('inf')))['key'] if results else 'N/A'

    # Filter out -1 cosine sims before calculating stats if desired, or include them
    valid_cosine_sims = [cs for cs in all_cosine_sims if cs >= 0] # Only positive sims for avg/min
    avg_cosine_sim = np.mean(valid_cosine_sims) if valid_cosine_sims else float('nan')
    min_cosine_sim = np.min(valid_cosine_sims) if valid_cosine_sims else float('nan')


    logging.info(f"Average Mean Absolute Difference (MAD): {avg_mean_ad:.6g}")
    logging.info(f"Average Max Absolute Difference:      {avg_max_ad:.6g}")
    logging.info(f"Overall Maximum Absolute Difference:  {overall_max_ad:.6g} (found in tensor '{overall_max_ad_key}')")
    logging.info(f"Average Cosine Similarity (valid>=0): {avg_cosine_sim:.6f}" if not math.isnan(avg_cosine_sim) else "Average Cosine Similarity (valid>=0): N/A")
    logging.info(f"Minimum Cosine Similarity (valid>=0): {min_cosine_sim:.6f}" if not math.isnan(min_cosine_sim) else "Minimum Cosine Similarity (valid>=0): N/A")


    # --- Top Differences ---
    # Sort by max absolute difference descending (handle potential missing keys)
    results.sort(key=lambda x: x.get("max_abs_diff", -float('inf')), reverse=True)

    logging.info(f"\n--- Top {min(top_n_diff, len(results))} Tensors by Max Absolute Difference (Numerically Compared Only) ---")
    for i in range(min(top_n_diff, len(results))):
        res = results[i]
        # Ensure keys exist before accessing
        key = res.get('key', 'ERROR_MISSING_KEY')
        max_ad_val = res.get('max_abs_diff', float('nan'))
        mean_ad_val = res.get('mean_abs_diff', float('nan'))
        cos_sim_val = res.get('cosine_sim', float('nan'))
        close_val = res.get('is_close', 'N/A')

        logging.info(
            f"{i+1}. Key: {key:<50} "
            f"MaxAD: {max_ad_val:.6g} | "
            f"MeanAD: {mean_ad_val:.6g} | "
            f"CosSim: {cos_sim_val:.4f} | "
            f"Close: {close_val}"
        )

    # --- Interpretation for LoRA ---
    logging.info("\n--- LoRA Compatibility Interpretation ---")
    # Prioritize architectural differences
    if unique_keys1 or unique_keys2 or mismatched_shape_keys:
        logging.error("Models have architectural differences (unique keys or mismatched shapes found). Direct LoRA swapping is NOT recommended.")
        if unique_keys1 or unique_keys2:
            logging.warning(" - Different sets of weights exist.")
        if mismatched_shape_keys:
            logging.warning(f" - Mismatched shapes found for keys like: {mismatched_shape_keys[0]}")
    elif comparison_error_keys:
         logging.warning("Some tensors could not be compared due to errors (other than shape mismatch). Check logs. LoRA compatibility might be affected.")
    else:
        # Assess based on numerical differences if architecture matches
        logging.info("Models appear to have the same architecture (matching keys and shapes). Assessing numerical similarity:")
        if avg_mean_ad < 1e-5 and overall_max_ad < 1e-3:
            logging.info(" -> Differences are very small. Models appear highly similar. High LoRA compatibility expected.")
        elif avg_mean_ad < 1e-4 and overall_max_ad < 5e-3:
            logging.info(" -> Differences are small. Models appear quite similar. Good LoRA compatibility expected.")
        elif avg_mean_ad < 1e-3 and overall_max_ad < 1e-2:
            logging.info(" -> Moderate differences detected. LoRAs might work but performance could vary, especially if targeting layers with larger differences.")
        else:
            logging.warning(" -> Significant numerical differences detected (Average MAD > 1e-3 or Overall MaxAD > 0.01). LoRA compatibility is questionable. Performance may degrade even with matching architecture.")

        if not math.isnan(min_cosine_sim) and min_cosine_sim < 0.98: # Stricter threshold for matching architecture
            logging.warning(f" -> Some tensors have lower cosine similarity (min >= 0: {min_cosine_sim:.4f}), indicating potential directional differences. This could affect LoRA.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare weights between two safetensors model files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model1_path", type=str, help="Path to the first .safetensors model file."
    )
    parser.add_argument(
        "model2_path", type=str, help="Path to the second .safetensors model file."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda", "mps"],
        help="Device to use for tensor comparisons ('cuda'/'mps' recommended if available).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4, # A reasonable default for FP16 comparison
        help="Absolute tolerance (atol) for torch.allclose check to consider tensors 'close'.",
    )
    parser.add_argument(
        "--top_n_diff",
        type=int,
        default=10,
        help="Report details for the top N tensors with the largest maximum absolute difference.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    compare_models(
        Path(args.model1_path),
        Path(args.model2_path),
        args.device,
        args.atol,
        args.top_n_diff,
    )

if __name__ == "__main__":
    main()