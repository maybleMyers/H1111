# --- START OF FILE bucket_tools.py ---

import math

# NOTE: It's generally recommended that width and height are divisible by 16 or even 32
# for optimal compatibility with VAEs and model architectures.
# This implementation uses a divisibility factor of 32 for all new buckets.
DIVISIBILITY_FACTOR = 32 # <--- Using 32 for generated buckets

# --- Helper Function to Generate Bucket Pairs ---

def generate_bucket_pairs(target_res, step=32, num_steps_half=6, max_ratio=2.5):
    """
    Generates a list of (height, width) pairs for a given target resolution bucket.

    Args:
        target_res (int): The target resolution for the bucket (e.g., 768).
                          Should ideally be divisible by 'step'.
        step (int): The required divisibility factor for height and width (e.g., 32).
        num_steps_half (int): Controls the range around the target_res to generate pairs.
                              Pairs will be generated with one dimension ranging roughly from
                              target_res - num_steps_half * step to target_res + num_steps_half * step.
        max_ratio (float): The maximum allowed aspect ratio (max_dim / min_dim) to prevent
                           extremely elongated shapes.

    Returns:
        list[tuple[int, int]]: A sorted list of (height, width) pairs.
    """
    if target_res % step != 0:
        print(f"Warning: target_res {target_res} is not divisible by step {step}. Adjusting.")
        target_res = round(target_res / step) * step

    pairs = set()

    # Add the square case
    pairs.add((target_res, target_res))

    # Iterate through potential heights around the target resolution
    for i in range(-num_steps_half, num_steps_half + 1):
        h_fixed = target_res + i * step
        if h_fixed <= 0:
            continue

        # Calculate width aiming to preserve area (target_res * target_res)
        ideal_w = (target_res * target_res) / h_fixed
        # Round width to the nearest multiple of 'step'
        w_rounded = round(ideal_w / step) * step
        w_rounded = max(step, w_rounded) # Ensure width is at least 'step'

        if w_rounded > 0:
            ratio = max(h_fixed, w_rounded) / min(h_fixed, w_rounded)
            if ratio <= max_ratio:
                pairs.add((h_fixed, w_rounded))

    # Iterate through potential widths around the target resolution (to catch different ratios)
    for i in range(-num_steps_half, num_steps_half + 1):
        w_fixed = target_res + i * step
        if w_fixed <= 0:
            continue

        # Calculate height aiming to preserve area
        ideal_h = (target_res * target_res) / w_fixed
        # Round height to the nearest multiple of 'step'
        h_rounded = round(ideal_h / step) * step
        h_rounded = max(step, h_rounded) # Ensure height is at least 'step'

        if h_rounded > 0:
            ratio = max(h_rounded, w_fixed) / min(h_rounded, w_fixed)
            if ratio <= max_ratio:
                pairs.add((h_rounded, w_fixed))

    # Convert set to list and sort
    sorted_pairs = sorted(list(pairs))
    return sorted_pairs

# --- Bucket Options ---

bucket_options = {
    # Original 640 bucket (divisible by 32) - KEEP AS IS FOR COMPATIBILITY
    640: [
        (416, 960), (448, 864), (480, 832), (512, 768), (544, 704),
        (576, 672), (608, 640), (640, 608), (672, 576), (704, 544),
        (768, 512), (832, 480), (864, 448), (960, 416),
    ],
    # Add new buckets from 672 up to 960, with dimensions divisible by 32
    # Using the helper function to generate these systematically
}

# Define the range of new bucket resolutions (step of 32)
new_bucket_keys = range(128, 960 + 1, DIVISIBILITY_FACTOR)

# Generate pairs for each new bucket key
for key in new_bucket_keys:
    bucket_options[key] = generate_bucket_pairs(
        target_res=key,
        step=DIVISIBILITY_FACTOR,
        num_steps_half=4, # Generates a decent range of aspect ratios
        max_ratio=2.7   # Allow slightly more elongated aspect ratios
    )

# --- Special additions/overrides ---

# Ensure HD resolutions are explicitly included in the 960 bucket
# (May already be generated, but adding ensures they exist and handles rounding edge cases)
if 960 in bucket_options:
    hd_pairs = {(704, 1280), (1280, 704)} # Use a set for easy addition
    # Check divisibility just in case (should be fine since 704/1280 are div by 32)
    hd_pairs_filtered = {
        (h, w) for h, w in hd_pairs
        if h % DIVISIBILITY_FACTOR == 0 and w % DIVISIBILITY_FACTOR == 0
    }

    existing_pairs = set(bucket_options[960])
    updated_pairs = existing_pairs.union(hd_pairs_filtered)
    bucket_options[960] = sorted(list(updated_pairs))
else:
    print("Warning: 960 key was not generated. Cannot add HD resolutions.")


# --- Function to Find Nearest Bucket ---

def find_nearest_bucket(h, w, resolution=640):
    """
    Finds the (bucket_h, bucket_w) pair within the specified resolution bucket
    that is closest to the aspect ratio of the input (h, w).

    Args:
        h (int): Original height.
        w (int): Original width.
        resolution (int): The key for the desired bucket in `bucket_options`.

    Returns:
        tuple[int, int]: The best matching (height, width) bucket dimensions,
                         or None if resolution key is invalid or no buckets exist.
    """
    # Ensure resolution is an integer
    try:
        resolution = int(resolution)
    except (TypeError, ValueError):
        print(f"Error: Invalid resolution value '{resolution}'. Must be an integer.")
        resolution = 640 # Default fallback
        print(f"Warning: Defaulting to resolution {resolution}.")

    if resolution not in bucket_options:
        available_keys = sorted(bucket_options.keys())
        if not available_keys:
            print("Error: No buckets defined.")
            return None # Cannot proceed

        # Find the numerically closest key
        closest_key = min(available_keys, key=lambda k: abs(k - resolution))
        print(f"Warning: Resolution key '{resolution}' not found. Using closest key: {closest_key}")
        resolution = closest_key

    if not bucket_options[resolution]:
         print(f"Error: No bucket options available for resolution {resolution}.")
         return None # Cannot proceed

    min_metric = float('inf')
    best_bucket = None

    # Aspect ratio of the input image
    if h <= 0: # Avoid division by zero or invalid aspect ratio
        print("Warning: Invalid input height (<=0). Using aspect ratio 1.0 for bucket selection.")
        target_aspect_ratio = 1.0
    else:
        target_aspect_ratio = w / h

    for (bucket_h, bucket_w) in bucket_options[resolution]:
        # Calculate aspect ratio difference
        if bucket_h <= 0: continue # Skip invalid buckets
        bucket_aspect_ratio = bucket_w / bucket_h
        metric = abs(target_aspect_ratio - bucket_aspect_ratio)

        # Optional: Add a small penalty for area difference to break ties
        target_area = h * w
        bucket_area = bucket_h * bucket_w
        area_diff_penalty = abs(target_area - bucket_area) / (target_area + 1e-6) * 0.01 # Small penalty
        metric += area_diff_penalty

        if metric < min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)

    if best_bucket is None:
        # Fallback if something went wrong, e.g., all buckets invalid
        print(f"Warning: Could not find best bucket for resolution {resolution}. Using first available.")
        best_bucket = bucket_options[resolution][0]

    return best_bucket