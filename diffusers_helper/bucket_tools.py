# --- START OF FILE bucket_tools.py ---

import math

# NOTE: It's generally recommended that width and height are divisible by 16 or even 32
# for optimal compatibility with VAEs and model architectures.
# This implementation uses a divisibility factor of 32 for all new buckets.
DIVISIBILITY_FACTOR = 32

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
    # Original 640 bucket (divisible by 16) - DO NOT MODIFY
    640: [
        (416, 960), (448, 864), (480, 832), (512, 768), (544, 704),
        (576, 672), (608, 640), (640, 608), (672, 576), (704, 544),
        (768, 512), (832, 480), (864, 448), (960, 416),
    ],
    # Add new buckets from 672 up to 960, with dimensions divisible by 32
    # Using the helper function to generate these systematically
}

# Define the range of new bucket resolutions (step of 32)
new_bucket_keys = range(672, 960 + 1, DIVISIBILITY_FACTOR)

# Generate pairs for each new bucket key
for key in new_bucket_keys:
    bucket_options[key] = generate_bucket_pairs(
        target_res=key,
        step=DIVISIBILITY_FACTOR,
        num_steps_half=7, # Generates a decent range of aspect ratios
        max_ratio=2.7   # Allow slightly more elongated aspect ratios
    )

# --- Special additions/overrides ---

# Ensure HD resolutions are explicitly included in the 960 bucket
# (May already be generated, but adding ensures they exist and handles rounding edge cases)
if 960 in bucket_options:
    hd_pairs = {(720, 1280), (1280, 720)} # Use a set for easy addition
    # Check divisibility just in case (should be fine since 720/1280 are div by 32)
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
                         or None if the resolution key doesn't exist.
    """
    if resolution not in bucket_options:
        print(f"Error: Resolution key '{resolution}' not found in bucket_options.")
        # Fallback strategy: Find the closest available key? Or just return None?
        # Let's try finding the closest key as a fallback.
        available_keys = sorted(bucket_options.keys())
        if not available_keys:
            print("Error: No buckets defined.")
            return None
        closest_key = min(available_keys, key=lambda k: abs(k - resolution))
        print(f"Warning: Falling back to closest available resolution key: {closest_key}")
        resolution = closest_key
        # return None # Or raise an error depending on desired behavior

    min_metric = float('inf')
    best_bucket = None

    # Aspect ratio of the input image
    target_aspect_ratio = w / h if h > 0 else float('inf')

    for (bucket_h, bucket_w) in bucket_options[resolution]:
        # Calculate aspect ratio difference (more robust than the original metric)
        bucket_aspect_ratio = bucket_w / bucket_h if bucket_h > 0 else float('inf')
        metric = abs(target_aspect_ratio - bucket_aspect_ratio)

        # Optional: Add a penalty for area difference if aspect ratios are very close
        # target_area = h * w
        # bucket_area = bucket_h * bucket_w
        # area_difference = abs(target_area - bucket_area) / target_area if target_area > 0 else 0
        # metric += area_difference * 0.1 # Small penalty for area difference

        if metric < min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
        # If metrics are identical, maybe prefer the one with closer area? (Less important)
        # elif metric == min_metric:
        #     current_best_area = best_bucket[0] * best_bucket[1]
        #     new_bucket_area = bucket_h * bucket_w
        #     if abs(target_area - new_bucket_area) < abs(target_area - current_best_area):
        #          best_bucket = (bucket_h, bucket_w)


    if best_bucket is None and bucket_options[resolution]:
        # If somehow no best bucket was found (e.g., input h=0), return the first option
        best_bucket = bucket_options[resolution][0]
    elif best_bucket is None:
         print(f"Error: No bucket options available for resolution {resolution}.")
         return None # Or raise error

    return best_bucket