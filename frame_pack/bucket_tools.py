# Base configuration for scaling bucket options
_BASE_RESOLUTION = 640
_BASE_BUCKET_OPTIONS = [
    (416, 960), (448, 864), (480, 832), (512, 768), (544, 704),
    (576, 672), (608, 640), (640, 608), (672, 576), (704, 544),
    (768, 512), (832, 480), (864, 448), (960, 416),
]

# Cache for generated bucket options to avoid redundant calculations
_generated_bucket_cache = {}

def _round_to_multiple(number, multiple):
    """Rounds a number to the nearest multiple of a given number."""
    if multiple == 0:
        # Default behavior: round to nearest int. Could also raise an error.
        return int(round(number)) 
    return int(multiple * round(float(number) / multiple))

def _adjust_resolution(resolution, divisor=32):
    """
    Adjusts a given resolution to the nearest multiple of 'divisor'.
    If the input resolution is positive but rounds to 0 (e.g., resolution=10, divisor=32),
    it's adjusted to 'divisor'.
    If the input resolution is non-positive (<=0), it defaults to 'divisor'.
    """
    if resolution <= 0:
        return divisor  # Default to minimum valid resolution for non-positive inputs
    
    adjusted = _round_to_multiple(resolution, divisor)
    
    # If resolution was positive but _round_to_multiple resulted in 0 
    # (e.g. input 10 for divisor 32 rounds to 0), ensure it's at least the divisor.
    if adjusted == 0: 
        return divisor 
    return adjusted

def generate_scaled_buckets(target_resolution_input, 
                            base_resolution=_BASE_RESOLUTION, 
                            base_options=_BASE_BUCKET_OPTIONS, 
                            divisor=32):
    """
    Generates scaled bucket options for a target resolution.
    
    The target_resolution_input is first adjusted to the nearest multiple of 'divisor'.
    Bucket dimensions are scaled from 'base_options' (which are for 'base_resolution')
    to the adjusted target resolution. These scaled dimensions are then rounded to the 
    nearest multiple of 'divisor' and ensured to be at least 'divisor'.
    
    Args:
        target_resolution_input (int): The desired target resolution.
        base_resolution (int): The resolution for which 'base_options' are defined.
        base_options (list of tuples): A list of (height, width) tuples for 'base_resolution'.
        divisor (int): The number that resolutions and bucket dimensions should be multiples of.

    Returns:
        list of tuples: Scaled and adjusted bucket options (height, width).
    """
    # Adjust the target resolution for scaling
    actual_target_resolution = _adjust_resolution(target_resolution_input, divisor)

    if actual_target_resolution in _generated_bucket_cache:
        return _generated_bucket_cache[actual_target_resolution]

    # Optimization: If adjusted target resolution matches base resolution.
    # This assumes base_options are already compliant with the divisor.
    # (Our _BASE_BUCKET_OPTIONS are multiples of 32, so this is fine for divisor=32).
    if actual_target_resolution == base_resolution:
        options_to_return = list(base_options) # Return a copy
        _generated_bucket_cache[actual_target_resolution] = options_to_return
        return options_to_return

    scaled_options = []
    seen_options = set() # To handle potential duplicates after rounding

    # Prevent division by zero if base_resolution is 0 (though _BASE_RESOLUTION is 640).
    if base_resolution == 0:
        # Fallback: return a single square bucket of the target resolution.
        # This case should not be hit with current constants.
        default_bucket = (actual_target_resolution, actual_target_resolution)
        _generated_bucket_cache[actual_target_resolution] = [default_bucket]
        return [default_bucket]
        
    scale_factor = float(actual_target_resolution) / base_resolution

    for base_h, base_w in base_options:
        scaled_h_float = base_h * scale_factor
        scaled_w_float = base_w * scale_factor

        scaled_h = _round_to_multiple(scaled_h_float, divisor)
        scaled_w = _round_to_multiple(scaled_w_float, divisor)

        # Ensure minimum dimension is at least the divisor
        scaled_h = max(scaled_h, divisor)
        scaled_w = max(scaled_w, divisor)
        
        bucket_tuple = (scaled_h, scaled_w)
        if bucket_tuple not in seen_options:
            scaled_options.append(bucket_tuple)
            seen_options.add(bucket_tuple)
    
    # If base_options was empty (not the case for internal use but could be if called externally),
    # scaled_options would be empty. Provide a default bucket in such a scenario.
    # actual_target_resolution is guaranteed to be >= divisor by _adjust_resolution.
    if not scaled_options: 
        default_bucket = (actual_target_resolution, actual_target_resolution)
        scaled_options.append(default_bucket)

    _generated_bucket_cache[actual_target_resolution] = scaled_options
    return scaled_options

def find_nearest_bucket(h, w, resolution=640):
    """
    Finds the nearest bucket for a given height (h) and width (w)
    at a specified target resolution.

    The 'resolution' parameter is the user's intended target resolution.
    This function will:
    1. Adjust this resolution to the nearest multiple of 32 (minimum 32).
    2. Generate a list of bucket options (height, width pairs) by scaling
       predefined base options (for 640px) to this adjusted resolution.
       All generated bucket dimensions will also be multiples of 32 and at least 32.
    3. Find the bucket from this generated list that is "nearest" to the
       aspect ratio of the input h, w. The nearness metric is
       abs(input_h * bucket_w - input_w * bucket_h).

    Args:
        h (int): The height of the image/item.
        w (int): The width of the image/item.
        resolution (int): The target resolution for which to find buckets.
                          Defaults to 640.

    Returns:
        tuple: A (bucket_h, bucket_w) tuple representing the best bucket found.
    """
    # generate_scaled_buckets handles the adjustment of 'resolution' internally
    # and uses a divisor of 32 by default for its calculations.
    # The problem statement implies a fixed divisor of 32 for this tool.
    current_bucket_options = generate_scaled_buckets(resolution, divisor=32)

    # Failsafe: If generate_scaled_buckets somehow returned an empty list (e.g., if _BASE_BUCKET_OPTIONS was empty),
    # provide a default bucket based on the adjusted resolution.
    if not current_bucket_options:
        adjusted_res_for_fallback = _adjust_resolution(resolution, 32)
        return (adjusted_res_for_fallback, adjusted_res_for_fallback)

    min_metric = float('inf')
    best_bucket = None 
    # Since current_bucket_options is guaranteed to be non-empty by the check above (or by generate_scaled_buckets's own logic
    # when _BASE_BUCKET_OPTIONS is populated), best_bucket will be assigned in the loop.

    for (bucket_h, bucket_w) in current_bucket_options:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric: # Using "<=" preserves original behavior (last encountered wins on ties)
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    
    return best_bucket