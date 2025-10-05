"""
Context Windows for WAN 2.2 Video Generation
Implements sliding window processing for long video generation with WAN models.
Based on context window techniques for managing temporal coherence in video diffusion models.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, List, Dict, Any, Tuple
import torch
import numpy as np
import collections
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import math

if TYPE_CHECKING:
    from wan.modules.model import WanModel

logger = logging.getLogger(__name__)


class ContextWindowABC(ABC):
    """Abstract base class for context windows."""
    
    def __init__(self):
        pass
    
    @abstractmethod
    def get_tensor(self, full: torch.Tensor) -> torch.Tensor:
        """Get torch.Tensor applicable to current window."""
        raise NotImplementedError("Not implemented.")
    
    @abstractmethod
    def add_window(self, full: torch.Tensor, to_add: torch.Tensor) -> torch.Tensor:
        """Apply torch.Tensor of window to the full tensor, in place. Returns reference to updated full tensor."""
        raise NotImplementedError("Not implemented.")


class ContextHandlerABC(ABC):
    """Abstract base class for context handlers."""
    
    def __init__(self):
        pass
    
    @abstractmethod
    def should_use_context(self, model: Any, conds: List[List[Dict]], x_in: torch.Tensor, 
                          timestep: torch.Tensor, model_options: Dict[str, Any]) -> bool:
        raise NotImplementedError("Not implemented.")
    
    @abstractmethod
    def get_resized_cond(self, cond_in: List[Dict], x_in: torch.Tensor, 
                        window: ContextWindowABC, device=None) -> List:
        raise NotImplementedError("Not implemented.")
    
    @abstractmethod
    def execute(self, calc_cond_batch: Callable, model: Any, conds: List[List[Dict]], 
               x_in: torch.Tensor, timestep: torch.Tensor, model_options: Dict[str, Any]):
        raise NotImplementedError("Not implemented.")


class IndexListContextWindow(ContextWindowABC):
    """Context window implementation using index lists."""
    
    def __init__(self, index_list: List[int], dim: int = 0):
        self.index_list = index_list
        self.context_length = len(index_list)
        self.dim = dim
    
    def get_tensor(self, full: torch.Tensor, device=None, dim=None) -> torch.Tensor:
        if dim is None:
            dim = self.dim
        if dim == 0 and full.shape[dim] == 1:
            return full
        idx = [slice(None)] * dim + [self.index_list]
        return full[idx].to(device) if device else full[idx]
    
    def add_window(self, full: torch.Tensor, to_add: torch.Tensor, dim=None) -> torch.Tensor:
        if dim is None:
            dim = self.dim
        idx = [slice(None)] * dim + [self.index_list]
        full[idx] += to_add
        return full


class IndexListCallbacks:
    """Callback types for context window processing."""
    EVALUATE_CONTEXT_WINDOWS = "evaluate_context_windows"
    COMBINE_CONTEXT_WINDOW_RESULTS = "combine_context_window_results"
    EXECUTE_START = "execute_start"
    EXECUTE_CLEANUP = "execute_cleanup"
    
    def init_callbacks(self):
        return {}


@dataclass
class ContextSchedule:
    """Context schedule configuration."""
    name: str
    func: Callable


@dataclass
class ContextFuseMethod:
    """Context fusion method configuration."""
    name: str
    func: Callable


ContextResults = collections.namedtuple("ContextResults", ['window_idx', 'sub_conds_out', 'sub_conds', 'window'])


class IndexListContextHandler(ContextHandlerABC):
    """Main handler for sliding window context processing."""
    
    def __init__(self, context_schedule: ContextSchedule, fuse_method: ContextFuseMethod,
                 context_length: int = 1, context_overlap: int = 0, context_stride: int = 1,
                 closed_loop: bool = False, dim: int = 0):
        self.context_schedule = context_schedule
        self.fuse_method = fuse_method
        self.context_length = context_length
        self.context_overlap = context_overlap
        self.context_stride = context_stride
        self.closed_loop = closed_loop
        self.dim = dim
        self._step = 0
        self.callbacks = {}
    
    def should_use_context(self, model: Any, conds: List[List[Dict]], x_in: torch.Tensor,
                          timestep: torch.Tensor, model_options: Dict[str, Any]) -> bool:
        """Check if context windows should be used based on input size."""
        if x_in.size(self.dim) > self.context_length:
            logger.info(f"Using context windows {self.context_length} for {x_in.size(self.dim)} frames.")
            return True
        return False
    
    def get_resized_cond(self, cond_in: List[Dict], x_in: torch.Tensor,
                        window: IndexListContextWindow, device=None) -> List:
        """Resize conditioning to match context window for WAN models."""
        if cond_in is None:
            return None
        
        # For WAN, cond_in is a list with a single dict
        if not cond_in:
            logger.warning("Empty condition list passed to get_resized_cond")
            return []
        
        resized_cond = []
        for actual_cond in cond_in:
            if not isinstance(actual_cond, dict):
                logger.warning(f"Expected dict in cond_in, got {type(actual_cond)}")
                continue
                
            resized_actual_cond = actual_cond.copy()
            
            for key in actual_cond:
                try:
                    cond_item = actual_cond[key]
                    if key == 'seq_len':
                        # Special handling for seq_len - recalculate based on window size
                        # For WAN models, seq_len is calculated as:
                        # seq_len = math.ceil((lat_h * lat_w) / (patch_h * patch_w) * lat_f)
                        # We need to adjust lat_f based on the window size
                        if cond_item is not None:
                            # Get window size from the actual indices
                            window_frames = len(window.index_list)
                            full_frames = x_in.size(self.dim)
                            
                            # Handle all numeric types including numpy and tensor
                            if hasattr(cond_item, 'item'):  # Handle tensors
                                original_seq_len = cond_item.item()
                            elif isinstance(cond_item, (int, float, np.integer, np.floating)):
                                original_seq_len = int(cond_item)
                            else:
                                original_seq_len = cond_item
                            
                            # Scale seq_len proportionally
                            new_seq_len = math.ceil(original_seq_len * window_frames / full_frames)
                            resized_actual_cond[key] = new_seq_len
                            logger.debug(f"Resized seq_len from {original_seq_len} to {new_seq_len} (window {window_frames}/{full_frames} frames)")
                        else:
                            resized_actual_cond[key] = cond_item
                            logger.warning(f"seq_len is None in conditioning")
                    elif isinstance(cond_item, list):
                        # Handle lists of tensors (e.g., 'y' for I2V models)
                        resized_list = []
                        for item in cond_item:
                            if isinstance(item, torch.Tensor):
                                # Check if this tensor has frames that need slicing
                                if self.dim < item.ndim and item.size(self.dim) == x_in.size(self.dim):
                                    # Slice the tensor to match the window
                                    sliced_item = window.get_tensor(item, device)
                                    resized_list.append(sliced_item)
                                    logger.debug(f"Sliced list item in '{key}' from shape {item.shape} to {sliced_item.shape}")
                                else:
                                    resized_list.append(item.to(device) if device else item)
                            else:
                                resized_list.append(item)
                        resized_actual_cond[key] = resized_list
                    elif isinstance(cond_item, torch.Tensor):
                        # Check if tensor matches expected dimensions
                        if self.dim < cond_item.ndim and cond_item.size(self.dim) == x_in.size(self.dim):
                            actual_cond_item = window.get_tensor(cond_item)
                            resized_actual_cond[key] = actual_cond_item.to(device) if device else actual_cond_item
                        else:
                            resized_actual_cond[key] = cond_item.to(device) if device else cond_item
                    elif isinstance(cond_item, dict):
                        new_cond_item = cond_item.copy()
                        for cond_key, cond_value in new_cond_item.items():
                            if isinstance(cond_value, torch.Tensor):
                                if self.dim < cond_value.ndim and cond_value.size(self.dim) == x_in.size(self.dim):
                                    new_cond_item[cond_key] = window.get_tensor(cond_value, device)
                        resized_actual_cond[key] = new_cond_item
                    else:
                        resized_actual_cond[key] = cond_item
                finally:
                    pass  # Cleanup handled by garbage collector
            
            resized_cond.append(resized_actual_cond)
        
        return resized_cond
    
    def set_step(self, timestep: torch.Tensor, model_options: Dict[str, Any]):
        """Set current step based on timestep."""
        if "transformer_options" in model_options and "sample_sigmas" in model_options["transformer_options"]:
            sample_sigmas = model_options["transformer_options"]["sample_sigmas"]
            mask = torch.isclose(sample_sigmas, timestep, rtol=0.0001)
            matches = torch.nonzero(mask)
            if torch.numel(matches) > 0:
                self._step = int(matches[0].item())
        else:
            # Fallback: estimate step from timestep value
            self._step = 0
    
    def get_context_windows(self, model: Any, x_in: torch.Tensor,
                           model_options: Dict[str, Any]) -> List[IndexListContextWindow]:
        """Get list of context windows for processing."""
        full_length = x_in.size(self.dim)
        logger.info(f"Creating context windows for {full_length} latent frames (dim={self.dim})")
        context_windows = self.context_schedule.func(full_length, self, model_options)
        logger.info(f"Created {len(context_windows)} windows with schedule '{self.context_schedule.name}'")
        for i, window in enumerate(context_windows):
            logger.debug(f"  Window {i}: frames {window[0]}-{window[-1]} ({len(window)} frames)")
        context_windows = [IndexListContextWindow(window, dim=self.dim) for window in context_windows]
        return context_windows
    
    def execute(self, calc_cond_batch: Callable, model: Any, conds: List[List[Dict]],
               x_in: torch.Tensor, timestep: torch.Tensor, model_options: Dict[str, Any]):
        """Execute context window processing for WAN models."""
        logger.info(f"=== Context Window Execution Started ===")
        logger.info(f"Input shape: {x_in.shape}, dim={self.dim}")
        self.set_step(timestep, model_options)
        context_windows = self.get_context_windows(model, x_in, model_options)
        enumerated_context_windows = list(enumerate(context_windows))
        logger.info(f"Will process {len(enumerated_context_windows)} windows")
        
        # For WAN models, we work with a single condition, not multiple
        # Initialize accumulation tensor (single, not list)
        conds_final = torch.zeros_like(x_in)
        if self.fuse_method.name == ContextFuseMethods.RELATIVE:
            counts_final = torch.ones(get_shape_for_dim(x_in, self.dim), device=x_in.device)
        else:
            counts_final = torch.zeros(get_shape_for_dim(x_in, self.dim), device=x_in.device)
        biases_final = [0.0] * x_in.shape[self.dim]
        
        # Process each window
        for enum_window in enumerated_context_windows:
            results = self.evaluate_context_windows(calc_cond_batch, model, x_in, conds, timestep,
                                                   [enum_window], model_options)
            for result in results:
                self.combine_context_window_results(x_in, result.sub_conds_out, result.sub_conds,
                                                   result.window, result.window_idx,
                                                   len(enumerated_context_windows), timestep,
                                                   conds_final, counts_final, biases_final)
        
        # Finalize results - return single tensor for WAN
        logger.info(f"=== Context Window Execution Complete ===")
        logger.info(f"Final output shape: {conds_final.shape}")
        if self.fuse_method.name == ContextFuseMethods.RELATIVE:
            return [conds_final]  # Wrap in list for compatibility with caller
        else:
            # Normalize by counts
            conds_final /= counts_final
            return [conds_final]  # Wrap in list for compatibility with caller
    
    def evaluate_context_windows(self, calc_cond_batch: Callable, model: Any, x_in: torch.Tensor,
                                conds: List[List[Dict]], timestep: torch.Tensor,
                                enumerated_context_windows: List[Tuple[int, IndexListContextWindow]],
                                model_options: Dict[str, Any], device=None, first_device=None) -> List:
        """Evaluate WAN model on context windows."""
        results = []
        
        for window_idx, window in enumerated_context_windows:
            logger.info(f"Processing window {window_idx}/{len(enumerated_context_windows)}: "
                       f"frames {window.index_list[0]}-{window.index_list[-1]} "
                       f"({len(window.index_list)} frames)")
            
            # Update model options with current window
            model_options["transformer_options"] = model_options.get("transformer_options", {})
            model_options["transformer_options"]["context_window"] = window
            
            # Get subsections of inputs
            logger.debug(f"Input x_in shape before window extraction: {x_in.shape}")
            sub_x = window.get_tensor(x_in, device)
            logger.debug(f"Window extracted sub_x shape: {sub_x.shape}")
            sub_timestep = window.get_tensor(timestep, device, dim=0) if timestep.ndim > 0 else timestep
            
            # For WAN models, resize the single condition
            sub_conds = [self.get_resized_cond(cond, x_in, window, device) for cond in conds]
            
            # Calculate conditions for this window
            # calc_cond_batch now returns a single tensor for WAN
            logger.debug(f"Calling calc_cond_batch with sub_x shape: {sub_x.shape}")
            sub_conds_out = calc_cond_batch(model, sub_conds, sub_x, sub_timestep, model_options)
            logger.debug(f"calc_cond_batch returned shape: {sub_conds_out.shape if isinstance(sub_conds_out, torch.Tensor) else 'not a tensor'}")
            
            # Ensure it's a tensor (not wrapped)
            if not isinstance(sub_conds_out, torch.Tensor):
                if isinstance(sub_conds_out, (list, tuple)) and len(sub_conds_out) > 0:
                    sub_conds_out = sub_conds_out[0]
            
            if device is not None:
                sub_conds_out = sub_conds_out.to(x_in.device)
            
            results.append(ContextResults(window_idx, sub_conds_out, sub_conds, window))
        
        return results
    
    def combine_context_window_results(self, x_in: torch.Tensor, sub_conds_out: torch.Tensor,
                                      sub_conds: List[Dict], window: IndexListContextWindow,
                                      window_idx: int, total_windows: int, timestep: torch.Tensor,
                                      conds_final: torch.Tensor, counts_final: torch.Tensor,
                                      biases_final: List[float]):
        """Combine results from context windows for WAN models."""
        # sub_conds_out is now a single tensor, not a list
        if self.fuse_method.name == ContextFuseMethods.RELATIVE:
            # Relative weighting based on position
            for pos, idx in enumerate(window.index_list):
                bias = 1 - abs(idx - (window.index_list[0] + window.index_list[-1]) / 2) / \
                       ((window.index_list[-1] - window.index_list[0] + 1e-2) / 2)
                bias = max(1e-2, bias)
                
                bias_total = biases_final[idx]
                prev_weight = (bias_total / (bias_total + bias))
                new_weight = (bias / (bias_total + bias))
                
                idx_window = [slice(None)] * self.dim + [idx]
                pos_window = [slice(None)] * self.dim + [pos]
                
                conds_final[idx_window] = conds_final[idx_window] * prev_weight + \
                                         sub_conds_out[pos_window] * new_weight
                biases_final[idx] = bias_total + bias
        else:
            # Weight-based fusion
            weights = get_context_weights(window.context_length, x_in.shape[self.dim],
                                         window.index_list, self, sigma=timestep)
            weights_tensor = match_weights_to_dim(weights, x_in, self.dim, device=x_in.device)
            
            # Apply to single tensor (not list)
            window.add_window(conds_final, sub_conds_out * weights_tensor)
            window.add_window(counts_final, weights_tensor)


def match_weights_to_dim(weights: List[float], x_in: torch.Tensor, dim: int,
                         device=None) -> torch.Tensor:
    """Convert weights list to tensor with proper dimensions."""
    total_dims = len(x_in.shape)
    weights_tensor = torch.tensor(weights, device=device)
    
    for _ in range(dim):
        weights_tensor = weights_tensor.unsqueeze(0)
    for _ in range(total_dims - dim - 1):
        weights_tensor = weights_tensor.unsqueeze(-1)
    
    return weights_tensor


def get_shape_for_dim(x_in: torch.Tensor, dim: int) -> List[int]:
    """Get shape list for specific dimension."""
    total_dims = len(x_in.shape)
    shape = []
    
    for i in range(total_dims):
        if i == dim:
            shape.append(x_in.shape[dim])
        else:
            shape.append(1)
    
    return shape


class ContextSchedules:
    """Available context schedule types."""
    STATIC_STANDARD = "standard_static"
    BATCHED = "batched"
    SLIDING = "sliding"




def create_windows_static_standard(num_frames: int, handler: IndexListContextHandler,
                                  model_options: Dict[str, Any]) -> List[List[int]]:
    """Create static windows with fixed overlap."""
    windows = []
    if num_frames <= handler.context_length:
        windows.append(list(range(num_frames)))
        return windows
    
    delta = handler.context_length - handler.context_overlap
    for start_idx in range(0, num_frames, delta):
        ending = start_idx + handler.context_length
        if ending >= num_frames:
            final_delta = ending - num_frames
            final_start_idx = start_idx - final_delta
            windows.append(list(range(final_start_idx, final_start_idx + handler.context_length)))
            break
        windows.append(list(range(start_idx, start_idx + handler.context_length)))
    
    return windows


def create_windows_batched(num_frames: int, handler: IndexListContextHandler,
                          model_options: Dict[str, Any]) -> List[List[int]]:
    """Create non-overlapping batched windows."""
    windows = []
    if num_frames <= handler.context_length:
        windows.append(list(range(num_frames)))
        return windows
    
    for start_idx in range(0, num_frames, handler.context_length):
        windows.append(list(range(start_idx, min(start_idx + handler.context_length, num_frames))))
    
    return windows


def create_windows_sliding(num_frames: int, handler: IndexListContextHandler,
                           model_options: Dict[str, Any]) -> List[List[int]]:
    """Create a simple, robust sliding window schedule."""
    windows = []
    if num_frames <= handler.context_length:
        windows.append(list(range(num_frames)))
        return windows

    delta = handler.context_length - handler.context_overlap
    if delta <= 0:
        # Fallback to avoid infinite loop if overlap is too large
        logger.warning(f"Overlap {handler.context_overlap} is not smaller than context length {handler.context_length}. "
                       f"Falling back to simple batched windows.")
        for start_idx in range(0, num_frames, handler.context_length):
            windows.append(list(range(start_idx, min(start_idx + handler.context_length, num_frames))))
        return windows

    for start_idx in range(0, num_frames, delta):
        end_idx = start_idx + handler.context_length
        
        # If the window extends past the end, shift it back to align with the end
        if end_idx > num_frames:
            start_idx = max(0, num_frames - handler.context_length)
            end_idx = num_frames
        
        window = list(range(start_idx, end_idx))
        
        # Avoid adding duplicate windows if the last window was already added
        if not windows or windows[-1] != window:
            windows.append(window)
        
        # Stop if we've reached the end of the frames
        if end_idx == num_frames:
            break
            
    return windows


# Mapping of schedule names to functions
CONTEXT_MAPPING = {
    ContextSchedules.STATIC_STANDARD: create_windows_static_standard,
    ContextSchedules.BATCHED: create_windows_batched,
    ContextSchedules.SLIDING: create_windows_sliding,
}


def get_matching_context_schedule(context_schedule: str) -> ContextSchedule:
    """Get context schedule by name."""
    func = CONTEXT_MAPPING.get(context_schedule, None)
    if func is None:
        raise ValueError(f"Unknown context_schedule '{context_schedule}'.")
    return ContextSchedule(context_schedule, func)


def get_context_weights(length: int, full_length: int, idxs: List[int],
                       handler: IndexListContextHandler, sigma: torch.Tensor = None) -> List[float]:
    """Get fusion weights for context window."""
    return handler.fuse_method.func(length, sigma=sigma, handler=handler,
                                   full_length=full_length, idxs=idxs)


def create_weights_flat(length: int, **kwargs) -> List[float]:
    """Create flat (uniform) weights."""
    return [1.0] * length


def create_weights_pyramid(length: int, **kwargs) -> List[float]:
    """Create pyramid weights (higher in center, lower at edges)."""
    if length % 2 == 0:
        max_weight = length // 2
        weight_sequence = list(range(1, max_weight + 1, 1)) + list(range(max_weight, 0, -1))
    else:
        max_weight = (length + 1) // 2
        weight_sequence = list(range(1, max_weight, 1)) + [max_weight] + list(range(max_weight - 1, 0, -1))
    return weight_sequence


def create_weights_overlap_linear(length: int, full_length: int, idxs: List[int],
                                 handler: IndexListContextHandler, **kwargs) -> List[float]:
    """Create linear blending weights for overlapping regions."""
    weights_torch = torch.ones((length))
    
    # Blend left-side on all except first window
    if min(idxs) > 0:
        ramp_up = torch.linspace(1e-37, 1, handler.context_overlap)
        weights_torch[:handler.context_overlap] = ramp_up
    
    # Blend right-side on all except last window
    if max(idxs) < full_length - 1:
        ramp_down = torch.linspace(1, 1e-37, handler.context_overlap)
        weights_torch[-handler.context_overlap:] = ramp_down
    
    return weights_torch.tolist()


class ContextFuseMethods:
    """Available fusion methods."""
    FLAT = "flat"
    PYRAMID = "pyramid"
    RELATIVE = "relative"
    OVERLAP_LINEAR = "overlap-linear"
    
    LIST = [PYRAMID, FLAT, OVERLAP_LINEAR]
    LIST_STATIC = [PYRAMID, RELATIVE, FLAT, OVERLAP_LINEAR]


# Mapping of fusion method names to functions
FUSE_MAPPING = {
    ContextFuseMethods.FLAT: create_weights_flat,
    ContextFuseMethods.PYRAMID: create_weights_pyramid,
    ContextFuseMethods.RELATIVE: create_weights_pyramid,
    ContextFuseMethods.OVERLAP_LINEAR: create_weights_overlap_linear,
}


def get_matching_fuse_method(fuse_method: str) -> ContextFuseMethod:
    """Get fusion method by name."""
    func = FUSE_MAPPING.get(fuse_method, None)
    if func is None:
        raise ValueError(f"Unknown fuse_method '{fuse_method}'.")
    return ContextFuseMethod(fuse_method, func)




class WanContextWindowsHandler:
    """WAN-specific context windows handler with optimizations for video generation."""
    
    def __init__(self, context_length: int = 81, context_overlap: int = 30,
                 context_schedule: str = "standard_static",
                 context_stride: int = 1, closed_loop: bool = False,
                 fuse_method: str = "pyramid"):
        """Initialize WAN context windows handler.
        
        Args:
            context_length: Length of context window in frames (default 81)
            context_overlap: Overlap between windows in frames (default 30)
            context_schedule: Schedule type for windows
            context_stride: Stride for uniform schedules
            closed_loop: Whether to close loop for cyclic videos
            fuse_method: Method for fusing window results
        """
        # Store original frame counts
        self.context_length_frames = context_length
        self.context_overlap_frames = context_overlap
        
        # Adjust for VAE stride (4x temporal compression)
        self.context_length = max(((context_length - 1) // 4) + 1, 1)
        self.context_overlap = max(((context_overlap - 1) // 4) + 1, 0)
        
        # Create handler with dimension 2 (temporal for [B, C, F, H, W])
        schedule = get_matching_context_schedule(context_schedule)
        method = get_matching_fuse_method(fuse_method)
        
        # Set dim based on whether we expect batch dimension
        # For [C, F, H, W] tensor, F is at index 1
        # For [B, C, F, H, W] tensor, F is at index 2
        # We'll use dim=1 for unbatched tensors
        self.handler = IndexListContextHandler(
            context_schedule=schedule,
            fuse_method=method,
            context_length=self.context_length,
            context_overlap=self.context_overlap,
            context_stride=context_stride,
            closed_loop=closed_loop,
            dim=1  # Temporal dimension for [C, F, H, W] tensors
        )
        
        logger.info(f"WAN Context Windows initialized: {self.context_length_frames} frames "
                   f"({self.context_length} latent), {self.context_overlap_frames} overlap "
                   f"({self.context_overlap} latent), schedule={context_schedule}, "
                   f"fuse={fuse_method}")
    
    def should_use(self, latent: torch.Tensor) -> bool:
        """Check if context windows should be used for given latent."""
        return self.handler.should_use_context(None, [], latent, torch.tensor(0), {})
    
    def apply_to_model(self, model: Any, model_options: Dict[str, Any]) -> Any:
        """Apply context window handler to model options."""
        model_options["context_handler"] = self.handler
        return model