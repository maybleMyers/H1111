import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Global CUDA stream for asynchronous weight transfers
TRANSFER_STREAM = torch.cuda.Stream()

# Maximum number of in-flight transfers to prevent unbounded memory growth
MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT", 2))

# Queue to track pending transfer events for synchronization
PENDING_EVENTS = []

class BouncingLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device="cuda"):
        global PENDING_EVENTS

        # Start async transfer on dedicated stream
        with torch.cuda.stream(TRANSFER_STREAM):
            # Transfer weights to GPU
            w = weight_cpu.to(device, non_blocking=True)
            b = bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None

            # CRITICAL: Cast to computation dtype INSIDE the transfer stream
            # This ensures dtype conversion happens with proper synchronization
            w_compute = w.to(x.dtype)
            b_compute = b.to(x.dtype) if b is not None else None

            # Record completion event
            evt = torch.cuda.Event()
            evt.record(TRANSFER_STREAM)
            PENDING_EVENTS.append(evt)

        # Throttle concurrent transfers
        if len(PENDING_EVENTS) > MAX_INFLIGHT:
            PENDING_EVENTS[0].synchronize()
            PENDING_EVENTS.pop(0)

        # Make compute stream wait for transfer completion
        torch.cuda.current_stream().wait_event(evt)

        # Ensure the event is truly synchronized
        # This is needed for proper memory coherency at high resolutions
        evt.synchronize()

        # Perform computation with properly synchronized weights
        out = F.linear(x, w_compute, b_compute)

        # Save for backward (though not used in inference)
        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        ctx.w_compute = w_compute
        ctx.b_compute = b_compute
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # Simplified backward pass - primarily for training, not used in inference
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device

        # Use cached weights if available from forward pass
        if hasattr(ctx, 'w_compute'):
            w_compute = ctx.w_compute
        else:
            # Transfer weights again if needed
            global PENDING_EVENTS
            with torch.cuda.stream(TRANSFER_STREAM):
                w = weight_cpu.to(device, non_blocking=True)
                w_compute = w.to(grad_out.dtype)
                evt = torch.cuda.Event()
                evt.record(TRANSFER_STREAM)
                PENDING_EVENTS.append(evt)

            if len(PENDING_EVENTS) > MAX_INFLIGHT:
                PENDING_EVENTS[0].synchronize()
                PENDING_EVENTS.pop(0)

            torch.cuda.current_stream().wait_event(evt)
            evt.synchronize()

        # Compute gradients
        grad_input = grad_out @ w_compute.to(grad_out.dtype)
        grad_weight = grad_out.t() @ x
        grad_bias = grad_out.sum(0) if bias_cpu is not None else None

        # Return gradients (keep on GPU for now - training would need CPU gradients)
        return grad_input.to(x.dtype), grad_weight.to(weight_cpu.dtype), grad_bias, None


class CPUBouncingLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device="cuda"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        weight_tensor = torch.empty(out_features, in_features, device="cpu", dtype=torch.float32)
        weight_tensor.share_memory_()
        self.weight = nn.Parameter(weight_tensor)

        if bias:
            bias_tensor = torch.empty(out_features, device="cpu", dtype=torch.float32)
            bias_tensor.share_memory_()
            self.bias = nn.Parameter(bias_tensor)
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

        # Register hook to ensure weights stay on CPU after state_dict loads
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    @staticmethod
    def _load_state_dict_pre_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Ensure loaded weights are on CPU with shared memory"""
        weight_key = f"{prefix}weight"
        bias_key = f"{prefix}bias"

        if weight_key in state_dict:
            weight = state_dict[weight_key]
            if not weight.is_cpu:
                state_dict[weight_key] = weight.cpu()
            # Ensure shared memory after loading
            if not state_dict[weight_key].is_shared():
                state_dict[weight_key].share_memory_()

        if bias_key in state_dict:
            bias = state_dict[bias_key]
            if not bias.is_cpu:
                state_dict[bias_key] = bias.cpu()
            if not state_dict[bias_key].is_shared():
                state_dict[bias_key].share_memory_()

    def _apply(self, fn):
        # Override to ensure weights ALWAYS stay on CPU
        # This prevents .to(device) calls from moving weights to GPU
        super()._apply(lambda t: t if t is self.weight or t is self.bias else fn(t))

        # Ensure weights are on CPU with shared memory
        if self.weight is not None:
            if not self.weight.is_cpu:
                self.weight.data = self.weight.data.cpu()
            if not self.weight.is_shared():
                self.weight.share_memory_()

        if self.bias is not None:
            if not self.bias.is_cpu:
                self.bias.data = self.bias.data.cpu()
            if not self.bias.is_shared():
                self.bias.share_memory_()

        return self

    def forward(self, x):
        return BouncingLinearFn.apply(x, self.weight, self.bias, self.device)

Linear = CPUBouncingLinear