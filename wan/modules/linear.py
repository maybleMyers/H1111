import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class BouncingLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device="cuda"):
        # Debug: Check if weights have valid values
        if not hasattr(BouncingLinearFn, '_debug_printed'):
            print(f"BouncingLinear Debug:")
            print(f"  Input x: shape={x.shape}, dtype={x.dtype}, device={x.device}")
            print(f"  Weight CPU: shape={weight_cpu.shape}, dtype={weight_cpu.dtype}, mean={weight_cpu.mean():.6f}, std={weight_cpu.std():.6f}")
            print(f"  Weight has NaN: {torch.isnan(weight_cpu).any()}, has Inf: {torch.isinf(weight_cpu).any()}")
            BouncingLinearFn._debug_printed = True

        # SIMPLIFIED APPROACH: Fully synchronous transfers
        # This ensures weights are fully transferred before use
        w = weight_cpu.to(device, non_blocking=False)
        b = bias_cpu.to(device, non_blocking=False) if bias_cpu is not None else None

        # Correctly cast weight to match float32 input activation
        w_compute = w.to(x.dtype)
        b_compute = b.to(x.dtype) if b is not None else None
        out = F.linear(x, w_compute, b_compute)

        # Save tensors for backward - keep weights on GPU for now
        ctx.save_for_backward(x, w, b)
        ctx.weight_cpu = weight_cpu
        ctx.bias_cpu = bias_cpu
        ctx.device = device
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w, b = ctx.saved_tensors
        weight_cpu = ctx.weight_cpu
        bias_cpu = ctx.bias_cpu

        # Weights are already on GPU from forward pass

        # For grad_input, we need float32 @ bfloat16 -> upcast weight
        w_compute_grad_input = w.to(grad_out.dtype)
        grad_input = grad_out @ w_compute_grad_input

        # For grad_weight, we need float32.T @ bfloat16
        grad_weight = grad_out.t() @ x

        grad_bias = grad_out.sum(0) if b is not None else None

        # Move gradients to CPU to match parameter storage
        grad_weight_cpu = grad_weight.to(weight_cpu.dtype).cpu()
        grad_bias_cpu = grad_bias.to(bias_cpu.dtype).cpu() if grad_bias is not None else None

        return grad_input.to(x.dtype), grad_weight_cpu, grad_bias_cpu, None


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