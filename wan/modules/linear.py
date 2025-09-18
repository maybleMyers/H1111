"""
CPU Linear Module

A memory-efficient linear layer implementation that keeps parameters on CPU
and transfers them to GPU on-demand using asynchronous CUDA streams.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

TRANSFER_STREAM = torch.cuda.Stream()
MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT", 2))
PENDING_EVENTS = []

class BouncingLinearFn(torch.autograd.Function):
    """
    Custom autograd function implementing the bouncing linear operation.
    """

    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device="cuda"):
        global PENDING_EVENTS

        with torch.cuda.stream(TRANSFER_STREAM):
            w = weight_cpu.to(device, non_blocking=True)
            b = bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
            evt = torch.cuda.Event()
            evt.record(TRANSFER_STREAM)
            PENDING_EVENTS.append(evt)

        if len(PENDING_EVENTS) > MAX_INFLIGHT:
            PENDING_EVENTS[0].synchronize()
            PENDING_EVENTS.pop(0)

        torch.cuda.current_stream().wait_event(evt)

        w_compute = w.to(x.dtype)
        b_compute = b.to(x.dtype) if b is not None else None
        out = F.linear(x, w_compute, b_compute)

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        return out

    @staticmethod
    def backward(ctx, grad_out):
        global PENDING_EVENTS
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device

        with torch.cuda.stream(TRANSFER_STREAM):
            w = weight_cpu.to(device, non_blocking=True)
            evt = torch.cuda.Event()
            evt.record(TRANSFER_STREAM)
            PENDING_EVENTS.append(evt)

        if len(PENDING_EVENTS) > MAX_INFLIGHT:
            PENDING_EVENTS[0].synchronize()
            PENDING_EVENTS.pop(0)

        torch.cuda.current_stream().wait_event(evt)

        w_compute = w.to(grad_out.dtype)
        x_compute = x.to(grad_out.dtype)

        grad_input = grad_out @ w_compute
        grad_weight = grad_out.t() @ x_compute
        grad_bias = grad_out.sum(0) if bias_cpu is not None else None

        # Return gradients with the correct dtype for the CPU parameters
        return grad_input.to(x.dtype), grad_weight.to(weight_cpu.dtype), grad_bias.to(bias_cpu.dtype) if bias_cpu is not None else None, None


class CPUBouncingLinear(nn.Module):
    """
    Linear layer with CPU-stored parameters that bounce to GPU on demand.
    """
    def __init__(self, in_features, out_features, bias=True, device="cuda"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        weight_tensor = torch.empty(out_features, in_features, device="cpu")
        weight_tensor.share_memory_()
        self.weight = nn.Parameter(weight_tensor)

        if bias:
            bias_tensor = torch.empty(out_features, device="cpu")
            bias_tensor.share_memory_()
            self.bias = nn.Parameter(bias_tensor)
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    # Best practice: Add _apply to prevent accidental moves to GPU
    def _apply(self, fn):
        super()._apply(fn)
        if self.weight is not None:
            self.weight.data = self.weight.data.cpu()
            if self.weight.grad is not None:
                self.weight.grad.data = self.weight.grad.data.cpu()
        if self.bias is not None:
            self.bias.data = self.bias.data.cpu()
            if self.bias.grad is not None:
                self.bias.grad.data = self.bias.grad.data.cpu()
        return self

    def forward(self, x):
        return BouncingLinearFn.apply(x, self.weight, self.bias, self.device)

Linear = CPUBouncingLinear