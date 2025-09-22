from concurrent.futures import ThreadPoolExecutor
import gc
import time
from typing import Optional
import torch
import torch.nn as nn


def clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def swap_weight_devices_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module, staging_buffers_a=None, staging_buffers_b=None, use_pinned_memory=False):
    """Original swap function without staging buffers (for backward compatibility)"""
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []

    # This is not working for all cases (e.g. SD3), so we need to find the corresponding modules
    # for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
    #     print(module_to_cpu.__class__, module_to_cuda.__class__)
    #     if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
    #         weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))
            else:
                if module_to_cuda.weight.data.device.type != device.type:
                    # print(
                    #     f"Module {module_to_cuda_name} not found in CPU model or shape mismatch, so not swapping and moving to device"
                    # )
                    module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

    # Use staging buffer version if buffers are requested
    if use_pinned_memory or staging_buffers_a is not None or staging_buffers_b is not None:
        return swap_weight_devices_cuda_staged(device, weight_swap_jobs, staging_buffers_a, staging_buffers_b, use_pinned_memory)

    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # cuda to cpu
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        stream.synchronize()

        # cpu to cuda
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

    stream.synchronize()
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    return None, None  # Return None staging buffers for compatibility


def swap_weight_devices_cuda_staged(device: torch.device, weight_swap_jobs, staging_buffers_a=None, staging_buffers_b=None, use_pinned_memory=False):
    """Optimized swap function using staging buffers with optional pinned memory"""
    if not weight_swap_jobs:
        return staging_buffers_a, staging_buffers_b

    # Initialize staging buffers if needed
    if staging_buffers_a is None:
        staging_buffers_a = []
        staging_buffers_b = []
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            if use_pinned_memory:
                try:
                    # Create staging buffer as pinned memory (as shared GPU RAM)
                    sbuf_a = torch.empty_like(cuda_data_view, device="cpu").pin_memory()
                    sbuf_b = torch.empty_like(cuda_data_view, device="cpu").pin_memory()
                    staging_buffers_a.append(sbuf_a)
                    staging_buffers_b.append(sbuf_b)
                except Exception as e:
                    print(f"Warning: Failed to allocate pinned memory, falling back to regular memory: {e}")
                    use_pinned_memory = False
                    # Clear any partially created buffers and recreate without pinning
                    staging_buffers_a = []
                    staging_buffers_b = []
                    for _, _, cuda_view, _ in weight_swap_jobs:
                        staging_buffers_a.append(torch.empty_like(cuda_view, device="cpu"))
                        staging_buffers_b.append(torch.empty_like(cuda_view, device="cpu"))
                    break
            else:
                staging_buffers_a.append(torch.empty_like(cuda_data_view, device="cpu"))
                staging_buffers_b.append(torch.empty_like(cuda_data_view, device="cpu"))

    # Create events for synchronization
    events = [torch.cuda.Event() for _ in weight_swap_jobs]

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Stage 1: Transfer from CUDA to staging buffer A and CPU to staging buffer B
        for event, sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
            events, staging_buffers_a, staging_buffers_b, weight_swap_jobs
        ):
            # cuda to staging buffer A
            sbuf_a.copy_(cuda_data_view.data, non_blocking=True)
            event.record(stream)

            # cpu to staging buffer B (cpu to pinned cpu)
            sbuf_b.copy_(module_to_cuda.weight.data)

            # staging buffer B to cuda
            cuda_data_view.copy_(sbuf_b, non_blocking=True)

        # Stage 2: Complete the transfers after synchronization
        for event, sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
            events, staging_buffers_a, staging_buffers_b, weight_swap_jobs
        ):
            # staging buffer A to cpu
            event.synchronize()
            cpu_data_view.copy_(sbuf_a)  # pinned cpu to cpu

            # update references
            module_to_cuda.weight.data = cuda_data_view
            module_to_cpu.weight.data = cpu_data_view

    stream.synchronize()  # synchronize staging buffer B to cuda
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    return staging_buffers_a, staging_buffers_b


def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    not tested
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    # device to cpu
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    synchronize_device()

    # cpu to device
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    synchronize_device()


def weighs_to_device(layer: nn.Module, device: torch.device):
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data = module.weight.data.to(device, non_blocking=True)


class Offloader:
    """
    common offloading class
    """

    def __init__(self, block_type: str, num_blocks: int, blocks_to_swap: int, device: torch.device, debug: bool = False, use_pinned_memory: bool = False):
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.debug = debug
        self.use_pinned_memory = use_pinned_memory

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"

        # Storage for staging buffers (per block)
        self.staging_buffers_a = {}
        self.staging_buffers_b = {}

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            # Get or create staging buffers for this block pair
            block_id = id(block_to_cuda)
            staging_a = self.staging_buffers_a.get(block_id)
            staging_b = self.staging_buffers_b.get(block_id)

            # Call swap function with staging buffers
            staging_a, staging_b = swap_weight_devices_cuda(
                self.device, block_to_cpu, block_to_cuda,
                staging_a, staging_b, self.use_pinned_memory
            )

            # Store staging buffers for reuse
            if staging_a is not None:
                self.staging_buffers_a[block_id] = staging_a
                self.staging_buffers_b[block_id] = staging_b
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(
                    f"[{self.block_type}] Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'}"
                )

            self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(f"[{self.block_type}] Moved blocks {bidx_to_cpu} and {bidx_to_cuda} in {time.perf_counter()-start_time:.2f}s")
            return bidx_to_cpu, bidx_to_cuda  # , event

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
        )

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"[{self.block_type}] Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        if self.debug:
            print(f"[{self.block_type}] Waited for block {block_idx}: {time.perf_counter()-start_time:.2f}s")


class ModelOffloader(Offloader):
    """
    supports forward offloading
    """

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        debug: bool = False,
        use_pinned_memory: bool = False,
    ):
        super().__init__(block_type, num_blocks, blocks_to_swap, device, debug, use_pinned_memory)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward  # forward only offloading: can be changed to True for inference

        if self.supports_backward:
            # register backward hooks
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def set_forward_only(self, forward_only: bool):
        self.forward_only = forward_only

    def __del__(self):
        if self.supports_backward:
            for handle in self.remove_handles:
                handle.remove()

    def create_backward_hook(self, blocks: list[nn.Module], block_index: int) -> Optional[callable]:
        # -1 for 0-based index
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        # create  hook
        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward")

        for b in blocks[0 : self.num_blocks - self.blocks_to_swap]:
            b.to(self.device)
            weighs_to_device(b, self.device)  # make sure weights are on device

        for b in blocks[self.num_blocks - self.blocks_to_swap :]:
            b.to(self.device)  # move block to device first
            weighs_to_device(b, "cpu")  # make sure weights are on cpu

        synchronize_device(self.device)
        clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int):
        # check if blocks_to_swap is enabled
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        # if supports_backward and backward is enabled, we swap blocks more than blocks_to_swap in backward pass
        if not self.forward_only and block_idx >= self.blocks_to_swap:
            return

        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        block_idx_to_cuda = block_idx_to_cuda % self.num_blocks  # this works for forward-only offloading
        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
