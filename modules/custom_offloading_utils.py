from collections import deque
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


def swap_weight_devices_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    Swap weights between two layers, moving layer_to_cpu's weights to CPU and layer_to_cuda's weights to GPU.
    Uses buffer reuse for large weight tensors to minimize GPU memory allocation.
    Also handles biases and other parameters with simple device transfers.
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    other_param_jobs = []  # For biases and other non-weight parameters

    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
        if module_to_cpu is None:
            continue

        # Handle weight parameter with buffer reuse
        if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
            if module_to_cpu.weight is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))
            elif module_to_cuda.weight.data.device.type != device.type:
                module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

        # Handle all other parameters (bias, etc.) - collect them for simple transfer
        for param_name, param in module_to_cuda.named_parameters(recurse=False):
            if param_name == "weight":  # Already handled above
                continue
            if param is not None:
                cpu_param = getattr(module_to_cpu, param_name, None)
                if cpu_param is not None:
                    other_param_jobs.append((module_to_cpu, module_to_cuda, param_name, cpu_param.data, param.data))

    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # cuda to cpu - weights (with buffer reuse)
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        # cuda to cpu - other params (simple transfer)
        for module_to_cpu, module_to_cuda, param_name, cpu_param_data, cuda_param_data in other_param_jobs:
            # If the GPU module's param is on GPU, move to CPU
            if cpu_param_data.device.type == device.type:
                setattr(module_to_cpu, param_name + "_data_backup", cpu_param_data)  # temporary backup
                getattr(module_to_cpu, param_name).data = cpu_param_data.to("cpu", non_blocking=True)

        stream.synchronize()

        # cpu to cuda - weights (reuse GPU buffer)
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

        # cpu to cuda - other params (simple transfer, reuse buffer if available)
        for module_to_cpu, module_to_cuda, param_name, cpu_param_data, cuda_param_data in other_param_jobs:
            backup_key = param_name + "_data_backup"
            if hasattr(module_to_cpu, backup_key):
                # Reuse the GPU buffer from the module that moved to CPU
                gpu_buffer = getattr(module_to_cpu, backup_key)
                gpu_buffer.copy_(cuda_param_data, non_blocking=True)
                getattr(module_to_cuda, param_name).data = gpu_buffer
                delattr(module_to_cpu, backup_key)
            else:
                # Fallback: simple transfer to GPU
                getattr(module_to_cuda, param_name).data = cuda_param_data.to(device, non_blocking=True)

    stream.synchronize()
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value


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
    """Move all parameters (weights, biases, and any other parameters) to the specified device."""
    for module in layer.modules():
        # Move all named parameters, not just weights
        for param_name, param in list(module.named_parameters(recurse=False)):
            if param is not None:
                param.data = param.data.to(device, non_blocking=True)


class Offloader:
    """
    common offloading class
    """

    def __init__(self, block_type: str, num_blocks: int, blocks_to_swap: int, device: torch.device, debug: bool = False):
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.debug = debug

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
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

        # Ensure CUDA operations from swap are complete
        if self.cuda_available:
            torch.cuda.synchronize()
            if self.debug:
                print(f"[{self.block_type}] Swap complete for block {block_idx}: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

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
        fixed_resident: bool = False,
        auto_tune_reserve_gb: Optional[float] = None,
    ):
        super().__init__(block_type, num_blocks, blocks_to_swap, device, debug)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward  # forward only offloading: can be changed to True for inference

        # Fixed-resident inference mode (opt-in): resident blocks never leave the GPU; the remaining
        # blocks are streamed CPU->GPU through a small set of reusable staging buffers. Weights are
        # immutable during inference, so the GPU->CPU copy of the classic rolling swap is skipped
        # entirely, cutting PCIe traffic from 2*num_blocks to blocks_to_swap transfers per forward.
        self.fixed_resident = fixed_resident and not supports_backward and self.cuda_available and blocks_to_swap > 0
        self.auto_tune_reserve_gb = auto_tune_reserve_gb
        if self.fixed_resident:
            self.num_staging = 2
            self.copy_stream = torch.cuda.Stream(device)
            self.cpu_masters = {}  # block_idx -> list of (param, cpu master tensor)
            self.staging = None  # num_staging lists of GPU tensors, each shaped like one block
            self.staging_release_evt = [None] * self.num_staging
            self.buf_for_block = {}  # block_idx -> staging set index
            self.forward_count = 0
            self.block_bytes = 0
            self.auto_tuned = auto_tune_reserve_gb is None

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

        if self.fixed_resident:
            self._prepare_fixed(blocks)
            return

        num_resident = self.num_blocks - self.blocks_to_swap
        if self.debug:
            print(f"[{self.block_type}] Prepare block devices: {num_resident} blocks on GPU, {self.blocks_to_swap} blocks on CPU")

        # Move only the first (num_blocks - blocks_to_swap) blocks to GPU
        # These are the blocks that will be on GPU initially
        for i, b in enumerate(blocks[0 : num_resident]):
            b.to(self.device)
            weighs_to_device(b, self.device)  # make sure all params are on device
            if self.debug and self.device.type == "cuda":
                print(f"  Block {i} moved to GPU. GPU memory: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB")

        # Keep the remaining blocks on CPU - they will be swapped in during forward pass
        # The swap mechanism will reuse GPU buffers from blocks moving to CPU
        for i, b in enumerate(blocks[num_resident:]):
            # Ensure all parameters are on CPU (they should already be from model loading)
            weighs_to_device(b, "cpu")

        synchronize_device(self.device)
        clean_memory_on_device(self.device)

        if self.debug and self.device.type == "cuda":
            print(f"[{self.block_type}] After prepare: GPU memory: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB")

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        if self.fixed_resident:
            future = self.futures.pop(block_idx, None)
            if future is None:
                return  # resident block, nothing to wait for
            done_evt = future.result()
            torch.cuda.current_stream().wait_event(done_evt)
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int):
        # check if blocks_to_swap is enabled
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.fixed_resident:
            self._release_fixed_block(block_idx)
            return

        # if supports_backward and backward is enabled, we swap blocks more than blocks_to_swap in backward pass
        if not self.forward_only and block_idx >= self.blocks_to_swap:
            return

        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        block_idx_to_cuda = block_idx_to_cuda % self.num_blocks  # this works for forward-only offloading
        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)

    # ----- fixed-resident inference mode -----

    def begin_forward(self, blocks: list[nn.Module]):
        """Call once at the start of each forward pass (fixed-resident mode only)."""
        if not self.fixed_resident or self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.forward_count += 1
        if not self.auto_tuned and self.forward_count == 2:
            # after one full forward, allocator reserve covers activations; measured free VRAM is real headroom
            self._auto_tune_promote()
        num_resident = self.num_blocks - self.blocks_to_swap
        for j in range(min(self.num_staging, self.blocks_to_swap)):
            self._submit_fixed_copy(num_resident + j, j)

    def _prepare_fixed(self, blocks: list[nn.Module]):
        # drain anything left over from a previous prepare / interrupted forward
        for future in self.futures.values():
            try:
                future.result()
            except Exception:
                pass
        self.futures.clear()
        self.buf_for_block.clear()
        self.staging_release_evt = [None] * self.num_staging
        torch.cuda.synchronize()

        num_resident = self.num_blocks - self.blocks_to_swap

        # resident blocks live on GPU permanently
        for b in blocks[:num_resident]:
            b.to(self.device)
            weighs_to_device(b, self.device)

        # streamed blocks: params stay on CPU as immutable masters; buffers (fp8 scales etc.) are tiny -> GPU
        template = None
        uniform = True
        self.cpu_masters = {}
        for i in range(num_resident, self.num_blocks):
            b = blocks[i]
            masters = []
            signature = []
            for name, p in b.named_parameters():
                if p.data.device.type != "cpu":
                    p.data = p.data.to("cpu")
                masters.append((p, p.data))
                signature.append((name, tuple(p.shape), p.dtype))
            for m in b.modules():
                for k, v in m._buffers.items():
                    if v is not None and v.device.type != self.device.type:
                        m._buffers[k] = v.to(self.device)
            self.cpu_masters[i] = masters
            if template is None:
                template = signature
            elif signature != template:
                uniform = False

        if not uniform or not template:
            print(f"[{self.block_type}] blocks are not uniform; falling back to standard block swap")
            self.fixed_resident = False
            self.prepare_block_devices_before_forward(blocks)
            return

        self.block_bytes = sum(t.nbytes for _, t in self.cpu_masters[num_resident])
        if self.staging is None:
            self.staging = [
                [torch.empty(shape, dtype=dtype, device=self.device) for _, shape, dtype in template]
                for _ in range(self.num_staging)
            ]

        synchronize_device(self.device)
        clean_memory_on_device(self.device)
        print(
            f"[{self.block_type}] fixed-resident block swap: {num_resident} blocks resident on GPU, "
            f"{self.blocks_to_swap} blocks streamed from CPU via {self.num_staging} staging buffers "
            f"({self.block_bytes / 1024**3:.2f} GB per block, upload-only)"
        )

    def _submit_fixed_copy(self, block_idx: int, set_idx: int):
        masters = self.cpu_masters[block_idx]
        bufs = self.staging[set_idx]
        release_evt = self.staging_release_evt[set_idx]
        self.staging_release_evt[set_idx] = None

        leftover = self.futures.pop(block_idx, None)
        if leftover is not None:
            leftover.result()

        def copy_job():
            with torch.cuda.stream(self.copy_stream):
                if release_evt is not None:
                    # don't overwrite the buffer while the previous occupant's kernels may still read it
                    self.copy_stream.wait_event(release_evt)
                for (param, master), buf in zip(masters, bufs):
                    buf.copy_(master, non_blocking=True)
                    param.data = buf
                done_evt = torch.cuda.Event()
                done_evt.record(self.copy_stream)
            return done_evt

        self.buf_for_block[block_idx] = set_idx
        self.futures[block_idx] = self.thread_pool.submit(copy_job)

    def _release_fixed_block(self, block_idx: int):
        set_idx = self.buf_for_block.pop(block_idx, None)
        if set_idx is None:
            return  # resident block

        # block may have been skipped (e.g. SLG): make sure its upload was issued before reusing the buffer
        future = self.futures.pop(block_idx, None)
        if future is not None:
            torch.cuda.current_stream().wait_event(future.result())

        # weights are immutable during inference: just point the params back at their CPU masters, no copy
        for param, master in self.cpu_masters[block_idx]:
            param.data = master

        evt = torch.cuda.Event()
        evt.record()  # current stream: fires once this block's kernels are done with the staging buffer
        self.staging_release_evt[set_idx] = evt

        next_idx = block_idx + self.num_staging
        if next_idx < self.num_blocks:
            self._submit_fixed_copy(next_idx, set_idx)

    def _auto_tune_promote(self):
        self.auto_tuned = True
        if self.block_bytes <= 0 or self.blocks_to_swap <= 1:
            return
        try:
            free_bytes, _total = torch.cuda.mem_get_info(self.device)
        except Exception:
            return
        reserve = int((self.auto_tune_reserve_gb or 0.0) * 1024**3)
        n = int((free_bytes - reserve) // self.block_bytes)
        n = max(0, min(n, self.blocks_to_swap - 1))
        if n <= 0:
            print(
                f"[{self.block_type}] auto block swap: no headroom to promote blocks "
                f"(free {free_bytes / 1024**3:.2f} GB, reserve {reserve / 1024**3:.2f} GB)"
            )
            return
        num_resident = self.num_blocks - self.blocks_to_swap
        promoted = 0
        for i in range(num_resident, num_resident + n):
            try:
                for param, master in self.cpu_masters[i]:
                    param.data = master.to(self.device)
            except torch.cuda.OutOfMemoryError:
                # roll back the partially-promoted block and stop promoting
                for param, master in self.cpu_masters[i]:
                    param.data = master
                torch.cuda.empty_cache()
                break
            self.cpu_masters.pop(i)
            promoted += 1
        if promoted == 0:
            return
        self.blocks_to_swap -= promoted
        print(
            f"[{self.block_type}] auto block swap: promoted {promoted} blocks to GPU "
            f"(free VRAM was {free_bytes / 1024**3:.2f} GB, reserve {reserve / 1024**3:.2f} GB); "
            f"now swapping {self.blocks_to_swap} of {self.num_blocks} blocks"
        )

    def reset_fixed_state(self):
        """Drain pending uploads and point every streamed param back at its CPU master.

        Used to get back to a clean state after an aborted forward (e.g. CUDA OOM).
        """
        if not self.fixed_resident:
            return
        for future in self.futures.values():
            try:
                future.result()
            except Exception:
                pass
        self.futures.clear()
        self.buf_for_block.clear()
        self.staging_release_evt = [None] * self.num_staging
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        for masters in self.cpu_masters.values():
            for param, master in masters:
                param.data = master

    def demote_blocks(self, blocks: list[nn.Module], n: int) -> int:
        """Move the last n resident blocks back to CPU streaming (OOM recovery). Returns count demoted."""
        if not self.fixed_resident:
            return 0
        num_resident = self.num_blocks - self.blocks_to_swap
        n = max(0, min(n, num_resident - 1))
        if n == 0:
            return 0
        for i in range(num_resident - n, num_resident):
            masters = []
            for name, p in blocks[i].named_parameters():
                p.data = p.data.to("cpu")
                masters.append((p, p.data))
            self.cpu_masters[i] = masters
        self.blocks_to_swap += n
        self.auto_tuned = True  # don't promote again after an OOM
        print(
            f"[{self.block_type}] OOM recovery: demoted {n} blocks back to CPU; "
            f"now swapping {self.blocks_to_swap} of {self.num_blocks} blocks"
        )
        return n


class ChunkedStreamingOffloader:
    """
    Inference-only weight streaming at sub-block granularity for a uniform block stack.

    The parameters of streamed blocks stay on CPU as immutable, pinned masters. Each block is
    split into an ordered list of chunks — groups of direct child modules in the order the
    block's forward visits them — and chunks are uploaded ahead of use into a small ring of
    pre-allocated GPU staging buffers (NUM_SETS buffer sets per chunk type) on a dedicated copy
    stream. A forward pre-hook on each chunk's first module releases every earlier chunk's
    staging buffer (weights are immutable during inference, so release is just repointing the
    params at their CPU masters — no download), tops the prefetch pipeline up one block ahead,
    and makes the compute stream wait on this chunk's upload event.

    Compared to the classic rolling swap (ModelOffloader): PCIe traffic is upload-only and
    proportional to blocks_to_swap, there are zero steady-state allocations (no fragmentation
    churn), no per-swap stream creation, no thread pool and no host-side synchronize —
    ordering is done entirely with CUDA events.

    The public surface mirrors what block-swap-aware transformers already call on
    ModelOffloader: prepare_block_devices_before_forward / begin_forward / wait_for_block /
    submit_move_blocks_forward (the last two are no-ops here; the hooks do the work).
    """

    NUM_SETS = 2  # staging sets per chunk type: double buffering across consecutive blocks

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        device: torch.device,
        chunk_groups: list[list[str]],
        debug: bool = False,
    ):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError(f"[{block_type}] chunked weight streaming requires a CUDA device, got {self.device}")

        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.num_resident = num_blocks - blocks_to_swap
        self.chunk_groups = [list(group) for group in chunk_groups]
        self.chunks_per_block = len(self.chunk_groups)
        self.lookahead = self.chunks_per_block  # prefetch one full block ahead of compute
        self.debug = debug

        self.copy_stream = torch.cuda.Stream(self.device)
        self.chunk_params = {}  # (block_idx, chunk_idx) -> list of (param, cpu master tensor)
        self.staging = None  # [chunk_idx][set_idx] -> list of GPU tensors shaped like that chunk
        self.set_release_evt = None  # [chunk_idx][set_idx] -> Event: last occupant's kernels enqueued
        self.chunk_order = []  # execution order over streamed blocks: [(block_idx, chunk_idx), ...]
        self.chunk_pos = {}  # inverse of chunk_order
        self.upload_evt = {}  # pos -> Event: chunk upload complete on the copy stream
        self.pos_staging = {}  # pos -> set_idx currently holding that chunk
        self._held = deque()  # submitted-and-not-released positions, ascending
        self._submitted = -1
        self._current_pos = -1
        self._hook_handles = []
        self._prepared = False

    # ----- ModelOffloader-compatible surface -----

    def wait_for_block(self, block_idx: int):
        pass  # the chunk pre-hooks wait at finer granularity

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int):
        pass  # uploads are submitted by the chunk pre-hooks

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.blocks_to_swap <= 0:
            return
        if self._prepared:
            self.reset()
            return

        for block in blocks[: self.num_resident]:
            block.to(self.device)
            weighs_to_device(block, self.device)

        gb = 1024**3
        pinned_bytes = 0
        pin_failed = False
        template = None
        for i in range(self.num_resident, self.num_blocks):
            block = blocks[i]
            covered = set()
            signature = []
            for k, group in enumerate(self.chunk_groups):
                masters = []
                for child_name in group:
                    module = block.get_submodule(child_name)
                    for param_name, param in module.named_parameters(prefix=child_name):
                        covered.add(param_name)
                        param.data = param.data.to("cpu")
                        if not pin_failed and not param.data.is_pinned():
                            try:
                                param.data = param.data.pin_memory()
                                pinned_bytes += param.data.nbytes
                            except RuntimeError:
                                # host RAM limit: keep this and all later masters pageable (correct, slower uploads)
                                pin_failed = True
                                print(f"[{self.block_type}] pin_memory failed; remaining masters stay pageable")
                        masters.append((param, param.data))
                        signature.append((param_name, tuple(param.shape), param.dtype))
                self.chunk_params[(i, k)] = masters
            all_names = {name for name, _ in block.named_parameters()}
            if covered != all_names:
                missing = sorted(all_names - covered)
                raise RuntimeError(f"[{self.block_type}] chunk groups miss block params, e.g. {missing[:5]}")
            if template is None:
                template = signature
            elif signature != template:
                raise RuntimeError(f"[{self.block_type}] streamed blocks are not uniform; cannot stream chunk-wise")
            # buffers (fp8 scale_weight etc.) are tiny -> live on the GPU permanently
            for module in block.modules():
                for buf_name, buf in module._buffers.items():
                    if buf is not None and buf.device.type != self.device.type:
                        module._buffers[buf_name] = buf.to(self.device)

        first = self.num_resident
        ring_bytes = 0
        self.staging = []
        self.set_release_evt = []
        for k in range(self.chunks_per_block):
            shapes = [(master.shape, master.dtype) for _, master in self.chunk_params[(first, k)]]
            sets = []
            for _ in range(self.NUM_SETS):
                sets.append([torch.empty(shape, dtype=dtype, device=self.device) for shape, dtype in shapes])
                ring_bytes += sum(int(t.nbytes) for t in sets[-1])
            self.staging.append(sets)
            self.set_release_evt.append([None] * self.NUM_SETS)

        for i in range(self.num_resident, self.num_blocks):
            for k in range(self.chunks_per_block):
                self.chunk_pos[(i, k)] = len(self.chunk_order)
                self.chunk_order.append((i, k))
            for k, group in enumerate(self.chunk_groups):
                lead = blocks[i].get_submodule(group[0])
                handle = lead.register_forward_pre_hook(self._make_pre_hook(self.chunk_pos[(i, k)]))
                self._hook_handles.append(handle)

        block_bytes = sum(
            int(master.nbytes) for k in range(self.chunks_per_block) for _, master in self.chunk_params[(first, k)]
        )
        synchronize_device(self.device)
        clean_memory_on_device(self.device)
        self._prepared = True
        print(
            f"[{self.block_type}] chunked weight streaming: {self.num_resident} blocks resident, "
            f"{self.blocks_to_swap} streamed in {self.chunks_per_block} chunks/block "
            f"({block_bytes / gb:.2f} GB/block, staging ring {ring_bytes / gb:.2f} GB, "
            f"pinned {pinned_bytes / gb:.2f} GB)"
        )

    def begin_forward(self, blocks: list[nn.Module]):
        """Call at the start of each forward pass: recycles the previous pass's buffers and primes the pipeline."""
        if self.blocks_to_swap <= 0 or not self._prepared:
            return
        # chunks still held from the previous forward: their kernels were enqueued long ago, release now
        while self._held:
            self._release(self._held.popleft())
        self._current_pos = -1
        self._submitted = -1
        target = min(self.lookahead, len(self.chunk_order)) - 1
        while self._submitted < target:
            self._submit(self._submitted + 1)

    def reset(self):
        """Drain the pipeline and point every streamed param back at its CPU master (abort recovery)."""
        if not self._prepared:
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        while self._held:
            self._release(self._held.popleft(), record_release=False)
        for k in range(self.chunks_per_block):
            self.set_release_evt[k] = [None] * self.NUM_SETS
        self.upload_evt.clear()
        self.pos_staging.clear()
        self._current_pos = -1
        self._submitted = -1

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    # ----- internals -----

    def _make_pre_hook(self, pos: int):
        def pre_hook(module, args):
            self._activate(pos)

        return pre_hook

    def _activate(self, pos: int):
        if pos == self._current_pos:
            return  # repeated call from a row-chunked helper inside the same chunk
        # everything before this chunk has its kernels enqueued: release those staging buffers
        while self._held and self._held[0] < pos:
            self._release(self._held.popleft())
        target = min(pos + self.lookahead, len(self.chunk_order) - 1)
        while self._submitted < target:
            self._submit(self._submitted + 1)
        evt = self.upload_evt.get(pos)
        if evt is not None:
            torch.cuda.current_stream().wait_event(evt)
        self._current_pos = pos

    def _set_for_pos(self, pos: int) -> int:
        block_idx, _ = self.chunk_order[pos]
        return (block_idx - self.num_resident) % self.NUM_SETS

    def _submit(self, pos: int):
        block_idx, k = self.chunk_order[pos]
        set_idx = self._set_for_pos(pos)
        release_evt = self.set_release_evt[k][set_idx]
        self.set_release_evt[k][set_idx] = None
        masters = self.chunk_params[(block_idx, k)]
        bufs = self.staging[k][set_idx]
        with torch.cuda.stream(self.copy_stream):
            if release_evt is not None:
                # don't overwrite the buffers while the previous occupant's kernels may still read them
                self.copy_stream.wait_event(release_evt)
            for (param, master), buf in zip(masters, bufs):
                buf.copy_(master, non_blocking=True)
                param.data = buf
            evt = torch.cuda.Event()
            evt.record(self.copy_stream)
        self.upload_evt[pos] = evt
        self.pos_staging[pos] = set_idx
        self._held.append(pos)
        self._submitted = pos
        if self.debug:
            print(f"[{self.block_type}] submit chunk {(block_idx, k)} -> staging set {set_idx}")

    def _release(self, pos: int, record_release: bool = True):
        block_idx, k = self.chunk_order[pos]
        set_idx = self.pos_staging.pop(pos)
        for param, master in self.chunk_params[(block_idx, k)]:
            param.data = master
        if record_release:
            evt = torch.cuda.Event()
            evt.record()  # compute stream: fires once this chunk's kernels are done with the buffers
            self.set_release_evt[k][set_idx] = evt
        self.upload_evt.pop(pos, None)
        if self.debug:
            print(f"[{self.block_type}] release chunk {(block_idx, k)} <- staging set {set_idx}")
