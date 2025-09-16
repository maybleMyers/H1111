import torch
import torch.distributed as dist
from einops import rearrange
from typing import Tuple
import ipdb

_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_RANK = None
_CONTEXT_PARALLEL_GROUP_SIZE = None
_CONTEXT_PARALLEL_GROUP_RANKS = None


def get_cp_rank_size():
    if _CONTEXT_PARALLEL_GROUP:
        return _CONTEXT_PARALLEL_RANK, _CONTEXT_PARALLEL_GROUP_SIZE
    else:
        return 0, 1


def local_shard(x: torch.Tensor, dim: int = 2) -> torch.Tensor:
    if not _CONTEXT_PARALLEL_GROUP:
        return x

    cp_rank, cp_size = get_cp_rank_size()
    return x.tensor_split(cp_size, dim=dim)[cp_rank]


def set_cp_group(cp_group, ranks, global_rank):
    global _CONTEXT_PARALLEL_GROUP, _CONTEXT_PARALLEL_RANK, _CONTEXT_PARALLEL_GROUP_SIZE, _CONTEXT_PARALLEL_GROUP_RANKS
    if _CONTEXT_PARALLEL_GROUP is not None:
        raise RuntimeError("CP group already initialized.")
    _CONTEXT_PARALLEL_GROUP = cp_group
    _CONTEXT_PARALLEL_RANK = dist.get_rank(cp_group)
    _CONTEXT_PARALLEL_GROUP_SIZE = dist.get_world_size(cp_group)
    _CONTEXT_PARALLEL_GROUP_RANKS = ranks

    assert _CONTEXT_PARALLEL_RANK == ranks.index(
        global_rank
    ), f"Rank mismatch: {global_rank} in {ranks} does not have position {_CONTEXT_PARALLEL_RANK} "
    assert _CONTEXT_PARALLEL_GROUP_SIZE == len(
        ranks
    ), f"Group size mismatch: {_CONTEXT_PARALLEL_GROUP_SIZE} != len({ranks})"


def get_cp_group():
    if _CONTEXT_PARALLEL_GROUP is None:
        raise RuntimeError("CP group not initialized")
    return _CONTEXT_PARALLEL_GROUP


def is_cp_active():
    return _CONTEXT_PARALLEL_GROUP is not None


class AllGatherIntoTensorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, reduce_dtype, group: dist.ProcessGroup):
        ctx.reduce_dtype = reduce_dtype
        ctx.group = group
        ctx.batch_size = x.size(0)
        group_size = _CONTEXT_PARALLEL_GROUP_SIZE

        x = x.contiguous()
        output = torch.empty(group_size * x.size(0), *x.shape[1:], dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(output, x, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        reduce_dtype = ctx.reduce_dtype
        batch_size = ctx.batch_size
        group_size = _CONTEXT_PARALLEL_GROUP_SIZE
        
        # Split gradient across dimension 0 (context parallel dimension)
        grad_chunks = grad_output.chunk(group_size, dim=0)
        rank = _CONTEXT_PARALLEL_RANK
        return grad_chunks[rank].contiguous(), None, None


def all_gather(tensor: torch.Tensor, group=None) -> torch.Tensor:
    if not _CONTEXT_PARALLEL_GROUP:
        return tensor

    # Use the provided group or default to _CONTEXT_PARALLEL_GROUP
    group = group if group is not None else _CONTEXT_PARALLEL_GROUP
    return AllGatherIntoTensorFunction.apply(tensor, torch.float32, group)


@torch.compiler.disable()
def _all_to_all_single(output, input, group):
    # Disable compilation since torch compile changes contiguity.
    assert input.is_contiguous(), "Input tensor must be contiguous."
    assert output.is_contiguous(), "Output tensor must be contiguous."
    return dist.all_to_all_single(output, input, group=group)


class CollectTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv: torch.Tensor, group: dist.ProcessGroup, num_heads: int):
        """Redistribute heads and receive tokens.

        Args:
            qkv: query, key or value. Shape: [B, M, 3 * num_heads * head_dim]
            group: The process group to use for communication
            num_heads: Total number of attention heads

        Returns:
            qkv: shape: [3, B, N, local_heads, head_dim]

        where M is the number of local tokens,
        N = cp_size * M is the number of global tokens,
        local_heads = num_heads // cp_size is the number of local heads.
        """
        ctx.group = group
        ctx.num_heads = num_heads
        cp_size = _CONTEXT_PARALLEL_GROUP_SIZE
        assert num_heads % cp_size == 0
        ctx.local_heads = num_heads // cp_size

        qkv = rearrange(
            qkv,
            "B M (qkv G h d) -> G M h B (qkv d)",
            qkv=3,
            G=cp_size,
            h=ctx.local_heads,
        ).contiguous()

        output_chunks = torch.empty_like(qkv)
        _all_to_all_single(output_chunks, qkv, group=group)

        return rearrange(output_chunks, "G M h B (qkv d) -> qkv B (G M) h d", qkv=3)

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        cp_size = _CONTEXT_PARALLEL_GROUP_SIZE
        local_heads = ctx.local_heads

        # Reverse the final rearrange
        grad_output = rearrange(grad_output, "qkv B (G M) h d -> G M h B (qkv d)", G=cp_size).contiguous()
        
        # Reverse the all-to-all
        grad_input = torch.empty_like(grad_output)
        _all_to_all_single(grad_input, grad_output, group=group)
        
        # Reverse the initial rearrange
        return rearrange(grad_input, "G M h B (qkv d) -> B M (qkv G h d)", 
                        qkv=3, G=cp_size, h=local_heads), None, None


def all_to_all_collect_tokens(x: torch.Tensor, num_heads: int, group=None) -> torch.Tensor:
    if not _CONTEXT_PARALLEL_GROUP:
        # Move QKV dimension to the front.
        #   B M (3 H d) -> 3 B M H d
        B, M, _ = x.size()
        x = x.view(B, M, 3, num_heads, -1)
        return x.permute(2, 0, 1, 3, 4)

    # Use the provided group or default to _CONTEXT_PARALLEL_GROUP
    group = group if group is not None else _CONTEXT_PARALLEL_GROUP
    return CollectTokens.apply(x, group, num_heads)


class CollectHeads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup):
        """Redistribute tokens and receive heads.

        Args:
            x: Output of attention. Shape: [B, N, local_heads, head_dim]
            group: The process group to use for communication

        Returns:
            Shape: [B, M, num_heads * head_dim]
        """
        ctx.group = group
        ctx.local_heads = x.size(2)
        ctx.head_dim = x.size(3)
        group_size = _CONTEXT_PARALLEL_GROUP_SIZE
        x = rearrange(x, "B (G M) h D -> G h M B D", G=group_size).contiguous()
        output = torch.empty_like(x)
        _all_to_all_single(output, x, group=group)
        del x
        return rearrange(output, "G h M B D -> B M (G h D)")

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        group_size = _CONTEXT_PARALLEL_GROUP_SIZE
        local_heads = ctx.local_heads
        head_dim = ctx.head_dim

        # Reverse the final rearrange
        grad_output = rearrange(grad_output, "B M (G h D) -> G h M B D", 
                              G=group_size, h=local_heads, D=head_dim)
        
        grad_output = grad_output.contiguous()

        # Reverse the all-to-all
        grad_input = torch.empty_like(grad_output)
        
        _all_to_all_single(grad_input, grad_output, group=group)
        
        # Reverse the initial rearrange
        return rearrange(grad_input, "G h M B D -> B (G M) h D", G=group_size), None


def all_to_all_collect_heads(x: torch.Tensor, group=None) -> torch.Tensor:
    if not _CONTEXT_PARALLEL_GROUP:
        # Merge heads.
        return x.view(x.size(0), x.size(1), x.size(2) * x.size(3))

    # Use the provided group or default to _CONTEXT_PARALLEL_GROUP
    group = group if group is not None else _CONTEXT_PARALLEL_GROUP
    return CollectHeads.apply(x, group)
