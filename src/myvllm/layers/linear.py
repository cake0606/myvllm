"""Tensor-parallel linear layers adapted from nano-vLLM."""

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

WeightLoader = Callable[..., None]


def divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0
    return numerator // denominator


def get_tp_rank_and_size() -> tuple[int, int]:
    """Use the active process group, or a single-rank local fallback."""

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def set_weight_loader(param: nn.Parameter, loader: WeightLoader) -> None:
    """Attach the loader consumed later by the generic model loader."""

    setattr(param, "weight_loader", loader)


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank, self.tp_size = get_tp_rank_and_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        set_weight_loader(self.weight, self.weight_loader)

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            set_weight_loader(self.bias, self.weight_loader)
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | str | None = None,
    ) -> None:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | str | None = None,
    ) -> None:
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """Shard output features across tensor-parallel ranks."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        _, tp_size = get_tp_rank_and_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | str | None = None,
    ) -> None:
        assert self.tp_dim is not None
        shard_size = param.data.size(self.tp_dim)
        start = self.tp_rank * shard_size
        shard = loaded_weight.narrow(self.tp_dim, start, shard_size)
        param.data.copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Store several independently loaded column projections in one weight."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ) -> None:
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | str | None = None,
    ) -> None:
        assert self.tp_dim is not None
        assert isinstance(loaded_shard_id, int)
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        destination = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        source = loaded_weight.chunk(self.tp_size, dim=self.tp_dim)[self.tp_rank]
        destination.copy_(source)


class QKVParallelLinear(ColumnParallelLinear):
    """Store packed query, key and value projections in one local weight."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ) -> None:
        _, tp_size = get_tp_rank_and_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | str | None = None,
    ) -> None:
        assert self.tp_dim is not None
        assert isinstance(loaded_shard_id, str)
        assert loaded_shard_id in {"q", "k", "v"}

        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size

        destination = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        source = loaded_weight.chunk(self.tp_size, dim=self.tp_dim)[self.tp_rank]
        destination.copy_(source)


class RowParallelLinear(LinearBase):
    """Shard input features, then sum partial outputs across TP ranks."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        _, tp_size = get_tp_rank_and_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, tp_dim=1)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | str | None = None,
    ) -> None:
        if param.data.ndim == 1:
            param.data.copy_(loaded_weight)
            return

        assert self.tp_dim is not None
        shard_size = param.data.size(self.tp_dim)
        start = self.tp_rank * shard_size
        shard = loaded_weight.narrow(self.tp_dim, start, shard_size)
        param.data.copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.tp_rank == 0 else None
        output = F.linear(x, self.weight, bias)
        if self.tp_size > 1:
            dist.all_reduce(output)
        return output
