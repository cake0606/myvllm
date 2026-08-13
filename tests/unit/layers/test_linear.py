import torch
import torch.nn.functional as F

from myvllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


def test_replicated_linear_loads_weight_and_matches_torch() -> None:
    layer = ReplicatedLinear(3, 2, bias=True)
    weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bias = torch.tensor([0.5, -0.5])
    x = torch.tensor([[2.0, 1.0, -1.0]])

    layer.weight_loader(layer.weight, weight)
    assert layer.bias is not None
    layer.weight_loader(layer.bias, bias)

    torch.testing.assert_close(layer(x), F.linear(x, weight, bias))


def test_parallel_linear_layers_fall_back_to_world_size_one() -> None:
    column = ColumnParallelLinear(4, 6)
    row = RowParallelLinear(4, 6)

    assert (column.tp_rank, column.tp_size, column.weight.shape) == (0, 1, (6, 4))
    assert (row.tp_rank, row.tp_size, row.weight.shape) == (0, 1, (6, 4))


def test_merged_column_loader_places_each_projection() -> None:
    layer = MergedColumnParallelLinear(2, [3, 4])
    gate = torch.arange(6, dtype=torch.float32).view(3, 2)
    up = torch.arange(8, dtype=torch.float32).view(4, 2) + 10

    layer.weight_loader(layer.weight, gate, loaded_shard_id=0)
    layer.weight_loader(layer.weight, up, loaded_shard_id=1)

    torch.testing.assert_close(layer.weight, torch.cat((gate, up)))


def test_qkv_loader_places_query_key_and_value() -> None:
    layer = QKVParallelLinear(
        hidden_size=4,
        head_size=2,
        total_num_heads=2,
        total_num_kv_heads=1,
    )
    query = torch.arange(16, dtype=torch.float32).view(4, 4)
    key = torch.arange(8, dtype=torch.float32).view(2, 4) + 20
    value = torch.arange(8, dtype=torch.float32).view(2, 4) + 40

    layer.weight_loader(layer.weight, query, loaded_shard_id="q")
    layer.weight_loader(layer.weight, key, loaded_shard_id="k")
    layer.weight_loader(layer.weight, value, loaded_shard_id="v")

    torch.testing.assert_close(layer.weight, torch.cat((query, key, value)))
