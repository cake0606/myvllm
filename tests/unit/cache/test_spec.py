from dataclasses import FrozenInstanceError, replace

import pytest

from myvllm.cache.spec import KVCacheSpec


def make_spec() -> KVCacheSpec:
    return KVCacheSpec(
        block_size=16,
        num_layers=28,
        num_kv_heads=4,
        head_dim=128,
        dtype="bfloat16",
        layout="block_major",
        tensor_parallel_size=2,
        tensor_parallel_rank=0,
        model_revision="revision-1",
    )


def test_kv_cache_spec_fields() -> None:
    spec = make_spec()

    assert spec.block_size == 16
    assert spec.num_layers == 28
    assert spec.num_kv_heads == 4
    assert spec.head_dim == 128
    assert spec.dtype == "bfloat16"
    assert spec.layout == "block_major"
    assert spec.tensor_parallel_size == 2
    assert spec.tensor_parallel_rank == 0
    assert spec.model_revision == "revision-1"


@pytest.mark.parametrize(
    ("dtype", "expected_item_size"),
    [
        ("fp8", 1),
        ("float16", 2),
        ("bfloat16", 2),
        ("float32", 4),
    ],
)
def test_dtype_item_size(dtype: str, expected_item_size: int) -> None:
    assert replace(make_spec(), dtype=dtype).dtype_item_size == expected_item_size


def test_kv_cache_capacity_in_bytes() -> None:
    spec = make_spec()

    expected_bytes_per_block = 2 * 28 * 16 * 4 * 128 * 2

    assert spec.bytes_per_block == expected_bytes_per_block
    assert spec.total_bytes(num_blocks=10) == 10 * expected_bytes_per_block


@pytest.mark.parametrize(
    "field_name",
    ["block_size", "num_layers", "num_kv_heads", "head_dim"],
)
@pytest.mark.parametrize("invalid_value", [0, -1])
def test_rejects_non_positive_dimensions(
    field_name: str,
    invalid_value: int,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        replace(make_spec(), **{field_name: invalid_value})


@pytest.mark.parametrize(
    "dtype",
    ["float16", "bfloat16", "float32", "fp8"],
)
def test_accepts_supported_dtype(dtype: str) -> None:
    assert replace(make_spec(), dtype=dtype).dtype == dtype


@pytest.mark.parametrize("dtype", ["", "auto", "float64", "int8"])
def test_rejects_unsupported_dtype(dtype: str) -> None:
    with pytest.raises(ValueError, match="dtype"):
        replace(make_spec(), dtype=dtype)


def test_rejects_unsupported_layout() -> None:
    with pytest.raises(ValueError, match="layout"):
        replace(make_spec(), layout="unknown")


@pytest.mark.parametrize("tensor_parallel_size", [0, -1])
def test_rejects_invalid_tensor_parallel_size(
    tensor_parallel_size: int,
) -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        replace(make_spec(), tensor_parallel_size=tensor_parallel_size)


@pytest.mark.parametrize("tensor_parallel_rank", [-1, 2])
def test_rejects_invalid_tensor_parallel_rank(
    tensor_parallel_rank: int,
) -> None:
    with pytest.raises(ValueError, match="tensor_parallel_rank"):
        replace(make_spec(), tensor_parallel_rank=tensor_parallel_rank)


@pytest.mark.parametrize("model_revision", ["", "   "])
def test_rejects_blank_model_revision(model_revision: str) -> None:
    with pytest.raises(ValueError, match="model_revision"):
        replace(make_spec(), model_revision=model_revision)


def test_kv_cache_spec_is_immutable() -> None:
    spec = make_spec()

    with pytest.raises(FrozenInstanceError):
        spec.block_size = 32  # type: ignore[misc]
