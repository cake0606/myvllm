import torch

from myvllm.cache.spec import KVCacheSpec
from myvllm.cache.storage import KVCacheStorage


def make_spec(dtype: str = "bfloat16") -> KVCacheSpec:
    return KVCacheSpec(
        block_size=4,
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        dtype=dtype,
        layout="block_major",
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
        model_revision="test-revision",
    )


def test_allocates_block_major_kv_tensor() -> None:
    spec = make_spec()
    storage = KVCacheStorage(spec, num_blocks=3, device="cpu")

    assert storage.num_blocks == 3
    assert storage.tensor.shape == (2, 2, 3, 4, 2, 8)
    assert storage.tensor.dtype is torch.bfloat16
    assert storage.tensor.device.type == "cpu"
    assert storage.tensor.nbytes == spec.total_bytes(num_blocks=3)


def test_layer_cache_is_a_view_of_shared_storage() -> None:
    storage = KVCacheStorage(make_spec(), num_blocks=3, device="cpu")

    key_cache, value_cache = storage.get_layer_kv_cache(layer_idx=1)

    assert key_cache.shape == (3, 4, 2, 8)
    assert value_cache.shape == (3, 4, 2, 8)

    key_cache[2, 3, 1, 7] = 11
    value_cache[0, 1, 0, 2] = 13

    assert storage.tensor[0, 1, 2, 3, 1, 7].item() == 11
    assert storage.tensor[1, 1, 0, 1, 0, 2].item() == 13


def test_maps_supported_cache_dtypes_to_torch() -> None:
    expected_dtypes = {
        "fp8": torch.float8_e4m3fn,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    for dtype, expected_dtype in expected_dtypes.items():
        storage = KVCacheStorage(
            make_spec(dtype),
            num_blocks=1,
            device="cpu",
        )
        assert storage.tensor.dtype is expected_dtype
