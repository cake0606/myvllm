from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from myvllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
)


def test_model_config() -> None:
    config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        tokenizer="Qwen/Qwen3-0.6B",
        revision="main",
        dtype="bfloat16",
        max_model_len=32_768,
        trust_remote_code=True,
    )

    assert config.model == "Qwen/Qwen3-0.6B"
    assert config.tokenizer == "Qwen/Qwen3-0.6B"
    assert config.revision == "main"
    assert config.dtype == "bfloat16"
    assert config.max_model_len == 32_768
    assert config.trust_remote_code is True


def test_model_config_defaults() -> None:
    config = ModelConfig(model="Qwen/Qwen3-0.6B")

    assert config.tokenizer is None
    assert config.revision is None
    assert config.dtype == "auto"
    assert config.max_model_len is None
    assert config.trust_remote_code is False


@pytest.mark.parametrize("model", ["", "   "])
def test_model_config_rejects_blank_model(model: str) -> None:
    with pytest.raises(ValueError, match="model"):
        ModelConfig(model=model)


@pytest.mark.parametrize("dtype", ["float64", "int8", ""])
def test_model_config_rejects_unsupported_dtype(dtype: str) -> None:
    with pytest.raises(ValueError, match="dtype"):
        ModelConfig(model="model", dtype=dtype)


@pytest.mark.parametrize("max_model_len", [0, -1])
def test_model_config_rejects_invalid_max_model_len(max_model_len: int) -> None:
    with pytest.raises(ValueError, match="max_model_len"):
        ModelConfig(model="model", max_model_len=max_model_len)


def test_scheduler_config_defaults() -> None:
    config = SchedulerConfig()

    assert config.max_num_batched_tokens == 2048
    assert config.max_num_seqs == 256


@pytest.mark.parametrize("max_num_batched_tokens", [0, -1])
def test_scheduler_config_rejects_invalid_token_budget(
    max_num_batched_tokens: int,
) -> None:
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        SchedulerConfig(max_num_batched_tokens=max_num_batched_tokens)


@pytest.mark.parametrize("max_num_seqs", [0, -1])
def test_scheduler_config_rejects_invalid_sequence_limit(
    max_num_seqs: int,
) -> None:
    with pytest.raises(ValueError, match="max_num_seqs"):
        SchedulerConfig(max_num_seqs=max_num_seqs)


def test_cache_config_defaults() -> None:
    config = CacheConfig()

    assert config.block_size == 256
    assert config.gpu_memory_utilization == 0.9
    assert config.cache_dtype == "auto"


@pytest.mark.parametrize("block_size", [0, -1])
def test_cache_config_rejects_invalid_block_size(block_size: int) -> None:
    with pytest.raises(ValueError, match="block_size"):
        CacheConfig(block_size=block_size)


@pytest.mark.parametrize(
    "gpu_memory_utilization",
    [0.0, -0.1, 1.1, inf, nan],
)
def test_cache_config_rejects_invalid_memory_utilization(
    gpu_memory_utilization: float,
) -> None:
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        CacheConfig(gpu_memory_utilization=gpu_memory_utilization)


def test_cache_config_accepts_supported_dtypes() -> None:
    for cache_dtype in (
        "auto",
        "float16",
        "bfloat16",
        "float32",
        "fp8",
    ):
        assert CacheConfig(cache_dtype=cache_dtype).cache_dtype == cache_dtype


def test_cache_config_rejects_unsupported_dtype() -> None:
    with pytest.raises(ValueError, match="cache_dtype"):
        CacheConfig(cache_dtype="int8")


def test_parallel_config_defaults() -> None:
    config = ParallelConfig()

    assert config.tensor_parallel_size == 1


@pytest.mark.parametrize("tensor_parallel_size", [0, -1])
def test_parallel_config_rejects_invalid_tp_size(
    tensor_parallel_size: int,
) -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        ParallelConfig(tensor_parallel_size=tensor_parallel_size)


def test_configs_are_immutable() -> None:
    config = ModelConfig(model="model")

    with pytest.raises(FrozenInstanceError):
        config.dtype = "float16"  # type: ignore[misc]
