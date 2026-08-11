"""Configuration data structures for myvllm."""

from dataclasses import dataclass
from math import isfinite

_MODEL_DTYPES = frozenset(
    {
        "auto",
        "float16",
        "bfloat16",
        "float32",
    }
)

_CACHE_DTYPES = frozenset(
    {
        "auto",
        "float16",
        "bfloat16",
        "float32",
        "fp8",
    }
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    "Configuration about model and tokenizer"

    model: str
    tokenizer: str | None = None
    revision: str | None = None
    dtype: str = "auto"
    max_model_len: int | None = None
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")

        if self.tokenizer is not None and not self.tokenizer.strip():
            raise ValueError("tokenizer must not be empty")

        if self.revision is not None and not self.revision.strip():
            raise ValueError("revision must not be empty")

        if self.dtype not in _MODEL_DTYPES:
            raise ValueError(
                f"dtype must be one of {sorted(_MODEL_DTYPES)}, got {self.dtype!r}"
            )

        if self.max_model_len is not None and self.max_model_len <= 0:
            raise ValueError("max_model_len must be greater than zero")


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 256

    def __post_init__(self) -> None:
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be greater than zero")

        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be greater than zero")


@dataclass(frozen=True, slots=True)
class CacheConfig:
    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    cache_dtype: str = "auto"

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be greater than zero")

        if (
            not isfinite(self.gpu_memory_utilization)
            or not 0 < self.gpu_memory_utilization <= 1
        ):
            raise ValueError(
                "gpu_memory_utilization must be finite and in the interval (0, 1]"
            )

        if self.cache_dtype not in _CACHE_DTYPES:
            raise ValueError(
                f"cache_dtype must be one of {sorted(_CACHE_DTYPES)}, "
                f"got {self.cache_dtype!r}"
            )


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    """Tensor-parallel execution configuration."""

    tensor_parallel_size: int = 1

    def __post_init__(self) -> None:
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be greater than zero")
