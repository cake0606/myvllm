"""KV-cache compatibility and layout specifications."""

from dataclasses import dataclass

_SUPPORTED_DTYPES = frozenset(
    {
        "float16",
        "bfloat16",
        "float32",
        "fp8",
    }
)

_SUPPORTED_LAYOUTS = frozenset(
    {
        "block_major",
    }
)


@dataclass(frozen=True, slots=True)
class KVCacheSpec:
    """Resolved KV-cache format for one tensor-parallel worker."""

    block_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    layout: str
    tensor_parallel_size: int
    tensor_parallel_rank: int
    model_revision: str

    def __post_init__(self) -> None:
        self._validate_positive_dimension("block_size", self.block_size)
        self._validate_positive_dimension("num_layers", self.num_layers)
        self._validate_positive_dimension("num_kv_heads", self.num_kv_heads)
        self._validate_positive_dimension("head_dim", self.head_dim)

        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"dtype must be one of {sorted(_SUPPORTED_DTYPES)}, got {self.dtype!r}"
            )

        if self.layout not in _SUPPORTED_LAYOUTS:
            raise ValueError(
                f"layout must be one of {sorted(_SUPPORTED_LAYOUTS)}, "
                f"got {self.layout!r}"
            )

        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be greater than zero")

        if not 0 <= self.tensor_parallel_rank < self.tensor_parallel_size:
            raise ValueError(
                "tensor_parallel_rank must be between zero and tensor_parallel_size - 1"
            )

        if not self.model_revision.strip():
            raise ValueError("model_revision must not be empty")

    @staticmethod
    def _validate_positive_dimension(field_name: str, value: int) -> None:
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than zero")

