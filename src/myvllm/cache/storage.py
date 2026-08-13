"""Physical tensor storage for the paged KV cache."""

import torch

from myvllm.cache.spec import KVCacheSpec

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "fp8": torch.float8_e4m3fn,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class KVCacheStorage:
    """Own the physical K/V tensor shared by all model layers.

    ``KVCacheManager`` assigns physical block IDs. This class gives those IDs
    actual tensor storage; it deliberately knows nothing about requests or
    logical block tables.
    """

    def __init__(
        self,
        spec: KVCacheSpec,
        num_blocks: int,
        device: str | torch.device = "cuda",
    ) -> None:
        self.spec = spec
        self.num_blocks = num_blocks

        # One allocation keeps all layers in a predictable block-major layout:
        # [K/V, layer, physical block, token slot, KV head, head dimension].
        # A physical block ID therefore indexes the same cache position in
        # every layer, which is exactly what the request block table describes.
        self.tensor = torch.empty(
            (
                2,
                spec.num_layers,
                num_blocks,
                spec.block_size,
                spec.num_kv_heads,
                spec.head_dim,
            ),
            dtype=_TORCH_DTYPES[spec.dtype],
            device=device,
        )

    def get_layer_kv_cache(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the K and V views used by one attention layer.

        Tensor indexing returns views, so attention writes directly into the
        shared allocation without copying per-layer cache tensors.
        """

        return self.tensor[0, layer_idx], self.tensor[1, layer_idx]
