from __future__ import annotations

from dataclasses import dataclass, field
from typing import NewType

BlockHash = NewType("BlockHash", bytes)


@dataclass(slots=True, eq=False)
class KVCacheBlock:
    """Metadata for one physical KV-cache block."""

    block_id: int
    ref_cnt: int = 0
    _block_hash: BlockHash | None = None
    _block_hash_num_tokens: int | None = None

    # These links are owned exclusively by FreeKVCacheBlockQueue.
    prev_free_block: KVCacheBlock | None = field(
        default=None,
        init=False,
        repr=False,
    )
    next_free_block: KVCacheBlock | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def is_free(self) -> bool:
        return self.ref_cnt == 0

    @property
    def is_cached(self) -> bool:
        return self.block_hash is not None

    @property
    def block_hash(self) -> BlockHash | None:
        return self._block_hash

    @property
    def block_hash_num_tokens(self) -> int | None:
        """Number of prefix tokens covered by ``block_hash``."""

        return self._block_hash_num_tokens

    def set_block_hash(
        self,
        block_hash: BlockHash,
        num_tokens: int,
    ) -> None:
        assert self.block_hash is None and self.block_hash_num_tokens is None, (
            "The block already has a hash"
        )
        assert num_tokens > 0, "num_tokens must be greater than zero"

        self._block_hash = block_hash
        self._block_hash_num_tokens = num_tokens

    def reset_hash(self) -> None:
        """Reset prefix-cache metadata when this block is evicted."""

        self._block_hash = None
        self._block_hash_num_tokens = None

