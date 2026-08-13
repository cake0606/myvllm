from myvllm.cache.block import KVCacheBlock
from myvllm.cache.block_pool import BlockPool
from myvllm.core.request import Request


class KVCacheManager:
    """Manage the physical KV-cache block table owned by each request."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_caching: bool = True,
    ) -> None:
        self.block_size = block_size
        self.block_pool = BlockPool(num_blocks, enable_caching)
        self.req_to_blocks: dict[str, list[KVCacheBlock]] = {}

    def _update_block_hashes(self, request: Request) -> None:
        if self.block_pool.enable_caching:
            request.update_block_hashes(self.block_size)

    def get_computed_blocks(
        self,
        request: Request,
    ) -> tuple[tuple[KVCacheBlock, ...], int]:
        """Return the longest continuous prefix found in the block cache."""

        if not self.block_pool.enable_caching:
            return (), 0

        self._update_block_hashes(request)

        # Keep at least one token for model execution. Even when the entire
        # prompt is cached, its final token must run to produce the next logits.
        max_cached_blocks = min(
            len(request.block_hashes),
            (request.num_tokens - 1) // self.block_size,
        )

        cached_blocks: list[KVCacheBlock] = []
        for block_hash in request.block_hashes[:max_cached_blocks]:
            block = self.block_pool.get_cached_block(block_hash)
            if block is None:
                break
            cached_blocks.append(block)

        return tuple(cached_blocks), len(cached_blocks) * self.block_size

    def allocate_slots(
        self,
        request: Request,
        num_scheduled_tokens: int,
        new_computed_blocks: tuple[KVCacheBlock, ...] = (),
    ) -> tuple[int, ...] | None:
        """
        Incrementally allocate the blocks needed by this scheduler step.
        Args:
            new_computed_blocks: prefix cache, only be used in the first time make block table

        """

        block_table = self.req_to_blocks.get(request.request_id, [])
        num_cached_tokens = len(new_computed_blocks) * self.block_size

        # 1. Convert the token boundary after this step into the total number
        # of physical blocks the request must own.
        num_tokens = (
            request.num_computed_tokens + num_cached_tokens + num_scheduled_tokens
        )
        num_required_blocks = (num_tokens + self.block_size - 1) // self.block_size

        # 2. Prefix blocks and blocks already in the table cover part of that
        # requirement; only the remainder needs fresh physical blocks.
        num_new_blocks = max(
            num_required_blocks - len(block_table) - len(new_computed_blocks),
            0,
        )

        # 3. A cached block with ref_cnt == 0 is still in the free queue.
        # Reusing it removes one free entry, just like allocating a new block.
        num_free_cached_blocks = sum(
            block.ref_cnt == 0 for block in new_computed_blocks
        )
        if (
            num_free_cached_blocks + num_new_blocks
            > self.block_pool.get_num_free_blocks()
        ):
            return None

        # Mutate only after the capacity check so a failed allocation leaves
        # both the request table and the free queue unchanged.
        if new_computed_blocks:
            self.block_pool.touch(list(new_computed_blocks))
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)

        if request.request_id not in self.req_to_blocks:
            block_table = self.req_to_blocks[request.request_id] = []
        block_table.extend(new_computed_blocks)
        block_table.extend(new_blocks)
        return tuple(block.block_id for block in block_table)

    def cache_computed_blocks(self, request: Request) -> None:
        """Register every fully computed block for later prefix reuse."""

        if not self.block_pool.enable_caching:
            return

        self._update_block_hashes(request)
        num_full_blocks = request.num_computed_tokens // self.block_size
        block_table = self.req_to_blocks[request.request_id]

        for block_idx in range(num_full_blocks):
            self.block_pool.cache_block(
                block_table[block_idx],
                request.block_hashes[block_idx],
                (block_idx + 1) * self.block_size,
            )

    def get_block_table(self, request_id: str) -> tuple[KVCacheBlock, ...]:
        return tuple(self.req_to_blocks.get(request_id, ()))

    def get_block_ids(self, request_id: str) -> tuple[int, ...]:
        return tuple(block.block_id for block in self.req_to_blocks.get(request_id, ()))

    def free(self, request_id: str) -> None:
        block_table = self.req_to_blocks.pop(request_id)
        # Free the logical tail first so recently useful prefix blocks enter
        # the eviction queue after less reusable suffix blocks.
        self.block_pool.free_blocks(reversed(block_table))

    def preempt(self, request_id: str) -> None:
        self.free(request_id)

    def get_num_free_blocks(self) -> int:
        return self.block_pool.get_num_free_blocks()
