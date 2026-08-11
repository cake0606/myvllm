from myvllm.cache.block import KVCacheBlock


class FreeKVCacheBlockQueue:
    """A doubly linked queue of free blocks in eviction order.

    The block at the front is reused first. The queue changes only linkage and
    ordering; reference counts are owned by the future ``BlockPool``.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self._head = KVCacheBlock(block_id=-1)
        self._tail = KVCacheBlock(block_id=-1)
        self._head.next_free_block = self._tail
        self._tail.prev_free_block = self._head
        self.num_free_blocks = 0

        self.append_n(blocks)

    def popleft(self) -> KVCacheBlock:
        """Remove and return the first free block."""

        first_block = self._head.next_free_block
        if first_block is None:
            raise RuntimeError("free block queue is corrupted")
        if first_block is self._tail:
            assert self.num_free_blocks == 0
            raise ValueError("no free blocks available")

        next_block = first_block.next_free_block
        if next_block is None:
            raise RuntimeError("free block queue is corrupted")

        self._head.next_free_block = next_block
        next_block.prev_free_block = self._head

        first_block.prev_free_block = None
        first_block.next_free_block = None
        self.num_free_blocks -= 1
        return first_block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Remove and return the first ``n`` free blocks."""

        if n < 0:
            raise ValueError("n must be non-negative")
        if n > self.num_free_blocks:
            raise ValueError(
                f"requested {n} blocks, but only {self.num_free_blocks} are free"
            )
        if n == 0:
            return []

        current = self._head.next_free_block
        blocks: list[KVCacheBlock] = []

        for _ in range(n):
            if current is None or current is self._tail:
                raise RuntimeError("free block queue is corrupted")

            next_block = current.next_free_block
            current.prev_free_block = None
            current.next_free_block = None
            blocks.append(current)
            current = next_block

        if current is None:
            raise RuntimeError("free block queue is corrupted")

        self._head.next_free_block = current
        current.prev_free_block = self._head
        self.num_free_blocks -= n
        return blocks

    def remove(self, block: KVCacheBlock) -> None:
        """Remove an arbitrary free block in O(1)."""

        if block is self._head or block is self._tail:
            raise RuntimeError("cannot remove a sentinel block")

        previous = block.prev_free_block
        following = block.next_free_block
        if previous is None or following is None:
            raise RuntimeError(f"block {block.block_id} is not in the free queue")

        previous.next_free_block = following
        following.prev_free_block = previous
        block.prev_free_block = None
        block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Append one block to the back of the free queue."""

        self._validate_insertable(block)
        last_block = self._tail.prev_free_block
        if last_block is None:
            raise RuntimeError("free block queue is corrupted")

        last_block.next_free_block = block
        block.prev_free_block = last_block
        block.next_free_block = self._tail
        self._tail.prev_free_block = block
        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """Append blocks while preserving their input order."""

        if not blocks:
            return

        self._validate_insertable_blocks(blocks)
        last_block = self._tail.prev_free_block
        if last_block is None:
            raise RuntimeError("free block queue is corrupted")

        for block in blocks:
            last_block.next_free_block = block
            block.prev_free_block = last_block
            last_block = block

        last_block.next_free_block = self._tail
        self._tail.prev_free_block = last_block
        self.num_free_blocks += len(blocks)

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        """Prepend blocks while preserving their input order."""

        if not blocks:
            return

        self._validate_insertable_blocks(blocks)
        first_block = self._head.next_free_block
        if first_block is None:
            raise RuntimeError("free block queue is corrupted")

        previous = self._head
        for block in blocks:
            previous.next_free_block = block
            block.prev_free_block = previous
            previous = block

        previous.next_free_block = first_block
        first_block.prev_free_block = previous
        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Return all free blocks in eviction order, mainly for tests."""

        blocks: list[KVCacheBlock] = []
        current = self._head.next_free_block

        while current is not None and current is not self._tail:
            blocks.append(current)
            current = current.next_free_block

        if current is None:
            raise RuntimeError("free block queue is corrupted")

        return blocks

    @staticmethod
    def _validate_insertable(block: KVCacheBlock) -> None:
        if block.ref_cnt != 0:
            raise RuntimeError(
                f"referenced block {block.block_id} cannot enter the free queue"
            )
        if block.prev_free_block is not None or block.next_free_block is not None:
            raise RuntimeError(f"block {block.block_id} is already linked")

    @classmethod
    def _validate_insertable_blocks(cls, blocks: list[KVCacheBlock]) -> None:
        seen_ids: set[int] = set()
        for block in blocks:
            if block.block_id in seen_ids:
                raise RuntimeError(f"duplicate block {block.block_id} in insertion")
            seen_ids.add(block.block_id)
            cls._validate_insertable(block)

