from collections.abc import Iterable

from myvllm.cache.block import BlockHash, KVCacheBlock


class FreeKVCacheBlockQueue:
    """A doubly linked queue of free blocks in eviction order.

    The block at the front is reused first. The queue changes only linkage and
    ordering; reference counts are owned by ``BlockPool``.
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


class BlockHashToBlockMap:
    """Map one prefix hash to one or more physical KV-cache blocks.

    The common single-block case avoids allocating an inner dictionary. If
    duplicate physical blocks contain the same prefix, the value is promoted
    to a dictionary keyed by block ID.
    """

    def __init__(self) -> None:
        self._cache: dict[
            BlockHash,
            KVCacheBlock | dict[int, KVCacheBlock],
        ] = {}

    def __len__(self) -> int:
        """Return the number of distinct prefix hashes."""

        return len(self._cache)

    def get_one_block(self, key: BlockHash) -> KVCacheBlock | None:
        """Return any physical block containing the given prefix."""

        blocks = self._cache.get(key)
        if blocks is None:
            return None
        if isinstance(blocks, KVCacheBlock):
            return blocks
        return next(iter(blocks.values()), None)

    def contains(self, key: BlockHash, block_id: int) -> bool:
        """Return whether ``key`` maps to the specified physical block."""

        blocks = self._cache.get(key)
        if blocks is None:
            return False
        if isinstance(blocks, KVCacheBlock):
            return blocks.block_id == block_id
        return block_id in blocks

    def insert(self, key: BlockHash, block: KVCacheBlock) -> None:
        """Insert a hash-to-block mapping.

        Re-inserting the same object is idempotent. Reusing a block ID for a
        different object is rejected because block IDs identify physical slots.
        """

        blocks = self._cache.get(key)
        if blocks is None:
            self._cache[key] = block
            return

        if isinstance(blocks, KVCacheBlock):
            if blocks is block:
                return
            if blocks.block_id == block.block_id:
                raise RuntimeError(
                    f"block ID {block.block_id} refers to different objects"
                )
            self._cache[key] = {
                blocks.block_id: blocks,
                block.block_id: block,
            }
            return

        existing = blocks.get(block.block_id)
        if existing is not None and existing is not block:
            raise RuntimeError(
                f"block ID {block.block_id} refers to different objects"
            )
        blocks[block.block_id] = block

    def pop(self, key: BlockHash, block_id: int) -> KVCacheBlock | None:
        """Remove only ``block_id`` from the mapping for ``key``."""

        blocks = self._cache.pop(key, None)
        if blocks is None:
            return None

        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # The requested ID did not match, so restore the mapping removed
            # by the outer dictionary pop.
            self._cache[key] = blocks
            return None

        block = blocks.pop(block_id, None)
        if blocks:
            self._cache[key] = blocks
        return block

    def clear(self) -> None:
        """Remove all prefix-hash mappings."""

        self._cache.clear()

class BlockPool:
    """Manage physical KV-cache allocation and prefix-cache residency."""

    def __init__(self, num_blocks: int, enable_caching: bool = True) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero")

        self.num_blocks = num_blocks
        self.enable_caching = enable_caching
        self.blocks = [KVCacheBlock(block_id) for block_id in range(num_blocks)]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.cached_block_hash_to_block = BlockHashToBlockMap()

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        return 1.0 - self.get_num_free_blocks() / self.num_blocks

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Allocate physical blocks from the front of the eviction queue."""

        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        num_free_blocks = self.get_num_free_blocks()
        if num_blocks > num_free_blocks:
            raise ValueError(
                f"requested {num_blocks} blocks, but only {num_free_blocks} are free"
            )

        blocks = self.free_block_queue.popleft_n(num_blocks)
        for block in blocks:
            assert block.ref_cnt == 0

            # A free cached block still contains reusable prefix KV. Once its
            # physical slot is selected for new contents, remove the stale
            # hash mapping before the storage is overwritten.
            if block.is_cached:
                self._evict_cached_block(block)

            block.ref_cnt = 1
        return blocks

    def touch(self, blocks: list[KVCacheBlock]) -> None:
        """Acquire references to prefix-cache blocks."""

        seen_ids: set[int] = set()
        for block in blocks:
            self._validate_owned_block(block)
            if block.block_id in seen_ids:
                raise RuntimeError(f"block {block.block_id} appears more than once")
            seen_ids.add(block.block_id)

        for block in blocks:
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Release blocks in caller-provided eviction-priority order.

        Uncached blocks are prepended because they cannot produce prefix hits.
        Cached blocks are appended to the LRU tail so their KV can remain
        reusable until the physical slot is selected for new contents.
        """

        blocks = list(ordered_blocks)
        seen_ids: set[int] = set()
        for block in blocks:
            self._validate_owned_block(block)
            if block.block_id in seen_ids:
                raise RuntimeError(f"block {block.block_id} appears more than once")
            if block.ref_cnt <= 0:
                raise RuntimeError(f"block {block.block_id} is already free")
            seen_ids.add(block.block_id)

        blocks_with_hash: list[KVCacheBlock] = []
        blocks_without_hash: list[KVCacheBlock] = []
        for block in blocks:
            block.ref_cnt -= 1
            if block.ref_cnt > 0:
                continue

            if block.block_hash is None and self.enable_caching:
                blocks_without_hash.append(block)
            else:
                blocks_with_hash.append(block)

        self.free_block_queue.prepend_n(blocks_without_hash)
        self.free_block_queue.append_n(blocks_with_hash)

    def cache_block(
        self,
        block: KVCacheBlock,
        block_hash: BlockHash,
        num_tokens: int,
    ) -> None:
        """Register a referenced, computed block for prefix lookup."""

        self._validate_owned_block(block)
        if not self.enable_caching:
            return
        if block.ref_cnt <= 0:
            raise RuntimeError(f"cannot cache unreferenced block {block.block_id}")
        if block.block_hash == block_hash:
            assert block.block_hash_num_tokens == num_tokens
            return
        if block.block_hash is not None:
            raise RuntimeError(f"block {block.block_id} already has another hash")

        block.set_block_hash(block_hash, num_tokens)
        self.cached_block_hash_to_block.insert(block_hash, block)

    def get_cached_block(self, block_hash: BlockHash) -> KVCacheBlock | None:
        """Look up a cached block without changing its reference count."""

        if not self.enable_caching:
            return None
        return self.cached_block_hash_to_block.get_one_block(block_hash)

    def acquire_cached_block(self, block_hash: BlockHash) -> KVCacheBlock | None:
        """Look up and acquire a cached block if it exists."""

        block = self.get_cached_block(block_hash)
        if block is not None:
            self.touch([block])
        return block

    def evict_block(self, block: KVCacheBlock) -> bool:
        """Remove a block from prefix lookup without changing its reference."""

        self._validate_owned_block(block)
        return self._evict_cached_block(block)

    def _evict_cached_block(self, block: KVCacheBlock) -> bool:
        """Remove a block's prefix mapping and reset its hash metadata."""

        block_hash = block.block_hash
        if block_hash is None:
            return False

        removed = self.cached_block_hash_to_block.pop(block_hash, block.block_id)
        if removed is not block:
            raise RuntimeError(
                f"prefix cache is inconsistent for block {block.block_id}"
            )

        block.reset_hash()
        return True

    def reset_prefix_cache(self) -> bool:
        """Clear all prefix hashes if no physical blocks are referenced."""

        if self.get_num_free_blocks() != self.num_blocks:
            return False

        self.cached_block_hash_to_block.clear()
        for block in self.blocks:
            block.reset_hash()
        return True

    def _validate_owned_block(self, block: KVCacheBlock) -> None:
        block_id = block.block_id
        if (
            block_id < 0
            or block_id >= self.num_blocks
            or self.blocks[block_id] is not block
        ):
            raise ValueError(f"block {block_id} does not belong to this pool")
