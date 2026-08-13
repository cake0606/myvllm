import pytest

from myvllm.cache.block import BlockHash, KVCacheBlock
from myvllm.cache.block_pool import BlockPool


def free_ids(pool: BlockPool) -> list[int]:
    return [block.block_id for block in pool.free_block_queue.get_all_free_blocks()]


def test_pool_initializes_all_blocks_as_free() -> None:
    pool = BlockPool(num_blocks=4)

    assert [block.block_id for block in pool.blocks] == [0, 1, 2, 3]
    assert pool.get_num_free_blocks() == 4
    assert pool.get_usage() == 0.0
    assert free_ids(pool) == [0, 1, 2, 3]


@pytest.mark.parametrize("num_blocks", [0, -1])
def test_pool_rejects_non_positive_capacity(num_blocks: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        BlockPool(num_blocks=num_blocks)


def test_get_new_blocks_allocates_from_queue_front() -> None:
    pool = BlockPool(num_blocks=4)

    allocated = pool.get_new_blocks(2)

    assert [block.block_id for block in allocated] == [0, 1]
    assert all(block.ref_cnt == 1 for block in allocated)
    assert pool.get_num_free_blocks() == 2
    assert pool.get_usage() == 0.5
    assert free_ids(pool) == [2, 3]


def test_get_new_blocks_zero_is_noop() -> None:
    pool = BlockPool(num_blocks=2)

    assert pool.get_new_blocks(0) == []
    assert free_ids(pool) == [0, 1]


@pytest.mark.parametrize("num_blocks", [-1, 3])
def test_get_new_blocks_rejects_invalid_count(num_blocks: int) -> None:
    pool = BlockPool(num_blocks=2)

    with pytest.raises(ValueError):
        pool.get_new_blocks(num_blocks)

    assert free_ids(pool) == [0, 1]


def test_free_uncached_blocks_prepends_in_caller_order() -> None:
    pool = BlockPool(num_blocks=5)
    blocks = pool.get_new_blocks(3)

    pool.free_blocks([blocks[2], blocks[1], blocks[0]])

    assert free_ids(pool) == [2, 1, 0, 3, 4]
    assert all(block.ref_cnt == 0 for block in blocks)


def test_free_cached_block_appends_to_lru_tail() -> None:
    pool = BlockPool(num_blocks=3)
    block = pool.get_new_blocks(1)[0]
    key = BlockHash(b"prefix")
    pool.cache_block(block, key, num_tokens=16)

    pool.free_blocks([block])

    assert free_ids(pool) == [1, 2, 0]
    assert block.ref_cnt == 0
    assert pool.get_cached_block(key) is block


def test_disabled_caching_uses_fifo_free_order() -> None:
    pool = BlockPool(num_blocks=3, enable_caching=False)
    block = pool.get_new_blocks(1)[0]

    pool.free_blocks([block])

    assert free_ids(pool) == [1, 2, 0]


def test_touch_removes_free_cached_block_and_increments_reference() -> None:
    pool = BlockPool(num_blocks=2)
    block = pool.get_new_blocks(1)[0]
    key = BlockHash(b"prefix")
    pool.cache_block(block, key, num_tokens=16)
    pool.free_blocks([block])

    acquired = pool.acquire_cached_block(key)

    assert acquired is block
    assert block.ref_cnt == 1
    assert free_ids(pool) == [1]


def test_touch_increments_reference_of_used_block() -> None:
    pool = BlockPool(num_blocks=2)
    block = pool.get_new_blocks(1)[0]

    pool.touch([block])
    pool.free_blocks([block])

    assert block.ref_cnt == 1
    assert free_ids(pool) == [1]


def test_free_rejects_already_free_block_without_mutating_queue() -> None:
    pool = BlockPool(num_blocks=2)

    with pytest.raises(RuntimeError, match="already free"):
        pool.free_blocks([pool.blocks[0]])

    assert pool.blocks[0].ref_cnt == 0
    assert free_ids(pool) == [0, 1]


def test_free_rejects_duplicate_block_atomically() -> None:
    pool = BlockPool(num_blocks=2)
    block = pool.get_new_blocks(1)[0]

    with pytest.raises(RuntimeError, match="more than once"):
        pool.free_blocks([block, block])

    assert block.ref_cnt == 1
    assert free_ids(pool) == [1]


def test_pool_rejects_foreign_block_with_same_id() -> None:
    pool = BlockPool(num_blocks=2)
    foreign = KVCacheBlock(0, ref_cnt=1)

    with pytest.raises(ValueError, match="does not belong"):
        pool.touch([foreign])
    with pytest.raises(ValueError, match="does not belong"):
        pool.free_blocks([foreign])
    with pytest.raises(ValueError, match="does not belong"):
        pool.cache_block(foreign, BlockHash(b"foreign"), num_tokens=16)


def test_cache_block_is_idempotent_for_same_hash_and_boundary() -> None:
    pool = BlockPool(num_blocks=2)
    block = pool.get_new_blocks(1)[0]
    key = BlockHash(b"prefix")

    pool.cache_block(block, key, num_tokens=16)
    pool.cache_block(block, key, num_tokens=16)

    assert block.block_hash == key
    assert block.block_hash_num_tokens == 16
    assert pool.get_cached_block(key) is block


def test_cache_block_rejects_unreferenced_block() -> None:
    pool = BlockPool(num_blocks=2)

    with pytest.raises(RuntimeError, match="unreferenced"):
        pool.cache_block(pool.blocks[0], BlockHash(b"prefix"), num_tokens=16)


def test_same_hash_can_reference_multiple_pool_blocks() -> None:
    pool = BlockPool(num_blocks=3)
    first, second = pool.get_new_blocks(2)
    key = BlockHash(b"shared-prefix")
    pool.cache_block(first, key, num_tokens=16)
    pool.cache_block(second, key, num_tokens=16)

    assert pool.cached_block_hash_to_block.contains(key, first.block_id)
    assert pool.cached_block_hash_to_block.contains(key, second.block_id)

    pool.evict_block(first)

    assert pool.get_cached_block(key) is second
    assert first.block_hash is None
    assert second.block_hash == key


def test_reallocating_free_cached_block_evicts_old_hash() -> None:
    pool = BlockPool(num_blocks=1)
    block = pool.get_new_blocks(1)[0]
    key = BlockHash(b"old-prefix")
    pool.cache_block(block, key, num_tokens=16)
    pool.free_blocks([block])

    allocated = pool.get_new_blocks(1)[0]

    assert allocated is block
    assert block.ref_cnt == 1
    assert block.block_hash is None
    assert pool.get_cached_block(key) is None


def test_evict_uncached_block_is_noop() -> None:
    pool = BlockPool(num_blocks=2)
    block = pool.get_new_blocks(1)[0]

    assert not pool.evict_block(block)


def test_reset_prefix_cache_requires_all_blocks_to_be_free() -> None:
    pool = BlockPool(num_blocks=2)
    block = pool.get_new_blocks(1)[0]
    key = BlockHash(b"prefix")
    pool.cache_block(block, key, num_tokens=16)

    assert not pool.reset_prefix_cache()
    assert pool.get_cached_block(key) is block

    pool.free_blocks([block])
    assert pool.reset_prefix_cache()
    assert pool.get_cached_block(key) is None
    assert block.block_hash is None
