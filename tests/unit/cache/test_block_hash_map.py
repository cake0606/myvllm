import pytest

from myvllm.cache.block import BlockHash, KVCacheBlock
from myvllm.cache.block_pool import BlockHashToBlockMap


def test_empty_map() -> None:
    block_map = BlockHashToBlockMap()
    key = BlockHash(b"missing")

    assert len(block_map) == 0
    assert block_map.get_one_block(key) is None
    assert not block_map.contains(key, 0)
    assert block_map.pop(key, 0) is None


def test_insert_and_lookup_single_block() -> None:
    block_map = BlockHashToBlockMap()
    key = BlockHash(b"prefix")
    block = KVCacheBlock(0)

    block_map.insert(key, block)

    assert len(block_map) == 1
    assert block_map.get_one_block(key) is block
    assert block_map.contains(key, 0)
    assert not block_map.contains(key, 1)


def test_duplicate_insert_is_idempotent() -> None:
    block_map = BlockHashToBlockMap()
    key = BlockHash(b"prefix")
    block = KVCacheBlock(0)

    block_map.insert(key, block)
    block_map.insert(key, block)

    assert len(block_map) == 1
    assert block_map.pop(key, 0) is block
    assert len(block_map) == 0


def test_rejects_different_objects_with_same_block_id() -> None:
    block_map = BlockHashToBlockMap()
    key = BlockHash(b"prefix")
    block_map.insert(key, KVCacheBlock(0))

    with pytest.raises(RuntimeError, match="different objects"):
        block_map.insert(key, KVCacheBlock(0))


def test_same_hash_can_map_to_multiple_blocks() -> None:
    block_map = BlockHashToBlockMap()
    key = BlockHash(b"shared-prefix")
    first = KVCacheBlock(0)
    second = KVCacheBlock(1)

    block_map.insert(key, first)
    block_map.insert(key, second)

    assert len(block_map) == 1
    assert block_map.contains(key, 0)
    assert block_map.contains(key, 1)
    assert block_map.get_one_block(key) in (first, second)


def test_pop_one_duplicate_preserves_other_block() -> None:
    block_map = BlockHashToBlockMap()
    key = BlockHash(b"shared-prefix")
    first = KVCacheBlock(0)
    second = KVCacheBlock(1)
    block_map.insert(key, first)
    block_map.insert(key, second)

    assert block_map.pop(key, 0) is first

    assert len(block_map) == 1
    assert not block_map.contains(key, 0)
    assert block_map.contains(key, 1)
    assert block_map.get_one_block(key) is second


def test_pop_unknown_id_restores_single_mapping() -> None:
    block_map = BlockHashToBlockMap()
    key = BlockHash(b"prefix")
    block = KVCacheBlock(0)
    block_map.insert(key, block)

    assert block_map.pop(key, 99) is None
    assert block_map.get_one_block(key) is block


def test_different_hashes_are_independent() -> None:
    block_map = BlockHashToBlockMap()
    first_key = BlockHash(b"first")
    second_key = BlockHash(b"second")
    first = KVCacheBlock(0)
    second = KVCacheBlock(1)
    block_map.insert(first_key, first)
    block_map.insert(second_key, second)

    assert block_map.pop(first_key, 0) is first
    assert block_map.get_one_block(first_key) is None
    assert block_map.get_one_block(second_key) is second


def test_clear_removes_all_hashes() -> None:
    block_map = BlockHashToBlockMap()
    block_map.insert(BlockHash(b"first"), KVCacheBlock(0))
    block_map.insert(BlockHash(b"second"), KVCacheBlock(1))

    block_map.clear()

    assert len(block_map) == 0

