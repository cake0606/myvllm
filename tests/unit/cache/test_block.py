from dataclasses import fields

import pytest

from myvllm.cache.block import BlockHash, KVCacheBlock, hash_block_tokens


def test_block_hash_is_deterministic_and_chained() -> None:
    first = hash_block_tokens(None, (1, 2))

    assert first == hash_block_tokens(None, (1, 2))
    assert first != hash_block_tokens(None, (2, 1))
    assert hash_block_tokens(first, (3, 4)) != hash_block_tokens(None, (3, 4))


def test_block_hash_rejects_empty_or_negative_tokens() -> None:
    with pytest.raises(ValueError, match="empty"):
        hash_block_tokens(None, ())

    with pytest.raises(ValueError, match="negative"):
        hash_block_tokens(None, (1, -1))


def test_new_block_has_empty_metadata() -> None:
    block = KVCacheBlock(block_id=3)

    assert block.block_id == 3
    assert block.ref_cnt == 0
    assert block.is_free
    assert not block.is_cached
    assert block.block_hash is None
    assert block.block_hash_num_tokens is None
    assert block.prev_free_block is None
    assert block.next_free_block is None


def test_set_and_reset_block_hash() -> None:
    block = KVCacheBlock(block_id=0)
    block_hash = BlockHash(b"hash")

    block.set_block_hash(block_hash, num_tokens=16)

    assert block.is_cached
    assert block.block_hash == block_hash
    assert block.block_hash_num_tokens == 16

    block.reset_hash()

    assert not block.is_cached
    assert block.block_hash is None
    assert block.block_hash_num_tokens is None


def test_rejects_replacing_existing_hash() -> None:
    block = KVCacheBlock(block_id=0)
    block.set_block_hash(BlockHash(b"first"), num_tokens=16)

    with pytest.raises(AssertionError, match="already has a hash"):
        block.set_block_hash(BlockHash(b"second"), num_tokens=32)


@pytest.mark.parametrize("num_tokens", [0, -1])
def test_rejects_non_positive_hash_coverage(num_tokens: int) -> None:
    block = KVCacheBlock(block_id=0)

    with pytest.raises(AssertionError, match="greater than zero"):
        block.set_block_hash(BlockHash(b"hash"), num_tokens=num_tokens)


def test_link_fields_do_not_participate_in_repr() -> None:
    field_by_name = {item.name: item for item in fields(KVCacheBlock)}

    assert field_by_name["prev_free_block"].repr is False
    assert field_by_name["next_free_block"].repr is False


def test_block_uses_slots() -> None:
    block = KVCacheBlock(block_id=0)

    assert not hasattr(block, "__dict__")
    with pytest.raises(AttributeError):
        block.unexpected_field = True  # type: ignore[attr-defined]
