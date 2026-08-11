import pytest

from myvllm.cache.block import KVCacheBlock
from myvllm.cache.block_pool import FreeKVCacheBlockQueue


def make_blocks(count: int) -> list[KVCacheBlock]:
    return [KVCacheBlock(block_id) for block_id in range(count)]


def assert_queue_state(
    queue: FreeKVCacheBlockQueue,
    expected: list[KVCacheBlock],
) -> None:
    assert queue.get_all_free_blocks() == expected
    assert queue.num_free_blocks == len(expected)

    previous = queue._head  # noqa: SLF001
    for block in expected:
        assert previous.next_free_block is block
        assert block.prev_free_block is previous
        previous = block

    assert previous.next_free_block is queue._tail  # noqa: SLF001
    assert queue._tail.prev_free_block is previous  # noqa: SLF001


def test_initialization_preserves_block_order() -> None:
    blocks = make_blocks(4)
    queue = FreeKVCacheBlockQueue(blocks)

    assert_queue_state(queue, blocks)


def test_empty_queue_rejects_popleft() -> None:
    queue = FreeKVCacheBlockQueue([])

    assert_queue_state(queue, [])
    with pytest.raises(ValueError, match="no free blocks"):
        queue.popleft()


def test_popleft_removes_and_unlinks_first_block() -> None:
    blocks = make_blocks(3)
    queue = FreeKVCacheBlockQueue(blocks)

    popped = queue.popleft()

    assert popped is blocks[0]
    assert popped.prev_free_block is None
    assert popped.next_free_block is None
    assert_queue_state(queue, blocks[1:])


def test_popleft_n_preserves_order_and_unlinks_blocks() -> None:
    blocks = make_blocks(5)
    queue = FreeKVCacheBlockQueue(blocks)

    popped = queue.popleft_n(3)

    assert popped == blocks[:3]
    assert all(block.prev_free_block is None for block in popped)
    assert all(block.next_free_block is None for block in popped)
    assert_queue_state(queue, blocks[3:])


def test_popleft_n_zero_is_noop() -> None:
    blocks = make_blocks(2)
    queue = FreeKVCacheBlockQueue(blocks)

    assert queue.popleft_n(0) == []
    assert_queue_state(queue, blocks)


@pytest.mark.parametrize("count", [-1, 3])
def test_popleft_n_rejects_invalid_count(count: int) -> None:
    blocks = make_blocks(2)
    queue = FreeKVCacheBlockQueue(blocks)

    with pytest.raises(ValueError):
        queue.popleft_n(count)

    assert_queue_state(queue, blocks)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_remove_unlinks_block_at_any_position(index: int) -> None:
    blocks = make_blocks(3)
    queue = FreeKVCacheBlockQueue(blocks)
    removed = blocks[index]

    queue.remove(removed)

    assert removed.prev_free_block is None
    assert removed.next_free_block is None
    assert_queue_state(queue, blocks[:index] + blocks[index + 1 :])


def test_remove_rejects_block_not_in_queue() -> None:
    queue = FreeKVCacheBlockQueue(make_blocks(2))
    other = KVCacheBlock(block_id=9)

    with pytest.raises(RuntimeError, match="not in the free queue"):
        queue.remove(other)


def test_append_to_empty_and_nonempty_queue() -> None:
    blocks = make_blocks(3)
    queue = FreeKVCacheBlockQueue([])

    queue.append(blocks[0])
    assert_queue_state(queue, blocks[:1])

    queue.append(blocks[1])
    assert_queue_state(queue, blocks[:2])


def test_append_n_preserves_input_order() -> None:
    blocks = make_blocks(5)
    queue = FreeKVCacheBlockQueue(blocks[:1])

    queue.append_n(blocks[2:5])

    assert_queue_state(queue, [blocks[0], blocks[2], blocks[3], blocks[4]])


def test_prepend_n_preserves_input_order() -> None:
    blocks = make_blocks(5)
    queue = FreeKVCacheBlockQueue(blocks[:2])

    queue.prepend_n(blocks[3:5])

    assert_queue_state(queue, [blocks[3], blocks[4], blocks[0], blocks[1]])


def test_rejects_inserting_referenced_block() -> None:
    block = KVCacheBlock(block_id=0, ref_cnt=1)
    queue = FreeKVCacheBlockQueue([])

    with pytest.raises(RuntimeError, match="referenced block"):
        queue.append(block)


def test_rejects_inserting_already_linked_block() -> None:
    blocks = make_blocks(2)
    queue = FreeKVCacheBlockQueue(blocks)

    with pytest.raises(RuntimeError, match="already linked"):
        queue.append(blocks[0])


def test_rejects_duplicate_block_in_bulk_insertion() -> None:
    block = KVCacheBlock(block_id=0)
    queue = FreeKVCacheBlockQueue([])

    with pytest.raises(RuntimeError, match="duplicate block"):
        queue.append_n([block, block])


def test_removed_block_can_be_appended_again() -> None:
    blocks = make_blocks(3)
    queue = FreeKVCacheBlockQueue(blocks)

    queue.remove(blocks[1])
    queue.append(blocks[1])

    assert_queue_state(queue, [blocks[0], blocks[2], blocks[1]])

