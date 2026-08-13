from myvllm.cache.kvcache_manager import KVCacheManager
from myvllm.core.request import Request
from myvllm.sampling_params import SamplingParams


def make_request(
    request_id: str,
    token_ids: tuple[int, ...],
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=16),
    )


def test_allocate_slots_grows_block_table_incrementally() -> None:
    manager = KVCacheManager(num_blocks=4, block_size=2)
    request = make_request("request-1", (1, 2, 3, 4))

    assert manager.allocate_slots(request, num_scheduled_tokens=1) == (0,)

    request.advance_computed_tokens(1)
    assert manager.allocate_slots(request, num_scheduled_tokens=3) == (0, 1)
    assert manager.get_block_ids(request.request_id) == (0, 1)
    assert manager.get_num_free_blocks() == 2


def test_cache_computed_blocks_registers_only_full_blocks() -> None:
    manager = KVCacheManager(num_blocks=3, block_size=2)
    request = make_request("request-1", (1, 2, 3))
    manager.allocate_slots(request, num_scheduled_tokens=1)

    request.advance_computed_tokens(1)
    manager.cache_computed_blocks(request)
    assert manager.get_block_table(request.request_id)[0].block_hash is None

    manager.allocate_slots(request, num_scheduled_tokens=2)
    request.advance_computed_tokens(2)
    manager.cache_computed_blocks(request)

    first, second = manager.get_block_table(request.request_id)
    assert first.block_hash == request.block_hashes[0]
    assert first.block_hash_num_tokens == 2
    assert second.block_hash is None


def test_reuses_longest_cached_prefix_and_shares_reference() -> None:
    manager = KVCacheManager(num_blocks=4, block_size=2)
    first = make_request("first", (1, 2, 3))

    first_ids = manager.allocate_slots(first, num_scheduled_tokens=3)
    assert first_ids is not None
    first.advance_computed_tokens(3)
    manager.cache_computed_blocks(first)

    second = make_request("second", (1, 2, 4))
    cached_blocks, num_cached_tokens = manager.get_computed_blocks(second)

    assert num_cached_tokens == 2
    assert tuple(block.block_id for block in cached_blocks) == first_ids[:1]

    second_ids = manager.allocate_slots(
        second,
        num_scheduled_tokens=1,
        new_computed_blocks=cached_blocks,
    )

    assert second_ids is not None
    assert second_ids[0] == first_ids[0]
    assert cached_blocks[0].ref_cnt == 2

    manager.free(second.request_id)
    assert cached_blocks[0].ref_cnt == 1
    manager.free(first.request_id)
    assert cached_blocks[0].ref_cnt == 0
    assert manager.get_num_free_blocks() == 4


def test_prefix_lookup_keeps_at_least_one_token_for_logits() -> None:
    manager = KVCacheManager(num_blocks=2, block_size=2)
    first = make_request("first", (1, 2))
    manager.allocate_slots(first, num_scheduled_tokens=2)
    first.advance_computed_tokens(2)
    manager.cache_computed_blocks(first)
    manager.free(first.request_id)

    second = make_request("second", (1, 2))

    assert manager.get_computed_blocks(second) == ((), 0)


def test_capacity_failure_is_atomic_with_free_cached_prefix() -> None:
    manager = KVCacheManager(num_blocks=1, block_size=2)
    first = make_request("first", (1, 2))
    first_ids = manager.allocate_slots(first, num_scheduled_tokens=2)
    assert first_ids == (0,)
    first.advance_computed_tokens(2)
    manager.cache_computed_blocks(first)
    manager.free(first.request_id)

    cached_block = manager.block_pool.blocks[0]
    second = make_request("second", (1, 2, 3))
    cached_blocks, num_cached_tokens = manager.get_computed_blocks(second)
    assert num_cached_tokens == 2

    result = manager.allocate_slots(
        second,
        num_scheduled_tokens=1,
        new_computed_blocks=cached_blocks,
    )

    assert result is None
    assert manager.get_block_table(second.request_id) == ()
    assert cached_block.ref_cnt == 0
    assert manager.get_num_free_blocks() == 1
