import pytest

from myvllm.core.request import Request, RequestStatus
from myvllm.sampling_params import SamplingParams


def make_request(
    *,
    status: RequestStatus = RequestStatus.WAITING,
    num_computed_tokens: int = 0,
) -> Request:
    return Request(
        request_id="request-1",
        prompt_token_ids=(10, 20, 30),
        sampling_params=SamplingParams(max_tokens=16),
        num_computed_tokens=num_computed_tokens,
        status=status,
    )


def test_new_request_starts_in_waiting_prefill() -> None:
    request = make_request()

    assert request.status is RequestStatus.WAITING
    assert request.num_prompt_tokens == 3
    assert request.num_output_tokens == 0
    assert request.num_tokens == 3
    assert request.num_computed_tokens == 0
    assert request.is_prefill
    assert not request.is_decode
    assert not request.is_terminal
    assert request.block_hashes == []


def test_request_hashes_only_new_complete_blocks() -> None:
    request = Request(
        request_id="request-1",
        prompt_token_ids=(1, 2, 3),
        sampling_params=SamplingParams(max_tokens=16),
    )

    request.update_block_hashes(block_size=2)
    first_hash = request.block_hashes[0]
    request.update_block_hashes(block_size=2)

    assert request.block_hashes == [first_hash]

    request.append_output_token(4)
    request.update_block_hashes(block_size=2)

    assert len(request.block_hashes) == 2
    assert request.block_hashes[0] == first_hash


def test_reset_computed_tokens_after_preemption() -> None:
    request = make_request(
        status=RequestStatus.RUNNING,
        num_computed_tokens=3,
    )

    request.reset_computed_tokens()

    assert request.num_computed_tokens == 0


def test_request_remains_prefill_until_entire_prompt_is_computed() -> None:
    request = make_request(num_computed_tokens=2)

    assert request.is_prefill
    assert not request.is_decode


def test_request_enters_decode_at_prompt_boundary() -> None:
    request = make_request(num_computed_tokens=3)

    assert not request.is_prefill
    assert request.is_decode


@pytest.mark.parametrize("request_id", ["", "   "])
def test_rejects_blank_request_id(request_id: str) -> None:
    with pytest.raises(ValueError, match="request_id"):
        Request(
            request_id=request_id,
            prompt_token_ids=(1,),
            sampling_params=SamplingParams(max_tokens=16),
        )


def test_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="prompt_token_ids"):
        Request(
            request_id="request-1",
            prompt_token_ids=(),
            sampling_params=SamplingParams(max_tokens=16),
        )


def test_rejects_negative_prompt_token_id() -> None:
    with pytest.raises(ValueError, match="prompt_token_ids"):
        Request(
            request_id="request-1",
            prompt_token_ids=(1, -1),
            sampling_params=SamplingParams(max_tokens=16),
        )


def test_rejects_negative_initial_output_token_id() -> None:
    with pytest.raises(ValueError, match="output_token_ids"):
        Request(
            request_id="request-1",
            prompt_token_ids=(1,),
            sampling_params=SamplingParams(max_tokens=16),
            output_token_ids=[-1],
        )


@pytest.mark.parametrize("num_computed_tokens", [-1, 4])
def test_rejects_invalid_initial_progress(num_computed_tokens: int) -> None:
    with pytest.raises(ValueError, match="num_computed_tokens"):
        make_request(num_computed_tokens=num_computed_tokens)


def test_output_token_lists_are_not_shared() -> None:
    first = make_request()
    second = make_request()

    first.output_token_ids.append(42)

    assert first.output_token_ids == [42]
    assert second.output_token_ids == []


def test_advance_computed_tokens_updates_progress() -> None:
    request = make_request(status=RequestStatus.RUNNING)

    request.advance_computed_tokens(2)

    assert request.num_computed_tokens == 2
    assert request.is_prefill


@pytest.mark.parametrize("count", [0, -1])
def test_rejects_non_positive_progress(count: int) -> None:
    request = make_request(status=RequestStatus.RUNNING)

    with pytest.raises(ValueError, match="count"):
        request.advance_computed_tokens(count)


def test_rejects_progress_beyond_available_tokens() -> None:
    request = make_request(status=RequestStatus.RUNNING)

    with pytest.raises(ValueError, match="num_computed_tokens"):
        request.advance_computed_tokens(4)


def test_append_output_token_extends_request() -> None:
    request = make_request(
        status=RequestStatus.RUNNING,
        num_computed_tokens=3,
    )

    request.append_output_token(42)

    assert request.output_token_ids == [42]
    assert request.num_output_tokens == 1
    assert request.num_tokens == 4
    assert request.is_decode


def test_rejects_negative_output_token_id() -> None:
    request = make_request(status=RequestStatus.RUNNING)

    with pytest.raises(ValueError, match="token_id"):
        request.append_output_token(-1)


@pytest.mark.parametrize(
    ("initial", "target"),
    [
        (RequestStatus.WAITING, RequestStatus.RUNNING),
        (RequestStatus.WAITING, RequestStatus.WAITING_FOR_REMOTE_KV),
        (RequestStatus.WAITING_FOR_REMOTE_KV, RequestStatus.RUNNING),
        (RequestStatus.RUNNING, RequestStatus.PREEMPTED),
        (RequestStatus.PREEMPTED, RequestStatus.RUNNING),
        (RequestStatus.RUNNING, RequestStatus.FINISHED),
        (RequestStatus.RUNNING, RequestStatus.ABORTED),
        (RequestStatus.RUNNING, RequestStatus.FAILED),
    ],
)
def test_allows_valid_status_transition(
    initial: RequestStatus,
    target: RequestStatus,
) -> None:
    request = make_request(status=initial)

    request.transition_to(target)

    assert request.status is target


def test_rejects_invalid_status_transition() -> None:
    request = make_request(status=RequestStatus.WAITING)

    with pytest.raises(ValueError, match="transition"):
        request.transition_to(RequestStatus.FINISHED)


@pytest.mark.parametrize(
    "status",
    [RequestStatus.FINISHED, RequestStatus.ABORTED, RequestStatus.FAILED],
)
def test_terminal_status_cannot_transition(status: RequestStatus) -> None:
    request = make_request(status=status)

    assert request.is_terminal
    with pytest.raises(ValueError, match="transition"):
        request.transition_to(RequestStatus.RUNNING)
