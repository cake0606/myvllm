from dataclasses import FrozenInstanceError

import pytest

from myvllm.core.scheduler_output import ScheduledRequest, SchedulerOutput


def make_scheduled_request(
    request_id: str = "request-1",
    *,
    num_scheduled_tokens: int = 3,
    should_sample: bool = False,
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id=request_id,
        num_scheduled_tokens=num_scheduled_tokens,
        block_ids=(0, 1),
        should_sample=should_sample,
    )


def test_scheduled_request() -> None:
    request = make_scheduled_request(should_sample=True)

    assert request.request_id == "request-1"
    assert request.num_scheduled_tokens == 3
    assert request.block_ids == (0, 1)
    assert request.should_sample is True


@pytest.mark.parametrize("request_id", ["", "   "])
def test_scheduled_request_rejects_blank_id(request_id: str) -> None:
    with pytest.raises(ValueError, match="request_id"):
        make_scheduled_request(request_id)


@pytest.mark.parametrize("num_scheduled_tokens", [0, -1])
def test_scheduled_request_rejects_non_positive_tokens(
    num_scheduled_tokens: int,
) -> None:
    with pytest.raises(ValueError, match="num_scheduled_tokens"):
        make_scheduled_request(num_scheduled_tokens=num_scheduled_tokens)


def test_scheduled_request_rejects_negative_block_id() -> None:
    with pytest.raises(ValueError, match="block_ids"):
        ScheduledRequest(
            request_id="request-1",
            num_scheduled_tokens=1,
            block_ids=(0, -1),
        )


def test_scheduler_output_sums_token_budget() -> None:
    output = SchedulerOutput(
        scheduled_requests=(
            make_scheduled_request("request-1", num_scheduled_tokens=3),
            make_scheduled_request("request-2", num_scheduled_tokens=1),
        )
    )

    assert output.total_num_scheduled_tokens == 4
    assert output.preempted_request_ids == frozenset()


def test_scheduler_output_rejects_duplicate_scheduled_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        SchedulerOutput(
            scheduled_requests=(
                make_scheduled_request(),
                make_scheduled_request(),
            )
        )


def test_scheduler_output_rejects_scheduled_and_preempted_overlap() -> None:
    with pytest.raises(ValueError, match="scheduled and preempted"):
        SchedulerOutput(
            scheduled_requests=(make_scheduled_request(),),
            preempted_request_ids=frozenset({"request-1"}),
        )


def test_scheduler_output_rejects_blank_preempted_id() -> None:
    with pytest.raises(ValueError, match="request_id"):
        SchedulerOutput(preempted_request_ids=frozenset({"   "}))


def test_scheduler_output_normalizes_input_containers() -> None:
    scheduled = [make_scheduled_request()]
    preempted = {"request-2"}

    output = SchedulerOutput(  # type: ignore[arg-type]
        scheduled_requests=scheduled,
        preempted_request_ids=preempted,
    )

    scheduled.clear()
    preempted.clear()

    assert len(output.scheduled_requests) == 1
    assert output.preempted_request_ids == frozenset({"request-2"})


def test_scheduler_output_dtos_are_immutable() -> None:
    request = make_scheduled_request()

    with pytest.raises(FrozenInstanceError):
        request.should_sample = True  # type: ignore[misc]
