"""Lightweight scheduler output for one engine step."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    """Work assigned to one request during the current scheduler step."""

    request_id: str
    num_scheduled_tokens: int
    block_ids: tuple[int, ...] = ()
    should_sample: bool = False

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")

        if self.num_scheduled_tokens <= 0:
            raise ValueError("num_scheduled_tokens must be greater than zero")

        if any(block_id < 0 for block_id in self.block_ids):
            raise ValueError("block_ids cannot contain negative IDs")


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    """All work selected by the scheduler for one engine step."""

    scheduled_requests: tuple[ScheduledRequest, ...] = ()
    preempted_request_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scheduled_requests", tuple(self.scheduled_requests))
        object.__setattr__(
            self,
            "preempted_request_ids",
            frozenset(self.preempted_request_ids),
        )

        scheduled_ids = [request.request_id for request in self.scheduled_requests]
        if len(scheduled_ids) != len(set(scheduled_ids)):
            raise ValueError("duplicate scheduled request IDs are not allowed")

        for request_id in self.preempted_request_ids:
            if not request_id.strip():
                raise ValueError("preempted request_id must not be empty")

        overlap = set(scheduled_ids) & self.preempted_request_ids
        if overlap:
            raise ValueError("a request cannot be scheduled and preempted in one step")

    @property
    def total_num_scheduled_tokens(self) -> int:
        return sum(request.num_scheduled_tokens for request in self.scheduled_requests)
