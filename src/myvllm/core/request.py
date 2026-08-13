"""Request state and lifecycle definitions."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from myvllm.cache.block import BlockHash, hash_block_tokens
from myvllm.sampling_params import SamplingParams


class RequestStatus(StrEnum):
    WAITING = "waiting"
    WAITING_FOR_REMOTE_KV = "waiting_for_remote_kv"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"
    ABORTED = "aborted"
    FAILED = "failed"


_TERMINAL_STATUSES = frozenset(
    {
        RequestStatus.FINISHED,
        RequestStatus.ABORTED,
        RequestStatus.FAILED,
    }
)

_ALLOWED_TRANSITIONS: dict[RequestStatus, frozenset[RequestStatus]] = {
    RequestStatus.WAITING: frozenset(
        {
            RequestStatus.RUNNING,
            RequestStatus.WAITING_FOR_REMOTE_KV,
            RequestStatus.ABORTED,
            RequestStatus.FAILED,
        }
    ),
    RequestStatus.WAITING_FOR_REMOTE_KV: frozenset(
        {
            RequestStatus.WAITING,
            RequestStatus.RUNNING,
            RequestStatus.ABORTED,
            RequestStatus.FAILED,
        }
    ),
    RequestStatus.RUNNING: frozenset(
        {
            RequestStatus.PREEMPTED,
            RequestStatus.FINISHED,
            RequestStatus.ABORTED,
            RequestStatus.FAILED,
        }
    ),
    RequestStatus.PREEMPTED: frozenset(
        {
            RequestStatus.WAITING,
            RequestStatus.RUNNING,
            RequestStatus.ABORTED,
            RequestStatus.FAILED,
        }
    ),
    RequestStatus.FINISHED: frozenset(),
    RequestStatus.ABORTED: frozenset(),
    RequestStatus.FAILED: frozenset(),
}


@dataclass(slots=True)
class Request:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling_params: SamplingParams

    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    block_hashes: list[BlockHash] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")

        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")

        if any(token_id < 0 for token_id in self.prompt_token_ids):
            raise ValueError("prompt_token_ids cannot contain negative token IDs")

        if any(token_id < 0 for token_id in self.output_token_ids):
            raise ValueError("output_token_ids cannot contain negative token IDs")

        self._validate_progress(self.num_computed_tokens)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + tuple(self.output_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < self.num_prompt_tokens

    @property
    def is_decode(self) -> bool:
        return self.num_computed_tokens >= self.num_prompt_tokens

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES

    def _validate_progress(self, num_computed_tokens: int) -> None:
        if not 0 <= num_computed_tokens <= self.num_tokens:
            raise ValueError("num_computed_tokens must be between zero and num_tokens")

    def advance_computed_tokens(self, count: int) -> None:
        if count <= 0:
            raise ValueError("count must be greater than zero")

        new_num_computed_tokens = self.num_computed_tokens + count
        self._validate_progress(new_num_computed_tokens)

        self.num_computed_tokens = new_num_computed_tokens

    def update_block_hashes(self, block_size: int) -> None:
        """Append chained hashes for newly completed token blocks."""

        if block_size <= 0:
            raise ValueError("block_size must be greater than zero")

        all_token_ids = self.all_token_ids
        start = len(self.block_hashes) * block_size

        parent_hash = self.block_hashes[-1] if self.block_hashes else None

        while start + block_size <= len(all_token_ids):
            end = start + block_size
            block_token_ids: Sequence[int] = all_token_ids[start:end]

            parent_hash = hash_block_tokens(
                parent_hash=parent_hash,
                token_ids=tuple(block_token_ids),
            )

            self.block_hashes.append(parent_hash)
            start = end

    def reset_computed_tokens(self) -> None:
        """Reset model progress after all request KV blocks are released."""

        self.num_computed_tokens = 0

    def append_output_token(self, token_id: int) -> None:
        if token_id < 0:
            raise ValueError("token_id must be non-negative")

        self.output_token_ids.append(token_id)

    def transition_to(self, new_status: RequestStatus) -> None:
        allowed_transitions = _ALLOWED_TRANSITIONS[self.status]

        if new_status not in allowed_transitions:
            raise ValueError(
                "invalid request status transition: "
                f"{self.status.value} -> {new_status.value}"
            )

        self.status = new_status
