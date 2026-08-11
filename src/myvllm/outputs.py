"""Public output data structures."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestOutput:
    """Cumulative generated tokens returned for one request."""

    request_id: str
    output_token_ids: tuple[int, ...]
    finished: bool

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")

        if any(token_id < 0 for token_id in self.output_token_ids):
            raise ValueError("output_token_ids cannot contain negative token IDs")
