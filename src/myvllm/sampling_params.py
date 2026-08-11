"""Sampling parameter data structures."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class SamplingParams:
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be greater than or equal to 1")

        if not isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")

        if not isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")

        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be -1 or greater than or equal to 1")

        if any(token_id < 0 for token_id in self.stop_token_ids):
            raise ValueError("stop_token_ids cannot contain negative token IDs")

        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
