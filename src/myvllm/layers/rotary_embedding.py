"""Rotary position embeddings adapted from nano-vLLM."""

from functools import lru_cache

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate the two halves of each attention head."""

    first_half, second_half = x.float().chunk(2, dim=-1)
    rotated_first = first_half * cos - second_half * sin
    rotated_second = second_half * cos + first_half * sin
    return torch.cat((rotated_first, rotated_second), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    cos_sin_cache: torch.Tensor

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size

        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)

        # Shape [max_position, 1, rotary_dim] broadcasts over attention heads.
        cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)
        self.register_buffer(
            "cos_sin_cache",
            cache.unsqueeze(1),
            persistent=False,
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return (
            apply_rotary_emb(query, cos, sin),
            apply_rotary_emb(key, cos, sin),
        )


@lru_cache(maxsize=1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
) -> RotaryEmbedding:
    """Reuse the immutable RoPE lookup table across decoder layers."""

    return RotaryEmbedding(head_size, rotary_dim, max_position, base)
