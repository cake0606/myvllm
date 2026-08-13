"""Normalization layers adapted from nano-vLLM."""

from typing import overload

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS normalization with an optional fused residual addition."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        normalized = x.float()
        variance = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return normalized.to(original_dtype) * self.weight

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_dtype = x.dtype

        # Keep the residual stream in the model dtype, but compute the sum and
        # variance in float32 to avoid accumulating normalization error.
        normalized = x.float() + residual.float()
        updated_residual = normalized.to(original_dtype)
        variance = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        output = normalized.to(original_dtype) * self.weight
        return output, updated_residual

    @overload
    def forward(
        self,
        x: torch.Tensor,
        residual: None = None,
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        return self.add_rms_forward(x, residual)
