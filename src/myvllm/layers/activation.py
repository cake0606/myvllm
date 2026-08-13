"""Activation layers adapted from nano-vLLM."""

import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    """Apply SiLU to the gate half and multiply it by the value half."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = x.chunk(2, dim=-1)
        return F.silu(gate) * value
