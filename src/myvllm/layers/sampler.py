"""Token sampler adapted from nano-vLLM."""

import torch
from torch import nn


class Sampler(nn.Module):
    """Sample one token per logits row, with temperature zero as greedy."""

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        greedy_mask = temperatures == 0
        safe_temperatures = temperatures.masked_fill(greedy_mask, 1)

        scaled_logits = logits.float() / safe_temperatures.unsqueeze(-1)
        probabilities = torch.softmax(scaled_logits, dim=-1)
        exponential_noise = torch.empty_like(probabilities).exponential_()
        sampled_tokens = (probabilities / exponential_noise.clamp_min_(1e-10)).argmax(
            dim=-1
        )
        greedy_tokens = logits.argmax(dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sampled_tokens)
