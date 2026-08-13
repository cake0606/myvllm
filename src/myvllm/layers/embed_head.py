"""Vocabulary embedding and LM head adapted from nano-vLLM."""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from myvllm.layers.linear import divide, get_tp_rank_and_size, set_weight_loader


class VocabParallelEmbedding(nn.Module):
    """Shard vocabulary rows across tensor-parallel ranks."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.tp_rank, self.tp_size = get_tp_rank_and_size()
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = divide(num_embeddings, self.tp_size)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        set_weight_loader(self.weight, self.weight_loader)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        shard_size = param.data.size(0)
        start = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(0, start, shard_size))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.embedding(token_ids, self.weight)

        local_mask = (token_ids >= self.vocab_start_idx) & (
            token_ids < self.vocab_end_idx
        )
        local_ids = (token_ids - self.vocab_start_idx).masked_fill(~local_mask, 0)
        output = F.embedding(local_ids, self.weight)
        output.mul_(local_mask.unsqueeze(-1))
        dist.all_reduce(output)
        return output


class ParallelLMHead(VocabParallelEmbedding):
    """Compute vocabulary logits without reading global execution context."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ) -> None:
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local_logits = F.linear(hidden_states, self.weight)
        if self.tp_size == 1:
            return local_logits

        logits_shards = [torch.empty_like(local_logits) for _ in range(self.tp_size)]
        dist.all_gather(logits_shards, local_logits)
        return torch.cat(logits_shards, dim=-1)
