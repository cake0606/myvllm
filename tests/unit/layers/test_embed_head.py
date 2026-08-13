import torch
import torch.nn.functional as F

from myvllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding


def test_vocab_embedding_loads_full_weight_in_world_size_one() -> None:
    layer = VocabParallelEmbedding(num_embeddings=4, embedding_dim=3)
    weight = torch.arange(12, dtype=torch.float32).view(4, 3)
    token_ids = torch.tensor([3, 1])

    layer.weight_loader(layer.weight, weight)

    torch.testing.assert_close(layer(token_ids), F.embedding(token_ids, weight))


def test_lm_head_computes_every_requested_logits_row() -> None:
    layer = ParallelLMHead(num_embeddings=4, embedding_dim=3)
    weight = torch.arange(12, dtype=torch.float32).view(4, 3)
    hidden_states = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    layer.weight_loader(layer.weight, weight)

    logits = layer(hidden_states)

    torch.testing.assert_close(logits, F.linear(hidden_states, weight))
