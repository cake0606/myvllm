import torch

from myvllm.layers.sampler import Sampler


def test_zero_temperature_uses_greedy_argmax() -> None:
    logits = torch.tensor([[1.0, 5.0, 3.0], [9.0, 2.0, 4.0]])
    temperatures = torch.zeros(2)

    sampled = Sampler()(logits, temperatures)

    torch.testing.assert_close(sampled, torch.tensor([1, 0]))


def test_positive_temperature_returns_one_token_per_row() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    sampled = Sampler()(logits, torch.ones(2))

    assert sampled.shape == (2,)
    assert torch.all((0 <= sampled) & (sampled < logits.shape[-1]))
