from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from myvllm.sampling_params import SamplingParams


def test_max_tokens_is_required() -> None:
    with pytest.raises(TypeError):
        SamplingParams()  # type: ignore[call-arg]


def test_default_sampling_options() -> None:
    params = SamplingParams(max_tokens=128)

    assert params.max_tokens == 128
    assert params.temperature == 1.0
    assert params.top_p == 1.0
    assert params.top_k == -1
    assert params.stop_token_ids == ()
    assert params.ignore_eos is False
    assert params.seed is None


def test_greedy_sampling_options() -> None:
    params = SamplingParams(
        max_tokens=32,
        temperature=0.0,
        stop_token_ids=(1, 2),
        seed=42,
    )

    assert params.temperature == 0.0
    assert params.stop_token_ids == (1, 2)
    assert params.seed == 42


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_rejects_invalid_max_tokens(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        SamplingParams(max_tokens=max_tokens)


@pytest.mark.parametrize("temperature", [-0.1, inf, nan])
def test_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(max_tokens=16, temperature=temperature)


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.1, inf, nan])
def test_rejects_invalid_top_p(top_p: float) -> None:
    with pytest.raises(ValueError, match="top_p"):
        SamplingParams(max_tokens=16, top_p=top_p)


@pytest.mark.parametrize("top_k", [-2, 0])
def test_rejects_invalid_top_k(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        SamplingParams(max_tokens=16, top_k=top_k)


def test_rejects_negative_stop_token_id() -> None:
    with pytest.raises(ValueError, match="stop_token_ids"):
        SamplingParams(max_tokens=16, stop_token_ids=(1, -1))


def test_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        SamplingParams(max_tokens=16, seed=-1)


def test_sampling_params_is_immutable() -> None:
    params = SamplingParams(max_tokens=16)

    with pytest.raises(FrozenInstanceError):
        params.max_tokens = 32  # type: ignore[misc]
