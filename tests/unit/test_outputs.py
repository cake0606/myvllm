from dataclasses import FrozenInstanceError

import pytest

from myvllm.outputs import RequestOutput


def test_request_output() -> None:
    output = RequestOutput(
        request_id="request-1",
        output_token_ids=(10, 20),
        finished=True,
    )

    assert output.request_id == "request-1"
    assert output.output_token_ids == (10, 20)
    assert output.finished is True


@pytest.mark.parametrize("request_id", ["", "   "])
def test_request_output_rejects_blank_request_id(request_id: str) -> None:
    with pytest.raises(ValueError, match="request_id"):
        RequestOutput(
            request_id=request_id,
            output_token_ids=(),
            finished=False,
        )


def test_request_output_rejects_negative_token_id() -> None:
    with pytest.raises(ValueError, match="output_token_ids"):
        RequestOutput(
            request_id="request-1",
            output_token_ids=(1, -1),
            finished=False,
        )


def test_request_output_is_immutable() -> None:
    output = RequestOutput(
        request_id="request-1",
        output_token_ids=(),
        finished=False,
    )

    with pytest.raises(FrozenInstanceError):
        output.finished = True  # type: ignore[misc]
