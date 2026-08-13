import torch
import torch.nn.functional as F

from myvllm.layers.activation import SiluAndMul


def test_silu_and_mul_matches_reference() -> None:
    x = torch.tensor([[1.0, -2.0, 3.0, 4.0]])

    output = SiluAndMul()(x)

    gate, up = x.chunk(2, dim=-1)
    torch.testing.assert_close(output, F.silu(gate) * up)
