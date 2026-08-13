import torch

from myvllm.layers.layernorm import RMSNorm


def reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(
        x.float().pow(2).mean(dim=-1, keepdim=True) + eps
    )
    return normalized.to(x.dtype) * weight


def test_rms_norm_matches_reference() -> None:
    layer = RMSNorm(hidden_size=4, eps=1e-5)
    layer.weight.data.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]])

    output = layer(x)

    torch.testing.assert_close(output, reference_rms_norm(x, layer.weight, 1e-5))


def test_fused_residual_add_returns_updated_residual() -> None:
    layer = RMSNorm(hidden_size=4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    residual = torch.tensor([[4.0, 3.0, 2.0, 1.0]])

    output, updated_residual = layer(x, residual)

    expected_residual = x + residual
    torch.testing.assert_close(updated_residual, expected_residual)
    torch.testing.assert_close(
        output,
        reference_rms_norm(expected_residual, layer.weight, layer.eps),
    )
