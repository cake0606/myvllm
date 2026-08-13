import torch

from myvllm.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb


def test_apply_rotary_embedding_matches_manual_rotation() -> None:
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    cos = torch.tensor([[[0.5, 0.25]]])
    sin = torch.tensor([[[0.75, 1.0]]])

    output = apply_rotary_emb(x, cos, sin)

    expected = torch.tensor([[[-1.75, -3.5, 2.25, 3.0]]])
    torch.testing.assert_close(output, expected)


def test_rotary_embedding_selects_cache_by_position() -> None:
    rope = RotaryEmbedding(
        head_size=4,
        rotary_dim=4,
        max_position_embeddings=8,
        base=10_000,
    )
    positions = torch.tensor([0, 3])
    query = torch.randn(2, 2, 4)
    key = torch.randn(2, 1, 4)

    rotated_query, rotated_key = rope(positions, query, key)

    cos_sin = rope.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    torch.testing.assert_close(rotated_query, apply_rotary_emb(query, cos, sin))
    torch.testing.assert_close(rotated_key, apply_rotary_emb(key, cos, sin))
