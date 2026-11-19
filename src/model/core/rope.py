"""Rotary position embeddings (RoPE), base 10,000 — the only position mechanism.

Spec §4: "RoPE, base 10,000, applied per layer; **no** learned position table".
Frequencies are derived from the base at call time; nothing positional is ever
stored as a parameter or a persistent buffer, so the trunk cannot grow a
learned position table by accident.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Computes cos/sin tables on the fly for a sequence slice.

    Invariant: ``base`` comes from ``RapportConfig.rope_base`` (10,000) and is
    never learned.
    """

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.base = base

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = self.base ** (
            -torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim
        )
        freqs = torch.outer(positions, inv_freq)  # [T, hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, hd]
        return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to ``x`` of shape [B, heads, T, hd]."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, hd]
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(t: Tensor) -> Tensor:
        half = t.shape[-1] // 2
        first, second = t[..., :half], t[..., half:]
        return torch.cat((-second, first), dim=-1)

    return x * cos + rotate_half(x) * sin
