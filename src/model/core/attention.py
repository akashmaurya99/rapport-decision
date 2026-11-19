"""Bidirectional multi-head self-attention with padding-only masking.

Spec §4: "SDPA, full bidirectional, no causal mask, padding mask only". The
mask argument is a padding mask (1 = real token, 0 = pad); nothing else is
accepted, and ``is_causal`` is always False — options may see each other and
earlier positions may see later ones.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .rope import RotaryEmbedding, apply_rope


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, head_dim: int, rope_base: float) -> None:
        super().__init__()
        if heads * head_dim != d_model:
            raise ValueError(
                f"heads ({heads}) * head_dim ({head_dim}) must equal d_model ({d_model})"
            )
        self.heads = heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = RotaryEmbedding(head_dim, base=rope_base)

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        batch, seq_len, dim = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(seq_len, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        mask = self._key_padding_mask(padding_mask, x)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        out = out.transpose(1, 2).reshape(batch, seq_len, dim)
        return self.o_proj(out)

    @staticmethod
    def _key_padding_mask(padding_mask: Tensor | None, like: Tensor) -> Tensor | None:
        """Broadcast [B, T] padding mask to an additive [B, 1, 1, T] float mask."""
        if padding_mask is None:
            return None
        keep = padding_mask.to(torch.bool)
        if keep.shape != like.shape[:2]:
            raise ValueError(
                f"padding_mask must be [B, T] = {tuple(like.shape[:2])},"
                f" got {tuple(keep.shape)}"
            )
        additive = torch.zeros_like(keep, dtype=like.dtype)
        # finfo.min rather than -inf keeps fully-padded rows finite instead of NaN.
        additive = additive.masked_fill(~keep, torch.finfo(like.dtype).min)
        return additive[:, None, None, :]
