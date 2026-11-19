"""Pre-RMSNorm encoder block: attention residual, then SwiGLU residual."""

from __future__ import annotations

from torch import Tensor, nn

from .attention import MultiHeadSelfAttention
from .ffn import SwiGLUFeedForward
from .norm import RMSNorm


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, head_dim: int, ffn: int,
                 rope_base: float, rms_eps: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=rms_eps)
        self.attn = MultiHeadSelfAttention(d_model, heads, head_dim, rope_base)
        self.norm2 = RMSNorm(d_model, eps=rms_eps)
        self.ffn = SwiGLUFeedForward(d_model, ffn)

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x
