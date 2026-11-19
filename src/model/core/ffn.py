"""SwiGLU feed-forward network, bias-free (Shazeer 2020).

Spec §3: at d_model 384, ffn 1024 is exactly 8d/3, matching the cost of a 4d
GELU FFN. All three projections are bias-free so the parameter formula
``3 · d · ffn`` holds exactly.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, ffn: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn, bias=False)
        self.up_proj = nn.Linear(d_model, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
