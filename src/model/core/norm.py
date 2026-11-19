"""RMSNorm without a bias — the pre-norm used by every encoder block."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned gain and no bias."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normed = x.float() * torch.rsqrt(variance + self.eps)
        return (normed * self.weight.float()).to(input_dtype)
