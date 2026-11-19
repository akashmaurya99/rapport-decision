"""WordPiece-sized token embedding for the trunk.

Spec §4: embedding is scaled by ``sqrt(d_model)``; RAPPORT-U uses a factorized
``vocab × 128 → 128 × d_model`` bottleneck (spec §3), RAPPORT-C/L use a plain
``vocab × d_model`` table. There is deliberately no positional table here.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TrunkEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, embedding_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        self.factorized = embedding_dim != d_model
        if self.factorized:
            self.factor_in = nn.Embedding(vocab_size, embedding_dim)
            self.factor_out = nn.Linear(embedding_dim, d_model, bias=False)
        else:
            self.weight = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        ids = input_ids if input_ids.dtype == torch.long else input_ids.long()
        if self.factorized:
            hidden = self.factor_out(self.factor_in(ids))
        else:
            hidden = self.weight(ids)
        return hidden * self.scale
