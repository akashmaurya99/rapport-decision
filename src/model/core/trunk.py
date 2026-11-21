"""The RAPPORT encoder trunk: embedding + N pre-RMSNorm blocks.

Spec §3/§4: RAPPORT-U/C/L share this module; sizes come from ``RapportConfig``.
The forward pass maps token ids ``[B, T]`` to hidden states ``[B, T, d_model]``.
There is no positional table anywhere in the tree — RoPE (base 10,000, see
``rope.py``) is the only position mechanism.
"""

from __future__ import annotations

from torch import Tensor, nn

from .block import EncoderBlock
from .config import RapportConfig
from .embedding import TrunkEmbedding


class RapportTrunk(nn.Module):
    def __init__(self, config: RapportConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = TrunkEmbedding(config.vocab_size, config.d_model, config.embedding_dim)
        self.blocks = nn.ModuleList(
            EncoderBlock(
                config.d_model,
                config.heads,
                config.head_dim,
                config.ffn,
                config.rope_base,
                config.rms_eps,
                config.dropout,
            )
            for _ in range(config.layers)
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Encode token ids.

        ``attention_mask`` is a padding mask ([B, T], 1 = real token, 0 = pad);
        it is the only mask the trunk consults. ``None`` means a full
        bidirectional window with no padding.
        """
        hidden = self.embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return hidden
