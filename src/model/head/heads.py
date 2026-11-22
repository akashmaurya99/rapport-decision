"""The six micro-heads (spec §4/§6).

* probability — CLS-analogue vector → 384→192 (GELU) → 1, sigmoid (74K);
* confidence — concat(CLS, top-1 probability, margin, entropy) → 387→128
  (GELU) → 1, sigmoid (50K);
* answerable — **absorbed into the probability head** (see below);
* risk — CLS-analogue vector → 384→192 (GELU) → 8, sigmoid (75K).

Answerability is documented as an absorption, not as a third head: spec §4's
layer table lists no separate answerability block, and the spec's compact
heads budget reconciles *exactly* without one (``SpecConstants.heads_total``
for RAPPORT-C is 497,370 = W_q + W_o + probability + confidence + risk +
task affine + task conditioning). ``answerable`` therefore re-emits the
probability head's sigmoid output — one shared 384→192→1 stack serves both
[[FR-MODEL-03]] outputs and the total added-parameter budget is unchanged.
The abstention *action* is not a head output at all; it derives from a cost
model over these quantities (spec §6) and is fitted in TASK-16.

Confidence is trained against the indicator "top-1 was correct" (TASK-11) —
it is a prediction *of correctness*, not a re-export of ``p_max``; it consumes
``p_max``, the margin to the runner-up and the entropy of the masked choice
distribution as features precisely so it can add information over ``p_max``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

_EPS = 1e-12


class ProbabilityHead(nn.Module):
    """CLS → 384→192 (GELU) → 1, sigmoid. Serves probability *and* answerable."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.input = nn.Linear(d_model, 192)
        self.activation = nn.GELU()
        self.output = nn.Linear(192, 1)

    def logits(self, cls_vector: Tensor) -> Tensor:
        """Raw pre-sigmoid logits (``[B, 1]``).

        The loss consumes these through ``binary_cross_entropy_with_logits``
        rather than applying ``log(sigmoid(x))`` to a probability that fp16
        rounds to exactly 0 or 1. See :func:`_bce_from_logits` in
        ``src/training/losses.py``.
        """
        if cls_vector.dim() != 2:
            raise ValueError(
                f"cls_vector must be [B, d], got shape {tuple(cls_vector.shape)}"
            )
        return self.output(self.activation(self.input(cls_vector)))

    def forward(self, cls_vector: Tensor) -> Tensor:
        return torch.sigmoid(self.logits(cls_vector))


class ConfidenceHead(nn.Module):
    """concat(CLS, p_max, margin, entropy) → 387→128 (GELU) → 1, sigmoid."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.input = nn.Linear(d_model + 3, 128)
        self.activation = nn.GELU()
        self.output = nn.Linear(128, 1)

    def logits(self, features: Tensor) -> Tensor:
        """Raw pre-sigmoid logits (``[B, 1]``); see :class:`ProbabilityHead`."""
        if features.dim() != 2:
            raise ValueError(
                f"features must be [B, d + 3], got shape {tuple(features.shape)}"
            )
        return self.output(self.activation(self.input(features)))

    def forward(self, features: Tensor) -> Tensor:
        return torch.sigmoid(self.logits(features))


class RiskHead(nn.Module):
    """CLS → 384→192 (GELU) → 8, multi-label sigmoid (spec §6, eight categories)."""

    def __init__(self, d_model: int, categories: int = 8) -> None:
        super().__init__()
        self.input = nn.Linear(d_model, 192)
        self.activation = nn.GELU()
        self.output = nn.Linear(192, categories)

    def logits(self, cls_vector: Tensor) -> Tensor:
        """Raw pre-sigmoid logits (``[B, categories]``); see :class:`ProbabilityHead`."""
        if cls_vector.dim() != 2:
            raise ValueError(
                f"cls_vector must be [B, d], got shape {tuple(cls_vector.shape)}"
            )
        return self.output(self.activation(self.input(cls_vector)))

    def forward(self, cls_vector: Tensor) -> Tensor:
        return torch.sigmoid(self.logits(cls_vector))


def confidence_features(cls_vector: Tensor, probabilities: Tensor) -> Tensor:
    """Build the 387-d confidence input from CLS and the masked distribution.

    ``probabilities`` is ``[B, n]`` with padding slots exactly 0 (padding
    cannot become the top choice and contributes 0 to the entropy). The margin
    is ``p_max − p_second`` over real slots; both terms require ``n ≥ 2``.
    Returns ``[B, d + 3]``.
    """
    if probabilities.dim() != 2:
        raise ValueError(
            f"probabilities must be [B, n], got shape {tuple(probabilities.shape)}"
        )
    # Static shape checks are evaluated eagerly, but ONNX tracing records a
    # Python bool and warns; during tracing the export validates shapes itself.
    if not torch.jit.is_tracing() and probabilities.shape[-1] < 2:
        raise ValueError(
            "probabilities must have at least 2 option slots to form a margin, "
            f"got {probabilities.shape[-1]}"
        )
    top2 = torch.topk(probabilities, k=2, dim=-1).values
    p_max = top2[:, 0]
    margin = top2[:, 0] - top2[:, 1]
    # In fp16 a literal 1e-12 clamps to 0, so log(0) · 0 would poison the entropy
    # with NaN. Use the smallest normal of the tensor's dtype as the floor;
    # for fp32 this is smaller than _EPS, so existing behaviour is unchanged.
    eps = max(_EPS, torch.finfo(probabilities.dtype).tiny)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(eps))).sum(dim=-1)
    scalars = torch.stack((p_max, margin, entropy), dim=-1)
    return torch.cat((cls_vector, scalars), dim=-1)
