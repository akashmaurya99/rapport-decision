"""Multi-task loss functions for supervised RAPPORT training (TASK-11).

Implements the seven terms of the total objective in
``.archgen/rapport/docs/training-strategy.md`` §3.2. All functions are pure
on tensors — no global state, no file I/O — and operate on a
:class:`src.model.head.RapportOutput` plus the batch labels.

Terms (``weights`` defaults from §3.2's table; the adaptive Kendall et al.
weighting with the 0.05 floor / 4.0 cap lands in TASK-14 and multiplies these
same per-term tensors):

* ``L_choice`` — listwise softmax cross-entropy over the masked option
  distribution, against ``chosen_index``. Rows with ``chosen_index == -1``
  are masked out; the mean is taken over the remaining rows.
* ``L_pref`` — pairwise BCE (Bradley–Terry on choice-logit differences) over
  the sampled pairs ``(example, winner_slot, loser_slot)``. A batch with no
  valid pair contributes exactly ``0.0`` (``active["preference"] is False``).
  Pairs whose winner or loser slot is padding are dropped before the mean.
* ``L_score`` — Huber(delta=0.1) between the sigmoid utility ``output.scores``
  and per-option utility targets. Padding slots are excluded; examples with
  ``score_mask == 0`` are excluded.
* ``L_prob`` — BCE + 0.5 * Brier between ``output.probability`` and a soft
  target in [0, 1] (§3.2: "a proper scoring rule, not a hard label").
  Examples with ``probability_mask == 0`` are masked out.
* ``L_conf`` — BCE between ``output.confidence`` and the indicator that the
  masked argmax of ``choice_logits`` equals ``chosen_index`` ("top-1 was
  correct"). Examples with ``chosen_index == -1`` are masked out.
* ``L_abstain`` — BCE on ``answerable``. Examples whose answerable label is
  unset (``answerable_mask == 0`` or ``answerable_target=None``) are masked
  out; with no labelled example the term is exactly ``0.0`` and excluded.
* ``L_risk`` — multi-label BCE over the categories with a per-category
  ``pos_weight``. Examples with no risk labels (``risk_mask == 0`` or
  ``risk_targets=None``) are masked out; with none labelled the term is
  exactly ``0.0`` and excluded. ``pos_weight`` is either passed explicitly or
  derived from the labelled examples in the batch as ``neg / pos`` per
  category (1.0 for categories with no positives — neutral weighting).

Masking semantics are uniform: an example excluded by its label mask
contributes nothing to that term's mean; a term with no unmasked example
contributes exactly ``0.0`` to the total and reports ``active[...] == False``.
``L_choice``/``L_conf`` use ``chosen_index == -1`` as the unset sentinel.

Degenerate inputs are handled explicitly, never by ``nan_to_num``:

* a row with fewer than two real options violates the data contract
  (``MIN_OPTIONS = 2``) and raises :class:`ValueError` (``L_choice`` is
  undefined there);
* ``chosen_index`` pointing at padding or outside ``[0, n)`` raises
  :class:`ValueError`;
* a non-finite loss — whether from non-finite model outputs or NaN targets
  caught downstream — raises :class:`LossNotFiniteError` naming the
  offending component via its ``component`` attribute. It never reaches the
  optimizer: the caller (TASK-12) counts the step and skips it.

The sigmoid-headed terms (probability, confidence, abstain, risk) are computed
with :func:`torch.nn.functional.binary_cross_entropy_with_logits` on the raw
head logits whenever :class:`~src.model.head.RapportOutput` supplies them.
That log-sum-exp form is finite even when an fp16 sigmoid saturates to exactly
0 or 1; the probability fallback exists only for synthetic/test outputs that
carry no logits and preserves the "surface degeneracy, never clamp" contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from src.model.head import RapportOutput

__all__ = [
    "DEFAULT_WEIGHTS",
    "LossBreakdown",
    "LossNotFiniteError",
    "compute_losses",
]

_BRIER_WEIGHT = 0.5
_HUBER_DELTA = 0.1
_TERM_NAMES = (
    "choice",
    "preference",
    "score",
    "probability",
    "confidence",
    "abstain",
    "risk",
)

# training-strategy.md §3.2, "Default weight" column.
DEFAULT_WEIGHTS: Mapping[str, float] = {
    "choice": 1.0,
    "preference": 1.0,
    "score": 0.5,
    "probability": 0.5,
    "confidence": 0.5,
    "abstain": 1.0,
    "risk": 0.5,
}


class LossNotFiniteError(Exception):
    """A loss component (or the total) evaluated to a non-finite value.

    Always raised — never swallowed — so the caller (the trainer, TASK-12)
    can count the event and skip the optimizer step. ``component`` names the
    offending term (one of ``choice``, ``preference``, ``score``,
    ``probability``, ``confidence``, ``abstain``, ``risk``, ``total``).
    """

    code = "LOSS_NOT_FINITE"

    def __init__(self, component: str, message: str) -> None:
        super().__init__(f"{self.code}: {message}")
        self.component = component
        self.message = message


@dataclass(frozen=True)
class LossBreakdown:
    """Every term, the weighted total, and which terms were active.

    Inactive terms are exactly ``0.0`` (they contribute nothing to the total)
    and report ``active[term] == False``; ``active`` therefore tells the
    trainer which means had data behind them. ``weights`` is the resolved
    weight vector actually applied to the total.
    """

    choice: Tensor
    preference: Tensor
    score: Tensor
    probability: Tensor
    confidence: Tensor
    abstain: Tensor
    risk: Tensor
    total: Tensor
    active: Mapping[str, bool] = field(default_factory=dict)
    weights: Mapping[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_float(t: Tensor) -> Tensor:
    return t if t.is_floating_point() else t.float()


def _zero_like(reference: Tensor) -> Tensor:
    return reference.new_zeros(())


def _real_mask(option_mask: Tensor, batch: int, width: int, device: torch.device) -> Tensor:
    mask = option_mask
    if mask.dim() != 2 or mask.shape[0] != batch or mask.shape[1] != width:
        raise ValueError(
            f"option_mask must be [batch={batch}, options={width}], "
            f"got shape {tuple(mask.shape)}"
        )
    if mask.dtype == torch.bool:
        real = mask
    else:
        numeric = _as_float(mask)
        values = numeric.unique()
        if not bool(((values == 0) | (values == 1)).all()):
            raise ValueError(
                f"option_mask must be binary (0/1), got values {values.tolist()}"
            )
        real = numeric.bool()
    real = real.to(device)
    counts = real.sum(dim=-1)
    if bool((counts < 2).any()):
        bad = int((counts < 2).nonzero(as_tuple=True)[0][0])
        raise ValueError(
            f"option_mask row {bad} has {int(counts[bad])} real option(s); "
            "L_choice requires at least two (the data contract guarantees >= 2)"
        )
    return real


def _bool_mask(
    mask: Tensor | None, batch: int, *, name: str, device: torch.device
) -> Tensor:
    if mask is None:
        return torch.ones(batch, dtype=torch.bool, device=device)
    if mask.dim() != 1 or mask.shape[0] != batch:
        raise ValueError(f"{name} must be [batch={batch}], got shape {tuple(mask.shape)}")
    return mask.bool().to(device)


def _float_target(t: Tensor, *, name: str, batch: int, device: torch.device) -> Tensor:
    target = _as_float(t)
    if target.dim() == 2 and target.shape[1] == 1:
        target = target.reshape(batch)
    if target.dim() != 1 or target.shape[0] != batch:
        raise ValueError(
            f"{name} must be [batch={batch}] or [batch={batch}, 1], "
            f"got shape {tuple(target.shape)}"
        )
    if not bool(torch.isfinite(target).all()):
        raise ValueError(f"{name} contains non-finite values")
    if bool(((target < 0) | (target > 1)).any()):
        raise ValueError(f"{name} values must lie in [0, 1]")
    return target.to(device)


def _bce_from_logits(
    logits: Tensor, target: Tensor, pos_weight: Tensor | None = None
) -> Tensor:
    """Numerically stable binary cross-entropy from raw logits.

    Uses PyTorch's log-sum-exp formulation rather than ``log(sigmoid(x))``,
    which under fp16 autocast saturates to exactly 0 or 1 and turns a finite
    label into ``-inf``. Returns the unreduced element-wise loss.
    """
    kwargs: dict[str, Tensor] = {}
    if pos_weight is not None:
        kwargs["pos_weight"] = pos_weight.float()
    return F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none", **kwargs
    )


def _bce_from_probs(
    p: Tensor, target: Tensor, pos_weight: Tensor | None = None
) -> Tensor:
    """Binary cross-entropy from probabilities, without a hard clamp.

    Fallback for outputs that carry no raw logits (synthetic doubles only).
    ``log`` of an exactly-zero probability yields ``-inf`` and a genuinely
    non-finite (``inf``) loss, which is surfaced by the caller as
    :class:`LossNotFiniteError` — never hidden. The ``where`` guards only keep
    the unused branch of a 0/1 target from producing ``0 * -inf = nan``.
    """
    safe_p = torch.where(target > 0, p, torch.ones_like(p))
    safe_q = torch.where(target < 1, 1.0 - p, torch.ones_like(p))
    pos = target * torch.log(safe_p)
    if pos_weight is not None:
        pos = pos_weight * pos
    neg = (1.0 - target) * torch.log(safe_q)
    return -(pos + neg)


def _bce_term(
    p: Tensor,
    logits: Tensor | None,
    target: Tensor,
    *,
    pos_weight: Tensor | None = None,
) -> Tensor:
    """Element-wise BCE, preferring the stable logits path when available."""
    if logits is not None:
        return _bce_from_logits(logits, target, pos_weight=pos_weight)
    return _bce_from_probs(p.float(), target, pos_weight=pos_weight)


def _check_finite(name: str, term: Tensor) -> None:
    if not bool(torch.isfinite(term).all()):
        raise LossNotFiniteError(
            name, f"{name} loss is non-finite (value: {term.detach().cpu().item()!r})"
        )


def _chosen(chosen_index: Tensor | None, batch: int) -> Tensor:
    assert chosen_index is not None
    if chosen_index.dim() != 1 or chosen_index.shape[0] != batch:
        raise ValueError(
            f"chosen_index must be [batch={batch}], got shape {tuple(chosen_index.shape)}"
        )
    if chosen_index.is_floating_point() or chosen_index.dtype == torch.bool:
        raise ValueError("chosen_index must be an integer tensor")
    return chosen_index.long()


def _check_chosen_slots(chosen: Tensor, mask: Tensor) -> None:
    real_counts = mask.sum(dim=-1)
    for row in (chosen >= 0).nonzero(as_tuple=True)[0].tolist():
        index = int(chosen[row])
        if index >= int(real_counts[row]):
            raise ValueError(
                f"chosen_index[{row}] = {index} points at padding "
                f"(row has {int(real_counts[row])} real options)"
            )


# ---------------------------------------------------------------------------
# terms (each returns (loss, active))
# ---------------------------------------------------------------------------


def _choice_loss(
    output: RapportOutput, mask: Tensor, chosen_index: Tensor | None
) -> tuple[Tensor, Tensor | None]:
    reference = output.choice_logits
    if chosen_index is None:
        return _zero_like(reference), None
    chosen = _chosen(chosen_index, mask.shape[0])
    _check_chosen_slots(chosen, mask)
    available = chosen >= 0
    if not bool(available.any()):
        return _zero_like(reference), None
    masked_logits = output.choice_logits.masked_fill(~mask, float("-inf"))
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    rows = available.nonzero(as_tuple=True)[0]
    return -log_probs[rows, chosen[rows]].mean(), rows


def _preference_loss(
    output: RapportOutput, mask: Tensor, pairs: Tensor | None
) -> tuple[Tensor, bool]:
    reference = output.choice_logits
    if pairs is None or pairs.numel() == 0:
        return _zero_like(reference), False
    if pairs.dim() != 2 or pairs.shape[1] != 3:
        raise ValueError(f"preference_pairs must be [n_pairs, 3], got shape {tuple(pairs.shape)}")
    if pairs.is_floating_point() or pairs.dtype == torch.bool:
        raise ValueError("preference_pairs must be an integer tensor")
    pairs_long = pairs.long()
    batch, width = mask.shape
    example, winner, loser = pairs_long[:, 0], pairs_long[:, 1], pairs_long[:, 2]
    if bool((example < 0).any()) or bool((example >= batch).any()):
        raise ValueError("preference_pairs contain an out-of-range example index")
    if bool((winner < 0).any()) or bool((loser < 0).any()):
        raise ValueError("preference_pairs contain a negative slot index")
    if bool((winner >= width).any()) or bool((loser >= width).any()):
        raise ValueError(f"preference_pairs contain a slot index >= {width}")
    valid = mask[example, winner] & mask[example, loser]
    if not bool(valid.any()):
        return _zero_like(reference), False
    ex, w, lo = example[valid], winner[valid], loser[valid]
    # Compute in FP32 even under autocast: logits arrive in fp16 from the
    # autocast graph, and every other term explicitly upcasts to FP32.
    diff = output.choice_logits[ex, w].float() - output.choice_logits[ex, lo].float()
    return F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff)), True


def _score_loss(
    output: RapportOutput,
    mask: Tensor,
    score_targets: Tensor | None,
    score_mask: Tensor | None,
) -> tuple[Tensor, bool]:
    reference = output.scores
    batch, width = mask.shape
    if score_targets is None:
        return _zero_like(reference), False
    targets = _as_float(score_targets)
    if targets.dim() != 2 or targets.shape[0] != batch or targets.shape[1] != width:
        raise ValueError(
            f"score_targets must be [batch={batch}, options={width}], "
            f"got shape {tuple(targets.shape)}"
        )
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("score_targets contains non-finite values")
    targets = targets.to(reference.device)
    rows = _bool_mask(score_mask, batch, name="score_mask", device=reference.device)
    valid = mask & rows.unsqueeze(1)
    if not bool(valid.any()):
        return _zero_like(reference), False
    # Huber in FP32 (scores may be fp16 under autocast; targets are fp32).
    diff = output.scores[valid].float() - targets[valid].float()
    return F.huber_loss(diff, torch.zeros_like(diff), delta=_HUBER_DELTA), True


def _probability_loss(
    output: RapportOutput,
    probability_target: Tensor | None,
    probability_mask: Tensor | None,
) -> tuple[Tensor, bool]:
    reference = output.probability
    batch = reference.shape[0]
    if probability_target is None:
        return _zero_like(reference), False
    target = _float_target(probability_target, name="probability_target", batch=batch,
                           device=reference.device)
    rows = _bool_mask(probability_mask, batch, name="probability_mask", device=reference.device)
    if not bool(rows.any()):
        return _zero_like(reference), False
    p = output.probability.reshape(batch)[rows].float()
    z = (
        output.probability_logits.reshape(batch)[rows]
        if output.probability_logits is not None
        else None
    )
    y = target[rows]
    bce = _bce_term(p, z, y)
    brier = (p - y) ** 2
    return (bce + _BRIER_WEIGHT * brier).mean(), True


def _confidence_loss(
    output: RapportOutput, mask: Tensor, chosen_index: Tensor | None
) -> tuple[Tensor, bool]:
    reference = output.confidence
    if chosen_index is None:
        return _zero_like(reference), False
    chosen = _chosen(chosen_index, mask.shape[0])
    available = chosen >= 0
    if not bool(available.any()):
        return _zero_like(reference), False
    masked_logits = output.choice_logits.masked_fill(~mask, float("-inf"))
    prediction = masked_logits.argmax(dim=-1)
    target = (prediction == chosen).float()
    p = output.confidence.reshape(mask.shape[0])[available].float()
    z = (
        output.confidence_logits.reshape(mask.shape[0])[available]
        if output.confidence_logits is not None
        else None
    )
    return _bce_term(p, z, target[available]).mean(), True


def _abstain_loss(
    output: RapportOutput,
    answerable_target: Tensor | None,
    answerable_mask: Tensor | None,
) -> tuple[Tensor, bool]:
    reference = output.answerable
    batch = reference.shape[0]
    if answerable_target is None:
        return _zero_like(reference), False
    target = _float_target(answerable_target, name="answerable_target", batch=batch,
                           device=reference.device)
    rows = _bool_mask(answerable_mask, batch, name="answerable_mask", device=reference.device)
    if not bool(rows.any()):
        return _zero_like(reference), False
    p = output.answerable.reshape(batch)[rows].float()
    # ``answerable`` is absorbed into the probability head, so it shares its logits.
    z = (
        output.probability_logits.reshape(batch)[rows]
        if output.probability_logits is not None
        else None
    )
    return _bce_term(p, z, target[rows]).mean(), True


def _risk_loss(
    output: RapportOutput,
    risk_targets: Tensor | None,
    risk_mask: Tensor | None,
    risk_pos_weight: Tensor | None,
) -> tuple[Tensor, bool]:
    reference = output.risk
    batch, categories = reference.shape
    if risk_targets is None:
        return _zero_like(reference), False
    targets = _as_float(risk_targets)
    if targets.dim() != 2 or targets.shape != (batch, categories):
        raise ValueError(
            f"risk_targets must be [batch={batch}, categories={categories}], "
            f"got shape {tuple(targets.shape)}"
        )
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("risk_targets contains non-finite values")
    if bool(((targets < 0) | (targets > 1)).any()):
        raise ValueError("risk_targets values must lie in [0, 1]")
    targets = targets.to(reference.device)
    rows = _bool_mask(risk_mask, batch, name="risk_mask", device=reference.device)
    if not bool(rows.any()):
        return _zero_like(reference), False
    active_targets = targets[rows]
    if risk_pos_weight is None:
        # Derived from the labelled examples in this batch: neg / pos per
        # category; categories with no positives keep a neutral weight of 1.0.
        pos = active_targets.sum(dim=0)
        neg = active_targets.shape[0] - pos
        pos_weight = torch.where(pos > 0, neg / pos.clamp_min(1.0), torch.ones_like(pos))
    else:
        pos_weight = _as_float(risk_pos_weight)
        if pos_weight.dim() != 1 or pos_weight.shape[0] != categories:
            raise ValueError(
                f"risk_pos_weight must be [categories={categories}], "
                f"got shape {tuple(pos_weight.shape)}"
            )
        if not bool(torch.isfinite(pos_weight).all()):
            raise ValueError("risk_pos_weight contains non-finite values")
        if bool((pos_weight < 0).any()):
            raise ValueError("risk_pos_weight must be non-negative")
        pos_weight = pos_weight.to(reference.device)
    p = output.risk[rows].float()
    z = output.risk_logits[rows] if output.risk_logits is not None else None
    per_category = _bce_term(p, z, active_targets, pos_weight=pos_weight)
    return per_category.mean(), True


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def compute_losses(
    output: RapportOutput,
    option_mask: Tensor,
    *,
    chosen_index: Tensor | None = None,
    preference_pairs: Tensor | None = None,
    score_targets: Tensor | None = None,
    score_mask: Tensor | None = None,
    probability_target: Tensor | None = None,
    probability_mask: Tensor | None = None,
    answerable_target: Tensor | None = None,
    answerable_mask: Tensor | None = None,
    risk_targets: Tensor | None = None,
    risk_mask: Tensor | None = None,
    risk_pos_weight: Tensor | None = None,
    weights: Mapping[str, float] | None = None,
) -> LossBreakdown:
    """Compute the seven loss terms and their weighted total.

    Label arguments are optional; a term whose labels are absent (or whose
    mask is all-false) is exactly ``0.0`` in the result and reports
    ``active[term] == False``. ``weights`` overrides :data:`DEFAULT_WEIGHTS`
    (training-strategy.md §3.2, fixed-weight arm); weights must be finite and
    non-negative.

    Raises :class:`LossNotFiniteError` if any component or the total is
    non-finite. Nothing here clamps, replaces or hides non-finite values.
    """
    reference = output.choice_logits
    if reference.dim() != 2:
        raise ValueError(
            f"output.choice_logits must be [batch, options], got {tuple(reference.shape)}"
        )
    batch, width = reference.shape
    if batch == 0:
        raise ValueError("compute_losses requires a non-empty batch")
    device = reference.device
    mask = _real_mask(option_mask, batch, width, device)

    resolved_weights = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        unknown = set(weights) - set(_TERM_NAMES)
        if unknown:
            raise ValueError(f"weights contains unknown terms: {sorted(unknown)}")
        for key, value in weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"weights[{key!r}] must be a real number, got {value!r}")
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"weights[{key!r}] must be finite, got {value!r}")
            if value < 0:
                raise ValueError(f"weights[{key!r}] must be non-negative, got {value!r}")
            resolved_weights[key] = float(value)

    choice, choice_rows = _choice_loss(output, mask, chosen_index)
    preference, preference_active = _preference_loss(output, mask, preference_pairs)
    score, score_active = _score_loss(output, mask, score_targets, score_mask)
    probability, probability_active = _probability_loss(
        output, probability_target, probability_mask
    )
    confidence, confidence_active = _confidence_loss(output, mask, chosen_index)
    abstain, abstain_active = _abstain_loss(output, answerable_target, answerable_mask)
    risk, risk_active = _risk_loss(output, risk_targets, risk_mask, risk_pos_weight)

    terms = {
        "choice": choice,
        "preference": preference,
        "score": score,
        "probability": probability,
        "confidence": confidence,
        "abstain": abstain,
        "risk": risk,
    }
    for name, term in terms.items():
        _check_finite(name, term)

    total = _zero_like(reference)
    for name, term in terms.items():
        total = total + resolved_weights[name] * term
    _check_finite("total", total)

    return LossBreakdown(
        choice=terms["choice"],
        preference=terms["preference"],
        score=terms["score"],
        probability=terms["probability"],
        confidence=terms["confidence"],
        abstain=terms["abstain"],
        risk=terms["risk"],
        total=total,
        active={
            "choice": choice_rows is not None,
            "preference": preference_active,
            "score": score_active,
            "probability": probability_active,
            "confidence": confidence_active,
            "abstain": abstain_active,
            "risk": risk_active,
        },
        weights=resolved_weights,
    )
