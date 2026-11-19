"""The utility-derived abstention policy and the risk–coverage curve.

Utilities are *constructor parameters*, never literals inside the formula.
Given utilities ``u_correct`` (utility of a correct answer), ``u_wrong``
(utility of an incorrect answer) and ``u_abstain`` (utility of abstaining), an
item is answered iff its estimated correctness probability ``p_hat`` makes the
expected utility of answering at least that of abstaining:

    p·u_correct + (1 − p)·u_wrong ≥ u_abstain
        ⟺  p ≥ (u_abstain − u_wrong) / (u_correct − u_wrong)

Hence **abstain iff** ``p_hat < (u_abstain − u_wrong) / (u_correct − u_wrong)``.
The declared default triple ``(1, −3, 0)`` — a correct answer is worth 1, a
wrong answer costs 3, abstaining costs 0 — yields ``(0 − (−3)) / (1 − (−3)) =
0.75``.

The risk–coverage curve (:func:`risk_coverage_curve`) sorts items by
confidence descending and reports, at each prefix ``m = 1..N``: the retained
fraction (coverage) ``m/N``, the selective accuracy of the retained set, and
its risk ``1 − accuracy``. The structure is plain JSON-serializable dicts and
lists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .errors import CalibrationInputError

#: Declared default utility triple (correct, wrong, abstain).
DEFAULT_U_CORRECT = 1.0
DEFAULT_U_WRONG = -3.0
DEFAULT_U_ABSTAIN = 0.0


def _finite(value: float, name: str) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationInputError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(f):
        raise CalibrationInputError(f"{name} must be finite, got {value!r}")
    return f


def abstention_threshold(
    u_correct: float = DEFAULT_U_CORRECT,
    u_wrong: float = DEFAULT_U_WRONG,
    u_abstain: float = DEFAULT_U_ABSTAIN,
) -> float:
    """Threshold on ``p_hat`` below which the policy abstains.

    With the default triple this is exactly ``0.75``. Raises
    :class:`CalibrationInputError` when ``u_correct == u_wrong`` (the utility
    of answering is independent of ``p`` — no decision boundary exists).
    """
    u_c = _finite(u_correct, "u_correct")
    u_w = _finite(u_wrong, "u_wrong")
    u_a = _finite(u_abstain, "u_abstain")
    if u_c == u_w:
        raise CalibrationInputError(
            "u_correct must differ from u_wrong: with equal utilities the answer/abstain "
            "decision is independent of p_hat"
        )
    return (u_a - u_w) / (u_c - u_w)


@dataclass(frozen=True)
class AbstentionPolicy:
    """Utility-derived abstention policy.

    Construct with the declared utility triple; :attr:`threshold` and
    :meth:`should_abstain` derive from it. The default triple yields the 0.75
    threshold declared in the product spec.
    """

    u_correct: float = DEFAULT_U_CORRECT
    u_wrong: float = DEFAULT_U_WRONG
    u_abstain: float = DEFAULT_U_ABSTAIN

    def __post_init__(self) -> None:
        _finite(self.u_correct, "u_correct")
        _finite(self.u_wrong, "u_wrong")
        _finite(self.u_abstain, "u_abstain")
        if self.u_correct == self.u_wrong:
            raise CalibrationInputError(
                "u_correct must differ from u_wrong: with equal utilities the "
                "answer/abstain decision is independent of p_hat"
            )

    @property
    def threshold(self) -> float:
        return (self.u_abstain - self.u_wrong) / (self.u_correct - self.u_wrong)

    def should_abstain(self, p_hat: float) -> bool:
        """True iff ``p_hat`` is strictly below :attr:`threshold`."""
        p = _finite(p_hat, "p_hat")
        if p < 0.0 or p > 1.0:
            raise CalibrationInputError(f"p_hat must be in [0, 1], got {p_hat!r}")
        return p < self.threshold

    def to_dict(self) -> dict[str, float | str]:
        return {
            "u_correct": float(self.u_correct),
            "u_wrong": float(self.u_wrong),
            "u_abstain": float(self.u_abstain),
            "threshold": self.threshold,
            "rule": "abstain iff p_hat < (u_abstain - u_wrong) / (u_correct - u_wrong)",
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AbstentionPolicy":
        try:
            u_correct = float(data["u_correct"])  # type: ignore[arg-type]
            u_wrong = float(data["u_wrong"])  # type: ignore[arg-type]
            u_abstain = float(data["u_abstain"])  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationInputError(
                f"abstention payload must carry numeric u_correct/u_wrong/u_abstain: {data!r}"
            ) from exc
        return cls(u_correct=u_correct, u_wrong=u_wrong, u_abstain=u_abstain)


def risk_coverage_curve(
    confidences: Sequence[float],
    correctness: Sequence[int | float | bool],
) -> dict[str, object]:
    """Empirical risk–coverage curve, JSON-serializable.

    Items are sorted by confidence descending (ties keep input order — the
    sort is stable). For each prefix ``m = 1..N`` the curve records the
    retained count, the coverage ``m/N``, the confidence of the ``m``-th
    retained item (``threshold`` — the operating point), the selective
    accuracy of the retained set and its risk ``1 − accuracy``.

    Returns a dict with parallel arrays (``sorted_p``, ``coverage``,
    ``selective_accuracy``, ``risk``), a per-point list (``points``) and the
    summary ``aurc`` (mean of the per-prefix risks, uniform width ``1/N``).
    """
    if confidences is None or correctness is None:
        raise CalibrationInputError("confidences and correctness must be sequences")
    if len(confidences) != len(correctness):
        raise CalibrationInputError(
            f"confidences and correctness must have equal length, got "
            f"{len(confidences)} and {len(correctness)}"
        )
    n = len(confidences)
    if n == 0:
        raise CalibrationInputError("expected at least one item")

    p: list[float] = []
    for i, v in enumerate(confidences):
        f = _finite(v, f"confidences[{i}]")
        if f < 0.0 or f > 1.0:
            raise CalibrationInputError(f"confidences[{i}] = {f} outside [0, 1]")
        p.append(f)
    y: list[int] = []
    for i, v in enumerate(correctness):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise CalibrationInputError(f"correctness[{i}] must be 0 or 1, got {v!r}")
        f = float(v)
        if f != 0.0 and f != 1.0:
            raise CalibrationInputError(f"correctness[{i}] must be 0 or 1, got {v!r}")
        y.append(int(f))

    order = sorted(range(n), key=lambda i: p[i], reverse=True)  # stable
    sorted_p = [p[i] for i in order]
    sorted_y = [y[i] for i in order]

    coverage: list[float] = []
    selective_accuracy: list[float] = []
    risk: list[float] = []
    points: list[dict[str, float | int]] = []
    running = 0
    for m in range(1, n + 1):
        running += sorted_y[m - 1]
        acc = running / m
        cov = m / n
        risks = 1.0 - acc
        coverage.append(cov)
        selective_accuracy.append(acc)
        risk.append(risks)
        points.append(
            {
                "retained": m,
                "coverage": cov,
                "threshold": sorted_p[m - 1],
                "selective_accuracy": acc,
                "risk": risks,
            }
        )

    return {
        "n_items": n,
        "sorted_p": sorted_p,
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "risk": risk,
        "points": points,
        "aurc": sum(risk) / n,
    }
