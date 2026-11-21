"""The utility-derived abstention policy and the risk–coverage curve.

Utilities are *constructor parameters*, never literals inside the formula.
Given utilities ``u_correct`` (utility of a correct answer), ``u_wrong``
(utility of an incorrect answer) and ``u_abstain`` (utility of abstaining), an
item is answered iff its estimated correctness probability ``p_hat`` makes the
expected utility of answering at least that of abstaining:

    p·u_correct + (1 − p)·u_wrong ≥ u_abstain
        ⟺  p ≥ (u_abstain − u_wrong) / (u_correct − u_wrong)   [if u_correct > u_wrong]

Hence for the common case ``u_correct > u_wrong``, **abstain iff**
``p_hat < (u_abstain − u_wrong) / (u_correct − u_wrong)``. When
``u_correct < u_wrong`` the inequality reverses and the side meaning "abstain"
flips; :meth:`AbstentionPolicy.should_abstain` therefore compares expected
utilities directly rather than assuming ``u_correct > u_wrong``. The declared
default triple ``(1, −3, 0)`` — a correct answer is worth 1, a wrong answer
costs 3, abstaining costs 0 — yields ``(0 − (−3)) / (1 − (−3)) = 0.75``.

The risk–coverage curve (:func:`risk_coverage_curve`) sorts items by
confidence descending into maximal equal-confidence blocks (each tied block is
one operating point) and reports, at each block boundary: the retained fraction
(coverage) ``M/N``, the selective accuracy of the retained set, and its risk
``1 − accuracy``. Permuting tied rows never changes the curve or its AURC. The
structure is plain JSON-serializable dicts and lists.
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
        """Crossover probability where answering and abstaining have equal utility.

        This is ``(u_abstain - u_wrong) / (u_correct - u_wrong)``. It is the
        decision boundary regardless of its sign, but *which side* of the
        boundary means "abstain" flips when ``u_correct < u_wrong`` — see
        :meth:`should_abstain`, which compares expected utilities directly.
        """
        return (self.u_abstain - self.u_wrong) / (self.u_correct - self.u_wrong)

    def should_abstain(self, p_hat: float) -> bool:
        """True iff answering has *strictly lower* expected utility than abstaining.

        Implemented from first principles — ``answer iff p·u_correct +
        (1 − p)·u_wrong >= u_abstain`` — so it is correct for both the declared
        triple and reversed utilities (``u_correct < u_wrong``), where dividing
        by the negative ``(u_correct − u_wrong)`` reverses the inequality. With
        the default ``(1, -3, 0)`` this is exactly ``p_hat < 0.75``.
        """
        p = _finite(p_hat, "p_hat")
        if p < 0.0 or p > 1.0:
            raise CalibrationInputError(f"p_hat must be in [0, 1], got {p_hat!r}")
        expected_answer = p * self.u_correct + (1.0 - p) * self.u_wrong
        return expected_answer < self.u_abstain

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
        policy = cls(u_correct=u_correct, u_wrong=u_wrong, u_abstain=u_abstain)
        if "threshold" in data:
            try:
                recorded = float(data["threshold"])  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise CalibrationInputError("abstention threshold must be numeric") from exc
            if abs(recorded - policy.threshold) > 1e-12:
                raise CalibrationInputError(
                    f"recorded abstention threshold {recorded} disagrees with the "
                    f"utility-derived threshold {policy.threshold}"
                )
        return policy


def risk_coverage_curve(
    confidences: Sequence[float],
    correctness: Sequence[int | float | bool],
) -> dict[str, object]:
    """Empirical risk–coverage curve, JSON-serializable.

    Items are grouped into maximal equal-confidence blocks ordered by
    confidence descending. For each block boundary the curve records the
    retained count ``M``, the coverage ``M/N``, the block's confidence
    (``threshold`` — the operating point), the selective accuracy of all
    retained rows and its risk ``1 − accuracy``. A tied block is evaluated as
    one indivisible operating point, so permuting tied rows leaves the curve
    and its AURC unchanged.

    Returns a dict with parallel arrays (``sorted_p`` — the full stable-sorted
    confidence list, and per-block ``coverage``/``selective_accuracy``/
    ``risk``), a per-block list (``points``) and the summary ``aurc``
    (``Σ_b (n_b/N)·risk_b``; equals the plain per-prefix mean with no ties).
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

    # Group equal confidences into one indivisible operating point.
    blocks: list[tuple[float, int, int]] = []  # (level, size, correct count)
    i = 0
    while i < n:
        j = i + 1
        level = p[order[i]]
        while j < n and p[order[j]] == level:
            j += 1
        members = order[i:j]
        blocks.append((level, len(members), sum(y[k] for k in members)))
        i = j

    coverage: list[float] = []
    selective_accuracy: list[float] = []
    risk: list[float] = []
    points: list[dict[str, float | int]] = []
    running_n = 0
    running_correct = 0
    running_errors = 0.0
    aurc = 0.0
    for level, size, n_correct in blocks:
        errors_in_block = size - n_correct
        for j in range(1, size + 1):
            # Same exact tie-order expectation as evaluation.metrics.aurc.
            aurc += (running_errors + errors_in_block * j / size) / (running_n + j) / n
        running_n += size
        running_correct += n_correct
        running_errors += errors_in_block
        acc = running_correct / running_n
        cov = running_n / n
        risks = 1.0 - acc
        coverage.append(cov)
        selective_accuracy.append(acc)
        risk.append(risks)
        points.append(
            {
                "retained": running_n,
                "coverage": cov,
                "threshold": level,
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
        "aurc": aurc,
    }
