"""The Stage C distillation usefulness gate (FR-DISTILL-02, risk R9, E8).

``distillation_gate`` is a **pure decision function**. It never reaches
through to training, never loads a model and never touches the cache on
disk — Stage C orchestration (TASK-25) calls it with the metrics each run
already produced. When the gate returns ``abort``, the report treats it as a
recorded **negative result** (Stage C is skipped), never as an error.

Rules (training-strategy.md §4; risks.md R9):

1. The teacher is unusable — the cached soft labels were missing or
   unreadable, or the teacher failed to load (E8's fail-to-load limb) —
   or the teacher does **not** beat the supervised student by at least
   ``min_teacher_margin`` points on dev (default 3.0).
2. The KD student is **not** at least ``min_kd_gain`` points above the
   supervised student on dev (default 1.0).

Either rule failing aborts distillation. ``reason`` names the failing
condition: ``teacher_unavailable``, ``teacher_margin_below_threshold`` or
``kd_gain_below_threshold`` (``ok`` on a pass). When both margins fail,
reason names the teacher limb; ``checks`` records each limb independently.

Metric contract
---------------
Metrics are mappings with a ``dev_accuracy`` key (fallback: ``accuracy``)
expressed in **percentage points** (0–100). For convenience, when every
provided accuracy lies in ``[0, 1]`` *and* both thresholds are at least 1.0, the
values are read as fractions and scaled to points (so ``0.71`` with the
default ``3.0`` margin means 71 points). Non-finite or absent student
accuracies are a :class:`GateInputError` (programmer error). For the
*teacher*, a missing/unreadable signal is not an error: the gate returns
``abort`` with ``reason='teacher_unavailable'``. The caller signals this
either by passing ``teacher_metrics=None``, by setting ``available:
false`` (or ``labels_available: false``) on the mapping, or by omitting a
finite ``dev_accuracy``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import GateInputError

#: Reason constants (stable; the report matches on these strings).
REASON_OK = "ok"
REASON_TEACHER_UNAVAILABLE = "teacher_unavailable"
REASON_TEACHER_MARGIN = "teacher_margin_below_threshold"
REASON_KD_GAIN = "kd_gain_below_threshold"

OUTCOME_PASS = "pass"
OUTCOME_ABORT = "abort"

_ACCURACY_KEYS = ("dev_accuracy", "accuracy")
_UNAVAILABLE_KEYS = ("available", "labels_available")


@dataclass(frozen=True)
class GateOutcome:
    """JSON-serializable verdict of :func:`distillation_gate`.

    ``teacher_margin`` is ``teacher_dev − supervised_dev`` (points) or
    ``None`` when the teacher is unavailable. ``kd_gain`` is
    ``kd_dev − supervised_dev`` (points); it is always computed because
    valid student metrics are required. ``negative_result`` is ``True``
    exactly when the outcome is ``abort`` — the report renders that as a
    recorded negative result.
    """

    outcome: str
    reason: str
    teacher_margin: float | None
    kd_gain: float | None
    teacher_accuracy: float | None
    supervised_accuracy: float | None
    kd_accuracy: float | None
    min_teacher_margin: float
    min_kd_gain: float
    checks: Mapping[str, bool]
    negative_result: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "teacher_margin": self.teacher_margin,
            "kd_gain": self.kd_gain,
            "teacher_accuracy": self.teacher_accuracy,
            "supervised_accuracy": self.supervised_accuracy,
            "kd_accuracy": self.kd_accuracy,
            "min_teacher_margin": self.min_teacher_margin,
            "min_kd_gain": self.min_kd_gain,
            "checks": dict(self.checks),
            "negative_result": self.negative_result,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GateOutcome":
        if not isinstance(payload, Mapping):
            raise GateInputError("GateOutcome.from_dict requires a mapping")
        return cls(
            outcome=str(payload["outcome"]),
            reason=str(payload["reason"]),
            teacher_margin=_optional_float(payload.get("teacher_margin")),
            kd_gain=_optional_float(payload.get("kd_gain")),
            teacher_accuracy=_optional_float(payload.get("teacher_accuracy")),
            supervised_accuracy=_optional_float(payload.get("supervised_accuracy")),
            kd_accuracy=_optional_float(payload.get("kd_accuracy")),
            min_teacher_margin=float(payload["min_teacher_margin"]),
            min_kd_gain=float(payload["min_kd_gain"]),
            checks={str(k): bool(v) for k, v in dict(payload.get("checks", {})).items()},
            negative_result=bool(payload["negative_result"]),
        )


def distillation_gate(
    teacher_metrics: Mapping[str, Any] | None,
    student_supervised_metrics: Mapping[str, Any],
    student_kd_metrics: Mapping[str, Any],
    min_teacher_margin: float = 3.0,
    min_kd_gain: float = 1.0,
) -> GateOutcome:
    """Decide whether Stage C distillation may proceed.

    Pure: same inputs, same verdict; no I/O, no training side effects.
    See the module docstring for the metric contract and reason strings.
    """
    margin_floor = _check_threshold(min_teacher_margin, "min_teacher_margin")
    gain_floor = _check_threshold(min_kd_gain, "min_kd_gain")

    supervised = _student_accuracy(student_supervised_metrics, "student_supervised_metrics")
    kd = _student_accuracy(student_kd_metrics, "student_kd_metrics")
    teacher = _teacher_accuracy(teacher_metrics)  # None => unavailable (fail-to-load limb)

    scale = _scale_factor(
        [a for a in (supervised, kd, teacher) if a is not None], margin_floor, gain_floor
    )
    supervised *= scale
    kd *= scale
    if teacher is not None:
        teacher *= scale

    teacher_margin = None if teacher is None else teacher - supervised
    kd_gain = kd - supervised

    margin_ok = teacher_margin is not None and teacher_margin >= margin_floor
    gain_ok = kd_gain >= gain_floor
    checks = {
        "teacher_available": teacher is not None,
        "teacher_margin": margin_ok,
        "kd_gain": gain_ok,
    }

    if teacher is None:
        return _abort(
            REASON_TEACHER_UNAVAILABLE, teacher_margin, kd_gain, teacher,
            supervised, kd, margin_floor, gain_floor, checks,
        )
    if not margin_ok:
        return _abort(
            REASON_TEACHER_MARGIN, teacher_margin, kd_gain, teacher,
            supervised, kd, margin_floor, gain_floor, checks,
        )
    if not gain_ok:
        return _abort(
            REASON_KD_GAIN, teacher_margin, kd_gain, teacher,
            supervised, kd, margin_floor, gain_floor, checks,
        )
    return GateOutcome(
        outcome=OUTCOME_PASS,
        reason=REASON_OK,
        teacher_margin=teacher_margin,
        kd_gain=kd_gain,
        teacher_accuracy=teacher,
        supervised_accuracy=supervised,
        kd_accuracy=kd,
        min_teacher_margin=margin_floor,
        min_kd_gain=gain_floor,
        checks=checks,
        negative_result=False,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _check_threshold(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateInputError(f"{field} must be a real number, got {value!r}", field=field)
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise GateInputError(f"{field} must be finite and >= 0, got {number!r}", field=field)
    return number


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1) and not isinstance(value, bool):
        return bool(value)
    raise GateInputError(f"{field} must be a boolean, got {value!r}", field=field)


def _accuracy_value(metrics: Mapping[str, Any]) -> float | None:
    """Finite accuracy from a mapping, or ``None`` when absent/invalid."""
    for key in _ACCURACY_KEYS:
        if key in metrics:
            raw = metrics[key]
            if isinstance(raw, bool):
                return None
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None
    return None


def _student_accuracy(metrics: Any, field: str) -> float:
    if not isinstance(metrics, Mapping):
        raise GateInputError(
            f"{field} must be a mapping with a dev_accuracy key, "
            f"got {type(metrics).__name__}",
            field=field,
        )
    value = _accuracy_value(metrics)
    if value is None:
        raise GateInputError(
            f"{field} must include a finite dev_accuracy (or accuracy) in percentage "
            f"points (0-100)",
            field=field,
        )
    return value


def _teacher_accuracy(metrics: Any) -> float | None:
    """The teacher's dev accuracy in points, or ``None`` when unavailable.

    ``None`` (the argument, not the value), any mapped ``available`` /
    ``labels_available`` flag set to a falsy value, or a missing/non-finite
    ``dev_accuracy`` all signal missing/unreadable cached soft labels (E8's
    fail-to-load limb) — the gate then records a negative result rather than
    raising. A non-mapping argument is malformed input and raises
    :class:`GateInputError`.
    """
    if metrics is None:
        return None
    if not isinstance(metrics, Mapping):
        raise GateInputError(
            f"teacher_metrics must be a mapping or None, got {type(metrics).__name__}",
            field="teacher_metrics",
        )
    for key in _UNAVAILABLE_KEYS:
        if key in metrics and not _as_bool(metrics[key], key):
            return None
    return _accuracy_value(metrics)


def _abort(
    reason: str,
    teacher_margin: float | None,
    kd_gain: float,
    teacher: float | None,
    supervised: float,
    kd: float,
    margin_floor: float,
    gain_floor: float,
    checks: Mapping[str, bool],
) -> GateOutcome:
    return GateOutcome(
        outcome=OUTCOME_ABORT,
        reason=reason,
        teacher_margin=teacher_margin,
        kd_gain=kd_gain,
        teacher_accuracy=teacher,
        supervised_accuracy=supervised,
        kd_accuracy=kd,
        min_teacher_margin=margin_floor,
        min_kd_gain=gain_floor,
        checks=checks,
        negative_result=True,
    )


def _scale_factor(
    accuracies: list[float], margin: float, gain: float
) -> float:
    """Return 100.0 when every provided accuracy is a fraction in [0, 1] and
    both thresholds are >= 1.0 (auto-normalisation), else 1.0."""
    provided = accuracies
    if provided and all(0.0 <= a <= 1.0 for a in provided) and min(margin, gain) >= 1.0:
        return 100.0
    return 1.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
