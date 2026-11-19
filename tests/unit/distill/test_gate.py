"""FR-DISTILL-02 gate (E8, risk R9) — pure decision function tests.

No cache, no training, no I/O: the gate decides from metrics alone.
"""

from __future__ import annotations

import json
import math

import pytest

from src.distill import (
    GateOutcome,
    distillation_gate,
)
from src.distill.errors import GateInputError
from src.distill.gate import (
    OUTCOME_ABORT,
    OUTCOME_PASS,
    REASON_KD_GAIN,
    REASON_OK,
    REASON_TEACHER_MARGIN,
    REASON_TEACHER_UNAVAILABLE,
)


def _m(dev_accuracy: float) -> dict[str, float]:
    return {"dev_accuracy": dev_accuracy}


def test_pass_when_teacher_clears_margin_and_kd_gain() -> None:
    outcome = distillation_gate(_m(75.0), _m(70.0), _m(72.0))
    assert outcome.outcome == OUTCOME_PASS
    assert outcome.reason == REASON_OK
    assert outcome.teacher_margin == pytest.approx(5.0)
    assert outcome.kd_gain == pytest.approx(2.0)
    assert outcome.negative_result is False
    assert outcome.checks == {
        "teacher_available": True,
        "teacher_margin": True,
        "kd_gain": True,
    }


def test_boundary_margins_pass() -> None:
    outcome = distillation_gate(_m(73.0), _m(70.0), _m(71.0))  # exactly +3.0 and +1.0
    assert outcome.outcome == OUTCOME_PASS
    assert outcome.teacher_margin == pytest.approx(3.0)
    assert outcome.kd_gain == pytest.approx(1.0)


def test_teacher_below_margin_aborts_with_named_reason() -> None:
    outcome = distillation_gate(_m(72.9), _m(70.0), _m(75.0))  # margin 2.9 < 3.0
    assert outcome.outcome == OUTCOME_ABORT
    assert outcome.reason == REASON_TEACHER_MARGIN
    assert outcome.teacher_margin == pytest.approx(2.9)
    assert outcome.negative_result is True
    assert outcome.checks["teacher_margin"] is False


def test_teacher_equal_to_student_aborts() -> None:
    outcome = distillation_gate(_m(70.0), _m(70.0), _m(75.0))
    assert outcome.outcome == OUTCOME_ABORT
    assert outcome.reason == REASON_TEACHER_MARGIN
    assert outcome.teacher_margin == 0.0


def test_teacher_weaker_than_student_aborts() -> None:
    outcome = distillation_gate(_m(65.0), _m(70.0), _m(80.0))
    assert outcome.outcome == OUTCOME_ABORT
    assert outcome.reason == REASON_TEACHER_MARGIN
    assert outcome.teacher_margin == pytest.approx(-5.0)


def test_kd_gain_below_one_point_aborts() -> None:
    outcome = distillation_gate(_m(80.0), _m(70.0), _m(70.9))  # kd gain 0.9 < 1.0
    assert outcome.outcome == OUTCOME_ABORT
    assert outcome.reason == REASON_KD_GAIN
    assert outcome.kd_gain == pytest.approx(0.9)
    assert outcome.teacher_margin == pytest.approx(10.0)
    assert outcome.negative_result is True
    assert outcome.checks["teacher_margin"] is True
    assert outcome.checks["kd_gain"] is False


def test_kd_student_equal_to_supervised_aborts() -> None:
    outcome = distillation_gate(_m(80.0), _m(70.0), _m(70.0))
    assert outcome.outcome == OUTCOME_ABORT
    assert outcome.reason == REASON_KD_GAIN


def test_teacher_none_means_teacher_unavailable() -> None:
    outcome = distillation_gate(None, _m(70.0), _m(72.0))
    assert outcome.outcome == OUTCOME_ABORT
    assert outcome.reason == REASON_TEACHER_UNAVAILABLE
    assert outcome.teacher_margin is None
    assert outcome.kd_gain == pytest.approx(2.0)  # still computed; teacher limb dominates
    assert outcome.teacher_accuracy is None
    assert outcome.checks["teacher_available"] is False
    assert outcome.negative_result is True


@pytest.mark.parametrize("flag", ["available", "labels_available"])
def test_teacher_available_false_means_unavailable(flag: str) -> None:
    metrics = {"dev_accuracy": 99.0, flag: False}  # would pass margin if trusted
    outcome = distillation_gate(metrics, _m(70.0), _m(72.0))
    assert outcome.outcome == OUTCOME_ABORT
    assert outcome.reason == REASON_TEACHER_UNAVAILABLE
    assert outcome.teacher_margin is None


def test_teacher_missing_dev_accuracy_means_unavailable() -> None:
    # cached soft labels missing/unreadable -> no teacher signal at all
    outcome = distillation_gate({"notes": "labels unreadable"}, _m(70.0), _m(72.0))
    assert outcome.outcome == OUTCOME_ABORT
    assert outcome.reason == REASON_TEACHER_UNAVAILABLE


def test_teacher_non_finite_dev_accuracy_means_unavailable() -> None:
    outcome = distillation_gate(_m(math.nan), _m(70.0), _m(72.0))
    assert outcome.outcome == OUTCOME_ABORT
    assert outcome.reason == REASON_TEACHER_UNAVAILABLE


def test_fraction_scale_is_auto_normalised() -> None:
    outcome = distillation_gate(_m(0.75), _m(0.70), _m(0.72))
    assert outcome.outcome == OUTCOME_PASS
    assert outcome.teacher_margin == pytest.approx(5.0)
    assert outcome.kd_gain == pytest.approx(2.0)


def test_outcome_is_json_serializable_and_round_trips() -> None:
    outcome = distillation_gate(_m(80.0), _m(70.0), _m(73.5))
    encoded = outcome.to_json()
    decoded = json.loads(encoded)
    for key in ("outcome", "reason", "teacher_margin", "kd_gain",
                "min_teacher_margin", "min_kd_gain", "negative_result"):
        assert key in decoded
    restored = GateOutcome.from_dict(decoded)
    assert restored == outcome


def test_abort_outcome_round_trips_with_null_margin() -> None:
    outcome = distillation_gate(None, _m(70.0), _m(72.0))
    decoded = json.loads(outcome.to_json())
    assert decoded["teacher_margin"] is None
    assert decoded["reason"] == "teacher_unavailable"
    assert GateOutcome.from_dict(decoded) == outcome


def test_student_metrics_must_be_mappings() -> None:
    with pytest.raises(GateInputError):
        distillation_gate(_m(80.0), "not-a-mapping", _m(72.0))  # type: ignore[arg-type]
    with pytest.raises(GateInputError):
        distillation_gate(_m(80.0), _m(70.0), None)  # type: ignore[arg-type]


def test_student_metrics_require_finite_dev_accuracy() -> None:
    with pytest.raises(GateInputError):
        distillation_gate(_m(80.0), {"dev_accuracy": "unreadable"}, _m(72.0))
    with pytest.raises(GateInputError):
        distillation_gate(_m(80.0), {"dev_accuracy": math.inf}, _m(72.0))
    with pytest.raises(GateInputError):
        distillation_gate(_m(80.0), {}, _m(72.0))


def test_teacher_metrics_non_mapping_raises() -> None:
    with pytest.raises(GateInputError):
        distillation_gate("not-a-mapping", _m(70.0), _m(72.0))  # type: ignore[arg-type]


def test_thresholds_are_validated() -> None:
    with pytest.raises(GateInputError):
        distillation_gate(_m(80.0), _m(70.0), _m(72.0), min_teacher_margin=-1.0)
    with pytest.raises(GateInputError):
        distillation_gate(_m(80.0), _m(70.0), _m(72.0), min_kd_gain=math.nan)


def test_margin_checked_before_gain() -> None:
    # both limbs fail -> teacher limb is named (teacher_margin is E8's first gate)
    outcome = distillation_gate(_m(70.0), _m(70.0), _m(70.0))
    assert outcome.reason == REASON_TEACHER_MARGIN


def test_custom_thresholds_are_honoured() -> None:
    outcome = distillation_gate(
        _m(71.5), _m(70.0), _m(70.5), min_teacher_margin=1.0, min_kd_gain=0.25
    )
    assert outcome.outcome == OUTCOME_PASS
    outcome = distillation_gate(
        _m(71.5), _m(70.0), _m(70.5), min_teacher_margin=2.0, min_kd_gain=0.25
    )
    assert outcome.outcome == OUTCOME_ABORT
    assert outcome.reason == REASON_TEACHER_MARGIN


def test_gate_has_no_training_dependency() -> None:
    import src.distill.gate as gate_module

    source_names = set(vars(gate_module))
    assert "train" not in source_names
    assert "trainer" not in source_names
