"""Tests for src/calibration/risk.py — per-category thresholds on dev safe rows.

Fixture for the binding-constraint case (category "violence"):
  positives: 4 x score 0.75, 2 x score 0.65          -> 6 positives
  safe rows: 0.9, 0.8, 0.7, 0.65, 0.62, then 15 x 0.01 -> 20 safe rows
  floor(0.10 * 20) = 2 safe rows may be flagged.
  t = 0.75 flags safe {0.9, 0.8} (2) and tp = 4  -> recall 4/6 ≈ 0.6667 (feasible)
  t = 0.70 would flag 3 safe rows (> budget)     -> infeasible
  t = 0.65 would flag 4 safe rows (> budget)     -> infeasible
Unconstrained, recall 1.0 would be reachable; the FPR budget caps it at 4/6.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.errors import CalibrationInputError  # noqa: E402
from src.calibration.risk import (  # noqa: E402
    DEFAULT_MAX_FPR,
    RISK_CATEGORIES,
    RISK_CATEGORY_COUNT,
    fit_risk_thresholds,
)


def _binding_fixture():
    scores = (
        [[0.75]] * 4
        + [[0.65]] * 2
        + [[0.9], [0.8], [0.7], [0.65], [0.62]]
        + [[0.01]] * 15
    )
    targets = [[1]] * 6 + [[0]] * 20
    return scores, targets


def test_categories_are_the_eight_declared_slots():
    assert RISK_CATEGORY_COUNT == 8
    assert RISK_CATEGORIES == (
        "violence",
        "self_harm",
        "sexual_content",
        "hate_speech",
        "harassment",
        "dangerous_content",
        "illegal_activity",
        "privacy_violation",
    )
    assert DEFAULT_MAX_FPR == 0.10


def test_fit_maximizes_recall_subject_to_fpr_budget():
    scores, targets = _binding_fixture()
    out = fit_risk_thresholds(scores, targets, categories=("violence",))
    entry = out["violence"]
    assert entry["threshold"] == pytest.approx(0.75, abs=1e-12)
    assert entry["recall"] == pytest.approx(4 / 6, abs=1e-12)
    assert entry["n_flagged_safe"] == 2
    assert entry["fpr"] == pytest.approx(0.1, abs=1e-12)
    assert entry["n_positive"] == 6
    assert entry["n_safe"] == 20
    assert entry["max_fpr"] == 0.10
    assert entry["reason"] == "fitted"
    assert entry["xstest_used_for_fitting"] is False
    json.dumps(out)  # JSON-serializable


def test_explicit_safe_mask_is_honoured():
    scores, targets = _binding_fixture()
    # safe_mask picks 5 high-score safe rows + 10 low-score safe rows ->
    # n_safe = 15, so floor(0.1 * 15) = 1 safe row may be flagged.
    safe_mask = [False] * 6 + [True, True, True, True, True] + [True] * 10 + [False] * 5
    out = fit_risk_thresholds(scores, targets, categories=("violence",), safe_mask=safe_mask)
    entry = out["violence"]
    assert entry["n_safe"] == 15
    assert entry["max_safe_flagged"] == 1
    assert entry["n_flagged_safe"] <= 1
    # Budget is 1: t = 0.75 flags {0.9, 0.8} (2 > 1) -> infeasible; t = 0.8 flags
    # {0.9, 0.8} (2 > 1) -> infeasible; t = 0.9 flags {0.9} (1 <= 1), tp = 0.
    assert entry["threshold"] == pytest.approx(0.9, abs=1e-12)
    assert entry["recall"] == pytest.approx(0.0, abs=1e-12)


def test_zero_budget_flags_nothing_on_safe_rows():
    # 5 safe rows -> floor(0.1 * 5) = 0 safe rows may be flagged.
    scores = [[0.8]] * 5 + [[0.9]] * 5
    targets = [[1]] * 5 + [[0]] * 5
    out = fit_risk_thresholds(scores, targets, categories=("violence",))
    entry = out["violence"]
    assert entry["max_safe_flagged"] == 0
    assert entry["n_flagged_safe"] == 0
    assert entry["recall"] == pytest.approx(0.0, abs=1e-12)
    assert entry["threshold"] == pytest.approx(max(0.9, 0.8) + 1.0, abs=1e-12)


def test_no_positive_rows_yields_null_threshold():
    scores = [[0.1]] * 4 + [[0.2]] * 16
    targets = [[0]] * 20
    out = fit_risk_thresholds(scores, targets, categories=("violence",))
    entry = out["violence"]
    assert entry["threshold"] is None
    assert entry["recall"] is None
    assert entry["reason"] == "no_positive_rows"
    assert entry["n_positive"] == 0
    json.dumps(out)


def test_no_safe_rows_gives_vacuous_fpr_and_recall_one():
    scores = [[0.3], [0.8], [0.5]]
    targets = [[1], [1], [1]]
    out = fit_risk_thresholds(
        scores, targets, categories=("violence",), safe_mask=[False, False, False]
    )
    entry = out["violence"]
    assert entry["reason"] == "no_safe_rows"
    assert entry["recall"] == pytest.approx(1.0, abs=1e-12)
    assert entry["fpr"] is None
    assert entry["n_safe"] == 0


def test_eight_category_fit_runs_columnwise():
    n = 30
    scores = [[0.01 * ((i * 7 + j * 13) % 10) + (0.5 if (i * 3 + j) % 11 == 0 else 0.0)
               for j in range(8)] for i in range(n)]
    targets = [[1 if (i + j) % 7 == 0 else 0 for j in range(8)] for i in range(n)]
    out = fit_risk_thresholds(scores, targets)
    assert list(out) == list(RISK_CATEGORIES)
    for entry in out.values():
        assert entry["fpr"] is None or entry["fpr"] <= 0.10 + 1e-12
        assert entry["n_flagged_safe"] <= entry["max_safe_flagged"]
    json.dumps(out)


def test_invalid_inputs_raise():
    scores, targets = _binding_fixture()
    with pytest.raises(CalibrationInputError):
        fit_risk_thresholds([], [], categories=("violence",))
    with pytest.raises(CalibrationInputError):
        fit_risk_thresholds(scores, targets[:-1], categories=("violence",))
    with pytest.raises(CalibrationInputError):
        fit_risk_thresholds(scores, targets, categories=("violence", "violence"))
    with pytest.raises(CalibrationInputError):
        fit_risk_thresholds(scores, targets, categories=())
    with pytest.raises(CalibrationInputError):
        fit_risk_thresholds(scores, targets, categories=("violence",), max_fpr=1.5)
    with pytest.raises(CalibrationInputError):
        fit_risk_thresholds(scores, targets, categories=("violence",), max_fpr=True)  # type: ignore[arg-type]
    with pytest.raises(CalibrationInputError):
        fit_risk_thresholds([[float("nan")]], [[1]], categories=("violence",))
    with pytest.raises(CalibrationInputError):
        fit_risk_thresholds([[0.5]], [[2]], categories=("violence",))
    with pytest.raises(CalibrationInputError):
        fit_risk_thresholds(scores, targets, categories=("violence",),
                            safe_mask=[True] * (len(scores) - 1))
    with pytest.raises(CalibrationInputError):
        # row width mismatch against the category count
        fit_risk_thresholds([[0.1, 0.2]], [[1, 1]], categories=("violence",))
