"""Tests for src/calibration/abstain.py — utility-derived policy and risk–coverage."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.abstain import (  # noqa: E402
    DEFAULT_U_ABSTAIN,
    DEFAULT_U_CORRECT,
    DEFAULT_U_WRONG,
    AbstentionPolicy,
    abstention_threshold,
    risk_coverage_curve,
)
from src.calibration.errors import CalibrationInputError  # noqa: E402


def test_default_utilities_yield_075():
    assert (DEFAULT_U_CORRECT, DEFAULT_U_WRONG, DEFAULT_U_ABSTAIN) == (1.0, -3.0, 0.0)
    assert abstention_threshold() == pytest.approx(0.75, abs=1e-12)
    policy = AbstentionPolicy()
    assert policy.threshold == pytest.approx(0.75, abs=1e-12)


def test_threshold_derives_from_constructor_utilities():
    # (u_a - u_w) / (u_c - u_w) = (0 - (-1)) / (2 - (-1)) = 1/3
    assert abstention_threshold(2.0, -1.0, 0.0) == pytest.approx(1.0 / 3.0, abs=1e-12)
    # (0.4 + 3) / (1 + 3) = 0.85
    policy = AbstentionPolicy(u_correct=1.0, u_wrong=-3.0, u_abstain=0.4)
    assert policy.threshold == pytest.approx(0.85, abs=1e-12)
    # abstain utility above the correct utility still computes (threshold > 1 -> always abstain)
    assert abstention_threshold(0.0, -1.0, 5.0) == pytest.approx(6.0, abs=1e-12)


def test_equal_answer_utilities_raise():
    with pytest.raises(CalibrationInputError):
        abstention_threshold(1.0, 1.0, 0.0)
    with pytest.raises(CalibrationInputError):
        AbstentionPolicy(u_correct=-1.0, u_wrong=-1.0)


def test_non_finite_utilities_raise():
    with pytest.raises(CalibrationInputError):
        AbstentionPolicy(u_correct=math.inf)
    with pytest.raises(CalibrationInputError):
        abstention_threshold(u_abstain=math.nan)


def test_should_abstain_boundary():
    policy = AbstentionPolicy()
    assert policy.should_abstain(0.74) is True
    assert policy.should_abstain(0.75) is False  # rule is p strictly below threshold
    assert policy.should_abstain(0.9) is False
    with pytest.raises(CalibrationInputError):
        policy.should_abstain(1.5)
    with pytest.raises(CalibrationInputError):
        policy.should_abstain(-0.1)


def test_policy_round_trip():
    policy = AbstentionPolicy(2.0, -4.0, 0.5)
    payload = policy.to_dict()
    assert payload["threshold"] == pytest.approx(policy.threshold)
    restored = AbstentionPolicy.from_dict(payload)
    assert restored == policy
    with pytest.raises(CalibrationInputError):
        AbstentionPolicy.from_dict({"u_correct": "not-a-number"})


def test_risk_coverage_curve_hand_computed():
    confidences = [0.9, 0.5, 0.7, 0.2]
    correctness = [1, 0, 1, 0]
    curve = risk_coverage_curve(confidences, correctness)
    # Sorted by confidence desc: 0.9(y=1), 0.7(y=1), 0.5(y=0), 0.2(y=0)
    assert curve["n_items"] == 4
    assert curve["sorted_p"] == [0.9, 0.7, 0.5, 0.2]
    assert curve["coverage"] == [0.25, 0.5, 0.75, 1.0]
    assert curve["selective_accuracy"] == [1.0, 1.0, pytest.approx(2 / 3), 0.5]
    assert curve["risk"] == [0.0, 0.0, pytest.approx(1 / 3), 0.5]
    assert curve["aurc"] == pytest.approx((0.0 + 0.0 + 1 / 3 + 0.5) / 4)
    points = curve["points"]
    assert len(points) == 4
    assert points[0]["retained"] == 1
    assert points[0]["threshold"] == 0.9
    assert points[-1]["coverage"] == 1.0
    # The whole structure must be JSON-serializable.
    json.dumps(curve)


def test_risk_coverage_curve_stable_ties():
    confidences = [0.5, 0.5, 0.5]
    correctness = [1, 0, 0]
    curve = risk_coverage_curve(confidences, correctness)
    # Ties keep input order: the y=1 item is retained first.
    assert curve["selective_accuracy"][0] == 1.0
    assert curve["selective_accuracy"][-1] == pytest.approx(1 / 3)


def test_risk_coverage_curve_invalid_input():
    with pytest.raises(CalibrationInputError):
        risk_coverage_curve([], [])
    with pytest.raises(CalibrationInputError):
        risk_coverage_curve([0.5], [1, 0])
    with pytest.raises(CalibrationInputError):
        risk_coverage_curve([1.5], [1])
    with pytest.raises(CalibrationInputError):
        risk_coverage_curve([0.5], [2])
    with pytest.raises(CalibrationInputError):
        risk_coverage_curve([0.5], [True])  # bools rejected, mirroring metrics
