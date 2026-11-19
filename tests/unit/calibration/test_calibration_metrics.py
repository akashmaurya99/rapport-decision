"""Tests for src/calibration/metrics.py — hand-computed fixtures.

Fixture pairs are (p, y) = [(0.9, 1), (0.8, 1), (0.4, 0), (0.1, 0)].

* bins = 5, edges [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1.0]:
  bin0 (0.1, y=0); bin2 (0.4, y=0); bin4 (0.9, y=1), (0.8, y=1).
  bin0: n=1, conf=0.1, acc=0.0  -> |0.1 - 0.0| = 0.1
  bin2: n=1, conf=0.4, acc=0.0  -> 0.4
  bin4: n=2, conf=0.85, acc=1.0 -> 0.15
  pooled ECE      = (1/4)(0.1) + (1/4)(0.4) + (2/4)(0.15) = 0.2
  bucket-macro    = (0.1 + 0.4 + 0.15) / 3            = 0.65 / 3 ≈ 0.216667
* brier = ((.1)² + (.2)² + (.4)² + (.1)²) / 4 = (0.01+0.04+0.16+0.01)/4 = 0.055
* nll   = -(ln .9 + ln .8 + ln .6 + ln .9) / 4 ≈ 0.236173
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration import metrics  # noqa: E402
from src.calibration.errors import CalibrationInputError  # noqa: E402

P = [0.9, 0.8, 0.4, 0.1]
Y = [1, 1, 0, 0]


def test_pooled_ece_hand_computed():
    assert metrics.ece(P, Y, bins=5) == pytest.approx(0.2, abs=1e-12)


def test_bucket_macro_ece_hand_computed():
    assert metrics.ece_bucket_macro(P, Y, bins=5) == pytest.approx(0.65 / 3, abs=1e-12)


def test_ece_empty_bins_are_skipped():
    # Only bins 0 and 4 non-empty; macro averages just those two.
    assert metrics.ece([0.1, 0.9], [0, 1], bins=5) == pytest.approx(
        0.5 * 0.1 + 0.5 * 0.1, abs=1e-12
    )
    assert metrics.ece_bucket_macro([0.1, 0.9], [0, 1], bins=5) == pytest.approx(0.1, abs=1e-12)


def test_ece_p_one_lands_in_last_bin():
    # p = 1.0 must be counted (right edge closed), not dropped.
    assert metrics.ece([1.0], [1], bins=15) == pytest.approx(0.0, abs=1e-12)
    assert metrics.ece([1.0], [0], bins=15) == pytest.approx(1.0, abs=1e-12)


def test_ece_edge_probability_lands_in_upper_bin():
    # bins=5; p = 0.4 is an exact edge and belongs to bin 2 ([.4,.6)), not bin 1.
    only = metrics.ece([0.4], [0], bins=5)
    assert only == pytest.approx(0.4, abs=1e-12)


def test_ece_default_bins_is_15():
    assert metrics.ECE_BINS == 15
    assert metrics.ece(P, Y) == pytest.approx(metrics.ece(P, Y, bins=15), abs=0.0)


def test_brier_hand_computed():
    assert metrics.brier_score(P, Y) == pytest.approx(0.055, abs=1e-12)


def test_nll_hand_computed():
    expected = -(math.log(0.9) + math.log(0.8) + math.log(1 - 0.4) + math.log(1 - 0.1)) / 4
    assert metrics.nll(P, Y) == pytest.approx(expected, abs=1e-12)


def test_nll_clamps_probabilities():
    # p = 0 and y = 1 would be inf without clamping; with eps it is finite.
    value = metrics.nll([0.0], [1])
    assert math.isfinite(value)
    assert value == pytest.approx(-math.log(metrics.NLL_EPS), abs=1e-6)


def test_perfect_calibration_ece_zero():
    assert metrics.ece([1.0, 1.0], [1, 1]) == pytest.approx(0.0, abs=1e-12)


def test_invalid_inputs_raise():
    with pytest.raises(CalibrationInputError):
        metrics.ece([], [])
    with pytest.raises(CalibrationInputError):
        metrics.ece([1.5], [1])  # out of [0, 1]
    with pytest.raises(CalibrationInputError):
        metrics.ece([0.5], [2])  # not binary
    with pytest.raises(CalibrationInputError):
        metrics.ece([0.5, 0.6], [1])  # length mismatch
    with pytest.raises(CalibrationInputError):
        metrics.ece([float("nan")], [1])
    with pytest.raises(CalibrationInputError):
        metrics.ece(P, Y, bins=0)
    with pytest.raises(CalibrationInputError):
        metrics.brier_score([], [])
    with pytest.raises(CalibrationInputError):
        metrics.nll(P, Y, eps=0.0)


def test_multiclass_nll_hand_computed():
    # Rows: [1, 0] with label 0, [0, 1] with label 0, temperature 1.
    p0 = math.e / (math.e + 1.0)
    expected = -(math.log(p0) + math.log(1.0 - p0)) / 2
    assert metrics.multiclass_nll([[1.0, 0.0], [0.0, 1.0]], [0, 0], 1.0) == pytest.approx(
        expected, abs=1e-12
    )


def test_multiclass_nll_rejects_bad_temperature():
    with pytest.raises(CalibrationInputError):
        metrics.multiclass_nll([[1.0, 0.0]], [0], 0.0)
    with pytest.raises(CalibrationInputError):
        metrics.multiclass_nll([[1.0, 0.0]], [0], -1.0)
    with pytest.raises(CalibrationInputError):
        metrics.multiclass_nll([[1.0]], [0], 1.0)  # need >= 2 options
    with pytest.raises(CalibrationInputError):
        metrics.multiclass_nll([[1.0, 0.0]], [2], 1.0)  # label out of range


def test_confidence_correctness_uses_argmax():
    logits = [[3.0, 0.0], [0.0, 3.0], [1.0, 2.0]]
    labels = [0, 1, 0]  # third row argmax is 1 -> incorrect
    confidences, correctness = metrics.confidence_correctness(logits, labels, 1.0)
    assert correctness == [1, 1, 0]
    assert confidences[0] == pytest.approx(metrics.softmax_probs([3.0, 0.0])[0])


def test_confidence_correctness_temperature_softens():
    logits = [[4.0, 0.0]] * 4
    labels = [0, 0, 0, 0]
    c_hot, _ = metrics.confidence_correctness(logits, labels, 0.5)
    c_cold, _ = metrics.confidence_correctness(logits, labels, 5.0)
    assert c_hot[0] > c_cold[0] > 0.5


def test_compute_calibration_metrics_aggregate():
    result = metrics.compute_calibration_metrics(P, Y, bins=5)
    assert result.n_items == 4
    assert result.bins == 5
    assert result.pooled_ece == pytest.approx(0.2, abs=1e-12)
    assert result.bucket_macro_ece == pytest.approx(0.65 / 3, abs=1e-12)
    assert result.brier == pytest.approx(0.055, abs=1e-12)
    assert result.nll == pytest.approx(metrics.nll(P, Y), abs=1e-12)
    assert result.to_dict()["n_items"] == 4


def test_softmax_probs_sum_to_one_and_stable():
    probs = metrics.softmax_probs([1000.0, 1000.0, 999.0])
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)
    assert all(math.isfinite(v) for v in probs)


def test_softmax_rejects_non_positive_temperature():
    with pytest.raises(CalibrationInputError):
        metrics.softmax_probs([1.0, 2.0], 0.0)
