"""Tests for src/calibration/map.py — persistence, JSON round-trip, apply."""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.abstain import AbstentionPolicy  # noqa: E402
from src.calibration.buckets import make_bucket_key_for_count  # noqa: E402
from src.calibration.errors import CalibrationInputError, CalibrationMapError  # noqa: E402
from src.calibration.map import (  # noqa: E402
    CalibrationEntry,
    CalibrationMap,
    calibration_path_for_run,
    load_calibration_map,
    save_calibration_map,
)
from src.calibration.risk import fit_risk_thresholds  # noqa: E402
from src.calibration.temperature import fit_temperature_map  # noqa: E402


def _small_dev(seed: int = 99):
    rng = np.random.RandomState(seed)
    logits, labels, buckets, groups = [], [], [], []
    specs = [("choice", 2, 220), ("scoring", 4, 55), ("preference", 2, 25)]
    for family, n_options, n in specs:
        for i in range(n):
            lab = int(rng.randint(n_options))
            z = rng.randn(n_options) * 0.5
            if rng.rand() < 0.75:
                z[lab] += 2.0
            logits.append(z.tolist())
            labels.append(lab)
            buckets.append(make_bucket_key_for_count(family, n_options))
            groups.append(f"{family}-g{i // 5}")
    return logits, labels, buckets, groups


@pytest.fixture()
def fitted_map() -> CalibrationMap:
    logits, labels, buckets, groups = _small_dev()
    return fit_temperature_map(logits, labels, buckets, 5, groups, run_id="t16-test")


def test_json_round_trip_is_exact(fitted_map):
    payload = fitted_map.to_dict()
    json.dumps(payload)  # JSON-serializable
    restored = CalibrationMap.from_dict(json.loads(json.dumps(payload)))
    assert restored == fitted_map


def test_save_and_load_explicit_path(fitted_map, tmp_path):
    target = tmp_path / "nested" / "calibration.json"
    save_calibration_map(fitted_map, target)
    assert target.exists()
    loaded = load_calibration_map(target)
    assert loaded == fitted_map
    # Deterministic bytes: saving twice yields identical output.
    other = tmp_path / "other.json"
    save_calibration_map(loaded, other)
    assert other.read_bytes() == target.read_bytes()


def test_default_run_path_convention(tmp_path):
    path = calibration_path_for_run("run123", runs_dir=tmp_path)
    assert path == tmp_path / "run123" / "calibration.json"


def test_run_id_validation(tmp_path):
    with pytest.raises(CalibrationMapError):
        calibration_path_for_run("", runs_dir=tmp_path)
    with pytest.raises(CalibrationMapError):
        calibration_path_for_run("a/b", runs_dir=tmp_path)
    with pytest.raises(CalibrationMapError):
        calibration_path_for_run("..", runs_dir=tmp_path)


def test_load_missing_and_invalid_json(tmp_path):
    with pytest.raises(CalibrationMapError) as exc_info:
        load_calibration_map(tmp_path / "nope.json")
    assert exc_info.value.code == "MAP_NOT_FOUND"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(CalibrationMapError) as exc_info:
        load_calibration_map(bad)
    assert exc_info.value.code == "MAP_INVALID_JSON"
    with pytest.raises(CalibrationMapError):
        CalibrationMap.from_dict({"folds": 5})


def test_map_freezes_default_abstention(fitted_map):
    assert fitted_map.abstention["u_correct"] == 1.0
    assert fitted_map.abstention["u_wrong"] == -3.0
    assert fitted_map.abstention["u_abstain"] == 0.0
    assert fitted_map.abstention["threshold"] == pytest.approx(0.75)
    policy = fitted_map.abstention_policy()
    assert policy.threshold == pytest.approx(0.75)


def test_custom_abstention_frozen_into_map():
    logits, labels, buckets, groups = _small_dev()
    policy = AbstentionPolicy(u_correct=2.0, u_wrong=-1.0, u_abstain=0.0)
    m = fit_temperature_map(logits, labels, buckets, 5, groups, abstention=policy)
    assert m.abstention["threshold"] == pytest.approx(1.0 / 3.0)
    assert m.abstention_policy() == policy


def test_risk_thresholds_frozen_into_map(fitted_map):
    scores = [[0.1] * 8, [0.9] * 8] + [[0.05] * 8] * 10
    targets = [[0] * 8, [1] * 8] + [[0] * 8] * 10
    thresholds = fit_risk_thresholds(scores, targets)
    frozen = fitted_map.with_risk_thresholds(thresholds)
    assert frozen.risk_thresholds is not None
    assert set(frozen.risk_thresholds) == set(thresholds)
    # Frozen values survive the JSON round-trip.
    restored = CalibrationMap.from_dict(json.loads(json.dumps(frozen.to_dict())))
    assert restored.risk_thresholds == frozen.risk_thresholds
    # The original map is unchanged (frozen maps are immutable snapshots).
    assert fitted_map.risk_thresholds is None


def test_risk_thresholds_attached_at_fit_time():
    logits, labels, buckets, groups = _small_dev()
    scores = [[0.1] * 8] * 5
    targets = [[0] * 8] * 5
    thresholds = fit_risk_thresholds(scores, targets)
    m = fit_temperature_map(logits, labels, buckets, 5, groups, risk_thresholds=thresholds)
    assert m.risk_thresholds == thresholds


def test_apply_temperature_entry(fitted_map):
    entry = fitted_map.entry_for("choice|2")
    assert entry.method == "temperature"
    out = fitted_map.apply([2.0, 0.1], "choice|2")
    assert math.isclose(sum(out.probabilities), 1.0, abs_tol=1e-9)
    assert out.confidence == pytest.approx(max(out.probabilities))
    assert out.method == "temperature"
    assert out.bucket_key == "choice|2"
    assert out.temperature == pytest.approx(entry.temperature)
    assert out.abstain is False  # confidence >> 0.75


def test_apply_abstains_below_frozen_threshold(fitted_map):
    out = fitted_map.apply([0.3, 0.2], "choice|2")  # near-uniform logits
    assert out.confidence < 0.75
    assert out.abstain is True


def test_apply_platt_entry(fitted_map):
    entry = fitted_map.entry_for("scoring|3-5")
    assert entry.method == "platt"
    out = fitted_map.apply([0.4, 1.2, 0.1, 2.0], "scoring|3-5")
    assert math.isclose(sum(out.probabilities), 1.0, abs_tol=1e-9)
    assert out.method == "platt"
    assert out.temperature == pytest.approx(1.0)


def test_apply_inherited_bucket_uses_pooled_temperature(fitted_map):
    out = fitted_map.apply([1.5, 0.0], "preference|2")
    assert out.method == "inherited_temperature"
    assert out.temperature == pytest.approx(fitted_map.pooled.temperature)


def test_apply_unknown_bucket_strict_and_lenient(fitted_map):
    with pytest.raises(CalibrationMapError):
        fitted_map.apply([1.0, 0.0], "choice|6-8")
    out = fitted_map.apply([1.0, 0.0], "choice|6-8", strict=False)
    assert out.bucket_key == "pooled"
    with pytest.raises(CalibrationInputError):
        fitted_map.apply([1.0, 0.0], "not-a-bucket")


def test_apply_pooled_when_no_bucket_given(fitted_map):
    out = fitted_map.apply([2.0, 0.1])
    assert out.bucket_key == "pooled"


def test_xstest_true_flag_is_rejected(fitted_map, tmp_path):
    assert fitted_map.xstest_used_for_fitting is False
    payload = fitted_map.to_dict()
    payload["xstest_used_for_fitting"] = True  # tampered file
    with pytest.raises(CalibrationMapError):
        CalibrationMap.from_dict(payload)
    # Constructing the map directly with a true flag is rejected too.
    with pytest.raises(CalibrationMapError):
        dataclasses.replace(fitted_map, xstest_used_for_fitting=True)


def test_malformed_frozen_abstention_policy_is_rejected(fitted_map):
    payload = fitted_map.to_dict()
    payload["abstention"] = {"u_correct": "not-a-number"}
    with pytest.raises(CalibrationMapError):
        CalibrationMap.from_dict(payload)


def test_calibration_entry_validation():
    entry = CalibrationEntry(
        bucket_key="choice|2",
        method="platt",
        temperature=1.0,
        n_items=60,
        ece_before=0.1,
        ece_after=0.05,
        folds=0,
        platt={"a": 1.0, "b": 0.0},
    )
    assert entry.platt == {"a": 1.0, "b": 0.0}
    with pytest.raises(CalibrationInputError):
        CalibrationEntry(
            bucket_key="choice|2",
            method="platt",
            temperature=1.0,
            n_items=60,
            ece_before=0.1,
            ece_after=0.05,
            folds=0,
            platt=None,  # missing Platt parameters
        )
    with pytest.raises(CalibrationInputError):
        CalibrationEntry(
            bucket_key="choice|2",
            method="temperature",
            temperature=0.0,  # must be > 0
            n_items=10,
            ece_before=0.1,
            ece_after=0.05,
            folds=5,
        )
    with pytest.raises(CalibrationInputError):
        CalibrationEntry(
            bucket_key="choice|2",
            method="inherited_temperature",
            temperature=1.2,
            n_items=10,
            ece_before=0.1,
            ece_after=0.05,
            folds=0,
            inherited=False,  # inherited method requires the flag
        )
