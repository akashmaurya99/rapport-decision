"""Tests for src/calibration/temperature.py — cross-fitted per-bucket fitting.

Synthetic dev sets are built with a seeded numpy RandomState (fixed seeds ->
bit-identical across runs). Behavioural assertions are structural plus
hand-derived; no assertion depends on machine float noise beyond 1e-12.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration import temperature  # noqa: E402
from src.calibration.buckets import declared_bucket_keys, make_bucket_key_for_count  # noqa: E402
from src.calibration.errors import CalibrationInputError  # noqa: E402


def _make_items(family: str, n_options: int, n: int, seed: int, p_correct: float = 0.75):
    """Deterministic synthetic bucket rows with group ids ``seed-g<k>`` (5/group)."""
    rng = np.random.RandomState(seed)
    rows, labels, groups = [], [], []
    per_group = 5
    for i in range(n):
        lab = int(rng.randint(n_options))
        z = rng.randn(n_options) * 0.5
        if rng.rand() < p_correct:
            z[lab] += 2.0
        rows.append(z.tolist())
        labels.append(lab)
        groups.append(f"seed{seed}-g{i // per_group}")
    key = make_bucket_key_for_count(family, n_options)
    return rows, labels, [key] * n, groups


def _dev_set():
    r_c2, l_c2, b_c2, g_c2 = _make_items("choice", 2, 210, seed=101, p_correct=0.80)
    r_s35, l_s35, b_s35, g_s35 = _make_items("scoring", 4, 60, seed=202, p_correct=0.62)
    r_p2, l_p2, b_p2, g_p2 = _make_items("preference", 2, 30, seed=303, p_correct=0.75)
    return (
        r_c2 + r_s35 + r_p2,
        l_c2 + l_s35 + l_p2,
        b_c2 + b_s35 + b_p2,
        g_c2 + g_s35 + g_p2,
    )


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def test_temperature_grid_is_451_points():
    grid = temperature.temperature_grid()
    assert len(grid) == 451
    assert grid[0] == pytest.approx(0.5)
    assert grid[-1] == pytest.approx(5.0)
    assert grid[1] - grid[0] == pytest.approx(0.01)


def test_grid_search_all_correct_confident_picks_grid_min():
    logits = [[4.0, 0.0]] * 8
    labels = [0] * 8
    # NLL decreases monotonically on the grid; the optimum sits at T = 0.5.
    assert temperature.grid_search_temperature(logits, labels) == pytest.approx(0.5, abs=1e-9)


def test_grid_search_all_wrong_confident_picks_grid_max():
    logits = [[4.0, 0.0]] * 8
    labels = [1] * 8
    # NLL decreases monotonically as T grows; the optimum sits at T = 5.0.
    assert temperature.grid_search_temperature(logits, labels) == pytest.approx(5.0, abs=1e-9)


def test_grid_search_is_deterministic():
    logits, labels, _, _ = _make_items("choice", 2, 40, seed=7)
    assert temperature.grid_search_temperature(logits, labels) == (
        temperature.grid_search_temperature(logits, labels)
    )


def test_grid_search_rejects_empty_grid_and_bad_temperatures():
    with pytest.raises(CalibrationInputError):
        temperature.grid_search_temperature([[1.0, 0.0]], [0], grid=[])
    with pytest.raises(CalibrationInputError):
        temperature.grid_search_temperature([[1.0, 0.0]], [0], grid=[0.0])


# ---------------------------------------------------------------------------
# Group-aware folds
# ---------------------------------------------------------------------------

def test_assign_folds_never_splits_a_group():
    groups = ["a", "a", "b", "b", "c", "d", "e", "f"]
    folds = temperature.assign_folds(groups, 2)
    assert len(folds) == 2
    for fold in folds:
        fold_groups = {groups[i] for i in fold}
        # each group appears in exactly one fold
        for other in folds:
            if other is fold:
                continue
            assert fold_groups.isdisjoint({groups[i] for i in other})
    assert sorted(i for fold in folds for i in fold) == list(range(len(groups)))


def test_assign_folds_is_deterministic_under_input_permutation():
    groups = [f"g{i}" for i in range(20) for _ in range(3)]
    permuted = list(reversed(groups))
    f1 = temperature.assign_folds(groups, 5)
    f2 = temperature.assign_folds(permuted, 5)

    def partition(fold_indices, grps):
        return sorted(tuple(sorted({grps[i] for i in fold})) for fold in fold_indices)

    assert partition(f1, groups) == partition(f2, permuted)


def test_assign_folds_reduces_to_unique_group_count():
    folds = temperature.assign_folds(["only"] * 6, 5)
    assert len(folds) == 1
    assert sorted(folds[0]) == list(range(6))


def test_assign_folds_validates_input():
    with pytest.raises(CalibrationInputError):
        temperature.assign_folds([], 2)
    with pytest.raises(CalibrationInputError):
        temperature.assign_folds(["a", ""], 2)
    with pytest.raises(CalibrationInputError):
        temperature.assign_folds(["a"], 0)


def test_cross_fitted_temperature_single_group_degenerate():
    logits, labels, _, _ = _make_items("choice", 2, 12, seed=11)
    groups = ["one-group"] * 12
    temp, fold_ts, eff = temperature.cross_fitted_temperature(logits, labels, groups, 5)
    assert eff == 1
    assert fold_ts == [temp]
    assert temp == pytest.approx(temperature.grid_search_temperature(logits, labels))


def test_cross_fitted_temperature_two_groups_two_folds():
    logits, labels, _, _ = _make_items("choice", 2, 20, seed=12)
    groups = ["g0"] * 10 + ["g1"] * 10
    _, fold_ts, eff = temperature.cross_fitted_temperature(logits, labels, groups, 5)
    assert eff == 2
    assert len(fold_ts) == 2
    assert all(t > 0 for t in fold_ts)


def test_geometric_mean_of_fold_temperatures():
    logits, labels, _, groups = _make_items("choice", 2, 60, seed=13)
    temp, fold_ts, eff = temperature.cross_fitted_temperature(logits, labels, groups, 5)
    assert eff == 5
    expected = math.exp(sum(math.log(t) for t in fold_ts) / len(fold_ts))
    assert temp == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# fit_temperature_map — bucket method selection and map structure
# ---------------------------------------------------------------------------

def test_fit_temperature_map_bucket_methods():
    logits, labels, buckets, groups = _dev_set()
    m = temperature.fit_temperature_map(logits, labels, buckets, 5, groups, run_id="t16")

    assert m.pooled.n_items == 300
    assert m.pooled.folds == 5
    assert m.pooled.temperature > 0.0
    assert len(m.buckets) == 3

    t_bucket = m.buckets["choice|2"]
    assert t_bucket.method == "temperature"
    assert t_bucket.n_items == 210
    assert t_bucket.folds == 5
    assert t_bucket.fold_temperatures is not None and len(t_bucket.fold_temperatures) == 5
    expected_t = math.exp(
        sum(math.log(t) for t in t_bucket.fold_temperatures) / 5
    )
    assert t_bucket.temperature == pytest.approx(expected_t, abs=1e-12)
    assert t_bucket.inherited is False

    platt_bucket = m.buckets["scoring|3-5"]
    assert platt_bucket.method == "platt"
    assert platt_bucket.n_items == 60
    assert platt_bucket.temperature == pytest.approx(1.0)
    assert platt_bucket.platt is not None
    assert math.isfinite(platt_bucket.platt["a"])
    assert math.isfinite(platt_bucket.platt["b"])

    inh_bucket = m.buckets["preference|2"]
    assert inh_bucket.method == "inherited_temperature"
    assert inh_bucket.inherited is True
    assert inh_bucket.n_items == 30
    assert inh_bucket.temperature == pytest.approx(m.pooled.temperature, abs=1e-12)

    # Every bucket stores its item count.
    for entry in [m.pooled, *m.buckets.values()]:
        assert isinstance(entry.n_items, int) and entry.n_items > 0
        assert math.isfinite(entry.ece_before) and math.isfinite(entry.ece_after)

    # Empty declared buckets (15 of 18) are skipped and recorded.
    assert len(m.skipped_buckets) == 15
    assert set(m.skipped_buckets) == set(declared_bucket_keys()) - set(m.buckets)


def test_fit_temperature_map_is_deterministic():
    logits, labels, buckets, groups = _dev_set()
    m1 = temperature.fit_temperature_map(logits, labels, buckets, 5, groups)
    m2 = temperature.fit_temperature_map(logits, labels, buckets, 5, groups)
    assert m1.to_dict() == m2.to_dict()


def test_fit_temperature_map_group_aware_default_still_works():
    logits, labels, buckets, _ = _dev_set()
    # groups omitted: each item its own group; fitting still succeeds.
    m = temperature.fit_temperature_map(logits, labels, buckets, 5)
    assert m.group_aware is False
    assert m.pooled.n_items == 300


def test_fit_temperature_map_records_pooled_fold_temperatures():
    logits, labels, buckets, groups = _dev_set()
    m = temperature.fit_temperature_map(logits, labels, buckets, 5, groups)
    assert m.pooled.fold_temperatures is not None
    assert len(m.pooled.fold_temperatures) == 5


def test_fit_temperature_map_rejects_malformed_inputs():
    logits, labels, buckets, groups = _dev_set()
    with pytest.raises(CalibrationInputError):
        temperature.fit_temperature_map(logits[:-1], labels, buckets, 5, groups)
    with pytest.raises(CalibrationInputError):
        temperature.fit_temperature_map(logits, labels, buckets[:-1], 5, groups)
    with pytest.raises(CalibrationInputError):
        temperature.fit_temperature_map(logits, labels, buckets, 5, groups[:-1])
    with pytest.raises(CalibrationInputError):
        # bucket/option-count mismatch: two-option rows labelled as "3-5"
        bad_buckets = list(buckets)
        idx = next(i for i, b in enumerate(bad_buckets) if b == "choice|2")
        bad_buckets[idx] = "choice|3-5"
        temperature.fit_temperature_map(logits, labels, bad_buckets, 5, groups)
    with pytest.raises(CalibrationInputError):
        temperature.fit_temperature_map(logits, labels, ["bad-key"] * len(logits), 5, groups)
    with pytest.raises(CalibrationInputError):
        temperature.fit_temperature_map(logits, labels, buckets, 1, groups)
    with pytest.raises(CalibrationInputError):
        bad_labels = list(labels)
        bad_labels[0] = 99
        temperature.fit_temperature_map(logits, bad_labels, buckets, 5, groups)


def test_fit_temperature_map_no_xstest_parameter():
    import inspect

    params = inspect.signature(temperature.fit_temperature_map).parameters
    assert not any("xstest" in name.lower() for name in params)
