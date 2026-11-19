"""Per-bucket temperature fitting with 5-fold group-aware cross-fitting.

Implements FR-CALIB-01. Development-set logits are fitted per bucket
(``task_family × option_count_bucket``), with the fold split taken on
``group_id`` so no group ever appears in both the train and the held-out fold.

Cross-fitting protocol (deterministic)
--------------------------------------
1. Fold assignment: groups are sorted by ``(−item_count, group_id)`` and greedily
   assigned to the least-loaded fold (ties → lowest fold index). This balances
   fold sizes and never depends on input order.
2. Per fold, the temperature is grid-searched over
   ``T ∈ [0.5, 5.0]`` in steps of ``0.01`` (451 points) to minimize the mean
   multiclass NLL on the other folds. Ties resolve to the *smaller* T (the
   grid is scanned ascending and only strictly better NLL replaces the
   incumbent).
3. The bucket temperature is the **geometric mean** of the fold temperatures,
   ``exp(mean(log T_k))``. The geometric mean was chosen over the arithmetic
   mean and the median because temperatures are scale parameters: it is the
   natural average on the log scale, it is unique and deterministic, and it
   cannot leave the admissible ``T > 0`` range.
4. If a bucket has fewer unique groups than folds, the fold count is reduced
   to ``n_groups``; with a single group no honest cross-fit exists and the
   temperature is grid-searched on all of the bucket's items with ``folds``
   recorded as ``1``.

Bucket sizing rules
-------------------
* ``< 50`` items — the bucket **inherits the pooled temperature** (fitted on
  all dev items) and is flagged ``inherited: true`` with
  ``method = "inherited_temperature"``.
* ``50–200`` items — **Platt scaling** (:mod:`src.calibration.platt`) replaces
  the single temperature; ``method = "platt"`` and ``platt = {"a", "b"}`` are
  stored. ``temperature`` carries the contract-mandated ``1.0`` placeholder
  (the Platt parameters, not a temperature, calibrate these buckets).
* ``> 200`` items — a single cross-fitted temperature;
  ``method = "temperature"``.
* Empty declared buckets are skipped and recorded in ``skipped_buckets``.
* Every bucket entry stores its ``n_items`` and its ``folds``.

``ece_before``/``ece_after`` per bucket are computed **in-sample** on dev
(``ece_after`` applies the bucket's fitted method). They describe the map as
frozen, not an unbiased generalization estimate.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from . import metrics
from .abstain import AbstentionPolicy
from .buckets import (
    declared_bucket_keys,
    option_count_bucket,
    parse_bucket_key,
)
from .errors import CalibrationInputError
from .map import CalibrationEntry, CalibrationMap
from .platt import fit_platt, logit, sigmoid

#: NLL grid search domain: [0.5, 5.0] in 0.01 steps (451 points).
T_GRID_MIN = 0.5
T_GRID_MAX = 5.0
T_GRID_STEP = 0.01

#: Bucket sizing thresholds (FR-CALIB-01).
MIN_BUCKET_ITEMS = 50
PLATT_MAX_ITEMS = 200

DEFAULT_FOLDS = 5


def temperature_grid(
    start: float = T_GRID_MIN, stop: float = T_GRID_MAX, step: float = T_GRID_STEP
) -> list[float]:
    """The deterministic ``[0.5, 5.0]`` grid in ``0.01`` steps (451 points)."""
    if not (math.isfinite(start) and math.isfinite(stop) and math.isfinite(step)):
        raise CalibrationInputError("grid bounds must be finite")
    if start <= 0.0 or stop < start or step <= 0.0:
        raise CalibrationInputError("require 0 < start <= stop and step > 0")
    n_steps = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(n_steps + 1)]


def grid_search_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    grid: Sequence[float] | None = None,
) -> float:
    """Argmin of mean multiclass NLL over the temperature grid.

    Deterministic: the grid is scanned ascending and only a strictly lower NLL
    replaces the incumbent, so ties resolve to the smaller temperature.
    """
    z_rows = metrics.validate_logit_rows(logits)
    y = metrics.validate_labels(labels, len(z_rows), [len(row) for row in z_rows])
    if grid is None:
        grid = temperature_grid()
    else:
        grid = [float(t) for t in grid]
        if not grid:
            raise CalibrationInputError("grid must be non-empty")
        for t in grid:
            if not math.isfinite(t) or t <= 0.0:
                raise CalibrationInputError(f"grid temperatures must be finite and > 0, got {t!r}")
    best_t = min(grid)
    best_nll = math.inf
    for t in sorted(grid):
        value = metrics.multiclass_nll(z_rows, y, t)
        if value < best_nll:
            best_nll = value
            best_t = t
    return best_t


def assign_folds(groups: Sequence[str], n_folds: int) -> list[list[int]]:
    """Assign item indices to folds, split by group.

    Groups are ordered by ``(−count, group_id)`` and greedily placed in the
    least-loaded fold (ties → lowest fold index). Returns one list of item
    indices per fold; every item appears in exactly one fold. The effective
    fold count is ``min(n_folds, n_unique_groups)``.
    """
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 1:
        raise CalibrationInputError(f"n_folds must be an int >= 1, got {n_folds!r}")
    if groups is None:
        raise CalibrationInputError("groups must be a sequence, got None")
    if len(groups) == 0:
        raise CalibrationInputError("expected at least one item")
    for i, g in enumerate(groups):
        if not isinstance(g, str) or not g:
            raise CalibrationInputError(f"groups[{i}] must be a non-empty str, got {g!r}")

    counts = Counter(groups)
    order = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    n_eff = min(n_folds, len(order))
    fold_load = [0] * n_eff
    fold_members: list[set[str]] = [set() for _ in range(n_eff)]
    for group_id, count in order:
        idx = min(range(n_eff), key=lambda i: (fold_load[i], i))
        fold_members[idx].add(group_id)
        fold_load[idx] += count
    return [
        [i for i, g in enumerate(groups) if g in fold_members[k]] for k in range(n_eff)
    ]


def cross_fitted_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    groups: Sequence[str],
    folds: int = DEFAULT_FOLDS,
    grid: Sequence[float] | None = None,
) -> tuple[float, list[float], int]:
    """Geometric-mean temperature from group-aware K-fold NLL grid search.

    Returns ``(temperature, fold_temperatures, effective_folds)``. With a
    single unique group, cross-fitting is impossible: the temperature is
    grid-searched on all items and ``effective_folds`` is ``1``.
    """
    z_rows = metrics.validate_logit_rows(logits)
    y = metrics.validate_labels(labels, len(z_rows), [len(row) for row in z_rows])
    if len(groups) != len(z_rows):
        raise CalibrationInputError(
            f"groups must have one entry per item, got {len(groups)} for {len(z_rows)} items"
        )
    fold_indices = assign_folds(groups, folds)
    n_eff = len(fold_indices)
    if n_eff < 2:
        temp = grid_search_temperature(z_rows, y, grid)
        return temp, [temp], 1

    fold_temperatures: list[float] = []
    for k in range(n_eff):
        val_set = set(fold_indices[k])
        train_idx = [i for i in range(len(z_rows)) if i not in val_set]
        train_logits = [z_rows[i] for i in train_idx]
        train_labels = [y[i] for i in train_idx]
        fold_temperatures.append(grid_search_temperature(train_logits, train_labels, grid))
    geo_mean = math.exp(sum(math.log(t) for t in fold_temperatures) / len(fold_temperatures))
    return geo_mean, fold_temperatures, n_eff


def apply_temperature(logits: Sequence[float], temperature: float) -> list[float]:
    """Softmax probabilities at ``temperature`` (``T = 1`` reproduces raw)."""
    return metrics.softmax_probs(logits, temperature)


def _bucket_ece(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature: float | None,
    platt: tuple[float, float] | None,
    bins: int,
) -> float:
    if platt is not None:
        confidences, correctness = _platt_confidence_correctness(logits, labels, platt)
    else:
        assert temperature is not None
        confidences, correctness = metrics.confidence_correctness(logits, labels, temperature)
    return metrics.ece(confidences, correctness, bins)


def _platt_confidence_correctness(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    platt: tuple[float, float],
) -> tuple[list[float], list[int]]:
    a, b = platt
    z_rows = metrics.validate_logit_rows(logits)
    y = metrics.validate_labels(labels, len(z_rows), [len(row) for row in z_rows])
    confidences: list[float] = []
    correctness: list[int] = []
    for row, label in zip(z_rows, y, strict=True):
        probs = metrics.softmax_probs(row, 1.0)
        argmax = max(range(len(probs)), key=lambda i: probs[i])
        confidences.append(sigmoid(a * logit(probs[argmax]) + b))
        correctness.append(1 if argmax == label else 0)
    return confidences, correctness


def _validate_fit_inputs(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    buckets: Sequence[str],
    folds: int,
    groups: Sequence[str] | None,
) -> tuple[list[list[float]], list[int], list[str], list[str], bool]:
    z_rows = metrics.validate_logit_rows(logits)
    n = len(z_rows)
    y = metrics.validate_labels(labels, n, [len(row) for row in z_rows])

    if buckets is None:
        raise CalibrationInputError("buckets must be a sequence, got None")
    if len(buckets) != n:
        raise CalibrationInputError(
            f"buckets must have one entry per item, got {len(buckets)} for {n} items"
        )
    bucket_keys = list(buckets)
    for i, key in enumerate(bucket_keys):
        _, occ_bucket = parse_bucket_key(key)  # raises on malformed keys
        expected = option_count_bucket(len(z_rows[i]))
        if occ_bucket != expected:
            raise CalibrationInputError(
                f"buckets[{i}] = {key!r} declares option-count bucket {occ_bucket!r} but "
                f"logits[{i}] has {len(z_rows[i])} options (bucket {expected!r})"
            )

    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise CalibrationInputError(f"folds must be an int >= 2, got {folds!r}")

    group_aware = groups is not None
    if groups is None:
        group_list = [str(i) for i in range(n)]
    else:
        if len(groups) != n:
            raise CalibrationInputError(
                f"groups must have one entry per item, got {len(groups)} for {n} items"
            )
        group_list = list(groups)
        for i, g in enumerate(group_list):
            if not isinstance(g, str) or not g:
                raise CalibrationInputError(f"groups[{i}] must be a non-empty str, got {g!r}")
    return z_rows, y, bucket_keys, group_list, group_aware


def fit_temperature_map(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    buckets: Sequence[str],
    folds: int = DEFAULT_FOLDS,
    groups: Sequence[str] | None = None,
    *,
    ece_bins: int = metrics.ECE_BINS,
    run_id: str | None = None,
    abstention: AbstentionPolicy | None = None,
    risk_thresholds: dict[str, dict[str, object]] | None = None,
) -> CalibrationMap:
    """Fit the per-bucket calibration map on **dev** logits.

    ``buckets[i]`` is the canonical bucket key for item ``i``
    (see :func:`src.calibration.buckets.make_bucket_key`). ``groups[i]`` is the
    ``group_id`` for item ``i``; the cross-fitting split never crosses a
    group. When ``groups`` is omitted each item is its own group (no two
    items share a fold assignment constraint) — pass dev ``group_id``s for the
    declared group-aware protocol.

    Returns a :class:`~src.calibration.map.CalibrationMap`. Risk thresholds
    are not fitted here; fit them with :func:`src.calibration.risk.fit_risk_thresholds`
    on dev and pass them via ``risk_thresholds=`` (or attach them later with
    ``CalibrationMap.with_risk_thresholds``) so they are frozen into the map
    before test is read. XSTest rows are never an input to this function.
    """
    if isinstance(ece_bins, bool) or not isinstance(ece_bins, int) or ece_bins < 1:
        raise CalibrationInputError(f"ece_bins must be an int >= 1, got {ece_bins!r}")
    z_rows, y, bucket_keys, group_list, group_aware = _validate_fit_inputs(
        logits, labels, buckets, folds, groups
    )
    n = len(z_rows)

    # --- Pooled temperature (all dev items), group-aware cross-fit ---
    pooled_t, pooled_fold_ts, pooled_eff_folds = cross_fitted_temperature(
        z_rows, y, group_list, folds
    )
    pooled_ece_before = _bucket_ece(z_rows, y, 1.0, None, ece_bins)
    pooled_ece_after = _bucket_ece(z_rows, y, pooled_t, None, ece_bins)
    pooled = CalibrationEntry(
        bucket_key="pooled",
        method="temperature",
        temperature=pooled_t,
        n_items=n,
        ece_before=pooled_ece_before,
        ece_after=pooled_ece_after,
        folds=pooled_eff_folds,
        task_family=None,
        option_count_bucket=None,
        fold_temperatures=tuple(pooled_fold_ts),
    )

    # --- Per-bucket fits ---
    by_bucket: dict[str, list[int]] = {}
    for i, key in enumerate(bucket_keys):
        by_bucket.setdefault(key, []).append(i)

    entries: dict[str, CalibrationEntry] = {}
    for key in sorted(by_bucket):
        idx = by_bucket[key]
        b_logits = [z_rows[i] for i in idx]
        b_labels = [y[i] for i in idx]
        b_groups = [group_list[i] for i in idx]
        family, occ_bucket = parse_bucket_key(key)
        n_items = len(idx)
        ece_before = _bucket_ece(b_logits, b_labels, 1.0, None, ece_bins)

        if n_items < MIN_BUCKET_ITEMS:
            entries[key] = CalibrationEntry(
                bucket_key=key,
                method="inherited_temperature",
                temperature=pooled_t,
                n_items=n_items,
                ece_before=ece_before,
                ece_after=_bucket_ece(b_logits, b_labels, pooled_t, None, ece_bins),
                folds=0,
                inherited=True,
                task_family=family,
                option_count_bucket=occ_bucket,
            )
            continue

        if n_items <= PLATT_MAX_ITEMS:
            conf_raw, correctness = metrics.confidence_correctness(b_logits, b_labels, 1.0)
            features = [logit(c) for c in conf_raw]
            a, b = fit_platt(features, correctness)
            ece_after = _bucket_ece(b_logits, b_labels, None, (a, b), ece_bins)
            entries[key] = CalibrationEntry(
                bucket_key=key,
                method="platt",
                temperature=1.0,  # contract placeholder; Platt params calibrate
                n_items=n_items,
                ece_before=ece_before,
                ece_after=ece_after,
                folds=0,
                task_family=family,
                option_count_bucket=occ_bucket,
                platt={"a": a, "b": b},
            )
            continue

        t, fold_ts, eff_folds = cross_fitted_temperature(b_logits, b_labels, b_groups, folds)
        entries[key] = CalibrationEntry(
            bucket_key=key,
            method="temperature",
            temperature=t,
            n_items=n_items,
            ece_before=ece_before,
            ece_after=_bucket_ece(b_logits, b_labels, t, None, ece_bins),
            folds=eff_folds,
            task_family=family,
            option_count_bucket=occ_bucket,
            fold_temperatures=tuple(fold_ts),
        )

    skipped = tuple(key for key in declared_bucket_keys() if key not in entries)

    policy = abstention if abstention is not None else AbstentionPolicy()

    return CalibrationMap(
        schema_version=1,
        folds=folds,
        fitted_on="dev",
        run_id=run_id,
        group_aware=group_aware,
        xstest_used_for_fitting=False,
        pooled=pooled,
        buckets=entries,
        skipped_buckets=skipped,
        abstention=policy.to_dict(),
        risk_thresholds=risk_thresholds,
    )
