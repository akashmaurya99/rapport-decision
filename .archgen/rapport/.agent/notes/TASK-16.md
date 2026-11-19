# TASK-16 — Per-bucket calibration and the abstention policy

Status: done

## What was built

- `src/calibration/buckets.py` — `BucketKey` algebra. Canonical key
  `"<task_family>|<option_count_bucket>"`; option-count bucket `{2, 3-5, 6-8}`;
  six task families → 18 declared buckets (`declared_bucket_keys()`);
  `option_count_bucket` enforces the contract's 2–8 options;
  `parse_bucket_key` is total (malformed keys raise `CalibrationInputError`).
- `src/calibration/metrics.py` — the module's **own** pooled ECE
  (`ece`, weighted), bucket-macro ECE (`ece_bucket_macro`, unweighted mean
  over non-empty bins), binary `brier_score`, `nll` (clamped at `1e-15`) and
  `multiclass_nll` (backs the grid search). `compute_calibration_metrics`
  aggregates the FR-CALIB-02 set. No import of `src.evaluation` — audited by
  `tests/unit/calibration/test_independence.py`. Bins use fixed edges
  `[i/bins, (i+1)/bins)`, last bin closed at 1.0 (same convention as the
  evaluation module, but a separate implementation, as FR-CALIB-02 requires).
- `src/calibration/temperature.py` — `fit_temperature_map(logits, labels,
  buckets, folds=5, groups=None, ...) -> CalibrationMap` (dev only).
  - **5-fold group-aware cross-fitting**, split by `group_id`, never by item.
    Fold assignment: groups sorted by `(−count, group_id)`, greedy
    least-loaded-fold placement, ties → lowest fold index → fully
    deterministic and input-order invariant. Fewer unique groups than folds →
    fold count reduced to `n_groups`; a single group → no honest cross-fit,
    temperature grid-searched on all items, `folds` recorded as `1`.
  - **NLL grid search** `T ∈ [0.5, 5.0]`, step `0.01` (451 points), ties
    resolve to the smaller `T` (ascending scan, strictly-better-only replace).
  - **Fold temperatures are averaged with a geometric mean**,
    `exp(mean(log T_k))`, chosen over arithmetic mean / median because `T` is
    a scale parameter (log-scale natural, always `T > 0`, unique,
    deterministic). Documented in the module docstring.
  - **Bucket sizing** (`n_items` stored on every entry): `< 50` → inherits the
    pooled temperature, `method="inherited_temperature"`, `inherited: true`;
    `50–200` → Platt scaling (`method="platt"`, `platt={"a","b"}`,
    `temperature` carries the contract's `> 0` placeholder at `1.0`);
    `> 200` → single cross-fitted temperature (`method="temperature"`,
    `fold_temperatures` persisted). Empty declared buckets are skipped and
    recorded in `skipped_buckets`.
  - `ece_before`/`ece_after` are **in-sample** on dev (`ece_after` applies the
    bucket's fitted method). They describe the frozen map, not an unbiased
    generalization estimate — noted in the docstring.
- `src/calibration/platt.py` — Platt scaling "correctness on score" as binary
  logistic regression on `x = logit(confidence)` with
  `p = sigmoid(a·x + b)` (`(a,b) = (1,0)` recovers raw confidence exactly;
  stable deterministic Newton start). numpy only, no sklearn. 2×2 Newton with
  `1e-8` ridge, `tol=1e-12`, `max_iter=100`; all-one-class targets saturate
  `b` at ±30 instead of diverging. `apply_platt_to_distribution` carries
  `cal = sigmoid(a·logit(conf) + b)` onto the argmax slot and renormalizes
  the rest proportionally (sum = 1).
- `src/calibration/abstain.py` — `AbstentionPolicy(u_correct=1.0,
  u_wrong=-3.0, u_abstain=0.0)`: utilities are constructor parameters, never
  literals in the formula. `threshold = (u_abstain − u_wrong) /
  (u_correct − u_wrong)`; default triple yields exactly **0.75**.
  `should_abstain(p)` is strictly-below (`p < threshold`).
  `abstention_threshold(...)` function twin for the declared API surface.
  `risk_coverage_curve(confidences, correctness)` sorts by confidence desc
  (stable ties) and emits a JSON-serializable dict: `sorted_p`, `coverage`
  (retained fraction `m/N`), `selective_accuracy`, `risk`, per-prefix
  `points` (`retained`, `coverage`, `threshold`, `selective_accuracy`,
  `risk`) and `aurc` (mean prefix risk, uniform `1/N` width).
  `u_correct == u_wrong` raises `CalibrationInputError` (no decision boundary).
- `src/calibration/risk.py` — `fit_risk_thresholds(scores, targets,
  *, max_fpr=0.10, categories=RISK_CATEGORIES, safe_mask=None)`. Per category
  the flagging threshold is grid-searched over `unique(scores) ∪
  {max+1}` to **maximize recall subject to flagging at most
  `floor(max_fpr · n_safe)` safe rows** (same convention as the evaluation
  module's `recall_at_fpr`); recall ties keep the smaller threshold (first
  feasible in the ascending grid — more of the budget used for equal
  recall). Default safe rows = all-zero-target dev rows (Aegis
  safe-labeled prompts carry an explicitly empty `risk_labels`); an explicit
  `safe_mask` selects them otherwise. Edge semantics: no positive rows →
  `threshold=None, recall=None` (`reason="no_positive_rows"`); no safe rows →
  vacuous FPR, threshold = `min(scores)`, recall 1.0 (`reason="no_safe_rows"`);
  otherwise always feasible at `max+1` (recall 0).
  **XSTest is never an input**: no dataset handle anywhere, module never reads
  files; `xstest_used_for_fitting` is hardcoded `False` on the map and on every
  per-category entry, and `from_dict` coerces a tampered `true` back to
  `false` (audit marker, not user data). `fit_temperature_map`/`fit_risk_thresholds`
  accept no `xstest` parameter (asserted by tests). XSTest remains the Y8
  measurement set.
- `src/calibration/map.py` — `CalibrationMap` / `CalibrationEntry` /
  `CalibratedOutput` frozen dataclasses. JSON round-trip is exact:
  `from_dict(to_dict(m)) == m` (floats are finite doubles; `fold_temperatures`
  stored as tuple internally, list in JSON). Storage convention
  `runs/<run_id>/calibration.json` via `calibration_path_for_run(run_id,
  runs_dir="runs")`; `save_calibration_map`/`load_calibration_map` accept
  explicit paths; save is byte-deterministic (stable ordering, trailing
  newline). `apply(logits, bucket_key=None, *, strict=True)` calibrates one
  item (temperature softmax / Platt / inherited pooled), returns calibrated
  probabilities, calibrated confidence, the abstain decision from the
  **frozen** policy, method and temperature used; unknown bucket →
  `CalibrationMapError` (strict) or pooled fallback (`strict=False`).
  `with_risk_thresholds(...)` returns a new frozen map with the per-category
  thresholds attached — that is how risk thresholds get frozen into
  `calibration.json` **before test is read** (or pass `risk_thresholds=`
  straight into `fit_temperature_map`).
- `src/calibration/errors.py` — `CalibrationError` (base, `code`) →
  `CalibrationInputError`, `CalibrationMapError` (codes `MAP_NOT_FOUND`,
  `MAP_INVALID_JSON`, …). No bare `ValueError` escapes the package.
- `src/calibration/__init__.py` — re-exports the public API.

## Interface decisions (for TASK-17 / TASK-22)

- `fit_temperature_map(logits, labels, buckets, folds=5, groups=None, *,
  ece_bins=15, run_id=None, abstention=None, risk_thresholds=None)` —
  per-item `buckets[i]` is the canonical key from `buckets.make_bucket_key`;
  option-count bucket is validated against `len(logits[i])`. Pass dev
  `groups=[group_id, ...]` — without them each item is its own group and the
  split is trivially group-safe but not the declared protocol.
- `CalibrationMap.buckets` is keyed by canonical key; `pooled` entry has
  `bucket_key="pooled"`. `skipped_buckets` lists declared buckets with no dev
  items (max 15 in the three-bucket fixture).
- Risk thresholds: call `fit_risk_thresholds(dev_scores, dev_targets)` on dev
  rows (scores = risk-head outputs, one column per category, column order =
  `RISK_CATEGORIES`), then `map = map.with_risk_thresholds(thresholds)` (or
  `risk_thresholds=` at fit time) and `save_calibration_map(map, path)`. The
  values are plain JSON dicts keyed by category.
- Platt buckets: `temperature` in the entry is the contract `> 0` placeholder
  (`1.0`); the Platt parameters in `entry.platt` carry the calibration.
  `Predictor` should consume `map.apply(logits, bucket_key)` rather than
  re-implement the transforms.
- `ece_after` is in-sample; cross-fit generalization is observable only via
  the fold temperatures themselves.
- Degenerate but valid: fitting with a single group per bucket yields
  `folds=1` (temperature grid-searched on all items of that bucket).

## Verification

- `.venv/bin/python -m pytest tests/unit/calibration -q` → **80 passed**
  (hand-computed ECE/Brier/NLL fixtures, fold-assignment determinism and
  disjointness, bucket method selection on 210/60/30-item buckets, JSON
  round-trip, risk-threshold binding fixture with FPR budget = 2/20, audit
  tests for the evaluation-module boundary and the XSTest flag).
- Full repo unit suite: **760 passed, 1 skipped** (was 680+1 before this task;
  no regressions, no other modules touched).
- `.venv/bin/python -m ruff check src/calibration tests/unit/calibration` → clean
- `.venv/bin/python -m mypy src/calibration` → Success, no issues (9 files)
- Test file basenames renamed to `test_calibration_metrics.py` /
  `test_calibration_risk.py` to avoid pytest's duplicate-basename collection
  collision with `tests/unit/evaluation/test_metrics.py` and
  `tests/unit/data/adapters/test_risk.py` (repo packages carry no
  `__init__.py` under `tests/unit/*`).
- Determinism: fitting twice yields identical `to_dict()`;
  `save_calibration_map` is byte-identical across runs; fold assignment is
  input-order invariant.
- No git operations performed; nothing outside `src/calibration/**` and
  `tests/unit/calibration/**` was edited.
