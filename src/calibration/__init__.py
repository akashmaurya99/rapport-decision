"""Calibration for RAPPORT (TASK-16, FR-CALIB-01/02/03).

Public surface:

* :mod:`src.calibration.temperature` — ``fit_temperature_map`` and the
  cross-fitting primitives.
* :mod:`src.calibration.map` — the :class:`CalibrationMap` JSON round-trip,
  ``runs/<run_id>/calibration.json`` persistence and ``apply``.
* :mod:`src.calibration.metrics` — the module's own pooled ECE, bucket-macro
  ECE, Brier score and NLL (no dependency on the evaluation module).
* :mod:`src.calibration.abstain` — the utility-derived abstention policy and
  the risk–coverage curve.
* :mod:`src.calibration.risk` — per-category risk thresholds fitted on dev
  safe rows (Aegis); XSTest is never an input here.
* :mod:`src.calibration.platt` / :mod:`src.calibration.buckets` — Platt
  scaling and the bucket-key algebra.
"""

from .abstain import (
    DEFAULT_U_ABSTAIN,
    DEFAULT_U_CORRECT,
    DEFAULT_U_WRONG,
    AbstentionPolicy,
    abstention_threshold,
    risk_coverage_curve,
)
from .buckets import (
    MAX_OPTIONS,
    MIN_OPTIONS,
    OPTION_COUNT_BUCKETS,
    TASK_FAMILIES,
    declared_bucket_keys,
    make_bucket_key,
    make_bucket_key_for_count,
    option_count_bucket,
    parse_bucket_key,
)
from .errors import CalibrationError, CalibrationInputError, CalibrationMapError
from .map import (
    MAP_FILENAME,
    SCHEMA_VERSION,
    CalibratedOutput,
    CalibrationEntry,
    CalibrationMap,
    calibration_path_for_run,
    load_calibration_map,
    save_calibration_map,
)
from .metrics import (
    ECE_BINS,
    NLL_EPS,
    CalibrationMetrics,
    brier_score,
    compute_calibration_metrics,
    confidence_correctness,
    ece,
    ece_bucket_macro,
    multiclass_nll,
    nll,
    softmax_probs,
)
from .platt import apply_platt, apply_platt_to_distribution, fit_platt, logit, sigmoid
from .risk import (
    DEFAULT_MAX_FPR,
    RISK_CATEGORIES,
    RISK_CATEGORY_COUNT,
    fit_risk_thresholds,
)
from .temperature import (
    DEFAULT_FOLDS,
    MIN_BUCKET_ITEMS,
    PLATT_MAX_ITEMS,
    T_GRID_MAX,
    T_GRID_MIN,
    T_GRID_STEP,
    apply_temperature,
    assign_folds,
    cross_fitted_temperature,
    fit_temperature_map,
    grid_search_temperature,
    temperature_grid,
)

__all__ = [
    # abstain
    "AbstentionPolicy",
    "DEFAULT_U_ABSTAIN",
    "DEFAULT_U_CORRECT",
    "DEFAULT_U_WRONG",
    "abstention_threshold",
    "risk_coverage_curve",
    # buckets
    "MAX_OPTIONS",
    "MIN_OPTIONS",
    "OPTION_COUNT_BUCKETS",
    "TASK_FAMILIES",
    "declared_bucket_keys",
    "make_bucket_key",
    "make_bucket_key_for_count",
    "option_count_bucket",
    "parse_bucket_key",
    # errors
    "CalibrationError",
    "CalibrationInputError",
    "CalibrationMapError",
    # map
    "MAP_FILENAME",
    "SCHEMA_VERSION",
    "CalibratedOutput",
    "CalibrationEntry",
    "CalibrationMap",
    "calibration_path_for_run",
    "load_calibration_map",
    "save_calibration_map",
    # metrics
    "ECE_BINS",
    "NLL_EPS",
    "CalibrationMetrics",
    "brier_score",
    "compute_calibration_metrics",
    "confidence_correctness",
    "ece",
    "ece_bucket_macro",
    "multiclass_nll",
    "nll",
    "softmax_probs",
    # platt
    "apply_platt",
    "apply_platt_to_distribution",
    "fit_platt",
    "logit",
    "sigmoid",
    # risk
    "DEFAULT_MAX_FPR",
    "RISK_CATEGORIES",
    "RISK_CATEGORY_COUNT",
    "fit_risk_thresholds",
    # temperature
    "DEFAULT_FOLDS",
    "MIN_BUCKET_ITEMS",
    "PLATT_MAX_ITEMS",
    "T_GRID_MAX",
    "T_GRID_MIN",
    "T_GRID_STEP",
    "apply_temperature",
    "assign_folds",
    "cross_fitted_temperature",
    "fit_temperature_map",
    "grid_search_temperature",
    "temperature_grid",
]
