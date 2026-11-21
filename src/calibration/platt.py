"""Platt scaling fitted with a small Newton-solver logistic regression.

Used for buckets with 50–200 dev items (FR-CALIB-01). The model is a binary
logistic regression of *correctness* on ``x = logit(confidence)`` where
``confidence`` is the max softmax probability of the raw logits. The
parameterisation ``p = sigmoid(a·x + b)`` is chosen so that ``(a, b) = (1, 0)``
recovers the raw confidence exactly — a stable, deterministic starting point.

Implemented with numpy only (no sklearn); the Newton system is at most 2×2
and solved with a tiny ridge for positive-definiteness. Degenerate all-one-class
targets saturate ``b`` at ±30 (sigmoid(30) ≈ 1 − 9e-14) rather than diverging.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .errors import CalibrationInputError

#: Clamp for converting probabilities to logits.
_LOGIT_EPS = 1e-6

#: Newton solver settings.
_MAX_ITER = 100
_TOL = 1e-12
_RIDGE = 1e-8

#: Saturation value used when every target in the fit set is the same class.
_SATURATED_INTERCEPT = 30.0


def logit(p: float) -> float:
    """Log-odds of ``p``, clamped away from 0 and 1."""
    clamped = min(max(p, _LOGIT_EPS), 1.0 - _LOGIT_EPS)
    return math.log(clamped / (1.0 - clamped))


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _validate_xy(
    features: Sequence[float], targets: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    if features is None or targets is None:
        raise CalibrationInputError("features and targets must be sequences")
    if len(features) != len(targets):
        raise CalibrationInputError(
            f"features and targets must have equal length, got {len(features)} and {len(targets)}"
        )
    if len(features) == 0:
        raise CalibrationInputError("expected at least one Platt sample")
    x = np.asarray([float(v) for v in features], dtype=np.float64)
    y = np.asarray([float(v) for v in targets], dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise CalibrationInputError("features must all be finite")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise CalibrationInputError("targets must all be 0 or 1")
    return x, y


def fit_platt(
    features: Sequence[float],
    targets: Sequence[int],
) -> tuple[float, float]:
    """Fit ``p = sigmoid(a·x + b)`` by Newton's method.

    Returns ``(a, b)``. Deterministic: fixed initialisation ``(1, 0)`` and a
    fixed iteration/tolerance schedule.
    """
    x, y = _validate_xy(features, targets)
    n_pos = float(np.sum(y == 1.0))
    if n_pos == 0.0:
        return 0.0, -_SATURATED_INTERCEPT
    if n_pos == float(len(y)):
        return 0.0, _SATURATED_INTERCEPT

    design = np.column_stack([x, np.ones_like(x)])
    w = np.array([1.0, 0.0], dtype=np.float64)
    ridge = np.eye(2, dtype=np.float64) * _RIDGE
    for _ in range(_MAX_ITER):
        z = design @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = design.T @ (p - y) + _RIDGE * w
        weights = p * (1.0 - p)
        hessian = (design.T * weights) @ design + ridge
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            break
        w = w - step
        if float(np.max(np.abs(step))) < _TOL:
            break
    a, b = float(w[0]), float(w[1])
    if not (math.isfinite(a) and math.isfinite(b)):
        raise CalibrationInputError("Platt fit diverged to non-finite parameters")
    return a, b


def apply_platt(confidence: float, a: float, b: float) -> float:
    """Apply a fitted Platt transform to a raw max-softmax confidence."""
    if not (0.0 <= confidence <= 1.0):
        raise CalibrationInputError(f"confidence must be in [0, 1], got {confidence!r}")
    return sigmoid(a * logit(confidence) + b)


def apply_platt_to_distribution(
    probs: Sequence[float], a: float, b: float
) -> tuple[list[float], float]:
    """Calibrate the *correctness* of a decision without changing its choice.

    Platt scaling estimates ``P(top-1 option is correct)`` — a scalar about the
    raw argmax option, **not** a categorical re-distribution over the options.
    Embedding that scalar into the argmax probability slot (an earlier design)
    could flip the decision: with ``(a, b) = (0, -3)`` and raw logits ``[4, 0]``
    the calibrated correctness is ``sigmoid(-3) ≈ 0.047``, which made the
    *second* option the argmax of the returned distribution. That is wrong: the
    choice must always follow the raw choice-logit argmax.

    This function therefore returns:

    * ``probabilities`` — the (normalised) raw option distribution, untouched by
      Platt scaling. Its argmax is exactly the raw argmax, so a caller that
      picks ``argmax(probabilities)`` picks the same option the model chose.
    * ``calibrated_confidence`` — the Platt-calibrated correctness probability
      ``sigmoid(a·logit(conf) + b)`` where ``conf = max(probabilities)``. This is
      a distinct scalar with *correctness* semantics; in particular it is not
      ``max(probabilities)`` when Platt scaling is active. Consumers (e.g. the
      abstention policy) must use this scalar, never the categorical maximum.

    The two returned values obey ``0 <= calibrated_confidence <= 1`` and
    ``sum(probabilities) == 1``; no inequality between them is asserted because
    none holds in general (correctness can be lower than ``max(probabilities)``).
    """
    p = [float(v) for v in probs]
    if not p:
        raise CalibrationInputError("probs must be non-empty")
    if any(not math.isfinite(v) or v < 0.0 or v > 1.0 for v in p):
        raise CalibrationInputError("probs must all be finite and in [0, 1]")
    total = sum(p)
    if not math.isfinite(total) or total <= 0.0:
        raise CalibrationInputError("probs must sum to a positive finite value")
    p = [v / total for v in p]
    argmax = max(range(len(p)), key=lambda i: p[i])
    conf = p[argmax]
    cal = apply_platt(conf, a, b)
    # Keep the option distribution intact — Platt only recalibrates correctness.
    return p, cal
