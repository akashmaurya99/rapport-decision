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
    """Platt-scale a full option distribution.

    The argmax slot carries the Platt-calibrated correctness probability
    ``cal = sigmoid(a·logit(conf) + b)``; the remaining slots keep their raw
    relative proportions and share the residual mass ``1 − cal``. Sums to 1.
    Returns ``(probabilities, calibrated_confidence)``.
    """
    p = [float(v) for v in probs]
    if not p:
        raise CalibrationInputError("probs must be non-empty")
    if any(v < 0.0 or v > 1.0 for v in p):
        raise CalibrationInputError("probs must all be in [0, 1]")
    total = sum(p)
    if not math.isfinite(total) or total <= 0.0:
        raise CalibrationInputError("probs must sum to a positive finite value")
    p = [v / total for v in p]
    argmax = max(range(len(p)), key=lambda i: p[i])
    conf = p[argmax]
    cal = apply_platt(conf, a, b)
    if conf >= 1.0 - 1e-9:
        # No mass left in the other slots; spread the residual uniformly.
        out = [0.0] * len(p)
        out[argmax] = cal
        residual = 1.0 - cal
        others = [i for i in range(len(p)) if i != argmax]
        if others:
            share = residual / len(others)
            for i in others:
                out[i] = share
        return out, cal
    residual = 1.0 - cal
    other_mass = 1.0 - conf
    factor = residual / other_mass
    out = [v * factor for v in p]
    out[argmax] = cal
    return out, cal
