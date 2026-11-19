"""Calibration metrics owned by the calibration module (FR-CALIB-02).

This module computes its own pooled ECE, bucket-macro ECE, Brier score and
NLL from scratch and **does not import (or depend on) the evaluation module**.
Implementation uses only the standard library and numpy-free math so the
numbers are reproducible anywhere.

Conventions
-----------

* **Pairs.** The calibration metrics operate on one binary pair ``(p, y)`` per
  item: ``p`` is a probability (typically the max-softmax confidence) and
  ``y ∈ {0, 1}`` whether the argmax prediction was correct.
* **ECE binning.** Fixed edges ``[i/bins, (i+1)/bins)`` for
  ``i = 0..bins-1`` with the right edge of the last bin closed, so ``p = 1.0``
  lands in the last bucket and exact-edge probabilities land in the upper
  bucket. Empty bins are skipped.
* **Pooled ECE** = ``Σ_b (n_b / N) · |acc_b − conf_b|`` over non-empty bins —
  each item counts in proportion to its bin's size.
* **Bucket-macro ECE** = unweighted mean of ``|acc_b − conf_b|`` over non-empty
  bins — every non-empty probability bucket counts equally.
* **Brier** = ``mean((p − y)²)``.
* **NLL** = binary log-loss with ``p`` clamped to ``[eps, 1 − eps]`` so the
  score is always finite. Multiclass NLL (:func:`multiclass_nll`) backs the
  temperature grid search.
* ``None`` is never emitted silently here; malformed input raises
  :class:`CalibrationInputError`. Empty input raises rather than returning a
  fabricated ``0.0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .errors import CalibrationInputError

#: Default bin count for the ECE family (fixed edges ``[0, 1/15, …, 1]``).
ECE_BINS = 15

#: Default probability clamp for NLL.
NLL_EPS = 1e-15


def _as_float_list(values: Sequence[float], name: str) -> list[float]:
    if values is None:
        raise CalibrationInputError(f"{name} must be a sequence, got None")
    out: list[float] = []
    for i, v in enumerate(values):
        try:
            f = float(v)
        except (TypeError, ValueError) as exc:
            raise CalibrationInputError(f"{name}[{i}] is not a float: {v!r}") from exc
        if not math.isfinite(f):
            raise CalibrationInputError(f"{name}[{i}] is not finite: {v!r}")
        out.append(f)
    return out


def _as_binary_list(values: Sequence[float | int], name: str) -> list[int]:
    if values is None:
        raise CalibrationInputError(f"{name} must be a sequence, got None")
    out: list[int] = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise CalibrationInputError(f"{name}[{i}] must be 0 or 1, got {v!r}")
        f = float(v)
        if f != 0.0 and f != 1.0:
            raise CalibrationInputError(f"{name}[{i}] must be 0 or 1, got {v!r}")
        out.append(int(f))
    return out


def _validate_pairs(
    probabilities: Sequence[float], targets: Sequence[float | int]
) -> tuple[list[float], list[int]]:
    p = _as_float_list(probabilities, "probabilities")
    y = _as_binary_list(targets, "targets")
    if len(p) != len(y):
        raise CalibrationInputError(
            f"probabilities and targets must have equal length, got {len(p)} and {len(y)}"
        )
    for i, v in enumerate(p):
        if v < 0.0 or v > 1.0:
            raise CalibrationInputError(f"probabilities[{i}] = {v} outside [0, 1]")
    if not p:
        raise CalibrationInputError("expected at least one (p, y) pair")
    return p, y


def _bin_index(p: float, bins: int) -> int:
    idx = int(p * bins)
    if idx >= bins:  # p == 1.0 lands in the last (closed) bin
        idx = bins - 1
    return idx


def _bin_stats(
    p: Sequence[float], y: Sequence[int], bins: int
) -> list[tuple[int, float, float]]:
    counts = [0] * bins
    conf_sums = [0.0] * bins
    acc_sums = [0.0] * bins
    for prob, target in zip(p, y, strict=True):
        idx = _bin_index(prob, bins)
        counts[idx] += 1
        conf_sums[idx] += prob
        acc_sums[idx] += target
    out: list[tuple[int, float, float]] = []
    for idx in range(bins):
        n = counts[idx]
        if n == 0:
            continue
        out.append((n, conf_sums[idx] / n, acc_sums[idx] / n))
    return out


def ece(
    probabilities: Sequence[float],
    targets: Sequence[float | int],
    bins: int = ECE_BINS,
) -> float:
    """Pooled (weighted) expected calibration error."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise CalibrationInputError(f"bins must be an int >= 1, got {bins!r}")
    p, y = _validate_pairs(probabilities, targets)
    stats = _bin_stats(p, y, bins)
    total = float(len(p))
    return sum(n / total * abs(acc - conf) for n, conf, acc in stats)


def ece_bucket_macro(
    probabilities: Sequence[float],
    targets: Sequence[float | int],
    bins: int = ECE_BINS,
) -> float:
    """Bucket-macro expected calibration error: unweighted mean |acc − conf|."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise CalibrationInputError(f"bins must be an int >= 1, got {bins!r}")
    p, y = _validate_pairs(probabilities, targets)
    stats = _bin_stats(p, y, bins)
    return sum(abs(acc - conf) for _, conf, acc in stats) / len(stats)


def brier_score(probabilities: Sequence[float], targets: Sequence[float | int]) -> float:
    """Mean squared error between probabilities and binary targets."""
    p, y = _validate_pairs(probabilities, targets)
    return sum((prob - target) ** 2 for prob, target in zip(p, y, strict=True)) / len(p)


def nll(
    probabilities: Sequence[float],
    targets: Sequence[float | int],
    eps: float = NLL_EPS,
) -> float:
    """Binary negative log-likelihood with probabilities clamped to [eps, 1-eps]."""
    if not math.isfinite(eps) or eps <= 0.0 or eps >= 0.5:
        raise CalibrationInputError(f"eps must be in (0, 0.5), got {eps!r}")
    p, y = _validate_pairs(probabilities, targets)
    total = 0.0
    for prob, target in zip(p, y, strict=True):
        clamped = min(max(prob, eps), 1.0 - eps)
        total -= target * math.log(clamped) + (1 - target) * math.log(1.0 - clamped)
    return total / len(p)


def softmax_probs(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Stable softmax of ``logits / temperature``."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise CalibrationInputError(f"temperature must be finite and > 0, got {temperature!r}")
    if logits is None:
        raise CalibrationInputError("logits must be a sequence, got None")
    z = _as_float_list(logits, "logits")
    if not z:
        raise CalibrationInputError("logits must be non-empty")
    scaled = [v / temperature for v in z]
    peak = max(scaled)
    exps = [math.exp(v - peak) for v in scaled]
    total = sum(exps)
    return [v / total for v in exps]


def multiclass_nll(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature: float,
) -> float:
    """Mean multiclass negative log-likelihood at a fixed temperature.

    Used by the temperature grid search. ``labels`` index into each row of
    ``logits``. Rows must contain the same structure as the rest of the
    pipeline (>= 2 options).
    """
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise CalibrationInputError(f"temperature must be finite and > 0, got {temperature!r}")
    z_rows = validate_logit_rows(logits)
    y = validate_labels(labels, len(z_rows), [len(row) for row in z_rows])
    total = 0.0
    for row, label in zip(z_rows, y, strict=True):
        probs = softmax_probs(row, temperature)
        total -= math.log(max(probs[label], NLL_EPS))
    return total / len(y)


def validate_logit_rows(logits: Sequence[Sequence[float]]) -> list[list[float]]:
    if logits is None:
        raise CalibrationInputError("logits must be a sequence of rows, got None")
    rows: list[list[float]] = []
    for i, row in enumerate(logits):
        if row is None:
            raise CalibrationInputError(f"logits[{i}] must be a sequence, got None")
        z = _as_float_list(row, f"logits[{i}]")
        if len(z) < 2:
            raise CalibrationInputError(
                f"logits[{i}] must contain at least 2 options, got {len(z)}"
            )
        rows.append(z)
    if not rows:
        raise CalibrationInputError("expected at least one logits row")
    return rows


def validate_labels(
    labels: Sequence[int], n_rows: int, row_lengths: list[int]
) -> list[int]:
    if labels is None:
        raise CalibrationInputError("labels must be a sequence, got None")
    if len(labels) != n_rows:
        raise CalibrationInputError(
            f"labels must have one entry per logits row, got {len(labels)} labels "
            f"for {n_rows} rows"
        )
    out: list[int] = []
    for i, v in enumerate(labels):
        if isinstance(v, bool) or not isinstance(v, int):
            raise CalibrationInputError(f"labels[{i}] must be an int, got {v!r}")
        if v < 0 or v >= row_lengths[i]:
            raise CalibrationInputError(
                f"labels[{i}] = {v} out of range for {row_lengths[i]} options"
            )
        out.append(v)
    return out


def confidence_correctness(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature: float | None = None,
) -> tuple[list[float], list[int]]:
    """Return ``(confidences, correctness)`` pairs.

    ``confidence`` is the max softmax probability per row — at the given
    ``temperature`` when supplied, else on the raw logits. ``correctness[i]``
    is 1 when the argmax option equals ``labels[i]``.
    """
    z_rows = validate_logit_rows(logits)
    y = validate_labels(labels, len(z_rows), [len(row) for row in z_rows])
    temp = 1.0 if temperature is None else float(temperature)
    confidences: list[float] = []
    correctness: list[int] = []
    for row, label in zip(z_rows, y, strict=True):
        probs = softmax_probs(row, temp)
        argmax = max(range(len(probs)), key=lambda i: probs[i])
        confidences.append(probs[argmax])
        correctness.append(1 if argmax == label else 0)
    return confidences, correctness


@dataclass(frozen=True)
class CalibrationMetrics:
    """Aggregate FR-CALIB-02 metrics for one set of (p, y) pairs."""

    pooled_ece: float
    bucket_macro_ece: float
    brier: float
    nll: float
    n_items: int
    bins: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "pooled_ece": self.pooled_ece,
            "bucket_macro_ece": self.bucket_macro_ece,
            "brier": self.brier,
            "nll": self.nll,
            "n_items": self.n_items,
            "bins": self.bins,
        }


def compute_calibration_metrics(
    probabilities: Sequence[float],
    targets: Sequence[float | int],
    bins: int = ECE_BINS,
) -> CalibrationMetrics:
    """Compute the full FR-CALIB-02 metric set for one (p, y) pair stream."""
    p, y = _validate_pairs(probabilities, targets)
    return CalibrationMetrics(
        pooled_ece=ece(p, y, bins),
        bucket_macro_ece=ece_bucket_macro(p, y, bins),
        brier=brier_score(p, y),
        nll=nll(p, y),
        n_items=len(p),
        bins=bins,
    )
