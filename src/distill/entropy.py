"""Teacher-distribution entropy and the 95th-percentile exclusion filter.

Stage C only distills examples whose teacher distribution carries signal. A
near-uniform (flat) teacher distribution is indistinguishable from noise, so
the training-strategy document (Stage C) excludes examples whose teacher
entropy exceeds the **95th percentile of the batch**.

``ENTROPY_EXCLUSION_PERCENTILE`` is the documented constant (95.0). The
percentile is computed with **linear interpolation between the closest ranks**
(numpy's default ``'linear'`` method): sort the values, take
``rank = p / 100 * (n - 1)``, and linearly interpolate between the floor and
ceil of the rank. Ties resolve deterministically. An example is marked
``exclude_from_distillation`` only when its entropy is **strictly above** the
percentile threshold — an entry exactly at the threshold is kept.

Entropy is measured in nats (natural log) of the softmax over the teacher's
raw option log-probs (temperature 1.0), independent of the distillation
temperature used by the KD loss.
"""

from __future__ import annotations

import math
from typing import Sequence

from .errors import DistillInputError

#: Documented constant: examples whose teacher entropy exceeds the 95th
#: percentile of the batch are excluded from distillation.
ENTROPY_EXCLUSION_PERCENTILE = 95.0


def entropy_from_logprobs(logprobs: Sequence[float]) -> float:
    """Natural-log (nats) entropy of ``softmax(logprobs)``.

    Uses the max-subtraction form so it is stable for large log-probs.
    Raises :class:`DistillInputError` for empty or non-finite input.
    """
    values = _finite_floats(logprobs, "logprobs")
    if not values:
        raise DistillInputError("logprobs must be non-empty", field="logprobs")
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    log_total = math.log(total)
    entropy = 0.0
    for v, e in zip(values, exps, strict=True):
        p = e / total
        # log p = (v - peak) - log_total
        entropy -= p * ((v - peak) - log_total)
    # Guard against a round-off negative zero on a one-hot distribution.
    return max(entropy, 0.0)


def percentile_linear(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0..100) with linear interpolation.

    Matches numpy's default ``'linear'`` method: sort the values, take
    ``rank = q / 100 * (n - 1)`` and interpolate linearly between the floor
    and ceil of the rank. Raises :class:`DistillInputError` on empty input,
    non-finite values or an out-of-range ``q``.
    """
    data = _finite_floats(values, "values")
    if not data:
        raise DistillInputError("values must be non-empty", field="values")
    if not (math.isfinite(q) and 0.0 <= q <= 100.0):
        raise DistillInputError(f"q must lie in [0, 100], got {q!r}", field="q")
    if len(data) == 1:
        return data[0]
    ordered = sorted(data)
    rank = (q / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def entropy_exclusion_mask(
    entropies: Sequence[float],
    percentile: float = ENTROPY_EXCLUSION_PERCENTILE,
) -> list[bool]:
    """``True`` at position ``i`` iff ``entropies[i]`` is strictly above the
    ``percentile``-th percentile of the batch (documented constant 95.0).

    Returns one flag per input, in input order. An empty batch yields an
    empty list.
    """
    data = _finite_floats(entropies, "entropies")
    if not data:
        return []
    threshold = percentile_linear(data, percentile)
    return [value > threshold for value in data]


def _finite_floats(values: Sequence[float], field: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise DistillInputError(f"{field} must be a sequence of numbers", field=field)
    out: list[float] = []
    for index, value in enumerate(values):
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise DistillInputError(
                f"{field}[{index}] must be a number, got {value!r}", field=field
            ) from exc
        if not math.isfinite(number):
            raise DistillInputError(
                f"{field}[{index}] must be finite, got {number!r}", field=field
            )
        out.append(number)
    return out
