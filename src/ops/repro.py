"""Fixed-seed reproducibility check (FR-OPS-03).

Re-executes a fixed number of steps of a *recorded*, deliberately tiny
deterministic workload — gradient descent on a quadratic objective over a
seed-initialised weight vector, implemented with numpy float64 — and asserts
the reproduced final loss matches the recorded value within a tolerance
(default 1e-4).

.. warning::
   This verifies the **determinism of the harness**: given the same seed and
   step count, the arithmetic is bit-stable on a platform. It does **not**
   verify the determinism of a trained network — real training adds CUDA
   kernel election, nondeterministic reductions and library autotuning, none
   of which this check exercises or claims to cover. FR-OPS-03's end-to-end
   form (kill-and-resume reproducing the next-step loss) is owned by the
   trainer; this module is the bounded, network-free gate that proves the
   replay machinery itself is sound.

The public :func:`verify_resume` also accepts two loss trajectories and
returns a plain ``bool`` ("did this resume reproduce that run?"), which is the
clean helper for the e2e wave; the numpy workload below remains the reference
implementation and the ``seed=...`` calling convention.

The recorded loss comes from either an explicit ``recorded_loss`` argument or
a JSONL log written by :class:`src.ops.logging.JsonlLogger`
(``recorded_path``), in which case the record with the highest ``step`` whose
``metrics``/``metric_set`` carries a finite ``loss`` is used.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, overload

import numpy as np

from .errors import OpsError, ReproductionError, ReproductionMismatchError
from .logging import iter_records

DEFAULT_REPLAY_STEPS = 50
DEFAULT_TOLERANCE = 1e-4
WORKLOAD_DIMS = 8
WORKLOAD_LEARNING_RATE = 0.1


@dataclass(frozen=True)
class ReproductionResult:
    """Outcome of one reproducibility check."""

    seed: int
    steps: int
    recorded_loss: float
    reproduced_loss: float
    abs_diff: float
    tolerance: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_reference_workload(seed: int, steps: int = DEFAULT_REPLAY_STEPS) -> float:
    """Run the deterministic reference workload and return the final loss.

    The workload: initialise ``w ~ N(0, 1)`` (8 dims) from
    ``numpy.random.default_rng(seed)`` (PCG64 — stable across platforms),
    then descend on ``f(w) = 0.5 * ||w - target||²`` with a fixed learning
    rate, where ``target`` is the fixed vector ``arange(1, 9) / 8``. All
    arithmetic is float64 with a fixed operation order, so the result is a
    pure function of ``(seed, steps)``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ReproductionError(f"seed must be an integer, got {seed!r}")
    if seed < 0:
        raise ReproductionError(f"seed must be >= 0, got {seed}")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ReproductionError(f"steps must be an integer, got {steps!r}")
    if steps < 1:
        raise ReproductionError(f"steps must be >= 1, got {steps}")

    rng = np.random.default_rng(seed)
    weights = rng.standard_normal(WORKLOAD_DIMS)
    target = np.arange(1, WORKLOAD_DIMS + 1, dtype=np.float64) / WORKLOAD_DIMS
    loss = math.inf
    for _ in range(steps):
        gradient = weights - target
        weights = weights - WORKLOAD_LEARNING_RATE * gradient
        residual = weights - target
        loss = 0.5 * float(np.dot(residual, residual))
    return loss


def _recorded_loss_from_jsonl(path: Path) -> float:
    best_step: int | None = None
    best_loss: float | None = None
    try:
        records = iter_records(path)
        for record in records:
            step = record.get("step")
            if not isinstance(step, int) or isinstance(step, bool):
                continue
            for key in ("metrics", "metric_set"):
                metrics = record.get(key)
                if not isinstance(metrics, dict):
                    continue
                loss = metrics.get("loss")
                if isinstance(loss, bool) or not isinstance(loss, (int, float)):
                    continue
                loss = float(loss)
                if not math.isfinite(loss):
                    continue
                if best_step is None or step >= best_step:
                    best_step = step
                    best_loss = loss
    except OpsError as exc:
        raise ReproductionError(f"cannot read recorded metrics from {path}: {exc}") from exc
    if best_loss is None:
        raise ReproductionError(
            f"no record with a finite 'loss' metric found in {path}", key="loss"
        )
    return best_loss


@overload
def verify_resume(
    trajectory_a: Any,
    trajectory_b: Any,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool: ...


@overload
def verify_resume(
    *,
    seed: int,
    replay_steps: int = DEFAULT_REPLAY_STEPS,
    tolerance: float = DEFAULT_TOLERANCE,
    recorded_loss: float | None = None,
    recorded_path: str | Path | None = None,
) -> ReproductionResult: ...


def verify_resume(*args: Any, **kwargs: Any) -> bool | ReproductionResult:
    """Compare two trajectories, or run the recorded-loss reference check.

    Two calling conventions are supported:

    * ``verify_resume(trajectory_a, trajectory_b, tolerance=1e-4) -> bool`` —
      the clean helper for the e2e wave: two sequences of per-step losses
      (e.g. an original run and a resumed run). Returns ``True`` only when
      both have the same length and every pair is within ``tolerance``.
    * ``verify_resume(seed=..., recorded_loss=... | recorded_path=...)`` —
      the original bounded numpy reference workload check, returning a
      :class:`ReproductionResult` (still used by the unit suite).

    Invalid inputs raise :class:`ReproductionError` in both modes.
    """
    if args:
        if len(args) < 2:
            raise ReproductionError("verify_resume requires two trajectories")
        if len(args) > 3:
            raise ReproductionError("verify_resume accepts at most three positional arguments")
        if kwargs and set(kwargs) - {"tolerance"}:
            raise ReproductionError(f"unexpected keyword arguments: {sorted(kwargs)}")
        tolerance = args[2] if len(args) > 2 else kwargs.get("tolerance", DEFAULT_TOLERANCE)
        return _verify_trajectories(args[0], args[1], tolerance)
    if "trajectory_a" in kwargs or "trajectory_b" in kwargs:
        unexpected = set(kwargs) - {"trajectory_a", "trajectory_b", "tolerance"}
        if unexpected:
            raise ReproductionError(f"unexpected keyword arguments: {sorted(unexpected)}")
        return _verify_trajectories(
            kwargs.get("trajectory_a"), kwargs.get("trajectory_b"),
            kwargs.get("tolerance", DEFAULT_TOLERANCE),
        )
    return _verify_reference(**kwargs)


def _as_trajectory(value: Any, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or value is None:
        raise ReproductionError(f"{name} must be a sequence of numbers, got {type(value).__name__}")
    try:
        items = list(value)
    except TypeError as exc:
        raise ReproductionError(f"{name} must be iterable: {exc}") from exc
    result: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float, np.integer, np.floating)):
            raise ReproductionError(f"{name} contains a non-numeric value: {item!r}")
        number = float(item)
        if not math.isfinite(number):
            raise ReproductionError(f"{name} contains a non-finite value: {item!r}")
        result.append(number)
    return tuple(result)


def _verify_trajectories(trajectory_a: Any, trajectory_b: Any, tolerance: Any) -> bool:
    """Element-wise comparison of two loss trajectories."""
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ReproductionError(f"tolerance must be a real number, got {tolerance!r}")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ReproductionError(f"tolerance must be finite and > 0, got {tolerance!r}")
    first = _as_trajectory(trajectory_a, "trajectory_a")
    second = _as_trajectory(trajectory_b, "trajectory_b")
    if not first or not second:
        raise ReproductionError("empty trajectories are not resume evidence")
    if len(first) != len(second):
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(first, second))


def _verify_reference(
    *,
    seed: int,
    replay_steps: int = DEFAULT_REPLAY_STEPS,
    tolerance: float = DEFAULT_TOLERANCE,
    recorded_loss: float | None = None,
    recorded_path: str | Path | None = None,
) -> ReproductionResult:
    """Re-execute ``replay_steps`` of the reference workload and compare losses.

    Exactly one of ``recorded_loss`` (explicit value) or ``recorded_path``
    (JSONL log to read the highest-step ``loss`` metric from) must be given.

    Returns a :class:`ReproductionResult` when the reproduced loss matches
    the recorded value within ``tolerance`` (default 1e-4); raises
    :class:`ReproductionMismatchError` otherwise (the exception carries the
    result as ``.result``) and :class:`ReproductionError` on bad input.
    """
    if (recorded_loss is None) == (recorded_path is None):
        raise ReproductionError("provide exactly one of recorded_loss or recorded_path")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ReproductionError(f"tolerance must be a real number, got {tolerance!r}")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ReproductionError(f"tolerance must be finite and > 0, got {tolerance!r}")

    if recorded_loss is None:
        path = Path(recorded_path)  # type: ignore[arg-type]
        if not path.is_file():
            raise ReproductionError(f"recorded log {path} does not exist")
        recorded = _recorded_loss_from_jsonl(path)
    else:
        if isinstance(recorded_loss, bool) or not isinstance(recorded_loss, (int, float)):
            raise ReproductionError(f"recorded_loss must be a real number, got {recorded_loss!r}")
        recorded = float(recorded_loss)
        if not math.isfinite(recorded):
            raise ReproductionError(f"recorded_loss must be finite, got {recorded_loss!r}")

    reproduced = run_reference_workload(seed, replay_steps)
    abs_diff = abs(reproduced - recorded)
    result = ReproductionResult(
        seed=seed,
        steps=replay_steps,
        recorded_loss=recorded,
        reproduced_loss=reproduced,
        abs_diff=abs_diff,
        tolerance=tolerance,
        passed=abs_diff <= tolerance,
    )
    if not result.passed:
        raise ReproductionMismatchError(
            f"reproduced loss {reproduced!r} does not match recorded loss {recorded!r} "
            f"within tolerance {tolerance:g} (|diff| = {abs_diff:g})",
            result=result,
        )
    return result
