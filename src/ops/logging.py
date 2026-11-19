"""Structured JSONL logging for runs (FR-OPS-01).

One JSON object per line. Every training step, evaluation and benchmark emits
exactly one record via :meth:`JsonlLogger.log` (or the ``log_step`` /
``log_eval`` / ``log_benchmark`` sugar).

Required keys per record
------------------------
* ``run_id`` — non-empty string; defaults to the logger's ``run_id``.
* ``step`` — integer >= 0 (booleans rejected).
* ``split`` — one of ``train``, ``dev``, ``test``, ``benchmark``, ``eval``.
* ``event`` — non-empty string (``step``, ``eval``, ``benchmark``, ...).
* ``wall_clock_seconds`` — finite float >= 0; when not supplied it is measured
  as the wall-clock time since the previous record emitted by this logger
  (since logger construction for the first record).
* ``metrics`` or ``metric_set`` — at least one, a dict with non-empty string
  keys and scalar values (``str`` / ``int`` / ``float`` / ``bool`` / ``None``;
  floats must be finite). Both may be present.

Plus ``timestamp`` (ISO-8601 UTC, added automatically) unless supplied in
``extra``.

Validation happens *before* the file is touched: a record that fails
validation raises :class:`JsonlRecordError` and nothing is written — a half
record can never land in the log.

Append semantics are atomic-ish: each record is written with a single
``open(path, "ab")`` + one ``write()`` of a complete line, then ``flush()``
and ``os.fsync()`` before the handle is closed. ``O_APPEND`` guarantees the
line lands at the end of the file; the fsync guarantees it is durable.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .errors import JsonlRecordError, JsonlWriteError, OpsError

VALID_SPLITS = ("train", "dev", "test", "benchmark", "eval")
REQUIRED_RECORD_KEYS = ("run_id", "step", "split", "event", "wall_clock_seconds")
METRIC_KEYS = ("metrics", "metric_set")
_AUTO_KEY = "timestamp"

# Values permitted inside a metrics dict: flat JSON scalars, finite floats.
_SCALAR_TYPES = (str, int, float, bool, type(None))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (_is_int(value) or isinstance(value, float)) and math.isfinite(float(value))


def _validate_run_id(value: Any, *, key: str = "run_id") -> str:
    if not isinstance(value, str) or not value.strip():
        raise JsonlRecordError(f"{key} must be a non-empty string, got {value!r}", key=key)
    return value


def _validate_step(value: Any) -> int:
    if not _is_int(value) or value < 0:
        raise JsonlRecordError(f"step must be an integer >= 0, got {value!r}", key="step")
    return value


def _validate_split(value: Any) -> str:
    if not isinstance(value, str) or value not in VALID_SPLITS:
        raise JsonlRecordError(
            f"split must be one of {', '.join(VALID_SPLITS)}, got {value!r}", key="split"
        )
    return value


def _validate_event(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JsonlRecordError(f"event must be a non-empty string, got {value!r}", key="event")
    return value


def _validate_wall_clock(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JsonlRecordError(
            f"wall_clock_seconds must be a real number, got {value!r}", key="wall_clock_seconds"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise JsonlRecordError(
            f"wall_clock_seconds must be finite and >= 0, got {value!r}", key="wall_clock_seconds"
        )
    return result


def _validate_metrics(value: Any, *, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise JsonlRecordError(f"{key} must be a dict, got {type(value).__name__}", key=key)
    if not value:
        raise JsonlRecordError(f"{key} must not be empty", key=key)
    for metric_key, metric_value in value.items():
        if not isinstance(metric_key, str) or not metric_key.strip():
            raise JsonlRecordError(
                f"{key} keys must be non-empty strings, got {metric_key!r}", key=key
            )
        if not isinstance(metric_value, _SCALAR_TYPES):
            raise JsonlRecordError(
                f"{key}[{metric_key!r}] must be a flat JSON scalar, "
                f"got {type(metric_value).__name__}",
                key=key,
            )
        if isinstance(metric_value, float) and not math.isfinite(metric_value):
            raise JsonlRecordError(
                f"{key}[{metric_key!r}] must be finite, got {metric_value!r}", key=key
            )
    return dict(value)


def build_record(
    *,
    run_id: Any,
    step: Any,
    split: Any,
    event: Any,
    wall_clock_seconds: Any,
    metrics: Any = None,
    metric_set: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate every field and assemble the record dict.

    Raises :class:`JsonlRecordError` on the first invalid field; the message
    and ``key`` attribute name the offending key.
    """
    if metrics is None and metric_set is None:
        raise JsonlRecordError(
            "a record requires at least one of 'metrics' or 'metric_set'", key="metrics"
        )
    record: dict[str, Any] = {}
    reserved = set(REQUIRED_RECORD_KEYS) | set(METRIC_KEYS) | {_AUTO_KEY}
    for key, value in (extra or {}).items():
        if not isinstance(key, str) or not key.strip():
            raise JsonlRecordError(
                f"extra keys must be non-empty strings, got {key!r}", key=_AUTO_KEY
            )
        if key in reserved:
            raise JsonlRecordError(
                f"extra key {key!r} collides with a reserved record key", key=key
            )
        record[key] = value
    record["run_id"] = _validate_run_id(run_id)
    record["step"] = _validate_step(step)
    record["split"] = _validate_split(split)
    record["event"] = _validate_event(event)
    record["wall_clock_seconds"] = _validate_wall_clock(wall_clock_seconds)
    if metrics is not None:
        record["metrics"] = _validate_metrics(metrics, key="metrics")
    if metric_set is not None:
        record["metric_set"] = _validate_metrics(metric_set, key="metric_set")
    record.setdefault(
        _AUTO_KEY, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    )
    return record


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield parsed records from a JSONL log, one dict per line.

    Raises :class:`JsonlRecordError` (with the 1-based line number in the
    message) if a line is not a JSON object, and :class:`OpsError` if the
    file cannot be read.
    """
    log_path = Path(path)
    try:
        handle = log_path.open("rb")
    except OSError as exc:
        raise OpsError(f"cannot read JSONL log {log_path}: {exc}") from exc
    with handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlRecordError(
                    f"{log_path}:{line_no} is not valid JSON: {exc}", key="<line>"
                ) from exc
            if not isinstance(parsed, dict):
                raise JsonlRecordError(
                    f"{log_path}:{line_no} is not a JSON object", key="<line>"
                )
            yield parsed


class JsonlLogger:
    """Append validated JSONL records to one log file.

    Parameters
    ----------
    path:
        Destination file; parent directories are created on first write.
    run_id:
        Default run id stamped on every record; overridable per call.
    """

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise JsonlRecordError(
                f"run_id must be a non-empty string, got {run_id!r}", key="run_id"
            )
        self.path = Path(path)
        self.run_id = run_id
        self.records_written = 0
        self._last_mark = time.perf_counter()

    def log(
        self,
        *,
        event: Any,
        step: Any,
        split: Any,
        metrics: Any = None,
        metric_set: Any = None,
        wall_clock_seconds: Any = None,
        run_id: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Validate and append one record; returns the record as written.

        When ``wall_clock_seconds`` is omitted it is measured as the elapsed
        time since the previous record emitted by this logger (or since the
        logger was constructed for the first record).
        """
        now = time.perf_counter()
        if wall_clock_seconds is None:
            wall_clock_seconds = now - self._last_mark
        record = build_record(
            run_id=self.run_id if run_id is None else run_id,
            step=step,
            split=split,
            event=event,
            wall_clock_seconds=wall_clock_seconds,
            metrics=metrics,
            metric_set=metric_set,
            extra=extra,
        )
        self._write(record)
        self._last_mark = time.perf_counter()
        self.records_written += 1
        return record

    def log_step(
        self,
        step: Any,
        metrics: Any,
        *,
        split: str = "train",
        wall_clock_seconds: Any = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Emit one ``event="step"`` training-step record."""
        return self.log(
            event="step",
            step=step,
            split=split,
            metrics=metrics,
            wall_clock_seconds=wall_clock_seconds,
            **extra,
        )

    def log_eval(
        self,
        step: Any,
        metrics: Any,
        *,
        split: str = "dev",
        wall_clock_seconds: Any = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Emit one ``event="eval"`` evaluation record."""
        return self.log(
            event="eval",
            step=step,
            split=split,
            metrics=metrics,
            wall_clock_seconds=wall_clock_seconds,
            **extra,
        )

    def log_benchmark(
        self,
        step: Any,
        metrics: Any,
        *,
        split: str = "benchmark",
        wall_clock_seconds: Any = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Emit one ``event="benchmark"`` benchmark record."""
        return self.log(
            event="benchmark",
            step=step,
            split=split,
            metrics=metrics,
            wall_clock_seconds=wall_clock_seconds,
            **extra,
        )

    def _write(self, record: dict[str, Any]) -> None:
        try:
            line = (
                json.dumps(record, allow_nan=False, separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError) as exc:
            raise JsonlRecordError(
                f"record is not JSON-serializable: {exc}", key="<record>"
            ) from exc
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise JsonlWriteError(f"cannot append to JSONL log {self.path}: {exc}") from exc
