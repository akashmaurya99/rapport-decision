"""The calibration map: persistence, JSON round-trip and application.

Storage convention (architecture contract): ``runs/<run_id>/calibration.json``.
:func:`calibration_path_for_run` renders that path; :func:`save_calibration_map`
and :func:`load_calibration_map` accept any explicit path.

The map is JSON-serializable and round-trips exactly:
``CalibrationMap.from_dict(m.to_dict()) == m`` for any map produced by this
package (all stored values are finite doubles, ints, strings and bools).
``xstest_used_for_fitting`` is hardcoded ``False`` — there is no argument
anywhere in this package that accepts XSTest rows, and any non-``false``
value found in a foreign file is coerced back to ``False`` on load (the
flag is an audit marker, not user data).

``apply`` produces calibrated probabilities, the calibrated confidence, the
abstain decision (from the utilities frozen into the map, defaulting to the
declared ``(1, −3, 0)`` triple → threshold 0.75) and the method/temperature
used. Unknown bucket keys raise under ``strict=True`` (default) and fall back
to the pooled entry under ``strict=False``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .abstain import AbstentionPolicy
from .buckets import parse_bucket_key
from .errors import CalibrationInputError, CalibrationMapError
from .metrics import softmax_probs
from .platt import apply_platt_to_distribution

SCHEMA_VERSION = 1
MAP_FILENAME = "calibration.json"

METHODS = ("temperature", "platt", "inherited_temperature")


@dataclass(frozen=True)
class CalibrationEntry:
    """Calibration state for one bucket (or the pooled entry).

    ``temperature`` is always > 0 (contract). For ``method="platt"`` it is the
    ``1.0`` placeholder mandated by the map contract — the Platt parameters in
    ``platt`` calibrate the bucket instead. For ``method="inherited_temperature"``
    it is the pooled temperature and ``inherited`` is ``True``.
    ``fold_temperatures`` records the per-fold grid-search optima for
    temperature buckets (the stored temperature is their geometric mean).
    """

    bucket_key: str
    method: str
    temperature: float
    n_items: int
    ece_before: float
    ece_after: float
    folds: int
    inherited: bool = False
    task_family: str | None = None
    option_count_bucket: str | None = None
    platt: dict[str, float] | None = None
    fold_temperatures: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bucket_key, str) or not self.bucket_key:
            raise CalibrationInputError("bucket_key must be a non-empty str")
        if self.method not in METHODS:
            raise CalibrationInputError(
                f"method must be one of {METHODS}, got {self.method!r}"
            )
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise CalibrationInputError(
                f"temperature must be finite and > 0, got {self.temperature!r}"
            )
        if isinstance(self.n_items, bool) or not isinstance(self.n_items, int) or self.n_items < 0:
            raise CalibrationInputError(f"n_items must be an int >= 0, got {self.n_items!r}")
        if not (math.isfinite(self.ece_before) and math.isfinite(self.ece_after)):
            raise CalibrationInputError("ece_before/ece_after must be finite")
        if isinstance(self.folds, bool) or not isinstance(self.folds, int) or self.folds < 0:
            raise CalibrationInputError(f"folds must be an int >= 0, got {self.folds!r}")
        if self.method == "inherited_temperature" and not self.inherited:
            raise CalibrationInputError("inherited_temperature buckets must set inherited=True")
        if self.method != "inherited_temperature" and self.inherited:
            raise CalibrationInputError("inherited=True requires method='inherited_temperature'")
        if self.method == "platt":
            if self.platt is None:
                raise CalibrationInputError("platt buckets must carry platt={'a', 'b'}")
            a, b = self.platt.get("a"), self.platt.get("b")
            if a is None or b is None:
                raise CalibrationInputError("platt must carry 'a' and 'b'")
            if not (math.isfinite(float(a)) and math.isfinite(float(b))):
                raise CalibrationInputError("platt 'a' and 'b' must be finite")
        if self.method == "temperature" and self.temperature <= 0.0:
            raise CalibrationInputError("temperature buckets require temperature > 0")
        if self.fold_temperatures is not None:
            for i, t in enumerate(self.fold_temperatures):
                if not math.isfinite(float(t)) or float(t) <= 0.0:
                    raise CalibrationInputError(
                        f"fold_temperatures[{i}] must be finite and > 0, got {t!r}"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_key": self.bucket_key,
            "method": self.method,
            "temperature": float(self.temperature),
            "n_items": int(self.n_items),
            "ece_before": float(self.ece_before),
            "ece_after": float(self.ece_after),
            "folds": int(self.folds),
            "inherited": bool(self.inherited),
            "task_family": self.task_family,
            "option_count_bucket": self.option_count_bucket,
            "platt": (
                None
                if self.platt is None
                else {k: float(v) for k, v in self.platt.items()}
            ),
            "fold_temperatures": (
                None
                if self.fold_temperatures is None
                else [float(t) for t in self.fold_temperatures]
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationEntry":
        if not isinstance(data, Mapping):
            raise CalibrationMapError(f"bucket entry must be an object, got {type(data).__name__}")
        try:
            fold_ts = data.get("fold_temperatures")
            platt = data.get("platt")
            return cls(
                bucket_key=str(data["bucket_key"]),
                method=str(data["method"]),
                temperature=float(data["temperature"]),
                n_items=int(data["n_items"]),
                ece_before=float(data["ece_before"]),
                ece_after=float(data["ece_after"]),
                folds=int(data["folds"]),
                inherited=bool(data.get("inherited", False)),
                task_family=(
                    None if data.get("task_family") is None else str(data["task_family"])
                ),
                option_count_bucket=(
                    None
                    if data.get("option_count_bucket") is None
                    else str(data["option_count_bucket"])
                ),
                platt=(
                    None
                    if platt is None
                    else {str(k): float(v) for k, v in dict(platt).items()}
                ),
                fold_temperatures=(
                    None if fold_ts is None else tuple(float(t) for t in fold_ts)
                ),
            )
        except KeyError as exc:
            raise CalibrationMapError(f"bucket entry missing required field {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise CalibrationMapError(f"bucket entry has an invalid field: {exc}") from exc


@dataclass(frozen=True)
class CalibratedOutput:
    """Result of applying the map to one item's logits."""

    probabilities: list[float]
    confidence: float
    abstain: bool
    method: str
    bucket_key: str
    temperature: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "probabilities": list(self.probabilities),
            "confidence": self.confidence,
            "abstain": self.abstain,
            "method": self.method,
            "bucket_key": self.bucket_key,
            "temperature": self.temperature,
        }


@dataclass(frozen=True)
class CalibrationMap:
    """Per-bucket calibration state frozen before test is read."""

    schema_version: int
    folds: int
    pooled: CalibrationEntry
    buckets: dict[str, CalibrationEntry]
    skipped_buckets: tuple[str, ...] = ()
    fitted_on: str = "dev"
    run_id: str | None = None
    group_aware: bool = True
    # Hardcoded: there is no fitting entry point that accepts XSTest rows.
    xstest_used_for_fitting: bool = False
    abstention: dict[str, Any] = field(
        default_factory=lambda: AbstentionPolicy().to_dict()
    )
    risk_thresholds: dict[str, dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if int(self.schema_version) != SCHEMA_VERSION:
            raise CalibrationMapError(
                f"unsupported schema_version {self.schema_version!r} (expected {SCHEMA_VERSION})"
            )
        if isinstance(self.folds, bool) or not isinstance(self.folds, int) or self.folds < 0:
            raise CalibrationMapError(f"folds must be an int >= 0, got {self.folds!r}")
        for key, entry in self.buckets.items():
            if key != entry.bucket_key:
                raise CalibrationMapError(
                    f"buckets[{key!r}] carries bucket_key {entry.bucket_key!r}"
                )
        # Audit integrity: the flag can never be true from this package.
        if self.xstest_used_for_fitting:
            object.__setattr__(self, "xstest_used_for_fitting", False)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "fitted_on": self.fitted_on,
            "run_id": self.run_id,
            "folds": int(self.folds),
            "group_aware": bool(self.group_aware),
            "xstest_used_for_fitting": False,
            "pooled": self.pooled.to_dict(),
            "buckets": {key: entry.to_dict() for key, entry in sorted(self.buckets.items())},
            "skipped_buckets": list(self.skipped_buckets),
            "abstention": dict(self.abstention),
            "risk_thresholds": (
                None
                if self.risk_thresholds is None
                else {k: dict(v) for k, v in self.risk_thresholds.items()}
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationMap":
        if not isinstance(data, Mapping):
            raise CalibrationMapError(
                f"calibration map must be a JSON object, got {type(data).__name__}"
            )
        try:
            pooled = CalibrationEntry.from_dict(data["pooled"])
            buckets_raw = data.get("buckets", {})
            if not isinstance(buckets_raw, Mapping):
                raise CalibrationMapError("'buckets' must be an object keyed by bucket key")
            buckets = {
                str(key): CalibrationEntry.from_dict(entry) for key, entry in buckets_raw.items()
            }
            abstention = data.get("abstention")
            if abstention is None:
                abstention = AbstentionPolicy().to_dict()
            elif not isinstance(abstention, Mapping):
                raise CalibrationMapError("'abstention' must be an object")
            risk_thresholds = data.get("risk_thresholds")
            if risk_thresholds is not None:
                if not isinstance(risk_thresholds, Mapping):
                    raise CalibrationMapError(
                        "'risk_thresholds' must be an object keyed by category"
                    )
                risk_thresholds = {str(k): dict(v) for k, v in risk_thresholds.items()}
            skipped = tuple(str(k) for k in data.get("skipped_buckets", ()))
            return cls(
                schema_version=int(data["schema_version"]),
                folds=int(data["folds"]),
                pooled=pooled,
                buckets=buckets,
                skipped_buckets=skipped,
                fitted_on=str(data.get("fitted_on", "dev")),
                run_id=(None if data.get("run_id") is None else str(data["run_id"])),
                group_aware=bool(data.get("group_aware", True)),
                xstest_used_for_fitting=False,
                abstention=dict(abstention),
                risk_thresholds=risk_thresholds,
            )
        except KeyError as exc:
            raise CalibrationMapError(f"calibration map missing required field {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise CalibrationMapError(f"calibration map has an invalid field: {exc}") from exc

    # ------------------------------------------------------------------
    # Derivation helpers
    # ------------------------------------------------------------------
    def with_risk_thresholds(
        self, risk_thresholds: Mapping[str, Mapping[str, Any]]
    ) -> "CalibrationMap":
        """Return a new map with the per-category risk thresholds frozen in."""
        if not isinstance(risk_thresholds, Mapping) or not risk_thresholds:
            raise CalibrationMapError("risk_thresholds must be a non-empty mapping")
        frozen = {str(k): dict(v) for k, v in risk_thresholds.items()}
        return replace(self, risk_thresholds=frozen)

    def abstention_policy(self) -> AbstentionPolicy:
        """Reconstruct the frozen abstention policy (defaults to declared triple)."""
        payload = self.abstention or {}
        try:
            return AbstentionPolicy.from_dict(payload)
        except CalibrationInputError:
            return AbstentionPolicy()

    def entry_for(self, bucket_key: str) -> CalibrationEntry:
        try:
            return self.buckets[bucket_key]
        except KeyError as exc:
            raise CalibrationMapError(
                f"no calibration entry for bucket {bucket_key!r} "
                f"(known: {sorted(self.buckets)}; skipped: {list(self.skipped_buckets)})"
            ) from exc

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    def apply(
        self,
        logits: Sequence[float],
        bucket_key: str | None = None,
        *,
        strict: bool = True,
    ) -> CalibratedOutput:
        """Calibrate one item's logits.

        ``bucket_key=None`` selects the pooled entry. An unknown bucket raises
        under ``strict=True`` (default) and falls back to pooled otherwise.
        """
        if bucket_key is not None:
            parse_bucket_key(bucket_key)  # validate format; raises on malformed keys
            entry = self.buckets.get(bucket_key)
            if entry is None:
                if strict:
                    raise CalibrationMapError(
                        f"unknown bucket {bucket_key!r}; known buckets: {sorted(self.buckets)}"
                    )
                entry = self.pooled
        else:
            entry = self.pooled

        if entry.method == "platt":
            assert entry.platt is not None
            probs = softmax_probs(logits, 1.0)
            calibrated, confidence = apply_platt_to_distribution(
                probs, float(entry.platt["a"]), float(entry.platt["b"])
            )
            method, temperature = "platt", entry.temperature
        else:
            calibrated = softmax_probs(logits, entry.temperature)
            confidence = max(calibrated)
            method, temperature = entry.method, entry.temperature

        policy = self.abstention_policy()
        return CalibratedOutput(
            probabilities=calibrated,
            confidence=confidence,
            abstain=policy.should_abstain(confidence),
            method=method,
            bucket_key=entry.bucket_key,
            temperature=temperature,
        )


def calibration_path_for_run(run_id: str, runs_dir: str | Path = "runs") -> Path:
    """Conventional map path: ``<runs_dir>/<run_id>/calibration.json``."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise CalibrationMapError("run_id must be a non-empty str")
    run_id = run_id.strip()
    if run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise CalibrationMapError(f"run_id {run_id!r} must not contain path separators")
    return Path(runs_dir) / run_id / MAP_FILENAME


def save_calibration_map(calibration_map: CalibrationMap, path: str | Path) -> Path:
    """Serialize the map to ``path`` (creating parent directories)."""
    if not isinstance(calibration_map, CalibrationMap):
        raise CalibrationMapError("save_calibration_map expects a CalibrationMap")
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(calibration_map.to_dict(), indent=2, sort_keys=False)
        target.write_text(payload + "\n", encoding="utf-8")
    except OSError as exc:
        raise CalibrationMapError(f"could not write calibration map to {target}: {exc}") from exc
    return target


def load_calibration_map(path: str | Path) -> CalibrationMap:
    """Load a map from ``path``; typed errors on every failure mode."""
    source = Path(path)
    if not source.exists():
        raise CalibrationMapError(f"calibration map not found at {source}", code="MAP_NOT_FOUND")
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise CalibrationMapError(f"could not read calibration map at {source}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CalibrationMapError(
            f"calibration map at {source} is not valid JSON: {exc}", code="MAP_INVALID_JSON"
        ) from exc
    return CalibrationMap.from_dict(parsed)
