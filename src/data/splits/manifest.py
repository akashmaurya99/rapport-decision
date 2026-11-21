"""SplitManifest — the audit record for one split run.

Layout mirrors the other provenance manifests in the repo: a frozen dataclass
with ``to_dict``/``from_dict`` and JSON helpers, strict parsing (missing or
undeclared keys are fatal), and typed errors throughout.

Required on every manifest (validated in ``__post_init__`` and on load):

- ``fingerprint`` — sha256 over the canonical sorted ``(example_id, split)``
  membership (see :mod:`src.data.splits.fingerprint`).
- ``counts`` — rows per family per split: ``{family: {train, dev, test: n}}``.
- ``group_overlap`` — number of groups spanning splits. **Any non-zero value
  is a fatal error**, on construction and on load.
- ``dedup_threshold`` — the near-dedup Jaccard threshold carried in from the
  clean pipeline (default 0.9, mirroring ``DEFAULT_NEAR_THRESHOLD``).
- ``ood_source`` — the withheld source (default ``UltraFeedback-binarized``);
  it contributes zero rows to train and dev.
- ``withheld_source`` — the withheld source recorded explicitly (the same
  identity as ``ood_source``; both fields are mandatory so the withholding is
  never implicit).

Also recorded: ``seed``, ``totals`` per split, ``ood_rows`` (rows of the OOD
source, all assigned to the evaluation side) and ``ratios``.

Optional audit fields ``excluded_thirteen_gram_rows`` and
``excluded_thirteen_gram_ids`` record rows the router quarantined because
they contributed a cross-split exact 13-gram. Emitted splits are guaranteed
free of such overlaps; a non-zero count is a *reported exclusion*, not a
silent label choice. They default to zero/empty and are optional on load so
older manifests stay readable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import (
    ManifestError,
    ManifestFieldError,
    ManifestFingerprintError,
    ManifestGroupOverlapError,
)
from .fingerprint import is_sha256_hex

#: Default withheld source: the designated source-held-out OOD set.
DEFAULT_OOD_SOURCE = "UltraFeedback-binarized"
#: Default near-dedup Jaccard threshold, carried from the clean pipeline.
DEFAULT_DEDUP_THRESHOLD = 0.9
#: The three split names, in train/dev/test order.
SPLIT_NAMES = ("train", "dev", "test")
#: Target row ratios within each family (stratified assignment).
RATIOS = {"train": 0.70, "dev": 0.15, "test": 0.15}

#: Newly introduced audit fields; accepted on load but not required, so
#: manifests written before the 13-gram exclusion policy stay readable.
_OPTIONAL_KEYS = frozenset(
    {"excluded_thirteen_gram_rows", "excluded_thirteen_gram_ids"}
)

_REQUIRED_KEYS = (
    "fingerprint",
    "counts",
    "group_overlap",
    "dedup_threshold",
    "ood_source",
    "withheld_source",
    "seed",
    "totals",
    "ood_rows",
    "ratios",
)


def ratios_dict() -> dict[str, float]:
    """A fresh copy of the target split ratios."""
    return dict(RATIOS)


def _is_non_bool_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_counts(value: Any, *, field_name: str) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping):
        raise ManifestFieldError(
            f"{field_name} must be a mapping of family to split counts, "
            f"got {type(value).__name__}",
            field=field_name,
        )
    counts: dict[str, dict[str, int]] = {}
    for family in sorted(value):
        row = value[family]
        if not isinstance(family, str) or not family.strip():
            raise ManifestFieldError(
                f"{field_name} families must be non-empty strings, got {family!r}",
                field=field_name,
            )
        if not isinstance(row, Mapping):
            raise ManifestFieldError(
                f"{field_name}[{family!r}] must be a mapping of split to row count",
                field=field_name,
            )
        if set(row) != set(SPLIT_NAMES):
            raise ManifestFieldError(
                f"{field_name}[{family!r}] must have exactly the keys "
                f"{list(SPLIT_NAMES)}, got {sorted(row)}",
                field=field_name,
            )
        family_counts: dict[str, int] = {}
        for split_name in SPLIT_NAMES:
            count = row[split_name]
            if not _is_non_bool_int(count) or count < 0:
                raise ManifestFieldError(
                    f"{field_name}[{family!r}][{split_name!r}] must be a "
                    f"non-negative integer, got {count!r}",
                    field=field_name,
                )
            family_counts[split_name] = count
        counts[family] = family_counts
    return counts


def _validate_totals(value: Any, *, field_name: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(SPLIT_NAMES):
        raise ManifestFieldError(
            f"{field_name} must be a mapping with exactly the keys "
            f"{list(SPLIT_NAMES)}",
            field=field_name,
        )
    totals: dict[str, int] = {}
    for split_name in SPLIT_NAMES:
        count = value[split_name]
        if not _is_non_bool_int(count) or count < 0:
            raise ManifestFieldError(
                f"{field_name}[{split_name!r}] must be a non-negative integer, "
                f"got {count!r}",
                field=field_name,
            )
        totals[split_name] = count
    return totals


def _validate_ratios(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(SPLIT_NAMES):
        raise ManifestFieldError(
            f"ratios must be a mapping with exactly the keys {list(SPLIT_NAMES)}",
            field="ratios",
        )
    ratios: dict[str, float] = {}
    for split_name in SPLIT_NAMES:
        ratio = value[split_name]
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ManifestFieldError(
                f"ratios[{split_name!r}] must be a number, got {ratio!r}",
                field="ratios",
            )
        if not 0.0 <= float(ratio) <= 1.0:
            raise ManifestFieldError(
                f"ratios[{split_name!r}] must lie in [0, 1], got {ratio!r}",
                field="ratios",
            )
        ratios[split_name] = float(ratio)
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ManifestFieldError(
            f"ratios must sum to 1, got {sum(ratios.values())}",
            field="ratios",
        )
    return ratios


def _validate_source(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestFieldError(
            f"{field_name} must be a non-empty string, got {value!r}",
            field=field_name,
        )
    return value.strip()


@dataclass(frozen=True)
class SplitManifest:
    """Audit record for one split run; serializable to/from JSON."""

    fingerprint: str
    counts: dict[str, dict[str, int]]
    group_overlap: int
    dedup_threshold: float
    ood_source: str
    withheld_source: str
    seed: int = 0
    totals: dict[str, int] = field(default_factory=lambda: {name: 0 for name in SPLIT_NAMES})
    ood_rows: int = 0
    ratios: dict[str, float] = field(default_factory=ratios_dict)
    excluded_thirteen_gram_rows: int = 0
    excluded_thirteen_gram_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not is_sha256_hex(self.fingerprint):
            raise ManifestFingerprintError(
                f"fingerprint must be a lowercase sha256 hex digest, "
                f"got {self.fingerprint!r}"
            )
        if _is_non_bool_int(self.group_overlap) and self.group_overlap != 0:
            # Fatal on construction too — a leaky manifest may never exist.
            raise ManifestGroupOverlapError(
                f"group_overlap is {self.group_overlap}; any positive overlap "
                f"is a fatal error (the build must fail)",
                overlap=int(self.group_overlap),
            )
        if not _is_non_bool_int(self.group_overlap):
            raise ManifestFieldError(
                f"group_overlap must be an integer, got {self.group_overlap!r}",
                field="group_overlap",
            )
        threshold = self.dedup_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ManifestFieldError(
                f"dedup_threshold must be a number, got {threshold!r}",
                field="dedup_threshold",
            )
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ManifestFieldError(
                f"dedup_threshold must lie in [0, 1], got {threshold!r}",
                field="dedup_threshold",
            )
        object.__setattr__(self, "dedup_threshold", threshold)
        if not _is_non_bool_int(self.seed) or self.seed < 0:
            raise ManifestFieldError(
                f"seed must be a non-negative integer, got {self.seed!r}",
                field="seed",
            )
        if not _is_non_bool_int(self.ood_rows) or self.ood_rows < 0:
            raise ManifestFieldError(
                f"ood_rows must be a non-negative integer, got {self.ood_rows!r}",
                field="ood_rows",
            )
        excluded = self.excluded_thirteen_gram_rows
        if not _is_non_bool_int(excluded) or excluded < 0:
            raise ManifestFieldError(
                f"excluded_thirteen_gram_rows must be a non-negative integer, "
                f"got {excluded!r}",
                field="excluded_thirteen_gram_rows",
            )
        ids = self.excluded_thirteen_gram_ids
        if isinstance(ids, (str, bytes)) or not isinstance(ids, (list, tuple)):
            raise ManifestFieldError(
                "excluded_thirteen_gram_ids must be a list/tuple of strings",
                field="excluded_thirteen_gram_ids",
            )
        if any(not isinstance(item, str) for item in ids):
            raise ManifestFieldError(
                "excluded_thirteen_gram_ids must contain only strings",
                field="excluded_thirteen_gram_ids",
            )
        if len(set(ids)) != len(ids):
            raise ManifestFieldError(
                "excluded_thirteen_gram_ids contains duplicates",
                field="excluded_thirteen_gram_ids",
            )
        if excluded != len(ids):
            raise ManifestFieldError(
                f"excluded_thirteen_gram_rows is {excluded} but "
                f"excluded_thirteen_gram_ids has {len(ids)} entries",
                field="excluded_thirteen_gram_rows",
            )
        object.__setattr__(self, "excluded_thirteen_gram_ids", tuple(ids))
        object.__setattr__(
            self, "ood_source", _validate_source(self.ood_source, field_name="ood_source")
        )
        object.__setattr__(
            self, "withheld_source",
            _validate_source(self.withheld_source, field_name="withheld_source"),
        )
        object.__setattr__(self, "counts", _validate_counts(self.counts, field_name="counts"))
        object.__setattr__(self, "totals", _validate_totals(self.totals, field_name="totals"))
        object.__setattr__(self, "ratios", _validate_ratios(self.ratios))

    # ------------------------------------------------------------------ dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "counts": {
                family: dict(split_counts)
                for family, split_counts in sorted(self.counts.items())
            },
            "group_overlap": self.group_overlap,
            "dedup_threshold": self.dedup_threshold,
            "ood_source": self.ood_source,
            "withheld_source": self.withheld_source,
            "seed": self.seed,
            "totals": dict(self.totals),
            "ood_rows": self.ood_rows,
            "ratios": dict(self.ratios),
            "excluded_thirteen_gram_rows": self.excluded_thirteen_gram_rows,
            "excluded_thirteen_gram_ids": list(self.excluded_thirteen_gram_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SplitManifest":
        if not isinstance(data, Mapping):
            raise ManifestFieldError(
                f"expected a mapping of manifest field to value, got {type(data).__name__}"
            )
        unknown = sorted(set(data) - set(_REQUIRED_KEYS) - _OPTIONAL_KEYS)
        if unknown:
            raise ManifestFieldError(
                f"undeclared field(s): {', '.join(unknown)}",
                field=unknown[0],
            )
        missing = [key for key in _REQUIRED_KEYS if key not in data]
        if missing:
            raise ManifestFieldError(
                f"required field {missing[0]!r} is absent",
                field=missing[0],
            )
        group_overlap = data["group_overlap"]
        if _is_non_bool_int(group_overlap) and group_overlap != 0:
            # Acceptance E4: any positive overlap is fatal on load.
            raise ManifestGroupOverlapError(
                f"manifest records group_overlap={group_overlap}; any positive "
                f"overlap is a fatal error (the build must fail)",
                overlap=int(group_overlap),
            )
        return cls(
            fingerprint=data["fingerprint"],
            counts=data["counts"],
            group_overlap=group_overlap,
            dedup_threshold=data["dedup_threshold"],
            ood_source=data["ood_source"],
            withheld_source=data["withheld_source"],
            seed=data["seed"],
            totals=data["totals"],
            ood_rows=data["ood_rows"],
            ratios=data["ratios"],
            excluded_thirteen_gram_rows=data.get("excluded_thirteen_gram_rows", 0),
            excluded_thirteen_gram_ids=tuple(data.get("excluded_thirteen_gram_ids", ())),
        )

    # ------------------------------------------------------------------ json
    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":") if indent is None else None,
            indent=indent,
        )

    @classmethod
    def from_json(cls, payload: str) -> "SplitManifest":
        if not isinstance(payload, str):
            raise ManifestFieldError(
                f"expected a JSON string, got {type(payload).__name__}",
                field="manifest",
            )
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ManifestFieldError(
                f"expected a JSON object at the top level, got {type(data).__name__}",
                field="manifest",
            )
        return cls.from_dict(data)


def write_manifest(manifest: SplitManifest, path: str | Path) -> Path:
    """Write ``manifest`` as JSON to ``path`` (parents created as needed)."""
    if not isinstance(manifest, SplitManifest):
        raise ManifestFieldError(
            f"expected a SplitManifest, got {type(manifest).__name__}",
            field="manifest",
        )
    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(manifest.to_json() + "\n", encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"could not write split manifest to {destination}: {exc}") from exc
    return destination


def read_manifest(path: str | Path) -> SplitManifest:
    """Read and strictly validate the manifest at ``path``."""
    source = Path(path)
    try:
        payload = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"could not read split manifest from {source}: {exc}") from exc
    return SplitManifest.from_json(payload)
