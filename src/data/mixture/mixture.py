"""Declarative training mixtures (FR-DATA-07).

A mixture is declared, never sampled ad hoc: each family names the source(s)
it draws from and a row cap. The **same seed and the same fingerprint
reproduce the same example order on every platform** — no RNG anywhere. Each
slot's position is derived from ``sha256`` over
``[domain, fingerprint, seed, family, source, index]`` (canonical compact
JSON) and all slots are sorted by that key, so allocation never depends on
dict iteration order, platform, or Python version.

Weight semantics (the "weight and cap" of FR-DATA-07):

- ``rows`` is the **cap**: exactly that many rows are taken from the source.
- ``weight`` (per entry in YAML, or the ``weights`` argument of
  :func:`sample_mixture`) is the family's relative **sampling weight**. The
  effective family weight resolves as: ``weights[family]`` if provided, else
  the common per-entry weight within the family (all entries in one family
  must agree, otherwise the declaration is ambiguous and rejected), else the
  family's row count — i.e. row-proportional shares by default. Weights are
  recorded in the result together with the expected vs actual shares and the
  absolute error between them; they are an audit signal, never silently
  dropped and never a reason to change the deterministic order.

Fingerprint semantics (FR-DATA-07 "from the recorded fingerprint"):

- ``fingerprint`` must be a lowercase sha256 hex digest (the split-manifest
  fingerprint recorded at acquisition time). Anything else is a
  :class:`MixtureFingerprintError`.
- When the mixture declaration pins a non-empty fingerprint (the placeholder
  ``""`` means "resolved at acquisition time") and the caller passes a
  different one, :func:`sample_mixture` raises
  :class:`MixtureFingerprintMismatchError` — the run refuses to train on an
  order derived from the wrong data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.data.acquire.manifest import STATUS_ADMITTED
from src.data.core.schema import TASK_FAMILIES, DecisionExample
from src.data.splits.fingerprint import is_sha256_hex

from .errors import (
    MixtureAvailabilityError,
    MixtureFamilyError,
    MixtureFingerprintError,
    MixtureFingerprintMismatchError,
    MixtureInputError,
    MixtureQuarantinedSourceError,
    MixtureSourceError,
)

_MAX_SEED = 2**32 - 1

# Domain separator for slot keys; a mixture-slot digest can never collide
# with a split-membership or content-id digest.
_SLOT_DOMAIN = "rapport:mixture-slot:v1"

_ENTRY_KEYS = frozenset({"source", "rows", "weight"})

DEFAULT_MIXTURE_NAME = "adhoc"


@dataclass(frozen=True)
class MixtureSource:
    """One declarative allocation: ``source`` contributes ``rows`` rows."""

    source: str
    rows: int
    weight: float | None = None

    def __post_init__(self) -> None:
        source = _validate_source_name(self.source, field_name="source")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "rows", _validate_rows(self.rows, field_name="rows"))
        object.__setattr__(self, "weight", _validate_weight(self.weight, field_name="weight"))


@dataclass(frozen=True)
class MixtureSlot:
    """One row slot in the deterministic global order."""

    order: int
    family: str
    source: str
    index: int
    key: str
    example_id: str


@dataclass(frozen=True)
class MixtureOrder:
    """The deterministic assignment of mixture slots to a training order."""

    seed: int
    fingerprint: str
    mixture: str
    total_rows: int
    slots: tuple[MixtureSlot, ...]
    weights: dict[str, float]
    family_rows: dict[str, int]
    family_shares: dict[str, float]
    expected_shares: dict[str, float]
    share_errors: dict[str, float]
    counts: dict[str, dict[str, int]]

    def example_ids(self) -> tuple[str, ...]:
        """The ordered slot ids, exactly as the trainer will consume them."""
        return tuple(slot.example_id for slot in self.slots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "mixture": self.mixture,
            "total_rows": self.total_rows,
            "weights": dict(self.weights),
            "family_rows": dict(self.family_rows),
            "family_shares": dict(self.family_shares),
            "expected_shares": dict(self.expected_shares),
            "share_errors": dict(self.share_errors),
            "counts": {family: dict(sources) for family, sources in self.counts.items()},
        }


# --- validation helpers ------------------------------------------------------


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _validate_seed(seed: object) -> int:
    if _is_bool(seed) or not isinstance(seed, int):
        raise MixtureInputError(f"seed must be an integer, got {seed!r}", field="seed")
    if seed < 0 or seed > _MAX_SEED:
        raise MixtureInputError(f"seed must lie in [0, {_MAX_SEED}], got {seed}", field="seed")
    return seed


def _validate_source_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MixtureSourceError(
            f"{field_name} must be a non-empty string, got {value!r}", field=field_name
        )
    return value.strip()


def _validate_rows(value: object, *, field_name: str) -> int:
    if _is_bool(value) or not isinstance(value, int):
        raise MixtureSourceError(
            f"rows must be a non-negative integer, got {value!r}", field=field_name
        )
    if value < 0:
        raise MixtureSourceError(
            f"rows must be a non-negative integer, got {value}", field=field_name
        )
    return value


def _validate_weight(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if _is_bool(value) or not isinstance(value, (int, float)):
        raise MixtureSourceError(
            f"weight must be a positive finite number, got {value!r}", field=field_name
        )
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0.0:
        raise MixtureSourceError(
            f"weight must be a positive finite number, got {value!r}", field=field_name
        )
    return weight


def _validate_fingerprint(fingerprint: object) -> str:
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise MixtureFingerprintError(
            f"fingerprint must be a non-empty sha256 hex digest, got {fingerprint!r}",
            field="fingerprint",
        )
    candidate = fingerprint.strip()
    if not is_sha256_hex(candidate):
        raise MixtureFingerprintError(
            "fingerprint must be a lowercase sha256 hex digest, "
            f"got {fingerprint!r}; record the split-manifest fingerprint at acquisition time",
            field="fingerprint",
        )
    return candidate


def _check_pinned_fingerprint(pinned: object, actual: str) -> None:
    """Compare the declaration-pinned fingerprint against the caller's."""
    if pinned in (None, ""):
        return
    if not isinstance(pinned, str) or not is_sha256_hex(pinned.strip()):
        raise MixtureFingerprintError(
            f"the mixture declaration pins an invalid fingerprint, got {pinned!r}",
            field="fingerprint",
        )
    expected = pinned.strip()
    if expected != actual:
        raise MixtureFingerprintMismatchError(
            f"fingerprint mismatch: the mixture was recorded against {expected} "
            f"but {actual} was passed; refusing to derive an example order from "
            "the wrong data",
            expected=expected,
            actual=actual,
        )


def parse_source_entry(value: object, *, family: str, field_name: str) -> MixtureSource:
    """Parse one declarative ``{source, rows, weight?}`` mapping."""
    if isinstance(value, MixtureSource):
        return value
    if not isinstance(value, Mapping):
        raise MixtureSourceError(
            f"{field_name} must be a mapping with keys 'source', 'rows' and optional "
            f"'weight', or a MixtureSource; got {value!r}",
            field=field_name,
        )
    unknown = sorted(str(key) for key in value if key not in _ENTRY_KEYS)
    if unknown:
        raise MixtureSourceError(
            f"{field_name} has undeclared key(s): {', '.join(unknown)}; "
            f"expected a subset of {sorted(_ENTRY_KEYS)}",
            field=field_name,
        )
    if "source" not in value:
        raise MixtureSourceError(f"{field_name} is missing required key 'source'", field=field_name)
    if "rows" not in value:
        raise MixtureSourceError(f"{field_name} is missing required key 'rows'", field=field_name)
    rows = _validate_rows(value["rows"], field_name=f"{field_name}.rows")
    weight = _validate_weight(value.get("weight"), field_name=f"{field_name}.weight")
    source = _validate_source_name(value["source"], field_name=f"{field_name}.source")
    return MixtureSource(source=source, rows=rows, weight=weight)


def parse_family_entries(
    family: str, value: object, *, field_name: str
) -> tuple[MixtureSource, ...]:
    """Parse the value of one family: a single entry mapping or a list of them."""
    raw: Sequence[object]
    if isinstance(value, Mapping):
        raw = (value,)
    elif isinstance(value, MixtureSource):
        raw = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = tuple(value)
    else:
        raise MixtureSourceError(
            f"{field_name} must be a mapping {{source, rows, weight?}} or a list of "
            f"such mappings, got {value!r}",
            field=field_name,
        )
    if not raw:
        raise MixtureSourceError(f"{field_name} declares no sources", field=field_name)
    entries = tuple(
        parse_source_entry(item, family=family, field_name=f"{field_name}[{offset}]")
        for offset, item in enumerate(raw)
    )
    seen: set[str] = set()
    for entry in entries:
        if entry.source in seen:
            raise MixtureSourceError(
                f"{field_name} lists source {entry.source!r} more than once; "
                "merge the allocations into one entry",
                field=field_name,
            )
        seen.add(entry.source)
    return entries


def parse_sources_mapping(
    sources: Mapping[str, object], *, field_name: str = "sources"
) -> dict[str, tuple[MixtureSource, ...]]:
    """Validate the declarative ``family -> entry | [entry, ...]`` mapping."""
    if not isinstance(sources, Mapping) or not sources:
        raise MixtureInputError(
            f"{field_name} must be a non-empty mapping of task family to source "
            f"entries, got {sources!r}",
            field=field_name,
        )
    parsed: dict[str, tuple[MixtureSource, ...]] = {}
    for raw_family, value in sources.items():
        family = raw_family if isinstance(raw_family, str) else str(raw_family)
        if family not in TASK_FAMILIES:
            raise MixtureFamilyError(
                f"unknown task family {raw_family!r}; expected one of {list(TASK_FAMILIES)}",
                field=field_name,
            )
        parsed[family] = parse_family_entries(family, value, field_name=f"{field_name}.{family}")
    return parsed


def _slot_key(fingerprint: str, seed: int, family: str, source: str, index: int) -> str:
    payload = [_SLOT_DOMAIN, fingerprint, seed, family, source, index]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _example_id_key(
    fingerprint: str, seed: int, family: str, source: str, example_id: str
) -> str:
    """Deterministic selection key for a real example inside one source bucket.

    Derived from the recorded fingerprint and seed, so which real rows a
    slot binds to is reproducible on every platform and independent of the
    order rows were supplied in.
    """
    payload = [_SLOT_DOMAIN, "example", fingerprint, seed, family, source, example_id]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def availability_rows_by_id(
    available_rows: Mapping[str, object] | Sequence[DecisionExample],
) -> dict[str, DecisionExample]:
    """Return eligible TRAIN rows keyed by their occurrence identity.

    Content hashes are intentionally option-order invariant, so the preference
    adapter's two de-biasing orientations share ``example_id``. A deterministic
    ``#occurrence=N`` suffix keeps those emissions (and only those) separately
    addressable. Dev/test/OOD rows are never eligible for optimization.
    """
    if isinstance(available_rows, Mapping):
        rows: list[object] = [
            row for group in available_rows.values() for row in _as_sequence(group)
        ]
    else:
        rows = list(available_rows)
    totals: dict[str, int] = {}
    for row in rows:
        if isinstance(row, DecisionExample) and row.split == "train":
            totals[row.example_id] = totals.get(row.example_id, 0) + 1
    seen: dict[str, int] = {}
    output: dict[str, DecisionExample] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, DecisionExample):
            raise MixtureAvailabilityError(
                f"available_rows[{position}] is a {type(row).__name__}, "
                "expected DecisionExample",
                family="",
                source="",
                requested=0,
                available=0,
            )
        if row.split != "train":
            continue
        occurrence = seen.get(row.example_id, 0)
        seen[row.example_id] = occurrence + 1
        key = (
            f"{row.example_id}#occurrence={occurrence}"
            if totals[row.example_id] > 1
            else row.example_id
        )
        output[key] = row
    return output


def _availability_index(
    available_rows: Mapping[str, object] | Sequence[DecisionExample] | None,
) -> dict[tuple[str, str], list[str]] | None:
    """Group eligible train rows by ``(task_family, source)``."""
    if available_rows is None:
        return None
    index: dict[tuple[str, str], list[str]] = {}
    for identity, row in availability_rows_by_id(available_rows).items():
        index.setdefault((row.task_family, row.source), []).append(identity)
    return {key: sorted(ids) for key, ids in index.items()}


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or isinstance(value, DecisionExample):
        return (value,)
    if isinstance(value, Sequence):
        return value
    if isinstance(value, Mapping):
        return (value,)
    return (value,)


def _effective_weights(
    parsed: Mapping[str, tuple[MixtureSource, ...]],
    weights: object,
) -> dict[str, float]:
    def entry_default(family: str) -> float:
        entry_weight = _entry_weight(parsed[family], family=family)
        if entry_weight is not None:
            return entry_weight
        return float(sum(entry.rows for entry in parsed[family]))

    if weights is None:
        return {family: entry_default(family) for family in parsed}
    if not isinstance(weights, Mapping):
        raise MixtureInputError(
            "weights must be a mapping of task family to a positive finite number, "
            f"got {weights!r}",
            field="weights",
        )
    unknown = sorted(str(family) for family in weights if family not in TASK_FAMILIES)
    if unknown:
        raise MixtureInputError(
            f"weights name unknown task family/families: {', '.join(unknown)}; "
            f"expected one of {list(TASK_FAMILIES)}",
            field="weights",
        )
    declared_elsewhere = sorted(family for family in weights if family not in parsed)
    if declared_elsewhere:
        # A weight for a family that contributes nothing to this mixture is
        # almost always a typo; refuse it rather than ignore it.
        raise MixtureInputError(
            f"weights name family/families that contribute no rows to this mixture: "
            f"{', '.join(declared_elsewhere)}",
            field="weights",
        )
    effective: dict[str, float] = {}
    for family in parsed:
        if family in weights:
            value = weights[family]
            if value is None:
                raise MixtureSourceError(
                    f"weights.{family} must be a positive finite number, got None",
                    field=f"weights.{family}",
                )
            validated = _validate_weight(value, field_name=f"weights.{family}")
            assert validated is not None  # narrowed by the None check above
            effective[family] = validated
        else:
            effective[family] = entry_default(family)
    return effective


def _entry_weight(entries: tuple[MixtureSource, ...], *, family: str) -> float | None:
    declared = [entry.weight for entry in entries if entry.weight is not None]
    if not declared:
        return None
    first = declared[0]
    for value in declared[1:]:
        if value != first:
            raise MixtureInputError(
                f"family {family!r} declares conflicting per-source weights {declared}; "
                "give every source in one family the same weight or set the family "
                "weight via the weights argument",
                field="weights",
            )
    return first


def sample_mixture(
    sources: Mapping[str, object] | Any,
    weights: Mapping[str, object] | None,
    seed: object,
    fingerprint: object,
    *,
    available_rows: Mapping[str, object] | Sequence[DecisionExample] | None = None,
    source_status: Any = None,
) -> MixtureOrder:
    """Return the deterministic assignment of mixture slots to a global order.

    ``sources`` is either the raw declarative mapping
    ``family -> {source, rows, weight?}`` (or a list of such entries per
    family) or a :class:`MixtureDeclaration` as returned by
    :func:`src.data.mixture.load_mixture` / :func:`load_preset`; a declaration
    that pins a fingerprint must match the ``fingerprint`` argument.

    ``weights`` is a mapping of family to relative sampling weight; it
    overrides per-entry weights for the families it names. Families it does
    not name fall back to their (common) entry weight, then to a
    row-proportional default.

    ``available_rows`` (real :class:`DecisionExample` records, flat or keyed
    by family) binds every slot to a distinct real ``example_id`` from the
    declared ``(family, source)``. Requests that exceed availability, or
    source/family combinations with no rows, are fatal. Without it the
    legacy synthetic slot ids are retained for declaration-only use.

    ``source_status`` is either a ``source -> status`` mapping (or manifest
    objects) or a directory of acquisition manifests. Any source marked
    ``quarantine`` is refused with :class:`MixtureQuarantinedSourceError`
    (TASK-07 acceptance), never silently dropped.
    """
    from .loader import MixtureDeclaration  # local import: loader imports this module

    pinned: object = None
    mixture_name = DEFAULT_MIXTURE_NAME
    if isinstance(sources, MixtureDeclaration):
        pinned = sources.fingerprint
        mixture_name = sources.name
        parsed = parse_sources_mapping(sources.sources, field_name="families")
    else:
        parsed = parse_sources_mapping(sources)

    seed_value = _validate_seed(seed)
    fingerprint_value = _validate_fingerprint(fingerprint)
    _check_pinned_fingerprint(pinned, fingerprint_value)

    if source_status is not None:
        from .loader import load_source_statuses

        statuses = load_source_statuses(source_status)
        for family, entries in parsed.items():
            for entry in entries:
                status = statuses.get(entry.source)
                # Supplying provenance opts into fail-closed enforcement:
                # missing, unknown and malformed statuses are not admission.
                if status != STATUS_ADMITTED:
                    raise MixtureQuarantinedSourceError(
                        f"source {entry.source!r} declared for family {family!r} has "
                        f"status {status!r}, not {STATUS_ADMITTED!r}; refusing to build "
                        "a training mixture without explicit admission",
                        source=entry.source,
                        status=status or "missing",
                    )

    if available_rows is not None and source_status is None:
        raise MixtureQuarantinedSourceError(
            "real-row training binding requires explicit source-status evidence; "
            "missing manifests must never authorize optimization",
            source="",
            status="missing",
        )
    availability = _availability_index(available_rows)

    effective_weights = _effective_weights(parsed, weights)

    family_rows = {
        family: sum(entry.rows for entry in entries) for family, entries in parsed.items()
    }
    total_rows = sum(family_rows.values())
    if total_rows <= 0:
        raise MixtureInputError(
            "the mixture declares zero rows in total; at least one family must "
            "allocate a positive number of rows",
            field="sources",
        )

    slots: list[MixtureSlot] = []
    counts: dict[str, dict[str, int]] = {}
    for family in sorted(parsed):
        family_counts = counts.setdefault(family, {})
        for entry in parsed[family]:
            family_counts[entry.source] = entry.rows
            for index in range(entry.rows):
                key = _slot_key(fingerprint_value, seed_value, family, entry.source, index)
                slots.append(
                    MixtureSlot(
                        order=-1,  # assigned after the global sort
                        family=family,
                        source=entry.source,
                        index=index,
                        key=key,
                        example_id=f"mixture:{fingerprint_value[:16]}:{family}:{entry.source}:{index}",
                    )
                )
    slots.sort(key=lambda slot: (slot.key, slot.family, slot.source, slot.index))

    # Bind real rows when supplied: the deterministic slot schedule above is
    # untouched; only the example_id each slot carries is resolved from the
    # available (family, source) bucket, with a global dedupe across slots.
    real_ids: dict[tuple[str, str], list[str]] = {}
    if availability is not None:
        used: set[str] = set()
        for family in sorted(parsed):
            for entry in parsed[family]:
                bucket = availability.get((family, entry.source), [])
                candidates = sorted(
                    (example_id for example_id in bucket if example_id not in used),
                    key=lambda example_id: _example_id_key(
                        fingerprint_value, seed_value, family, entry.source, example_id
                    ),
                )
                if len(candidates) < entry.rows:
                    raise MixtureAvailabilityError(
                        f"family {family!r} source {entry.source!r} requests "
                        f"{entry.rows} rows but only {len(candidates)} distinct "
                        "available rows remain after dedupe",
                        family=family,
                        source=entry.source,
                        requested=entry.rows,
                        available=len(candidates),
                    )
                chosen = candidates[: entry.rows]
                used.update(chosen)
                real_ids[(family, entry.source)] = chosen

    pointers: dict[tuple[str, str], int] = {}
    ordered_slots: list[MixtureSlot] = []
    for position, slot in enumerate(slots):
        if availability is None:
            example_id = slot.example_id
        else:
            bucket_key = (slot.family, slot.source)
            cursor = pointers.get(bucket_key, 0)
            example_id = real_ids[bucket_key][cursor]
            pointers[bucket_key] = cursor + 1
        ordered_slots.append(
            MixtureSlot(
                order=position,
                family=slot.family,
                source=slot.source,
                index=slot.index,
                key=slot.key,
                example_id=example_id,
            )
        )
    ordered = tuple(ordered_slots)

    total_weight = sum(effective_weights.values())
    family_shares = {family: rows / total_rows for family, rows in family_rows.items()}
    expected_shares = {family: effective_weights[family] / total_weight for family in parsed}
    share_errors = {
        family: abs(family_shares[family] - expected_shares[family]) for family in parsed
    }

    return MixtureOrder(
        seed=seed_value,
        fingerprint=fingerprint_value,
        mixture=mixture_name,
        total_rows=total_rows,
        slots=ordered,
        weights=dict(effective_weights),
        family_rows=family_rows,
        family_shares=family_shares,
        expected_shares=expected_shares,
        share_errors=share_errors,
        counts=counts,
    )
