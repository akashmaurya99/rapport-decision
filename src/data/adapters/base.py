"""Shared adapter machinery: protocol, result types, code constants.

One :class:`Adapter` protocol, one ``to_decision_example(rows) -> AdapterResult``
method on every adapter. The result carries *every* input row's fate:
``examples`` holds the normalized :class:`DecisionExample` records (with
``source``, ``task_family`` and ``group_id`` set), ``rejected`` holds one
:class:`RejectedRecord` per unmappable row — ``(record_index, typed_reason_code)``
— and ``meta`` holds one :class:`ExampleMeta` per emitted example, aligned by
position with ``examples``. Nothing is silently skipped: a row either yields
an example or a rejection.

Option order is shuffled at ingest with a recorded seed. Every shuffle derives
a per-row RNG from ``sha256(seed_domain || master_seed || row_fingerprint)``,
so the shuffle is deterministic for the same master seed and the same row
content, regardless of row order. The master seed is recorded in the report
(``report["shuffle_seed"]``), the per-row derived seed in ``meta.shuffle_seed``,
and the emitted-position → source-position permutation in ``meta.original_order``;
:func:`restore_original_order` rebuilds the source option order (and the
chosen index) from those two fields.

``example_id`` is never computed here: every example goes through
``normalize_example`` (TASK-03's builder), which recomputes — and verifies —
the content hash.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, NamedTuple, Protocol, runtime_checkable

from src.data.core.errors import DataError
from src.data.core.normalize import normalize_example
from src.data.core.schema import DecisionExample

DEFAULT_SEED = 13
_SHUFFLE_DOMAIN = "rapport:adapter-shuffle:v1"
_REDUCE_DOMAIN = "rapport:mmlu-pro-reduce:v1"
_GROUP_DOMAIN = "rapport:adapter-group:v1"

# Default caps. Truncation is reserved for the preference family (HH-RLHF
# transcripts); every other family rejects an overlong field with
# ADAPTOR_TOO_LONG rather than silently dropping content.
DEFAULT_MAX_STATE_CHARS = 8_000
DEFAULT_MAX_FIELD_CHARS = 100_000

# --- Typed rejection codes (adapter level) ----------------------------------
# Core-contract violations flow through with their TASK-03 codes (e.g.
# DATA_DUPLICATE_OPTION); these ADAPTOR_* codes mark failures the adapters
# detect before the contract is reached.
ADAPTOR_MISSING_FIELD = "ADAPTOR_MISSING_FIELD"
ADAPTOR_BAD_TYPE = "ADAPTOR_BAD_TYPE"
ADAPTOR_NO_GOLD = "ADAPTOR_NO_GOLD"
ADAPTOR_BAD_INDEX = "ADAPTOR_BAD_INDEX"
ADAPTOR_EMPTY_STATE = "ADAPTOR_EMPTY_STATE"
ADAPTOR_TOO_LONG = "ADAPTOR_TOO_LONG"
ADAPTOR_NO_OPTIONS = "ADAPTOR_NO_OPTIONS"
ADAPTOR_BAD_SCORE = "ADAPTOR_BAD_SCORE"
ADAPTOR_CONTRADICTORY_LABEL = "ADAPTOR_CONTRADICTORY_LABEL"
ADAPTOR_RISK_LABEL_LIMIT = "ADAPTOR_RISK_LABEL_LIMIT"
ADAPTOR_META_MISSING = "ADAPTOR_META_MISSING"

ALL_ADAPTOR_CODES = (
    ADAPTOR_MISSING_FIELD,
    ADAPTOR_BAD_TYPE,
    ADAPTOR_NO_GOLD,
    ADAPTOR_BAD_INDEX,
    ADAPTOR_EMPTY_STATE,
    ADAPTOR_TOO_LONG,
    ADAPTOR_NO_OPTIONS,
    ADAPTOR_BAD_SCORE,
    ADAPTOR_CONTRADICTORY_LABEL,
    ADAPTOR_RISK_LABEL_LIMIT,
    ADAPTOR_META_MISSING,
)


class AdapterError(Exception):
    """Raised for adapter-level failures outside the per-row reject path."""

    code = "ADAPTOR_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message


class _RecordFailure(Exception):
    """Internal control flow: one input row failed; carry the typed reason."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


class RejectedRecord(NamedTuple):
    """One unmappable input row.

    ``record_index`` is the 0-based position of the row in the input iterable.
    ``reason_code`` is a stable typed code (``ADAPTOR_*`` or a TASK-03
    ``DATA_*`` code when the failure is a contract violation). ``detail``
    names the offending field/value; it is for humans, never for parsing.
    """

    record_index: int
    reason_code: str
    detail: str = ""


@dataclass(frozen=True)
class ExampleMeta:
    """Adapter-side metadata for one emitted example.

    ``DecisionExample`` is frozen and its field set is fixed by the contract,
    so provenance that the contract has no slot for lives here, aligned by
    position with ``AdapterResult.examples``:

    - ``original_order``: emitted-position → source-position permutation of
      the option list. Retained for reconstruction via
      :func:`restore_original_order`.
    - ``shuffle_seed``: the per-row derived seed used for the ingest shuffle
      (``None`` when the adapter emits a deterministic permutation instead).
    - ``reduction``: reduction method applied, e.g. ``"mmlu-pro-10-to-8"``.
    - ``reduction_seed``: the seed used for non-gold distractor removal.
    - ``dropped_indices``: source option indices removed by the reduction.
    - ``kept_indices``: source indices of the surviving options, in the
      pre-shuffle kept order; ``None`` when no reduction was applied (the
      kept order is the source order). With ``original_order`` this fully
      determines the source option permutation of a reduced example.
    - ``truncated``: True when a text field was truncated at ingest.
    - ``origin`` / ``subset``: provenance tags (``"mmlu-auxiliary-train"``,
      ``"truthfulqa"``, the AbstentionBench subset name, ...).
    """

    original_order: tuple[int, ...] | None = None
    shuffle_seed: int | None = None
    reduction: str | None = None
    reduction_seed: int | None = None
    dropped_indices: tuple[int, ...] = ()
    kept_indices: tuple[int, ...] | None = None
    truncated: bool = False
    origin: str | None = None
    subset: str | None = None


@dataclass(frozen=True)
class AdapterResult:
    """The outcome of mapping one batch of raw rows.

    ``meta`` is aligned by position with ``examples``. ``report`` carries the
    adapter-level counters: input rows, converted, rejected, the recorded
    ingest seed and any adapter-specific figures (the MMLU-Pro reduction
    count, the auxiliary_train-derived binary row count, per-subset conversion
    rates). No figure in ``report`` pools rates that the contract requires to
    stay per-subset.
    """

    examples: list[DecisionExample] = field(default_factory=list)
    rejected: list[RejectedRecord] = field(default_factory=list)
    meta: list[ExampleMeta] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Adapter(Protocol):
    """The single adapter interface.

    Every adapter declares ``source``, ``license``, ``task_family`` and
    ``name`` as class attributes and maps a batch of raw rows with one
    method — :meth:`to_decision_example`.
    """

    source: str
    license: str
    task_family: str
    name: str

    def to_decision_example(self, rows: Iterable[Mapping[str, Any]]) -> AdapterResult:
        """Map raw rows; never raise for a bad *row* (reject it instead).

        Raises :class:`AdapterError` only for adapter misconfiguration.
        """
        ...


# --- shared helpers -----------------------------------------------------------


def fingerprint_row(row: Mapping[str, Any]) -> str:
    """Stable fingerprint of one raw row (canonical JSON, sha256)."""
    blob = json.dumps(row, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _derive_seed(domain: str, master: int, material: str) -> int:
    digest = hashlib.sha256(f"{domain}:{master}:{material}".encode("utf-8")).hexdigest()
    return int(digest, 16)


def shuffle_order(n: int, *, master_seed: int, row_fingerprint: str) -> tuple[list[int], int]:
    """Return ``(order, per_row_seed)`` for shuffling ``n`` options.

    ``order[j]`` is the source index of the option that lands in emitted
    position ``j``. Deterministic given ``master_seed`` and the row content;
    independent of row order in the batch.
    """
    per_row_seed = _derive_seed(_SHUFFLE_DOMAIN, master_seed, row_fingerprint)
    order = list(range(n))
    random.Random(per_row_seed).shuffle(order)
    return order, per_row_seed


def reduce_seed(master: int, row_fingerprint: str) -> int:
    """Per-row seed for the MMLU-Pro 10→8 non-gold removal."""
    return _derive_seed(_REDUCE_DOMAIN, master, row_fingerprint)


def apply_order(items: list[Any], order: list[int]) -> list[Any]:
    """``apply_order(xs, order)[j] == xs[order[j]]``."""
    return [items[i] for i in order]


def derive_group_id(source: str, row_fingerprint: str) -> str:
    """Deterministic fallback group id when the source row declares none."""
    blob = f"{_GROUP_DOMAIN}:{source}:{row_fingerprint}".encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    return f"{source}:{digest[:16]}"


def cap_text(text: str, *, max_chars: int, field: str) -> tuple[str, bool]:
    """Truncate at a word boundary; returns ``(text, truncated)``.

    A text that is already short enough is returned unchanged. Truncation
    breaks at a whitespace boundary when one exists in the second half of the
    window; the result always fits within ``max_chars``. ``field`` is accepted
    for signature symmetry with :func:`length_cap` and error messages.
    """
    if len(text) <= max_chars:
        return text, False
    window = text[:max_chars]
    cut = window.rfind(" ")
    if cut >= max_chars // 2:
        window = window[:cut]
    return window.rstrip(), True


def require_str(row: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> str:
    value = row.get(key)
    if value is None:
        raise _RecordFailure(ADAPTOR_MISSING_FIELD, f"field {key!r} is absent")
    if not isinstance(value, str):
        raise _RecordFailure(
            ADAPTOR_BAD_TYPE, f"field {key!r} must be a string, got {type(value).__name__}"
        )
    text = value.strip()
    if not text and not allow_empty:
        raise _RecordFailure(ADAPTOR_MISSING_FIELD, f"field {key!r} is empty")
    return text


def length_cap(text: str, *, max_chars: int, field: str) -> str:
    if len(text) > max_chars:
        raise _RecordFailure(
            ADAPTOR_TOO_LONG, f"field {field!r} is {len(text)} chars (cap {max_chars})"
        )
    return text


def group_id_for(row: Mapping[str, Any], source: str, row_fingerprint: str) -> str:
    value = row.get("group_id") or row.get("id")
    if value is None:
        return derive_group_id(source, row_fingerprint)
    if not isinstance(value, str) or not value.strip():
        raise _RecordFailure(ADAPTOR_BAD_TYPE, "group_id must be a non-empty string")
    return value.strip()


def split_for(row: Mapping[str, Any], default: str) -> str:
    value = row.get("split", default)
    if not isinstance(value, str):
        raise _RecordFailure(ADAPTOR_BAD_TYPE, "split must be a string")
    return value.strip() or default


def finalize(raw: Mapping[str, Any]) -> DecisionExample:
    """Normalize one assembled record; map contract violations onto rejections."""
    try:
        return normalize_example(raw)
    except DataError as exc:
        raise _RecordFailure(exc.code, exc.message) from exc


def base_report(
    *,
    name: str,
    source: str,
    license: str,
    task_family: str,
    seed: int,
    input_rows: int,
    examples: list[DecisionExample],
    rejected: list[RejectedRecord],
    extra: dict[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "adapter": name,
        "source": source,
        "license": license,
        "task_family": task_family,
        "input_rows": input_rows,
        "converted": len(examples),
        "rejected": len(rejected),
        "shuffle_seed": seed,
    }
    report.update(extra)
    return report


def restore_original_order(
    example: DecisionExample, meta: ExampleMeta
) -> DecisionExample:
    """Rebuild ``example`` with options in the source (pre-shuffle) order.

    ``meta.original_order[j]`` is the source position of the option emitted at
    position ``j``, so source option ``i`` was emitted at ``inverse[i]`` where
    ``inverse`` is the inverse permutation. Chosen indices map through the
    same permutation and scores travel with their options. The content id is
    recomputed through the TASK-03 builder (parsing re-validates). For reduced
    (10→8) examples the dropped options are gone by design; the restored
    record holds the kept options in source order.
    """
    if meta.original_order is None or len(meta.original_order) != len(example.options):
        raise AdapterError(
            "meta.original_order is absent or inconsistent with the option count; "
            "the original order cannot be reconstructed",
            code=ADAPTOR_META_MISSING,
        )
    order = list(meta.original_order)
    inverse = [0] * len(order)
    for emitted_pos, source_pos in enumerate(order):
        inverse[source_pos] = emitted_pos

    options = [example.options[inverse[i]] for i in range(len(order))]
    raw: dict[str, Any] = {
        "task_family": example.task_family,
        "goal": example.goal,
        "state": example.state,
        "options": [{"text": opt.text, "label": opt.label} for opt in options],
        "source": example.source,
        "license": example.license,
        "group_id": example.group_id,
        "split": example.split,
    }
    if example.chosen_index is not None:
        raw["chosen_index"] = order[example.chosen_index]
    if example.scores is not None:
        raw["scores"] = [example.scores[inverse[i]] for i in range(len(order))]
    if example.answerable is not None:
        raw["answerable"] = example.answerable
    if example.risk_labels is not None:
        raw["risk_labels"] = list(example.risk_labels)
    return normalize_example(raw)
