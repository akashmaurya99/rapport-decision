"""Risk adapter: Aegis-style multi-label rows → the ``risk`` family.

Field mapping (``nvidia/Aegis-AI-Content-Safety-Dataset-2.0``, CC-BY-4.0;
XSTest is evaluation-only and is *not* handled here):

- ``prompt`` (+ optional ``response``) → ``state``
  (``"<prompt>\\n\\nResponse:\\n<response>"`` when a response is present)
- ``categories`` (list of strings, may be empty) → ``risk_labels``.
  An **explicit empty list** marks the row safe and maps to
  ``risk_labels == ()`` — an annotated "touches no category" reading, which is
  what the all-zero risk target needs. An **absent** ``categories`` field is a
  rejected row (``ADAPTOR_MISSING_FIELD``): unannotated is not the same claim
  as safe. Duplicates are removed (first occurrence wins); more than eight
  labels is rejected with ``ADAPTOR_RISK_LABEL_LIMIT`` (the contract's
  ``RISK_CATEGORY_COUNT``) rather than silently truncated. Label membership
  is not validated — the eight-slot taxonomy is not declared anywhere in this
  repository (see TASK-03's ruling), and nothing here invents it.
- ``options = ["safe", "unsafe"]`` satisfy the contract's ≥2-option rule; the
  gold lives in ``risk_labels``, so ``chosen_index`` is deliberately unset
  (risk is not a choice family)
- ``goal`` is the fixed directive ``"Which harm categories does this content
  touch?"``
- ``group_id`` comes from ``group_id``/``id`` when present, else a
  deterministic content-hash fallback
- ``split`` defaults to ``train``

Option order is shuffled at ingest with the recorded seed (the two polarity
options); ``meta.original_order`` retains the permutation. ``example_id`` is
computed by the TASK-03 builder.

Note for downstream (TASK-11): explicitly-safe rows carry ``risk_labels ==
()``; rows with no risk annotation never reach the loss (they are rejected
here). If the risk loss is masked on empty labels, safe rows are masked too —
flag this interpretation question in the adapter notes rather than change the
encoding silently.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from src.data.core.schema import RISK_CATEGORY_COUNT

from .base import (
    ADAPTOR_BAD_TYPE,
    ADAPTOR_EMPTY_STATE,
    ADAPTOR_MISSING_FIELD,
    ADAPTOR_RISK_LABEL_LIMIT,
    DEFAULT_MAX_FIELD_CHARS,
    DEFAULT_MAX_STATE_CHARS,
    DEFAULT_SEED,
    AdapterResult,
    ExampleMeta,
    RejectedRecord,
    _RecordFailure,
    apply_order,
    base_report,
    finalize,
    fingerprint_row,
    group_id_for,
    length_cap,
    shuffle_order,
    split_for,
)

GOAL = "Which harm categories does this content touch?"
SAFE, UNSAFE = "safe", "unsafe"


def _labels(row: Mapping[str, Any]) -> list[str]:
    value = row.get("categories")
    if value is None:
        raise _RecordFailure(ADAPTOR_MISSING_FIELD, "field 'categories' is absent")
    if not isinstance(value, (list, tuple)):
        raise _RecordFailure(
            ADAPTOR_BAD_TYPE,
            f"field 'categories' must be a list of strings, got {type(value).__name__}",
        )
    labels: list[str] = []
    seen: set[str] = set()
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise _RecordFailure(
                ADAPTOR_BAD_TYPE,
                f"risk label at index {i} must be a non-empty string",
            )
        label = item.strip()
        if label not in seen:
            seen.add(label)
            labels.append(label)
    if len(labels) > RISK_CATEGORY_COUNT:
        raise _RecordFailure(
            ADAPTOR_RISK_LABEL_LIMIT,
            f"at most {RISK_CATEGORY_COUNT} risk labels are declared, got {len(labels)}",
        )
    return labels


class RiskAdapter:
    """Aegis 2.0 (CC-BY-4.0) multi-label safety rows → ``risk`` examples."""

    name = "risk"
    source = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
    license = "CC-BY-4.0"
    task_family = "risk"
    default_split = "train"

    def __init__(
        self,
        *,
        seed: int = DEFAULT_SEED,
        max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
        max_field_chars: int = DEFAULT_MAX_FIELD_CHARS,
        source: str | None = None,
        license: str | None = None,
        default_split: str | None = None,
    ) -> None:
        self.seed = seed
        self.max_state_chars = max_state_chars
        self.max_field_chars = max_field_chars
        if source is not None:
            self.source = source
        if license is not None:
            self.license = license
        if default_split is not None:
            self.default_split = default_split

    def _convert_row(
        self, record_index: int, row: Mapping[str, Any]
    ) -> tuple[dict[str, Any], ExampleMeta]:
        if not isinstance(row, Mapping):
            raise _RecordFailure(
                ADAPTOR_BAD_TYPE, f"row must be a mapping, got {type(row).__name__}"
            )
        fingerprint = fingerprint_row(row)
        split = split_for(row, self.default_split)
        group_id = group_id_for(row, self.source, fingerprint)
        labels = _labels(row)

        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            if prompt is None:
                raise _RecordFailure(ADAPTOR_MISSING_FIELD, "field 'prompt' is absent")
            raise _RecordFailure(ADAPTOR_BAD_TYPE, "field 'prompt' must be a string")
        if not prompt.strip():
            raise _RecordFailure(ADAPTOR_EMPTY_STATE, "field 'prompt' is empty")
        state = prompt.strip()
        response = row.get("response")
        if response is not None:
            if not isinstance(response, str):
                raise _RecordFailure(ADAPTOR_BAD_TYPE, "field 'response' must be a string")
            state = f"{state}\n\nResponse:\n{response.strip()}"
        state = length_cap(state, max_chars=self.max_state_chars, field="state")

        order, per_row_seed = shuffle_order(
            2, master_seed=self.seed, row_fingerprint=fingerprint
        )
        emitted = apply_order([SAFE, UNSAFE], order)

        raw: dict[str, Any] = {
            "task_family": "risk",
            "goal": GOAL,
            "state": state,
            "options": [{"text": t} for t in emitted],
            "source": self.source,
            "license": self.license,
            "group_id": group_id,
            "split": split,
            "risk_labels": labels,
        }
        meta = ExampleMeta(
            original_order=tuple(order),
            shuffle_seed=per_row_seed,
            origin="aegis-2.0",
        )
        return raw, meta

    def to_decision_example(self, rows: Iterable[Mapping[str, Any]]) -> AdapterResult:
        examples, metas, rejected = [], [], []
        rows_seen = 0
        safe_rows = 0
        labeled_rows = 0
        for record_index, row in enumerate(rows):
            rows_seen += 1
            try:
                raw, meta = self._convert_row(record_index, row)
                example = finalize(raw)
                examples.append(example)
                metas.append(meta)
                if example.risk_labels:
                    labeled_rows += 1
                else:
                    safe_rows += 1
            except _RecordFailure as failure:
                rejected.append(
                    RejectedRecord(record_index, failure.reason_code, failure.detail)
                )
        report = base_report(
            name=self.name,
            source=self.source,
            license=self.license,
            task_family=self.task_family,
            seed=self.seed,
            input_rows=rows_seen,
            examples=examples,
            rejected=rejected,
            extra={
                "safe_rows": safe_rows,
                "labeled_rows": labeled_rows,
            },
        )
        return AdapterResult(examples=examples, rejected=rejected, meta=metas, report=report)
