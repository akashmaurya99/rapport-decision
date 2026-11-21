"""Abstention adapter: AbstentionBench-derived rows → the ``abstention`` family.

AbstentionBench (arXiv 2506.09038) is 20 sub-datasets with *heterogeneous*
"unanswerable" definitions — this adapter never pools them. Conversion is
reported **per subset**: for every ``subset`` value present in the input
rows, ``report["per_subset"][subset]`` carries

.. code-block:: json

    {"input": N, "converted": M, "rejected": N - M, "conversion_rate": M / N}

There is deliberately no pooled top-level conversion rate; the sub-datasets
are not commensurable. Rejected rows whose missing/invalid ``subset`` cannot
be bucketed are counted in ``report["unbucketed_rejections"]``, never in a
subset's rate. The adapter declares no canonical subset list — none is
declared in this repository — and reports exactly the subsets the input
contains.

Row shape (offline fixtures mirror the admitted 12k-row derived source):

- ``subset`` (required, non-empty string) — the sub-dataset id
- ``question`` (required) and optional ``context`` → ``state``
  (``"<question>\\n\\nContext:\\n<context>"`` when a context is present)
- answerability, either ``answerable: bool`` or ``unanswerable: bool``;
  providing both is fine only when they agree, otherwise the row is rejected
  with ``ADAPTOR_CONTRADICTORY_LABEL``
- ``goal`` is the fixed directive ``"Is the question answerable from the
  given state?"``
- ``options = ["Answerable", "Unanswerable"]``, shuffled at ingest with the
  recorded seed; ``chosen_index`` tracks the gold through the shuffle and
  ``answerable`` carries it as the contract's boolean
- ``group_id`` comes from ``group_id``/``id`` when present, else a
  deterministic content-hash fallback
- ``split`` defaults to ``train``

Example-level provenance (``subset``) is recorded in ``meta.subset``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .base import (
    ADAPTOR_BAD_TYPE,
    ADAPTOR_CONTRADICTORY_LABEL,
    ADAPTOR_EMPTY_STATE,
    ADAPTOR_MISSING_FIELD,
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

GOAL = "Is the question answerable from the given state?"
YES, NO = "Answerable", "Unanswerable"


def _resolve_answerable(row: Mapping[str, Any]) -> bool:
    ans = row.get("answerable")
    unans = row.get("unanswerable")
    if ans is None and unans is None:
        raise _RecordFailure(
            ADAPTOR_MISSING_FIELD, "provide either 'answerable' or 'unanswerable' (bool)"
        )
    for key, value in (("answerable", ans), ("unanswerable", unans)):
        if value is not None and not isinstance(value, bool):
            raise _RecordFailure(
                ADAPTOR_BAD_TYPE,
                f"field {key!r} must be a boolean, got {type(value).__name__}",
            )
    if ans is not None and unans is not None:
        if ans == unans:
            raise _RecordFailure(
                ADAPTOR_CONTRADICTORY_LABEL,
                f"answerable={ans} and unanswerable={unans} cannot both hold",
            )
        return bool(ans)
    if ans is not None:
        return bool(ans)
    assert unans is not None
    return not bool(unans)


class AbstentionAdapter:
    """AbstentionBench-derived rows; per-subset conversion rates are reported,
    never pooled."""

    name = "abstention"
    source = "AbstentionBench-derived"
    license = "AbstentionBench (arXiv 2506.09038) — see paper"
    task_family = "abstention"
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
        subset = row.get("subset")
        if not isinstance(subset, str) or not subset.strip():
            if subset is None:
                raise _RecordFailure(ADAPTOR_MISSING_FIELD, "field 'subset' is absent")
            raise _RecordFailure(
                ADAPTOR_BAD_TYPE,
                f"field 'subset' must be a non-empty string, got {type(subset).__name__}",
            )
        subset = subset.strip()

        fingerprint = fingerprint_row(row)
        split = split_for(row, self.default_split)
        group_id = group_id_for(row, self.source, fingerprint)
        answerable = _resolve_answerable(row)

        question = row.get("question")
        if not isinstance(question, str):
            if question is None:
                raise _RecordFailure(ADAPTOR_MISSING_FIELD, "field 'question' is absent")
            raise _RecordFailure(ADAPTOR_BAD_TYPE, "field 'question' must be a string")
        if not question.strip():
            raise _RecordFailure(ADAPTOR_EMPTY_STATE, "field 'question' is empty")
        state = question.strip()
        context = row.get("context")
        if context is not None:
            if not isinstance(context, str):
                raise _RecordFailure(ADAPTOR_BAD_TYPE, "field 'context' must be a string")
            state = f"{state}\n\nContext:\n{context.strip()}"
        state = length_cap(state, max_chars=self.max_state_chars, field="state")

        order, per_row_seed = shuffle_order(
            2, master_seed=self.seed, row_fingerprint=fingerprint
        )
        emitted = apply_order([YES, NO], order)
        gold_position = order.index(0 if answerable else 1)

        raw = {
            "task_family": "abstention",
            "goal": GOAL,
            "state": state,
            "options": [{"text": t} for t in emitted],
            "source": self.source,
            "license": self.license,
            "group_id": group_id,
            "split": split,
            "chosen_index": gold_position,
            "answerable": answerable,
        }
        meta = ExampleMeta(
            original_order=tuple(order),
            shuffle_seed=per_row_seed,
            origin="abstentionbench",
            subset=subset,
        )
        return raw, meta

    def to_decision_example(self, rows: Iterable[Mapping[str, Any]]) -> AdapterResult:
        examples, metas, rejected = [], [], []
        rows_seen = 0
        per_subset: dict[str, dict[str, Any]] = {}
        unbucketed_rejections = 0
        for record_index, row in enumerate(rows):
            rows_seen += 1
            subset_key: str | None = None
            if isinstance(row, Mapping):
                value = row.get("subset")
                if isinstance(value, str) and value.strip():
                    subset_key = value.strip()
            try:
                raw, meta = self._convert_row(record_index, row)
                example = finalize(raw)
                examples.append(example)
                metas.append(meta)
                bucket = per_subset.setdefault(
                    meta.subset or "<unknown>",
                    {"input": 0, "converted": 0, "rejected": 0, "conversion_rate": 0.0},
                )
                bucket["input"] += 1
                bucket["converted"] += 1
            except _RecordFailure as failure:
                rejected.append(
                    RejectedRecord(record_index, failure.reason_code, failure.detail)
                )
                if subset_key is not None:
                    bucket = per_subset.setdefault(
                        subset_key,
                        {"input": 0, "converted": 0, "rejected": 0, "conversion_rate": 0.0},
                    )
                    bucket["input"] += 1
                    bucket["rejected"] += 1
                else:
                    unbucketed_rejections += 1
        for bucket in per_subset.values():
            bucket["conversion_rate"] = (
                bucket["converted"] / bucket["input"] if bucket["input"] else 0.0
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
                "per_subset": per_subset,
                "n_subsets": len(per_subset),
                "unbucketed_rejections": unbucketed_rejections,
                # Deliberately no pooled conversion rate: the 20 sub-datasets
                # have heterogeneous unanswerable definitions.
            },
        )
        return AdapterResult(examples=examples, rejected=rejected, meta=metas, report=report)
