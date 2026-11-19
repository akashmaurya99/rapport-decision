"""Scoring adapter: HelpSteer2 rows → the ``scoring`` family.

Field mapping (``nvidia/HelpSteer2``, CC-BY-4.0):

- ``prompt`` + ``response`` → ``state`` as ``"<prompt>\\n\\nResponse: <response>"``
- the three *quality* attributes ``helpfulness``, ``correctness`` and
  ``coherence`` (each an integer/float rating 0–4) become the options, and
  their ratings normalized to ``rating / 4`` become the ``scores`` — utility
  floats in ``[0, 1]``, one per option, aligned positionally
- ``complexity`` and ``verbosity`` are **not** quality axes (higher is not
  better, so they carry no utility signal) and are excluded from the mapping;
  the exclusion is recorded in ``report["excluded_attributes"]``, never silent
- ``goal`` is the fixed directive ``"Rate the response on each quality
  aspect."``
- ``group_id`` comes from ``group_id``/``id`` when present, else a
  deterministic content-hash fallback
- ``split`` defaults to ``train``

Option order is shuffled at ingest with the recorded seed; the scores are
permuted along with the options (score *of* an aspect travels with it), and
``meta.original_order`` retains the permutation for reconstruction via
``base.restore_original_order``. ``example_id`` is computed by the TASK-03
builder.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .base import (
    ADAPTOR_BAD_SCORE,
    ADAPTOR_BAD_TYPE,
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

GOAL = "Rate the response on each quality aspect."
QUALITY_ATTRIBUTES = ("helpfulness", "correctness", "coherence")
EXCLUDED_ATTRIBUTES = ("complexity", "verbosity")
MAX_RATING = 4.0


def _rating(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise _RecordFailure(ADAPTOR_MISSING_FIELD, f"field {key!r} is absent")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _RecordFailure(
            ADAPTOR_BAD_TYPE,
            f"field {key!r} must be a number, got {type(value).__name__}",
        )
    rating = float(value)
    if not rating == rating:  # NaN
        raise _RecordFailure(ADAPTOR_BAD_SCORE, f"field {key!r} is NaN")
    if not 0.0 <= rating <= MAX_RATING:
        raise _RecordFailure(
            ADAPTOR_BAD_SCORE, f"field {key!r} is {rating}; expected 0..4"
        )
    return rating


class ScoringAdapter:
    """``nvidia/HelpSteer2`` multi-attribute ratings → per-aspect utilities."""

    name = "scoring"
    source = "nvidia/HelpSteer2"
    license = "CC-BY-4.0"
    task_family = "scoring"
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

        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            if prompt is None:
                raise _RecordFailure(ADAPTOR_MISSING_FIELD, "field 'prompt' is absent")
            raise _RecordFailure(ADAPTOR_BAD_TYPE, "field 'prompt' must be a string")
        if not prompt.strip():
            raise _RecordFailure(ADAPTOR_EMPTY_STATE, "field 'prompt' is empty")
        response = row.get("response")
        if not isinstance(response, str):
            if response is None:
                raise _RecordFailure(ADAPTOR_MISSING_FIELD, "field 'response' is absent")
            raise _RecordFailure(ADAPTOR_BAD_TYPE, "field 'response' must be a string")
        if not response.strip():
            raise _RecordFailure(ADAPTOR_EMPTY_STATE, "field 'response' is empty")
        state = f"{prompt.strip()}\n\nResponse:\n{response.strip()}"
        state = length_cap(state, max_chars=self.max_state_chars, field="state")

        ratings = [_rating(row, attr) for attr in QUALITY_ATTRIBUTES]
        utilities = [r / MAX_RATING for r in ratings]

        order, per_row_seed = shuffle_order(
            len(QUALITY_ATTRIBUTES), master_seed=self.seed, row_fingerprint=fingerprint
        )
        emitted_attributes = apply_order(list(QUALITY_ATTRIBUTES), order)
        emitted_scores = apply_order(utilities, order)

        raw = {
            "task_family": "scoring",
            "goal": GOAL,
            "state": state,
            "options": [{"text": attr} for attr in emitted_attributes],
            "source": self.source,
            "license": self.license,
            "group_id": group_id,
            "split": split,
            "scores": emitted_scores,
        }
        meta = ExampleMeta(
            original_order=tuple(order),
            shuffle_seed=per_row_seed,
            origin="helpsteer2",
        )
        return raw, meta

    def to_decision_example(self, rows: Iterable[Mapping[str, Any]]) -> AdapterResult:
        examples, metas, rejected = [], [], []
        rows_seen = 0
        for record_index, row in enumerate(rows):
            rows_seen += 1
            try:
                raw, meta = self._convert_row(record_index, row)
                examples.append(finalize(raw))
                metas.append(meta)
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
                "attributes_mapped": list(QUALITY_ATTRIBUTES),
                "excluded_attributes": list(EXCLUDED_ATTRIBUTES),
                "excluded_reason": "descriptive axes, not utility/quality scales",
            },
        )
        return AdapterResult(examples=examples, rejected=rejected, meta=metas, report=report)
