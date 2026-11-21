"""Probability adapter: TruthfulQA + auxiliary_train-derived binaries → ``probability``.

Two row shapes, dispatched by keys (both admitted):

1. **auxiliary_train-style** (``cais/mmlu`` ``auxiliary_train``, MIT — the
   admitted training split; MMLU test/dev never enter training). Formula: for
   a 4-option auxiliary_train row, emit **one row per option**:

   ``"Is '<option>' the correct answer?"``

   with ``options = ["Yes", "No"]`` (shuffled at ingest, recorded seed),
   ``chosen_index`` pointing at the gold answer and ``answerable`` set to the
   proposition's truth value (``True`` iff the queried option is the gold —
   hence both values appear, exactly ``n_options`` rows per parent, of which
   exactly one carries ``answerable=True``). The gold lives in ``chosen_index``
   over the two options; ``answerable`` doubles as the binary gold for the
   proposition, matching the contract's ``answerable`` slot. All derived rows
   of one parent share the parent's ``group_id`` so they can never straddle a
   split. The number of emitted derived rows is reported as
   ``report["derived_binary_rows"]``.

2. **TruthfulQA-style** (``truthfulqa/truthful_qa``, Apache-2.0). For each row,
   one proposition row for ``best_answer`` (gold: Yes) and, when
   ``incorrect_answers`` is non-empty, one for ``incorrect_answers[0]`` (gold:
   No). Same Yes/No encoding, same shuffle, same shared ``group_id``. Counts
   are reported as ``truthfulqa_yes_rows`` / ``truthfulqa_no_rows``.

Anything else is rejected with ``ADAPTOR_MISSING_FIELD``. ``goal`` is the
fixed directive ``"Is this proposition true?"`` (the spec's truth-verification
question). ``example_id`` is computed by the TASK-03 builder; option order is
shuffled at ingest and reconstructed via ``meta.original_order``.

Provenance is **per shape**: auxiliary_train-derived rows carry the derived
source id ``cais/mmlu#auxiliary_train-derived-binary`` and the MIT license,
never the TruthfulQA/Apache-2.0 class attributes (those apply to shape 2
only). When an acquisition manifest is supplied for ``cais/mmlu``, its
license is used in preference to the MIT default; a quarantined manifest is
never trusted. The resolved provenance is recorded in
``report["provenance"]``.

Note on ``answerable`` for this family: the spec masks the abstention loss
when ``answerable`` is unset, and probability rows are not abstention rows —
the per-task loss masking is keyed off ``task_family`` there. Setting
``answerable`` on probability rows is still safe (nothing consumes it outside
the abstention family) and carries the binary gold in the contract's only
boolean slot.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .base import (
    ADAPTOR_BAD_TYPE,
    ADAPTOR_EMPTY_STATE,
    ADAPTOR_MISSING_FIELD,
    DEFAULT_MAX_FIELD_CHARS,
    DEFAULT_MAX_STATE_CHARS,
    DEFAULT_SEED,
    AdapterError,
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
from .choice import _parse_gold

GOAL = "Is this proposition true?"
YES, NO = "Yes", "No"

#: Shape-1 provenance. The derived rows are not TruthfulQA data.
MMLU_SOURCE = "cais/mmlu#auxiliary_train-derived-binary"
MMLU_LICENSE = "MIT"


def license_from_manifest(manifest: Any | None, *, default: str) -> str:
    """Resolve a license from an acquisition manifest, never from quarantine.

    ``manifest`` is duck-typed (a :class:`~src.data.acquire.SourceManifest`
    in production, a simple object or mapping in tests). No manifest means the
    caller explicitly uses the adapter's declared provenance. Once a manifest
    is supplied, only an admitted source with a real license is accepted;
    quarantine or missing provenance can never fall back to an invented MIT.
    """
    if manifest is None:
        return default
    if isinstance(manifest, Mapping):
        status = manifest.get("status")
        license_ = manifest.get("license")
    else:
        status = getattr(manifest, "status", None)
        license_ = getattr(manifest, "license", None)
    normalized_status = str(status).strip().lower() if status is not None else ""
    if normalized_status != "admitted":
        raise AdapterError(
            f"cais/mmlu acquisition status {normalized_status or 'missing'!r} is not admitted"
        )
    if not isinstance(license_, str) or not license_.strip():
        raise AdapterError("admitted cais/mmlu manifest carries no usable license")
    return license_.strip()


class ProbabilityAdapter:
    """TruthfulQA (Apache-2.0) + ``cais/mmlu`` auxiliary_train → binary propositions."""

    name = "probability"
    source = "truthfulqa/truthful_qa"
    license = "Apache-2.0"
    task_family = "probability"
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
        mmlu_source: str = MMLU_SOURCE,
        mmlu_license: str | None = None,
        mmlu_manifest: Any | None = None,
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
        self.mmlu_source = mmlu_source
        self.mmlu_license = license_from_manifest(
            mmlu_manifest, default=MMLU_LICENSE if mmlu_license is None else mmlu_license
        )

    # -- shared bits -----------------------------------------------------------

    def _binary_row(
        self,
        *,
        state: str,
        is_gold: bool,
        group_id: str,
        split: str,
        fingerprint: str,
        origin: str,
        source: str,
        license_: str,
    ) -> tuple[dict[str, Any], ExampleMeta]:
        order, per_row_seed = shuffle_order(
            2, master_seed=self.seed, row_fingerprint=f"{fingerprint}:{state}"
        )
        emitted = apply_order([YES, NO], order)
        gold_position = order.index(0 if is_gold else 1)
        raw = {
            "task_family": "probability",
            "goal": GOAL,
            "state": state,
            "options": [{"text": t} for t in emitted],
            "source": source,
            "license": license_,
            "group_id": group_id,
            "split": split,
            "chosen_index": gold_position,
            "answerable": bool(is_gold),
        }
        meta = ExampleMeta(
            original_order=tuple(order),
            shuffle_seed=per_row_seed,
            origin=origin,
        )
        return raw, meta

    # -- shape 1: auxiliary_train-style ----------------------------------------

    def _from_auxiliary_train(
        self, record_index: int, row: Mapping[str, Any], fingerprint: str
    ) -> list[tuple[dict[str, Any], ExampleMeta]]:
        question = row.get("question")
        if not isinstance(question, str):
            if question is None:
                raise _RecordFailure(ADAPTOR_MISSING_FIELD, "field 'question' is absent")
            raise _RecordFailure(ADAPTOR_BAD_TYPE, "field 'question' must be a string")
        if not question.strip():
            raise _RecordFailure(ADAPTOR_EMPTY_STATE, "field 'question' is empty")
        choices = row.get("choices")
        if choices is None:
            raise _RecordFailure(ADAPTOR_MISSING_FIELD, "field 'choices' is absent")
        if not isinstance(choices, (list, tuple)):
            raise _RecordFailure(
                ADAPTOR_BAD_TYPE,
                f"field 'choices' must be a list of strings, got {type(choices).__name__}",
            )
        if len(choices) < 2:
            raise _RecordFailure(
                ADAPTOR_MISSING_FIELD,
                f"field 'choices' needs at least 2 options, got {len(choices)}",
            )
        options: list[str] = []
        for i, item in enumerate(choices):
            if not isinstance(item, str) or not item.strip():
                raise _RecordFailure(
                    ADAPTOR_MISSING_FIELD,
                    f"option at index {i} is absent or not a non-empty string",
                )
            capped = length_cap(item.strip(), max_chars=self.max_field_chars, field=f"choices[{i}]")
            options.append(capped)
        gold = _parse_gold(row.get("answer"), len(options))
        split = split_for(row, self.default_split)
        group_id = group_id_for(row, self.mmlu_source, fingerprint)
        question = length_cap(question.strip(), max_chars=self.max_state_chars, field="question")

        derived: list[tuple[dict[str, Any], ExampleMeta]] = []
        for i, option in enumerate(options):
            state = length_cap(
                f"{question}\n\nIs '{option}' the correct answer?",
                max_chars=self.max_state_chars,
                field="state",
            )
            derived.append(
                self._binary_row(
                    state=state,
                    is_gold=(i == gold),
                    group_id=group_id,
                    split=split,
                    fingerprint=fingerprint,
                    origin="mmlu-auxiliary-train",
                    source=self.mmlu_source,
                    license_=self.mmlu_license,
                )
            )
        return derived

    # -- shape 2: TruthfulQA-style ---------------------------------------------

    def _from_truthfulqa(
        self, record_index: int, row: Mapping[str, Any], fingerprint: str
    ) -> list[tuple[dict[str, Any], ExampleMeta]]:
        question = row.get("question")
        if not isinstance(question, str):
            if question is None:
                raise _RecordFailure(ADAPTOR_MISSING_FIELD, "field 'question' is absent")
            raise _RecordFailure(ADAPTOR_BAD_TYPE, "field 'question' must be a string")
        if not question.strip():
            raise _RecordFailure(ADAPTOR_EMPTY_STATE, "field 'question' is empty")
        question = length_cap(question.strip(), max_chars=self.max_state_chars, field="question")
        split = split_for(row, self.default_split)
        group_id = group_id_for(row, self.source, fingerprint)

        best = row.get("best_answer")
        if not isinstance(best, str) or not best.strip():
            if best is None:
                raise _RecordFailure(ADAPTOR_MISSING_FIELD, "field 'best_answer' is absent")
            raise _RecordFailure(ADAPTOR_BAD_TYPE, "field 'best_answer' must be a non-empty string")

        derived: list[tuple[dict[str, Any], ExampleMeta]] = []
        propositions = [(best.strip(), True)]
        incorrect = row.get("incorrect_answers")
        if incorrect is not None:
            if not isinstance(incorrect, (list, tuple)):
                raise _RecordFailure(
                    ADAPTOR_BAD_TYPE,
                    f"field 'incorrect_answers' must be a list, got {type(incorrect).__name__}",
                )
            first_bad = next(
                (v for v in incorrect if not isinstance(v, str) or not v.strip()), None
            )
            if first_bad is not None:
                raise _RecordFailure(
                    ADAPTOR_MISSING_FIELD, "field 'incorrect_answers' holds an empty entry"
                )
            if incorrect:
                propositions.append((incorrect[0].strip(), False))
        for proposition, is_gold in propositions:
            state = length_cap(
                f"{question}\n\nProposition: {proposition}",
                max_chars=self.max_state_chars,
                field="state",
            )
            derived.append(
                self._binary_row(
                    state=state,
                    is_gold=is_gold,
                    group_id=group_id,
                    split=split,
                    fingerprint=fingerprint,
                    origin="truthfulqa",
                    source=self.source,
                    license_=self.license,
                )
            )
        return derived

    # -- entry point -----------------------------------------------------------

    def to_decision_example(self, rows: Iterable[Mapping[str, Any]]) -> AdapterResult:
        examples, metas, rejected = [], [], []
        derived = 0
        yes_rows = 0
        no_rows = 0
        rows_seen = 0
        for record_index, row in enumerate(rows):
            rows_seen += 1
            try:
                if not isinstance(row, Mapping):
                    raise _RecordFailure(
                        ADAPTOR_BAD_TYPE, f"row must be a mapping, got {type(row).__name__}"
                    )
                fingerprint = fingerprint_row(row)
                if "choices" in row or "answer" in row:
                    built = self._from_auxiliary_train(record_index, row, fingerprint)
                elif "best_answer" in row:
                    built = self._from_truthfulqa(record_index, row, fingerprint)
                else:
                    raise _RecordFailure(
                        ADAPTOR_MISSING_FIELD,
                        "row is neither auxiliary_train-style "
                        "('question' + 'choices' + 'answer') nor TruthfulQA-style "
                        "('question' + 'best_answer')",
                    )
                emitted: list[tuple[Any, ExampleMeta]] = []
                for raw, meta in built:
                    emitted.append((finalize(raw), meta))
                for example, meta in emitted:
                    examples.append(example)
                    metas.append(meta)
                    if meta.origin == "mmlu-auxiliary-train":
                        derived += 1
                    else:
                        yes_rows += 1 if example.answerable else 0
                        no_rows += 0 if example.answerable else 1
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
                "derived_binary_rows": derived,
                "truthfulqa_yes_rows": yes_rows,
                "truthfulqa_no_rows": no_rows,
                "provenance": {
                    "mmlu_auxiliary_train_derived": {
                        "source": self.mmlu_source,
                        "license": self.mmlu_license,
                    },
                    "truthfulqa": {"source": self.source, "license": self.license},
                },
            },
        )
        return AdapterResult(examples=examples, rejected=rejected, meta=metas, report=report)
