"""Choice adapter: MMLU-style and MMLU-Pro rows → the ``choice`` family.

Scope note: MMLU-Pro (``TIGER-Lab/MMLU-Pro``) is **evaluation-only** — it is
admitted as the choice benchmark and never enters the training mixture. The
adapter exists so the eval set can be mapped at ingest, and so the same
code path serves ``cais/mmlu`` ``auxiliary_train`` (the admitted *training*
split, disjoint from MMLU test/dev by construction). Both shapes are the
same mapping with different declared ``source``/``license``/default ``split``.

Field mapping (MMLU-style rows):

- ``question`` → ``state`` (required, non-empty)
- ``choices`` (list of option texts) → ``options``; the source positional
  letters (``A``…``J``) are kept as the option labels, so a kept option
  retains its original source label even after reduction and shuffling
- ``answer`` → gold; an integer index into ``choices`` or a letter
  (``A``…``J``), case-insensitive
- ``goal`` is the fixed decision directive ``"Pick the correct option."``
- ``group_id`` comes from ``group_id``/``id`` when present, else a
  deterministic content-hash fallback (see ``base.derive_group_id``)
- ``split`` defaults to the adapter default (``train`` for ``cais/mmlu``
  auxiliary_train, ``test`` for MMLU-Pro); a row-level ``split`` overrides

MMLU-Pro 10→8 reduction. A row whose ``choices`` number more than eight
options is reduced **before** shuffling: the gold option and seven seeded
non-gold distractors are kept, the two seeded non-gold distractors are
dropped. The per-row reduction RNG seeds on
``sha256("rapport:mmlu-pro-reduce:v1" || reduction_seed || row_fingerprint)``,
so the reduction is deterministic for a recorded ``reduction_seed`` and is
independent of row order. Every reduced example carries
``meta.reduction == "mmlu-pro-10-to-8"``, ``meta.reduction_seed`` and
``meta.dropped_indices`` (the source indices that were dropped), and the
adapter reports the count in ``report["reduced_10_to_8"]`` (plus the seed and
the method under ``report["reduction_seed"]`` / ``report["reduction_method"]``).
Only non-gold indices are ever dropped — the gold option is always retained.

Option order is shuffled at ingest with the recorded seed (see ``base``);
``meta.original_order`` retains the emitted-position → kept-list-position
permutation (for unreduced rows the kept list *is* the source list), and
``meta.kept_indices`` maps kept positions back to source indices when a
reduction was applied. ``base.restore_original_order`` rebuilds the
pre-shuffle record. ``example_id`` is computed by the TASK-03 builder.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Mapping

from src.data.core.schema import MAX_OPTIONS

from .base import (
    ADAPTOR_BAD_INDEX,
    ADAPTOR_BAD_TYPE,
    ADAPTOR_EMPTY_STATE,
    ADAPTOR_MISSING_FIELD,
    ADAPTOR_NO_GOLD,
    ADAPTOR_NO_OPTIONS,
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
    reduce_seed,
    shuffle_order,
    split_for,
)

GOAL = "Pick the correct option."
REDUCTION_METHOD = "mmlu-pro-10-to-8"
REDUCTION_METHOD_GENERIC = "seeded-non-gold-removal-to-8"
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _parse_gold(value: Any, n_options: int) -> int:
    if isinstance(value, bool):
        raise _RecordFailure(ADAPTOR_NO_GOLD, f"gold answer {value!r} is a boolean, not an index")
    if isinstance(value, int):
        index = value
    elif isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            index = int(text)
        elif len(text) == 1 and text.upper() in _LETTERS[:n_options]:
            index = _LETTERS.index(text.upper())
        else:
            raise _RecordFailure(
                ADAPTOR_NO_GOLD,
                f"gold answer {value!r} is neither an index nor a letter "
                f"A..{_LETTERS[n_options - 1]}",
            )
    else:
        raise _RecordFailure(
            ADAPTOR_NO_GOLD, f"gold answer has type {type(value).__name__}; expected int or letter"
        )
    if not 0 <= index < n_options:
        raise _RecordFailure(
            ADAPTOR_BAD_INDEX, f"gold index {index} is outside 0..{n_options - 1}"
        )
    return index


def _reduce_to_eight(
    gold: int, *, n_options: int, reduction_seed: int, row_fingerprint: str
) -> tuple[list[int], list[int], int]:
    """Choose the kept source indices: gold + seven seeded non-gold distractors.

    Returns ``(kept_sorted, dropped, per_row_reduction_seed)``. Only non-gold
    indices are ever dropped; the gold option is always retained.
    """
    per_row_seed = reduce_seed(reduction_seed, row_fingerprint)
    non_gold = [i for i in range(n_options) if i != gold]
    rng = random.Random(per_row_seed)
    rng.shuffle(non_gold)
    keep_distractors = sorted(non_gold[: MAX_OPTIONS - 1])
    kept = sorted([gold, *keep_distractors])
    dropped = [i for i in range(n_options) if i not in kept]
    return kept, dropped, per_row_seed


class ChoiceAdapter:
    """MMLU-style 4+ option multiple choice.

    Defaults map ``cais/mmlu`` ``auxiliary_train`` (MIT) admitted for
    training. See the module docstring for the mapping and the reduction
    rule. Rows with more than eight options are reduced by seeded non-gold
    removal; rows with fewer than two options are rejected.
    """

    name = "choice"
    source = "cais/mmlu#auxiliary_train"
    license = "MIT"
    task_family = "choice"
    default_split = "train"

    def __init__(
        self,
        *,
        seed: int = DEFAULT_SEED,
        reduction_seed: int = DEFAULT_SEED,
        max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
        max_field_chars: int = DEFAULT_MAX_FIELD_CHARS,
        source: str | None = None,
        license: str | None = None,
        default_split: str | None = None,
    ) -> None:
        self.seed = seed
        self.reduction_seed = reduction_seed
        self.max_state_chars = max_state_chars
        self.max_field_chars = max_field_chars
        if source is not None:
            self.source = source
        if license is not None:
            self.license = license
        if default_split is not None:
            self.default_split = default_split

    # -- mapping --------------------------------------------------------------

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

        question = row.get("question")
        if not isinstance(question, str):
            if question is None:
                raise _RecordFailure(ADAPTOR_MISSING_FIELD, "field 'question' is absent")
            raise _RecordFailure(ADAPTOR_BAD_TYPE, "field 'question' must be a string")
        if not question.strip():
            raise _RecordFailure(ADAPTOR_EMPTY_STATE, "field 'question' is empty")
        state = length_cap(question.strip(), max_chars=self.max_state_chars, field="question")

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
                ADAPTOR_NO_OPTIONS, f"need at least 2 options, got {len(choices)}"
            )
        options_src: list[str] = []
        for i, item in enumerate(choices):
            if not isinstance(item, str):
                raise _RecordFailure(
                    ADAPTOR_BAD_TYPE,
                    f"option at index {i} must be a string, got {type(item).__name__}",
                )
            options_src.append(
                length_cap(item, max_chars=self.max_field_chars, field=f"choices[{i}]")
            )

        gold = _parse_gold(row.get("answer"), len(options_src))

        # -- reduction (before shuffling): more than 8 options → keep gold + 7 seeded non-gold
        reduction: str | None = None
        dropped: tuple[int, ...] = ()
        per_row_reduction_seed: int | None = None
        original_option_count = len(options_src)
        source_indices = list(range(original_option_count))
        if original_option_count > MAX_OPTIONS:
            source_indices, dropped_list, per_row_reduction_seed = _reduce_to_eight(
                gold,
                n_options=original_option_count,
                reduction_seed=self.reduction_seed,
                row_fingerprint=fingerprint,
            )
            dropped = tuple(dropped_list)
            options_src = [options_src[i] for i in source_indices]
            gold = source_indices.index(gold)
            reduction = (
                REDUCTION_METHOD
                if original_option_count == 10
                else REDUCTION_METHOD_GENERIC
            )

        order, per_row_seed = shuffle_order(
            len(options_src), master_seed=self.seed, row_fingerprint=fingerprint
        )
        emitted = apply_order(options_src, order)
        emitted_gold = order.index(gold)
        # Option labels are the source letters of the (kept) source indices,
        # shuffled along with the options.
        source_labels = [_LETTERS[i] for i in source_indices]
        emitted_labels = apply_order(source_labels, order)
        # meta.original_order refers to the kept list (a true permutation of
        # range(len(example.options))); meta.kept_indices maps kept positions
        # back to source indices when a reduction was applied.
        kept_order = tuple(order)

        raw = {
            "task_family": "choice",
            "goal": GOAL,
            "state": state,
            "options": [{"text": t, "label": label} for t, label in zip(emitted, emitted_labels)],
            "source": self.source,
            "license": self.license,
            "group_id": group_id,
            "split": split,
            "chosen_index": emitted_gold,
        }
        meta = ExampleMeta(
            original_order=kept_order,
            shuffle_seed=per_row_seed,
            reduction=reduction,
            reduction_seed=per_row_reduction_seed,
            dropped_indices=dropped,
            kept_indices=tuple(source_indices) if reduction is not None else None,
            origin="mmlu-pro" if self.default_split == "test" else "mmlu-auxiliary-train",
        )
        return raw, meta

    def to_decision_example(self, rows: Iterable[Mapping[str, Any]]) -> AdapterResult:
        examples, metas, rejected = [], [], []
        reduced_10 = 0
        reduced_other = 0
        rows_seen = 0
        for record_index, row in enumerate(rows):
            rows_seen += 1
            try:
                raw, meta = self._convert_row(record_index, row)
                examples.append(finalize(raw))
                metas.append(meta)
                if meta.reduction == REDUCTION_METHOD:
                    reduced_10 += 1
                elif meta.reduction is not None:
                    reduced_other += 1
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
                "reduction_method": REDUCTION_METHOD,
                "reduction_seed": self.reduction_seed,
                # Total >8-option reductions (subset accounting below).
                "reduced_over_8_to_8": reduced_10 + reduced_other,
                "reduced_10_to_8": reduced_10,
                "reduced_other_to_8": reduced_other,
                "max_options": MAX_OPTIONS,
            },
        )
        return AdapterResult(examples=examples, rejected=rejected, meta=metas, report=report)


class MmluProAdapter(ChoiceAdapter):
    """``TIGER-Lab/MMLU-Pro`` — evaluation-only choice benchmark.

    Same mapping as :class:`ChoiceAdapter`; declared source is
    ``TIGER-Lab/MMLU-Pro``, license MIT, default split ``test``. Never admitted
    for training; exists so the 12,102-row eval set maps into the unified
    schema with the 10→8 reduction applied and reported.
    """

    name = "mmlu-pro"
    source = "TIGER-Lab/MMLU-Pro"
    license = "MIT"
    default_split = "test"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("source", self.source)
        kwargs.setdefault("license", self.license)
        kwargs.setdefault("default_split", self.default_split)
        super().__init__(**kwargs)
