"""Preference adapter: HH-RLHF-style chosen/rejected pairs → ``preference``.

Field mapping (Anthropic HH-RLHF rows):

- ``chosen`` / ``rejected`` → the two candidate responses (required)
- ``prompt`` → ``state`` when present (the conversation up to the final
  assistant turn)
- otherwise the shared human prefix of the two transcripts is extracted: HH-RLHF
  rows carry the *full* transcript in both fields, sharing every turn up to the
  last ``assistant`` turn. When the shared prefix cannot be established, the
  full transcripts become the options and ``state`` is empty — which the
  contract explicitly allows for the preference family
- ``goal`` is the fixed directive ``"Pick the response that better satisfies
  the request."`` (the spec's ``<goal>`` slot)
- ``group_id`` comes from ``group_id``/``id`` when present, else a
  deterministic content-hash fallback shared by both emissions of a pair

Both orders are emitted — deterministically, per the family's de-biasing
augmentation byte: emission A keeps ``[chosen, rejected]`` with
``chosen_index == 0``, emission B reverses to ``[rejected, chosen]`` with
``chosen_index == 1``. Both emissions of one source row share ``group_id``
(so a pair can never straddle a split). The random ingest shuffle is
deliberately not applied on top of this deterministic two-permutation
emission; ``meta.original_order`` records the (trivial) permutation per
emission and ``meta.shuffle_seed`` is ``None``.

Extremely long transcripts are truncated at a word boundary to
``max_state_chars`` (default 8000) and flagged via ``meta.truncated``; a
field longer than ``max_field_chars`` (default 100000) is rejected with
``ADAPTOR_TOO_LONG``. Truncated options are never silently accepted: an empty
option text after normalization is rejected through the contract.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .base import (
    ADAPTOR_BAD_TYPE,
    ADAPTOR_MISSING_FIELD,
    DEFAULT_MAX_FIELD_CHARS,
    DEFAULT_MAX_STATE_CHARS,
    DEFAULT_SEED,
    AdapterResult,
    ExampleMeta,
    RejectedRecord,
    _RecordFailure,
    base_report,
    cap_text,
    finalize,
    fingerprint_row,
    group_id_for,
    length_cap,
    split_for,
)

GOAL = "Pick the response that better satisfies the request."


def _split_transcript(transcript: str) -> tuple[str, str] | None:
    """Split ``<shared prefix>\\n\\nAssistant: <response>`` off the last turn."""
    for marker in ("\n\nAssistant: ", "\n\nassistant: "):
        idx = transcript.rfind(marker)
        if idx != -1:
            return transcript[:idx], transcript[idx + len(marker) :].strip()
    return None


class PreferenceAdapter:
    """Anthropic HH-RLHF (MIT) chosen/rejected pairs; default split ``train``.

    Emits two examples per pair (both orders); see the module docstring.
    """

    name = "preference"
    source = "Anthropic/hh-rlhf"
    license = "MIT"
    task_family = "preference"
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
    ) -> tuple[list[dict[str, Any]], list[ExampleMeta], bool]:
        if not isinstance(row, Mapping):
            raise _RecordFailure(
                ADAPTOR_BAD_TYPE, f"row must be a mapping, got {type(row).__name__}"
            )
        fingerprint = fingerprint_row(row)
        split = split_for(row, self.default_split)
        group_id = group_id_for(row, self.source, fingerprint)

        chosen = row.get("chosen")
        rejected = row.get("rejected")
        for key, value in (("chosen", chosen), ("rejected", rejected)):
            if value is None:
                raise _RecordFailure(ADAPTOR_MISSING_FIELD, f"field {key!r} is absent")
            if not isinstance(value, str):
                raise _RecordFailure(
                    ADAPTOR_BAD_TYPE,
                    f"field {key!r} must be a string, got {type(value).__name__}",
                )
        assert isinstance(chosen, str) and isinstance(rejected, str)
        chosen_text = length_cap(chosen, max_chars=self.max_field_chars, field="chosen")
        rejected_text = length_cap(rejected, max_chars=self.max_field_chars, field="rejected")

        prompt = row.get("prompt")
        truncated = False
        if prompt is not None:
            if not isinstance(prompt, str):
                raise _RecordFailure(ADAPTOR_BAD_TYPE, "field 'prompt' must be a string")
            state, truncated = cap_text(
                prompt.strip(), max_chars=self.max_state_chars, field="prompt"
            )
            option_chosen, option_rejected = chosen_text, rejected_text
        else:
            split_chosen = _split_transcript(chosen_text)
            split_rejected = _split_transcript(rejected_text)
            if (
                split_chosen is not None
                and split_rejected is not None
                and split_chosen[0] == split_rejected[0]
            ):
                state, truncated = cap_text(
                    split_chosen[0].strip(),
                    max_chars=self.max_state_chars,
                    field="state",
                )
                option_chosen, option_rejected = split_chosen[1], split_rejected[1]
            else:
                # Unparseable transcripts: keep the full transcripts as the
                # options; the contract allows an empty state for preference.
                state = ""
                option_chosen, option_rejected = chosen_text, rejected_text

        raws: list[dict[str, Any]] = []
        metas: list[ExampleMeta] = []
        for emission, (first, second, chosen_index) in enumerate(
            (
                (option_chosen, option_rejected, 0),  # order A: chosen at index 0
                (option_rejected, option_chosen, 1),  # order B: chosen at index 1
            )
        ):
            raws.append(
                {
                    "task_family": "preference",
                    "goal": GOAL,
                    "state": state,
                    "options": [{"text": first}, {"text": second}],
                    "source": self.source,
                    "license": self.license,
                    "group_id": group_id,
                    "split": split,
                    "chosen_index": chosen_index,
                }
            )
            metas.append(
                ExampleMeta(
                    original_order=(0, 1) if emission == 0 else (1, 0),
                    shuffle_seed=None,
                    truncated=truncated,
                    origin="hh-rlhf",
                )
            )
        return raws, metas, truncated

    def to_decision_example(self, rows: Iterable[Mapping[str, Any]]) -> AdapterResult:
        examples, metas, rejected = [], [], []
        rows_seen = 0
        truncated_rows = 0
        for record_index, row in enumerate(rows):
            rows_seen += 1
            try:
                raws, row_metas, truncated = self._convert_row(record_index, row)
                built = []
                for raw, meta in zip(raws, row_metas):
                    built.append((finalize(raw), meta))
                for example, meta in built:
                    examples.append(example)
                    metas.append(meta)
                if truncated:
                    truncated_rows += 1
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
                "both_orders_emitted": len(examples),
                "pairs_converted": len(examples) // 2,
                "truncated_rows": truncated_rows,
                "shuffle": "deterministic-both-orders",
            },
        )
        return AdapterResult(examples=examples, rejected=rejected, meta=metas, report=report)
