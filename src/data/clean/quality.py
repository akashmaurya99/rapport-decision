"""Quality rules Q1–Q6: named, counted, structural — never difficulty-based.

Each rule is a pure predicate on a :class:`DecisionExample` (``True`` means
"violates"). The :class:`QualityFilter` runs the predicates in the order the strategy
lists them, assigns every rejected row to exactly one rule — the
first that fires — and increments that rule's counter in a shared
:class:`RejectionReport`. Because each removed row is charged to exactly one
rule, ``report.total_rejected == len(input) - len(kept)`` always holds, and
:meth:`RejectionReport.verify` fails loudly if it ever stops holding.

These rules are deliberately structural. No rule invokes a learned model,
no rule reads difficulty, embedding agreement, baseline correctness,
perplexity, label confidence, or any other signal of how *hard* an example
is — a hard example for being hard is never deleted here. Difficulty-based
selection lives in a later pipeline stage and applies to training-time
sampling only, not to cleaning.

Token counting note: at this layer "tokens" means whitespace-delimited words
(``str.split``), matching the cost profile of a data-cleaning pass. The
512-token budget in Q5 is therefore approximate; the tokenizer enforces the
exact budget at encode time (TASK-04). Q1's "<3-token option" is applied
literally as written in the dataset strategy, including to single-word
options — *except* for the families whose option lists are a fixed
polarity/attribute vocabulary (see :data:`FIXED_OPTION_VOCABULARIES`). Those
vocabularies are produced by the adapters, not authored by a human, so a
one-token "Yes"/"No"/"safe"/"unsafe"/"helpfulness" is a complete, legitimate
option. Short options outside the family's declared vocabulary are still
rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from src.data.core.schema import DecisionExample

from .report import RejectionReport

#: Q5 rejects rows whose whitespace-token total exceeds this budget.
DEFAULT_MAX_TOKENS = 512
#: Minimum whitespace-token count an option must carry (Q1).
MIN_OPTION_TOKENS = 3

#: Families with adapter-owned fixed option vocabularies. Q1 is skipped for
#: an option whose (case-folded) text is declared here: these are structural
#: polarity/attribute labels, never free-text author options. Adding a family
#: here is an explicit claim that its adapter owns the option vocabulary.
FIXED_OPTION_VOCABULARIES: dict[str, frozenset[str]] = {
    "scoring": frozenset({"helpfulness", "correctness", "coherence"}),
    "probability": frozenset({"yes", "no"}),
    "abstention": frozenset({"answerable", "unanswerable"}),
    "risk": frozenset({"safe", "unsafe"}),
}

# Q2 residue patterns: an HTML-ish tag, an entity reference, or markdown bold.
_TAG_RESIDUE = re.compile(r"<[^>\n]{0,200}>")
_ENTITY_RESIDUE = re.compile(r"&(?:[a-zA-Z]{2,10}|#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6});")
_BOLD_RESIDUE = re.compile(r"\*\*")

#: Placeholder strings Q6 rejects as stand-ins for a real option, matched
#: case-insensitively against the normalized option text.
PLACEHOLDER_OPTIONS = frozenset(
    {"n/a", "na", "tbd", "tba", "todo", "xxx", "unknown", "placeholder"}
)
#: Real options that must never be rejected by Q6 — "none of the above" style
#: answers are genuine options, as the dataset strategy explicitly records.
REAL_OPTIONS = frozenset(
    {
        "none of the above",
        "none of these",
        "all of the above",
        "all of these",
        "i don't know",
        "cannot be determined",
    }
)


def _iter_text_fields(example: DecisionExample) -> list[str]:
    return [example.goal, example.state, *(option.text for option in example.options)]


def q1_short_option(example: DecisionExample) -> bool:
    """Q1 — an option is empty or carries fewer than 3 whitespace tokens.

    Family-aware: fixed polarity/attribute vocabularies declared in
    :data:`FIXED_OPTION_VOCABULARIES` are exempt, because the adapter (not a
    human author) owns those exact option strings. A short option that is
    *not* part of the family's declared vocabulary is still rejected.
    """
    exempt = FIXED_OPTION_VOCABULARIES.get(example.task_family, frozenset())

    def ascii_fold(text: str) -> str:
        # Unicode casefold maps long-s (ſ) to ASCII s and would admit option
        # spellings no adapter emits. Only ASCII case is normalized here.
        return "".join(char.lower() if "A" <= char <= "Z" else char for char in text.strip())

    return any(
        len(option.text.split()) < MIN_OPTION_TOKENS
        and ascii_fold(option.text) not in exempt
        for option in example.options
    )


def q2_markup_residue(example: DecisionExample) -> bool:
    """Q2 — any text field still carries markup residue (<br>, &amp;, **)."""
    for text in _iter_text_fields(example):
        if (
            _TAG_RESIDUE.search(text)
            or _ENTITY_RESIDUE.search(text)
            or _BOLD_RESIDUE.search(text)
        ):
            return True
    return False


def q3_duplicate_option(example: DecisionExample) -> bool:
    """Q3 — two options are identical after normalization."""
    texts = [option.text for option in example.options]
    return len(set(texts)) != len(texts)


def q4_chosen_index_out_of_range(example: DecisionExample) -> bool:
    """Q4 — chosen_index is present but not an integer within option range.

    A missing ``chosen_index`` is not out of range and is not charged to Q4;
    whether a family requires the field is the contract's question, not this
    rule's.
    """
    chosen = example.chosen_index
    if chosen is None:
        return False
    if isinstance(chosen, bool) or not isinstance(chosen, int):
        return True
    return not 0 <= chosen < len(example.options)


def q5_too_long(
    example: DecisionExample, max_tokens: int = DEFAULT_MAX_TOKENS
) -> bool:
    """Q5 — goal + state + options exceed the 512 whitespace-token budget."""
    total = sum(len(text.split()) for text in _iter_text_fields(example))
    return total > max_tokens


def q6_punctuation_only_or_placeholder(example: DecisionExample) -> bool:
    """Q6 — an option is only punctuation/symbols or a placeholder.

    "none of the above" and its siblings are real options and are never
    rejected here (see :data:`REAL_OPTIONS`).
    """
    for option in example.options:
        text = option.text.strip()
        folded = text.casefold()
        if folded in REAL_OPTIONS:
            continue
        if not any(character.isalnum() for character in text):
            return True  # only punctuation / symbols / no alphanumeric at all
        if folded in PLACEHOLDER_OPTIONS:
            return True
    return False


#: Rules in evaluation order; names match the dataset strategy's Q1..Q6.
RULES: tuple[tuple[str, Callable[[DecisionExample], bool]], ...] = (
    ("Q1", q1_short_option),
    ("Q2", q2_markup_residue),
    ("Q3", q3_duplicate_option),
    ("Q4", q4_chosen_index_out_of_range),
    ("Q5", q5_too_long),
    ("Q6", q6_punctuation_only_or_placeholder),
)


@dataclass(frozen=True)
class QualityResult:
    """Output of :class:`QualityFilter`."""

    kept: tuple[DecisionExample, ...]
    report: RejectionReport

    @property
    def rejected_count(self) -> int:
        return self.report.total_rejected


class QualityFilter:
    """Apply Q1..Q6 in order; first failing rule claims each rejected row."""

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        self.max_tokens = max_tokens

    def _rules(self) -> tuple[tuple[str, Callable[[DecisionExample], bool]], ...]:
        budget = self.max_tokens
        if budget == DEFAULT_MAX_TOKENS:
            return RULES
        return tuple(
            (name, (lambda example: q5_too_long(example, budget)) if name == "Q5" else predicate)
            for name, predicate in RULES
        )

    def run(self, examples: Sequence[DecisionExample]) -> QualityResult:
        report = RejectionReport()
        kept: list[DecisionExample] = []
        for example in examples:
            rejected_by: str | None = None
            for name, predicate in self._rules():
                if predicate(example):
                    rejected_by = name
                    break
            if rejected_by is None:
                kept.append(example)
            else:
                report.increment(rejected_by, example.example_id)
        report.verify(len(examples) - len(kept))
        return QualityResult(kept=tuple(kept), report=report)
