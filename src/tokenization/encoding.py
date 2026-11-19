"""Budget-aware encoding of ``DecisionExample`` records and length bucketing.

Joint sequence layout (the RAPPORT input format, architecture-spec §5)::

    <task>{task_family}<goal>{goal}<state>{state}<opt>{option_1}</opt><opt>…</opt>

The state runs until the first ``<opt>`` marker; the five special tokens in
``src.tokenization.tokenizer.SPECIAL_TOKENS`` are the complete marker set —
there is deliberately no ``</state>`` (contract TASK-04 / FR-TOK-01).

Budget and truncation (FR-TOK-02)
----------------------------------

``encode_example(example, budget=512)`` assembles the joint sequence and
guarantees ``len(input_ids) <= budget``. When the sequence exceeds the
budget the **state is truncated from the end**; the goal prefix and the
option suffix (every ``<opt>``/``</opt>`` marker and the option texts, in
order) are kept intact. Two documented fallbacks, only reachable when the
state alone cannot absorb the overflow:

1. goal truncated from the end (state becomes empty);
2. if the option block alone still exceeds the budget, the option block is
   truncated from the end (trailing markers may be lost). Pathological;
   flagged like every other truncation.

Every truncation sets ``truncated=True`` on the returned record — the input
:class:`~src.data.core.schema.DecisionExample` is frozen and is **never**
mutated; a new :class:`EncodedExample` is returned instead. Truncations are
also counted **per source** in a module-level counter. ``truncation_counts()``
returns a copy, ``reset_truncation_counts()`` clears it (tests use this for
isolation), and when ``encode_example`` is given a ``manifest`` mapping the
counter snapshot is written into it under ``manifest["truncations_per_source"]``
together with ``manifest["truncation_policy"]``. The run-manifest writer lives
in ``src/config/manifest.py`` (TASK-01), which this task does not own; the
caller merges these keys, or passes the ``manifest`` mapping through.

Length bucketing (FR-TOK-02)
----------------------------

``bucket_examples`` pads every example to one of at most three fixed widths
(derived from ``budget``: ``budget//4``, ``budget//2``, ``budget`` — 128 /
256 / 512 at the default budget), so a run emits **at most three distinct
batch shapes**. Padding policy: right-pad with ``[PAD]`` (id 0) up to the
bucket width, ``attention_mask=0`` beyond the real token length. Each batch
records the padding and truncation policies verbatim. Examples keep their
input order within a bucket (stable), buckets are emitted short → medium →
long, and the last (partial) batch of a bucket may be smaller than
``batch_size`` — documented, never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, MutableMapping, Sequence

from src.data.core.schema import DecisionExample, Option

from .errors import BucketOverflowError, BudgetTooSmallError
from .tokenizer import (
    DEFAULT_TOKENIZER,
    PAD_ID,
    STATE_ID,
    RapportTokenizer,
)

DEFAULT_BUDGET = 512
PADDING_POLICY = (
    "right-pad with [PAD] (id 0) to the bucket width; "
    "attention_mask=0 beyond the real token length"
)
TRUNCATION_POLICY = (
    "state truncated from the end to fit the 512-token budget; goal prefix and "
    "option suffix kept intact; goal truncated from the end only when the option "
    "block alone exceeds the budget; every truncation is flagged on the encoded "
    "example and counted per source in truncations_per_source"
)
TRUNCATION_MANIFEST_KEY = "truncations_per_source"
TRUNCATION_POLICY_MANIFEST_KEY = "truncation_policy"

# Module-level per-source truncation counter; surfaced through the manifest.
TRUNCATION_COUNTS: dict[str, int] = {}


def truncation_counts() -> dict[str, int]:
    """Return a copy of the per-source truncation counts (for the manifest)."""
    return dict(TRUNCATION_COUNTS)


def reset_truncation_counts() -> None:
    """Clear the per-source truncation counter (used by tests)."""
    TRUNCATION_COUNTS.clear()


@dataclass(frozen=True)
class EncodedExample:
    """An encoded example. Never mutates the frozen ``DecisionExample``."""

    example: DecisionExample
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    truncated: bool
    source: str
    truncation_policy: str


def _options_suffix(options: Sequence[Option]) -> str:
    return "".join(f"<opt>{option.text}</opt>" for option in options)


def encode_example(
    example: DecisionExample,
    budget: int = DEFAULT_BUDGET,
    manifest: MutableMapping[str, Any] | None = None,
    tokenizer: RapportTokenizer | None = None,
) -> EncodedExample:
    """Build the joint sequence for ``example`` under ``budget`` tokens.

    Truncates the state from the end when the sequence overflows; sets
    ``truncated`` on the returned copy (the input example is untouched) and
    increments the per-source truncation counter for ``example.source``. If
    ``manifest`` is a mutable mapping, it is updated in place with
    ``truncations_per_source`` (a snapshot of the counter) and
    ``truncation_policy``.
    """
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise BudgetTooSmallError("budget must be a positive integer", field="budget")
    tok = tokenizer if tokenizer is not None else DEFAULT_TOKENIZER

    head = tok.encode(f"<task>{example.task_family}<goal>")
    goal_ids = tok.encode(example.goal)
    state_ids = tok.encode(example.state)
    options_ids = tok.encode(_options_suffix(example.options))

    truncated = False
    total = len(head) + len(goal_ids) + 1 + len(state_ids) + len(options_ids)
    if total > budget:
        truncated = True
        state_room = budget - (len(head) + len(goal_ids) + 1 + len(options_ids))
        if state_room >= 0:
            state_ids = state_ids[:state_room]
        else:
            # Fallback 1: the goal gives way; the option tail stays intact.
            state_ids = []
            goal_room = budget - (len(head) + 1 + len(options_ids))
            if goal_room >= 0:
                goal_ids = goal_ids[:goal_room]
            else:
                # Fallback 2 (pathological): the option block alone exceeds
                # the budget; it is truncated from the end and flagged.
                goal_ids = []
                options_room = budget - (len(head) + 1)
                options_ids = options_ids[:options_room] if options_room >= 0 else []
                if options_room < 0:
                    head = head[: max(budget - 1, 0)]

    ids = [*head, *goal_ids, STATE_ID, *state_ids, *options_ids]
    if len(ids) > budget:  # absolute guarantee; only the pathological path reaches this
        ids = ids[:budget]

    if truncated:
        TRUNCATION_COUNTS[example.source] = TRUNCATION_COUNTS.get(example.source, 0) + 1
        if manifest is not None:
            manifest[TRUNCATION_MANIFEST_KEY] = truncation_counts()
            manifest[TRUNCATION_POLICY_MANIFEST_KEY] = TRUNCATION_POLICY
    elif manifest is not None:
        manifest.setdefault(TRUNCATION_POLICY_MANIFEST_KEY, TRUNCATION_POLICY)

    return EncodedExample(
        example=example,
        input_ids=tuple(ids),
        attention_mask=tuple(1 for _ in ids),
        truncated=truncated,
        source=example.source,
        truncation_policy=TRUNCATION_POLICY,
    )


@dataclass(frozen=True)
class BucketedBatch:
    """One fixed-shape batch; ``padded_length`` is the batch's tensor width."""

    bucket: str
    padded_length: int
    batch_size: int
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    examples: tuple[DecisionExample, ...]
    truncated: tuple[bool, ...]
    padding_policy: str
    truncation_policy: str


_BUCKET_NAMES = ("short", "medium", "long")


def default_bucket_widths(budget: int = DEFAULT_BUDGET) -> tuple[int, ...]:
    """The three bucket widths for ``budget``: ``budget//4``, ``budget//2``, ``budget``."""
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise BudgetTooSmallError("budget must be a positive integer", field="budget")
    widths = sorted({max(1, budget // 4), max(1, budget // 2), budget})
    return tuple(widths)


def bucket_examples(
    examples: Iterable[DecisionExample | EncodedExample],
    *,
    budget: int = DEFAULT_BUDGET,
    batch_size: int = 32,
    bucket_widths: Sequence[int] | None = None,
    tokenizer: RapportTokenizer | None = None,
) -> list[BucketedBatch]:
    """Bucket ``examples`` into at most three padded batch shapes.

    Accepts ``DecisionExample`` records (encoded with ``budget``; truncations
    are counted per source as usual) or pre-encoded :class:`EncodedExample`
    records (used as-is, no counter side effects). Input order is preserved
    within each bucket; buckets are emitted short → medium → long; each
    example is padded with ``[PAD]`` to its bucket width and its
    ``attention_mask`` is 0 beyond the real token length.

    Raises :class:`BucketOverflowError` if an encoded example is longer than
    the largest bucket width.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise BudgetTooSmallError("batch_size must be a positive integer", field="batch_size")
    widths = tuple(bucket_widths) if bucket_widths is not None else default_bucket_widths(budget)
    if not widths:
        raise BudgetTooSmallError("bucket_widths must not be empty", field="bucket_widths")
    if len(widths) > 3:
        raise BudgetTooSmallError(
            "at most three bucket widths are allowed (FR-TOK-02)", field="bucket_widths"
        )
    if tuple(sorted(widths)) != tuple(widths):
        raise BudgetTooSmallError("bucket_widths must be sorted ascending", field="bucket_widths")
    if any(not isinstance(w, int) or isinstance(w, bool) or w < 1 for w in widths):
        raise BudgetTooSmallError("bucket_widths must be positive integers", field="bucket_widths")

    encoded: list[EncodedExample] = []
    for item in examples:
        if isinstance(item, EncodedExample):
            encoded.append(item)
        else:
            encoded.append(encode_example(item, budget=budget, tokenizer=tokenizer))

    max_width = widths[-1]
    groups: list[list[EncodedExample]] = [[] for _ in widths]
    for item in encoded:
        length = len(item.input_ids)
        if length > max_width:
            raise BucketOverflowError(
                f"encoded example of length {length} exceeds the largest bucket "
                f"width {max_width}",
                field="examples",
            )
        for index, width in enumerate(widths):
            if length <= width:
                groups[index].append(item)
                break

    batches: list[BucketedBatch] = []
    for index, (width, members) in enumerate(zip(widths, groups)):
        name = _BUCKET_NAMES[index] if index < len(_BUCKET_NAMES) else f"bucket-{index}"
        for start in range(0, len(members), batch_size):
            chunk = members[start : start + batch_size]
            padded_ids: list[tuple[int, ...]] = []
            padded_mask: list[tuple[int, ...]] = []
            for item in chunk:
                real = len(item.input_ids)
                pad = width - real
                padded_ids.append((*item.input_ids, *([PAD_ID] * pad)))
                padded_mask.append((*item.attention_mask, *([0] * pad)))
            batches.append(
                BucketedBatch(
                    bucket=name,
                    padded_length=width,
                    batch_size=len(chunk),
                    input_ids=tuple(padded_ids),
                    attention_mask=tuple(padded_mask),
                    examples=tuple(item.example for item in chunk),
                    truncated=tuple(item.truncated for item in chunk),
                    padding_policy=PADDING_POLICY,
                    truncation_policy=TRUNCATION_POLICY,
                )
            )
    return batches
