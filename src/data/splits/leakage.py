"""Leakage checks between splits. Fatal, never a warning.

Three checks, in increasing order of cost:

1. :func:`assert_no_group_overlap` — no ``group_id`` may appear in more than
   one split (FR-DATA-06). Returns the overlap count (always ``0`` when it
   returns); raises :class:`GroupOverlapError` otherwise.
2. :func:`assert_no_option_text_leak` — no identical normalized option text
   of at least ``MIN_OPTION_TEXT_TOKENS`` tokens may appear in more than one
   split. Failures raise :class:`OptionTextLeakageError` listing the count of
   violating texts.
3. :func:`assert_no_thirteen_gram_overlap` — no exact ``N_GRAM`` (13)
   consecutive token sequence may appear in more than one split. Failures
   raise :class:`NGramLeakageError` with per-split-pair counts.

The 13-gram check uses a double 64-bit Rabin-Karp rolling hash over
interned token ids, giving a 128-bit gram key. Token ids are assigned in
input order within the call, which is deterministic for a given input
sequence; only equality of gram keys is ever observed, so the counts are
independent of input permutation. False positives are negligible
(~n²/2¹²⁹ for n grams).
"""

from __future__ import annotations

from typing import Sequence

from src.data.core.schema import DecisionExample

from .errors import (
    GroupOverlapError,
    NGramLeakageError,
    OptionTextLeakageError,
    SplitInputError,
)

#: Minimum token count for an option text to participate in the leak check.
MIN_OPTION_TEXT_TOKENS = 40
#: Width of the exact n-gram checked across splits.
N_GRAM = 13

_SPLIT_NAMES = ("train", "dev", "test")
_MAX_SAMPLES = 5
_MASK64 = (1 << 64) - 1
_BASE_A = 1000003
_BASE_B = 1000033


_SPLIT_INDEX = {name: index for index, name in enumerate(_SPLIT_NAMES)}


def split_pair_key(left: str, right: str) -> str:
    """Canonical, order-independent key for a pair of split names."""
    left_index = _SPLIT_INDEX.get(left, 99)
    right_index = _SPLIT_INDEX.get(right, 99)
    first, second = (left, right) if left_index <= right_index else (right, left)
    return f"{first}/{second}"


def token_count(text: str) -> int:
    """Number of whitespace-separated tokens in ``text``."""
    return len(text.split())


def _validate_split_name(split: str) -> None:
    if split not in _SPLIT_NAMES:
        raise SplitInputError(
            f"example carries split {split!r}, expected one of {list(_SPLIT_NAMES)}",
            field="split",
        )


def assert_no_group_overlap(examples: Sequence[DecisionExample]) -> int:
    """Raise :class:`GroupOverlapError` if any ``group_id`` spans splits.

    Returns ``0`` when the splits are group-disjoint (the only success case).
    This is the assertion that fails the build on a leaky split.
    """
    splits_by_group: dict[str, set[str]] = {}
    for example in examples:
        _validate_split_name(example.split)
        splits_by_group.setdefault(example.group_id, set()).add(example.split)
    offenders = {
        group_id: sorted(splits)
        for group_id, splits in splits_by_group.items()
        if len(splits) > 1
    }
    if offenders:
        sample = tuple(sorted(offenders)[:_MAX_SAMPLES])
        detail_rows = sorted(offenders.items())[:_MAX_SAMPLES]
        detail = ", ".join(f"{g}->({'/'.join(s)})" for g, s in detail_rows)
        raise GroupOverlapError(
            f"{len(offenders)} group_id(s) appear in more than one split: {detail}",
            overlap=len(offenders),
            groups=sample,
        )
    return 0


def find_shared_option_texts(
    examples: Sequence[DecisionExample],
    *,
    min_tokens: int = MIN_OPTION_TEXT_TOKENS,
) -> dict[str, tuple[str, ...]]:
    """Map violating option texts to the splits they appear in.

    Only texts with at least ``min_tokens`` tokens are considered. Returns an
    empty dict when no text appears in more than one split.
    """
    if min_tokens < 1:
        raise SplitInputError(
            f"min_tokens must be a positive integer, got {min_tokens!r}",
            field="min_tokens",
        )
    splits_by_text: dict[str, set[str]] = {}
    for example in examples:
        _validate_split_name(example.split)
        for option in example.options:
            text = option.text
            if token_count(text) >= min_tokens:
                splits_by_text.setdefault(text, set()).add(example.split)
    return {
        text: tuple(sorted(splits))
        for text, splits in splits_by_text.items()
        if len(splits) > 1
    }


def assert_no_option_text_leak(
    examples: Sequence[DecisionExample],
    *,
    min_tokens: int = MIN_OPTION_TEXT_TOKENS,
) -> None:
    """Raise :class:`OptionTextLeakageError` on cross-split option-text leaks."""
    shared = find_shared_option_texts(examples, min_tokens=min_tokens)
    if not shared:
        return
    pair_counts: dict[str, int] = {}
    for splits in shared.values():
        for i, left in enumerate(splits):
            for right in splits[i + 1 :]:
                key = split_pair_key(left, right)
                pair_counts[key] = pair_counts.get(key, 0) + 1
    sample = tuple(sorted(shared)[:_MAX_SAMPLES])
    pair_detail = ", ".join(f"{pair}={count}" for pair, count in sorted(pair_counts.items()))
    raise OptionTextLeakageError(
        f"{len(shared)} identical normalized option text(s) of at least "
        f"{min_tokens} tokens appear in more than one split ({pair_detail}); "
        f"first offenders: {sample!r}",
        count=len(shared),
        texts=sample,
        pair_counts=pair_counts,
    )


def _document_tokens(example: DecisionExample) -> list[str]:
    tokens = example.goal.split()
    if example.state:
        tokens += example.state.split()
    for option in example.options:
        tokens += option.text.split()
    return tokens


def find_thirteen_gram_overlaps(
    examples: Sequence[DecisionExample],
    *,
    n_gram: int = N_GRAM,
) -> dict[str, int]:
    """Count shared exact n-grams per split pair.

    Returns a mapping like ``{"dev/test": 3}`` containing only pairs with at
    least one shared gram. The n-gram spans the normalized goal, state and
    option texts of each example.
    """
    if not isinstance(n_gram, int) or isinstance(n_gram, bool) or n_gram < 1:
        raise SplitInputError(
            f"n_gram must be a positive integer, got {n_gram!r}",
            field="n_gram",
        )
    intern: dict[str, int] = {}
    per_split: dict[str, set[int]] = {name: set() for name in _SPLIT_NAMES}
    mod = 1 << 64
    pow_a = pow(_BASE_A, n_gram, mod)
    pow_b = pow(_BASE_B, n_gram, mod)
    for example in examples:
        _validate_split_name(example.split)
        target = per_split[example.split]
        ids: list[int] = []
        for token in _document_tokens(example):
            token_id = intern.get(token)
            if token_id is None:
                token_id = len(intern) + 1
                intern[token] = token_id
            ids.append(token_id)
        hash_a = 0
        hash_b = 0
        for index, token_id in enumerate(ids):
            hash_a = (hash_a * _BASE_A + token_id) & _MASK64
            hash_b = (hash_b * _BASE_B + token_id) & _MASK64
            if index >= n_gram:
                lead = ids[index - n_gram]
                hash_a = (hash_a - lead * pow_a) & _MASK64
                hash_b = (hash_b - lead * pow_b) & _MASK64
            if index >= n_gram - 1:
                target.add((hash_a << 64) | hash_b)
    overlaps: dict[str, int] = {}
    for i, left in enumerate(_SPLIT_NAMES):
        for right in _SPLIT_NAMES[i + 1 :]:
            shared = len(per_split[left] & per_split[right])
            if shared:
                overlaps[split_pair_key(left, right)] = shared
    return overlaps


def assert_no_thirteen_gram_overlap(
    examples: Sequence[DecisionExample],
    *,
    n_gram: int = N_GRAM,
) -> None:
    """Raise :class:`NGramLeakageError` on cross-split n-gram leaks."""
    overlaps = find_thirteen_gram_overlaps(examples, n_gram=n_gram)
    if not overlaps:
        return
    total = sum(overlaps.values())
    detail = ", ".join(f"{pair}={count}" for pair, count in sorted(overlaps.items()))
    raise NGramLeakageError(
        f"exact {n_gram}-gram overlap between splits: {detail} (total={total})",
        counts=overlaps,
    )


def assert_no_leakage(examples: Sequence[DecisionExample]) -> None:
    """Run the option-text and n-gram checks; raise on any failure."""
    assert_no_option_text_leak(examples)
    assert_no_thirteen_gram_overlap(examples)
