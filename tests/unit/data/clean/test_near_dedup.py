"""Near dedup: MinHash over 5-gram shingles at Jaccard ≥ 0.9 (threshold recorded)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.clean import (
    DEFAULT_NEAR_THRESHOLD,
    NearDedup,
    canonical_document,
    estimate_jaccard,
    jaccard_index,
    shingles,
)

_LONG_STATE = (
    "The service holds customer data across three regions and the rollout window "
    "is four hours with on call staff monitoring the error budget closely "
) * 3
_NEAR_STATE = _LONG_STATE.replace("three regions", "four regions")
_FAR_STATE = (
    "An entirely different document about a cooking recipe for a sourdough loaf "
    "with a preferment and a long overnight cold proof in the refrigerator "
) * 3


def _near_pair(example_factory):
    a = example_factory(state=_LONG_STATE, group_id="g-near-a")
    b = example_factory(state=_NEAR_STATE, group_id="g-near-b")
    return a, b


def test_near_dedup_removes_near_duplicate_and_records_threshold(example_factory):
    a, b = _near_pair(example_factory)
    result = NearDedup().run([a, b])

    assert result.near_removed == 1
    assert result.removed_ids == (b.example_id,)
    assert result.kept == (a,)
    assert result.threshold == pytest.approx(0.9)
    assert result.threshold == DEFAULT_NEAR_THRESHOLD
    assert result.num_perm == 128


def test_near_dedup_keeps_dissimilar_rows(example_factory):
    a = example_factory(state=_LONG_STATE, group_id="g-a")
    far = example_factory(state=_FAR_STATE, group_id="g-far")

    result = NearDedup().run([a, far])

    assert result.near_removed == 0
    assert len(result.kept) == 2


def test_near_dedup_is_option_order_insensitive(example_factory):
    a = example_factory(group_id="g-order-a")
    # Same options in reverse order, chosen_index adjusted to track the same
    # gold text; shingles are built from the sorted option texts.
    reversed_options = (
        "Delay the rollout by two full weeks",
        "Roll out to all traffic immediately",
        "Roll out to ten percent of traffic first",
    )
    b = example_factory(options=reversed_options, chosen_index=2, group_id="g-order-b")

    assert canonical_document(a) == canonical_document(b)
    result = NearDedup().run([a, b])
    assert result.near_removed == 1


def test_near_dedup_never_merges_rows_with_different_gold_labels(example_factory):
    a = example_factory(chosen_index=0, group_id="g-label-a")
    b = example_factory(chosen_index=1, group_id="g-label-b")  # identical content, other gold

    result = NearDedup().run([a, b])

    # Same content would be Jaccard 1.0, but different chosen_index means this
    # is a contradiction candidate, not a duplicate. Both rows must survive.
    assert result.near_removed == 0
    assert len(result.kept) == 2


def test_near_dedup_is_deterministic(example_factory):
    rows = [
        example_factory(state=_LONG_STATE, group_id="g-det-a"),
        example_factory(state=_NEAR_STATE, group_id="g-det-b"),
        example_factory(state=_FAR_STATE, group_id="g-det-c"),
    ]
    first = NearDedup().run(rows)
    second = NearDedup().run(rows)
    assert first.removed_ids == second.removed_ids
    assert first.candidate_pairs == second.candidate_pairs


def test_minhash_estimate_tracks_true_jaccard():
    base = set(shingles(_LONG_STATE))
    near = set(shingles(_NEAR_STATE))
    assert jaccard_index(base, near) > 0.8
    # For identical sets the estimate must be exactly 1.0.
    assert estimate_jaccard(
        NearDedup()._hasher.signature(base), NearDedup()._hasher.signature(base)
    ) == pytest.approx(1.0)


def test_estimate_jaccard_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        estimate_jaccard(np.zeros(4, dtype=np.uint64), np.zeros(5, dtype=np.uint64))


def test_shingles_short_text_is_single_shingle():
    assert shingles("abc") == {"abc"}
    assert shingles("") == set()
    assert shingles("abcdef", size=3) == {"abc", "bcd", "cde", "def"}
