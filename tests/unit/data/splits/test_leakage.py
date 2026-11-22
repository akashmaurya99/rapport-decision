"""Tests for the leakage checks: group overlap, option-text and 13-gram leaks."""

from __future__ import annotations

import dataclasses

import pytest
from splits_test_support import make_example

from src.data.splits import (
    GroupOverlapError,
    NGramLeakageError,
    OptionTextLeakageError,
    SplitInputError,
    assert_no_group_overlap,
    assert_no_leakage,
    assert_no_option_text_leak,
    assert_no_thirteen_gram_overlap,
    find_shared_option_texts,
    find_thirteen_gram_overlaps,
    split,
    split_pair_key,
    token_count,
)

SHARED_SPAN = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike".split()
)



def _n_token_text(n: int) -> str:
    return " ".join(f"tok{index}" for index in range(n))


def _assign(examples, split_name):
    return [dataclasses.replace(example, split=split_name) for example in examples]


class TestSplitPairKey:
    def test_order_independent_and_train_dev_test_ordered(self):
        assert split_pair_key("test", "train") == "train/test"
        assert split_pair_key("dev", "train") == "train/dev"
        assert split_pair_key("test", "dev") == "dev/test"


class TestGroupOverlap:
    def test_disjoint_splits_return_zero(self):
        examples = _assign(
            [make_example(uid=9001, group_id="g-a"), make_example(uid=9002, group_id="g-b")],
            "train",
        )
        assert assert_no_group_overlap(examples) == 0

    def test_shared_group_across_splits_is_a_fatal_typed_error(self):
        train = _assign([make_example(uid=9101, group_id="g-shared")], "train")
        dev = _assign([make_example(uid=9102, group_id="g-shared")], "dev")
        with pytest.raises(GroupOverlapError) as excinfo:
            assert_no_group_overlap(train + dev)
        error = excinfo.value
        assert error.code == "SPLIT_GROUP_OVERLAP"
        assert error.overlap == 1
        assert error.groups == ("g-shared",)
        assert "1 group_id" in str(error)

    def test_three_way_group_counts_once(self):
        train = _assign([make_example(uid=9201, group_id="g-three")], "train")
        dev = _assign([make_example(uid=9202, group_id="g-three")], "dev")
        test = _assign([make_example(uid=9203, group_id="g-three")], "test")
        with pytest.raises(GroupOverlapError) as excinfo:
            assert_no_group_overlap(train + dev + test)
        assert excinfo.value.overlap == 1

    def test_multiple_overlapping_groups_report_the_count(self):
        examples = _assign(
            [
                make_example(uid=9301, group_id="g-one"),
                make_example(uid=9302, group_id="g-one"),
                make_example(uid=9303, group_id="g-two"),
                make_example(uid=9304, group_id="g-two"),
            ],
            "train",
        )
        examples[1] = dataclasses.replace(examples[1], split="dev")
        examples[2] = dataclasses.replace(examples[2], split="test")
        with pytest.raises(GroupOverlapError) as excinfo:
            assert_no_group_overlap(examples)
        assert excinfo.value.overlap == 2
        assert excinfo.value.groups == ("g-one", "g-two")

    def test_unknown_split_name_is_rejected(self):
        with pytest.raises(SplitInputError):
            assert_no_group_overlap(_assign([make_example(uid=9400)], "ood"))


class TestOptionTextLeak:
    def test_identical_forty_token_option_across_splits_is_fatal(self):
        shared = _n_token_text(40)
        train = _assign(
            [make_example(
                uid=9501, group_id="g-1",
                options=(shared, "train-only-a aa bb", "train-only-b cc dd"),
            )],
            "train",
        )
        test = _assign(
            [make_example(
                uid=9502, group_id="g-2",
                options=(shared, "test-only-a ee ff", "test-only-b gg hh"),
            )],
            "test",
        )
        with pytest.raises(OptionTextLeakageError) as excinfo:
            assert_no_option_text_leak(train + test)
        error = excinfo.value
        assert error.code == "SPLIT_OPTION_TEXT_LEAK"
        assert error.count == 1
        assert error.texts == (shared,)
        assert error.pair_counts == {"train/test": 1}

    def test_thirty_nine_token_option_is_not_flagged(self):
        shared = _n_token_text(39)
        train = _assign(
            [make_example(
                uid=9601, group_id="g-1", options=(shared, "a-one aa bb", "a-two cc dd")
            )],
            "train",
        )
        dev = _assign(
            [make_example(
                uid=9602, group_id="g-2", options=(shared, "a-three ee ff", "a-four gg hh")
            )],
            "dev",
        )
        assert find_shared_option_texts(train + dev) == {}
        assert_no_option_text_leak(train + dev)  # must not raise

    def test_forty_token_option_in_one_split_is_fine(self):
        shared = _n_token_text(45)
        first = _assign(
            [make_example(
                uid=9701, group_id="g-1", options=(shared, "b-one aa bb", "b-two cc dd")
            )],
            "train",
        )
        second = _assign(
            [make_example(
                uid=9702, group_id="g-2", options=(shared, "b-three ee ff", "b-four gg hh")
            )],
            "train",
        )
        assert find_shared_option_texts(first + second) == {}

    def test_option_leak_error_reports_counts_for_multiple_texts(self):
        shared_a = _n_token_text(41)
        shared_b = " ".join(f"other{index:02d}" for index in range(44))
        rows = [
            make_example(
                uid=uid,
                group_id=f"g-{uid}",
                options=(text, f"unique-{uid}-x aa bb", f"unique-{uid}-y cc dd"),
            )
            for uid, text in (
                (9801, shared_a), (9802, shared_a), (9803, shared_b), (9804, shared_b)
            )
        ]
        rows[1] = dataclasses.replace(rows[1], split="dev")
        rows[2] = dataclasses.replace(rows[2], split="test")
        with pytest.raises(OptionTextLeakageError) as excinfo:
            assert_no_option_text_leak(rows)
        error = excinfo.value
        assert error.count == 2
        assert set(error.pair_counts) == {"train/dev", "train/test"}
        assert error.pair_counts["train/dev"] == 1
        assert error.pair_counts["train/test"] == 1

    def test_min_tokens_boundary_uses_constant(self):
        assert token_count(_n_token_text(40)) == 40
        assert token_count(_n_token_text(39)) == 39

    def test_token_count(self):
        assert token_count("one two  three") == 3
        assert token_count("") == 0


class TestThirteenGramOverlap:
    def test_exact_thirteen_gram_across_splits_is_fatal(self):
        span = " ".join(SHARED_SPAN)
        train = _assign(
            [make_example(uid=9901, group_id="g-1", goal=f"goal-9901 pre9901a {span} post9901b")],
            "train",
        )
        dev = _assign(
            [make_example(uid=9902, group_id="g-2", goal=f"goal-9902 pre9902a {span} post9902b")],
            "dev",
        )
        with pytest.raises(NGramLeakageError) as excinfo:
            assert_no_thirteen_gram_overlap(train + dev)
        error = excinfo.value
        assert error.code == "SPLIT_13GRAM_OVERLAP"
        assert error.counts == {"train/dev": 1}
        assert error.total == 1

    def test_twelve_shared_tokens_are_not_flagged(self):
        span = " ".join(SHARED_SPAN[:12])
        train = _assign(
            [make_example(uid=9911, group_id="g-1", goal=f"goal-9911 pre9911a {span} post9911b")],
            "train",
        )
        dev = _assign(
            [make_example(uid=9912, group_id="g-2", goal=f"goal-9912 pre9912a {span} post9912b")],
            "dev",
        )
        assert find_thirteen_gram_overlaps(train + dev) == {}
        assert_no_thirteen_gram_overlap(train + dev)  # must not raise

    def test_shared_gram_inside_one_split_is_fine(self):
        span = " ".join(SHARED_SPAN)
        first = _assign(
            [make_example(uid=9921, group_id="g-1", goal=f"goal-9921 pre9921a {span} post9921b")],
            "train",
        )
        second = _assign(
            [make_example(uid=9922, group_id="g-2", goal=f"goal-9922 pre9922a {span} post9922b")],
            "train",
        )
        assert find_thirteen_gram_overlaps(first + second) == {}

    def test_pair_counts_cover_all_three_pairs(self):
        span = " ".join(SHARED_SPAN)
        rows = []
        for split_name, uid in (("train", 9931), ("dev", 9932), ("test", 9933)):
            rows.append(
                dataclasses.replace(
                    make_example(
                        uid=uid, group_id=f"g-{uid}",
                        goal=f"goal-{uid} pre{uid}a {span} post{uid}b",
                    ),
                    split=split_name,
                )
            )
        counts = find_thirteen_gram_overlaps(rows)
        assert counts == {"dev/test": 1, "train/dev": 1, "train/test": 1}

    def test_n_gram_validation(self):
        with pytest.raises(SplitInputError):
            find_thirteen_gram_overlaps([], n_gram=0)


class TestEndToEndThroughRouter:
    def test_split_raises_on_cross_split_shared_long_option_text(self):
        shared = _n_token_text(45)
        examples = []
        for uid in range(8000, 8060):
            examples.append(
                make_example(
                    uid=uid,
                    group_id=f"g-{uid}",
                    options=(shared, f"unique-{uid}-x aa bb", f"unique-{uid}-y cc dd"),
                )
            )
        with pytest.raises(OptionTextLeakageError) as excinfo:
            split(examples, seed=13)
        assert excinfo.value.count == 1

    def test_split_reports_exclusion_for_cross_split_thirteen_gram_overlap(self):
        # A fixed directive shared across splits used to abort the whole
        # build. It is now quarantined and counted in the manifest, with a
        # hard assertion on the rows that remain.
        span = " ".join(SHARED_SPAN)
        examples = []
        for uid in range(8100, 8160):
            examples.append(
                make_example(
                    uid=uid,
                    group_id=f"g-{uid}",
                    goal=f"goal-{uid} pre{uid}a {span} post{uid}b",
                )
            )
        # Keep a clean tail so quarantine does not remove the entire corpus.
        for uid in range(9000, 9030):
            examples.append(
                make_example(uid=uid, group_id=f"g-{uid}", goal=f"unique clean goal {uid}")
            )
        result = split(examples, seed=13)
        assert result.manifest.excluded_thirteen_gram_rows > 0
        assert len(result.manifest.excluded_thirteen_gram_ids) == (
            result.manifest.excluded_thirteen_gram_rows
        )
        assert len(result.examples) + result.manifest.excluded_thirteen_gram_rows == len(
            examples
        )
        assert find_thirteen_gram_overlaps(result.examples) == {}
        assert_no_leakage(result.examples)  # residual hard check passes

    def test_split_passes_on_clean_corpus(self):
        examples = [
            make_example(uid=8200 + uid, group_id=f"g-{8200 + uid}") for uid in range(40)
        ]
        result = split(examples, seed=13)
        assert assert_no_leakage(result.examples) is None
