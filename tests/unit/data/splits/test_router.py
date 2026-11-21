"""Tests for the split router: assignment, ratios, OOD handling, determinism."""

from __future__ import annotations

import pytest
from conftest import make_example

from src.data.core.schema import TASK_FAMILIES
from src.data.splits import (
    DEFAULT_OOD_SOURCE,
    SplitInputError,
    SplitManifest,
    SplitResult,
    split,
)
from src.data.splits.router import SplitResult as RouterSplitResult


def _corpus(n_per_family: int, families=("choice", "preference"), source="fixture"):
    examples = []
    uid = 0
    for family in families:
        for _ in range(n_per_family):
            uid += 1
            examples.append(make_example(uid=uid, family=family, source=source))
    return examples


class TestAssignmentRule:
    def test_every_group_lands_in_exactly_one_split(self):
        examples = _corpus(30)
        result = split(examples, seed=13)
        seen: dict[str, str] = {}
        for example in result.examples:
            prior = seen.setdefault(example.group_id, example.split)
            assert prior == example.split

    def test_no_group_in_two_splits_across_seeds(self):
        examples = _corpus(25)
        for seed in range(10):
            result = split(examples, seed=seed)
            groups = {}
            for example in result.examples:
                groups.setdefault(example.group_id, set()).add(example.split)
            assert all(len(splits) == 1 for splits in groups.values())

    def test_ratio_is_exact_seventy_fifteen_fifteen_per_family_singleton_groups(self):
        # 100 singleton groups per family -> exact 70/15/15 row counts.
        examples = _corpus(100, families=("choice", "preference"))
        result = split(examples, seed=13)
        for family in ("choice", "preference"):
            counts = result.manifest.counts[family]
            assert counts == {"train": 70, "dev": 15, "test": 15}

    def test_row_counts_add_up_per_family(self):
        examples = _corpus(17, families=sorted(TASK_FAMILIES))
        result = split(examples, seed=13)
        for family in sorted(TASK_FAMILIES):
            counts = result.manifest.counts[family]
            expected = sum(1 for e in examples if e.task_family == family)
            assert sum(counts.values()) == expected

    def test_totals_match_examples(self):
        examples = _corpus(30)
        result = split(examples, seed=13)
        assert result.manifest.totals == {
            name: len(result.by_split(name)) for name in ("train", "dev", "test")
        }
        assert len(result.examples) == len(examples)

    def test_split_output_carries_assigned_split_and_preserves_example_id(self):
        examples = _corpus(10)
        original_ids = [e.example_id for e in examples]
        result = split(examples, seed=13)
        assert [e.example_id for e in result.examples] == original_ids
        assert {e.split for e in result.examples} == {"train", "dev", "test"}

    def test_group_spanning_families_keeps_one_split(self):
        shared_group = "group-shared"
        examples = [
            make_example(uid=1001, family="choice", group_id=shared_group),
            make_example(uid=1002, family="preference", group_id=shared_group),
        ]
        result = split(examples, seed=13)
        assert result.examples[0].split == result.examples[1].split

    def test_input_order_is_preserved(self):
        examples = _corpus(12)
        result = split(examples, seed=13)
        assert [e.example_id for e in result.examples] == [e.example_id for e in examples]


class TestCoarseGroups:
    def test_coarse_groups_still_reach_every_split(self):
        # Four atomic groups of three rows used to yield 9/3/0: the small
        # test quota was starved by the large train quota.
        examples = []
        uid = 50000
        for group in range(4):
            for _ in range(3):
                uid += 1
                examples.append(
                    make_example(uid=uid, family="choice", group_id=f"coarse-{group}")
                )
        result = split(examples, seed=13)
        counts = result.manifest.counts["choice"]
        assert all(counts[name] > 0 for name in ("train", "dev", "test"))
        assert sum(counts.values()) == len(examples)
        assert len(result.examples) == len(examples)
        assert result.manifest.excluded_thirteen_gram_rows == 0


class TestOodHandling:
    def test_ood_source_contributes_zero_rows_to_train_and_dev(self):
        ood = [
            make_example(uid=2000 + i, family="preference", source=DEFAULT_OOD_SOURCE)
            for i in range(12)
        ]
        regular = _corpus(30)
        result = split(regular + ood, seed=13)
        ood_ids = {e.example_id for e in ood}
        assert all(e.example_id not in ood_ids for e in result.train)
        assert all(e.example_id not in ood_ids for e in result.dev)
        assert ood_ids <= {e.example_id for e in result.test}

    def test_ood_identity_is_recorded_in_the_manifest(self):
        ood = [
            make_example(uid=3000 + i, family="preference", source=DEFAULT_OOD_SOURCE)
            for i in range(4)
        ]
        result = split(_corpus(15) + ood, seed=13)
        assert result.manifest.ood_source == DEFAULT_OOD_SOURCE
        assert result.manifest.withheld_source == DEFAULT_OOD_SOURCE
        assert result.manifest.ood_rows == 4

    def test_train_and_dev_counts_exclude_ood_rows(self):
        ood = [
            make_example(uid=4000 + i, family="preference", source=DEFAULT_OOD_SOURCE)
            for i in range(9)
        ]
        regular = _corpus(20, families=("preference",))
        result = split(regular + ood, seed=13)
        preference = result.manifest.counts["preference"]
        assert preference["train"] == len(result.train)
        assert preference["dev"] == len(result.dev)
        # preference test rows = non-OOD remainder + the 9 OOD rows
        assert preference["test"] == len(result.test)

    def test_group_shared_between_ood_and_regular_source_follows_ood(self):
        mixed_group = "group-mixed"
        ood = make_example(
            uid=5001, family="preference", group_id=mixed_group, source=DEFAULT_OOD_SOURCE
        )
        regular = make_example(uid=5002, family="choice", group_id=mixed_group, source="fixture")
        result = split([ood, regular] + _corpus(10), seed=13)
        by_id = {e.example_id: e for e in result.examples}
        assert by_id[ood.example_id].split == "test"
        assert by_id[regular.example_id].split == "test"

    def test_custom_ood_source_is_honored(self):
        ood = [
            make_example(uid=6000 + i, family="choice", source="custom-ood")
            for i in range(5)
        ]
        result = split(_corpus(20) + ood, seed=13, ood_source="custom-ood")
        assert result.manifest.ood_source == "custom-ood"
        assert result.manifest.withheld_source == "custom-ood"
        assert all(e.split == "test" for e in result.ood)
        assert all(e.split != "train" and e.split != "dev" for e in result.ood)


class TestDeterminism:
    def test_two_calls_produce_identical_assignments_and_fingerprints(self):
        examples = _corpus(20, families=("choice", "preference", "risk"))
        first = split(examples, seed=13)
        second = split(examples, seed=13)
        assert [e.split for e in first.examples] == [e.split for e in second.examples]
        assert first.manifest == second.manifest
        assert first.fingerprint == second.fingerprint

    def test_manifest_json_is_byte_identical_across_calls(self):
        examples = _corpus(10)
        first = split(examples, seed=13).manifest.to_json()
        second = split(examples, seed=13).manifest.to_json()
        assert first == second

    def test_input_permutation_keeps_assignment_and_fingerprint(self):
        examples = _corpus(15)
        forward = split(examples, seed=13)
        shuffled = split(list(reversed(examples)), seed=13)
        forward_by_id = {e.example_id: e.split for e in forward.examples}
        shuffled_by_id = {e.example_id: e.split for e in shuffled.examples}
        assert forward_by_id == shuffled_by_id
        assert forward.fingerprint == shuffled.fingerprint

    def test_different_seeds_produce_valid_splits(self):
        examples = _corpus(20)
        for seed in (1, 7, 99):
            result = split(examples, seed=seed)
            groups: dict[str, set[str]] = {}
            for example in result.examples:
                groups.setdefault(example.group_id, set()).add(example.split)
            assert all(len(s) == 1 for s in groups.values())


class TestManifestContents:
    def test_manifest_records_fingerprint_counts_threshold_and_zero_overlap(self):
        examples = _corpus(20)
        result = split(examples, seed=13)
        manifest = result.manifest
        assert isinstance(manifest, SplitManifest)
        assert len(manifest.fingerprint) == 64
        assert set(manifest.counts) == {"choice", "preference"}
        for counts in manifest.counts.values():
            assert set(counts) == {"train", "dev", "test"}
        assert manifest.group_overlap == 0
        assert manifest.dedup_threshold == 0.9
        assert manifest.ood_source == DEFAULT_OOD_SOURCE
        assert manifest.withheld_source == DEFAULT_OOD_SOURCE
        assert manifest.seed == 13

    def test_dedup_threshold_is_carried_from_the_clean_pipeline(self):
        # clean_examples returns CleanResult.dedup_threshold (default 0.9);
        # a different threshold chosen upstream must be recorded verbatim.
        result = split(_corpus(5), seed=13, dedup_threshold=0.85)
        assert result.manifest.dedup_threshold == 0.85
        assert result.manifest.to_dict()["dedup_threshold"] == 0.85


class TestInputValidation:
    def test_seed_must_be_a_non_bool_int(self):
        with pytest.raises(SplitInputError):
            split(_corpus(3), seed=True)  # type: ignore[arg-type]
        with pytest.raises(SplitInputError):
            split(_corpus(3), seed=-1)
        with pytest.raises(SplitInputError):
            split(_corpus(3), seed=1.5)  # type: ignore[arg-type]

    def test_ood_source_must_be_a_non_empty_string(self):
        with pytest.raises(SplitInputError):
            split(_corpus(3), seed=13, ood_source="   ")

    def test_dedup_threshold_must_lie_in_unit_interval(self):
        with pytest.raises(SplitInputError):
            split(_corpus(3), seed=13, dedup_threshold=1.5)
        with pytest.raises(SplitInputError):
            split(_corpus(3), seed=13, dedup_threshold=True)  # type: ignore[arg-type]

    def test_non_decision_example_rows_are_rejected(self):
        with pytest.raises(SplitInputError):
            split([{"not": "an example"}], seed=13)  # type: ignore[list-item]

    def test_result_is_a_split_result(self):
        result = split(_corpus(3), seed=13)
        assert isinstance(result, SplitResult)
        assert isinstance(result, RouterSplitResult)

    def test_replace_does_not_mutate_input_examples(self):
        examples = _corpus(5)
        original = [(e.example_id, e.split) for e in examples]
        split(examples, seed=13)
        assert [(e.example_id, e.split) for e in examples] == original


class TestEmptyInput:
    def test_empty_input_yields_a_valid_empty_manifest(self):
        result = split([], seed=13)
        assert result.examples == ()
        assert result.manifest.counts == {}
        assert result.manifest.totals == {"train": 0, "dev": 0, "test": 0}
        assert result.manifest.group_overlap == 0
        assert result.manifest.ood_rows == 0
