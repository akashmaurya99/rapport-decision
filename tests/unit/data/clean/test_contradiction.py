"""Contradiction audit: disagreeing gold groups are quarantined, never resolved."""

from __future__ import annotations

import json

from src.data.clean import ContradictionAudit, group_key
from src.data.clean.contradiction import write_contested
from src.data.core import from_dict
from src.data.core.serialize import to_dict


def _pair(example_factory):
    a = example_factory(chosen_index=0, group_id="g-contra-a")
    b = example_factory(chosen_index=1, group_id="g-contra-b")  # same content, other gold
    return a, b


def test_group_key_excludes_chosen_index(example_factory):
    a, b = _pair(example_factory)
    assert group_key(a) == group_key(b)


def test_contradictory_group_is_quarantined_and_counted(example_factory):
    a, b = _pair(example_factory)
    result = ContradictionAudit().run([a, b])

    assert result.kept == ()
    assert result.contested == (a, b)
    assert result.contested_count == 2
    assert result.contested_groups == 1
    assert set(result.contested_ids) == {a.example_id, b.example_id}


def test_no_label_is_silently_chosen(example_factory):
    a, b = _pair(example_factory)
    # Three rows, two-vs-one gold split: the majority row must NOT win.
    c = example_factory(chosen_index=0, group_id="g-contra-c")

    result = ContradictionAudit().run([a, b, c])

    assert result.kept == ()
    assert result.contested_count == 3
    kept_golds = [row.chosen_index for row in result.kept]
    assert kept_golds == []


def test_agreeing_group_is_kept(example_factory):
    a = example_factory(chosen_index=0, group_id="g-agree-a")
    b = example_factory(chosen_index=0, group_id="g-agree-b")

    result = ContradictionAudit().run([a, b])

    assert result.kept == (a, b)
    assert result.contested == ()
    assert result.contested_groups == 0


def test_contested_rows_are_excluded_from_clean_output(example_factory):
    a, b = _pair(example_factory)
    clean = example_factory(
        goal="Choose the archival storage class for the monthly backups", group_id="g-clean"
    )
    result = ContradictionAudit().run([a, b, clean])

    assert result.kept == (clean,)
    contested_ids = {row.example_id for row in result.contested}
    assert {row.example_id for row in result.kept}.isdisjoint(contested_ids)


def test_contested_file_is_written_and_reported(example_factory, tmp_path):
    a, b = _pair(example_factory)
    destination = tmp_path / "contested.jsonl"

    result = ContradictionAudit().run([a, b], contested_path=destination)

    assert result.contested_path == destination
    lines = destination.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == result.contested_count == 2
    roundtrip = [from_dict(json.loads(line)) for line in lines]
    assert [row.example_id for row in roundtrip] == list(result.contested_ids)


def test_contested_path_with_no_contradictions_writes_empty_file(example_factory, tmp_path):
    a = example_factory(chosen_index=0, group_id="g-solo")
    destination = tmp_path / "contested.jsonl"

    result = ContradictionAudit().run([a], contested_path=destination)

    assert destination.read_text(encoding="utf-8") == ""
    assert result.contested_count == 0


def test_rows_without_chosen_index_join_the_contested_group(example_factory):
    # probability rows may carry a chosen_index or not; a group that mixes
    # labeled and unlabeled rows with a labeled disagreement elsewhere is
    # quarantined wholesale — no row is picked out as the "winner".
    a = example_factory(family="probability", chosen_index=0, group_id="g-mix-a")
    b = example_factory(family="probability", chosen_index=1, group_id="g-mix-b")
    unlabeled = example_factory(family="probability", chosen_index=None, group_id="g-mix-c")

    result = ContradictionAudit().run([a, b, unlabeled])

    assert result.contested_count == 3
    assert set(result.contested_ids) == {a.example_id, b.example_id, unlabeled.example_id}
    assert result.kept == ()


def test_groups_with_identical_full_label_payloads_are_not_contested(example_factory):
    a = example_factory(
        family="scoring", chosen_index=None, scores=(0.5, 0.5, 0.5), group_id="g-s-a"
    )
    b = example_factory(
        family="scoring", chosen_index=None, scores=(0.5, 0.5, 0.5), group_id="g-s-b"
    )

    result = ContradictionAudit().run([a, b])

    assert result.kept == (a, b)
    assert result.contested == ()


def test_conflicting_scores_are_contested(example_factory):
    a = example_factory(
        family="scoring", chosen_index=None, scores=(0.5, 0.5, 0.5), group_id="g-s-a"
    )
    b = example_factory(
        family="scoring", chosen_index=None, scores=(0.25, 0.75, 0.5), group_id="g-s-b"
    )

    result = ContradictionAudit().run([a, b])

    assert result.kept == ()
    assert result.contested_count == 2


def test_write_contested_returns_row_count(example_factory, tmp_path):
    a, b = _pair(example_factory)
    destination = tmp_path / "contested.jsonl"
    assert write_contested([a, b], destination) == 2
    assert to_dict(a)["example_id"] == a.example_id
