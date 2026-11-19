"""End-to-end cleaning pipeline: stage order, CleanReport, contested plumbing."""

from __future__ import annotations

import json

from src.data.clean import CleanReport, clean_examples

_LONG_STATE = (
    "The service holds customer data across three regions and the rollout window "
    "is four hours with on call staff monitoring the error budget closely "
) * 3
_NEAR_STATE = _LONG_STATE.replace("three regions", "four regions")


def test_clean_report_shape_and_totals(example_factory):
    keep = example_factory(group_id="g-keep")
    exact_dup = example_factory(group_id="g-exact-dup")  # same content as keep
    near_dup = example_factory(state=_NEAR_STATE, group_id="g-near-dup")
    # The contested pair uses a distinct goal so it does not collide with the
    # keep row on the (source-independent) exact-dedup key.
    contested_a = example_factory(
        goal="Choose the backup region for the nightly archive restore",
        chosen_index=0,
        group_id="g-contra-a",
    )
    contested_b = example_factory(
        goal="Choose the backup region for the nightly archive restore",
        chosen_index=1,
        group_id="g-contra-b",
    )
    q1_rejected = example_factory(
        options=("Yes", "Roll out to ten percent of traffic first",
                 "Delay the rollout by two full weeks"),
        chosen_index=0,
        group_id="g-q1",
    )
    rows = [keep, exact_dup, near_dup, contested_a, contested_b, q1_rejected]

    result = clean_examples(rows)
    report = result.report

    assert isinstance(report, CleanReport)
    assert report.exact_removed == 1
    assert report.near_removed == 0  # near-dup's state is too different; kept
    assert report.per_rule["Q1"] == 1
    assert sum(report.per_rule.values()) == 1
    assert report.contested == 2
    assert report.kept == len(result.kept)
    assert report.kept + report.exact_removed + report.near_removed + sum(
        report.per_rule.values()
    ) + report.contested == len(rows)


def test_contested_rows_are_not_in_clean_output(example_factory, tmp_path):
    a = example_factory(chosen_index=0, group_id="g-contra-a")
    b = example_factory(chosen_index=1, group_id="g-contra-b")
    destination = tmp_path / "contested.jsonl"

    result = clean_examples([a, b], contested_path=destination)

    assert result.kept == ()
    assert {row.example_id for row in result.contested} == {a.example_id, b.example_id}
    assert set(result.contested_ids).isdisjoint(row.example_id for row in result.kept)
    lines = destination.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert result.contested_path == destination
    assert result.dedup_threshold == 0.9


def test_exact_duplicate_removal_precedes_contradiction_audit(example_factory):
    # Byte-identical rows are deduped before the audit, so an exact-duplicate
    # pair does not look like a "group" of two with one label.
    a = example_factory(chosen_index=0, group_id="g-same-a")
    a_dup = example_factory(chosen_index=0, group_id="g-same-b")

    result = clean_examples([a, a_dup])

    assert result.report.exact_removed == 1
    assert result.report.contested == 0
    assert result.report.kept == 1


def test_near_duplicate_pair_is_counted_and_kept_first_wins(example_factory):
    a = example_factory(state=_LONG_STATE, group_id="g-near-a")
    b = example_factory(state=_NEAR_STATE, group_id="g-near-b")

    result = clean_examples([a, b])

    assert result.report.near_removed == 1
    assert [row.group_id for row in result.kept] == ["g-near-a"]


def test_near_duplicates_with_conflicting_labels_reach_contested(example_factory):
    # Identical content, different gold: near-dedup must not merge them; the
    # contradiction audit quarantines both.
    a = example_factory(chosen_index=0, state=_LONG_STATE, group_id="g-clash-a")
    b = example_factory(chosen_index=1, state=_LONG_STATE, group_id="g-clash-b")

    result = clean_examples([a, b])

    assert result.report.near_removed == 0
    assert result.report.contested == 2
    assert result.kept == ()


def test_clean_examples_is_deterministic(example_factory):
    rows = [
        example_factory(state=_LONG_STATE, group_id="g-det-a"),
        example_factory(state=_NEAR_STATE, group_id="g-det-b"),
        example_factory(group_id="g-det-c"),
        example_factory(chosen_index=1, group_id="g-det-d"),
        example_factory(chosen_index=0, group_id="g-det-e"),
    ]
    first = clean_examples(rows)
    second = clean_examples(rows)
    assert [row.example_id for row in first.kept] == [row.example_id for row in second.kept]
    assert first.contested_ids == second.contested_ids
    assert first.report == second.report


def test_clean_report_is_serializable_shape(example_factory):
    result = clean_examples([example_factory(group_id="g-only")])
    payload = {
        "exact_removed": result.report.exact_removed,
        "near_removed": result.report.near_removed,
        "per_rule": result.report.per_rule,
        "contested": result.report.contested,
        "kept": result.report.kept,
    }
    assert json.loads(json.dumps(payload))["kept"] == 1
