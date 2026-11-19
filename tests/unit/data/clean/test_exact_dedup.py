"""Exact dedup: byte-identical normalized rows are removed, per-source counts."""

from __future__ import annotations

from src.data.clean import ExactDedup, exact_key


def test_exact_dedup_removes_byte_identical_rows(example_factory):
    first = example_factory(group_id="g-1")
    duplicate = example_factory(group_id="g-2")  # same content, different group
    distinct = example_factory(goal="Choose the backup region for the archive", group_id="g-3")

    result = ExactDedup().run([first, duplicate, distinct])

    assert result.kept == (first, distinct)
    assert result.removed_ids == (duplicate.example_id,)
    assert result.exact_removed == 1


def test_exact_dedup_reports_per_source_counts(example_factory):
    a = example_factory(source="alpha", group_id="g-1")
    b = example_factory(
        source="beta",
        goal="Choose the backup region for the nightly archive restore",
        group_id="g-2",
    )
    a_dup = example_factory(source="alpha", group_id="g-3")
    b_dup1 = example_factory(
        source="beta",
        goal="Choose the backup region for the nightly archive restore",
        group_id="g-4",
    )
    b_dup2 = example_factory(
        source="beta",
        goal="Choose the backup region for the nightly archive restore",
        group_id="g-5",
    )

    result = ExactDedup().run([a, b, a_dup, b_dup1, b_dup2])

    assert result.per_source == {"alpha": 1, "beta": 2}
    assert result.exact_removed == 3
    assert sum(result.per_source.values()) == result.exact_removed


def test_exact_dedup_keeps_first_occurrence_in_input_order(example_factory):
    first = example_factory(group_id="g-first")
    second = example_factory(group_id="g-second")
    third = example_factory(group_id="g-third")

    result = ExactDedup().run([first, second, third])

    assert result.kept[0].group_id == "g-first"


def test_exact_dedup_key_is_option_order_invariant(example_factory):
    row = example_factory()
    permuted_options = [{"text": text} for text in sorted(
        ("Roll out to ten percent of traffic first",
         "Roll out to all traffic immediately",
         "Delay the rollout by two full weeks"), reverse=True)]
    from src.data.core import normalize_example

    shuffled = normalize_example(
        {
            "task_family": "choice",
            "goal": "Pick the safest plan for the deployment rollout",
            "state": "The service holds customer data and the rollout window is four hours",
            "options": permuted_options,
            "chosen_index": 2,
            "source": "fixture",
            "license": "MIT",
            "group_id": "group-perm",
            "split": "train",
        }
    )
    # Same option texts, permuted order, chosen_index tracks the same gold text.
    assert exact_key(row)[:4] == exact_key(shuffled)[:4]


def test_exact_dedup_key_includes_chosen_index(example_factory):
    a = example_factory(chosen_index=0, group_id="g-1")
    b = example_factory(chosen_index=1, group_id="g-2")
    assert exact_key(a) != exact_key(b)

    result = ExactDedup().run([a, b])
    assert result.exact_removed == 0


def test_exact_dedup_result_is_deterministic(example_factory):
    rows = [example_factory(group_id=f"g-{i}") for i in range(6)]
    rows += list(rows)  # exact duplicates of every row
    first = ExactDedup().run(rows)
    second = ExactDedup().run(rows)
    assert first.removed_ids == second.removed_ids
    assert first.per_source == second.per_source
    assert [row.example_id for row in first.kept] == [row.example_id for row in second.kept]
