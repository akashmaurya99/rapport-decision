"""Abstention adapter: per-subset conversion rates, never pooled."""

from __future__ import annotations

from src.data.adapters.abstention import AbstentionAdapter


def row(subset="subset_a", question="When was the Treaty of Westphalia signed?", answerable=False):
    return {
        "subset": subset,
        "question": question,
        "answerable": answerable,
        "id": f"{subset}-1",
    }


def test_answerable_row_maps_with_yes_gold():
    result = AbstentionAdapter(seed=7).to_decision_example([row(answerable=True)])
    example = result.examples[0]
    assert example.task_family == "abstention"
    assert example.answerable is True
    assert example.options[example.chosen_index].text == "Answerable"
    assert example.source == "AbstentionBench-derived"
    assert example.group_id == "subset_a-1"
    assert result.meta[0].subset == "subset_a"


def test_unanswerable_row_maps_with_no_gold():
    result = AbstentionAdapter(seed=7).to_decision_example([row(answerable=False)])
    example = result.examples[0]
    assert example.answerable is False
    assert example.options[example.chosen_index].text == "Unanswerable"


def test_unanswerable_field_alias_is_supported():
    result = AbstentionAdapter().to_decision_example(
        [{"subset": "s", "question": "q", "unanswerable": True}]
    )
    assert result.examples[0].answerable is False


def test_contradictory_labels_rejected():
    result = AbstentionAdapter().to_decision_example(
        [
            {"subset": "s", "question": "q", "answerable": True, "unanswerable": True},
            {"subset": "s", "question": "q", "answerable": False, "unanswerable": False},
        ]
    )
    assert [r.reason_code for r in result.rejected] == [
        "ADAPTOR_CONTRADICTORY_LABEL",
        "ADAPTOR_CONTRADICTORY_LABEL",
    ]


def test_per_subset_conversion_rates_reported_never_pooled():
    rows = [
        row("subset_a", answerable=True),
        row("subset_a", answerable=False),
        row("subset_b", question=""),  # rejected: empty state
        row("subset_b", answerable=True),
        row("subset_b", answerable=False),
    ]
    result = AbstentionAdapter().to_decision_example(rows)
    per_subset = result.report["per_subset"]
    assert per_subset["subset_a"] == {
        "input": 2,
        "converted": 2,
        "rejected": 0,
        "conversion_rate": 1.0,
    }
    assert per_subset["subset_b"]["input"] == 3
    assert per_subset["subset_b"]["converted"] == 2
    assert per_subset["subset_b"]["rejected"] == 1
    assert abs(per_subset["subset_b"]["conversion_rate"] - 2 / 3) < 1e-12
    # No pooled rate anywhere at the top level.
    assert "conversion_rate" not in result.report
    assert "pooled" not in str(result.report.keys()).lower()
    assert result.report["n_subsets"] == 2


def test_missing_subset_cannot_be_bucketed():
    result = AbstentionAdapter().to_decision_example(
        [{"question": "q", "answerable": True}]
    )
    assert result.rejected[0].reason_code == "ADAPTOR_MISSING_FIELD"
    assert result.report["unbucketed_rejections"] == 1
    assert result.report["per_subset"] == {}


def test_missing_answerability_rejected():
    result = AbstentionAdapter().to_decision_example([{"subset": "s", "question": "q"}])
    assert result.rejected[0].reason_code == "ADAPTOR_MISSING_FIELD"


def test_answerable_must_be_bool():
    result = AbstentionAdapter().to_decision_example(
        [{"subset": "s", "question": "q", "answerable": "yes"}]
    )
    assert result.rejected[0].reason_code == "ADAPTOR_BAD_TYPE"


def test_context_extends_state():
    result = AbstentionAdapter().to_decision_example(
        [{"subset": "s", "question": "q", "context": "docs", "answerable": True}]
    )
    assert "Context:" in result.examples[0].state
    assert result.examples[0].state.startswith("q")


def test_zero_to_one_conversion_rate_when_all_fail():
    rows = [row("subset_c", question="") for _ in range(3)]
    result = AbstentionAdapter().to_decision_example(rows)
    assert result.report["per_subset"]["subset_c"]["conversion_rate"] == 0.0
