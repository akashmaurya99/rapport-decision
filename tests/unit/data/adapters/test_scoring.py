"""Scoring adapter: HelpSteer2 multi-attribute → utility floats in [0,1]."""

from __future__ import annotations

import math

from src.data.adapters import restore_original_order
from src.data.adapters.scoring import ScoringAdapter


def helpsteer_row(helpfulness=4, correctness=3, coherence=2, response="A fine answer."):
    return {
        "prompt": "Explain photosynthesis.",
        "response": response,
        "helpfulness": helpfulness,
        "correctness": correctness,
        "coherence": coherence,
        "complexity": 3,
        "verbosity": 2,
        "id": "hs-1",
    }


def test_maps_attributes_to_utility_scores():
    result = ScoringAdapter(seed=5).to_decision_example([helpsteer_row()])
    assert len(result.examples) == 1
    example = result.examples[0]
    assert example.task_family == "scoring"
    assert example.source == "nvidia/HelpSteer2"
    assert example.license == "CC-BY-4.0"
    assert example.group_id == "hs-1"
    assert len(example.options) == 3
    assert len(example.scores) == 3
    assert all(0.0 <= s <= 1.0 for s in example.scores)
    by_aspect = dict(zip([o.text for o in example.options], example.scores))
    assert by_aspect["helpfulness"] == 1.0
    assert by_aspect["correctness"] == 0.75
    assert by_aspect["coherence"] == 0.5


def test_scores_travel_with_shuffled_options():
    adapter = ScoringAdapter(seed=31)
    result = adapter.to_decision_example([helpsteer_row()])
    example, meta = result.examples[0], result.meta[0]
    assert meta.shuffle_seed is not None
    restored = restore_original_order(example, meta)
    assert [o.text for o in restored.options] == ["helpfulness", "correctness", "coherence"]
    assert list(restored.scores) == [1.0, 0.75, 0.5]


def test_excluded_attributes_are_reported_not_smuggled():
    result = ScoringAdapter().to_decision_example([helpsteer_row()])
    assert result.report["attributes_mapped"] == ["helpfulness", "correctness", "coherence"]
    assert result.report["excluded_attributes"] == ["complexity", "verbosity"]
    assert "complexity" not in [o.text for o in result.examples[0].options]


def test_rating_out_of_range_and_bad_types_rejected():
    for bad in (-1, 5, 4.5, math.nan):
        row = helpsteer_row(helpfulness=bad)
        result = ScoringAdapter().to_decision_example([row])
        assert result.rejected[0].reason_code == "ADAPTOR_BAD_SCORE", bad
    result = ScoringAdapter().to_decision_example([helpsteer_row(helpfulness="high")])
    assert result.rejected[0].reason_code == "ADAPTOR_BAD_TYPE"


def test_missing_rating_rejected():
    row = helpsteer_row()
    del row["correctness"]
    result = ScoringAdapter().to_decision_example([row])
    assert result.rejected[0].reason_code == "ADAPTOR_MISSING_FIELD"


def test_missing_response_rejected_with_empty_state_code_for_empty():
    row = helpsteer_row(response="   ")
    result = ScoringAdapter().to_decision_example([row])
    assert result.rejected[0].reason_code == "ADAPTOR_EMPTY_STATE"
    row = helpsteer_row()
    del row["response"]
    assert ScoringAdapter().to_decision_example([row]).rejected[0].reason_code == (
        "ADAPTOR_MISSING_FIELD"
    )


def test_scores_are_floats_and_parse_cleanly():
    result = ScoringAdapter(seed=2).to_decision_example([helpsteer_row(4, 4, 4)])
    assert all(isinstance(s, float) for s in result.examples[0].scores)
    assert set(result.examples[0].scores) == {1.0}
