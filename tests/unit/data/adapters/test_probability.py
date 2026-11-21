"""Probability adapter: auxiliary_train-derived binaries and TruthfulQA."""

from __future__ import annotations

import pytest

from src.data.adapters.base import AdapterError
from src.data.adapters.probability import ProbabilityAdapter


def aux_row():
    return {
        "question": "What is the capital of France?",
        "choices": ["London", "Paris", "Berlin", "Madrid"],
        "answer": "B",
        "id": "aux-1",
    }


def tqa_row(with_incorrect=True):
    row = {
        "question": "What happens if you eat watermelon seeds?",
        "best_answer": "Nothing happens.",
        "incorrect_answers": ["A watermelon grows in your stomach."],
        "id": "tqa-1",
    }
    if not with_incorrect:
        row["incorrect_answers"] = []
    return row


def test_auxiliary_train_row_yields_one_binary_row_per_option():
    result = ProbabilityAdapter(seed=17).to_decision_example([aux_row()])
    assert len(result.examples) == 4
    assert result.report["derived_binary_rows"] == 4
    assert result.report["input_rows"] == 1
    for example in result.examples:
        assert example.task_family == "probability"
        assert example.source == "cais/mmlu#auxiliary_train-derived-binary"
        assert example.license == "MIT"
        assert example.group_id == "aux-1"  # shared with parent: never straddle a split
        assert [o.text for o in example.options] in (["Yes", "No"], ["No", "Yes"])
        assert "Is '" in example.state and "the correct answer?" in example.state


def test_auxiliary_rows_never_inherit_truthfulqa_provenance():
    result = ProbabilityAdapter(seed=17).to_decision_example([aux_row(), tqa_row()])
    derived = [e for e in result.examples if "correct answer?" in e.state]
    originals = [e for e in result.examples if "Proposition:" in e.state]
    assert derived and originals
    assert all(e.source == "cais/mmlu#auxiliary_train-derived-binary" for e in derived)
    assert all(e.license == "MIT" for e in derived)
    assert all(e.source == "truthfulqa/truthful_qa" for e in originals)
    assert all(e.license == "Apache-2.0" for e in originals)
    assert result.report["provenance"]["mmlu_auxiliary_train_derived"] == {
        "source": "cais/mmlu#auxiliary_train-derived-binary",
        "license": "MIT",
    }


def test_manifest_license_is_honored_but_quarantine_is_not():
    class Manifest:
        def __init__(self, status, license):
            self.status = status
            self.license = license

    admitted = ProbabilityAdapter(mmlu_manifest=Manifest("admitted", "MIT-recorded"))
    assert admitted.mmlu_license == "MIT-recorded"
    with pytest.raises(AdapterError, match="not admitted"):
        ProbabilityAdapter(mmlu_manifest=Manifest("quarantine", "unknown"))
    with pytest.raises(AdapterError, match="no usable license"):
        ProbabilityAdapter(mmlu_manifest=Manifest("admitted", ""))


def test_exactly_one_derived_row_marks_the_gold():
    result = ProbabilityAdapter(seed=17).to_decision_example([aux_row()])
    golds = [e.answerable for e in result.examples]
    assert golds.count(True) == 1
    gold_example = next(e for e in result.examples if e.answerable)
    assert gold_example.options[gold_example.chosen_index].text == "Yes"
    for example in result.examples:
        chosen_text = example.options[example.chosen_index].text
        assert (chosen_text == "Yes") == example.answerable


def test_the_gold_option_text_appears_in_its_own_proposition():
    result = ProbabilityAdapter(seed=17).to_decision_example([aux_row()])
    gold_example = next(e for e in result.examples if e.answerable)
    assert "Paris" in gold_example.state


def test_derived_rows_are_stable_and_shuffled():
    a = ProbabilityAdapter(seed=17).to_decision_example([aux_row()])
    b = ProbabilityAdapter(seed=17).to_decision_example([aux_row()])
    assert [e.example_id for e in a.examples] == [e.example_id for e in b.examples]
    # Shuffled at ingest with recorded seeds, one per derived row.
    assert all(m.shuffle_seed is not None for m in a.meta)
    assert len(set(m.shuffle_seed for m in a.meta)) > 1


def test_truthfulqa_row_emits_yes_and_no_propositions():
    result = ProbabilityAdapter(seed=5).to_decision_example([tqa_row()])
    assert len(result.examples) == 2
    assert result.report["truthfulqa_yes_rows"] == 1
    assert result.report["truthfulqa_no_rows"] == 1
    assert all(e.group_id == "tqa-1" for e in result.examples)
    gold_values = {e.answerable for e in result.examples}
    assert gold_values == {True, False}


def test_truthfulqa_without_incorrect_answers_emits_single_yes_row():
    result = ProbabilityAdapter(seed=5).to_decision_example([tqa_row(with_incorrect=False)])
    assert len(result.examples) == 1
    assert result.examples[0].answerable is True


def test_unrecognizable_row_rejected_with_typed_reason():
    result = ProbabilityAdapter().to_decision_example([{"hello": "world"}])
    assert result.rejected[0].reason_code == "ADAPTOR_MISSING_FIELD"
    assert result.report["converted"] == 0


def test_gold_out_of_range_in_aux_row_rejected():
    row = aux_row()
    row["answer"] = "Z"
    result = ProbabilityAdapter().to_decision_example([row])
    assert result.rejected[0].reason_code == "ADAPTOR_NO_GOLD"


def test_empty_question_rejected():
    row = aux_row()
    row["question"] = "   "
    assert ProbabilityAdapter().to_decision_example([row]).rejected[0].reason_code == (
        "ADAPTOR_EMPTY_STATE"
    )


def test_report_counts_mixed_batch():
    rows = [aux_row(), tqa_row(), {"nonsense": True}]
    result = ProbabilityAdapter(seed=3).to_decision_example(rows)
    assert result.report["input_rows"] == 3
    assert result.report["converted"] == 6
    assert result.report["rejected"] == 1
    assert result.report["derived_binary_rows"] == 4


@pytest.mark.parametrize("seed", [1, 13, 99])
def test_derived_gold_position_tracks_shuffle(seed):
    result = ProbabilityAdapter(seed=seed).to_decision_example([aux_row()])
    for example in result.examples:
        gold_text = example.options[example.chosen_index].text
        assert gold_text == ("Yes" if example.answerable else "No")
