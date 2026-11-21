"""Choice adapter: MMLU auxiliary_train and MMLU-Pro 10→8 reduction."""

from __future__ import annotations

from src.data.adapters import restore_original_order
from src.data.adapters.choice import ChoiceAdapter, MmluProAdapter
from src.data.core.schema import compute_example_id


def aux_row(answer="B", question="What is the capital of France?"):
    return {
        "question": question,
        "choices": ["London", "Paris", "Berlin", "Madrid"],
        "answer": answer,
        "id": "mmlu-aux-1",
    }


def pro_row(answer="C"):
    return {
        "question": "Which statement about photosynthesis is correct?",
        "choices": [f"option-{c}" for c in "ABCDEFGHIJ"],
        "answer": answer,
        "id": "mmlu-pro-1",
    }


def test_auxiliary_train_row_maps_into_choice_family():
    result = ChoiceAdapter(seed=42).to_decision_example([aux_row()])
    assert len(result.examples) == 1
    example = result.examples[0]
    assert example.task_family == "choice"
    assert example.source == "cais/mmlu#auxiliary_train"
    assert example.license == "MIT"
    assert example.group_id == "mmlu-aux-1"
    assert example.split == "train"
    assert len(example.options) == 4
    assert example.example_id == compute_example_id(
        "choice", example.goal, example.state, [o.text for o in example.options]
    )
    assert result.report["reduced_10_to_8"] == 0
    assert result.report["reduction_seed"] == 13


def test_gold_accepts_letter_and_index_and_digit_string():
    for answer in ("B", 1, "1", "b"):
        result = ChoiceAdapter(seed=1).to_decision_example([aux_row(answer=answer)])
        assert len(result.examples) == 1, answer
        # Paris is index 1 in source order; find where it landed.
        example = result.examples[0]
        assert example.options[example.chosen_index].text == "Paris"


def test_gold_letter_out_of_range_and_garbage_are_rejected():
    assert ChoiceAdapter().to_decision_example([aux_row(answer="Z")]).rejected[0].reason_code == (
        "ADAPTOR_NO_GOLD"
    )
    assert ChoiceAdapter().to_decision_example([aux_row(answer=7)]).rejected[0].reason_code == (
        "ADAPTOR_BAD_INDEX"
    )
    assert ChoiceAdapter().to_decision_example([aux_row(answer=True)]).rejected[0].reason_code == (
        "ADAPTOR_NO_GOLD"
    )
    assert ChoiceAdapter().to_decision_example([aux_row(answer=None)]).rejected[0].reason_code == (
        "ADAPTOR_NO_GOLD"
    )


def test_option_labels_keep_source_letters():
    result = ChoiceAdapter(seed=5).to_decision_example([aux_row()])
    labels = sorted(o.label for o in result.examples[0].options)
    assert labels == ["A", "B", "C", "D"]


def test_mmlu_pro_ten_to_eight_reduction_reported_with_seed():
    adapter = MmluProAdapter(seed=11, reduction_seed=1234)
    result = adapter.to_decision_example([pro_row(answer="C")])
    assert result.report["reduced_10_to_8"] == 1
    assert result.report["reduction_seed"] == 1234
    assert result.report["reduction_method"] == "mmlu-pro-10-to-8"
    example, meta = result.examples[0], result.meta[0]
    assert len(example.options) == 8
    assert meta.reduction == "mmlu-pro-10-to-8"
    assert meta.reduction_seed is not None
    assert len(meta.dropped_indices) == 2
    assert meta.kept_indices is not None and len(meta.kept_indices) == 8
    # Gold is always retained: source index 2 (option-C) survives.
    assert 2 in meta.kept_indices
    assert example.options[example.chosen_index].text == "option-C"
    # Dropped indices are exactly the non-gold ones not kept.
    assert 2 not in meta.dropped_indices


def test_mmlu_pro_reduction_deterministic_and_seed_dependent():
    row = pro_row(answer="A")
    a = MmluProAdapter(seed=3, reduction_seed=100).to_decision_example([row])
    b = MmluProAdapter(seed=3, reduction_seed=100).to_decision_example([row])
    assert [m.dropped_indices for m in a.meta] == [m.dropped_indices for m in b.meta]
    assert a.examples[0].example_id == b.examples[0].example_id
    c = MmluProAdapter(seed=3, reduction_seed=200).to_decision_example([row])
    # Different seeds may coincide; every seed must retain the gold and land on 8.
    assert len(c.examples[0].options) == 8
    assert c.examples[0].options[c.examples[0].chosen_index].text == "option-A"


def test_mmlu_pro_defaults_to_test_split_and_eval_source():
    result = MmluProAdapter().to_decision_example([pro_row()])
    assert result.examples[0].split == "test"
    assert result.examples[0].source == "TIGER-Lab/MMLU-Pro"
    assert result.report["adapter"] == "mmlu-pro"


def test_row_split_overrides_default():
    row = aux_row()
    row["split"] = "dev"
    result = ChoiceAdapter().to_decision_example([row])
    assert result.examples[0].split == "dev"


def test_unmappable_rows_are_counted_and_rejected():
    rows = [
        aux_row(),
        {"question": "", "choices": ["a", "b"], "answer": 0},  # empty state
        {"question": "q", "choices": ["only-one"], "answer": 0},  # too few options
        {"choices": ["a", "b"], "answer": 0},  # missing question
        {"question": "q", "answer": 0},  # missing choices
        {"question": "q", "choices": ["a", "a"], "answer": 0},  # duplicate options
    ]
    result = ChoiceAdapter().to_decision_example(rows)
    codes = [r.reason_code for r in result.rejected]
    assert codes == [
        "ADAPTOR_EMPTY_STATE",
        "ADAPTOR_NO_OPTIONS",
        "ADAPTOR_MISSING_FIELD",
        "ADAPTOR_MISSING_FIELD",
        "DATA_DUPLICATE_OPTION",
    ]
    assert result.report["input_rows"] == 6
    assert result.report["converted"] == 1
    assert result.report["rejected"] == 5


def test_overlong_question_rejected_with_too_long():
    row = aux_row(question="x" * 100)
    result = ChoiceAdapter(max_state_chars=50).to_decision_example([row])
    assert result.rejected[0].reason_code == "ADAPTOR_TOO_LONG"


def test_option_order_shuffled_with_recorded_seed_and_reconstructable():
    adapter = ChoiceAdapter(seed=77)
    result = adapter.to_decision_example([pro_row(answer="B")])
    example, meta = result.examples[0], result.meta[0]
    assert meta.shuffle_seed is not None
    assert result.report["shuffle_seed"] == 77
    restored = restore_original_order(example, meta)
    kept = [f"option-{c}" for c in "ABCDEFGHIJ"]
    expected = [kept[i] for i in meta.kept_indices]
    assert [o.text for o in restored.options] == expected


def test_example_ids_are_unique_across_questions():
    rows = [aux_row(question=f"Question number {i}?") for i in range(5)]
    result = ChoiceAdapter(seed=9).to_decision_example(rows)
    ids = [e.example_id for e in result.examples]
    assert len(set(ids)) == 5


def test_fallback_group_id_is_deterministic_and_nonempty():
    rows = [{k: v for k, v in aux_row().items() if k != "id"}]
    first = ChoiceAdapter().to_decision_example(rows).examples[0].group_id
    second = ChoiceAdapter().to_decision_example(rows).examples[0].group_id
    assert first == second
    assert first.strip()


def test_nine_option_row_reduces_with_generic_method_name():
    row = aux_row()
    row["choices"] = [f"choice-{i}" for i in range(9)]
    result = ChoiceAdapter(seed=2).to_decision_example([row])
    assert result.report["reduced_over_8_to_8"] == 1
    assert result.report["reduced_10_to_8"] == 0
    assert result.report["reduced_other_to_8"] == 1
    assert result.meta[0].reduction == "seeded-non-gold-removal-to-8"
    assert len(result.examples[0].options) == 8
