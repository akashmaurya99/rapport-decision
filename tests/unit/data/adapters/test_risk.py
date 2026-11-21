"""Risk adapter: Aegis-style multi-label rows → the ``risk`` family."""

from __future__ import annotations

from src.data.adapters.risk import RiskAdapter


def aegis_row(categories=("hate", "violence"), prompt="Here is a prompt."):
    return {
        "prompt": prompt,
        "categories": list(categories),
        "id": "aegis-1",
    }


def test_labeled_row_preserves_risk_labels_and_source():
    result = RiskAdapter(seed=9).to_decision_example([aegis_row(("hate", "violence"))])
    example = result.examples[0]
    assert example.task_family == "risk"
    assert example.source == "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
    assert example.license == "CC-BY-4.0"
    assert example.group_id == "aegis-1"
    assert set(example.risk_labels) == {"hate", "violence"}
    assert example.chosen_index is None  # not a choice family
    assert result.report["labeled_rows"] == 1
    assert result.report["safe_rows"] == 0


def test_explicit_empty_categories_means_safe_with_empty_labels():
    result = RiskAdapter().to_decision_example([aegis_row(())])
    example = result.examples[0]
    assert example.risk_labels == ()
    assert result.report["safe_rows"] == 1


def test_absent_categories_field_rejected_not_treated_as_safe():
    row = {"prompt": "p", "id": "aegis-2"}
    result = RiskAdapter().to_decision_example([row])
    assert result.rejected[0].reason_code == "ADAPTOR_MISSING_FIELD"
    assert len(result.examples) == 0


def test_duplicate_labels_are_deduplicated_first_wins():
    result = RiskAdapter().to_decision_example([aegis_row(("hate", "hate", "violence"))])
    assert result.examples[0].risk_labels == ("hate", "violence")


def test_more_than_eight_labels_rejected_with_typed_reason():
    row = aegis_row(tuple(f"cat-{i}" for i in range(9)))
    result = RiskAdapter().to_decision_example([row])
    assert result.rejected[0].reason_code == "ADAPTOR_RISK_LABEL_LIMIT"


def test_eight_labels_are_accepted():
    row = aegis_row(tuple(f"cat-{i}" for i in range(8)))
    result = RiskAdapter().to_decision_example([row])
    assert len(result.examples) == 1
    assert len(result.examples[0].risk_labels) == 8


def test_options_are_safe_unsafe_polarity_and_shuffled():
    result = RiskAdapter(seed=4).to_decision_example([aegis_row()])
    example, meta = result.examples[0], result.meta[0]
    assert sorted(o.text for o in example.options) == ["safe", "unsafe"]
    assert meta.shuffle_seed is not None
    assert meta.original_order in ((0, 1), (1, 0))


def test_response_is_folded_into_state():
    row = aegis_row()
    row["response"] = "assistant answer"
    result = RiskAdapter().to_decision_example([row])
    assert "Response:" in result.examples[0].state


def test_empty_prompt_rejected():
    result = RiskAdapter().to_decision_example([aegis_row(prompt="  ")])
    assert result.rejected[0].reason_code == "ADAPTOR_EMPTY_STATE"


def test_non_list_categories_rejected():
    row = aegis_row()
    row["categories"] = "hate"
    result = RiskAdapter().to_decision_example([row])
    assert result.rejected[0].reason_code == "ADAPTOR_BAD_TYPE"


def test_case_variant_of_the_same_category_is_deduplicated():
    row = aegis_row(("Violence", "violence"))
    result = RiskAdapter().to_decision_example([row])
    assert len(result.examples) == 1
    assert result.examples[0].risk_labels == ("Violence",)


def test_non_string_label_rejected():
    row = aegis_row(("hate", 42))  # type: ignore[arg-type]
    result = RiskAdapter().to_decision_example([row])
    assert result.rejected[0].reason_code == "ADAPTOR_BAD_TYPE"
