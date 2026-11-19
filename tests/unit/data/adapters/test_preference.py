"""Preference adapter: HH-RLHF chosen/rejected, both orders."""

from __future__ import annotations

from src.data.adapters.preference import PreferenceAdapter


def hh_row(prompt: str = "Human: How do I boil an egg?"):
    return {
        "prompt": prompt,
        "chosen": "Boil water for ten minutes, then cool the eggs.",
        "rejected": "Just microwave it.",
        "id": "hh-1",
    }


def transcript_row():
    return {
        "chosen": "\n\nHuman: How do I boil an egg?\n\nAssistant: Boil water for ten minutes.",
        "rejected": "\n\nHuman: How do I boil an egg?\n\nAssistant: Just microwave it.",
        "id": "hh-2",
    }


def test_both_orders_emitted_chosen_at_zero_in_order_A():
    result = PreferenceAdapter().to_decision_example([hh_row()])
    assert len(result.examples) == 2
    first, second = result.examples
    assert [o.text for o in first.options] == [
        "Boil water for ten minutes, then cool the eggs.",
        "Just microwave it.",
    ]
    assert first.chosen_index == 0
    assert [o.text for o in second.options] == [
        "Just microwave it.",
        "Boil water for ten minutes, then cool the eggs.",
    ]
    assert second.chosen_index == 1
    # Both emissions share group_id so the pair can never straddle a split.
    assert first.group_id == second.group_id == "hh-1"
    assert result.report["both_orders_emitted"] == 2
    assert result.report["pairs_converted"] == 1


def test_transcript_prefix_is_extracted_as_state():
    result = PreferenceAdapter().to_decision_example([transcript_row()])
    assert len(result.examples) == 2
    for example in result.examples:
        assert example.state == "Human: How do I boil an egg?"
    assert [o.text for o in result.examples[0].options] == [
        "Boil water for ten minutes.",
        "Just microwave it.",
    ]


def test_unparseable_transcripts_fall_back_to_empty_state():
    row = {
        "chosen": "alpha response",
        "rejected": "beta response",
        "id": "hh-x",
    }
    result = PreferenceAdapter().to_decision_example([row])
    assert len(result.examples) == 2
    assert all(e.state == "" for e in result.examples)  # allowed for preference


def test_deterministic_orders_no_rng_shuffle():
    result = PreferenceAdapter(seed=123).to_decision_example([hh_row(), hh_row()])
    metas = result.meta
    assert all(m.shuffle_seed is None for m in metas)
    assert [m.original_order for m in metas] == [(0, 1), (1, 0), (0, 1), (1, 0)]
    assert result.report["shuffle"] == "deterministic-both-orders"


def test_shared_group_id_derived_when_missing_is_stable():
    row = {k: v for k, v in hh_row().items() if k != "id"}
    a = PreferenceAdapter().to_decision_example([row])
    b = PreferenceAdapter().to_decision_example([row])
    assert a.examples[0].group_id == b.examples[0].group_id
    assert a.examples[0].group_id == a.examples[1].group_id


def test_missing_fields_rejected_with_typed_codes():
    result = PreferenceAdapter().to_decision_example(
        [{"chosen": "only chosen"}, {"rejected": "only rejected"}]
    )
    assert [r.reason_code for r in result.rejected] == [
        "ADAPTOR_MISSING_FIELD",
        "ADAPTOR_MISSING_FIELD",
    ]
    assert result.report["converted"] == 0


def test_identical_responses_rejected_once_as_duplicate_options():
    row = hh_row()
    row["rejected"] = row["chosen"]
    result = PreferenceAdapter().to_decision_example([row])
    assert len(result.rejected) == 1
    assert result.rejected[0].record_index == 0
    assert result.rejected[0].reason_code == "DATA_DUPLICATE_OPTION"
    assert len(result.examples) == 0


def test_overlong_state_is_truncated_and_flagged():
    row = hh_row(prompt=" ".join(["word"] * 500))
    result = PreferenceAdapter(max_state_chars=100).to_decision_example([row])
    assert result.report["truncated_rows"] == 1
    assert all(len(e.state) <= 100 for e in result.examples)
    assert all(m.truncated for m in result.meta)


def test_family_source_license_and_goal():
    result = PreferenceAdapter().to_decision_example([hh_row()])
    for example in result.examples:
        assert example.task_family == "preference"
        assert example.source == "Anthropic/hh-rlhf"
        assert example.license == "MIT"
        assert example.goal == "Pick the response that better satisfies the request."
