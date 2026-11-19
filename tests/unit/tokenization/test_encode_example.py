"""FR-TOK-02 (budget half): encode_example honours the 512-token budget."""

from __future__ import annotations

import itertools

import pytest

from src.data.core.normalize import normalize_example
from src.data.core.schema import DecisionExample
from src.tokenization import (
    DEFAULT_TOKENIZER,
    SPECIAL_TOKEN_IDS,
    TRUNCATION_POLICY,
    bucket_examples,
    encode_example,
    reset_truncation_counts,
    truncation_counts,
)
from src.tokenization import errors as tok_errors

_GROUP_COUNTER = itertools.count()


def make_example(
    *,
    goal: str = "Pick the cheapest migration path",
    state: str = "Three Postgres primaries across two regions and the budget allows two more.",
    option_texts: tuple[str, ...] = (
        "Lift and shift both regions",
        "Consolidate to one region and add read replicas",
    ),
    source: str = "fixture/test",
) -> DecisionExample:
    return normalize_example(
        {
            "task_family": "choice",
            "goal": goal,
            "state": state,
            "options": [{"text": text, "label": None} for text in option_texts],
            "source": source,
            "license": "CC-BY-4.0",
            "group_id": f"group-{next(_GROUP_COUNTER)}",
            "split": "train",
            "chosen_index": 0,
        }
    )


def long_state_example(source: str = "fixture/long", state_words: int = 1200) -> DecisionExample:
    state = " ".join(f"constraint{i} holds for the workload" for i in range(state_words // 6))
    return make_example(state=state, source=source)


def setup_function() -> None:
    reset_truncation_counts()


def test_short_example_roundtrips_without_truncation() -> None:
    example = make_example()
    encoded = encode_example(example)
    assert encoded.truncated is False
    assert len(encoded.input_ids) <= 512
    assert len(encoded.attention_mask) == len(encoded.input_ids)
    assert all(mask == 1 for mask in encoded.attention_mask)
    assert encoded.source == example.source
    assert encoded.truncation_policy == TRUNCATION_POLICY
    assert truncation_counts() == {}


def test_joint_sequence_layout_markers_in_order() -> None:
    example = make_example()
    encoded = encode_example(example)
    tok = DEFAULT_TOKENIZER
    markers = [
        tok.id_to_token[i]
        for i in encoded.input_ids
        if i in SPECIAL_TOKEN_IDS.values()
    ]
    assert markers == ["<task>", "<goal>", "<state>", "<opt>", "</opt>", "<opt>", "</opt>"]


def test_long_state_is_truncated_from_the_end_at_512() -> None:
    example = long_state_example()
    encoded = encode_example(example)
    assert len(encoded.input_ids) == 512
    assert encoded.truncated is True
    # The input record is frozen and never mutated.
    assert type(encoded.example) is DecisionExample
    assert not hasattr(encoded.example, "truncated")


def test_truncation_keeps_goal_prefix_and_option_tail_intact() -> None:
    example = long_state_example()
    encoded = encode_example(example)
    tok = DEFAULT_TOKENIZER

    head = tok.encode(f"<task>{example.task_family}<goal>")
    goal_ids = tok.encode(example.goal)
    state_marker = [4]
    options_ids = tok.encode(
        "".join(f"<opt>{o.text}</opt>" for o in example.options)
    )
    prefix = [*head, *goal_ids, *state_marker]
    assert list(encoded.input_ids[: len(prefix)]) == prefix
    assert list(encoded.input_ids[-len(options_ids):]) == options_ids


def test_truncation_is_flagged_per_source_in_the_counter() -> None:
    first = long_state_example(source="fixture/alpha")
    second = long_state_example(source="fixture/beta")
    third = long_state_example(source="fixture/alpha")
    for example in (first, second, third, make_example(source="fixture/alpha")):
        encode_example(example)
    assert truncation_counts() == {"fixture/alpha": 2, "fixture/beta": 1}


def test_truncation_count_is_written_into_the_manifest_mapping() -> None:
    example = long_state_example(source="fixture/manifest")
    manifest: dict[str, object] = {}
    encoded = encode_example(example, manifest=manifest)
    assert encoded.truncated is True
    assert manifest["truncations_per_source"] == {"fixture/manifest": 1}
    assert manifest["truncation_policy"] == TRUNCATION_POLICY


def test_untruncated_call_still_records_the_policy_in_the_manifest() -> None:
    manifest: dict[str, object] = {}
    encode_example(make_example(), manifest=manifest)
    assert manifest["truncation_policy"] == TRUNCATION_POLICY
    assert "truncations_per_source" not in manifest


def test_smaller_budget_is_respected_exactly() -> None:
    example = long_state_example()
    encoded = encode_example(example, budget=64)
    assert len(encoded.input_ids) == 64
    assert encoded.truncated is True


def test_huge_goal_falls_back_to_truncating_the_goal_option_tail_intact() -> None:
    # Goal + options nearly fill the budget: the state vanishes and the goal
    # is truncated from the end; the option suffix survives.
    example = make_example(
        goal="decide " + " ".join(f"consideration{i}" for i in range(600)),
        state="small state",
        option_texts=("option alpha", "option beta"),
    )
    encoded = encode_example(example, budget=256)
    assert len(encoded.input_ids) <= 256
    assert encoded.truncated is True
    tok = DEFAULT_TOKENIZER
    options_ids = tok.encode(
        "".join(f"<opt>{o.text}</opt>" for o in example.options)
    )
    assert list(encoded.input_ids[-len(options_ids):]) == options_ids


def test_pathological_option_block_overflow_never_exceeds_the_budget() -> None:
    example = make_example(
        goal="pick one",
        state="tiny state",
        option_texts=tuple(f"option number {i} " + "filler " * 200 for i in range(8)),
    )
    encoded = encode_example(example, budget=64)
    assert len(encoded.input_ids) <= 64
    assert encoded.truncated is True


def test_budget_validation() -> None:
    example = make_example()
    with pytest.raises(tok_errors.BudgetTooSmallError):
        encode_example(example, budget=0)
    with pytest.raises(tok_errors.BudgetTooSmallError):
        encode_example(example, budget=True)  # type: ignore[arg-type]


def test_module_counter_is_reset_between_tests() -> None:
    encode_example(long_state_example(source="fixture/reset"))
    assert truncation_counts().get("fixture/reset") == 1
    reset_truncation_counts()
    assert truncation_counts() == {}


def test_truncated_flag_lives_on_the_encoded_copy_not_the_input() -> None:
    example = long_state_example()
    before = example
    encoded = encode_example(example)
    assert encoded.truncated is True
    assert example is before
    assert example == before  # frozen record untouched
    # Same length, same fields, same identity semantics as a fresh encode.
    assert encoded.example == before


def test_bucket_examples_accepts_encoded_examples_without_recounting() -> None:
    example = long_state_example(source="fixture/precounted")
    encoded = encode_example(example)
    reset_truncation_counts()
    bucket_examples([encoded])
    assert truncation_counts() == {}
