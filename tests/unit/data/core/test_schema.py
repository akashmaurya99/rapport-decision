"""Tests for src/data/core/schema.py — frozen records and content identity."""

from __future__ import annotations

import hashlib
import json

import pytest

from src.data.core import (
    DecisionExample,
    Option,
    compute_example_id,
)
from src.data.core.errors import BadTypeError, IdMismatchError
from src.data.core.schema import SPLITS, TASK_FAMILIES

GOAL = "Choose the cheapest migration path"
STATE = "Three primaries across two regions, budget for two more instances."
TEXTS = ("Alpha path", "Beta path", "Gamma path", "Delta path")


def _id_for(texts, family="choice", goal=GOAL, state=STATE):
    return compute_example_id(family, goal, state, list(texts))


def test_contract_declares_six_families_and_three_splits():
    assert TASK_FAMILIES == frozenset(
        {"choice", "preference", "scoring", "probability", "abstention", "risk"}
    )
    assert SPLITS == frozenset({"train", "dev", "test"})


def test_example_id_is_option_order_invariant():
    from itertools import permutations

    base = _id_for(TEXTS)
    assert _id_for(list(reversed(TEXTS))) == base
    assert _id_for([TEXTS[2], TEXTS[0], TEXTS[3], TEXTS[1]]) == base
    for permutation in permutations(TEXTS):
        assert _id_for(list(permutation)) == base


def test_example_id_changes_with_state_goal_family_and_texts():
    base = _id_for(TEXTS)
    assert _id_for(TEXTS, state=STATE + " Extended.") != base
    assert _id_for(TEXTS, goal=GOAL + " Now.") != base
    assert _id_for(TEXTS, family="preference") != base
    assert _id_for([TEXTS[0], TEXTS[1], "Omega path", TEXTS[3]]) != base


def test_example_id_is_nfc_invariant():
    nfc_texts = ["café au lait", "thé complet"]
    nfd_texts = ["cafe\u0301 au lait", "the\u0301 complet"]
    assert _id_for(nfc_texts) == _id_for(nfd_texts)


def test_example_id_matches_manual_sha256():
    payload = ["rapport:decision-example:v1", "choice", GOAL, STATE, sorted(TEXTS)]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    assert _id_for(TEXTS) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_option_is_frozen_and_labeled():
    option = Option(text="Alpha path", label="A")
    with pytest.raises(AttributeError):
        option.text = "mutated"  # type: ignore[misc]
    assert option.label == "A"
    assert Option(text="x").label is None


def test_decision_example_is_frozen():
    example = _example(compute_example_id("choice", GOAL, STATE, TEXTS[:2]))
    with pytest.raises(AttributeError):
        example.goal = "mutated"  # type: ignore[misc]


def _example(example_id: str, **overrides) -> DecisionExample:
    values = {
        "example_id": example_id,
        "task_family": "choice",
        "goal": GOAL,
        "state": STATE,
        "options": tuple(Option(text=t) for t in TEXTS[:2]),
        "source": "fixture",
        "license": "MIT",
        "group_id": "group-1",
        "split": "train",
        "chosen_index": 0,
    }
    values.update(overrides)
    return DecisionExample(**values)


def test_lists_are_coerced_to_tuples():
    example_id = compute_example_id("choice", GOAL, STATE, TEXTS[:2])
    example = _example(
        example_id,
        options=[Option(text=t) for t in TEXTS[:2]],
        scores=[0.5, 0.5],
        risk_labels=["data_loss"],
    )
    assert isinstance(example.options, tuple)
    assert isinstance(example.scores, tuple)
    assert isinstance(example.risk_labels, tuple)
    assert example.scores == (0.5, 0.5)


def test_stale_example_id_raises_typed_error():
    with pytest.raises(IdMismatchError) as excinfo:
        _example("0" * 64)
    assert excinfo.value.code == "DATA_ID_MISMATCH"


def test_non_string_option_text_raises_typed_error():
    example_id = compute_example_id("choice", GOAL, STATE, TEXTS[:2])
    with pytest.raises(BadTypeError):
        _example(example_id, options=(Option(text=1), Option(text="b")))  # type: ignore[arg-type]
