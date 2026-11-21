"""Shared-machinery tests: shuffle determinism, restore, result plumbing."""

from __future__ import annotations

import pytest

from src.data.adapters import (
    ADAPTOR_META_MISSING,
    AdapterError,
    ExampleMeta,
    RejectedRecord,
    restore_original_order,
    shuffle_order,
)
from src.data.adapters.base import fingerprint_row
from src.data.adapters.choice import MmluProAdapter


def _mmlu_row(answer: str = "B") -> dict:
    return {
        "question": "What is the capital of France?",
        "choices": ["London", "Paris", "Berlin", "Madrid"],
        "answer": answer,
        "id": "row-1",
    }


def test_rejected_record_shape():
    rec = RejectedRecord(3, "ADAPTOR_NO_GOLD", "detail")
    assert rec.record_index == 3
    assert rec.reason_code == "ADAPTOR_NO_GOLD"
    index, code = rec[:2]
    assert (index, code) == (3, "ADAPTOR_NO_GOLD")


def test_fingerprint_row_accepts_non_string_mapping_keys():
    mixed = {"a": 1, 2: "two", "nested": {3: "three"}}
    fingerprint = fingerprint_row(mixed)
    assert isinstance(fingerprint, str) and len(fingerprint) == 64
    assert fingerprint == fingerprint_row(mixed)
    # Coercion is stable regardless of dict insertion order.
    assert fingerprint == fingerprint_row({2: "two", "nested": {3: "three"}, "a": 1})


def test_shuffle_is_deterministic_per_row_content():
    fp = fingerprint_row({"a": 1})
    order_a, seed_a = shuffle_order(8, master_seed=13, row_fingerprint=fp)
    order_b, seed_b = shuffle_order(8, master_seed=13, row_fingerprint=fp)
    assert order_a == order_b
    assert seed_a == seed_b
    assert sorted(order_a) == list(range(8))


def test_shuffle_varies_with_seed_and_content():
    fp = fingerprint_row({"a": 1})
    orders = {
        tuple(shuffle_order(8, master_seed=seed, row_fingerprint=fp)[0])
        for seed in (1, 2, 3, 4, 5)
    }
    assert len(orders) > 1


def test_shuffle_independent_of_row_order():
    row1, row2 = {"q": "one"}, {"q": "two"}
    o1 = shuffle_order(4, master_seed=13, row_fingerprint=fingerprint_row(row1))[0]
    o2 = shuffle_order(4, master_seed=13, row_fingerprint=fingerprint_row(row2))[0]
    # Each row's order depends on its own fingerprint, not on batch position.
    again = shuffle_order(4, master_seed=13, row_fingerprint=fingerprint_row(row1))[0]
    assert o1 == again
    assert sorted(o1) == sorted(o2) == [0, 1, 2, 3]


def test_restore_original_order_roundtrip():
    adapter = MmluProAdapter(seed=99)
    result = adapter.to_decision_example([_mmlu_row()])
    assert len(result.examples) == 1
    example, meta = result.examples[0], result.meta[0]
    restored = restore_original_order(example, meta)
    assert [o.text for o in restored.options] == ["London", "Paris", "Berlin", "Madrid"]
    assert restored.chosen_index == 1
    assert restored.example_id == example.example_id  # id is order-invariant


def test_restore_requires_original_order():
    adapter = MmluProAdapter()
    result = adapter.to_decision_example([_mmlu_row()])
    example = result.examples[0]
    with pytest.raises(AdapterError) as excinfo:
        restore_original_order(example, ExampleMeta())
    assert excinfo.value.code == ADAPTOR_META_MISSING


def test_result_alignment_and_report_counts():
    result = MmluProAdapter(seed=7).to_decision_example([_mmlu_row(), {"question": ""}])
    assert result.report["input_rows"] == 2
    assert result.report["converted"] == len(result.examples) == len(result.meta)
    assert result.report["rejected"] == len(result.rejected) == 1


def test_non_mapping_rows_are_rejected_not_raised():
    adapter = MmluProAdapter()
    result = adapter.to_decision_example(["garbage", 42, None])  # type: ignore[list-item]
    assert len(result.examples) == 0
    assert [r.reason_code for r in result.rejected] == ["ADAPTOR_BAD_TYPE"] * 3
