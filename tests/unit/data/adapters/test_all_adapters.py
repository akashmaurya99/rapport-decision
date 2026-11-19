"""Cross-adapter contract checks: every adapter satisfies the shared rules."""

from __future__ import annotations

import json

import pytest

from src.data.adapters import ALL_ADAPTERS
from src.data.adapters.base import fingerprint_row
from src.data.core.serialize import from_json, to_json


def _rows():
    return {
        "choice": {
            "question": "Capital of France?",
            "choices": ["London", "Paris"],
            "answer": "B",
        },
        "mmlu-pro": {
            "question": "Capital of France?",
            "choices": [f"o{i}" for i in range(10)],
            "answer": "A",
        },
        "preference": {"chosen": "good answer", "rejected": "bad answer"},
        "scoring": {
            "prompt": "Explain entropy.",
            "response": "Entropy is disorder.",
            "helpfulness": 3,
            "correctness": 3,
            "coherence": 3,
        },
        "probability": {
            "question": "What is 2+2?",
            "choices": ["3", "4"],
            "answer": "B",
        },
        "abstention": {"subset": "s", "question": "Ungoogleable?", "answerable": False},
        "risk": {"prompt": "say something", "categories": ["hate"]},
    }


@pytest.mark.parametrize("adapter_cls", ALL_ADAPTERS, ids=lambda c: c.name)
def test_examples_carry_source_family_and_group_id(adapter_cls):
    adapter = adapter_cls(seed=3)
    result = adapter.to_decision_example([_rows()[adapter.name]])
    assert result.examples, adapter_cls.__name__
    for example in result.examples:
        assert example.source == adapter.source
        assert example.task_family == adapter.task_family
        assert example.group_id.strip()
        assert example.split in {"train", "dev", "test"}


@pytest.mark.parametrize("adapter_cls", ALL_ADAPTERS, ids=lambda c: c.name)
def test_report_has_required_counters(adapter_cls):
    adapter = adapter_cls(seed=3)
    result = adapter.to_decision_example([_rows()[adapter.name]])
    report = result.report
    for key in (
        "adapter",
        "source",
        "license",
        "task_family",
        "input_rows",
        "converted",
        "rejected",
        "shuffle_seed",
    ):
        assert key in report, key
    assert report["converted"] == len(result.examples)
    assert report["rejected"] == len(result.rejected)
    assert report["input_rows"] == len(result.examples) + len(result.rejected) or (
        # probability/preference expand rows into multiple examples
        adapter_cls.name in {"probability", "preference"}
    )


@pytest.mark.parametrize("adapter_cls", ALL_ADAPTERS, ids=lambda c: c.name)
def test_examples_roundtrip_through_the_serializer(adapter_cls):
    adapter = adapter_cls(seed=3)
    result = adapter.to_decision_example([_rows()[adapter.name]])
    for example in result.examples:
        restored = from_json(to_json(example))
        assert restored.example_id == example.example_id


@pytest.mark.parametrize("adapter_cls", ALL_ADAPTERS, ids=lambda c: c.name)
def test_garbage_rows_never_raise(adapter_cls):
    adapter = adapter_cls(seed=3)
    result = adapter.to_decision_example([None, 7, "text", {"x": "y"}])  # type: ignore[list-item]
    assert result.rejected, adapter_cls.__name__
    assert len(result.examples) + len(result.rejected) >= 4


@pytest.mark.parametrize("adapter_cls", ALL_ADAPTERS, ids=lambda c: c.name)
def test_meta_aligned_with_examples(adapter_cls):
    adapter = adapter_cls(seed=3)
    result = adapter.to_decision_example([_rows()[adapter.name]])
    assert len(result.meta) == len(result.examples)


@pytest.mark.parametrize("adapter_cls", ALL_ADAPTERS, ids=lambda c: c.name)
def test_no_silent_skips_total_accounting(adapter_cls):
    adapter = adapter_cls(seed=3)
    rows = [_rows()[adapter.name], None, _rows()[adapter.name]]  # type: ignore[list-item]
    result = adapter.to_decision_example(rows)
    accounted = len(result.rejected)
    assert accounted == result.report["rejected"]
    # every non-rejected row produced at least one example
    assert len(result.examples) >= 2


def test_fingerprint_row_is_order_insensitive_for_mappings():
    assert fingerprint_row({"a": 1, "b": 2}) == fingerprint_row({"b": 2, "a": 1})


def test_serialized_examples_have_no_adapter_meta_keys():
    # meta lives in AdapterResult, never inside the serialized record.
    adapter = ALL_ADAPTERS[0](seed=3)
    result = adapter.to_decision_example([_rows()[adapter.name]])
    blob = json.loads(to_json(result.examples[0]))
    assert "meta" not in blob
    assert "original_order" not in blob
