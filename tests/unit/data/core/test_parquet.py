"""Tests for src/data/core/parquet.py — optional pyarrow-backed serialization.

Skipped entirely when pyarrow is not importable: parquet support is optional,
never silently faked.
"""

from __future__ import annotations

import pytest

pyarrow = pytest.importorskip("pyarrow", reason="pyarrow is an optional dependency")

from src.data.core import (  # noqa: E402
    from_dict,
    normalize_example,
    read_parquet,
    write_parquet,
)
from src.data.core.serialize import to_dict  # noqa: E402


def _example(**overrides):
    record = {
        "task_family": "choice",
        "goal": "Choose the cheapest migration path",
        "state": "Three primaries across two regions.",
        "options": [
            {"text": "Lift and shift", "label": "A"},
            {"text": "Consolidate regions", "label": "B"},
        ],
        "chosen_index": 1,
        "scores": [0.3, 0.7],
        "answerable": True,
        "risk_labels": ["data_loss"],
        "source": "fixture",
        "license": "MIT",
        "group_id": "group-1",
        "split": "train",
    }
    record.update(overrides)
    return normalize_example(record)


def test_parquet_round_trip_preserves_everything(tmp_path):
    rich = _example()
    minimal = normalize_example(
        {
            "task_family": "preference",
            "goal": "Pick the better reply",
            "state": "",
            "options": [{"text": "Reply one"}, {"text": "Reply two"}],
            "chosen_index": 0,
            "source": "fixture",
            "license": "MIT",
            "group_id": "group-2",
            "split": "dev",
        }
    )
    path = tmp_path / "examples.parquet"
    write_parquet([rich, minimal], path)
    loaded = read_parquet(path)
    assert loaded == [rich, minimal]
    assert loaded[0].example_id == rich.example_id
    assert loaded[1].answerable is None


def test_parquet_row_is_one_row_per_example(tmp_path):
    examples = [
        _example(group_id=f"group-{i}")
        for i in range(3)
    ]
    path = tmp_path / "rows.parquet"
    write_parquet(examples, path)
    import pyarrow.parquet as pq

    assert pq.read_table(path).num_rows == 3


def test_parquet_round_trip_through_to_dict(tmp_path):
    example = _example()
    path = tmp_path / "dict_roundtrip.parquet"
    write_parquet([example], path)
    loaded = read_parquet(path)
    assert to_dict(loaded[0]) == to_dict(example)
    assert from_dict(to_dict(loaded[0])) == loaded[0]
