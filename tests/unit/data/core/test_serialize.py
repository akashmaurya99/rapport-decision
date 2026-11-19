"""Tests for src/data/core/serialize.py and the fixture round-trip.

The fixture at ``fixtures/canonical_example.json`` must round-trip through
parse → validate → serialize with an identical ``example_id``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.core import (
    from_dict,
    from_json,
    normalize_example,
    read_json,
    to_dict,
    to_json,
    validate_example,
    write_json,
)
from src.data.core.errors import DataError, IdMismatchError, JsonInvalidError

FIXTURES = Path(__file__).parent / "fixtures"
CANONICAL = FIXTURES / "canonical_example.json"

# Golden hash of the fixture's (task_family, goal, state, sorted option texts).
EXPECTED_ID = "555154b59729bb5cd9d9d52f556d6f09ad63a91cbecfe54dbfb4e579f73015a6"


def _fixture_raw() -> dict:
    return json.loads(CANONICAL.read_text(encoding="utf-8"))


def test_fixture_exists_and_parses_with_identical_hash():
    example = from_json(CANONICAL.read_text(encoding="utf-8"))
    assert example.example_id == EXPECTED_ID
    assert example.example_id == _fixture_raw()["example_id"]


def test_fixture_round_trip_parse_validate_serialize():
    raw = _fixture_raw()
    assert validate_example(raw) == []
    example = from_dict(raw)
    serialized = to_dict(example)
    assert serialized["example_id"] == EXPECTED_ID
    reparsed = from_dict(serialized)
    assert reparsed == example
    assert to_json(reparsed) == to_json(example)


def test_fixture_json_file_round_trip(tmp_path):
    example = read_json(CANONICAL)
    target = tmp_path / "copy.json"
    write_json(example, target)
    reread = read_json(target)
    assert reread == example
    assert reread.example_id == EXPECTED_ID


def test_to_dict_omits_absent_optionals():
    example = normalize_example(
        {
            "task_family": "abstention",
            "goal": "Decide whether to answer",
            "state": "No context is available at all.",
            "options": [{"text": "Answer"}, {"text": "Abstain"}],
            "source": "fixture",
            "license": "MIT",
            "group_id": "group-9",
            "split": "dev",
        }
    )
    data = to_dict(example)
    assert "chosen_index" not in data
    assert "scores" not in data
    assert "answerable" not in data
    assert "risk_labels" not in data
    assert set(data) == {
        "example_id",
        "task_family",
        "goal",
        "state",
        "options",
        "source",
        "license",
        "group_id",
        "split",
    }


def test_to_dict_includes_present_optionals():
    example = from_dict(_fixture_raw())
    data = to_dict(example)
    assert data["chosen_index"] == 1
    assert data["scores"] == [0.2, 0.65, 0.15]
    assert data["answerable"] is True
    assert data["risk_labels"] == ["data_loss"]
    assert data["options"][0]["label"] == "A"


def test_option_without_label_serializes_without_label():
    example = normalize_example(
        {
            "task_family": "choice",
            "goal": "Pick one",
            "state": "Two options are available.",
            "options": [{"text": "Alpha"}, {"text": "Beta"}],
            "chosen_index": 0,
            "source": "fixture",
            "license": "MIT",
            "group_id": "group-10",
            "split": "test",
        }
    )
    for option in to_dict(example)["options"]:
        assert "label" not in option


def test_from_dict_rejects_unknown_keys():
    raw = _fixture_raw()
    raw["provenance"] = "should not be accepted"
    with pytest.raises(DataError) as excinfo:
        from_dict(raw)
    assert excinfo.value.code == "DATA_UNKNOWN_FIELD"


def test_from_dict_rejects_tampered_hash():
    raw = _fixture_raw()
    raw["example_id"] = "f" * 64
    with pytest.raises(IdMismatchError):
        from_dict(raw)


def test_from_json_rejects_invalid_json():
    with pytest.raises(JsonInvalidError) as excinfo:
        from_json("{not json")
    assert excinfo.value.code == "DATA_JSON_INVALID"


def test_from_json_rejects_non_object_top_level():
    with pytest.raises(JsonInvalidError) as excinfo:
        from_json("[1, 2, 3]")
    assert excinfo.value.code == "DATA_JSON_INVALID"


def test_to_json_preserves_non_ascii():
    example = normalize_example(
        {
            "task_family": "choice",
            "goal": "Choisissez le chemin le moins cher",
            "state": "Budget serré, équipe réduite.",
            "options": [{"text": "Voie une"}, {"text": "Voie deux"}],
            "chosen_index": 0,
            "source": "fixture",
            "license": "MIT",
            "group_id": "group-11",
            "split": "train",
        }
    )
    payload = to_json(example)
    assert "é" in payload
    assert from_json(payload) == example
