"""Tests for src/calibration/buckets.py — bucket-key algebra."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration import buckets  # noqa: E402
from src.calibration.errors import CalibrationInputError  # noqa: E402


def test_option_count_bucket_mapping():
    assert buckets.option_count_bucket(2) == "2"
    assert buckets.option_count_bucket(3) == "3-5"
    assert buckets.option_count_bucket(4) == "3-5"
    assert buckets.option_count_bucket(5) == "3-5"
    assert buckets.option_count_bucket(6) == "6-8"
    assert buckets.option_count_bucket(7) == "6-8"
    assert buckets.option_count_bucket(8) == "6-8"


def test_option_count_bucket_rejects_out_of_contract():
    for bad in (1, 9, 0, -1, 100):
        with pytest.raises(CalibrationInputError):
            buckets.option_count_bucket(bad)
    with pytest.raises(CalibrationInputError):
        buckets.option_count_bucket(True)  # type: ignore[arg-type]
    with pytest.raises(CalibrationInputError):
        buckets.option_count_bucket(2.0)  # type: ignore[arg-type]


def test_declared_buckets_are_18():
    declared = buckets.declared_bucket_keys()
    assert len(declared) == 18
    assert len(set(declared)) == 18
    assert declared[0] == "choice|2"
    assert declared[-1] == "risk|6-8"
    # Six families x three option-count buckets, in deterministic order.
    assert declared == tuple(
        f"{family}|{bucket}"
        for family in buckets.TASK_FAMILIES
        for bucket in buckets.OPTION_COUNT_BUCKETS
    )


def test_make_and_parse_round_trip():
    for key in buckets.declared_bucket_keys():
        family, bucket = buckets.parse_bucket_key(key)
        assert buckets.make_bucket_key(family, bucket) == key
    assert buckets.make_bucket_key_for_count("choice", 7) == "choice|6-8"


def test_parse_rejects_malformed_keys():
    for bad in ("", "choice", "choice|", "|2", "unknown|2", "choice|4-6", "a|b|c", "choice|9"):
        with pytest.raises(CalibrationInputError):
            buckets.parse_bucket_key(bad)
    with pytest.raises(CalibrationInputError):
        buckets.parse_bucket_key(None)  # type: ignore[arg-type]


def test_make_bucket_key_validates():
    with pytest.raises(CalibrationInputError):
        buckets.make_bucket_key("unknown", "2")
    with pytest.raises(CalibrationInputError):
        buckets.make_bucket_key("choice", "4-6")
