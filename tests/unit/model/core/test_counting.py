"""Tests for src/model/core/counting.py — per-component accounting vs the spec."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.model.core import (
    ParameterCountMismatchError,
    ParameterReport,
    RapportConfig,
    RapportTrunk,
    assert_within_spec,
    count_parameters,
    expected_parameter_counts,
)


def test_compact_trunk_counts_match_spec_exactly():
    config = RapportConfig.compact()
    report = count_parameters(RapportTrunk(config), spec=config)
    assert report.trunk == 21_242_880
    assert report.components["attention"] == 7_077_888
    assert report.components["swiglu"] == 14_155_776
    assert report.components["norms"] == 9_216
    assert report.components["embedding"] == 11_720_448
    assert report.total == 32_963_328


def test_ultra_and_large_trunks_count_within_spec():
    for name in ("ultra", "large"):
        config = getattr(RapportConfig, name)()
        report = count_parameters(RapportTrunk(config), spec=config)
        expected = expected_parameter_counts(config)
        assert abs(report.trunk - expected["trunk"]) <= 0.01 * expected["trunk"]
        assert report.trunk == config.spec.trunk


def test_counter_fails_the_build_when_spec_and_implementation_disagree():
    compact = RapportConfig.compact()
    eleven_layers = replace(compact, layers=11)  # implementation drifted from the spec
    with pytest.raises(ParameterCountMismatchError) as excinfo:
        count_parameters(RapportTrunk(eleven_layers), spec=compact)
    message = str(excinfo.value)
    assert "trunk" in message and "attention" in message


def test_counter_fails_when_spec_constant_is_wrong():
    compact = RapportConfig.compact()
    tampered = replace(compact, spec=replace(compact.spec, swiglu_total=1_000_000))
    with pytest.raises(ParameterCountMismatchError) as excinfo:
        count_parameters(RapportTrunk(compact), spec=tampered)
    assert "swiglu" in str(excinfo.value)


def test_assert_within_spec_tolerance_boundary():
    compact = RapportConfig.compact()
    base = expected_parameter_counts(compact)

    def report_with(attention: int) -> ParameterReport:
        components = {
            "embedding": base["embedding"],
            "attention": attention,
            "swiglu": base["swiglu"],
            "norms": base["norms"],
            "other": 0,
        }
        trunk = attention + base["swiglu"] + base["norms"]
        return ParameterReport(components=components, trunk=trunk, total=trunk + base["embedding"])

    within = report_with(round(base["attention"] * 1.009))
    assert_within_spec(within, compact.spec)  # +0.9% — inside the 1% band
    outside = report_with(round(base["attention"] * 1.02))
    with pytest.raises(ParameterCountMismatchError):
        assert_within_spec(outside, compact.spec)


def test_count_parameters_without_spec_never_raises():
    compact = RapportConfig.compact()
    eleven_layers = replace(compact, layers=11)
    report = count_parameters(RapportTrunk(eleven_layers))
    assert report.trunk != compact.spec.trunk


def test_report_rows_exposes_trunk_and_total():
    compact = RapportConfig.compact()
    report = count_parameters(RapportTrunk(compact))
    rows = report.rows()
    assert rows["trunk"] == 21_242_880
    assert rows["total"] == rows["trunk"] + rows["embedding"]
