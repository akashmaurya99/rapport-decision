"""Tests for license resolution — the admission gate (FR-DATA-02).

No network anywhere: Hub metadata is passed in as stubs/mappings.
"""

from __future__ import annotations

import pytest

from src.data.acquire import (
    LicenseResolutionError,
    declared_rows_from_metadata,
    resolve_license,
)


class TestMappingResolution:
    def test_dict_metadata_with_license(self):
        resolution = resolve_license(
            "cais/mmlu", hub_metadata={"card_data": {"license": "MIT"}}
        )
        assert resolution.license == "MIT"
        assert resolution.origin == "hub_metadata"

    def test_dict_metadata_top_level_license(self):
        resolution = resolve_license("x", hub_metadata={"license": "apache-2.0"})
        assert resolution.license == "apache-2.0"

    def test_dict_metadata_top_level_beats_card_data(self):
        resolution = resolve_license(
            "x", hub_metadata={"license": "mit", "card_data": {"license": "apache-2.0"}}
        )
        assert resolution.license == "mit"

    def test_dict_metadata_null_license_is_missing(self):
        resolution = resolve_license("x", hub_metadata={"card_data": {"license": None}})
        assert resolution.license is None
        assert resolution.origin == "hub_metadata"

    def test_licenses_mapping_wins_over_hub(self):
        resolution = resolve_license(
            "x",
            hub_metadata={"license": "apache-2.0"},
            licenses={"x": "MIT"},
        )
        assert resolution.license == "MIT"
        assert resolution.origin == "licenses_mapping"

    def test_licenses_mapping_none_is_explicit_unlicensed(self):
        resolution = resolve_license("x", licenses={"x": None})
        assert resolution.license is None
        assert resolution.origin == "licenses_mapping"

    def test_licenses_mapping_missing_source_falls_back_to_hub(self):
        resolution = resolve_license(
            "other",
            hub_metadata={"license": "mit"},
            licenses={"x": "MIT"},
        )
        assert resolution.license == "mit"
        assert resolution.origin == "hub_metadata"

    def test_licenses_mapping_missing_source_and_no_hub_is_missing(self):
        resolution = resolve_license("other", licenses={"x": "MIT"})
        assert resolution.license is None
        assert resolution.origin == "missing"


class TestObjectResolution:
    def test_duck_typed_hub_metadata(self, fake_hub, fake_card):
        resolution = resolve_license("x", hub_metadata=fake_hub(license="MIT"))
        assert resolution.license == "MIT"
        assert resolution.origin == "hub_metadata"

    def test_card_data_with_none_license(self, fake_hub):
        resolution = resolve_license("x", hub_metadata=fake_hub(license=None))
        assert resolution.license is None

    def test_hub_metadata_without_card_data(self, fake_hub):
        resolution = resolve_license("x", hub_metadata=fake_hub(declare_license_slot=False))
        assert resolution.license is None

    def test_direct_license_attribute(self, fake_card):
        resolution = resolve_license("x", hub_metadata=fake_card(license="apache-2.0"))
        assert resolution.license == "apache-2.0"

    def test_multi_license_list_is_joined(self, fake_hub):
        resolution = resolve_license("x", hub_metadata=fake_hub(license=["mit", "apache-2.0"]))
        assert resolution.license == "mit OR apache-2.0"

    def test_empty_license_list_is_missing(self, fake_hub):
        resolution = resolve_license("x", hub_metadata=fake_hub(license=[]))
        assert resolution.license is None

    @pytest.mark.parametrize("sentinel", ["", "unknown", "UNKNOWN", "None", "n/a", "null", "  "])
    def test_sentinel_strings_mean_missing(self, sentinel):
        resolution = resolve_license("x", hub_metadata={"license": sentinel})
        assert resolution.license is None

    def test_whitespace_padding_is_stripped(self):
        resolution = resolve_license("x", hub_metadata={"license": "  MIT  "})
        assert resolution.license == "MIT"

    def test_uninterpretable_metadata_raises_typed_error(self):
        with pytest.raises(LicenseResolutionError) as excinfo:
            resolve_license("x", hub_metadata=object())
        assert excinfo.value.code == "DATA_LICENSE_RESOLUTION_ERROR"

    def test_uninterpretable_card_data_raises_typed_error(self):
        class Card:
            pass

        class Hub:
            card_data = Card()

        with pytest.raises(LicenseResolutionError):
            resolve_license("x", hub_metadata=Hub())

    def test_non_string_license_value_raises_typed_error(self):
        with pytest.raises(LicenseResolutionError):
            resolve_license("x", hub_metadata={"license": 42})

    def test_empty_source_raises(self):
        with pytest.raises(LicenseResolutionError):
            resolve_license("", hub_metadata={"license": "mit"})


class TestDeclaredRows:
    def test_num_rows_attribute(self, fake_hub):
        assert declared_rows_from_metadata(fake_hub(license="MIT", num_rows=231_400)) == 231_400

    def test_num_rows_in_mapping(self):
        assert declared_rows_from_metadata({"num_rows": "169352"}) == 169_352
        assert declared_rows_from_metadata({"num_examples": 385_000}) == 385_000

    def test_num_rows_nested_in_card_data_mapping(self):
        assert declared_rows_from_metadata({"card_data": {"num_rows": 231_400}}) == 231_400
        assert declared_rows_from_metadata({"card_data": {"license": "mit"}}) is None

    def test_missing_or_garbage_declared_rows(self):
        assert declared_rows_from_metadata(None) is None
        assert declared_rows_from_metadata({}) is None
        assert declared_rows_from_metadata({"num_rows": "lots"}) is None
        assert declared_rows_from_metadata({"num_rows": -5}) is None
        assert declared_rows_from_metadata({"num_rows": True}) is None


class TestRealHubClasses:
    """Exercise the resolution path against the real huggingface_hub types.

    Constructed in-process from payloads — no network call is made.
    """

    def test_datasetinfo_with_license(self):
        datasets = pytest.importorskip("huggingface_hub", reason="huggingface_hub present")
        info = datasets.DatasetInfo(id="x/y", author="x", card_data={"license": "mit"})
        resolution = resolve_license("x/y", hub_metadata=info)
        assert resolution.license == "mit"
        assert resolution.origin == "hub_metadata"

    def test_datasetinfo_without_license(self):
        datasets = pytest.importorskip("huggingface_hub", reason="huggingface_hub present")
        info = datasets.DatasetInfo(id="x/y", author="x", card_data={})
        assert resolve_license("x/y", hub_metadata=info).license is None
        info_no_card = datasets.DatasetInfo(id="x/y", author="x")
        assert resolve_license("x/y", hub_metadata=info_no_card).license is None


class TestRealHubShapes:
    """Mirror huggingface_hub.DatasetInfo without importing it."""

    def test_datasetinfo_like_object(self):
        from dataclasses import dataclass

        @dataclass
        class CardData:
            license: str | None = None

        @dataclass
        class DatasetInfo:
            card_data: CardData | None

        assert resolve_license("x", hub_metadata=DatasetInfo(CardData("MIT"))).license == "MIT"
        assert resolve_license("x", hub_metadata=DatasetInfo(None)).license is None
        assert resolve_license("x", hub_metadata=DatasetInfo(CardData())).license is None
