"""Tests for SplitManifest: validation, JSON round-trip and fatal overlap."""

from __future__ import annotations

import json

import pytest
from conftest import make_example

from src.data.splits import (
    DEFAULT_OOD_SOURCE,
    RATIOS,
    SPLIT_NAMES,
    ManifestError,
    ManifestFieldError,
    ManifestFingerprintError,
    ManifestGroupOverlapError,
    SplitManifest,
    read_manifest,
    split,
    write_manifest,
)


def _build_manifest(**overrides) -> SplitManifest:
    manifest_kwargs = dict(
        fingerprint="a" * 64,
        counts={"choice": {"train": 7, "dev": 1, "test": 2}},
        group_overlap=0,
        dedup_threshold=0.9,
        ood_source=DEFAULT_OOD_SOURCE,
        withheld_source=DEFAULT_OOD_SOURCE,
        seed=13,
        totals={"train": 7, "dev": 1, "test": 2},
        ood_rows=3,
        ratios=dict(RATIOS),
    )
    manifest_kwargs.update(overrides)
    return SplitManifest(**manifest_kwargs)


def _build_manifest_dict(**overrides):
    data = _build_manifest().to_dict()
    data.update(overrides)
    return data


class TestConstruction:
    def test_valid_manifest_is_frozen_and_complete(self):
        manifest = _build_manifest()
        assert manifest.fingerprint == "a" * 64
        assert manifest.counts == {"choice": {"train": 7, "dev": 1, "test": 2}}
        assert manifest.group_overlap == 0
        assert manifest.dedup_threshold == 0.9
        assert manifest.ood_source == DEFAULT_OOD_SOURCE
        assert manifest.withheld_source == DEFAULT_OOD_SOURCE
        assert manifest.seed == 13
        assert manifest.totals == {"train": 7, "dev": 1, "test": 2}
        assert manifest.ood_rows == 3
        assert manifest.ratios == dict(RATIOS)
        with pytest.raises(Exception):
            manifest.group_overlap = 1  # type: ignore[misc]

    def test_positive_group_overlap_is_fatal_on_construction(self):
        with pytest.raises(ManifestGroupOverlapError) as excinfo:
            _build_manifest(group_overlap=2)
        assert excinfo.value.code == "SPLIT_MANIFEST_GROUP_OVERLAP"
        assert excinfo.value.overlap == 2

    def test_negative_group_overlap_is_fatal(self):
        with pytest.raises(ManifestGroupOverlapError):
            _build_manifest(group_overlap=-1)

    def test_non_integer_group_overlap_is_a_field_error(self):
        with pytest.raises(ManifestFieldError):
            _build_manifest(group_overlap="0")  # type: ignore[arg-type]
        with pytest.raises(ManifestFieldError):
            _build_manifest(group_overlap=True)  # type: ignore[arg-type]

    def test_fingerprint_must_be_sha256_hex(self):
        with pytest.raises(ManifestFingerprintError):
            _build_manifest(fingerprint="not-a-digest")
        with pytest.raises(ManifestFingerprintError):
            _build_manifest(fingerprint="A" * 64)  # uppercase is not canonical
        with pytest.raises(ManifestFingerprintError):
            _build_manifest(fingerprint="a" * 63)

    def test_dedup_threshold_domain(self):
        with pytest.raises(ManifestFieldError):
            _build_manifest(dedup_threshold=1.5)
        with pytest.raises(ManifestFieldError):
            _build_manifest(dedup_threshold=-0.1)
        with pytest.raises(ManifestFieldError):
            _build_manifest(dedup_threshold="0.9")  # type: ignore[arg-type]

    def test_dedup_threshold_is_stored_as_float(self):
        assert _build_manifest(dedup_threshold=1).dedup_threshold == 1.0  # type: ignore[arg-type]

    def test_ood_source_and_withheld_source_must_be_non_empty(self):
        with pytest.raises(ManifestFieldError):
            _build_manifest(ood_source="")
        with pytest.raises(ManifestFieldError):
            _build_manifest(withheld_source="  ")

    def test_seed_and_ood_rows_must_be_non_negative_ints(self):
        with pytest.raises(ManifestFieldError):
            _build_manifest(seed=-1)
        with pytest.raises(ManifestFieldError):
            _build_manifest(seed=True)  # type: ignore[arg-type]
        with pytest.raises(ManifestFieldError):
            _build_manifest(ood_rows=-1)

    def test_counts_must_have_exactly_train_dev_test_per_family(self):
        with pytest.raises(ManifestFieldError):
            _build_manifest(counts={"choice": {"train": 7, "dev": 1}})
        with pytest.raises(ManifestFieldError):
            _build_manifest(counts={"choice": {"train": 7, "dev": 1, "test": 2, "ood": 5}})

    def test_counts_must_be_non_negative_non_bool_ints(self):
        with pytest.raises(ManifestFieldError):
            _build_manifest(counts={"choice": {"train": -1, "dev": 1, "test": 2}})
        with pytest.raises(ManifestFieldError):
            _build_manifest(counts={"choice": {"train": True, "dev": 1, "test": 2}})  # type: ignore[dict-item]

    def test_totals_must_have_exactly_train_dev_test(self):
        with pytest.raises(ManifestFieldError):
            _build_manifest(totals={"train": 7, "dev": 1})

    def test_ratios_must_sum_to_one(self):
        with pytest.raises(ManifestFieldError):
            _build_manifest(ratios={"train": 0.7, "dev": 0.15, "test": 0.10})

    def test_ratios_must_cover_train_dev_test(self):
        with pytest.raises(ManifestFieldError):
            _build_manifest(ratios={"train": 0.7, "dev": 0.3})


class TestStrictLoad:
    def test_missing_required_field_is_fatal(self):
        data = _build_manifest_dict()
        del data["fingerprint"]
        with pytest.raises(ManifestFieldError) as excinfo:
            SplitManifest.from_dict(data)
        assert excinfo.value.field == "fingerprint"

    def test_unknown_field_is_fatal(self):
        data = _build_manifest_dict(surprise="boo")
        with pytest.raises(ManifestFieldError):
            SplitManifest.from_dict(data)

    def test_positive_overlap_in_stored_manifest_is_fatal(self):
        data = _build_manifest_dict(group_overlap=4)
        with pytest.raises(ManifestGroupOverlapError) as excinfo:
            SplitManifest.from_dict(data)
        assert excinfo.value.overlap == 4

    def test_from_json_rejects_non_object_payloads(self):
        with pytest.raises(ManifestError):
            SplitManifest.from_json("[1, 2, 3]")
        with pytest.raises(ManifestError):
            SplitManifest.from_json("not json {")

    def test_from_dict_rejects_non_mapping(self):
        with pytest.raises(ManifestFieldError):
            SplitManifest.from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_all_required_keys_are_required(self):
        for key in (
            "fingerprint",
            "counts",
            "group_overlap",
            "dedup_threshold",
            "ood_source",
            "withheld_source",
            "seed",
            "totals",
            "ood_rows",
            "ratios",
        ):
            data = _build_manifest_dict()
            del data[key]
            with pytest.raises(ManifestError):
                SplitManifest.from_dict(data)


class TestJsonRoundTrip:
    def test_to_json_from_json_round_trip(self):
        manifest = _build_manifest()
        restored = SplitManifest.from_json(manifest.to_json())
        assert restored == manifest

    def test_to_dict_from_dict_round_trip(self):
        manifest = _build_manifest()
        assert SplitManifest.from_dict(manifest.to_dict()) == manifest

    def test_to_json_is_deterministic(self):
        manifest = _build_manifest()
        assert manifest.to_json() == manifest.to_json()
        assert manifest.to_json(indent=None) == manifest.to_json(indent=None)

    def test_to_dict_emits_counts_sorted_by_family(self):
        manifest = _build_manifest(
            counts={
                "risk": {"train": 1, "dev": 0, "test": 0},
                "choice": {"train": 2, "dev": 1, "test": 0},
            }
        )
        assert list(manifest.to_dict()["counts"]) == ["choice", "risk"]


class TestFileHelpers:
    def test_write_then_read_round_trip(self, tmp_path):
        examples = [make_example(uid=7000 + uid, group_id=f"g-{uid}") for uid in range(8)]
        result = split(examples, seed=13)
        path = tmp_path / "nested" / "split_manifest.json"
        write_manifest(result.manifest, path)
        assert read_manifest(path) == result.manifest

    def test_read_manifest_rejects_positive_overlap_on_disk(self, tmp_path):
        payload = _build_manifest_dict(group_overlap=3)
        path = tmp_path / "split_manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ManifestGroupOverlapError):
            read_manifest(path)

    def test_read_manifest_missing_file_is_a_typed_error(self, tmp_path):
        with pytest.raises(ManifestError):
            read_manifest(tmp_path / "absent.json")

    def test_written_file_is_valid_json_with_required_keys(self, tmp_path):
        examples = [make_example(uid=7100 + uid, group_id=f"g-{uid}") for uid in range(6)]
        result = split(examples, seed=13)
        path = tmp_path / "split_manifest.json"
        write_manifest(result.manifest, path)
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "fingerprint",
            "counts",
            "group_overlap",
            "dedup_threshold",
            "ood_source",
            "withheld_source",
        ):
            assert key in data
        assert data["group_overlap"] == 0


class TestRatiosAndSplitNames:
    def test_ratios_match_the_seventy_fifteen_fifteen_contract(self):
        assert RATIOS == {"train": 0.70, "dev": 0.15, "test": 0.15}

    def test_split_names(self):
        assert SPLIT_NAMES == ("train", "dev", "test")
