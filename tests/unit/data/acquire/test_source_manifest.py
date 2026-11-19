"""Tests for SourceManifest serialization and integrity."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.data.acquire import (
    ManifestInvalidError,
    SourceManifest,
    compute_shard_set_sha256,
    load_manifest,
    sha256_file,
    write_manifest,
)
from src.data.acquire.manifest import MANIFEST_FILENAME

FIXED = datetime(2026, 11, 19, 12, 0, 0, tzinfo=timezone.utc)


def make_manifest(**overrides) -> SourceManifest:
    payload = {
        "source": "cais/mmlu",
        "license": "MIT",
        "declared_rows": 231_400,
        "observed_rows": 30_000,
        "sha256": "0" * 64,
        "acquired_at": FIXED.isoformat(timespec="seconds"),
        "status": "admitted",
        "shards": [],
        "shard_sha256": {},
        "shard_bytes": {},
        "quarantine_reason": None,
        "license_origin": "hub_metadata",
    }
    payload.update(overrides)
    return SourceManifest(**payload)


def shard_manifest(tmp_path: Path) -> SourceManifest:
    shard = tmp_path / "src-00000.parquet"
    shard.write_bytes(b"\x00\x01parquet-bytes")
    digest = sha256_file(shard)
    overall = compute_shard_set_sha256([(shard.name, digest, shard.stat().st_size)])
    return SourceManifest(
        source="src",
        license="MIT",
        declared_rows=10,
        observed_rows=10,
        sha256=overall,
        acquired_at=FIXED.isoformat(timespec="seconds"),
        status="admitted",
        shards=[shard],
        shard_sha256={str(shard): digest},
        shard_bytes={str(shard): shard.stat().st_size},
        license_origin="hub_metadata",
    )


class TestRoundTrip:
    def test_to_dict_from_dict_round_trip(self, tmp_path):
        manifest = shard_manifest(tmp_path)
        restored = SourceManifest.from_dict(manifest.to_dict())
        assert restored == manifest

    def test_to_json_from_json_round_trip(self, tmp_path):
        manifest = shard_manifest(tmp_path)
        assert SourceManifest.from_json(manifest.to_json()) == manifest

    def test_write_and_load_manifest(self, tmp_path):
        manifest = shard_manifest(tmp_path)
        target = write_manifest(manifest, tmp_path)
        assert target == tmp_path / MANIFEST_FILENAME
        assert load_manifest(target) == manifest

    def test_shards_paths_survive_str_round_trip(self, tmp_path):
        manifest = shard_manifest(tmp_path)
        data = manifest.to_dict()
        assert all(isinstance(p, str) for p in data["shards"])
        assert SourceManifest.from_dict(data).shards == manifest.shards

    def test_total_shard_bytes_property(self, tmp_path):
        manifest = shard_manifest(tmp_path)
        assert manifest.total_shard_bytes == sum(manifest.shard_bytes.values())


class TestValidation:
    def test_bad_status_rejected(self):
        with pytest.raises(ManifestInvalidError):
            make_manifest(status="maybe")

    def test_bad_sha256_rejected(self):
        with pytest.raises(ManifestInvalidError):
            make_manifest(sha256="xyz")

    def test_negative_observed_rows_rejected(self):
        with pytest.raises(ManifestInvalidError):
            make_manifest(observed_rows=-1)

    def test_bool_observed_rows_rejected(self):
        with pytest.raises(ManifestInvalidError):
            make_manifest(observed_rows=True)

    def test_bad_acquired_at_rejected(self):
        with pytest.raises(ManifestInvalidError):
            make_manifest(acquired_at="not a date")

    def test_empty_source_rejected(self):
        with pytest.raises(ManifestInvalidError):
            make_manifest(source="  ")

    def test_shard_hash_must_be_hex(self, tmp_path):
        shard = tmp_path / "s.parquet"
        shard.write_bytes(b"x")
        with pytest.raises(ManifestInvalidError):
            make_manifest(
                sha256=compute_shard_set_sha256([(shard.name, "f" * 64, 1)]),
                shards=[shard],
                shard_sha256={str(shard): "not-hex"},
                shard_bytes={str(shard): 1},
            )

    def test_shard_bytes_must_be_nonnegative(self, tmp_path):
        shard = tmp_path / "s.parquet"
        shard.write_bytes(b"x")
        digest = compute_shard_set_sha256([(shard.name, "f" * 64, 0)])
        with pytest.raises(ManifestInvalidError):
            make_manifest(
                sha256=digest,
                shards=[shard],
                shard_sha256={str(shard): "f" * 64},
                shard_bytes={str(shard): -1},
            )

    def test_shard_records_must_cover_shards(self, tmp_path):
        shard = tmp_path / "s.parquet"
        shard.write_bytes(b"x")
        with pytest.raises(ManifestInvalidError, match="missing records"):
            make_manifest(
                sha256=compute_shard_set_sha256([(shard.name, "f" * 64, 1)]),
                shards=[shard],
                shard_sha256={str(shard): "f" * 64},
                shard_bytes={},
            )

    def test_shard_records_must_not_reference_unknown_shards(self, tmp_path):
        shard = tmp_path / "s.parquet"
        shard.write_bytes(b"x")
        digest = compute_shard_set_sha256([(shard.name, "f" * 64, 1)])
        other = str(tmp_path / "other.parquet")
        with pytest.raises(ManifestInvalidError, match="not present"):
            make_manifest(
                sha256=digest,
                shards=[shard],
                shard_sha256={str(shard): "f" * 64, other: "f" * 64},
                shard_bytes={str(shard): 1, other: 1},
            )

    def test_unknown_keys_rejected(self):
        data = make_manifest().to_dict()
        data["surprise"] = 1
        with pytest.raises(ManifestInvalidError, match="unknown keys"):
            SourceManifest.from_dict(data)

    def test_missing_required_keys_rejected(self):
        data = make_manifest().to_dict()
        del data["observed_rows"]
        with pytest.raises(ManifestInvalidError, match="missing required keys"):
            SourceManifest.from_dict(data)

    def test_non_object_manifest_rejected(self):
        with pytest.raises(ManifestInvalidError):
            SourceManifest.from_dict([1, 2, 3])

    def test_invalid_json_rejected(self):
        with pytest.raises(ManifestInvalidError):
            SourceManifest.from_json("{not json")

    def test_load_manifest_missing_file(self, tmp_path):
        with pytest.raises(ManifestInvalidError):
            load_manifest(tmp_path / "nope.json")


class TestDigest:
    def test_empty_shard_set_digest_is_stable(self):
        digest = compute_shard_set_sha256([])
        assert isinstance(digest, str) and len(digest) == 64
        assert digest == compute_shard_set_sha256([])

    def test_digest_depends_on_order(self):
        a = ("a.parquet", "1" * 64, 10)
        b = ("b.parquet", "2" * 64, 20)
        assert compute_shard_set_sha256([a, b]) != compute_shard_set_sha256([b, a])

    def test_digest_depends_on_size(self):
        a = ("a.parquet", "1" * 64, 10)
        a2 = ("a.parquet", "1" * 64, 11)
        assert compute_shard_set_sha256([a]) != compute_shard_set_sha256([a2])

    def test_digest_rejects_bad_hash(self):
        with pytest.raises(ManifestInvalidError):
            compute_shard_set_sha256([("a.parquet", "zz", 10)])
