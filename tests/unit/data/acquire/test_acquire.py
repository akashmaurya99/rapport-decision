"""Tests for streaming acquisition: cap, idempotency, quarantine.

The network is stubbed out everywhere; nothing here touches the Hub.
"""

from __future__ import annotations

import json

import pytest

pyyarrow = pytest.importorskip("pyarrow", reason="pyarrow is required for parquet shards")
import pyarrow.parquet as pq  # noqa: E402

from src.data.acquire import (  # noqa: E402
    MAX_CORPUS_BYTES,
    AcquireError,
    CorpusTooLargeError,
    DownloadError,
    ManifestIntegrityError,
    ManifestInvalidError,
    MissingShardError,
    RowFilterError,
    ShardHashMismatchError,
    acquire_source,
    load_manifest,
    sha256_file,
)
from src.data.acquire.manifest import MANIFEST_FILENAME, compute_shard_set_sha256  # noqa: E402


class TestHappyPath:
    def test_admitted_source_writes_manifest_and_shards(
        self, tmp_path, rows, stub_downloader, fixed_now
    ):
        stub = stub_downloader(rows(25))
        manifest = acquire_source(
            "src",
            output_dir=tmp_path,
            declared_rows=1000,
            licenses={"src": "MIT"},
            downloader=stub,
            now=fixed_now,
            rows_per_shard=10,
        )
        assert manifest.status == "admitted"
        assert manifest.source == "src"
        assert manifest.license == "MIT"
        assert manifest.license_origin == "licenses_mapping"
        assert manifest.declared_rows == 1000
        assert manifest.observed_rows == 25
        assert manifest.acquired_at == fixed_now.isoformat(timespec="seconds")
        assert len(manifest.shards) == 3  # 10 + 10 + 5
        for shard in manifest.shards:
            assert shard.is_file()
            assert manifest.shard_sha256[str(shard)] == sha256_file(shard)
            assert manifest.shard_bytes[str(shard)] == shard.stat().st_size
        # manifest-level digest matches the shard set
        expected = compute_shard_set_sha256(
            [
                (p.name, manifest.shard_sha256[str(p)], manifest.shard_bytes[str(p)])
                for p in manifest.shards
            ]
        )
        assert manifest.sha256 == expected
        assert manifest.total_shard_bytes == sum(manifest.shard_bytes.values())
        # the manifest on disk matches the returned one
        on_disk = load_manifest(tmp_path / MANIFEST_FILENAME)
        assert on_disk == manifest

    def test_parquet_shards_round_trip(self, tmp_path, rows, stub_downloader, fixed_now):
        stub = stub_downloader(rows(8))
        manifest = acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "apache-2.0"},
            downloader=stub,
            now=fixed_now,
            rows_per_shard=4,
            columns=("id", "goal"),
        )
        table = pq.read_table(manifest.shards[0])
        assert table.num_rows == 4
        assert set(table.column_names) == {"id", "goal"}  # column projection applied
        assert table.column("id")[0].as_py() == "row-0"

    def test_row_filter_applied_before_counting(self, tmp_path, rows, stub_downloader, fixed_now):
        stub = stub_downloader(rows(10))
        manifest = acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            downloader=stub,
            now=fixed_now,
            row_filter=lambda row: int(row["id"].split("-")[1]) % 2 == 0,
        )
        assert manifest.observed_rows == 5
        total = sum(pq.read_table(shard).num_rows for shard in manifest.shards)
        assert total == 5

    def test_declared_rows_fall_back_to_hub_metadata(
        self, tmp_path, rows, stub_downloader, fixed_now, fake_hub
    ):
        stub = stub_downloader(rows(3))
        manifest = acquire_source(
            "src",
            output_dir=tmp_path,
            hub_metadata=fake_hub(license="MIT", num_rows=77),
            downloader=stub,
            now=fixed_now,
        )
        assert manifest.declared_rows == 77

    def test_split_is_forwarded_to_downloader(self, tmp_path, rows, stub_downloader, fixed_now):
        stub = stub_downloader(rows(2))
        acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            split="train",
            downloader=stub,
            now=fixed_now,
        )
        assert stub.calls == [("src", "train")]

    def test_empty_source_admitted_with_zero_rows(self, tmp_path, stub_downloader, fixed_now):
        stub = stub_downloader([])
        manifest = acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            downloader=stub,
            now=fixed_now,
        )
        assert manifest.status == "admitted"
        assert manifest.observed_rows == 0
        assert manifest.shards == []
        assert manifest.sha256 == compute_shard_set_sha256([])

    def test_200mb_cap_uses_mebibytes(self):
        assert MAX_CORPUS_BYTES == 200 * 1024 * 1024


class TestQuarantine:
    def test_missing_license_quarantines_and_never_downloads(
        self, tmp_path, rows, stub_downloader, fixed_now
    ):
        stub = stub_downloader(rows(10))
        manifest = acquire_source(
            "src",
            output_dir=tmp_path,
            downloader=stub,
            now=fixed_now,
        )
        assert manifest.status == "quarantine"
        assert manifest.license is None
        assert manifest.observed_rows == 0
        assert manifest.shards == []
        assert manifest.quarantine_reason
        assert "no license" in manifest.quarantine_reason
        assert stub.calls == []  # downloader never invoked
        # quarantined corpus leaves no parquet on disk
        assert list(tmp_path.glob("*.parquet")) == []
        # mixture contract: the record is loadable and refuses admission
        on_disk = load_manifest(tmp_path / MANIFEST_FILENAME)
        assert on_disk.status == "quarantine"

    def test_hub_metadata_without_license_quarantines(
        self, tmp_path, stub_downloader, fixed_now, fake_hub
    ):
        stub = stub_downloader([])
        manifest = acquire_source(
            "src",
            output_dir=tmp_path,
            hub_metadata=fake_hub(license=None),  # e.g. HellaSwag: no license on the card
            downloader=stub,
            now=fixed_now,
        )
        assert manifest.status == "quarantine"
        assert manifest.license_origin == "hub_metadata"

    def test_declared_share_alike_license_quarantines(
        self, tmp_path, stub_downloader, fixed_now, fake_hub
    ):
        stub = stub_downloader([])
        manifest = acquire_source(
            "src",
            output_dir=tmp_path,
            hub_metadata=fake_hub(license="CC-BY-SA-4.0"),  # e.g. ai2_arc
            quarantine_licenses=frozenset({"cc-by-sa-4.0"}),
            downloader=stub,
            now=fixed_now,
        )
        assert manifest.status == "quarantine"
        assert manifest.license == "CC-BY-SA-4.0"
        assert "share-alike" in manifest.quarantine_reason
        assert stub.calls == []

    def test_over_cap_source_refused_and_recorded(
        self, tmp_path, rows, stub_downloader, fixed_now
    ):
        stub = stub_downloader(rows(40))
        # tiny cap: even one shard blows it
        with pytest.raises(CorpusTooLargeError) as excinfo:
            acquire_source(
                "src",
                output_dir=tmp_path,
                licenses={"src": "MIT"},
                downloader=stub,
                now=fixed_now,
                max_bytes=1,
                rows_per_shard=5,
            )
        exc = excinfo.value
        assert exc.code == "DATA_CORPUS_TOO_LARGE"
        assert exc.limit == 1
        assert exc.bytes_observed > 1
        # refusal is recorded as a quarantine manifest, no shards on disk
        assert list(tmp_path.glob("*.parquet")) == []
        assert list(tmp_path.glob(".acquire-tmp-*")) == []
        manifest = load_manifest(tmp_path / MANIFEST_FILENAME)
        assert manifest.status == "quarantine"
        assert manifest.quarantine_reason
        assert str(exc.limit) in manifest.quarantine_reason
        assert manifest.observed_rows > 0

    def test_cap_boundary_exactly_at_limit_is_admitted(
        self, tmp_path, rows, stub_downloader, fixed_now
    ):
        stub = stub_downloader(rows(10))
        first = acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            downloader=stub,
            now=fixed_now,
            max_bytes=MAX_CORPUS_BYTES,
        )
        total = first.total_shard_bytes
        # re-run at exactly the measured size: admitted (cap is <=)
        redownloaded = stub_downloader(rows(10))
        second = acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            downloader=redownloaded,
            now=fixed_now,
            max_bytes=total,
            refresh=True,
        )
        assert second.total_shard_bytes == total
        # one byte below the limit: refused
        with pytest.raises(CorpusTooLargeError):
            acquire_source(
                "src",
                output_dir=tmp_path,
                licenses={"src": "MIT"},
                downloader=stub_downloader(rows(10)),
                now=fixed_now,
                max_bytes=total - 1,
                refresh=True,
            )


class TestIdempotency:
    def test_rerun_returns_existing_manifest_without_downloading(
        self, tmp_path, rows, stub_downloader, fixed_now
    ):
        first_stub = stub_downloader(rows(7))
        first = acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            downloader=first_stub,
            now=fixed_now,
        )
        assert len(first_stub.calls) == 1

        def forbidden_downloader(source, *, split=None):
            raise AssertionError("download must not happen on an idempotent re-run")

        second = acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            downloader=forbidden_downloader,
            now=fixed_now,  # would produce a different timestamp if re-created
        )
        assert second == first
        assert second.acquired_at == first.acquired_at

    def test_rerun_after_tampering_raises_and_names_the_shard(
        self, tmp_path, rows, stub_downloader, fixed_now
    ):
        stub = stub_downloader(rows(6))
        manifest = acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            downloader=stub,
            now=fixed_now,
            rows_per_shard=3,
        )
        victim = manifest.shards[0]
        victim.write_bytes(victim.read_bytes() + b"tampered")

        def forbidden_downloader(source, *, split=None):
            raise AssertionError("must not re-download on hash mismatch")

        with pytest.raises(ShardHashMismatchError) as excinfo:
            acquire_source(
                "src",
                output_dir=tmp_path,
                licenses={"src": "MIT"},
                downloader=forbidden_downloader,
                now=fixed_now,
            )
        exc = excinfo.value
        assert exc.shard == str(victim)
        assert str(victim) in str(exc)
        assert exc.expected == manifest.shard_sha256[str(victim)]
        assert exc.actual == sha256_file(victim)

    def test_rerun_with_missing_shard_raises_and_names_the_shard(
        self, tmp_path, rows, stub_downloader, fixed_now
    ):
        stub = stub_downloader(rows(4))
        manifest = acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            downloader=stub,
            now=fixed_now,
            rows_per_shard=2,
        )
        victim = manifest.shards[1]
        victim.unlink()
        with pytest.raises(MissingShardError) as excinfo:
            acquire_source(
                "src",
                output_dir=tmp_path,
                licenses={"src": "MIT"},
                downloader=stub_downloader(rows(4)),
                now=fixed_now,
            )
        assert excinfo.value.shard == str(victim)

    def test_tampered_manifest_digest_raises_integrity_error(
        self, tmp_path, rows, stub_downloader, fixed_now
    ):
        stub = stub_downloader(rows(3))
        acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            downloader=stub,
            now=fixed_now,
        )
        data = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        data["sha256"] = "f" * 64
        (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ManifestIntegrityError):
            acquire_source(
                "src",
                output_dir=tmp_path,
                licenses={"src": "MIT"},
                downloader=stub_downloader(rows(3)),
                now=fixed_now,
            )

    def test_manifest_with_dropped_shard_record_raises_and_never_downloads(
        self, tmp_path, rows, stub_downloader, fixed_now
    ):
        acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            downloader=stub_downloader(rows(4)),
            now=fixed_now,
            rows_per_shard=2,
        )
        data = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        data["shard_bytes"] = {}  # manifest no longer accounts for its shards
        (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ManifestInvalidError):
            acquire_source(
                "src",
                output_dir=tmp_path,
                licenses={"src": "MIT"},
                downloader=stub_downloader(rows(4)),
                now=fixed_now,
            )

    def test_manifest_with_swapped_shard_order_raises_integrity_error(
        self, tmp_path, rows, stub_downloader, fixed_now
    ):
        acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            downloader=stub_downloader(rows(4)),
            now=fixed_now,
            rows_per_shard=2,
        )
        data = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        data["shards"] = list(reversed(data["shards"]))
        (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ManifestIntegrityError):
            acquire_source(
                "src",
                output_dir=tmp_path,
                licenses={"src": "MIT"},
                downloader=stub_downloader(rows(4)),
                now=fixed_now,
            )

    def test_corrupt_manifest_raises_manifest_invalid(
        self, tmp_path, rows, stub_downloader, fixed_now
    ):
        acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            downloader=stub_downloader(rows(2)),
            now=fixed_now,
        )
        (tmp_path / MANIFEST_FILENAME).write_text("{corrupt", encoding="utf-8")
        with pytest.raises(ManifestInvalidError):
            acquire_source(
                "src",
                output_dir=tmp_path,
                licenses={"src": "MIT"},
                downloader=stub_downloader(rows(2)),
                now=fixed_now,
            )

    def test_quarantine_manifest_is_idempotent(self, tmp_path, stub_downloader, fixed_now):
        first = acquire_source(
            "src",
            output_dir=tmp_path,
            downloader=stub_downloader([]),
            now=fixed_now,
        )
        assert first.status == "quarantine"

        def forbidden_downloader(source, *, split=None):
            raise AssertionError("quarantine must come from the recorded verdict")

        # Even with a license now provided, the recorded verdict stands;
        # refresh=True is the documented escape hatch.
        second = acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            downloader=forbidden_downloader,
            now=fixed_now,
        )
        assert second == first

    def test_refresh_forces_reacquisition(self, tmp_path, rows, stub_downloader, fixed_now):
        stub = stub_downloader(rows(5))
        acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            downloader=stub,
            now=fixed_now,
        )
        fresh_stub = stub_downloader(rows(5))
        manifest = acquire_source(
            "src",
            output_dir=tmp_path,
            licenses={"src": "MIT"},
            downloader=fresh_stub,
            now=fixed_now,
            refresh=True,
        )
        assert len(fresh_stub.calls) == 1
        assert manifest.observed_rows == 5


class TestFailureModes:
    def test_downloader_raising_yields_download_error_and_no_manifest(
        self, tmp_path, fixed_now
    ):
        def broken(source, *, split=None):
            yield {"id": "ok"}
            raise RuntimeError("connection reset by peer")

        with pytest.raises(DownloadError) as excinfo:
            acquire_source(
                "src",
                output_dir=tmp_path,
                licenses={"src": "MIT"},
                downloader=broken,
                now=fixed_now,
            )
        assert excinfo.value.source == "src"
        assert not (tmp_path / MANIFEST_FILENAME).exists()
        assert list(tmp_path.glob("*.parquet")) == []
        assert list(tmp_path.glob(".acquire-tmp-*")) == []

    def test_downloader_failing_to_start_yields_download_error(self, tmp_path, fixed_now):
        def broken(source, *, split=None):
            raise RuntimeError("hub 503")
            yield  # pragma: no cover - make it a generator

        with pytest.raises(DownloadError):
            acquire_source(
                "src",
                output_dir=tmp_path,
                licenses={"src": "MIT"},
                downloader=broken,
                now=fixed_now,
            )
        assert not (tmp_path / MANIFEST_FILENAME).exists()

    def test_non_mapping_row_is_a_typed_error(self, tmp_path, fixed_now):
        def bad_rows(source, *, split=None):
            yield 42

        with pytest.raises(AcquireError, match="row 0"):
            acquire_source(
                "src",
                output_dir=tmp_path,
                licenses={"src": "MIT"},
                downloader=bad_rows,
                now=fixed_now,
            )

    def test_row_filter_raising_is_typed_error_naming_row(
        self, tmp_path, rows, stub_downloader, fixed_now
    ):
        stub = stub_downloader(rows(3))

        def exploding_filter(row):
            raise ValueError("boom")

        with pytest.raises(RowFilterError) as excinfo:
            acquire_source(
                "src",
                output_dir=tmp_path,
                licenses={"src": "MIT"},
                downloader=stub,
                row_filter=exploding_filter,
                now=fixed_now,
            )
        assert excinfo.value.row_index == 0
        assert excinfo.value.source == "src"

    def test_invalid_arguments(self, tmp_path, stub_downloader, rows):
        with pytest.raises(AcquireError):
            acquire_source(
                "",
                output_dir=tmp_path,
                licenses={"": "MIT"},
                downloader=stub_downloader(rows(1)),
            )
        with pytest.raises(AcquireError):
            acquire_source(
                "src",
                output_dir=tmp_path,
                licenses={"src": "MIT"},
                downloader=stub_downloader(rows(1)),
                max_bytes=0,
            )
        with pytest.raises(AcquireError):
            acquire_source(
                "src",
                output_dir=tmp_path,
                licenses={"src": "MIT"},
                downloader=stub_downloader(rows(1)),
                rows_per_shard=0,
            )
