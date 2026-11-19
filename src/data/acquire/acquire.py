"""Streaming acquisition of external sources with provenance manifests.

One call to :func:`acquire_source` performs, in order:

1. **Idempotency check** — if a manifest already exists in ``output_dir`` and
   every recorded shard re-hashes to its recorded sha256, the existing
   manifest is returned untouched and *no download happens*. A missing or
   tampered shard raises a typed error naming the shard; corrupt JSON in the
   manifest raises :class:`ManifestInvalidError`. Acquisition is never
   silently re-run on top of bad state.
2. **License resolution** — from the caller's ``licenses=`` mapping, or a
   passed-in Hub metadata object. No license ⇒ status ``quarantine``: the
   manifest is written, no shards are materialized, and the mixture
   (TASK-13) refuses the source. Licenses listed in ``quarantine_licenses``
   (e.g. share-alike) are quarantined the same way.
3. **Streaming write** — rows flow from the injected ``downloader`` callable
   (default: the live Hub downloader in ``src.data.acquire.hub``, which tests
   never exercise) through an optional predicate and column projection, and
   land in parquet shards of at most ``rows_per_shard`` rows. Nothing is
   materialized in memory beyond one shard.
4. **Cap enforcement, before bytes hit disk** — each shard is encoded to an
   in-memory buffer first; if the running total plus the new shard's byte
   size would exceed ``max_bytes`` (default 200 MB, the per-run ceiling from
   the storage budget), the acquisition aborts *before writing the shard*:
   the partial corpus is deleted, a quarantine manifest recording the
   observed byte count and the cap is written, and
   :class:`CorpusTooLargeError` is raised. No corpus over the cap is ever
   materialized on disk, transiently or otherwise.

Raw sources are streamed through ``output_dir/.acquire-tmp-<uuid>/`` and
renamed into place only after the run completes; the manifest is written
last and is the commit point. A crash mid-run leaves no manifest, so the
next run starts clean (``refresh=True`` forces the same).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn, Protocol

from .errors import (
    AcquireError,
    AcquireWriteError,
    CorpusTooLargeError,
    DownloadError,
    RowFilterError,
)
from .license import declared_rows_from_metadata, resolve_license
from .manifest import (
    STATUS_ADMITTED,
    STATUS_QUARANTINE,
    SourceManifest,
    compute_shard_set_sha256,
    load_manifest,
    manifest_path,
    shard_basename,
    verify_manifest,
    write_manifest,
)

#: Per-run ceiling on materialized corpus bytes (storage budget, §7 of the
#: dataset strategy). 200 MB = 200 * 1024 * 1024 bytes.
MAX_CORPUS_BYTES = 200 * 1024 * 1024

#: Rows per shard; bounds shard size and keeps parquet files manageable.
DEFAULT_ROWS_PER_SHARD = 10_000


class Downloader(Protocol):
    """Injection seam for the row source.

    Tests stub this with an in-memory iterator; the live implementation in
    ``src.data.acquire.hub`` streams from the Hugging Face Hub with
    ``datasets.load_dataset(..., streaming=True)``. The protocol itself
    performs no network access.
    """

    def __call__(
        self, source: str, *, split: str | None = None
    ) -> Iterator[Mapping[str, Any]]: ...


class _CapExceeded(Exception):
    """Internal signal: writing the next shard would exceed the byte cap."""

    def __init__(self, bytes_observed: int, observed_rows: int) -> None:
        super().__init__(f"corpus byte cap would be exceeded ({bytes_observed} bytes)")
        self.bytes_observed = bytes_observed
        self.observed_rows = observed_rows


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.isoformat(timespec="seconds")


def _sanitize_shard_stem(source: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip("_")
    return stem or "source"


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - pyarrow is present in the venv
        raise AcquireWriteError(
            "pyarrow is not installed; acquisition cannot write parquet shards"
        ) from exc
    return pa, pq


def _encode_shard(table_rows: list[dict[str, Any]]) -> tuple[bytes, str, int]:
    """Encode rows to a parquet shard in memory; return (payload, sha256, nbytes)."""
    pa, pq = _pyarrow()
    try:
        table = pa.Table.from_pylist(table_rows)
    except Exception as exc:
        raise AcquireWriteError(f"could not build parquet table: {exc}") from exc
    try:
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        payload = sink.getvalue().to_pybytes()
    except OSError as exc:
        raise AcquireWriteError(f"could not encode parquet shard: {exc}") from exc
    except Exception as exc:
        raise AcquireWriteError(f"could not encode parquet shard: {exc}") from exc
    return payload, hashlib.sha256(payload).hexdigest(), len(payload)


def _refuse_over_cap(
    *,
    source: str,
    bytes_observed: int,
    observed_rows: int,
    max_bytes: int,
    declared_rows: int | None,
    license_: str | None,
    license_origin: str | None,
    output_dir: Path,
    moment: datetime,
) -> NoReturn:
    """Record a refusal manifest for an over-cap source, then raise.

    The partial corpus is not left on disk (the caller removes the temp
    shards). The manifest records the observed byte count and the cap so
    the refusal is auditable.
    """
    manifest = SourceManifest(
        source=source,
        license=license_,
        declared_rows=declared_rows,
        observed_rows=observed_rows,
        sha256=compute_shard_set_sha256([]),
        acquired_at=_iso(moment),
        status=STATUS_QUARANTINE,
        shards=[],
        shard_sha256={},
        shard_bytes={},
        quarantine_reason=(
            f"corpus exceeds per-run cap: {bytes_observed} bytes would be materialized, "
            f"limit is {max_bytes} bytes; no shards were written"
        ),
        license_origin=license_origin,
    )
    write_manifest(manifest, output_dir)
    raise CorpusTooLargeError(
        f"source {source!r} exceeded the per-run byte cap: {bytes_observed} bytes would "
        f"be materialized, limit is {max_bytes} bytes; acquisition refused and recorded "
        f"as quarantine in the manifest",
        source=source,
        bytes_observed=bytes_observed,
        limit=max_bytes,
    )


def _acquire_into_tmp(
    *,
    source: str,
    tmp_dir: Path,
    split: str | None,
    columns: tuple[str, ...] | None,
    row_filter: Callable[[Mapping[str, Any]], bool] | None,
    rows_per_shard: int,
    max_bytes: int,
    downloader: Downloader,
) -> tuple[list[tuple[str, Path, str, int]], int]:
    """Stream rows into ``tmp_dir``; return shard records and row count.

    Shard records are ``(shard_name, tmp_path, sha256, nbytes)``. Raises
    :class:`_CapExceeded` *before writing a shard that would breach
    ``max_bytes``* — no over-cap file ever lands on disk.
    """
    try:
        iterator = iter(downloader(source, split=split))
    except Exception as exc:
        raise DownloadError(
            f"downloader for source {source!r} failed to start: {exc}", source=source
        ) from exc

    shards: list[tuple[str, Path, str, int]] = []
    total_bytes = 0
    observed_rows = 0
    batch: list[dict[str, Any]] = []
    shard_index = 0

    def flush() -> None:
        nonlocal batch, shard_index, total_bytes
        if not batch:
            return
        payload, digest, nbytes = _encode_shard(batch)
        if total_bytes + nbytes > max_bytes:
            raise _CapExceeded(total_bytes + nbytes, observed_rows)
        shard_name = f"{_sanitize_shard_stem(source)}-{shard_index:05d}.parquet"
        shard_path = tmp_dir / shard_name
        try:
            shard_path.write_bytes(payload)
        except OSError as exc:
            raise AcquireWriteError(
                f"could not write shard {shard_path} (disk full?): {exc}",
                path=str(shard_path),
            ) from exc
        shards.append((shard_name, shard_path, digest, nbytes))
        total_bytes += nbytes
        batch = []
        shard_index += 1

    try:
        row_index = 0
        while True:
            try:
                row = next(iterator)
            except StopIteration:
                break
            except Exception as exc:
                raise DownloadError(
                    f"downloader for source {source!r} failed at row {row_index}: {exc}",
                    source=source,
                ) from exc

            if not isinstance(row, Mapping):
                raise AcquireError(
                    f"row {row_index} from source {source!r} is not a mapping "
                    f"(got {type(row).__name__})",
                    field="row",
                )
            projected: dict[str, Any] = {}
            for key, value in row.items():
                key_str = key if isinstance(key, str) else str(key)
                if columns is not None and key_str not in columns:
                    continue
                projected[key_str] = value
            if row_filter is not None:
                try:
                    keep = bool(row_filter(projected))
                except Exception as exc:
                    raise RowFilterError(
                        f"row_filter for source {source!r} raised at row {row_index}: {exc}",
                        source=source,
                        row_index=row_index,
                    ) from exc
                if not keep:
                    row_index += 1
                    continue
            batch.append(projected)
            observed_rows += 1
            row_index += 1
            if len(batch) >= rows_per_shard:
                flush()
        flush()
    except _CapExceeded:
        raise
    except AcquireError:
        raise
    except Exception as exc:  # defensive: anything from the iterator itself
        raise DownloadError(
            f"downloader for source {source!r} failed: {exc}", source=source
        ) from exc

    return shards, observed_rows


def acquire_source(
    source: str,
    *,
    output_dir: str | Path,
    declared_rows: int | None = None,
    hub_metadata: Any | None = None,
    licenses: Mapping[str, str | None] | None = None,
    quarantine_licenses: frozenset[str] | set[str] | tuple[str, ...] | list[str] = (),
    split: str | None = None,
    columns: tuple[str, ...] | list[str] | None = None,
    row_filter: Callable[[Mapping[str, Any]], bool] | None = None,
    max_bytes: int = MAX_CORPUS_BYTES,
    rows_per_shard: int = DEFAULT_ROWS_PER_SHARD,
    downloader: Downloader | None = None,
    now: datetime | None = None,
    refresh: bool = False,
) -> SourceManifest:
    """Acquire ``source`` as parquet shards plus a provenance manifest.

    Returns the manifest. On re-run with unchanged shards, the existing
    manifest is returned untouched and no download is attempted. A source
    whose bytes would exceed ``max_bytes`` is refused with
    :class:`CorpusTooLargeError` and recorded as a quarantine manifest.

    ``refresh=True`` ignores (and overwrites) any existing manifest — the
    documented escape hatch for re-admitting a source after a license
    review. Without it, an existing valid manifest always wins.
    """
    if not isinstance(source, str) or not source.strip():
        raise AcquireError("source must be a non-empty string", field="source")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise AcquireError(
            f"max_bytes must be a positive integer, got {max_bytes!r}", field="max_bytes"
        )
    if (
        not isinstance(rows_per_shard, int)
        or isinstance(rows_per_shard, bool)
        or rows_per_shard < 1
    ):
        raise AcquireError(
            f"rows_per_shard must be >= 1, got {rows_per_shard!r}", field="rows_per_shard"
        )

    out = Path(output_dir)
    moment = now if now is not None else _utc_now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AcquireWriteError(
            f"cannot create output directory {out}: {exc}", path=str(out)
        ) from exc

    existing = manifest_path(out)
    if not refresh and existing.is_file():
        manifest = load_manifest(existing)
        verify_manifest(manifest)
        return manifest

    resolution = resolve_license(source, hub_metadata=hub_metadata, licenses=licenses)
    declared = (
        declared_rows
        if declared_rows is not None
        else declared_rows_from_metadata(hub_metadata)
    )
    if declared is not None and (
        not isinstance(declared, int) or isinstance(declared, bool) or declared < 0
    ):
        raise AcquireError(
            f"declared_rows must be a non-negative integer or null, got {declared!r}",
            field="declared_rows",
        )

    license_ = resolution.license
    quarantine_set = {entry.strip().lower() for entry in quarantine_licenses}
    license_casefolded = license_.lower() if license_ else None

    quarantine_reason: str | None = None
    if license_ is None:
        quarantine_reason = (
            "no license declared by the source (resolution origin: "
            f"{resolution.origin}); refusing to materialize the corpus"
        )
    elif license_casefolded in quarantine_set:
        quarantine_reason = (
            f"license {license_!r} is quarantined by policy "
            f"(share-alike or unreviewed); refusing to materialize the corpus"
        )

    if quarantine_reason is not None:
        manifest = SourceManifest(
            source=source,
            license=license_,
            declared_rows=declared,
            observed_rows=0,
            sha256=compute_shard_set_sha256([]),
            acquired_at=_iso(moment),
            status=STATUS_QUARANTINE,
            shards=[],
            shard_sha256={},
            shard_bytes={},
            quarantine_reason=quarantine_reason,
            license_origin=resolution.origin,
        )
        write_manifest(manifest, out)
        return manifest

    if downloader is None:
        from .hub import hub_downloader

        downloader = hub_downloader()

    columns_tuple = tuple(columns) if columns is not None else None
    tmp_dir = out / f".acquire-tmp-{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        shard_records, observed_rows = _acquire_into_tmp(
            source=source,
            tmp_dir=tmp_dir,
            split=split,
            columns=columns_tuple,
            row_filter=row_filter,
            rows_per_shard=rows_per_shard,
            max_bytes=max_bytes,
            downloader=downloader,
        )

        # Defense in depth: the cap is enforced before every shard is
        # written, so this can never fire — it documents the invariant the
        # storage budget relies on.
        if sum(record[3] for record in shard_records) > max_bytes:  # pragma: no cover
            raise CorpusTooLargeError(
                f"source {source!r} shards total {sum(r[3] for r in shard_records)} bytes, "
                f"limit is {max_bytes}",
                source=source,
                bytes_observed=sum(record[3] for record in shard_records),
                limit=max_bytes,
            )

        final_records: list[tuple[Path, str, int]] = []
        for shard_name, tmp_shard, digest, nbytes in shard_records:
            target = out / shard_name
            try:
                os.replace(tmp_shard, target)
            except OSError as exc:
                raise AcquireWriteError(
                    f"could not move shard {tmp_shard} into place at {target}: {exc}",
                    path=str(target),
                ) from exc
            final_records.append((target, digest, nbytes))

        shard_sha256 = {str(path): digest for path, digest, _ in final_records}
        shard_bytes = {str(path): nbytes for path, _, nbytes in final_records}
        digest = compute_shard_set_sha256(
            [
                (shard_basename(path), shard_sha256[str(path)], shard_bytes[str(path)])
                for path, _, _ in final_records
            ]
        )
        manifest = SourceManifest(
            source=source,
            license=license_,
            declared_rows=declared,
            observed_rows=observed_rows,
            sha256=digest,
            acquired_at=_iso(moment),
            status=STATUS_ADMITTED,
            shards=[path for path, _, _ in final_records],
            shard_sha256=shard_sha256,
            shard_bytes=shard_bytes,
            license_origin=resolution.origin,
        )
        write_manifest(manifest, out)
        return manifest
    except _CapExceeded as exc:
        _refuse_over_cap(
            source=source,
            bytes_observed=exc.bytes_observed,
            observed_rows=exc.observed_rows,
            max_bytes=max_bytes,
            declared_rows=declared,
            license_=license_,
            license_origin=resolution.origin,
            output_dir=out,
            moment=moment,
        )
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
