"""Source acquisition and provenance manifests for RAPPORT.

Public surface: :func:`acquire_source` (streaming, capped, idempotent),
:class:`SourceManifest` + :func:`load_manifest`/:func:`write_manifest`/
:func:`verify_manifest` (the provenance record and its integrity check),
:func:`resolve_license` (Hub metadata or explicit mapping; no network in
this package), and the typed errors in ``src.data.acquire.errors``.

The live Hub entry point lives in ``src.data.acquire.hub`` and is never
exercised by the test suite.
"""

from .acquire import (
    DEFAULT_ROWS_PER_SHARD,
    MAX_CORPUS_BYTES,
    Downloader,
    acquire_source,
)
from .errors import (
    AcquireError,
    AcquireWriteError,
    CorpusTooLargeError,
    DownloadError,
    LicenseResolutionError,
    ManifestIntegrityError,
    ManifestInvalidError,
    MissingShardError,
    RowFilterError,
    ShardHashMismatchError,
    ShardVerificationError,
)
from .license import LicenseResolution, declared_rows_from_metadata, resolve_license
from .manifest import (
    MANIFEST_FILENAME,
    STATUS_ADMITTED,
    STATUS_QUARANTINE,
    SourceManifest,
    compute_shard_set_sha256,
    load_manifest,
    manifest_path,
    sha256_file,
    verify_manifest,
    write_manifest,
)

__all__ = [
    "DEFAULT_ROWS_PER_SHARD",
    "MANIFEST_FILENAME",
    "MAX_CORPUS_BYTES",
    "STATUS_ADMITTED",
    "STATUS_QUARANTINE",
    "AcquireError",
    "AcquireWriteError",
    "CorpusTooLargeError",
    "Downloader",
    "DownloadError",
    "LicenseResolution",
    "LicenseResolutionError",
    "ManifestIntegrityError",
    "ManifestInvalidError",
    "MissingShardError",
    "RowFilterError",
    "ShardHashMismatchError",
    "ShardVerificationError",
    "SourceManifest",
    "acquire_source",
    "compute_shard_set_sha256",
    "declared_rows_from_metadata",
    "load_manifest",
    "manifest_path",
    "resolve_license",
    "sha256_file",
    "verify_manifest",
    "write_manifest",
]
