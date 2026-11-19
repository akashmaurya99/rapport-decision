"""Typed errors raised by the acquisition module.

Every error carries a stable ``code`` attribute so callers (and tests) can
match on the failure class without parsing message text. Codes are part of
the public contract, mirroring ``src/data/core/errors.py``: a code is never
renamed or reused for a different failure class.

The acquisition stance is fail-loudly: a partial download is deleted, a
tampered shard aborts the run, an over-cap corpus is refused and recorded —
never silently truncated, never silently re-downloaded.
"""

from __future__ import annotations

from src.data.core.errors import DataError


class AcquireError(DataError):
    """Base class for errors raised by this package."""

    code = "DATA_ACQUIRE_ERROR"


class LicenseResolutionError(AcquireError):
    """License could not be read from the provided metadata object.

    Raised only when a metadata object was *passed* but cannot be
    interpreted (not a mapping, not an object with a readable license
    slot). A missing license is not an error — it yields a quarantine
    manifest; an unreadable one is a bug in the caller or a Hub schema
    change, and is surfaced instead of guessed.
    """

    code = "DATA_LICENSE_RESOLUTION_ERROR"


class DownloadError(AcquireError):
    """The downloader callable raised while streaming rows for ``source``."""

    code = "DATA_DOWNLOAD_FAILED"

    def __init__(
        self, message: str, *, source: str | None = None, field: str | None = None
    ) -> None:
        super().__init__(message, field=field)
        self.source = source


class RowFilterError(AcquireError):
    """A ``row_filter`` predicate raised at a specific row index."""

    code = "DATA_ROW_FILTER_FAILED"

    def __init__(
        self, message: str, *, source: str, row_index: int, field: str | None = None
    ) -> None:
        super().__init__(message, field=field)
        self.source = source
        self.row_index = row_index


class AcquireWriteError(AcquireError):
    """A shard or the manifest could not be written (e.g. disk exhausted)."""

    code = "DATA_ACQUIRE_WRITE_FAILED"

    def __init__(self, message: str, *, path: str | None = None, field: str | None = None) -> None:
        super().__init__(message, field=field)
        self.path = path


class CorpusTooLargeError(AcquireError):
    """The source exceeded the per-run byte cap; it is refused and recorded."""

    code = "DATA_CORPUS_TOO_LARGE"

    def __init__(
        self,
        message: str,
        *,
        source: str,
        bytes_observed: int,
        limit: int,
        field: str | None = None,
    ) -> None:
        super().__init__(message, field=field)
        self.source = source
        self.bytes_observed = bytes_observed
        self.limit = limit


class ShardVerificationError(AcquireError):
    """Base class for shard verification failures on an existing corpus."""

    code = "DATA_SHARD_VERIFICATION_FAILED"

    def __init__(self, message: str, *, shard: str, field: str | None = None) -> None:
        super().__init__(message, field=field)
        self.shard = shard


class MissingShardError(ShardVerificationError):
    """A shard recorded in the manifest does not exist on disk."""

    code = "DATA_SHARD_MISSING"


class ShardHashMismatchError(ShardVerificationError):
    """A shard's recomputed sha256 differs from the manifest's record."""

    code = "DATA_SHARD_HASH_MISMATCH"

    def __init__(
        self,
        message: str,
        *,
        shard: str,
        expected: str,
        actual: str,
        field: str | None = None,
    ) -> None:
        super().__init__(message, shard=shard, field=field)
        self.expected = expected
        self.actual = actual


class ManifestInvalidError(AcquireError):
    """A manifest file exists but cannot be parsed/validated.

    Raised instead of silently re-downloading: a corrupt manifest is an
    operator problem (or tampering), never a signal to fetch the corpus
    again.
    """

    code = "DATA_MANIFEST_INVALID"


class ManifestIntegrityError(AcquireError):
    """The manifest's overall sha256 does not match its recorded shard set."""

    code = "DATA_MANIFEST_INTEGRITY"
