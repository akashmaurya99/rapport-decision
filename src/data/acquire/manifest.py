"""Source acquisition manifests: schema, (de)serialization, integrity.

A :class:`SourceManifest` records, for every acquired source, the license
that was in force, the declared and observed row counts, the sha256 of
every shard written, an ISO-8601 acquisition timestamp, and an admission
verdict (``admitted`` | ``quarantine``). The mixture module (TASK-13) reads
``status`` and refuses quarantined sources; nothing else in the pipeline
second-guesses it.

``sha256`` at the top level is a digest over the *ordered shard set* (
canonical JSON of ``[basename, sha256, size]`` triples, itself hashed), so
it fingerprints the corpus as acquired, not any single file. Empty corpora
(quarantine verdicts, zero-row sources) hash the empty list.

Paths in ``shards`` are stored exactly as written by the caller (the
acquisition module writes absolute paths under ``output_dir`` when given
one). Per-shard hashes and byte counts are keyed by the same path string in
``shard_sha256`` / ``shard_bytes``.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .errors import (
    ManifestIntegrityError,
    ManifestInvalidError,
    MissingShardError,
    ShardHashMismatchError,
)

MANIFEST_FILENAME = "source-manifest.json"

STATUS_ADMITTED = "admitted"
STATUS_QUARANTINE = "quarantine"
_VALID_STATUSES = frozenset({STATUS_ADMITTED, STATUS_QUARANTINE})

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def compute_shard_set_sha256(shards: list[tuple[str, str, int]]) -> str:
    """Digest a ``[(name, sha256, nbytes)]`` shard set.

    The payload is canonical JSON (sorted keys, compact separators, UTF-8)
    of the ordered list; order matters because shard order is acquisition
    order. The empty list hashes to the sha256 of ``"[]"``.
    """
    for name, digest, nbytes in shards:
        if not _SHA256_RE.fullmatch(digest):
            raise ManifestInvalidError(f"shard {name!r} has a malformed sha256 record: {digest!r}")
        if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes < 0:
            raise ManifestInvalidError(f"shard {name!r} has an invalid byte count: {nbytes!r}")
    payload = json.dumps(shards, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Stream-hash a file; never loads the whole shard into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def shard_basename(path: str | Path) -> str:
    return Path(path).name


@dataclass(frozen=True)
class SourceManifest:
    """Provenance record for one acquired source (FR-DATA-02)."""

    source: str
    license: str | None
    declared_rows: int | None
    observed_rows: int
    sha256: str
    acquired_at: str
    status: str
    shards: list[Path] = field(default_factory=list)
    shard_sha256: dict[str, str] = field(default_factory=dict)
    shard_bytes: dict[str, int] = field(default_factory=dict)
    quarantine_reason: str | None = None
    license_origin: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ManifestInvalidError("source must be a non-empty string", field="source")
        if self.license is not None and (
            not isinstance(self.license, str) or not self.license.strip()
        ):
            raise ManifestInvalidError(
                "license must be a non-empty string or null", field="license"
            )
        for name, value in (("declared_rows", self.declared_rows),):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ManifestInvalidError(
                    f"{name} must be a non-negative integer or null, got {value!r}",
                    field=name,
                )
        if not isinstance(self.observed_rows, int) or isinstance(self.observed_rows, bool):
            raise ManifestInvalidError("observed_rows must be an integer", field="observed_rows")
        if self.observed_rows < 0:
            raise ManifestInvalidError(
                f"observed_rows must be >= 0, got {self.observed_rows}", field="observed_rows"
            )
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ManifestInvalidError(
                f"sha256 must be 64 lowercase hex characters, got {self.sha256!r}", field="sha256"
            )
        if not isinstance(self.acquired_at, str):
            raise ManifestInvalidError(
                "acquired_at must be an ISO-8601 string", field="acquired_at"
            )
        try:
            datetime.fromisoformat(self.acquired_at)
        except ValueError as exc:
            raise ManifestInvalidError(
                f"acquired_at is not a parseable ISO-8601 timestamp: {self.acquired_at!r}",
                field="acquired_at",
            ) from exc
        if self.status not in _VALID_STATUSES:
            raise ManifestInvalidError(
                f"status must be one of {sorted(_VALID_STATUSES)}, got {self.status!r}",
                field="status",
            )
        object.__setattr__(self, "shards", [Path(p) for p in self.shards])
        for name, digest in self.shard_sha256.items():
            if not _SHA256_RE.fullmatch(digest):
                raise ManifestInvalidError(
                    f"shard_sha256[{name!r}] is not 64 lowercase hex: {digest!r}",
                    field="shard_sha256",
                )
        for name, nbytes in self.shard_bytes.items():
            if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes < 0:
                raise ManifestInvalidError(
                    f"shard_bytes[{name!r}] must be a non-negative integer, got {nbytes!r}",
                    field="shard_bytes",
                )
        self._check_shard_consistency()

    def _check_shard_consistency(self) -> None:
        shard_keys = [str(p) for p in self.shards]
        if len(set(shard_keys)) != len(shard_keys):
            raise ManifestInvalidError("shards contains duplicate paths", field="shards")
        record_pairs = (
            ("shard_sha256", self.shard_sha256),
            ("shard_bytes", self.shard_bytes),
        )
        for label, mapping in record_pairs:
            extra = set(mapping) - set(shard_keys)
            if extra:
                raise ManifestInvalidError(
                    f"{label} records shards not present in shards: {sorted(extra)}",
                    field=label,
                )
            missing = set(shard_keys) - set(mapping)
            if missing:
                raise ManifestInvalidError(
                    f"{label} is missing records for shards: {sorted(missing)}",
                    field=label,
                )

    @property
    def total_shard_bytes(self) -> int:
        """Sum of per-shard byte counts; the 200 MB cap is checked against this."""
        return sum(self.shard_bytes.values())

    @property
    def admitted(self) -> bool:
        return self.status == STATUS_ADMITTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "license": self.license,
            "declared_rows": self.declared_rows,
            "observed_rows": self.observed_rows,
            "sha256": self.sha256,
            "acquired_at": self.acquired_at,
            "status": self.status,
            "shards": [str(p) for p in self.shards],
            "shard_sha256": dict(self.shard_sha256),
            "shard_bytes": dict(self.shard_bytes),
            "quarantine_reason": self.quarantine_reason,
            "license_origin": self.license_origin,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: Any) -> "SourceManifest":
        if not isinstance(data, dict):
            raise ManifestInvalidError(
                f"manifest must be a JSON object, got {type(data).__name__}"
            )
        required = {
            "source",
            "license",
            "declared_rows",
            "observed_rows",
            "sha256",
            "acquired_at",
            "status",
            "shards",
        }
        unknown = set(data) - required - {
            "shard_sha256",
            "shard_bytes",
            "quarantine_reason",
            "license_origin",
        }
        if unknown:
            raise ManifestInvalidError(f"manifest contains unknown keys: {sorted(unknown)}")
        missing = required - set(data)
        if missing:
            raise ManifestInvalidError(f"manifest is missing required keys: {sorted(missing)}")
        shards = data["shards"]
        if not isinstance(shards, list) or not all(isinstance(p, (str, Path)) for p in shards):
            raise ManifestInvalidError("shards must be a list of path strings", field="shards")
        shard_sha256 = data.get("shard_sha256") or {}
        shard_bytes = data.get("shard_bytes") or {}
        if not isinstance(shard_sha256, dict) or not isinstance(shard_bytes, dict):
            raise ManifestInvalidError(
                "shard_sha256 and shard_bytes must be objects", field="shard_sha256"
            )
        declared = data["declared_rows"]
        if declared is not None and (
            not isinstance(declared, int) or isinstance(declared, bool) or declared < 0
        ):
            raise ManifestInvalidError(
                f"declared_rows must be a non-negative integer or null, got {declared!r}",
                field="declared_rows",
            )
        return cls(
            source=data["source"],
            license=data["license"],
            declared_rows=declared,
            observed_rows=data["observed_rows"],
            sha256=data["sha256"],
            acquired_at=data["acquired_at"],
            status=data["status"],
            shards=[Path(p) for p in shards],
            shard_sha256={str(k): v for k, v in shard_sha256.items()},
            shard_bytes={str(k): v for k, v in shard_bytes.items()},
            quarantine_reason=data.get("quarantine_reason"),
            license_origin=data.get("license_origin"),
        )

    @classmethod
    def from_json(cls, text: str) -> "SourceManifest":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ManifestInvalidError(f"manifest is not valid JSON: {exc}") from exc
        return cls.from_dict(data)


def manifest_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / MANIFEST_FILENAME


def load_manifest(path: str | Path) -> SourceManifest:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ManifestInvalidError(f"manifest not found: {path}") from exc
    except OSError as exc:
        raise ManifestInvalidError(f"manifest cannot be read at {path}: {exc}") from exc
    return SourceManifest.from_json(text)


def write_manifest(manifest: SourceManifest, output_dir: str | Path) -> Path:
    """Write the manifest atomically (temp file + rename).

    The manifest is the commit point of an acquisition: shards are renamed
    into place first, the manifest is written last. A crash mid-run leaves
    no manifest, so the next run re-acquires instead of trusting partial
    state.
    """
    directory = Path(output_dir)
    target = directory / MANIFEST_FILENAME
    tmp = directory / f".{MANIFEST_FILENAME}.tmp-{uuid.uuid4().hex}"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        tmp.write_text(manifest.to_json(), encoding="utf-8")
        tmp.replace(target)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise ManifestInvalidError(f"cannot write manifest to {target}: {exc}") from exc
    return target


def verify_manifest(manifest: SourceManifest) -> None:
    """Re-hash every shard recorded in ``manifest``; raise on any mismatch.

    Shard paths in the manifest are used as recorded (absolute when the
    acquisition was run with an absolute ``output_dir``).

    Raises :class:`MissingShardError` (shard deleted),
    :class:`ShardHashMismatchError` (shard bytes changed), or
    :class:`ManifestIntegrityError` (the manifest's overall digest does not
    match its own shard records). Never re-downloads, never mutates.
    """
    recomputed: list[tuple[str, str, int]] = []
    for shard in manifest.shards:
        key = str(shard)
        path = Path(shard)
        if not path.is_file():
            raise MissingShardError(
                f"shard {key} recorded in manifest for source {manifest.source!r} "
                f"is missing on disk",
                shard=key,
            )
        expected = manifest.shard_sha256[key]
        actual = sha256_file(path)
        if actual != expected:
            raise ShardHashMismatchError(
                f"shard {key} for source {manifest.source!r} failed verification: "
                f"expected sha256 {expected}, recomputed {actual}",
                shard=key,
                expected=expected,
                actual=actual,
            )
        # Size is implied by the hash; stat it only for the digest triple.
        recomputed.append((shard_basename(shard), actual, path.stat().st_size))

    overall = compute_shard_set_sha256(recomputed)
    if overall != manifest.sha256:
        raise ManifestIntegrityError(
            f"manifest for source {manifest.source!r} failed integrity check: "
            f"overall sha256 {manifest.sha256} does not match the recomputed "
            f"shard set digest {overall}"
        )
