"""Run registry (FR-OPS-02).

Scans a runs directory (``runs/<run_id>/manifest.json`` layout) and indexes
every valid run by its ``run_id``. A run is *valid* only when its
``manifest.json`` exists, parses as UTF-8 JSON object, and carries every
required RunManifest field with the right type and format (sha256 hashes,
portable run id matching its directory, non-negative int seed, version map) —
the required set per ``.archgen/rapport/architecture.yaml``.

A run that fails any of those checks is **excluded**, never silently listed:
malformed encodings, unreadable files and duplicate ids are excluded rather
than aborting the whole scan.
the number of exclusions and their per-directory reasons are exposed via
:attr:`RunRegistry.excluded_count` and :attr:`RunRegistry.exclusions`.

``list()`` returns the indexed run ids sorted lexicographically; ``get``
returns the manifest path plus the parsed manifest. Registry entries also
carry the checkpoint paths the run produced (files under ``checkpoints/`` and
top-level ``*.pt`` / ``*.pth`` / ``*.ckpt`` / ``*.safetensors`` files).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .errors import RunRegistryError, UnknownRunError

CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors")

# Every field the RunManifest contract marks required (plus the two honesty
# fields, git_dirty/compute_capability, which are optional by design).
REQUIRED_MANIFEST_FIELDS = (
    "run_id",
    "git_sha",
    "config_hash",
    "data_fingerprint",
    "seed",
    "device",
    "versions",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _valid_run_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value not in (".", "..")
        and not value.endswith(".")
        and bool(_PORTABLE_ID.fullmatch(value))
    )


def _valid_versions(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("python"), str)
        and bool(value["python"].strip())
    )


_FIELD_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "run_id": _valid_run_id,
    "git_sha": lambda v: isinstance(v, str) and bool(v.strip()),
    "config_hash": lambda v: isinstance(v, str) and bool(_HEX64.fullmatch(v)),
    # An empty fingerprint is the documented placeholder before acquisition.
    "data_fingerprint": lambda v: isinstance(v, str)
    and (v == "" or bool(_PORTABLE_ID.fullmatch(v))),
    "seed": lambda v: type(v) is int and 0 <= v <= 2**32 - 1,
    "device": lambda v: isinstance(v, str) and bool(v.strip()),
    "versions": _valid_versions,
}


@dataclass(frozen=True)
class RegistryEntry:
    """One indexed run: manifest location, parsed content and outputs."""

    run_id: str
    manifest_path: Path
    manifest: dict[str, Any]
    config_hash: str
    data_fingerprint: str
    checkpoint_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "manifest_path": str(self.manifest_path),
            "config_hash": self.config_hash,
            "data_fingerprint": self.data_fingerprint,
            "checkpoint_paths": list(self.checkpoint_paths),
            "manifest": self.manifest,
        }


@dataclass(frozen=True)
class Exclusion:
    """One run directory that was excluded from the registry, with the reason."""

    run_dir: Path
    reason: str


@dataclass(frozen=True)
class RegistryScan:
    """Counts from the most recent scan."""

    total_run_dirs: int
    indexed: int
    excluded: int


def _validate_manifest(parsed: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Return (manifest, None) when valid, else (None, stable reason)."""
    if not isinstance(parsed, dict):
        return None, "manifest_not_an_object"
    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in parsed or parsed[field] is None:
            return None, f"missing_{field}"
        if not _FIELD_VALIDATORS[field](parsed[field]):
            return None, f"invalid_{field}"
    return parsed, None


def _checkpoint_paths(run_dir: Path) -> tuple[str, ...]:
    paths: list[Path] = []
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        paths.extend(p for p in checkpoints_dir.rglob("*") if p.is_file())
    for child in run_dir.iterdir():
        if child.is_file() and child.suffix in CHECKPOINT_SUFFIXES:
            paths.append(child)
    return tuple(sorted(str(p) for p in set(paths)))


class RunRegistry:
    """Indexes every valid run under a runs directory by ``run_id``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._entries: dict[str, RegistryEntry] = {}
        self._exclusions: tuple[Exclusion, ...] = ()
        self.refresh()

    # ------------------------------------------------------------------ scan
    def refresh(self) -> RegistryScan:
        """Rescan the runs directory; the registry reflects the result."""
        if not self.root.exists():
            raise RunRegistryError(f"runs directory {self.root} does not exist")
        if not self.root.is_dir():
            raise RunRegistryError(f"runs root {self.root} is not a directory")

        exclusions: list[Exclusion] = []
        candidates: list[tuple[Path, dict[str, Any]]] = []
        total_dirs = 0
        try:
            children = sorted(self.root.iterdir())
        except OSError as exc:
            raise RunRegistryError(f"cannot list runs root {self.root}: {exc}") from exc
        for child in children:
            try:
                is_dir = child.is_dir()
            except OSError:
                exclusions.append(Exclusion(child, "unreadable_run_dir"))
                continue
            if not is_dir:
                continue
            total_dirs += 1
            manifest_path = child / "manifest.json"
            try:
                exists = manifest_path.is_file()
            except OSError:
                exclusions.append(Exclusion(child, "unreadable_manifest"))
                continue
            if not exists:
                exclusions.append(Exclusion(child, "missing_manifest"))
                continue
            try:
                raw = manifest_path.read_bytes()
            except OSError:
                exclusions.append(Exclusion(child, "unreadable_manifest"))
                continue
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except UnicodeError:
                exclusions.append(Exclusion(child, "invalid_manifest_encoding"))
                continue
            except json.JSONDecodeError:
                exclusions.append(Exclusion(child, "invalid_manifest_json"))
                continue
            manifest, reason = _validate_manifest(parsed)
            if manifest is None:
                assert reason is not None
                exclusions.append(Exclusion(child, reason))
                continue
            candidates.append((child, manifest))

        # Deterministic duplicate handling: any run_id claimed by more than one
        # directory is excluded everywhere it appears.
        counts: dict[str, int] = {}
        for _, manifest in candidates:
            counts[manifest["run_id"]] = counts.get(manifest["run_id"], 0) + 1

        entries: dict[str, RegistryEntry] = {}
        for run_dir, manifest in candidates:
            run_id = manifest["run_id"]
            if counts[run_id] > 1:
                exclusions.append(Exclusion(run_dir, "duplicate_run_id"))
                continue
            # A manifest may never claim a different run than its directory.
            if run_id != run_dir.name:
                exclusions.append(Exclusion(run_dir, "run_id_mismatch"))
                continue
            try:
                checkpoint_paths = _checkpoint_paths(run_dir)
            except OSError:
                exclusions.append(Exclusion(run_dir, "unreadable_checkpoints"))
                continue
            entries[run_id] = RegistryEntry(
                run_id=run_id,
                manifest_path=run_dir / "manifest.json",
                manifest=manifest,
                config_hash=manifest["config_hash"],
                data_fingerprint=manifest["data_fingerprint"],
                checkpoint_paths=checkpoint_paths,
            )

        # Artefacts directly in root (non-directory files) are not run dirs and
        # are ignored entirely — they are not runs and count as neither.
        self._entries = entries
        self._exclusions = tuple(exclusions)
        return RegistryScan(
            total_run_dirs=total_dirs,
            indexed=len(entries),
            excluded=len(exclusions),
        )

    # -------------------------------------------------------------- queries
    def list(self) -> list[str]:
        """Sorted list of indexed run ids."""
        return sorted(self._entries)

    def get(self, run_id: str) -> RegistryEntry:
        """Return the entry for ``run_id``; raises :class:`UnknownRunError` otherwise."""
        try:
            return self._entries[run_id]
        except KeyError:
            raise UnknownRunError(
                f"run {run_id!r} is not indexed "
                f"({len(self._entries)} indexed, {len(self._exclusions)} excluded)"
            ) from None

    @property
    def excluded_count(self) -> int:
        return len(self._exclusions)

    @property
    def exclusions(self) -> tuple[Exclusion, ...]:
        return self._exclusions

    @property
    def entries(self) -> tuple[RegistryEntry, ...]:
        return tuple(self._entries[run_id] for run_id in self.list())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, run_id: object) -> bool:
        return run_id in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.list())
