"""Run registry (FR-OPS-02).

Scans a runs directory (``runs/<run_id>/manifest.json`` layout) and indexes
every valid run by its ``run_id``. A run is *valid* only when its
``manifest.json`` exists, parses as a JSON object, and carries non-empty
string ``run_id``, ``config_hash`` and ``data_fingerprint`` fields — the full
required set per the RunManifest contract in ``.archgen/rapport/architecture.yaml``.

A run that fails any of those checks is **excluded**, never silently listed:
the number of exclusions and their per-directory reasons are exposed via
:attr:`RunRegistry.excluded_count` and :attr:`RunRegistry.exclusions`.

``list()`` returns the indexed run ids sorted lexicographically; ``get``
returns the manifest path plus the parsed manifest. Registry entries also
carry the checkpoint paths the run produced (files under ``checkpoints/`` and
top-level ``*.pt`` / ``*.pth`` / ``*.ckpt`` / ``*.safetensors`` files).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .errors import RunRegistryError, UnknownRunError

CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors")

REQUIRED_MANIFEST_FIELDS = ("run_id", "config_hash", "data_fingerprint")


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
    """Return (manifest, None) when valid, else (None, reason)."""
    if not isinstance(parsed, dict):
        return None, "manifest_not_an_object"
    for field in REQUIRED_MANIFEST_FIELDS:
        value = parsed.get(field)
        if value is None:
            return None, f"missing_{field}"
        if not isinstance(value, str) or not value.strip():
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
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            total_dirs += 1
            manifest_path = child / "manifest.json"
            if not manifest_path.is_file():
                exclusions.append(Exclusion(child, "missing_manifest"))
                continue
            try:
                parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                exclusions.append(Exclusion(child, "invalid_manifest_json"))
                continue
            except OSError:
                exclusions.append(Exclusion(child, "unreadable_manifest"))
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
            entries[run_id] = RegistryEntry(
                run_id=run_id,
                manifest_path=run_dir / "manifest.json",
                manifest=manifest,
                config_hash=manifest["config_hash"],
                data_fingerprint=manifest["data_fingerprint"],
                checkpoint_paths=_checkpoint_paths(run_dir),
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
