"""Run manifest assembly and writing.

A manifest is written for every run (FR-CONFIG-02). It records the git sha
(and whether the tree was dirty — recorded, never refused), a sha256 over the
resolved config, the data fingerprint read from the environment placeholder,
the seed, the device, and the versions of python, torch, transformers,
datasets and onnxruntime (absent libraries are recorded as null, never
invented).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .errors import ManifestWriteError
from .loader import ResolvedConfig

_VERSION_PACKAGES = ("torch", "transformers", "datasets", "onnxruntime")


@dataclass(frozen=True)
class RunManifest:
    """The record of one run, serialized to ``runs/<run_id>/manifest.json``."""

    run_id: str
    git_sha: str
    git_dirty: bool | None
    config_hash: str
    data_fingerprint: str
    seed: int
    device: str
    compute_capability: str | None
    determinism: str
    versions: dict[str, str | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_config_hash(resolved: ResolvedConfig) -> str:
    """sha256 over the resolved config, serialized as canonical YAML."""
    canonical = yaml.safe_dump(resolved.to_dict(), sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def read_git_state() -> tuple[str, bool | None]:
    """Return (sha, dirty) for the current repository, tolerating absence.

    A failure to resolve git yields ``("unknown", None)`` — the manifest then
    records the absence honestly instead of failing the run.
    """
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return "unknown", None
    porcelain = _git("status", "--porcelain")
    dirty = None if porcelain is None else bool(porcelain)
    return sha, dirty


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in _VERSION_PACKAGES:
        versions[name] = _package_version(name)
    return versions


def _describe_device(device: str) -> tuple[str, str | None]:
    """Return (device string, compute capability) for the manifest."""
    if device != "cuda":
        return device, None
    try:
        import torch
    except ImportError:
        return "cuda (torch not importable)", None
    if not torch.cuda.is_available():
        return "cuda (unavailable)", None
    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    major, minor = torch.cuda.get_device_capability(index)
    return f"cuda:{index} {name} (sm_{major}{minor})", f"{major}.{minor}"


def build_run_manifest(
    resolved: ResolvedConfig,
    *,
    now: datetime | None = None,
) -> RunManifest:
    """Assemble the manifest for one resolved config.

    ``now`` exists so tests can pin the timestamp embedded in ``run_id``;
    production callers leave it as ``None``.
    """
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    config_hash = compute_config_hash(resolved)
    git_sha, git_dirty = read_git_state()
    device, compute_capability = _describe_device(resolved.device)
    return RunManifest(
        run_id=f"{moment:%Y%m%dT%H%M%SZ}-{config_hash[:8]}",
        git_sha=git_sha,
        git_dirty=git_dirty,
        config_hash=config_hash,
        data_fingerprint=resolved.data_fingerprint,
        seed=resolved.seed,
        device=device,
        compute_capability=compute_capability,
        determinism=resolved.determinism,
        versions=read_versions(),
    )


def write_run_manifest(manifest: RunManifest, runs_dir: str | Path) -> Path:
    """Write ``manifest`` as ``<runs_dir>/<run_id>/manifest.json``.

    Raises ManifestWriteError (with the underlying cause chained) if the
    directory or file cannot be created/written.
    """
    target_dir = Path(runs_dir) / manifest.run_id
    target = target_dir / "manifest.json"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ManifestWriteError(f"cannot write manifest to {target}: {exc}") from exc
    return target
