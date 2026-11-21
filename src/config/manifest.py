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
import os
import platform
import re
import secrets
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .errors import ManifestRunIdError, ManifestWriteError
from .loader import REPO_ROOT, ResolvedConfig

_VERSION_PACKAGES = ("torch", "transformers", "datasets", "onnxruntime")
_LOCK_FILE = REPO_ROOT / "requirements.lock"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
_SEPARATORS: tuple[str, ...] = tuple(s for s in (os.sep, os.altsep, "/", "\\") if s)


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
    # sha256 of requirements.lock at build time; None when the lock is absent.
    # Optional so hand-built manifests (e2e) stay source-compatible.
    dependency_lock_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_run_id(value: object) -> str:
    """Validate ``value`` as a single, portable, non-traversing directory name."""
    if not isinstance(value, str) or not value.strip():
        raise ManifestRunIdError(f"run_id must be a non-empty string, got {value!r}")
    if value != value.strip() or "\x00" in value or any(ord(c) < 32 for c in value):
        raise ManifestRunIdError(f"run_id contains control or surrounding whitespace: {value!r}")
    if Path(value).is_absolute() or any(sep in value for sep in _SEPARATORS):
        raise ManifestRunIdError(
            f"run_id must not be absolute or contain path separators: {value!r}"
        )
    if value in (".", "..") or value.endswith("."):
        raise ManifestRunIdError(f"run_id must not be '.' or '..' or end with a dot: {value!r}")
    if not _RUN_ID_RE.fullmatch(value):
        raise ManifestRunIdError(
            f"run_id {value!r} is not portable; use letters, digits, '.', '_' or '-' only"
        )
    if value.upper() in _RESERVED_NAMES:
        raise ManifestRunIdError(f"run_id {value!r} is a reserved filesystem name")
    return value


def compute_config_hash(resolved: ResolvedConfig) -> str:
    """sha256 over the resolved config, serialized as canonical YAML."""
    canonical = yaml.safe_dump(resolved.to_dict(), sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd or REPO_ROOT),
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


def read_git_state(repo_root: str | Path | None = None) -> tuple[str, bool | None]:
    """Return (sha, dirty) for the repository at ``repo_root``.

    Defaults to ``REPO_ROOT`` (the checkout that owns this source file) rather
    than the process CWD, so a run launched from a different directory still
    records the correct commit. A failure to resolve git yields
    ``("unknown", None)`` — recorded honestly instead of failing the run.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    sha = _git("rev-parse", "HEAD", cwd=root)
    if sha is None:
        return "unknown", None
    porcelain = _git("status", "--porcelain", cwd=root)
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


def read_dependency_lock_digest(path: str | Path | None = None) -> str | None:
    """sha256 of the pinned requirements lock, or None when it is absent."""
    lock = Path(path) if path is not None else _LOCK_FILE
    try:
        return hashlib.sha256(lock.read_bytes()).hexdigest()
    except OSError:
        return None


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:  # some filesystems do not support directory fsync
        pass
    finally:
        os.close(fd)


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
    # Timestamp + hash alone collide for runs started in the same second with
    # the same config. pid + random nonce make the id collision-resistant.
    nonce = secrets.token_hex(6)
    return RunManifest(
        run_id=f"{moment:%Y%m%dT%H%M%SZ}-{config_hash[:8]}-{os.getpid():x}-{nonce}",
        git_sha=git_sha,
        git_dirty=git_dirty,
        config_hash=config_hash,
        data_fingerprint=resolved.data_fingerprint,
        seed=resolved.seed,
        device=device,
        compute_capability=compute_capability,
        determinism=resolved.determinism,
        versions=read_versions(),
        dependency_lock_sha256=read_dependency_lock_digest(),
    )


def write_run_manifest(manifest: RunManifest, runs_dir: str | Path) -> Path:
    """Write ``manifest`` as ``<runs_dir>/<run_id>/manifest.json``.

    Raises ManifestWriteError (with the underlying cause chained) if the
    directory or file cannot be created/written.
    """
    run_id = validate_run_id(manifest.run_id)
    root = Path(runs_dir).resolve()
    target_dir = root / run_id
    # Resolve the candidate before *any* mkdir/write: an absolute run_id,
    # a nested path or a symlink must never escape the runs root.
    if target_dir.resolve().parent != root or target_dir == root:
        raise ManifestRunIdError(
            f"run_id {run_id!r} resolves outside the runs root {root}", key="run_id"
        )

    target = target_dir / "manifest.json"
    payload = (json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp: Path | None = None
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        if target_dir.resolve().parent != root:  # re-check after mkdir (TOCTOU)
            raise ManifestRunIdError(
                f"run_id {run_id!r} resolves outside the runs root {root}", key="run_id"
            )
        tmp = target_dir / f".manifest.json.{os.getpid()}.{secrets.token_hex(4)}.tmp"
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        _fsync_dir(target_dir)
    except OSError as exc:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        raise ManifestWriteError(f"cannot write manifest to {target}: {exc}") from exc
    return target
