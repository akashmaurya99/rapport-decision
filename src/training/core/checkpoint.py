"""Atomic checkpoint I/O with sha256 sidecars (TASK-12).

Design (training-strategy.md §8):

* file format ``checkpoint_step{N}.pt``;
* written to ``*.tmp`` first, ``fsync``ed, then ``os.replace``ed into place —
  a partial checkpoint is never loadable under its final name;
* a ``<file>.sha256`` sidecar carrying the checkpoint's sha256 digest is
  written the same way;
* on load, the sidecar digest is verified against the file **before**
  ``torch.load`` is reached — a mismatch raises
  :class:`CheckpointHashMismatchError` and the payload is never deserialized.

``latest_checkpoint`` implements the "resume auto" discovery used by the Colab
re-run procedure (t4-optimization.md §10): highest step wins.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from .errors import (
    CheckpointError,
    CheckpointHashMismatchError,
    CheckpointSidecarMissingError,
)

CHECKPOINT_PREFIX = "checkpoint_step"
CHECKPOINT_SUFFIX = ".pt"
SIDECAR_SUFFIX = ".sha256"
TMP_SUFFIX = ".tmp"
_FORMAT_VERSION = 1
_NAME_RE = re.compile(r"^checkpoint_step(\d+)\.pt$")
_CHUNK = 1 << 20


def checkpoint_filename(step: int) -> str:
    return f"{CHECKPOINT_PREFIX}{step}{CHECKPOINT_SUFFIX}"


def checkpoint_path(directory: str | Path, step: int) -> Path:
    return Path(directory) / checkpoint_filename(step)


def sidecar_path(path: str | Path) -> Path:
    return Path(str(path) + SIDECAR_SUFFIX)


def step_from_filename(path: str | Path) -> int | None:
    match = _NAME_RE.match(Path(path).name)
    return int(match.group(1)) if match else None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + TMP_SUFFIX)
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def save_checkpoint(payload: dict[str, Any], directory: str | Path, step: int) -> Path:
    """Serialize ``payload`` atomically and write its ``.sha256`` sidecar."""
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["format_version"] = _FORMAT_VERSION
    body["step"] = int(step)

    final = checkpoint_path(directory, step)
    tmp = final.parent / (final.name + TMP_SUFFIX)
    with open(tmp, "wb") as handle:
        torch.save(body, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, final)
    _fsync_dir(final.parent)

    digest = sha256_file(final)
    _atomic_write(sidecar_path(final), f"{digest}  {final.name}\n".encode("utf-8"))
    return final


def read_sidecar(path: str | Path) -> str:
    sidecar = sidecar_path(path)
    try:
        text = sidecar.read_text(encoding="utf-8")
    except OSError as exc:
        raise CheckpointSidecarMissingError(
            f"cannot read sha256 sidecar {sidecar}: {exc}; refusing to load the checkpoint",
            key="sidecar",
        ) from exc
    token = text.split()[0] if text.split() else ""
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise CheckpointSidecarMissingError(
            f"sha256 sidecar {sidecar} does not carry a lowercase hex digest",
            key="sidecar",
        )
    return token


def verify_checkpoint(path: str | Path) -> str:
    """Return the file's sha256 after checking it against the sidecar.

    Raises :class:`CheckpointSidecarMissingError` when the sidecar is absent
    or malformed, and :class:`CheckpointHashMismatchError` when the file does
    not hash to the sidecar's digest. On mismatch the file is never opened
    for deserialization.
    """
    path = Path(path)
    expected = read_sidecar(path)
    actual = sha256_file(path)
    if actual != expected:
        raise CheckpointHashMismatchError(
            f"checkpoint {path} sha256 mismatch: sidecar expects {expected}, "
            f"file hashes to {actual}; the checkpoint is rejected and never loaded",
            path=str(path),
            expected=expected,
            actual=actual,
        )
    return actual


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """Verify the checkpoint's hash, then deserialize it.

    The hash check always runs first; ``torch.load`` is never reached for a
    checkpoint whose file does not match its sidecar.
    """
    import torch

    path = Path(path)
    verify_checkpoint(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"checkpoint {path} could not be deserialized: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"checkpoint {path} did not contain a mapping payload")
    return payload


def latest_checkpoint(directory: str | Path) -> Path | None:
    """Return the highest-step ``checkpoint_step*.pt`` in ``directory``, if any."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for child in directory.iterdir():
        if not child.is_file():
            continue
        step = step_from_filename(child)
        if step is None:
            continue
        if best is None or step > best[0]:
            best = (step, child)
    return best[1] if best else None
