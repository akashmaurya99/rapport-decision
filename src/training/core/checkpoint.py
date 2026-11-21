"""Atomic checkpoint I/O with sha256 sidecars (TASK-12).

Design (training-strategy.md §8):

* file format ``checkpoint_step{N}.pt``;
* written to ``*.tmp`` first, ``fsync``ed, then ``os.replace``ed into place —
  a partial checkpoint is never loadable under its final name;
* a ``<file>.sha256`` sidecar carrying the checkpoint's sha256 digest is
  written the same way;
* on load, the sidecar digest is verified against the file **before**
  ``torch.load`` is reached — a mismatch raises
  :class:`CheckpointHashMismatchError` and the payload is never deserialized;
* deserialization uses ``torch.load(..., weights_only=True)``. A matching
  sidecar only proves the bytes were not altered on disk — it does **not**
  make an untrusted pickle safe to unpickle, so the unsafe pickle path is
  never taken (SECURITY: the sidecar is unkeyed and anyone who can write
  ``runs/`` can forge both the checkpoint and its digest).

The trainer's payload contains numpy RNG state (``numpy.random.get_state()``
returns a tuple holding an ndarray), which the weights-only unpickler refuses.
Rather than reverting to unsafe loading, :func:`save_checkpoint` converts
numpy values into a small, explicitly marked dict of primitives (dtype string,
shape, raw bytes) and :func:`load_checkpoint_payload` restores them before the
payload is returned. The trainer's ``_restore_rng`` therefore still receives
the original tuple-of-(str, ndarray, int, int, float) shape.

Format version 2 is intentionally distinct from v1: v1 checkpoints were
written with the unsafe loader and cannot be read under weights-only rules,
so they are rejected with a clear error instead of being silently downgraded.

``latest_checkpoint`` implements the "resume auto" discovery used by the Colab
re-run procedure (t4-optimization.md §10): highest step wins.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .errors import (
    CheckpointError,
    CheckpointHashMismatchError,
    CheckpointSidecarMissingError,
)

CHECKPOINT_PREFIX = "checkpoint_step"
CHECKPOINT_SUFFIX = ".pt"
SIDECAR_SUFFIX = ".sha256"
TMP_SUFFIX = ".tmp"
_FORMAT_VERSION = 2
_SAFE_MARKER = "__rapport_numpy_value__"
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


def _safe_key(key: Any) -> Any:
    """Mapping keys must stay hashable and weights-only friendly."""
    if isinstance(key, np.generic):
        return key.item()
    if isinstance(key, np.ndarray) and key.size == 1:
        return key.item()
    return key


def _to_safe(value: Any) -> Any:
    """Recursively turn numpy values into weights-only-safe primitives.

    Supported values are the types a real trainer checkpoint contains:
    tensors, strings, numbers, booleans, bytes, ``None``, lists, tuples,
    mappings and numpy arrays/scalars. Anything else is a programming error
    and is rejected at save time with :class:`CheckpointError` rather than
    producing a checkpoint that cannot be loaded safely.
    """
    import torch

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise CheckpointError("object arrays are not supported in safe checkpoints")
        array = np.ascontiguousarray(value)
        return {
            _SAFE_MARKER: "ndarray",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": array.tobytes(),
        }
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise CheckpointError("object scalars are not supported in safe checkpoints")
        return {
            _SAFE_MARKER: "scalar",
            "dtype": value.dtype.str,
            "data": value.tobytes(),
        }
    if isinstance(value, Mapping):
        if _SAFE_MARKER in value:
            # Escape literal user mappings that happen to contain our reserved
            # key; otherwise decoding would mistake them for ndarray metadata.
            return {
                _SAFE_MARKER: "mapping",
                "items": [(_safe_key(key), _to_safe(item)) for key, item in value.items()],
            }
        return {_safe_key(key): _to_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        # Preserve tuple-ness: random.setstate()/np.random.setstate() require it.
        return tuple(_to_safe(item) for item in value)
    if isinstance(value, list):
        return [_to_safe(item) for item in value]
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, (str, int, float, bool, bytes, type(None))) or torch.is_tensor(value):
        return value
    raise CheckpointError(
        f"unsupported checkpoint value type {type(value).__name__}; checkpoint payloads "
        "must use tensors or weights-only-safe primitives"
    )


def _from_safe(value: Any) -> Any:
    """Inverse of :func:`_to_safe`, with strict, typed payload validation."""
    try:
        if isinstance(value, Mapping):
            if _SAFE_MARKER in value:
                tag = value[_SAFE_MARKER]
                if tag == "mapping":
                    items = value["items"]
                    if not isinstance(items, (list, tuple)):
                        raise CheckpointError("invalid escaped mapping payload")
                    return {key: _from_safe(item) for key, item in items}
                if tag not in {"ndarray", "scalar"}:
                    raise CheckpointError(f"unknown safe checkpoint value tag {tag!r}")
                dtype = np.dtype(value["dtype"])
                if dtype.hasobject:
                    raise CheckpointError("object arrays are not supported in safe checkpoints")
                data = bytes(value["data"])
                if tag == "ndarray":
                    shape = value["shape"]
                    if not isinstance(shape, (list, tuple)) or any(
                        type(dim) is not int or dim < 0 for dim in shape
                    ):
                        raise CheckpointError("invalid ndarray shape in checkpoint")
                    expected = dtype.itemsize
                    for dim in shape:
                        expected *= dim
                    if len(data) != expected:
                        raise CheckpointError("ndarray byte length does not match dtype/shape")
                    array = np.frombuffer(data, dtype=dtype).reshape(tuple(shape))
                    return np.array(array, copy=True)  # writable, like the original
                if len(data) != dtype.itemsize:
                    raise CheckpointError("scalar byte length does not match dtype")
                return np.frombuffer(data, dtype=dtype)[0]
            return {key: _from_safe(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(_from_safe(item) for item in value)
        if isinstance(value, list):
            return [_from_safe(item) for item in value]
        return value
    except CheckpointError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(f"malformed safe checkpoint value: {exc}") from exc


def save_checkpoint(payload: dict[str, Any], directory: str | Path, step: int) -> Path:
    """Serialize ``payload`` atomically (weights-only safe) plus its sidecar."""
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    body = _to_safe(payload)
    if not isinstance(body, dict):
        raise CheckpointError("checkpoint payload must be a mapping")
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
    """Verify the checkpoint's hash, then deserialize it with weights only.

    The hash check always runs first; ``torch.load`` is never reached for a
    checkpoint whose file does not match its sidecar. The sidecar is unkeyed
    (a hash is integrity, not authenticity), so ``weights_only=True`` is the
    actual security boundary: a malicious pickle with a *valid* sidecar is
    rejected before any reducer can execute, raising :class:`CheckpointError`.
    """
    import torch

    path = Path(path)
    verify_checkpoint(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(
            f"checkpoint {path} could not be deserialized safely (weights_only=True): {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"checkpoint {path} did not contain a mapping payload")
    version = payload.get("format_version")
    if version != _FORMAT_VERSION:
        raise CheckpointError(
            f"checkpoint {path} has format_version {version!r}, expected "
            f"{_FORMAT_VERSION}; refusing to load an incompatible or foreign checkpoint "
            "(v1 checkpoints used unsafe pickle loading and are intentionally rejected)",
            key="format_version",
        )
    return _from_safe(payload)


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
