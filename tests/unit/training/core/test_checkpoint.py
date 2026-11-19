"""Unit tests for src.training.core.checkpoint — atomic writes and hashing."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.core.checkpoint import (  # noqa: E402
    checkpoint_filename,
    checkpoint_path,
    latest_checkpoint,
    load_checkpoint_payload,
    save_checkpoint,
    sha256_file,
    sidecar_path,
    step_from_filename,
    verify_checkpoint,
)
from src.training.core.errors import (  # noqa: E402
    CheckpointHashMismatchError,
    CheckpointSidecarMissingError,
)


def _payload(step: int) -> dict:
    return {
        "step": step,
        "epoch": 1,
        "sampler_offset": 42,
        "model": {"w": torch.ones(2, 2)},
        "rng": {"torch": torch.get_rng_state()},
    }


def test_filename_format() -> None:
    assert checkpoint_filename(500) == "checkpoint_step500.pt"
    assert checkpoint_filename(0) == "checkpoint_step0.pt"
    assert step_from_filename("checkpoint_step123.pt") == 123
    assert step_from_filename("checkpoint_step.pt") is None
    assert step_from_filename("other.pt") is None


def test_save_is_atomic_and_writes_sidecar(tmp_path: Path) -> None:
    final = save_checkpoint(_payload(7), tmp_path, 7)
    assert final == checkpoint_path(tmp_path, 7)
    assert final.name == "checkpoint_step7.pt"
    assert not (tmp_path / "checkpoint_step7.pt.tmp").exists()
    sidecar = sidecar_path(final)
    assert sidecar.exists()
    text = sidecar.read_text(encoding="utf-8")
    digest, filename = text.split()
    assert filename == final.name
    assert digest == hashlib.sha256(final.read_bytes()).hexdigest()


def test_verify_checkpoint_returns_digest(tmp_path: Path) -> None:
    final = save_checkpoint(_payload(3), tmp_path, 3)
    assert verify_checkpoint(final) == sha256_file(final)


def test_hash_mismatch_is_rejected_and_never_loaded(tmp_path: Path) -> None:
    final = save_checkpoint(_payload(5), tmp_path, 5)
    # Corrupt the file but leave the sidecar untouched.
    final.write_bytes(b"this is not a torch checkpoint")
    with pytest.raises(CheckpointHashMismatchError) as excinfo:
        load_checkpoint_payload(final)
    assert excinfo.value.expected != excinfo.value.actual
    assert excinfo.value.path == str(final)


def test_hash_mismatch_after_single_byte_flip(tmp_path: Path) -> None:
    final = save_checkpoint(_payload(9), tmp_path, 9)
    data = bytearray(final.read_bytes())
    data[len(data) // 2] ^= 0x01
    final.write_bytes(bytes(data))
    with pytest.raises(CheckpointHashMismatchError):
        verify_checkpoint(final)


def test_missing_sidecar_is_refused(tmp_path: Path) -> None:
    final = save_checkpoint(_payload(11), tmp_path, 11)
    sidecar_path(final).unlink()
    with pytest.raises(CheckpointSidecarMissingError):
        verify_checkpoint(final)
    with pytest.raises(CheckpointSidecarMissingError):
        load_checkpoint_payload(final)


def test_malformed_sidecar_is_refused(tmp_path: Path) -> None:
    final = save_checkpoint(_payload(13), tmp_path, 13)
    sidecar_path(final).write_text("not-a-digest\n", encoding="utf-8")
    with pytest.raises(CheckpointSidecarMissingError):
        verify_checkpoint(final)


def test_latest_checkpoint_picks_highest_step(tmp_path: Path) -> None:
    for step in (100, 500, 250):
        save_checkpoint(_payload(step), tmp_path, step)
    (tmp_path / "checkpoint_stepabc.pt").write_bytes(b"junk")
    (tmp_path / "unrelated.pt").write_bytes(b"junk")
    latest = latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint_step500.pt"
    assert latest_checkpoint(tmp_path / "does-not-exist") is None


def test_tmp_file_not_left_behind_on_success(tmp_path: Path) -> None:
    save_checkpoint(_payload(1), tmp_path, 1)
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_step_from_filename_regex(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint_step42.pt"
    assert step_from_filename(path) == 42
    assert re.fullmatch(r"checkpoint_step\d+\.pt", checkpoint_filename(42))
