"""Unit tests for src.training.core.checkpoint — atomic writes and hashing."""

from __future__ import annotations

import hashlib
import random
import re
import sys
from pathlib import Path

import numpy as np
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
    CheckpointError,
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


def test_unknown_format_version_is_refused(tmp_path: Path) -> None:
    final = save_checkpoint(_payload(3), tmp_path, 3)
    payload = torch.load(final, map_location="cpu", weights_only=True)
    payload["format_version"] = 999
    torch.save(payload, final)
    # Keep the hash consistent so the refusal is specifically about version.
    sidecar_path(final).write_text(
        f"{hashlib.sha256(final.read_bytes()).hexdigest()}  {final.name}\n", encoding="utf-8"
    )
    with pytest.raises(CheckpointError, match="format_version") as excinfo:
        load_checkpoint_payload(final)
    assert excinfo.value.key == "format_version"


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


def _trainer_like_payload() -> dict:
    """Mirror the real trainer.state_dict() types, including numpy RNG state."""
    return {
        "step": 7,
        "epoch": 1,
        "sampler_offset": 42,
        "model": {"w": torch.arange(4, dtype=torch.float32).reshape(2, 2)},
        "optimizer": {
            "state": {0: {"exp_avg": torch.zeros(2, 2), "step": 3}},
            "param_groups": [{"lr": 3e-4, "betas": (0.9, 0.98), "params": [0]}],
        },
        "scheduler": {"base_lrs": [3e-4], "last_epoch": 7, "_step_count": 8},
        "scaler": {"scale": 65536.0, "growth_factor": 2.0, "enabled": True},
        "weighter": {"log_s": torch.zeros(7)},
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": [],
        },
        "config": {"seed": 13, "fp16": True, "weight_mode": "fixed", "keep_last_n": None},
    }


def test_real_trainer_like_payload_round_trips_with_weights_only(tmp_path: Path) -> None:
    original = _trainer_like_payload()
    final = save_checkpoint(original, tmp_path, 7)
    restored = load_checkpoint_payload(final)

    assert restored["config"] == original["config"]
    assert torch.equal(restored["model"]["w"], original["model"]["w"])
    assert torch.equal(restored["rng"]["torch"], original["rng"]["torch"])

    # The RNG states must come back in the exact shape _restore_rng() expects.
    python_state = restored["rng"]["python"]
    assert isinstance(python_state, tuple)
    random.setstate(python_state)  # raises if the tuple structure was lost

    numpy_state = restored["rng"]["numpy"]
    assert isinstance(numpy_state, tuple) and len(numpy_state) == 5
    assert isinstance(numpy_state[1], np.ndarray)
    assert numpy_state[1].dtype == np.dtype(np.uint32)
    np.random.set_state(numpy_state)

    # No numpy marker dicts should leak into the returned payload.
    assert "__rapport_numpy_value__" not in repr(restored)


_REDUCER_RAN: list[str] = []


def _malicious_reducer() -> dict:
    """Reducer a malicious checkpoint would ask torch.load to execute."""
    _REDUCER_RAN.append("executed")
    return {"format_version": 2, "step": 0}


class _MaliciousPickle:
    def __reduce__(self):  # noqa: ANN204 - pickle protocol requires this shape
        return _malicious_reducer, ()


def test_malicious_pickle_with_valid_sidecar_is_rejected(tmp_path: Path, monkeypatch) -> None:
    """A forged-but-hash-consistent pickle must not run code (RCE regression)."""
    import torch

    monkeypatch.setattr(sys.modules[__name__], "_REDUCER_RAN", [])
    malicious = tmp_path / "checkpoint_step99.pt"
    torch.save(_MaliciousPickle(), malicious)
    # The attacker can write the sidecar too — integrity alone is not safety.
    sidecar_path(malicious).write_text(
        f"{hashlib.sha256(malicious.read_bytes()).hexdigest()}  {malicious.name}\n",
        encoding="utf-8",
    )

    assert verify_checkpoint(malicious) == sha256_file(malicious)  # sidecar is valid
    with pytest.raises(CheckpointError, match="weights_only=True"):
        load_checkpoint_payload(malicious)
    assert _REDUCER_RAN == []  # the reducer never executed


def test_reserved_marker_mapping_is_escaped_and_round_trips(tmp_path: Path) -> None:
    payload = {"metadata": {"__rapport_numpy_value__": "literal-user-value"}}
    path = save_checkpoint(payload, tmp_path, 1)
    restored = load_checkpoint_payload(path)
    assert restored["metadata"] == payload["metadata"]


def test_object_array_is_rejected_at_save_time(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="object arrays"):
        save_checkpoint(
            {"bad": np.array([{"process_local": 1}], dtype=object)}, tmp_path, 1
        )


def test_malformed_safe_marker_raises_typed_checkpoint_error(tmp_path: Path) -> None:
    payload = {
        "format_version": 2,
        "step": 1,
        "value": {"__rapport_numpy_value__": "ndarray"},
    }
    path = tmp_path / "checkpoint_step1.pt"
    torch.save(payload, path)
    sidecar_path(path).write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(CheckpointError):
        load_checkpoint_payload(path)

