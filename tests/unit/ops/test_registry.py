"""Tests for src/ops/registry.py — FR-OPS-02 run registry indexing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ops import (
    RunRegistry,
    RunRegistryError,
    UnknownRunError,
)

_UNSET: object = object()
_HASH = "c0ffee" * 10 + "c0ff"  # sha256-shaped: 64 lowercase hex chars


def make_run(
    root: Path,
    name: str,
    *,
    run_id: object = _UNSET,
    config_hash: object = _UNSET,
    data_fingerprint: object = _UNSET,
    git_sha: str = "abc123",
    seed: int = 13,
    checkpoints: tuple[str, ...] = (),
    manifest_extras: bool = True,
    write_manifest: bool = True,
) -> Path:
    """Create a run directory; ``None`` for a field omits it from the manifest."""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    if write_manifest:
        rid = name if run_id is _UNSET else run_id
        ch = _HASH if config_hash is _UNSET else config_hash
        fp = "deadbeef" * 8 if data_fingerprint is _UNSET else data_fingerprint
        payload: dict[str, object] = {}
        if rid is not None:
            payload["run_id"] = rid
        if ch is not None:
            payload["config_hash"] = ch
        if fp is not None:
            payload["data_fingerprint"] = fp
        if manifest_extras:
            payload.update(
                {
                    "git_sha": git_sha,
                    "seed": seed,
                    "device": "cpu",
                    "determinism": "strict",
                    "versions": {"python": "3.12.0", "torch": "2.x"},
                }
            )
        (run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    for checkpoint in checkpoints:
        target = run_dir / checkpoint
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"ckpt")
    return run_dir


def test_valid_run_is_indexed(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    make_run(root, "run-a")

    registry = RunRegistry(root)

    assert registry.list() == ["run-a"]
    assert len(registry) == 1
    assert "run-a" in registry
    entry = registry.get("run-a")
    assert entry.manifest_path == root / "run-a" / "manifest.json"
    assert entry.manifest["git_sha"] == "abc123"
    assert entry.config_hash == _HASH
    assert entry.data_fingerprint == "deadbeef" * 8
    assert entry.checkpoint_paths == ()
    assert registry.excluded_count == 0


def test_list_is_sorted(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    for name in ("run-c", "run-a", "run-b"):
        make_run(root, name)
    registry = RunRegistry(root)
    assert registry.list() == ["run-a", "run-b", "run-c"]


def test_run_without_manifest_is_excluded_and_counted(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    make_run(root, "no-manifest", write_manifest=False)
    make_run(root, "good")

    registry = RunRegistry(root)

    assert registry.list() == ["good"]
    assert registry.excluded_count == 1
    assert registry.exclusions[0].run_dir == root / "no-manifest"
    assert registry.exclusions[0].reason == "missing_manifest"


@pytest.mark.parametrize(
    ("config_hash", "data_fingerprint", "reason"),
    [
        (None, "deadbeef" * 8, "missing_config_hash"),
        ("", "deadbeef" * 8, "invalid_config_hash"),
        (_HASH, None, "missing_data_fingerprint"),
        (_HASH, "   ", "invalid_data_fingerprint"),
    ],
)
def test_run_missing_fingerprint_or_hash_is_excluded(
    tmp_path: Path,
    config_hash: str | None,
    data_fingerprint: str | None,
    reason: str,
) -> None:
    root = tmp_path / "runs"
    make_run(root, "bad", config_hash=config_hash, data_fingerprint=data_fingerprint)

    registry = RunRegistry(root)

    assert registry.list() == []
    assert registry.excluded_count == 1
    assert registry.exclusions[0].reason == reason


def test_manifest_missing_run_id_is_excluded(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run_dir = root / "no-id"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"config_hash": "x" * 8, "data_fingerprint": "y" * 8}), encoding="utf-8"
    )
    registry = RunRegistry(root)
    assert registry.list() == []
    assert registry.exclusions[0].reason == "missing_run_id"


def test_corrupt_manifest_is_excluded(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run_dir = root / "corrupt"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{not json", encoding="utf-8")
    registry = RunRegistry(root)
    assert registry.list() == []
    assert registry.exclusions[0].reason == "invalid_manifest_json"


def test_manifest_that_is_not_an_object_is_excluded(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run_dir = root / "listy"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")
    registry = RunRegistry(root)
    assert registry.list() == []
    assert registry.exclusions[0].reason == "manifest_not_an_object"


def test_duplicate_run_id_excluded_everywhere(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    make_run(root, "dir-a", run_id="same-id")
    make_run(root, "dir-b", run_id="same-id")
    make_run(root, "unique")

    registry = RunRegistry(root)

    assert registry.list() == ["unique"]
    assert registry.excluded_count == 2
    assert {e.reason for e in registry.exclusions} == {"duplicate_run_id"}


def test_get_unknown_run_raises(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    make_run(root, "good")
    registry = RunRegistry(root)
    with pytest.raises(UnknownRunError) as excinfo:
        registry.get("nope")
    assert "nope" in str(excinfo.value)


def test_missing_root_raises_typed_error(tmp_path: Path) -> None:
    with pytest.raises(RunRegistryError):
        RunRegistry(tmp_path / "does-not-exist")


def test_file_as_root_raises_typed_error(tmp_path: Path) -> None:
    target = tmp_path / "a-file"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(RunRegistryError):
        RunRegistry(target)


def test_refresh_picks_up_new_and_removed_runs(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    registry = RunRegistry(root)
    assert registry.list() == []

    make_run(root, "fresh")
    summary = registry.refresh()
    assert summary.indexed == 1
    assert summary.excluded == 0
    assert registry.list() == ["fresh"]

    (root / "fresh" / "manifest.json").unlink()
    summary = registry.refresh()
    assert summary.indexed == 0
    assert summary.excluded == 1
    assert registry.list() == []


def test_checkpoint_paths_are_collected_and_sorted(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    make_run(
        root,
        "with-ckpts",
        checkpoints=(
            "checkpoints/step-1000/model.pt",
            "checkpoints/step-500/model.pt",
            "top-level.pt",
            "notes.txt",
        ),
    )
    registry = RunRegistry(root)
    entry = registry.get("with-ckpts")
    names = [Path(p).name for p in entry.checkpoint_paths]
    assert "notes.txt" not in names
    assert entry.checkpoint_paths == tuple(sorted(entry.checkpoint_paths))
    assert set(names) == {"model.pt", "model.pt", "top-level.pt"}


def test_refresh_returns_scan_counts(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    make_run(root, "good")
    make_run(root, "bad", write_manifest=False)
    (root / "stray-file.txt").write_text("not a run dir", encoding="utf-8")

    registry = RunRegistry(root)
    assert len(registry) == 1
    assert registry.excluded_count == 1


def test_exclusion_does_not_silently_list_anything(tmp_path: Path) -> None:
    """FR-OPS-02: excluded runs never appear in list()/get()."""
    root = tmp_path / "runs"
    for i in range(3):
        make_run(root, f"bad-{i}", config_hash=None)
    make_run(root, "good")

    registry = RunRegistry(root)

    assert registry.list() == ["good"]
    assert registry.excluded_count == 3
    for exclusion in registry.exclusions:
        assert exclusion.reason == "missing_config_hash"
    with pytest.raises(UnknownRunError):
        registry.get("bad-0")


def test_invalid_utf8_manifest_is_excluded_not_fatal(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    make_run(root, "good")
    bad = root / "bad-encoding"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_bytes(b'{"run_id": "\xff\xfe"}')

    registry = RunRegistry(root)

    assert registry.list() == ["good"]
    assert registry.excluded_count == 1
    assert registry.exclusions[0].run_dir == bad
    assert registry.exclusions[0].reason == "invalid_manifest_encoding"


def test_directory_run_id_mismatch_is_excluded(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    make_run(root, "claims-another-id", run_id="different-id")
    make_run(root, "good")

    registry = RunRegistry(root)

    assert registry.list() == ["good"]
    mismatch = [e for e in registry.exclusions if e.reason == "run_id_mismatch"]
    assert len(mismatch) == 1
    assert mismatch[0].run_dir == root / "claims-another-id"


@pytest.mark.parametrize(
    ("remove", "reason"),
    [
        ("git_sha", "missing_git_sha"),
        ("seed", "missing_seed"),
        ("device", "missing_device"),
        ("versions", "missing_versions"),
    ],
)
def test_missing_contract_fields_are_excluded(tmp_path: Path, remove, reason) -> None:
    import json as _json

    root = tmp_path / "runs"
    run_dir = root / "incomplete"
    make_run(root, "incomplete")
    path = run_dir / "manifest.json"
    payload = _json.loads(path.read_text(encoding="utf-8"))
    del payload[remove]
    path.write_text(_json.dumps(payload), encoding="utf-8")

    registry = RunRegistry(root)
    assert registry.list() == []
    assert registry.exclusions[0].reason == reason


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("config_hash", "short", "invalid_config_hash"),
        ("data_fingerprint", 12345, "invalid_data_fingerprint"),
        ("seed", True, "invalid_seed"),
        ("seed", -1, "invalid_seed"),
        ("device", 7, "invalid_device"),
        ("versions", {}, "invalid_versions"),
        ("run_id", "../escape", "invalid_run_id"),
    ],
)
def test_wrong_field_types_or_formats_are_excluded(tmp_path: Path, field, value, reason) -> None:
    import json as _json

    root = tmp_path / "runs"
    run_dir = root / "wrong-type"
    make_run(root, "wrong-type")
    path = run_dir / "manifest.json"
    payload = _json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(_json.dumps(payload), encoding="utf-8")

    registry = RunRegistry(root)
    assert registry.list() == []
    assert registry.exclusions[0].reason == reason


def test_unreadable_manifest_does_not_abort_scan(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runs"
    make_run(root, "good")
    make_run(root, "unreadable")

    original = Path.read_bytes

    def flaky_read(self: Path) -> bytes:
        if self.name == "manifest.json" and self.parent.name == "unreadable":
            raise OSError("permission denied")
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", flaky_read)
    registry = RunRegistry(root)
    assert registry.list() == ["good"]
    assert [e.reason for e in registry.exclusions] == ["unreadable_manifest"]


def test_entry_to_dict_is_json_serializable(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    make_run(root, "good", checkpoints=("checkpoints/m.pt",))
    entry = RunRegistry(root).get("good")
    payload = entry.to_dict()
    json.dumps(payload)
    assert payload["manifest_path"].endswith("manifest.json")
