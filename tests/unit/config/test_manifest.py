"""Tests for src/config/manifest.py — run manifest shape and writing."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from src.config import (
    ManifestRunIdError,
    ManifestWriteError,
    RunManifest,
    build_run_manifest,
    compute_config_hash,
    load_config,
    read_dependency_lock_digest,
    read_git_state,
    validate_run_id,
    write_run_manifest,
)

BASE_YAML = Path(__file__).resolve().parents[3] / "configs" / "base.yaml"


@pytest.fixture()
def resolved(tmp_path):
    return load_config(BASE_YAML, env={})


def test_config_hash_is_sha256_of_resolved_yaml(resolved):
    expected = hashlib.sha256(
        yaml.safe_dump(resolved.to_dict(), sort_keys=True, default_flow_style=False).encode("utf-8")
    ).hexdigest()
    assert compute_config_hash(resolved) == expected
    assert re.fullmatch(r"[0-9a-f]{64}", compute_config_hash(resolved))


def test_manifest_records_required_fields(resolved):
    manifest = build_run_manifest(resolved, now=datetime(2026, 11, 19, 20, 16, tzinfo=timezone.utc))
    payload = manifest.to_dict()
    for field in (
        "run_id",
        "git_sha",
        "git_dirty",
        "config_hash",
        "data_fingerprint",
        "seed",
        "device",
        "compute_capability",
        "determinism",
        "versions",
    ):
        assert field in payload
    assert payload["seed"] == 13
    assert payload["determinism"] == "strict"
    assert payload["config_hash"] == compute_config_hash(resolved)
    assert payload["data_fingerprint"] == ""  # placeholder default, recorded as such
    assert payload["git_sha"]  # non-empty string, "unknown" outside a git repo
    assert payload["git_dirty"] is None or isinstance(payload["git_dirty"], bool)


def test_manifest_run_id_embeds_timestamp_hash_pid_and_nonce(resolved):
    moment = datetime(2026, 11, 19, 20, 16, 5, tzinfo=timezone.utc)
    manifest = build_run_manifest(resolved, now=moment)
    assert re.fullmatch(r"20261119T201605Z-[0-9a-f]{8}-[0-9a-f]+-[0-9a-f]{12}", manifest.run_id)
    assert manifest.run_id.startswith(f"20261119T201605Z-{manifest.config_hash[:8]}-")


def test_same_timestamp_same_config_yields_distinct_run_ids(resolved):
    moment = datetime(2026, 11, 19, 20, 16, 5, tzinfo=timezone.utc)
    ids = {build_run_manifest(resolved, now=moment).run_id for _ in range(32)}
    assert len(ids) == 32  # pid + random nonce defeat second-level collisions


def test_manifest_versions_cover_python_torch_transformers_datasets(resolved):
    manifest = build_run_manifest(resolved)
    versions = manifest.versions
    assert versions["python"] == platform.python_version()
    for name in ("torch", "transformers", "datasets", "onnxruntime"):
        assert name in versions  # null when the package is absent, never invented


def test_data_fingerprint_comes_from_env(tmp_path):
    resolved = load_config(BASE_YAML, env={"RAPPORT_DATA_FINGERPRINT": "cafe1234"})
    manifest = build_run_manifest(resolved)
    assert manifest.data_fingerprint == "cafe1234"


def test_write_run_manifest_writes_json_at_runs_dir(resolved, tmp_path):
    manifest = build_run_manifest(resolved, now=datetime(2026, 11, 19, 20, 16, tzinfo=timezone.utc))
    runs_dir = tmp_path / "runs"
    target = write_run_manifest(manifest, runs_dir)
    assert target == runs_dir / manifest.run_id / "manifest.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == manifest.to_dict()
    assert payload["versions"]["python"] == platform.python_version()


def test_write_run_manifest_fails_typed_on_unwritable_dir(resolved, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    manifest = build_run_manifest(resolved)
    with pytest.raises(ManifestWriteError) as excinfo:
        write_run_manifest(manifest, blocker / "runs")
    assert excinfo.value.code == "MANIFEST_WRITE_FAILED"


def test_git_state_is_reported_not_refused():
    sha, dirty = read_git_state()
    assert (isinstance(sha, str) and sha) or sha == "unknown"
    assert dirty is None or isinstance(dirty, bool)


@pytest.mark.parametrize(
    "run_id",
    ["", "   ", ".", "..", "../escape", "a/b", "a\\b", "/tmp/absolute", "run id", "..\\evil"],
)
def test_traversing_or_non_portable_run_ids_are_rejected(run_id):
    with pytest.raises(ManifestRunIdError) as excinfo:
        validate_run_id(run_id)
    assert excinfo.value.code == "MANIFEST_INVALID_RUN_ID"


@pytest.mark.parametrize("run_id", ["../escape", "/tmp/absolute", "nested/run"])
def test_write_rejects_traversal_before_creating_anything(resolved, tmp_path, run_id):
    manifest = build_run_manifest(resolved)
    object.__setattr__(manifest, "run_id", run_id)  # frozen dataclass: simulate a forged ID
    runs = tmp_path / "runs"
    with pytest.raises(ManifestRunIdError):
        write_run_manifest(manifest, runs)
    assert not runs.exists()


def test_write_is_atomic_without_leftover_tmp_files(resolved, tmp_path):
    manifest = build_run_manifest(resolved, now=datetime(2026, 11, 19, 20, 16, tzinfo=timezone.utc))
    runs = tmp_path / "runs"
    target = write_run_manifest(manifest, runs)
    assert [p.name for p in target.parent.iterdir()] == ["manifest.json"]
    assert json.loads(target.read_text(encoding="utf-8")) == manifest.to_dict()


def test_git_state_is_read_relative_to_repo_root_from_any_cwd(resolved, tmp_path, monkeypatch):
    expected = read_git_state()
    monkeypatch.chdir(tmp_path)
    assert read_git_state() == expected
    manifest = build_run_manifest(resolved)
    assert manifest.git_sha == expected[0]


def test_lock_digest_is_recorded_when_lock_exists(resolved, tmp_path, monkeypatch):
    import src.config.manifest as manifest_module

    lock = tmp_path / "requirements.lock"
    lock.write_text("torch==2.15.0\n", encoding="utf-8")
    monkeypatch.setattr(manifest_module, "_LOCK_FILE", lock)
    digest = read_dependency_lock_digest()
    assert digest is not None and re.fullmatch(r"[0-9a-f]{64}", digest)
    assert build_run_manifest(resolved).dependency_lock_sha256 == digest


def test_run_manifest_is_frozen_dataclass(resolved):
    manifest = build_run_manifest(resolved)
    assert isinstance(manifest, RunManifest)
    with pytest.raises(AttributeError):
        manifest.seed = 1  # type: ignore[misc]
