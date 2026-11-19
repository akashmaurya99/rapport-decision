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
    ManifestWriteError,
    RunManifest,
    build_run_manifest,
    compute_config_hash,
    load_config,
    read_git_state,
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


def test_manifest_run_id_embeds_timestamp_and_hash_prefix(resolved):
    moment = datetime(2026, 11, 19, 20, 16, 5, tzinfo=timezone.utc)
    manifest = build_run_manifest(resolved, now=moment)
    assert re.fullmatch(r"20261119T201605Z-[0-9a-f]{8}", manifest.run_id)
    assert manifest.run_id.endswith(manifest.config_hash[:8])


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


def test_run_manifest_is_frozen_dataclass(resolved):
    manifest = build_run_manifest(resolved)
    assert isinstance(manifest, RunManifest)
    with pytest.raises(AttributeError):
        manifest.seed = 1  # type: ignore[misc]
