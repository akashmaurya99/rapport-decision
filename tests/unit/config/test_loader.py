"""Tests for src/config/loader.py — resolution, overrides and typed errors."""

from __future__ import annotations

import pytest
import yaml

from src.config import (
    SCHEMA,
    ConfigEnvError,
    ConfigFileUnreadableError,
    ConfigTypeMismatchError,
    ConfigUnknownKeyError,
    ConfigValueError,
    ConfigYamlError,
    load_config,
)


def _write(tmp_path, content) -> object:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_valid_config_loads_with_yaml_values(tmp_path):
    path = _write(
        tmp_path,
        "seed: 99\ndevice: cpu\nfp16: false\nepochs: 4\nlearning_rate: 1.0e-4\n",
    )
    resolved = load_config(path, env={})
    assert resolved.seed == 99
    assert resolved.device == "cpu"
    assert resolved.fp16 is False
    assert resolved.epochs == 4
    assert resolved.learning_rate == pytest.approx(1.0e-4)


def test_defaults_apply_for_missing_keys(tmp_path):
    path = _write(tmp_path, "seed: 7\n")
    resolved = load_config(path, env={})
    assert resolved.seed == 7
    for key, spec in SCHEMA.items():
        if key != "seed":
            assert getattr(resolved, key) == spec.default


def test_unknown_key_fails_with_typed_error_naming_key(tmp_path):
    path = _write(tmp_path, "seed: 13\nbogus_key: 1\n")
    with pytest.raises(ConfigUnknownKeyError) as excinfo:
        load_config(path, env={})
    err = excinfo.value
    assert err.code == "CONFIG_UNKNOWN_KEY"
    assert "bogus_key" in str(err)
    assert err.key == "bogus_key"


def test_env_overrides_yaml(tmp_path):
    path = _write(tmp_path, "seed: 5\ndevice: cuda\ndata_dir: data\n")
    resolved = load_config(path, env={"RAPPORT_SEED": "99", "RAPPORT_DEVICE": "cpu"})
    assert resolved.seed == 99
    assert resolved.device == "cpu"
    assert resolved.data_dir == "data"  # untouched key keeps the YAML value


def test_env_override_beats_default_and_parses_types(tmp_path):
    path = _write(tmp_path, "")
    resolved = load_config(
        path,
        env={
            "RAPPORT_SEED": "123",
            "RAPPORT_FP16": "false",
            "RAPPORT_EPOCHS": "2",
            "RAPPORT_LEARNING_RATE": "1e-3",
            "RAPPORT_DATA_FINGERPRINT": "deadbeef",
        },
    )
    assert resolved.seed == 123
    assert resolved.fp16 is False
    assert resolved.epochs == 2
    assert resolved.learning_rate == pytest.approx(1e-3)
    assert resolved.data_fingerprint == "deadbeef"


def test_env_value_that_cannot_be_coerced_fails_naming_var(tmp_path):
    path = _write(tmp_path, "")
    with pytest.raises(ConfigEnvError) as excinfo:
        load_config(path, env={"RAPPORT_SEED": "thirteen"})
    assert excinfo.value.code == "CONFIG_BAD_ENV_VALUE"
    assert "RAPPORT_SEED" in str(excinfo.value)


def test_unreadable_yaml_file_fails_typed(tmp_path):
    with pytest.raises(ConfigFileUnreadableError) as excinfo:
        load_config(tmp_path / "does-not-exist.yaml", env={})
    assert excinfo.value.code == "CONFIG_FILE_UNREADABLE"


def test_unparsable_yaml_fails_typed(tmp_path):
    path = _write(tmp_path, "seed: [1, 2\n")
    with pytest.raises(ConfigYamlError) as excinfo:
        load_config(path, env={})
    assert excinfo.value.code == "CONFIG_YAML_INVALID"


def test_non_mapping_yaml_fails_typed(tmp_path):
    path = _write(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ConfigYamlError) as excinfo:
        load_config(path, env={})
    assert excinfo.value.code == "CONFIG_YAML_INVALID"


def test_wrong_type_fails_naming_key(tmp_path):
    path = _write(tmp_path, "seed: thirteen\n")
    with pytest.raises(ConfigTypeMismatchError) as excinfo:
        load_config(path, env={})
    assert excinfo.value.code == "CONFIG_TYPE_MISMATCH"
    assert "seed" in str(excinfo.value)


def test_bool_is_not_accepted_as_int(tmp_path):
    path = _write(tmp_path, "seed: true\n")
    with pytest.raises(ConfigTypeMismatchError):
        load_config(path, env={})


def test_out_of_domain_value_fails_naming_key(tmp_path):
    path = _write(tmp_path, "epochs: 0\n")
    with pytest.raises(ConfigValueError) as excinfo:
        load_config(path, env={})
    assert excinfo.value.code == "CONFIG_BAD_VALUE"
    assert "epochs" in str(excinfo.value)


def test_unknown_device_fails_naming_key(tmp_path):
    path = _write(tmp_path, "device: tpu\n")
    with pytest.raises(ConfigValueError) as excinfo:
        load_config(path, env={})
    assert "device" in str(excinfo.value)


def test_unknown_determinism_fails_naming_key(tmp_path):
    path = _write(tmp_path, "determinism: paranoid\n")
    with pytest.raises(ConfigValueError) as excinfo:
        load_config(path, env={})
    assert "determinism" in str(excinfo.value)


def test_resolved_config_round_trips_through_dict(tmp_path):
    path = _write(tmp_path, "seed: 21\n")
    resolved = load_config(path, env={})
    payload = resolved.to_dict()
    assert payload["seed"] == 21
    assert set(payload) == set(SCHEMA)


def test_base_yaml_is_valid_yaml_and_passes_the_schema():
    from pathlib import Path

    base = Path(__file__).resolve().parents[3] / "configs" / "base.yaml"
    with base.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)  # YAML-validity check
    assert isinstance(document, dict)
    resolved = load_config(base, env={})  # schema-validity check
    assert resolved.seed == 13
    assert resolved.device == "cuda"
    assert resolved.fp16 is True
    assert resolved.determinism == "strict"
    assert resolved.teacher == "Qwen/Qwen3-1.7B"
    assert resolved.data_fingerprint == ""
    assert set(document) <= set(SCHEMA)  # no unknown keys
