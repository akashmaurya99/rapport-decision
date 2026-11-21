"""Tests for the single validated experiment-config boundary.

``load_experiment_config`` is the API the e2e runner calls; these tests pin
its contract: base defaults < experiment YAML < RAPPORT_* env overrides,
unknown-key rejection, validation, and ``to_resolved()`` precedence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    SCHEMA,
    SHARED_FIELDS,
    ConfigEnvError,
    ConfigTypeMismatchError,
    ConfigUnknownKeyError,
    ConfigValueError,
    ExperimentConfig,
    load_config,
    load_experiment_config,
)

EXPERIMENTS = Path(__file__).resolve().parents[3] / "configs" / "experiments"
BASE_YAML = Path(__file__).resolve().parents[3] / "configs" / "base.yaml"


def _write(tmp_path, content: str) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_real_experiment_yaml_and_to_resolved() -> None:
    config = load_experiment_config(EXPERIMENTS / "starter.yaml", env={})
    assert isinstance(config, ExperimentConfig)
    assert config.value("arm") == "starter"
    assert config.value("mixture") == "starter"
    assert config.value("ood_source", "fallback") == "fallback"  # absent key

    resolved = config.to_resolved()
    assert resolved.learning_rate == pytest.approx(3.0e-4)  # lr alias, not stale 3e-5
    assert resolved.epochs == 8
    assert resolved.batch_size == 64
    assert resolved.max_length == 512
    assert resolved.seed == 13
    assert resolved.device == "cuda"  # from authoritative base defaults
    assert resolved.determinism == "strict"
    assert resolved.data_dir == "data"
    assert resolved.runs_dir == "runs"
    assert resolved.fp16 is True
    assert resolved.teacher == "Qwen/Qwen3-1.7B"
    assert resolved.data_fingerprint == ""


def test_all_real_experiment_configs_load() -> None:
    names = sorted(path.stem for path in EXPERIMENTS.glob("*.yaml"))
    assert names  # the config directory is present in the source checkout
    for path in sorted(EXPERIMENTS.glob("*.yaml")):
        config = load_experiment_config(path, env={})
        resolved = config.to_resolved()
        assert resolved.learning_rate == pytest.approx(3.0e-4)
        assert config.value("arm")


def test_unknown_key_is_rejected_by_name(tmp_path) -> None:
    path = _write(tmp_path, "arm: test\nlerning_rate: 1e-3\n")
    with pytest.raises(ConfigUnknownKeyError) as excinfo:
        load_experiment_config(path, env={})
    assert excinfo.value.key == "lerning_rate"
    assert excinfo.value.code == "CONFIG_UNKNOWN_KEY"
    assert "lerning_rate" in str(excinfo.value)


def test_wrong_type_is_rejected(tmp_path) -> None:
    path = _write(tmp_path, "arm: test\nepochs: eight\n")
    with pytest.raises(ConfigTypeMismatchError) as excinfo:
        load_experiment_config(path, env={})
    assert excinfo.value.key == "epochs"


def test_out_of_domain_values_are_rejected(tmp_path) -> None:
    for content, key in (
        ("arm: test\nepochs: 0\n", "epochs"),
        ("arm: test\nlr: -1\n", "learning_rate"),  # canonicalised alias
        ("arm: test\nlayers: 0\n", "layers"),
        ("arm: test\ndevice: tpu\n", "device"),
        ("arm: ../escape\n", "arm"),
        ("arm: test\nseeds: [13, -1]\n", "seeds"),
    ):
        with pytest.raises(ConfigValueError) as excinfo:
            load_experiment_config(_write(tmp_path, content), env={})
        assert excinfo.value.key == key
    with pytest.raises(ConfigUnknownKeyError):
        load_experiment_config(
            _write(tmp_path, "arm: test\nvariants: [{wrong: x}]\n"), env={}
        )


def test_alias_and_canonical_learning_rate_cannot_both_be_declared(tmp_path) -> None:
    path = _write(tmp_path, "arm: test\nlr: 1e-3\nlearning_rate: 2e-3\n")
    with pytest.raises(ConfigValueError):
        load_experiment_config(path, env={})


def test_nested_variant_overrides_are_validated(tmp_path) -> None:
    path = _write(tmp_path, "arm: test\nvariants: [{name: bad, lr: .inf}]\n")
    with pytest.raises(ConfigValueError):
        load_experiment_config(path, env={})
    path = _write(tmp_path, "arm: test\nvariants: [{name: bad, learning_rate: -1}]\n")
    with pytest.raises(ConfigValueError):
        load_experiment_config(path, env={})


def test_all_twelve_env_overrides_apply_and_win(tmp_path) -> None:
    path = _write(tmp_path, "arm: test\nfp16: true\nepochs: 8\n")
    env = {
        "RAPPORT_SEED": "77",
        "RAPPORT_DEVICE": "cpu",
        "RAPPORT_DATA_DIR": "custom-data",
        "RAPPORT_RUNS_DIR": "custom-runs",
        "RAPPORT_FP16": "false",
        "RAPPORT_DETERMINISM": "relaxed",
        "RAPPORT_EPOCHS": "3",
        "RAPPORT_BATCH_SIZE": "7",
        "RAPPORT_LEARNING_RATE": "2.5e-4",
        "RAPPORT_MAX_LENGTH": "256",
        "RAPPORT_TEACHER": "teacher-x",
        "RAPPORT_DATA_FINGERPRINT": "abc123",
    }
    config = load_experiment_config(path, env=env)
    assert len(config.env_overrides) == 12
    resolved = config.to_resolved()
    assert resolved.to_dict() == {
        "seed": 77,
        "device": "cpu",
        "data_dir": "custom-data",
        "runs_dir": "custom-runs",
        "fp16": False,
        "determinism": "relaxed",
        "epochs": 3,
        "batch_size": 7,
        "learning_rate": pytest.approx(2.5e-4),
        "max_length": 256,
        "teacher": "teacher-x",
        "data_fingerprint": "abc123",
    }
    # The alias is updated too, for callers using .value("lr").
    assert config.value("lr") == pytest.approx(2.5e-4)


@pytest.mark.parametrize(
    ("var", "raw"),
    [
        ("RAPPORT_SEED", "thirteen"),
        ("RAPPORT_DEVICE", "tpu"),
        ("RAPPORT_FP16", "maybe"),
        ("RAPPORT_EPOCHS", "0"),
        ("RAPPORT_BATCH_SIZE", "-2"),
        ("RAPPORT_LEARNING_RATE", "not-a-float"),
        ("RAPPORT_MAX_LENGTH", "99999"),
    ],
)
def test_invalid_env_values_raise_typed_config_env_error(tmp_path, var, raw) -> None:
    path = _write(tmp_path, "arm: test\n")
    with pytest.raises(ConfigEnvError) as excinfo:
        load_experiment_config(path, env={var: raw})
    assert excinfo.value.code == "CONFIG_BAD_ENV_VALUE"
    assert var in str(excinfo.value)


def test_non_string_env_value_raises_config_env_error(tmp_path) -> None:
    with pytest.raises(ConfigEnvError):
        load_experiment_config(_write(tmp_path, "arm: test\n"), env={"RAPPORT_SEED": 13})


def test_explicit_overrides_win_over_env_and_yaml(tmp_path) -> None:
    path = _write(tmp_path, "arm: test\nseed: 3\n")
    env = {"RAPPORT_SEED": "5", "RAPPORT_DEVICE": "cuda"}
    config = load_experiment_config(path, env=env)
    resolved = config.to_resolved({"seed": 9, "device": "cpu", "data_fingerprint": "live"})
    assert resolved.seed == 9
    assert resolved.device == "cpu"
    assert resolved.data_fingerprint == "live"
    # to_resolved is pure: the document itself is unchanged.
    assert config.value("seed") == 5


def test_unknown_explicit_override_is_rejected(tmp_path) -> None:
    config = load_experiment_config(_write(tmp_path, "arm: test\n"), env={})
    with pytest.raises(ConfigUnknownKeyError) as excinfo:
        config.to_resolved({"lerning_rate": 1e-3})
    assert excinfo.value.key == "lerning_rate"


def test_invalid_explicit_override_is_rejected(tmp_path) -> None:
    config = load_experiment_config(_write(tmp_path, "arm: test\n"), env={})
    with pytest.raises(ConfigValueError):
        config.to_resolved({"epochs": 0})


def test_document_is_read_only_and_shared_fields_are_the_twelve(tmp_path) -> None:
    from types import MappingProxyType

    config = load_experiment_config(_write(tmp_path, "arm: test\n"), env={})
    assert isinstance(config.document, MappingProxyType)
    with pytest.raises(TypeError):
        config.document["seed"] = 1  # type: ignore[index]
    assert len(SHARED_FIELDS) == 12
    assert set(SHARED_FIELDS) == set(SCHEMA)


def test_base_yaml_lr_drift_is_resolved() -> None:
    base = load_config(BASE_YAML, env={})
    assert base.learning_rate == pytest.approx(3.0e-4)
    assert SCHEMA["learning_rate"].default == pytest.approx(3.0e-4)
    resolved = load_experiment_config(EXPERIMENTS / "starter.yaml", env={}).to_resolved()
    assert resolved.learning_rate == base.learning_rate
