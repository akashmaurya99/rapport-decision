"""Unit tests for TASK-14 — the fixed-weight arm registered for ablation A4."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.training.losses import DEFAULT_WEIGHTS
from src.training.multitask import DEFAULT_TASK_NAMES, build_fixed_weight_config
from src.training.multitask import arms as arms_module

REPO_ROOT = Path(__file__).resolve().parents[4]
A4_YAML = REPO_ROOT / "configs" / "experiments" / "a4-weights.yaml"


def test_a4_yaml_exists_where_expected():
    # The task brief referred to configs/experiments/a4-weighting.yaml; the
    # actual file shipped by TASK-13 is a4-weights.yaml.
    assert A4_YAML.is_file()


def test_fixed_arm_consistent_with_a4_yaml():
    template = yaml.safe_load(A4_YAML.read_text(encoding="utf-8"))
    config = build_fixed_weight_config("A4")

    assert config["arm"] == "A4"
    assert config["title"] == template["title"] == "weights"
    assert config["question"] == template["question"]
    assert config["weight_mode"] == "fixed"
    assert template["weight_mode"] == "learned"  # the template stays the learned arm
    assert config["weight_floor"] == pytest.approx(template["weight_floor"]) == pytest.approx(0.05)
    assert config["weight_cap"] == pytest.approx(template["weight_cap"]) == pytest.approx(4.0)

    # Everything unrelated to the weighting scheme is inherited verbatim.
    for key, value in template.items():
        if key in ("weight_mode",):
            continue
        assert config[key] == value, f"key {key!r} drifted from the A4 template"

    # The fixed arm carries the §3.2 default weights the Trainer resolves.
    assert config["weights"] == {
        task: float(DEFAULT_WEIGHTS[task]) for task in DEFAULT_TASK_NAMES
    }
    assert set(config["weights"]) == set(DEFAULT_TASK_NAMES)

    # The ablation's identifying fields survive.
    assert config["seeds"] == [13, 29, 47]
    assert config["mixture"] == "starter"
    assert config["mixture_rows"] == 55000
    assert config["epochs"] == 8
    assert config["optimizer"] == "adamw"


def test_fixed_arm_is_yaml_serializable():
    config = build_fixed_weight_config("A4")
    dumped = yaml.safe_dump(config)
    assert yaml.safe_load(dumped)["weights"] == {
        task: float(DEFAULT_WEIGHTS[task]) for task in DEFAULT_TASK_NAMES
    }


def test_fixed_arm_does_not_mutate_template_file():
    before = A4_YAML.read_bytes()
    build_fixed_weight_config("A4")
    assert A4_YAML.read_bytes() == before


def test_unknown_arm_rejected():
    with pytest.raises(ValueError, match="A4"):
        build_fixed_weight_config("A5")
    with pytest.raises(ValueError, match="A4"):
        build_fixed_weight_config("")
    with pytest.raises(ValueError, match="A4"):
        build_fixed_weight_config(None)  # type: ignore[arg-type]


def test_explicit_missing_config_path_raises(tmp_path):
    with pytest.raises((OSError, ValueError)):
        build_fixed_weight_config("A4", config_path=tmp_path / "does-not-exist.yaml")


def test_fallback_template_used_when_default_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(arms_module, "_DEFAULT_CONFIG_PATH", tmp_path / "missing.yaml")
    config = build_fixed_weight_config("A4")
    on_disk = yaml.safe_load(A4_YAML.read_text(encoding="utf-8"))
    for key, value in on_disk.items():
        if key == "weight_mode":
            continue
        assert config[key] == value
    assert config["weight_mode"] == "fixed"
    assert config["weights"] == {
        task: float(DEFAULT_WEIGHTS[task]) for task in DEFAULT_TASK_NAMES
    }


def test_fallback_matches_on_disk_template():
    on_disk = yaml.safe_load(A4_YAML.read_text(encoding="utf-8"))
    assert arms_module._FALLBACK_TEMPLATE == on_disk
