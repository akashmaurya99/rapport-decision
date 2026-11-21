"""Unit tests for src.training.core.config (TrainingConfig)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.core import TrainerConfigError, TrainingConfig  # noqa: E402


def test_defaults_match_design() -> None:
    config = TrainingConfig()
    # training-strategy.md §3.1 / §8 and t4-optimization.md §2.
    assert config.checkpoint_every == 500
    assert config.warmup_ratio == 0.05
    assert config.scaler_init_scale == float(2**16)
    assert config.scaler_backoff_factor == 0.5
    assert config.scaler_growth_interval == 2000
    assert config.grad_clip_norm == 1.0
    assert config.learning_rate == pytest.approx(3.0e-4)
    assert config.min_learning_rate == pytest.approx(1.0e-5)
    assert config.weight_decay == pytest.approx(0.01)
    assert (config.beta1, config.beta2) == (0.9, 0.98)
    assert config.eps == pytest.approx(1.0e-8)
    assert config.fp16 is True
    assert config.epochs == 8
    assert config.batch_size == 64
    assert config.steps == 14_450
    assert config.nonfinite_per_epoch == 10


def test_new_fields_default_to_backward_compatible_values() -> None:
    config = TrainingConfig()
    assert config.head_lr is None
    assert config.determinism == "strict"
    assert config.weight_mode == "fixed"
    assert config.keep_last_n is None
    # They round-trip through to_dict()/from_mapping() like every other key.
    assert TrainingConfig.from_mapping(config.to_dict()) == config


def test_new_fields_validate() -> None:
    from src.training.core import TrainerConfigError

    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"head_lr": 0.0})
    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"head_lr": float("nan")})
    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"head_lr": "1e-3"})
    assert TrainingConfig.from_mapping({"head_lr": 1.0e-3}).head_lr == pytest.approx(1.0e-3)

    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"determinism": "sloppy"})
    assert TrainingConfig.from_mapping({"determinism": "relaxed"}).determinism == "relaxed"

    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"weight_mode": "auto"})
    for mode in ("learned", "fixed", "uniform"):
        assert TrainingConfig.from_mapping({"weight_mode": mode}).weight_mode == mode

    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"keep_last_n": 0})
    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"keep_last_n": -1})
    assert TrainingConfig.from_mapping({"keep_last_n": 3}).keep_last_n == 3


def test_to_dict_round_trip() -> None:
    config = TrainingConfig(steps=100, checkpoint_every=7)
    rebuilt = TrainingConfig.from_mapping(config.to_dict())
    assert rebuilt == config


def test_from_mapping_accepts_partial() -> None:
    config = TrainingConfig.from_mapping({"steps": 50, "checkpoint_every": 10})
    assert config.steps == 50
    assert config.checkpoint_every == 10
    assert config.epochs == 8  # untouched default


def test_from_mapping_rejects_unknown_key() -> None:
    with pytest.raises(TrainerConfigError) as excinfo:
        TrainingConfig.from_mapping({"checkpoint_everty": 10})
    assert "checkpoint_everty" in str(excinfo.value)


def test_from_mapping_rejects_bad_types() -> None:
    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"steps": "100"})
    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"fp16": 1})  # bool slot, not int
    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"checkpoint_every": True})  # bool is not int


def test_from_mapping_rejects_out_of_range() -> None:
    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"steps": 0})
    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"checkpoint_every": 0})
    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"warmup_ratio": 1.0})
    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"learning_rate": 0.0})
    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"min_learning_rate": 1.0, "learning_rate": 0.001})
    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"device": "tpu"})
    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping({"nonfinite_per_epoch": 0})


def test_from_mapping_rejects_non_mapping() -> None:
    with pytest.raises(TrainerConfigError):
        TrainingConfig.from_mapping([("steps", 100)])  # type: ignore[arg-type]


def test_resolved_device_cpu_here() -> None:
    assert TrainingConfig(device="cpu").resolved_device() == "cpu"
    assert TrainingConfig(device="cuda").resolved_device() == "cuda"
    # "auto" resolves to a concrete, valid device.
    assert TrainingConfig(device="auto").resolved_device() in {"cpu", "cuda"}
