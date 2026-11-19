"""Configuration for the FP16 trainer core (TASK-12).

A small, flat, frozen dataclass. All keys carry their design defaults from
``.archgen/rapport/docs/training-strategy.md`` §3.1 and §8 and
``.archgen/rapport/docs/t4-optimization.md`` §2:

* AdamW, betas (0.9, 0.98), eps 1e-8, weight decay 0.01;
* learning rate 3e-4 (encoder-LR arm) with a cosine decay to
  ``min_learning_rate`` (1e-5) after a 5% linear warmup;
* gradient clipping by global norm 1.0;
* ``GradScaler`` with init scale 2**16, backoff 0.5, growth interval 2000;
* checkpoints every ``checkpoint_every`` steps (default **500**) — a config
  key, not a hardcoded constant — and at every epoch end;
* ``steps`` is the planned total optimizer-step budget (§8: "The step
  budget is a config value").

``from_mapping`` builds a config from a plain mapping (e.g. a YAML document)
and refuses unknown keys; types are checked strictly (a bool is never an
int) and every value is range-checked.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import TrainerConfigError

_DEVICE_DOMAIN = ("auto", "cpu", "cuda", "mps")

# Fields whose values must be JSON/YAML-friendly scalars; used by to_dict().
@dataclass(frozen=True)
class TrainingConfig:
    """Resolved configuration for :class:`src.training.core.Trainer`."""

    # Horizon (training-strategy.md §8). 115.6k examples at batch 64 is
    # 1,806 steps/epoch; 8 epochs ≈ 14,450 steps.
    steps: int = 14_450
    epochs: int = 8
    batch_size: int = 64
    seed: int = 13
    device: str = "auto"  # "auto" resolves to cuda when available, else cpu

    # Optimizer (training-strategy.md §3.1).
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 1.0e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1.0e-8
    warmup_ratio: float = 0.05  # 5% linear warmup, then cosine to min LR
    grad_clip_norm: float = 1.0

    # Precision (t4-optimization.md §2). FP16 autocast + GradScaler.
    fp16: bool = True
    scaler_init_scale: float = float(2**16)
    scaler_backoff_factor: float = 0.5
    scaler_growth_factor: float = 2.0
    scaler_growth_interval: int = 2000

    # Checkpointing (training-strategy.md §8).
    checkpoint_every: int = 500
    checkpoint_dir: str = "checkpoints"

    # E5/FR-TRAIN-04: more than this many non-finite steps in one epoch is
    # divergence, not bad luck.
    nonfinite_per_epoch: int = 10

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrainingConfig":
        """Build a config from a plain mapping, rejecting unknown keys."""
        if not isinstance(mapping, Mapping):
            raise TrainerConfigError(
                f"config must be a mapping, got {type(mapping).__name__}", key="config"
            )
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise TrainerConfigError(
                f"unknown config key(s) {unknown}; declared keys are {sorted(known)}",
                key=unknown[0],
            )
        values: dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            if field.name in mapping:
                values[field.name] = mapping[field.name]
        config = cls(**values)
        _validate(config)
        return config

    def resolved_device(self) -> str:
        """Resolve ``auto`` to a concrete device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - torch is a hard dependency
            return "cpu"


def _validate(config: TrainingConfig) -> None:
    def fail(key: str, problem: str) -> None:
        raise TrainerConfigError(f"config key {key!r}: {problem}", key=key)

    def is_int(value: Any) -> bool:
        return type(value) is int

    def is_float(value: Any) -> bool:
        return type(value) in (int, float) and type(value) is not bool

    def is_bool(value: Any) -> bool:
        return type(value) is bool

    if not is_int(config.steps) or config.steps < 1:
        fail("steps", f"expected an integer >= 1, got {config.steps!r}")
    if not is_int(config.epochs) or config.epochs < 1:
        fail("epochs", f"expected an integer >= 1, got {config.epochs!r}")
    if not is_int(config.batch_size) or config.batch_size < 1:
        fail("batch_size", f"expected an integer >= 1, got {config.batch_size!r}")
    if not is_int(config.seed) or not (0 <= config.seed <= 2**32 - 1):
        fail("seed", f"expected an integer in [0, 2**32 - 1], got {config.seed!r}")
    if not isinstance(config.device, str) or config.device not in _DEVICE_DOMAIN:
        fail("device", f"expected one of {list(_DEVICE_DOMAIN)}, got {config.device!r}")
    if not is_float(config.learning_rate) or not math.isfinite(config.learning_rate):
        fail("learning_rate", f"expected a finite float, got {config.learning_rate!r}")
    if config.learning_rate <= 0:
        fail("learning_rate", f"expected > 0, got {config.learning_rate!r}")
    if not is_float(config.min_learning_rate) or not math.isfinite(config.min_learning_rate):
        fail("min_learning_rate", f"expected a finite float, got {config.min_learning_rate!r}")
    if config.min_learning_rate <= 0:
        fail("min_learning_rate", f"expected > 0, got {config.min_learning_rate!r}")
    if config.min_learning_rate > config.learning_rate:
        fail("min_learning_rate", "must not exceed learning_rate")
    if not is_float(config.weight_decay) or not math.isfinite(config.weight_decay):
        fail("weight_decay", f"expected a finite float, got {config.weight_decay!r}")
    if config.weight_decay < 0:
        fail("weight_decay", f"expected >= 0, got {config.weight_decay!r}")
    for name in ("beta1", "beta2"):
        value = getattr(config, name)
        if not is_float(value) or not math.isfinite(value) or not (0.0 <= value < 1.0):
            fail(name, f"expected a finite float in [0, 1), got {value!r}")
    if not is_float(config.eps) or not math.isfinite(config.eps) or config.eps <= 0:
        fail("eps", f"expected a finite float > 0, got {config.eps!r}")
    if not is_float(config.warmup_ratio) or not math.isfinite(config.warmup_ratio):
        fail("warmup_ratio", f"expected a finite float, got {config.warmup_ratio!r}")
    if not (0.0 <= config.warmup_ratio < 1.0):
        fail("warmup_ratio", f"expected in [0, 1), got {config.warmup_ratio!r}")
    if not is_float(config.grad_clip_norm) or not math.isfinite(config.grad_clip_norm):
        fail("grad_clip_norm", f"expected a finite float, got {config.grad_clip_norm!r}")
    if config.grad_clip_norm <= 0:
        fail("grad_clip_norm", f"expected > 0, got {config.grad_clip_norm!r}")
    if not is_bool(config.fp16):
        fail("fp16", f"expected a bool, got {config.fp16!r}")
    if not is_float(config.scaler_init_scale) or config.scaler_init_scale <= 0:
        fail("scaler_init_scale", f"expected a float > 0, got {config.scaler_init_scale!r}")
    if not is_float(config.scaler_backoff_factor):
        fail("scaler_backoff_factor", f"expected a float, got {config.scaler_backoff_factor!r}")
    if not (0.0 < config.scaler_backoff_factor < 1.0):
        fail("scaler_backoff_factor", f"expected in (0, 1), got {config.scaler_backoff_factor!r}")
    if not is_float(config.scaler_growth_factor) or config.scaler_growth_factor <= 1.0:
        fail("scaler_growth_factor", f"expected a float > 1.0, got {config.scaler_growth_factor!r}")
    if not is_int(config.scaler_growth_interval) or config.scaler_growth_interval < 1:
        fail(
            "scaler_growth_interval",
            f"expected an integer >= 1, got {config.scaler_growth_interval!r}",
        )
    if not is_int(config.checkpoint_every) or config.checkpoint_every < 1:
        fail("checkpoint_every", f"expected an integer >= 1, got {config.checkpoint_every!r}")
    if not isinstance(config.checkpoint_dir, str) or not config.checkpoint_dir.strip():
        fail("checkpoint_dir", f"expected a non-empty string, got {config.checkpoint_dir!r}")
    if not is_int(config.nonfinite_per_epoch) or config.nonfinite_per_epoch < 1:
        fail("nonfinite_per_epoch", f"expected an integer >= 1, got {config.nonfinite_per_epoch!r}")
