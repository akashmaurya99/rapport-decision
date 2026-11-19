"""Fixed-weight arm registered for ablation A4.

A4 (``configs/experiments/a4-weights.yaml``, docs/evaluation.md §5) asks
"does uncertainty weighting collapse a task?" and runs three arms: ``learned``
(this package, floor 0.05 / cap 4.0), ``fixed`` (fixed weights) and
``uniform`` (equal weight, loss-normalised). This module provides the
``fixed`` arm definition in code so it can be registered / asserted without
editing anything under ``configs/``.

``build_fixed_weight_config`` loads the on-disk A4 experiment config as the
template (falling back to an embedded copy of it when the file is not
reachable) and returns a complete, YAML-serializable config dict with
``weight_mode: fixed`` and an explicit uniform weight vector
(``1.0`` for every task). Everything not specific to the weighting scheme —
seeds [13, 29, 47], mixture ``starter`` (55k rows), 8 epochs, AdamW, lr
3.0e-4, fp16 — is inherited from the template unchanged, so the fixed arm is
exactly the learned arm with the weighting replaced.

Note on the uniform vector: docs/training-strategy.md §3.2 lists per-term
*default* weights (choice 1.0, pref 1.0, score 0.5, prob 0.5, conf 0.5,
abstain 1.0, risk 0.5) and the A4 YAML comments describe the fixed arm as
"fixed weights as per docs/training-strategy.md §3.2 defaults". The contract
for this task fixes the arm at *uniform 1.0 across tasks* instead; that is
what this function emits, and the deviation is deliberate — uniform 1.0 is
the control that isolates the effect of learning the weights at all.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .weighting import DEFAULT_TASK_NAMES

__all__ = ["FIXED_WEIGHT_ARM", "build_fixed_weight_config"]

#: The only arm registered here.
FIXED_WEIGHT_ARM = "A4"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "experiments" / "a4-weights.yaml"

# Embedded copy of configs/experiments/a4-weights.yaml (TASK-13), used only
# when the on-disk file is unreachable. Kept key-for-key identical; the tests
# assert consistency with the real file so drift fails loudly.
_FALLBACK_TEMPLATE: dict[str, Any] = {
    "arm": "A4",
    "title": "weights",
    "question": "does uncertainty weighting collapse a task?",
    "init": "bge-small",
    "variant": "compact",
    "layers": 12,
    "hidden": 384,
    "readout": "cosine",
    "weight_mode": "learned",
    "weight_floor": 0.05,
    "weight_cap": 4.0,
    "head_only": False,
    "ordering": "curriculum",
    "mixture": "starter",
    "mixture_rows": 55000,
    "seeds": [13, 29, 47],
    "data_fingerprint": "",
    "epochs": 8,
    "batch_size": 64,
    "lr": 3.0e-4,
    "head_lr": 1.0e-3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "schedule": "cosine",
    "grad_clip": 1.0,
    "dropout": 0.1,
    "label_smoothing": 0.0,
    "optimizer": "adamw",
    "fp16": True,
    "variants": [
        {"name": "learned", "weight_mode": "learned"},
        {"name": "fixed", "weight_mode": "fixed"},
        {"name": "uniform", "weight_mode": "uniform"},
    ],
}


def build_fixed_weight_config(
    arm: str = FIXED_WEIGHT_ARM,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the fixed-weight arm definition for ``arm`` (only ``"A4"``).

    The result is a plain, YAML-serializable dict mirroring
    ``configs/experiments/a4-weights.yaml`` with:

    * ``arm: "A4"``, ``title: "weights"`` (all template keys preserved),
    * ``weight_mode: "fixed"``,
    * ``weights``: explicit uniform vector — ``1.0`` for every canonical task
      (``choice``, ``preference``, ``score``, ``probability``, ``confidence``,
      ``abstain``, ``risk``),
    * ``weight_floor`` / ``weight_cap``: ``0.05`` / ``4.0`` (unchanged; the
      floor/cap only bind the learned arm, recorded for symmetry).

    Parameters
    ----------
    arm:
        Must be ``"A4"`` — the only arm registered here; anything else raises
        ``ValueError``.
    config_path:
        Optional path to the template YAML. Defaults to the repo's
        ``configs/experiments/a4-weights.yaml``; when that is missing, the
        embedded fallback (key-for-key identical) is used.
    """
    if not isinstance(arm, str) or arm != FIXED_WEIGHT_ARM:
        raise ValueError(
            f"only the {FIXED_WEIGHT_ARM!r} fixed-weight arm is registered, got {arm!r}"
        )

    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    try:
        template = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(template, dict):
            raise ValueError(f"{path} did not parse to a mapping")
    except (OSError, yaml.YAMLError, ValueError):
        if config_path is not None:
            raise
        template = copy.deepcopy(_FALLBACK_TEMPLATE)

    config = copy.deepcopy(template)
    config["arm"] = FIXED_WEIGHT_ARM
    config["weight_mode"] = "fixed"
    config["weight_floor"] = 0.05
    config["weight_cap"] = 4.0
    config["weights"] = {task: 1.0 for task in DEFAULT_TASK_NAMES}
    return config
