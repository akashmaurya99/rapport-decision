"""Declarative schema for the resolved run configuration.

The schema is flat on purpose: every supported key, its Python type, its
default and a value-domain check live in one table so the loader, the tests
and the documentation cannot drift apart. Adding a key means adding one row
here plus a line in ``configs/base.yaml`` — nothing else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

_MAX_SEED = 2**32 - 1  # numpy's legacy seed domain; torch/CUDA accept far more


def _fingerprint(value: Any) -> str | None:
    if not isinstance(value, str):
        return "expected a string"
    if value == "":
        return None  # documented pre-acquisition placeholder
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        return "expected empty, a 64-character sha256 digest, or a portable fingerprint name"
    return None


def _in(domain: set[str]) -> Callable[[Any], str | None]:
    def check(value: Any) -> str | None:
        if value not in domain:
            return f"expected one of {sorted(domain)}, got {value!r}"
        return None

    return check


def _at_least(minimum: float, *, exclusive: bool = False) -> Callable[[Any], str | None]:
    def check(value: Any) -> str | None:
        import math

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return f"expected a finite number, got {value!r}"
        ok = value > minimum if exclusive else value >= minimum
        if not ok:
            op = ">" if exclusive else ">="
            return f"expected {op} {minimum}, got {value}"
        return None

    return check


def _between(low: float, high: float) -> Callable[[Any], str | None]:
    def check(value: Any) -> str | None:
        if not (low <= value <= high):
            return f"expected between {low} and {high}, got {value}"
        return None

    return check


def _non_empty(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "expected a non-empty string"
    return None


@dataclass(frozen=True)
class KeySpec:
    """Type, default and domain check for one config key."""

    kind: str  # one of: int, float, bool, str
    default: Any
    validate: Callable[[Any], str | None]


SCHEMA: dict[str, KeySpec] = {
    "seed": KeySpec("int", 13, _between(0, _MAX_SEED)),
    "device": KeySpec("str", "cuda", _in({"cpu", "cuda", "mps"})),
    "data_dir": KeySpec("str", "data", _non_empty),
    "runs_dir": KeySpec("str", "runs", _non_empty),
    "fp16": KeySpec("bool", True, lambda v: None),
    "determinism": KeySpec("str", "strict", _in({"strict", "relaxed"})),
    "epochs": KeySpec("int", 10, _at_least(1)),
    "batch_size": KeySpec("int", 16, _at_least(1)),
    # Kept in sync with configs/base.yaml and the experiment YAMLs (lr: 3e-4).
    "learning_rate": KeySpec("float", 3.0e-4, _at_least(0.0, exclusive=True)),
    "max_length": KeySpec("int", 512, _between(1, 4096)),
    "teacher": KeySpec("str", "Qwen/Qwen3-1.7B", _non_empty),
    # Placeholder fed from RAPPORT_DATA_FINGERPRINT until TASK-07 lands a real
    # split-manifest fingerprint; an empty string means "not provided".
    "data_fingerprint": KeySpec("str", "", _fingerprint),
}

# RAPPORT_<KEY> overrides the YAML value for every declared key.
ENV_PREFIX = "RAPPORT_"


def env_var_for(key: str) -> str:
    return f"{ENV_PREFIX}{key.upper()}"


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
#
# ``configs/base.yaml`` is the authoritative source of the *shared* defaults;
# ``BASE_DEFAULTS`` below is the packaged fallback used when the installed
# distribution has no YAML files on disk. Keep the two in sync. Experiment
# YAMLs under ``configs/experiments/`` fully specify their own values and use
# different key names (notably ``lr`` instead of ``learning_rate``), so they
# are validated against their own schema and canonicalised before merging.

BASE_DEFAULTS: dict[str, Any] = {
    "seed": 13,
    "device": "cuda",
    "data_dir": "data",
    "runs_dir": "runs",
    "fp16": True,
    "determinism": "strict",
    "epochs": 10,
    "batch_size": 16,
    "learning_rate": 3.0e-4,
    "max_length": 512,
    "teacher": "Qwen/Qwen3-1.7B",
    "data_fingerprint": "",
}

# The twelve fields of ``ResolvedConfig``. These are the only keys with
# RAPPORT_* environment overrides and the only keys to_resolved() consumes.
SHARED_FIELDS: tuple[str, ...] = tuple(BASE_DEFAULTS)

# Experiment YAML alias -> canonical ResolvedConfig field.
KEY_ALIASES: dict[str, str] = {"lr": "learning_rate"}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def _portable_identifier(value: Any) -> str | None:
    if not isinstance(value, str):
        return "expected a string"
    if not value or len(value) > 128:
        return "expected a non-empty string of at most 128 characters"
    if value.upper() in _WINDOWS_RESERVED:
        return f"{value!r} is a reserved filesystem name"
    if not _IDENTIFIER_RE.fullmatch(value) or value.endswith("."):
        return "expected a portable name: letters, digits, '.', '_' or '-' only"
    return None


def _non_empty_list(value: Any) -> str | None:
    if not isinstance(value, (list, tuple)):
        return "expected a non-empty list"
    if not value:
        return "expected a non-empty list"
    return None


def _seed_list(value: Any) -> str | None:
    problem = _non_empty_list(value)
    if problem:
        return problem
    for seed in value:
        if type(seed) is not int or not 0 <= seed <= _MAX_SEED:
            return f"expected seed integers in 0..{_MAX_SEED}, got {seed!r}"
    return None


def _variant_list(value: Any) -> str | None:
    problem = _non_empty_list(value)
    if problem:
        return problem
    for variant in value:
        if not isinstance(variant, dict):
            return f"expected a mapping, got {type(variant).__name__}"
        name = variant.get("name")
        if not isinstance(name, str) or not name.strip():
            return "every variant requires a non-empty 'name'"
    return None


def _float_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if type(value) not in (int, float) or isinstance(value, bool):
        return f"expected a float or null, got {type(value).__name__}"
    import math

    if not math.isfinite(value):
        return f"expected a finite float or null, got {value}"
    if value <= 0:
        return f"expected > 0 or null, got {value}"
    return None


def _bool_list(value: Any) -> str | None:
    problem = _non_empty_list(value)
    if problem:
        return problem
    if any(type(item) is not bool for item in value):
        return "expected a list of booleans"
    return None


def _unit(value: Any) -> str | None:
    return _between(0.0, 1.0)(value)


def _dropout(value: Any) -> str | None:
    if type(value) not in (int, float) or isinstance(value, bool):
        return f"expected a float in [0, 1), got {value!r}"
    if not 0.0 <= value < 1.0:
        return f"expected a float in [0, 1), got {value}"
    return None


EXPERIMENT_SCHEMA: dict[str, KeySpec] = {
    # Identity / provenance.
    "arm": KeySpec("str", "", _portable_identifier),
    "title": KeySpec("str", "", _non_empty),
    "question": KeySpec("str", "", _non_empty),
    # Model / ablation knobs. Free-form strings stay non-empty rather than
    # closed enums so new arms do not require a schema change.
    "init": KeySpec("str", "", _non_empty),
    "init_alternates": KeySpec("str_list", (), _non_empty_list),
    "variant": KeySpec("str", "", _non_empty),
    "layers": KeySpec("int", 0, _at_least(1)),
    "hidden": KeySpec("int", 0, _at_least(1)),
    "readout": KeySpec("str", "", _non_empty),
    "readout_alternates": KeySpec("str_list", (), _non_empty_list),
    "weight_mode": KeySpec("str", "", _non_empty),
    "weight_floor": KeySpec("float", 0.0, _at_least(0.0)),
    "weight_cap": KeySpec("float", 0.0, _at_least(0.0)),
    "head_only": KeySpec("bool", False, lambda v: None),
    "head_only_alternates": KeySpec("bool_list", (), _bool_list),
    "ordering": KeySpec("str", "", _non_empty),
    "ordering_alternates": KeySpec("str_list", (), _non_empty_list),
    "mixture": KeySpec("str", "", _non_empty),
    "mixture_rows": KeySpec("int", 0, _at_least(1)),
    "seeds": KeySpec("int_list", (), _seed_list),
    "kd": KeySpec("bool", False, lambda v: None),
    "context": KeySpec("str", "", lambda v: None),
    "max_length": KeySpec("int", 0, _between(1, 4096)),
    "infer_max_length": KeySpec("int", 0, _between(1, 4096)),
    "ood_source": KeySpec("str", "", _non_empty),
    "y2_comparator": KeySpec("str", "", _non_empty),
    "variants": KeySpec("mapping_list", (), _variant_list),
    # Training loop knobs (some are shared with ResolvedConfig).
    "epochs": KeySpec("int", 0, _at_least(1)),
    "batch_size": KeySpec("int", 0, _at_least(1)),
    "lr": KeySpec("float", 0.0, _at_least(0.0, exclusive=True)),
    "head_lr": KeySpec("float_or_none", None, _float_or_none),
    "weight_decay": KeySpec("float", 0.0, _at_least(0.0)),
    "keep_last_n": KeySpec("int", 3, _at_least(1)),  # checkpoint retention (disk budget)
    "warmup_ratio": KeySpec("float", 0.0, _unit),
    "schedule": KeySpec("str", "", _non_empty),
    "grad_clip": KeySpec("float", 0.0, _at_least(0.0)),
    "dropout": KeySpec("float", 0.0, _dropout),
    "label_smoothing": KeySpec("float", 0.0, _unit),
    "optimizer": KeySpec("str", "", _non_empty),
    "fp16": KeySpec("bool", False, lambda v: None),
    "data_fingerprint": KeySpec("str", "", _fingerprint),
    "teacher": SCHEMA["teacher"],
    "learning_rate": SCHEMA["learning_rate"],
}

# Base/shared keys that experiments may also spell explicitly.
for _shared in SHARED_FIELDS:
    EXPERIMENT_SCHEMA.setdefault(_shared, SCHEMA[_shared])
