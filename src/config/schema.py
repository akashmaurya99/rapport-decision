"""Declarative schema for the resolved run configuration.

The schema is flat on purpose: every supported key, its Python type, its
default and a value-domain check live in one table so the loader, the tests
and the documentation cannot drift apart. Adding a key means adding one row
here plus a line in ``configs/base.yaml`` — nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

_MAX_SEED = 2**32 - 1  # numpy's legacy seed domain; torch/CUDA accept far more


def _in(domain: set[str]) -> Callable[[Any], str | None]:
    def check(value: Any) -> str | None:
        if value not in domain:
            return f"expected one of {sorted(domain)}, got {value!r}"
        return None

    return check


def _at_least(minimum: float, *, exclusive: bool = False) -> Callable[[Any], str | None]:
    def check(value: Any) -> str | None:
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
    "learning_rate": KeySpec("float", 3.0e-5, _at_least(0.0, exclusive=True)),
    "max_length": KeySpec("int", 512, _between(1, 4096)),
    "teacher": KeySpec("str", "Qwen/Qwen3-1.7B", _non_empty),
    # Placeholder fed from RAPPORT_DATA_FINGERPRINT until TASK-07 lands a real
    # split-manifest fingerprint; an empty string means "not provided".
    "data_fingerprint": KeySpec("str", "", lambda v: None),
}

# RAPPORT_<KEY> overrides the YAML value for every declared key.
ENV_PREFIX = "RAPPORT_"


def env_var_for(key: str) -> str:
    return f"{ENV_PREFIX}{key.upper()}"
