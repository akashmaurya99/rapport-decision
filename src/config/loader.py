"""Pure config resolution: YAML file plus RAPPORT_* environment overrides.

Resolution order is ``schema defaults < YAML document < environment``. The
function is pure in that it performs no global mutation; its only inputs are
the file at ``path`` and the ``env`` mapping (defaults to ``os.environ``).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import (
    ConfigEnvError,
    ConfigFileUnreadableError,
    ConfigTypeMismatchError,
    ConfigUnknownKeyError,
    ConfigValueError,
    ConfigYamlError,
)
from .schema import SCHEMA, env_var_for


@dataclass(frozen=True)
class ResolvedConfig:
    """The validated run configuration, after defaults and overrides applied."""

    seed: int
    device: str
    data_dir: str
    runs_dir: str
    fp16: bool
    determinism: str
    epochs: int
    batch_size: int
    learning_rate: float
    max_length: int
    teacher: str
    data_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_env_value(key: str, raw: str, kind: str) -> Any:
    """Coerce one environment-string into the schema's Python type."""
    var = env_var_for(key)
    try:
        if kind == "int":
            return int(raw.strip(), 10)
        if kind == "float":
            return float(raw.strip())
        if kind == "bool":
            lowered = raw.strip().lower()
            truthy = {"1", "true", "yes", "on"}
            falsy = {"0", "false", "no", "off"}
            if lowered in truthy:
                return True
            if lowered in falsy:
                return False
            raise ValueError(f"expected one of {sorted(truthy | falsy)}")
        return raw  # str passes through untouched
    except ValueError as exc:
        raise ConfigEnvError(
            f"{var} could not be parsed as {kind}: {raw!r} ({exc})", key=key
        ) from exc


def _check_type(key: str, value: Any, kind: str) -> Any:
    """Enforce the declared Python type strictly.

    bool is a subclass of int, so ``isinstance`` would admit ``seed: true``;
    the schema therefore compares exact types, except float slots which accept
    both int and float (YAML writes ``3`` for a whole-number float).
    """
    if kind == "int":
        ok = type(value) is int
        expected = "int"
    elif kind == "float":
        ok = type(value) in (int, float) and type(value) is not bool
        expected = "float"
        if ok:
            return float(value)
    elif kind == "bool":
        ok = type(value) is bool
        expected = "bool"
    else:  # str
        ok = type(value) is str
        expected = "str"
    if not ok:
        raise ConfigTypeMismatchError(
            f"config key {key!r} expects {expected}, got {type(value).__name__}: {value!r}",
            key=key,
        )
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigFileUnreadableError(f"cannot read config file {path}: {exc}") from exc
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigYamlError(f"invalid YAML in {path}: {exc}") from exc
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ConfigYamlError(
            f"config file {path} must contain a mapping at the top level, "
            f"got {type(document).__name__}"
        )
    return document


def load_config(path: str | Path, env: Mapping[str, str] | None = None) -> ResolvedConfig:
    """Resolve ``path`` (YAML) with ``RAPPORT_*`` overrides from ``env``.

    Parameters
    ----------
    path:
        Path to the YAML config (e.g. ``configs/base.yaml``).
    env:
        Environment mapping to consult for overrides. Defaults to
        ``os.environ``. Tests pass an explicit mapping.

    Raises
    ------
    ConfigFileUnreadableError
        The file cannot be read.
    ConfigYamlError
        The document is not valid YAML or not a top-level mapping.
    ConfigUnknownKeyError
        The document declares a key outside the schema; the error names the key.
    ConfigTypeMismatchError
        A value's type does not match its schema slot.
    ConfigEnvError
        A ``RAPPORT_*`` override cannot be coerced.
    ConfigValueError
        A value is outside its accepted domain.
    """
    environment = os.environ if env is None else env
    document = _read_yaml(Path(path))

    unknown = sorted(set(document) - set(SCHEMA))
    if unknown:
        # Fail fast, naming the first offender alphabetically for determinism.
        key = unknown[0]
        raise ConfigUnknownKeyError(
            f"unknown config key {key!r} in {path}; declared keys are {sorted(SCHEMA)}",
            key=key,
        )

    resolved: dict[str, Any] = {}
    for key, spec in SCHEMA.items():
        if key in document:
            value = document[key]
        else:
            value = spec.default
        override = environment.get(env_var_for(key))
        if override is not None:
            value = _coerce_env_value(key, override, spec.kind)
        value = _check_type(key, value, spec.kind)
        problem = spec.validate(value)
        if problem is not None:
            raise ConfigValueError(f"config key {key!r}: {problem}", key=key)
        resolved[key] = value

    return ResolvedConfig(**{field.name: resolved[field.name] for field in fields(ResolvedConfig)})
