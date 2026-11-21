"""Pure config resolution: YAML file plus RAPPORT_* environment overrides.

Resolution order is ``schema defaults < YAML document < environment``. The
functions are pure: no global mutation; their only inputs are the file at
``path`` and the ``env`` mapping (defaults to ``os.environ``).

Two public entry points exist:

* :func:`load_config` — the legacy, flat ``configs/base.yaml`` path.
* :func:`load_experiment_config` — the validated boundary for experiment
  YAMLs (``configs/experiments/*.yaml``). It merges authoritative base
  defaults, rejects unknown keys, validates every declared field, applies
  ``RAPPORT_*`` overrides and exposes :meth:`ExperimentConfig.to_resolved`.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import MappingProxyType
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
from .schema import (
    BASE_DEFAULTS,
    EXPERIMENT_SCHEMA,
    KEY_ALIASES,
    SCHEMA,
    SHARED_FIELDS,
    KeySpec,
    env_var_for,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_BASE_YAML = REPO_ROOT / "configs" / "base.yaml"


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


@dataclass(frozen=True)
class ExperimentConfig:
    """A fully validated experiment YAML, ready to become a ResolvedConfig.

    ``document`` is the merged, canonicalised and validated view of the YAML
    (base defaults < experiment document < RAPPORT_* env overrides). It is a
    read-only mapping. Aliases are kept alongside their canonical keys so
    ``value("lr")`` and ``value("learning_rate")`` both work.
    """

    path: Path
    document: Mapping[str, Any]
    env_overrides: Mapping[str, str]

    def value(self, key: str, default: Any = None) -> Any:
        """Return a validated document value (or ``default`` when absent)."""
        return self.document.get(key, default)

    def to_resolved(
        self, explicit_overrides: Mapping[str, Any] | None = None
    ) -> ResolvedConfig:
        """Build the 12-field :class:`ResolvedConfig` for this experiment.

        ``explicit_overrides`` (e.g. CLI-derived seed, device or runtime data
        fingerprint) wins over both YAML and ``RAPPORT_*`` values. Unknown or
        invalid overrides raise the same typed errors as ``load_config``.
        """
        values: dict[str, Any] = {
            key: self.document.get(key, SCHEMA[key].default) for key in SHARED_FIELDS
        }
        if explicit_overrides:
            unknown = sorted(set(explicit_overrides) - set(SHARED_FIELDS))
            if unknown:
                raise ConfigUnknownKeyError(
                    f"unknown resolved-config override {unknown[0]!r}; "
                    f"allowed keys are {list(SHARED_FIELDS)}",
                    key=unknown[0],
                )
            values.update(explicit_overrides)
        for key, spec in SCHEMA.items():
            value = _check_type(key, values[key], spec.kind)
            problem = spec.validate(value)
            if problem is not None:
                raise ConfigValueError(f"config key {key!r}: {problem}", key=key)
            values[key] = value
        return ResolvedConfig(**values)


def _coerce_env_value(key: str, raw: str, kind: str) -> Any:
    """Coerce one environment-string into the schema's Python type."""
    var = env_var_for(key)
    if not isinstance(raw, str):
        raise ConfigEnvError(
            f"{var} must be a string from the environment, got {type(raw).__name__}: {raw!r}",
            key=key,
        )
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

    unknown = sorted(str(key) for key in set(document) - set(SCHEMA))
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


def _check_experiment_type(key: str, value: Any, kind: str) -> Any:
    """Type check for the wider experiment schema (lists, mappings, scalars)."""
    if kind in ("int", "float", "bool", "str"):
        return _check_type(key, value, kind)
    if kind == "float_or_none":
        if value is not None:
            value = _check_type(key, value, "float")
        return value
    expected: dict[str, tuple[type, ...]] = {
        "str_list": (str,),
        "int_list": (int,),
        "any_list": (),
        "bool_list": (bool,),
        "mapping_list": (dict,),
    }
    if kind not in expected:
        raise ConfigTypeMismatchError(
            f"internal schema error: unknown kind {kind!r} for {key!r}", key=key
        )
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ConfigTypeMismatchError(
            f"config key {key!r} expects a list, got {type(value).__name__}: {value!r}",
            key=key,
        )
    item_types = expected[kind]
    if item_types:
        for item in value:
            ok = type(item) in item_types if kind == "int_list" else isinstance(item, item_types)
            if not ok:
                expected_name = {str: "str", int: "int", dict: "mapping"}[item_types[0]]
                raise ConfigTypeMismatchError(
                    f"config key {key!r} expects a list of {expected_name}, "
                    f"got {type(item).__name__}: {item!r}",
                    key=key,
                )
    return list(value)


def _spec_for(key: str) -> KeySpec:
    if key in EXPERIMENT_SCHEMA:
        return EXPERIMENT_SCHEMA[key]
    return SCHEMA[key]


def _base_defaults() -> dict[str, Any]:
    """Read configs/base.yaml when present, else the packaged fallback.

    The YAML file is authoritative in a source checkout; the packaged copy in
    :data:`~src.config.schema.BASE_DEFAULTS` keeps an installed distribution
    self-contained (YAML config files are not package data).
    """
    defaults = dict(BASE_DEFAULTS)
    if not _BASE_YAML.is_file():
        return defaults
    document = _read_yaml(_BASE_YAML)
    unknown = sorted(str(key) for key in set(document) - set(SCHEMA))
    if unknown:
        raise ConfigUnknownKeyError(
            f"unknown key {unknown[0]!r} in base config {_BASE_YAML}", key=unknown[0]
        )
    defaults.update(document)
    return defaults


def load_experiment_config(
    path: str | Path, env: Mapping[str, str] | None = None
) -> ExperimentConfig:
    """Load and validate an experiment YAML as the single config boundary.

    Order of precedence (last wins)::

        configs/base.yaml (authoritative shared defaults)
        < experiment YAML document
        < RAPPORT_* environment overrides

    Unknown keys (e.g. ``lerning_rate``) are rejected by name, every value is
    type- and domain-checked, and ``lr`` is canonicalised to ``learning_rate``.
    Use :meth:`ExperimentConfig.to_resolved` to obtain the 12-field run config.
    """
    environment = os.environ if env is None else env
    source = Path(path)
    document = _read_yaml(source)

    if "lr" in document and "learning_rate" in document:
        raise ConfigValueError(
            "declare exactly one of 'lr' and 'learning_rate', not both",
            key="learning_rate",
        )
    allowed = set(EXPERIMENT_SCHEMA) | set(KEY_ALIASES)
    unknown = sorted(str(key) for key in set(document) - allowed)
    if unknown:
        key = unknown[0]
        raise ConfigUnknownKeyError(
            f"unknown config key {key!r} in {source}; declared keys are {sorted(allowed)}",
            key=key,
        )

    canonical: dict[str, Any] = {}
    for key, value in document.items():
        target = KEY_ALIASES.get(key, key)
        canonical[target] = value
        if target != key:
            canonical[key] = value  # keep the alias readable via .value()

    merged = _base_defaults()
    merged.update(canonical)

    # Nested ablation variants are configuration too: validate every override
    # against the same schema instead of accepting misspellings/NaN/Infinity.
    variants = document.get("variants", [])
    if isinstance(variants, (list, tuple)):
        for index, variant in enumerate(variants):
            if not isinstance(variant, dict):
                continue  # _variant_list reports the structural problem
            unknown_variant = sorted(
                str(key) for key in set(variant) - allowed - {"name"}
            )
            if unknown_variant:
                raise ConfigUnknownKeyError(
                    f"unknown variant override key {unknown_variant[0]!r} "
                    f"in variants[{index}]",
                    key=unknown_variant[0],
                )
            for key, value in variant.items():
                if key == "name":
                    continue
                spec = _spec_for(key)
                checked = _check_experiment_type(key, value, spec.kind)
                problem = spec.validate(checked)
                if problem is not None:
                    raise ConfigValueError(
                        f"variants[{index}].{key}: {problem}", key=key
                    )

    for key, value in list(merged.items()):
        spec = _spec_for(key)
        value = _check_experiment_type(key, value, spec.kind)
        problem = spec.validate(value)
        if problem is not None:
            raise ConfigValueError(f"config key {key!r}: {problem}", key=key)
        merged[key] = value

    applied_env: dict[str, str] = {}
    for key in SHARED_FIELDS:
        var = env_var_for(key)
        raw = environment.get(var)
        if raw is None:
            continue
        value = _coerce_env_value(key, raw, SCHEMA[key].kind)
        try:
            value = _check_type(key, value, SCHEMA[key].kind)
        except ConfigTypeMismatchError as exc:
            raise ConfigEnvError(f"{var}: {exc.message}", key=key) from exc
        problem = SCHEMA[key].validate(value)
        if problem is not None:
            raise ConfigEnvError(f"{var}: {problem}", key=key)
        merged[key] = value
        if key == "learning_rate":
            merged["lr"] = value
        applied_env[var] = raw

    return ExperimentConfig(
        path=source,
        document=MappingProxyType(merged),
        env_overrides=MappingProxyType(applied_env),
    )
