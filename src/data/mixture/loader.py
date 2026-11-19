"""Loader for declarative ``mixture.yaml`` files and the packaged presets.

Format (``family -> {source, rows, weight?}``, extended to a list of such
mappings for multi-source families)::

    name: starter                     # optional; defaults to the file stem
    fingerprint: ""                   # optional; "" = resolved at acquisition time
    families:
      choice: {source: "cais/mmlu#auxiliary_train", rows: 12000}
      preference:
        - {source: "Anthropic/hh-rlhf", rows: 16000}
        - {source: "nvidia/HelpSteer2", rows: 4000, weight: 1.0}

The two packaged presets (``starter`` 55k and ``full`` 115.6k, from
docs/dataset-strategy.md §6) live in ``src/data/mixture/presets/`` and are
loaded with :func:`load_preset`; experiment configs reference them by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from src.data.core.schema import TASK_FAMILIES
from src.data.splits.fingerprint import is_sha256_hex

from .errors import MixtureConfigError, MixtureConfigFieldError, MixtureFamilyError
from .mixture import (
    MixtureOrder,
    MixtureSource,
    parse_family_entries,
    sample_mixture,
)

PRESETS_DIR = Path(__file__).resolve().parent / "presets"
PRESET_NAMES = ("starter", "full")
_KNOWN_FAMILIES = frozenset(TASK_FAMILIES)
_TOP_LEVEL_KEYS = frozenset({"name", "fingerprint", "families"})
_DEFAULT_NAME = "mixture"


@dataclass(frozen=True)
class MixtureDeclaration:
    """A parsed, validated mixture declaration.

    ``fingerprint`` is the recorded data fingerprint pinned at acquisition
    time, or ``""`` while it is unresolved. Pinning is what turns the
    fingerprint placeholder in experiment configs into a hard check: a caller
    passing any other fingerprint gets a
    :class:`MixtureFingerprintMismatchError` from :func:`sample_mixture`.
    """

    name: str
    fingerprint: str
    sources: dict[str, tuple[MixtureSource, ...]]

    @property
    def total_rows(self) -> int:
        return sum(entry.rows for entries in self.sources.values() for entry in entries)

    def family_rows(self) -> dict[str, int]:
        return {
            family: sum(entry.rows for entry in entries)
            for family, entries in self.sources.items()
        }

    def counts(self) -> dict[str, dict[str, int]]:
        return {
            family: {entry.source: entry.rows for entry in entries}
            for family, entries in self.sources.items()
        }

    def sample(self, seed: int, fingerprint: str | None = None) -> MixtureOrder:
        """Sample the declaration against ``fingerprint``.

        Defaults to the pinned fingerprint; passing a different one (or a
        non-sha256 value) raises a typed error from :func:`sample_mixture`.
        """
        target = self.fingerprint if fingerprint is None else fingerprint
        if not target:
            raise MixtureConfigError(
                f"mixture {self.name!r} pins no fingerprint; pass the "
                "acquisition-time fingerprint recorded in the split manifest",
                field="fingerprint",
            )
        return sample_mixture(self, None, seed, target)


def _validate_optional_fingerprint(value: object, *, field_name: str = "fingerprint") -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise MixtureConfigFieldError(
            f"{field_name} must be a string (\"\" = unresolved at acquisition time), "
            f"got {value!r}",
            field=field_name,
        )
    candidate = value.strip()
    if candidate and not is_sha256_hex(candidate):
        raise MixtureConfigFieldError(
            f"{field_name} must be a lowercase sha256 hex digest or \"\", got {value!r}",
            field=field_name,
        )
    return candidate


def _validate_name(value: object, *, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise MixtureConfigFieldError(
            f"name must be a non-empty string, got {value!r}", field="name"
        )
    return value.strip()


def parse_mixture_document(
    data: object, *, default_name: str = _DEFAULT_NAME
) -> MixtureDeclaration:
    """Validate an already-parsed mapping (the YAML document)."""
    if not isinstance(data, Mapping):
        raise MixtureConfigError(
            f"mixture document must be a mapping at the top level, "
            f"got {type(data).__name__}"
        )
    unknown = sorted(str(key) for key in data if key not in _TOP_LEVEL_KEYS)
    if unknown:
        raise MixtureConfigFieldError(
            f"undeclared field(s): {', '.join(unknown)}; expected a subset of "
            f"{sorted(_TOP_LEVEL_KEYS)}",
            field=unknown[0],
        )
    if "families" not in data:
        raise MixtureConfigFieldError(
            "mixture document is missing required field 'families'", field="families"
        )
    families = data["families"]
    if not isinstance(families, Mapping) or not families:
        raise MixtureConfigFieldError(
            f"'families' must be a non-empty mapping of task family to source "
            f"entries, got {families!r}",
            field="families",
        )
    parsed: dict[str, tuple[MixtureSource, ...]] = {}
    for raw_family, value in families.items():
        family = raw_family if isinstance(raw_family, str) else str(raw_family)
        if family not in _KNOWN_FAMILIES:
            raise MixtureFamilyError(
                f"unknown task family {raw_family!r}; expected one of {sorted(_KNOWN_FAMILIES)}",
                field="families",
            )
        parsed[family] = parse_family_entries(
            family, value, field_name=f"families.{family}"
        )
    name = _validate_name(data.get("name"), default=default_name)
    fingerprint = _validate_optional_fingerprint(data.get("fingerprint"))
    return MixtureDeclaration(name=name, fingerprint=fingerprint, sources=parsed)


def load_mixture(path: str | Path) -> MixtureDeclaration:
    """Parse and validate a ``mixture.yaml`` file."""
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MixtureConfigError(f"mixture file not found: {target}") from exc
    except OSError as exc:
        raise MixtureConfigError(f"mixture file cannot be read at {target}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise MixtureConfigError(f"mixture file {target} is not valid YAML: {exc}") from exc
    return parse_mixture_document(data, default_name=target.stem)


def preset_path(name: str) -> Path:
    """Absolute path of a packaged preset mixture (``starter`` or ``full``)."""
    if name not in PRESET_NAMES:
        raise MixtureConfigError(
            f"unknown preset {name!r}; expected one of {list(PRESET_NAMES)}", field="name"
        )
    return PRESETS_DIR / f"{name}.yaml"


def load_preset(name: str) -> MixtureDeclaration:
    """Load a packaged preset mixture by name (``starter`` or ``full``)."""
    return load_mixture(preset_path(name))
