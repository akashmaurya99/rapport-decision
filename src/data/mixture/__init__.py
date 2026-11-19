"""Declarative training mixtures for RAPPORT (FR-DATA-07).

Public surface: :func:`sample_mixture` (deterministic assignment seeded by
``(seed, fingerprint)``), :func:`load_mixture` / :func:`load_preset` /
:func:`preset_path` (declarative ``mixture.yaml`` and the packaged 55k
starter and 115.6k full presets), :func:`parse_mixture_document`,
:class:`MixtureSource`, :class:`MixtureDeclaration`, :class:`MixtureSlot`,
:class:`MixtureOrder`, and the typed errors in :mod:`.errors`.
"""

from .errors import (
    MixtureConfigError,
    MixtureConfigFieldError,
    MixtureError,
    MixtureFamilyError,
    MixtureFingerprintError,
    MixtureFingerprintMismatchError,
    MixtureInputError,
    MixtureSourceError,
)
from .loader import (
    PRESET_NAMES,
    PRESETS_DIR,
    MixtureDeclaration,
    load_mixture,
    load_preset,
    parse_mixture_document,
    preset_path,
)
from .mixture import (
    DEFAULT_MIXTURE_NAME,
    MixtureOrder,
    MixtureSlot,
    MixtureSource,
    parse_family_entries,
    parse_source_entry,
    parse_sources_mapping,
    sample_mixture,
)

__all__ = [
    "DEFAULT_MIXTURE_NAME",
    "PRESET_NAMES",
    "PRESETS_DIR",
    "MixtureConfigError",
    "MixtureConfigFieldError",
    "MixtureDeclaration",
    "MixtureError",
    "MixtureFamilyError",
    "MixtureFingerprintError",
    "MixtureFingerprintMismatchError",
    "MixtureInputError",
    "MixtureOrder",
    "MixtureSlot",
    "MixtureSource",
    "MixtureSourceError",
    "load_mixture",
    "load_preset",
    "parse_family_entries",
    "parse_mixture_document",
    "parse_source_entry",
    "parse_sources_mapping",
    "preset_path",
    "sample_mixture",
]
