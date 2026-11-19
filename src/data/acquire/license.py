"""License resolution for acquisition.

Resolution sources, in precedence order:

1. an explicit ``licenses=`` mapping passed by the caller (operator override;
   e.g. a reviewed license for a source whose Hub card declares none, or a
   spreadsheet of SPDX ids),
2. a passed-in Hub metadata object (``huggingface_hub.DatasetInfo``, or any
   object shaped like it — see below),
3. nothing → the license is recorded as ``None`` and the source is
   quarantined.

No network access happens here. The Hub object is *passed in*; tests pass
stubs and fakes. A source with no resolvable license is never an exception —
it is a quarantine decision recorded in the manifest (FR-DATA-02).

Hub metadata object handling, in order:

- a mapping: ``license`` at the top level, then a ``card_data`` mapping's
  ``license``;
- an object with a ``license`` attribute that is a string/list/None
  (duck-typed; covers ``DatasetCardData`` and hand-written fakes);
- an object with a ``card_data`` attribute, where ``card_data`` is a mapping
  or an object with a ``license`` attribute (covers ``DatasetInfo`` from
  ``huggingface_hub``, where ``card_data`` is a ``DatasetCardData``).

Anything else raises :class:`LicenseResolutionError` — fail loudly rather
than guess a license.

License values may be a string or a list of strings (the Hub allows both).
Lists are normalized to ``" OR "``-joined SPDX disjunction-ish strings after
empty entries are dropped; an empty list is treated as missing.
Sentinel strings that mean "no license declared" (``""``, ``"unknown"``,
``"none"``, ``"n/a"``, ``"null"``, case-insensitive) resolve to ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import LicenseResolutionError

#: Sentinel strings (case-insensitive, after stripping) that mean
#: "no license was declared", not a license.
_MISSING_SENTINELS = frozenset({"", "unknown", "none", "n/a", "null", "not specified"})

_LICENSE_ORIGIN_MAPPING = "licenses_mapping"
_LICENSE_ORIGIN_HUB = "hub_metadata"
_LICENSE_ORIGIN_MISSING = "missing"


@dataclass(frozen=True)
class LicenseResolution:
    """The outcome of resolving one source's license.

    ``license`` is the normalized SPDX-ish identifier, or ``None`` when the
    source declares none. ``origin`` records where the answer came from so
    the manifest can say *how* the license was known, not just what it was.
    """

    source: str
    license: str | None
    origin: str


def _normalize_license_value(value: Any) -> str | None:
    """Normalize a raw license value to a single string or ``None``."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [_normalize_license_value(item) for item in value]
        joined = [part for part in parts if part is not None]
        if not joined:
            return None
        return " OR ".join(joined)
    if not isinstance(value, str):
        raise LicenseResolutionError(
            f"license value must be a string or a list of strings, "
            f"got {type(value).__name__}",
            field="license",
        )
    text = value.strip()
    if text.lower() in _MISSING_SENTINELS:
        return None
    return text


def _license_from_mapping(mapping: Mapping[str, Any]) -> str | None | Any:
    """Return the raw license value from a mapping, or a sentinel.

    Returns ``None`` both when the key is present-but-null and when absent —
    the caller treats both as "not declared here" and moves to the next
    resolution source.
    """
    card = mapping.get("card_data")
    if "license" in mapping:
        return mapping["license"]
    if isinstance(card, Mapping) and "license" in card:
        return card["license"]
    return None


def _read_hub_license(source: str, hub_metadata: Any) -> str | None:
    if isinstance(hub_metadata, Mapping):
        return _normalize_license_value(_license_from_mapping(hub_metadata))

    direct = getattr(hub_metadata, "license", None)
    if direct is not None or hasattr(hub_metadata, "license"):
        # An explicit ``license = None`` attribute (e.g. DatasetCardData with
        # no license key) means "declared nothing" — that is a valid answer,
        # not an unreadable object.
        return _normalize_license_value(direct)

    card = getattr(hub_metadata, "card_data", None)
    if card is None and hasattr(hub_metadata, "card_data"):
        return None
    if isinstance(card, Mapping):
        return _normalize_license_value(_license_from_mapping(card))
    if card is not None:
        card_license = getattr(card, "license", None)
        if card_license is None and not hasattr(card, "license"):
            raise LicenseResolutionError(
                f"card_data for source {source!r} is a {type(card).__name__} "
                f"with no readable 'license' attribute",
                field="license",
            )
        return _normalize_license_value(card_license)

    raise LicenseResolutionError(
        f"hub metadata for source {source!r} is a {type(hub_metadata).__name__} "
        f"with no readable 'license' or 'card_data' slot",
        field="license",
    )


def resolve_license(
    source: str,
    *,
    hub_metadata: Any | None = None,
    licenses: Mapping[str, str | None] | None = None,
) -> LicenseResolution:
    """Resolve the license for ``source``.

    ``licenses`` (operator mapping) wins over ``hub_metadata``. A source
    listed in ``licenses`` with a ``None`` value is treated as explicitly
    unlicensed — distinct from absent from the mapping only in ``origin``,
    and both yield a quarantine. An uninterpretable metadata object raises
    :class:`LicenseResolutionError`.
    """
    if not isinstance(source, str) or not source.strip():
        raise LicenseResolutionError("source must be a non-empty string", field="source")

    if licenses is not None:
        if not isinstance(licenses, Mapping):
            raise LicenseResolutionError(
                f"licenses must be a mapping of source -> license string or None, "
                f"got {type(licenses).__name__}",
                field="licenses",
            )
        if source in licenses:
            return LicenseResolution(
                source=source,
                license=_normalize_license_value(licenses[source]),
                origin=_LICENSE_ORIGIN_MAPPING,
            )

    if hub_metadata is not None:
        return LicenseResolution(
            source=source,
            license=_read_hub_license(source, hub_metadata),
            origin=_LICENSE_ORIGIN_HUB,
        )

    return LicenseResolution(source=source, license=None, origin=_LICENSE_ORIGIN_MISSING)


def declared_rows_from_metadata(hub_metadata: Any | None) -> int | None:
    """Best-effort declared row count from a Hub metadata object.

    Only explicit, unambiguous integer slots are read: a ``num_rows``
    attribute/key, or ``num_examples`` in a mapping. Anything else yields
    ``None`` — a missing declared count is recorded as null, never invented.
    """
    if hub_metadata is None:
        return None

    def _as_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return None

    def _from_mapping(mapping: Mapping[str, Any]) -> int | None:
        for key in ("num_rows", "num_examples"):
            value = _as_int(mapping.get(key))
            if value is not None:
                return value
        return None

    if isinstance(hub_metadata, Mapping):
        value = _from_mapping(hub_metadata)
        if value is not None:
            return value
        card = hub_metadata.get("card_data")
        return _from_mapping(card) if isinstance(card, Mapping) else None

    direct = _as_int(getattr(hub_metadata, "num_rows", None))
    if direct is not None:
        return direct
    card = getattr(hub_metadata, "card_data", None)
    if isinstance(card, Mapping):
        return _from_mapping(card)
    return _as_int(getattr(card, "num_rows", None)) if card is not None else None
