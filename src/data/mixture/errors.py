"""Typed errors raised by the data-mixture module.

Mirrors the ``src/data/core`` and ``src/data/splits`` pattern: every error
carries a stable ``code`` attribute so callers and tests can match on the
failure class without parsing message text. Codes are part of the public
contract of this package and are never renamed or reused.

Mixture failures are *fatal* — they raise, never warn. A silently mis-built
mixture is how a reported metric stops being reproducible.
"""

from __future__ import annotations

from src.data.core.errors import DataError


class MixtureError(DataError):
    """Base class for errors raised by the mixture package."""

    code = "MIXTURE_ERROR"


class MixtureInputError(MixtureError):
    """An argument to :func:`src.data.mixture.sample_mixture` was malformed."""

    code = "MIXTURE_BAD_INPUT"


class MixtureFamilyError(MixtureInputError):
    """A family key is not one of the six declared task families."""

    code = "MIXTURE_BAD_FAMILY"


class MixtureSourceError(MixtureInputError):
    """A source entry is missing/invalid, or a source repeats within a family."""

    code = "MIXTURE_BAD_SOURCE"


class MixtureFingerprintError(MixtureError):
    """The fingerprint is absent, or not a lowercase sha256 hex digest."""

    code = "MIXTURE_FINGERPRINT"


class MixtureFingerprintMismatchError(MixtureFingerprintError):
    """The caller passed a fingerprint other than the one recorded at acquisition.

    ``expected`` is the fingerprint pinned in the mixture declaration;
    ``actual`` is the one passed by the caller. The two differ, so any example
    order derived from them would diverge — refuse instead of silently
    sampling a different mixture than the one that was recorded.
    """

    code = "MIXTURE_FINGERPRINT_MISMATCH"

    def __init__(self, message: str, *, expected: str, actual: str) -> None:
        super().__init__(message, field="fingerprint")
        self.expected = expected
        self.actual = actual


class MixtureConfigError(MixtureError):
    """A mixture YAML file could not be read, parsed or validated."""

    code = "MIXTURE_CONFIG_INVALID"


class MixtureConfigFieldError(MixtureConfigError):
    """A mixture YAML field is missing, unknown or has the wrong type/domain."""

    code = "MIXTURE_CONFIG_FIELD"
