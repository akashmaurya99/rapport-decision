"""Typed errors raised by the data-splits module.

Mirrors the ``src/data/core`` and ``src/data/clean`` pattern: every error
carries a stable ``code`` attribute so callers and tests can match on the
failure class without parsing message text. Codes are part of the public
contract of this package.

Leakage failures are *fatal* — they raise, never warn. Each leakage error
carries the offending counts as attributes so CI can log exactly what
leaked without parsing the message.
"""

from __future__ import annotations

from src.data.core.errors import DataError


class SplitError(DataError):
    """Base class for errors raised by the splits package."""

    code = "SPLIT_ERROR"


class SplitInputError(SplitError):
    """An argument to :func:`src.data.splits.split` was malformed."""

    code = "SPLIT_BAD_INPUT"


class GroupOverlapError(SplitError):
    """A ``group_id`` ended up in more than one split. Fatal.

    ``overlap`` is the number of groups spanning splits; ``groups`` lists up
    to five offending group ids (sorted, for stable messages).
    """

    code = "SPLIT_GROUP_OVERLAP"

    def __init__(
        self,
        message: str,
        *,
        overlap: int,
        groups: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message, field="group_id")
        self.overlap = overlap
        self.groups = groups


class OptionTextLeakageError(SplitError):
    """Identical normalized option text of ≥40 tokens appears in two splits. Fatal.

    ``count`` is the number of distinct violating option texts; ``texts``
    lists up to five of them (sorted, for stable messages); ``pair_counts``
    tallies violations per split pair (``"dev/test"`` etc.).
    """

    code = "SPLIT_OPTION_TEXT_LEAK"

    def __init__(
        self,
        message: str,
        *,
        count: int,
        texts: tuple[str, ...] = (),
        pair_counts: dict[str, int] | None = None,
    ) -> None:
        super().__init__(message, field="options")
        self.count = count
        self.texts = texts
        self.pair_counts = dict(pair_counts or {})


class NGramLeakageError(SplitError):
    """An exact 13-gram appears in more than one split. Fatal.

    ``counts`` maps the offending split pair (``"dev/test"`` ...) to the
    number of shared 13-grams; ``total`` is their sum.
    """

    code = "SPLIT_13GRAM_OVERLAP"

    def __init__(self, message: str, *, counts: dict[str, int]) -> None:
        super().__init__(message)
        self.counts = dict(counts)
        self.total = sum(self.counts.values())


# --- Manifest ---------------------------------------------------------------


class ManifestError(SplitError):
    """A split manifest failed to serialize, parse or validate."""

    code = "SPLIT_MANIFEST_INVALID"


class ManifestFieldError(ManifestError):
    """A manifest field is missing, unknown or has the wrong type/domain."""

    code = "SPLIT_MANIFEST_FIELD"


class ManifestFingerprintError(ManifestError):
    """The manifest fingerprint is not a lowercase sha256 hex digest."""

    code = "SPLIT_MANIFEST_FINGERPRINT"


class ManifestGroupOverlapError(ManifestError):
    """The manifest records a positive ``group_overlap``. Fatal.

    ``overlap`` is the recorded value. Any non-zero overlap — in a freshly
    built manifest or one loaded from disk — is a hard error, never a warning.
    """

    code = "SPLIT_MANIFEST_GROUP_OVERLAP"

    def __init__(self, message: str, *, overlap: int) -> None:
        super().__init__(message, field="group_overlap")
        self.overlap = overlap
