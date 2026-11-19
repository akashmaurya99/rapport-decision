"""Typed errors raised by the data-clean module.

Mirrors the ``src/data/core`` pattern: every error carries a stable ``code``
attribute so callers and tests can match on the failure class without parsing
message text. Codes are part of the public contract of this package.
"""

from __future__ import annotations

from src.data.core.errors import DataError


class CleanError(DataError):
    """Base class for errors raised by the cleaning pipeline."""

    code = "CLEAN_ERROR"


class ReportMismatchError(CleanError):
    """A rejection report's per-rule counters do not sum to the removed rows.

    Raised by :meth:`RejectionReport.verify` when the invariant
    ``len(input) - len(kept) == sum(per_rule.values())`` is violated. The
    pipeline never swallows this: it means a rule double-counted or a row
    vanished without a counter, and either is a bug.
    """

    code = "CLEAN_REPORT_MISMATCH"


class ContestedWriteError(CleanError):
    """The contested quarantine file could not be written."""

    code = "CLEAN_CONTESTED_WRITE"
