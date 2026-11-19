"""Typed errors raised by the tokenization package.

Every error carries a stable ``code`` attribute so callers (and tests) can
match on the failure class without parsing message text. The code is included
in ``str(error)`` for log readability, mirroring the ``src/data`` and
``src/config`` patterns. Codes are part of the public contract: downstream
tasks may match on them, so a code is never renamed or reused.
"""

from __future__ import annotations


class TokenizationError(Exception):
    """Base class for errors raised by this package."""

    code = "TOK_ERROR"

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(f"{self.code}: {message}")
        self.message = message
        self.field = field


class BudgetTooSmallError(TokenizationError):
    """The requested budget (or batch_size / bucket widths) is not valid."""

    code = "TOK_BUDGET_INVALID"


class BucketOverflowError(TokenizationError):
    """An encoded example does not fit the largest bucket width."""

    code = "TOK_BUCKET_OVERFLOW"
