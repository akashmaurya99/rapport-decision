"""Typed errors raised by the data core module.

Every error carries a stable ``code`` attribute so callers (and tests) can
match on the failure class without parsing message text. The code is included
in ``str(error)`` for log readability, mirroring the ``src/config`` pattern.

Codes are part of the public contract: downstream tasks (adapters, cleaning,
splits) may match on them, so a code is never renamed or reused for a
different failure class.
"""

from __future__ import annotations


class DataError(Exception):
    """Base class for errors raised by this package."""

    code = "DATA_ERROR"

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(f"{self.code}: {message}")
        self.message = message
        self.field = field


class DataValidationError(DataError):
    """A record failed validation against the DecisionExample contract."""

    code = "DATA_VALIDATION_ERROR"


# --- Contract violations, one distinct code each ---------------------------


class EmptyGoalError(DataValidationError):
    """The goal is absent, not a string, or empty after normalization."""

    code = "DATA_EMPTY_GOAL"


class EmptyStateError(DataValidationError):
    """The state is empty, which the contract only allows for preference."""

    code = "DATA_EMPTY_STATE"


class MissingFieldError(DataValidationError):
    """A required field is absent from the record."""

    code = "DATA_MISSING_FIELD"


class EmptyFieldError(DataValidationError):
    """A required string field normalizes to empty (source, license, group_id)."""

    code = "DATA_EMPTY_FIELD"


class TooFewOptionsError(DataValidationError):
    """Fewer than two options were provided."""

    code = "DATA_TOO_FEW_OPTIONS"


class TooManyOptionsError(DataValidationError):
    """More than eight options were provided (the contract's default path)."""

    code = "DATA_TOO_MANY_OPTIONS"


class EmptyOptionError(DataValidationError):
    """An option's text is empty after normalization."""

    code = "DATA_EMPTY_OPTION"


class DuplicateOptionError(DataValidationError):
    """Two or more options are identical after normalization."""

    code = "DATA_DUPLICATE_OPTION"


class ChosenIndexOutOfRangeError(DataValidationError):
    """``chosen_index`` is not within ``0 <= index < len(options)``."""

    code = "DATA_CHOSEN_INDEX_OUT_OF_RANGE"


class ChosenIndexMissingError(DataValidationError):
    """``chosen_index`` is absent for a choice or preference family."""

    code = "DATA_MISSING_CHOSEN_INDEX"


class BadTaskFamilyError(DataValidationError):
    """``task_family`` is not one of the six declared families."""

    code = "DATA_BAD_TASK_FAMILY"


class BadSplitError(DataValidationError):
    """``split`` is not one of train, dev or test."""

    code = "DATA_BAD_SPLIT"


class ScoresLengthMismatchError(DataValidationError):
    """``scores`` is present but its length does not equal the option count."""

    code = "DATA_SCORES_LENGTH_MISMATCH"


class ScoreOutOfRangeError(DataValidationError):
    """A score lies outside the normalized utility range 0..1."""

    code = "DATA_SCORE_OUT_OF_RANGE"


class RiskLabelInvalidError(DataValidationError):
    """A risk label is empty, not a string, duplicated, or exceeds eight labels."""

    code = "DATA_RISK_LABEL_INVALID"


class BadTypeError(DataValidationError):
    """A field has the wrong JSON-level type for its contract slot."""

    code = "DATA_BAD_TYPE"


# --- Serialization and integrity --------------------------------------------


class UnknownFieldError(DataError):
    """The record contains a key the contract does not declare."""

    code = "DATA_UNKNOWN_FIELD"


class IdMismatchError(DataError):
    """A provided ``example_id`` does not match the recomputed content hash."""

    code = "DATA_ID_MISMATCH"


class JsonInvalidError(DataError):
    """The payload is not valid JSON, or not a JSON object at the top level."""

    code = "DATA_JSON_INVALID"


class HtmlStripFailedError(DataError):
    """The stdlib HTML parser failed to scan a text field."""

    code = "DATA_HTML_STRIP_FAILED"


# --- Optional parquet support ------------------------------------------------


class ParquetUnavailableError(DataError):
    """pyarrow is not importable; parquet support is optional, not invented."""

    code = "DATA_PARQUET_UNAVAILABLE"


class ParquetError(DataError):
    """A parquet read or write failed."""

    code = "DATA_PARQUET_ERROR"
