"""DecisionExample contract and normalization for RAPPORT.

Public surface: ``Option``/``DecisionExample`` (frozen records),
``normalize_example``/``validate_example`` (fail-fast and collect-all
validation with stable codes), ``normalize_text``/``strip_html``/
``collapse_whitespace`` (the normalization pipeline), ``compute_example_id``
(sha256 content identity, option-order invariant), and the
``to_dict``/``from_dict``/``to_json``/``from_json`` round-trip. Parquet
helpers (``write_parquet``/``read_parquet``) are optional and require
pyarrow at call time.
"""

from .errors import (
    BadSplitError,
    BadTaskFamilyError,
    BadTypeError,
    ChosenIndexMissingError,
    ChosenIndexOutOfRangeError,
    DataError,
    DataValidationError,
    DuplicateOptionError,
    EmptyFieldError,
    EmptyGoalError,
    EmptyOptionError,
    EmptyStateError,
    HtmlStripFailedError,
    IdMismatchError,
    JsonInvalidError,
    MissingFieldError,
    ParquetError,
    ParquetUnavailableError,
    RiskLabelInvalidError,
    ScoreOutOfRangeError,
    ScoresLengthMismatchError,
    TooFewOptionsError,
    TooManyOptionsError,
    UnknownFieldError,
)
from .normalize import (
    collapse_whitespace,
    normalize_example,
    normalize_text,
    strip_html,
    validate_example,
)
from .parquet import read_parquet, write_parquet
from .schema import (
    MAX_OPTIONS,
    MIN_OPTIONS,
    RISK_CATEGORY_COUNT,
    SPLITS,
    TASK_FAMILIES,
    DecisionExample,
    Option,
    compute_example_id,
)
from .serialize import (
    from_dict,
    from_json,
    read_json,
    to_dict,
    to_json,
    write_json,
)

__all__ = [
    "MAX_OPTIONS",
    "MIN_OPTIONS",
    "RISK_CATEGORY_COUNT",
    "SPLITS",
    "TASK_FAMILIES",
    "BadSplitError",
    "BadTaskFamilyError",
    "BadTypeError",
    "ChosenIndexMissingError",
    "ChosenIndexOutOfRangeError",
    "DataError",
    "DataValidationError",
    "DecisionExample",
    "DuplicateOptionError",
    "EmptyFieldError",
    "EmptyGoalError",
    "EmptyOptionError",
    "EmptyStateError",
    "HtmlStripFailedError",
    "IdMismatchError",
    "JsonInvalidError",
    "MissingFieldError",
    "Option",
    "ParquetError",
    "ParquetUnavailableError",
    "RiskLabelInvalidError",
    "ScoreOutOfRangeError",
    "ScoresLengthMismatchError",
    "TooFewOptionsError",
    "TooManyOptionsError",
    "UnknownFieldError",
    "collapse_whitespace",
    "compute_example_id",
    "from_dict",
    "from_json",
    "normalize_example",
    "normalize_text",
    "read_json",
    "read_parquet",
    "strip_html",
    "to_dict",
    "to_json",
    "validate_example",
    "write_json",
    "write_parquet",
]
