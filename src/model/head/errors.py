"""Typed errors raised by the model-head package.

Every error carries a stable ``code`` attribute so callers (and tests) can
match on the failure class without parsing message text. The code is included
in ``str(error)`` for log readability, mirroring the ``src/config``,
``src/data`` and ``src/tokenization`` patterns. Codes are part of the public
contract: downstream tasks may match on them, so a code is never renamed or
reused.
"""

from __future__ import annotations


class ModelError(Exception):
    """Base class for errors raised by this package."""

    code = "MODEL_ERROR"

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(f"{self.code}: {message}")
        self.message = message
        self.field = field


class InvalidOptionCountError(ModelError):
    """The declared option count is outside the accepted 2..8 range (E2).

    Raised by input validation, before any encoder compute happens: either a
    batch slot width that is 1 or greater than 8, or a row of ``option_mask``
    whose real-option count falls outside 2..8.
    """

    code = "MODEL_INVALID_OPTION_COUNT"


class OptionMaskError(ModelError):
    """The input layout is structurally invalid.

    Covers malformed ``input_ids`` (wrong rank or dtype) and ``option_mask``
    failures: width/batch mismatch, non-binary values, non-prefix (padding not
    trailing) layouts, and more or fewer ``<opt>`` markers in a row than the
    mask declares. (``<goal>`` marker count is reported by :class:`GoalMarkerError`.)
    """

    code = "MODEL_OPTION_MASK_INVALID"


class GoalMarkerError(ModelError):
    """The input ids do not contain exactly one ``<goal>`` marker per row."""

    code = "MODEL_GOAL_MARKER_INVALID"


class AttentionMaskError(ModelError):
    """``attention_mask`` has the wrong shape or non-binary values."""

    code = "MODEL_ATTENTION_MASK_INVALID"


class InvalidTaskIdError(ModelError):
    """``task_ids`` is not a 1-D integer vector in range for the configured tasks."""

    code = "MODEL_INVALID_TASK_ID"


class TaskMarkerError(ModelError):
    """The ``<task>`` marker is missing, duplicated, left-padded or not first."""

    code = "MODEL_TASK_MARKER_INVALID"


class DeviceMismatchError(ModelError):
    """An input tensor lives on a different device than ``input_ids``."""

    code = "MODEL_DEVICE_MISMATCH"
