"""Typed errors for the calibration module (TASK-16).

Every failure raised by ``src/calibration`` inherits from
:class:`CalibrationError` so callers can catch one base type. Malformed
*inputs* (bad shapes, out-of-domain values) raise :class:`CalibrationInputError`;
problems reading or writing a :class:`~src.calibration.map.CalibrationMap`
raise :class:`CalibrationMapError`. No bare ``ValueError`` escapes this
package.
"""

from __future__ import annotations


class CalibrationError(Exception):
    """Base error for the calibration package.

    ``code`` is a stable machine-readable token; ``message`` is human-facing.
    """

    code = "CALIBRATION_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class CalibrationInputError(CalibrationError):
    """Malformed calibration input (shapes, domains, fold counts, …)."""

    code = "CALIBRATION_INPUT"


class CalibrationMapError(CalibrationError):
    """A calibration map could not be loaded, saved or applied."""

    code = "CALIBRATION_MAP"
