"""Typed errors raised by the distillation package (TASK-15).

Every error carries a stable ``code`` attribute so callers (and tests) can
match on the failure class without parsing message text, mirroring the
``src/training.losses`` and ``src/calibration.errors`` patterns. Codes are
part of the public contract: downstream stages (Stage C in ``src/training``)
may match on them, so a code is never renamed or reused.
"""

from __future__ import annotations


class DistillError(Exception):
    """Base class for errors raised by this package."""

    code = "DISTILL_ERROR"

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(f"{self.code}: {message}")
        self.message = message
        self.field = field


class DistillInputError(DistillError):
    """A caller passed a malformed argument (wrong type, empty, non-finite)."""

    code = "DISTILL_INPUT"


class TeacherUnavailableError(DistillError):
    """The local teacher could not be loaded or produced no usable output.

    Covers E8's fail-to-load limb: the teacher is expected to run locally
    inside the runtime (a one-time download, then fully offline). Any failure
    to construct the model, or any missing/unreadable cached soft labels, is
    surfaced through this class so the gate can record ``teacher_unavailable``
    instead of crashing the pipeline.
    """

    code = "DISTILL_TEACHER_UNAVAILABLE"


class GateInputError(DistillInputError):
    """The gate received metrics that are structurally invalid.

    A missing/unreadable *teacher* is NOT an error — it is a recorded negative
    result (see :func:`src.distill.gate.distillation_gate`). This error is
    reserved for malformed inputs: wrong types, non-finite values in the
    student metrics, or non-finite thresholds.
    """

    code = "DISTILL_GATE_INPUT"
