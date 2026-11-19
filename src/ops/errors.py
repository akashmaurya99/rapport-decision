"""Typed errors raised by the ops module.

Every error carries a stable ``code`` attribute so callers (and tests) can
match on the failure class without parsing message text. The code is included
in ``str(error)`` for log readability. Mirrors the pattern in
``src/config/errors.py``; the ops module deliberately does not import the
config package so it stays usable in minimal environments.
"""

from __future__ import annotations


class OpsError(Exception):
    """Base class for errors raised by the ops module."""

    code = "OPS_ERROR"

    def __init__(self, message: str, *, key: str | None = None) -> None:
        super().__init__(f"{self.code}: {message}")
        self.message = message
        self.key = key


class JsonlRecordError(OpsError):
    """A JSONL record is missing required keys or carries invalid values.

    Raised *before* anything is written to disk, so a half record can never
    land in the log. ``key`` names the offending record key when applicable.
    """

    code = "OPS_JSONL_RECORD_INVALID"


class JsonlWriteError(OpsError):
    """The log file could not be written (directory missing, permission, ...)."""

    code = "OPS_JSONL_WRITE_FAILED"


class RunRegistryError(OpsError):
    """The run registry could not scan or index the runs directory."""

    code = "OPS_REGISTRY_ERROR"


class UnknownRunError(RunRegistryError):
    """A run id was requested that is not indexed (absent or excluded)."""

    code = "OPS_RUN_UNKNOWN"


class ReproductionError(OpsError):
    """The reproducibility check could not be executed as requested."""

    code = "OPS_REPRODUCTION_ERROR"


class ReproductionMismatchError(ReproductionError):
    """The reproduced metric does not match the recorded value within tolerance.

    Carries the full :class:`~src.ops.repro.ReproductionResult` as ``result``
    so callers can report the exact delta without re-running the workload.
    """

    code = "OPS_REPRODUCTION_MISMATCH"

    def __init__(self, message: str, *, result: object | None = None) -> None:
        super().__init__(message, key="loss")
        self.result = result
