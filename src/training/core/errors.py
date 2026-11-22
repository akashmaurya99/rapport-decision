"""Typed errors raised by the training core (TASK-12).

Every error carries a stable ``code`` attribute so callers (and tests) can
match on the failure class without parsing message text. The code is
included in ``str(error)`` for log readability, mirroring ``src/ops/errors.py``
and ``src/config/errors.py``.
"""

from __future__ import annotations


class TrainerError(Exception):
    """Base class for errors raised by the training core."""

    code = "TRAINER_ERROR"

    def __init__(self, message: str, *, key: str | None = None) -> None:
        super().__init__(f"{self.code}: {message}")
        self.message = message
        self.key = key


class TrainerConfigError(TrainerError):
    """The trainer configuration is missing, mistyped or out of range."""

    code = "TRAINER_CONFIG_INVALID"


class CheckpointError(TrainerError):
    """A checkpoint could not be written, located or restored."""

    code = "CHECKPOINT_ERROR"


class CheckpointNotFoundError(CheckpointError):
    """No usable checkpoint exists at the requested path or directory."""

    code = "CHECKPOINT_NOT_FOUND"


class CheckpointSidecarMissingError(CheckpointError):
    """The ``.sha256`` sidecar for a checkpoint file is absent.

    Without the sidecar there is nothing to verify the file against, so the
    checkpoint is refused rather than trusted blindly.
    """

    code = "CHECKPOINT_SIDECAR_MISSING"


class CheckpointHashMismatchError(CheckpointError):
    """The checkpoint file's sha256 does not match its ``.sha256`` sidecar.

    The file is corrupt or tampered with. The checkpoint is rejected and is
    **never loaded** — ``torch.load`` is not reached on this path.
    ``expected`` is the sidecar digest, ``actual`` the file's digest.
    """

    code = "CHECKPOINT_HASH_MISMATCH"

    def __init__(
        self,
        message: str,
        *,
        path: str,
        expected: str,
        actual: str,
    ) -> None:
        super().__init__(message, key="checkpoint")
        self.path = path
        self.expected = expected
        self.actual = actual


class CheckpointBeyondHorizonError(CheckpointError):
    """The checkpoint's step is at or beyond the configured step horizon.

    Refusing to resume past the planned budget keeps a stale or foreign
    checkpoint from silently extending a finished run. ``step`` is the
    checkpoint's step, ``steps`` the configured horizon
    (:attr:`TrainingConfig.steps`).
    """

    code = "CHECKPOINT_BEYOND_HORIZON"

    def __init__(self, message: str, *, step: int, steps: int) -> None:
        super().__init__(message, key="step")
        self.step = step
        self.steps = steps


class TrainingDivergedError(TrainerError):
    """Too many consecutive non-finite gradients in one epoch (E5).

    Raised when the streak of skipped (non-finite) steps exceeds
    ``nonfinite_per_epoch`` (default 10). Only a persistent streak is
    divergence: transient fp16 overflows that the GradScaler backs off from are
    cleared by the next successful optimizer step, so an otherwise healthy long
    run is not aborted by early recoverable skips.
    """

    code = "TRAINING_DIVERGED"

    def __init__(self, message: str, *, nonfinite_steps: int, epoch: int) -> None:
        super().__init__(message, key="nonfinite_steps")
        self.nonfinite_steps = nonfinite_steps
        self.epoch = epoch
