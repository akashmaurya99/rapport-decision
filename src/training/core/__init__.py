"""FP16 trainer core for RAPPORT — public surface (TASK-12)."""

from .checkpoint import (
    checkpoint_filename,
    checkpoint_path,
    latest_checkpoint,
    load_checkpoint_payload,
    save_checkpoint,
    sha256_file,
    sidecar_path,
    step_from_filename,
    verify_checkpoint,
)
from .config import TrainingConfig
from .errors import (
    CheckpointBeyondHorizonError,
    CheckpointError,
    CheckpointHashMismatchError,
    CheckpointNotFoundError,
    CheckpointSidecarMissingError,
    TrainerConfigError,
    TrainerError,
    TrainingDivergedError,
)
from .trainer import StepResult, Trainer, build_optimizer, build_scheduler

__all__ = [
    "CheckpointBeyondHorizonError",
    "CheckpointError",
    "CheckpointHashMismatchError",
    "CheckpointNotFoundError",
    "CheckpointSidecarMissingError",
    "StepResult",
    "Trainer",
    "TrainerConfigError",
    "TrainerError",
    "TrainingConfig",
    "TrainingDivergedError",
    "build_optimizer",
    "build_scheduler",
    "checkpoint_filename",
    "checkpoint_path",
    "latest_checkpoint",
    "load_checkpoint_payload",
    "save_checkpoint",
    "sha256_file",
    "sidecar_path",
    "step_from_filename",
    "verify_checkpoint",
]

__version__ = "0.1.0"
