"""Structured logging, run registry and reproducibility checks for RAPPORT.

Public surface
--------------
* :class:`JsonlLogger` — one validated JSON object per line (FR-OPS-01).
* :class:`RunRegistry` — indexes every valid run under a runs directory by
  ``run_id``; invalid runs are excluded and counted, never silently listed
  (FR-OPS-02).
* :func:`verify_resume` — fixed-seed, bounded replay of a recorded tiny
  deterministic workload; asserts the reproduced loss matches the recorded
  value within 1e-4 (FR-OPS-03). Verifies harness determinism, not a network.
"""

from .errors import (
    JsonlRecordError,
    JsonlWriteError,
    OpsError,
    ReproductionError,
    ReproductionMismatchError,
    RunRegistryError,
    UnknownRunError,
)
from .logging import (
    METRIC_KEYS,
    REQUIRED_RECORD_KEYS,
    VALID_SPLITS,
    JsonlLogger,
    build_record,
    iter_records,
)
from .registry import (
    CHECKPOINT_SUFFIXES,
    REQUIRED_MANIFEST_FIELDS,
    Exclusion,
    RegistryEntry,
    RegistryScan,
    RunRegistry,
)
from .repro import (
    DEFAULT_REPLAY_STEPS,
    DEFAULT_TOLERANCE,
    ReproductionResult,
    run_reference_workload,
    verify_resume,
)

__all__ = [
    "CHECKPOINT_SUFFIXES",
    "DEFAULT_REPLAY_STEPS",
    "DEFAULT_TOLERANCE",
    "METRIC_KEYS",
    "REQUIRED_MANIFEST_FIELDS",
    "REQUIRED_RECORD_KEYS",
    "VALID_SPLITS",
    "Exclusion",
    "JsonlLogger",
    "JsonlRecordError",
    "JsonlWriteError",
    "OpsError",
    "RegistryEntry",
    "RegistryScan",
    "ReproductionError",
    "ReproductionMismatchError",
    "ReproductionResult",
    "RunRegistry",
    "RunRegistryError",
    "UnknownRunError",
    "build_record",
    "iter_records",
    "run_reference_workload",
    "verify_resume",
]

__version__ = "0.1.0"
