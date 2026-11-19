"""Marker readout, the six micro-heads and the RAPPORT model assembly.

See ``model.py`` for the forward contract; ``readout.py`` and ``heads.py``
for the readout and the micro-heads; ``errors.py`` for the typed errors.
No ``src/model/__init__.py`` is created or touched — ``src.model`` stays a
namespace package, ``src.model.core`` (TASK-05) is imported unchanged.
"""

from .errors import (
    AttentionMaskError,
    GoalMarkerError,
    InvalidOptionCountError,
    InvalidTaskIdError,
    ModelError,
    OptionMaskError,
)
from .heads import ConfidenceHead, ProbabilityHead, RiskHead, confidence_features
from .model import (
    MAX_OPTIONS,
    MIN_OPTIONS,
    NUM_TASKS,
    RISK_CATEGORIES,
    RapportModel,
    RapportOutput,
)
from .readout import MASK_NEG, MarkerReadout, ReadoutOutput

__all__ = [
    "MAX_OPTIONS",
    "MIN_OPTIONS",
    "NUM_TASKS",
    "RISK_CATEGORIES",
    "MASK_NEG",
    "AttentionMaskError",
    "ConfidenceHead",
    "GoalMarkerError",
    "InvalidOptionCountError",
    "InvalidTaskIdError",
    "MarkerReadout",
    "ModelError",
    "OptionMaskError",
    "ProbabilityHead",
    "RapportModel",
    "RapportOutput",
    "ReadoutOutput",
    "RiskHead",
    "confidence_features",
]
