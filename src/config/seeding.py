"""Seed python, numpy, torch and CUDA, and pin down determinism.

FR-CONFIG-03: every run calls :func:`set_seed` once at startup. ``strict``
determinism additionally asks torch to use deterministic algorithms and sets
``CUBLAS_WORKSPACE_CONFIG`` (required by ``torch.use_deterministic_algorithms``
on CUDA). ``setdefault`` is used so an operator-provided value is never
silently overwritten.
"""

from __future__ import annotations

import os
import random

from .errors import ConfigValueError, DependencyMissingError

_VALID_DETERMINISM = frozenset({"strict", "relaxed"})
_MAX_SEED = 2**32 - 1


def set_seed(seed: int, determinism: str = "strict") -> None:
    """Seed python, numpy, torch and (when available) CUDA.

    Parameters
    ----------
    seed:
        Non-negative integer below 2**32.
    determinism:
        ``strict`` enables deterministic torch algorithms and disables the
        cudnn benchmark autotuner; ``relaxed`` seeds everything and explicitly
        restores deterministic algorithms, cudnn deterministic mode and the
        cudnn autotuner, so switching modes in-process takes effect.

    Raises
    ------
    ConfigValueError
        ``seed`` or ``determinism`` is outside the accepted domain.
    DependencyMissingError
        numpy or torch is not importable; the name of the missing package is
        carried in the error message.
    """
    if type(seed) is not int or not 0 <= seed <= _MAX_SEED:
        raise ConfigValueError(
            f"seed must be an int between 0 and {_MAX_SEED}, got {seed!r}", key="seed"
        )
    if determinism not in _VALID_DETERMINISM:
        raise ConfigValueError(
            f"determinism must be one of {sorted(_VALID_DETERMINISM)}, got {determinism!r}",
            key="determinism",
        )

    random.seed(seed)

    try:
        import numpy as np
    except ImportError as exc:
        raise DependencyMissingError(f"numpy is required to seed the run: {exc}") from exc
    np.random.seed(seed)

    try:
        import torch
    except ImportError as exc:
        raise DependencyMissingError(f"torch is required to seed the run: {exc}") from exc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_determinism(determinism)


def configure_determinism(determinism: str = "strict") -> None:
    """Apply process-level kernel settings without touching RNG streams.

    Resume restores RNG state from a checkpoint, but deterministic-kernel flags
    are process state and are not checkpointed. Keeping this separate prevents
    a resumed run from either losing strict determinism or being re-seeded.
    """
    if determinism not in _VALID_DETERMINISM:
        raise ConfigValueError(
            f"determinism must be one of {sorted(_VALID_DETERMINISM)}, got {determinism!r}",
            key="determinism",
        )
    try:
        import torch
    except ImportError as exc:
        raise DependencyMissingError(f"torch is required to configure determinism: {exc}") from exc
    if determinism == "strict" and torch.cuda.is_available():
        # Must be visible to cuBLAS before its handle is created; an
        # operator-provided value wins over the default.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    strict = determinism == "strict"
    if strict:
        # MPS has no deterministic index_put_ kernel; requiring it raises
        # RuntimeError mid-backward instead of degrading gracefully. Request
        # determinism with warn_only there; CPU and CUDA keep the hard
        # guarantee.
        mps_only = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and not torch.cuda.is_available()
        )
        torch.use_deterministic_algorithms(True, warn_only=bool(mps_only))
    else:
        torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = strict
    torch.backends.cudnn.benchmark = not strict
