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
        cudnn benchmark autotuner; ``relaxed`` seeds everything but allows
        non-deterministic kernels.

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
        if determinism == "strict":
            # Must be visible to cuBLAS before its handle is created; an
            # operator-provided value wins over the default.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if determinism == "strict":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
