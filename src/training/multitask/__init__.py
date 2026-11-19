"""Adaptive multi-task weighting and per-task logging (TASK-14).

Implements the Kendall et al. (2018) uncertainty weighting scheme from
``.archgen/rapport/docs/training-strategy.md`` §3.2:

.. code-block:: text

    L = Σ_t  w_t · L_t + s_t        with  w_t = exp(-s_t)
        floor ≤ w_t ≤ cap           (default floor=0.05, cap=4.0)

where ``s_t`` is a learned per-task log-variance (``log_s_t``, an
:class:`torch.nn.Parameter` trained with the rest of the model). ``s_t`` is
clamped internally so ``exp(-s_t)`` always stays inside ``[floor, cap]``;
because the clamp saturates, a task pinned at the floor contributes a
constant weight and zero gradient through the weighting term — that is the
known uncertainty-weighting failure mode this module is built to surface.

Surfacing collapse (FR-TRAIN-03)
--------------------------------
:meth:`TaskWeighter.log_epoch` writes one ``event="task_weights"`` JSONL
record through :class:`src.ops.logging.JsonlLogger` every epoch. The record
carries, as top-level keys, the resolved per-task weight vector
(``weights``: task → float), the per-task active flag (``active``: task →
bool) and the epoch number; the flat ``metrics`` block carries
``epoch``/``num_tasks`` plus any scalar metrics the caller passes (e.g.
per-task losses as ``loss_<task>``).

Two consecutive epochs at the floor for the same task emit a
``WEIGHT_COLLAPSE`` event. The event is a structured dict::

    {"event": "WEIGHT_COLLAPSE", "task": <task>, "consecutive_floor_epochs": 2}

returned by :meth:`TaskWeighter.collapse_events` (and every emission is also
JSONL-written via the supplied logger as an ``event="WEIGHT_COLLAPSE"``
record). The streak resets as soon as the task leaves the floor; a fresh
collapse later re-emits.

Manifest / run registry (FR-TRAIN-03)
-------------------------------------
:meth:`TaskWeighter.to_manifest_fragment` returns the fragment TASK-01's
manifest builder embeds into ``runs/<run_id>/manifest.json``. Once embedded,
:class:`src.ops.registry.RunRegistry` indexes the fragment with the rest of
the manifest, so ``registry.get(run_id).manifest["task_weights"]`` yields the
resolved per-task weight vector for the run. This package never writes the
manifest itself (``src/config/**`` is TASK-01's file) and never mutates the
registry (it is a read-only scanner); both go through those components'
public contracts.

Fixed-weight arm (ablation A4)
------------------------------
:func:`build_fixed_weight_config` returns the fixed-weight arm definition
(uniform weights 1.0 across tasks) consistent with
``configs/experiments/a4-weights.yaml`` (the A4 ablation: "does uncertainty
weighting collapse a task?"). No file under ``configs/`` is written; the arm
is registered in code and consistency is covered by tests.
"""

from .arms import FIXED_WEIGHT_ARM, build_fixed_weight_config
from .weighting import (
    COLLAPSE_FLOOR_EPOCHS,
    DEFAULT_TASK_NAMES,
    WEIGHT_COLLAPSE_EVENT,
    WEIGHT_LOG_EVENT,
    TaskWeighter,
)

__all__ = [
    "COLLAPSE_FLOOR_EPOCHS",
    "DEFAULT_TASK_NAMES",
    "FIXED_WEIGHT_ARM",
    "TaskWeighter",
    "WEIGHT_COLLAPSE_EVENT",
    "WEIGHT_LOG_EVENT",
    "build_fixed_weight_config",
]

__version__ = "0.1.0"
