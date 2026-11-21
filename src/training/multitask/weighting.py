"""Kendall-style uncertainty weighting with a floor, a cap and collapse detection.

Design authority: ``.archgen/rapport/docs/training-strategy.md`` §3.2 and
``.archgen/rapport/docs/evaluation.md`` §5 (ablation A4). See the package
docstring for the full contract.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "COLLAPSE_FLOOR_EPOCHS",
    "DEFAULT_TASK_NAMES",
    "WEIGHT_COLLAPSE_EVENT",
    "WEIGHT_LOG_EVENT",
    "TaskWeighter",
]

#: Canonical task order, matching ``src.training.losses`` term names.
DEFAULT_TASK_NAMES: tuple[str, ...] = (
    "choice",
    "preference",
    "score",
    "probability",
    "confidence",
    "abstain",
    "risk",
)

#: ``event`` value for the per-epoch weight record.
WEIGHT_LOG_EVENT = "task_weights"

#: ``event`` value for the collapse record surfaced by this module.
WEIGHT_COLLAPSE_EVENT = "WEIGHT_COLLAPSE"

#: Consecutive floor epochs before a task is surfaced as WEIGHT_COLLAPSE.
COLLAPSE_FLOOR_EPOCHS = 2

# Tolerance for "at the floor"; exp(-s_max) can land 1 ulp above 0.05.
_FLOOR_EPS = 1e-6

_FLAT_SCALAR_TYPES = (str, int, float, bool, type(None))


def _is_flat_scalar(value: Any) -> bool:
    if not isinstance(value, _FLAT_SCALAR_TYPES):
        return False
    return not (isinstance(value, float) and not math.isfinite(value))


class TaskWeighter(nn.Module):
    """Learn per-task loss weights with a floor and a cap (Kendall et al. 2018).

    Parameters
    ----------
    num_tasks:
        Number of tasks; integer >= 1 (booleans rejected).
    floor:
        Lower bound on each resolved weight ``w_t = exp(-s_t)``. Default 0.05.
    cap:
        Upper bound on each resolved weight. Default 4.0. Must be >= ``floor``;
        both must be finite with ``floor > 0``.
    init_s:
        Initial value(s) for the per-task log-variance ``s_t``. A single float
        (default ``0.0``, i.e. all weights start at 1.0) or a sequence of
        ``num_tasks`` finite floats. Seeded onto ``log_s`` in order.
    task_names:
        Optional per-task names. Defaults to :data:`DEFAULT_TASK_NAMES` when
        ``num_tasks == 7``, else ``task_0 … task_{n-1}``. Names must be
        unique, non-empty strings.

    Notes
    -----
    ``s_t`` is stored as the ``log_s`` parameter and clamped internally to
    ``[-log(cap), -log(floor)]`` so ``exp(-s_t)`` always lies in
    ``[floor, cap]``. The clamp saturates: at the boundary the weight is
    constant and the weighting term passes no gradient. This is intentional —
    it is the guard that makes a collapse detectable instead of silently
    zeroing a task.

    Collapse tracking state (``_floor_streak`` / emitted events) is runtime
    state, not checkpoint state; it is not part of ``state_dict()``.
    """

    def __init__(
        self,
        num_tasks: int,
        floor: float = 0.05,
        cap: float = 4.0,
        init_s: float | Sequence[float] = 0.0,
        task_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(num_tasks, bool) or not isinstance(num_tasks, int):
            raise ValueError(f"num_tasks must be an integer, got {num_tasks!r}")
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be >= 1, got {num_tasks}")
        if isinstance(floor, bool) or not isinstance(floor, (int, float)):
            raise ValueError(f"floor must be a real number, got {floor!r}")
        if isinstance(cap, bool) or not isinstance(cap, (int, float)):
            raise ValueError(f"cap must be a real number, got {cap!r}")
        floor_f = float(floor)
        cap_f = float(cap)
        if not math.isfinite(floor_f) or not math.isfinite(cap_f):
            raise ValueError(f"floor and cap must be finite, got floor={floor_f}, cap={cap_f}")
        if floor_f <= 0.0:
            raise ValueError(f"floor must be > 0, got {floor_f}")
        if cap_f < floor_f:
            raise ValueError(f"cap must be >= floor, got cap={cap_f}, floor={floor_f}")

        if task_names is None:
            if num_tasks == len(DEFAULT_TASK_NAMES):
                names = DEFAULT_TASK_NAMES
            else:
                names = tuple(f"task_{i}" for i in range(num_tasks))
        else:
            if len(task_names) != num_tasks:
                raise ValueError(
                    f"task_names must have num_tasks={num_tasks} entries, got {len(task_names)}"
                )
            names = tuple(task_names)
            for name in names:
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"task names must be non-empty strings, got {name!r}")
            if len(set(names)) != len(names):
                raise ValueError(f"task_names must be unique, got {names}")

        if isinstance(init_s, bool):
            raise ValueError(f"init_s must be a float or a sequence of floats, got {init_s!r}")
        if isinstance(init_s, (int, float)):
            init_values = [float(init_s)] * num_tasks
        else:
            init_values = [float(v) for v in init_s]
            if len(init_values) != num_tasks:
                raise ValueError(
                    f"init_s must have num_tasks={num_tasks} entries, got {len(init_values)}"
                )
        for value in init_values:
            if not math.isfinite(value):
                raise ValueError(f"init_s entries must be finite, got {value!r}")

        self.floor = floor_f
        self.cap = cap_f
        self.num_tasks = num_tasks
        self.task_names = names
        # s bounds keeping exp(-s) inside [floor, cap].
        self._s_min = -math.log(cap_f)
        self._s_max = -math.log(floor_f)

        self.log_s = nn.Parameter(torch.tensor(init_values, dtype=torch.float32))

        self._floor_streak: dict[str, int] = {name: 0 for name in names}
        self._collapse_events: list[dict[str, Any]] = []

    # ------------------------------------------------------------- weights
    def _clamped_s(self) -> Tensor:
        return self.log_s.clamp(self._s_min, self._s_max)

    def weight_tensor(self) -> Tensor:
        """Resolved weights as a tensor (differentiable, clamped)."""
        return torch.exp(-self._clamped_s())

    def weights(self) -> dict[str, float]:
        """Detached resolved weights, task name → float, clamped to [floor, cap]."""
        with torch.no_grad():
            resolved = self.weight_tensor().tolist()
        return {name: float(value) for name, value in zip(self.task_names, resolved)}

    def forward(
        self,
        losses: Mapping[str, Tensor] | Sequence[Tensor],
        active: Mapping[str, bool] | None = None,
    ) -> Tensor:
        """Return ``Σ_t w_t · L_t + s_t`` over the supplied per-task losses.

        ``losses`` may be a mapping (every task name must be present; unknown
        keys are rejected) or a sequence ordered like ``task_names``. Each
        entry must be a finite scalar tensor. ``active`` optionally marks
        tasks inactive; an inactive task contributes ``0`` for its weighted
        term (its ``s_t`` regulariser is still applied, matching Kendall et
        al.). Inactive therefore means "no data this step", not "excluded
        from weighting".
        """
        return self.weighted_sum(losses, active) + self._clamped_s().sum()

    # Public compatibility alias: identical to forward(), including the Kendall
    # regulariser. Do not replace it with weighted_sum(), which omits Σs.
    loss = forward

    def weighted_sum(
        self,
        losses: Mapping[str, Tensor] | Sequence[Tensor],
        active: Mapping[str, bool] | None = None,
    ) -> Tensor:
        """Return ``Σ_t w_t · L_t`` over active tasks, without the ``s_t`` term.

        This is the fixed / uniform arm's weighted total: the Kendall
        regulariser ``Σ_t s_t`` is deliberately excluded so the resolved
        fixed weights reproduce a plain weighted sum exactly.
        """
        per_task = self._resolve_losses(losses)
        if active is not None:
            for key in active:
                if key not in self.task_names:
                    raise ValueError(f"active contains unknown task {key!r}")
        clamped_s = self._clamped_s()
        total = clamped_s.new_zeros(())
        for index, name in enumerate(self.task_names):
            if active is not None and not bool(active.get(name, True)):
                continue
            total = total + torch.exp(-clamped_s[index]) * per_task[index]
        return total

    def _resolve_losses(self, losses: Mapping[str, Tensor] | Sequence[Tensor]) -> list[Tensor]:
        if isinstance(losses, Mapping):
            missing = [name for name in self.task_names if name not in losses]
            if missing:
                raise ValueError(f"losses is missing task(s): {', '.join(missing)}")
            unknown = [key for key in losses if key not in self.task_names]
            if unknown:
                raise ValueError(f"losses has unknown task(s): {', '.join(unknown)}")
            ordered = [losses[name] for name in self.task_names]
        elif isinstance(losses, Sequence) and not isinstance(losses, (str, bytes)):
            if len(losses) != self.num_tasks:
                raise ValueError(
                    f"losses sequence must have num_tasks={self.num_tasks} entries, "
                    f"got {len(losses)}"
                )
            ordered = list(losses)
        else:
            raise TypeError(f"losses must be a mapping or a sequence, got {type(losses).__name__}")
        resolved: list[Tensor] = []
        for index, (name, value) in enumerate(zip(self.task_names, ordered)):
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"losses[{name!r}] must be a Tensor, got {type(value).__name__}"
                )
            tensor = value.float()
            if tensor.numel() != 1:
                raise ValueError(
                    f"losses[{name!r}] must be scalar, got shape {tuple(tensor.shape)}"
                )
            resolved.append(tensor.reshape(()))
        return resolved

    # --------------------------------------------------- collapses / streaks
    def _update_collapse_tracking(
        self,
        resolved_weights: Mapping[str, float],
        *,
        epoch: int,
        logger: Any | None,
        split: str,
    ) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        for name in self.task_names:
            at_floor = resolved_weights[name] <= self.floor + _FLOOR_EPS
            if at_floor:
                self._floor_streak[name] += 1
            else:
                self._floor_streak[name] = 0
            streak = self._floor_streak[name]
            if streak != COLLAPSE_FLOOR_EPOCHS:
                continue
            event = {
                "event": WEIGHT_COLLAPSE_EVENT,
                "task": name,
                "consecutive_floor_epochs": streak,
            }
            self._collapse_events.append(dict(event))
            emitted.append(event)
            if logger is not None:
                logger.log(
                    event=WEIGHT_COLLAPSE_EVENT,
                    step=epoch,
                    split=split,
                    metrics={"epoch": epoch, "consecutive_floor_epochs": streak},
                    task=name,
                    consecutive_floor_epochs=streak,
                )
        return emitted

    def collapse_events(self) -> list[dict[str, Any]]:
        """Return copies of every WEIGHT_COLLAPSE event emitted so far.

        Each event is exactly ``{"event": "WEIGHT_COLLAPSE", "task": <task>,
        "consecutive_floor_epochs": 2}``. Emission happens inside
        :meth:`log_epoch`; when no logger was supplied, events are still
        tracked here but not JSONL-written anywhere.
        """
        return [dict(event) for event in self._collapse_events]

    def floor_streaks(self) -> dict[str, int]:
        """Current consecutive-floor-epoch streak per task (0 when off the floor)."""
        return dict(self._floor_streak)

    # ------------------------------------------------------------ logging
    def log_epoch(
        self,
        logger: Any | None,
        epoch: int,
        metrics: Mapping[str, Any] | None = None,
        *,
        split: str = "train",
        step: int | None = None,
    ) -> dict[str, Any] | None:
        """Emit one ``task_weights`` record for ``epoch`` and update collapse tracking.

        ``metrics`` may carry:

        * ``"active"`` — mapping task → bool; merged into the record as the
          per-task active flag. Defaults to all-True.
        * ``"losses"`` — mapping task → finite float; logged as
          ``loss_<task>`` entries in the flat metrics block.
        * any other flat scalar entry — merged verbatim into the metrics block.

        Nested or non-scalar leftover values are rejected with ``ValueError``
        before anything is logged. The record returned by ``logger.log`` is
        returned here; when ``logger`` is ``None`` nothing is written and this
        method returns ``None`` (collapse tracking still runs, so
        :meth:`collapse_events` keeps working with no logger attached).
        """
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise ValueError(f"epoch must be an integer >= 0, got {epoch!r}")
        if epoch < 0:
            raise ValueError(f"epoch must be an integer >= 0, got {epoch}")
        if step is None:
            step = epoch
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"step must be an integer >= 0, got {step!r}")
        if not isinstance(split, str) or not split.strip():
            raise ValueError(f"split must be a non-empty string, got {split!r}")
        if logger is not None and not hasattr(logger, "log"):
            raise TypeError(
                "logger must provide a log(...) method (e.g. src.ops.logging.JsonlLogger)"
            )

        active_flags: dict[str, bool] = {name: True for name in self.task_names}
        payload: dict[str, Any] = {"epoch": epoch, "num_tasks": self.num_tasks}
        if metrics is not None:
            if not isinstance(metrics, Mapping):
                raise TypeError(f"metrics must be a mapping, got {type(metrics).__name__}")
            active_value = metrics.get("active")
            if active_value is not None:
                if not isinstance(active_value, Mapping):
                    raise ValueError(f"metrics['active'] must be a mapping, got {active_value!r}")
                for name, flag in active_value.items():
                    if name not in self.task_names:
                        raise ValueError(f"metrics['active'] has unknown task {name!r}")
                    active_flags[name] = bool(flag)
            losses_value = metrics.get("losses")
            if losses_value is not None:
                if not isinstance(losses_value, Mapping):
                    raise ValueError(f"metrics['losses'] must be a mapping, got {losses_value!r}")
                for name, value in losses_value.items():
                    if name not in self.task_names:
                        raise ValueError(f"metrics['losses'] has unknown task {name!r}")
                    if not _is_flat_scalar(value):
                        raise ValueError(
                            f"metrics['losses'][{name!r}] must be a flat scalar, got {value!r}"
                        )
                    payload[f"loss_{name}"] = float(value)
            for key, value in metrics.items():
                if key in ("active", "losses"):
                    continue
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(f"metrics keys must be non-empty strings, got {key!r}")
                if not _is_flat_scalar(value):
                    raise ValueError(
                        f"metrics[{key!r}] must be a flat JSON scalar, got {value!r}"
                    )
                payload[key] = value

        resolved = self.weights()
        record = None
        if logger is not None:
            record = logger.log(
                event=WEIGHT_LOG_EVENT,
                step=step,
                split=split,
                metrics=payload,
                weights=resolved,
                active=active_flags,
            )
        self._update_collapse_tracking(resolved, epoch=epoch, logger=logger, split=split)
        return record

    # ----------------------------------------------------------- manifest
    def to_manifest_fragment(self) -> dict[str, Any]:
        """Return the manifest fragment TASK-01's builder embeds (FR-TRAIN-03).

        Shape::

            {
                "task_weight_mode": "learned",
                "task_weight_floor": 0.05,
                "task_weight_cap": 4.0,
                "task_weights": {"choice": 1.0, ...},
            }

        All values are JSON scalars / flat dicts so the fragment serializes
        cleanly into ``manifest.json``; once embedded, ``RunRegistry`` surfaces
        it under ``registry.get(run_id).manifest["task_weights"]``. This method
        never touches the filesystem — embedding is the manifest builder's job.
        """
        return {
            "task_weight_mode": "learned",
            "task_weight_floor": self.floor,
            "task_weight_cap": self.cap,
            "task_weights": self.weights(),
        }
