"""FP16 trainer core for RAPPORT (TASK-12).

One process. Design authorities:

* ``.archgen/rapport/docs/training-strategy.md`` §3.1 (AdamW, 5% linear
  warmup then cosine to 1e-5, global-norm clip 1.0, weight decay excluded on
  norms/biases/embeddings) and §8 (500-step cadence, epoch-end checkpoint,
  contents: model/optimizer/scaler/scheduler/step/epoch/sampler offset/RNG
  states of python+numpy+torch+CUDA, checkpoint every N steps as a config
  key);
* ``.archgen/rapport/docs/t4-optimization.md`` §2 (FP16 autocast, FP32 loss,
  GradScaler init 2**16 / backoff 0.5 / growth interval 2000, non-finite
  gradients skip the step and are counted — never corrupt weights).

Non-finite handling (E5 / FR-TRAIN-04)
--------------------------------------
A non-finite value is detected at two points:

1. ``LossNotFiniteError`` from
   :func:`src.training.losses.compute_losses` — the loss never reaches the
   optimizer at all.
2. A non-finite gradient detected after ``scaler.unscale_`` and before
   clipping — the optimizer step is skipped and the scaler backs off.

Either way: ``nonfinite_steps`` (total) and ``nonfinite_steps_epoch``
(per epoch) are incremented, the global ``step`` counter does **not**
advance, the scheduler is not stepped, and the batch is still consumed from
the sampler (``sampler_offset`` advances). More than
``config.nonfinite_per_epoch`` (default 10) non-finite steps in one epoch
raises :class:`TrainingDivergedError`.

Resume contract
---------------
:meth:`Trainer.save` writes ``checkpoint_step{N}.pt`` atomically
(``*.tmp`` → fsync → rename → fsync(dir)) plus a ``.sha256`` sidecar.
:meth:`Trainer.load` verifies the sidecar against the file **before**
deserializing — a mismatch raises :class:`CheckpointHashMismatchError` and
the payload is never loaded — and refuses a checkpoint whose step is at or
beyond ``config.steps`` with :class:`CheckpointBeyondHorizonError`. All
state (model, optimizer, scaler, scheduler, step, epoch, sampler offset,
counters, python/numpy/torch/CUDA RNG) is restored, so the resumed run
follows the same loss trajectory as the uninterrupted one.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from src.config.seeding import configure_determinism, set_seed
from src.ops.logging import JsonlLogger
from src.training.losses import DEFAULT_WEIGHTS, LossNotFiniteError, compute_losses
from src.training.multitask import DEFAULT_TASK_NAMES, TaskWeighter

from .checkpoint import (
    latest_checkpoint,
    load_checkpoint_payload,
    save_checkpoint,
    sidecar_path,
    step_from_filename,
)
from .config import TrainingConfig, _validate
from .errors import (
    CheckpointBeyondHorizonError,
    CheckpointError,
    CheckpointNotFoundError,
    TrainerError,
    TrainingDivergedError,
)

__all__ = [
    "StepResult",
    "Trainer",
    "build_optimizer",
    "build_scheduler",
]

_MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "option_mask", "task_ids")
_LOSS_KEYS = (
    "chosen_index",
    "preference_pairs",
    "score_targets",
    "score_mask",
    "probability_target",
    "probability_mask",
    "answerable_target",
    "answerable_mask",
    "risk_targets",
    "risk_mask",
    "risk_pos_weight",
)
_NO_DECAY_TOKENS = ("bias", "norm", "embedding", "task_vector", "ln_", "rmsnorm")


def _is_head_param(name: str) -> bool:
    """True for readout / head parameters, false for trunk (encoder) parameters.

    The production model names them ``readout.*``, ``probability_head.*``,
    ``confidence_head.*`` and ``risk_head.*``; generic ``head.*`` and
    ``*.head.*`` layouts are accepted too.
    """
    lowered = name.lower()
    if lowered.startswith("head") or lowered.startswith("readout"):
        return True
    parts = lowered.split(".")
    return "head" in parts or any(part.endswith("_head") for part in parts)


@dataclass(frozen=True)
class StepResult:
    """Outcome of one consumed batch."""

    step: int
    epoch: int
    loss: float | None
    skipped: bool
    reason: str | None
    sampler_offset: int
    nonfinite_steps: int
    grad_norm: float | None = None
    active: Mapping[str, bool] | None = None


def build_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    *,
    weighter: TaskWeighter | None = None,
) -> torch.optim.AdamW:
    """AdamW with no decay on norms/biases/embeddings and an optional head LR.

    Parameters are bucketed by ``(is_head, no_decay)``. Head/readout buckets
    get ``config.head_lr`` when set (otherwise the encoder LR); norm/bias and
    other no-decay parameters always get weight decay 0. A trainable
    ``TaskWeighter.log_s`` (learned arm) joins the no-decay body group.
    """
    named: list[tuple[str, nn.Parameter]] = list(model.named_parameters())
    if weighter is not None:
        named.extend(
            (f"weighter.{name}", param)
            for name, param in weighter.named_parameters()
            if param.requires_grad
        )
    buckets: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, param in named:
        if not param.requires_grad:
            continue
        lowered = name.lower()
        no_decay = param.ndim < 2 or any(token in lowered for token in _NO_DECAY_TOKENS)
        buckets.setdefault((_is_head_param(name), no_decay), []).append(param)
    groups: list[dict[str, Any]] = []
    for (is_head, no_decay), params in buckets.items():
        groups.append(
            {
                "params": params,
                "lr": config.head_lr if (is_head and config.head_lr is not None)
                else config.learning_rate,
                "weight_decay": 0.0 if no_decay else config.weight_decay,
            }
        )
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    """5% linear warmup then cosine decay to ``min_learning_rate`` (§3.1)."""
    total = max(1, config.steps)
    warmup = max(1, int(total * config.warmup_ratio))
    min_ratio = config.min_learning_rate / config.learning_rate

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _build_weighter(config: TrainingConfig) -> TaskWeighter:
    """Build the A4 weighting arm for this config.

    * ``learned`` — ``log_s`` is trainable and starts at 0 (all weights 1.0);
      ``TaskWeighter.forward`` adds the Kendall ``Σ s_t`` regulariser.
    * ``fixed`` — ``log_s = -log(DEFAULT_WEIGHTS)`` is frozen, reproducing the
      §3.2 default weighted sum exactly (backward-compatible default).
    * ``uniform`` — ``log_s = 0`` is frozen, so every task has weight 1.0.
    """
    if config.weight_mode == "learned":
        return TaskWeighter(num_tasks=len(DEFAULT_TASK_NAMES))
    initial = (
        [-math.log(DEFAULT_WEIGHTS[name]) for name in DEFAULT_TASK_NAMES]
        if config.weight_mode == "fixed"
        else [0.0] * len(DEFAULT_TASK_NAMES)
    )
    weighter = TaskWeighter(num_tasks=len(DEFAULT_TASK_NAMES), init_s=initial)
    weighter.log_s.requires_grad_(False)
    return weighter


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": [],
    }
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:  # pragma: no cover - best effort on odd drivers
            state["cuda"] = []
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(tuple(numpy_state))  # type: ignore[arg-type]
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_states = state.get("cuda") or []
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(cuda_states))


class Trainer:
    """One-process FP16 trainer with checkpointed resume.

    Parameters
    ----------
    config:
        A :class:`TrainingConfig` (or a mapping to build one via
        :meth:`TrainingConfig.from_mapping`).
    model:
        The module to train (RAPPORT's ``RapportModel`` in production; a
        stub satisfying the same interface in unit tests). May be omitted at
        construction and attached later via ``attach``; :meth:`fit` and
        :meth:`train_batch` require an attached model.
    optimizer, scheduler, scaler:
        Optional pre-built objects; when omitted they are built from the
        config and the attached model.
    logger:
        Optional :class:`src.ops.logging.JsonlLogger`. When absent, logging
        is a no-op, but the step counter, the hooks and one JSONL record per
        step still exist whenever a logger is attached.
    sampler:
        Optional object carrying an integer ``offset`` attribute (the
        deterministic sampler's position). The trainer reads and advances it
        and round-trips it through checkpoints. Without one, the trainer
        keeps an internal counter.
    """

    def __init__(
        self,
        config: TrainingConfig | Mapping[str, Any],
        model: nn.Module | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scaler: torch.amp.GradScaler | None = None,
        logger: JsonlLogger | None = None,
        sampler: Any | None = None,
    ) -> None:
        if isinstance(config, TrainingConfig):
            _validate(config)
            self.config = config
        else:
            self.config = TrainingConfig.from_mapping(config)
        self.device = self.config.resolved_device()
        self.sampler = sampler
        self.logger = logger

        self.step = 0
        self.epoch = 0
        self._sampler_offset = 0
        self.nonfinite_steps = 0
        self.nonfinite_steps_epoch = 0

        self.model: nn.Module | None = None
        self.weighter: TaskWeighter | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.scaler: torch.amp.GradScaler | None = None
        if model is not None:
            self.attach(model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)

    # ------------------------------------------------------------------ wiring
    def attach(
        self,
        model: nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> None:
        """Attach ``model`` and build (or adopt) optimizer/scheduler/scaler."""
        self.model = model
        self.model.to(self.device)
        self.weighter = _build_weighter(self.config)
        self.weighter.to(self.device)
        if optimizer is not None and self.weighter.log_s.requires_grad:
            # A caller-supplied optimizer must own every trainable parameter.
            # Silently omitting log_s both freezes learned weighting and leaves
            # its gradients outside optimizer.zero_grad().
            if not any(
                parameter is self.weighter.log_s
                for group in optimizer.param_groups
                for parameter in group["params"]
            ):
                optimizer.add_param_group(
                    {
                        "params": [self.weighter.log_s],
                        "lr": self.config.learning_rate,
                        "weight_decay": 0.0,
                    }
                )
        self.optimizer = (
            optimizer
            if optimizer is not None
            else build_optimizer(model, self.config, weighter=self.weighter)
        )
        self.scheduler = (
            scheduler if scheduler is not None else build_scheduler(self.optimizer, self.config)
        )
        if scaler is not None:
            self.scaler = scaler
        else:
            scaler_device = "cuda" if self.device == "cuda" else "cpu"
            self.scaler = torch.amp.GradScaler(
                scaler_device,
                init_scale=self.config.scaler_init_scale,
                growth_factor=self.config.scaler_growth_factor,
                backoff_factor=self.config.scaler_backoff_factor,
                growth_interval=self.config.scaler_growth_interval,
                enabled=self.config.fp16,
            )

    def _require_model(self) -> nn.Module:
        if self.model is None or self.optimizer is None:
            raise TrainerError("no model attached; pass model= to Trainer() or call attach()")
        return self.model

    @property
    def sampler_offset(self) -> int:
        sampler_offset = getattr(self.sampler, "offset", None) if self.sampler is not None else None
        if isinstance(sampler_offset, int) and not isinstance(sampler_offset, bool):
            return sampler_offset
        return self._sampler_offset

    def _consume_sampler(self) -> None:
        self._sampler_offset += 1
        if self.sampler is not None and hasattr(self.sampler, "offset"):
            try:
                self.sampler.offset = self._sampler_offset
            except Exception:
                # A read-only sampler still gets its internal mirror advanced.
                pass

    # ------------------------------------------------------------------ steps
    def _register_nonfinite(self, reason: str) -> None:
        self.nonfinite_steps += 1
        self.nonfinite_steps_epoch += 1
        if self.nonfinite_steps_epoch > self.config.nonfinite_per_epoch:
            raise TrainingDivergedError(
                f"{self.nonfinite_steps_epoch} non-finite steps in epoch {self.epoch} "
                f"(limit {self.config.nonfinite_per_epoch}); last reason: {reason}",
                nonfinite_steps=self.nonfinite_steps_epoch,
                epoch=self.epoch,
            )

    def _weighted_total(self, breakdown: Any) -> torch.Tensor:
        """Resolve the A4 arm's weighted total from the loss breakdown.

        ``fixed`` keeps the exact §3.2 weighted sum; ``uniform`` uses the
        weighter's equal-weight sum; ``learned`` uses the weighter's Kendall
        objective (weighted sum plus the ``Σ s_t`` regulariser), whose ``s_t``
        gradient flows to ``log_s`` in the optimizer's no-decay group.
        """
        if self.weighter is None:
            return breakdown.total
        if self.config.weight_mode == "fixed":
            return breakdown.total
        terms = {name: getattr(breakdown, name) for name in DEFAULT_TASK_NAMES}
        if self.config.weight_mode == "learned":
            return self.weighter(terms, breakdown.active)
        return self.weighter.weighted_sum(terms, breakdown.active)

    def _emit_step_record(
        self,
        *,
        loss: float | None,
        skipped: bool,
        reason: str | None,
        grad_norm: float | None,
    ) -> None:
        if self.logger is None:
            return
        metrics: dict[str, Any] = {
            "skipped": skipped,
            "nonfinite_steps": self.nonfinite_steps,
            "sampler_offset": self.sampler_offset,
        }
        if loss is not None and math.isfinite(loss):
            metrics["loss"] = loss
        if grad_norm is not None and math.isfinite(grad_norm):
            metrics["grad_norm"] = grad_norm
        if reason is not None:
            metrics["reason"] = reason
        self.logger.log_step(self.step, metrics, split="train", epoch=self.epoch)

    def train_batch(self, batch: Mapping[str, Any]) -> StepResult:
        """Consume one batch; returns a :class:`StepResult`.

        The batch is a mapping carrying the model inputs
        (``input_ids``, ``attention_mask``, ``option_mask``, ``task_ids``)
        plus the optional :func:`compute_losses` label kwargs
        (``chosen_index``, ``preference_pairs``, ...).
        """
        if not isinstance(batch, Mapping):
            raise TrainerError(f"batch must be a mapping, got {type(batch).__name__}")
        model = self._require_model()
        assert self.optimizer is not None and self.scaler is not None and self.scheduler is not None

        self._consume_sampler()
        model.train()
        # Clear gradients through the OPTIMIZER, not just model.parameters():
        # in learned weighting mode log_s is a trainable parameter owned by the
        # optimizer but not by the model. model.zero_grad() would leave its
        # gradient accumulating across steps, making weight updates depend on
        # step history and breaking kill/resume trajectory reproducibility.
        self.optimizer.zero_grad(set_to_none=True)

        model_kwargs = {key: batch[key] for key in _MODEL_INPUT_KEYS if key in batch}
        if "input_ids" not in model_kwargs:
            raise TrainerError("batch is missing 'input_ids'")
        if "option_mask" not in batch:
            raise TrainerError("batch is missing 'option_mask'")
        label_kwargs = {key: batch[key] for key in _LOSS_KEYS if key in batch}

        # DataLoaders conventionally produce CPU tensors. Model parameters and
        # every label/index tensor must agree with the trainer device; moving
        # only model inputs still breaks advanced indexing in choice/preference
        # losses. Preserve non-tensor metadata and container structure.
        def on_device(value: Any) -> Any:
            if torch.is_tensor(value):
                return value.to(self.device)
            if isinstance(value, Mapping):
                return {key: on_device(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(on_device(item) for item in value)
            if isinstance(value, list):
                return [on_device(item) for item in value]
            return value

        model_kwargs = {key: on_device(value) for key, value in model_kwargs.items()}
        label_kwargs = {key: on_device(value) for key, value in label_kwargs.items()}
        batch = dict(batch)
        batch["option_mask"] = on_device(batch["option_mask"])

        device_type = self.device if self.device in ("cpu", "cuda", "mps") else "cpu"
        with torch.autocast(
            device_type=device_type,
            dtype=torch.float16,
            enabled=self.config.fp16,
        ):
            output = model(**model_kwargs)
            try:
                breakdown = compute_losses(output, batch["option_mask"], **label_kwargs)
            except LossNotFiniteError as exc:
                reason = f"loss_not_finite:{exc.component}"
                self._register_nonfinite(reason)
                self._emit_step_record(loss=None, skipped=True, reason=reason, grad_norm=None)
                return StepResult(
                    step=self.step,
                    epoch=self.epoch,
                    loss=None,
                    skipped=True,
                    reason=reason,
                    sampler_offset=self.sampler_offset,
                    nonfinite_steps=self.nonfinite_steps,
                )

        # A batch with no active loss term yields a detached zero. Backward on
        # it would raise a raw RuntimeError ("element 0 of tensors does not
        # require grad"). Treat it as a typed skip instead: no optimizer step,
        # no scheduler step, no step counter advance, no nonfinite counter
        # change (nothing diverged — there was simply nothing to learn from).
        if not any(breakdown.active.values()):
            reason = "no_active_terms"
            self._emit_step_record(loss=None, skipped=True, reason=reason, grad_norm=None)
            return StepResult(
                step=self.step,
                epoch=self.epoch,
                loss=None,
                skipped=True,
                reason=reason,
                sampler_offset=self.sampler_offset,
                nonfinite_steps=self.nonfinite_steps,
                active=breakdown.active,
            )

        # The loss is carried in FP32 (t4-optimization.md §2); casting from
        # the fp16 graph is exact and keeps the scaler boundary finite.
        loss = self._weighted_total(breakdown).float()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        # One global-norm sync replaces a per-parameter isfinite() loop: any
        # inf/NaN gradient makes the returned total norm inf/NaN. clip_grad_norm_
        # must run after unscale_ and before the step, exactly as before.
        optimized_parameters = list(
            {
                id(parameter): parameter
                for group in self.optimizer.param_groups
                for parameter in group["params"]
                if parameter.requires_grad
            }.values()
        )
        grad_norm_t = torch.nn.utils.clip_grad_norm_(
            optimized_parameters, self.config.grad_clip_norm
        )
        if not bool(torch.isfinite(grad_norm_t)):
            reason = "gradient_not_finite"
            # Let the scaler back off before the next attempt.
            self.scaler.update()
            self._register_nonfinite(reason)
            self._emit_step_record(loss=None, skipped=True, reason=reason, grad_norm=None)
            return StepResult(
                step=self.step,
                epoch=self.epoch,
                loss=None,
                skipped=True,
                reason=reason,
                sampler_offset=self.sampler_offset,
                nonfinite_steps=self.nonfinite_steps,
                active=breakdown.active,
            )

        grad_norm = float(grad_norm_t)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.step += 1

        loss_value = float(loss.detach())
        self._emit_step_record(loss=loss_value, skipped=False, reason=None, grad_norm=grad_norm)

        if self.step % self.config.checkpoint_every == 0:
            self.save()

        return StepResult(
            step=self.step,
            epoch=self.epoch,
            loss=loss_value,
            skipped=False,
            reason=None,
            sampler_offset=self.sampler_offset,
            nonfinite_steps=self.nonfinite_steps,
            grad_norm=grad_norm,
            active=breakdown.active,
        )

    def fit(
        self,
        dataloader: Sequence[Mapping[str, Any]] | Any,
        *,
        epochs: int | None = None,
    ) -> list[StepResult]:
        """Train for ``epochs`` passes over ``dataloader`` (default: config).

        A checkpoint is written at the end of every epoch; every
        ``checkpoint_every`` successful steps one is written mid-epoch.
        ``dataloader`` must be re-iterable across epochs (like a standard
        ``DataLoader``); it is iterated once per epoch.
        """
        self._require_model()
        # Kernel flags are process state and must be restored in every process;
        # RNG streams are seeded only for a fresh run (load() restores them).
        configure_determinism(self.config.determinism)
        if self.step == 0 and self.epoch == 0:
            set_seed(self.config.seed, self.config.determinism)
        if epochs is not None:
            if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
                raise TrainerError(f"epochs must be an integer >= 1, got {epochs!r}")
            total_epochs = epochs
        else:
            # Resume-aware: only the epochs left in the planned budget run.
            total_epochs = max(0, self.config.epochs - self.epoch)
        history: list[StepResult] = []
        for _ in range(total_epochs):
            epoch_start = len(history)
            # sampler_offset counts consumed batches, including typed skips.
            # A mid-epoch resume must skip exactly that many rows rather than
            # replaying them. This requires a re-iterable, fixed-order loader;
            # shuffled loaders need an explicit deterministic sampler.
            skip = 0
            if self.epoch == 0 and self.sampler_offset:
                try:
                    skip = self.sampler_offset % len(dataloader)
                except TypeError as exc:
                    raise TrainerError(
                        "mid-epoch resume requires a sized, fixed-order dataloader"
                    ) from exc
            for batch in islice(dataloader, skip, None):
                if self.step >= self.config.steps:
                    break
                history.append(self.train_batch(batch))
            # The epoch counter denotes COMPLETED epochs: increment before the
            # epoch-end checkpoint so resume does not replay this epoch.
            self.epoch += 1
            self.nonfinite_steps_epoch = 0
            self._log_epoch(history[epoch_start:])
            self.save()  # epoch-end checkpoint
            if self.logger is not None:
                self.logger.log(
                    event="epoch_end",
                    step=self.step,
                    split="train",
                    metrics={
                        "epoch": self.epoch,
                        "step": self.step,
                        "nonfinite_steps": self.nonfinite_steps,
                    },
                )
            if self.step >= self.config.steps:
                break
        return history

    def _log_epoch(self, history: list[StepResult]) -> None:
        """Emit the per-epoch task-weight record (A4 / FR-TRAIN-03)."""
        if self.weighter is None:
            return
        active: dict[str, bool] = {name: False for name in DEFAULT_TASK_NAMES}
        for result in history:
            if result.active:
                for name, flag in result.active.items():
                    active[name] = active[name] or bool(flag)
        losses = [result.loss for result in history if result.loss is not None]
        metrics: dict[str, Any] = {"active": active}
        if losses:
            metrics["loss"] = sum(losses) / len(losses)
        self.weighter.log_epoch(
            self.logger, epoch=self.epoch, metrics=metrics, step=self.step
        )

    # ------------------------------------------------------------ checkpoints
    def state_dict(self) -> dict[str, Any]:
        """The full checkpoint payload (model, optimizer, scaler, scheduler, ...)."""
        self._require_model()
        assert self.optimizer is not None and self.scheduler is not None and self.scaler is not None
        return {
            "step": self.step,
            "epoch": self.epoch,
            "sampler_offset": self.sampler_offset,
            "nonfinite_steps": self.nonfinite_steps,
            "nonfinite_steps_epoch": self.nonfinite_steps_epoch,
            "model": self.model.state_dict(),  # type: ignore[union-attr]
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "weighter": self.weighter.state_dict() if self.weighter is not None else None,
            "rng": _rng_state(),
            "config": self.config.to_dict(),
        }

    def save(self, directory: str | Path | None = None) -> Path:
        """Write ``checkpoint_step{N}.pt`` atomically plus its ``.sha256`` sidecar."""
        directory = Path(directory) if directory is not None else Path(self.config.checkpoint_dir)
        payload = self.state_dict()
        path = save_checkpoint(payload, directory, self.step)
        self._prune_checkpoints(directory)
        return path

    def _prune_checkpoints(self, directory: Path) -> None:
        """Keep only the N most recent ``checkpoint_step*.pt`` files, if asked.

        ``keep_last_n=None`` (the default) keeps everything. Only files whose
        names match the checkpoint pattern inside ``directory`` are ever
        considered, and the just-written checkpoint is never deleted.
        """
        keep = self.config.keep_last_n
        if keep is None or not directory.is_dir():
            return
        candidates: list[tuple[int, Path]] = []
        for child in directory.iterdir():
            step = step_from_filename(child)
            if step is None or not child.is_file():
                continue
            candidates.append((step, child))
        candidates.sort(key=lambda item: item[0], reverse=True)
        for step, child in candidates[keep:]:
            if step == self.step:  # never remove the current run's newest artifact
                continue
            child.unlink(missing_ok=True)
            sidecar_path(child).unlink(missing_ok=True)

    def load(self, source: str | Path) -> int:
        """Restore all state from a checkpoint file or a directory.

        The sidecar hash is verified before ``torch.load`` — a mismatch
        raises :class:`CheckpointHashMismatchError` and nothing is loaded. A
        checkpoint whose step is at or beyond ``config.steps`` raises
        :class:`CheckpointBeyondHorizonError` and nothing is loaded either.
        Returns the restored step.
        """
        self._require_model()
        assert self.optimizer is not None and self.scheduler is not None and self.scaler is not None
        assert self.model is not None

        path = Path(source)
        if path.is_dir():
            latest = latest_checkpoint(path)
            if latest is None:
                raise CheckpointNotFoundError(
                    f"no checkpoint_step*.pt found in {path}", key="checkpoint"
                )
            path = latest
        elif not path.is_file():
            raise CheckpointNotFoundError(f"checkpoint {path} does not exist", key="checkpoint")

        # Horizon check on the filename step — before the payload is read.
        name_step = step_from_filename(path)
        if name_step is not None and name_step >= self.config.steps:
            raise CheckpointBeyondHorizonError(
                f"checkpoint {path} is at step {name_step} >= configured horizon "
                f"{self.config.steps}; refusing to resume a finished run",
                step=name_step,
                steps=self.config.steps,
            )

        payload = load_checkpoint_payload(path)  # verifies hash + format version

        saved_config = payload.get("config")
        comparable_saved = (
            {key: saved_config[key] for key in self.config.resume_dict() if key in saved_config}
            if isinstance(saved_config, Mapping)
            else saved_config
        )
        if comparable_saved != self.config.resume_dict():
            raise CheckpointError(
                f"checkpoint {path} was written with a different training config "
                "(trajectory-defining settings); "
                "refusing to resume with mismatched settings",
                key="config",
            )

        try:
            payload_step = int(payload["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"checkpoint {path} is missing a valid 'step'") from exc
        if name_step is not None and payload_step != name_step:
            raise CheckpointError(
                f"checkpoint {path} filename says step {name_step} but payload says {payload_step}"
            )
        if payload_step >= self.config.steps:
            raise CheckpointBeyondHorizonError(
                f"checkpoint {path} is at step {payload_step} >= configured horizon "
                f"{self.config.steps}; refusing to resume a finished run",
                step=payload_step,
                steps=self.config.steps,
            )

        # Validate the complete payload BEFORE mutating live state. A late
        # scaler/scheduler/RNG failure must not leave a hybrid trainer behind.
        for key in ("model", "optimizer", "scaler", "scheduler", "rng"):
            if not isinstance(payload.get(key), Mapping):
                raise CheckpointError(
                    f"checkpoint {path} is missing a valid {key!r} state", key=key
                )
        rng = payload["rng"]
        for key in ("python", "numpy", "torch"):
            if rng.get(key) is None:
                raise CheckpointError(
                    f"checkpoint {path} RNG state is missing {key!r}", key="rng"
                )
        if self.weighter is not None and not isinstance(payload.get("weighter"), Mapping):
            raise CheckpointError(
                f"checkpoint {path} is missing a valid 'weighter' state", key="weighter"
            )
        counters: dict[str, int] = {}
        for key in ("epoch", "sampler_offset", "nonfinite_steps", "nonfinite_steps_epoch"):
            value = payload.get(key)
            if type(value) is not int or value < 0:
                raise CheckpointError(
                    f"checkpoint {path} has invalid {key!r}: {value!r}", key=key
                )
            counters[key] = value

        before = copy.deepcopy(self.state_dict())
        rng_before = _rng_state()
        try:
            self.model.load_state_dict(payload["model"])
            self.optimizer.load_state_dict(payload["optimizer"])
            self.scaler.load_state_dict(dict(payload["scaler"]))
            self.scheduler.load_state_dict(payload["scheduler"])
            if self.weighter is not None:
                self.weighter.load_state_dict(payload["weighter"])
            _restore_rng(rng)
            configure_determinism(self.config.determinism)
        except Exception as exc:
            # Roll back every mutation, including RNG and counters.
            self.model.load_state_dict(before["model"])
            self.optimizer.load_state_dict(before["optimizer"])
            self.scaler.load_state_dict(before["scaler"])
            self.scheduler.load_state_dict(before["scheduler"])
            if self.weighter is not None and before.get("weighter") is not None:
                self.weighter.load_state_dict(before["weighter"])
            _restore_rng(rng_before)
            if isinstance(exc, CheckpointError):
                raise
            raise CheckpointError(f"checkpoint {path} could not be restored: {exc}") from exc

        self.step = payload_step
        self.epoch = counters["epoch"]
        self.sampler_offset_restore(counters["sampler_offset"])
        self.nonfinite_steps = counters["nonfinite_steps"]
        self.nonfinite_steps_epoch = counters["nonfinite_steps_epoch"]
        return self.step

    def sampler_offset_restore(self, offset: int) -> None:
        """Set both the internal counter and an attached sampler's offset."""
        offset = int(offset)
        self._sampler_offset = offset
        if self.sampler is not None and hasattr(self.sampler, "offset"):
            try:
                self.sampler.offset = offset
            except Exception:
                pass
