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

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from src.ops.logging import JsonlLogger
from src.training.losses import LossNotFiniteError, compute_losses

from .checkpoint import (
    latest_checkpoint,
    load_checkpoint_payload,
    save_checkpoint,
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


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """AdamW with weight decay excluded on norms, biases and embeddings (§3.1)."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if param.ndim < 2 or any(token in lowered for token in _NO_DECAY_TOKENS):
            no_decay.append(param)
        else:
            decay.append(param)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": config.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
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


def _grads_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        grad = param.grad
        if grad is not None and not bool(torch.isfinite(grad).all()):
            return False
    return True


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
        self.optimizer = optimizer if optimizer is not None else build_optimizer(model, self.config)
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
        model.zero_grad(set_to_none=True)

        model_kwargs = {key: batch[key] for key in _MODEL_INPUT_KEYS if key in batch}
        if "input_ids" not in model_kwargs:
            raise TrainerError("batch is missing 'input_ids'")
        if "option_mask" not in batch:
            raise TrainerError("batch is missing 'option_mask'")
        label_kwargs = {key: batch[key] for key in _LOSS_KEYS if key in batch}

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

        # The loss is carried in FP32 (t4-optimization.md §2); casting from
        # the fp16 graph is exact and keeps the scaler boundary finite.
        loss = breakdown.total.float()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        if not _grads_finite(model):
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
            )

        grad_norm_t = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.config.grad_clip_norm
        )
        grad_norm = float(grad_norm_t) if torch.isfinite(grad_norm_t) else None

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
        total_epochs = self.config.epochs if epochs is None else epochs
        if isinstance(total_epochs, bool) or not isinstance(total_epochs, int) or total_epochs < 1:
            raise TrainerError(f"epochs must be an integer >= 1, got {total_epochs!r}")
        history: list[StepResult] = []
        for _ in range(total_epochs):
            for batch in dataloader:
                history.append(self.train_batch(batch))
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
            self.epoch += 1
            self.nonfinite_steps_epoch = 0
        return history

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
            "rng": _rng_state(),
            "config": self.config.to_dict(),
        }

    def save(self, directory: str | Path | None = None) -> Path:
        """Write ``checkpoint_step{N}.pt`` atomically plus its ``.sha256`` sidecar."""
        directory = Path(directory) if directory is not None else Path(self.config.checkpoint_dir)
        payload = self.state_dict()
        return save_checkpoint(payload, directory, self.step)

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

        payload = load_checkpoint_payload(path)  # verifies the hash first

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

        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload.get("scaler") or {})
        self.scheduler.load_state_dict(payload["scheduler"])
        rng = payload.get("rng")
        if isinstance(rng, Mapping):
            _restore_rng(rng)

        self.step = payload_step
        self.epoch = int(payload.get("epoch", 0))
        self.sampler_offset_restore(int(payload.get("sampler_offset", 0)))
        self.nonfinite_steps = int(payload.get("nonfinite_steps", 0))
        self.nonfinite_steps_epoch = int(payload.get("nonfinite_steps_epoch", 0))
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
