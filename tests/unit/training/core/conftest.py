"""Shared fixtures/helpers for the trainer-core unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.core import RapportConfig, SpecConstants  # noqa: E402
from src.model.head import RapportModel, RapportOutput  # noqa: E402
from src.tokenization.tokenizer import GOAL_ID, OPT_ID, TASK_ID  # noqa: E402

SEQ_LEN = 16
N_OPTIONS = 3
BATCH = 2


def tiny_config() -> RapportConfig:
    spec = SpecConstants(
        attention_total=0,
        swiglu_total=0,
        norms_total=0,
        trunk=0,
        embedding=0,
        heads_total=0,
        total=0,
        rows=(),
    )
    return RapportConfig(
        variant="tiny-test",
        layers=1,
        d_model=32,
        heads=2,
        head_dim=16,
        ffn=64,
        vocab_size=30_522,
        embedding_dim=32,
        spec=spec,
    )


def make_model(seed: int = 13) -> RapportModel:
    torch.manual_seed(seed)
    return RapportModel(tiny_config())


def make_batch(seed: int, batch_size: int = BATCH, n_options: int = N_OPTIONS) -> dict:
    """A deterministic synthetic batch carrying model inputs + loss labels."""
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(10, 30_522, (batch_size, SEQ_LEN), generator=generator)
    ids[:, 0] = TASK_ID
    ids[:, 1] = GOAL_ID
    position = 2
    for slot in range(n_options):
        ids[:, position] = OPT_ID
        position += 2
    option_mask = torch.ones(batch_size, n_options, dtype=torch.long)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "option_mask": option_mask,
        "task_ids": torch.zeros(batch_size, dtype=torch.long),
        "chosen_index": torch.randint(0, n_options, (batch_size,), generator=generator),
        "score_targets": torch.rand(batch_size, n_options, generator=generator),
        "probability_target": torch.rand(batch_size, generator=generator),
        "answerable_target": (torch.rand(batch_size, generator=generator) > 0.5).float(),
        "risk_targets": (torch.rand(batch_size, 8, generator=generator) > 0.7).float(),
        "preference_pairs": torch.tensor(
            [[row, 0, 1] for row in range(batch_size)], dtype=torch.long
        ),
    }


@pytest.fixture
def batch_factory():
    return make_batch


class InfGrad(torch.autograd.Function):
    """Forward returns zeros (finite); backward injects +inf gradients."""

    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        return x * 0.0

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        return torch.full_like(grad_out, float("inf"))


class StubRapportModel(torch.nn.Module):
    """Minimal nn.Module satisfying the trainer's model interface.

    Produces a finite :class:`RapportOutput` whose ``choice_logits`` depend
    on a parameter through :class:`InfGrad` — so the loss is finite but the
    gradients are not (the FR-TRAIN-04 / E5 test double).
    """

    def __init__(self, batch: int = BATCH, options: int = N_OPTIONS) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(batch, options))
        self.batch = batch
        self.options = options

    def forward(self, **kwargs):  # noqa: ARG002
        base = InfGrad.apply(self.dummy)
        logits = base + 1.0
        batch, options = logits.shape
        return RapportOutput(
            choice_logits=logits,
            scores=torch.full((batch, options), 0.5),
            probability=torch.full((batch, 1), 0.5),
            confidence=torch.full((batch, 1), 0.5),
            answerable=torch.full((batch, 1), 0.5),
            risk=torch.full((batch, 8), 0.5),
            score_vector=logits,
        )


class NanGrad(torch.autograd.Function):
    """Forward returns zeros (finite); backward injects NaN gradients."""

    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        return x * 0.0

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        return torch.full_like(grad_out, float("nan"))


class NanGradModel(torch.nn.Module):
    """Produces finite outputs but NaN gradients (global-norm skip test double)."""

    def __init__(self, batch: int = BATCH, options: int = N_OPTIONS) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(batch, options))
        self.batch = batch
        self.options = options

    def forward(self, **kwargs):  # noqa: ARG002
        logits = NanGrad.apply(self.dummy) + 1.0
        batch, options = logits.shape
        return RapportOutput(
            choice_logits=logits,
            scores=torch.full((batch, options), 0.5),
            probability=torch.full((batch, 1), 0.5),
            confidence=torch.full((batch, 1), 0.5),
            answerable=torch.full((batch, 1), 0.5),
            risk=torch.full((batch, 8), 0.5),
            score_vector=logits,
        )


class NanLogitModel(torch.nn.Module):
    """Produces NaN choice logits so ``compute_losses`` raises LossNotFiniteError."""

    def __init__(self, batch: int = BATCH, options: int = N_OPTIONS) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.batch = batch
        self.options = options

    def forward(self, **kwargs):  # noqa: ARG002
        batch, options = self.batch, self.options
        logits = torch.full((batch, options), float("nan")) + self.dummy
        return RapportOutput(
            choice_logits=logits,
            scores=torch.full((batch, options), 0.5),
            probability=torch.full((batch, 1), 0.5),
            confidence=torch.full((batch, 1), 0.5),
            answerable=torch.full((batch, 1), 0.5),
            risk=torch.full((batch, 8), 0.5),
            score_vector=logits,
        )
