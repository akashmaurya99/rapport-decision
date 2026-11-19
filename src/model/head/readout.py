"""Marker readout: gather marker hidden states, cosine scoring (spec §4/§6).

The readout is the only place option *slots* exist. Given the trunk's hidden
states it gathers:

* the ``<goal>`` marker hiddens — the query vectors, projected by ``W_q``;
* each ``<opt>`` marker hidden — the option vectors, projected by ``W_o``;

and scores every option as a per-task-temperature cosine similarity with a
per-task affine before the listwise softmax.

Permutation equivariance (FR-MODEL-02) holds **by construction** here: every
operation is either slot-wise (projections, cosine, affine, sigmoid) or a
row-wise reduction over slots that does not depend on slot indices (softmax,
masking). No parameter is indexed by option slot — there is no ordinal
embedding, no positional table, no slot-specific bias. Permuting the gathered
option vectors permutes the emitted scores by the same permutation exactly.

Why the exact-1e-4 guarantee is stated at the readout level: token positions
are inherent to the sequence (RoPE is the trunk's only position mechanism,
spec §4/§6), so re-encoding a sequence whose option blocks are swapped moves
marker token positions and perturbs the hidden states the trunk emits. That
deviation is a property of the encoder, not of the readout — the spec itself
declares "positions are inherent to the sequence". The permutation contract
therefore binds the readout: the same encoded sequence, with the option slots
permuted, yields exactly permuted scores. End-to-end perturbation is measured
by TASK-19's robustness harness.

Padding slots (``option_mask == 0``) are ignored everywhere: their affines are
replaced by ``MASK_NEG`` before the softmax (so choice logits are directly
listwise-softmax CE compatible with TASK-11), their utility scores are zeroed,
and the choice distribution sums to 1 over real options only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Finite stand-in for -inf in the padding slots of choice_logits. Large enough
# that softmax over the slots assigns exactly 0 to padding, finite enough that
# downstream cross-entropy (TASK-11) and ONNX export (TASK-23) never see inf/nan.
MASK_NEG = -1e9


@dataclass(frozen=True)
class ReadoutOutput:
    """The tensors the readout hands to the heads.

    ``probabilities`` is the masked listwise distribution over real options;
    padding slots carry exactly 0. ``score_vector`` is the raw cosine score
    divided by the per-task temperature (spec §6's ``s_i``); ``scores`` is the
    sigmoid calibrated utility in ``[0, 1]``; ``choice_logits`` is the
    affine-scaled score (spec's "scores before softmax") with padding slots set
    to ``MASK_NEG``.
    """

    choice_logits: Tensor
    score_vector: Tensor
    scores: Tensor
    probabilities: Tensor


class MarkerReadout(nn.Module):
    """Gather markers, project, cosine-score, affine-scale.

    Parameters are exactly the spec §4 rows: ``W_q`` and ``W_o`` (bias-free,
    d×d), the per-task learned temperature ``τ_task`` (parameterized as
    ``log_tau`` so τ > 0 by construction) and the per-task affine
    (``task_scale``/``task_bias``, spec's "Task affine" row). Nothing else.
    """

    def __init__(self, d_model: int, num_tasks: int) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be a positive integer, got {d_model}")
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be a positive integer, got {num_tasks}")
        self.d_model = d_model
        self.num_tasks = num_tasks
        self.query = nn.Linear(d_model, d_model, bias=False)  # W_q
        self.option = nn.Linear(d_model, d_model, bias=False)  # W_o
        # τ_task = exp(log_tau); zero init means τ = 1 for every task.
        self.log_tau = nn.Parameter(torch.zeros(num_tasks))
        # Per-task affine on scores before softmax; identity at init.
        self.task_scale = nn.Parameter(torch.ones(num_tasks))
        self.task_bias = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self,
        hidden: Tensor,
        goal_positions: Tensor,
        option_positions: Tensor,
        task_ids: Tensor,
        option_mask: Tensor,
    ) -> ReadoutOutput:
        """Score every option slot of every example.

        ``goal_positions`` is ``[B]`` (index of the single ``<goal>`` marker
        per row); ``option_positions`` is ``[B, n]`` (index of the *j*-th
        ``<opt>`` marker per row; padding slots may hold any in-range index and
        are ignored via ``option_mask``). ``option_mask`` is ``[B, n]`` with
        1 = real option and 0 = padding slot.
        """
        if hidden.dim() != 3:
            raise ValueError(f"hidden must be [B, T, d], got shape {tuple(hidden.shape)}")
        batch, _, d_model = hidden.shape
        if d_model != self.d_model:
            raise ValueError(f"expected hidden width {self.d_model}, got {d_model}")

        goal_index = goal_positions.to(torch.long).view(batch, 1, 1)
        goal_index = goal_index.expand(-1, 1, d_model)
        query = hidden.gather(1, goal_index).squeeze(1)  # [B, d]

        option_index = option_positions.to(torch.long).unsqueeze(-1).expand(-1, -1, d_model)
        options = hidden.gather(1, option_index)  # [B, n, d]

        query = self.query(query)  # W_q q
        options = self.option(options)  # W_o o_i

        cos = F.cosine_similarity(options, query.unsqueeze(1), dim=-1, eps=1e-8)  # [B, n]
        tau = torch.exp(self.log_tau[task_ids.to(torch.long)]).unsqueeze(1)  # [B, 1]
        raw = cos / tau  # spec §6 s_i

        scale = self.task_scale[task_ids.to(torch.long)].unsqueeze(1)  # [B, 1]
        bias = self.task_bias[task_ids.to(torch.long)].unsqueeze(1)  # [B, 1]
        logits = scale * raw + bias

        keep = option_mask.to(torch.bool)
        choice_logits = logits.masked_fill(~keep, MASK_NEG)
        score_vector = raw.masked_fill(~keep, 0.0)
        scores = torch.sigmoid(logits).masked_fill(~keep, 0.0)
        probabilities = torch.softmax(choice_logits, dim=-1)  # padding slots underflow to 0

        return ReadoutOutput(
            choice_logits=choice_logits,
            score_vector=score_vector,
            scores=scores,
            probabilities=probabilities,
        )
