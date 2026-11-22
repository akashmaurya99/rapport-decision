"""RAPPORT model assembly: conditioned trunk, marker readout, six heads.

Forward signature (contract):

    RapportModel.forward(input_ids, attention_mask, option_mask, task_ids)
        -> RapportOutput

where ``RapportOutput`` carries the six head tensors plus the score vector:

    choice_logits[n]  scores[n]  probability[1]  confidence[1]  answerable[1]  risk[8]

Single pass (FR-MODEL-04): the state and all options are encoded jointly by
one trunk call; there is no per-option forward pass anywhere in the default
path — options are gathered *from* the shared hidden states.

CLS analogue (spec §4/§6 "CLS state vector"): RAPPORT's marker set
(``<task>``, ``<goal>``, ``<state>``, ``<opt>``, ``</opt>``) has no dedicated
``<cls>`` token, so the CLS-analogue vector ``c`` is the hidden state of the
``<task>`` marker — the token that always opens the joint sequence (position 0;
``encode_example`` builds ``<task>…``). The position is *derived from the input
ids* and gathered from the hidden states, never hardcoded as ``hidden[:, 0]``.
Left-padded batches (or a missing/duplicate task marker) are refused with a
typed ``TaskMarkerError`` rather than silently reading a pad row.

Task conditioning (spec §4 row 2): a learned per-task vector is added at the
``<task>`` marker position(s) *before* the encoder blocks, via a private
subclass of ``RapportTrunk`` (TASK-05's trunk is wired unchanged — its own
forward is untouched). The vectors are indexed by ``task_ids``; ids 0-5 are
the six task families in schema order (choice, preference, scoring,
probability, abstention, risk) and ids 6-7 are reserved (spec §4: "6 families
+ 2 reserved"), exactly 8 slots.

Answerable absorption: ``answerable`` is the probability head's sigmoid
output, re-emitted — one shared stack, zero added parameters (see
``src/model/head/heads.py`` and the TASK-06 notes for the full accounting).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.model.core import RapportConfig, RapportTrunk
from src.tokenization.tokenizer import GOAL_ID, OPT_ID, TASK_ID

from .errors import (
    AttentionMaskError,
    DeviceMismatchError,
    GoalMarkerError,
    InvalidOptionCountError,
    InvalidTaskIdError,
    OptionMaskError,
    TaskMarkerError,
)
from .heads import ConfidenceHead, ProbabilityHead, RiskHead, confidence_features
from .readout import MASK_NEG, MarkerReadout

NUM_TASKS = 8  # six families + two reserved (spec §4, task-conditioning row)
MIN_OPTIONS = 2  # spec §5 / schema MIN_OPTIONS
MAX_OPTIONS = 8  # spec §5 / schema MAX_OPTIONS
RISK_CATEGORIES = 8  # spec §6 r_k, k = 1..8

__all__ = [
    "MAX_OPTIONS",
    "MIN_OPTIONS",
    "NUM_TASKS",
    "RISK_CATEGORIES",
    "MASK_NEG",
    "RapportModel",
    "RapportOutput",
]


@dataclass(frozen=True)
class RapportOutput:
    """All six head outputs plus the raw score vector.

    Shape contract (FR-MODEL-03), for batch ``B`` and ``n`` option slots:

    * ``choice_logits`` — ``[B, n]`` listwise logits (padding slots ``MASK_NEG``);
    * ``scores`` — ``[B, n]`` sigmoid utility in ``[0, 1]`` (padding slots 0);
    * ``probability`` — ``[B, 1]``;
    * ``confidence`` — ``[B, 1]``;
    * ``answerable`` — ``[B, 1]`` (absorbed into the probability head);
    * ``risk`` — ``[B, 8]``;
    * ``score_vector`` — ``[B, n]`` raw cosine score divided by the per-task
      temperature (spec §6 ``s_i``; padding slots 0).

    The three optional ``*_logits`` fields carry the raw pre-sigmoid values the
    losses consume through :func:`torch.nn.functional.binary_cross_entropy_with_logits`.
    They exist because an fp16 sigmoid saturates to exactly 0 or 1, which makes
    a hand-rolled ``log(p)`` BCE produce ``-inf``. Omitted only for synthetic
    outputs (e.g. unit-test doubles); when absent the loss falls back to the
    probability path and raises ``LossNotFiniteError`` on a degenerate value.
    """

    choice_logits: Tensor
    scores: Tensor
    probability: Tensor
    confidence: Tensor
    answerable: Tensor
    risk: Tensor
    score_vector: Tensor
    probability_logits: Tensor | None = None
    confidence_logits: Tensor | None = None
    risk_logits: Tensor | None = None


class _ConditionedTrunk(RapportTrunk):
    """RapportTrunk with the spec §4 task-conditioning row wired in.

    TASK-05's ``RapportTrunk`` is reused unchanged: embedding and blocks are
    built by the parent and this subclass only inserts the per-task vector
    addition between the embedding and the blocks (exactly where spec §4 puts
    it: "[B, 384] added at the <task> position"). TASK-05's files are not
    modified.
    """

    def __init__(self, config: RapportConfig, num_tasks: int) -> None:
        super().__init__(config)
        self.task_vector = nn.Embedding(num_tasks, config.d_model)
        # Token embeddings are scaled by sqrt(d_model) by TrunkEmbedding; an
        # N(0, 1) task vector would be a ~5% perturbation of that residual
        # stream. Match the token scale so task conditioning is not washed out.
        with torch.no_grad():
            self.task_vector.weight.normal_(0.0, math.sqrt(config.d_model))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        task_ids: Tensor | None = None,
    ) -> Tensor:
        hidden = self.embedding(input_ids)
        if task_ids is not None:
            vectors = self.task_vector(task_ids.to(torch.long))  # [B, d]
            at_task = input_ids == TASK_ID  # [B, T]
            if attention_mask is not None:
                at_task = at_task & attention_mask.bool()
            hidden = hidden + at_task.unsqueeze(-1) * vectors.unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return hidden


class RapportModel(nn.Module):
    """The full RAPPORT model: conditioned trunk → marker readout → six heads."""

    def __init__(self, config: RapportConfig | None = None, num_tasks: int = NUM_TASKS) -> None:
        super().__init__()
        if not isinstance(num_tasks, int) or isinstance(num_tasks, bool) or num_tasks < 1:
            raise InvalidTaskIdError(
                f"num_tasks must be a positive integer, got {num_tasks!r}",
                field="num_tasks",
            )
        self.config = config if config is not None else RapportConfig.compact()
        self.num_tasks = num_tasks
        self.risk_categories = RISK_CATEGORIES
        d_model = self.config.d_model

        self.trunk = _ConditionedTrunk(self.config, num_tasks)
        self.readout = MarkerReadout(d_model, num_tasks)
        self.probability_head = ProbabilityHead(d_model)
        self.confidence_head = ConfidenceHead(d_model)
        self.risk_head = RiskHead(d_model, RISK_CATEGORIES)

    # -- input validation (E2): runs before any compute ----------------------

    def _validate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        option_mask: Tensor,
        task_ids: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor, Tensor]:
        detail = ""
        if isinstance(input_ids, Tensor):
            detail = f" with shape {tuple(input_ids.shape)}"
        if not isinstance(input_ids, Tensor) or input_ids.dim() != 2:
            raise OptionMaskError(
                f"input_ids must be a 2-D [B, T] tensor, got "
                f"{type(input_ids).__name__}{detail}",
                field="input_ids",
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise OptionMaskError(
                f"input_ids must be an int32 or int64 tensor, got {input_ids.dtype}",
                field="input_ids",
            )
        batch = input_ids.shape[0]
        ids = input_ids.to(torch.long)

        # Device agreement is checked before any compute: mixing a GPU
        # input_ids with CPU mask/task tensors otherwise fails deep inside
        # embedding/gather with an opaque backend error.
        for name, tensor in (
            ("attention_mask", attention_mask),
            ("option_mask", option_mask),
            ("task_ids", task_ids),
        ):
            if isinstance(tensor, Tensor) and tensor.device != input_ids.device:
                raise DeviceMismatchError(
                    f"{name} is on device {tensor.device} but input_ids is on "
                    f"{input_ids.device}",
                    field=name,
                )

        if attention_mask is not None:
            if not isinstance(attention_mask, Tensor) or attention_mask.dim() != 2:
                raise AttentionMaskError(
                    "attention_mask must be a 2-D [B, T] tensor, got "
                    f"{type(attention_mask).__name__}",
                    field="attention_mask",
                )
            if tuple(attention_mask.shape) != tuple(input_ids.shape):
                raise AttentionMaskError(
                    f"attention_mask must match input_ids shape "
                    f"{tuple(input_ids.shape)}, got {tuple(attention_mask.shape)}",
                    field="attention_mask",
                )
            attn = attention_mask
            if not bool(((attn == 0) | (attn == 1)).all()):
                raise AttentionMaskError(
                    "attention_mask must be binary (0 = pad, 1 = real token)",
                    field="attention_mask",
                )
            attention_mask = attn

        if not isinstance(option_mask, Tensor) or option_mask.dim() != 2:
            raise OptionMaskError(
                f"option_mask must be a 2-D [B, n] tensor, got "
                f"{type(option_mask).__name__}",
                field="option_mask",
            )
        if tuple(option_mask.shape[:1]) != (batch,):
            raise OptionMaskError(
                f"option_mask batch {option_mask.shape[0]} != input_ids batch {batch}",
                field="option_mask",
            )
        width = option_mask.shape[1]
        if not bool(((option_mask == 0) | (option_mask == 1)).all()):
            raise OptionMaskError("option_mask must be binary (0 = pad, 1 = option)",
                                  field="option_mask")
        # E2: 1 or 9 (or any out-of-range) slot width is refused outright.
        if width < MIN_OPTIONS or width > MAX_OPTIONS:
            raise InvalidOptionCountError(
                f"option slot width {width} is outside the accepted "
                f"{MIN_OPTIONS}..{MAX_OPTIONS} range",
                field="option_mask",
            )
        counts = option_mask.to(torch.long).sum(dim=1)
        for row, count in enumerate(counts.tolist()):
            if count < MIN_OPTIONS or count > MAX_OPTIONS:
                raise InvalidOptionCountError(
                    f"example {row} declares {count} option(s); "
                    f"the accepted range is {MIN_OPTIONS}..{MAX_OPTIONS}",
                    field="option_mask",
                )
        # Prefix layout: padding is trailing; no zero may precede a one.
        keep = option_mask.to(torch.bool)
        if bool((keep[:, :-1].logical_not() & keep[:, 1:]).any()):
            raise OptionMaskError(
                "option_mask must be a prefix mask (real options first, "
                "padding slots trailing); found a padding slot before an option",
                field="option_mask",
            )
        # Only *unmasked* markers count: a marker parked in the attention-masked
        # tail would be gathered from a row that contributes nothing to the
        # encoding, silently conditioning the readout on a pad position.
        valid = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones_like(ids, dtype=torch.bool)
        )
        markers = ((ids == OPT_ID) & valid).sum(dim=1)
        for row, (found, declared) in enumerate(zip(markers.tolist(), counts.tolist())):
            if found != declared:
                raise OptionMaskError(
                    f"example {row}: option_mask declares {declared} option(s) but "
                    f"the input_ids contain {found} unmasked '<opt>' marker(s)",
                    field="option_mask",
                )

        goal_markers = ((ids == GOAL_ID) & valid).sum(dim=1)
        for row, found in enumerate(goal_markers.tolist()):
            if found != 1:
                raise GoalMarkerError(
                    f"example {row}: expected exactly one '<goal>' marker, found {found}",
                    field="input_ids",
                )

        if not isinstance(task_ids, Tensor) or task_ids.dim() != 1:
            raise InvalidTaskIdError(
                f"task_ids must be a 1-D [B] integer tensor, got "
                f"{type(task_ids).__name__}",
                field="task_ids",
            )
        if tuple(task_ids.shape) != (batch,):
            raise InvalidTaskIdError(
                f"task_ids shape {tuple(task_ids.shape)} != batch {batch}",
                field="task_ids",
            )
        if task_ids.dtype not in (torch.int32, torch.int64, torch.bool):
            raise InvalidTaskIdError(
                f"task_ids must have an integer dtype, got {task_ids.dtype}",
                field="task_ids",
            )
        tasks = task_ids.to(torch.long)
        if bool((tasks < 0).any()) or bool((tasks >= self.num_tasks).any()):
            raise InvalidTaskIdError(
                f"task_ids must lie in [0, {self.num_tasks}), "
                f"got min {int(tasks.min())} and max {int(tasks.max())}",
                field="task_ids",
            )
        task_positions = self._task_positions(ids, attention_mask)
        return ids, attention_mask, keep.to(torch.long), tasks, task_positions

    # -- task-marker position (CLS analogue) ------------------------------------

    @staticmethod
    def _task_positions(ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        """Return the ``[B]`` index of the single unmasked ``<task>`` marker.

        The marker must open the sequence at position 0 — the layout
        ``encode_example`` emits. A left-padded batch (position 0 is pad) is
        refused with a typed error rather than letting the probability,
        confidence and risk heads read a pad row as if it were CLS.
        """
        valid = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones_like(ids, dtype=torch.bool)
        )
        if bool((~valid[:, :-1] & valid[:, 1:]).any()):
            raise TaskMarkerError(
                "input_ids must be right-padded; found a padding position "
                "before a real token",
                field="attention_mask",
            )
        at_task = (ids == TASK_ID) & valid
        counts = at_task.sum(dim=1)
        bad = (counts != 1).nonzero(as_tuple=True)[0]
        if bad.numel():
            row = int(bad[0])
            raise TaskMarkerError(
                f"example {row}: expected exactly one unmasked '<task>' marker, "
                f"found {int(counts[row])}",
                field="input_ids",
            )
        positions = at_task.to(torch.long).argmax(dim=1)
        if bool((positions != 0).any()):
            raise TaskMarkerError(
                "the '<task>' marker must open the sequence at position 0",
                field="input_ids",
            )
        return positions

    # -- marker positions ------------------------------------------------------

    @staticmethod
    def _marker_positions(
        ids: Tensor, option_mask: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Locate the ``<goal>`` and the ``<opt>`` markers in every row.

        Returns ``(goal_positions [B], option_positions [B, n])``. Padding
        slots point at index 0 and are ignored by the readout through the
        mask. Marker order is the slot order (spec §5: options share one
        ``<opt>`` marker, slot *j* is the *j*-th marker in the sequence).
        Only unmasked positions are considered; both result tensors live on
        ``ids.device`` so they can gather GPU/mps hidden states directly.
        """
        batch = ids.shape[0]
        width = option_mask.shape[1]
        valid = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones_like(ids, dtype=torch.bool)
        )
        goal = torch.empty(batch, dtype=torch.long, device=ids.device)
        options = torch.zeros(batch, width, dtype=torch.long, device=ids.device)
        for row in range(batch):
            row_valid = valid[row]
            goal_row = ((ids[row] == GOAL_ID) & row_valid).nonzero(as_tuple=True)[0]
            # Validation already guaranteed exactly one unmasked <goal> per row.
            goal[row] = goal_row[0]
            marker_row = ((ids[row] == OPT_ID) & row_valid).nonzero(as_tuple=True)[0]
            # Padding slots stay at 0 (the options tensor is zero-initialised).
            take = min(width, marker_row.numel())
            if take:
                options[row, :take] = marker_row[:take]
        return goal, options

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        option_mask: Tensor,
        task_ids: Tensor,
    ) -> RapportOutput:
        ids, attention_mask, keep, tasks, task_positions = self._validate(
            input_ids, attention_mask, option_mask, task_ids
        )
        hidden = self.trunk(ids, attention_mask, tasks)  # single joint pass

        goal_positions, option_positions = self._marker_positions(ids, keep, attention_mask)
        readout = self.readout(hidden, goal_positions, option_positions, tasks, keep)

        # CLS analogue: hidden state at the derived <task> marker position.
        cls_index = task_positions.view(-1, 1, 1).expand(-1, -1, hidden.shape[-1])
        cls_vector = hidden.gather(1, cls_index).squeeze(1)
        probability_logits = self.probability_head.logits(cls_vector)  # [B, 1]
        probability = torch.sigmoid(probability_logits)
        confidence_input = confidence_features(cls_vector, readout.probabilities)
        confidence_logits = self.confidence_head.logits(confidence_input)  # [B, 1]
        confidence = torch.sigmoid(confidence_logits)
        risk_logits = self.risk_head.logits(cls_vector)  # [B, 8]
        risk = torch.sigmoid(risk_logits)

        return RapportOutput(
            choice_logits=readout.choice_logits,
            scores=readout.scores,
            probability=probability,
            confidence=confidence,
            answerable=probability,  # absorbed into the probability head (documented)
            risk=risk,
            score_vector=readout.score_vector,
            probability_logits=probability_logits,
            confidence_logits=confidence_logits,
            risk_logits=risk_logits,
        )
