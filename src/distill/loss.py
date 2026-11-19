"""The knowledge-distillation loss for Stage C (FR-DISTILL-02 support).

``distillation_loss`` computes

.. math::

    L_{KD} = T^2 \\cdot KL(\\mathrm{softmax}(t/T) \\,\\|\\, \\mathrm{softmax}(s/T))

where ``t`` is the teacher's option log-probs and ``s`` is the student's
option scores (``choice_logits``). The student softmax is computed in FP32
regardless of the incoming dtype; the teacher branch is detached so gradients
flow only into the student. The reduction over a batch is the mean, so a
single-example input (1-D) and a batch (2-D) both return a scalar tensor.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .errors import DistillInputError

DEFAULT_TEMPERATURE = 2.0


def distillation_loss(
    student_scores: Tensor | Sequence[float],
    teacher_logprobs: Tensor | Sequence[float],
    temperature: float = DEFAULT_TEMPERATURE,
) -> Tensor:
    """``T^2 * KL(softmax(t/T) || softmax(s/T))`` as a scalar FP32 tensor.

    Parameters
    ----------
    student_scores:
        Student option scores/logits, shape ``[n]`` (one example) or
        ``[batch, n]``. May carry gradients.
    teacher_logprobs:
        Teacher option log-probs at the answer position, same shape as
        ``student_scores``. Detached inside — never receives gradients.
    temperature:
        Distillation temperature ``T`` (Stage C uses ``2.0``). Must be
        finite and ``> 0``.
    """
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise DistillInputError(
            f"temperature must be a real number, got {temperature!r}", field="temperature"
        )
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise DistillInputError(f"temperature must be finite and > 0, got {temperature!r}")

    student = _as_float32(student_scores, "student_scores", detach=False)
    teacher = _as_float32(teacher_logprobs, "teacher_logprobs", detach=True)

    if student.shape != teacher.shape:
        raise DistillInputError(
            f"student_scores and teacher_logprobs must have the same shape, "
            f"got {tuple(student.shape)} vs {tuple(teacher.shape)}"
        )
    if student.dim() not in (1, 2):
        raise DistillInputError(
            f"student_scores must be [n] or [batch, n], got shape {tuple(student.shape)}"
        )
    if student.shape[-1] < 2:
        raise DistillInputError("at least two options are required per example")

    reduce_dim = -1
    log_prob_teacher = F.log_softmax(teacher / temperature, dim=reduce_dim)
    log_prob_student = F.log_softmax(student / temperature, dim=reduce_dim)
    prob_teacher = log_prob_teacher.exp()
    kl = (prob_teacher * (log_prob_teacher - log_prob_student)).sum(dim=reduce_dim)
    loss = kl * (temperature**2)
    result = loss.mean() if student.dim() == 2 else loss
    if not bool(torch.isfinite(result)):
        raise DistillInputError(
            f"distillation loss is non-finite (value: {result.detach().cpu().item()!r})",
            field="loss",
        )
    return result


def _as_float32(
    value: Tensor | Sequence[float], field: str, *, detach: bool
) -> Tensor:
    if isinstance(value, Tensor):
        tensor = value
    else:
        if isinstance(value, (str, bytes)):
            raise DistillInputError(f"{field} must be a tensor or a sequence of numbers")
        try:
            tensor = torch.as_tensor(list(value))
        except (TypeError, ValueError) as exc:
            raise DistillInputError(f"{field} must contain only numbers: {exc}") from exc
    if detach:
        tensor = tensor.detach()
    try:
        tensor = tensor.to(dtype=torch.float32)
    except (RuntimeError, TypeError) as exc:
        raise DistillInputError(f"{field} must be convertible to float32: {exc}") from exc
    if not bool(torch.isfinite(tensor).all()):
        raise DistillInputError(f"{field} contains non-finite values")
    return tensor
