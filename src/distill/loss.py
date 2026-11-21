"""The knowledge-distillation loss for Stage C (FR-DISTILL-02 support).

``distillation_loss`` computes

.. math::

    L_{KD} = T^2 \\cdot KL(\\mathrm{softmax}(t/T) \\,\\|\\, \\mathrm{softmax}(s/T))

where ``t`` is the teacher's option log-probs and ``s`` is the student's
option scores (``choice_logits``). The student softmax is computed in FP32
regardless of the incoming dtype; the teacher branch is detached so gradients
flow only into the student. The reduction over a batch is the mean, so a
single-example input (1-D) and a batch (2-D) both return a scalar tensor.

An optional ``option_mask`` (1 = real option, 0 = padded slot) is applied
to both branches *before* softmax/KL, so changing a padded logit cannot
change the loss. Masked contributions are explicitly zeroed rather than left
as ``0 * nan`` from ``-inf - -inf``. The default (``None``) means every slot
is real, preserving the original behaviour.
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
    *,
    option_mask: Tensor | Sequence[Sequence[int]] | None = None,
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
    option_mask:
        Optional mask with the same shape as ``student_scores``; ``1`` marks
        a real option and ``0`` a padded slot. Padded slots are excluded from
        both softmaxes and from the KL sum. ``None`` keeps all slots.
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

    valid = _option_mask(option_mask, student, field="option_mask")
    reduce_dim = -1
    student_masked = student.masked_fill(~valid, float("-inf"))
    teacher_masked = teacher.masked_fill(~valid, float("-inf"))
    log_prob_teacher = F.log_softmax(teacher_masked / temperature, dim=reduce_dim)
    log_prob_student = F.log_softmax(student_masked / temperature, dim=reduce_dim)
    prob_teacher = log_prob_teacher.exp()
    kl_terms = prob_teacher * (log_prob_teacher - log_prob_student)
    # A fully masked slot is -inf - -inf = nan; zero it explicitly.
    kl_terms = kl_terms.masked_fill(~valid, 0.0)
    kl = kl_terms.sum(dim=reduce_dim)
    loss = kl * (temperature**2)
    result = loss.mean() if student.dim() == 2 else loss
    if not bool(torch.isfinite(result)):
        raise DistillInputError(
            f"distillation loss is non-finite (value: {result.detach().cpu().item()!r})",
            field="loss",
        )
    return result


def _option_mask(
    value: Tensor | Sequence[Sequence[int]] | None, reference: Tensor, *, field: str
) -> Tensor:
    """Return a boolean ``valid`` mask shaped like ``reference`` (all True if unset)."""
    if value is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if isinstance(value, Tensor):
        mask = value
    else:
        if isinstance(value, (str, bytes)):
            raise DistillInputError(f"{field} must be a tensor or a sequence of masks")
        try:
            mask = torch.as_tensor(list(value))
        except (TypeError, ValueError) as exc:
            raise DistillInputError(f"{field} must contain only 0/1 values: {exc}") from exc
    if mask.shape != reference.shape:
        raise DistillInputError(
            f"{field} must have the same shape as student_scores, "
            f"got {tuple(mask.shape)} vs {tuple(reference.shape)}",
            field=field,
        )
    if not bool(torch.isfinite(mask).all()) or not bool(((mask == 0) | (mask == 1)).all()):
        raise DistillInputError(f"{field} must contain only 0 and 1", field=field)
    valid = mask.to(device=reference.device, dtype=torch.bool)
    if not bool(valid.any(dim=-1).all()):
        raise DistillInputError(f"{field} leaves an example with no real option", field=field)
    return valid


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
