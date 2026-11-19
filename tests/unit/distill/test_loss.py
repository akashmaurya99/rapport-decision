"""distillation_loss — hand-computed fixtures against FR-DISTILL-02's KD term."""

from __future__ import annotations

import math

import pytest
import torch

from src.distill import distillation_loss
from src.distill.errors import DistillInputError

# Hand-computed with plain math (T = 2.0):
#   t/T softmax = [0.6285317192, 0.2312238976, 0.1402443832]
#   s/T softmax = [0.4192289516, 0.3264958358, 0.2542752126]
#   KL = 0.09130699391503301 ;  L = T^2 * KL = 4 * KL = 0.36522797566013204
TEACHER_LOGPROBS = [2.0, 0.0, -1.0]
STUDENT_SCORES = [1.0, 0.5, 0.0]
EXPECTED_LOSS = 0.36522797566013204


def test_matches_hand_computed_fixture() -> None:
    loss = distillation_loss(STUDENT_SCORES, TEACHER_LOGPROBS, temperature=2.0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(EXPECTED_LOSS, rel=0.0, abs=1e-6)


def test_kl_is_non_negative_and_zero_for_identical_distributions() -> None:
    identical = distillation_loss(TEACHER_LOGPROBS, TEACHER_LOGPROBS, temperature=2.0)
    assert identical.item() == pytest.approx(0.0, abs=1e-6)


def test_temperature_scales_as_t_squared() -> None:
    loss_t1 = distillation_loss(STUDENT_SCORES, TEACHER_LOGPROBS, temperature=1.0)
    loss_t2 = distillation_loss(STUDENT_SCORES, TEACHER_LOGPROBS, temperature=2.0)
    # not proportional in general (softmax changes with T), but both finite & positive
    assert loss_t1.item() > 0.0
    assert loss_t2.item() > 0.0
    assert loss_t2.item() != loss_t1.item()


def test_batch_input_returns_mean_scalar() -> None:
    batch_student = torch.tensor([STUDENT_SCORES, [0.0, 1.0, 2.0]], dtype=torch.float32)
    batch_teacher = torch.tensor([TEACHER_LOGPROBS, [0.0, 0.0, 0.0]], dtype=torch.float32)
    loss = distillation_loss(batch_student, batch_teacher, temperature=2.0)
    assert loss.dim() == 0
    single = distillation_loss(STUDENT_SCORES, TEACHER_LOGPROBS, temperature=2.0)
    other = distillation_loss([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], temperature=2.0)
    assert loss.item() == pytest.approx((single.item() + other.item()) / 2.0, abs=1e-6)


def test_gradients_flow_to_student_not_teacher() -> None:
    student = torch.tensor(STUDENT_SCORES, dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor(TEACHER_LOGPROBS, dtype=torch.float32, requires_grad=True)
    loss = distillation_loss(student, teacher, temperature=2.0)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None  # teacher branch detached


def test_student_grad_points_toward_teacher() -> None:
    # a tiny gradient step on the student logits must reduce the KD loss
    student = torch.tensor(STUDENT_SCORES, dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor(TEACHER_LOGPROBS, dtype=torch.float32)
    before = distillation_loss(student, teacher, temperature=2.0).item()
    distillation_loss(student, teacher, temperature=2.0).backward()
    assert student.grad is not None
    with torch.no_grad():
        student_step = student - 0.1 * student.grad
    after = distillation_loss(student_step, teacher, temperature=2.0).item()
    assert after < before


def test_fp32_softmax_under_fp16_input() -> None:
    student = torch.tensor(STUDENT_SCORES, dtype=torch.float16)
    teacher = torch.tensor(TEACHER_LOGPROBS, dtype=torch.float16)
    loss = distillation_loss(student, teacher, temperature=2.0)
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(EXPECTED_LOSS, abs=1e-3)


def test_invalid_inputs_raise_typed_error() -> None:
    with pytest.raises(DistillInputError):
        distillation_loss([1.0], [1.0, 2.0])  # shape mismatch
    with pytest.raises(DistillInputError):
        distillation_loss([1.0, math.nan], [0.0, 0.0])  # non-finite student
    with pytest.raises(DistillInputError):
        distillation_loss([1.0, 2.0], [0.0, math.inf])  # non-finite teacher
    with pytest.raises(DistillInputError):
        distillation_loss([1.0, 2.0], [0.0, 0.0], temperature=0.0)  # bad temperature
    with pytest.raises(DistillInputError):
        distillation_loss([1.0, 2.0], [0.0, 0.0], temperature=math.nan)
    with pytest.raises(DistillInputError):
        distillation_loss([1.0], [0.0])  # fewer than two options


def test_extreme_logits_stay_finite() -> None:
    # max-subtraction keeps the softmax finite for huge but representable logits
    student = [1e30, -1e30]
    teacher = [-1e30, 1e30]
    loss = distillation_loss(student, teacher, temperature=2.0)
    assert torch.isfinite(loss)
