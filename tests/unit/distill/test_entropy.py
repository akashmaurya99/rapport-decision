"""The 95th-percentile entropy exclusion filter (documented constant)."""

from __future__ import annotations

import math

import pytest

from src.distill import (
    ENTROPY_EXCLUSION_PERCENTILE,
    entropy_exclusion_mask,
    entropy_from_logprobs,
    percentile_linear,
)
from src.distill.errors import DistillInputError


def test_documented_constant_is_95() -> None:
    assert ENTROPY_EXCLUSION_PERCENTILE == 95.0


def test_entropy_of_uniform_distribution_is_max() -> None:
    uniform = entropy_from_logprobs([0.0, 0.0, 0.0, 0.0])
    assert uniform == pytest.approx(math.log(4.0), abs=1e-9)
    peaked = entropy_from_logprobs([100.0, 0.0, 0.0, 0.0])
    assert peaked == pytest.approx(0.0, abs=1e-6)


def test_entropy_handles_shift_invariance() -> None:
    base = entropy_from_logprobs([1.0, 2.0, 3.0])
    shifted = entropy_from_logprobs([1001.0, 1002.0, 1003.0])
    assert base == pytest.approx(shifted, abs=1e-6)


def test_percentile_linear_matches_numpy_semantics() -> None:
    values = list(range(1, 21))  # 1..20
    # rank = 0.95 * (n - 1) = 18.05 -> between 19 and 20
    expected = 19.0 + 0.05 * (20.0 - 19.0)
    assert percentile_linear(values, 95.0) == pytest.approx(expected)
    assert percentile_linear([5.0], 95.0) == 5.0
    assert percentile_linear(values, 0.0) == 1.0
    assert percentile_linear(values, 100.0) == 20.0


def test_fixture_marks_only_top_entropy_entry() -> None:
    # 20 low-entropy examples + 1 flat (max-entropy) example = flat teacher signal
    entropies = [0.05] * 19 + [0.20, math.log(8.0)]
    mask = entropy_exclusion_mask(entropies, ENTROPY_EXCLUSION_PERCENTILE)
    assert sum(mask) == 1
    assert mask[-1] is True
    assert not any(mask[:-1])


def test_strictly_above_threshold_is_required() -> None:
    # value exactly at the 95th percentile is kept, not excluded
    values = [0.0] * 19 + [1.0, 1.0]
    mask = entropy_exclusion_mask(values, 95.0)
    # threshold is exactly 1.0 here, so nothing may be excluded
    assert mask == [False] * 21


def test_empty_batch_returns_empty_mask() -> None:
    assert entropy_exclusion_mask([]) == []


def test_cache_build_marks_flat_teacher_entries(tmp_path, example_factory,
                                                stub_teacher_factory) -> None:
    # batch of 4 examples; stub returns progressively flatter distributions
    class FlatTeacher:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, example, prompt):
            self.calls += 1
            vec = {
                1: [50.0, 0.0, 0.0],
                2: [50.0, 0.0, 0.0],
                3: [50.0, 0.0, 0.0],
                4: [0.0, 0.0, 0.0],  # flat -> excluded
            }[self.calls]
            return vec

    from src.distill import DistillOptions, build_soft_label_cache

    examples = [
        example_factory(goal=f"Decision goal number {i} for rollout planning") for i in range(4)
    ]
    result = build_soft_label_cache(
        examples, "stub", "Goal: {goal}\nAnswer:", DistillOptions(cache_dir=tmp_path,
                                                                  teacher=FlatTeacher())
    )
    flags = [entry.exclude_from_distillation for entry in result.labels.values()]
    assert flags.count(True) <= 1
    flat_id = examples[-1].example_id
    assert result.labels[flat_id].exclude_from_distillation is True


def test_percentile_rejects_bad_input() -> None:
    with pytest.raises(DistillInputError):
        percentile_linear([], 95.0)
    with pytest.raises(DistillInputError):
        percentile_linear([1.0], 101.0)
    with pytest.raises(DistillInputError):
        entropy_from_logprobs([])
    with pytest.raises(DistillInputError):
        entropy_from_logprobs([0.0, math.inf])
