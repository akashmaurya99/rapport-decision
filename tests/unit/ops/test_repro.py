"""Tests for src/ops/repro.py — FR-OPS-03 fixed-seed reproducibility check."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ops import (
    JsonlLogger,
    ReproductionError,
    ReproductionMismatchError,
    run_reference_workload,
    verify_resume,
)

SEED = 13
STEPS = 50
REFERENCE_LOSS_SEED_13 = 0.00022519580952978248


def test_reference_workload_is_deterministic() -> None:
    first = run_reference_workload(SEED, STEPS)
    second = run_reference_workload(SEED, STEPS)
    assert first == second  # bit-exact: same seed, same platform, same op order


def test_reference_workload_changes_with_seed() -> None:
    assert run_reference_workload(13, STEPS) != run_reference_workload(29, STEPS)


def test_reference_workload_matches_recorded_constant() -> None:
    # Regression guard: the recorded 50-step loss for seed 13 must not drift.
    assert run_reference_workload(SEED, STEPS) == pytest.approx(REFERENCE_LOSS_SEED_13, abs=1e-12)


def test_reference_workload_validates_arguments() -> None:
    with pytest.raises(ReproductionError):
        run_reference_workload(-1, STEPS)
    with pytest.raises(ReproductionError):
        run_reference_workload(True, STEPS)  # type: ignore[arg-type]
    with pytest.raises(ReproductionError):
        run_reference_workload(SEED, 0)
    with pytest.raises(ReproductionError):
        run_reference_workload(SEED, 1.5)  # type: ignore[arg-type]


def test_verify_resume_passes_with_exact_recorded_loss() -> None:
    recorded = run_reference_workload(SEED, STEPS)
    result = verify_resume(seed=SEED, replay_steps=STEPS, recorded_loss=recorded)
    assert result.passed
    assert result.abs_diff == 0.0
    assert result.steps == 50
    assert result.tolerance == 1e-4
    assert result.reproduced_loss == pytest.approx(recorded)


def test_verify_resume_passes_within_tolerance() -> None:
    recorded = run_reference_workload(SEED, STEPS) + 5e-5
    result = verify_resume(seed=SEED, replay_steps=STEPS, recorded_loss=recorded)
    assert result.passed
    assert result.abs_diff == pytest.approx(5e-5)


def test_verify_resume_raises_on_mismatch_beyond_tolerance() -> None:
    recorded = run_reference_workload(SEED, STEPS) + 1e-3
    with pytest.raises(ReproductionMismatchError) as excinfo:
        verify_resume(seed=SEED, replay_steps=STEPS, recorded_loss=recorded)
    assert excinfo.value.result is not None
    assert not excinfo.value.result.passed
    assert excinfo.value.key == "loss"


def test_verify_resume_requires_exactly_one_recorded_source(tmp_path: Path) -> None:
    with pytest.raises(ReproductionError):
        verify_resume(seed=SEED)
    with pytest.raises(ReproductionError):
        verify_resume(
            seed=SEED, recorded_loss=0.1, recorded_path=tmp_path / "log.jsonl"
        )


def test_verify_resume_validates_tolerance() -> None:
    with pytest.raises(ReproductionError):
        verify_resume(seed=SEED, recorded_loss=0.1, tolerance=0.0)
    with pytest.raises(ReproductionError):
        verify_resume(seed=SEED, recorded_loss=0.1, tolerance=float("nan"))
    with pytest.raises(ReproductionError):
        verify_resume(seed=SEED, recorded_loss=0.1, tolerance="1e-4")  # type: ignore[arg-type]


def test_verify_resume_rejects_non_finite_recorded_loss() -> None:
    with pytest.raises(ReproductionError):
        verify_resume(seed=SEED, recorded_loss=float("inf"))


def test_verify_resume_reads_recorded_loss_from_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "metrics.jsonl"
    recorded = run_reference_workload(SEED, STEPS)
    logger = JsonlLogger(log_path, run_id="repro-run")
    logger.log(event="step", step=0, split="train", metrics={"loss": 1.9}, wall_clock_seconds=0.01)
    logger.log(event="step", step=STEPS, split="train", metrics={"loss": recorded},
               wall_clock_seconds=0.01)

    result = verify_resume(seed=SEED, replay_steps=STEPS, recorded_path=log_path)

    assert result.passed
    assert result.recorded_loss == pytest.approx(recorded)


def test_verify_resume_uses_highest_step_loss_in_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "metrics.jsonl"
    recorded = run_reference_workload(SEED, STEPS)
    logger = JsonlLogger(log_path, run_id="repro-run")
    # Write the final loss first, then a stale lower-step loss; the check must
    # prefer the highest step regardless of file order.
    logger.log(event="step", step=STEPS, split="train", metrics={"loss": recorded})
    logger.log(event="step", step=1, split="train", metrics={"loss": 12345.0})

    result = verify_resume(seed=SEED, replay_steps=STEPS, recorded_path=log_path)
    assert result.passed


def test_verify_resume_metric_set_key_is_read(tmp_path: Path) -> None:
    log_path = tmp_path / "metrics.jsonl"
    recorded = run_reference_workload(SEED, STEPS)
    logger = JsonlLogger(log_path, run_id="repro-run")
    logger.log(event="eval", step=STEPS, split="eval", metric_set={"loss": recorded})

    result = verify_resume(seed=SEED, replay_steps=STEPS, recorded_path=log_path)
    assert result.passed


def test_verify_resume_missing_recorded_file(tmp_path: Path) -> None:
    with pytest.raises(ReproductionError):
        verify_resume(seed=SEED, recorded_path=tmp_path / "absent.jsonl")


def test_verify_resume_log_without_loss_metric(tmp_path: Path) -> None:
    log_path = tmp_path / "metrics.jsonl"
    logger = JsonlLogger(log_path, run_id="repro-run")
    logger.log(event="eval", step=STEPS, split="dev", metrics={"accuracy": 0.7})

    with pytest.raises(ReproductionError) as excinfo:
        verify_resume(seed=SEED, recorded_path=log_path)
    assert excinfo.value.key == "loss"


def test_verify_resume_default_is_50_steps_and_tolerance() -> None:
    recorded = run_reference_workload(SEED, 50)
    result = verify_resume(seed=SEED, recorded_loss=recorded)
    assert result.steps == 50
    assert result.tolerance == pytest.approx(1e-4)


def test_result_to_dict_round_trip() -> None:
    recorded = run_reference_workload(SEED, STEPS)
    result = verify_resume(seed=SEED, recorded_loss=recorded)
    payload = result.to_dict()
    assert set(payload) == {
        "seed",
        "steps",
        "recorded_loss",
        "reproduced_loss",
        "abs_diff",
        "tolerance",
        "passed",
    }
