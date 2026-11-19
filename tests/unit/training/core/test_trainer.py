"""Unit tests for src.training.core.trainer — FP16 Trainer.

Covers the TASK-12 acceptance criteria:

* checkpoints every ``checkpoint_every`` steps and at epoch end, preserving
  the sampler offset, resuming to the same loss trajectory (FR-TRAIN-02);
* non-finite gradients skip the optimizer step, increment the counter and do
  not advance the global step counter (E5 / FR-TRAIN-04);
* a checkpoint whose file hash mismatches is rejected with a typed error and
  never loaded (E6);
* model, optimizer, scaler and scheduler state round-trip with byte-equal
  tensors (FR-TRAIN-02).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conftest import NanLogitModel, StubRapportModel, make_batch, make_model  # noqa: E402

from src.ops.logging import JsonlLogger, iter_records  # noqa: E402
from src.training.core import (  # noqa: E402
    CheckpointBeyondHorizonError,
    CheckpointHashMismatchError,
    CheckpointNotFoundError,
    Trainer,
    TrainerError,
    TrainingConfig,
    TrainingDivergedError,
    checkpoint_path,
    sidecar_path,
    verify_checkpoint,
)


def _config(tmp_path: Path, **overrides) -> TrainingConfig:
    values = {
        "steps": 100,
        "epochs": 1,
        "fp16": False,
        "checkpoint_dir": str(tmp_path / "ckpts"),
    }
    values.update(overrides)
    return TrainingConfig(**values)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# construction / wiring
# ---------------------------------------------------------------------------


def test_trainer_builds_adamw_with_split_weight_decay(tmp_path: Path) -> None:
    trainer = Trainer(_config(tmp_path), make_model())
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    groups = trainer.optimizer.param_groups
    decay_wd = {g["weight_decay"] for g in groups}
    assert 0.0 in decay_wd
    assert any(abs(wd - 0.01) < 1e-12 for wd in decay_wd)
    norms_in_decay = any(
        "norm" in n or "bias" in n or "embedding" in n
        for group in groups
        if group["weight_decay"] > 0
        for p in group["params"]
        for n, q in trainer.model.named_parameters()
        if q is p
    )
    assert not norms_in_decay
    # AdamW betas/eps from the design (§3.1).
    assert trainer.optimizer.param_groups[0]["betas"] == (0.9, 0.98)
    assert trainer.optimizer.param_groups[0]["eps"] == pytest.approx(1.0e-8)


def test_scheduler_is_warmup_then_cosine(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=100, warmup_ratio=0.05)
    trainer = Trainer(config, make_model())
    lrs = [trainer.optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        trainer.optimizer.step()  # steady state: optimizer before scheduler
        trainer.scheduler.step()
        lrs.append(trainer.optimizer.param_groups[0]["lr"])
    peak = max(lrs)
    peak_index = lrs.index(peak)
    assert peak_index == 4  # 5% of 100 steps warmup; LambdaLR fn(4) == 1.0
    assert lrs[-1] == pytest.approx(config.min_learning_rate, rel=1e-4)
    # Warmup is monotonically non-decreasing up to the peak.
    assert all(a <= b for a, b in zip(lrs[: peak_index + 1], lrs[1 : peak_index + 1]))


def test_scaler_configures_design_values(tmp_path: Path) -> None:
    config = _config(tmp_path, fp16=True)
    trainer = Trainer(config, make_model())
    state = trainer.scaler.state_dict()
    assert state["scale"] == float(2**16)
    assert state["backoff_factor"] == pytest.approx(0.5)
    assert state["growth_interval"] == 2000


def test_config_mapping_accepted(tmp_path: Path) -> None:
    trainer = Trainer(
        {"steps": 10, "checkpoint_every": 2, "checkpoint_dir": str(tmp_path / "c")},
        make_model(),
    )
    assert trainer.config.steps == 10
    assert trainer.config.checkpoint_every == 2


def test_missing_model_raises(tmp_path: Path) -> None:
    trainer = Trainer(_config(tmp_path))
    with pytest.raises(TrainerError):
        trainer.fit(iter([]))


# ---------------------------------------------------------------------------
# checkpoint cadence, epoch end, sampler offset
# ---------------------------------------------------------------------------


def test_checkpoint_every_n_steps_and_epoch_end(tmp_path: Path) -> None:
    config = _config(tmp_path, checkpoint_every=2, steps=10)
    trainer = Trainer(config, make_model())
    batches = [make_batch(1000 + i) for i in range(5)]
    for batch in batches:
        trainer.train_batch(batch)
    ckpt_dir = Path(config.checkpoint_dir)
    assert checkpoint_path(ckpt_dir, 2).exists()
    assert checkpoint_path(ckpt_dir, 4).exists()
    assert not checkpoint_path(ckpt_dir, 6).exists()
    sidecar = sidecar_path(checkpoint_path(ckpt_dir, 2))
    assert sidecar.exists()
    verify_checkpoint(checkpoint_path(ckpt_dir, 2))  # does not raise
    # No .tmp litter.
    assert not list(ckpt_dir.glob("*.tmp"))


def test_epoch_end_checkpoint_written(tmp_path: Path) -> None:
    config = _config(tmp_path, checkpoint_every=500, epochs=1, steps=10)
    trainer = Trainer(config, make_model())
    batches = [make_batch(2000 + i) for i in range(3)]
    trainer.fit(iter(batches), epochs=1)
    assert checkpoint_path(config.checkpoint_dir, 3).exists()
    assert trainer.epoch == 1


def test_sampler_offset_advances_and_persists(tmp_path: Path) -> None:
    class ToySampler:
        def __init__(self) -> None:
            self.offset = 0

    config = _config(tmp_path, steps=10)
    sampler = ToySampler()
    trainer = Trainer(config, make_model(), sampler=sampler)
    for i, seed in enumerate((3000, 3001, 3002)):
        result = trainer.train_batch(make_batch(seed))
        assert result.sampler_offset == i + 1
    assert sampler.offset == 3
    path = trainer.save()

    # Fresh trainer with a fresh sampler resumes at the stored offset.
    fresh_sampler = ToySampler()
    resumer = Trainer(config, make_model(seed=99), sampler=fresh_sampler)
    resumer.load(path)
    assert resumer.sampler_offset == 3
    assert fresh_sampler.offset == 3


def test_sampler_offset_preserved_for_skipped_batches(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10, nonfinite_per_epoch=100)
    trainer = Trainer(config, StubRapportModel())
    batch = make_batch(4000)
    result = trainer.train_batch(batch)
    assert result.skipped
    assert trainer.sampler_offset == 1  # consumed even though it was skipped


def test_checkpoint_payload_contents(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10)
    trainer = Trainer(config, make_model())
    for i in range(2):
        trainer.train_batch(make_batch(5000 + i))
    path = trainer.save()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in (
        "model",
        "optimizer",
        "scaler",
        "scheduler",
        "step",
        "epoch",
        "sampler_offset",
        "rng",
    ):
        assert key in payload
    assert set(payload["rng"]) >= {"python", "numpy", "torch", "cuda"}
    assert payload["step"] == 2


# ---------------------------------------------------------------------------
# hash mismatch / beyond horizon (E6, load-time refusal)
# ---------------------------------------------------------------------------


def test_hash_mismatch_rejected_and_never_loaded(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10)
    trainer = Trainer(config, make_model())
    trainer.train_batch(make_batch(6000))
    path = trainer.save()
    # Destroy the payload entirely; only the sidecar describes the original.
    path.write_bytes(b"definitely not a checkpoint")

    fresh = Trainer(config, make_model(seed=77))
    before = {k: v.clone() for k, v in fresh.model.state_dict().items()}
    with pytest.raises(CheckpointHashMismatchError):
        fresh.load(path)
    after = fresh.model.state_dict()
    for key, value in before.items():
        assert torch.equal(value, after[key])


def test_beyond_horizon_refused_and_never_loaded(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=4)
    trainer = Trainer(config, make_model())
    for i in range(4):
        trainer.train_batch(make_batch(7000 + i))
    trainer.save()  # checkpoint_step4.pt, horizon is 4

    fresh = Trainer(config, make_model(seed=88))
    with torch.no_grad():
        for param in fresh.model.parameters():
            param.zero_()
    with pytest.raises(CheckpointBeyondHorizonError) as excinfo:
        fresh.load(str(tmp_path / "ckpts"))
    assert excinfo.value.step == 4
    assert excinfo.value.steps == 4
    # Refusal happened before deserialization: weights are still zeros.
    assert all(
        bool((param == 0).all()) for param in fresh.model.parameters()
    )


def test_load_missing_checkpoint_raises(tmp_path: Path) -> None:
    trainer = Trainer(_config(tmp_path), make_model())
    with pytest.raises(CheckpointNotFoundError):
        trainer.load(tmp_path / "nope.pt")
    with pytest.raises(CheckpointNotFoundError):
        trainer.load(tmp_path / "empty-dir")


def test_load_restores_step_epoch_counters(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10)
    trainer = Trainer(config, make_model())
    trainer.epoch = 3
    trainer.sampler_offset_restore(41)
    trainer.train_batch(make_batch(8000))
    path = trainer.save()

    fresh = Trainer(config, make_model(seed=55))
    fresh.load(path)
    assert fresh.step == trainer.step
    assert fresh.epoch == 3
    assert fresh.sampler_offset == trainer.sampler_offset
    assert fresh.nonfinite_steps == trainer.nonfinite_steps


# ---------------------------------------------------------------------------
# state round-trip: byte-equal tensors (FR-TRAIN-02)
# ---------------------------------------------------------------------------


def _assert_state_dicts_equal(a, b, path: str = "state") -> None:
    assert type(a) is type(b) or isinstance(a, dict)
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys differ: {set(a) ^ set(b)}"
        for key in a:
            _assert_state_dicts_equal(a[key], b[key], f"{path}.{key}")
        return
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f"{path}: tensors differ"
        return
    assert a == b, f"{path}: {a!r} != {b!r}"


def test_model_optimizer_scaler_scheduler_round_trip_bytes_equal(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10, fp16=True)
    _seed_all(1234)
    trainer = Trainer(config, make_model(seed=1234))
    for i in range(3):
        trainer.train_batch(make_batch(9000 + i))
    path = trainer.save()

    fresh = Trainer(config, make_model(seed=9999))
    fresh.load(path)

    _assert_state_dicts_equal(
        trainer.model.state_dict(), fresh.model.state_dict(), "model"
    )
    _assert_state_dicts_equal(
        trainer.optimizer.state_dict(), fresh.optimizer.state_dict(), "optimizer"
    )
    assert trainer.scaler.state_dict() == fresh.scaler.state_dict()
    _assert_state_dicts_equal(
        trainer.scheduler.state_dict(), fresh.scheduler.state_dict(), "scheduler"
    )


def test_rng_state_round_trip(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10)
    _seed_all(4321)
    trainer = Trainer(config, make_model(seed=4321))
    for i in range(2):
        trainer.train_batch(make_batch(9100 + i))
    path = trainer.save()
    # save() consumes no RNG, so the streams below continue from the saved state.
    expected_torch = torch.rand(4)
    expected_python = random.random()
    expected_numpy = np.random.rand(4)

    fresh = Trainer(config, make_model(seed=1111))
    fresh.load(path)
    assert torch.equal(torch.rand(4), expected_torch)
    assert random.random() == expected_python
    np.testing.assert_array_equal(np.random.rand(4), expected_numpy)


def test_scheduler_last_epoch_round_trip(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=50)
    trainer = Trainer(config, make_model())
    for i in range(6):
        trainer.train_batch(make_batch(9300 + i))
    path = trainer.save()
    fresh = Trainer(config, make_model(seed=314))
    fresh.load(path)
    assert (
        fresh.scheduler.state_dict()["last_epoch"]
        == trainer.scheduler.state_dict()["last_epoch"]
    )
    assert fresh.scheduler.state_dict()["_last_lr"] == trainer.scheduler.state_dict()["_last_lr"]


# ---------------------------------------------------------------------------
# resume to the same loss trajectory (FR-TRAIN-02)
# ---------------------------------------------------------------------------


def test_kill_resume_reproduces_loss_trajectory(tmp_path: Path) -> None:
    batches = [make_batch(10000 + i) for i in range(50)]
    config = _config(tmp_path, steps=100, epochs=1)

    # Uninterrupted reference run.
    _seed_all(13)
    reference = Trainer(config, make_model(seed=13))
    reference_losses: list[float | None] = []
    reference_skipped: list[bool] = []
    for batch in batches:
        result = reference.train_batch(batch)
        reference_losses.append(result.loss)
        reference_skipped.append(result.skipped)

    # Interrupted run: stop after 30 batches and checkpoint.
    _seed_all(13)
    interrupted = Trainer(config, make_model(seed=13))
    for batch in batches[:30]:
        interrupted.train_batch(batch)
    interrupted.save()
    assert interrupted.sampler_offset == 30

    # Resumed run: fresh process state, different model init, reload.
    _seed_all(999)
    resumed = Trainer(config, make_model(seed=999))
    resumed.load(Path(config.checkpoint_dir))
    assert resumed.sampler_offset == 30
    resumed_losses: list[float | None] = []
    resumed_skipped: list[bool] = []
    for batch in batches[30:]:
        result = resumed.train_batch(batch)
        resumed_losses.append(result.loss)
        resumed_skipped.append(result.skipped)

    assert resumed_skipped == reference_skipped[30:]
    for index, (expected, actual) in enumerate(zip(reference_losses[30:], resumed_losses)):
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            assert abs(expected - actual) <= 1e-4, f"step {30 + index} loss diverged"


def test_resume_with_fp16_autocast_matches_uninterrupted(tmp_path: Path) -> None:
    """FP16 (emulated on CPU) resume keeps the skip pattern and losses aligned."""
    batches = [make_batch(11000 + i) for i in range(8)]
    config = _config(tmp_path, steps=100, epochs=1, fp16=True)

    _seed_all(13)
    reference = Trainer(config, make_model(seed=13))
    reference_results = [reference.train_batch(batch) for batch in batches]

    _seed_all(13)
    interrupted = Trainer(config, make_model(seed=13))
    for batch in batches[:4]:
        interrupted.train_batch(batch)
    interrupted.save()

    _seed_all(999)
    resumed = Trainer(config, make_model(seed=999))
    resumed.load(Path(config.checkpoint_dir))
    resumed_results = [resumed.train_batch(batch) for batch in batches[4:]]

    for expected, actual in zip(reference_results[4:], resumed_results):
        assert expected.skipped == actual.skipped
        if expected.loss is None:
            assert actual.loss is None
        else:
            assert actual.loss is not None
            assert abs(expected.loss - actual.loss) <= 1e-4


# ---------------------------------------------------------------------------
# non-finite gradient / loss handling (E5 / FR-TRAIN-04)
# ---------------------------------------------------------------------------


def test_nonfinite_gradient_skips_step_and_counts(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10, nonfinite_per_epoch=100)
    trainer = Trainer(config, StubRapportModel())
    model_params_before = [p.detach().clone() for p in trainer.model.parameters()]
    scheduler_epoch_before = trainer.scheduler.state_dict()["last_epoch"]

    result = trainer.train_batch(make_batch(12000))

    assert result.skipped is True
    assert result.reason == "gradient_not_finite"
    assert trainer.step == 0  # global step counter NOT advanced
    assert trainer.nonfinite_steps == 1
    assert trainer.nonfinite_steps_epoch == 1
    assert trainer.scheduler.state_dict()["last_epoch"] == scheduler_epoch_before
    for param, snapshot in zip(trainer.model.parameters(), model_params_before):
        assert torch.equal(param, snapshot)


def test_nonfinite_loss_skips_step_and_counts(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10, nonfinite_per_epoch=100)
    trainer = Trainer(config, NanLogitModel())
    result = trainer.train_batch(make_batch(12100))
    assert result.skipped is True
    assert result.reason is not None and result.reason.startswith("loss_not_finite:")
    assert trainer.step == 0
    assert trainer.nonfinite_steps == 1
    assert trainer.sampler_offset == 1


def test_too_many_nonfinite_steps_in_epoch_raises_diverged(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=100, nonfinite_per_epoch=3)
    trainer = Trainer(config, StubRapportModel())
    batch = make_batch(12200)
    for _ in range(3):
        trainer.train_batch(batch)
    with pytest.raises(TrainingDivergedError) as excinfo:
        trainer.train_batch(batch)
    assert excinfo.value.nonfinite_steps == 4
    assert excinfo.value.epoch == 0


def test_nonfinite_counter_resets_each_epoch(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=100, nonfinite_per_epoch=2, epochs=2)
    trainer = Trainer(config, StubRapportModel())
    # Two inf-gradient steps: within the limit of 2 for epoch 0.
    logger_batches = [make_batch(12300 + i) for i in range(2)]
    for batch in logger_batches:
        trainer.train_batch(batch)
    assert trainer.nonfinite_steps_epoch == 2
    # Simulate an epoch boundary.
    trainer.epoch += 1
    trainer.nonfinite_steps_epoch = 0
    result = trainer.train_batch(make_batch(12302))
    assert result.skipped
    assert trainer.nonfinite_steps_epoch == 1


# ---------------------------------------------------------------------------
# JSONL logging hooks
# ---------------------------------------------------------------------------


def test_one_jsonl_record_per_step(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    logger = JsonlLogger(log_path, run_id="task-12-test")
    config = _config(tmp_path, steps=10)
    trainer = Trainer(config, make_model(), logger=logger)
    for i in range(3):
        trainer.train_batch(make_batch(13000 + i))

    records = list(iter_records(log_path))
    assert len(records) == 3
    for index, record in enumerate(records):
        assert record["event"] == "step"
        assert record["split"] == "train"
        assert isinstance(record["step"], int)
        assert isinstance(record["wall_clock_seconds"], float)
        assert record["run_id"] == "task-12-test"
        metrics = record["metrics"]
        assert metrics["skipped"] is False
        assert np.isfinite(metrics["loss"])
        assert metrics["nonfinite_steps"] == 0
        assert metrics["sampler_offset"] == index + 1


def test_skipped_step_emits_record_without_loss(tmp_path: Path) -> None:
    log_path = tmp_path / "run-skip.jsonl"
    logger = JsonlLogger(log_path, run_id="task-12-skip")
    config = _config(tmp_path, steps=10, nonfinite_per_epoch=100)
    trainer = Trainer(config, StubRapportModel(), logger=logger)
    trainer.train_batch(make_batch(13100))
    records = list(iter_records(log_path))
    assert len(records) == 1
    record = records[0]
    assert record["metrics"]["skipped"] is True
    assert record["metrics"]["reason"] == "gradient_not_finite"
    assert "loss" not in record["metrics"]


def test_epoch_end_event_emitted(tmp_path: Path) -> None:
    log_path = tmp_path / "run-epoch.jsonl"
    logger = JsonlLogger(log_path, run_id="task-12-epoch")
    config = _config(tmp_path, steps=10, epochs=2, nonfinite_per_epoch=100)
    trainer = Trainer(config, StubRapportModel(), logger=logger)
    trainer.fit(iter([make_batch(13200)]), epochs=2)
    records = list(iter_records(log_path))
    epoch_records = [r for r in records if r["event"] == "epoch_end"]
    assert len(epoch_records) == 2
    assert all(r["split"] == "train" for r in epoch_records)


def test_records_are_valid_json_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "run-json.jsonl"
    logger = JsonlLogger(log_path, run_id="task-12-json")
    trainer = Trainer(_config(tmp_path, steps=10), make_model(), logger=logger)
    trainer.train_batch(make_batch(13300))
    with open(log_path, encoding="utf-8") as handle:
        lines = [line for line in handle.read().splitlines() if line.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["step"] == 1


def test_no_logger_is_a_noop_but_steps_still_counted(tmp_path: Path) -> None:
    trainer = Trainer(_config(tmp_path, steps=10), make_model())
    result = trainer.train_batch(make_batch(13400))
    assert trainer.step == 1
    assert result.loss is not None


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


def test_fit_returns_history_and_counts_epochs(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10, epochs=2)
    trainer = Trainer(config, make_model())
    batches = [make_batch(14000 + i) for i in range(3)]
    history = trainer.fit(batches, epochs=2)  # re-iterable, like a DataLoader
    assert len(history) == 6
    assert trainer.epoch == 2
