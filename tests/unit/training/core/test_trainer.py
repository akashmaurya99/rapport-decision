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

import hashlib
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

from conftest import (  # noqa: E402
    NanGradModel,
    NanLogitModel,
    StubRapportModel,
    make_batch,
    make_model,
)

from src.ops.logging import JsonlLogger, iter_records  # noqa: E402
from src.training.core import (  # noqa: E402
    CheckpointBeyondHorizonError,
    CheckpointError,
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
from src.training.core.trainer import _is_head_param  # noqa: E402
from src.training.multitask import DEFAULT_TASK_NAMES, WEIGHT_LOG_EVENT  # noqa: E402


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


def test_learned_weighting_resume_is_bit_exact(tmp_path: Path) -> None:
    """log_s must not accumulate stale gradients across steps or resume.

    Regression: train_batch used model.zero_grad(), which misses log_s because
    it is owned by the optimizer/weighter, not by model.parameters(). The
    accumulated gradient made the Kendall objective depend on step history and
    broke kill/resume equality (measured 2.2e-4 drift over 12 steps).
    """
    batches = [make_batch(11000 + i) for i in range(8)]
    config = _config(tmp_path, steps=100, epochs=1, weight_mode="learned")

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

    assert resumed.weighter is not None and reference.weighter is not None
    assert torch.equal(resumed.weighter.log_s, reference.weighter.log_s)
    for expected, actual in zip(reference_results[4:], resumed_results):
        assert expected.loss is not None and actual.loss is not None
        assert abs(expected.loss - actual.loss) <= 1e-6


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


# ---------------------------------------------------------------------------
# FIX 1 — resume/horizon defects in fit()
# ---------------------------------------------------------------------------


def _rewrite_payload(path: Path, mutate) -> None:
    """Edit a checkpoint payload while keeping its sha256 sidecar consistent."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mutate(payload)
    torch.save(payload, path)
    sidecar_path(path).write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n", encoding="utf-8"
    )


def test_fit_resume_runs_only_remaining_epochs(tmp_path: Path) -> None:
    """Default ``fit()`` must not replay epochs already completed (1+2==3)."""
    config = _config(tmp_path, steps=100, epochs=3, checkpoint_every=100)
    batches = [make_batch(19000 + i) for i in range(3)]

    first = Trainer(config, make_model())
    first.fit(batches, epochs=1)
    assert (first.step, first.epoch) == (3, 1)

    first_ckpt = checkpoint_path(Path(config.checkpoint_dir), 3)
    resumed = Trainer(config, make_model(seed=99))
    resumed.load(first_ckpt)
    assert resumed.epoch == 1

    history = resumed.fit(batches, epochs=None)
    assert (resumed.step, resumed.epoch) == (9, 3)
    assert len(history) == 6  # 2 remaining epochs x 3 batches (not 3 x 3)

    final_ckpt = checkpoint_path(Path(config.checkpoint_dir), 9)
    final_payload = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    assert final_payload["epoch"] == 3  # epoch-end save records the completed epoch


def test_fit_stops_at_configured_step_horizon(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=2, epochs=5, checkpoint_every=100)
    trainer = Trainer(config, make_model())
    history = trainer.fit([make_batch(20000 + i) for i in range(6)], epochs=5)
    assert len(history) == 2
    assert (trainer.step, trainer.epoch) == (2, 1)


# ---------------------------------------------------------------------------
# FIX 2 — label-free / fully-masked batches are typed skips
# ---------------------------------------------------------------------------


def test_label_free_batch_is_typed_skip_and_logged(tmp_path: Path) -> None:
    log_path = tmp_path / "no-labels.jsonl"
    logger = JsonlLogger(log_path, run_id="no-labels")
    config = _config(tmp_path, steps=10)
    trainer = Trainer(config, make_model(), logger=logger)
    full = make_batch(18000)
    label_free = {
        key: full[key] for key in ("input_ids", "attention_mask", "option_mask", "task_ids")
    }

    result = trainer.train_batch(label_free)

    assert result.skipped is True
    assert result.reason == "no_active_terms"
    assert result.active is not None and not any(result.active.values())
    assert (trainer.step, trainer.nonfinite_steps, trainer.sampler_offset) == (0, 0, 1)
    records = list(iter_records(log_path))
    assert len(records) == 1
    assert records[0]["metrics"]["skipped"] is True
    assert records[0]["metrics"]["reason"] == "no_active_terms"
    assert "loss" not in records[0]["metrics"]


def test_all_chosen_unset_without_other_labels_is_skipped(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10)
    trainer = Trainer(config, make_model())
    full = make_batch(18001)
    batch = {key: full[key] for key in ("input_ids", "attention_mask", "option_mask", "task_ids")}
    batch["chosen_index"] = torch.full((2,), -1, dtype=torch.long)
    result = trainer.train_batch(batch)
    assert result.skipped is True and result.reason == "no_active_terms"
    assert trainer.step == 0 and trainer.nonfinite_steps == 0


def test_all_chosen_unset_with_other_labels_still_steps(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10)
    trainer = Trainer(config, make_model())
    batch = make_batch(18002)
    batch["chosen_index"] = torch.full((2,), -1, dtype=torch.long)
    result = trainer.train_batch(batch)
    assert result.skipped is False
    assert result.active is not None
    assert result.active["choice"] is False and result.active["confidence"] is False
    assert result.active["score"] is True
    assert trainer.step == 1


# ---------------------------------------------------------------------------
# FIX 7 — one global-norm sync replaces per-parameter isfinite() checks
# ---------------------------------------------------------------------------


def test_nan_gradient_skips_step_without_counter_regression(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10, nonfinite_per_epoch=100)
    trainer = Trainer(config, NanGradModel())
    before = [p.detach().clone() for p in trainer.model.parameters()]
    result = trainer.train_batch(make_batch(17000))
    assert result.skipped is True and result.reason == "gradient_not_finite"
    assert (trainer.step, trainer.nonfinite_steps, trainer.nonfinite_steps_epoch) == (0, 1, 1)
    for param, snapshot in zip(trainer.model.parameters(), before):
        assert torch.equal(param, snapshot)


# ---------------------------------------------------------------------------
# FIX 3 — checkpoint validation gaps
# ---------------------------------------------------------------------------


def test_load_refuses_mismatched_config(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10)
    trainer = Trainer(config, make_model())
    trainer.train_batch(make_batch(21000))
    path = trainer.save()

    other = Trainer(_config(tmp_path, steps=10, weight_decay=0.02), make_model(seed=99))
    before = {k: v.clone() for k, v in other.model.state_dict().items()}
    with pytest.raises(CheckpointError, match="different training config") as excinfo:
        other.load(path)
    assert excinfo.value.key == "config"
    for key, value in before.items():
        assert torch.equal(value, other.model.state_dict()[key])


def test_load_missing_or_malformed_scaler_is_typed_error(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10)
    trainer = Trainer(config, make_model())
    trainer.train_batch(make_batch(21100))
    path = trainer.save()

    _rewrite_payload(path, lambda payload: payload.pop("scaler", None))
    fresh = Trainer(config, make_model(seed=99))
    with pytest.raises(CheckpointError, match="scaler") as excinfo:
        fresh.load(path)
    assert excinfo.value.key == "scaler"

    _rewrite_payload(path, lambda payload: payload.__setitem__("scaler", "not-a-state"))
    with pytest.raises(CheckpointError, match="scaler") as excinfo:
        fresh.load(path)
    assert excinfo.value.key == "scaler"


# ---------------------------------------------------------------------------
# FIX 4 — parameter-group LR (head_lr)
# ---------------------------------------------------------------------------


def test_head_lr_splits_head_and_trunk_parameters(tmp_path: Path) -> None:
    config = _config(tmp_path, learning_rate=3.0e-4, head_lr=1.0e-3)
    model = make_model()
    trainer = Trainer(config, model)
    name_by_param = {id(p): name for name, p in model.named_parameters()}

    # LambdaLR already applied the warmup factor to param_groups["lr"].
    factor = trainer.optimizer.param_groups[0]["lr"] / config.learning_rate
    head_groups = [
        g for g in trainer.optimizer.param_groups
        if abs(g["lr"] - factor * 1.0e-3) < 1e-12
    ]
    body_groups = [
        g for g in trainer.optimizer.param_groups
        if abs(g["lr"] - factor * 3.0e-4) < 1e-12
    ]
    assert head_groups and body_groups

    head_names = {name_by_param[id(p)] for g in head_groups for p in g["params"]}
    body_names = {name_by_param[id(p)] for g in body_groups for p in g["params"]}
    assert head_names and all(_is_head_param(name) for name in head_names)
    assert all(not _is_head_param(name) for name in body_names)
    assert head_names | body_names == set(name_by_param.values())
    # Norms/biases still get no weight decay in both groups.
    assert any(g["weight_decay"] == 0.0 for g in head_groups)
    assert any(g["weight_decay"] == 0.0 for g in body_groups)


def test_head_lr_none_keeps_single_learning_rate(tmp_path: Path) -> None:
    config = _config(tmp_path, learning_rate=3.0e-4, head_lr=None)
    trainer = Trainer(config, make_model())
    assert len({round(g["lr"], 12) for g in trainer.optimizer.param_groups}) == 1


# ---------------------------------------------------------------------------
# FIX 5 — determinism is applied at fit() start, never in train_batch()
# ---------------------------------------------------------------------------


def test_fit_applies_seed_and_determinism(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[int, str]] = []

    def fake_set_seed(seed: int, determinism: str) -> None:
        calls.append((seed, determinism))
        torch.manual_seed(777)  # sentinel so we can observe the reseed

    monkeypatch.setattr("src.training.core.trainer.set_seed", fake_set_seed)
    config = _config(tmp_path, steps=10, seed=77, determinism="relaxed")
    trainer = Trainer(config, make_model())

    trainer.fit([], epochs=1)  # empty loader: no RNG consumed after the seed

    assert calls == [(77, "relaxed")]
    assert torch.equal(torch.get_rng_state(), torch.Generator().manual_seed(777).get_state())


def test_fit_does_not_reseed_a_resumed_trainer(tmp_path: Path, monkeypatch) -> None:
    """load() restores RNG state; fit() must not overwrite it on resume."""
    calls: list[tuple[int, str]] = []
    monkeypatch.setattr(
        "src.training.core.trainer.set_seed",
        lambda seed, determinism: calls.append((seed, determinism)),
    )
    config = _config(tmp_path, steps=10, seed=77)
    trainer = Trainer(config, make_model())
    trainer.step = 5
    trainer.epoch = 1

    trainer.fit([], epochs=1)

    assert calls == []


def test_train_batch_does_not_reseed(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[int, str]] = []
    monkeypatch.setattr(
        "src.training.core.trainer.set_seed",
        lambda seed, determinism: calls.append((seed, determinism)),
    )
    trainer = Trainer(_config(tmp_path, steps=10), make_model())
    trainer.train_batch(make_batch(22000))
    assert calls == []


# ---------------------------------------------------------------------------
# FIX 6 — multitask weighting is wired into the trainer
# ---------------------------------------------------------------------------


def test_learned_mode_registers_log_s_in_no_decay_group(tmp_path: Path) -> None:
    config = _config(tmp_path, weight_mode="learned")
    trainer = Trainer(config, make_model())
    log_s = trainer.weighter.log_s
    assert log_s.requires_grad
    owning = [g for g in trainer.optimizer.param_groups if any(p is log_s for p in g["params"])]
    assert len(owning) == 1
    assert owning[0]["weight_decay"] == 0.0
    assert owning[0]["lr"] == trainer.optimizer.param_groups[0]["lr"]

    fixed = Trainer(_config(tmp_path, weight_mode="fixed"), make_model())
    assert not fixed.weighter.log_s.requires_grad
    assert all(
        not any(p is fixed.weighter.log_s for p in g["params"])
        for g in fixed.optimizer.param_groups
    )


def test_learned_vs_fixed_produce_different_losses_and_weight_updates(tmp_path: Path) -> None:
    fixed = Trainer(_config(tmp_path, weight_mode="fixed"), make_model(seed=13))
    learned = Trainer(_config(tmp_path, weight_mode="learned"), make_model(seed=13))
    before = learned.weighter.log_s.detach().clone()

    fixed_result = fixed.train_batch(make_batch(23000))
    learned_result = learned.train_batch(make_batch(23000))
    assert fixed_result.loss is not None and learned_result.loss is not None
    assert abs(fixed_result.loss - learned_result.loss) > 1e-4

    assert not torch.equal(learned.weighter.log_s, before)
    assert any(
        abs(learned.weighter.weights()[name] - 1.0) > 1e-6 for name in DEFAULT_TASK_NAMES
    )

    fixed_weights = fixed.weighter.weights()
    fixed.train_batch(make_batch(23001))
    assert fixed.weighter.weights() == fixed_weights
    assert torch.equal(fixed.weighter.log_s, fixed.weighter.log_s.detach().clone())


def test_weighter_state_round_trip_and_epoch_logging(tmp_path: Path, ) -> None:
    config = _config(tmp_path, weight_mode="learned", epochs=1, steps=10)
    log_path = tmp_path / "weights.jsonl"
    logger = JsonlLogger(log_path, run_id="weights-test")
    trainer = Trainer(config, make_model(), logger=logger)
    for i in range(2):
        trainer.train_batch(make_batch(23100 + i))
    path = trainer.save()
    assert "weighter" in torch.load(path, map_location="cpu", weights_only=False)

    fresh = Trainer(config, make_model(seed=99))
    fresh.load(path)
    assert torch.equal(fresh.weighter.log_s, trainer.weighter.log_s)

    trainer.fit([make_batch(23200)], epochs=1)
    records = [r for r in iter_records(log_path) if r["event"] == WEIGHT_LOG_EVENT]
    assert records
    assert set(records[0]["weights"]) == set(DEFAULT_TASK_NAMES)


# ---------------------------------------------------------------------------
# FIX 9 — checkpoint retention
# ---------------------------------------------------------------------------


def test_keep_last_n_prunes_only_old_numbered_checkpoints(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=100, checkpoint_every=1, keep_last_n=2)
    directory = Path(config.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    user_file = directory / "notes.txt"
    user_file.write_text("do not delete", encoding="utf-8")

    trainer = Trainer(config, make_model())
    for i in range(4):
        trainer.train_batch(make_batch(24000 + i))

    names = sorted(p.name for p in directory.glob("checkpoint_step*.pt"))
    assert names == ["checkpoint_step3.pt", "checkpoint_step4.pt"]
    assert not sidecar_path(checkpoint_path(directory, 2)).exists()
    assert user_file.read_text(encoding="utf-8") == "do not delete"
    assert not any(p.name.endswith(".tmp") for p in directory.iterdir())


def test_keep_last_n_none_keeps_every_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=100, checkpoint_every=1)
    trainer = Trainer(config, make_model())
    for i in range(3):
        trainer.train_batch(make_batch(24100 + i))
    assert len(list(Path(config.checkpoint_dir).glob("checkpoint_step*.pt"))) == 3


# ---------------------------------------------------------------------------
# Wave-4 regression tests: device transfer, log_s safety, mid-epoch resume
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_cpu_batch_is_transferred_to_the_trainer_device(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10, device="mps")
    trainer = Trainer(config, make_model())
    result = trainer.train_batch(make_batch(25000))  # entirely CPU-resident
    assert result.step == 1 and result.loss is not None
    assert next(trainer.model.parameters()).device.type == "mps"


@pytest.mark.parametrize("fp16", [False, True])
def test_nan_log_s_gradient_is_a_typed_skip_for_both_precisions(
    tmp_path: Path, fp16: bool
) -> None:
    config = _config(tmp_path, steps=10, fp16=fp16, weight_mode="learned")
    trainer = Trainer(config, make_model())
    before = trainer.weighter.log_s.detach().clone()
    original = trainer._weighted_total

    def poisoned(breakdown):
        return original(breakdown) + trainer.weighter.log_s.sum() * float("nan")

    trainer._weighted_total = poisoned
    result = trainer.train_batch(make_batch(25001))
    assert result.skipped is True and result.reason == "gradient_not_finite"
    assert result.step == 0 and result.nonfinite_steps == 1
    assert torch.equal(trainer.weighter.log_s, before)
    assert torch.isfinite(trainer.weighter.log_s).all()


def test_mid_epoch_fit_resume_skips_already_consumed_batches(tmp_path: Path) -> None:
    batches = [dict(make_batch(25100 + index), tag=index) for index in range(4)]
    config = _config(tmp_path, steps=100, epochs=1)
    first = Trainer(config, make_model())
    for batch in batches[:2]:
        first.train_batch(batch)
    first.save()

    resumed = Trainer(config, make_model(seed=99))
    resumed.load(Path(config.checkpoint_dir))
    seen: list[int] = []
    original = resumed.train_batch

    def recording(batch):
        seen.append(batch["tag"])
        return original(batch)

    resumed.train_batch = recording
    resumed.fit(batches)
    assert seen == [2, 3]
    assert resumed.step == 4 and resumed.epoch == 1


def test_custom_optimizer_registers_learned_log_s(tmp_path: Path) -> None:
    model = make_model()
    custom = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(_config(tmp_path, weight_mode="learned"), model, optimizer=custom)
    assert any(
        parameter is trainer.weighter.log_s
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    )
    before = trainer.weighter.log_s.detach().clone()
    trainer.train_batch(make_batch(25002))
    assert not torch.equal(before, trainer.weighter.log_s)


def test_load_rejects_malformed_rng_without_mutating_state(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=10)
    trainer = Trainer(config, make_model())
    trainer.train_batch(make_batch(25003))
    path = trainer.save()
    _rewrite_payload(path, lambda payload: payload.update({"rng": "not-a-mapping"}))

    other = Trainer(config, make_model(seed=99))
    before = {key: value.clone() for key, value in other.model.state_dict().items()}
    with pytest.raises(CheckpointError, match="rng"):
        other.load(path)
    assert all(torch.equal(before[key], value) for key, value in other.model.state_dict().items())


def test_checkpoint_relocation_is_allowed_but_optimizer_mismatch_is_not(tmp_path: Path) -> None:
    first = Trainer(_config(tmp_path / "a", steps=10), make_model())
    first.train_batch(make_batch(25004))
    path = first.save()
    moved = Trainer(_config(tmp_path / "b", steps=10), make_model(seed=99))
    moved.load(path)
    assert moved.step == 1
    assert Path(moved.config.checkpoint_dir) == tmp_path / "b" / "ckpts"
    mismatch = Trainer(_config(tmp_path / "b", steps=10, learning_rate=1e-2), make_model())
    with pytest.raises(CheckpointError, match="different training config"):
        mismatch.load(path)
