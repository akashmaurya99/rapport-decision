"""Unit tests for TASK-14 — adaptive multi-task weighting (TaskWeighter)."""

from __future__ import annotations

import json
import math

import pytest
import torch

from src.ops.logging import JsonlLogger, iter_records
from src.ops.registry import RunRegistry
from src.training.multitask import (
    COLLAPSE_FLOOR_EPOCHS,
    DEFAULT_TASK_NAMES,
    WEIGHT_COLLAPSE_EVENT,
    WEIGHT_LOG_EVENT,
    TaskWeighter,
)

TOL = 1e-6


class RecordingLogger:
    """Minimal stub: captures every ``log(**kwargs)`` call as a dict."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def log(self, **kwargs):  # noqa: ANN001 - stub
        self.calls.append(kwargs)
        return kwargs


def _make_weighter(**kwargs) -> TaskWeighter:
    kwargs.setdefault("num_tasks", len(DEFAULT_TASK_NAMES))
    return TaskWeighter(**kwargs)


def _pin_floor(weighter: TaskWeighter, task: str) -> None:
    with torch.no_grad():
        index = weighter.task_names.index(task)
        weighter.log_s[index] = 50.0  # exp(-50) << floor; clamps to the floor


def _pin_off_floor(weighter: TaskWeighter, task: str) -> None:
    with torch.no_grad():
        index = weighter.task_names.index(task)
        weighter.log_s[index] = 0.0


# ------------------------------------------------------------- construction
def test_defaults_and_names():
    weighter = _make_weighter()
    assert weighter.task_names == DEFAULT_TASK_NAMES
    assert weighter.floor == pytest.approx(0.05)
    assert weighter.cap == pytest.approx(4.0)
    assert isinstance(weighter.log_s, torch.nn.Parameter)
    assert dict(weighter.weights()) == pytest.approx({t: 1.0 for t in DEFAULT_TASK_NAMES})


def test_init_s_sequence_and_broadcast():
    weighter = TaskWeighter(3, init_s=[0.0, 1.0, -1.0], task_names=["a", "b", "c"])
    assert weighter.log_s.tolist() == pytest.approx([0.0, 1.0, -1.0])
    broadcast = TaskWeighter(2, init_s=0.5, task_names=["a", "b"])
    assert broadcast.log_s.tolist() == pytest.approx([0.5, 0.5])


def test_custom_num_tasks_get_generated_names():
    weighter = TaskWeighter(2, init_s=0.0)
    assert weighter.task_names == ("task_0", "task_1")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_tasks": True},
        {"num_tasks": 0},
        {"num_tasks": 2.5},
        {"num_tasks": 2, "floor": 0.0},
        {"num_tasks": 2, "floor": -0.1},
        {"num_tasks": 2, "floor": 5.0, "cap": 4.0},
        {"num_tasks": 2, "floor": float("nan")},
        {"num_tasks": 2, "cap": float("inf")},
        {"num_tasks": 2, "init_s": [0.0]},
        {"num_tasks": 2, "init_s": float("nan")},
        {"num_tasks": 2, "task_names": ["only-one", "one", "two"]},
        {"num_tasks": 2, "task_names": ["dup", "dup"]},
        {"num_tasks": 2, "task_names": ["", "b"]},
    ],
)
def test_invalid_construction_raises(kwargs):
    with pytest.raises((ValueError, TypeError)):
        TaskWeighter(**kwargs)


# ----------------------------------------------------------- floor and cap
def test_floor_cap_enforced_across_s_sweep():
    weighter = _make_weighter()
    for raw in [x / 20.0 for x in range(-2000, 2001)]:  # -100.0 … 100.0
        with torch.no_grad():
            weighter.log_s.fill_(raw)
        resolved = weighter.weights()
        for value in resolved.values():
            assert weighter.floor - TOL <= value <= weighter.cap + TOL


def test_floor_cap_hit_at_extremes():
    weighter = _make_weighter()
    with torch.no_grad():
        weighter.log_s.fill_(1e6)
    assert all(v == pytest.approx(weighter.floor, abs=1e-6) for v in weighter.weights().values())
    with torch.no_grad():
        weighter.log_s.fill_(-1e6)
    assert all(v == pytest.approx(weighter.cap, abs=1e-6) for v in weighter.weights().values())


def test_weights_returns_detached_floats():
    weighter = _make_weighter()
    resolved = weighter.weights()
    assert all(type(v) is float for v in resolved.values())
    assert list(resolved) == list(DEFAULT_TASK_NAMES)


def test_weight_tensor_is_differentiable():
    weighter = _make_weighter()
    total = weighter.weight_tensor().sum()
    total.backward()
    assert weighter.log_s.grad is not None
    assert torch.isfinite(weighter.log_s.grad).all()


# ------------------------------------------------------------------ forward
def test_forward_matches_hand_computed_kendall_loss():
    weighter = TaskWeighter(2, task_names=["a", "b"], init_s=[0.5, -0.25])
    losses = {"a": torch.tensor(2.0), "b": torch.tensor(3.0)}
    total = weighter(losses)
    s = [0.5, -0.25]
    expected = sum(math.exp(-v) * loss for v, loss in zip(s, [2.0, 3.0])) + sum(s)
    assert total.item() == pytest.approx(expected, abs=1e-5)


def test_forward_sequence_matches_mapping():
    weighter = TaskWeighter(2, task_names=["a", "b"], init_s=[0.1, 0.2])
    via_map = weighter({"a": torch.tensor(1.5), "b": torch.tensor(0.5)})
    via_seq = weighter([torch.tensor(1.5), torch.tensor(0.5)])
    assert via_map.item() == pytest.approx(via_seq.item(), abs=1e-6)


def test_forward_inactive_task_contributes_no_weighted_term_but_keeps_s():
    weighter = TaskWeighter(2, task_names=["a", "b"], init_s=[0.3, 0.4])
    losses = {"a": torch.tensor(2.0), "b": torch.tensor(9.0)}
    full = weighter(losses)
    partial = weighter(losses, active={"b": False})
    s = [0.3, 0.4]
    expected = (math.exp(-s[0]) * 2.0 + s[0]) + s[1]
    assert partial.item() == pytest.approx(expected, abs=1e-5)
    assert full.item() != pytest.approx(partial.item(), abs=1e-5)


def test_forward_rejects_bad_losses():
    weighter = TaskWeighter(2, task_names=["a", "b"])
    with pytest.raises(ValueError, match="missing"):
        weighter({"a": torch.tensor(1.0)})
    with pytest.raises(ValueError, match="unknown"):
        weighter({"a": torch.tensor(1.0), "b": torch.tensor(1.0), "c": torch.tensor(1.0)})
    with pytest.raises(ValueError, match="num_tasks"):
        weighter([torch.tensor(1.0)])
    with pytest.raises(ValueError, match="scalar"):
        weighter({"a": torch.tensor([1.0, 2.0]), "b": torch.tensor(1.0)})
    with pytest.raises(TypeError, match="Tensor"):
        weighter({"a": 1.0, "b": torch.tensor(1.0)})
    with pytest.raises(ValueError, match="active"):
        weighter({"a": torch.tensor(1.0), "b": torch.tensor(1.0)}, active={"zz": False})


def test_forward_gradient_reaches_log_s():
    weighter = _make_weighter()
    losses = {name: torch.tensor(1.0 + i) for i, name in enumerate(DEFAULT_TASK_NAMES)}
    weighter(losses).backward()
    assert weighter.log_s.grad is not None
    assert torch.isfinite(weighter.log_s.grad).all()
    assert weighter.log_s.grad.abs().sum() > 0


# -------------------------------------------------------- collapse detection
def test_collapse_event_after_two_consecutive_floor_epochs():
    weighter = _make_weighter()
    logger = RecordingLogger()
    _pin_floor(weighter, "risk")

    weighter.log_epoch(logger, epoch=0, metrics=None)
    assert weighter.collapse_events() == []
    assert weighter.floor_streaks()["risk"] == 1

    weighter.log_epoch(logger, epoch=1, metrics=None)
    events = weighter.collapse_events()
    assert events == [
        {"event": WEIGHT_COLLAPSE_EVENT, "task": "risk", "consecutive_floor_epochs": 2}
    ]
    assert weighter.floor_streaks()["risk"] == 2

    collapse_calls = [c for c in logger.calls if c["event"] == WEIGHT_COLLAPSE_EVENT]
    assert len(collapse_calls) == 1
    assert collapse_calls[0]["task"] == "risk"
    assert collapse_calls[0]["consecutive_floor_epochs"] == COLLAPSE_FLOOR_EPOCHS == 2
    assert collapse_calls[0]["metrics"]["consecutive_floor_epochs"] == 2


def test_collapse_not_reemitted_while_pinned_and_resets_off_floor():
    weighter = _make_weighter()
    logger = RecordingLogger()
    _pin_floor(weighter, "score")

    for epoch in range(4):
        weighter.log_epoch(logger, epoch=epoch, metrics=None)
    # Exactly one event, emitted at the second floor epoch.
    assert [e["consecutive_floor_epochs"] for e in weighter.collapse_events()] == [2]
    assert weighter.floor_streaks()["score"] == 4

    _pin_off_floor(weighter, "score")
    weighter.log_epoch(logger, epoch=4, metrics=None)
    assert weighter.floor_streaks()["score"] == 0

    _pin_floor(weighter, "score")
    weighter.log_epoch(logger, epoch=5, metrics=None)
    weighter.log_epoch(logger, epoch=6, metrics=None)
    events = weighter.collapse_events()
    assert len(events) == 2
    assert all(e["task"] == "score" for e in events)


def test_collapse_tracked_without_logger():
    weighter = _make_weighter()
    _pin_floor(weighter, "abstain")
    assert weighter.log_epoch(None, epoch=0, metrics=None) is None
    weighter.log_epoch(None, epoch=1, metrics=None)
    events = weighter.collapse_events()
    assert events == [
        {"event": WEIGHT_COLLAPSE_EVENT, "task": "abstain", "consecutive_floor_epochs": 2}
    ]


def test_floor_streak_tolerates_ulp_above_floor():
    weighter = _make_weighter()
    with torch.no_grad():
        # s_max is -log(floor); setting exactly there yields exp(-s) ≈ floor ± 1 ulp.
        weighter.log_s[weighter.task_names.index("confidence")] = weighter._s_max
    w = weighter.weights()["confidence"]
    assert w <= weighter.floor + 1e-6
    weighter.log_epoch(None, epoch=0, metrics=None)
    assert weighter.floor_streaks()["confidence"] == 1


# ----------------------------------------------------------- epoch logging
def test_log_epoch_record_shape_with_recording_logger():
    weighter = _make_weighter()
    logger = RecordingLogger()
    record = weighter.log_epoch(
        logger,
        epoch=7,
        metrics={"losses": {t: 1.0 for t in DEFAULT_TASK_NAMES}, "lr": 3e-4},
    )
    assert record is logger.calls[0]
    assert record["event"] == WEIGHT_LOG_EVENT
    assert record["weights"] == pytest.approx({t: 1.0 for t in DEFAULT_TASK_NAMES})
    assert record["active"] == {t: True for t in DEFAULT_TASK_NAMES}
    assert record["metrics"]["epoch"] == 7
    assert record["metrics"]["num_tasks"] == len(DEFAULT_TASK_NAMES)
    assert record["metrics"]["loss_choice"] == pytest.approx(1.0)
    assert record["metrics"]["lr"] == pytest.approx(3e-4)


def test_log_epoch_active_flags_and_validation():
    weighter = _make_weighter()
    logger = RecordingLogger()
    record = weighter.log_epoch(logger, epoch=0, metrics={"active": {"risk": False}})
    assert record["active"]["risk"] is False
    assert record["active"]["choice"] is True
    with pytest.raises(ValueError, match="unknown task"):
        weighter.log_epoch(logger, epoch=1, metrics={"active": {"nope": True}})
    with pytest.raises(ValueError, match="flat JSON scalar"):
        weighter.log_epoch(logger, epoch=1, metrics={"nested": {"x": 1}})
    with pytest.raises(ValueError, match="non-empty"):
        weighter.log_epoch(logger, epoch=1, metrics={"": 1})
    with pytest.raises(ValueError, match="epoch"):
        weighter.log_epoch(logger, epoch=-1, metrics=None)
    with pytest.raises(ValueError, match="epoch"):
        weighter.log_epoch(logger, epoch=True, metrics=None)


def test_log_epoch_writes_jsonl_via_real_logger(tmp_path):
    log_path = tmp_path / "run" / "train.jsonl"
    logger = JsonlLogger(log_path, run_id="run-test")
    weighter = _make_weighter()
    _pin_floor(weighter, "preference")

    weighter.log_epoch(logger, epoch=0, metrics={"losses": {"choice": 0.25}})
    weighter.log_epoch(logger, epoch=1, metrics={"losses": {"choice": 0.25}})

    records = list(iter_records(log_path))
    weight_records = [r for r in records if r["event"] == WEIGHT_LOG_EVENT]
    collapse_records = [r for r in records if r["event"] == WEIGHT_COLLAPSE_EVENT]
    assert len(weight_records) == 2
    for record in weight_records:
        assert record["run_id"] == "run-test"
        assert isinstance(record["step"], int)
        assert set(record["weights"]) == set(DEFAULT_TASK_NAMES)
        assert set(record["active"]) == set(DEFAULT_TASK_NAMES)
        assert all(isinstance(v, bool) for v in record["active"].values())
        assert record["metrics"]["epoch"] in (0, 1)
    assert weight_records[0]["metrics"]["loss_choice"] == pytest.approx(0.25)
    assert len(collapse_records) == 1
    assert collapse_records[0]["task"] == "preference"
    assert collapse_records[0]["consecutive_floor_epochs"] == 2
    assert collapse_records[0]["metrics"]["consecutive_floor_epochs"] == 2


def test_weights_logged_every_epoch(tmp_path):
    log_path = tmp_path / "train.jsonl"
    logger = JsonlLogger(log_path, run_id="run-loop")
    weighter = _make_weighter()
    for epoch in range(3):
        weighter.log_epoch(logger, epoch=epoch, metrics=None)
    events = [r["event"] for r in iter_records(log_path)]
    assert events == [WEIGHT_LOG_EVENT, WEIGHT_LOG_EVENT, WEIGHT_LOG_EVENT]


# ------------------------------------------------------------- manifest
def test_to_manifest_fragment_shape_and_serializable():
    weighter = _make_weighter()
    fragment = weighter.to_manifest_fragment()
    assert fragment["task_weight_mode"] == "learned"
    assert fragment["task_weight_floor"] == pytest.approx(0.05)
    assert fragment["task_weight_cap"] == pytest.approx(4.0)
    assert fragment["task_weights"] == pytest.approx({t: 1.0 for t in DEFAULT_TASK_NAMES})
    json.dumps(fragment)  # must be JSON-serializable


def test_manifest_fragment_reaches_the_run_registry(tmp_path):
    weighter = _make_weighter()
    with torch.no_grad():
        weighter.log_s[weighter.task_names.index("score")] = 1.5
    fragment = weighter.to_manifest_fragment()

    run_dir = tmp_path / "runs" / "run-manifest"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "run-manifest",
        "config_hash": "deadbeef" * 8,
        "data_fingerprint": "ab" * 32,
        "git_sha": "deadbee",
        "seed": 13,
        "device": "cuda:0",
        "versions": {"python": "3.11.0"},
        **fragment,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    registry = RunRegistry(tmp_path / "runs")
    entry = registry.get("run-manifest")
    assert entry.manifest["task_weights"] == pytest.approx(fragment["task_weights"])
    assert entry.manifest["task_weight_floor"] == pytest.approx(0.05)
    assert entry.manifest["task_weight_cap"] == pytest.approx(4.0)


def test_collapse_events_returns_copies():
    weighter = _make_weighter()
    _pin_floor(weighter, "choice")
    weighter.log_epoch(None, epoch=0, metrics=None)
    weighter.log_epoch(None, epoch=1, metrics=None)
    events = weighter.collapse_events()
    events[0]["task"] = "mutated"
    assert weighter.collapse_events()[0]["task"] == "choice"
