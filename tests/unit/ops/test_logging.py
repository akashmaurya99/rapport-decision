"""Tests for src/ops/logging.py — JSONL record validation and append semantics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.ops import (
    VALID_SPLITS,
    JsonlLogger,
    JsonlRecordError,
    build_record,
    iter_records,
)


@pytest.fixture()
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "runs" / "run-1" / "metrics.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_log_writes_one_line_with_required_keys(log_path: Path) -> None:
    logger = JsonlLogger(log_path, run_id="run-1")
    record = logger.log_step(1, {"loss": 0.5}, wall_clock_seconds=0.25)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed == record
    for key in ("run_id", "step", "split", "event", "wall_clock_seconds", "metrics", "timestamp"):
        assert key in parsed
    assert parsed["run_id"] == "run-1"
    assert parsed["step"] == 1
    assert parsed["split"] == "train"
    assert parsed["event"] == "step"
    assert parsed["wall_clock_seconds"] == 0.25
    assert parsed["metrics"] == {"loss": 0.5}


def test_eval_and_benchmark_events(log_path: Path) -> None:
    logger = JsonlLogger(log_path, run_id="run-1")
    logger.log_eval(10, {"accuracy": 0.75})
    logger.log_benchmark(0, {"p50_ms": 11.2, "throughput": 301.0})

    records = _read_jsonl(log_path)
    assert [r["event"] for r in records] == ["eval", "benchmark"]
    assert records[0]["split"] == "dev"
    assert records[1]["split"] == "benchmark"


def test_all_declared_splits_are_accepted(log_path: Path) -> None:
    logger = JsonlLogger(log_path, run_id="run-1")
    for index, split in enumerate(VALID_SPLITS):
        logger.log(event="step", step=index, split=split, metrics={"loss": 1.0})
    assert [r["split"] for r in _read_jsonl(log_path)] == list(VALID_SPLITS)


def test_missing_metrics_raises_typed_error_and_writes_nothing(log_path: Path) -> None:
    logger = JsonlLogger(log_path, run_id="run-1")
    with pytest.raises(JsonlRecordError) as excinfo:
        logger.log(event="step", step=1, split="train")
    assert excinfo.value.key == "metrics"
    assert not log_path.exists()


def test_metric_set_alone_satisfies_the_requirement(log_path: Path) -> None:
    logger = JsonlLogger(log_path, run_id="run-1")
    logger.log(event="eval", step=3, split="test", metric_set={"ece": 0.04})
    (record,) = _read_jsonl(log_path)
    assert record["metric_set"] == {"ece": 0.04}
    assert "metrics" not in record


def test_both_metric_keys_are_kept(log_path: Path) -> None:
    logger = JsonlLogger(log_path, run_id="run-1")
    logger.log(
        event="eval",
        step=3,
        split="test",
        metrics={"loss": 0.1},
        metric_set={"loss": 0.1, "ece": 0.02},
    )
    (record,) = _read_jsonl(log_path)
    assert record["metrics"] == {"loss": 0.1}
    assert record["metric_set"] == {"loss": 0.1, "ece": 0.02}


def test_invalid_split_rejected(log_path: Path) -> None:
    logger = JsonlLogger(log_path, run_id="run-1")
    with pytest.raises(JsonlRecordError) as excinfo:
        logger.log(event="step", step=1, split="validation", metrics={"loss": 1.0})
    assert excinfo.value.key == "split"
    assert not log_path.exists()


@pytest.mark.parametrize("bad_step", [-1, 1.5, True, "1", None])
def test_invalid_step_rejected(log_path: Path, bad_step: object) -> None:
    logger = JsonlLogger(log_path, run_id="run-1")
    with pytest.raises(JsonlRecordError) as excinfo:
        logger.log(event="step", step=bad_step, split="train", metrics={"loss": 1.0})
    assert excinfo.value.key == "step"
    assert not log_path.exists()


@pytest.mark.parametrize("bad_event", ["", "   ", 7, None])
def test_invalid_event_rejected(log_path: Path, bad_event: object) -> None:
    logger = JsonlLogger(log_path, run_id="run-1")
    with pytest.raises(JsonlRecordError) as excinfo:
        logger.log(event=bad_event, step=1, split="train", metrics={"loss": 1.0})
    assert excinfo.value.key == "event"


@pytest.mark.parametrize(
    "bad_metrics",
    [{}, [], "loss", {"loss": object()}, {"loss": float("nan")}, {"loss": float("inf")},
     {"": 1.0}, {1: 2.0}, {"nested": {"a": 1}}, {"items": [1, 2]}],
)
def test_invalid_metrics_rejected(log_path: Path, bad_metrics: object) -> None:
    logger = JsonlLogger(log_path, run_id="run-1")
    with pytest.raises(JsonlRecordError):
        logger.log(event="step", step=1, split="train", metrics=bad_metrics)
    assert not log_path.exists()


@pytest.mark.parametrize("bad_wall", [-0.1, float("nan"), float("inf"), "1.5", True])
def test_invalid_wall_clock_rejected(log_path: Path, bad_wall: object) -> None:
    logger = JsonlLogger(log_path, run_id="run-1")
    with pytest.raises(JsonlRecordError) as excinfo:
        logger.log(event="step", step=1, split="train", metrics={"loss": 1.0},
                   wall_clock_seconds=bad_wall)
    assert excinfo.value.key == "wall_clock_seconds"


def test_wall_clock_auto_measured_and_non_negative(log_path: Path) -> None:
    logger = JsonlLogger(log_path, run_id="run-1")
    first = logger.log(event="step", step=0, split="train", metrics={"loss": 1.0})
    second = logger.log(event="step", step=1, split="train", metrics={"loss": 0.9})
    assert first["wall_clock_seconds"] >= 0.0
    assert second["wall_clock_seconds"] >= 0.0


def test_extra_keys_recorded_and_reserved_collision_rejected(log_path: Path) -> None:
    logger = JsonlLogger(log_path, run_id="run-1")
    record = logger.log(
        event="step", step=1, split="train", metrics={"loss": 1.0}, optimizer="adamw"
    )
    assert record["optimizer"] == "adamw"
    with pytest.raises(JsonlRecordError):
        logger.log(
            event="step", step=2, split="train", metrics={"loss": 1.0}, timestamp="x"
        )


def test_run_id_override_per_record(log_path: Path) -> None:
    logger = JsonlLogger(log_path, run_id="run-1")
    record = logger.log(
        event="step", step=1, split="train", metrics={"loss": 1.0}, run_id="run-2"
    )
    assert record["run_id"] == "run-2"


def test_logger_rejects_empty_run_id(log_path: Path) -> None:
    with pytest.raises(JsonlRecordError):
        JsonlLogger(log_path, run_id="  ")


def test_append_binary_mode_and_fsync_per_record(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsync_calls: list[int] = []
    real_fsync = os.fsync

    def counting_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", counting_fsync)
    logger = JsonlLogger(log_path, run_id="run-1")
    for step in range(3):
        logger.log(event="step", step=step, split="train", metrics={"loss": 1.0 - step})
    assert len(fsync_calls) == 3
    assert logger.records_written == 3


def test_two_loggers_append_to_the_same_file(log_path: Path) -> None:
    first = JsonlLogger(log_path, run_id="run-a")
    second = JsonlLogger(log_path, run_id="run-b")
    first.log(event="step", step=0, split="train", metrics={"loss": 1.0})
    second.log(event="step", step=1, split="train", metrics={"loss": 0.5})

    records = _read_jsonl(log_path)
    assert [r["run_id"] for r in records] == ["run-a", "run-b"]


def test_parent_directories_are_created(log_path: Path) -> None:
    assert not log_path.parent.exists()
    JsonlLogger(log_path, run_id="run-1").log(
        event="step", step=0, split="train", metrics={"loss": 1.0}
    )
    assert log_path.is_file()


def test_iter_records_roundtrip(log_path: Path) -> None:
    logger = JsonlLogger(log_path, run_id="run-1")
    for step in range(5):
        logger.log(event="step", step=step, split="train", metrics={"loss": 1.0 - step * 0.1})
    records = list(iter_records(log_path))
    assert len(records) == 5
    assert [r["step"] for r in records] == list(range(5))


def test_iter_records_reports_corrupt_line(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True)
    log_path.write_text('{"run_id": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(JsonlRecordError) as excinfo:
        list(iter_records(log_path))
    assert ":2" in str(excinfo.value)


def test_iter_records_rejects_non_object_line(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True)
    log_path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(JsonlRecordError):
        list(iter_records(log_path))


def test_build_record_validates_before_writing() -> None:
    with pytest.raises(JsonlRecordError) as excinfo:
        build_record(
            run_id="",
            step=0,
            split="train",
            event="step",
            wall_clock_seconds=0.1,
            metrics={"loss": 1.0},
        )
    assert excinfo.value.key == "run_id"


def test_metric_values_may_be_bool_none_str_int() -> None:
    record = build_record(
        run_id="run-1",
        step=0,
        split="eval",
        event="eval",
        wall_clock_seconds=0,
        metrics={"converged": True, "note": "ok", "count": 3, "extra": None},
    )
    assert record["metrics"]["converged"] is True
