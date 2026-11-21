"""Unit tests for the TASK-24 benchmark boundary (FR-INFER-03, FR-CONFIG-02).

All tests are CPU-only and use the tiny fixture model; the full-default run
(warmup=50, iters=1000, batches 1 and 32) is exercised end to end. No network
is touched; the optional ``onnx``-dependent temporary-export path is allowed
to produce integer serialized sizes, and assertions tolerate ``None`` for
hosts without it.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
import torch

from src.inference.bench import (
    ENVIRONMENT_KEYS,
    PRECISIONS,
    BenchReport,
    environment_block,
    run_benchmark,
)
from src.model.core import RapportConfig, SpecConstants  # noqa: E402
from src.model.head import RapportModel  # noqa: E402
from src.tokenization import RapportTokenizer  # noqa: E402


def _tiny_config() -> RapportConfig:
    spec = SpecConstants(
        attention_total=8,
        swiglu_total=8,
        norms_total=1,
        trunk=17,
        embedding=8,
        heads_total=8,
        total=33,
        rows=(),
    )
    return RapportConfig(
        variant="RAPPORT-TEST",
        layers=1,
        d_model=32,
        heads=2,
        head_dim=16,
        ffn=64,
        vocab_size=128,
        embedding_dim=32,
        spec=spec,
    )


@pytest.fixture(scope="module")
def tokenizer() -> RapportTokenizer:
    return RapportTokenizer()


@pytest.fixture(scope="module")
def model() -> RapportModel:
    torch.manual_seed(7)
    return RapportModel(_tiny_config()).eval()


@pytest.fixture(scope="module")
def report(model: RapportModel, tokenizer: RapportTokenizer) -> BenchReport:
    # FR-INFER-03 / acceptance: default warmup=50, iters=1000, batches (1, 32).
    return run_benchmark(model, tokenizer)


# ---------------------------------------------------------------------------
# environment_block (FR-CONFIG-02)
# ---------------------------------------------------------------------------


def test_environment_block_has_exactly_the_declared_keys() -> None:
    block = environment_block()
    assert list(block) == list(ENVIRONMENT_KEYS)


def test_environment_block_records_torch_and_python_versions() -> None:
    block = environment_block()
    assert block["torch_version"] == str(torch.__version__)
    import platform

    assert block["python_version"] == platform.python_version()


def test_environment_block_never_raises_on_cuda_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated CUDA failure")

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    monkeypatch.setattr(torch.cuda, "get_device_name", _boom)
    monkeypatch.setattr(torch.cuda, "get_device_capability", _boom)
    monkeypatch.setattr(torch.cuda, "current_device", _boom)

    block = environment_block()
    assert block["gpu"] is None
    assert block["compute_capability"] is None
    assert block["torch_version"] == str(torch.__version__)
    assert block["python_version"]


def test_environment_block_tolerates_total_cuda_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "is_available",
        "get_device_name",
        "get_device_capability",
        "current_device",
    ):
        def _failing(*args: Any, **kwargs: Any) -> Any:
            raise OSError("no cuda")

        monkeypatch.setattr(torch.cuda, name, _failing)
    block = environment_block()
    assert block["gpu"] is None
    assert block["compute_capability"] is None


# ---------------------------------------------------------------------------
# flags_against (FR-CONFIG-02)
# ---------------------------------------------------------------------------


def _report_with_environment(environment: dict[str, Any]) -> BenchReport:
    base = environment_block()
    base.update(environment)
    return BenchReport(
        batch=[1],
        warmup=0,
        iters=1,
        p50_ms={"1": 1.0},
        p95_ms={"1": 1.0},
        p99_ms={"1": 1.0},
        throughput_decisions_per_s=1.0,
        peak_rss_mb=1.0,
        peak_vram_mb=None,
        environment=base,
        precision_note="test",
        precision=["fp32"],
        serialized_sizes={"fp32": None, "fp16": None, "int8": None},
        measurements={},
        seq_len=64,
        latency_method="test",
        quantization_gate={},
        flags_against_previous=[],
    )


def test_flags_against_identical_environments_returns_empty() -> None:
    report = _report_with_environment({})
    assert report.flags_against(report) == []
    assert report.flags_against(dict(report.environment)) == []


def test_flags_against_detects_gpu_and_torch_changes() -> None:
    current = _report_with_environment({"gpu": "NVIDIA T4", "torch_version": "2.15.0"})
    previous = _report_with_environment({"gpu": None, "torch_version": "2.0.0"})
    flags = current.flags_against(previous)
    assert any("gpu" in flag for flag in flags)
    assert any("torch_version" in flag for flag in flags)
    assert all(isinstance(flag, str) for flag in flags)


def test_flags_against_accepts_bare_mapping() -> None:
    current = _report_with_environment({"gpu": "NVIDIA T4"})
    previous = dict(current.environment)
    previous["gpu"] = None
    flags = current.flags_against(previous)
    assert len(flags) == 1
    assert "gpu" in flags[0]


def test_flags_against_rejects_unsupported_previous() -> None:
    report = _report_with_environment({})
    with pytest.raises(TypeError):
        report.flags_against(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run_benchmark (FR-INFER-03)
# ---------------------------------------------------------------------------


def test_run_benchmark_reports_both_batches_with_contract_defaults(report: BenchReport) -> None:
    assert report.batch == [1, 32]
    assert report.warmup == 50
    assert report.iters == 1000

    assert set(report.p50_ms) == {"1", "32"}
    assert set(report.p95_ms) == {"1", "32"}
    assert set(report.p99_ms) == {"1", "32"}
    for values in (report.p50_ms, report.p95_ms, report.p99_ms):
        assert all(v is not None and v > 0 for v in values.values())

    assert report.throughput_decisions_per_s is not None
    assert report.throughput_decisions_per_s > 0
    fp32 = report.measurements["fp32"]
    assert fp32["32"]["throughput_decisions_per_s"] > 0
    assert fp32["1"]["throughput_decisions_per_s"] > 0


def test_run_benchmark_p95_reuses_export_helper_and_matches_report(report: BenchReport) -> None:
    # export.benchmark_p95 is nearest-rank; our per-batch p95 values are floats
    # in the same family as the p50/p99 neighbours.
    for size in ("1", "32"):
        p50 = report.measurements["fp32"][size]["p50_ms"]
        p95 = report.measurements["fp32"][size]["p95_ms"]
        p99 = report.measurements["fp32"][size]["p99_ms"]
        assert p50 is not None and p95 is not None and p99 is not None
        assert p50 <= p95 <= p99 * 1.5  # ordering holds up to timing noise


def test_run_benchmark_reports_peak_memory(report: BenchReport) -> None:
    assert report.peak_rss_mb > 0
    if torch.cuda.is_available():
        assert report.peak_vram_mb is not None and report.peak_vram_mb >= 0
    else:
        assert report.peak_vram_mb is None


def test_run_benchmark_reports_serialized_sizes_for_all_three_variants(
    report: BenchReport,
) -> None:
    assert set(report.serialized_sizes) == {"fp32", "fp16", "int8"}
    for value in report.serialized_sizes.values():
        assert value is None or (isinstance(value, int) and value > 0)


def test_run_benchmark_environment_block_matches_contract(report: BenchReport) -> None:
    assert set(report.environment) == set(ENVIRONMENT_KEYS)
    assert report.environment["torch_version"] == str(torch.__version__)
    if not torch.cuda.is_available():
        assert report.environment["gpu"] is None
        assert report.environment["compute_capability"] is None
        assert report.environment["cuda_version"] is None


def test_run_benchmark_documents_cpu_emulation(report: BenchReport) -> None:
    note = report.precision_note
    assert "fp32" in note
    assert "fp16" in note
    assert "int8" in note
    if not torch.cuda.is_available():
        assert "CPU-emulated" in note


def test_run_benchmark_gate_stand_in_is_recorded(report: BenchReport) -> None:
    gate = report.quantization_gate
    assert gate["status"] in {"not_evaluated", "passed", "failed", "error"}
    if gate["status"] == "not_evaluated":
        assert "gate" in gate["reason"]


def test_run_benchmark_report_is_json_serializable(report: BenchReport) -> None:
    payload = report.to_dict()
    blob = json.dumps(payload)
    decoded = json.loads(blob)
    assert decoded["batch"] == [1, 32]
    assert decoded["warmup"] == 50
    assert decoded["iters"] == 1000
    assert set(decoded["environment"]) == set(ENVIRONMENT_KEYS)
    assert set(PRECISIONS) <= set(decoded["measurements"])
    # The FP32-ORT gate baseline is an extra measurement key when an FP32
    # ONNX artifact was available; torch precisions are always present.
    assert set(decoded["measurements"]["fp32"]) == {"1", "32"}


def test_run_benchmark_to_json_round_trip(report: BenchReport) -> None:
    decoded = json.loads(report.to_json())
    assert decoded["batch"] == [1, 32]
    assert decoded["serialized_sizes"] == report.serialized_sizes
    assert decoded["environment"]["torch_version"] == report.environment["torch_version"]


def test_run_benchmark_flags_previous_report(
    model: RapportModel, tokenizer: RapportTokenizer
) -> None:
    baseline = _report_with_environment({"gpu": "NVIDIA T4", "torch_version": "2.0.0"})
    result = run_benchmark(
        model,
        tokenizer,
        batch_sizes=(1,),
        warmup=0,
        iters=1,
        precision=("fp32",),
        gate_report=None,
        previous_report=baseline,
    )
    assert any("gpu" in flag for flag in result.flags_against_previous)
    assert any("torch_version" in flag for flag in result.flags_against_previous)


def test_run_benchmark_without_previous_report_has_empty_flags(
    model: RapportModel, tokenizer: RapportTokenizer
) -> None:
    result = run_benchmark(
        model, tokenizer, batch_sizes=(1,), warmup=0, iters=1, precision=("fp32",)
    )
    assert result.flags_against_previous == []


def test_run_benchmark_gate_report_is_evaluated(
    model: RapportModel, tokenizer: RapportTokenizer
) -> None:
    gate_report = {
        "choice_accuracy_fp32": 0.80,
        "choice_accuracy_int8": 0.795,
        "bucket_macro_ece_fp32": 0.03,
        "bucket_macro_ece_int8": 0.045,
        "cosine_score_path_fp32": True,
        "int8_size_bytes": 30 * 1024 * 1024,
        "fp32_p95_ms": 100.0,
        "int8_p95_ms": 40.0,
    }
    result = run_benchmark(
        model,
        tokenizer,
        batch_sizes=(1,),
        warmup=0,
        iters=1,
        precision=("fp32", "int8"),
        gate_report=gate_report,
    )
    # The caller's declared 2.5x speedup is ignored in favor of THIS run's
    # synchronized FP32-ORT/INT8 measurements; on this host it does not reach 2x.
    assert result.quantization_gate["status"] == "failed"
    assert result.quantization_gate["artifact"] == "fp16"
    assert result.quantization_gate["margins"]["p95_latency_speedup"] < 2.5


def test_run_benchmark_gate_report_invalid_is_recorded_as_error(
    model: RapportModel, tokenizer: RapportTokenizer
) -> None:
    result = run_benchmark(
        model,
        tokenizer,
        batch_sizes=(1,),
        warmup=0,
        iters=1,
        precision=("fp32", "int8"),
        gate_report={"fp32_p95_ms": 100.0},
    )
    assert result.quantization_gate["status"] == "error"


def test_run_benchmark_rejects_invalid_arguments(
    model: RapportModel, tokenizer: RapportTokenizer
) -> None:
    with pytest.raises(TypeError):
        run_benchmark(model, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        run_benchmark(model, tokenizer, iters=0)
    with pytest.raises(ValueError):
        run_benchmark(model, tokenizer, warmup=-1)
    with pytest.raises(ValueError):
        run_benchmark(model, tokenizer, batch_sizes=())
    with pytest.raises(ValueError):
        run_benchmark(model, tokenizer, batch_sizes=(0,))
    with pytest.raises(ValueError):
        run_benchmark(model, tokenizer, precision=("int4",))  # type: ignore[arg-type]


def test_run_benchmark_uses_declared_defaults() -> None:
    import inspect

    signature = inspect.signature(run_benchmark)
    assert signature.parameters["batch_sizes"].default == (1, 32)
    assert signature.parameters["warmup"].default == 50
    assert signature.parameters["iters"].default == 1000
    assert signature.parameters["precision"].default == ("fp32", "fp16", "int8")


def test_run_benchmark_batches_are_deterministic(
    model: RapportModel, tokenizer: RapportTokenizer
) -> None:
    first = run_benchmark(model, tokenizer, batch_sizes=(1,), warmup=0, iters=1)
    second = run_benchmark(model, tokenizer, batch_sizes=(1,), warmup=0, iters=1)
    assert first.measurements["fp32"]["1"]["iterations"] == 1
    assert second.measurements["fp32"]["1"]["iterations"] == 1
    # Determinism of the fixture is observable through the stable seq_len and
    # precision selection, not wall-clock numbers.
    assert first.seq_len == second.seq_len == 64
    assert first.precision == second.precision == ["fp32", "fp16", "int8"]


def test_flags_against_on_bench_report_dataclass() -> None:
    report = _report_with_environment({"driver": "535.1"})
    changed = dataclasses.replace(
        report, environment={**report.environment, "driver": "545.23"}
    )
    flags = changed.flags_against(report)
    assert len(flags) == 1
    assert "driver" in flags[0]


# ---------------------------------------------------------------------------
# audit finding 6: synchronized single-array timing, INT8 gate baseline
# ---------------------------------------------------------------------------


def test_measure_derives_p50_p95_p99_from_one_latency_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import bench as bm

    calls: list[int] = []

    class _Session:
        def run(self, output_names, feed):  # noqa: ANN001, ANN202
            calls.append(1)

    ticks = iter(float(i) / 1000.0 for i in range(100))
    monkeypatch.setattr(bm.time, "perf_counter", lambda: next(ticks))
    measured = bm._measure(
        _Session(), {"x": torch.zeros(1)}, rows=1,
        warmup=2, iters=3, device_type="cpu",
    )
    # warmup + timed passes only — no separate p95 pass.
    assert len(calls) == 5
    assert measured["p50_ms"] == pytest.approx(1.0)
    assert measured["p95_ms"] == pytest.approx(1.0)
    assert measured["p99_ms"] == pytest.approx(1.0)


def test_int8_gate_baseline_is_fp32_ort_not_torch(
    model: RapportModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    from src.inference import bench as bm

    class _FakeORT:
        def __init__(self, path: str) -> None:
            self.path = path

        def run(self, output_names, feed):  # noqa: ANN001, ANN202
            return (torch.zeros(1, 2),)

    loaded: list[str] = []

    def _fake_load(path: Path):
        loaded.append(str(path))
        return _FakeORT(str(path)), None

    monkeypatch.setattr(bm, "_load_ort_session", _fake_load)
    monkeypatch.setattr(
        bm, "_resolve_artifacts",
        lambda model, tokenizer, artifacts_dir, int8_artifact: (
            {"fp32": 10, "fp16": None, "int8": 10},
            Path("/tmp/fake-int8.onnx"), "", None, Path("/tmp/fake-fp32.onnx"),
        ),
    )
    result = bm.run_benchmark(
        model, RapportTokenizer(), batch_sizes=(1,), iters=2, warmup=0,
        precision=("fp32", "int8"),
    )
    assert any("fake-fp32.onnx" in p for p in loaded)
    assert "fp32_ort" in result.measurements
    baseline = result.measurements["fp32_ort"]["1"]["p95_ms"]
    assert baseline is not None and baseline > 0
    # The stand-in gate baseline is the FP32-ORT number, not torch's p95.
    assert result.quantization_gate["stand_in_report"]["fp32_p95_ms"] == pytest.approx(baseline)


def test_skipped_top_level_throughput_stays_none(
    model, tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.inference import bench as bm
    monkeypatch.setattr(
        bm, "_resolve_artifacts",
        lambda model, tokenizer, artifacts_dir, int8_artifact: (
            {"fp32": None, "fp16": None, "int8": None}, None, "no artifact", None, None,
        ),
    )
    result = run_benchmark(
        model, tokenizer, batch_sizes=(1,), iters=2, warmup=0,
        precision=("int8",),
    )
    assert result.measurements["int8"]["1"]["throughput_decisions_per_s"] is None
    assert result.throughput_decisions_per_s is None
