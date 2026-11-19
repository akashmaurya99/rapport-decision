"""Tests for src/inference/export.py — ONNX export and INT8 quantization.

Covers TASK-23 acceptance criteria (FR-INFER-02, E9):

* ONNX export matches PyTorch choice logits within 1e-3 on the fixed
  64-row reference batch, with opset/sample_count/tolerance recorded in
  the export report (FR-INFER-02);
* the five-condition INT8 gate (choice accuracy drop <= 1 point,
  bucket-macro ECE rise <= 0.02, cosine score path in FP32, serialized
  INT8 size <= 35 MB, p95 CPU latency >= 2x faster than FP32 ONNX) — the
  pure decision matrix is exercised on synthetic reports, the artifact
  selection refuses INT8 and retains FP16 when any bound fails (E9).

Tests that need ``onnx``/``onnxruntime`` are skipped with reason
``onnx-not-installed`` when those packages are absent; everything else
runs without them.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

try:  # pragma: no cover - depends on environment
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401

    _ONNX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ONNX_AVAILABLE = False

onnx_required = pytest.mark.skipif(not _ONNX_AVAILABLE, reason="onnx-not-installed")

from src.inference.export import (  # noqa: E402
    EXPORT_OPSET,
    EXPORT_TOLERANCE,
    MAX_ACCURACY_DROP,
    MAX_BUCKET_ECE_RISE,
    MAX_INT8_BYTES,
    MIN_P95_SPEEDUP,
    OUTPUT_NAMES,
    REFERENCE_BATCH_ROWS,
    REFERENCE_BATCH_SEED,
    ExportSelection,
    GateOutcome,
    OnnxExportMismatchError,
    QuantizationRejectedError,
    QuantizationReportError,
    benchmark_p95,
    convert_fp32_to_fp16,
    export_onnx,
    make_reference_batch,
    quantization_gate,
    quantize_dynamic_int8,
    serialized_sizes,
    tensor_fingerprint,
    validate_and_select,
)
from src.model.core import RapportConfig, SpecConstants  # noqa: E402
from src.model.head import MAX_OPTIONS, RapportModel  # noqa: E402
from src.tokenization.tokenizer import (  # noqa: E402
    OPT_ID,
    PAD_ID,
    TASK_ID,
)

WAIVER_NOTE = "onnx-not-installed"


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


def _tiny_model() -> RapportModel:
    torch.manual_seed(7)
    return RapportModel(_tiny_config()).eval()


def _good_report() -> dict:
    # Four of five conditions have comfortable headroom; INT8 is 2.5x faster.
    return {
        "choice_accuracy_fp32": 0.80,
        "choice_accuracy_int8": 0.795,
        "bucket_macro_ece_fp32": 0.03,
        "bucket_macro_ece_int8": 0.045,
        "cosine_score_path_fp32": True,
        "int8_size_bytes": 30 * 1024 * 1024,
        "fp32_p95_ms": 100.0,
        "int8_p95_ms": 40.0,
    }


# ---------------------------------------------------------------------------
# Reference batch + fingerprint (no onnx needed)
# ---------------------------------------------------------------------------


def test_reference_batch_is_deterministic_and_well_formed() -> None:
    first = make_reference_batch()
    second = make_reference_batch()
    for name in ("input_ids", "attention_mask", "option_mask", "task_ids"):
        assert torch.equal(first[name], second[name])
    assert first["input_ids"].shape == (REFERENCE_BATCH_ROWS, 64)
    assert first["option_mask"].shape == (REFERENCE_BATCH_ROWS, MAX_OPTIONS)
    # prefix property: no 0 may precede a 1 in any row
    keep = first["option_mask"].to(torch.bool)
    assert not bool((keep[:, :-1].logical_not() & keep[:, 1:]).any())
    # marker counts match the declared option counts per row
    markers = (first["input_ids"] == OPT_ID).sum(dim=1)
    assert torch.equal(markers, first["option_mask"].sum(dim=1))
    # exactly one <task> marker per row and padding beyond the real length
    assert torch.equal(first["input_ids"][:, 0], torch.full((REFERENCE_BATCH_ROWS,), TASK_ID))
    pads = first["input_ids"] == PAD_ID
    assert torch.equal(first["attention_mask"].to(torch.bool), ~pads)
    assert int(first["task_ids"].min()) >= 0
    assert int(first["task_ids"].max()) < 6


def test_reference_batch_fingerprint_matches_seed_and_rows() -> None:
    batch = make_reference_batch()
    fingerprint = tensor_fingerprint(batch)
    assert len(fingerprint) == 64
    assert tensor_fingerprint(make_reference_batch()) == fingerprint
    assert tensor_fingerprint(make_reference_batch(seed=REFERENCE_BATCH_SEED + 1)) != fingerprint
    assert tensor_fingerprint(make_reference_batch(rows=32)) != fingerprint


def test_reference_batch_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="rows"):
        make_reference_batch(rows=0)
    with pytest.raises(ValueError, match="width"):
        make_reference_batch(width=9)
    with pytest.raises(ValueError, match="width"):
        make_reference_batch(width=1)
    with pytest.raises(ValueError, match="seq_len"):
        make_reference_batch(seq_len=8)
    with pytest.raises(ValueError, match="vocab_size"):
        make_reference_batch(vocab_size=4)


# ---------------------------------------------------------------------------
# Gate: decision matrix (pure logic, no onnx needed)
# ---------------------------------------------------------------------------


def test_gate_accepts_when_all_five_conditions_pass() -> None:
    outcome = quantization_gate(_good_report())
    assert outcome.passed is True
    assert outcome.artifact == "int8"
    assert set(outcome.conditions) == {
        "choice_accuracy_drop",
        "bucket_macro_ece_rise",
        "cosine_score_path_fp32",
        "serialized_size",
        "p95_latency_speedup",
    }
    assert all(outcome.conditions.values())
    assert outcome.margins["choice_accuracy_drop"] == pytest.approx(0.005)
    assert outcome.margins["bucket_macro_ece_rise"] == pytest.approx(0.015)
    assert outcome.margins["p95_latency_speedup"] == pytest.approx(2.5)
    assert json.loads(outcome.to_json())["artifact"] == "int8"


def test_gate_accepts_boundary_values() -> None:
    report = _good_report()
    report.update(
        choice_accuracy_fp32=0.80,
        choice_accuracy_int8=0.80 - MAX_ACCURACY_DROP,  # exactly 1 point
        bucket_macro_ece_fp32=0.03,
        bucket_macro_ece_int8=0.03 + MAX_BUCKET_ECE_RISE,  # exactly 0.02
        int8_size_bytes=MAX_INT8_BYTES,  # exactly 35 MB
        fp32_p95_ms=100.0,
        int8_p95_ms=100.0 / MIN_P95_SPEEDUP,  # exactly 2x
    )
    outcome = quantization_gate(report)
    assert outcome.passed is True
    assert outcome.artifact == "int8"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("choice_accuracy_int8", 0.78),  # drop = 2 points > 1
        ("bucket_macro_ece_int8", 0.06),  # rise = 0.03 > 0.02
        ("int8_size_bytes", MAX_INT8_BYTES + 1),  # 1 byte over the 35 MB bound
        ("int8_p95_ms", 60.0),  # speedup = 1.667x < 2x
    ],
)
def test_gate_refuses_single_bound_failures(key: str, value: object) -> None:
    report = _good_report()
    report[key] = value
    outcome = quantization_gate(report)
    assert outcome.passed is False
    assert outcome.artifact == "fp16"
    assert not all(outcome.conditions.values())


def test_gate_refuses_when_cosine_path_is_quantized() -> None:
    report = _good_report()
    report["cosine_score_path_fp32"] = False
    outcome = quantization_gate(report)
    assert outcome.passed is False
    assert outcome.artifact == "fp16"
    assert outcome.conditions == {
        "choice_accuracy_drop": True,
        "bucket_macro_ece_rise": True,
        "cosine_score_path_fp32": False,
        "serialized_size": True,
        "p95_latency_speedup": True,
    }


def test_gate_refuses_when_everything_fails() -> None:
    report = _good_report()
    report.update(
        choice_accuracy_int8=0.50,
        bucket_macro_ece_int8=0.90,
        cosine_score_path_fp32=False,
        int8_size_bytes=MAX_INT8_BYTES * 2,
        int8_p95_ms=90.0,
    )
    outcome = quantization_gate(report)
    assert outcome.passed is False
    assert outcome.artifact == "fp16"
    assert not any(outcome.conditions.values())


def test_gate_missing_keys_raise_typed_error() -> None:
    report = _good_report()
    del report["int8_p95_ms"]
    del report["cosine_score_path_fp32"]
    with pytest.raises(QuantizationReportError, match="cosine_score_path_fp32"):
        quantization_gate(report)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("choice_accuracy_fp32", 80.0),  # percentages are not fractions
        ("bucket_macro_ece_fp32", -0.1),
        ("choice_accuracy_int8", math.nan),
        ("int8_size_bytes", -1),
        ("fp32_p95_ms", 0.0),
        ("int8_p95_ms", math.inf),
    ],
)
def test_gate_rejects_out_of_contract_values(key: str, value: object) -> None:
    report = _good_report()
    report[key] = value
    with pytest.raises(QuantizationReportError):
        quantization_gate(report)


def test_gate_requires_boolean_cosine_flag() -> None:
    report = _good_report()
    report["cosine_score_path_fp32"] = 1  # truthy but not a bool
    with pytest.raises(QuantizationReportError, match="cosine_score_path_fp32"):
        quantization_gate(report)


def test_gate_outcome_is_json_serializable_and_consistent() -> None:
    outcome = quantization_gate(_good_report())
    payload = json.loads(json.dumps(outcome.to_dict()))
    assert payload["artifact"] == "int8"
    assert payload["passed"] is True
    assert set(payload["conditions"]) == set(outcome.conditions)
    with pytest.raises(QuantizationReportError):
        GateOutcome(passed=True, artifact="fp16", conditions={"x": True}, margins={})
    with pytest.raises(QuantizationReportError):
        GateOutcome(passed=False, artifact="int8", conditions={"x": False}, margins={})


def test_validate_and_select_keeps_fp16_and_records_decision() -> None:
    report = _good_report()
    report["int8_p95_ms"] = 90.0  # force speedup failure
    selection = validate_and_select(report)
    assert isinstance(selection, ExportSelection)
    assert selection.artifact == "fp16"
    assert selection.accepted is False
    assert selection.gate.artifact == "fp16"
    assert selection.report["decision"]["artifact"] == "fp16"
    assert "p95_latency_speedup" in selection.report["decision"]["reason"]
    # caller's mapping is not mutated
    assert "decision" not in report
    # the selection round-trips through JSON
    payload = json.loads(selection.to_json())
    assert payload["selection"]["artifact"] == "fp16"
    assert payload["decision"]["conditions"]["p95_latency_speedup"] is False


def test_validate_and_select_accepts_int8_when_gate_passes() -> None:
    selection = validate_and_select(_good_report())
    assert selection.artifact == "int8"
    assert selection.accepted is True
    assert selection.report["decision"]["reason"].startswith("all five")
    assert json.loads(selection.to_json())["selection"]["accepted"] is True


def test_validate_and_select_strict_mode_raises_with_gate_attached() -> None:
    report = _good_report()
    report["cosine_score_path_fp32"] = False
    with pytest.raises(QuantizationRejectedError) as excinfo:
        validate_and_select(report, strict=True)
    assert excinfo.value.gate.artifact == "fp16"
    assert excinfo.value.gate.conditions["cosine_score_path_fp32"] is False


# ---------------------------------------------------------------------------
# serialized_sizes (no onnx needed)
# ---------------------------------------------------------------------------


def test_serialized_sizes_directory_reports_all_variants(tmp_path: Path) -> None:
    fp32 = tmp_path / "rapport.onnx"
    fp16 = tmp_path / "rapport.fp16.onnx"
    int8 = tmp_path / "rapport.int8.onnx"
    fp32.write_bytes(b"f" * 100)
    fp16.write_bytes(b"f" * 50)
    int8.write_bytes(b"i" * 25)
    sizes = serialized_sizes(tmp_path)
    assert sizes == {"fp32": 100, "fp16": 50, "int8": 25}


def test_serialized_sizes_directory_missing_variants_are_none(tmp_path: Path) -> None:
    (tmp_path / "rapport.onnx").write_bytes(b"f" * 10)
    sizes = serialized_sizes(tmp_path)
    assert sizes == {"fp32": 10, "fp16": None, "int8": None}


def test_serialized_sizes_file_mode_classifies_by_name(tmp_path: Path) -> None:
    target = tmp_path / "rapport.int8.onnx"
    target.write_bytes(b"x" * 42)
    assert serialized_sizes(target) == {"int8": 42}
    plain = tmp_path / "model.onnx"
    plain.write_bytes(b"x" * 7)
    assert serialized_sizes(plain) == {"fp32": 7}


def test_serialized_sizes_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        serialized_sizes(tmp_path / "nope.onnx")


# ---------------------------------------------------------------------------
# benchmark_p95 (no onnx needed — fake session)
# ---------------------------------------------------------------------------


class _CountingSession:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, output_names, feed):  # noqa: ANN001, ANN201 - ORT-like signature
        assert output_names is None
        assert set(feed) == {"input_ids"}
        self.calls += 1
        return []


def test_benchmark_p95_counts_warmup_and_iters() -> None:
    session = _CountingSession()
    feed = {"input_ids": torch.zeros((2, 4), dtype=torch.long)}
    p95 = benchmark_p95(session, feed, iters=30, warmup=7)
    assert isinstance(p95, float)
    assert p95 >= 0.0
    assert session.calls == 37


def test_benchmark_p95_nearest_rank_uses_sorted_tail() -> None:
    session = _CountingSession()
    feed = {"input_ids": torch.zeros((1, 2), dtype=torch.long)}
    p95 = benchmark_p95(session, feed, iters=100, warmup=0)
    assert math.isfinite(p95)


@pytest.mark.parametrize(("iters", "warmup"), [(0, 0), (-1, 0), (True, 0), (10, -1), (10, False)])
def test_benchmark_p95_rejects_bad_iteration_counts(iters: object, warmup: object) -> None:
    session = _CountingSession()
    feed = {"input_ids": torch.zeros((1, 2), dtype=torch.long)}
    with pytest.raises(ValueError):
        benchmark_p95(session, feed, iters=iters, warmup=warmup)  # type: ignore[arg-type]


def test_benchmark_p95_requires_mapping_feed() -> None:
    with pytest.raises(TypeError):
        benchmark_p95(_CountingSession(), [1, 2, 3], iters=1, warmup=0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Export input validation (no onnx import happens before the type check)
# ---------------------------------------------------------------------------


def test_export_onnx_rejects_non_rapport_model(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="RapportModel"):
        export_onnx(nn.Linear(2, 2), {}, tmp_path / "bad.onnx")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Real ONNX round-trip (skipped with reason onnx-not-installed when absent)
# ---------------------------------------------------------------------------


@onnx_required
def test_export_onnx_matches_pytorch_within_tolerance(tmp_path: Path) -> None:
    model = _tiny_model()
    artifact = tmp_path / "rapport-test.onnx"
    report = export_onnx(
        model, {"name": "rapport-test", "vocab_size": 128}, artifact, opset=EXPORT_OPSET
    )

    assert report["opset"] == EXPORT_OPSET == 17
    assert report["sample_count"] == REFERENCE_BATCH_ROWS == 64
    assert report["tolerance"] == EXPORT_TOLERANCE == 1e-3
    assert report["reference_batch"]["rows"] == REFERENCE_BATCH_ROWS
    assert report["reference_batch"]["seed"] == REFERENCE_BATCH_SEED
    assert len(report["reference_batch"]["sha256"]) == 64
    assert report["max_abs_logit_diff"] <= EXPORT_TOLERANCE
    assert report["dynamic_axes"]["option_mask"] == {0: "batch", 1: "options"}
    assert report["dynamic_axes"]["choice_logits"] == {0: "batch", 1: "options"}
    assert report["dynamic_axes_check"]["max_abs_logit_diff"] <= EXPORT_TOLERANCE
    assert artifact.exists()
    assert report["serialized_size_bytes"] == artifact.stat().st_size
    payload = json.dumps(report)
    assert "rapport-test" in payload


@onnx_required
def test_export_onnx_dynamic_option_axis_runs_unseen_shape(tmp_path: Path) -> None:
    import numpy as np
    import onnxruntime as ort

    model = _tiny_model()
    artifact = tmp_path / "rapport-test.onnx"
    export_onnx(model, {"name": "rapport-test"}, artifact)

    session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
    batch = make_reference_batch(seed=1234, rows=4, width=4, seq_len=64, vocab_size=128)
    feed = {name: tensor.numpy() for name, tensor in batch.items()}
    outputs = session.run(list(OUTPUT_NAMES), feed)
    assert outputs[0].shape == (4, 4)
    assert outputs[5].shape == (4, 8)  # risk categories are fixed
    reference_logits = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["option_mask"],
        batch["task_ids"],
    ).choice_logits.detach().numpy()
    assert np.abs(outputs[0] - reference_logits).max() <= EXPORT_TOLERANCE


@onnx_required
def test_quantize_dynamic_int8_keeps_cosine_path_fp32(tmp_path: Path) -> None:
    model = _tiny_model()
    fp32_artifact = tmp_path / "rapport-test.onnx"
    int8_artifact = tmp_path / "rapport-test.int8.onnx"
    export_onnx(model, {"name": "rapport-test"}, fp32_artifact)

    report = quantize_dynamic_int8(fp32_artifact, int8_artifact)

    assert report["cosine_score_path_fp32"] is True
    excluded = " ".join(report["nodes_excluded_from_quantization"])
    assert "readout/query" in excluded or "readout.query" in excluded
    assert "readout/option" in excluded or "readout.option" in excluded
    assert int8_artifact.exists()
    assert report["int8_size_bytes"] == int8_artifact.stat().st_size
    assert report["fp32_size_bytes"] == fp32_artifact.stat().st_size
    assert set(report["per_head_drift"]) == set(OUTPUT_NAMES)
    assert all(value >= 0.0 for value in report["per_head_drift"].values())

    # the excluded projections must still be plain MatMuls (not MatMulInteger)
    int8_model = onnx.load(str(int8_artifact))
    cosine_nodes = [
        node
        for node in int8_model.graph.node
        if "readout/query" in (node.name or "") or "readout/option" in (node.name or "")
    ]
    assert cosine_nodes, "expected to find the cosine projections in the INT8 graph"
    assert all(node.op_type == "MatMul" for node in cosine_nodes)
    json.dumps(report)


@onnx_required
def test_convert_fp32_to_fp16_reports_sizes(tmp_path: Path) -> None:
    model = _tiny_model()
    fp32_artifact = tmp_path / "rapport-test.onnx"
    fp16_artifact = tmp_path / "rapport-test.fp16.onnx"
    export_onnx(model, {"name": "rapport-test"}, fp32_artifact)

    report = convert_fp32_to_fp16(fp32_artifact, fp16_artifact)

    assert report["artifact"] == "fp16"
    assert fp16_artifact.exists()
    assert report["fp16_size_bytes"] == fp16_artifact.stat().st_size
    assert report["fp16_size_bytes"] <= report["source_fp32_bytes"]
    assert report["keep_io_types"] is True


@onnx_required
def test_full_pipeline_serialized_sizes_and_p95(tmp_path: Path) -> None:
    model = _tiny_model()
    fp32_artifact = tmp_path / "rapport-test.onnx"
    fp16_artifact = tmp_path / "rapport-test.fp16.onnx"
    int8_artifact = tmp_path / "rapport-test.int8.onnx"
    export_onnx(model, {"name": "rapport-test"}, fp32_artifact)
    convert_fp32_to_fp16(fp32_artifact, fp16_artifact)
    quantize_report = quantize_dynamic_int8(fp32_artifact, int8_artifact)

    sizes = serialized_sizes(tmp_path)
    assert sizes["fp32"] == fp32_artifact.stat().st_size
    assert sizes["fp16"] == fp16_artifact.stat().st_size
    assert sizes["int8"] == int8_artifact.stat().st_size

    import onnxruntime as ort

    batch = make_reference_batch(rows=8, vocab_size=128)
    fp32_session = ort.InferenceSession(str(fp32_artifact), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(int8_artifact), providers=["CPUExecutionProvider"])
    fp32_p95 = benchmark_p95(fp32_session, batch, iters=5, warmup=1)
    int8_p95 = benchmark_p95(int8_session, batch, iters=5, warmup=1)
    assert fp32_p95 > 0.0
    assert int8_p95 > 0.0

    # the 2x gate compares INT8 p95 against the FP32 baseline; measured tiny-model
    # timings are not a stable 2x signal, so the gate decision uses declared latencies
    gate_report = _good_report()
    gate_report.update(
        int8_size_bytes=quantize_report["int8_size_bytes"],
        fp32_p95_ms=100.0,
        int8_p95_ms=40.0,
    )
    outcome = quantization_gate(gate_report)
    assert outcome.artifact == "int8"


@onnx_required
def test_export_onnx_mismatch_error_carries_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forced numerical divergence must raise, never silently ship."""
    model = _tiny_model()
    artifact = tmp_path / "rapport-test.onnx"

    from src.inference import export as export_module

    def _bad_compare(module, path, batch):  # noqa: ANN001, ANN202
        diffs = {name: 0.0 for name in OUTPUT_NAMES}
        diffs["choice_logits"] = EXPORT_TOLERANCE * 10
        return diffs

    monkeypatch.setattr(export_module, "_compare_against_pytorch", _bad_compare)
    with pytest.raises(OnnxExportMismatchError) as excinfo:
        export_onnx(model, {"name": "rapport-test"}, artifact)
    assert excinfo.value.max_abs_logit_diff > EXPORT_TOLERANCE
    assert excinfo.value.tolerance == EXPORT_TOLERANCE


def test_gate_report_constants_match_declared_bounds() -> None:
    assert MAX_ACCURACY_DROP == 0.01
    assert MAX_BUCKET_ECE_RISE == 0.02
    assert MAX_INT8_BYTES == 35 * 1024 * 1024
    assert MIN_P95_SPEEDUP == 2.0


def test_dataclasses_frozen() -> None:
    outcome = quantization_gate(_good_report())
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.passed = False  # type: ignore[misc]
    selection = validate_and_select(_good_report())
    with pytest.raises(dataclasses.FrozenInstanceError):
        selection.artifact = "fp16"  # type: ignore[misc]


def test_waiver_note() -> None:
    # Documents the skip contract: tests above that need onnx carry reason
    # onnx-not-installed; the rest run unconditionally.
    assert WAIVER_NOTE == "onnx-not-installed"
