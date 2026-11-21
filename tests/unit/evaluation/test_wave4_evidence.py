"""Wave-4 regression tests for the repaired acceptance-evidence gates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    MetricInputError,
    MetricRecord,
    aurc,
    compute_metrics,
)
from src.evaluation.report import render_verdict  # noqa: E402

FP = "ab" * 32
RISK_CATEGORIES = (
    "violence",
    "self_harm",
    "sexual_content",
    "hate_speech",
    "harassment",
    "dangerous_content",
    "illegal_activity",
    "privacy_violation",
)


def _runs() -> list[dict]:
    return [
        {
            "run_id": "run-13",
            "config_hash": "cfg",
            "data_fingerprint": FP,
            "seed": 13,
            "device": "cuda:0",
            "git_sha": "deadbee",
            "determinism": "strict",
            "versions": {"python": "3.12.0"},
        }
    ]


def _rows() -> dict:
    return {row.id: row for row in render_verdict({"runs": _runs(), "metrics": {}})}


def _risk_measurements(*, tp: int = 90, fp: int = 0, fn: int = 10, tn: int = 1000):
    measurements = {}
    for category in RISK_CATEGORIES:
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        measurements[category] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "recall": recall,
            "fpr": fpr,
        }
    return measurements


def _y7_block(**overrides):
    block = {
        "cohort": "aegis",
        "split_manifest_fingerprint": FP,
        "thresholds_from": "dev",
        "risk_thresholds_fingerprint": "cd" * 32,
        "fpr_threshold": 0.10,
        "measurements": _risk_measurements(),
        "keyword_risk_baseline": {
            "recall": {category: 0.4 for category in RISK_CATEGORIES}
        },
    }
    block.update(overrides)
    return block


def _y8_block(**overrides):
    block = {
        "cohort": "xstest",
        "split_manifest_fingerprint": FP,
        "thresholds_from": "dev",
        "risk_thresholds_fingerprint": "cd" * 32,
        "xstest_over_refusal_rate": 0.05,
        "n_safe": 200,
        "n_flagged": 10,
    }
    block.update(overrides)
    return block


def test_y7_rejects_empty_or_contradictory_counts() -> None:
    zero = render_verdict(
        {
            "runs": _runs(),
            "metrics": {},
            "y7": _y7_block(
                measurements=_risk_measurements(tp=0, fp=0, fn=0, tn=0)
            ),
        }
    )
    assert {row.id: row for row in zero}["Y7"].status == "not-measured"

    lying = render_verdict(
        {
            "runs": _runs(),
            "metrics": {},
            "y7": _y7_block(
                measurements=_risk_measurements(tp=90, fp=50, fn=10, tn=50)
            ),
        }
    )
    assert {row.id: row for row in lying}["Y7"].status == "not-measured"


def test_y8_requires_a_consistent_safe_cohort() -> None:
    rows = render_verdict({"runs": _runs(), "metrics": {}, "y8": _y8_block()})
    assert {row.id: row for row in rows}["Y8"].status != "not-measured"

    broken = render_verdict(
        {"runs": _runs(), "metrics": {}, "y8": _y8_block(n_flagged=5)}
    )
    assert {row.id: row for row in broken}["Y8"].status == "not-measured"

    missing = render_verdict(
        {"runs": _runs(), "metrics": {}, "y8": _y8_block(n_safe=None)}
    )
    assert {row.id: row for row in missing}["Y8"].status == "not-measured"


def test_y9_rejects_unsynchronized_or_emulated_timing() -> None:
    benchmark = {
        "device": "NVIDIA T4",
        "precision": "fp16",
        "batch": [1, 32],
        "latency_method": "unsynchronized perf_counter",
        "p50_batch1_ms": 1.0,
        "throughput_batch32": 100.0,
    }
    rows = render_verdict({"runs": _runs(), "metrics": {}, "benchmark": benchmark})
    assert {row.id: row for row in rows}["Y9"].status == "not-measured"

    emulated = dict(benchmark, device="T4 emulated on CPU")
    rows = render_verdict({"runs": _runs(), "metrics": {}, "benchmark": emulated})
    assert {row.id: row for row in rows}["Y9"].status == "not-measured"


def test_y11_rejects_impossible_counts() -> None:
    rows = render_verdict(
        {"runs": _runs(), "metrics": {}, "benchmark": {"parameters": -1, "fp32_size_bytes": -4}}
    )
    assert {row.id: row for row in rows}["Y11"].status == "not-measured"


def test_y12_requires_consistent_gate_and_retained_artifact() -> None:
    block = {
        "gate_outcome": {
            "passed": True,
            "artifact": "fp16",
            "conditions": {
                "choice_accuracy_drop": True,
                "bucket_macro_ece_rise": True,
                "cosine_score_path_fp32": True,
                "serialized_size": True,
                "p95_latency_speedup": True,
            },
        },
        "retained_artifact": "fp16",
        "validated_artifacts": {
            "fp16": {
                "path": "/artifacts/fp16.onnx",
                "size_bytes": 10,
                "onnx_checker": True,
                "ort_session": True,
                "parity_validated": True,
                "sha256": "ab" * 32,
            }
        },
        "accuracy_drop_points": 0.1,
    }
    rows = render_verdict({"runs": _runs(), "metrics": {}, "int8_export": block})
    assert {row.id: row for row in rows}["Y12"].status == "not-measured"


def test_y14_rejects_an_unbound_scalar() -> None:
    rows = render_verdict(
        {"runs": _runs(), "metrics": {}, "reproducibility_delta_points": 0.0}
    )
    assert {row.id: row for row in rows}["Y14"].status == "not-measured"


def test_aurc_uses_the_exact_tie_order_expectation() -> None:
    # Distinct confidence plus a three-row tied block: the expected AURC is
    # the mean over every ordering of the tied rows.
    confidences = [0.9, 0.5, 0.5, 0.5]
    correct = [1, 1, 0, 0]
    assert aurc(confidences, correct) == pytest.approx(0.3194444444444444, abs=1e-9)


def test_over_refusal_is_unmeasured_with_an_incomplete_threshold_map() -> None:
    record = MetricRecord(
        chosen_index=0,
        predicted_index=0,
        scores=(0.9, 0.1),
        gold_scores=(0.9, 0.1),
        probabilities=(0.9, 0.1),
        risk_targets=(0, 0, 0, 0, 0, 0, 0, 0),
        risk_probabilities=(0.0, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99),
    )
    partial = {"violence": {"threshold": 0.9}}
    result = compute_metrics([record], risk_thresholds=partial)
    assert result.over_refusal_rate is None
    assert result.over_refusal_n_safe == 1
    assert result.over_refusal_n_flagged is None

    with pytest.raises(MetricInputError):
        compute_metrics([record], risk_thresholds={"violence": {"threshold": float("nan")}})
