"""Tests for src/evaluation/report.py — the single evaluation report (FR-EVAL-04/05).

Synthetic evidence only: three seeds {13, 29, 47}, a fake held-out source and
stub baselines. The report never trains anything; every input below is a
recorded-run-shaped dict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.ablate import REQUIRED_SEEDS  # noqa: E402
from src.evaluation.report import (  # noqa: E402
    BLOCKING_CRITERIA,
    build_report,
    exit_code,
    render_verdict,
)

FINGERPRINT = "ab" * 32
FINGERPRINT_OTHER = "cd" * 32
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


def make_metrics(
    *,
    accuracy: float = 0.70,
    ece_pooled: float = 0.02,
    ece_bucket_macro: float = 0.03,
    sel80: float = 0.78,
    over_refusal: float = 0.05,
    aegis_recall: float | dict[str, float] = 0.90,
    full_acc: float | None = None,
) -> dict:
    if isinstance(aegis_recall, dict):
        aegis_per_category = aegis_recall
    else:
        aegis_per_category = {category: aegis_recall for category in RISK_CATEGORIES}
    full = accuracy if full_acc is None else full_acc
    return {
        "choice": {"test": {"accuracy": accuracy, "macro_f1": accuracy - 0.01}},
        "probability": {
            "test": {
                "ece_pooled": ece_pooled,
                "ece_bucket_macro": ece_bucket_macro,
                "selective_accuracy_80": sel80,
                "accuracy": full,
                "brier": 0.1,
                "nll": 0.3,
            }
        },
        "abstention": {"test": {"aurc": 0.2}},
        "risk": {
            "test": {
                "over_refusal_rate": over_refusal,
                "risk_recall_at_fpr": dict(aegis_per_category),
            }
        },
    }


def make_robustness(*, ood_accuracy: float = 0.72, fingerprint: str | None = FINGERPRINT,
                    within_tolerance: bool = True, accuracy_delta: float = 0.001) -> dict:
    ood = {
        "accuracy": ood_accuracy,
        "ood_source": "UltraFeedback-binarized",
        "n_rows": 8000,
        "empty": False,
        "by_family": {},
    }
    if fingerprint is not None:
        ood["split_manifest_fingerprint"] = fingerprint
    return {
        "seed": 13,
        "split_manifest_fingerprint": fingerprint,
        "ood_source": "UltraFeedback-binarized",
        "rows": {"test": 12000},
        "permutation": {
            "overall": {
                "within_tolerance": within_tolerance,
                "accuracy_delta": accuracy_delta,
                "score_max_deviation": 0.0,
                "score_tolerance": 1e-4,
                "families": ["choice"],
            }
        },
        "option_count_sweep": {"overall": {}},
        "ood": ood,
        "context": {},
        "length_sweep": {},
        "notes": [],
    }


def make_benchmark() -> dict:
    return {
        "p50_batch1_ms": 9.0,
        "p95_batch1_ms": 11.0,
        "throughput_batch32": 350.0,
        "parameters": 33_460_000,
        "fp32_size_bytes": 130 * 1024 * 1024,
        "int8": {"accuracy_drop_points": 0.4},
    }


def make_keyword_baseline(recall: float = 0.4) -> dict:
    return {
        "recall": {category: recall for category in RISK_CATEGORIES},
        "false_positive_rate": {category: 0.02 for category in RISK_CATEGORIES},
        "over_refusal_rate": 0.08,
        "n_items": 200,
    }


def make_runs(
    *,
    metrics: dict | None = None,
    robustness: dict | None = "default",
    benchmark: dict | None = "default",
    evidence: dict | None = None,
    fingerprint: str | None = FINGERPRINT,
    seeds: tuple[int, ...] = REQUIRED_SEEDS,
) -> list[dict]:
    base_metrics = metrics if metrics is not None else make_metrics()
    runs = []
    for seed in seeds:
        run = {
            "run_id": f"run-{seed}",
            "config_hash": f"cfg-{seed}",
            "data_fingerprint": fingerprint,
            "seed": seed,
            "device": "cuda:0",
            "git_sha": "deadbee",
            "determinism": "deterministic",
            "versions": {"python": "3.12.0", "torch": "2.4.0"},
            "metrics": json.loads(json.dumps(base_metrics)),
        }
        if seed == seeds[0]:
            if robustness == "default":
                run["robustness"] = make_robustness()
            elif robustness is not None:
                run["robustness"] = robustness
            if benchmark == "default":
                run["benchmark"] = make_benchmark()
            elif benchmark is not None:
                run["benchmark"] = benchmark
            run["evidence"] = {
                "reproducibility_delta_points": 0.1,
                "resume": {"abs_diff": 1e-5},
                "keyword_risk_baseline": make_keyword_baseline(),
            }
            if evidence:
                run["evidence"].update(evidence)
        runs.append(run)
    return runs


def make_baselines(*, b3_accuracy: float = 0.55, b4_ood_accuracy: float = 0.60) -> dict:
    return {
        "B3-tfidf-logreg": {
            "test": {"accuracy": b3_accuracy},
            "test_per_family": {"choice": {"accuracy": b3_accuracy}},
            "split_manifest_fingerprint": FINGERPRINT,
        },
        "B4-frozen-embedding-cosine": {
            "test": {"accuracy": 0.50},
            "test_per_family": {"choice": {"accuracy": 0.50}},
            "ood": {"accuracy": b4_ood_accuracy},
            "split_manifest_fingerprint": FINGERPRINT,
        },
    }


def make_ablations() -> dict:
    return {
        "A1": {
            "values": {13: 0.70, 29: 0.72, 47: 0.71},
            "reference": {
                "name": "random",
                "values": {13: 0.60, 29: 0.62, 47: 0.61},
            },
        },
        "A2": {"status": "pass"},
        "A3": {"status": "pass"},
        "A4": {"weight_collapse": False},
        "A5": {"status": "pass"},
        "A6": {"stage_gains": [1.5, 2.0]},
        "A7": {"status": "pass"},
        "A8": {"acc512": 0.80, "acc1024": 0.79},
    }


def test_block_list_covers_all_fourteen():
    assert BLOCKING_CRITERIA == tuple(f"Y{i}" for i in range(1, 15))


def test_happy_path_writes_report_and_exits_zero(tmp_path):
    result = build_report(make_runs(), make_baselines(), make_ablations(), tmp_path)
    assert result.exit_code == 0
    assert result.report_json_path.exists()
    assert result.report_md_path.exists()
    data = json.loads(result.report_json_path.read_text())
    assert [arm["arm"] for arm in data["arms"]] == [f"A{i}" for i in range(1, 9)]
    assert [row["id"] for row in data["verdicts"]] == [f"Y{i}" for i in range(1, 15)]
    statuses = {row["id"]: row["status"] for row in data["verdicts"]}
    assert all(status == "pass" for status in statuses.values()), statuses
    md = result.report_md_path.read_text()
    for section in ("## §3", "## §5", "## §6", "## §7", "## §8", "## Negative results"):
        assert section in md
    assert "not distinguishable at n=3 seeds" not in md or "CLAIM" in md
    assert "none observed" in md


def test_failed_arm_is_included_in_report_and_negative_results(tmp_path):
    ablations = make_ablations()
    ablations["A4"] = {"weight_collapse": True, "summary": "task collapsed below floor"}
    result = build_report(make_runs(), make_baselines(), ablations, tmp_path)
    data = json.loads(result.report_json_path.read_text())
    a4 = next(arm for arm in data["arms"] if arm["arm"] == "A4")
    assert a4["status"] == "fail"
    md = result.report_md_path.read_text()
    assert "A4" in md and "fail" in md
    assert any("A4" in entry for entry in data["negative_results"])


def test_arm_missing_data_renders_not_measured_without_fabrication(tmp_path):
    ablations = {"A1": make_ablations()["A1"]}  # every other arm missing
    result = build_report(make_runs(), make_baselines(), ablations, tmp_path)
    data = json.loads(result.report_json_path.read_text())
    by_id = {arm["arm"]: arm for arm in data["arms"]}
    assert by_id["A3"]["status"] == "not-measured"
    assert by_id["A3"]["mean"] is None
    assert by_id["A3"]["per_seed"] == {}
    assert by_id["A3"]["missing"]
    assert "A3" in result.report_md_path.read_text()


def test_y1_fail_when_model_below_b3_plus_margin(tmp_path):
    runs = make_runs(metrics=make_metrics(accuracy=0.50))
    result = build_report(runs, make_baselines(b3_accuracy=0.55), make_ablations(), tmp_path)
    statuses = {row.id: row for row in result.verdicts}
    assert statuses["Y1"].status == "fail"
    assert result.exit_code == 1
    md = result.report_md_path.read_text()
    assert "Y1" in md
    assert "Blocking criteria FAILED" in md


def test_y1_pass_when_model_clears_b3_margin(tmp_path):
    result = build_report(make_runs(), make_baselines(b3_accuracy=0.55), make_ablations(), tmp_path)
    statuses = {row.id: row for row in result.verdicts}
    assert statuses["Y1"].status == "pass"


def test_y7_missing_input_renders_not_measured_never_pass(tmp_path):
    runs = make_runs()
    for run in runs:
        run["metrics"]["risk"]["test"].pop("risk_recall_at_fpr", None)
        run.get("evidence", {}).pop("keyword_risk_baseline", None)
    result = build_report(runs, make_baselines(), make_ablations(), tmp_path)
    statuses = {row.id: row for row in result.verdicts}
    assert statuses["Y7"].status == "not-measured"
    assert statuses["Y7"].status != "pass"
    assert any("aegis" in key for key in statuses["Y7"].missing)


def test_y7_missing_keyword_comparator_renders_not_measured(tmp_path):
    runs = make_runs()
    for run in runs:
        run.get("evidence", {}).pop("keyword_risk_baseline", None)
    result = build_report(runs, make_baselines(), make_ablations(), tmp_path)
    statuses = {row.id: row for row in result.verdicts}
    assert statuses["Y7"].status == "not-measured"
    assert any("keyword" in key for key in statuses["Y7"].missing)


def test_y7_pass_and_fail_by_weakest_category(tmp_path):
    passing = build_report(make_runs(), make_baselines(), make_ablations(), tmp_path)
    assert {row.id: row for row in passing.verdicts}["Y7"].status == "pass"

    weak = {category: 0.90 for category in RISK_CATEGORIES}
    weak["self_harm"] = 0.50
    runs = make_runs(metrics=make_metrics(aegis_recall=weak))
    failing = build_report(runs, make_baselines(), make_ablations(), tmp_path)
    row = {r.id: r for r in failing.verdicts}["Y7"]
    assert row.status == "fail"
    assert failing.exit_code == 1


def test_y8_missing_renders_not_measured_never_pass(tmp_path):
    runs = make_runs()
    for run in runs:
        run["metrics"]["risk"]["test"].pop("over_refusal_rate", None)
    result = build_report(runs, make_baselines(), make_ablations(), tmp_path)
    statuses = {row.id: row for row in result.verdicts}
    assert statuses["Y8"].status == "not-measured"
    assert statuses["Y8"].status != "pass"


def test_y8_fail_when_over_refusal_above_limit(tmp_path):
    runs = make_runs(metrics=make_metrics(over_refusal=0.25))
    result = build_report(runs, make_baselines(), make_ablations(), tmp_path)
    row = {r.id: r for r in result.verdicts}["Y8"]
    assert row.status == "fail"
    assert result.exit_code == 1


def test_y2_blocked_without_fingerprint_but_accepts_one(tmp_path):
    runs = make_runs(robustness=make_robustness(fingerprint=None))
    result = build_report(runs, make_baselines(), make_ablations(), tmp_path)
    row = {r.id: r for r in result.verdicts}["Y2"]
    assert row.status == "not-measured"
    assert any("fingerprint" in key for key in row.missing)

    ok = build_report(make_runs(), make_baselines(), make_ablations(), tmp_path)
    assert {r.id: r for r in ok.verdicts}["Y2"].status == "pass"


def test_y2_fail_when_ood_gap_below_margin(tmp_path):
    runs = make_runs(robustness=make_robustness(ood_accuracy=0.62))
    result = build_report(
        runs, make_baselines(b4_ood_accuracy=0.60), make_ablations(), tmp_path
    )
    row = {r.id: r for r in result.verdicts}["Y2"]
    assert row.status == "fail"
    assert result.exit_code == 1


def test_y6_fail_when_permutation_out_of_tolerance(tmp_path):
    runs = make_runs(robustness=make_robustness(within_tolerance=False, accuracy_delta=0.001))
    result = build_report(runs, make_baselines(), make_ablations(), tmp_path)
    row = {r.id: r for r in result.verdicts}["Y6"]
    assert row.status == "fail"
    assert result.exit_code == 1


def test_empty_negative_results_section_exits_nonzero(tmp_path):
    result = build_report(
        make_runs(),
        make_baselines(),
        make_ablations(),
        tmp_path,
        negative_results=[],  # explicit empty: no failures to synthesize from either
    )
    data = json.loads(result.report_json_path.read_text())
    assert data["negative_results"] == []
    assert data["negative_results_empty"] is True
    assert "INVALID" in result.report_md_path.read_text()
    assert result.exit_code == 1


def test_exit_code_semantics():
    row = render_verdict({"metrics": {}, "baselines": {}, "robustness": {}})[0]
    assert row.status == "not-measured"
    assert exit_code([]) == 0
    assert exit_code([{"id": "Y1", "status": "pass", "blocking": True}]) == 0
    assert exit_code([{"id": "Y1", "status": "fail", "blocking": True}]) == 1
    assert exit_code([{"id": "Y1", "status": "fail", "blocking": False}]) == 0
    assert exit_code([], negative_results=[]) == 1
    assert exit_code([], negative_results=["none observed"]) == 0


def test_metrics_render_per_family_per_split_with_three_seeds(tmp_path):
    result = build_report(make_runs(), make_baselines(), make_ablations(), tmp_path)
    md = result.report_md_path.read_text()
    data = json.loads(result.report_json_path.read_text())
    cell = data["metrics"]["main"]["choice"]["test"]["accuracy"]
    assert cell["n_seeds"] == 3
    assert cell["per_seed"] == {"13": 0.70, "29": 0.70, "47": 0.70}
    assert cell["mean"] == pytest.approx(0.70)
    assert "## §3" in md
    for seed in ("13", "29", "47"):
        assert seed in md
    assert "±" in md


def test_claim_boundary_not_distinguishable_is_not_printed_as_claim(tmp_path):
    ablations = make_ablations()
    # flat reference == arm: gap 0.04 mean, per-seed diffs [0.00, -0.04, 0.02]
    ablations["A1"] = {
        "values": {13: 0.70, 29: 0.66, 47: 0.72},
        "reference": {"name": "random", "values": {13: 0.70, 29: 0.70, 47: 0.70}},
    }
    result = build_report(make_runs(), make_baselines(), ablations, tmp_path)
    md = result.report_md_path.read_text()
    assert "not distinguishable at n=3 seeds" in md
    data = json.loads(result.report_json_path.read_text())
    claim = next(c for c in data["claims"] if c["arm"] == "A1")
    assert claim["distinguishable"] is False


def test_distinguishable_claim_is_printed(tmp_path):
    result = build_report(make_runs(), make_baselines(), make_ablations(), tmp_path)
    md = result.report_md_path.read_text()
    assert "CLAIM" in md
    assert "A1 beats random" in md


def test_inconsistent_fingerprints_warn(tmp_path):
    runs = make_runs(fingerprint=FINGERPRINT)
    runs[1]["data_fingerprint"] = FINGERPRINT_OTHER
    runs[2]["data_fingerprint"] = FINGERPRINT_OTHER
    result = build_report(runs, make_baselines(), make_ablations(), tmp_path)
    data = json.loads(result.report_json_path.read_text())
    assert data["manifest"]["fingerprint_consistent"] is False
    assert any("fingerprint" in warning for warning in data["warnings"])


def test_verdicts_missing_evidence_all_not_measured(tmp_path):
    result = build_report([], {}, {}, tmp_path)
    assert all(row.status == "not-measured" for row in result.verdicts)
    assert all(row.missing for row in result.verdicts)
    assert result.exit_code == 0  # not-measured is not a fail


def test_y14_duplicate_seed_gap_fail_and_pass(tmp_path):
    # two recorded runs of seed 13: 0.60 and 0.70 → 10 points apart (> 0.5)
    runs = make_runs()
    for run in runs:
        if run["run_id"] == "run-13":
            run["metrics"]["choice"]["test"]["accuracy"] = 0.60
            run["evidence"].pop("reproducibility_delta_points", None)
    repeat = json.loads(json.dumps(runs[0]))
    repeat["run_id"] = "run-13-repeat"
    repeat["metrics"]["choice"]["test"]["accuracy"] = 0.70
    runs = runs + [repeat]
    result = build_report(runs, make_baselines(), make_ablations(), tmp_path)
    row = {r.id: r for r in result.verdicts}["Y14"]
    assert row.status == "fail"

    close = make_runs()
    for run in close:
        if run["run_id"] == "run-13":
            run["metrics"]["choice"]["test"]["accuracy"] = 0.700
            run["evidence"].pop("reproducibility_delta_points", None)
    repeat = json.loads(json.dumps(close[0]))
    repeat["run_id"] = "run-13-repeat"
    repeat["metrics"]["choice"]["test"]["accuracy"] = 0.704  # 0.4 points apart
    close = close + [repeat]
    result = build_report(close, make_baselines(), make_ablations(), tmp_path)
    row = {r.id: r for r in result.verdicts}["Y14"]
    assert row.status == "pass"


def test_verdict_rows_rendered_in_order(tmp_path):
    result = build_report(make_runs(), make_baselines(), make_ablations(), tmp_path)
    assert [row.id for row in result.verdicts] == [f"Y{i}" for i in range(1, 15)]


def test_blocking_fail_elsewhere_does_not_get_hidden(tmp_path):
    runs = make_runs(metrics=make_metrics(ece_pooled=0.30))
    result = build_report(runs, make_baselines(), make_ablations(), tmp_path)
    row = {r.id: r for r in result.verdicts}["Y4"]
    assert row.status == "fail"
    assert result.exit_code == 1
    assert "Y4" in result.report_md_path.read_text()


def test_registry_like_runs_are_consumed(tmp_path):
    # RunRegistry-like injection: build a real registry over two run dirs.
    runs_root = tmp_path / "runs"
    for seed in REQUIRED_SEEDS:
        run_dir = runs_root / f"seed-{seed}"
        run_dir.mkdir(parents=True)
        manifest = {
            "run_id": f"seed-{seed}",
            "config_hash": "cfg",
            "data_fingerprint": FINGERPRINT,
            "seed": seed,
            "device": "cuda:0",
            "git_sha": "deadbee",
            "determinism": "deterministic",
            "versions": {"torch": "2.4.0"},
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    from src.ops.registry import RunRegistry

    registry = RunRegistry(runs_root)
    result = build_report(registry, make_baselines(), {}, tmp_path / "out")
    data = json.loads(result.report_json_path.read_text())
    assert data["manifest"]["run_ids"] == [f"seed-{seed}" for seed in REQUIRED_SEEDS]
    assert data["manifest"]["seeds"] == list(REQUIRED_SEEDS)
    assert [arm["arm"] for arm in data["arms"]] == [f"A{i}" for i in range(1, 9)]


def test_baseline_metrics_like_objects_are_consumed(tmp_path):
    class _BaselineMetricsLike:
        def __init__(self, name, test_acc, b4_ood=None):
            self._dump = {
                "baseline": name,
                "test": {"accuracy": test_acc},
                "test_per_family": {"choice": {"accuracy": test_acc}},
                "ood": {"accuracy": b4_ood} if b4_ood is not None else {},
                "split_manifest_fingerprint": FINGERPRINT,
            }

        def to_dict(self):
            return self._dump

    baselines = {
        "B3-tfidf-logreg": _BaselineMetricsLike("B3-tfidf-logreg", 0.55),
        "B4-frozen-embedding-cosine": _BaselineMetricsLike(
            "B4-frozen-embedding-cosine", 0.50, b4_ood=0.60
        ),
    }
    result = build_report(make_runs(), baselines, make_ablations(), tmp_path)
    statuses = {row.id: row for row in result.verdicts}
    assert statuses["Y1"].status == "pass"
    assert statuses["Y2"].status == "pass"


def test_report_md_embeds_recorded_risk_payload(tmp_path):
    result = build_report(make_runs(), make_baselines(), make_ablations(), tmp_path)
    md = result.report_md_path.read_text()
    assert "risk_recall_at_fpr" in md
    assert "violence" in md
