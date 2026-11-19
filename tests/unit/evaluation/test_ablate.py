"""Tests for src/evaluation/ablate.py — ablation arms A1–A8 (FR-EVAL-04).

Synthetic, offline evidence only. Covers the claim-printing boundary at one
seed standard deviation, arm adjudication (pass/fail/not-measured), arms
missing data, and the keyword-free paths that do not depend on recorded runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.ablate import (  # noqa: E402
    ARM_IDS,
    REQUIRED_SEEDS,
    arm_failures,
    claim_distinguishable,
    collect_claims,
    compare_arms,
    evaluate_arms,
    seed_std,
)

FINGERPRINT = "ab" * 32


def test_required_seeds_match_preregistered():
    assert REQUIRED_SEEDS == (13, 29, 47)
    assert ARM_IDS == ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8")


def test_seed_std_sample_estimator():
    assert seed_std([0.5]) == 0.0
    assert seed_std([]) == 0.0
    # sample std of [0, 1, 2] is exactly 1.0
    assert seed_std([0.0, 1.0, 2.0]) == 1.0


def test_claim_distinguishable_boundary_at_one_seed_std():
    # diffs [0, 1, 2]: mean == std == 1.0 — NOT strictly greater, no claim
    assert claim_distinguishable([0.0, 1.0, 2.0]) is False
    # diffs [0, 2, 3]: mean 5/3 > std sqrt(14/9) ≈ 1.247 — distinguishable
    assert claim_distinguishable([0.0, 2.0, 3.0]) is True
    # single seed cannot establish a claim
    assert claim_distinguishable([5.0]) is False
    # zero-variance positive gap is distinguishable (std 0)
    assert claim_distinguishable([2.0, 2.0, 2.0]) is True


def test_compare_arms_claim_text_and_threshold():
    claim = compare_arms(
        "A1", {13: 0.70, 29: 0.75, 47: 0.73},
        "random", {13: 0.60, 29: 0.70, 47: 0.65},
        metric="choice_accuracy",
    )
    assert claim is not None
    assert claim["n_seeds"] == 3
    # diffs [0.10, 0.05, 0.08]: mean 0.0767, std 0.0252 — distinguishable
    assert claim["distinguishable"] is True
    assert "beats" in claim["text"]
    assert claim["mean_gap_points"] > 0.0

    quiet = compare_arms(
        "A1", {13: 0.70, 29: 0.66, 47: 0.72},
        "random", {13: 0.70, 29: 0.70, 47: 0.70},
        metric="choice_accuracy",
    )
    assert quiet is not None
    assert quiet["distinguishable"] is False
    assert "not distinguishable at n=3 seeds" in quiet["text"]


def test_compare_arms_no_overlap_returns_none():
    assert compare_arms("A1", {13: 0.7}, "ref", {29: 0.6}, metric="m") is None


def test_missing_arm_renders_not_measured_never_omitted():
    results = evaluate_arms({})
    assert [arm.arm for arm in results] == list(ARM_IDS)
    for arm in results:
        assert arm.status == "not-measured"
        assert arm.per_seed == {}
        assert arm.mean is None
        assert arm.missing, arm.arm


def test_partial_arm_data_keeps_failed_arm():
    results = evaluate_arms({
        "A4": {"weight_collapse": True, "summary": "task collapsed below floor"},
    })
    by_id = {arm.arm: arm for arm in results}
    assert by_id["A4"].status == "fail"
    assert "collapsed" in by_id["A4"].summary
    # every other arm is still rendered
    assert [arm.arm for arm in results] == list(ARM_IDS)


def test_arm_failures_lists_failed_arms_only():
    results = evaluate_arms({"A4": {"weight_collapse": True}, "A7": {"status": "pass"}})
    failures = arm_failures(results)
    assert len(failures) == 1
    assert failures[0].startswith("arm A4 FAIL")


def test_a1_gap_adjudicated_with_margin():
    passing = evaluate_arms({
        "A1": {
            "values": {13: 0.70, 29: 0.72, 47: 0.71},
            "reference": {"name": "random", "values": {13: 0.60, 29: 0.62, 47: 0.61}},
        },
    })
    arm = {a.arm: a for a in passing}["A1"]
    assert arm.status == "pass"
    assert arm.comparison is not None
    assert arm.comparison["distinguishable"] is True

    failing = evaluate_arms({
        "A1": {
            "values": {13: 0.63, 29: 0.64, 47: 0.62},
            "reference": {"name": "random", "values": {13: 0.61, 29: 0.62, 47: 0.60}},
        },
    })
    arm = {a.arm: a for a in failing}["A1"]
    assert arm.status == "fail"


def test_a1_missing_reference_renders_not_measured():
    results = evaluate_arms({"A1": {"values": {13: 0.7, 29: 0.7, 47: 0.7}}})
    arm = {a.arm: a for a in results}["A1"]
    assert arm.status == "not-measured"
    assert any("reference" in key for key in arm.missing)


def test_a4_weight_collapse_outcomes():
    collapsed = evaluate_arms({"A4": {"weight_collapse": True}})
    assert {a.arm: a for a in collapsed}["A4"].status == "fail"
    healthy = evaluate_arms({"A4": {"weight_collapse": False}})
    assert {a.arm: a for a in healthy}["A4"].status == "pass"
    unwatched = evaluate_arms({"A4": {"per_seed": {13: 0.1}}})
    assert {a.arm: a for a in unwatched}["A4"].status == "not-measured"
    by_epochs = evaluate_arms({"A4": {"collapse_epochs": 2}})
    assert {a.arm: a for a in by_epochs}["A4"].status == "fail"


def test_a6_stage_gains_rule():
    ok = evaluate_arms({"A6": {"stage_gains": [2.5, 1.5, 3.0]}})
    assert {a.arm: a for a in ok}["A6"].status == "pass"
    weak = evaluate_arms({"A6": {"stage_gains": [2.5, 0.5, 3.0]}})
    assert {a.arm: a for a in weak}["A6"].status == "fail"
    missing = evaluate_arms({"A6": {}})
    assert {a.arm: a for a in missing}["A6"].status == "not-measured"


def test_a8_uses_length_extension_verdict():
    ok = evaluate_arms({"A8": {"acc512": 0.80, "acc1024": 0.79}})
    assert {a.arm: a for a in ok}["A8"].status == "pass"
    bad = evaluate_arms({"A8": {"acc512": 0.80, "acc1024": 0.77}})
    assert {a.arm: a for a in bad}["A8"].status == "fail"
    missing = evaluate_arms({"A8": {"acc512": 0.80}})
    arm = {a.arm: a for a in missing}["A8"]
    assert arm.status == "not-measured"
    assert any("acc1024" in key for key in arm.missing)


def test_explicit_status_is_honoured_but_validated():
    results = evaluate_arms([
        {"arm": "A2", "status": "FAIL", "summary": "cosine lost to cross-attn"}
    ])
    arm = {a.arm: a for a in results}["A2"]
    assert arm.status == "fail"
    assert "cosine" in arm.summary
    bogus = evaluate_arms([{"arm": "A2", "status": "winner"}])
    assert {a.arm: a for a in bogus}["A2"].status == "not-measured"


def test_list_form_and_mapping_form_agree():
    as_map = evaluate_arms({"A1": {"status": "pass"}})
    as_list = evaluate_arms([{"arm": "A1", "status": "pass"}])
    assert {a.arm: a.status for a in as_map} == {a.arm: a.status for a in as_list}


def test_collect_claims_only_comparison_arms():
    results = evaluate_arms({
        "A1": {
            "values": {13: 0.70, 29: 0.72, 47: 0.71},
            "reference": {"name": "random", "values": {13: 0.60, 29: 0.62, 47: 0.61}},
        },
        "A4": {"weight_collapse": False},
    })
    claims = collect_claims(results)
    assert [claim["arm"] for claim in claims] == ["A1"]
    assert claims[0]["distinguishable"] is True


def test_unknown_arm_ids_are_appended_not_dropped():
    results = evaluate_arms([{"arm": "A9", "status": "fail", "summary": "off-protocol arm"}])
    assert results[-1].arm == "A9"
    assert results[-1].status == "fail"
    assert len(results) == len(ARM_IDS) + 1
