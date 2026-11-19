"""Tests for src/evaluation/baselines.py — FR-EVAL-02 fixtures.

Fixtures are synthetic and offline; the shared 40-row fixture was designed so
the classical baselines have an above-chance signal to find (separable
marker tokens). The repo root is placed on ``sys.path`` here because this
suite owns no conftest of its own.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.evaluation.metrics as metrics  # noqa: E402
from src.data.core.schema import (  # noqa: E402
    DecisionExample,
    Option,
    compute_example_id,
)
from src.data.splits.errors import ManifestError  # noqa: E402
from src.data.splits.manifest import SplitManifest, write_manifest  # noqa: E402
from src.evaluation.baselines import (  # noqa: E402
    ABSTENTION_RANDOM_SEED,
    ALL_BASELINES,
    B1_MAJORITY,
    B3_TFIDF,
    EMBEDDING_SOURCE_HASH,
    BaselineInputError,
    BaselineMetrics,
    FrozenEmbeddingCosineBaseline,
    MajorityClassBaseline,
    TfidfLogisticBaseline,
    load_manifest_fingerprint,
    run_baselines,
    y1_verdict,
)

FINGERPRINT = "a1" * 32  # lowercase sha256 hex
OTHER_FINGERPRINT = "b2" * 32


def _example(family, goal, state, options, chosen_index, split, group_id, scores=None,
             risk_labels=None):
    example_id = compute_example_id(family, goal, state, list(options))
    return DecisionExample(
        example_id=example_id,
        task_family=family,
        goal=goal,
        state=state,
        options=tuple(Option(text=t) for t in options),
        source="fixture",
        license="MIT",
        group_id=group_id,
        split=split,
        chosen_index=chosen_index,
        scores=scores,
        risk_labels=risk_labels,
    )


def _choice2(goal, split, gid, chosen):
    return _example("choice", goal, f"state for {goal}", ("Alpha option", "Beta option"),
                    chosen, split, gid)


def _preference2(goal, split, gid, chosen):
    return _example("preference", goal, f"state for {goal}", ("Alpha option", "Beta option"),
                    chosen, split, gid)


_OPTS3 = ("Gamma path", "Delta path", "Epsilon path")


def _choice3(goal, split, gid, chosen):
    return _example("choice", goal, f"state for {goal}", _OPTS3, chosen, split, gid)


def _abstention(goal, split, gid, chosen):
    return _example("abstention", goal, f"state for {goal}", ("Answerable", "Unanswerable"),
                    chosen, split, gid)


def fixture_rows():
    """40 rows: 24 train / 8 dev / 8 test, split tags consistent with the manifest."""
    train = []
    for i in range(5):
        train.append(_choice2(f"goal alpha case {i}", "train", f"c2a-{i}", 0))
        train.append(_choice2(f"goal beta case {i}", "train", f"c2b-{i}", 1))
    for i in range(2):
        train.append(_preference2(f"prefer alpha case {i}", "train", f"p2a-{i}", 0))
        train.append(_preference2(f"prefer beta case {i}", "train", f"p2b-{i}", 1))
    train.append(_choice3("choose gamma marker", "train", "c3-gamma", 0))
    train.append(_choice3("pick delta marker", "train", "c3-delta", 1))
    train.append(_choice3("pick epsilon marker", "train", "c3-epsilon", 2))
    train.append(_choice3("pick delta again", "train", "c3-delta2", 1))
    train.append(_abstention("abstain one", "train", "ab-1", 1))
    train.append(_abstention("abstain two", "train", "ab-2", 0))
    train.append(_abstention("abstain three", "train", "ab-3", None))
    train.append(_abstention("abstain four", "train", "ab-4", None))
    train.append(_example("probability", "Is this true?", "proposition", ("Yes", "No"),
                          1, "train", "prob-1"))
    train.append(_example("scoring", "rank the paths", "ranking state",
                          ("Alpha option", "Beta option"),
                          None, "train", "score-1", scores=(0.9, 0.1)))
    assert len(train) == 24

    dev = [
        _choice2("goal alpha dev a", "dev", "d-c-1", 0),
        _choice2("goal beta dev b", "dev", "d-c-2", 1),
        _choice2("goal alpha dev c", "dev", "d-c-3", 0),
        _choice3("pick delta dev", "dev", "d-c3", 1),
        _preference2("prefer alpha dev", "dev", "d-p-1", 0),
        _preference2("prefer beta dev", "dev", "d-p-2", 1),
        _abstention("abstain dev", "dev", "d-ab", 1),
        _example("probability", "Is this true?", "dev proposition", ("Yes", "No"),
                 0, "dev", "d-prob"),
    ]
    assert len(dev) == 8

    test = [
        _choice2("goal alpha test a", "test", "t-c-1", 0),
        _choice2("goal beta test b", "test", "t-c-2", 1),
        _choice2("goal alpha test c", "test", "t-c-3", 0),
        _choice3("pick delta test", "test", "t-c3", 1),
        _preference2("prefer alpha test", "test", "t-p-1", 0),
        _preference2("prefer beta test", "test", "t-p-2", 1),
        _abstention("abstain test", "test", "t-ab", 0),
        _example("risk", "risk goal", "risk state", ("Alpha option", "Beta option"),
                 None, "test", "t-risk", risk_labels=("violence",)),
    ]
    assert len(test) == 8
    return train, dev, test


@pytest.fixture(scope="module")
def rows():
    return fixture_rows()


@pytest.fixture(scope="module")
def run_results(rows):
    train, dev, test = rows
    return run_baselines(train, dev, test, FINGERPRINT)


def _result(run_results, baseline):
    return next(r for r in run_results if r.baseline == baseline)


# --- module wiring -----------------------------------------------------------


def test_fingerprint_must_be_sha256_hex(rows):
    train, dev, test = rows
    with pytest.raises(BaselineInputError):
        MajorityClassBaseline().run(train, dev, test, "not-a-fingerprint")
    with pytest.raises(BaselineInputError):
        MajorityClassBaseline().run(train, dev, test, "A1" * 32)  # uppercase rejected


def test_rows_with_wrong_split_tag_raise(rows):
    train, dev, test = rows
    doctored_dev = list(dev)
    doctored_dev[0] = _example("choice", "goal alpha dev a", "state X",
                               ("Alpha option", "Beta option"), 0, "train", "d-c-1")
    with pytest.raises(BaselineInputError):
        MajorityClassBaseline().run(train, doctored_dev, test, FINGERPRINT)


def test_non_decision_example_rows_raise(rows):
    train, dev, test = rows
    with pytest.raises(BaselineInputError):
        MajorityClassBaseline().run(train, dev, [dict(goal="x")], FINGERPRINT)  # type: ignore[arg-type]


def test_chosen_index_out_of_range_raises(rows):
    train, dev, test = rows
    bad = _example("choice", "bad index", "state", ("Alpha option", "Beta option"),
                   2, "test", "t-bad")
    with pytest.raises(BaselineInputError):
        MajorityClassBaseline().run(train, dev, [bad], FINGERPRINT)


def test_run_baselines_returns_three_results_same_fingerprint(run_results):
    assert [r.baseline for r in run_results] == list(ALL_BASELINES)
    assert {r.split_manifest_fingerprint for r in run_results} == {FINGERPRINT}


def test_result_is_json_serializable(run_results):
    payload = json.dumps([r.to_dict() for r in run_results])
    assert FINGERPRINT in payload


def test_metric_block_schema_matches_metric_result(run_results):
    reference = set(metrics.compute_metrics([]).to_dict())
    for res in run_results:
        assert set(res.dev.to_dict()) == reference
        assert set(res.test.to_dict()) == reference
        for block in res.dev_per_family.values():
            assert set(block.to_dict()) == reference


def test_pooled_metrics_cover_scored_stream_only(rows, run_results):
    train, dev, test = rows
    for res in run_results:
        assert res.n_train == len(train)
        assert res.n_dev == len(dev)
        assert res.n_test == len(test)
        # 6 scored dev rows (4 choice + 2 preference); abstention/probability excluded
        assert res.dev.n_items == 6
        assert res.test.n_items == 6
        assert set(res.dev_per_family) == {"choice", "preference"}
        assert set(res.test_per_family) == {"choice", "preference"}


def test_skipped_counts_are_reported(run_results):
    res = _result(run_results, B1_MAJORITY)
    assert "abstention" not in res.skipped["train"]  # abstention handled separately
    assert "abstention" not in res.skipped["dev"]
    assert "abstention" not in res.skipped["test"]
    assert res.skipped["train"]["probability"] == 1
    assert res.skipped["train"]["scoring"] == 1
    assert res.skipped["dev"]["probability"] == 1
    assert res.skipped["test"]["risk"] == 1


def test_choice_row_without_chosen_index_is_skipped():
    no_gold = _example("choice", "no gold row", "state", ("Alpha option", "Beta option"),
                       None, "train", "c-nogold")
    train = [_choice2("goal alpha case", "train", "ng-0", 0), no_gold]
    dev = [_choice2("goal beta dev q", "dev", "ng-d", 1)]
    res = MajorityClassBaseline().run(train, dev, [], FINGERPRINT)
    assert res.skipped["train"]["no-chosen-index"] == 1
    assert res.n_train == 2
    assert res.dev.n_items == 1


# --- B1 majority class --------------------------------------------------------


def test_b1_predicts_majority_class():
    train = [_choice2(f"goal alpha case {i}", "train", f"m-{i}", 1) for i in range(5)]
    train += [_choice2("goal beta case x", "train", "m-x", 0)]
    dev = [_choice2("goal alpha dev q", "dev", "m-d", 0)]
    res = MajorityClassBaseline().run(train, dev, [], FINGERPRINT)
    assert res.dev.accuracy == pytest.approx(0.0)  # deterministic majority → always 1
    assert res.dev_per_family["choice"].accuracy == pytest.approx(0.0)


def test_b1_tie_breaks_to_lowest_index():
    train = [_choice2("goal alpha case 0", "train", "t0", 0),
             _choice2("goal beta case 0", "train", "t1", 1)]
    dev = [_choice2("goal beta dev q", "dev", "t-d", 1)]
    res = MajorityClassBaseline().run(train, dev, [], FINGERPRINT)
    assert res.dev.accuracy == pytest.approx(0.0)  # tie → index 0 predicted


def test_b1_probabilities_are_empirical_frequencies():
    runner = MajorityClassBaseline()
    train = [_choice2(f"goal alpha case {i}", "train", f"f-{i}", 0) for i in range(3)]
    train += [_choice2("goal beta case f", "train", "f-b", 1)]
    dev = [_choice2("goal alpha dev q", "dev", "f-d", 0)]
    res = runner.run(train, dev, [], FINGERPRINT)
    # empirical 0.75/0.25 → Brier/accuracy consistent with p=(0.75, 0.25)
    probs = 0.75
    assert res.dev.brier == pytest.approx((probs - 1) ** 2 * probs + (1 - probs) ** 2 * (1 - probs))


def test_b1_bucket_fallback_to_family_pool():
    # train has only 3-option rows (majority 0); dev has a 2-option row (unseen bucket)
    train = [_choice3("choose gamma marker", "train", "fb-0", 0),
             _choice3("choose gamma marker again", "train", "fb-1", 0),
             _choice3("pick delta marker", "train", "fb-2", 1)]
    dev = [_choice2("goal alpha dev q", "dev", "fb-d", 0)]
    res = MajorityClassBaseline().run(train, dev, [], FINGERPRINT)
    # family-pooled distribution [0.667, 0.333, 0] → truncated to 2 → renormalized
    assert res.dev.accuracy == pytest.approx(1.0)


def test_b1_empty_train_falls_back_uniform():
    dev = [_choice2("goal alpha dev q", "dev", "e-d", 0)]
    res = MajorityClassBaseline().run([], dev, [], FINGERPRINT)
    assert res.dev.n_items == 1
    assert any("no labeled" in n for n in res.notes)


# --- B3 TF-IDF logistic regression --------------------------------------------


def test_b3_learns_separable_choice_rows(rows, run_results):
    train, dev, test = rows
    res = _result(run_results, B3_TFIDF)
    assert res.dev.accuracy == pytest.approx(1.0)
    assert res.test.accuracy == pytest.approx(1.0)


def test_b3_deterministic_across_runs(rows):
    train, dev, test = rows
    first = TfidfLogisticBaseline().run(train, dev, test, FINGERPRINT)
    second = TfidfLogisticBaseline().run(train, dev, test, FINGERPRINT)
    assert first.dev.to_dict() == second.dev.to_dict()
    assert first.test.to_dict() == second.test.to_dict()


def test_b3_convergence_recorded_not_asserted(run_results):
    res = _result(run_results, B3_TFIDF)
    converged = [n for n in res.notes if "converged=" in n]
    assert converged, "B3 must record per-bucket convergence in the notes"
    for line in converged:
        assert "n_iters=" in line
        assert "final_objective=" in line


def test_b3_objective_decreases_over_training(rows):
    train, _, _ = rows
    runner = TfidfLogisticBaseline()
    scored = [r for r in train if r.task_family in ("choice", "preference")
              and r.chosen_index is not None]
    runner.fit(scored)
    for model in runner.models.values():
        assert model.history[-1] < model.history[0]
        assert len(model.history) - 1 <= 200


def test_b3_unseen_option_count_bucket_falls_back(rows):
    train, dev, test = rows
    two_option_train = [r for r in train if len(r.options) == 2 and r.task_family in
                        ("choice", "preference") and r.chosen_index is not None]
    res = TfidfLogisticBaseline().run(two_option_train, dev, [], FINGERPRINT)
    assert res.dev.n_items == 6
    assert any("unseen in train" in n for n in res.notes)
    for probs in (r for r in res.dev_per_family.values()):
        assert probs.n_items > 0


def test_b3_empty_train_emits_uniform_predictions(rows):
    _, dev, _ = rows
    res = TfidfLogisticBaseline().run([], dev, [], FINGERPRINT)
    assert res.dev.n_items == 6
    assert any("no labeled" in n for n in res.notes)


def test_b3_metrics_match_direct_compute_metrics(run_results):
    # pooled dev block must equal compute_metrics over the same scored stream
    res = _result(run_results, B3_TFIDF)
    records = res.dev.n_items
    assert records == 6
    assert 0.0 <= res.dev.macro_f1 <= 1.0
    assert res.dev.brier is not None
    assert res.dev.ece_pooled is not None


# --- B4 frozen-embedding cosine -------------------------------------------------


def test_b4_hash_fallback_is_labeled_and_runs(rows):
    train, dev, test = rows
    res = FrozenEmbeddingCosineBaseline(embedding_source="hash").run(
        train, dev, test, FINGERPRINT)
    assert res.embedding_source == EMBEDDING_SOURCE_HASH
    assert res.dev.n_items == 6
    assert res.dev.accuracy == pytest.approx(1.0)


def test_b4_auto_falls_back_when_bge_unavailable(rows, monkeypatch):
    train, dev, test = rows

    def boom(self):
        raise OSError("no local model files")

    monkeypatch.setattr(FrozenEmbeddingCosineBaseline, "_load_bge", boom)
    res = FrozenEmbeddingCosineBaseline(embedding_source="auto").run(
        train, dev, test, FINGERPRINT)
    assert res.embedding_source == EMBEDDING_SOURCE_HASH
    assert any("falling back to hash-ngram" in n for n in res.notes)


def test_b4_explicit_bge_request_raises_when_unavailable(rows, monkeypatch):
    train, dev, test = rows

    def boom(self):
        raise OSError("no local model files")

    monkeypatch.setattr(FrozenEmbeddingCosineBaseline, "_load_bge", boom)
    with pytest.raises(BaselineInputError):
        FrozenEmbeddingCosineBaseline(embedding_source="bge").run(
            train, dev, test, FINGERPRINT)


def test_b4_invalid_source_rejected():
    with pytest.raises(BaselineInputError):
        FrozenEmbeddingCosineBaseline(embedding_source="glmp")


def test_b4_deterministic_across_runs(rows):
    train, dev, test = rows
    first = FrozenEmbeddingCosineBaseline(embedding_source="hash").run(
        train, dev, test, FINGERPRINT)
    second = FrozenEmbeddingCosineBaseline(embedding_source="hash").run(
        train, dev, test, FINGERPRINT)
    assert first.dev.to_dict() == second.dev.to_dict()


def test_b4_probabilities_sum_to_one(rows):
    train, dev, _ = rows
    runner = FrozenEmbeddingCosineBaseline(embedding_source="hash")
    scored = [r for r in dev if r.task_family in ("choice", "preference")]
    runner.fit([])
    distributions = runner.predict_distribution(scored)
    assert all(abs(sum(p) - 1.0) < 1e-6 for p in distributions)


# --- abstention stream ----------------------------------------------------------


def test_abstention_scored_separately_with_random_default(rows):
    _, dev, test = rows
    res = MajorityClassBaseline().run(*rows, FINGERPRINT)
    assert res.dev_abstention is not None
    assert res.dev_abstention.n_rows == 1
    assert res.dev_abstention.n_scored == 1
    assert res.dev_abstention.expected_random_accuracy == pytest.approx(0.5)
    # seeded draws must match an independent recomputation
    expected_draw = random.Random(ABSTENTION_RANDOM_SEED).randrange(2)
    assert res.dev_abstention.accuracy == pytest.approx(1.0 if expected_draw == 1 else 0.0)
    assert res.test_abstention is not None
    assert res.test_abstention.n_rows == 1
    # abstention rows are never mixed into the pooled metrics
    assert res.dev.n_items == 6


def test_abstention_without_chosen_index(rows):
    train, dev, _ = rows
    no_gold = _example("abstention", "unanswerable", "no gold", ("Answerable", "Unanswerable"),
                       None, "dev", "no-gold")
    res = MajorityClassBaseline().run(train, [no_gold], [], FINGERPRINT)
    assert res.dev_abstention is not None
    assert res.dev_abstention.n_rows == 1
    assert res.dev_abstention.n_scored == 0
    assert res.dev_abstention.accuracy is None
    assert res.dev_abstention.expected_random_accuracy is None
    assert res.skipped["dev"] == {}


# --- y1_verdict hard gate -------------------------------------------------------


def test_y1_verdict_pass_at_exactly_eight_points():
    assert y1_verdict(0.63, 0.55) == "pass"  # exactly +8.0 points


def test_y1_verdict_fail_just_below_margin():
    assert y1_verdict(0.629, 0.55) == "fail"


def test_y1_verdict_fail_when_below_baseline():
    assert y1_verdict(0.50, 0.55) == "fail"


def test_y1_verdict_fail_closed_on_missing_accuracy():
    assert y1_verdict(None, 0.55) == "fail"
    assert y1_verdict(0.70, None) == "fail"
    assert y1_verdict(None, None) == "fail"


def test_y1_verdict_custom_margin():
    assert y1_verdict(0.60, 0.55, margin=5.0) == "pass"
    assert y1_verdict(0.60, 0.55, margin=6.0) == "fail"


def test_y1_verdict_out_of_domain_raises():
    with pytest.raises(BaselineInputError):
        y1_verdict(1.5, 0.55)
    with pytest.raises(BaselineInputError):
        y1_verdict(0.6, -0.1)
    with pytest.raises(BaselineInputError):
        y1_verdict(float("nan"), 0.55)
    with pytest.raises(BaselineInputError):
        y1_verdict(0.6, float("inf"))
    with pytest.raises(BaselineInputError):
        y1_verdict(0.6, 0.55, margin=-1.0)


# --- manifest helper --------------------------------------------------------------


def _tiny_manifest() -> SplitManifest:
    return SplitManifest(
        fingerprint=FINGERPRINT,
        counts={"choice": {"train": 2, "dev": 1, "test": 1}},
        group_overlap=0,
        dedup_threshold=0.9,
        ood_source="UltraFeedback-binarized",
        withheld_source="UltraFeedback-binarized",
        seed=13,
        totals={"train": 2, "dev": 1, "test": 1},
        ood_rows=0,
        ratios={"train": 0.7, "dev": 0.15, "test": 0.15},
    )


def test_load_manifest_fingerprint_roundtrip(tmp_path):
    path = write_manifest(_tiny_manifest(), tmp_path / "splits.json")
    assert load_manifest_fingerprint(path) == FINGERPRINT


def test_load_manifest_fingerprint_rejects_garbage(tmp_path):
    path = tmp_path / "garbage.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest_fingerprint(path)


def test_baselines_echo_the_manifest_fingerprint(tmp_path, rows):
    """Acceptance: baselines run from the same split manifest as the model."""
    train, dev, test = rows
    path = write_manifest(_tiny_manifest(), tmp_path / "splits.json")
    fingerprint = load_manifest_fingerprint(path)
    for res in run_baselines(train, dev, test, fingerprint):
        assert res.split_manifest_fingerprint == fingerprint


def test_all_baselines_emit_well_formed_probabilities(run_results):
    for res in run_results:
        assert isinstance(res, BaselineMetrics)
        for block in (res.dev, res.test, *res.dev_per_family.values(),
                      *res.test_per_family.values()):
            assert isinstance(block, metrics.MetricResult)
            assert block.accuracy is None or 0.0 <= block.accuracy <= 1.0
            assert block.macro_f1 is None or 0.0 <= block.macro_f1 <= 1.0
