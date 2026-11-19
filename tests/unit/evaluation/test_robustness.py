"""Tests for src/evaluation/robustness.py — FR-EVAL-03 / E7.

Fixtures are synthetic and offline: a tiny ``RapportModel`` (ultra variant)
or deterministic stub models standing in for it, a tiny synthetic corpus and a
locally built ``SplitManifest``. The repo root is placed on ``sys.path`` here
because this suite owns no conftest of its own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.core.schema import (  # noqa: E402
    DecisionExample,
    Option,
    compute_example_id,
)
from src.data.splits.manifest import SplitManifest  # noqa: E402
from src.evaluation.robustness import (  # noqa: E402
    LENGTH_BUDGETS,
    LENGTH_EXTENSION_TOLERANCE,
    SCORE_TOLERANCE,
    RobustnessInputError,
    length_extension_verdict,
    run_robustness,
)
from src.model.core import RapportConfig  # noqa: E402
from src.model.head import RapportModel  # noqa: E402
from src.tokenization import DEFAULT_TOKENIZER  # noqa: E402
from src.tokenization.encoding import TRUNCATION_COUNTS  # noqa: E402
from src.tokenization.tokenizer import OPT_CLOSE_ID, OPT_ID  # noqa: E402

FINGERPRINT = "a1" * 32
TOKENIZER = DEFAULT_TOKENIZER

_EMPTY_STATE = "Alpha option wins the described decision scenario today."


def _bucket_fn(n: int) -> str:
    if n == 2:
        return "2"
    if n <= 5:
        return "3-5"
    return "6-8"


def _row(family, goal, state, options, chosen, split, source, scores=None, gid=None):
    gid = gid or f"gid-{family}-{goal}"
    example_id = compute_example_id(family, goal, state, list(options))
    return DecisionExample(
        example_id=example_id,
        task_family=family,
        goal=goal,
        state=state,
        options=tuple(Option(text=t) for t in options),
        source=source,
        license="MIT",
        group_id=gid,
        split=split,
        chosen_index=chosen,
        scores=scores,
    )


def _manifest() -> SplitManifest:
    return SplitManifest(
        fingerprint=FINGERPRINT,
        counts={
            "choice": {"train": 2, "dev": 0, "test": 6},
            "preference": {"train": 1, "dev": 0, "test": 2},
            "scoring": {"train": 0, "dev": 0, "test": 1},
        },
        group_overlap=0,
        dedup_threshold=0.5,
        ood_source="UltraFeedback-binarized",
        withheld_source="UltraFeedback-binarized",
        seed=20261119,
        totals={"train": 3, "dev": 0, "test": 9},
        ood_rows=2,
    )


def _corpus():
    train = [
        _row("choice", "alpha train goal", f"{_EMPTY_STATE} train split variant alpha.",
             ("Alpha option", "Beta option"), 0, "train", "fixture-train"),
        _row("choice", "beta train goal", f"{_EMPTY_STATE} train split variant beta.",
             ("Gamma option", "Delta option"), 1, "train", "fixture-train"),
        _row("preference", "prefer train goal", f"{_EMPTY_STATE} train split variant pref.",
             ("Alpha option", "Beta option"), 1, "train", "fixture-train"),
    ]
    test = [
        _row("choice", "test two alpha", f"{_EMPTY_STATE} Alpha option is the stronger pick here.",
             ("Alpha option", "Beta option"), 0, "test", "fixture-test"),
        _row("choice", "test two beta", f"{_EMPTY_STATE} Beta option is the stronger pick here.",
             ("Alpha option", "Beta option"), 1, "test", "fixture-test"),
        _row("choice", "test three gamma", f"{_EMPTY_STATE} Gamma path looks best now.",
             ("Gamma path", "Delta path", "Epsilon path"), 0, "test", "fixture-test"),
        _row("choice", "test three delta", f"{_EMPTY_STATE} Delta path looks best now.",
             ("Gamma path", "Delta path", "Epsilon path"), 1, "test", "fixture-test"),
        _row("choice", "test four options", f"{_EMPTY_STATE} The fourth option wins this case.",
             ("Alpha option", "Beta option", "Gamma option", "Delta option"),
             3, "test", "fixture-test"),
        _row("choice", "test five options", f"{_EMPTY_STATE} The fifth option wins this case.",
             ("Alpha option", "Beta option", "Gamma option", "Delta option", "Epsilon option"),
             4, "test", "fixture-test"),
        _row("preference", "prefer test alpha", f"{_EMPTY_STATE} Alpha option is preferred now.",
             ("Alpha option", "Beta option"), 0, "test", "fixture-test"),
        _row("preference", "prefer test beta", f"{_EMPTY_STATE} Beta option is preferred now.",
             ("Alpha option", "Beta option"), 1, "test", "fixture-test"),
        _row("scoring", "rank the paths", f"{_EMPTY_STATE} Ranking scenario for the paths.",
             ("Alpha option", "Beta option"), None, "test", "fixture-test",
             scores=(0.9, 0.1)),
    ]
    corpus = {"choice": train[:2] + test[:6],
              "preference": train[2:] + test[6:8],
              "scoring": test[8:]}
    ood = [
        _row("choice", "ood heldout", f"{_EMPTY_STATE} Heldout option looks best here.",
             ("Alpha option", "Beta option"), 0, "test", "heldout-source"),
        _row("choice", "ood leaked", f"{_EMPTY_STATE} Leaked option looks best here.",
             ("Alpha option", "Beta option"), 1, "test", "fixture-train"),
    ]
    return corpus, ood


def _split_rows_by_family():
    corpus, _ = _corpus()
    return corpus


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(20261119)
    return RapportModel(RapportConfig.ultra()).eval()


def _run(model, corpus=None, ood=None, seed=20261119, split_manifest="a1" * 32):
    if corpus is None or ood is None:
        default_corpus, default_ood = _corpus()
        corpus = default_corpus if corpus is None else corpus
        ood = default_ood if ood is None else ood
    return run_robustness(model, TOKENIZER, corpus, split_manifest, ood, _bucket_fn,
                          seed=seed)


# ---------------------------------------------------------------------------
# stub models (deterministic stand-ins with known score behaviour)
# ---------------------------------------------------------------------------


class _HashModel(torch.nn.Module):
    """Score per slot = stable function of that option block's token ids.

    Perfectly permutation-invariant: re-encoding with permuted option blocks
    leaves every slot's content tokens untouched, so every score is exactly
    reproduced at the permuted position.
    """

    def forward(self, input_ids, attention_mask, option_mask, task_ids):
        import hashlib

        n = option_mask.shape[1]
        ids = input_ids[0].tolist()
        opt_positions = [i for i, t in enumerate(ids) if t == OPT_ID]
        scores = []
        for j in range(n):
            start = opt_positions[j] + 1
            end = ids.index(OPT_CLOSE_ID, start)
            digest = hashlib.sha256(str(ids[start:end]).encode("utf-8")).hexdigest()
            scores.append(int(digest, 16) % 10007 / 10007.0)
        return _StubOutput(scores)


class _PositionalModel(torch.nn.Module):
    """Score per slot only — ignores option content entirely.

    Permuting the options leaves the scores attached to the slots, so the
    permutation deviation is guaranteed positive.
    """

    def forward(self, input_ids, attention_mask, option_mask, task_ids):
        n = option_mask.shape[1]
        scores = [0.9 - 0.1 * j for j in range(n)]
        return _StubOutput(scores)


class _StubOutput:
    def __init__(self, scores):
        tensor = torch.tensor([list(scores)], dtype=torch.float32)
        self.scores = tensor
        self.choice_logits = tensor.clone()
        self.confidence = torch.full((1, 1), 0.6, dtype=torch.float32)
        self.probability = torch.full((1, 1), 0.6, dtype=torch.float32)


# ---------------------------------------------------------------------------
# report shape, determinism, manifest handling
# ---------------------------------------------------------------------------


def test_report_is_json_serializable_grouped_by_family_and_deterministic(model):
    corpus, ood = _corpus()
    first = _run(model, corpus, ood)
    second = _run(model, corpus, ood)
    data = first.to_dict()
    round_tripped = json.loads(json.dumps(data))
    assert round_tripped == data
    assert first.to_dict() == second.to_dict()

    assert data["split_manifest_fingerprint"] == FINGERPRINT
    assert data["ood_source"] is None  # str manifest carries no ood_source
    assert data["rows"] == {"train": 3, "evaluated": 9, "ood": 2}
    for block in ("permutation", "option_count_sweep", "length_sweep"):
        assert "choice" in data[block]
        assert "overall" in data[block]
    for transform in ("contradictory", "missing_information"):
        assert "choice" in data["context"][transform]["by_family"]
        assert "overall" in data["context"][transform]
    assert set(data["context"]) == {"contradictory", "missing_information"}
    assert data["notes"], "notes must document the harness's judgment calls"


def test_manifest_object_echoes_ood_source(model):
    corpus, ood = _corpus()
    report = _run(model, corpus, ood, split_manifest=_manifest())
    assert report.split_manifest_fingerprint == FINGERPRINT
    assert report.ood_source == "UltraFeedback-binarized"
    assert report.ood["ood_source"] == "UltraFeedback-binarized"


def test_train_rows_are_used_only_for_the_source_crosscheck(model):
    corpus, ood = _corpus()
    report = _run(model, corpus, ood)
    assert report.rows["train"] == 3
    assert report.rows["evaluated"] == 9  # train rows never evaluated


def test_truncation_counter_and_model_mode_restored(model):
    before = dict(TRUNCATION_COUNTS)
    training_before = model.training
    corpus, ood = _corpus()
    _run(model, corpus, ood)
    assert dict(TRUNCATION_COUNTS) == before
    assert model.training == training_before


# ---------------------------------------------------------------------------
# permutation block (E7)
# ---------------------------------------------------------------------------


def test_permutation_enumerates_all_n_for_n_le_4_and_samples_otherwise(model):
    corpus, ood = _corpus()
    report = _run(model, corpus, ood)
    by_n = report.permutation["choice"]["by_option_count"]
    assert by_n["2"]["enumeration"] == "full"
    assert by_n["2"]["n_permutations"] == 2
    assert by_n["3"]["enumeration"] == "full"
    assert by_n["3"]["n_permutations"] == 6
    assert by_n["4"]["enumeration"] == "full"
    assert by_n["4"]["n_permutations"] == 24
    assert by_n["5"]["enumeration"] == "sampled"
    assert by_n["5"]["n_permutations"] == 8
    for entry in by_n.values():
        assert len(entry["permutations"]) == entry["n_permutations"]
        assert entry["score_tolerance"] == SCORE_TOLERANCE == 1e-4


def test_permutation_sampled_set_is_deterministic_for_fixed_seed(model):
    corpus, ood = _corpus()
    first = _run(model, corpus, ood, seed=20261119)
    second = _run(model, corpus, ood, seed=20261119)
    assert first.permutation == second.permutation


def test_permutation_reports_accuracy_delta_as_max_minus_min(model):
    corpus, ood = _corpus()
    report = _run(model, corpus, ood)
    for family in ("choice", "preference", "scoring"):
        for entry in report.permutation[family]["by_option_count"].values():
            accuracies = [p["accuracy"] for p in entry["permutations"]]
            defined = [a for a in accuracies if a is not None]
            if not defined:
                assert entry["accuracy_delta"] is None
                continue
            assert entry["accuracy_delta"] == pytest.approx(max(defined) - min(defined))


def test_content_hash_stub_is_exactly_permutation_invariant():
    corpus, ood = _corpus()
    stub_corpus = {"choice": [r for r in corpus["choice"] if len(r.options) == 3]}
    report = _run(_HashModel(), stub_corpus, ood)
    entry = report.permutation["choice"]["by_option_count"]["3"]
    assert entry["score_max_deviation"] == 0.0
    assert entry["within_tolerance"] is True
    assert entry["accuracy_delta"] == 0.0
    assert report.permutation["choice"]["score_max_deviation"] == 0.0
    assert report.permutation["choice"]["within_tolerance"] is True


def test_positional_stub_deviation_exceeds_tolerance():
    corpus, ood = _corpus()
    stub_corpus = {"choice": [r for r in corpus["choice"] if len(r.options) == 3]}
    report = _run(_PositionalModel(), stub_corpus, ood)
    entry = report.permutation["choice"]["by_option_count"]["3"]
    assert entry["score_max_deviation"] > SCORE_TOLERANCE
    assert entry["within_tolerance"] is False
    assert report.permutation["overall"]["within_tolerance"] is False


# ---------------------------------------------------------------------------
# option-count sweep + held-out source (FR-EVAL-03)
# ---------------------------------------------------------------------------


def test_option_count_sweep_reports_accuracy_and_bucket_ece_for_2_to_8(model):
    corpus, ood = _corpus()
    report = _run(model, corpus, ood)
    for family in ("choice", "preference", "scoring", "overall"):
        sweep = report.option_count_sweep[family]
        assert set(sweep) == {str(n) for n in range(2, 9)}
        for n_str, entry in sweep.items():
            assert entry["n_options"] == int(n_str)
            assert entry["bucket"] == _bucket_fn(int(n_str))
            assert entry["ece_bins"] == 15
            if entry["n_scored_rows"] == 0:
                assert entry["accuracy"] is None
                assert entry["bucket_ece"] is None
            else:
                assert entry["bucket_ece"] is not None
    choice = report.option_count_sweep["choice"]
    assert choice["2"]["n_rows"] == 2
    assert choice["3"]["n_rows"] == 2
    assert choice["4"]["n_rows"] == 1
    assert choice["5"]["n_rows"] == 1
    assert choice["6"]["n_rows"] == 0
    assert choice["5"]["bucket"] == "3-5"
    assert choice["2"]["bucket"] == "2"
    assert choice["4"]["bucket"] == "3-5"


def test_held_out_source_block_excludes_sources_with_training_rows(model):
    corpus, ood = _corpus()
    report = _run(model, corpus, ood)
    block = report.ood
    assert block["empty"] is False
    assert block["n_rows"] == 1
    assert block["n_excluded"] == 1
    assert block["excluded_sources"] == ["fixture-train"]
    assert block["sources"] == ["heldout-source"]
    assert block["by_family"]["choice"]["n_rows"] == 1
    assert block["split_manifest_fingerprint"] == FINGERPRINT
    json.dumps(block)


def test_held_out_source_block_present_but_empty_for_empty_input(model):
    corpus, _ = _corpus()
    report = _run(model, corpus, [])
    block = report.ood
    assert block["empty"] is True
    assert block["n_rows"] == 0
    assert block["by_family"] == {}
    assert block["accuracy"] is None
    assert block["bucket_ece"] is None
    assert block["abstention_rate"] is None
    assert block["note"]
    json.dumps(block)


def test_held_out_source_block_present_but_empty_when_all_sources_seen_in_train(model):
    corpus, ood = _corpus()
    leaked_only = [ood[1]]  # source "fixture-train" appears in the train rows
    report = _run(model, corpus, leaked_only)
    block = report.ood
    assert block["empty"] is True
    assert block["n_rows"] == 0
    assert block["n_excluded"] == 1
    assert block["excluded_sources"] == ["fixture-train"]


# ---------------------------------------------------------------------------
# contradictory / missing-information probes (FR-EVAL-03)
# ---------------------------------------------------------------------------


def _assert_delta_consistency(entry):
    clean = entry["abstention_rate_clean"]
    transformed = entry["abstention_rate_transformed"]
    delta = entry["abstention_rate_delta"]
    if transformed is None:
        # transform skipped every row in this group (e.g. no gold option)
        assert delta is None
        return
    assert clean is not None and delta is not None
    assert delta == pytest.approx(transformed - clean)
    if clean == 0.0:
        assert entry["abstention_rate_relative_change"] is None
    else:
        assert entry["abstention_rate_relative_change"] == pytest.approx(delta / clean)
    if entry["accuracy_clean"] is not None and entry["accuracy_transformed"] is not None:
        expected = entry["accuracy_transformed"] - entry["accuracy_clean"]
        assert entry["accuracy_delta"] == pytest.approx(expected)


def test_contradictory_and_missing_information_report_abstention_deltas(model):
    corpus, ood = _corpus()
    report = _run(model, corpus, ood)
    contradictory = report.context["contradictory"]
    missing = report.context["missing_information"]
    assert contradictory["transform"] == "swap_gold_with_plausible_distractor"
    assert missing["transform"] == "drop_decisive_sentence"
    assert contradictory["abstention_threshold"] == 0.75
    for block in (contradictory, missing):
        for family in ("choice", "preference", "scoring", "overall"):
            entry = block["by_family"].get(family) or block["overall"]
            _assert_delta_consistency(entry)
    # the scoring row has no gold option: it is skipped by the contradiction
    # probe but still measured for abstention/accuracy on the missing-info probe
    assert contradictory["overall"]["n_skipped_no_gold"] == 1
    assert missing["overall"]["n_skipped_no_gold"] == 0
    assert missing["overall"]["n_transformed"] == report.rows["evaluated"]


def test_missing_information_reports_emptied_states():
    row = _row(
        "preference",
        "empted state probe",
        "Decisive markerword decides the outcome here.",
        ("markerword option", "plain option"),
        0,
        "test",
        "fixture-test",
    )
    corpus = {"preference": [row]}
    report = _run(_PositionalModel(), corpus, [])
    entry = report.context["missing_information"]["by_family"]["preference"]
    assert entry["n_state_emptied"] == 1


def test_contradiction_swaps_gold_with_highest_utility_distractor(model):
    corpus = {
        "choice": [
            _row("choice", "swap probe", f"{_EMPTY_STATE} Swapprobe details.",
                 ("gold choice text", "distractor text one", "distractor text two"),
                 0, "test", "fixture-test", scores=(0.5, 0.9, 0.1)),
        ]
    }
    report = _run(_PositionalModel(), corpus, [])
    entry = report.context["contradictory"]["by_family"]["choice"]
    # highest-utility non-gold option is index 1; transformed example keeps the
    # content multiset and chosen_index is not remapped
    assert entry["n_transformed"] == 1
    assert entry["n_scored_rows"] == 1


# ---------------------------------------------------------------------------
# length sweep (FR-EVAL-03)
# ---------------------------------------------------------------------------


def test_length_extension_verdict_rule():
    assert length_extension_verdict(0.90, 0.88) == "PASS"  # exactly 2 points is not > 2
    assert length_extension_verdict(0.90, 0.879) == "FAIL"  # exceeds 2 points
    assert length_extension_verdict(None, 0.5) == "not-measured"
    assert length_extension_verdict(0.5, None) == "not-measured"
    assert LENGTH_EXTENSION_TOLERANCE == 0.02


def test_length_sweep_reports_budgets_choice_accuracy_and_bucket_ece(model):
    corpus, ood = _corpus()
    report = _run(model, corpus, ood)
    for family in ("choice", "preference", "scoring", "overall"):
        sweep = report.length_sweep[family]
        assert sweep["length_extension"] in {"PASS", "FAIL", "not-measured"}
        for budget in LENGTH_BUDGETS:
            entry = sweep[str(budget)]
            assert entry["budget"] == budget
            assert entry["n_rows"] >= 0
            assert set(entry) >= {
                "choice_accuracy", "bucket_ece", "ece_bins", "n_truncated",
                "n_skipped_no_gold",
            }
            assert entry["ece_bins"] == 15
        if sweep["512"]["choice_accuracy"] is not None:
            assert sweep["drop_512_to_1024"] == pytest.approx(
                sweep["512"]["choice_accuracy"] - sweep["1024"]["choice_accuracy"]
            )
    assert LENGTH_BUDGETS == (512, 768, 1024)


# ---------------------------------------------------------------------------
# rows without chosen_index / zero-row families
# ---------------------------------------------------------------------------


def test_rows_without_chosen_index_scored_but_not_counted_in_accuracy(model):
    corpus, ood = _corpus()
    report = _run(model, corpus, ood)
    scoring = report.option_count_sweep["scoring"]["2"]
    assert scoring["n_rows"] == 1
    assert scoring["n_scored_rows"] == 0
    assert scoring["accuracy"] is None
    assert scoring["bucket_ece"] is None
    scoring_perm = report.permutation["scoring"]["by_option_count"]["2"]
    assert scoring_perm["n_scored_rows"] == 0
    assert scoring_perm["accuracy_delta"] is None
    assert scoring_perm["score_max_deviation"] is not None  # scores still measured
    assert report.length_sweep["scoring"]["512"]["n_skipped_no_gold"] == 1


def test_family_with_zero_evaluation_rows_is_reported_not_fabricated(model):
    train_only = [
        _row("probability", "prob train only", f"{_EMPTY_STATE} Probability train row.",
             ("Yes", "No"), 1, "train", "fixture-train"),
    ]
    corpus = {"probability": train_only}
    report = _run(model, corpus, [])
    permutation = report.permutation["probability"]
    assert permutation["n_rows"] == 0
    assert permutation["accuracy_delta"] is None
    assert permutation["score_max_deviation"] is None
    assert permutation["within_tolerance"] is None
    sweep = report.option_count_sweep["probability"]
    assert all(entry["n_rows"] == 0 for entry in sweep.values())
    assert all(entry["accuracy"] is None for entry in sweep.values())
    assert report.length_sweep["probability"]["length_extension"] == "not-measured"
    assert report.context["contradictory"]["by_family"]["probability"]["n_rows"] == 0


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_invalid_seed_rejected(model):
    corpus, ood = _corpus()
    with pytest.raises(RobustnessInputError) as excinfo:
        _run(model, corpus, ood, seed=True)
    assert excinfo.value.code == "ROBUSTNESS_INPUT_INVALID"
    with pytest.raises(RobustnessInputError):
        _run(model, corpus, ood, seed=1 << 64)


def test_invalid_split_manifest_rejected(model):
    corpus, ood = _corpus()
    with pytest.raises(RobustnessInputError):
        _run(model, corpus, ood, split_manifest="zz" * 32)
    with pytest.raises(RobustnessInputError):
        _run(model, corpus, ood, split_manifest="A1" * 32)
    with pytest.raises(RobustnessInputError):
        _run(model, corpus, ood, split_manifest=123)


def test_invalid_bucket_fn_rejected(model):
    corpus, ood = _corpus()
    with pytest.raises(RobustnessInputError):
        run_robustness(model, TOKENIZER, corpus, "a1" * 32, ood, lambda n: 3)
    with pytest.raises(RobustnessInputError):
        run_robustness(
            model, TOKENIZER, corpus, "a1" * 32, ood,
            lambda n: 1 / 0,  # noqa: E731 - deliberately broken callable
        )
    with pytest.raises(RobustnessInputError):
        run_robustness(model, TOKENIZER, corpus, "a1" * 32, ood, "not-callable")


def test_unknown_family_and_bad_option_counts_rejected(model):
    row = _row("todos", "bad family", "state text here", ("Alpha option", "Beta option"),
               0, "test", "fixture-test")
    with pytest.raises(RobustnessInputError):
        run_robustness(model, TOKENIZER, {"todos": [row]}, "a1" * 32, [], _bucket_fn)
    nine_options = _row("choice", "nine options", "state text here",
                        tuple(f"option {i}" for i in range(9)), 0, "test", "fixture-test")
    with pytest.raises(RobustnessInputError):
        run_robustness(model, TOKENIZER, {"choice": [nine_options]}, "a1" * 32, [],
                       _bucket_fn)
    out_of_range = _row("choice", "chosen out of range", "state text here",
                        ("Alpha option", "Beta option"), 2, "test", "fixture-test")
    with pytest.raises(RobustnessInputError):
        run_robustness(model, TOKENIZER, {"choice": [out_of_range]}, "a1" * 32, [],
                       _bucket_fn)
