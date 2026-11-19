"""Quality rules Q1–Q6: named counters, structural only, report invariant."""

from __future__ import annotations

import pytest

from src.data.clean import RULE_ORDER, QualityFilter, RejectionReport
from src.data.clean.errors import ReportMismatchError
from src.data.clean.quality import (
    q1_short_option,
    q2_markup_residue,
    q3_duplicate_option,
    q4_chosen_index_out_of_range,
    q5_too_long,
    q6_punctuation_only_or_placeholder,
)

_LONG_OPTION = (
    "A sufficiently long option text that carries well over three whitespace tokens"
)


def _valid(example_factory, **overrides):
    defaults = {
        "goal": "Pick the safest plan for the deployment rollout",
        "state": "The service holds customer data and the rollout window is four hours",
        "options": (
            "Roll out to ten percent of traffic first",
            "Roll out to all traffic immediately",
            "Delay the rollout by two full weeks",
        ),
    }
    defaults.update(overrides)
    return example_factory(**defaults)


# --- One test per rule, each proving its named counter increments -----------


def test_q1_rejects_option_with_fewer_than_three_tokens(example_factory):
    example = _valid(example_factory, options=("Yes", _LONG_OPTION, _LONG_OPTION + " again"))
    assert q1_short_option(example)
    result = QualityFilter().run([example])
    assert result.report.per_rule["Q1"] == 1
    assert result.kept == ()


def test_q2_rejects_entity_residue(example_factory):
    # Normalization decodes &amp; once; residue that survives (e.g. from
    # double-encoded entities) is caught on the raw text by Q2.
    example = _valid(example_factory)
    object.__setattr__(
        example, "state", "The rollout window &amp; the error budget both matter here"
    )
    assert q2_markup_residue(example)
    result = QualityFilter().run([example])
    assert result.report.per_rule["Q2"] == 1


def test_q2_rejects_tag_residue(example_factory):
    example = _valid(example_factory)
    object.__setattr__(example, "state", "The rollout<br>window matters here a lot")
    assert q2_markup_residue(example)


def test_q2_rejects_markdown_bold_residue(example_factory):
    example = _valid(example_factory, goal="Pick the **safest** plan for rollout now")
    assert q2_markup_residue(example)


def test_q3_counts_duplicate_options_even_if_constructed_directly(example_factory):
    example = _valid(example_factory)
    object.__setattr__(
        example, "options", example.options + (example.options[0],)
    )
    assert q3_duplicate_option(example)
    result = QualityFilter().run([example])
    # Q1 fires first? No — all options are 3+ tokens, so Q1 passes and Q3 claims it.
    assert result.report.per_rule["Q3"] == 1


def test_q4_rejects_out_of_range_chosen_index(example_factory):
    example = _valid(example_factory)
    object.__setattr__(example, "chosen_index", 99)
    assert q4_chosen_index_out_of_range(example)
    result = QualityFilter().run([example])
    assert result.report.per_rule["Q4"] == 1


def test_q4_allows_missing_chosen_index(example_factory):
    example = _valid(example_factory, family="scoring", chosen_index=None)
    assert not q4_chosen_index_out_of_range(example)


def test_q5_rejects_rows_over_the_token_budget(example_factory):
    long_state = " ".join(["word"] * 600)
    example = _valid(example_factory, state=long_state)
    assert q5_too_long(example)
    result = QualityFilter().run([example])
    assert result.report.per_rule["Q5"] == 1


def test_q6_rejects_punctuation_only_option(example_factory):
    # "! ! !" is three whitespace tokens (Q1 passes) but no alphanumerics (Q6).
    example = _valid(example_factory, options=("! ! !", _LONG_OPTION, _LONG_OPTION + " again"))
    assert q6_punctuation_only_or_placeholder(example)
    result = QualityFilter().run([example])
    assert result.report.per_rule["Q6"] == 1


@pytest.mark.parametrize("placeholder", ["N/A", "n/a", "TBD", "unknown", "xxx"])
def test_q6_rejects_placeholder_options(example_factory, placeholder):
    example = _valid(
        example_factory, options=(placeholder, _LONG_OPTION, _LONG_OPTION + " again")
    )
    assert q6_punctuation_only_or_placeholder(example)


@pytest.mark.parametrize(
    "real",
    ["none of the above", "None of the above", "all of the above", "none of these"],
)
def test_q6_never_rejects_none_of_the_above(example_factory, real):
    example = _valid(example_factory, options=(real, _LONG_OPTION, _LONG_OPTION + " again"))
    assert not q6_punctuation_only_or_placeholder(example)


# --- Report invariant: counters sum to removed rows --------------------------


def test_rejection_report_sums_to_removed_rows(example_factory):
    rows = [
        _valid(example_factory, group_id="g-ok-1"),
        _valid(example_factory, group_id="g-ok-2"),
        _valid(
            example_factory,
            options=("Yes", _LONG_OPTION, _LONG_OPTION + " x"),
            group_id="g-q1",
        ),
        _valid(
            example_factory,
            state="The rollout **window** matters for the error budget",
            group_id="g-q2",
        ),
        _valid(example_factory, state=" ".join(["word"] * 600), group_id="g-q5"),
        _valid(
            example_factory,
            options=("! ! !", _LONG_OPTION, _LONG_OPTION + " x"),
            group_id="g-q6",
        ),
    ]
    total_in = len(rows)

    result = QualityFilter().run(rows)
    removed = total_in - len(result.kept)

    assert result.report.total_rejected == removed
    assert sum(result.report.per_rule.values()) == removed
    assert result.report.per_rule["Q1"] == 1
    assert result.report.per_rule["Q2"] == 1
    assert result.report.per_rule["Q5"] == 1
    assert result.report.per_rule["Q6"] == 1
    result.report.verify(removed)  # must not raise


def test_verify_raises_on_mismatch():
    report = RejectionReport()
    with pytest.raises(ReportMismatchError):
        report.verify(1)


def test_increment_rejects_unknown_rule():
    report = RejectionReport()
    with pytest.raises(ReportMismatchError):
        report.increment("Q7", "id-1")


def test_first_failing_rule_claims_the_row(example_factory):
    # An option that is both short (Q1) and punctuation-only (Q6): Q1 claims it.
    example = _valid(example_factory, options=("!", _LONG_OPTION, _LONG_OPTION + " x"))
    result = QualityFilter().run([example])
    assert result.report.per_rule["Q1"] == 1
    assert result.report.per_rule["Q6"] == 0


def test_rule_order_covers_exactly_q1_to_q6():
    assert RULE_ORDER == ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
    assert set(RULE_ORDER) == {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}


# --- No rule deletes a hard example for being hard ---------------------------


def test_hard_but_valid_example_survives(example_factory):
    # Unusual content, valid structure: decimal-heavy, negations, long clauses.
    example = _valid(
        example_factory,
        goal="Estimate the probability the exchange settles before midnight UTC",
        state=(
            "The quoted spread is 0.0031 versus 0.0047 yesterday, the book shows "
            "seventeen resting orders, three of them icebergs, and latency drifted "
            "from 40 milliseconds to 110 milliseconds after the last config push"
        ),
        options=(
            "It settles before midnight because the drift is within the normal noise band",
            "It does not settle before midnight because the iceberg orders distort the book",
            "The outcome cannot be decided from the information given in the state fields",
        ),
    )
    result = QualityFilter().run([example])
    assert result.kept == (example,)
    assert result.report.total_rejected == 0


def test_quality_filter_never_reads_scores_or_difficulty(example_factory):
    # A scoring example with extreme (0.0 / 1.0) but in-range utilities is kept:
    # no rule inspects score values, margin size, or how "hard" a row is.
    example = example_factory(
        family="scoring",
        options=(
            "This option is plainly the best of the three choices presented here",
            "This option is plainly the worst of the three choices presented here",
            "This option is mediocre yet entirely acceptable for the stated use case",
        ),
        chosen_index=None,
        scores=(1.0, 0.0, 0.5),
    )
    result = QualityFilter().run([example])
    assert result.kept == (example,)


# --- Details trail ------------------------------------------------------------

def test_details_lists_rejected_ids_per_rule(example_factory):
    example = _valid(example_factory, options=("Yes", _LONG_OPTION, _LONG_OPTION + " x"))
    result = QualityFilter().run([example])
    assert result.report.details["Q1"] == [example.example_id]
