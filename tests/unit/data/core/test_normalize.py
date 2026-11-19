"""Tests for src/data/core/normalize.py — the normalization pipeline and validation.

Covers E3 (duplicated option after normalization rejected at construction with
a distinct error code) and every rejection code the contract names.
"""

from __future__ import annotations

import pytest

from src.data.core import (
    collapse_whitespace,
    normalize_example,
    normalize_text,
    strip_html,
    validate_example,
)
from src.data.core.errors import (
    BadSplitError,
    BadTaskFamilyError,
    BadTypeError,
    ChosenIndexMissingError,
    ChosenIndexOutOfRangeError,
    DataError,
    DuplicateOptionError,
    EmptyGoalError,
    EmptyOptionError,
    EmptyStateError,
    IdMismatchError,
    MissingFieldError,
    RiskLabelInvalidError,
    ScoreOutOfRangeError,
    ScoresLengthMismatchError,
    TooFewOptionsError,
    TooManyOptionsError,
    UnknownFieldError,
)


def _raw(**overrides):
    record = {
        "task_family": "choice",
        "goal": "Pick a plan",
        "state": "Budget is 10k",
        "options": [{"text": "Alpha"}, {"text": "Beta"}],
        "chosen_index": 0,
        "source": "fixture",
        "license": "MIT",
        "group_id": "group-1",
        "split": "train",
    }
    record.update(overrides)
    return record


def _code(excinfo):
    return excinfo.value.code


# --- The four rejections the contract names ---------------------------------


@pytest.mark.parametrize("goal", ["", "   ", "\n\t", "<p></p>", "<b>  </b>"])
def test_empty_goal_is_rejected_with_distinct_code(goal):
    with pytest.raises(EmptyGoalError) as excinfo:
        normalize_example(_raw(goal=goal))
    assert _code(excinfo) == "DATA_EMPTY_GOAL"


@pytest.mark.parametrize("options", [[], [{"text": "only"}]])
def test_fewer_than_two_options_rejected_with_distinct_code(options):
    with pytest.raises(TooFewOptionsError) as excinfo:
        normalize_example(_raw(options=options, chosen_index=None))
    assert _code(excinfo) == "DATA_TOO_FEW_OPTIONS"


def test_duplicate_option_after_normalization_rejected_with_distinct_code():
    with pytest.raises(DuplicateOptionError) as excinfo:
        normalize_example(_raw(options=[{"text": "Yes"}, {"text": "  Yes\n"}]))
    assert _code(excinfo) == "DATA_DUPLICATE_OPTION"


def test_duplicate_option_after_html_and_entity_normalization():
    with pytest.raises(DuplicateOptionError):
        normalize_example(
            _raw(options=[{"text": "Tom &amp; Jerry"}, {"text": "<b>Tom & Jerry</b>"}])
        )


def test_duplicate_option_after_nfc_normalization():
    with pytest.raises(DuplicateOptionError):
        normalize_example(_raw(options=[{"text": "café"}, {"text": "cafe\u0301"}]))


@pytest.mark.parametrize("index", [2, 7, -1, 99])
def test_chosen_index_out_of_range_rejected_with_distinct_code(index):
    with pytest.raises(ChosenIndexOutOfRangeError) as excinfo:
        normalize_example(_raw(chosen_index=index))
    assert _code(excinfo) == "DATA_CHOSEN_INDEX_OUT_OF_RANGE"


# --- Other validation codes -------------------------------------------------


def test_nine_options_rejected():
    options = [{"text": f"opt-{i}"} for i in range(9)]
    with pytest.raises(TooManyOptionsError) as excinfo:
        normalize_example(_raw(options=options, chosen_index=0))
    assert _code(excinfo) == "DATA_TOO_MANY_OPTIONS"


def test_eighth_option_is_allowed():
    options = [{"text": f"opt-{i}"} for i in range(8)]
    example = normalize_example(_raw(options=options, chosen_index=7))
    assert len(example.options) == 8


def test_empty_option_text_rejected():
    with pytest.raises(EmptyOptionError) as excinfo:
        normalize_example(_raw(options=[{"text": "Alpha"}, {"text": "<i></i>"}]))
    assert _code(excinfo) == "DATA_EMPTY_OPTION"


def test_unknown_task_family_rejected():
    with pytest.raises(BadTaskFamilyError) as excinfo:
        normalize_example(_raw(task_family="mcq"))
    assert _code(excinfo) == "DATA_BAD_TASK_FAMILY"


def test_unknown_split_rejected():
    with pytest.raises(BadSplitError) as excinfo:
        normalize_example(_raw(split="validation"))
    assert _code(excinfo) == "DATA_BAD_SPLIT"


def test_missing_required_field_rejected():
    raw = _raw()
    del raw["source"]
    with pytest.raises(MissingFieldError) as excinfo:
        normalize_example(raw)
    assert _code(excinfo) == "DATA_MISSING_FIELD"


def test_unknown_key_rejected():
    with pytest.raises(UnknownFieldError) as excinfo:
        normalize_example(_raw(note="not in the contract"))
    assert _code(excinfo) == "DATA_UNKNOWN_FIELD"


def test_empty_state_rejected_for_non_preference_family():
    with pytest.raises(EmptyStateError) as excinfo:
        normalize_example(_raw(state="   "))
    assert _code(excinfo) == "DATA_EMPTY_STATE"


def test_empty_state_allowed_for_preference_family():
    example = normalize_example(_raw(task_family="preference", state="", chosen_index=0))
    assert example.state == ""


def test_chosen_index_required_for_choice_family():
    raw = _raw()
    del raw["chosen_index"]
    with pytest.raises(ChosenIndexMissingError) as excinfo:
        normalize_example(raw)
    assert _code(excinfo) == "DATA_MISSING_CHOSEN_INDEX"


def test_chosen_index_not_required_for_scoring_family():
    raw = _raw(task_family="scoring")
    del raw["chosen_index"]
    example = normalize_example(raw)
    assert example.chosen_index is None


def test_boolean_chosen_index_rejected():
    with pytest.raises(BadTypeError):
        normalize_example(_raw(chosen_index=True))


def test_scores_length_must_match_options():
    with pytest.raises(ScoresLengthMismatchError) as excinfo:
        normalize_example(_raw(scores=[1.0]))
    assert _code(excinfo) == "DATA_SCORES_LENGTH_MISMATCH"


def test_scores_must_be_within_unit_interval():
    with pytest.raises(ScoreOutOfRangeError) as excinfo:
        normalize_example(_raw(scores=[0.5, 1.5]))
    assert _code(excinfo) == "DATA_SCORE_OUT_OF_RANGE"


def test_boolean_score_rejected():
    with pytest.raises(BadTypeError):
        normalize_example(_raw(scores=[1.0, True]))


def test_nine_risk_labels_rejected():
    with pytest.raises(RiskLabelInvalidError):
        normalize_example(_raw(risk_labels=[f"r{i}" for i in range(9)]))


def test_duplicate_risk_labels_rejected():
    with pytest.raises(RiskLabelInvalidError):
        normalize_example(_raw(risk_labels=["self-harm", "self-harm"]))


def test_tampered_example_id_rejected():
    example = normalize_example(_raw())
    raw = _raw(example_id="0" * 64)
    with pytest.raises(IdMismatchError):
        normalize_example(raw)
    assert example.example_id != "0" * 64


def test_provided_valid_example_id_is_accepted():
    example = normalize_example(_raw())
    again = normalize_example(_raw(example_id=example.example_id))
    assert again.example_id == example.example_id


def test_validate_example_collects_multiple_errors():
    raw = _raw(goal="", options=[{"text": "Alpha"}], split="nope")
    codes = [e.code for e in validate_example(raw)]
    assert "DATA_EMPTY_GOAL" in codes
    assert "DATA_TOO_FEW_OPTIONS" in codes
    assert "DATA_BAD_SPLIT" in codes


def test_validate_example_returns_empty_list_for_valid_record():
    assert validate_example(_raw()) == []


def test_normalize_example_raises_first_error_not_all():
    raw = _raw(goal="", split="nope")
    with pytest.raises(DataError):
        normalize_example(raw)


def test_options_must_be_a_list_not_a_string():
    with pytest.raises(BadTypeError):
        normalize_example(_raw(options="Alpha Beta"))


# --- The normalization pipeline ----------------------------------------------


def test_nfc_applied_to_goal_state_and_options():
    example = normalize_example(
        _raw(
            goal="cafe\u0301",            # NFD é
            state="the\u0301 cafe\u0301",
            options=[{"text": "ye\u0301s"}, {"text": "no"}],
        )
    )
    assert example.goal == "café"
    assert example.state == "thé café"
    assert example.options[0].text == "yés"


def test_html_tags_stripped_and_entities_decoded():
    assert strip_html("<p>Tom &amp; Jerry &lt;3</p>") == "Tom & Jerry <3"
    assert strip_html("a<script>var x = 1;</script>b") == "a b"
    assert strip_html("x<style>.y{color:red}</style>y") == "x y"
    assert collapse_whitespace("  a\n\tb\r\nc   d ") == "a b c d"


def test_markup_rich_inputs_normalize_to_plain_text():
    example = normalize_example(
        _raw(
            goal="<p>Pick <strong>wisely</strong></p>",
            options=[{"text": "<b>Alpha</b> path"}, {"text": "Beta &amp; gamma"}],
        )
    )
    assert example.goal == "Pick wisely"
    assert example.options[0].text == "Alpha path"
    assert example.options[1].text == "Beta & gamma"


def test_nbsp_and_newlines_collapse():
    assert normalize_text("a\xa0\xa0b\n\nc") == "a b c"


def test_nothing_is_invented_for_absent_optionals():
    example = normalize_example(_raw())
    assert example.scores is None
    assert example.answerable is None
    assert example.risk_labels is None
    assert example.source == "fixture"


def test_empty_option_label_is_recorded_as_absent():
    example = normalize_example(
        _raw(options=[{"text": "Alpha", "label": "  <i></i> "}, {"text": "Beta"}])
    )
    assert example.options[0].label is None


def test_option_order_is_preserved_in_the_record():
    example = normalize_example(_raw(options=[{"text": "Beta"}, {"text": "Alpha"}]))
    assert [o.text for o in example.options] == ["Beta", "Alpha"]
