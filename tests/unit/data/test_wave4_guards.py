"""Wave-4 regression tests for the repaired data-integrity guards.

Covers: train-only mixture eligibility, fail-closed source admission,
preference de-biasing augmentation surviving cleaning, the Q1 vocabulary
boundary, and the empty-corpus 13-gram guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.clean import clean_examples  # noqa: E402
from src.data.clean.quality import q1_short_option  # noqa: E402
from src.data.core.normalize import normalize_example  # noqa: E402
from src.data.mixture import (  # noqa: E402
    MixtureAvailabilityError,
    MixtureQuarantinedSourceError,
    sample_mixture,
)
from src.data.splits import split  # noqa: E402
from src.data.splits.errors import SplitInputError  # noqa: E402

FP = "ab" * 32


def _example(family: str, source: str, index: int, split_name: str = "train"):
    options = [
        {"text": f"{family} option {index} alpha beta"},
        {"text": f"{family} option {index} gamma delta"},
    ]
    return normalize_example(
        {
            "task_family": family,
            "goal": f"choose for case {index}",
            "state": f"context {index} unique state text",
            "options": options,
            "source": source,
            "license": "MIT",
            "group_id": f"group-{family}-{index}",
            "split": split_name,
            "chosen_index": 0,
        }
    )


def test_mixture_binds_only_train_rows() -> None:
    rows = [
        _example("choice", "fixture/choice", 0, "train"),
        _example("choice", "fixture/choice", 1, "dev"),
        _example("choice", "fixture/choice", 2, "test"),
    ]
    order = sample_mixture(
        {"choice": {"source": "fixture/choice", "rows": 1}},
        None,
        13,
        FP,
        available_rows=rows,
        source_status={"fixture/choice": "admitted"},
    )
    assert len(order.slots) == 1
    bound = order.slots[0].example_id
    assert bound == rows[0].example_id  # never the dev/test row


def test_mixture_requires_explicit_source_admission() -> None:
    rows = [_example("choice", "fixture/choice", 0)]
    with pytest.raises(MixtureQuarantinedSourceError):
        sample_mixture(
            {"choice": {"source": "fixture/choice", "rows": 1}},
            None,
            13,
            FP,
            available_rows=rows,
        )
    with pytest.raises(MixtureQuarantinedSourceError):
        sample_mixture(
            {"choice": {"source": "fixture/choice", "rows": 1}},
            None,
            13,
            FP,
            available_rows=rows,
            source_status={"fixture/choice": "unknown"},
        )


def test_mixture_reports_only_eligible_train_availability() -> None:
    rows = [_example("choice", "fixture/choice", 0, "dev")]
    with pytest.raises(MixtureAvailabilityError) as excinfo:
        sample_mixture(
            {"choice": {"source": "fixture/choice", "rows": 1}},
            None,
            13,
            FP,
            available_rows=rows,
            source_status={"fixture/choice": "admitted"},
        )
    assert excinfo.value.available == 0


def test_preference_two_order_augmentation_survives_cleaning() -> None:
    from src.data.adapters.preference import PreferenceAdapter

    row = {
        "prompt": "Which reply is better?",
        "chosen": "a warm helpful reply with details",
        "rejected": "a curt unhelpful reply",
    }
    result = PreferenceAdapter(seed=3).to_decision_example([row])
    assert len(result.examples) == 2
    cleaned = clean_examples(result.examples)
    assert len(cleaned.kept) == 2
    assert {example.chosen_index for example in cleaned.kept} == {0, 1}
    assert cleaned.report.near_removed == 0


def test_q1_rejects_unicode_casefold_impostors() -> None:
    from src.data.core.schema import Option, compute_example_id

    for text in ("ſafe", "Yeſ", "helpfulneſſ", "Unanſwerable"):
        example = _example("risk", "fixture/risk", 0)
        options = (Option(text=text), example.options[1])
        forged = type(example)(
            example_id=compute_example_id(
                "risk", example.goal, example.state, [option.text for option in options]
            ),
            task_family="risk",
            goal=example.goal,
            state=example.state,
            options=options,
            source=example.source,
            license=example.license,
            group_id=example.group_id,
            split=example.split,
        )
        assert q1_short_option(forged) is True, text


def test_small_families_still_reach_every_split() -> None:
    """Rounded targets must not make dev/test structurally unreachable."""
    rows = [
        _example("choice", "fixture/choice", index, "train")
        for index in range(4)
    ]
    result = split(rows, seed=13)
    for name in ("train", "dev", "test"):
        assert len(result.by_split(name)) >= 1, name


def test_split_refuses_to_emit_an_empty_corpus() -> None:
    span = " ".join(f"shared-token-{index}" for index in range(13))
    rows = [
        normalize_example(
            {
                "task_family": "choice",
                "goal": f"goal {index} {span} tail",
                "state": f"state {index}",
                "options": [
                    {"text": f"alpha option {index} one two"},
                    {"text": f"beta option {index} three four"},
                ],
                "source": "fixture/source",
                "license": "MIT",
                "group_id": f"group-{index}",
                "split": "train",
                "chosen_index": 0,
            }
        )
        for index in range(12)
    ]
    with pytest.raises(SplitInputError, match="no trainable corpus remains"):
        split(rows, seed=13)
