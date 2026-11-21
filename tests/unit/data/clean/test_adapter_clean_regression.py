"""Regression: real adapter output survives Q1 and label conflicts are quarantined.

The original Q1 rejected every one-token fixed-vocabulary option, which
deleted four of the six task families at cleaning time. These tests feed
actual adapter output through ``clean_examples`` (not hand-built options) so
that drift between the adapters and the quality rule is caught.
"""

from __future__ import annotations

import pytest

from src.data.adapters import (
    AbstentionAdapter,
    ChoiceAdapter,
    PreferenceAdapter,
    ProbabilityAdapter,
    RiskAdapter,
    ScoringAdapter,
)
from src.data.clean import clean_examples, label_signature
from src.data.core import normalize_example

FAMILIES = ("choice", "preference", "scoring", "probability", "abstention", "risk")


def _adapter_rows() -> list[tuple[str, object, list[dict]]]:
    return [
        (
            "choice",
            ChoiceAdapter(seed=3),
            [
                {
                    "question": f"Which city hosts the national archive number {i}?",
                    "choices": [
                        "the city of Amsterdam on the coast",
                        "the city of Lyon near the river",
                        "the city of Turin under the Alps",
                    ],
                    "answer": "A",
                    "id": f"choice-{i}",
                }
                for i in range(2)
            ],
        ),
        (
            "preference",
            PreferenceAdapter(seed=3),
            [
                {
                    "chosen": (
                        f"Human: what is the plan {i}?\n\nAssistant: use the documented rollout"
                    ),
                    "rejected": (
                        f"Human: what is the plan {i}?\n\nAssistant: skip the documented checks"
                    ),
                    "id": f"pref-{i}",
                }
                for i in range(2)
            ],
        ),
        (
            "scoring",
            ScoringAdapter(seed=3),
            [
                {
                    "prompt": f"How should the operator handle incident {i}?",
                    "response": "Follow the written runbook and record the result.",
                    "helpfulness": 4,
                    "correctness": 3,
                    "coherence": 2,
                    "id": f"score-{i}",
                }
                for i in range(2)
            ],
        ),
        (
            "probability",
            ProbabilityAdapter(seed=3),
            [
                {
                    "question": f"Does the artifact {i} belong to the release set?",
                    "best_answer": "Yes it belongs to the release set.",
                    "incorrect_answers": ["No it belongs to the sandbox set."],
                    "id": f"tqa-{i}",
                }
                for i in range(2)
            ],
        ),
        (
            "abstention",
            AbstentionAdapter(seed=3),
            [
                {
                    "subset": "synthetic",
                    "question": f"Who approved the deployment window {i}?",
                    "answerable": bool(i),
                    "id": f"abs-{i}",
                }
                for i in range(2)
            ],
        ),
        (
            "risk",
            RiskAdapter(seed=3),
            [
                {
                    "prompt": f"Describe the safety review for case {i}.",
                    "categories": ["violence"] if i else [],
                    "id": f"risk-{i}",
                }
                for i in range(2)
            ],
        ),
    ]


def test_all_six_families_retain_rows_after_cleaning_real_adapter_output():
    examples = []
    for _family, adapter, rows in _adapter_rows():
        result = adapter.to_decision_example(rows)  # type: ignore[attr-defined]
        assert not result.rejected, result.rejected
        examples.extend(result.examples)

    cleaned = clean_examples(examples)

    surviving = {example.task_family for example in cleaned.kept}
    assert surviving == set(FAMILIES)
    assert cleaned.report.per_rule.get("Q1", 0) == 0
    assert len(cleaned.contested) == 0


def _same_content(family: str, **label_fields):
    raw = {
        "task_family": family,
        "goal": "Rate the response on each quality aspect.",
        "state": "The same prompt with the same response for every competing annotation",
        "options": [{"text": text} for text in ("helpfulness", "correctness", "coherence")],
        "source": "fixture",
        "license": "MIT",
        "group_id": "group-conflict",
        "split": "train",
    }
    raw.update(label_fields)
    return normalize_example(raw)


@pytest.mark.parametrize(
    ("family", "left", "right"),
    [
        ("scoring", {"scores": (1.0, 0.0, 0.5)}, {"scores": (0.0, 1.0, 0.5)}),
        (
            "abstention",
            {"chosen_index": 0, "answerable": True},
            {"chosen_index": 1, "answerable": False},
        ),
        (
            "probability",
            {"chosen_index": 0, "answerable": True},
            {"chosen_index": 1, "answerable": False},
        ),
        (
            "risk",
            {"risk_labels": ("violence",)},
            {"risk_labels": ("self-harm",)},
        ),
    ],
)
def test_conflicting_annotations_are_contested_not_merged(family, left, right):
    a = _same_content(family, **left)
    b = _same_content(family, **right)
    assert label_signature(a) != label_signature(b)

    result = clean_examples([a, b])

    assert result.kept == ()
    assert len(result.contested) == 2
    assert set(result.contested_ids) == {a.example_id, b.example_id}
