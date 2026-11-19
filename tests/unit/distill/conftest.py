"""Fixtures for the distillation unit tests.

The repo root goes on ``sys.path`` so the source tree is imported as
``src.distill`` from the repository root, matching the other unit-test
packages. Every teacher here is a deterministic **stub** — no test performs
a live teacher download or any network access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.core import normalize_example  # noqa: E402
from src.data.core.schema import DecisionExample  # noqa: E402


def make_example(
    *,
    family: str = "choice",
    goal: str = "Pick the safest plan for the deployment rollout",
    state: str = "The service holds customer data and the rollout window is four hours",
    options: tuple[str, ...] = (
        "Roll out to ten percent of traffic first",
        "Roll out to all traffic immediately",
        "Delay the rollout by two full weeks",
    ),
    chosen_index: int | None = 0,
    source: str = "fixture",
    license: str = "MIT",
    group_id: str = "group-1",
    split: str = "train",
    **extra: Any,
) -> DecisionExample:
    record: dict[str, Any] = {
        "task_family": family,
        "goal": goal,
        "state": state,
        "options": [{"text": text} for text in options],
        "source": source,
        "license": license,
        "group_id": group_id,
        "split": split,
    }
    if chosen_index is not None:
        record["chosen_index"] = chosen_index
    record.update(extra)
    return normalize_example(record)


class StubTeacher:
    """Deterministic offline teacher.

    Returns ``base_offsets + seed_bias(example.example_id)`` — a fixed logit
    vector per example order, identical across processes and runs. Counts
    its own forwards in ``calls`` so tests can assert on the teacher side,
    independently of the module's counter.
    """

    def __init__(self, logprobs: Sequence[float] | None = None) -> None:
        self.logprobs = list(logprobs) if logprobs is not None else [2.0, 0.0, -1.0]
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, example: DecisionExample, prompt: str) -> list[float]:
        self.calls += 1
        self.prompts.append(prompt)
        vec = list(self.logprobs)
        while len(vec) < len(example.options):
            vec.append(0.0)
        return vec[: len(example.options)]


@pytest.fixture
def example_factory():
    return make_example


@pytest.fixture
def stub_teacher_factory():
    return StubTeacher
