"""Fixtures for the data-clean unit tests."""

from __future__ import annotations

from typing import Any

import pytest

from src.data.core import normalize_example
from src.data.core.schema import DecisionExample


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


@pytest.fixture
def example_factory():
    return make_example
