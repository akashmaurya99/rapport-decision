"""Shared example factory for the data-splits unit tests.

Named uniquely (not ``conftest``) so ``from splits_test_support import ...``
cannot collide with another directory's ``conftest`` module when several test
directories are collected together.
"""

from __future__ import annotations

import itertools
from typing import Any

from src.data.core import normalize_example
from src.data.core.schema import DecisionExample

_counter = itertools.count(1)


def make_example(
    *,
    uid: int | None = None,
    family: str = "choice",
    goal: str | None = None,
    state: str | None = None,
    options: tuple[str, ...] | None = None,
    chosen_index: int | None = 0,
    source: str = "fixture",
    license: str = "MIT",
    group_id: str | None = None,
    split: str = "train",
    **extra: Any,
) -> DecisionExample:
    if uid is None:
        uid = next(_counter)
    goal = goal if goal is not None else (
        f"goal-{uid} tok-a{uid} tok-b{uid} tok-c{uid} pick the safest rollout plan"
    )
    state = state if state is not None else (
        f"state-{uid} tok-d{uid} tok-e{uid} tok-f{uid} tok-g{uid} the service holds "
        f"customer data with a four hour window"
    )
    options = options or (
        f"option-alpha-{uid} unq{uid}x roll out ten percent first",
        f"option-beta-{uid} unq{uid}y roll out all traffic now",
        f"option-gamma-{uid} unq{uid}z delay the rollout two weeks",
    )
    group_id = group_id if group_id is not None else f"group-{uid}"
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
    if family in ("choice", "preference"):
        record["chosen_index"] = 0 if chosen_index is None else chosen_index
    elif chosen_index is not None:
        record["chosen_index"] = chosen_index
    record.update(extra)
    return normalize_example(record)
