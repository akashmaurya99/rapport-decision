"""Deterministic synthetic fixtures for the end-to-end acceptance smoke run.

No real corpus is downloaded by this repository. These helpers manufacture a
tiny, fully validated set of :class:`DecisionExample` rows spanning all six
task families so the acceptance CLI can exercise every pipeline stage without
pretending that real data (or a CUDA device) was present. Every row carries a
content-addressed id computed by the production schema code, and every training
batch is derived from a fixed per-step seed so a kill/resume pair sees exactly
the same tensor stream.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from src.data.core.normalize import normalize_example
from src.data.core.schema import TASK_FAMILIES, DecisionExample
from src.tokenization.tokenizer import GOAL_ID, OPT_ID, TASK_ID

FAMILIES: tuple[str, ...] = (
    "choice",
    "preference",
    "scoring",
    "probability",
    "abstention",
    "risk",
)
DEFAULT_OOD_SOURCE = "UltraFeedback-binarized"
DEFAULT_SOURCE = "synthetic-fixture"
SEQ_LEN = 24
N_OPTIONS = 3
OPTION_POSITIONS: tuple[int, ...] = (4, 6, 8)


def _raw_row(family: str, index: int, *, ood_source: str) -> dict[str, Any]:
    is_ood = family == "choice" and index % 11 == 10
    source = ood_source if is_ood else DEFAULT_SOURCE
    if is_ood:
        split = "test"
    elif index % 10 < 7:
        split = "train"
    elif index % 10 < 9:
        split = "dev"
    else:
        split = "test"
    options = [
        {"text": f"{family} option {index} letter {letter}", "label": letter}
        for letter in ("a", "b", "c")
    ]
    raw: dict[str, Any] = {
        "task_family": family,
        "goal": f"choose the best {family} answer for case {index}",
        "state": f"synthetic context {index}: the road is wet and the lamp is on",
        "options": options,
        "source": source,
        "license": "synthetic",
        "group_id": f"group-{family}-{index // 4}",
        "split": split,
        "chosen_index": index % N_OPTIONS,
        "scores": [0.15, 0.55, 0.30],
        "answerable": bool(index % 5),
    }
    if family == "risk":
        raw["risk_labels"] = [f"synthetic-category-{index % 8}"]
    return raw


def make_rows(
    rows_per_family: int, *, ood_source: str = DEFAULT_OOD_SOURCE
) -> list[DecisionExample]:
    """Build ``rows_per_family`` validated rows for each of the six families."""
    if (
        not isinstance(rows_per_family, int)
        or isinstance(rows_per_family, bool)
        or rows_per_family < 2
    ):
        raise ValueError("rows_per_family must be an integer >= 2")
    if set(FAMILIES) != set(TASK_FAMILIES):
        raise ValueError("synthetic fixture families drifted from the schema contract")
    rows: list[DecisionExample] = []
    for family in FAMILIES:
        for index in range(rows_per_family):
            rows.append(normalize_example(_raw_row(family, index, ood_source=ood_source)))
    return rows


def data_fingerprint(rows: list[DecisionExample]) -> str:
    """Content fingerprint of an ordered row set (source and split included)."""
    payload = "\n".join(f"{row.example_id}:{row.source}:{row.split}" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_batch(step_seed: int, *, batch_size: int = 2) -> dict[str, torch.Tensor]:
    """Deterministic trainer batch with all multi-task labels attached.

    The tensor stream is a pure function of ``step_seed``; it does not consume
    global RNG state, which is what makes a fresh-process resume reproduce the
    uninterrupted loss trajectory.
    """
    generator = torch.Generator().manual_seed(step_seed)
    ids = torch.randint(10, 30_522, (batch_size, SEQ_LEN), generator=generator)
    ids[:, 0] = TASK_ID
    ids[:, 1] = GOAL_ID
    for position in OPTION_POSITIONS:
        ids[:, position] = OPT_ID
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "option_mask": torch.ones(batch_size, N_OPTIONS, dtype=torch.long),
        "task_ids": torch.zeros(batch_size, dtype=torch.long),
        "chosen_index": torch.randint(0, N_OPTIONS, (batch_size,), generator=generator),
        "score_targets": torch.rand(batch_size, N_OPTIONS, generator=generator),
        "probability_target": torch.rand(batch_size, generator=generator),
        "answerable_target": (torch.rand(batch_size, generator=generator) > 0.5).float(),
        "risk_targets": (torch.rand(batch_size, 8, generator=generator) > 0.7).float(),
        "preference_pairs": torch.tensor(
            [[row, 0, 1] for row in range(batch_size)], dtype=torch.long
        ),
    }
