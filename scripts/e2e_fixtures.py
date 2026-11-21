"""Deterministic synthetic fixtures for the end-to-end acceptance smoke run.

No real corpus is downloaded by this repository. These helpers manufacture a
tiny, fully validated set of :class:`DecisionExample` rows spanning all six
task families so the acceptance CLI can exercise every pipeline stage without
pretending that real data (or a CUDA device) was present. Every row carries a
content-addressed id computed by the production schema code, and every training
batch is derived from a fixed per-step seed so a kill/resume pair sees exactly
the same tensor stream.

Small fixture sizes are deliberately *not* train-only: dev/test and at least
one withheld-source (OOD) row are present from ``rows_per_family >= 2`` so the
split and robustness stages always have something to report. Labels are
family-appropriate (scoring carries ``scores`` and no ``chosen_index``; risk
carries ``risk_labels`` and no ``chosen_index``; choice/preference carry
``chosen_index``; probability/abstention carry ``chosen_index`` and
``answerable``). State text is drawn from a varied template pool so a fixed
13-token span cannot collide across splits and abort the clean build.
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

# Distinct openings keep any 13-token window unique across rows that land in
# different splits. The index token on every template is the final anchor.
_STATE_TEMPLATES: tuple[str, ...] = (
    "synthetic context {i}: the road is wet and the lamp is on",
    "audit record {i}: a delivery truck blocks the narrow lane at dawn",
    "archive room {i}: the thermostat holds eighteen degrees all night",
    "harbor shift {i}: the crane operator logs a delayed container move",
    "field note {i}: the river gauge rose after the upstream storm",
    "kitchen log {i}: the sourdough proof ran long on the lower shelf",
    "rail depot {i}: a spare coupling was fitted to the evening car",
    "lab bench {i}: the centrifuge spun a sealed sample for eleven minutes",
    "office memo {i}: the budget review moved to the following Tuesday",
    "greenhouse {i}: the vent opened when the afternoon sun hit the glass",
    "studio take {i}: the microphone stand slipped during the second verse",
    "clinic intake {i}: the triage desk recorded a routine follow up",
)


def _split_for(index: int, rows_per_family: int, *, is_ood: bool) -> str:
    """Deterministic train/dev/test assignment for small fixture sizes."""
    if is_ood:
        return "test"
    if rows_per_family == 2:
        return ("train", "dev")[index]
    if rows_per_family == 3:
        return ("train", "dev", "test")[index]
    return ("train", "dev", "train", "test")[index % 4]


def _raw_row(
    family: str,
    index: int,
    *,
    ood_source: str,
    rows_per_family: int,
) -> dict[str, Any]:
    # Small sizes have no index % 11 == 10 row, so withhold the last row of
    # the choice family instead; larger sizes keep the original schedule.
    if rows_per_family < 11:
        is_ood = family == "choice" and index == rows_per_family - 1
    else:
        is_ood = family == "choice" and index % 11 == 10
    source = ood_source if is_ood else DEFAULT_SOURCE
    split = _split_for(index, rows_per_family, is_ood=is_ood)

    options = [
        {"text": f"{family} option {index} letter {letter}", "label": letter}
        for letter in ("a", "b", "c")
    ]
    raw: dict[str, Any] = {
        "task_family": family,
        "goal": f"choose the best {family} answer for case {index}",
        # Family prefix keeps the 13-gram spanning goal-tail + state distinct
        # across families: the real split router quarantines cross-split exact
        # 13-grams, and otherwise identical states at the same index would
        # exclude every row when families land in different splits.
        "state": f"{family} " + _STATE_TEMPLATES[index % len(_STATE_TEMPLATES)].format(i=index),
        "options": options,
        "source": source,
        "license": "synthetic",
        # One group per fixture row: small smoke corpora can realize train/dev/
        # test instead of collapsing every row of a family into one atomic split.
        "group_id": f"group-{family}-{index}",
        "split": split,
    }
    # Family-appropriate labels: only the families that consume a field get
    # it, so the fixtures mirror the adapters' real output contracts.
    if family in ("choice", "preference", "probability", "abstention"):
        raw["chosen_index"] = index % N_OPTIONS
    if family == "scoring":
        raw["scores"] = [0.15, 0.55, 0.30]
    if family in ("probability", "abstention"):
        raw["answerable"] = bool(index % 2)
    if family == "risk":
        raw["risk_labels"] = [f"synthetic-category-{index % 8}"]
    return raw


def make_raw_rows(
    rows_per_family: int, *, ood_source: str = DEFAULT_OOD_SOURCE
) -> list[dict[str, Any]]:
    """Build the raw (unnormalized) fixture mappings for the acquire stage.

    The e2e runner acquires these first and then runs them through the same
    production ingest path as real rows: :func:`normalize_example` finalizes
    each one (the six source-specific adapters consume source-format rows,
    which these unified-layout fixtures deliberately are not).
    """
    if (
        not isinstance(rows_per_family, int)
        or isinstance(rows_per_family, bool)
        or rows_per_family < 2
    ):
        raise ValueError("rows_per_family must be an integer >= 2")
    if set(FAMILIES) != set(TASK_FAMILIES):
        raise ValueError("synthetic fixture families drifted from the schema contract")
    return [
        _raw_row(family, index, ood_source=ood_source, rows_per_family=rows_per_family)
        for family in FAMILIES
        for index in range(rows_per_family)
    ]


def make_rows(
    rows_per_family: int, *, ood_source: str = DEFAULT_OOD_SOURCE
) -> list[DecisionExample]:
    """Build ``rows_per_family`` validated rows for each of the six families."""
    return [normalize_example(raw) for raw in make_raw_rows(
        rows_per_family, ood_source=ood_source
    )]


def data_fingerprint(rows: list[DecisionExample]) -> str:
    """Content fingerprint of an ordered row set (source and split included).

    This is intentionally *not* :func:`src.data.splits.membership_fingerprint`:
    the e2e runner fingerprints the ordered synthetic fixture including each
    row's ``source`` and ``split``, so a changed fixture order or OOD
    assignment invalidates the run fingerprint and a resume is refused. The
    canonical split fingerprint, by contrast, is order-invariant and hashes
    only ``(example_id, split)`` membership. Both are deterministic; their
    semantics differ by design, so neither replaces the other.
    """
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
