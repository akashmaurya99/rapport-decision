"""Deterministic, group-aware, family-stratified splitting.

Assignment rule
---------------
* Every row of the designated OOD source (default ``UltraFeedback-binarized``)
  is assigned to the evaluation side (the ``test`` split) — never train/dev.
  Any *other* row that shares a ``group_id`` with an OOD row follows its
  group to the evaluation side, so no group can ever straddle the OOD
  boundary and the leakage checks stay meaningful.
* All remaining rows are split 70/15/15 **by row count within each family**
  (stratified). Rows of one ``group_id`` are atomic: they land in one split.
* The assignment is a pure function of ``(seed, group_id)``: groups are
  ordered by ``sha256("rapport:split:v1" || seed || group_id)`` and each
  group is placed into the split with the largest row-count deficit against
  its family's 70/15/15 target (water-filling). Ties prefer train, then dev,
  then test. No RNG state, wall clock or platform artifact is consulted, so
  the router is deterministic given ``(seed, inputs)`` on every machine.

Groups spanning families: a group is assigned once — in the first family
(processed in sorted family order) that claims it — and every row of the
group, in every family, follows that assignment. Stratification is then
still exact per family up to group atomicity.

After assignment, three leakage checks run unconditionally and are fatal:
group disjointness, identical normalized option text of ≥40 tokens across
splits, and exact 13-gram overlap across splits.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, replace
from typing import Sequence

from src.data.core.schema import TASK_FAMILIES, DecisionExample

from .errors import SplitInputError
from .fingerprint import membership_fingerprint
from .leakage import (
    assert_no_group_overlap,
    assert_no_leakage,
)
from .manifest import (
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_OOD_SOURCE,
    RATIOS,
    SPLIT_NAMES,
    SplitManifest,
    ratios_dict,
)

_MAX_SEED = (1 << 64) - 1
_HASH_DOMAIN = "rapport:split:v1"
_FAMILY_ORDER = tuple(sorted(TASK_FAMILIES))
_PRIORITY = {"train": 2, "dev": 1, "test": 0}  # tie-break: prefer train, then dev

__all__ = [
    "DEFAULT_DEDUP_THRESHOLD",
    "DEFAULT_OOD_SOURCE",
    "RATIOS",
    "SPLIT_NAMES",
    "SplitResult",
    "split",
]


@dataclass(frozen=True)
class SplitResult:
    """The split rows plus the audit manifest."""

    examples: tuple[DecisionExample, ...]
    manifest: SplitManifest

    def by_split(self, name: str) -> tuple[DecisionExample, ...]:
        """All examples assigned to ``name`` (a copy; order is input order)."""
        if name not in SPLIT_NAMES:
            raise SplitInputError(
                f"unknown split {name!r}, expected one of {list(SPLIT_NAMES)}",
                field="split",
            )
        return tuple(example for example in self.examples if example.split == name)

    @property
    def train(self) -> tuple[DecisionExample, ...]:
        return self.by_split("train")

    @property
    def dev(self) -> tuple[DecisionExample, ...]:
        return self.by_split("dev")

    @property
    def test(self) -> tuple[DecisionExample, ...]:
        return self.by_split("test")

    @property
    def ood(self) -> tuple[DecisionExample, ...]:
        """The rows of the withheld OOD source (all on the evaluation side)."""
        return tuple(
            example for example in self.examples if example.source == self.manifest.ood_source
        )

    @property
    def fingerprint(self) -> str:
        return self.manifest.fingerprint


def _group_digest(seed: int, group_id: str) -> str:
    payload = f"{_HASH_DOMAIN}:{seed}:{group_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitInputError(f"seed must be an integer, got {seed!r}", field="seed")
    if seed < 0 or seed > _MAX_SEED:
        raise SplitInputError(f"seed must lie in [0, {_MAX_SEED}], got {seed}", field="seed")
    return seed


def _validate_source(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SplitInputError(
            f"{field_name} must be a non-empty string, got {value!r}",
            field=field_name,
        )
    return value.strip()


def _validate_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SplitInputError(
            f"dedup_threshold must be a number, got {value!r}", field="dedup_threshold"
        )
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise SplitInputError(
            f"dedup_threshold must lie in [0, 1], got {threshold!r}", field="dedup_threshold"
        )
    return threshold


def _targets(row_count: int) -> dict[str, int]:
    train = int(row_count * RATIOS["train"] + 0.5)
    dev = int(row_count * RATIOS["dev"] + 0.5)
    return {"train": train, "dev": dev, "test": row_count - train - dev}


def split(
    examples: Sequence[DecisionExample],
    seed: int,
    *,
    ood_source: str = DEFAULT_OOD_SOURCE,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> SplitResult:
    """Split ``examples`` into train/dev/test, deterministically.

    See the module docstring for the assignment rule and the leakage checks.
    Raises :class:`SplitInputError` on malformed arguments and the fatal
    typed leakage errors from :mod:`src.data.splits.leakage` on any leak.
    """
    seed = _validate_seed(seed)
    ood_source = _validate_source(ood_source, field_name="ood_source")
    dedup_threshold = _validate_threshold(dedup_threshold)
    try:
        rows = list(examples)
    except TypeError as exc:
        raise SplitInputError(
            f"examples must be a sequence of DecisionExample records: {exc}",
            field="examples",
        ) from exc
    for index, row in enumerate(rows):
        if not isinstance(row, DecisionExample):
            raise SplitInputError(
                f"examples[{index}] is a {type(row).__name__}, expected DecisionExample",
                field="examples",
            )

    # Step 1 — OOD groups: every group containing a row of the withheld
    # source is pinned to the evaluation side (test), wholesale.
    ood_groups = {row.group_id for row in rows if row.source == ood_source}
    assignment: dict[str, str] = {group_id: "test" for group_id in ood_groups}

    # Step 2 — family-stratified water-filling over the remaining groups.
    family_group_rows: dict[str, Counter[str]] = {family: Counter() for family in _FAMILY_ORDER}
    for row in rows:
        if row.group_id in assignment:
            continue
        family_group_rows[row.task_family][row.group_id] += 1

    family_rows: dict[str, int] = {
        family: sum(counts.values()) for family, counts in family_group_rows.items()
    }

    for family in _FAMILY_ORDER:
        counts = family_group_rows[family]
        if not counts:
            continue
        targets = _targets(family_rows[family])
        current = {name: 0 for name in SPLIT_NAMES}
        ordered = sorted(
            counts.items(), key=lambda item: (_group_digest(seed, item[0]), item[0])
        )
        for group_id, size in ordered:
            existing = assignment.get(group_id)
            if existing is not None:
                # Group already claimed by an earlier family (or pinned to the
                # OOD side): its rows in this family follow that assignment.
                current[existing] += size
                continue
            chosen = max(
                SPLIT_NAMES,
                key=lambda name: (targets[name] - current[name], _PRIORITY[name]),
            )
            assignment[group_id] = chosen
            current[chosen] += size

    # Step 3 — materialize the split records (split is not part of the
    # content hash, so example_id is unchanged by the assignment).
    output = tuple(
        row
        if row.split == assignment[row.group_id]
        else replace(row, split=assignment[row.group_id])
        for row in rows
    )

    # Step 4 — fatal leakage verification on the final assignment.
    overlap = assert_no_group_overlap(output)
    assert_no_leakage(output)

    # Step 5 — counts, fingerprint, manifest.
    counts_out: dict[str, dict[str, int]] = {}
    totals = {name: 0 for name in SPLIT_NAMES}
    ood_rows = 0
    for example in output:
        family_counts = counts_out.setdefault(
            example.task_family, {name: 0 for name in SPLIT_NAMES}
        )
        family_counts[example.split] += 1
        totals[example.split] += 1
        if example.source == ood_source:
            ood_rows += 1
    fingerprint = membership_fingerprint(
        (example.example_id, example.split) for example in output
    )
    manifest = SplitManifest(
        fingerprint=fingerprint,
        counts=counts_out,
        group_overlap=overlap,
        dedup_threshold=dedup_threshold,
        ood_source=ood_source,
        withheld_source=ood_source,
        seed=seed,
        totals=totals,
        ood_rows=ood_rows,
        ratios=ratios_dict(),
    )
    return SplitResult(examples=output, manifest=manifest)
