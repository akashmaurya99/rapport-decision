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
  group is placed into the split furthest *below* its target **relative to
  that target's size** (proportional water-filling). Ties prefer train, then
  dev, then test. Relative (not absolute) deficit prevents a small test
  quota from being starved by a large train quota when groups are coarse.
  No RNG state, wall clock or platform artifact is consulted, so the router
  is deterministic given ``(seed, inputs)`` on every machine.

Groups spanning families: a group is assigned once — in the first family
(processed in sorted family order) that claims it — and every row of the
group, in every family, follows that assignment. Stratification is then
still exact per family up to group atomicity.

After assignment, group disjointness and identical ≥40-token option text
across splits are fatal. Rows contributing a cross-split exact 13-gram are
instead moved to a *reported exclusion bucket*: they are omitted from the
emitted splits and counted in the split manifest, and a hard assertion then
verifies no overlap remains in the emitted rows. This keeps a fixed adapter
directive from aborting a whole build while never hiding a residual leak.
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
    assert_no_option_text_leak,
    find_thirteen_gram_offender_ids,
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


def _targets(row_count: int, group_count: int = 0) -> dict[str, int]:
    """Row targets per split, with a floor so no split is structurally empty.

    Rounding 70/15/15 can produce a zero target for dev or test on small
    corpora, which would make that split unreachable (a family would have no
    dev rows at all). When the family has enough atomic groups to fill every
    split, each split keeps at least one row and train absorbs the remainder.
    Large corpora are unaffected: their rounded targets already exceed one.
    """
    train = int(row_count * RATIOS["train"] + 0.5)
    dev = int(row_count * RATIOS["dev"] + 0.5)
    test = row_count - train - dev
    if group_count >= 2:
        dev = max(dev, 1)
    if group_count >= 3:
        test = max(test, 1)
    train = max(row_count - dev - test, 1)
    return {"train": train, "dev": dev, "test": test}


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
        targets = _targets(family_rows[family], len(counts))
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
            # Relative deficit: how far below this split's own quota it is.
            # Absolute deficits starve a small quota when groups are coarse.
            chosen = max(
                SPLIT_NAMES,
                key=lambda name: (
                    (targets[name] - current[name]) / targets[name]
                    if targets[name]
                    else -1.0,
                    _PRIORITY[name],
                ),
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

    # Step 4 — fatal checks plus the reported 13-gram exclusion bucket.
    overlap = assert_no_group_overlap(output)
    # Option-text leakage is fatal on the full assignment (never quarantined):
    # a 45-token identical option is an authored duplicate, not boilerplate.
    assert_no_option_text_leak(output)
    offenders = find_thirteen_gram_offender_ids(output)
    emitted = (
        tuple(row for row in output if row.example_id not in offenders)
        if offenders
        else output
    )
    # Hard assertion: nothing leaky may survive in the emitted splits, and a
    # nonempty corpus may never silently become an empty "successful" build.
    assert_no_leakage(emitted)
    if rows and not emitted:
        raise SplitInputError(
            f"13-gram quarantine removed all {len(rows)} rows; no trainable corpus remains"
        )

    # Step 5 — counts, fingerprint, manifest (of the emitted rows only).
    counts_out: dict[str, dict[str, int]] = {}
    totals = {name: 0 for name in SPLIT_NAMES}
    ood_rows = 0
    for example in emitted:
        family_counts = counts_out.setdefault(
            example.task_family, {name: 0 for name in SPLIT_NAMES}
        )
        family_counts[example.split] += 1
        totals[example.split] += 1
        if example.source == ood_source:
            ood_rows += 1
    fingerprint = membership_fingerprint(
        (example.example_id, example.split) for example in emitted
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
        excluded_thirteen_gram_rows=len(offenders),
        excluded_thirteen_gram_ids=tuple(sorted(offenders)),
    )
    return SplitResult(examples=emitted, manifest=manifest)
