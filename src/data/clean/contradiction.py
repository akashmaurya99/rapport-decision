"""Contradiction audit: groups with more than one gold label are quarantined.

Grouping key: ``(task_family, goal, state, sorted option texts)`` — exactly
the content identity TASK-03 hashes, minus the label. Within a group, every
non-``None`` gold ``chosen_index`` must agree. If two rows disagree, the
*entire group* — every row, including rows with no ``chosen_index`` — is
moved to the ``contested`` output list. No label is ever picked: there is no
branch in this module that selects, averages, or "majority-votes" a gold
index, because silently resolving a contradiction is how a benchmark gets
poisoned.

Quarantined rows are excluded from the clean output and reported: the count
of contested rows, the number of contested groups, and the example ids are
all on the result record. Optionally, the quarantined rows are written as
JSONL to ``contested_path`` so the contested file exists on disk with the
same content as the in-memory list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.data.core.schema import DecisionExample
from src.data.core.serialize import to_dict

from .errors import ContestedWriteError


def group_key(example: DecisionExample) -> tuple:
    """Return the contradiction grouping key for ``example``.

    ``(task_family, goal, state, sorted option texts)`` — ``chosen_index``
    is deliberately excluded; it is the coordinate the audit disagrees over.
    """
    return (
        example.task_family,
        example.goal,
        example.state,
        tuple(sorted(option.text for option in example.options)),
    )


@dataclass(frozen=True)
class ContradictionResult:
    """Output of :class:`ContradictionAudit`."""

    kept: tuple[DecisionExample, ...]
    contested: tuple[DecisionExample, ...]
    contested_ids: tuple[str, ...]
    contested_groups: int
    contested_path: Path | None = None

    @property
    def contested_count(self) -> int:
        return len(self.contested)


def write_contested(contested: Sequence[DecisionExample], path: str | Path) -> int:
    """Write quarantined rows to a JSONL contested file; return the count."""
    destination = Path(path)
    try:
        with destination.open("w", encoding="utf-8") as handle:
            for example in contested:
                handle.write(
                    json.dumps(to_dict(example), ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
    except OSError as exc:
        raise ContestedWriteError(
            f"could not write contested rows to {destination}: {exc}"
        ) from exc
    return len(contested)


class ContradictionAudit:
    """Quarantine groups whose gold ``chosen_index`` differs across rows."""

    def run(
        self,
        examples: Sequence[DecisionExample],
        *,
        contested_path: str | Path | None = None,
    ) -> ContradictionResult:
        groups: dict[tuple, list[DecisionExample]] = {}
        order: list[tuple] = []
        for example in examples:
            key = group_key(example)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(example)

        kept: list[DecisionExample] = []
        contested: list[DecisionExample] = []
        contested_groups = 0
        for key in order:
            rows = groups[key]
            golds = {
                row.chosen_index for row in rows if row.chosen_index is not None
            }
            if len(golds) > 1:
                contested.extend(rows)
                contested_groups += 1
            else:
                kept.extend(rows)

        if contested_path is not None:
            write_contested(contested, contested_path)
            destination: Path | None = Path(contested_path)
        else:
            destination = None
        return ContradictionResult(
            kept=tuple(kept),
            contested=tuple(contested),
            contested_ids=tuple(row.example_id for row in contested),
            contested_groups=contested_groups,
            contested_path=destination,
        )
