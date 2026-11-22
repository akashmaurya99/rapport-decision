"""Cleaning pipeline: exact dedup → near dedup → contradiction audit → Q1–Q6.

The order matches the dataset strategy's pipeline (raw → ingest-normalize →
exact dedup → near dedup → contradiction audit → quality rules). Rows arrive
already normalized by TASK-03; this package never normalizes or re-labels.

Every stage feeds the next in input order, so results are deterministic.
Contested rows are excluded from the clean output and surfaced separately;
nothing is silently resolved or dropped without a counter. The returned
:class:`CleanResult` carries a :class:`CleanReport` with ``exact_removed``,
``near_removed``, ``per_rule``, ``contested`` and ``kept``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from src.data.core.schema import DecisionExample

from .contradiction import ContradictionAudit
from .dedup import DEFAULT_NEAR_THRESHOLD, ExactDedup, NearDedup
from .quality import DEFAULT_MAX_TOKENS, QualityFilter
from .report import CleanReport


@dataclass(frozen=True)
class CleanResult:
    """Full output of one cleaning pass.

    ``dedup_threshold`` records the near-dedup Jaccard threshold in effect
    (the dataset strategy wants it recorded in the split manifest).
    ``contested_ids`` lists the quarantined rows' example ids for audit.
    """

    kept: tuple[DecisionExample, ...]
    contested: tuple[DecisionExample, ...]
    report: CleanReport
    dedup_threshold: float
    contested_ids: tuple[str, ...] = field(default_factory=tuple)
    contested_path: Path | None = None


def clean_examples(
    examples: Sequence[DecisionExample],
    *,
    near_threshold: float = DEFAULT_NEAR_THRESHOLD,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    contested_path: str | Path | None = None,
    skip_near_dedup: bool = False,
) -> CleanResult:
    """Run the cleaning pipeline and return kept rows, contested rows, report.

    ``skip_near_dedup`` is for curated corpora whose rows are already unique by
    construction (each prompt/question appears once). Near-dedup generates
    O(n²) candidates and is intended for scraped, redundant corpora; skipping it
    here avoids hours of work while keeping exact dedup, contradiction and Q1-Q6.
    """
    exact = ExactDedup().run(examples)
    if skip_near_dedup:
        from .dedup import NearDedupResult

        near = NearDedupResult(
            kept=exact.kept,
            removed_ids=(),
            per_source={},
            threshold=near_threshold,
        )
    else:
        near = NearDedup(threshold=near_threshold).run(exact.kept)
    contradiction = ContradictionAudit().run(near.kept, contested_path=contested_path)
    quality = QualityFilter(max_tokens=max_tokens).run(contradiction.kept)

    report = CleanReport(
        exact_removed=exact.exact_removed,
        near_removed=near.near_removed,
        per_rule=dict(quality.report.per_rule),
        contested=contradiction.contested_count,
        kept=len(quality.kept),
    )
    return CleanResult(
        kept=quality.kept,
        contested=contradiction.contested,
        report=report,
        dedup_threshold=near.threshold,
        contested_ids=contradiction.contested_ids,
        contested_path=contradiction.contested_path,
    )
