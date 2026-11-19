"""Shared report records for the cleaning pipeline.

``RejectionReport`` is the shared counter object handed to every quality
rule; ``CleanReport`` is the single summary record every output of the
pipeline carries. Keeping both in one module avoids import cycles between
the stage modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ReportMismatchError

#: Names of the six structural quality rules, in evaluation order.
RULE_ORDER: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")


@dataclass
class RejectionReport:
    """Per-rule rejection counters plus an audit trail of removed ids.

    The invariant this record exists to guarantee:

        ``total_rejected == sum(per_rule.values())``

    and, for a filter run that started with ``n`` rows and kept ``k``,

        ``total_rejected == n - k``

    The quality filter assigns each rejected row to exactly one rule (the
    first that fires, in :data:`RULE_ORDER`), so the sum can never exceed the
    number of removed rows. :meth:`verify` checks this loudly instead of
    trusting it.

    ``details`` maps each rule name to the example ids it rejected, for
    audit. It is informational only and never influences filtering.
    """

    per_rule: dict[str, int] = field(
        default_factory=lambda: {rule: 0 for rule in RULE_ORDER}
    )
    details: dict[str, list[str]] = field(
        default_factory=lambda: {rule: [] for rule in RULE_ORDER}
    )

    def increment(self, rule: str, example_id: str) -> None:
        if rule not in self.per_rule:
            raise ReportMismatchError(f"unknown quality rule {rule!r}")
        self.per_rule[rule] += 1
        self.details[rule].append(example_id)

    @property
    def total_rejected(self) -> int:
        """Sum of the per-rule counters; equals the number of removed rows."""
        return sum(self.per_rule.values())

    def verify(self, removed_rows: int) -> None:
        """Assert the report sums to ``removed_rows``; raise if not."""
        total = self.total_rejected
        if total != removed_rows:
            raise ReportMismatchError(
                f"per-rule counters sum to {total} but {removed_rows} rows were "
                f"removed; the report does not account for every removed row"
            )


@dataclass(frozen=True)
class CleanReport:
    """Summary of one pass through the cleaning pipeline.

    ``exact_removed``  — rows removed as byte-identical normalized duplicates.
    ``near_removed``   — rows removed as near-duplicates (MinHash estimate
                         at or above the recorded threshold).
    ``per_rule``       — quality-rule counters; ``sum(per_rule.values())``
                         equals the number of rows quality filtering removed.
    ``contested``      — rows quarantined to the contested list because their
                         group carries more than one gold ``chosen_index``.
    ``kept``           — rows that survive every stage.
    """

    exact_removed: int
    near_removed: int
    per_rule: dict[str, int]
    contested: int
    kept: int
