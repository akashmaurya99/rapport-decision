"""Data cleaning for RAPPORT: dedup, quality rules, contradiction audit.

Public surface: :func:`clean_examples` (the full pipeline), the four stage
classes (:class:`ExactDedup`, :class:`NearDedup`, :class:`ContradictionAudit`,
:class:`QualityFilter`), the report records (:class:`CleanReport`,
:class:`RejectionReport`, :data:`RULE_ORDER`), and the helpers used by tests
(:func:`exact_key`, :func:`canonical_document`, :func:`shingles`,
:func:`jaccard_index`, :class:`MinHasher`, :func:`estimate_jaccard`,
:func:`group_key`, :func:`write_contested`, the per-stage predicates).
"""

from .contradiction import (
    ContradictionAudit,
    ContradictionResult,
    group_key,
    write_contested,
)
from .dedup import (
    DEFAULT_NEAR_THRESHOLD,
    NUM_PERM,
    SHINGLE_SIZE,
    ExactDedup,
    ExactDedupResult,
    MinHasher,
    NearDedup,
    NearDedupResult,
    canonical_document,
    estimate_jaccard,
    exact_key,
    gold_signature,
    jaccard_index,
    shingles,
)
from .errors import CleanError, ContestedWriteError, ReportMismatchError
from .pipeline import CleanResult, clean_examples
from .quality import (
    DEFAULT_MAX_TOKENS,
    PLACEHOLDER_OPTIONS,
    REAL_OPTIONS,
    RULES,
    QualityFilter,
    QualityResult,
    q1_short_option,
    q2_markup_residue,
    q3_duplicate_option,
    q4_chosen_index_out_of_range,
    q5_too_long,
    q6_punctuation_only_or_placeholder,
)
from .report import RULE_ORDER, CleanReport, RejectionReport

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_NEAR_THRESHOLD",
    "NUM_PERM",
    "PLACEHOLDER_OPTIONS",
    "REAL_OPTIONS",
    "RULES",
    "RULE_ORDER",
    "SHINGLE_SIZE",
    "CleanError",
    "CleanReport",
    "CleanResult",
    "ContradictionAudit",
    "ContradictionResult",
    "ContestedWriteError",
    "ExactDedup",
    "ExactDedupResult",
    "MinHasher",
    "NearDedup",
    "NearDedupResult",
    "QualityFilter",
    "QualityResult",
    "RejectionReport",
    "ReportMismatchError",
    "canonical_document",
    "clean_examples",
    "estimate_jaccard",
    "exact_key",
    "group_key",
    "jaccard_index",
    "q1_short_option",
    "q2_markup_residue",
    "q3_duplicate_option",
    "q4_chosen_index_out_of_range",
    "q5_too_long",
    "q6_punctuation_only_or_placeholder",
    "shingles",
    "write_contested",
    "gold_signature",
]
