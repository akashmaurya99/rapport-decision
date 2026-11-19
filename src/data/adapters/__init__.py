"""Six task-family adapters: raw source rows → the unified schema.

One :class:`~src.data.adapters.base.Adapter` protocol, one family module per
task family. Every adapter declares ``source``, ``license``, ``task_family``
and ``name``, shuffles option order at ingest with a recorded seed, retains
``meta.original_order`` for reconstruction, and reports its figures
(conversion counts, the MMLU-Pro 10→8 reduction count, the
auxiliary_train-derived binary row count, per-subset conversion rates) in
``AdapterResult.report``. Unmappable rows are counted and rejected with a
typed reason code — never silently skipped.
"""

from .abstention import AbstentionAdapter
from .base import (
    ADAPTOR_BAD_INDEX,
    ADAPTOR_BAD_SCORE,
    ADAPTOR_BAD_TYPE,
    ADAPTOR_CONTRADICTORY_LABEL,
    ADAPTOR_EMPTY_STATE,
    ADAPTOR_META_MISSING,
    ADAPTOR_MISSING_FIELD,
    ADAPTOR_NO_GOLD,
    ADAPTOR_NO_OPTIONS,
    ADAPTOR_RISK_LABEL_LIMIT,
    ADAPTOR_TOO_LONG,
    ALL_ADAPTOR_CODES,
    DEFAULT_MAX_FIELD_CHARS,
    DEFAULT_MAX_STATE_CHARS,
    DEFAULT_SEED,
    Adapter,
    AdapterError,
    AdapterResult,
    ExampleMeta,
    RejectedRecord,
    base_report,
    fingerprint_row,
    restore_original_order,
    shuffle_order,
)
from .choice import ChoiceAdapter, MmluProAdapter
from .preference import PreferenceAdapter
from .probability import ProbabilityAdapter
from .risk import RiskAdapter
from .scoring import ScoringAdapter

__all__ = [
    "Adapter",
    "AdapterError",
    "AdapterResult",
    "AbstentionAdapter",
    "ALL_ADAPTOR_CODES",
    "ADAPTOR_BAD_INDEX",
    "ADAPTOR_BAD_SCORE",
    "ADAPTOR_BAD_TYPE",
    "ADAPTOR_CONTRADICTORY_LABEL",
    "ADAPTOR_EMPTY_STATE",
    "ADAPTOR_META_MISSING",
    "ADAPTOR_MISSING_FIELD",
    "ADAPTOR_NO_GOLD",
    "ADAPTOR_NO_OPTIONS",
    "ADAPTOR_RISK_LABEL_LIMIT",
    "ADAPTOR_TOO_LONG",
    "ChoiceAdapter",
    "DEFAULT_MAX_FIELD_CHARS",
    "DEFAULT_MAX_STATE_CHARS",
    "DEFAULT_SEED",
    "ExampleMeta",
    "MmluProAdapter",
    "PreferenceAdapter",
    "ProbabilityAdapter",
    "RejectedRecord",
    "RiskAdapter",
    "ScoringAdapter",
    "base_report",
    "fingerprint_row",
    "restore_original_order",
    "shuffle_order",
]

ALL_ADAPTERS = (
    ChoiceAdapter,
    MmluProAdapter,
    PreferenceAdapter,
    ScoringAdapter,
    ProbabilityAdapter,
    AbstentionAdapter,
    RiskAdapter,
)
