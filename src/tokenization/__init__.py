"""Tokenizer, special-token handling and length bucketing for RAPPORT.

Public surface: ``RapportTokenizer``/``DEFAULT_TOKENIZER`` (the deterministic
stand-in tokenizer; RAPPORT training initializes from ``bge-small-en-v1.5``,
see TASK-13/TASK-15), the five special-token ids, ``encode_example`` (the
512-token joint-sequence budget with state-from-the-end truncation),
``bucket_examples`` (at most three padded batch shapes, right-pad, masked
tail), the per-source truncation counter, and the :class:`EncodedExample` /
:class:`BucketedBatch` records.
"""

from .encoding import (
    DEFAULT_BUDGET,
    PADDING_POLICY,
    TRUNCATION_COUNTS,
    TRUNCATION_MANIFEST_KEY,
    TRUNCATION_POLICY,
    TRUNCATION_POLICY_MANIFEST_KEY,
    BucketedBatch,
    EncodedExample,
    bucket_examples,
    default_bucket_widths,
    encode_example,
    reset_truncation_counts,
    truncation_counts,
)
from .errors import (
    BucketOverflowError,
    BudgetTooSmallError,
    TokenizationError,
)
from .tokenizer import (
    DEFAULT_TOKENIZER,
    GOAL_ID,
    OPT_CLOSE_ID,
    OPT_ID,
    PAD_ID,
    PAD_TOKEN,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    STATE_ID,
    TASK_ID,
    UNK_ID,
    UNK_TOKEN,
    RapportTokenizer,
)

__all__ = [
    "DEFAULT_BUDGET",
    "DEFAULT_TOKENIZER",
    "GOAL_ID",
    "OPT_CLOSE_ID",
    "OPT_ID",
    "PAD_ID",
    "PAD_TOKEN",
    "PADDING_POLICY",
    "SPECIAL_TOKEN_IDS",
    "SPECIAL_TOKENS",
    "STATE_ID",
    "TASK_ID",
    "TRUNCATION_COUNTS",
    "TRUNCATION_MANIFEST_KEY",
    "TRUNCATION_POLICY",
    "TRUNCATION_POLICY_MANIFEST_KEY",
    "UNK_ID",
    "UNK_TOKEN",
    "BucketedBatch",
    "BudgetTooSmallError",
    "BucketOverflowError",
    "EncodedExample",
    "RapportTokenizer",
    "TokenizationError",
    "bucket_examples",
    "default_bucket_widths",
    "encode_example",
    "reset_truncation_counts",
    "truncation_counts",
]
