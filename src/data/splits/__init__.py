"""Deterministic group-aware splits, leakage checks and fingerprints.

Public surface: :func:`split` (the router), :class:`SplitResult`,
:class:`SplitManifest` (plus ``read_manifest``/``write_manifest`` JSON
helpers), :func:`membership_fingerprint`, and the leakage assertions
(:func:`assert_no_group_overlap`, :func:`assert_no_option_text_leak`,
:func:`assert_no_thirteen_gram_overlap`, :func:`assert_no_leakage`) with
their diagnostic ``find_*`` counterparts. Every failure mode raises a typed
:class:`SplitError` subclass carrying a stable ``code``.
"""

from .errors import (
    GroupOverlapError,
    ManifestError,
    ManifestFieldError,
    ManifestFingerprintError,
    ManifestGroupOverlapError,
    NGramLeakageError,
    OptionTextLeakageError,
    SplitError,
    SplitInputError,
)
from .fingerprint import is_sha256_hex, membership_fingerprint
from .leakage import (
    MIN_OPTION_TEXT_TOKENS,
    N_GRAM,
    assert_no_group_overlap,
    assert_no_leakage,
    assert_no_option_text_leak,
    assert_no_thirteen_gram_overlap,
    find_shared_option_texts,
    find_thirteen_gram_offender_ids,
    find_thirteen_gram_overlaps,
    split_pair_key,
    token_count,
)
from .manifest import (
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_OOD_SOURCE,
    RATIOS,
    SPLIT_NAMES,
    SplitManifest,
    ratios_dict,
    read_manifest,
    write_manifest,
)
from .router import SplitResult, split

__all__ = [
    "DEFAULT_DEDUP_THRESHOLD",
    "DEFAULT_OOD_SOURCE",
    "GroupOverlapError",
    "ManifestError",
    "ManifestFieldError",
    "ManifestFingerprintError",
    "ManifestGroupOverlapError",
    "MIN_OPTION_TEXT_TOKENS",
    "N_GRAM",
    "NGramLeakageError",
    "OptionTextLeakageError",
    "RATIOS",
    "SPLIT_NAMES",
    "SplitError",
    "SplitInputError",
    "SplitManifest",
    "SplitResult",
    "assert_no_group_overlap",
    "assert_no_leakage",
    "assert_no_option_text_leak",
    "assert_no_thirteen_gram_overlap",
    "find_shared_option_texts",
    "find_thirteen_gram_offender_ids",
    "find_thirteen_gram_overlaps",
    "is_sha256_hex",
    "membership_fingerprint",
    "read_manifest",
    "ratios_dict",
    "split",
    "split_pair_key",
    "token_count",
    "write_manifest",
]
