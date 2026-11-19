"""Bucket keys for per-bucket calibration (FR-CALIB-01).

A ``BucketKey`` is the pair ``(task_family, option_count_bucket)`` rendered as
the canonical string ``"<task_family>|<option_count_bucket>"`` — e.g.
``"choice|3-5"``. The option-count bucket is one of ``"2"`` (exactly two
options), ``"3-5"`` (three to five options) and ``"6-8"`` (six to eight
options), per the data contract's 2–8 option range. Six families × three
buckets yields the 18 declared buckets; any declared bucket with no dev items
is skipped during fitting and recorded in ``skipped_buckets``.
"""

from __future__ import annotations

from .errors import CalibrationInputError

MIN_OPTIONS = 2
MAX_OPTIONS = 8

TASK_FAMILIES: tuple[str, ...] = (
    "choice",
    "preference",
    "scoring",
    "probability",
    "abstention",
    "risk",
)

OPTION_COUNT_BUCKETS: tuple[str, ...] = ("2", "3-5", "6-8")

BUCKET_SEPARATOR = "|"


def option_count_bucket(option_count: int) -> str:
    """Map an option count to its declared bucket.

    Raises :class:`CalibrationInputError` for counts outside the 2–8 contract.
    """
    if isinstance(option_count, bool) or not isinstance(option_count, int):
        raise CalibrationInputError(
            f"option_count must be an int, got {type(option_count).__name__}"
        )
    if option_count < MIN_OPTIONS or option_count > MAX_OPTIONS:
        raise CalibrationInputError(
            f"option_count {option_count} outside the {MIN_OPTIONS}-{MAX_OPTIONS} contract"
        )
    if option_count == 2:
        return "2"
    if option_count <= 5:
        return "3-5"
    return "6-8"


def make_bucket_key(task_family: str, option_count_bucket_name: str) -> str:
    """Render the canonical ``"<task_family>|<option_count_bucket>"`` key."""
    if task_family not in TASK_FAMILIES:
        raise CalibrationInputError(
            f"task_family {task_family!r} not one of {TASK_FAMILIES}"
        )
    if option_count_bucket_name not in OPTION_COUNT_BUCKETS:
        raise CalibrationInputError(
            f"option_count_bucket {option_count_bucket_name!r} "
            f"not one of {OPTION_COUNT_BUCKETS}"
        )
    return f"{task_family}{BUCKET_SEPARATOR}{option_count_bucket_name}"


def make_bucket_key_for_count(task_family: str, option_count: int) -> str:
    """Build the canonical key directly from a family and a raw option count."""
    return make_bucket_key(task_family, option_count_bucket(option_count))


def parse_bucket_key(key: str) -> tuple[str, str]:
    """Parse a canonical key into ``(task_family, option_count_bucket)``.

    Raises :class:`CalibrationInputError` on any malformed key — unknown
    family, unknown option-count bucket, missing separator or extra segments.
    """
    if not isinstance(key, str):
        raise CalibrationInputError(f"bucket key must be a str, got {type(key).__name__}")
    parts = key.split(BUCKET_SEPARATOR)
    if len(parts) != 2:
        raise CalibrationInputError(
            f"bucket key {key!r} must be '<task_family>{BUCKET_SEPARATOR}"
            f"<option_count_bucket>'"
        )
    family, bucket = parts
    if family not in TASK_FAMILIES:
        raise CalibrationInputError(
            f"bucket key {key!r}: task_family {family!r} not one of {TASK_FAMILIES}"
        )
    if bucket not in OPTION_COUNT_BUCKETS:
        raise CalibrationInputError(
            f"bucket key {key!r}: option_count_bucket {bucket!r} not one of "
            f"{OPTION_COUNT_BUCKETS}"
        )
    return family, bucket


def declared_bucket_keys() -> tuple[str, ...]:
    """The 18 declared buckets, in deterministic (family, bucket) order."""
    return tuple(
        make_bucket_key(family, bucket)
        for family in TASK_FAMILIES
        for bucket in OPTION_COUNT_BUCKETS
    )
