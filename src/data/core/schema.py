"""The DecisionExample contract: option and example records, content identity.

``DecisionExample`` is a frozen dataclass. The one invariant the class enforces
itself is content integrity: ``example_id`` must equal the sha256 of
``(task_family, NFC goal, NFC state, the sorted NFC option texts)``. Passing a
stale or tampered ``example_id`` raises :class:`IdMismatchError` rather than
being recorded, so a hash is never trusted from the outside.

Construct examples through ``normalize_example`` (``src/data/core/normalize.py``);
direct construction is supported for tests and round-trips but performs no
normalization.

The eight risk category names are deliberately *not* enumerated here: no
document in this repository declares them (Aegis 2.0 ships 13 categories; the
risk head consumes eight slots). Membership is therefore not validated — see
``.archgen/rapport/.agent/notes/TASK-03.md``.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Any

from .errors import BadTypeError, IdMismatchError

TASK_FAMILIES = frozenset(
    {"choice", "preference", "scoring", "probability", "abstention", "risk"}
)
SPLITS = frozenset({"train", "dev", "test"})

MIN_OPTIONS = 2
MAX_OPTIONS = 8
RISK_CATEGORY_COUNT = 8

# Domain separator prefixed to the hashed payload so a decision-example digest
# can never collide with a digest of some other JSON list in the project.
_ID_DOMAIN = "rapport:decision-example:v1"


@dataclass(frozen=True)
class Option:
    """One candidate option. ``text`` is non-empty and unique per example
    after normalization; ``label`` is the source label when the source is
    multiple-choice, otherwise ``None`` (never invented)."""

    text: str
    label: str | None = None


def compute_example_id(
    task_family: str,
    goal: str,
    state: str,
    option_texts: list[str] | tuple[str, ...],
) -> str:
    """Return the sha256 content hash of an example.

    The payload is the domain separator, the task family, the (already NFC)
    goal and state, and the *sorted* option texts, serialized as compact JSON.
    Sorting the option texts makes the hash invariant to option permutation;
    the task family, goal, state and texts themselves are hashed verbatim, so
    distinct states produce distinct ids.
    """
    if not isinstance(task_family, str) or not isinstance(goal, str) or not isinstance(state, str):
        raise BadTypeError(
            "task_family, goal and state must be strings",
            field="task_family" if not isinstance(task_family, str) else None,
        )
    texts: list[str] = []
    for index, text in enumerate(option_texts):
        if not isinstance(text, str):
            raise BadTypeError(
                f"option text at index {index} must be a string, "
                f"got {type(text).__name__}",
                field="options",
            )
        texts.append(text)
    payload: list[Any] = [
        _ID_DOMAIN,
        unicodedata.normalize("NFC", task_family),
        unicodedata.normalize("NFC", goal),
        unicodedata.normalize("NFC", state),
        sorted(unicodedata.normalize("NFC", t) for t in texts),
    ]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class DecisionExample:
    """One validated row of the unified schema.

    Collections are stored as tuples so the record is hashable and immutable.
    ``example_id`` is verified — never trusted — against the recomputed
    content hash on every construction.
    """

    example_id: str
    task_family: str
    goal: str
    state: str
    options: tuple[Option, ...]
    source: str
    license: str
    group_id: str
    split: str
    chosen_index: int | None = None
    scores: tuple[float, ...] | None = None
    answerable: bool | None = None
    risk_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        options = self.options
        if isinstance(options, list):
            options = tuple(options)
            object.__setattr__(self, "options", options)
        scores = self.scores
        if isinstance(scores, list):
            try:
                scores = tuple(float(s) for s in scores)
            except (TypeError, ValueError) as exc:
                raise BadTypeError(
                    f"scores must be numbers: {exc}", field="scores"
                ) from exc
            object.__setattr__(self, "scores", scores)
        risk_labels = self.risk_labels
        if isinstance(risk_labels, list):
            risk_labels = tuple(risk_labels)
            object.__setattr__(self, "risk_labels", risk_labels)

        try:
            texts = [opt.text for opt in options]
            expected = compute_example_id(self.task_family, self.goal, self.state, texts)
        except (BadTypeError, AttributeError, TypeError) as exc:
            if isinstance(exc, BadTypeError):
                raise
            raise BadTypeError(
                f"options must be Option records with string text: {exc}", field="options"
            ) from exc
        if self.example_id != expected:
            raise IdMismatchError(
                f"example_id {self.example_id!r} does not match the recomputed "
                f"content hash {expected!r}",
                field="example_id",
            )


# Field names in serialized form, in canonical order. Used by the serializer
# and by unknown-key rejection. Not part of the hash payload.
OPTIONAL_FIELDS = ("chosen_index", "scores", "answerable", "risk_labels")
REQUIRED_FIELDS = (
    "task_family",
    "goal",
    "state",
    "options",
    "source",
    "license",
    "group_id",
    "split",
)
SERIALIZED_FIELDS = ("example_id", *REQUIRED_FIELDS, *OPTIONAL_FIELDS)
