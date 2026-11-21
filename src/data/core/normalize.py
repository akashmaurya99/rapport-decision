"""Normalization and validation for the DecisionExample contract.

Pipeline for every text field: Unicode NFC → strip HTML (tags dropped,
``<script>``/``<style>`` bodies removed, entities decoded) → collapse all
whitespace runs to a single space and strip the ends → Unicode NFC again.

``normalize_example`` raises the *first* violation as a typed error; it never
raises a bare exception for malformed input. ``validate_example`` runs the
same checks and returns every violation found, so callers can audit a batch
without exception plumbing.

Nothing is recorded that the source did not provide: optional fields that are
absent stay absent (``None``), a provided ``example_id`` is verified against
the recomputed content hash and never silently overwritten, and unknown keys
are rejected rather than dropped. HTML is stripped, never escaped into the
record.
"""

from __future__ import annotations

import unicodedata
from html.parser import HTMLParser
from typing import Any, Mapping

from .errors import (
    BadSplitError,
    BadTaskFamilyError,
    BadTypeError,
    ChosenIndexMissingError,
    ChosenIndexOutOfRangeError,
    DataError,
    DuplicateOptionError,
    EmptyFieldError,
    EmptyGoalError,
    EmptyOptionError,
    EmptyStateError,
    HtmlStripFailedError,
    IdMismatchError,
    MissingFieldError,
    RiskLabelInvalidError,
    ScoreOutOfRangeError,
    ScoresLengthMismatchError,
    TooFewOptionsError,
    TooManyOptionsError,
    UnknownFieldError,
)
from .schema import (
    MAX_OPTIONS,
    MIN_OPTIONS,
    REQUIRED_FIELDS,
    RISK_CATEGORY_COUNT,
    SERIALIZED_FIELDS,
    SPLITS,
    TASK_FAMILIES,
    DecisionExample,
    Option,
    compute_example_id,
)

CHOICE_FAMILIES = frozenset({"choice", "preference"})
# Families whose state may be empty, per the contract note on DecisionExample.state.
STATE_OPTIONAL_FAMILIES = frozenset({"preference"})

_SKIP_TAGS = frozenset({"script", "style"})


class _HTMLTextExtractor(HTMLParser):
    """Collect character data, dropping tags and ``<script>``/``<style>`` bodies.

    ``convert_charrefs=True`` decodes named and numeric character references
    (``&amp;``, ``&#39;``, ...) so entity decoding is part of tag stripping.
    Chunks are joined with a single space — never concatenated without a
    separator — so ``foo<br>bar`` normalizes to ``foo bar`` and adjacent
    markup can never fuse two words into one.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_startendtag(self, tag: str, attrs: Any) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            self._skip_depth -= 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._chunks.append(data)

    def text(self) -> str:
        return " ".join(self._chunks)


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities; returns text without markup."""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(text)
        extractor.close()
    except Exception as exc:  # HTMLParser is tolerant; this is belt and braces.
        raise HtmlStripFailedError(f"could not strip HTML from text: {exc}") from exc
    return extractor.text()


def collapse_whitespace(text: str) -> str:
    """Collapse every whitespace run (incl. NBSP and newlines) to one space."""
    return " ".join(text.split())


def normalize_text(text: str, *, field: str | None = None) -> str:
    """Apply the full normalization pipeline to one text field."""
    if not isinstance(text, str):
        raise BadTypeError(
            f"expected a string, got {type(text).__name__}", field=field
        )
    normalized = unicodedata.normalize("NFC", text)
    without_markup = strip_html(normalized)
    collapsed = collapse_whitespace(without_markup)
    return unicodedata.normalize("NFC", collapsed)


def normalize_example(raw: Mapping[str, Any]) -> DecisionExample:
    """Normalize and validate one raw record; raise the first violation.

    Accepts any mapping shaped like the serialized record (``to_dict`` output,
    adapter output, a parsed fixture). Raises a typed :class:`DataError`
    subclass on the first problem — the code attribute is stable.
    """
    fields, errors = _analyze(raw)
    if errors:
        raise errors[0]
    if fields is None:  # unreachable: no errors ⇒ normalization succeeded
        raise DataError("normalization failed without a recorded error")
    return _build_example(fields, raw)


def validate_example(raw: Mapping[str, Any]) -> list[DataError]:
    """Return every contract violation found in ``raw``, or an empty list.

    Never raises for malformed input; structural problems (options not a
    list, for example) are reported as one error and the dependent checks are
    skipped rather than guessed at.
    """
    _, errors = _analyze(raw)
    return errors


def _analyze(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[DataError]]:
    errors: list[DataError] = []
    if not isinstance(raw, Mapping):
        return None, [
            BadTypeError(
                f"expected a mapping of field name to value, got {type(raw).__name__}"
            )
        ]

    unknown = sorted(set(raw) - set(SERIALIZED_FIELDS))
    if unknown:
        errors.append(
            UnknownFieldError(f"undeclared field(s): {', '.join(unknown)}", field=unknown[0])
        )

    for name in REQUIRED_FIELDS:
        if name not in raw or raw[name] is None:
            errors.append(MissingFieldError(f"required field {name!r} is absent", field=name))

    family = _as_str(raw.get("task_family"), "task_family", errors)
    goal = _text_field(raw, "goal", errors)
    state = _text_field(raw, "state", errors)
    source = _text_field(raw, "source", errors, required_non_empty=True)
    license_ = _text_field(raw, "license", errors, required_non_empty=True)
    group_id = _text_field(raw, "group_id", errors, required_non_empty=True)
    split = _as_str(raw.get("split"), "split", errors)

    if family is not None and family not in TASK_FAMILIES:
        errors.append(
            BadTaskFamilyError(
                f"expected one of {sorted(TASK_FAMILIES)}, got {family!r}",
                field="task_family",
            )
        )
    if split is not None and split not in SPLITS:
        errors.append(
            BadSplitError(
                f"expected one of {sorted(SPLITS)}, got {split!r}", field="split"
            )
        )

    if goal is not None and goal == "":
        errors.append(
            EmptyGoalError("goal is empty after normalization", field="goal")
        )
    if state is not None and state == "":
        allowed = family is not None and family in STATE_OPTIONAL_FAMILIES
        if not allowed:
            errors.append(
                EmptyStateError(
                    "state is empty after normalization; the contract allows this "
                    "only for the preference family",
                    field="state",
                )
            )

    option_texts, option_labels = _options(raw.get("options"), errors)

    if option_texts is not None:
        if len(option_texts) < MIN_OPTIONS:
            errors.append(
                TooFewOptionsError(
                    f"expected at least {MIN_OPTIONS} options, got {len(option_texts)}",
                    field="options",
                )
            )
        if len(option_texts) > MAX_OPTIONS:
            errors.append(
                TooManyOptionsError(
                    f"expected at most {MAX_OPTIONS} options, got {len(option_texts)}",
                    field="options",
                )
            )
        for index, text in enumerate(option_texts):
            if text == "":
                errors.append(
                    EmptyOptionError(
                        f"option at index {index} is empty after normalization",
                        field="options",
                    )
                )
        seen: dict[str, int] = {}
        for index, text in enumerate(option_texts):
            if text in seen:
                errors.append(
                    DuplicateOptionError(
                        f"options at indexes {seen[text]} and {index} are identical "
                        f"after normalization: {text!r}",
                        field="options",
                    )
                )
            else:
                seen[text] = index

    _chosen_index(raw.get("chosen_index", _ABSENT), option_texts, family, errors)
    _scores(raw.get("scores", _ABSENT), option_texts, errors)
    _answerable(raw.get("answerable", _ABSENT), errors)
    _risk_labels(raw.get("risk_labels", _ABSENT), errors)
    _example_id(raw.get("example_id", _ABSENT), family, goal, state, option_texts, errors)

    if errors:
        return None, errors
    if option_texts is None or option_labels is None:
        # option structural error was already recorded; be defensive anyway
        return None, [
            DataError("option normalization produced no result", field="options")
        ]
    return {
        "task_family": family,
        "goal": goal,
        "state": state,
        "source": source,
        "license": license_,
        "group_id": group_id,
        "split": split,
        "option_texts": option_texts,
        "option_labels": option_labels,
    }, []


class _Absent:
    pass


_ABSENT = _Absent()


def _as_str(value: Any, field_name: str, errors: list[DataError]) -> str | None:
    if value is None:
        return None  # missing-field error already recorded by the caller
    if not isinstance(value, str):
        errors.append(
            BadTypeError(
                f"field {field_name!r} must be a string, got {type(value).__name__}",
                field=field_name,
            )
        )
        return None
    return value.strip()


def _text_field(
    raw: Mapping[str, Any],
    name: str,
    errors: list[DataError],
    *,
    required_non_empty: bool = False,
) -> str | None:
    value = raw.get(name)
    if value is None:
        return None  # missing-field error already recorded
    if not isinstance(value, str):
        errors.append(
            BadTypeError(
                f"field {name!r} must be a string, got {type(value).__name__}",
                field=name,
            )
        )
        return None
    try:
        normalized = normalize_text(value, field=name)
    except DataError as exc:
        errors.append(exc)
        return None
    if required_non_empty and normalized == "":
        errors.append(
            EmptyFieldError(f"field {name!r} is empty after normalization", field=name)
        )
    return normalized


def _options(
    value: Any, errors: list[DataError]
) -> tuple[list[str] | None, list[str | None] | None]:
    if value is None:
        return None, None  # missing-field error already recorded
    if not isinstance(value, (list, tuple)) or isinstance(value, str):
        errors.append(
            BadTypeError(
                f"field 'options' must be a list, got {type(value).__name__}",
                field="options",
            )
        )
        return None, None
    texts: list[str] = []
    labels: list[str | None] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            errors.append(
                BadTypeError(
                    f"option at index {index} must be an object, "
                    f"got {type(item).__name__}",
                    field="options",
                )
            )
            continue
        extra = sorted(set(item) - {"text", "label"})
        if extra:
            errors.append(
                UnknownFieldError(
                    f"option at index {index} declares undeclared field(s): "
                    f"{', '.join(extra)}",
                    field="options",
                )
            )
        text_value = item.get("text")
        if text_value is None:
            errors.append(
                MissingFieldError(f"option at index {index} has no 'text'", field="options")
            )
            continue
        try:
            texts.append(normalize_text(text_value, field="options"))
        except DataError as exc:
            errors.append(exc)
            continue
        label_value = item.get("label")
        if label_value is None:
            labels.append(None)
        elif not isinstance(label_value, str):
            errors.append(
                BadTypeError(
                    f"option label at index {index} must be a string, "
                    f"got {type(label_value).__name__}",
                    field="options",
                )
            )
            labels.append(None)
        else:
            label = normalize_text(label_value, field="options")
            labels.append(label if label else None)  # empty label ≡ not provided
    if len(texts) != len(labels):
        # A structural error above skipped an entry; text checks stay valid.
        return texts, None
    return texts, labels


def _chosen_index(
    value: Any,
    option_texts: list[str] | None,
    family: str | None,
    errors: list[DataError],
) -> None:
    if value is _ABSENT:
        if family is not None and family in CHOICE_FAMILIES:
            errors.append(
                ChosenIndexMissingError(
                    f"chosen_index is required for the {family!r} family",
                    field="chosen_index",
                )
            )
        return
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(
            BadTypeError(
                f"chosen_index must be an integer, got {type(value).__name__}",
                field="chosen_index",
            )
        )
        return
    if option_texts is not None and not 0 <= value < len(option_texts):
        errors.append(
            ChosenIndexOutOfRangeError(
                f"chosen_index {value} is outside 0..{len(option_texts) - 1}",
                field="chosen_index",
            )
        )


def _scores(
    value: Any, option_texts: list[str] | None, errors: list[DataError]
) -> None:
    if value is _ABSENT:
        return
    if not isinstance(value, (list, tuple)) or isinstance(value, str):
        errors.append(
            BadTypeError(
                f"scores must be a list of numbers, got {type(value).__name__}",
                field="scores",
            )
        )
        return
    if option_texts is not None and len(value) != len(option_texts):
        errors.append(
            ScoresLengthMismatchError(
                f"scores has {len(value)} entries but the example has "
                f"{len(option_texts)} options",
                field="scores",
            )
        )
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            errors.append(
                BadTypeError(
                    f"score at index {index} must be a number, "
                    f"got {type(item).__name__}",
                    field="scores",
                )
            )
            continue
        if not 0.0 <= float(item) <= 1.0:
            errors.append(
                ScoreOutOfRangeError(
                    f"score at index {index} is {item}; expected 0..1", field="scores"
                )
            )


def _answerable(value: Any, errors: list[DataError]) -> None:
    if value is _ABSENT:
        return
    if not isinstance(value, bool):
        errors.append(
            BadTypeError(
                f"answerable must be a boolean, got {type(value).__name__}",
                field="answerable",
            )
        )


def _risk_labels(value: Any, errors: list[DataError]) -> None:
    if value is _ABSENT:
        return
    if not isinstance(value, (list, tuple)) or isinstance(value, str):
        errors.append(
            BadTypeError(
                f"risk_labels must be a list of strings, got {type(value).__name__}",
                field="risk_labels",
            )
        )
        return
    if len(value) > RISK_CATEGORY_COUNT:
        errors.append(
            RiskLabelInvalidError(
                f"at most {RISK_CATEGORY_COUNT} risk labels are declared, "
                f"got {len(value)}",
                field="risk_labels",
            )
        )
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(
                RiskLabelInvalidError(
                    f"risk label at index {index} must be a non-empty string",
                    field="risk_labels",
                )
            )
            continue
        label = unicodedata.normalize("NFC", item.strip())
        # Category names are case-insensitive identifiers: "Violence" and
        # "violence" name the same category and must not both occupy slots.
        key = label.casefold()
        if key in seen:
            errors.append(
                RiskLabelInvalidError(
                    f"risk label {label!r} appears more than once", field="risk_labels"
                )
            )
        seen.add(key)


def _example_id(
    value: Any,
    family: str | None,
    goal: str | None,
    state: str | None,
    option_texts: list[str] | None,
    errors: list[DataError],
) -> None:
    if value is _ABSENT:
        return
    if not isinstance(value, str):
        errors.append(
            BadTypeError(
                f"example_id must be a string, got {type(value).__name__}",
                field="example_id",
            )
        )
        return
    if family is None or goal is None or state is None or option_texts is None:
        return  # the fields needed to verify the id failed validation already
    expected = compute_example_id(family, goal, state, option_texts)
    if value != expected:
        errors.append(
            IdMismatchError(
                f"example_id {value!r} does not match the recomputed content "
                f"hash {expected!r}",
                field="example_id",
            )
        )


def _build_example(
    fields: dict[str, Any], raw: Mapping[str, Any]
) -> DecisionExample:
    family = fields["task_family"]
    texts: list[str] = fields["option_texts"]
    labels: list[str | None] = fields["option_labels"]
    options = tuple(Option(text=text, label=label) for text, label in zip(texts, labels))
    example_id = compute_example_id(family, fields["goal"], fields["state"], texts)

    chosen = raw.get("chosen_index", None)
    scores_raw = raw.get("scores", None)
    scores = (
        tuple(float(s) for s in scores_raw) if scores_raw is not None else None
    )
    answerable = raw.get("answerable", None)
    risk_labels_raw = raw.get("risk_labels", None)
    risk_labels = (
        tuple(unicodedata.normalize("NFC", s.strip()) for s in risk_labels_raw)
        if risk_labels_raw is not None
        else None
    )

    return DecisionExample(
        example_id=example_id,
        task_family=family,
        goal=fields["goal"],
        state=fields["state"],
        options=options,
        source=fields["source"],
        license=fields["license"],
        group_id=fields["group_id"],
        split=fields["split"],
        chosen_index=chosen,
        scores=scores,
        answerable=answerable,
        risk_labels=risk_labels,
    )
