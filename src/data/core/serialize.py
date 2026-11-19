"""Dictionary and JSON serialization for DecisionExample.

``to_dict`` emits exactly the fields the example provides — absent optional
fields are omitted, never written as null, and ``example_id`` is included
because content hash is the primary key. ``from_dict`` is the parse side of
the round-trip: it delegates to :func:`normalize_example`, so parsing is
validation, and a serialized record can only come back as a hash-consistent
example. Unknown keys are rejected, not dropped.

The JSON serializer is UTF-8, compact by default, and never escapes
non-ASCII (``ensure_ascii=False``); callers writing files should encode
UTF-8 explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import JsonInvalidError
from .normalize import normalize_example
from .schema import DecisionExample


def to_dict(example: DecisionExample) -> dict[str, Any]:
    """Serialize to a plain dict; omitted optionals are simply absent."""
    options: list[dict[str, Any]] = []
    for option in example.options:
        item: dict[str, Any] = {"text": option.text}
        if option.label is not None:
            item["label"] = option.label
        options.append(item)

    data: dict[str, Any] = {
        "example_id": example.example_id,
        "task_family": example.task_family,
        "goal": example.goal,
        "state": example.state,
        "options": options,
        "source": example.source,
        "license": example.license,
        "group_id": example.group_id,
        "split": example.split,
    }
    if example.chosen_index is not None:
        data["chosen_index"] = example.chosen_index
    if example.scores is not None:
        data["scores"] = list(example.scores)
    if example.answerable is not None:
        data["answerable"] = example.answerable
    if example.risk_labels is not None:
        data["risk_labels"] = list(example.risk_labels)
    return data


def from_dict(data: dict[str, Any]) -> DecisionExample:
    """Parse a serialized record; parsing is validation.

    Unknown keys are rejected; a provided ``example_id`` must match the
    recomputed content hash. Raises the same typed errors as
    :func:`normalize_example`.
    """
    return normalize_example(data)


def to_json(example: DecisionExample, *, indent: int | None = None) -> str:
    """Serialize to a JSON string (UTF-8, non-ASCII preserved)."""
    return json.dumps(
        to_dict(example),
        ensure_ascii=False,
        separators=(",", ":") if indent is None else None,
        indent=indent,
    )


def from_json(payload: str) -> DecisionExample:
    """Parse a JSON string into an example; raise a typed error otherwise."""
    if not isinstance(payload, str):
        raise JsonInvalidError(
            f"expected a JSON string, got {type(payload).__name__}"
        )
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise JsonInvalidError(f"payload is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise JsonInvalidError(
            f"expected a JSON object at the top level, got {type(data).__name__}"
        )
    return from_dict(data)


def write_json(example: DecisionExample, path: str | Path, *, indent: int | None = 2) -> None:
    """Write one example as JSON (UTF-8); overwrites an existing file."""
    Path(path).write_text(to_json(example, indent=indent) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> DecisionExample:
    """Read one example from a JSON file; typed errors on any failure."""
    target = Path(path)
    try:
        payload = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise JsonInvalidError(f"could not read {target}: {exc}", field="path") from exc
    return from_json(payload)
