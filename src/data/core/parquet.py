"""Optional parquet serialization for DecisionExample.

pyarrow is present in the development venv but is **not** a declared
dependency of this project, so this module imports it lazily and raises
:class:`ParquetUnavailableError` when it is missing — parquet support is
recorded here, not invented elsewhere. Callers that need hard guarantees
should declare pyarrow themselves.

The parquet layout is one row per example: scalar string columns for the
contract's required fields, nullable columns for the optional ones, and a
``list<struct<text:string,label:string>>`` column for the options, matching
the storage note in the architecture contract ("one parquet row; content
hash is the primary key").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import DataError, ParquetError, ParquetUnavailableError
from .normalize import normalize_example
from .schema import DecisionExample
from .serialize import to_dict


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ParquetUnavailableError(
            "pyarrow is not installed; parquet serialization is unavailable "
            "and nothing else in this module can run"
        ) from exc
    return pa, pq


def _schema() -> Any:
    pa, _ = _pyarrow()
    option_type = pa.list_(pa.struct([("text", pa.string()), ("label", pa.string())]))
    return pa.schema(
        [
            ("example_id", pa.string()),
            ("task_family", pa.string()),
            ("goal", pa.string()),
            ("state", pa.string()),
            ("options", option_type),
            ("chosen_index", pa.int64()),
            ("scores", pa.list_(pa.float64())),
            ("answerable", pa.bool_()),
            ("risk_labels", pa.list_(pa.string())),
            ("source", pa.string()),
            ("license", pa.string()),
            ("group_id", pa.string()),
            ("split", pa.string()),
        ]
    )


def write_parquet(examples: list[DecisionExample], path: str | Path) -> None:
    """Write examples to a parquet file; overwrites an existing file."""
    pa, pq = _pyarrow()
    rows = [to_dict(e) for e in examples]
    try:
        table = pa.Table.from_pylist(rows, schema=_schema())
    except Exception as exc:
        raise ParquetError(f"could not build parquet table: {exc}") from exc
    try:
        pq.write_table(table, str(path))
    except Exception as exc:
        raise ParquetError(f"could not write parquet file {path}: {exc}") from exc


def read_parquet(path: str | Path) -> list[DecisionExample]:
    """Read examples from a parquet file; parsing is validation."""
    _, pq = _pyarrow()
    try:
        table = pq.read_table(str(path))
    except Exception as exc:
        raise ParquetError(f"could not read parquet file {path}: {exc}") from exc
    try:
        rows = table.to_pylist()
    except Exception as exc:
        raise ParquetError(f"could not decode parquet rows from {path}: {exc}") from exc
    examples: list[DecisionExample] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ParquetError(f"row {index} is not a mapping")
        # Nulls in optional columns mean "not provided"; drop them so the
        # round-trip records exactly what the source provided.
        cleaned = {key: value for key, value in row.items() if value is not None}
        try:
            examples.append(normalize_example(cleaned))
        except DataError:
            raise
        except Exception as exc:
            raise ParquetError(f"row {index} failed to parse: {exc}") from exc
    return examples
