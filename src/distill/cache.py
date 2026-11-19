"""Content-addressed soft-label cache for the local teacher (FR-DISTILL-01).

Layout (relative to ``options.cache_dir``, default ``cache/teacher``)::

    <teacher_id>/<example_id>.json   one float-array payload per example
    <teacher_id>/index.json          index of known-good entries

``<teacher_id>`` is filesystem-sanitized (``[A-Za-z0-9._-]`` kept, everything
else replaced by ``_``) so ids such as ``Qwen/Qwen3-1.7B`` cannot escape the
directory; the raw id is stored inside every file for fidelity.
``template_hash`` is the sha256 hex digest of the template string. The
cache key is the triple ``(teacher_id, template_hash, example_id)`` — a
rerun with an identical triple performs **zero teacher forwards**.

Corruption handling: every payload carries ``payload_hash`` (sha256 of the
canonical JSON of the payload fields). A file whose payload hash does not
match — corrupt bytes, a hand-edited value, a truncated write — is treated
as a **miss** and recomputed exactly once; the recomputed entry (with its
new hash) is written back and is never silently trusted. A file whose
recorded ``teacher_id`` / ``template_hash`` / ``example_id`` does not match
the request is likewise a miss.

The teacher's forward passes are counted by ``teacher_forwards`` on the
returned :class:`SoftLabelCacheResult`; the counter is incremented only on
actual teacher calls (never on a cache hit), so tests assert it is ``0`` on
a second invocation and that ``cache_hit_rate == 1.0``.

Writes are atomic (write to a sibling ``*.tmp`` file, fsync, ``os.replace``),
so a crashed run never leaves a half-written entry that a later run would
silently trust.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..data.core import DecisionExample
from .entropy import ENTROPY_EXCLUSION_PERCENTILE, entropy_exclusion_mask, entropy_from_logprobs
from .errors import DistillInputError, TeacherUnavailableError
from .teacher import DEFAULT_TEMPLATE, Teacher, load_local_teacher, render_prompt

CACHE_VERSION = 1
INDEX_VERSION = 1
DEFAULT_CACHE_ROOT = "cache/teacher"
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def template_hash(template: str) -> str:
    """sha256 hex digest of the template string (UTF-8)."""
    if not isinstance(template, str) or not template:
        raise DistillInputError("template must be a non-empty string", field="template")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def sanitize_teacher_id(teacher_id: str) -> str:
    """Filesystem-safe directory name for a teacher id.

    Keeps ``[A-Za-z0-9._-]``; replaces every other run of characters with a
    single ``_``. The raw id is preserved inside each cache file.
    """
    if not isinstance(teacher_id, str) or not teacher_id.strip():
        raise DistillInputError("teacher_id must be a non-empty string", field="teacher_id")
    sanitized = _SANITIZE_RE.sub("_", teacher_id.strip()).strip("._")
    return sanitized or "teacher"


@dataclass(frozen=True)
class DistillOptions:
    """Options for :func:`build_soft_label_cache`.

    ``teacher`` is the injected teacher callable (tests pass a deterministic
    stub). When ``None``, the live local-LM path is constructed via
    :func:`src.distill.teacher.load_local_teacher` — that path performs at
    most a one-time download and then runs offline, and is never exercised
    by the unit tests.
    """

    cache_dir: str | Path = DEFAULT_CACHE_ROOT
    teacher: Teacher | None = None
    device: str = "cpu"
    entropy_percentile: float = ENTROPY_EXCLUSION_PERCENTILE


@dataclass(frozen=True)
class SoftLabelEntry:
    """One example's teacher distribution as returned by the cache."""

    example_id: str
    option_count: int
    logprobs: tuple[float, ...]
    entropy: float
    exclude_from_distillation: bool
    cache_hit: bool
    recomputed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "option_count": self.option_count,
            "logprobs": list(self.logprobs),
            "entropy": self.entropy,
            "exclude_from_distillation": self.exclude_from_distillation,
            "cache_hit": self.cache_hit,
            "recomputed": self.recomputed,
        }


@dataclass(frozen=True)
class SoftLabelCacheResult:
    """Summary of one :func:`build_soft_label_cache` invocation.

    ``teacher_forwards`` counts actual teacher calls only — it is ``0`` when
    every request was served from the cache. ``hits`` are positions served
    from the on-disk cache; ``misses`` is ``total - hits`` (positions that
    needed the teacher, including duplicates that reused a same-batch
    forward). ``cache_hit_rate`` is ``hits / total`` (``1.0`` for an empty
    batch).
    """

    teacher_id: str
    template_hash: str
    cache_dir: str
    total: int
    hits: int
    misses: int
    recomputed: int
    teacher_forwards: int
    labels: Mapping[str, SoftLabelEntry] = field(default_factory=dict)

    @property
    def cache_hit_rate(self) -> float:
        return self.hits / self.total if self.total else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "teacher_id": self.teacher_id,
            "template_hash": self.template_hash,
            "cache_dir": self.cache_dir,
            "total": self.total,
            "hits": self.hits,
            "misses": self.misses,
            "recomputed": self.recomputed,
            "teacher_forwards": self.teacher_forwards,
            "cache_hit_rate": self.cache_hit_rate,
            "labels": {key: entry.to_dict() for key, entry in self.labels.items()},
        }


def build_soft_label_cache(
    examples: Sequence[DecisionExample],
    teacher_id: str,
    template: str = DEFAULT_TEMPLATE,
    options: DistillOptions | None = None,
) -> SoftLabelCacheResult:
    """Compute (or read) the teacher's option distribution per example.

    For every example, the cache is consulted under the triple
    ``(teacher_id, template_hash, example_id)``. A valid hit performs no
    teacher forward; a miss or a corrupt entry triggers exactly one teacher
    call whose result is written back with a fresh payload hash.

    The returned entries carry ``exclude_from_distillation=True`` when their
    teacher entropy is strictly above the ``entropy_percentile``-th percentile
    (default: the documented 95.0) of the *requested batch*.
    """
    options = options or DistillOptions()
    if not isinstance(teacher_id, str) or not teacher_id.strip():
        raise DistillInputError("teacher_id must be a non-empty string", field="teacher_id")
    teacher_id = teacher_id.strip()
    thash = template_hash(template)
    example_list = list(examples)
    for position, example in enumerate(example_list):
        _check_example(example, position)

    teacher_dir = Path(options.cache_dir) / sanitize_teacher_id(teacher_id)
    teacher_dir.mkdir(parents=True, exist_ok=True)
    cache_index = _load_index(teacher_dir, teacher_id)

    # --- first pass: serve from cache or schedule (re)computation -----------
    # position -> (logprobs, entropy, recomputed)
    resolved: dict[int, tuple[list[float], float, bool]] = {}
    cache_hit_positions: set[int] = set()
    pending: list[int] = []
    inflight: dict[str, int] = {}  # example_id -> position that owns the forward
    recomputed = 0
    for position, example in enumerate(example_list):
        entry_path = teacher_dir / f"{example.example_id}.json"
        cached = _read_entry(entry_path,
                             teacher_id=teacher_id, template_hash=thash,
                             example_id=example.example_id,
                             option_count=len(example.options))
        if cached is not None:
            logprobs, entropy = cached
            resolved[position] = (logprobs, entropy, False)
            cache_hit_positions.add(position)
            continue
        corrupt = entry_path.exists()  # exists but failed validation: corrupt
        owner = inflight.get(example.example_id)
        if owner is None:
            inflight[example.example_id] = position
            pending.append(position)
            if corrupt:
                recomputed += 1  # corrupt entry: a miss, recomputed exactly once
        # else: a duplicate of an example already pending in this batch;
        # it will reuse that forward's result below (no extra call).

    # --- second pass: teacher forwards for the unique misses -----------------
    teacher_forwards = 0
    teacher_callable = _resolve_teacher(options, teacher_id) if pending else None
    for position in pending:
        example = example_list[position]
        prompt = render_prompt(template, example)
        assert teacher_callable is not None
        try:
            raw = teacher_callable(example, prompt)
        except Exception as exc:
            raise TeacherUnavailableError(
                f"teacher {teacher_id!r} failed on example {example.example_id!r}: {exc}"
            ) from exc
        teacher_forwards += 1
        logprobs = _coerce_logprobs(raw, len(example.options), example.example_id)
        entropy = entropy_from_logprobs(logprobs)
        payload_hash = _write_entry(teacher_dir / f"{example.example_id}.json",
                                    teacher_id=teacher_id, template_hash=thash,
                                    example_id=example.example_id, logprobs=logprobs,
                                    entropy=entropy)
        cache_index["entries"][example.example_id] = {
            "template_hash": thash,
            "option_count": len(example.options),
            "entropy": entropy,
            "file": f"{example.example_id}.json",
            "payload_hash": payload_hash,
        }
        resolved[position] = (logprobs, entropy, True)

    # duplicates of a recomputed example reuse that forward (no extra call);
    # they are counted as misses (the example needed the teacher), but only
    # one forward was actually performed for the whole batch.
    for position, example in enumerate(example_list):
        if position in resolved:
            continue
        owner = inflight[example.example_id]
        resolved[position] = resolved[owner]

    if pending:
        _write_index(teacher_dir / "index.json", cache_index)

    # --- batch entropy filter (95th percentile, documented constant) ---------
    ordered = [resolved[i] for i in range(len(example_list))]
    entropies = [item[1] for item in ordered]
    excluded = entropy_exclusion_mask(entropies, options.entropy_percentile)

    hits = len(cache_hit_positions)
    labels: dict[str, SoftLabelEntry] = {}
    for position, example in enumerate(example_list):
        logprobs, entropy, was_recomputed = resolved[position]
        labels[example.example_id] = SoftLabelEntry(
            example_id=example.example_id,
            option_count=len(example.options),
            logprobs=tuple(logprobs),
            entropy=entropy,
            exclude_from_distillation=bool(excluded[position]),
            cache_hit=position in cache_hit_positions,
            recomputed=was_recomputed,
        )
    return SoftLabelCacheResult(
        teacher_id=teacher_id,
        template_hash=thash,
        cache_dir=str(teacher_dir),
        total=len(example_list),
        hits=hits,
        misses=len(example_list) - hits,
        recomputed=recomputed,
        teacher_forwards=teacher_forwards,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _check_example(example: object, index: int) -> None:
    example_id = getattr(example, "example_id", None)
    options = getattr(example, "options", None)
    if not isinstance(example_id, str) or not example_id:
        raise DistillInputError(f"examples[{index}] must carry a non-empty example_id")
    if options is None or len(options) < 2:  # type: ignore[arg-type]
        raise DistillInputError(
            f"examples[{index}] must carry at least two options (id {example_id!r})"
        )


def _resolve_teacher(options: DistillOptions, teacher_id: str) -> Teacher:
    if options.teacher is not None:
        return options.teacher
    return load_local_teacher(teacher_id, device=options.device)


def _coerce_logprobs(raw: Sequence[float], option_count: int, example_id: str) -> list[float]:
    if isinstance(raw, (str, bytes)):
        raise TeacherUnavailableError(
            f"teacher returned a string instead of logprobs for {example_id!r}"
        )
    try:
        values = [float(v) for v in raw]
    except (TypeError, ValueError) as exc:
        raise TeacherUnavailableError(
            f"teacher returned non-numeric logprobs for {example_id!r}: {exc}"
        ) from exc
    if len(values) != option_count:
        raise TeacherUnavailableError(
            f"teacher returned {len(values)} logprobs for {example_id!r}, "
            f"expected {option_count}"
        )
    if not all(math.isfinite(v) for v in values):
        raise TeacherUnavailableError(f"teacher returned non-finite logprobs for {example_id!r}")
    return values


_PAYLOAD_FIELDS = (
    "version",
    "teacher_id",
    "template_hash",
    "example_id",
    "option_count",
    "logprobs",
    "entropy",
    "created_unix",
)


def _payload_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {key: payload[key] for key in _PAYLOAD_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_entry(
    path: Path,
    *,
    teacher_id: str,
    template_hash: str,
    example_id: str,
    option_count: int,
) -> tuple[list[float], float] | None:
    """Return ``(logprobs, entropy)`` for a *valid* entry, else ``None``.

    Any structural problem — unreadable file, bad JSON, wrong key triple,
    non-finite values, or a payload-hash mismatch — is a miss, never an
    error and never silently trusted.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != CACHE_VERSION:
        return None
    if payload.get("teacher_id") != teacher_id.strip():
        return None
    if payload.get("template_hash") != template_hash:
        return None
    if payload.get("example_id") != example_id:
        return None
    if payload.get("option_count") != option_count:
        return None
    logprobs = payload.get("logprobs")
    entropy = payload.get("entropy")
    if not isinstance(logprobs, list) or len(logprobs) != option_count:
        return None
    if isinstance(entropy, bool) or not isinstance(entropy, (int, float)):
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in logprobs):
        return None
    values = [float(v) for v in logprobs]
    entropy_value = float(entropy)
    if not all(math.isfinite(v) for v in values) or not math.isfinite(entropy_value):
        return None
    expected = payload.get("payload_hash")
    if not isinstance(expected, str) or expected != _payload_hash(payload):
        return None
    return values, entropy_value


def _write_entry(
    path: Path,
    *,
    teacher_id: str,
    template_hash: str,
    example_id: str,
    logprobs: Sequence[float],
    entropy: float,
) -> str:
    payload: dict[str, Any] = {
        "version": CACHE_VERSION,
        "teacher_id": teacher_id.strip(),
        "template_hash": template_hash,
        "example_id": example_id,
        "option_count": len(logprobs),
        "logprobs": [float(v) for v in logprobs],
        "entropy": float(entropy),
        "created_unix": time.time(),
    }
    payload["payload_hash"] = _payload_hash(payload)
    _atomic_write_json(path, payload)
    return str(payload["payload_hash"])


def _load_index(teacher_dir: Path, teacher_id: str) -> dict[str, Any]:
    path = teacher_dir / "index.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        if (
            isinstance(index, dict)
            and index.get("version") == INDEX_VERSION
            and index.get("teacher_id") == teacher_id.strip()
            and isinstance(index.get("entries"), dict)
        ):
            return index
    except (OSError, json.JSONDecodeError):
        pass
    return {"version": INDEX_VERSION, "teacher_id": teacher_id.strip(), "entries": {}}


def _write_index(path: Path, index: Mapping[str, Any]) -> None:
    _atomic_write_json(path, dict(index))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
