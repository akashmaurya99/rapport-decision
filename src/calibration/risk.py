"""Per-category risk thresholds fitted on the dev split (FR-CALIB-03).

For each of the eight declared risk categories, the flagging threshold is
grid-searched on **dev** rows to *maximize recall subject to a false-positive
rate of at most ``max_fpr`` (default 0.10) on the dev safe rows*. The dev safe
rows are the Aegis safe-labeled prompts (explicitly empty ``risk_labels``);
pass ``safe_mask=...`` to select them, or leave it ``None`` to derive safe
rows as the all-zero-target dev rows.

**The XSTest safe set is never used for fitting.** It is the Y8 measurement
set: over-refusal is measured on it, never fitted on it. This module accepts
no dataset handle and never reads files; it is the caller's responsibility to
pass dev (and only dev) rows. ``CalibrationMap.xstest_used_for_fitting`` is
hardcoded ``False`` as an audit flag.

Fitting rule per category: at most ``floor(max_fpr · n_safe)`` safe rows may be
flagged (the same convention as the metrics module's ``recall_at_fpr``). The
threshold is chosen to maximize recall over the candidate grid
``unique(scores) ∪ {max(scores) + 1}``; ties in recall keep the **smaller**
threshold (more of the budget is used for the same recall — the flags already
fit under the budget by construction). Categories infeasible at every useful
threshold fall back to ``max(scores) + 1`` (flag nothing, recall 0.0), which
is always feasible. Categories with no positive rows cannot have their recall
measured; their threshold is set to ``None`` and ``recall`` to ``None``.
Categories with no safe rows face a vacuous FPR constraint; the threshold is
set to ``min(scores)`` (flag everything, recall 1.0, ``fpr = None``).

Results are plain JSON-serializable dicts, ready to be frozen into the
calibration map via ``CalibrationMap(risk_thresholds=...)`` before the test
split is read.
"""

from __future__ import annotations

import math
from typing import Sequence

from .errors import CalibrationInputError

#: The eight declared RAPPORT risk-head labels (FR-MODEL-03), slot order.
RISK_CATEGORIES: tuple[str, ...] = (
    "violence",
    "self_harm",
    "sexual_content",
    "hate_speech",
    "harassment",
    "dangerous_content",
    "illegal_activity",
    "privacy_violation",
)

RISK_CATEGORY_COUNT = len(RISK_CATEGORIES)

#: Default false-positive-rate budget on the dev safe rows.
DEFAULT_MAX_FPR = 0.10


def _validate_scores_targets(
    scores: Sequence[Sequence[float]],
    targets: Sequence[Sequence[int | float]],
    n_categories: int,
) -> tuple[list[list[float]], list[list[int]]]:
    if scores is None or targets is None:
        raise CalibrationInputError("scores and targets must be sequences")
    if len(scores) != len(targets):
        raise CalibrationInputError(
            f"scores and targets must have equal length, got {len(scores)} and {len(targets)}"
        )
    if len(scores) == 0:
        raise CalibrationInputError("expected at least one dev row")
    s_out: list[list[float]] = []
    t_out: list[list[int]] = []
    for i, (s_row, t_row) in enumerate(zip(scores, targets, strict=True)):
        if s_row is None or t_row is None:
            raise CalibrationInputError(f"row {i}: scores and targets must be sequences")
        if len(s_row) != n_categories:
            raise CalibrationInputError(
                f"row {i}: expected {n_categories} category scores, got {len(s_row)}"
            )
        if len(t_row) != n_categories:
            raise CalibrationInputError(
                f"row {i}: expected {n_categories} category targets, got {len(t_row)}"
            )
        s_vals: list[float] = []
        for j, v in enumerate(s_row):
            try:
                f = float(v)
            except (TypeError, ValueError) as exc:
                raise CalibrationInputError(f"row {i} category {j}: not a float: {v!r}") from exc
            if not math.isfinite(f):
                raise CalibrationInputError(f"row {i} category {j}: not finite: {v!r}")
            s_vals.append(f)
        t_vals: list[int] = []
        for j, v in enumerate(t_row):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise CalibrationInputError(
                    f"row {i} category {j}: target must be 0 or 1, got {v!r}"
                )
            f = float(v)
            if f != 0.0 and f != 1.0:
                raise CalibrationInputError(
                    f"row {i} category {j}: target must be 0 or 1, got {v!r}"
                )
            t_vals.append(int(f))
        s_out.append(s_vals)
        t_out.append(t_vals)
    return s_out, t_out


def _validate_safe_mask(
    safe_mask: Sequence[bool] | None,
    targets: list[list[int]],
) -> list[bool]:
    n = len(targets)
    if safe_mask is None:
        # Safe rows are the all-zero-target dev rows (Aegis safe-labeled
        # prompts carry an explicitly empty risk_labels list).
        return [all(t == 0 for t in row) for row in targets]
    if len(safe_mask) != n:
        raise CalibrationInputError(
            f"safe_mask must have one entry per row, got {len(safe_mask)} for {n} rows"
        )
    mask: list[bool] = []
    for i, v in enumerate(safe_mask):
        if not isinstance(v, bool):
            raise CalibrationInputError(f"safe_mask[{i}] must be a bool, got {v!r}")
        mask.append(v)
    return mask


def fit_risk_thresholds(
    scores: Sequence[Sequence[float]],
    targets: Sequence[Sequence[int | float]],
    *,
    max_fpr: float = DEFAULT_MAX_FPR,
    categories: Sequence[str] = RISK_CATEGORIES,
    safe_mask: Sequence[bool] | None = None,
) -> dict[str, dict[str, object]]:
    """Fit one flagging threshold per risk category on dev rows.

    ``scores[i][j]`` is the model's risk score for category ``j`` of dev row
    ``i``; ``targets[i][j]`` is the binary label (1 = category present).
    ``safe_mask[i]`` marks the dev safe rows (Aegis safe-labeled prompts);
    when ``None``, safe rows are the all-zero-target rows. Categories are
    matched to ``categories`` column-wise (default: the eight declared
    categories).

    Returns a JSON-serializable dict keyed by category name. Each value
    carries ``threshold`` (``None`` when the category has no positive dev
    row), ``recall`` (``None`` likewise), ``fpr`` (``None`` when there are no
    safe rows), ``n_positive``, ``n_safe``, ``n_flagged_safe``, the budget
    ``max_fpr``, the feasible ``reason`` and the number of safe rows allowed
    to be flagged (``max_safe_flagged``).
    """
    cats = tuple(categories)
    if not cats:
        raise CalibrationInputError("categories must be non-empty")
    for c in cats:
        if not isinstance(c, str) or not c:
            raise CalibrationInputError(f"categories entries must be non-empty strings, got {c!r}")
    if len(set(cats)) != len(cats):
        raise CalibrationInputError("categories must not contain duplicates")
    if isinstance(max_fpr, bool) or not isinstance(max_fpr, (int, float)):
        raise CalibrationInputError(f"max_fpr must be a real number, got {max_fpr!r}")
    if not math.isfinite(float(max_fpr)) or not (0.0 <= float(max_fpr) <= 1.0):
        raise CalibrationInputError(f"max_fpr must be in [0, 1], got {max_fpr!r}")
    max_fpr_f = float(max_fpr)

    s_rows, t_rows = _validate_scores_targets(scores, targets, len(cats))
    n = len(s_rows)
    safe = _validate_safe_mask(safe_mask, t_rows)

    out: dict[str, dict[str, object]] = {}
    for j, category in enumerate(cats):
        col_scores = [s_rows[i][j] for i in range(n)]
        n_positive = sum(1 for i in range(n) if t_rows[i][j] == 1)
        safe_idx = [i for i in range(n) if safe[i]]
        n_safe = len(safe_idx)
        budget = math.floor(max_fpr_f * n_safe)

        base = {
            "category": category,
            "max_fpr": max_fpr_f,
            "n_positive": n_positive,
            "n_safe": n_safe,
            "max_safe_flagged": budget,
            "xstest_used_for_fitting": False,
        }

        if n_positive == 0:
            # Recall is undefined with no positive rows; flag nothing.
            out[category] = {
                **base,
                "threshold": None,
                "recall": None,
                "fpr": 0.0 if n_safe else None,
                "n_flagged_safe": 0,
                "reason": "no_positive_rows",
            }
            continue

        if n_safe == 0:
            # Vacuous FPR constraint: flag everything.
            out[category] = {
                **base,
                "threshold": min(col_scores),
                "recall": 1.0,
                "fpr": None,
                "n_flagged_safe": 0,
                "reason": "no_safe_rows",
            }
            continue

        pos_idx = [i for i in range(n) if t_rows[i][j] == 1]
        candidates = sorted(set(col_scores)) + [max(col_scores) + 1.0]
        best_recall = -1.0
        best_threshold = candidates[-1]
        best_flagged = 0
        for t in candidates:
            flagged_safe = sum(1 for i in safe_idx if col_scores[i] >= t)
            if flagged_safe > budget:
                continue
            tp = sum(1 for i in pos_idx if col_scores[i] >= t)
            recall = tp / n_positive
            # Maximize recall; ties in recall keep the smaller threshold
            # (the grid is scanned ascending; only strictly better wins).
            if recall > best_recall:
                best_recall = recall
                best_threshold = t
                best_flagged = flagged_safe

        out[category] = {
            **base,
            "threshold": float(best_threshold),
            "recall": float(best_recall),
            "fpr": best_flagged / n_safe,
            "n_flagged_safe": best_flagged,
            "reason": "fitted",
        }
    return out
