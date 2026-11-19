"""Ablation arms A1–A8 — adjudication for the single evaluation report (FR-EVAL-04).

One report covers every arm, every seed and every declared metric; no arm is
omitted for a negative result. Arms with no recorded data are rendered
``not-measured`` — never omitted, never fabricated.

Pre-registered rules (``.archgen/rapport/docs/evaluation.md`` §5):

| Arm | Pass condition |
|-----|----------------|
| A1  | bge-small ≥ random init by ≥ 3 points, else the claim is reported unsupported |
| A2  | cosine readout within 2 points of joint cross-attention, else revise the spec |
| A3  | RAPPORT-C within 2 points of RAPPORT-L on the held-out source, else promote RAPPORT-L |
| A4  | no task collapsed below the floor for 2 consecutive epochs (``WEIGHT_COLLAPSE``) |
| A5  | full fine-tune ≥ head-only by ≥ 3 points, else revisit initialization |
| A6  | each ladder stage's gain over its predecessor ≥ 1 point, or the stage is dropped |
| A7  | curriculum ≥ uniform ordering by ≥ 1 point |
| A8  | max 2-point choice-accuracy drop at 1024 tokens (reuses ``length_extension_verdict``) |

Claim-printing rule (pre-registered, applies everywhere a gap is printed):
a claim ("arm A beats arm B") is printed only when ``mean(A − B)`` exceeds one
seed standard deviation of the per-seed differences; otherwise the report
renders ``not distinguishable at n=3 seeds``.

Deterministic: no wall-clock, no network, no training. All inputs are recorded
runs/evidence dicts. Seed standard deviations use the sample estimator
(``ddof=1``); with fewer than two paired seeds a comparison is *not* printed as
distinguishable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .robustness import length_extension_verdict

#: Seeds every arm is measured at (pre-registered, docs/evaluation.md §1).
REQUIRED_SEEDS: tuple[int, ...] = (13, 29, 47)

ARM_IDS: tuple[str, ...] = tuple(f"A{i}" for i in range(1, 9))

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_NOT_MEASURED = "not-measured"
_VALID_STATUSES = (STATUS_PASS, STATUS_FAIL, STATUS_NOT_MEASURED)

#: Factor of the seed standard deviation a mean gap must exceed for a claim to
#: be printed (docs/evaluation.md §1 rule 2).
SEED_STD_FACTOR = 1.0

#: Boundary tolerances (points) absorbing binary floating-point noise.
_EPS_POINTS = 1e-9

ARM_QUESTIONS: dict[str, str] = {
    "A1": "does pretraining help at 55k?",
    "A2": "is cosine enough?",
    "A3": "what does capacity buy?",
    "A4": "does uncertainty weighting collapse a task?",
    "A5": "how much does the trunk learn?",
    "A6": "payoff per example",
    "A7": "is the curriculum earning its complexity?",
    "A8": "is RoPE extension safe?",
}


# ---------------------------------------------------------------------------
# seed statistics and paired claims
# ---------------------------------------------------------------------------

def seed_std(values: Sequence[float]) -> float:
    """Sample standard deviation of per-seed values (``ddof=1``); 0.0 below n=2."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def claim_distinguishable(diffs: Sequence[float]) -> bool:
    """True iff ``mean(diffs)`` strictly exceeds ``SEED_STD_FACTOR · std(diffs)``.

    ``diffs`` are the per-seed paired differences ``A − B``. With fewer than two
    paired seeds no claim can be established (conservative: returns False).
    """
    paired = [float(d) for d in diffs]
    if len(paired) < 2:
        return False
    mean = sum(paired) / len(paired)
    return mean > SEED_STD_FACTOR * seed_std(paired) + _EPS_POINTS


def _as_seed_map(values: Any, *, default_seeds: Sequence[int] = REQUIRED_SEEDS) -> dict[int, float]:
    """Normalize per-seed values to ``{seed: float}``.

    Accepts ``{seed: value}`` mappings (int-like keys) or plain sequences,
    which pair positionally with ``default_seeds`` (longer sequences are
    truncated with a note; non-finite values are dropped). Anything else
    yields an empty map — callers render ``not-measured``.
    """
    out: dict[int, float] = {}
    if values is None:
        return out
    if isinstance(values, Mapping):
        for raw_key, raw_value in values.items():
            try:
                seed = int(raw_key)  # accepts "13" as well as 13
            except (TypeError, ValueError):
                continue
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                continue
            number = float(raw_value)
            if math.isfinite(number):
                out[seed] = number
        return out
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        for seed, raw_value in zip(default_seeds, values):
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                continue
            number = float(raw_value)
            if math.isfinite(number):
                out[int(seed)] = number
    return out


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def compare_arms(
    arm_name: str,
    arm_values: Any,
    reference_name: str,
    reference_values: Any,
    *,
    metric: str,
    scale: float = 100.0,
) -> dict[str, Any] | None:
    """Paired per-seed comparison of two arms/conditions.

    Returns a JSON-safe claim dict, or ``None`` when no seed overlaps. The
    claim is marked ``distinguishable`` only when the mean gap exceeds one
    seed standard deviation of the per-seed differences; otherwise the
    ``text`` renders ``not distinguishable at n=3 seeds``.
    """
    arm_map = _as_seed_map(arm_values)
    ref_map = _as_seed_map(reference_values)
    common = sorted(set(arm_map) & set(ref_map))
    if not common:
        return None
    diffs = [arm_map[s] - ref_map[s] for s in common]
    gap = sum(diffs) / len(diffs)
    std = seed_std(diffs)
    distinguishable = claim_distinguishable(diffs)
    if distinguishable:
        text = (
            f"{arm_name} beats {reference_name} by {gap * scale:+.2f} points on {metric} "
            f"(mean gap exceeds one seed std of {std * scale:.2f} points)"
        )
    else:
        text = (
            f"not distinguishable at n={len(common)} seeds: {arm_name} vs {reference_name} "
            f"gap {gap * scale:+.2f} points, one seed std {std * scale:.2f} points"
        )
    return {
        "arm": arm_name,
        "reference": reference_name,
        "metric": metric,
        "seeds": common,
        "n_seeds": len(common),
        "mean_a": sum(arm_map[s] for s in common) / len(common),
        "mean_b": sum(ref_map[s] for s in common) / len(common),
        "mean_gap_points": gap * scale,
        "seed_std_points": std * scale,
        "per_seed_diff_points": [d * scale for d in diffs],
        "distinguishable": distinguishable,
        "text": text,
    }


# ---------------------------------------------------------------------------
# arm results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArmResult:
    """One ablation arm, rendered pass/fail/not-measured (never omitted)."""

    arm: str
    question: str
    status: str
    summary: str
    metric: str
    split: str
    per_seed: dict[int, float] = field(default_factory=dict)
    mean: float | None = None
    std: float | None = None
    n_seeds: int = 0
    comparison: dict[str, Any] | None = None
    missing: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "question": self.question,
            "status": self.status,
            "summary": self.summary,
            "metric": self.metric,
            "split": self.split,
            "per_seed": {str(k): v for k, v in sorted(self.per_seed.items())},
            "mean": self.mean,
            "std": self.std,
            "n_seeds": self.n_seeds,
            "comparison": dict(self.comparison) if self.comparison else None,
            "missing": list(self.missing),
            "notes": list(self.notes),
        }


def _stats(values: Mapping[int, float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    vals = [values[s] for s in sorted(values)]
    mean = sum(vals) / len(vals)
    return mean, seed_std(vals) if len(vals) >= 2 else 0.0


def _comparison_from_entry(
    entry: Mapping[str, Any],
    *,
    arm_id: str,
    arm_values: Any,
    metric: str,
) -> tuple[dict[str, Any] | None, str, tuple[str, ...]]:
    """Extract ``reference`` from an arm entry; returns (claim, reference_name, missing)."""
    reference = entry.get("reference")
    reference_name = str(entry.get("reference_name") or "") or None
    reference_values: Any = None
    if isinstance(reference, Mapping):
        reference_name = reference_name or str(reference.get("name") or "reference")
        reference_values = reference.get("values")
        if reference_values is None:
            reference_values = reference.get("per_seed")
    else:
        if reference is not None:
            reference_values = reference  # bare sequence of per-seed values
        if reference_values is None:
            reference_values = entry.get("reference_values")
        if reference_values is None:
            reference_values = entry.get("reference_per_seed")
        reference_name = reference_name or (
            str(entry.get("reference_name")) if entry.get("reference_name") else "reference"
        )
    if reference_values is None:
        return None, reference_name or "reference", (f"ablations.{arm_id}.reference.values",)
    claim = compare_arms(
        arm_id, arm_values, str(reference_name or "reference"), reference_values, metric=metric
    )
    if claim is None:
        return None, str(reference_name or "reference"), (
            f"ablations.{arm_id}.reference.values (no paired seeds)",
        )
    return claim, str(reference_name or "reference"), ()


def _comparison_arm(
    entry: Mapping[str, Any],
    arm_id: str,
    *,
    margin_points: float,
    rule: str,
    metric: str,
    split: str,
    question: str,
) -> ArmResult:
    raw_values = entry.get("values")
    if raw_values is None:
        raw_values = entry.get("per_seed")
    values = _as_seed_map(raw_values)
    mean, std = _stats(values)
    claim, reference_name, ref_missing = _comparison_from_entry(
        entry, arm_id=arm_id, arm_values=values, metric=metric
    )
    missing: tuple[str, ...] = () if values else (f"ablations.{arm_id}.values",)
    missing = missing + ref_missing
    if missing:
        return ArmResult(
            arm=arm_id,
            question=question,
            status=STATUS_NOT_MEASURED,
            summary=(
                f"insufficient recorded data for arm {arm_id}; rendered not-measured, "
                f"missing: {', '.join(missing)}"
            ),
            metric=metric,
            split=split,
            per_seed=values,
            mean=mean,
            std=std,
            n_seeds=len(values),
            missing=missing,
        )
    assert claim is not None
    gap = float(claim["mean_gap_points"])
    if rule == "ge":
        ok = gap >= margin_points - _EPS_POINTS
    elif rule == "within":
        ok = abs(gap) <= margin_points + _EPS_POINTS
    elif rule == "ge_neg":
        ok = gap >= -margin_points - _EPS_POINTS
    else:  # pragma: no cover - internal invariant
        raise AssertionError(f"unknown rule {rule!r}")
    status = STATUS_PASS if ok else STATUS_FAIL
    summary = (
        f"gap vs {reference_name} = {gap:+.2f} points "
        f"(pass condition: {rule} {margin_points:.1f} points) — {claim['text']}"
    )
    return ArmResult(
        arm=arm_id,
        question=question,
        status=status,
        summary=summary,
        metric=metric,
        split=split,
        per_seed=values,
        mean=mean,
        std=std,
        n_seeds=len(values),
        comparison=claim,
    )


def _adjudicate(arm_id: str, entry: Mapping[str, Any] | None) -> ArmResult:
    question = ARM_QUESTIONS.get(arm_id, "undeclared arm")
    if entry is None:
        return ArmResult(
            arm=arm_id,
            question=question,
            status=STATUS_NOT_MEASURED,
            summary=(
                f"no recorded run for arm {arm_id}; rendered not-measured, never omitted"
            ),
            metric="",
            split="",
            missing=(f"ablations.{arm_id}",),
        )
    if not isinstance(entry, Mapping):
        return ArmResult(
            arm=arm_id,
            question=question,
            status=STATUS_NOT_MEASURED,
            summary=f"malformed ablation entry for arm {arm_id}; rendered not-measured",
            metric="",
            split="",
            missing=(f"ablations.{arm_id}",),
            notes=("entry is not a mapping",),
        )

    explicit_status = entry.get("status")
    if explicit_status is None:
        explicit_status = entry.get("verdict")
    if explicit_status is not None:
        normalized = str(explicit_status).strip().lower()
        raw_values = entry.get("values")
        if raw_values is None:
            raw_values = entry.get("per_seed")
        values = _as_seed_map(raw_values)
        mean, std = _stats(values)
        notes: list[str] = []
        if normalized not in _VALID_STATUSES:
            notes.append(f"unrecognised status {normalized!r}; rendered not-measured")
            normalized = STATUS_NOT_MEASURED
        summary = str(entry.get("summary") or entry.get("note") or f"recorded status: {normalized}")
        return ArmResult(
            arm=arm_id,
            question=str(entry.get("question") or question),
            status=normalized,
            summary=summary,
            metric=str(entry.get("metric") or ""),
            split=str(entry.get("split") or ""),
            per_seed=values,
            mean=mean,
            std=std,
            n_seeds=len(values),
            missing=() if normalized != STATUS_NOT_MEASURED else (f"ablations.{arm_id}",),
            notes=tuple(notes),
        )

    metric = str(entry.get("metric") or "choice_accuracy")
    split = str(entry.get("split") or "test")

    if arm_id == "A4":
        collapse = entry.get("weight_collapse")
        collapse_epochs = _finite_number(entry.get("collapse_epochs"))
        missing: tuple[str, ...] = ()
        if collapse is None and collapse_epochs is None:
            missing = (f"ablations.{arm_id}.weight_collapse",)
        collapsed = bool(collapse) or (collapse_epochs is not None and collapse_epochs >= 2)
        raw_values = entry.get("values")
        if raw_values is None:
            raw_values = entry.get("per_seed")
        values = _as_seed_map(raw_values)
        mean, std = _stats(values)
        if missing:
            status, summary = (
                STATUS_NOT_MEASURED,
                "WEIGHT_COLLAPSE watch has no recorded outcome; rendered not-measured",
            )
        elif collapsed:
            status, summary = (
                STATUS_FAIL,
                "WEIGHT_COLLAPSE observed: a task collapsed below the floor for 2 "
                "consecutive epochs (uncertainty weighting failed the pass condition)",
            )
        else:
            status, summary = (
                STATUS_PASS,
                "no task collapsed below the floor for 2 consecutive epochs",
            )
        return ArmResult(
            arm=arm_id,
            question=question,
            status=status,
            summary=summary,
            metric=metric,
            split=split,
            per_seed=values,
            mean=mean,
            std=std,
            n_seeds=len(values),
            missing=missing,
        )

    if arm_id == "A6":
        gains_raw = entry.get("stage_gains")
        raw_values = entry.get("values")
        if raw_values is None:
            raw_values = entry.get("per_seed")
        values = _as_seed_map(raw_values)
        mean, std = _stats(values)
        gains: list[float] = []
        malformed = False
        if isinstance(gains_raw, Sequence) and not isinstance(gains_raw, (str, bytes)):
            for raw in gains_raw:
                number = _finite_number(raw)
                if number is None:
                    malformed = True
                else:
                    gains.append(number)
        if gains_raw is None or malformed or not gains:
            missing = (f"ablations.{arm_id}.stage_gains",)
            return ArmResult(
                arm=arm_id,
                question=question,
                status=STATUS_NOT_MEASURED,
                summary="ladder stage gains not recorded; rendered not-measured",
                metric=metric,
                split=split,
                per_seed=values,
                mean=mean,
                std=std,
                n_seeds=len(values),
                missing=missing,
                notes=("stage_gains must be a non-empty sequence of finite point gains",)
                if malformed
                else (),
            )
        weakest = min(gains)
        ok = all(g >= 1.0 - _EPS_POINTS for g in gains)
        return ArmResult(
            arm=arm_id,
            question=question,
            status=STATUS_PASS if ok else STATUS_FAIL,
            summary=(
                f"weakest ladder stage gained {weakest:+.2f} points over its predecessor; "
                f"pass condition: every stage ≥ +1.0 point or is dropped"
            ),
            metric=metric,
            split=split,
            per_seed=values,
            mean=mean,
            std=std,
            n_seeds=len(values),
        )

    if arm_id == "A8":
        acc512 = _finite_number(entry.get("acc512"))
        if acc512 is None:
            acc512 = _finite_number(entry.get("acc_512"))
        acc1024 = _finite_number(entry.get("acc1024"))
        if acc1024 is None:
            acc1024 = _finite_number(entry.get("acc_1024"))
        length_block = entry.get("length_extension")
        if isinstance(length_block, Mapping):
            if acc512 is None:
                acc512 = _finite_number(length_block.get("acc512"))
            if acc512 is None:
                acc512 = _finite_number(length_block.get("acc_512"))
            if acc1024 is None:
                acc1024 = _finite_number(length_block.get("acc1024"))
            if acc1024 is None:
                acc1024 = _finite_number(length_block.get("acc_1024"))
        missing = tuple(
            name
            for name, number in (("ablations.A8.acc512", acc512), ("ablations.A8.acc1024", acc1024))
            if number is None
        )
        verdict = length_extension_verdict(acc512, acc1024) if not missing else "not-measured"
        status = {"PASS": STATUS_PASS, "FAIL": STATUS_FAIL}.get(verdict, STATUS_NOT_MEASURED)
        detail = (
            f"choice accuracy 512→1024 tokens: {acc512:.4f} → {acc1024:.4f}"
            if acc512 is not None and acc1024 is not None
            else "length-sweep accuracies not recorded"
        )
        return ArmResult(
            arm=arm_id,
            question=question,
            status=status,
            summary=(
                f"{detail}; drop must not exceed 2 points — verdict {verdict}"
            ),
            metric="choice_accuracy",
            split=split,
            missing=missing,
        )

    comparison_rules = {
        "A1": ("ge", 3.0),
        "A2": ("within", 2.0),
        "A3": ("ge_neg", 2.0),
        "A5": ("ge", 3.0),
        "A7": ("ge", 1.0),
    }
    if arm_id in comparison_rules:
        rule, margin = comparison_rules[arm_id]
        return _comparison_arm(
            entry,
            arm_id,
            margin_points=margin,
            rule=rule,
            metric=metric,
            split=split,
            question=question,
        )

    # Undeclared arm id: render explicitly, never pass silently.
    explicit = str(entry.get("status") or STATUS_NOT_MEASURED).strip().lower()
    status = explicit if explicit in _VALID_STATUSES else STATUS_NOT_MEASURED
    raw_values = entry.get("values")
    if raw_values is None:
        raw_values = entry.get("per_seed")
    values = _as_seed_map(raw_values)
    mean, std = _stats(values)
    return ArmResult(
        arm=arm_id,
        question=question,
        status=status,
        summary=str(entry.get("summary") or f"undeclared arm {arm_id}"),
        metric=metric,
        split=split,
        per_seed=values,
        mean=mean,
        std=std,
        n_seeds=len(values),
        notes=("arm id is not one of A1–A8",),
    )


def normalize_ablation_entries(ablations: Any) -> dict[str, Any]:
    """Normalize ``ablations`` to ``{arm_id: entry}``.

    Accepts a mapping ``{arm: entry}`` or a sequence of entry mappings each
    carrying an ``arm`` (or ``id``/``name``) key. Anything else yields an
    empty mapping.
    """
    out: dict[str, Any] = {}
    if ablations is None:
        return out
    if isinstance(ablations, Mapping):
        for key, value in ablations.items():
            out[str(key).strip().upper()] = value
        return out
    if isinstance(ablations, Sequence) and not isinstance(ablations, (str, bytes)):
        for index, item in enumerate(ablations):
            if not isinstance(item, Mapping):
                continue
            raw_arm = item.get("arm") or item.get("id") or item.get("name")
            if raw_arm is None:
                out[f"A?{index}"] = item
                continue
            out[str(raw_arm).strip().upper()] = item
    return out


def evaluate_arms(ablations: Any) -> list[ArmResult]:
    """Evaluate A1–A8 in order; arms without data render ``not-measured``.

    Undeclared arm ids present in the input are appended after A8 — included,
    never silently dropped.
    """
    entries = normalize_ablation_entries(ablations)
    results: list[ArmResult] = []
    for arm_id in ARM_IDS:
        results.append(_adjudicate(arm_id, entries.get(arm_id)))
    for extra_id in sorted(key for key in entries if key not in ARM_IDS):
        results.append(_adjudicate(extra_id, entries[extra_id]))
    return results


def arm_failures(arms: Sequence[ArmResult]) -> list[str]:
    """Failed arms as negative-result entries (failed arms are always kept)."""
    return [f"arm {arm.arm} FAIL: {arm.summary}" for arm in arms if arm.status == STATUS_FAIL]


def collect_claims(arms: Sequence[ArmResult]) -> list[dict[str, Any]]:
    """Collect the per-arm comparison claim dicts, in arm order."""
    claims: list[dict[str, Any]] = []
    for arm in arms:
        if arm.comparison:
            claim = dict(arm.comparison)
            claim["arm"] = arm.arm
            claims.append(claim)
    return claims
