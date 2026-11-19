"""Robustness harness for the RAPPORT decision model (FR-EVAL-03, E7).

This module measures how the decision model behaves under perturbations of a
fixed evaluation corpus. It is a *measurement* harness: it never fails the
build on a bad measurement — pass/fail style verdicts are recorded in the
report (``score_max_deviation`` against ``SCORE_TOLERANCE``, ``length_extension``
against ``LENGTH_EXTENSION_TOLERANCE``) for a downstream report to interpret.

Entry point: :func:`run_robustness`.

Evaluation corpus
-----------------
``split_rows_by_family`` maps ``task_family -> rows``. Every row is a
:class:`~src.data.core.schema.DecisionExample`. Rows tagged ``split == "train"``
are **never evaluated**; they are used only as the source cross-check for the
held-out-source block ("zero training rows"). All remaining rows (typically the
test split) are the evaluation corpus, per family, and every block of the
report groups by family. Families with zero evaluation rows are reported with
``n_rows: 0`` and ``None`` metrics — reported, never fabricated.

Rows without a ``chosen_index`` are scored (their scores feed the permutation
deviation) but are skipped for choice-accuracy/ECE metrics, which are reported
as ``None`` when no scored row exists.

Blocks
------
* ``permutation`` — for every option-count group in a family, re-encodes the
  same example set under each permutation of its option slots. For ``n <= 4``
  all ``n!`` permutations are enumerated; for ``n > 4`` eight permutations are
  sampled deterministically (``random.Random(f"{seed}:{family}:{n}")``).
  Reports per permutation choice accuracy (chosen rows only), ``accuracy_delta``
  (max − min across permutations) and ``score_max_deviation`` — the maximum
  ``|scores_perm[j] − scores_base[perm[j]]|`` over rows, slots and
  permutations (permutation ``(0, …, n-1)`` is the base). FR-MODEL-02 requires
  this deviation to lie within ``1e-4``; the harness records the measurement
  and ``within_tolerance`` — it does not assert it.
* ``option_count_sweep`` — accuracy and bucket ECE for each ``n`` in 2..8.
  Bucket ECE is :func:`src.evaluation.metrics.ece` over ``(confidence,
  correct)`` pairs with the metrics module's fixed bin edges (15). The bucket
  label is ``bucket_fn(n)`` (the option-count bucket); ``bucket_fn`` is
  injected so the harness stays agnostic of any particular bucketing scheme.
* ``ood`` — held-out-source block, computed only on OOD rows whose source
  contributes zero training rows (cross-checked against the train rows in
  ``split_rows_by_family``; the split manifest carries no source list, so a
  fingerprint cross-check is not performed and is noted as such). Rows from a
  source seen in train are excluded and counted. If no eligible rows remain
  (including an empty input), the block is present with ``empty: true`` and
  ``None`` metrics — never fabricated.
* ``context`` — two documented transforms applied to the same example set:

  * ``contradictory`` (``swap_gold_with_plausible_distractor``): the gold
    option's text (and label) is exchanged with its plausible distractor —
    the non-gold option with the highest gold utility (``row.scores``, ties to
    the lowest index; without scores, the first non-gold option in index
    order). The ``chosen_index`` is *not* remapped: the held-out label now
    contradicts the option content, which is the point of the probe.
    Option order (and hence ``example_id``, which is order-invariant) is
    unchanged in content terms; rows without ``chosen_index`` are skipped and
    counted.
  * ``missing_information`` (``drop_decisive_sentence``): the decisive
    sentence is removed from the state. The decisive sentence is the one with
    the greatest word-token overlap with the gold option's text (ties break to
    the earliest sentence; when no sentence shares a word with the gold
    option, the final sentence is removed — documented fallback). The
    ``example_id`` is recomputed after the state changes. Removing the only
    sentence yields an empty state, recorded as ``n_state_emptied``.

  For each transform the report gives, per family and pooled: the abstention
  rate on the clean set and on the transformed set, the change in abstention
  rate (``abstention_rate_delta``), its relative change, and the same trio of
  figures for choice accuracy.
* ``length_sweep`` — the same examples re-encoded at budgets 512, 768 and
  1024 through :func:`src.tokenization.encoding.encode_example`
  (``budget=…``). Reports choice accuracy and bucket ECE per budget (choice
  accuracy over rows with a ``chosen_index``, bucket ECE over the same rows).
  A drop of choice accuracy from 512 to 1024 exceeding
  ``LENGTH_EXTENSION_TOLERANCE`` (2 points) sets ``length_extension`` to
  ``"FAIL"`` — the 1024-token extension claim is a measured extrapolation, not
  an assumption; otherwise ``"PASS"``. When either accuracy is undefined the
  verdict is ``"not-measured"``.

Abstention rule
---------------
Abstention is defined on the model's confidence head output: a row abstains
iff ``confidence < 0.75``. The threshold derives from the declared default
utility triple ``u_correct = 1``, ``u_wrong = -3``, ``u_abstain = 0`` — the
point where answering beats abstaining in expectation — ``(0 - (-3)) / (1 -
(-3)) = 0.75``. (This mirrors the predictor's default abstention policy; the
harness restates the constant instead of importing ``src.calibration`` —
**all metrics come from :mod:`src.evaluation.metrics` only**.)

Determinism
-----------
The report is a pure function of its inputs for a fixed ``seed``: sampled
permutations are seeded per ``(seed, family, n)``, families and option counts
are iterated in sorted order, and no wall clock or hash randomization is
consulted (``PYTHONHASHSEED`` is irrelevant — all ordering is explicit). The
report is JSON-serializable via :meth:`RobustnessReport.to_dict`.

Side effects
------------
``encode_example`` increments a module-global truncation counter. To keep the
harness side-effect free, that counter is snapshotted before the run and
restored afterwards; per-budget truncation counts are reported instead
(``n_truncated`` in each ``length_sweep`` entry). The model is temporarily set
to eval mode during the run and its previous mode is restored.
"""

from __future__ import annotations

import itertools
import math
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from src.data.core.schema import DecisionExample, compute_example_id
from src.data.splits.manifest import SplitManifest
from src.evaluation import metrics
from src.tokenization.encoding import TRUNCATION_COUNTS, encode_example

__all__ = [
    "ABSTENTION_THRESHOLD",
    "DEFAULT_SEED",
    "LENGTH_BUDGETS",
    "LENGTH_EXTENSION_TOLERANCE",
    "PERMUTATION_SAMPLE_SIZE",
    "SCORE_TOLERANCE",
    "RobustnessInputError",
    "RobustnessReport",
    "length_extension_verdict",
    "run_robustness",
]

# Schema order of the six task families; matches the task-conditioning ids
# documented in src/model/head/model.py (ids 0-5, 6-7 reserved).
TASK_FAMILY_ORDER: tuple[str, ...] = (
    "choice",
    "preference",
    "scoring",
    "probability",
    "abstention",
    "risk",
)
_TASK_ID: dict[str, int] = {name: index for index, name in enumerate(TASK_FAMILY_ORDER)}

#: Fixed seed for :func:`run_robustness` (the task's agreed archgen seed).
DEFAULT_SEED = 20261119

#: Token budgets exercised by the length sweep (FR-EVAL-03).
LENGTH_BUDGETS: tuple[int, ...] = (512, 768, 1024)
#: A choice-accuracy drop larger than this from 512 to 1024 tokens fails the
#: 1024 extension claim ("exceeding 2 points").
LENGTH_EXTENSION_TOLERANCE = 0.02
#: Per-score permutation tolerance declared by FR-MODEL-02 / task E7.
SCORE_TOLERANCE = 1e-4
#: All permutations are enumerated for at most this many options; beyond it,
#: ``PERMUTATION_SAMPLE_SIZE`` seeded samples are drawn.
FULL_ENUMERATION_MAX = 4
#: Number of seeded permutations sampled for ``n > FULL_ENUMERATION_MAX``.
PERMUTATION_SAMPLE_SIZE = 8
#: Abstain iff the confidence head emits less than this. Derives from the
#: declared default utility triple (correct=1, wrong=-3, abstain=0).
ABSTENTION_THRESHOLD = 0.75

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_WORD_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|[\r\n]+")


class RobustnessInputError(ValueError):
    """A malformed argument reached :func:`run_robustness`.

    ``field`` names the offending argument; ``code`` is stable so callers can
    match on the failure class without parsing the message.
    """

    code = "ROBUSTNESS_INPUT_INVALID"

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(f"{self.code}: {message}")
        self.message = message
        self.field = field


@dataclass(frozen=True)
class RobustnessReport:
    """JSON-serializable robustness report, grouped by family.

    Blocks (``permutation``, ``option_count_sweep``, ``ood``, ``context``,
    ``length_sweep``) are nested dicts; ``to_dict()`` returns a plain-dict
    view that round-trips through :func:`json.dumps`.
    """

    seed: int
    split_manifest_fingerprint: str
    ood_source: str | None
    rows: dict[str, int]
    permutation: dict[str, Any]
    option_count_sweep: dict[str, Any]
    ood: dict[str, Any]
    context: dict[str, Any]
    length_sweep: dict[str, Any]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict JSON-safe view (tuples become lists)."""
        return {
            "seed": self.seed,
            "split_manifest_fingerprint": self.split_manifest_fingerprint,
            "ood_source": self.ood_source,
            "rows": dict(self.rows),
            "permutation": self.permutation,
            "option_count_sweep": self.option_count_sweep,
            "ood": self.ood,
            "context": self.context,
            "length_sweep": self.length_sweep,
            "notes": list(self.notes),
        }


def length_extension_verdict(acc512: float | None, acc1024: float | None) -> str:
    """Verdict on the 1024-token extension claim.

    ``"FAIL"`` when the 512→1024 choice-accuracy drop exceeds
    :data:`LENGTH_EXTENSION_TOLERANCE` (2 points); ``"PASS"`` otherwise;
    ``"not-measured"`` when either accuracy is undefined.

    The comparison absorbs binary floating-point noise (``0.9 - 0.88`` is
    ``0.020000000000000004``): a drop of exactly 2 points is ``"PASS"``,
    mirroring the repo's 1e-9 boundary convention.
    """
    if acc512 is None or acc1024 is None:
        return "not-measured"
    return "FAIL" if (acc512 - acc1024) > LENGTH_EXTENSION_TOLERANCE + 1e-9 else "PASS"


# ---------------------------------------------------------------------------
# internal scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ScoredRow:
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    confidence: float
    predicted_index: int
    abstain: bool
    truncated: bool


def _clamp01(value: float) -> float:
    if value != value:  # NaN guard (sigmoid output cannot be NaN, defensive only)
        return 0.0
    return min(1.0, max(0.0, value))


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    return tuple(v / total for v in exps)


def _score_row(
    model: torch.nn.Module,
    tokenizer: Any,
    row: DecisionExample,
    *,
    budget: int = LENGTH_BUDGETS[0],
) -> _ScoredRow:
    encoded = encode_example(row, budget=budget, tokenizer=tokenizer)
    n = len(row.options)
    ids = torch.tensor([list(encoded.input_ids)], dtype=torch.long)
    attention = torch.tensor([list(encoded.attention_mask)], dtype=torch.long)
    option_mask = torch.ones((1, n), dtype=torch.long)
    task_ids = torch.tensor([_TASK_ID[row.task_family]], dtype=torch.long)
    with torch.no_grad():
        output = model(ids, attention, option_mask, task_ids)
    scores = tuple(float(v) for v in output.scores[0].tolist())
    logits = [float(v) for v in output.choice_logits[0].tolist()]
    probabilities = _softmax(logits)
    confidence = _clamp01(float(output.confidence.reshape(-1)[0].item()))
    predicted = max(range(n), key=probabilities.__getitem__)
    return _ScoredRow(
        scores=scores,
        probabilities=probabilities,
        confidence=confidence,
        predicted_index=predicted,
        abstain=confidence < ABSTENTION_THRESHOLD,
        truncated=bool(encoded.truncated),
    )


_ScoreCache = dict[tuple[str, tuple[str, ...], int], _ScoredRow]


def _score_cached(
    model: torch.nn.Module,
    tokenizer: Any,
    row: DecisionExample,
    *,
    budget: int,
    cache: _ScoreCache,
) -> _ScoredRow:
    key = (row.example_id, tuple(option.text for option in row.options), budget)
    hit = cache.get(key)
    if hit is None:
        hit = _score_row(model, tokenizer, row, budget=budget)
        cache[key] = hit
    return hit


def _chosen_pairs(
    rows: Sequence[DecisionExample], scored: Sequence[_ScoredRow]
) -> list[tuple[int, _ScoredRow]]:
    pairs: list[tuple[int, _ScoredRow]] = []
    for row, s in zip(rows, scored, strict=True):
        chosen = row.chosen_index
        if chosen is None:
            continue
        pairs.append((int(chosen), s))
    return pairs


def _gold_metrics(
    pairs: Sequence[tuple[int, _ScoredRow]],
    *,
    ece_bins: int = metrics.ECE_BINS,
) -> dict[str, Any]:
    if not pairs:
        return {"accuracy": None, "bucket_ece": None, "ece_bins": ece_bins}
    gold = [c for c, _ in pairs]
    pred = [s.predicted_index for _, s in pairs]
    confidences = [s.confidence for _, s in pairs]
    correct = [int(g == p) for g, p in zip(gold, pred, strict=True)]
    return {
        "accuracy": metrics.accuracy(gold, pred),
        "bucket_ece": metrics.ece(confidences, correct, bins=ece_bins),
        "ece_bins": ece_bins,
    }


def _abstention_rate(scored: Sequence[_ScoredRow]) -> float | None:
    if not scored:
        return None
    return sum(1 for s in scored if s.abstain) / len(scored)


def _relative_change(delta: float, clean: float | None) -> float | None:
    if clean is None or clean == 0.0:
        return None
    return delta / clean


# ---------------------------------------------------------------------------
# input validation / corpus partitioning
# ---------------------------------------------------------------------------


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RobustnessInputError(f"seed must be an integer, got {seed!r}", field="seed")
    if seed < 0 or seed >= 1 << 64:
        raise RobustnessInputError(f"seed must lie in [0, 2^64), got {seed}", field="seed")
    return seed


def _validate_rows(rows: object, *, field: str) -> list[DecisionExample]:
    if not isinstance(rows, Sequence):
        raise RobustnessInputError(f"{field} must be a sequence of DecisionExample rows",
                                   field=field)
    out: list[DecisionExample] = []
    for index, row in enumerate(rows):
        if not isinstance(row, DecisionExample):
            raise RobustnessInputError(
                f"{field}[{index}] is a {type(row).__name__}, expected DecisionExample",
                field=field,
            )
        _validate_row(row, field=f"{field}[{index}]")
        out.append(row)
    return out


def _validate_row(row: DecisionExample, *, field: str) -> None:
    if row.task_family not in _TASK_ID:
        raise RobustnessInputError(
            f"{field}: unknown task_family {row.task_family!r}; "
            f"expected one of {list(TASK_FAMILY_ORDER)}",
            field=field,
        )
    n_options = len(row.options)
    if n_options < 2 or n_options > 8:
        raise RobustnessInputError(
            f"{field}: {n_options} option(s); the model accepts 2..8",
            field=field,
        )
    chosen = row.chosen_index
    if chosen is not None:
        if isinstance(chosen, bool) or not isinstance(chosen, int):
            raise RobustnessInputError(
                f"{field}: chosen_index must be an integer or None, got {chosen!r}",
                field=field,
            )
        if chosen < 0 or chosen >= n_options:
            raise RobustnessInputError(
                f"{field}: chosen_index {chosen} out of range [0, {n_options})",
                field=field,
            )


def _bucket_label(bucket_fn: Callable[[int], str], n: int) -> str:
    try:
        label = bucket_fn(n)
    except RobustnessInputError:
        raise
    except Exception as exc:  # bucket_fn is caller-provided; wrap failures
        raise RobustnessInputError(
            f"bucket_fn raised for n={n}: {exc}", field="bucket_fn"
        ) from exc
    if not isinstance(label, str) or not label:
        raise RobustnessInputError(
            f"bucket_fn must return a non-empty str for n={n}, got {label!r}",
            field="bucket_fn",
        )
    return label


def _resolve_manifest(split_manifest: object) -> tuple[str, str | None]:
    if isinstance(split_manifest, SplitManifest):
        return split_manifest.fingerprint, split_manifest.ood_source
    if isinstance(split_manifest, str):
        if not _SHA256_HEX.fullmatch(split_manifest):
            raise RobustnessInputError(
                "split_manifest as a str must be a lowercase sha256 hex digest",
                field="split_manifest",
            )
        return split_manifest, None
    raise RobustnessInputError(
        "split_manifest must be a SplitManifest or a lowercase sha256 hex "
        f"fingerprint string, got {type(split_manifest).__name__}",
        field="split_manifest",
    )


def _permutations_for(n: int, seed: int, family: str) -> tuple[tuple[tuple[int, ...], ...], str]:
    all_perms = list(itertools.permutations(range(n)))
    if n <= FULL_ENUMERATION_MAX:
        return tuple(all_perms), "full"
    rng = random.Random(f"{seed}:{family}:{n}")
    return tuple(sorted(rng.sample(all_perms, PERMUTATION_SAMPLE_SIZE))), "sampled"


def _permute_row(row: DecisionExample, perm: tuple[int, ...]) -> DecisionExample:
    options = tuple(row.options[i] for i in perm)
    chosen = row.chosen_index
    if chosen is not None:
        chosen = perm.index(chosen)
    # example_id is order-invariant, so the content hash is unchanged.
    return replace(row, options=options, chosen_index=chosen)


def _manifest_rows_by_family(
    split_rows_by_family: object,
) -> tuple[list[str], list[DecisionExample], dict[str, list[DecisionExample]]]:
    if not isinstance(split_rows_by_family, Mapping):
        raise RobustnessInputError(
            "split_rows_by_family must be a mapping of task_family -> rows",
            field="split_rows_by_family",
        )
    families: list[str] = []
    train_rows: list[DecisionExample] = []
    eval_rows: dict[str, list[DecisionExample]] = {}
    for family in sorted(split_rows_by_family):
        if not isinstance(family, str):
            raise RobustnessInputError(
                f"split_rows_by_family keys must be family-name strings, "
                f"got {type(family).__name__}",
                field="split_rows_by_family",
            )
        if family not in _TASK_ID:
            raise RobustnessInputError(
                f"split_rows_by_family key {family!r} is not a known task family "
                f"(expected one of {list(TASK_FAMILY_ORDER)})",
                field="split_rows_by_family",
            )
        rows = _validate_rows(split_rows_by_family[family], field=f"split_rows_by_family[{family}]")
        for row in rows:
            if row.task_family != family:
                raise RobustnessInputError(
                    f"split_rows_by_family[{family!r}] contains a {row.task_family!r} row "
                    f"(example_id={row.example_id[:16]}…)",
                    field="split_rows_by_family",
                )
        families.append(family)
        eval_rows[family] = [row for row in rows if row.split != "train"]
        train_rows.extend(row for row in rows if row.split == "train")
    return families, train_rows, eval_rows


# ---------------------------------------------------------------------------
# transforms (context-robustness probes)
# ---------------------------------------------------------------------------


def _word_tokens(text: str) -> list[str]:
    return _WORD_TOKEN.findall(text.lower())


def _swap_gold_with_plausible_distractor(
    row: DecisionExample,
) -> tuple[DecisionExample, bool] | None:
    """Exchange the gold option's content with its plausible distractor.

    The plausible distractor is the non-gold option with the highest gold
    utility (``row.scores``), ties to the lowest index; without scores, the
    first non-gold option in index order. ``chosen_index`` is deliberately
    *not* remapped — the label now contradicts the option content. Rows
    without a ``chosen_index`` have no gold option to swap; they return
    ``None`` (skipped and counted by the caller). Option texts keep their
    multiset, so the order-invariant ``example_id`` is unchanged. Never
    empties the state, so the second element of the tuple is always
    ``False`` — both transforms share one protocol.
    """
    chosen = row.chosen_index
    if chosen is None:
        return None
    n = len(row.options)
    distractor: int | None = None
    if row.scores is not None and len(row.scores) == n:
        for j in range(n):
            if j == chosen:
                continue
            # Strict greater keeps the lowest index on ties.
            if distractor is None or row.scores[j] > row.scores[distractor]:
                distractor = j
    else:
        for j in range(n):
            if j != chosen:
                distractor = j
                break
    assert distractor is not None  # n >= 2 guarantees a non-gold slot
    options = list(row.options)
    options[chosen], options[distractor] = options[distractor], options[chosen]
    # example_id is order-invariant, so the hash stays valid.
    return replace(row, options=tuple(options)), False


def _split_sentences(state: str) -> list[str]:
    return [piece for piece in _SENTENCE_SPLIT.split(state.strip()) if piece.strip()]


def _gold_text(row: DecisionExample) -> str:
    chosen = row.chosen_index
    if chosen is not None:
        return row.options[chosen].text
    return row.options[0].text  # documented fallback for rows without gold


def _drop_decisive_sentence(row: DecisionExample) -> tuple[DecisionExample, bool]:
    """Remove the state's decisive sentence. Returns (row, state_emptied).

    Decisive = greatest word-token overlap with the gold option's text; ties
    break to the earliest sentence; when no sentence shares a word with the
    gold option, the final sentence is removed (documented fallback). The
    ``example_id`` is recomputed because the state content changed. Removing
    the only sentence yields an empty state (``state_emptied=True``).
    """
    gold_words = set(_word_tokens(_gold_text(row)))
    sentences = _split_sentences(row.state)
    if not sentences:
        new_state = ""
        example_id = compute_example_id(
            row.task_family,
            row.goal,
            new_state,
            [option.text for option in row.options],
        )
        return replace(row, state=new_state, example_id=example_id), True
    overlaps = [len(set(_word_tokens(s)) & gold_words) for s in sentences]
    if all(o == 0 for o in overlaps):
        decisive = len(sentences) - 1
    else:
        # Strict greater keeps the earliest sentence on ties.
        decisive = 0
        for i in range(1, len(overlaps)):
            if overlaps[i] > overlaps[decisive]:
                decisive = i
    remaining = [s for i, s in enumerate(sentences) if i != decisive]
    new_state = " ".join(remaining)
    example_id = compute_example_id(
        row.task_family,
        row.goal,
        new_state,
        [option.text for option in row.options],
    )
    return replace(row, state=new_state, example_id=example_id), not new_state


# ---------------------------------------------------------------------------
# report blocks; all take the evaluated-rows mapping and a shared score cache
# ---------------------------------------------------------------------------


def _score_all(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[DecisionExample],
    *,
    budget: int,
    cache: _ScoreCache,
) -> list[_ScoredRow]:
    return [_score_cached(model, tokenizer, row, budget=budget, cache=cache) for row in rows]


def _permutation_block(
    model: torch.nn.Module,
    tokenizer: Any,
    eval_rows: Mapping[str, list[DecisionExample]],
    families: Sequence[str],
    seed: int,
    cache: _ScoreCache,
) -> dict[str, Any]:
    per_family: dict[str, Any] = {}
    for family in families:
        rows = eval_rows[family]
        by_option_count: dict[str, Any] = {}
        for n in sorted({len(r.options) for r in rows}):
            subgroup = [r for r in rows if len(r.options) == n]
            permutations, mode = _permutations_for(n, seed, family)
            identity = tuple(range(n))
            base = _score_all(
                model, tokenizer, subgroup, budget=LENGTH_BUDGETS[0], cache=cache
            )
            chosen_idx = [i for i, r in enumerate(subgroup) if r.chosen_index is not None]
            accuracies: list[float | None] = []
            payload: list[dict[str, Any]] = []
            deviation = 0.0
            for perm in permutations:
                if perm == identity:
                    scored = base
                else:
                    scored = _score_all(
                        model,
                        tokenizer,
                        [_permute_row(r, perm) for r in subgroup],
                        budget=LENGTH_BUDGETS[0],
                        cache=cache,
                    )
                if chosen_idx:
                    correct = 0
                    for i in chosen_idx:
                        chosen = subgroup[i].chosen_index
                        assert chosen is not None  # guarded by chosen_idx
                        if scored[i].predicted_index == perm.index(chosen):
                            correct += 1
                    accuracy: float | None = correct / len(chosen_idx)
                else:
                    accuracy = None
                accuracies.append(accuracy)
                for i in range(len(subgroup)):
                    slot_scores = scored[i].scores
                    base_scores = base[i].scores
                    for j in range(n):
                        d = abs(slot_scores[j] - base_scores[perm[j]])
                        if d > deviation:
                            deviation = d
                payload.append({"permutation": list(perm), "accuracy": accuracy})
            defined = [a for a in accuracies if a is not None]
            delta = (max(defined) - min(defined)) if defined else None
            by_option_count[str(n)] = {
                "n_options": n,
                "n_rows": len(subgroup),
                "n_scored_rows": len(chosen_idx),
                "enumeration": mode,
                "n_permutations": len(permutations),
                "permutations": payload,
                "accuracy_delta": delta,
                "score_max_deviation": deviation,
                "score_tolerance": SCORE_TOLERANCE,
                "within_tolerance": deviation <= SCORE_TOLERANCE,
            }
        per_family[family] = _permutation_family_entry(rows, by_option_count)
    return {**per_family, "overall": _permutation_overall(per_family)}


def _permutation_family_entry(
    rows: Sequence[DecisionExample],
    by_option_count: Mapping[str, Any],
) -> dict[str, Any]:
    entries = list(by_option_count.values())
    deltas = [e["accuracy_delta"] for e in entries if e["accuracy_delta"] is not None]
    deviations = [
        e["score_max_deviation"] for e in entries if e["score_max_deviation"] is not None
    ]
    withins = [e["within_tolerance"] for e in entries if e["within_tolerance"] is not None]
    return {
        "n_rows": len(rows),
        "n_scored_rows": sum(1 for r in rows if r.chosen_index is not None),
        "by_option_count": dict(by_option_count),
        "accuracy_delta": max(deltas) if deltas else None,
        "score_max_deviation": max(deviations) if deviations else None,
        "score_tolerance": SCORE_TOLERANCE,
        "within_tolerance": all(withins) if withins else None,
    }


def _permutation_overall(per_family: Mapping[str, Any]) -> dict[str, Any]:
    entries = list(per_family.values())
    deltas = [e["accuracy_delta"] for e in entries if e["accuracy_delta"] is not None]
    deviations = [
        e["score_max_deviation"] for e in entries if e["score_max_deviation"] is not None
    ]
    withins = [e["within_tolerance"] for e in entries if e["within_tolerance"] is not None]
    return {
        "families": sorted(per_family),
        "n_rows": sum(e["n_rows"] for e in entries),
        "n_scored_rows": sum(e["n_scored_rows"] for e in entries),
        "accuracy_delta": max(deltas) if deltas else None,
        "score_max_deviation": max(deviations) if deviations else None,
        "score_tolerance": SCORE_TOLERANCE,
        "within_tolerance": all(withins) if withins else None,
    }


def _option_count_sweep(
    model: torch.nn.Module,
    tokenizer: Any,
    eval_rows: Mapping[str, list[DecisionExample]],
    families: Sequence[str],
    bucket_fn: Callable[[int], str],
    cache: _ScoreCache,
) -> dict[str, Any]:
    per_family: dict[str, Any] = {}
    pooled: dict[str, Any] = {}
    for family in families:
        rows = eval_rows[family]
        per_n: dict[str, Any] = {}
        for n in range(2, 9):
            group = [r for r in rows if len(r.options) == n]
            scored = _score_all(model, tokenizer, group, budget=LENGTH_BUDGETS[0], cache=cache)
            pairs = _chosen_pairs(group, scored)
            per_n[str(n)] = {
                "n_options": n,
                "bucket": _bucket_label(bucket_fn, n),
                "n_rows": len(group),
                "n_scored_rows": len(pairs),
                **_gold_metrics(pairs),
            }
        per_family[family] = per_n
    for n in range(2, 9):
        group = [r for family in families for r in eval_rows[family] if len(r.options) == n]
        scored = _score_all(model, tokenizer, group, budget=LENGTH_BUDGETS[0], cache=cache)
        pairs = _chosen_pairs(group, scored)
        pooled[str(n)] = {
            "n_options": n,
            "bucket": _bucket_label(bucket_fn, n),
            "n_rows": len(group),
            "n_scored_rows": len(pairs),
            **_gold_metrics(pairs),
        }
    return {**per_family, "overall": pooled}


def _ood_block(
    model: torch.nn.Module,
    tokenizer: Any,
    ood_rows: Sequence[DecisionExample],
    train_rows: Sequence[DecisionExample],
    *,
    cache: _ScoreCache,
    fingerprint: str,
    ood_source: str | None,
) -> dict[str, Any]:
    train_sources = {r.source for r in train_rows}
    excluded = [r for r in ood_rows if r.source in train_sources]
    eligible = [r for r in ood_rows if r.source not in train_sources]
    base = {
        "ood_source": ood_source,
        "split_manifest_fingerprint": fingerprint,
        "n_excluded": len(excluded),
        "excluded_sources": sorted({r.source for r in excluded}),
        "n_train_sources": len(train_sources),
        "ece_bins": metrics.ECE_BINS,
    }
    if not eligible:
        return {
            **base,
            "empty": True,
            "n_rows": 0,
            "sources": [],
            "by_family": {},
            "accuracy": None,
            "bucket_ece": None,
            "abstention_rate": None,
            "note": (
                "no OOD rows remain after excluding sources present in train; "
                "metrics are None rather than fabricated"
            ),
        }
    by_family: dict[str, Any] = {}
    for family in sorted({r.task_family for r in eligible}):
        group = [r for r in eligible if r.task_family == family]
        scored = _score_all(model, tokenizer, group, budget=LENGTH_BUDGETS[0], cache=cache)
        pairs = _chosen_pairs(group, scored)
        by_family[family] = {
            "n_rows": len(group),
            "n_scored_rows": len(pairs),
            "sources": sorted({r.source for r in group}),
            "abstention_rate": _abstention_rate(scored),
            **_gold_metrics(pairs),
        }
    all_scored = _score_all(model, tokenizer, eligible, budget=LENGTH_BUDGETS[0], cache=cache)
    pooled_pairs = _chosen_pairs(eligible, all_scored)
    return {
        **base,
        "empty": False,
        "n_rows": len(eligible),
        "sources": sorted({r.source for r in eligible}),
        "by_family": by_family,
        "abstention_rate": _abstention_rate(all_scored),
        **_gold_metrics(pooled_pairs),
    }


_TRANSFORM_DESCRIPTIONS = {
    "contradictory": (
        "swap_gold_with_plausible_distractor: the gold option's text swaps with "
        "the non-gold option of highest gold utility (ties to the lowest index; "
        "first non-gold option when no scores); chosen_index is deliberately "
        "not remapped — the label contradicts the content"
    ),
    "missing_information": (
        "drop_decisive_sentence: remove the state sentence with maximal word-token "
        "overlap with the gold option text (ties to the earliest; the final sentence "
        "when no overlap); example_id is recomputed"
    ),
}

_TRANSFORMS: Mapping[str, Callable[[DecisionExample], tuple[DecisionExample, bool] | None]] = {
    "contradictory": _swap_gold_with_plausible_distractor,
    "missing_information": _drop_decisive_sentence,
}


def _context_stats(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[DecisionExample],
    *,
    transform: Callable[[DecisionExample], tuple[DecisionExample, bool] | None],
    cache: _ScoreCache,
) -> dict[str, Any]:
    scored_clean = _score_all(model, tokenizer, rows, budget=LENGTH_BUDGETS[0], cache=cache)
    clean_pairs = _chosen_pairs(rows, scored_clean)
    clean_metrics = _gold_metrics(clean_pairs)
    clean_abstention = _abstention_rate(scored_clean)

    transformed: list[DecisionExample] = []
    skipped_no_gold = 0
    state_emptied = 0
    for row in rows:
        result = transform(row)
        if result is None:
            skipped_no_gold += 1
            continue
        new_row, emptied = result
        transformed.append(new_row)
        state_emptied += int(emptied)
    scored_t = _score_all(model, tokenizer, transformed, budget=LENGTH_BUDGETS[0], cache=cache)
    t_pairs = _chosen_pairs(transformed, scored_t)
    t_metrics = _gold_metrics(t_pairs)
    transformed_abstention = _abstention_rate(scored_t)

    delta = (
        transformed_abstention - clean_abstention
        if transformed_abstention is not None and clean_abstention is not None
        else None
    )
    clean_accuracy = clean_metrics["accuracy"]
    transformed_accuracy = t_metrics["accuracy"]
    accuracy_delta = (
        transformed_accuracy - clean_accuracy
        if transformed_accuracy is not None and clean_accuracy is not None
        else None
    )
    return {
        "n_rows": len(rows),
        "n_transformed": len(transformed),
        "n_skipped_no_gold": skipped_no_gold,
        "n_state_emptied": state_emptied,
        "n_scored_rows": len(t_pairs),
        "abstention_rate_clean": clean_abstention,
        "abstention_rate_transformed": transformed_abstention,
        "abstention_rate_delta": delta,
        "abstention_rate_relative_change": (
            _relative_change(delta, clean_abstention) if delta is not None else None
        ),
        "accuracy_clean": clean_accuracy,
        "accuracy_transformed": transformed_accuracy,
        "accuracy_delta": accuracy_delta,
        "accuracy_relative_change": (
            _relative_change(accuracy_delta, clean_accuracy) if accuracy_delta is not None else None
        ),
        "bucket_ece_clean": clean_metrics["bucket_ece"],
        "bucket_ece_transformed": t_metrics["bucket_ece"],
    }


def _context_block(
    model: torch.nn.Module,
    tokenizer: Any,
    eval_rows: Mapping[str, list[DecisionExample]],
    families: Sequence[str],
    cache: _ScoreCache,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pooled_rows = [row for family in families for row in eval_rows[family]]
    for key, transform in _TRANSFORMS.items():
        by_family = {
            family: _context_stats(
                model, tokenizer, eval_rows[family], transform=transform, cache=cache
            )
            for family in families
        }
        out[key] = {
            "transform": _TRANSFORM_DESCRIPTIONS[key].split(":", 1)[0],
            "description": _TRANSFORM_DESCRIPTIONS[key],
            "abstention_threshold": ABSTENTION_THRESHOLD,
            "by_family": by_family,
            "overall": _context_stats(
                model, tokenizer, pooled_rows, transform=transform, cache=cache
            ),
        }
    return out


def _length_sweep_family(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[DecisionExample],
    *,
    cache: _ScoreCache,
) -> dict[str, Any]:
    by_budget: dict[str, Any] = {}
    accuracies: dict[int, float | None] = {}
    for budget in LENGTH_BUDGETS:
        scored = _score_all(model, tokenizer, rows, budget=budget, cache=cache)
        pairs = _chosen_pairs(rows, scored)
        stats = _gold_metrics(pairs)
        accuracy = stats["accuracy"]
        accuracies[budget] = accuracy
        by_budget[str(budget)] = {
            "budget": budget,
            "n_rows": len(rows),
            "n_scored_rows": len(pairs),
            "n_skipped_no_gold": len(rows) - len(pairs),
            "choice_accuracy": accuracy,
            "bucket_ece": stats["bucket_ece"],
            "ece_bins": stats["ece_bins"],
            "n_truncated": sum(1 for s in scored if s.truncated),
        }
    acc512 = accuracies[LENGTH_BUDGETS[0]]
    acc1024 = accuracies[LENGTH_BUDGETS[-1]]
    drop = acc512 - acc1024 if acc512 is not None and acc1024 is not None else None
    return {
        **by_budget,
        "drop_512_to_1024": drop,
        "length_extension": length_extension_verdict(acc512, acc1024),
    }


def _length_sweep_block(
    model: torch.nn.Module,
    tokenizer: Any,
    eval_rows: Mapping[str, list[DecisionExample]],
    families: Sequence[str],
    cache: _ScoreCache,
) -> dict[str, Any]:
    per_family = {
        family: _length_sweep_family(model, tokenizer, eval_rows[family], cache=cache)
        for family in families
    }
    pooled_rows = [row for family in families for row in eval_rows[family]]
    pooled = _length_sweep_family(model, tokenizer, pooled_rows, cache=cache)
    return {**per_family, "overall": pooled}


_NOTES: tuple[str, ...] = (
    "evaluation corpus = rows with split != 'train' from split_rows_by_family; "
    "train-tagged rows are used only as the source cross-check for the ood block",
    "abstention = confidence head below 0.75 (declared default utility triple "
    "correct=1, wrong=-3, abstain=0: (0 - (-3)) / (1 - (-3)) = 0.75)",
    "bucket ECE is src.evaluation.metrics.ece over (confidence, correct) pairs "
    "with fixed bin edges; bucket label = bucket_fn(n_options)",
    "metrics are read from src.evaluation.metrics; src.calibration is never imported",
    "permutation: all n! enumerated for n <= 4, eight seeded permutations for "
    "n > 4 (random.Random(f'{seed}:{family}:{n}')); deviation compares "
    "scores_perm[j] to base scores of option perm[j]",
    "accuracy metrics are computed on rows with a chosen_index; rows without "
    "one are scored and counted, but contribute None to accuracy/ECE",
    "contradiction transform keeps chosen_index (label contradicts content) and "
    "the content multiset (order-invariant example_id is unchanged)",
    "missing-information transform recomputes example_id; a state emptied to '' "
    "is counted as n_state_emptied",
    "length sweep re-encodes with encode_example(budget=…); a >2-point drop "
    "from 512 to 1024 sets length_extension to FAIL",
    "families with zero evaluation rows are reported with n_rows=0 and None "
    "metrics, never fabricated",
)


def run_robustness(
    model: torch.nn.Module,
    tokenizer: Any,
    split_rows_by_family: Mapping[str, Sequence[DecisionExample]],
    split_manifest: SplitManifest | str,
    ood_rows: Sequence[DecisionExample],
    bucket_fn: Callable[[int], str],
    seed: int = DEFAULT_SEED,
) -> RobustnessReport:
    """Measure decision-model robustness; see the module docstring for the contract."""
    seed = _validate_seed(seed)
    if not isinstance(model, torch.nn.Module):
        raise RobustnessInputError(
            f"model must be a torch.nn.Module, got {type(model).__name__}",
            field="model",
        )
    if not callable(getattr(tokenizer, "encode", None)):
        raise RobustnessInputError(
            f"tokenizer must expose a callable encode(), got {type(tokenizer).__name__}",
            field="tokenizer",
        )
    if not callable(bucket_fn):
        raise RobustnessInputError(
            f"bucket_fn must be callable, got {type(bucket_fn).__name__}",
            field="bucket_fn",
        )
    fingerprint, ood_source = _resolve_manifest(split_manifest)
    ood_rows_list = _validate_rows(ood_rows, field="ood_rows")
    families, train_rows, eval_rows = _manifest_rows_by_family(split_rows_by_family)

    snapshot = dict(TRUNCATION_COUNTS)
    was_training = bool(model.training)
    cache: _ScoreCache = {}
    try:
        model.eval()
        permutation = _permutation_block(model, tokenizer, eval_rows, families, seed, cache)
        option_count_sweep = _option_count_sweep(
            model, tokenizer, eval_rows, families, bucket_fn, cache
        )
        ood = _ood_block(
            model,
            tokenizer,
            ood_rows_list,
            train_rows,
            cache=cache,
            fingerprint=fingerprint,
            ood_source=ood_source,
        )
        context = _context_block(model, tokenizer, eval_rows, families, cache)
        length_sweep = _length_sweep_block(model, tokenizer, eval_rows, families, cache)
    finally:
        TRUNCATION_COUNTS.clear()
        TRUNCATION_COUNTS.update(snapshot)
        model.train(was_training)

    rows_count = {
        "train": len(train_rows),
        "evaluated": sum(len(rows) for rows in eval_rows.values()),
        "ood": len(ood_rows_list),
    }
    return RobustnessReport(
        seed=seed,
        split_manifest_fingerprint=fingerprint,
        ood_source=ood_source,
        rows=rows_count,
        permutation=permutation,
        option_count_sweep=option_count_sweep,
        ood=ood,
        context=context,
        length_sweep=length_sweep,
        notes=_NOTES,
    )
