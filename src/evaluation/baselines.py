"""Classical and frozen-embedding baselines for RAPPORT (FR-EVAL-02).

Three baselines run over the same split manifest as the model and emit
metrics in the same schema as :mod:`src.evaluation.metrics` (``MetricResult``,
imported, never redefined):

* ``B1-majority`` — majority class per family (B1). One distribution per
  ``(family, option-count)`` bucket taken from train; falls back to the
  family-pooled distribution, then to the global-pooled distribution, then to
  the uniform distribution. Probabilities are the empirical class frequencies.
  Ties break deterministically to the lowest option index.
* ``B3-tfidf-logreg`` — TF-IDF (char 3–5-gram + word 1–2-gram) over the joint
  ``goal + state + options`` text, hashed to a fixed ``TFIDF_FEATURE_DIMS``
  feature space, binarized (document presence), IDF-smoothed
  (``log((1 + N) / (1 + df)) + 1``), L2-normalized rows, followed by a
  multinomial (softmax) logistic regression **implemented in numpy**
  (full-batch gradient descent, L2 penalty ``‖W‖² / (2 · C · N)`` with fixed
  ``C = LOGREG_C = 1.0``, learning rate ``LOGREG_LR``, deterministic cap of
  ``LOGREG_MAX_ITERS = 200`` iterations, early stop when the relative
  objective improvement drops below ``LOGREG_TOL``; the realized iteration
  count and convergence flag are recorded in the run notes). One model is
  fitted per option-count bucket (2..8 options) because option indices are
  not comparable across buckets; a bucket unseen in train falls back to the
  train-global majority class distribution.
* ``B4-frozen-embedding-cosine`` — frozen ``bge-small-en-v1.5`` (mean-pooled,
  L2-normalized, cosine choice between the ``goal + state`` query and each
  option) when ``transformers`` can load the model **locally**
  (``local_files_only=True``; this module never touches the network).
  Otherwise a deterministic hashed char n-gram embedding (numpy, 384 dims,
  matching bge-small's hidden size; labelled ``hash-ngram``) is used. The
  resolved source is recorded on the result as
  ``embedding_source ∈ {"bge-small-en-v1.5", "hash-ngram"}``.

Conventions
-----------
* **Labels.** Only ``choice`` and ``preference`` rows carrying a
  ``chosen_index`` are consumed, per the pre-registered design (FR-EVAL-02).
  Every other family — and ``choice``/``preference`` rows without a chosen
  index — is skipped and counted in ``skipped`` (keyed by family, plus the
  key ``no-chosen-index`` for scored-family rows lacking one). A
  ``chosen_index`` that points outside ``[0, n_options)`` is a contract
  violation and raises :class:`BaselineInputError` (fail-fast).
* **Abstention is scored separately.** Abstention rows are excluded from the
  training and metric streams entirely. They are scored with a
  *uniform-random-over-options* default (``random.Random(ABSTENTION_RANDOM_SEED)``,
  seed 0 — deterministic across runs) and recorded on the result as
  ``dev_abstention`` / ``test_abstention`` (an :class:`AbstentionRandomScore`
  with ``n_rows``, ``n_scored``, the seeded-draw ``accuracy`` and the
  closed-form ``expected_random_accuracy = mean(1 / n_options)``).
* **Metric schema.** ``dev``/``test`` are :class:`MetricResult` objects, as
  are each entry of ``dev_per_family``/``test_per_family`` (only families
  present in the split appear; a split lacking a family reports nothing for
  it rather than fabricating rows). Calibration pairs use the confidence-vs-
  correctness view (the baselines emit no binary-probability head). ``scores``
  carry the per-option probabilities (baseline raw scores have no utility-
  scale meaning; within-row ranking is preserved). ``gold_scores`` carry the
  row's gold utilities when present, else the one-hot vector at
  ``chosen_index`` (documented stand-in so NDCG/Spearman/Pearson/utility-MAE
  remain defined). Metrics uncomputable for a baseline are ``None`` — e.g.
  the per-category risk metrics and over-refusal rate, because the classical
  baselines have no risk head — they are never fabricated. Per-family metric
  blocks are the same schema as the pooled block.
* **Same manifest as the model.** ``run`` requires the caller to echo the
  split-manifest fingerprint; the fingerprint must be a lowercase sha256 hex
  digest (validated via :func:`src.data.splits.fingerprint.is_sha256_hex`)
  and is recorded verbatim on the result. The rows handed in must carry
  matching ``split`` tags (``train_rows`` → ``split="train"`` etc.), so a
  manifest cannot be echoed for rows of another split.
  :func:`load_manifest_fingerprint` consumes the split-manifest JSON path and
  returns its fingerprint (strictly validated via
  ``src.data.splits.manifest.read_manifest``) — the entry point for
  "baselines run from the same split manifest as the model".
* **Determinism.** No wall clock, no network, no unseeded RNG anywhere. The
  TF-IDF solver is full-batch with zero initialization; the abstention
  default uses ``Random(0)``. Two runs over the same rows produce identical
  results.
* **Dependencies.** numpy is required. scipy is used *only* to hold the
  binarized TF-IDF matrix as a sparse CSR (it ships with the declared
  scikit-learn dependency); if scipy is absent, a dense numpy matrix is used
  instead (memory-hungry above ~65k cells — acceptable for fixtures, not for
  the 115.6k-row acceptance mixture). The logistic regression itself is
  always implemented in numpy, per the pre-registered design; scikit-learn is
  deliberately not used for it (solver behaviour would otherwise drift with
  sklearn versions).

``B3`` is a **hard gate**: :func:`y1_verdict` implements the pre-registered
Y1 criterion — pass iff ``model_accuracy − baseline_accuracy ≥ margin``
(margin in percentage points, default ``Y1_MARGIN_POINTS = 8.0``).
"""

from __future__ import annotations

import math
import operator
import random
import re
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Sequence

import numpy as np

from src.data.core.schema import DecisionExample
from src.data.splits.fingerprint import is_sha256_hex, membership_fingerprint
from src.data.splits.manifest import read_manifest
from src.evaluation.metrics import (
    MetricRecord,
    MetricResult,
    compute_metrics,
)
from src.evaluation.metrics import (
    accuracy as _binary_accuracy,
)

__all__ = [
    "ABSTENTION_RANDOM_SEED",
    "ALL_BASELINES",
    "B1_MAJORITY",
    "B3_TFIDF",
    "B4_COSINE",
    "BaselineError",
    "BaselineInputError",
    "BaselineMetrics",
    "BaselineRunner",
    "AbstentionRandomScore",
    "EMBEDDING_SOURCE_BGE",
    "EMBEDDING_SOURCE_HASH",
    "FrozenEmbeddingCosineBaseline",
    "HASH_EMBED_DIMS",
    "LOGREG_C",
    "LOGREG_MAX_ITERS",
    "LOGREG_LR",
    "LOGREG_TOL",
    "MajorityClassBaseline",
    "SCORED_FAMILIES",
    "TfidfLogisticBaseline",
    "TFIDF_FEATURE_DIMS",
    "Y1_MARGIN_POINTS",
    "load_manifest_fingerprint",
    "run_baselines",
    "y1_verdict",
]

#: Baseline identifiers, weakest to strongest.
B1_MAJORITY = "B1-majority"
B3_TFIDF = "B3-tfidf-logreg"
B4_COSINE = "B4-frozen-embedding-cosine"
ALL_BASELINES: tuple[str, ...] = (B1_MAJORITY, B3_TFIDF, B4_COSINE)

#: Families carrying the choice-style ``chosen_index`` label these baselines consume.
SCORED_FAMILIES: tuple[str, ...] = ("choice", "preference")

#: bge model id (loaded local-only; never downloaded by this module).
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_SOURCE_BGE = "bge-small-en-v1.5"
EMBEDDING_SOURCE_HASH = "hash-ngram"

#: TF-IDF feature-space size (hashed, binarized document presence).
TFIDF_FEATURE_DIMS = 50_000
TFIDF_CHAR_NGRAMS: tuple[int, ...] = (3, 4, 5)
TFIDF_WORD_NGRAMS: tuple[int, ...] = (1, 2)

#: Logistic-regression hyper-parameters (fixed by the pre-registered design).
LOGREG_C = 1.0
LOGREG_MAX_ITERS = 200
LOGREG_LR = 0.5
LOGREG_TOL = 1e-9

#: Hashed char n-gram embedding geometry (384 == bge-small hidden size).
HASH_EMBED_DIMS = 384
HASH_EMBED_NGRAMS: tuple[int, ...] = (3, 4, 5)

#: Seed for the abstention uniform-random-over-options default.
ABSTENTION_RANDOM_SEED = 0

#: Y1 pre-registered margin, in percentage points (docs/evaluation.md §8).
Y1_MARGIN_POINTS = 8.0

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_WORD_JOIN = " "


class BaselineError(ValueError):
    """Base class for baseline failures.

    Carries a stable ``code`` so callers can match on the failure class
    without parsing message text.
    """

    code = "BASELINE_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(f"{self.code}: {message}")
        self.message = message


class BaselineInputError(BaselineError):
    """Malformed input to a baseline (bad fingerprint, bad rows, bad ranges)."""

    code = "BASELINE_INPUT_INVALID"


# ---------------------------------------------------------------------------
# text featurization (shared by B3 and the hash-ngram embedding fallback)
# ---------------------------------------------------------------------------

def row_text(row: DecisionExample) -> str:
    """The joint text scored by the classical baselines: ``goal + state + options``."""
    return f"{row.goal}\n{row.state}\n{_WORD_JOIN.join(opt.text for opt in row.options)}"


def _word_tokens(lowered_text: str) -> list[str]:
    return _TOKEN_RE.findall(lowered_text)


def _feature_tokens(text: str) -> list[str]:
    """Namespaced feature tokens: ``w1:``/``w2:`` word grams, ``c{n}:`` char grams."""
    lowered = text.lower()
    tokens: list[str] = []
    words = _word_tokens(lowered)
    for n in TFIDF_WORD_NGRAMS:
        for i in range(0, len(words) - n + 1):
            tokens.append(f"w{n}:{_WORD_JOIN.join(words[i:i + n])}")
    for n in TFIDF_CHAR_NGRAMS:
        if len(lowered) < n:
            continue
        for i in range(len(lowered) - n + 1):
            tokens.append(f"c{n}:{lowered[i:i + n]}")
    return tokens


def _hash_token(token: str) -> int:
    """Deterministic 31-bit hash (CRC-32); stable across platforms and runs."""
    return zlib.crc32(token.encode("utf-8"))


def _char_grams(text: str, n: int) -> list[str]:
    lowered = text.lower()
    if len(lowered) < n:
        return []
    return [lowered[i:i + n] for i in range(len(lowered) - n + 1)]


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], n_classes), dtype=np.float64)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


class _TfidfVectorizer:
    """Hashed, binarized TF-IDF with smoothed IDF and L2-normalized rows.

    No vocabulary is stored — hashing makes the feature space fixed, which
    bounds memory regardless of corpus size (``TFIDF_FEATURE_DIMS``). Rows
    carry document-presence binarized counts (never raw term frequencies).
    """

    def __init__(self, feature_dims: int = TFIDF_FEATURE_DIMS) -> None:
        if isinstance(feature_dims, bool) or not isinstance(feature_dims, int) or feature_dims < 1:
            raise BaselineInputError(
                f"feature_dims must be a positive integer, got {feature_dims!r}")
        self.feature_dims = feature_dims
        self.idf_: np.ndarray | None = None
        self.n_docs_: int = 0

    def fit(self, docs: Sequence[str]) -> "_TfidfVectorizer":
        df = np.zeros(self.feature_dims, dtype=np.int64)
        for doc in docs:
            seen = {_hash_token(tok) % self.feature_dims for tok in _feature_tokens(doc)}
            for h in seen:
                df[h] += 1
        n = len(docs)
        self.idf_ = np.log((1.0 + n) / (1.0 + df)) + 1.0
        self.n_docs_ = n
        return self

    def transform(self, docs: Sequence[str]) -> Any:
        """Return a CSR matrix (scipy) or a dense float32 array (fallback)."""
        if self.idf_ is None:
            raise BaselineInputError("vectorizer is not fitted")
        try:
            from scipy import sparse
        except ImportError:
            sparse = None  # type: ignore[assignment]
        n = len(docs)
        if sparse is not None:
            indptr = np.zeros(n + 1, dtype=np.int64)
            indices: list[np.ndarray] = []
            data: list[np.ndarray] = []
            for i, doc in enumerate(docs):
                hashed = sorted(
                    {_hash_token(tok) % self.feature_dims for tok in _feature_tokens(doc)}
                )
                if not hashed:
                    indices.append(np.empty(0, dtype=np.int32))
                    data.append(np.empty(0, dtype=np.float32))
                    indptr[i + 1] = indptr[i]
                    continue
                vec = self.idf_[np.asarray(hashed, dtype=np.int64)]
                norm = float(np.linalg.norm(vec))
                if norm > 0.0:
                    vec = vec / norm
                indices.append(np.asarray(hashed, dtype=np.int32))
                data.append(vec.astype(np.float32))
                indptr[i + 1] = indptr[i] + len(hashed)
            flat_indices = np.concatenate(indices) if indices else np.empty(0, dtype=np.int32)
            flat_data = np.concatenate(data) if data else np.empty(0, dtype=np.float32)
            return sparse.csr_matrix(
                (flat_data, flat_indices, indptr), shape=(n, self.feature_dims)
            )
        # Dense fallback (fixtures only — O(n_docs × feature_dims) memory).
        if n * self.feature_dims > 2**26:
            raise BaselineInputError(
                f"dense TF-IDF fallback would need {n * self.feature_dims} cells; "
                "install scipy (ships with the declared scikit-learn dependency)"
            )
        mat = np.zeros((n, self.feature_dims), dtype=np.float32)
        for i, doc in enumerate(docs):
            hashed = sorted(
                {_hash_token(tok) % self.feature_dims for tok in _feature_tokens(doc)}
            )
            if not hashed:
                continue
            vec = self.idf_[np.asarray(hashed, dtype=np.int64)]
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                mat[i, np.asarray(hashed, dtype=np.int64)] = (vec / norm).astype(np.float32)
        return mat


class _SoftmaxLogReg:
    """Multinomial logistic regression, full-batch gradient descent, numpy.

    Objective: ``mean cross-entropy + ‖W‖² / (2 · C · N)`` with ``C = LOGREG_C``.
    Bias is unpenalized. Zero initialization ⇒ deterministic. Stops when the
    relative objective improvement falls below ``LOGREG_TOL`` or after
    ``LOGREG_MAX_ITERS`` iterations (the deterministic cap — convergence is
    recorded, never asserted, so an under-converged fit is reported, not
    raised).
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        c: float = LOGREG_C,
        lr: float = LOGREG_LR,
        tol: float = LOGREG_TOL,
        max_iters: int = LOGREG_MAX_ITERS,
    ) -> None:
        if n_classes < 1:
            raise BaselineInputError(f"n_classes must be >= 1, got {n_classes}")
        self.c = float(c)
        self.lr = float(lr)
        self.tol = float(tol)
        self.max_iters = int(max_iters)
        self.weight = np.zeros((n_features, n_classes), dtype=np.float64)
        self.bias = np.zeros(n_classes, dtype=np.float64)
        self.classes_: np.ndarray | None = None
        self.history: list[float] = []
        self.n_iters = 0
        self.converged = False

    def _objective(self, x_mat: Any, y_one_hot: np.ndarray, n: int) -> float:
        logits = x_mat @ self.weight + self.bias
        logits = logits - logits.max(axis=1, keepdims=True)
        logsumexp = np.log(np.exp(logits).sum(axis=1))
        log_probs = logits - logsumexp[:, None]
        ce = -float((y_one_hot * log_probs).sum()) / n
        penalty = float((self.weight**2).sum()) / (2.0 * self.c * n)
        return ce + penalty

    def fit(self, x_mat: Any, y: np.ndarray) -> "_SoftmaxLogReg":
        n = int(y.shape[0])
        if n == 0:
            self.n_iters = 0
            self.converged = True
            return self
        n_classes = int(self.weight.shape[1])
        y_one_hot = _one_hot(y.astype(np.int64), n_classes)
        prev = self._objective(x_mat, y_one_hot, n)
        self.history.append(prev)
        converged = False
        for it in range(1, self.max_iters + 1):
            logits = x_mat @ self.weight + self.bias
            probs = _softmax_rows(logits)
            grad_p = probs - y_one_hot
            grad_w = x_mat.T @ grad_p / n + self.weight / (self.c * n)
            grad_b = grad_p.mean(axis=0)
            self.weight -= self.lr * np.asarray(grad_w, dtype=np.float64)
            self.bias -= self.lr * grad_b
            obj = self._objective(x_mat, y_one_hot, n)
            self.history.append(obj)
            rel = abs(prev - obj) / max(abs(prev), 1e-12)
            self.n_iters = it
            prev = obj
            if rel < self.tol:
                converged = True
                break
        self.converged = converged
        return self

    def predict_proba(self, x_mat: Any) -> np.ndarray:
        logits = x_mat @ self.weight + self.bias
        return _softmax_rows(logits)


def _hash_embedding(text: str, dims: int = HASH_EMBED_DIMS) -> np.ndarray:
    """Deterministic hashed char n-gram embedding (the documented offline fallback)."""
    vec = np.zeros(dims, dtype=np.float64)
    lowered = text.lower()
    for n in HASH_EMBED_NGRAMS:
        for gram in _char_grams(lowered, n):
            vec[zlib.crc32(gram.encode("utf-8")) % dims] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0.0 else vec


@dataclass(frozen=True)
class AbstentionRandomScore:
    """Uniform-random-over-options default, scored on abstention rows.

    ``seed`` is fixed (``ABSTENTION_RANDOM_SEED``) so the draws are identical
    on every run. ``accuracy`` uses the seeded draws and is ``None`` when no
    abstention row carries a usable ``chosen_index``; ``expected_random_accuracy``
    is the closed-form ``mean(1 / n_options)`` over the scored rows.
    """

    n_rows: int
    n_scored: int
    accuracy: float | None
    expected_random_accuracy: float | None
    seed: int = ABSTENTION_RANDOM_SEED

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_scored": self.n_scored,
            "accuracy": self.accuracy,
            "expected_random_accuracy": self.expected_random_accuracy,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class BaselineMetrics:
    """One baseline's metrics over dev and test, in the ``MetricResult`` schema.

    ``dev``/``test`` are pooled :class:`MetricResult` blocks over the scored
    (``choice`` + ``preference``) rows; ``dev_per_family``/``test_per_family``
    carry the same schema per family, only for families present in the split.
    ``embedding_source`` is set only by the B4 runner. ``skipped`` maps split
    name → reason key → rows not consumed (family names for non-scored
    families and the key ``no-chosen-index`` for scored-family rows without
    one). ``notes`` records every judgment call made for this run.
    """

    baseline: str
    split_manifest_fingerprint: str
    embedding_source: str | None
    n_train: int
    n_dev: int
    n_test: int
    skipped: dict[str, dict[str, int]]
    dev: MetricResult
    test: MetricResult
    dev_per_family: dict[str, MetricResult]
    test_per_family: dict[str, MetricResult]
    dev_abstention: AbstentionRandomScore | None
    test_abstention: AbstentionRandomScore | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "split_manifest_fingerprint": self.split_manifest_fingerprint,
            "embedding_source": self.embedding_source,
            "n_train": self.n_train,
            "n_dev": self.n_dev,
            "n_test": self.n_test,
            "skipped": {split: dict(counts) for split, counts in self.skipped.items()},
            "dev": self.dev.to_dict(),
            "test": self.test.to_dict(),
            "dev_per_family": {k: v.to_dict() for k, v in self.dev_per_family.items()},
            "test_per_family": {k: v.to_dict() for k, v in self.test_per_family.items()},
            "dev_abstention": self.dev_abstention.to_dict() if self.dev_abstention else None,
            "test_abstention": self.test_abstention.to_dict() if self.test_abstention else None,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# row partitioning
# ---------------------------------------------------------------------------

def _partition(
    rows: Sequence[DecisionExample],
) -> tuple[list[DecisionExample], dict[str, int], list[DecisionExample]]:
    """Split rows into (scored, skipped-counts, abstention). Fail-fast on bad labels."""
    scored: list[DecisionExample] = []
    skipped: dict[str, int] = {}
    abstention: list[DecisionExample] = []
    for row in rows:
        family = row.task_family
        n_options = len(row.options)
        if family not in SCORED_FAMILIES and family != "abstention":
            skipped[family] = skipped.get(family, 0) + 1
            continue
        chosen = row.chosen_index
        if chosen is None:
            # An unlabeled abstention row is scored with n_scored=0 (documented);
            # but an INVALID index must never be silently skipped below.
            if family != "abstention":
                skipped["no-chosen-index"] = skipped.get("no-chosen-index", 0) + 1
            if family == "abstention":
                abstention.append(row)
            continue
        chosen_int = _require_index(chosen, "chosen_index")
        if not 0 <= chosen_int < n_options:
            raise BaselineInputError(
                f"chosen_index {chosen_int} out of range [0, {n_options}) "
                f"(example_id={row.example_id[:16]}…)"
            )
        if family == "abstention":
            abstention.append(row)
        else:
            scored.append(row)
    return scored, skipped, abstention


def _require_index(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise BaselineInputError(f"{name} must be an integer, got {value!r}")
    try:
        return int(operator.index(value))  # type: ignore[arg-type]
    except TypeError as exc:
        raise BaselineInputError(f"{name} must be an integer, got {value!r}") from exc


def _onehot_scores(chosen: int, n_options: int) -> tuple[float, ...]:
    return tuple(1.0 if i == chosen else 0.0 for i in range(n_options))


def _record_for(row: DecisionExample, probs: Sequence[float]) -> MetricRecord:
    n_options = len(row.options)
    chosen = _require_index(row.chosen_index, "chosen_index")  # validated by _partition
    if row.scores is not None:
        gold_scores = tuple(float(v) for v in row.scores)
    else:
        gold_scores = _onehot_scores(chosen, n_options)
    prob_tuple = tuple(float(p) for p in probs)
    return MetricRecord(
        chosen_index=chosen,
        predicted_index=int(np.argmax(prob_tuple)),
        scores=prob_tuple,
        gold_scores=gold_scores,
        probabilities=prob_tuple,
        confidence=float(max(prob_tuple)),
    )


def _evaluate_scored(
    rows: Sequence[DecisionExample],
    runner: "BaselineRunner",
) -> tuple[MetricResult, dict[str, MetricResult]]:
    """Pooled + per-family MetricResults over the scored stream (empty-safe)."""
    if len(rows) == 0:
        return compute_metrics([]), {}
    distributions = runner.predict_distribution(list(rows))
    records = [
        _record_for(row, probs) for row, probs in zip(rows, distributions, strict=True)
    ]
    pooled = compute_metrics(records)
    per_family: dict[str, MetricResult] = {}
    for family in SCORED_FAMILIES:
        family_records = [
            rec for rec, row in zip(records, rows, strict=True) if row.task_family == family
        ]
        if family_records:
            per_family[family] = compute_metrics(family_records)
    return pooled, per_family


def _score_abstention(rows: Sequence[DecisionExample]) -> AbstentionRandomScore | None:
    """Score abstention rows with the uniform-random-over-options default."""
    if len(rows) == 0:
        return None
    rng = random.Random(ABSTENTION_RANDOM_SEED)
    gold: list[int] = []
    preds: list[int] = []
    expected = 0.0
    for row in rows:
        chosen = row.chosen_index
        if chosen is None:
            continue  # documented: unlabeled rows are counted but not scored
        # _partition already validated these; never silently skip an invalid
        # abstention index — that would quietly shrink the scored cohort.
        chosen_int = _require_index(chosen, "chosen_index")
        n_options = len(row.options)
        if not 0 <= chosen_int < n_options:
            raise BaselineInputError(
                f"chosen_index {chosen_int} out of range [0, {n_options}) "
                f"(example_id={row.example_id[:16]}…)"
            )
        gold.append(chosen_int)
        preds.append(rng.randrange(n_options))
        expected += 1.0 / n_options
    if not gold:
        return AbstentionRandomScore(
            n_rows=len(rows),
            n_scored=0,
            accuracy=None,
            expected_random_accuracy=None,
        )
    return AbstentionRandomScore(
        n_rows=len(rows),
        n_scored=len(gold),
        accuracy=_binary_accuracy(gold, preds),
        expected_random_accuracy=expected / len(gold),
    )


# ---------------------------------------------------------------------------
# base runner
# ---------------------------------------------------------------------------

class BaselineRunner(ABC):
    """One baseline, one pass over the split manifest.

    Subclasses implement :meth:`fit` (train on the scored stream only) and
    :meth:`predict_distribution` (one probability vector per option, summing
    to 1). The shared :meth:`run` does everything else: fingerprint and split
    tag validation, partitioning, metric emission, abstention scoring and
    note collection.
    """

    #: Stable baseline identifier (see ALL_BASELINES).
    baseline: ClassVar[str]

    def __init__(self) -> None:
        self.fit_notes: list[str] = []
        self._fitted = False

    # -- subclass hooks -------------------------------------------------------

    @abstractmethod
    def fit(self, train_rows: Sequence[DecisionExample]) -> None:
        """Fit on the scored (``choice`` + ``preference``) train rows."""

    @abstractmethod
    def predict_distribution(
        self, rows: Sequence[DecisionExample]
    ) -> list[list[float]]:
        """Return one per-option probability vector per row (sums to 1)."""

    # -- shared entry point ---------------------------------------------------

    def run(
        self,
        train_rows: Sequence[DecisionExample],
        dev_rows: Sequence[DecisionExample],
        test_rows: Sequence[DecisionExample],
        split_manifest_fingerprint: str,
    ) -> BaselineMetrics:
        fingerprint = split_manifest_fingerprint
        if not is_sha256_hex(fingerprint):
            raise BaselineInputError(
                f"split_manifest_fingerprint must be a lowercase sha256 hex digest, "
                f"got {fingerprint!r}"
            )
        splits: dict[str, list[DecisionExample]] = {}
        for name, rows in (("train", train_rows), ("dev", dev_rows), ("test", test_rows)):
            materialized = list(rows)
            for index, row in enumerate(materialized):
                if not isinstance(row, DecisionExample):
                    raise BaselineInputError(
                        f"{name}_rows[{index}] must be a DecisionExample, "
                        f"got {type(row).__name__}"
                    )
                if row.split != name:
                    raise BaselineInputError(
                        f"{name}_rows[{index}] carries split={row.split!r} "
                        f"(example_id={row.example_id[:16]}…); rows must come from "
                        f"the {name} split of the manifest being echoed"
                    )
            splits[name] = materialized

        # Echoing a well-formed digest is not provenance: recompute it from the
        # actual row membership and refuse a mismatch. This catches a caller
        # passing a fingerprint for a different record stream.
        actual_fingerprint = membership_fingerprint(
            (row.example_id, row.split)
            for name in ("train", "dev", "test")
            for row in splits[name]
        )
        if actual_fingerprint != fingerprint:
            raise BaselineInputError(
                "split_manifest_fingerprint does not match the actual (example_id, split) "
                f"membership: expected {fingerprint}, computed {actual_fingerprint}"
            )

        scored_train, skipped_train, _ = _partition(splits["train"])
        scored_dev, skipped_dev, abst_dev = _partition(splits["dev"])
        scored_test, skipped_test, abst_test = _partition(splits["test"])

        self.fit_notes = []
        self.fit(scored_train)
        self._fitted = True
        if not scored_train:
            self.fit_notes.append(
                "no labeled (choice/preference) train rows; predictions are the "
                "documented fallback (uniform where no distribution exists)"
            )

        dev_metrics, dev_per_family = _evaluate_scored(scored_dev, self)
        test_metrics, test_per_family = _evaluate_scored(scored_test, self)
        dev_abstention = _score_abstention(abst_dev)
        test_abstention = _score_abstention(abst_test)

        notes = list(self.fit_notes)
        notes.append(
            f"abstention rows scored separately with uniform-random-over-options "
            f"default (seed={ABSTENTION_RANDOM_SEED}); recorded in dev_abstention/"
            f"test_abstention, never mixed into the pooled metrics"
        )
        notes.append(
            "probability/scoring/risk rows carry no choice-style label for these "
            "baselines and are not scored; risk metrics and the over-refusal rate "
            "are undefined (None) by construction"
        )
        notes.append(
            "scores carry per-option probabilities (within-row ranking preserved); "
            "gold_scores are the row's gold utilities when present, else the one-hot "
            "vector at chosen_index"
        )
        return BaselineMetrics(
            baseline=self.baseline,
            split_manifest_fingerprint=fingerprint,
            embedding_source=self.embedding_source_resolved(),
            n_train=len(splits["train"]),
            n_dev=len(splits["dev"]),
            n_test=len(splits["test"]),
            skipped={"train": skipped_train, "dev": skipped_dev, "test": skipped_test},
            dev=dev_metrics,
            test=test_metrics,
            dev_per_family=dev_per_family,
            test_per_family=test_per_family,
            dev_abstention=dev_abstention,
            test_abstention=test_abstention,
            notes=tuple(notes),
        )

    def embedding_source_resolved(self) -> str | None:
        """Resolved embedding provenance — only B4 sets this."""
        return None

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise BaselineInputError(
                f"{self.baseline}: predict_distribution called before fit()"
            )


class MajorityClassBaseline(BaselineRunner):
    """B1 — majority class per family, from train frequencies.

    Distributions are keyed ``(family, n_options)`` with deterministic
    fallbacks: ``(family, n_options)`` → family-pooled → global-pooled →
    uniform. Ties break to the lowest option index. Probabilities are the
    train class frequencies (they sum to 1 and double as calibration input).
    """

    baseline: ClassVar[str] = B1_MAJORITY

    def __init__(self) -> None:
        super().__init__()
        self._bucket: dict[tuple[str, int], np.ndarray] = {}
        self._family: dict[str, np.ndarray] = {}
        self._global: np.ndarray | None = None

    @staticmethod
    def _distribution(counts: np.ndarray) -> np.ndarray:
        total = float(counts.sum())
        if total <= 0.0:
            return np.full(len(counts), 1.0 / len(counts))
        return counts / total

    @staticmethod
    def _bump(mapping: dict[Any, np.ndarray], key: Any, index: int) -> None:
        counts = mapping.get(key)
        if counts is None:
            counts = np.zeros(index + 1, dtype=np.float64)
            mapping[key] = counts
        elif index >= len(counts):
            grown = np.zeros(index + 1, dtype=np.float64)
            grown[: len(counts)] = counts
            counts = grown
            mapping[key] = counts
        counts[index] += 1.0

    def fit(self, train_rows: Sequence[DecisionExample]) -> None:
        bucket_counts: dict[tuple[str, int], np.ndarray] = {}
        family_counts: dict[str, np.ndarray] = {}
        global_counts = np.zeros(0, dtype=np.float64)
        for row in train_rows:
            chosen = _require_index(row.chosen_index, "chosen_index")  # pre-validated
            self._bump(bucket_counts, (row.task_family, len(row.options)), chosen)
            self._bump(family_counts, row.task_family, chosen)
            if chosen >= len(global_counts):
                grown = np.zeros(chosen + 1, dtype=np.float64)
                grown[: len(global_counts)] = global_counts
                global_counts = grown
            global_counts[chosen] += 1.0
        self._bucket = {key: self._distribution(c) for key, c in bucket_counts.items()}
        self._family = {key: self._distribution(c) for key, c in family_counts.items()}
        self._global = self._distribution(global_counts) if global_counts.size else None
        self.fit_notes.append(
            f"B1 majority-class: fitted on {len(train_rows)} scored train rows; "
            "probabilities are train class frequencies; ties break to the lowest index"
        )
        self._fitted = True

    @staticmethod
    def _truncate_renormalize(vec: np.ndarray, size: int) -> np.ndarray:
        if len(vec) >= size:
            out = vec[:size].copy()
        else:
            out = np.zeros(size, dtype=np.float64)
            out[: len(vec)] = vec
        total = float(out.sum())
        if total <= 0.0:
            return np.full(size, 1.0 / size, dtype=np.float64)
        return out / total

    def predict_distribution(self, rows: Sequence[DecisionExample]) -> list[list[float]]:
        self._require_fitted()
        out: list[list[float]] = []
        for row in rows:
            n_options = len(row.options)
            dist: np.ndarray | None = None
            bucket = self._bucket.get((row.task_family, n_options))
            if bucket is not None:
                dist = self._truncate_renormalize(bucket, n_options)
            elif row.task_family in self._family:
                dist = self._truncate_renormalize(self._family[row.task_family], n_options)
            elif self._global is not None:
                dist = self._truncate_renormalize(self._global, n_options)
            else:
                dist = np.full(n_options, 1.0 / n_options, dtype=np.float64)
            out.append([float(v) for v in dist])
        return out


class TfidfLogisticBaseline(BaselineRunner):
    """B3 — TF-IDF (char 3–5 + word 1–2) + numpy softmax logistic regression.

    One logistic-regression model is fitted per option-count bucket, because
    option indices are not comparable across buckets with different sizes.
    A bucket unseen in train falls back to the train-global class frequencies
    (uniform when train is empty), and this fallback is recorded in the notes.
    Convergence is *recorded* (``converged`` + ``n_iters`` in the notes), the
    deterministic cap is ``LOGREG_MAX_ITERS``; both are documented, neither is
    asserted.
    """

    baseline: ClassVar[str] = B3_TFIDF

    def __init__(
        self,
        *,
        feature_dims: int = TFIDF_FEATURE_DIMS,
        c: float = LOGREG_C,
        lr: float = LOGREG_LR,
        tol: float = LOGREG_TOL,
        max_iters: int = LOGREG_MAX_ITERS,
    ) -> None:
        super().__init__()
        self.feature_dims = feature_dims
        self.c = c
        self.lr = lr
        self.tol = tol
        self.max_iters = max_iters
        self.vectorizer: _TfidfVectorizer | None = None
        self.models: dict[int, _SoftmaxLogReg] = {}
        self._global_counts: np.ndarray | None = None
        self._unseen_reported: set[int] = set()

    def fit(self, train_rows: Sequence[DecisionExample]) -> None:
        docs = [row_text(row) for row in train_rows]
        self.vectorizer = _TfidfVectorizer(self.feature_dims).fit(docs)
        bucket_rows: dict[int, list[int]] = {}
        global_counts = np.zeros(0, dtype=np.float64)
        for i, row in enumerate(train_rows):
            chosen = _require_index(row.chosen_index, "chosen_index")  # pre-validated
            bucket_rows.setdefault(len(row.options), []).append(i)
            if chosen >= len(global_counts):
                grown = np.zeros(chosen + 1, dtype=np.float64)
                grown[: len(global_counts)] = global_counts
                global_counts = grown
            global_counts[chosen] += 1.0
        self._global_counts = global_counts if global_counts.size else None
        self._unseen_reported = set()
        self.models = {}
        for n_options in sorted(bucket_rows):
            idxs = bucket_rows[n_options]
            chosen_arr = np.array(
                [_require_index(train_rows[i].chosen_index, "chosen_index") for i in idxs],
                dtype=np.int64,
            )
            classes, mapped = np.unique(chosen_arr, return_inverse=True)
            model = _SoftmaxLogReg(
                self.feature_dims,
                len(classes),
                c=self.c,
                lr=self.lr,
                tol=self.tol,
                max_iters=self.max_iters,
            )
            model.classes_ = classes
            subset_docs = [docs[i] for i in idxs]
            x_mat = self.vectorizer.transform(subset_docs)
            model.fit(x_mat, mapped)
            self.models[n_options] = model
            final_obj = model.history[-1] if model.history else float("nan")
            self.fit_notes.append(
                f"B3 bucket n_options={n_options}: train_rows={len(idxs)} "
                f"classes={classes.tolist()} converged={model.converged} "
                f"n_iters={model.n_iters}/{self.max_iters} final_objective={final_obj:.6f}"
            )
        self._fitted = True

    def _fallback_probs(self, n_options: int) -> list[float]:
        if self._global_counts is not None:
            counts = self._global_counts
            if len(counts) >= n_options:
                vec = counts[:n_options].astype(np.float64)
            else:
                vec = np.zeros(n_options, dtype=np.float64)
                vec[: len(counts)] = counts
            total = float(vec.sum())
            if total > 0.0:
                return [float(v) / total for v in vec]
        return [1.0 / n_options] * n_options

    def predict_distribution(self, rows: Sequence[DecisionExample]) -> list[list[float]]:
        self._require_fitted()
        assert self.vectorizer is not None  # set in fit()
        group: dict[int, list[int]] = {}
        for i, row in enumerate(rows):
            group.setdefault(len(row.options), []).append(i)
        results: list[list[float] | None] = [None] * len(rows)
        for n_options in sorted(group):
            idxs = group[n_options]
            model = self.models.get(n_options)
            if model is None:
                for i in idxs:
                    results[i] = self._fallback_probs(n_options)
                if n_options not in self._unseen_reported:
                    self._unseen_reported.add(n_options)
                    self.fit_notes.append(
                        f"B3 bucket n_options={n_options} unseen in train; "
                        "used train-global frequencies as fallback"
                    )
                continue
            subset = [rows[i] for i in idxs]
            x_mat = self.vectorizer.transform([row_text(r) for r in subset])
            probs = model.predict_proba(x_mat)
            assert model.classes_ is not None
            for j, i in enumerate(idxs):
                vec = np.zeros(n_options, dtype=np.float64)
                vec[model.classes_] = probs[j]
                out_sum = float(vec.sum())
                results[i] = [float(v) / out_sum for v in vec]
        final: list[list[float]] = []
        for i, resolved in enumerate(results):
            if resolved is None:  # pragma: no cover - defensive; every row is assigned
                raise BaselineInputError(f"B3 internal error: row {i} was not scored")
            final.append(resolved)
        return final

    def embedding_source_resolved(self) -> str | None:  # pragma: no cover - B3 has none
        return None


class FrozenEmbeddingCosineBaseline(BaselineRunner):
    """B4 — frozen ``bge-small-en-v1.5`` embeddings + cosine choice.

    The query is ``goal + state``; each option's text is embedded separately;
    per-option cosine similarity is turned into probabilities by an untempered
    softmax (temperature fixed at 1.0 — cosines in [-1, 1] are already
    well-scaled). The model is loaded **local-only** (``local_files_only=True``)
    so this module can never touch the network; if the import or the local
    load fails — or ``embedding_source="hash"`` is forced — a deterministic
    hashed char 3–5-gram embedding (numpy, ``HASH_EMBED_DIMS`` dims, L2
    normalized) is used instead, and the output records
    ``embedding_source="hash-ngram"``. Requesting ``embedding_source="bge"``
    explicitly raises :class:`BaselineInputError` when bge cannot be loaded
    (an explicit request is never silently downgraded).
    """

    baseline: ClassVar[str] = B4_COSINE

    def __init__(self, embedding_source: str = "auto", *,
                 batch_size: int = 32, max_length: int = 512) -> None:
        super().__init__()
        source = (embedding_source or "auto").lower()
        if source not in ("auto", "hash", "bge"):
            raise BaselineInputError(
                f"embedding_source must be one of 'auto', 'hash', 'bge', got {embedding_source!r}"
            )
        self.requested_source = source
        self.batch_size = batch_size
        self.max_length = max_length
        self._resolved: str | None = None
        self._tokenizer: Any = None
        self._model: Any = None

    def embedding_source_resolved(self) -> str | None:
        self._require_fitted()
        return self._resolved

    def fit(self, train_rows: Sequence[DecisionExample]) -> None:
        # Frozen embeddings are stateless; resolving the source happens once here.
        resolved = self._resolve_source()
        self._resolved = resolved
        if resolved == EMBEDDING_SOURCE_HASH:
            self.fit_notes.append(
                "B4 embedding_source=hash-ngram: deterministic hashed char 3-5-gram "
                f"embedding (numpy, {HASH_EMBED_DIMS} dims, L2-normalized); "
                "bge-small-en-v1.5 unavailable or bypassed — offline-safe fallback"
            )
        else:
            self.fit_notes.append(
                f"B4 embedding_source={EMBEDDING_SOURCE_BGE}: local_files_only load of "
                "BAAI/bge-small-en-v1.5; mean pooling over the attention mask; "
                "L2-normalized; cosine choice with softmax temperature 1.0"
            )
        self.fit_notes.append(
            "B4 is stateless across train: it consumes no labels; fit() only "
            "resolves the embedding source"
        )
        self._fitted = True

    def _resolve_source(self) -> str:
        if self.requested_source == "hash":
            self._resolved = EMBEDDING_SOURCE_HASH
            return EMBEDDING_SOURCE_HASH
        try:
            tokenizer, model = self._load_bge()
        except Exception as exc:
            if self.requested_source == "bge":
                raise BaselineInputError(
                    f"embedding_source='bge' was requested but "
                    f"{BGE_MODEL_NAME} could not be loaded locally ({type(exc).__name__}: {exc})"
                ) from exc
            self.fit_notes.append(
                f"B4 bge-small-en-v1.5 unavailable locally "
                f"({type(exc).__name__}); falling back to hash-ngram"
            )
            self._resolved = EMBEDDING_SOURCE_HASH
            return EMBEDDING_SOURCE_HASH
        self._tokenizer, self._model = tokenizer, model
        self._resolved = EMBEDDING_SOURCE_BGE
        return EMBEDDING_SOURCE_BGE

    def _load_bge(self) -> tuple[Any, Any]:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME, local_files_only=True)
        model = AutoModel.from_pretrained(BGE_MODEL_NAME, local_files_only=True)
        model.eval()
        return tokenizer, model

    def _embed_bge(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        vectors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start:start + self.batch_size])
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                output = self._model(**enc)
                last = output.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(last.dtype)
                pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy().astype(np.float64))
        return np.concatenate(vectors, axis=0)

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._resolved == EMBEDDING_SOURCE_BGE:
            return self._embed_bge(texts)
        return np.stack([_hash_embedding(t) for t in texts])

    def predict_distribution(self, rows: Sequence[DecisionExample]) -> list[list[float]]:
        self._require_fitted()
        if len(rows) == 0:
            return []
        queries = self._embed([f"{r.goal}\n{r.state}" for r in rows])
        option_texts: list[str] = []
        spans: list[tuple[int, int]] = []
        for row in rows:
            start = len(option_texts)
            option_texts.extend(opt.text for opt in row.options)
            spans.append((start, len(option_texts)))
        options = self._embed(option_texts)
        out: list[list[float]] = []
        for i, (start, end) in enumerate(spans):
            cosines = np.clip(options[start:end] @ queries[i], -1.0, 1.0)
            probs = _softmax_rows(cosines.reshape(1, -1))[0]
            out.append([float(v) for v in probs])
        return out


# ---------------------------------------------------------------------------
# Y1 hard gate + manifest/bootstrap helpers + module entry point
# ---------------------------------------------------------------------------

def y1_verdict(
    model_acc: float | None,
    baseline_acc: float | None,
    margin: float = Y1_MARGIN_POINTS,
) -> str:
    """Y1 gate (docs/evaluation.md §8): pass iff ``model − baseline ≥ margin``.

    ``model_acc``/``baseline_acc`` are fractions in ``[0, 1]`` (the units of
    :func:`metrics.accuracy`); ``margin`` is in **percentage points** (default
    ``8.0`` — RAPPORT-C must clear B3 by 8 points in-distribution). Fail-closed:
    a missing accuracy yields ``"fail"``. Out-of-domain inputs raise
    :class:`BaselineInputError` rather than silently passing or failing.
    """
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise BaselineInputError(f"margin must be a real number, got {margin!r}")
    margin_f = float(margin)
    if not math.isfinite(margin_f) or margin_f < 0.0:
        raise BaselineInputError(f"margin must be a finite, non-negative number, got {margin!r}")
    for name, value in (("model_acc", model_acc), ("baseline_acc", baseline_acc)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BaselineInputError(f"{name} must be a float in [0, 1] or None, got {value!r}")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise BaselineInputError(f"{name} must lie in [0, 1], got {value!r}")
    if model_acc is None or baseline_acc is None:
        return "fail"
    gap_points = (float(model_acc) - float(baseline_acc)) * 100.0
    # 1e-9-point tolerance absorbs binary floating-point noise around the
    # pre-registered boundary (e.g. 0.63 − 0.55 landing a hair below 8.0).
    return "pass" if gap_points >= margin_f - 1e-9 else "fail"


def load_manifest_fingerprint(path: str | Path) -> str:
    """Read the split-manifest JSON at ``path``; return its (validated) fingerprint.

    Strictly validates the manifest through
    :func:`src.data.splits.manifest.read_manifest` — a manifest with a
    non-zero ``group_overlap`` can never exist in the first place, so the
    fingerprint returned here is always safe to echo to
    :meth:`BaselineRunner.run`.
    """
    manifest = read_manifest(path)
    return manifest.fingerprint


def run_baselines(
    train_rows: Sequence[DecisionExample],
    dev_rows: Sequence[DecisionExample],
    test_rows: Sequence[DecisionExample],
    split_manifest_fingerprint: str,
    *,
    baselines: Sequence[str] = ALL_BASELINES,
) -> tuple[BaselineMetrics, ...]:
    """Run the requested baselines (default: all three) over one manifest."""
    registry: dict[str, type[BaselineRunner]] = {
        B1_MAJORITY: MajorityClassBaseline,
        B3_TFIDF: TfidfLogisticBaseline,
        B4_COSINE: FrozenEmbeddingCosineBaseline,
    }
    results: list[BaselineMetrics] = []
    for name in baselines:
        cls = registry.get(name)
        if cls is None:
            raise BaselineInputError(
                f"unknown baseline {name!r}; expected one of {list(registry)}"
            )
        results.append(cls().run(train_rows, dev_rows, test_rows, split_manifest_fingerprint))
    return tuple(results)
