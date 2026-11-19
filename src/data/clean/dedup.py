"""Deduplication stage: exact and near duplicates.

Exact dedup keys a row on ``(task_family, goal, state, sorted option texts,
chosen_index)`` over the already-normalized fields (normalization happened in
``src.data.core.normalize_example``, so "byte-identical normalized rows" are
byte-identical here). The first occurrence in input order is kept; later
occurrences are removed and counted per source. The key builder is imported
against ``src.data.core`` semantics — the texts are the NFC, HTML-stripped,
whitespace-collapsed fields produced by TASK-03, and the option texts are
sorted so the key is invariant to option permutation.

Near dedup is a deterministic 128-permutation MinHash over character 5-gram
shingles of a canonical document (``goal``, ``state``, then the *sorted*
option texts, newline-joined — order-insensitive over options). Similarity is
the fraction of equal MinHash positions, an unbiased estimate of the Jaccard
index of the shingle sets. Rows whose estimated similarity is at or above the
threshold (default 0.9, recorded on the result) are unioned into clusters;
each cluster keeps its earliest row and the rest are removed.

``datasketch`` is intentionally not required: the MinHasher below is pure
numpy. Its structure — :class:`NearDedup._candidate_pairs` as the seam
between signature generation and pair evaluation — is the drop-in point for a
banded-LSH implementation. The bundled implementation evaluates **all** pairs
in O(n²); that is only acceptable at fixture scale (hundreds to low
thousands of rows) and is documented as such. A production-scale run must
replace ``_candidate_pairs`` with an LSH index before the dataset grows past
that.

Near dedup never merges two rows whose gold option *text* differs: those
are contradiction candidates, not duplicates, and merging them would
silently choose one gold label. They survive to the contradiction audit
untouched. Comparing the gold text (rather than the raw ``chosen_index``)
keeps the guard permutation-aware: the same gold text reached through a
permuted option list is still a duplicate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np

from src.data.core.schema import DecisionExample

#: Jaccard-estimate threshold at or above which two rows are near-duplicates.
DEFAULT_NEAR_THRESHOLD = 0.9
#: Number of MinHash permutations.
NUM_PERM = 128
#: Character shingle width.
SHINGLE_SIZE = 5
#: Fixed seed: identical input always yields identical signatures.
DEFAULT_SEED = 1729


def exact_key(example: DecisionExample) -> tuple:
    """Return the normalized exact-dedup key for ``example``.

    ``(task_family, goal, state, sorted option texts, chosen_index)``. The
    texts are the fields TASK-03 normalization produced; sorting the option
    texts makes the key option-order invariant, matching the philosophy of
    ``compute_example_id``.
    """
    return (
        example.task_family,
        example.goal,
        example.state,
        tuple(sorted(option.text for option in example.options)),
        example.chosen_index,
    )


@dataclass(frozen=True)
class ExactDedupResult:
    """Output of :class:`ExactDedup`."""

    kept: tuple[DecisionExample, ...]
    removed_ids: tuple[str, ...]
    per_source: dict[str, int] = field(default_factory=dict)

    @property
    def exact_removed(self) -> int:
        return len(self.removed_ids)


class ExactDedup:
    """Remove byte-identical normalized rows, keeping the first occurrence."""

    def run(self, examples: Sequence[DecisionExample]) -> ExactDedupResult:
        seen: dict[tuple, int] = {}
        kept: list[DecisionExample] = []
        removed_ids: list[str] = []
        per_source: dict[str, int] = {}
        for example in examples:
            key = exact_key(example)
            if key in seen:
                removed_ids.append(example.example_id)
                per_source[example.source] = per_source.get(example.source, 0) + 1
                continue
            seen[key] = len(kept)
            kept.append(example)
        return ExactDedupResult(
            kept=tuple(kept),
            removed_ids=tuple(removed_ids),
            per_source=per_source,
        )


# --- Near dedup --------------------------------------------------------------


def gold_signature(example: DecisionExample) -> str | None:
    """Return the normalized text of the gold option, or ``None`` if unset.

    Used as the pair-acceptance guard for near dedup: equality of the gold
    *text* means "same decision, possibly permuted presentation"; difference
    means "conflicting label" and the pair must not be merged.
    """
    chosen = example.chosen_index
    if chosen is None:
        return None
    if isinstance(chosen, bool) or not isinstance(chosen, int):
        return None
    if not 0 <= chosen < len(example.options):
        return None
    return example.options[chosen].text


def canonical_document(example: DecisionExample) -> str:
    """Build the option-order-insensitive document used for shingling."""
    options = "\n".join(sorted(option.text for option in example.options))
    return f"{example.goal}\n{example.state}\n{options}"


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    """Return the set of character ``size``-shingles of ``text``.

    Texts shorter than one shingle yield the text itself as a single shingle,
    so short rows still have a comparable signature.
    """
    if not text:
        return set()
    if len(text) < size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def jaccard_index(a: set[str], b: set[str]) -> float:
    """Exact Jaccard index of two shingle sets (used by tests and audits)."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def _hash_shingle(shingle: str) -> np.uint64:
    digest = hashlib.sha256(shingle.encode("utf-8")).digest()
    return np.uint64(int.from_bytes(digest[:8], byteorder="little"))


class MinHasher:
    """Deterministic 64-bit wraparound MinHash over hashed shingles.

    ``a`` and ``b`` are drawn once from a seeded numpy Generator; every
    shingle is hashed with sha256 and reduced to 64 bits; the signature is
    ``min over shingles of (a * h + b) mod 2**64`` computed with numpy uint64
    wraparound arithmetic (the standard universal family; it needs no prime
    modulus and vectorizes). Equal-minima fraction is an unbiased estimator of
    the shingle-set Jaccard index with standard error ≈ sqrt(J(1-J)/128).
    """

    def __init__(self, num_perm: int = NUM_PERM, seed: int = DEFAULT_SEED) -> None:
        if num_perm < 1:
            raise ValueError("num_perm must be at least 1")
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.seed = seed
        # Multipliers stay below 2**61 so products fit in int64 comfortably
        # even before wraparound; add 1 so no multiplier is zero.
        self._a = rng.integers(0, 2**61, size=num_perm).astype(np.uint64) + np.uint64(1)
        self._b = rng.integers(0, 2**61, size=num_perm).astype(np.uint64)

    def signature(self, shingle_set: Iterable[str]) -> np.ndarray:
        sig = np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        hashes = [_hash_shingle(shingle) for shingle in shingle_set]
        if not hashes:
            return sig
        hashed = np.array(hashes, dtype=np.uint64)
        values = self._a[:, np.newaxis] * hashed[np.newaxis, :] + self._b[:, np.newaxis]
        return values.min(axis=1)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    if sig_a.shape != sig_b.shape:
        raise ValueError("signatures must have the same shape")
    return float(np.mean(sig_a == sig_b))


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        root = index
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[index] != root:
            self.parent[index], index = root, self.parent[index]
        return root

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


@dataclass(frozen=True)
class NearDedupResult:
    """Output of :class:`NearDedup`; the threshold is always recorded."""

    kept: tuple[DecisionExample, ...]
    removed_ids: tuple[str, ...]
    per_source: dict[str, int] = field(default_factory=dict)
    threshold: float = DEFAULT_NEAR_THRESHOLD
    num_perm: int = NUM_PERM
    candidate_pairs: int = 0

    @property
    def near_removed(self) -> int:
        return len(self.removed_ids)


class NearDedup:
    """Remove near-duplicates at an estimated Jaccard ≥ ``threshold``.

    Pair generation is naive and O(n²) — see the module docstring. An LSH
    (e.g. banded MinHash) implementation is a drop-in replacement for
    :meth:`_candidate_pairs`; nothing else in the class depends on how pairs
    are produced.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_NEAR_THRESHOLD,
        *,
        num_perm: int = NUM_PERM,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within 0..1")
        self.threshold = threshold
        self._hasher = MinHasher(num_perm=num_perm, seed=seed)

    def _candidate_pairs(
        self, signatures: Sequence[np.ndarray]
    ) -> Iterator[tuple[int, int]]:
        """Yield all ``(i, j)`` with ``i < j``. O(n²); fixture scale only."""
        total = len(signatures)
        for i in range(total):
            for j in range(i + 1, total):
                yield i, j

    def run(self, examples: Sequence[DecisionExample]) -> NearDedupResult:
        signatures = [
            self._hasher.signature(shingles(canonical_document(example)))
            for example in examples
        ]
        union_find = _UnionFind(len(examples))
        pair_count = 0
        for i, j in self._candidate_pairs(signatures):
            pair_count += 1
            left, right = examples[i], examples[j]
            # Rows with different gold labels are contradiction candidates,
            # not duplicates: merging them would silently choose a label.
            if gold_signature(left) != gold_signature(right):
                continue
            if estimate_jaccard(signatures[i], signatures[j]) >= self.threshold:
                union_find.union(i, j)

        cluster_kept: dict[int, int] = {}
        removed_ids: list[str] = []
        per_source: dict[str, int] = {}
        removed_indexes: set[int] = set()
        for index in range(len(examples)):
            root = union_find.find(index)
            if root in cluster_kept:
                removed_indexes.add(index)
                removed_ids.append(examples[index].example_id)
                source = examples[index].source
                per_source[source] = per_source.get(source, 0) + 1
            else:
                cluster_kept[root] = index

        kept = tuple(
            example for index, example in enumerate(examples) if index not in removed_indexes
        )
        return NearDedupResult(
            kept=kept,
            removed_ids=tuple(removed_ids),
            per_source=per_source,
            threshold=self.threshold,
            num_perm=self._hasher.num_perm,
            candidate_pairs=pair_count,
        )
