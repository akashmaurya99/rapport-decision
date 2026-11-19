# TASK-08 notes — Deduplication, quality filtering and contradiction audit

Status: done. Owner files only: `src/data/clean/**`, `tests/unit/data/clean/**`
(TASK-03 / TASK-07 files untouched). `datasketch` is NOT installed, so the
MinHash is implemented from numpy (128 perms, char 5-gram shingles).

## Module map
- `src/data/clean/dedup.py` — `ExactDedup`, `NearDedup`, `MinHasher`,
  `exact_key`, `group_key`-style helpers (`canonical_document`, `shingles`,
  `jaccard_index`, `estimate_jaccard`, `gold_signature`).
- `src/data/clean/quality.py` — Q1–Q6 as module-level pure predicates
  (`q1_short_option` … `q6_punctuation_only_or_placeholder`), registry in
  `RULES` (evaluation order), applied by `QualityFilter` into a shared
  `RejectionReport`. No rule references difficulty, scores, embeddings, or a
  learned model — structural only.
- `src/data/clean/contradiction.py` — `ContradictionAudit` + `write_contested`
  (JSONL via `to_dict`); grouping key excludes `chosen_index`.
- `src/data/clean/report.py` — `RejectionReport` (named counters,
  `total_rejected` property = sum of per-rule counts; `verify(removed_rows)`
  raises `ReportMismatchError` on mismatch) and the frozen `CleanReport`
  (`exact_removed`, `near_removed`, `per_rule`, `contested`, `kept`).
- `src/data/clean/pipeline.py` — `clean_examples(...)`: exact → near →
  contradiction → quality (strategy order), returns `CleanResult` carrying
  `CleanReport`, `dedup_threshold`, `contested_ids`, optional `contested_path`.

## Design decisions (contract left these open)
- **Exact-dedup key** = `(task_family, goal, state, sorted option texts,
  chosen_index)` over the TASK-03-normalized fields (“SFD” read as
  “normalized as per TASK-03”; `src.data.core` is imported, not reimplemented).
  `source` and `group_id` are NOT in the key (per the contract), so
  content-identical rows coming from two sources are one duplicate; the
  removed row is attributed to its own source in `per_source`.
- **First occurrence wins** in input order for both dedups — deterministic.
- **Near dedup**: canonical document = `goal\nstate\n` + newline-joined
  *sorted* option texts → option-order invariant shingles. Char 5-grams;
  short texts (<5 chars) shingle to themselves. Hashing: sha256 → 64-bit le
  ints; signature = row-wise min of `(a*h + b) mod 2**64` with uint64
  wraparound (numpy universal family, no prime modulus needed). Threshold
  default 0.9, recorded on `NearDedupResult.threshold` and propagated to
  `CleanResult.dedup_threshold`. `_candidate_pairs` is the documented O(n²)
  seam for a drop-in banded-LSH; fixture scale only — replace before large
  runs.
- **Pair-acceptance guard uses gold option TEXT** (`gold_signature`), not the
  raw index: identical content with conflicting gold is a contradiction
  candidate and is never merged (silently choosing a label is exactly what
  criterion 3 forbids) — but the same gold text reached through a *permuted*
  option list still dedups. Contested groups reach `ContradictionAudit`
  untouched.
- **QualityFilter assigns each rejected row to exactly one rule** (first
  failing rule in registry order), so `sum(per_rule.values()) == removed rows`
  by construction; `QualityFilter.run` calls `report.verify(...)` and tests
  assert the invariant. Scores/difficulty are never read (test: extreme but
  in-range utilities survive).
- **Q1 applied literally** from the strategy ("empty or <3-token option",
  tokens = whitespace splits) — this rejects single-word options like “Yes”;
  flagged here because it is the literal spec wording, worth review before
  the mixture run. **Q5 tokens are whitespace-delimited**; the strategy's real
  512-token budget is enforced by the tokenizer (TASK-04).
- **Q6**: rejects option with no Unicode alnum (punctuation-only) or a
  placeholder from `PLACEHOLDER_OPTIONS` (`n/a`, `na`, `tbd`, `tba`, `todo`,
  `xxx`, `unknown`, `placeholder`); `REAL_OPTIONS` ("none of the above",
  "all of the above", "none of these", "all of these", "i don't know",
  "cannot be determined") are never rejected.
- **Q4** = chosen_index present but not an int / out of range; missing
  chosen_index is *not* out-of-range (contract question, not a rule's).
- **Contradiction audit**: group by `(task_family, goal, state, sorted option
  texts)`; if the set of non-None golds has >1 value, the WHOLE group
  (including any unlabeled rows) is quarantined to the contested list. No
  majority vote anywhere. Written as JSONL when `contested_path` is given;
  reported via count + `contested_ids`.
- Stages pass tuples in input order; `clean_examples` is deterministic.

## Verification
`pytest tests/unit/data -q` 324 passed (clean: 57); full suite
`pytest tests/unit -q` 637 passed, 1 skipped; `ruff check` and
`mypy src/data/clean` clean. TASK-03/TASK-07 files not edited.
