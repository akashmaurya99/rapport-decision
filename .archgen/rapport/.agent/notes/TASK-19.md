# TASK-19 notes — Robustness harness: permutation, option-count sweep, held-out source

Done: 2026-11-19. Status: done (single wave, one pass). No git run.

## Files shipped
- `src/evaluation/robustness.py` (new)
- `tests/unit/evaluation/test_robustness.py` (new, 24 tests)

Nothing else touched (no `src/evaluation/__init__.py`; namespace packages, as
TASK-17/TASK-18). `tasks.yaml` not hand-edited (status via `archgen_tasks`).

## Public surface (FR-EVAL-03 / E7)
- `run_robustness(model, tokenizer, split_rows_by_family, split_manifest,
  ood_rows, bucket_fn, seed=20261119) -> RobustnessReport`
- `RobustnessReport` — frozen dataclass; `to_dict()` JSON-safe (round-trips
  through `json.dumps`); blocks: `permutation`, `option_count_sweep`, `ood`,
  `context`, `length_sweep` — all grouped by family plus a pooled `overall`
  (except `ood`, which is a dedicated block with its own pooled metrics).
- `RobustnessInputError` (`ROBUSTNESS_INPUT_INVALID`); field named in `.field`.
- `length_extension_verdict(acc512, acc1024) -> "PASS"|"FAIL"|"not-measured"`.
- Constants: `DEFAULT_SEED=20261119`, `LENGTH_BUDGETS=(512,768,1024)`,
  `LENGTH_EXTENSION_TOLERANCE=0.02`, `SCORE_TOLERANCE=1e-4`,
  `PERMUTATION_SAMPLE_SIZE=8`, `ABSTENTION_THRESHOLD=0.75`.

## Contract decisions (design notes the contract left open)
- **Evaluation corpus rule.** `split_rows_by_family` maps family → rows; rows
  with `split == "train"` are NEVER evaluated. They are used only as the source
  cross-check for the held-out-source block (sources contributing training rows
  are excluded from the `ood` block and counted in `n_excluded`). Everything
  else is evaluation corpus (typically the test split); per family, with
  `overall` pooled across families.
- **OOD block.** `empty: true` with `None` metrics and a `note` when no eligible
  rows remain (including empty input) — never fabricated. Fingerprint echoed;
  the split manifest carries no source list, so the cross-check is row-based;
  no fingerprint→source cross-check is possible or faked.
- **Abstention** is defined on the confidence head: abstain iff confidence
  `< 0.75` (restated constant, derives from the declared default utility triple
  correct=1/wrong=−3/abstain=0; `src.calibration` is never imported — all
  metrics come from `src.evaluation.metrics`).
- **Bucket ECE** = `src.evaluation.metrics.ece` over `(confidence, correct)`
  pairs, fixed bin edges (`ECE_BINS=15`). Bucket label = `bucket_fn(n)`
  (injected; harness stays agnostic of bucketing scheme).
- **Permutation** (E7): all `n!` for `n ≤ 4`; 8 seeded samples for `n > 4`
  (`random.Random(f"{seed}:{family}:{n}")`, sampled set sorted → canonical).
  Deviation compares `scores_perm[j]` against base scores of option `perm[j]`
  (order-invariant example_id ⇒ base rows reusable). Accuracy delta =
  max−min across permutations (chosen rows only). Measurement is RECORDED with
  `score_tolerance=1e-4` + `within_tolerance`, never asserted in the harness.
- **Contradictory** transform: gold option's Option (text+label) swaps with the
  highest-gold-utility non-gold option (`row.scores`; ties → lowest index;
  no scores → first non-gold). `chosen_index` deliberately NOT remapped (label
  contradicts content); content multiset unchanged so `example_id` stays valid.
  Rows without `chosen_index` skipped (`n_skipped_no_gold`).
- **Missing-information** transform: decisive sentence removed — max word-token
  overlap with the gold option text (ties → earliest; no overlap → final
  sentence, documented fallback). `example_id` recomputed (state changed);
  emptying the state is counted (`n_state_emptied`).
- **Length sweep** re-encodes the same examples at 512/768/1024 via
  `encode_example(budget=…)`. Drop 512→1024 > 2 points sets
  `length_extension="FAIL"` (the 1024 RoPE extension claim), else `"PASS"`;
  `"not-measured"` when either accuracy is undefined. The comparison absorbs
  binary FP noise (`0.9−0.88 > 0.02`) with a 1e-9 epsilon (repo convention).
- **Rows without chosen_index** are scored and counted, but excluded from
  accuracy/ECE (`n_scored_rows` vs `n_rows`, metrics `None`) — matches the
  plan's edge case (permutation of a row without chosen index: still reported).
- **Families with zero evaluation rows** → reported `n_rows=0`, `None` metrics.
- **Empty-state corpus** guards: scoring missing → `n_rows=0` entries for all
  of 2..8 and all budgets; `overall` still aggregates reported correctly.
- **Side effects:** `TRUNCATION_COUNTS` global snapshotted and restored around
  the run (reported instead as `n_truncated` per budget); model temporarily set
  to `eval()` and restored afterwards; deterministic for fixed seed.
- **Score cache** keyed by `(example_id, option texts in slot order, budget)`
  makes the run single-pass; safe because `example_id` is order-invariant
  (permutations share it but the option-order tuple disambiguates).

## Verification
- `pytest tests/unit/evaluation/test_robustness.py -q` → 24 passed
- `pytest tests/unit/evaluation -q` → 144 passed (no regressions)
- `ruff check` (both files) → clean; `mypy src/evaluation/robustness.py` → clean
- JSON round-trip verified in a test and by manual dump.

## Stubs used in tests
`_HashModel` (score = sha256 of the option block's token ids → exactly
permutation-invariant ⇒ deviation must be 0.0) and `_PositionalModel` (score =
slot index only ⇒ deviation > 1e-4 ⇒ `within_tolerance` False). E7 logic is
tested deterministically; the tiny ultra RapportModel fixture exercises the
real path (RoPE is inherently positional, so its deviation is recorded, not
asserted — observed ≈2e-5 to 4e-5 on the fixture).
