# TASK-10 notes — Deterministic group-aware splits, leakage checks and fingerprints

Done: 2026-11-19. Single wave; TASK-03/TASK-07/TASK-08/TASK-09 untouched (only read).

## Files shipped

- `src/data/splits/errors.py` — `SplitError(DataError)` base, `code="SPLIT_ERROR"`.
  Typed failures, all fatal (never warnings):
  `SplitInputError` (SPLIT_BAD_INPUT) — malformed arguments;
  `GroupOverlapError` (SPLIT_GROUP_OVERLAP) — `overlap: int` + `groups: tuple[str, ...]`
  (≤5 samples), message lists the count and offenders;
  `OptionTextLeakageError` (SPLIT_OPTION_TEXT_LEAK) — `count` of violating texts,
  `texts` (≤5 samples), `pair_counts` (per split pair, `"train/dev"` style keys);
  `NGramLeakageError` (SPLIT_13GRAM_OVERLAP) — `counts` per split pair + `total`;
  `ManifestError` / `ManifestFieldError` (SPLIT_MANIFEST_INVALID / SPLIT_MANIFEST_FIELD,
  with `field=`), `ManifestFingerprintError` (SPLIT_MANIFEST_FINGERPRINT),
  `ManifestGroupOverlapError` (SPLIT_MANIFEST_GROUP_OVERLAP, `overlap=`).
- `src/data/splits/fingerprint.py` — `membership_fingerprint`: sha256 over canonical
  sorted `(example_id, split)` pairs (domain separator `rapport:split-membership:v1`,
  compact JSON); order-invariant; multiset semantics. `is_sha256_hex` validates digests.
- `src/data/splits/leakage.py` — `assert_no_group_overlap` (returns 0; raises
  GroupOverlapError), `find_shared_option_texts` / `assert_no_option_text_leak`
  (identical normalized option text, `MIN_OPTION_TEXT_TOKENS = 40`, whitespace-token
  count), `find_thirteen_gram_overlaps` / `assert_no_thirteen_gram_overlap`
  (`N_GRAM = 13`), `assert_no_leakage`. 13-grams are a double 64-bit Rabin-Karp rolling
  hash (bases 1000003/1000033) over interned token ids → 128-bit gram keys; token ids
  are assigned in input order, so identical grams hash identically within a call and
  reported counts are permutation-invariant. Pair keys are train/dev/test ordered.
- `src/data/splits/manifest.py` — `SplitManifest` (frozen dataclass) + `to_dict` /
  `from_dict` (strict: unknown/missing keys fatal) + `to_json` / `from_json` +
  `read_manifest` / `write_manifest` helpers (parents created; OSError → ManifestError).
  Fields: `fingerprint`, `counts` (per family per split, exactly train/dev/test keys,
  non-negative non-bool ints), `group_overlap` (**any non-zero value is a fatal error on
  construction, on load and from disk** — E4), `dedup_threshold` (float in [0,1]; carried
  from the clean pipeline via the `split(..., dedup_threshold=)` parameter, default 0.9
  = `DEFAULT_NEAR_THRESHOLD`), `ood_source` (required, default
  `UltraFeedback-binarized`), `withheld_source` (required, recorded explicitly — same
  identity), plus `seed`, `totals`, `ood_rows`, `ratios` (0.70/0.15/0.15). JSON storage
  layout matches the architecture.yaml SplitManifest contract (additional fields are
  additive, the declared five are all present).
- `src/data/splits/router.py` — `split(examples, seed, *, ood_source=DEFAULT_OOD_SOURCE,
  dedup_threshold=DEFAULT_DEDUP_THRESHOLD) -> SplitResult`. Assignment rule:
  (1) any group containing a row of the withheld source is pinned wholesale to the
  evaluation side (the `test` split — contract allows only train/dev/test); any other
  row sharing that group follows it, so the OOD boundary is never straddled;
  (2) remaining rows are split 70/15/15 **by row count within each family** (stratified,
  half-up targets); groups are atomic and ordered by
  `sha256("rapport:split:v1" || seed || group_id)`, water-filling into the split with
  the largest row-count deficit (ties prefer train → dev → test);
  (3) a group spanning families is assigned once (first family in sorted order) and all
  its rows follow; records get `dataclasses.replace(split=...)` — `example_id` is
  unchanged (split is not part of the content hash);
  (4) the three leakage checks run unconditionally on the final assignment — group
  disjointness (the assertion that fails the build), option-text ≥40 tokens, exact
  13-gram — all fatal typed errors listing counts;
  (5) manifest is built with fingerprint over `(example_id, split)` membership.
  `SplitResult` carries `examples`, `manifest`, `.train/.dev/.test/.ood`, `.by_split()`,
  `.fingerprint`. Pure function of `(seed, inputs)` — no RNG/wall-clock/platform
  artifacts; verified by two-calls and input-permutation tests.

## Decisions

- OOD rows land in `test` (the contract's only evaluation split); train/dev counts and
  the per-family `counts` therefore exclude them, while `test` counts (and `ood_rows`)
  include them. `ood_source` and `withheld_source` are both required and carry the same
  identity, so the withholding is never implicit.
- Assignment hash is salted with the seed; "assignment by sha256 of group_id" holds
  (the group_id is the payload) while `(seed, inputs)` fully determines the output.
- Leakage checks compare across all three splits including test; a build fails rather
  than ships a near-duplicate across the OOD boundary.
- Singleton groups give exact 70/15/15 per family (test: 100 rows/family → 70/15/15);
  larger groups approximate it up to one group's row count.
- `example_id` never changes: split is outside the content-hash payload.

## Verification

- `.venv/bin/python -m pytest tests/unit/data/splits -q` — 91 passed.
- `.venv/bin/python -m ruff check src/data/splits tests/unit/data/splits` — clean.
- `.venv/bin/python -m mypy src/data/splits` — clean.
- `.venv/bin/python -m pytest tests/unit/data -q` — 415 passed (no collateral damage).
