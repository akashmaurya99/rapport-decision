# TASK-09 notes — Six task-family adapters into the unified schema

Done: 2026-11-19. Single wave; siblings TASK-03/TASK-05 untouched.

## Files shipped

- `src/data/adapters/base.py` — `Adapter` protocol (`@runtime_checkable`) with the single
  method `to_decision_example(rows) -> AdapterResult`. `AdapterResult` carries `examples`,
  `rejected` (`RejectedRecord` NamedTuple `(record_index, reason_code, detail="")`), aligned
  `meta` (`ExampleMeta`, one per example) and a `report` dict. Typed codes
  `ADAPTOR_MISSING_FIELD`, `ADAPTOR_BAD_TYPE`, `ADAPTOR_NO_GOLD`, `ADAPTOR_BAD_INDEX`,
  `ADAPTOR_EMPTY_STATE`, `ADAPTOR_TOO_LONG`, `ADAPTOR_NO_OPTIONS`, `ADAPTOR_BAD_SCORE`,
  `ADAPTOR_CONTRADICTORY_LABEL`, `ADAPTOR_RISK_LABEL_LIMIT`, `ADAPTOR_META_MISSING`
  (the four contracted example codes are present verbatim, incl. the `ADAPTOR_` spelling).
  Core-contract violations flow through with their TASK-03 `DATA_*` codes as the reason
  (e.g. `DATA_DUPLICATE_OPTION`); both families of code are stable and typed.
  Shared machinery: per-row RNG derived from `sha256("rapport:adapter-shuffle:v1" || seed ||
  row_fingerprint)` (order-independent, deterministic for the recorded master seed),
  `shuffle_order`, `apply_order`, `reduce_seed`, `derive_group_id`, `cap_text` (word-boundary
  truncation), `length_cap`, `finalize` (normalize_example → first `DataError` becomes the
  rejection reason), `base_report`, `restore_original_order` (rebuilds the pre-shuffle record
  via the TASK-03 builder; maps options/scores/chosen_index/labels through the inverse
  permutation; recomputes `example_id`).
- `src/data/adapters/choice.py` — `ChoiceAdapter` (defaults `cais/mmlu`, MIT, split `train`)
  and `MmluProAdapter` (`TIGER-Lab/MMLU-Pro`, MIT, split `test`, evaluation-only). Mapping:
  `question→state`, `choices→options` (positional letters kept as source labels),
  `answer` gold as index or letter (case-insensitive), `goal` = fixed directive.
  Rows with >8 options are reduced **before** shuffling: gold + seven seeded non-gold
  distractors kept, the rest dropped (only non-gold indices ever dropped). Reduction RNG is
  content-hash seeded (`rapport:mmlu-pro-reduce:v1 || reduction_seed || row_fingerprint`),
  order-independent. 10-option rows → `meta.reduction == "mmlu-pro-10-to-8"`; other >8 counts →
  `"seeded-non-gold-removal-to-8"`. `meta.reduction_seed`, `meta.dropped_indices`,
  `meta.kept_indices` recorded on every reduced example; `report["reduced_10_to_8"]`,
  `reduced_over_8_to_8`, `reduced_other_to_8`, `reduction_seed`, `reduction_method` reported.
- `src/data/adapters/preference.py` — HH-RLHF (`Anthropic/hh-rlhf`, MIT). Prompt via
  `prompt` field, else the shared `\n\nAssistant: ` prefix extracted from the two transcripts;
  unparseable transcripts fall back to empty state (contract allows empty state for
  preference). **Both orders are emitted deterministically** — emission A
  `[chosen, rejected]` chosen_index=0, emission B `[rejected, chosen]` chosen_index=1
  (the "both orders, chosen=0" rule); no RNG shuffle on top (the two permutations are the
  augmentation), `meta.shuffle_seed` is None, `meta.original_order` records the trivial
  permutation. Both emissions share `group_id`. Overlong prompt/state truncated at word
  boundary to `max_state_chars` (default 8000) with `meta.truncated=True`; beyond
  `max_field_chars` (100k) → `ADAPTOR_TOO_LONG`.
- `src/data/adapters/scoring.py` — HelpSteer2 (`nvidia/HelpSteer2`, CC-BY-4.0). The three
  *quality* attributes (helpfulness/correctness/coherence, ratings 0–4) are the options;
  `rating/4 ∈ [0,1]` are the `scores` (one utility float per option, positionally aligned).
  `complexity`/`verbosity` are deliberately excluded as non-utility descriptive axes —
  the exclusion is reported (`report["excluded_attributes"]`), never silent. Scores permute
  with the shuffled options; restore round-trips.
- `src/data/adapters/probability.py` — one adapter, two admitted shapes:
  (1) auxiliary_train-style (`question`+`choices`+`answer`) → **one derived row per option**:
  `"Is '<option>' the correct answer?"`, options `[Yes, No]` shuffled at ingest,
  `chosen_index` = position of gold, `answerable = (option is gold)` (both values appear;
  exactly one `answerable=True` row per parent), all derived rows share the parent's
  `group_id` (can never straddle a split). Derived count reported as
  `report["derived_binary_rows"]`.
  (2) TruthfulQA-style (`question`+`best_answer`[+`incorrect_answers`]) → one Yes-gold row
  for `best_answer` and one No-gold row for `incorrect_answers[0]`; counts in
  `truthfulqa_yes_rows` / `truthfulqa_no_rows`. Anything else → `ADAPTOR_MISSING_FIELD`.
- `src/data/adapters/abstention.py` — AbstentionBench-derived rows
  (`subset`, `question`, optional `context`, `answerable` bool or `unanswerable` bool;
  both required to agree else `ADAPTOR_CONTRADICTORY_LABEL`). Options
  `[Answerable, Unanswerable]` shuffled; `chosen_index` tracks the gold; `answerable`
  carries it. **Per-subset conversion rates are reported in
  `report["per_subset"][subset] = {input, converted, rejected, conversion_rate}` and are
  deliberately never pooled** (the 20 sub-datasets have heterogeneous unanswerable
  definitions); rejections whose subset is missing/invalid are counted in
  `report["unbucketed_rejections"]`, not in any subset's rate. No canonical subset list is
  declared in this repo, so none is hard-coded; exactly the subsets present in the input are
  reported. Example-level provenance in `meta.subset`.
- `src/data/adapters/risk.py` — Aegis 2.0 (`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`,
  CC-BY-4.0). `prompt` (+optional `response`) → state; `categories` (required; explicit
  empty list = annotated-safe → `risk_labels == ()`; absent → `ADAPTOR_MISSING_FIELD`,
  unannotated ≠ safe); dedup first-wins; >8 labels → `ADAPTOR_RISK_LABEL_LIMIT`; membership
  unvalidated (taxonomy not declared in this repo — TASK-03 ruling). Options
  `["safe", "unsafe"]` (contract ≥2 options); the gold lives in `risk_labels`, so
  `chosen_index` is unset by design.
- `src/data/adapters/__init__.py` — exports (protocol, result types, codes, six/seven
  adapters, `restore_original_order`).
- `tests/unit/data/adapters/` — 116 tests, all offline (in-memory fixtures, no `datasets`
  import, no network).

## Contract rulings (gaps the contract left open, decided here)

1. **`meta` cannot live on `DecisionExample`.** The schema is frozen with a fixed field set
   (`example_id, task_family, goal, state, options, source, license, group_id, split,
   chosen_index, scores, answerable, risk_labels`), serialization rejects unknown keys, and
   `src/data/core/**` is TASK-03's (read-only here). So `meta.*` required by the contract
   (`reduction_seed`, `original_order`, …) is delivered as `ExampleMeta` in
   `AdapterResult.meta`, aligned by position with `examples`. Reconstruction goes through
   `base.restore_original_order(example, meta)`; nothing is written into the serialized record.
2. **`example_id` is permutation-invariant** (hash of sorted option texts, TASK-03), so the
   two preference emissions of one pair share an `example_id`, as do the identical-text
   derived Yes/No rows of identical aux rows. This is inherent to the contracted id. Flag
   for **TASK-08**: exact dedup keyed on `example_id` alone would collapse the both-orders
   preference augmentation (and same-content Yes/No rows with different chosen) — dedup must
   key on `(example_id, chosen_index)` for those families, or keep both orders explicitly.
   Flag for **TASK-10**: split by `group_id` (pairs share it) — derived rows inherit the
   parent's `group_id` by construction, which is the required leakage property, but the
   shared id must not be treated as a row-level unique key in split manifests.
3. **Probability `answerable` = proposition truth value** (`True` iff the queried option is
   the gold), the only boolean slot the contract offers on a non-abstention row; the gold
   also lives in `chosen_index` over `[Yes, No]` (consistent pair). This makes derived rows
   double-encoded but consistent, and satisfies the note "with answerable=true/false".
   Probability rows are not abstention-family rows, so the abstention loss must be keyed off
   `task_family` (or off `answerable` only within the abstention family) — if TASK-11 masks
   on bare `answerable is None` across families, all-Yes/mostly-No rows would pollute the
   answerability head; flag it there in review.
4. **Preference is NOT RNG-shuffled at ingest.** The contracted "both orders" emission is a
   deterministic two-permutation augmentation; stacking a random shuffle on it can collapse
   both emissions to the same post-shuffle order (duplicate rows, same permutation-invariant
   `example_id`) and defeats the augmentation. `meta.shuffle_seed=None`,
   `report["shuffle"]="deterministic-both-orders"`; `original_order` still recorded, as
   retention requires. All other families shuffle options at ingest with the recorded seed.
5. **Scoring keeps 3 of 5 HelpSteer2 attributes.** complexity/verbosity are not
   better/worse scales; mapping them to "utility" would fabricate signal. Exclusion is
   reported, not silent. Options = the three quality aspects (2–4 options per the strategy
   doc), scores = rating/4 ∈ [0,1] aligned positionally.
6. **Risk safe rows carry `risk_labels == ()`** (annotated "touches nothing"), and
   unannotated rows (absent `categories`) are rejected. **Flag for TASK-11**: if the risk
   loss is masked on "no risk labels present", explicitly-safe rows are masked too — the
   all-zero target needed to train over-refusal would never be seen. The adapter records the
   distinction as deliberately as it can: absent → rejected; `()` → annotated safe.
7. **Reduction accounting.** 10-option rows report under `reduced_10_to_8` with method
   `mmlu-pro-10-to-8`; any other >8 count reduces by the same seeded rule and reports under
   `reduced_other_to_8` / `reduced_over_8_to_8` with method `seeded-non-gold-removal-to-8`.
   Gold is structurally undroppable; labels (source letters) follow their option through
   reduction and shuffle.
8. **`original_order` refers to the kept list** for reduced rows (a true permutation of
   `range(len(options))`, so `restore_original_order` works uniformly); `meta.kept_indices`
   maps kept positions back to source indices and `meta.dropped_indices` lists the two
   removed source indices. Full source reconstruction = restore + kept/dropped indices.
9. **Shuffle/reduction seeds derive from row *content*, not batch position**
   (`sha256(domain || master || fingerprint)`), so ingest order never changes the output.
   The master seed is in `report["shuffle_seed"]`; the per-row derived seed in
   `meta.shuffle_seed`.
10. **AbstentionBench has no repo-declared 20-subset list**, so none is invented; rates are
    computed per `subset` present in the input, never pooled. The license string is
    `"AbstentionBench (arXiv 2506.09038) — see paper"` (the Hub/paper declare no SPDX id);
    non-empty, so the contract's license slot is satisfied honestly.
11. **Derived rows share `record_index` of the parent** in `RejectedRecord`; derived
    examples are built in a local buffer and only emitted when *all* of them pass
    validation — a late failure rejects the parent without half-emitting it.
12. **Truncation policy:** only the preference family truncates (word boundary,
    `max_state_chars`, flagged). Every other family rejects an overlong field with
    `ADAPTOR_TOO_LONG` rather than silently cutting content.
13. **Absent vs empty vs wrong-type are distinct rejections:** absent → `ADAPTOR_MISSING_FIELD`,
    wrong JSON type → `ADAPTOR_BAD_TYPE`, present-but-empty (where a non-empty value is
    required) → `ADAPTOR_EMPTY_STATE`; `answer`/`gold` unparseable → `ADAPTOR_NO_GOLD`;
    gold index out of range → `ADAPTOR_BAD_INDEX`; overlong → `ADAPTOR_TOO_LONG`;
    >8 risk labels → `ADAPTOR_RISK_LABEL_LIMIT`; contradictory abstention labels →
    `ADAPTOR_CONTRADICTORY_LABEL`; NaN/out-of-range 0–4 rating → `ADAPTOR_BAD_SCORE`.
    Non-mapping rows → `ADAPTOR_BAD_TYPE`. Nothing is silently skipped.
14. **MMLU-Pro is evaluation-only** and labelled as such; `MmluProAdapter` defaults to
    split `test`. The training-admitted `cais/mmlu` auxiliary_train is `ChoiceAdapter`'s
    default. No live corpora are read anywhere; `datasets` is never imported.

## Verification

- `.venv/bin/python -m pytest tests/unit/data/adapters -q` → **116 passed** (115 + the 9-option generic-reduction case).
- `.venv/bin/python -m pytest tests/unit -q` → **515 passed, 1 skipped** (sibling suites untouched).
- `.venv/bin/python -m ruff check src/data/adapters tests/unit/data/adapters` → clean.
- `.venv/bin/python -m mypy src/data/adapters` → no issues (8 files).
- Adversarial spot checks: tuple rows, whitespace-only subset (unbucketed), dual-agreeing
  abstention labels, NaN rating, transcript extraction, kept-list restore after 10→8
  reduction, gold retention across seeds, shared derived `group_id`.

## Blockers / honest caveats

- The three `meta`-vs-`DecisionExample` tensions above (rulings 1, 3, 6) are resolved by
  reporting/encoding choices, not by changing the TASK-03 contract — the schema is frozen
  and core was out of scope.
- AbstentionBench's 20 sub-dataset names come from the paper behind arXiv 2506.09038; this
  repository declares no canonical list, so per-subset reporting is input-driven. If the
  orchestrator wants the 20 literal names validated against the paper, that is a docs/data
  decision (and needs network), not an adapter one.
- Risk category *membership* remains unvalidated by design (TASK-03 ruling stands).
- `PreferenceAdapter` emits two rows per pair; mixture weights/quotas are TASK-13's, the
  both-orders convention is documented here rather than hidden in a flag.
- `example_id` collisions across preference emissions and identical derived Yes/No rows are
  inherent to the permutation-invariant id (ruling 2) — surfaced here for TASK-08/TASK-10,
  not fixed by mutating ids.
- No `datasets` import, no network, no parquet; nothing outside owned files was edited.
