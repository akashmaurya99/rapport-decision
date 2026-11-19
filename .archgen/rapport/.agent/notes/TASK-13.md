# TASK-13 notes — Declarative mixtures and experiment configs

Done: 2026-11-19. Single wave; TASK-03/TASK-07/TASK-09/TASK-10 untouched (only read). No git run.

## Files shipped

- `src/data/mixture/errors.py` — `MixtureError(DataError)` base `MIXTURE_ERROR`.
  Typed, fatal (never warnings): `MixtureInputError` (MIXTURE_BAD_INPUT),
  `MixtureFamilyError` (MIXTURE_BAD_FAMILY — not one of the six task families),
  `MixtureSourceError` (MIXTURE_BAD_SOURCE — missing/invalid entry, duplicate
  source within a family), `MixtureFingerprintError` (MIXTURE_FINGERPRINT —
  absent or not a lowercase sha256 hex digest),
  `MixtureFingerprintMismatchError` (MIXTURE_FINGERPRINT_MISMATCH — carries
  `expected`/`actual`), `MixtureConfigError` (MIXTURE_CONFIG_INVALID — YAML
  file), `MixtureConfigFieldError` (MIXTURE_CONFIG_FIELD — unknown/missing/
  mistyped field, `field=`).
- `src/data/mixture/mixture.py` — `MixtureSource` (declarative allocation
  `{source, rows, weight?}`), `MixtureSlot`, `MixtureOrder`, and
  `sample_mixture(sources, weights, seed, fingerprint) -> MixtureOrder`.
  `sources` is the raw declarative mapping `family -> entry | [entries]`
  **or** a `MixtureDeclaration` (pinning check). No RNG anywhere: each slot
  gets `sha256(canonical_json([domain, fingerprint, seed, family, source,
  index]))` (`rapport:mixture-slot:v1`) and all slots are sorted by that key
  (ties totally ordered by family/source/index), so the assignment is
  byte-identical on any platform and independent of declaration iteration
  order. `example_ids()` returns the ordered ids the trainer consumes
  (`mixture:<fp16>:<family>:<source>:<index>`); two calls with the same
  `(seed, fingerprint)` produce identical ordered ids (tested, incl. the
  packaged presets). `MixtureOrder` records per-family rows/shares, effective
  weights, expected (weight) shares and abs share error — audit only, never a
  gate. Weight semantics: `rows` is the cap; `weights` argument overrides
  per-entry weights per family; families it doesn't name fall back to the
  (common, must-agree) entry weight, else row-proportional default. `weights`
  naming an unknown or absent family is an error (typo guard).
- `src/data/mixture/loader.py` — `load_mixture(path)`,
  `parse_mixture_document`, packaged presets via `load_preset`/`preset_path`
  (`starter`, `full`) from `src/data/mixture/presets/`. Top-level keys:
  `name?`, `fingerprint?` (`""` = unresolved at acquisition time; else must
  be sha256 hex), `families` (required). Unknown keys/families/fields are
  typed errors. `MixtureDeclaration.sample(seed, fingerprint=None)` refuses to
  sample while the pinned fingerprint is unresolved and raises
  `MixtureFingerprintMismatchError` on any other fingerprint.
- `src/data/mixture/presets/starter.yaml` — 55,000 rows (choice 12,000
  mmlu-auxiliary_train · preference 20,000 = HH-RLHF 16,000 + HelpSteer2
  4,000 · scoring 5,000 HelpSteer2 · probability 4,600 = TruthfulQA 1,600 +
  auxiliary_train-derived-binary 3,000 · abstention 6,600 · risk 6,800 Aegis).
  Probability split is NOT pinned by docs/dataset-strategy.md §6; TruthfulQA
  kept whole (1,600) with 3,000 derived rows to hit exactly 4,600.
  UltraFeedback rows: zero.
- `src/data/mixture/presets/full.yaml` — 115,600 rows (preference 33,000 =
  HH-RLHF 23,000 + HelpSteer2 10,000 · choice 30,000 mmlu auxiliary_train ·
  risk 18,000 Aegis · abstention 12,000 AbstentionBench-derived · probability
  11,600 = TruthfulQA 1,600 + derived-binary 10,000 · scoring 11,000
  HelpSteer2). UltraFeedback-binarized and XSTest are absent by design.
- `configs/experiments/{a1-init,a2-readout,a3-depth,a4-weights,a5-head-only,
  a6-data,a7-ordering,a8-context}.yaml` — one YAML per arm of
  docs/evaluation.md §5, `starter.yaml` (55k, Stage B), `acceptance.yaml`
  (115.6k, Stage B; records `ood_source: "UltraFeedback-binarized"` and
  `y2_comparator: "B3"`), `README.md` (arm→file mapping, knob domains,
  why ablations default to 55k). Every config names arm, init,
  `data_fingerprint: ""` (placeholder; resolved at acquisition time — key
  exists, asserted), variant (ultra/compact/large), epochs 8, batch_size 64,
  `lr: 3.0e-4` (head_lr 1.0e-3), seeds {13,29,47} (A6 seed 13 only — the one
  arm allowed a subset). A-arm conditions are in each file's `variants:` list
  with the arm under test as top-level defaults. No config computes Y-verdicts
  (TASK-21).

## Decisions / deviations to record

- **Y2 comparator**: the task contract pins `y2_comparator: "B3"`
  (TF-IDF (char+word n-grams) + logistic regression) even though
  docs/evaluation.md §8 registers Y2 against B4 (frozen bge-small + cosine)
  and Y1 against B3. Recorded verbatim from the contract; flag for T21/T20 to
  adjudicate against evaluation.md, which is the registered acceptance table.
- **Ablation data scale**: all arms default to the 55k starter (A1 is pinned
  to 55k by the doc; A6 spans 55k/115.6k/+KD/uniform-order). Acceptance
  re-runs the winning stack at 115.6k. Documented in configs/experiments/README.md.
- **A6 seed subset**: seed 13 only for all four data conditions, documented in
  the config header and README (allowed explicitly by the design notes).
- **A8 default**: `512-train-1024-infer` (doc lists it first; the RoPE
  extension arm), `512-512` as the control, both in `variants:`.
- **Shared `HelpSteer2`**: allowed in two families (10k preference + 11k
  scoring); the duplicate-source rule is per family only. Starter total
  HelpSteer2 9,000, full total 21,000 ≤ 21,362 ✓.
- Config YAMLs are experiment declarations for T21 tooling, not inputs to
  `src/config`'s flat run schema; they intentionally use `lr` and arm metadata.

## Verification

- `pytest tests/unit/data/mixture -q` — 168 passed
- `pytest tests/unit/data -q` — 583 passed (no collateral damage)
- `ruff check src/data/mixture tests/unit/data/mixture` — clean
- `mypy src/data/mixture` — clean
- Starter/full counts asserted within 1% of docs/dataset-strategy.md §6 (in
  fact exact: 55,000 / 115,600) in tests/unit/data/mixture/test_presets.py.
