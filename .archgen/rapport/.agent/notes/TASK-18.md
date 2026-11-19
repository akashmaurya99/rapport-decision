# TASK-18 notes — Classical and frozen-embedding baselines

Done: 2026-11-19. Status: done (single wave, one pass). No git run.

## Files shipped
- `src/evaluation/baselines.py` (new)
- `tests/unit/evaluation/test_baselines.py` (new, 40 tests)

Nothing else touched. `src/evaluation/__init__.py` deliberately NOT created
(namespace packages, consistent with TASK-17). `src/data/splits/` consumed, not
edited. `tasks.yaml` not hand-edited (status via `archgen_tasks`).

## Public surface (FR-EVAL-02)
- `BaselineRunner` — ABC; shared `run(train_rows, dev_rows, test_rows, split_manifest_fingerprint) -> BaselineMetrics`.
  Subclasses implement `fit(scored_train_rows)` + `predict_distribution(rows)`.
- `MajorityClassBaseline` (`B1-majority`), `TfidfLogisticBaseline` (`B3-tfidf-logreg`),
  `FrozenEmbeddingCosineBaseline` (`B4-frozen-embedding-cosine`).
- `BaselineMetrics` — frozen dataclass; `to_dict()` JSON-safe. Dev/test pooled
  blocks plus per-family blocks (`dev_per_family`/`test_per_family`), both
  `MetricResult` (imported from `src.evaluation.metrics`, never redefined).
  `skipped` = split → reason → count (family keys + `no-chosen-index`).
  `notes: tuple[str, ...]` records every judgment call in the run.
- `AbstentionRandomScore` — `n_rows/n_scored/accuracy/expected_random_accuracy/seed`.
- `y1_verdict(model_acc, baseline_acc, margin=8.0) -> "pass"|"fail"` — pure,
  fail-closed on missing input; out-of-domain raises `BaselineInputError`.
  Accuracies are fractions in `[0, 1]` (as emitted by `metrics.accuracy`);
  margin is percentage points (default `Y1_MARGIN_POINTS = 8.0`). A 1e-9-point
  tolerance absorbs binary float noise around the pre-registered boundary.
- `run_baselines(...)` — convenience entry point (all three baselines, one manifest).
- `load_manifest_fingerprint(path)` — consumes the split-manifest JSON path
  (strict parse via `read_manifest`), returns its validated sha256 fingerprint.
- `BaselineInputError` (`BASELINE_INPUT_INVALID`), `BaselineError` base.

## Design decisions / contract honouring
- **Same manifest as the model.** `run` requires a lowercase sha256 hex
  fingerprint (validated with `is_sha256_hex`), echoes it on the result, and
  enforces `split` tags (`train_rows` → `split="train"`, etc.), so a manifest
  cannot be echoed for another split's rows.
- **Scored stream = choice + preference rows with `chosen_index`.** All other
  families skipped (counted by family); scored-family rows without a chosen
  index counted under `no-chosen-index`; a chosen index out of range raises
  (fail-fast, contract violation).
- **Abstention never enters the metric stream.** It is scored separately with a
  uniform-random-over-options default (`random.Random(0)`); `expected_random_accuracy`
  is the closed form `mean(1/n_options)`; recorded as `dev_abstention`/`test_abstention`.
- **Metric schema is `MetricResult` everywhere** (pooled + per-family, dev + test);
  everything uncomputable is `None` (e.g. risk metrics / over-refusal — classical
  baselines have no risk head). `scores` carry per-option probabilities
  (documented: raw baseline scores have no utility scale; within-row ranking
  preserved). `gold_scores` = row utilities when present, else one-hot at
  `chosen_index` (documented stand-in so NDCG/Spearman/Pearson/utility-MAE stay defined).
- **B3.** Joint text `goal\nstate\noptions`. Features: word 1–2-grams + char 3–5-grams,
  namespaced (`w1:`/`c3:`…), CRC-32 hashed to `TFIDF_FEATURE_DIMS = 50_000`, binarized
  document presence, smooth IDF `log((1+N)/(1+df)) + 1`, L2-normalized rows.
  Softmax logistic regression **in numpy** (scikit-learn deliberately not used for it, so
  solver behaviour cannot drift with sklearn versions; sklearn *is* declared as a dependency
  but unused here): full-batch GD, zero init, objective
  `mean cross-entropy + ‖W‖²/(2·C·N)` with fixed `C = LOGREG_C = 1.0`, `lr = 0.5`,
  deterministic cap `LOGREG_MAX_ITERS = 200`, early stop at relative objective improvement
  < `LOGREG_TOL = 1e-9`. Convergence is **recorded, not asserted** — per-bucket
  `converged=`/`n_iters=`/`final_objective=` lines land in `notes`; on the 40-row
  fixture the 2-option bucket converges at iter 178. One model per option-count
  bucket (option indices not comparable across buckets); unseen bucket → train-global
  class frequencies fallback (uniform if train empty), recorded in notes.
- **B4.** Query = `goal + state`, options embedded separately; cosine; softmax temperature
  fixed at 1.0. bge-small path loads **local-only** (`local_files_only=True`, mean pooling
  over attention mask, L2-normalized) — verified on this machine: bge files are NOT in the
  HF cache, load raises `OSError`, and the runner falls back to the deterministic hashed
  char 3–5-gram embedding (numpy, `HASH_EMBED_DIMS = 384`, the bge hidden size). Output
  labels it explicitly: `embedding_source="hash-ngram"` vs `"bge-small-en-v1.5"`.
  Explicit `embedding_source="bge"` with bge unavailable raises `BaselineInputError`
  (never silently downgraded); `embedding_source="hash"` forces the fallback; default
  `"auto"` tries bge, falls back + records the reason. Zero network anywhere.
- **B1.** Majority per family, refined per option-count bucket with family-pooled,
  then global-pooled, then uniform fallback. Ties → lowest option index (argmax picks
  first max). Probabilities are empirical train class frequencies (sum to 1).
- **scipy** is used ONLY to hold the binarized TF-IDF matrix as CSR (ships with the
  declared scikit-learn dependency). If scipy is absent, a dense numpy matrix is used
  (hard cap ~2^26 cells; beyond that a typed error tells the user to install scipy).
  All fitting math is numpy/scipy-sparse regardless.

## Fixtures / verification
- 40 synthetic rows (24 train / 8 dev / 8 test; 14 choice + 4 preference train etc.),
  split tags enforced and consistent with the manifest.
- `.venv/bin/python -m pytest tests/unit/evaluation/test_baselines.py -q` → **40 passed**.
- `ruff check src/evaluation/baselines.py tests/unit/evaluation/test_baselines.py` → clean.
- `mypy src/evaluation/baselines.py` → clean. Full evaluation dir → 120 passed (80 metrics + 40 baselines).
- `tests/unit` overall: 978 passed with `--ignore=tests/unit/data/mixture`;
  collection of `test_loader.py` under `tests/unit` aborts the whole run with an
  **import-file mismatch** vs `tests/unit/config/test_loader.py` (duplicate basename,
  no `__init__.py`; also reproduced after deleting all `__pycache__` — it is a
  pre-existing repo condition, unrelated to this task; nothing outside ownership was
  touched to fix it).

## Honest caveats / notes for downstream tasks (TASK-21)
- `run_baselines` is provided as the named entry point (`architecture.yaml` lists
  `run_baselines`); it constructs fresh baselines per call — instantiate once per arm.
- The Y1 gate compares accuracies in `[0, 1]` fractions with the margin in points;
  callers must not pass percentages. `y1_verdict(None, acc)` == `"fail"` (fail-closed);
  report should render `not-measured` separately when accuracies are missing.
- B3's NDCG/Spearman/utility-MAE rest on one-hot stand-in gold utilities for
  choice/preference rows; interpret those ranking numbers accordingly.
- B4's cosine distribution is temperature-1.0; no per-bucket calibration was fitted
  (calibration fitting is TASK-16's job, on dev).
- Per-family `MetricResult` blocks exist for `choice`/`preference` only (the labeled
  stream); other families are counted in `skipped`.
