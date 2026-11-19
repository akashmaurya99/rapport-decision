# TASK-21 notes — Ablation arms and the single evaluation report

Done: 2026-11-19. Status: done (single pass after two review rounds).

## Files shipped
- `src/evaluation/ablate.py` — ablation arms A1–A8 adjudication (FR-EVAL-04).
  - `REQUIRED_SEEDS = (13, 29, 47)`, `ARM_IDS = A1..A8`.
  - `seed_std` (sample std, ddof=1), `claim_distinguishable` (claim printed iff
    `mean(A−B) > 1 seed-std` of per-seed diffs; <2 paired seeds → never
    distinguishable), `compare_arms` (paired per-seed comparison, JSON-safe
    claim dict with `distinguishable` flag; text renders `not distinguishable
    at n=3 seeds` otherwise).
  - Arm adjudication from pre-registered rules (docs/evaluation.md §5): A1 gap
    ≥+3 pts vs random; A2 |gap| ≤2 pts vs cross-attn; A3 within −2.0 pts of
    RAPPORT-L; A4 `weight_collapse`/`collapse_epochs ≥2` → fail, missing key →
    not-measured; A5 ≥+3 pts vs head-only; A6 `stage_gains` all ≥+1.0 pt;
    A7 ≥+1.0 pt vs uniform; A8 reuses TASK-19's `length_extension_verdict`.
    Explicit `status`/`verdict` on the entry is honoured (validated against
    pass|fail|not-measured; unknown → not-measured). Arms with no data render
    `not-measured` with the missing keys named — never omitted, never given
    fabricated numbers. Undeclared arm ids are appended after A8.
  - `evaluate_arms(ablations)` accepts a mapping `{arm: entry}` or a list of
    entries carrying `arm`/`id`/`name`. `collect_claims`, `arm_failures` helpers.

- `src/evaluation/report.py` — the single report (FR-EVAL-04/05).
  - `build_report(runs, baselines, ablations, out_dir, *, negative_results=None,
    main_group="main")` → writes `report.json` + `report.md` under `out_dir`.
    `runs` accepts: list of manifest dicts, `RunRegistry`-like (`list()`+`get()`;
    verified against the real `RunRegistry`), mapping `{id: run}`, or objects
    with a mapping `manifest` attr / `to_dict()`. Each run may carry `metrics`
    (`{family: {split: {metric: number|{"per_seed"}|{"values","seeds"}|
    {"mean","std"}|<dict payload>}}}`), `robustness` (RobustnessReport or
    dict), `benchmark` (dict), and `evidence` (verdict inputs, deep-merged
    across runs).
  - Sections: §1 manifest (config hashes, data fingerprints + consistency flag,
    seeds, device, git sha, determinism, versions); §3 per-family per-split
    mean ± std with seeds {13, 29, 47} (opaque payloads like per-category
    `risk_recall_at_fpr` rendered verbatim under "Recorded payload"); §5 arms
    A1–A8 (+claims, printed only past the 1 seed-std bar); §6 robustness blocks;
    §7 benchmark; §8 verdicts Y1–Y14; **Negative results** (mandatory).
  - `render_verdict(evidence)` returns structured `VerdictRow`s (Y1..Y14,
    blocking, detail, missing keys). `exit_code(rows, negative_results=None)` →
    1 iff any blocking row is `fail`, or `negative_results` given and empty.
    **All Y1–Y14 are treated as blocking** (fail-closed) — the
    pre-registration names no non-blocking subset. `main()` CLI
    (`python -m src.evaluation.report --bundle bundle.json --out-dir DIR`)
    exits with the report code; TASK-25's `scripts/e2e.py --report` can import
    `build_report`/`exit_code` directly.
  - Missing evidence → `not-measured` with keys named — never a silent pass.
    Y7 requires per-category recall at FPR ≤ 0.10 on held-out Aegis rows (all 8
    categories, `fpr_threshold` honoured if present) AND the keyword-rule
    comparator (`keyword_risk_baseline` imported from
    `src/evaluation/metrics.py`; accepts a `KeywordBaselineResult`-like object,
    a mapping with `recall`, raw `texts`+`risk_targets`, or the metrics-cell
    `risk_recall_at_fpr` payload). Either input missing → not-measured.
    Y8 reads the XSTest over-refusal rate (evidence key, or risk-family
    metric cell, preferring split `xstest`) — missing → not-measured.
    Y1 consumes `y1_verdict(margin=8.0)` (missing → not-measured, consistent
    with the global rule). Y2 reads the OOD block from the robustness report
    (requires `split_manifest_fingerprint`, `ood_source` alone is not enough);
    comparator = B4, keys `y2.comparator_accuracy` / `y2.b4_accuracy` /
    `baselines.B4.ood*`. Y6 uses permutation `within_tolerance` +
    `accuracy_delta ≤ 0.005`. Y12: drop ≤ 1.0 pt or `artifact_rejected` → pass.
    Y13: `resume.abs_diff ≤ 1e-4`. Y14: `reproducibility_delta_points` or
    duplicate-seed max pairwise gap computed from runs.
  - Negative results: failed arms (always kept), failed blocking verdicts, and
    non-distinguishable comparisons are synthesised in. With
    `negative_results=None` the section can never come out empty (falls back to
    `none observed` + an explicit neutral finding). An explicit `[]` disables
    the fallback → section empty → `INVALID` banner + exit 1.
  - Inconsistent `data_fingerprint` across runs → manifest warning,
    `fingerprint_consistent: false` (comparisons across fingerprints are not
    valid for claims); runs missing `run_id` are rendered as `run[i]` with a
    warning, never dropped. No wall-clock anywhere; byte-identical output
    across invocations (verified).

- `tests/unit/evaluation/test_ablate.py` (19 tests), `tests/unit/evaluation/test_report.py`
  (25 tests) — synthetic evidence only. Claim boundary at exactly the 1-seed-std
  threshold ([0,1,2] mean == std → no claim; [0,2,3] → distinguishable);
  empty-negative-results exit 1; Y1 fail → exit 1; Y7/Y8 missing → not-measured
  (never pass); failed arm kept; arm not-measured without fabricated numbers;
  distinguishable vs not-distinguishable rendering; RunRegistry-like injection;
  registry fingerprint-mismatch warning; duplicate-seed Y14 both branches.

## Results
- `pytest tests/unit/evaluation` → 193 passed.
- `ruff check` (my 4 files) → clean; `mypy` → clean.

## Decisions / notes
- "Blocking" is not enumerated in docs/evaluation.md §8, so all fourteen
  criteria are blocking (fail-closed). Non-blocking exits would weaken the gate;
  the CLI (TASK-25) reads the same code path.
- not-measured verdicts do NOT fail the command; only blocking `fail` and the
  empty negative-results section do.
- Y1 missing → not-measured (global missing-key rule wins); `y1_verdict` is
  still the verdict source whenever both accuracies exist.
- Y2 requires the fingerprint key itself; accepting `ood_source` as a
  fingerprint would defeat the "input dict accepts the fingerprint" intent.
- Sample seed std (ddof=1) throughout; n<2 → not distinguishable.
- `report.py` exceeds the 1000-LOC repo guideline — unavoidable: file ownership
  pins exactly these four files, and splitting would create unowned modules.
