# TASK-25 — End-to-end acceptance run and report

## What shipped

- `scripts/e2e.py` — the single acceptance CLI. Command surface:
  `--config PATH`, `--resume {auto,fresh}`, `--starter`, `--synthetic`,
  `--kill-after-step N`, plus isolated test/ops overrides (`--runs-dir`,
  `--data-dir`, `--rows-per-family`, `--steps`, `--seed`, `--evidence-file`,
  `--reference-losses`).
- `scripts/e2e_fixtures.py` — deterministic synthetic `DecisionExample` rows
  spanning all six families (with a withheld-source OOD block) and a pure
  per-step-seed training batch; no corpus download, no CUDA fabrication.
- `scripts/__init__.py` — makes `scripts` a package (also fixes mypy module
  identity for `scripts.*` imports).
- `tests/unit/scripts/conftest.py` + `tests/unit/scripts/test_e2e.py` — 12 tests.

## Design decisions

- **Data honesty.** With neither `--synthetic` nor a real parquet directory,
  the runner prints `synthetic fallback (NOT real corpus evidence)`. Synthetic
  mode uses `rows_per_family` (default 40); the configured mixture size
  (115.6k / 55k) is recorded in `state.json` and `sample.json` as the *planned*
  corpus, never claimed as executed data. CPU-only runs disable FP16 and print
  so; T4 latency/throughput keys are deliberately absent from benchmark
  evidence, so Y9/Y10 render `not-measured` instead of invented numbers.
- **Run identity / idempotency.** `run_id = <arm>-<mode>-<fingerprint[:12]>`
  is deterministic. `manifest.json` is written via `write_run_manifest(manifest,
  runs_dir)` (the API expects the runs root, not the run dir). `state.json`
  carries the experiment/config/run fingerprints and per-stage markers; stage
  markers are preserved across re-invocations with the same fingerprint, so
  re-running skips the completed stages.
- **Resume.** `discover_resume()` uses `RunRegistry` (real API), then verifies
  each candidate's `.sha256` with TASK-12's `verify_checkpoint` *before*
  loading. A checkpoint from another config raises
  `E2E_CONFIG_FINGERPRINT_MISMATCH`; a corrupt/missing sidecar propagates
  `CHECKPOINT_HASH_MISMATCH` / `CHECKPOINT_SIDECAR_MISSING` to the CLI boundary
  (stderr + exit 2) — never a silent retrain. Without `--resume auto` the
  trainer is built fresh and the chosen path is printed in both cases.
- **Preemption.** `--kill-after-step N` saves a TASK-12 checkpoint at step N,
  logs `simulated_kill`, then `os._exit(1)`. The resumed process re-initialises
  the model with a different seed, calls `Trainer.load()` (restoring model,
  optimizer, scaler, scheduler, step, sampler offset and RNG), and continues.
- **Training shape.** A one-layer `RapportConfig` (`d_model=32`) keeps the CPU
  smoke short; batches come from `e2e_fixtures.make_batch`, a pure function of
  the global step. Default synthetic horizon is 50 steps (config's real
  `planned_steps` is 14,456 at 115.6k / batch 64 / 8 epochs).
- **Report / exit.** `build_report()` renders all Y1–Y14 rows;
  `result.exit_code` (internally `exit_code()`) is returned verbatim from
  `main()`. `--evidence-file` lets an operator/tests inject recorded evidence
  (e.g. a failing `resume.abs_diff`) — the report logic, not the CLI, decides
  the verdict and the exit code. `--starter` writes `smoke.json` with
  `verdict: false` and produces no `report.md`/`report.json`.

## Verification (all green)

- `.venv/bin/python -m pytest tests/unit/scripts -q` → **12 passed**
  (fixtures, E10 kill/resume equality, 14 verdict rows + OOD/Y2 block, blocking
  failure exit 1, stage reuse, starter smoke, bad sidecar typed errors, wrong
  config typed error, synthetic fallback announcement).
- `.venv/bin/python -m ruff check scripts tests/unit/scripts` → all checks passed.
- `.venv/bin/python -m mypy scripts` → no issues.
- Manual default-scale run of the acceptance command (kill at step 20, resume):
  RESUMED from `checkpoint_step20.pt`, 50/50 steps, max loss difference vs the
  uninterrupted run **0.0** (≪ 1e-4), `exit_code=0`, 14/14 verdict rows.

## Notes for reviewers

- No `src/**` files were touched; all upstream modules were reused as-is.
- No git commands were run.
- `Skill compliance: N/A — no capability skill was required for this task`
  (the task contract declares no capabilities).
