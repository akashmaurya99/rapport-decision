# TASK-20 notes — Structured logging, run registry, reproducibility checks

Done: 2026-11-19. Status: done (single wave, one pass).

## Files shipped
- `src/ops/logging.py` — `JsonlLogger` (FR-OPS-01). One JSON object per line; every
  record validated **before** write (missing/invalid keys → typed
  `JsonlRecordError`, nothing written). Append semantics: `open(path, "ab")` +
  single `write()` of the full line + `flush()` + `os.fsync()` per record.
  Required keys enforced: `run_id`, `step` (int ≥ 0, bool rejected), `split`
  (train|dev|test|benchmark|eval), `event` (non-empty str), `wall_clock_seconds`
  (finite float ≥ 0; auto-measured from the previous record when omitted), and
  `metrics` or `metric_set` (dict of non-empty str keys → flat finite JSON
  scalars). `timestamp` (ISO-8601 UTC) added automatically; extra keys allowed
  unless they collide with a reserved key. Sugar: `log_step` / `log_eval` /
  `log_benchmark` (event/split defaults). Module-level `iter_records(path)`
  round-trips a log, raising `JsonlRecordError` with the 1-based line number on
  corrupt lines. `json.dumps(..., allow_nan=False)` — NaN can never be written.
- `src/ops/registry.py` — `RunRegistry` (FR-OPS-02). Scans immediate child dirs
  of a runs root (`runs/<run_id>/manifest.json` layout). A run is indexed only
  if `manifest.json` exists, parses as a JSON object, and carries non-empty
  string `run_id` / `config_hash` / `data_fingerprint`. Otherwise **excluded**,
  never silently listed: `excluded_count` + `exclusions` (per-dir reason:
  missing_manifest / invalid_manifest_json / manifest_not_an_object /
  unreadable_manifest / missing_* / invalid_* / duplicate_run_id). Duplicate
  `run_id`s are excluded everywhere they appear (deterministic). `list()` →
  sorted run ids; `get(run_id)` → `RegistryEntry` (manifest path, parsed
  manifest, config_hash, data_fingerprint, checkpoint paths from
  `checkpoints/**` + top-level `*.pt/*.pth/*.ckpt/*.safetensors`). `refresh()`
  rescan returns a `RegistryScan` summary; missing/file root → `RunRegistryError`.
  `get` on unknown/excluded id → `UnknownRunError`.
- `src/ops/repro.py` — `verify_resume` (FR-OPS-03). Re-executes a **recorded**,
  tiny deterministic workload: gradient descent on `f(w) = 0.5·||w − target||²`,
  `w ~ N(0,1)` (8 dims) from `numpy.random.default_rng(seed)`, fixed lr 0.1,
  float64, fixed op order — pure function of `(seed, steps)`. Compares the
  reproduced loss against the recorded value (explicit `recorded_loss=` or read
  from a JSONL log via `recorded_path=` — highest-`step` record carrying a
  finite `loss` metric, from `metrics` or `metric_set`) and **asserts within
  1e-4** (`DEFAULT_TOLERANCE`) else raises `ReproductionMismatchError` (carries
  `.result`). Defaults: `replay_steps=50`. Exactly one of
  `recorded_loss`/`recorded_path` required.
  **Scope note (in module docstring): verifies determinism of the harness
  (this arithmetic replay), NOT of a trained network** — real kill-resume
  determinism is the trainer's domain (TASK-12).
- `src/ops/errors.py` — `OpsError` base (config-style `code`/`key`), `JsonlRecordError`,
  `JsonlWriteError`, `RunRegistryError`, `UnknownRunError`, `ReproductionError`,
  `ReproductionMismatchError`. Standalone (no import of `src.config`) so ops
  stays dependency-light; mirrors the config error pattern.
- `src/ops/__init__.py` — public surface + `__all__`.
- `tests/unit/ops/{conftest,test_logging,test_registry,test_repro}.py` — 79 tests.

## Results
- `pytest tests/unit/ops -q` → 79 passed; full `tests/unit` → 280 passed, 1 skipped.
- `ruff check src/ops tests/unit/ops` → clean; `mypy src/ops` → clean.

## Notes / decisions
- Wall-clock auto-measure: when `wall_clock_seconds` is omitted, the logger
  stamps elapsed `perf_counter` time since the previous record (logger
  construction for the first) — satisfies "one record with wall-clock time"
  without caller bookkeeping; explicit values always win.
- Registry indexes by the **manifest's** `run_id`, not the directory name (per
  the RunManifest contract); dirs are only a scanning unit. Non-directory files
  directly in the root are ignored (not runs, count as neither).
- `excluded` counts run *directories*, so a run missing `manifest.json` is
  counted even though no manifest was ever globbed.
- Reference workload constant (seed 13, 50 steps): 2.2519580952978248e-04 —
  pinned in a regression test (`abs=1e-12`).
- Test helper `make_run` uses an `_UNSET` sentinel so "omit field" (exclusion)
  is distinguishable from "default value"; first draft of the test suite had
  that conflation and was fixed before landing.
- Nothing outside `src/ops/**` and `tests/unit/ops/**` touched. No git commands run.
