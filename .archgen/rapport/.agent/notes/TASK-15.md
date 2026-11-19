# TASK-15 — Local-teacher soft-label distillation and its gate

**Status:** done. **Owner:** src/distill/**, tests/unit/distill/** (nothing else touched).

## What was built

- `src/distill/cache.py` — `build_soft_label_cache(examples, teacher_id, template, options)`.
  Content-addressed key `(teacher_id, template_hash[=sha256(template)], example_id)`.
  Layout: `<cache_dir>/<sanitized_teacher_id>/<example_id>.json` + `index.json`
  (default root `cache/teacher`). `teacher_id` is filesystem-sanitized
  (`[A-Za-z0-9._-]`, rest → `_`) so `Qwen/Qwen3-1.7B` cannot escape the dir; raw id is
  stored inside each file. Writes are atomic (tmp + fsync + `os.replace`).
  - Counter `teacher_forwards` on the result increments **only** on actual teacher calls;
    second invocation = 0 forwards, `cache_hit_rate == 1.0` (asserted in tests).
  - Corruption: every payload carries `payload_hash` (sha256 of canonical JSON of the
    payload fields). Mismatch / bad JSON / wrong recorded triple / non-finite values →
    treated as a **miss**, recomputed exactly once, new hash written back. Never silently trusted.
  - `index.json` merges validated entries (files are the source of truth; the index is audit).
    An all-hit run performs no writes at all.
  - Duplicated example_ids in one batch share a single forward (counted as misses, not hits).
- `src/distill/gate.py` — `distillation_gate(teacher_metrics, student_supervised_metrics,
  student_kd_metrics, min_teacher_margin=3.0, min_kd_gain=1.0) -> GateOutcome`. Pure; never
  reaches into training (TASK-25 orchestrates). Reasons: `ok`, `teacher_unavailable`,
  `teacher_margin_below_threshold`, `kd_gain_below_threshold`. Margin checked before gain;
  teacher limb dominates. `GateOutcome` (frozen dataclass) is JSON-serializable
  (`to_dict`/`to_json`/`from_dict`), carries `teacher_margin`, `kd_gain`, `reason`,
  per-limb `checks`, `negative_result=True` on abort (= recorded negative result, E8 / R9).
  - Metric contract: `dev_accuracy` (fallback `accuracy`) in **percentage points**;
    when all provided accuracies lie in [0, 1] and both thresholds are >= 1.0 they are
    auto-read as fractions ×100. Absent/non-finite student accuracy → typed `GateInputError`
    (programmer error). Missing/unreadable teacher signal → abort `teacher_unavailable`,
    signalled by `teacher_metrics=None`, an `available`/`labels_available` falsy flag, or a
    missing/non-finite teacher `dev_accuracy`.
- `src/distill/loss.py` — `distillation_loss(student_scores, teacher_logprobs, temperature=2.0)`:
  `T² · KL(softmax(t/T) ‖ softmax(s/T))`, softmax in FP32, teacher detached, mean over batch,
  scalar tensor. Hand-computed fixture: KL = 0.09130699391503301, loss = 0.36522797566013204.
- `src/distill/entropy.py` — documented constant `ENTROPY_EXCLUSION_PERCENTILE = 95.0`;
  `entropy_from_logprobs` (nats, max-subtracted softmax), `percentile_linear` (linear
  interpolation, numpy "linear" semantics), `entropy_exclusion_mask` — entries strictly
  **above** the batch 95th percentile get `exclude_from_distillation=True`.
- `src/distill/teacher.py` — `Teacher` callable protocol (`teacher(example, prompt) -> one
  log-prob per option, in option order, at the answer position`), `render_prompt`
  (`{goal}`/`{state}`/`{options}`, lettered options), `DEFAULT_TEMPLATE`, and the live path
  `load_local_teacher` / `LocalCausalLMTeacher` (lazy transformers import; one-time HF
  download, then fully offline; sums option-completion token log-probs; **never exercised
  by tests**; any load failure → `TeacherUnavailableError`, E8's fail-to-load limb).
- `src/distill/errors.py` — `DistillError` base (`DISTILL_*` codes), `DistillInputError`,
  `TeacherUnavailableError`, `GateInputError`.

## Verification

`.venv/bin/python -m pytest tests/unit/distill -q` → **54 passed**.
`.venv/bin/python -m ruff check src/distill tests/unit/distill` → clean.
`.venv/bin/python -m mypy src/distill` → clean (7 files).
Tests are fully offline: the teacher is always an injected deterministic stub
(`tests/unit/distill/conftest.py::StubTeacher`); no live download, no network.

## Notes / for the orchestrator

- `cache/teacher/` is runtime output and should be gitignored / not committed
  (repo is not a git repo right now, so nothing was added). Suggest TASK-25's acceptance
  script point `DistillOptions.cache_dir` at the run dir or leave the default.
- Cache key includes `template_hash` but the file name is only `<example_id>.json`
  (per the spec's path); a template change is detected via the recorded hash and treated
  as a miss → recompute under the same path. There is no cross-template coexistence for
  the same example id (last write wins) — fine for Stage C's single template.
- Gate accuracies are percentage points; the fraction auto-normalisation edge
  (all values ≤ 1.0 and both thresholds ≥ 1.0) is documented in `src/distill/gate.py`.
- FR-DISTILL-01 (criterion 3) is architectural: the runtime path loads the model locally
  and performs no network calls beyond the one-time download. Not something a unit test
  can assert, hence the offline-stub strategy; the live path is deliberately untested.
- Stage C (TASK-25) consumes this package: build/refresh cache → gate → KD arm only if pass;
  `abort` must surface as a recorded negative result (`GateOutcome.negative_result`).
