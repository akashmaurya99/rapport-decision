# TASK-14 notes — Adaptive multi-task weighting and per-task logging

Done: 2026-11-19. Status: done (single wave, one pass).

## Files shipped
- `src/training/multitask/weighting.py` — `TaskWeighter(num_tasks, floor=0.05, cap=4.0,
  init_s=0.0, task_names=None)`. Kendall et al. 2018: `w_t = exp(-s_t)`,
  `L = Σ_t w_t·L_t + s_t` (`s_t` clamped internally to `[-log(cap), -log(floor)]` so
  `exp(-s_t)` always lies in `[0.05, 4.0]`; the clamp saturates, which is what makes a
  collapse detectable). `log_s` is an `nn.Parameter` trained with the model.
  `weights()` returns detached floats (task → float). `forward(losses, active=None)`
  accepts a Mapping (strict: all tasks required, unknown keys rejected) or an ordered
  Sequence of scalar Tensors; `active={task: False}` zeroes a task's weighted term while
  keeping its `s_t` regulariser. `loss` is aliased to `forward`.
  `log_epoch(logger, epoch, metrics=None, *, split="train", step=None)` writes one
  `event="task_weights"` record through `src.ops.logging.JsonlLogger` with the resolved
  per-task weight vector and per-task active flag as top-level record keys
  (`weights` task→float, `active` task→bool — nested dicts are legal as JsonlLogger
  `extra` values; the flat `metrics` block carries `epoch`, `num_tasks`, optional
  `loss_<task>` and any flat-scalar caller entries). `logger=None` skips the write but
  keeps collapse tracking. `log_epoch` is intended to be called once per epoch.
  `to_manifest_fragment()` returns `{task_weight_mode: "learned", task_weight_floor,
  task_weight_cap, task_weights: {task: float}}` for TASK-01's manifest builder to embed;
  once embedded, `RunRegistry.get(run_id).manifest["task_weights"]` surfaces the vector
  (test-covered). It never touches the filesystem — `src/config/**` untouched.
- `src/training/multitask/arms.py` — `build_fixed_weight_config(arm="A4", *, config_path=None)`.
  Loads `configs/experiments/a4-weights.yaml` (default; key-for-key embedded fallback used
  only when the default path is unreachable; an explicit bad `config_path` raises), returns
  a YAML-serializable dict with `weight_mode: "fixed"` and an explicit uniform `weights`
  vector (1.0 × the 7 canonical tasks). All other template keys (seeds [13,29,47],
  starter 55k mixture, 8 epochs, AdamW lr 3e-4, fp16 …) inherited verbatim. Only `"A4"`
  is registered; other arms raise `ValueError`. No file under `configs/` is written.
- `src/training/multitask/__init__.py` — public surface + `__all__`.
- `tests/unit/training/multitask/{conftest,test_weighting,test_arms}.py` — 44 tests.

## Acceptance criteria mapping
1. Floor 0.05 / cap 4.0 enforced by clamping `s_t`; sweep test over s ∈ [-100, 100]
   (2001 points) plus extreme-value saturation tests; logged every epoch via
   `log_epoch`; a task at the floor for 2 consecutive epochs emits
   `{"event": "WEIGHT_COLLAPSE", "task": ..., "consecutive_floor_epochs": 2}` (scalar
   consecutive-epoch counter, streak resets off-floor, re-emits on a fresh collapse;
   NOT re-emitted every epoch while pinned). JSONL-written via the supplied logger;
   `collapse_events()` returns copies of all emitted events (also without a logger).
   Floor detection uses `weight <= floor + 1e-6` to absorb 1-ulp slack from `exp(-s_max)`.
2. Resolved vector written to the manifest via `to_manifest_fragment()` (embedded by
   TASK-01's builder — this task owns no manifest-writing code) and surfaces through
   `RunRegistry` once embedded (integration test builds a tmp `runs/<id>/manifest.json`,
   scans it, asserts `entry.manifest["task_weights"]`). The JSONL `task_weights` record
   is the per-epoch log channel; both go through ops components' public contracts.
3. Fixed-weight arm registered for A4: `build_fixed_weight_config("A4")`; consistency
   with the on-disk a4-weights.yaml (and with the fallback template) is test-enforced,
   including "template file not mutated".

## Deviations / decisions
- The brief referenced `configs/experiments/a4-weighting.yaml`; the file TASK-13 actually
  shipped is `configs/experiments/a4-weights.yaml` (same purpose, title `weights`, arm
  `A4`). Implementation and tests use the real file; the fallback template is asserted
  equal to it so any drift fails loudly.
- A4 YAML comments the fixed arm as "fixed weights as per docs/training-strategy.md
  §3.2 defaults" (score/prob/conf/risk = 0.5). The task contract pins the arm at
  **uniform 1.0 across tasks** instead; `build_fixed_weight_config` emits uniform 1.0
  and documents the deviation in its module docstring.
- Clamp saturation ⇒ zero gradient through the weighting term at the boundary; the
  `+ s_t` regularizer is computed on the clamped `s_t` (literal reading of "s_t is
  clamped so exp(-s_t) stays in [floor, cap]"). Kendall's `+ s` term would otherwise
  fight the clamp asymmetrically.
- Collapse tracking is scalar consecutive-epoch counting (per spec "2 consecutive
  epochs"), not EWMA.
- Registry cannot be written by design (read-only scanner over `manifest.json`); the
  FR-TRAIN-03 "written to the manifest and to the run registry" chain is satisfied
  through the manifest fragment, which is what the registry indexes.
- Runtime streak state is deliberately not in `state_dict()`.
- `configs/`, `src/config/**`, `src/ops/**`, `src/training/losses.py`,
  `src/training/core/**` untouched.

## Verification
- `.venv/bin/python -m pytest tests/unit/training/multitask -q` → 44 passed
- `.venv/bin/python -m ruff check src/training/multitask tests/unit/training/multitask` → clean
- `.venv/bin/python -m mypy src/training/multitask` → Success, no issues

## Follow-ups (not this task)
- Trainer (TASK-12 caller) should pass `active=` from the losses' `active` mapping and
  call `log_epoch` at epoch end; checkpoint/resume of streak state is out of scope.
- TASK-21's report consumes `WEIGHT_COLLAPSE` records to gate A4's pass condition.
