# TASK-12 — FP16 trainer core with checkpoint and resume

Status: done

## What was built

- `src/training/core/errors.py` — typed errors: `TrainerError` base (`code`+
  `key`, config-style), `TrainerConfigError`, `CheckpointError` →
  `CheckpointNotFoundError`, `CheckpointSidecarMissingError`,
  `CheckpointHashMismatchError` (carries `.path/.expected/.actual`),
  `CheckpointBeyondHorizonError` (carries `.step/.steps`), and
  `TrainingDivergedError` (carries `.nonfinite_steps/.epoch`).
- `src/training/core/config.py` — `TrainingConfig` frozen dataclass; defaults
  per training-strategy.md §3.1/§8 and t4-optimization.md §2 (AdamW β
  (0.9, 0.98) ε 1e-8, lr 3e-4 → cosine to 1e-5 after 5% warmup, WD 0.01,
  clip 1.0, scaler init 2**16 / backoff 0.5 / growth interval 2000,
  `checkpoint_every=500` — a config key — `steps` horizon default 14,450,
  `nonfinite_per_epoch=10`, `fp16=True`, `device=auto`). `from_mapping`
  rejects unknown keys, bool-as-int, out-of-range values.
- `src/training/core/checkpoint.py` — atomic save (`*.tmp` → write+fsync →
  `os.replace` → dir fsync), `.sha256` sidecar written the same way
  (`<digest>  <filename>\n`), `verify_checkpoint` (sidecar parsed and hex
  checked — missing/malformed sidecar → `CheckpointSidecarMissingError`,
  mismatch → `CheckpointHashMismatchError`), `load_checkpoint_payload`
  (hash verified **before** `torch.load`, which passes
  `weights_only=False`), `latest_checkpoint` (resume-auto, highest step
  wins), `checkpoint_step{N}.pt` naming, `step_from_filename`.
- `src/training/core/trainer.py` — `Trainer(config, model=None, *,
  optimizer|scheduler|scaler|logger|sampler|…)`. FP16 via
  `torch.autocast(device_type, dtype=torch.float16)`; autocast is emulated
  on CPU and verified to run. Loss carried in FP32 (`breakdown.total.float()`,
  §2). Gradient clipping global norm 1.0. `build_optimizer` — AdamW with
  weight decay excluded on norms/biases/embeddings/task_vector (ndim<2 or
  name token). `build_scheduler` — 5% linear warmup then cosine to
  `min_learning_rate` (LambdaLR). `Trainer.train_batch(batch)` consumes one
  mapping batch (`input_ids`/`attention_mask`/`option_mask`/`task_ids` +
  the `compute_losses` label kwargs), returns frozen `StepResult`.
  `fit(dataloader, epochs=…)` re-iterates the dataloader per epoch, writes
  an epoch-end checkpoint each epoch + every `checkpoint_every` successful
  steps mid-epoch (named by optimizer step count).
- Non-finite handling (E5/FR-TRAIN-04): `LossNotFiniteError` **before**
  backward, or non-finite grads detected after `scaler.unscale_` (checked
  before clipping). Either way: `nonfinite_steps` + `nonfinite_steps_epoch`
  incremented, optimizer/scheduler **not** stepped, global `step` counter
  does not advance, sampler offset still advances, JSONL record still
  emitted, scaler backs off. More than `nonfinite_per_epoch` (10) in an
  epoch → `TrainingDivergedError`.
- Checkpoint payload: model/optimizer/scaler/scheduler state_dicts,
  `step`, `epoch`, `sampler_offset`, `nonfinite_steps[_epoch]`, `rng`
  (python `random`, legacy `numpy.random` `get_state`/`set_state`,
  `torch.get_rng_state`, `torch.cuda.get_rng_state_all` when CUDA exists —
  empty list otherwise), plus `config` and `format_version`.
- Resume: `load()` resolves a file or directory (auto = latest), checks the
  filename step against the horizon **before** reading the payload, verifies
  the sha256 sidecar **before** `torch.load`, cross-checks payload step vs
  filename, refuses `step >= config.steps` with
  `CheckpointBeyondHorizonError` (nothing is deserialized/loaded), restores
  all state incl. RNG and sampler offset (attached sampler's `.offset`
  attribute is read/advanced/restored; internal counter otherwise).
- JSONL logging optional: accepts a `JsonlLogger`; one `step` record per
  consumed batch (skipped batches included, with `skipped`/`reason` and no
  `loss` key since NaN must never hit the logger) + one `epoch_end` record
  per epoch. No logger → no-op, counters still advance.
- `src/training/core/__init__.py` — public surface.

## Verification

- `.venv/bin/python -m pytest tests/unit/training/core -q` → **47 passed**
  (checkpoint 10, config 8, trainer 29).
- `.venv/bin/python -m ruff check src/training/core tests/unit/training/core` → clean.
- `.venv/bin/python -m mypy src/training/core` → clean (5 files).
- Full unit suite: `pytest tests/unit --ignore=tests/unit/data/mixture` →
  **938 passed, 1 skipped**. NOTE: the repo's full `pytest tests/unit`
  collection is broken by a pre-existing basename collision
  `tests/unit/config/test_loader.py` vs `tests/unit/data/mixture/test_loader.py`
  (present before this task; not owned here, left alone). This task's own
  config test was renamed `test_training_config.py` to avoid the same class
  of collision with `tests/unit/model/core/test_config.py`.
- Kill-resume FR-TRAIN-02 proof: 50-step toy workload (tiny RapportModel,
  fixed seeds, fp32 and fp16 modes), interrupted at batch/step 30, resumed
  from the step-30 checkpoint into a fresh process with a differently-seeded
  model → per-step losses match the uninterrupted run within **1e-4** (in
  fact exactly, incl. the identical skip pattern under fp16). 50/50 loss
  equality + `sampler_offset == 30` after reload.
- Hash-mismatch E6: file corrupted (single-bit flip and full-garbage
  overwrite) with intact sidecar → `CheckpointHashMismatchError`; model
  weights verified untouched (never loaded). Missing/malformed sidecar →
  `CheckpointSidecarMissingError`. Beyond-horizon (step == horizon) →
  `CheckpointBeyondHorizonError` before any deserialization; zeroed model
  verified still zeroed.
- Byte-equal round-trip: after 3 steps (fp16 on), model/optimizer/scaler/
  scheduler state_dicts compare exactly (`torch.equal` per tensor;
  scheduler `last_epoch`/`_last_lr` exact) after save→fresh-instance→load.
- Non-finite gradient (stub model with an autograd.Function injecting +inf)
  → step skipped, `nonfinite_steps == 1`, `step == 0`, scheduler/params
  untouched; >3 (test limit) in-epoch → `TrainingDivergedError`;
  loss-side NaN path (RapportOutput with NaN logits → `LossNotFiniteError`)
  → same skip accounting, reason `loss_not_finite:<component>`.
- `shasum -a 256 checkpoint_step30.pt` matches the `.sha256` sidecar
  byte-for-byte.

## Environment note (FP16 on CPU)

On CPU, `torch.autocast('cpu', float16)` is emulated and GradScaler's
2**16 init scale overflows the emulation on the first steps (fp16 max
65504) — observed: all-inf grads at 2**16, 1 bad param at 2**14, clean at
2**12; the scaler backs off deterministically (65536→…→8192) and training
proceeds; the JSONL path and the skip machinery handle it correctly. On
CUDA (the T4 target) this does not occur for mean-reduced losses. The toy
tests therefore use `fp16=False` for optimizer-step counting/cadence/
byte-equality tests, and exercise fp16 explicitly in the resume-parity test
and a completion smoke run (17/20 steps OK, 3 skips, well under the
10/epoch divergence limit). RNG restore covers python/numpy/torch; CUDA
states are saved when CUDA is available (empty otherwise).

## Not in scope / deferred

- SIGINT checkpointing, last-3 retention and best-dev checkpoint (spec §8,
  not in the contract).
- Kendall uncertainty weighting / per-task loss logging → TASK-14.
- `verify_resume` (FR-OPS-03) lives in `src/ops/repro.py` (TASK-20) and is
  deliberately untouched.
