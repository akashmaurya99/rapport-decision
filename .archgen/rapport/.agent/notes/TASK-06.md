# TASK-06 — Marker readout, six heads and model assembly

Status: done (pending orchestrator snapshot)

## What was built

- `src/model/head/errors.py` — typed errors, repo pattern (`code` on the
  class, included in `str(...)`): `ModelError` base with
  `InvalidOptionCountError` (`MODEL_INVALID_OPTION_COUNT`), `OptionMaskError`
  (`MODEL_OPTION_MASK_INVALID`), `GoalMarkerError` (`MODEL_GOAL_MARKER_INVALID`),
  `AttentionMaskError` (`MODEL_ATTENTION_MASK_INVALID`), `InvalidTaskIdError`
  (`MODEL_INVALID_TASK_ID`).
- `src/model/head/readout.py` — `MarkerReadout`: gathers hidden states at the
  `<goal>` marker and each `<opt>` marker, projects by `W_q` / `W_o`
  (bias-free 384→384, spec §4 rows: 147K each), scores options as cosine
  similarity divided by a per-task learned temperature `τ_task =
  exp(log_tau)` (8 params), then a per-task affine `scale·s + bias` (16
  params, identity at init — spec's "Task affine" row, applied *before*
  softmax as the spec requires). `ReadoutOutput` carries `choice_logits`,
  `scores` (sigmoid utility in [0,1]), `score_vector` (raw s_i, divided by τ),
  and `probabilities` (the masked listwise softmax). Padding slots
  (`option_mask == 0`): `choice_logits = -1e9` (`MASK_NEG`), `scores` and
  `score_vector` = 0, softmax assigns exactly 0 (finite stand-in for −inf so
  TASK-11's CE and ONNX never see inf/nan; still listwise-softmax-CE ready).
- `src/model/head/heads.py` — `ProbabilityHead` (CLS → 384→192 GELU → 1,
  sigmoid), `ConfidenceHead` (concat(CLS, p_max, margin, entropy) → 387→128
  GELU → 1, sigmoid), `RiskHead` (CLS → 384→192 GELU → 8, multi-label sigmoid),
  and `confidence_features` (hand-computed fixture in tests). Imports no data
  layer.
- `src/model/head/model.py` — `RapportModel(RapportConfig | None, num_tasks=8)`;
  `RapportOutput` frozen dataclass (six tensors + `score_vector`);
  `_ConditionedTrunk(RapportTrunk)` — **private subclass of TASK-05's trunk,
  core untouched**, inserting the spec §4 task-conditioning row (learned
  per-task vector added at `<task>` marker position(s) *before* the blocks,
  8 × 384 = 3,072 params); everything else is inherited unchanged.
- `src/model/head/__init__.py` — exports; **`src/model/__init__.py` was NOT
  created** — `src.model` stays a namespace package (TASK-05 note); nothing of
  TASK-05's is shadowed.
- Tests in `tests/unit/model/head/`: `test_model.py` (shapes, E2 validation,
  budgets), `test_permutation.py` (FR-MODEL-02), `test_heads.py` (unit heads),
  `head_helpers.py` (joint-sequence fixtures), `conftest.py` (sys.path setup
  only, repo idiom).

## Contract details worth knowing

- **Forward signature** (as specified): `forward(input_ids, attention_mask,
  option_mask, task_ids) -> RapportOutput`. `attention_mask` is required but
  may be `None` (means full bidirectional window). `task_ids` [B] int64,
  0–5 = the six families in schema order (choice, preference, scoring,
  probability, abstention, risk), 6–7 reserved (spec §4), exactly 8 slots.
- **CLS analogue**: RAPPORT's marker set has no `<cls>` token; `c :=
  hidden[:, 0]`, i.e. the `<task>` marker — the position a BERT-style CLS
  occupies and where `encode_example` always opens the joint sequence.
  Documented in the module docstring and tested to feed the probability/risk/
  confidence heads.
- **Answerable is absorbed into the probability head** (picked ONE per the
  task): `answerable` re-emits the probability head's sigmoid output — one
  shared 384→192→1 stack, **zero added parameters**. Spec §4's layer table
  lists no answerability block and `SpecConstants.heads_total` reconciles
  exactly without one. Documented in `heads.py` and here; TASK-11's
  answerability loss will hit the same parameters as the probability loss —
  masked independently per label as designed.
- **Permutation equivariance (FR-MODEL-02)**: exact 1e-4 is guaranteed and
  tested at the readout/heads level (the only option-slot-exposed path).
  Token positions are inherent to the sequence (RoPE is the trunk's only
  position mechanism; spec §6: "positions are inherent to the sequence"), so
  re-encoding a permuted option order moves marker positions and perturbs
  trunk hidden states measurably (~0.5 norm observed). What FR-MODEL-02
  forbids — ordinal embeddings / slot-indexed paths — is tested directly:
  readout equivariance (atol 1e-6), full-forward equivariance with a
  content-aware, position-free stub trunk swapped in for RoPE (all n! for
  n ≤ 4), a structural scan banning `ordinal|slot_emb|opt_index|option_pos|
  order_emb|position` in every parameter name, and a no-slot-axis check on the
  readout parameters (only task-axis and d×d projections). End-to-end accuracy
  under re-encoded permutation is TASK-19's measurement, explicitly not
  claimed here. Test options use distinct leading-word texts so the
  content-based equivariance test is non-vacuous (asserted).
- **E2 typed refusal before compute**: `InvalidOptionCountError` for slot
  width 1 or > 8, and for any per-row real count outside 2..8 — raised inside
  `RapportModel.forward` *before* the trunk is called (tested with a spy:
  zero trunk calls on validation failure).
- **Joint single pass (FR-MODEL-04)**: one trunk call encodes state + all
  options; tested via spy counters (trunk == 1 call, readout == 1 call).
  No per-option forward pass exists in the default path; options are gathered
  from the shared hidden states. The batch is right-padded to a common width;
  mixing option counts in one batch is supported (rectangular `[B, n]` outputs
  with padding slots inert — TASK-22/19 will read `option_mask`).
- **mixed-count batches**: outputs are `[B, W]` with `W = option_mask.shape[1]`;
  for a uniform-n batch `W == n` and the "exactly n scores per example"
  criterion holds literally (tested for n ∈ 2..8, batch 1 and 32). Per-row real
  counts are what validation enforces; padding slots are inert everywhere.
- **`RapportOutput`** carries all six tensors + `score_vector` (raw cos/τ, the
  spec §6 `s_i`); it does not carry `probabilities` (TASK-11 can derive it
  from `choice_logits`, it is still exposed on `ReadoutOutput`).
- **Non-prefix masks rejected** (`OptionMaskError`): padding must be trailing;
  the joint layout right-pads options, so this is the pipeline's native form —
  permute-within-real-slots is the supported permutation.
- Stale design notes: `count_parameters(model, compact_spec)` passes for the
  full model — all TASK-06 params land in `other`; trunk/embedding/attention/
  swiglu/norms match the spec exactly (count test: `|total − spec.total| ≤ 16`).

## Verification

- `.venv/bin/python -m pytest tests/unit/model/head -q` → **65 passed**.
- `.venv/bin/python -m pytest tests/ -q` → **580 passed, 1 skipped**
  (full repo suite; nothing else was touched).
- `.venv/bin/python -m ruff check src/model/head tests/unit/model/head` → clean.
- `.venv/bin/python -m mypy src/model/head` → no issues.
- Parameter accounting (own count test, vs `RapportConfig.compact()` spec):
  trunk 21,242,880 (exact), embedding 11,720,448 (exact), heads **497,378**
  (spec `heads_total` 497,370 + 8 `log_tau` — the spec's Scores row lists
  "n_tasks" but `SpecConstants.heads_total` apparently does not absorb them);
  total **33,460,706** (spec 33,460,698; Δ = 8, asserted ≤ 16 in test);
  **heads ≤ 0.6M** (497,378). All within spec tolerance; counter raises on
  trunk/embedding drift.
- n ∈ {2..8} × batch {1, 32}: all shapes per FR-MODEL-03 (choice_logits[n],
  scores[n], probability/confidence/answerable [B,1], risk [B,8], + score_vector[n]);
  n = 1 and n = 9 refused by input validation with `InvalidOptionCountError`
  before any trunk compute (spy-verified).
- Permutation: `hidden`-state readout equivariance exact (≤ 1e-6) for all
  n!, n ≤ 4, plus two sampled seeds; full-forward stub-trunk equivariance
  (atol 1e-4); CLS-side outputs permutation-invariant; confidence features
  permutation-invariant (atol 1e-6, hand-checked entropy/margin math).
- Padding slots inert: `choice_logits == MASK_NEG`, `scores == 0`,
  softmax ignores them (mixed 2/4-option batch tested).

## Notes for the orchestrator / downstream (TASK-11, 12, 16, 19, 22, 23, 25)

- **TASK-11**: consume `RapportOutput.choice_logits` for listwise softmax CE;
  padding slots are `MASK_NEG = -1e9` (finite) so unmasked `log_softmax` sums
  exclude padding exactly; still pass `option_mask` for the masked path.
  `answerable` is not an independent output — decide the abstention loss
  target against `probability` (they share parameters). `probabilities`
  (masked distribution, padding = 0) is available from `model.readout(...)`
  if a consumer wants it without re-softmaxing.
- **TASK-12**: model instantiates on CPU fp32; no data files needed;
  `RapportModel(RapportConfig.compact())` is the checkpoint root.
- **TASK-16**: per-bucket temperatures fit on dev *logits* — use
  `choice_logits` (pre-affine scores are `score_vector`). Abstention policy
  threshold 0.75 derives from the utility triple (1, −3, 0); nothing here
  blocks that.
- **TASK-19**: the 1e-4 permutation harness will enumerate all n! — expect
  exact equality through `readout` (≤1e-6), not through re-encoded sequences
  (RoPE positions; deviation ~0.1–1.0); the harness's "per-score max deviation"
  must be measured on the same forward (or via the slot-level path), or the
  test will fail by construction.
- **TASK-22**: `RapportModel.forward` is the intended default path; the
  predictor maps `chosen_index` from `argmax(choice_logits)` over real slots
  and `probabilities` from the masked softmax.
- **TASK-23**: the cosine path is fp32-safe (all-fp32 CPU here); `MASK_NEG`
  is finite for ONNX export; `answerable` folding into `probability` means
  INT8 per-head checks cover one shared block for both outputs.
- **Does not exist in this task**: loss functions, training loop,
  calibration, predictor, export — all later tasks.
