# RAPPORT — Complete Technical Reference

This is the single repository-level engineering document for RAPPORT. It records
what the code actually implements after the audit-driven repair baseline
(`git commit 586c1d6`), the ML/DL concepts used, the data and training pipeline,
evaluation semantics, deployment path, and the limits that must not be claimed
as measured results.

Architecture intent and pre-registered experiment design remain under
`.archgen/rapport/` (`architecture.yaml`, `docs/*.md`, `tasks.yaml`). This
document describes the **implemented system**.

---

## 1. Executive summary

RAPPORT is a compact, non-autoregressive decision model. It reads a bounded
decision state plus a variable set of 2–8 candidate options and returns, in one
forward pass:

- a chosen option index,
- a score for every option,
- a listwise probability distribution over the real options,
- a calibrated correctness/confidence estimate,
- an abstention decision,
- an 8-dimensional risk vector.

It is a bounded decision model, not a general assistant or chat model, and it
does not generate text.

| Fact | Value |
|---|---|
| Production variant | `compact` / RAPPORT-C |
| Parameters | **33,460,706 actual** (spec table 33,460,698; +8 = `readout.log_tau`) |
| Layers × hidden | 12 × 384 |
| Attention | 6 heads × 64 head_dim, full bidirectional SDPA, padding mask only |
| FFN | SwiGLU 384 → 1024 → 384 |
| Position mechanism | RoPE base 10,000; no learned position table |
| Vocabulary table | 30,522 token slots |
| Stand-in tokenizer | 257 real pieces |
| Max training sequence | 512 tokens |
| Options per example | 2–8 |
| Starter mixture | 55,000 declared rows |
| Acceptance mixture | 115,600 declared rows |
| Training hardware target | NVIDIA T4, 16 GB, FP16 autocast + GradScaler |
| Runtime | Python 3.11+, PyTorch 2.x; validated on Python 3.12.12 / torch 2.15.0 |
| Tests at repair baseline | 1634 passed, 1 skipped |

---

## 2. Repository layout

| Path | Responsibility |
|---|---|
| `src/config/` | YAML resolution, environment overrides, manifests, seeding |
| `src/data/core/` | `DecisionExample`, normalization, serialization, parquet I/O |
| `src/data/acquire/` | licensed source acquisition, shard hashes, source manifests |
| `src/data/clean/` | exact/near dedup, Q1–Q6 quality rules, contradiction audit |
| `src/data/adapters/` | six source-to-schema adapters |
| `src/data/splits/` | group-aware deterministic split, leakage checks, split manifest |
| `src/data/mixture/` | declarative 55k/115.6k presets, train-only row binding |
| `src/tokenization/` | stand-in tokenizer, special tokens, encoding, length bucketing |
| `src/model/core/` | embedding, RoPE, attention, RMSNorm, SwiGLU, encoder trunk, counting |
| `src/model/head/` | marker readout, six heads, validation, model assembly |
| `src/training/` | losses, Trainer, checkpoints, config, multitask weighting |
| `src/distill/` | local-teacher runner, soft-label cache, KD loss, gating |
| `src/calibration/` | bucket temperature maps, Platt scaling, ECE metrics, abstention, risk thresholds |
| `src/evaluation/` | task metrics, baselines, robustness, ablations, Y1–Y14 report |
| `src/inference/` | `Predictor`, ONNX export, INT8 quantization, benchmark |
| `src/ops/` | JSONL logging, run registry, reproducibility helpers |
| `scripts/e2e.py` | complete pipeline CLI |
| `configs/` | base config, experiment configs, mixture presets |
| `requirements.lock` | pinned environment |
| `tests/unit/` | unit/regression tests mirroring `src/` |

---

## 3. Model architecture and parameter accounting

### 3.1 Declared variants

| Variant | Layers | d_model | Heads | head_dim | FFN | Embedding dim | Actual parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| RAPPORT-U (`ultra`) | 6 | 320 | 5 | 64 | 896 | 128 | 11,944,162 |
| RAPPORT-C (`compact`) | 12 | 384 | 6 | 64 | 1024 | 384 | 33,460,706 |
| RAPPORT-L (`large`) | 16 | 448 | 7 | 64 | 1192 | 448 | 52,803,170 |

`compact` is the production/Y11-eligible variant. Its FP32 weight tensor alone
is about 127 MB; full training state (weights + gradients + AdamW moments +
scaler/scheduler state) is much larger, so checkpoint retention is bounded to
the latest three by default.

### 3.2 Forward path

Input:

```text
input_ids       [B, T]            int64 token ids
attention_mask  [B, T]            1 = real token, 0 = padding
option_mask     [B, N]            1 = real option, 0 = padding
task_ids        [B]               task family id
```

Output (`RapportOutput`):

```text
choice_logits   [B, N]            masked listwise logits
scores          [B, N]            sigmoid-calibrated utility per option
score_vector    [B, N]            raw cosine / temperature
probability     [B, 1]            probability/answerable head output
confidence      [B, 1]            confidence head output
answerable      [B, 1]            same tensor/graph node as probability
risk            [B, 8]            sigmoid risk probabilities
```

### 3.3 Model components

- **Token embedding** (`TrunkEmbedding`): maps ids to `d_model`; outputs are
  multiplied by `sqrt(d_model)`. RAPPORT-U uses a factorized
  `vocab × 128 → 128 × d_model` bottleneck; C/L use a full table.
- **Task conditioning**: a learned task vector is added at the `<task>` marker
  position. It is normalized to the same scale as the token embedding.
- **RoPE** (`RotaryEmbedding`): half-split rotary embedding, base 10,000,
  computed in FP32 and cast to the input dtype after trig. It is applied inside
  every attention layer; there is no learned absolute position embedding.
- **Multi-head self-attention** (`MultiHeadSelfAttention`): `Q`, `K`, `V`
  projections, RoPE on `Q/K`, PyTorch `F.scaled_dot_product_attention` with
  `is_causal=False`. Padding is an additive mask using `finfo(dtype).min`, not
  `-inf`, so fully padded rows remain finite.
- **Encoder block** (`EncoderBlock`): pre-RMSNorm, attention residual,
  pre-RMSNorm, SwiGLU FFN residual. Optional dropout is applied to attention
  probabilities and residual branches only when configured and training.
- **RMSNorm**: bias-free, computed in FP32 with learned gain, cast back to input
  dtype.
- **SwiGLU FFN**: `down(silu(gate(x)) * up(x))`, bias-free.
- **Marker readout**: gathers the hidden state at `<goal>` and at each `<opt>`
  marker, projects them with `W_q` and `W_o`, and computes cosine similarity.
  Padding markers are excluded from the gather.
- **Task temperature and affine**: `tau = exp(log_tau)` (zero-init means 1),
  plus per-task scale/bias; scores are `scale * (cos / tau) + bias`.
- **Heads**:
  - `ProbabilityHead`: CLS → 384→192 GELU → 1 sigmoid; serves both probability
    and answerable (the tensors are intentionally the same).
  - `ConfidenceHead`: concat(CLS, top-1 option score, top-1/second margin,
    entropy) → 387→128 GELU → 1 sigmoid.
  - `RiskHead`: CLS → 384→192 GELU → 8 sigmoid.
- **Typed validation**: option count 2–8; marker presence/order; binary masks;
  device agreement; task id range. Invalid inputs raise typed `ModelError`
  subclasses before compute.

### 3.4 Risk categories

Eight fixed slots, in order:

`violence`, `self_harm`, `sexual_content`, `hate_speech`, `harassment`,
`dangerous_content`, `illegal_activity`, `privacy_violation`.

---

## 4. Data lifecycle

The full real-data path is
`raw parquet → adapters → clean → split → mixture → tokenize → batch`.

### 4.1 Acquisition and provenance

- Licensed sources are written to parquet shards with SHA-256 hashes and source
  manifests recording license, status (`admitted` / `quarantine`), observed and
  raw row counts, adapter identity, shard layout, rejected rows, and per-shard
  adapter reports.
- A corpus with no license is quarantined. The mixture refuses sources that are
  not explicitly admitted.
- Raw-shard and unified-shard layouts are distinct; reusing a raw acquisition
  with an adapter requires `refresh=True`.
- No real corpus is shipped in the repository. Without `--data-dir`, the CLI
  uses deterministic synthetic fixtures and labels them as synthetic.

### 4.2 Six task-family adapters

| Family | Source shape | Emitted labels |
|---|---|---|
| choice | MMLU-style multiple choice | `chosen_index` |
| preference | chosen/rejected response pairs | two option orders, `chosen_index` |
| scoring | attribute ratings | normalized `scores` per option |
| probability | TruthfulQA proposition / MMLU binary derivation | `answerable` as binary truth |
| abstention | answerable/unanswerable examples | `answerable`, `chosen_index` |
| risk | Aegis-style prompts and categories | `risk_labels`, no `chosen_index` |

All rows are normalized to `DecisionExample` with NFC text, HTML stripping,
whitespace collapse, content-hash `example_id`, unique option texts, source,
license, group_id and split.

### 4.3 Cleaning

- Exact dedup removes byte-identical normalized content.
- Near dedup uses MinHash over character shingles; default Jaccard threshold
  0.9.
- Label payloads (gold text, scores, answerable, risk labels) participate in the
  duplicate/contradiction decision: conflicting annotations are quarantined,
  never silently merged.
- Quality rules Q1–Q6 are structural, with per-rule counters. Q1 exempts only
  declared fixed option vocabularies such as `Yes`/`No`, `safe`/`unsafe`.
- A corpus that quarantines every row fails the build.

### 4.4 Splitting

- Group-aware deterministic 70/15/15 targets with relative water-filling.
- Small groups still get non-empty train/dev/test when enough groups exist.
- OOD source `UltraFeedback-binarized` is pinned to test and never appears in
  train or dev; rows sharing its group also follow it to test.
- Fatal checks: no group overlap, no long option-text overlap, no residual
  13-gram overlap.
- Split fingerprint is `sha256` over `(example_id, split)` membership.

### 4.5 Mixtures

| Preset | Rows | Family counts |
|---|---:|---|
| starter | 55,000 | choice 12k, preference 20k, scoring 5k, probability 4.6k, abstention 6.6k, risk 6.8k |
| full | 115,600 | choice 30k, preference 33k, scoring 11k, probability 11.6k, abstention 12k, risk 18k |

Only **train** rows can bind to mixture slots. Dev/test/OOD rows are ineligible.
Every declared source must be admitted by a manifest; missing or unknown status
is a typed error. Preference's two option orders are separately addressable as
occurrences of the same content hash.

### 4.6 Tokenization and batching

- Stand-in tokenizer: deterministic, dependency-free, word-piece-like; printed
  ASCII pieces, `##` continuations, a fixed set of digraph/suffix pieces, space
  as `Ġ`, 257 total pieces.
- Special marker tokens: `<task>`, `<goal>`, `<state>`, `<opt>`, `</opt>`.
- `encode_example` enforces a 512-token budget. Long states are truncated from
  the end; goal/option blocks are preserved where possible.
- Truncations are counted per source in a caller-owned manifest, never in global
  mutable state on production paths.
- `bucket_examples` emits at most three padded widths (`budget//4`, `budget//2`,
  `budget`) and builds right-padded `input_ids`, `attention_mask`, `option_mask`
  and `task_ids`.

---

## 5. Training

### 5.1 Training configuration

Production `starter.yaml` / `acceptance.yaml` use:

| Setting | Value |
|---|---|
| seed | 13 (first configured seed; `RAPPORT_SEED` and `--seed` can override) |
| device | CUDA if available, else CPU; FP16 only on CUDA |
| epochs | 8 |
| batch size | 64 |
| encoder LR | 3.0e-4 |
| head LR | 1.0e-3 |
| min LR | 1.0e-5 |
| weight decay | 0.01 |
| warmup | 5% linear |
| schedule | cosine to min LR |
| gradient clip | global norm 1.0 |
| dropout | 0.1 |
| optimizer | AdamW, betas (0.9, 0.98), eps 1e-8 |
| weight mode | `fixed` by default; learned/fixed/uniform supported |
| checkpoint cadence | configured via `checkpoint_every`; retention 3 |
| determinism | strict (backend-aware on MPS) |

Planned optimizer steps:

- starter: `ceil(55,000/64) × 8 = 6,880`
- acceptance: `ceil(115,600/64) × 8 = 14,456`

### 5.2 Optimizer and precision

- AdamW parameters are split into no-decay vs decay and head vs trunk groups.
  Norms, biases, embeddings, task vectors and `log_s` are no-decay.
- FP16 autocast wraps forward and loss; loss is computed in FP32; `GradScaler`
  scales the backward pass, gradients are unscaled, finite-checked, clipped,
  then applied with `scaler.step`.
- Non-finite loss or gradient produces a typed skip: no optimizer step, no
  scheduler step, no step increment, with per-epoch divergence counting.
- A batch whose labels make every task inactive is a typed skip, not a crash.
- Deterministic algorithms and cuDNN flags are process configuration, applied
  separately from RNG seeding. On MPS, `warn_only=True` is used because some
  MPS kernels lack deterministic implementations.

### 5.3 Seven-task multitask loss

Default fixed weights:

| Term | Weight | Objective | Labels |
|---|---:|---|---|
| choice | 1.0 | masked cross-entropy | `chosen_index` |
| preference | 1.0 | pairwise BCE on logit difference | `preference_pairs` |
| score | 0.5 | Huber (δ=0.1) | `scores`, `score_mask` |
| probability | 0.5 | BCE + 0.5·Brier | `probability_target`, mask |
| confidence | 0.5 | BCE on correctness | derived argmax vs `chosen_index` |
| abstain | 1.0 | BCE | `answerable_target`, mask |
| risk | 0.5 | per-category weighted BCE | `risk_targets`, mask |

Important properties:

- Padding slots never contribute to choice loss or confidence target.
- Every term is gated by a mask; absent labels mean `active=False`, not a
  fabricated zero target.
- Losses are computed in FP32 where required and checked for finiteness; no
  clamping hides NaNs.
- `weight_mode=learned` uses Kendall-style uncertainty weighting:
  `Σ_t exp(-s_t) L_t + s_t`, with `s_t = log_s_t` clamped so weights stay in
  `[0.05, 4.0]`; `log_s` is a trainable optimizer parameter and participates in
  checkpointing.
- `weight_mode=fixed` reproduces the §3.2 default weighted sum.
- `weight_mode=uniform` gives all active tasks weight 1.0 and omits the
  regularizer.

### 5.4 Checkpoint and resume

Each checkpoint is `checkpoint_stepN.pt` plus a `.sha256` sidecar.

- Payload: step, epoch, sampler offset, non-finite counters, model state,
  optimizer state, scaler state, scheduler state, weighter state, RNG states,
  resolved training config.
- Format v2 is `weights_only=True` safe; numpy RNG arrays are stored as explicit
  dtype/shape/bytes records and restored on load.
- Writes are atomic: temp file → fsync → `os.replace` → directory fsync; the
  sidecar is verified before deserialization.
- Load is transactional: the full schema is validated before any live state is
  overwritten.
- Resume accepts relocation to a new checkpoint directory but refuses real
  optimizer/LR/horizon/config mismatches.
- Full kill/resume equality is regression-tested to 1e-4; learned-weighting
  resume is bit-exact.

---

## 6. Calibration, confidence and abstention

Implemented in `src/calibration/`; independent of `src/evaluation/` by design.

- **Bucket keys**: task family × option-count bucket.
- **Temperature scaling**: single positive temperature per bucket; fitted on dev
  logits with 5-fold group-aware cross-fitting and NLL grid search.
- **Platt scaling**: used only for 50–200 item buckets, fitting
  `P(top-1 correct)`; it never changes the chosen option.
- **Small buckets** (<50 items) inherit the pooled temperature and are flagged.
- **ECE metrics**: pooled and bucket-macro ECE, Brier score, NLL; 15 fixed
  confidence bins.
- **Abstention policy**: utility triple `(u_correct=1, u_wrong=-3,
  u_abstain=0)` gives threshold `(0 - (-3)) / (1 - (-3)) = 0.75`. The decision
  is made by comparing expected utilities, which also handles reversed
  utility triples correctly.
- **Risk thresholds**: per-category dev thresholds chosen to hold FPR ≤ 0.10 on
  safe dev rows; XSTest is never used for fitting. A tampered
  `xstest_used_for_fitting` flag or malformed frozen policy is rejected.
- **Calibration map**: frozen before test is read; stores bucket temperatures,
  item counts, ECE before/after, folds, abstention policy and risk thresholds.

---

## 7. Evaluation

### 7.1 Metrics

`src/evaluation/metrics.py` implements:

- accuracy, macro-F1,
- NDCG@k (with `k > n` handled),
- Spearman and Pearson correlation,
- utility MAE,
- Brier score, NLL, pooled ECE, bucket-macro ECE,
- AURC and selective accuracy at 70/80/90% coverage, with tie-aware risk
  integration and honest achieved-coverage reporting,
- per-category risk recall at frozen FPR threshold,
- over-refusal rate on safe rows,
- keyword-rule risk baseline.

Non-finite or malformed evidence is `None` or a typed error, never a numeric
pass.

### 7.2 Baselines

- **B1**: majority class.
- **B2** / optional embedding fallback: hash-ngram or local frozen embeddings,
  explicitly labeled when the real embedding source is unavailable.
- **B3**: TF-IDF char+word n-grams → logistic regression, fitted on train only.
- **B4**: frozen embedding cosine, fitted/used on train evidence only; carries
  its own embedding source label.
- Keyword rule comparator for risk.

### 7.3 Robustness harness

- permutation invariance over option order,
- option-count sweep (2–8),
- held-out source block with train/dev exclusion,
- contradictory/missing-information context transforms over the same retained
  row IDs,
- 512→1024 length sweep,
- finite-output validation; NaN/infinity makes a measurement not-measured.

### 7.4 Y1–Y14 acceptance verdicts

All Y rows are blocking. Missing evidence renders `not-measured`; acceptance
then exits non-zero. Starter mode is explicitly non-acceptance.

| Criterion | Threshold |
|---|---|
| Y1 choice accuracy vs B3 | +8.0 points |
| Y2 held-out-source accuracy vs B4 | +5.0 points |
| Y3 bucket-macro ECE | ≤ 0.05 |
| Y4 pooled ECE | ≤ 0.05 |
| Y5 selective accuracy at 80% vs full | +5.0 points, same cohort |
| Y6 permutation | within tolerance and accuracy delta ≤ 0.5 points |
| Y7 risk recall at FPR ≤ 0.10 | ≥ 0.85, Aegis cohort, frozen dev thresholds |
| Y8 XSTest safe over-refusal | ≤ 0.10, frozen thresholds, nonzero cohort |
| Y9 p50 batch-1 latency | ≤ 12 ms on T4 FP16, synchronized |
| Y10 batch-32 throughput | ≥ 300 decisions/s on T4 FP16 |
| Y11 parameters / FP32 size | ≤ 35M / ≤ 140 MB |
| Y12 INT8 accuracy drop | ≤ 1.0 point, real five-condition gate |
| Y13 kill/resume next-step loss gap | ≤ 1e-4 |
| Y14 seed reproducibility | ≤ 0.5 accuracy points across duplicate runs |

Per-seed verdicts are aggregated with precedence `fail > not-measured > pass`,
so a measured failure can never be erased by a later missing seed.

---

## 8. Inference, export and benchmark

- `Predictor` requires trained weights for config-based construction; random
  weights are never served as a trained model.
- Request validation enforces 2–8 options; the model is set to `eval()` and
  called under `no_grad()`.
- Temperature scaling changes ranking only if the mapping is monotone; Platt
  correctness is kept separate from categorical choice probabilities.
- ONNX export uses dynamic axes for batch, sequence and option dimensions,
  checks the graph, loads an ORT session, and compares FP32 output against both
  the export wrapper and the real model.
- ONNX FP16 conversion refuses to emit an artifact if the converter produces an
  invalid graph (as it does on the validated ONNX build).
- INT8 dynamic quantization preserves the cosine projection path in FP32,
  evaluates per-head drift, and refuses the artifact on validation failure.
- Five-condition export gate: choice accuracy drop, bucket ECE rise, cosine
  path, serialized size, p95 latency speedup.
- Benchmark derives p50/p95/p99 from one synchronized latency array. INT8
  speedup uses the FP32 ONNX Runtime session as baseline, never torch FP32.
  Skipped measurements remain `None`; they are never reported as zero.

---

## 9. Operations, reproducibility and security

- Single validated config boundary: base YAML < experiment YAML < `RAPPORT_*`
  env < explicit overrides. Unknown keys are rejected by name.
- The runner refuses experiment knobs it cannot honor and prints declared but
  unexecuted alternates; examples: pretrained `init`, curriculum ordering, KD.
- Three seed sources with precedence: config first seed < `RAPPORT_SEED` <
  `--seed`.
- Run manifests record run id, git SHA/dirty state, config hash, data
  fingerprint, seed, device, determinism, versions, and dependency lock digest.
- `code_fingerprint` covers all `src/` Python, `scripts/`, mixture presets and
  the dependency lock; changing training code invalidates stage artifacts and
  resume identity.
- JSONL logs: step metrics, eval events, benchmark events, warnings.
- Registry validates required manifest fields and directory/ID agreement and
  excludes malformed runs without aborting.
- Checkpoints are safe (`weights_only=True`); path traversal in arm/run_id is
  rejected; JSON artifacts are atomically written.
- Packaging exposes `src.*` and `rapport`; mixture presets are package data.

---

## 10. Complete technology stack

### Runtime and libraries

- Python 3.11+ (validated 3.12.12)
- PyTorch 2.15.0 (`torch.nn`, `torch.nn.functional`, SDPA, `GradScaler`,
  autocast, `clip_grad_norm_`)
- NumPy 2.5.5
- PyYAML
- pandas / PyArrow for parquet I/O
- transformers 5.23.0 and datasets 5.2.0 are declared for source/teacher
  integration; the shipped tokenizer is a local stand-in
- scikit-learn 1.9.1 for TF-IDF/logistic baselines
- ONNX 1.23.2 and ONNX Runtime 1.32.0 for export/quantization/benchmark
- psutil for memory telemetry
- pytest, ruff, mypy for verification
- setuptools packaging with pinned `requirements.lock`

### Engineering patterns

- typed error classes with stable codes,
- atomic file writes and SHA-256 sidecars,
- content-hashed row, dataset, config and code identities,
- fail-closed evidence aggregation,
- dataclass-based immutable configs/results,
- registry/log/manifest provenance,
- dependency pinning and CI smoke tests.

---

## 11. ML/DL concepts index

| Concept | Where it is used |
|---|---|
| Transformer encoder | `RapportTrunk`: embedding + N pre-norm blocks |
| Self-attention | `MultiHeadSelfAttention`; all tokens/options attend bidirectionally |
| Scaled dot-product attention | PyTorch SDPA with default `1/sqrt(head_dim)` scaling |
| Padding mask | additive `finfo.min` mask over padded keys; no causal mask |
| Multi-head attention | compact: 6 heads × 64 dims |
| RoPE | half-split rotary embedding, base 10,000, per layer |
| Relative position encoding property | RoPE dot products depend on token distance |
| RMSNorm | pre-norm in every block, FP32 internals, learned gain |
| Pre-norm residual blocks | `x + attn(norm1(x))`, then `x + ffn(norm2(x))` |
| SwiGLU FFN | gated feed-forward with SiLU, bias-free |
| Embedding scaling | token embeddings multiplied by `sqrt(d_model)` |
| Task conditioning | learned task vector added at `<task>` marker |
| Marker-based set readout | gather `<goal>`/`<opt>` hidden states; option order equivariance |
| Cosine similarity scoring | normalized dot product between projected query/options |
| Learned temperature | `tau = exp(log_tau)`, positive by construction |
| Listwise softmax | masked softmax over real options only |
| Multitask learning | seven losses with fixed/learned/uniform weighting |
| Uncertainty weighting | Kendall `Σ exp(-s)L + s` with bounded weights |
| Cross-entropy | choice and preference BCE |
| Pairwise ranking loss | winner-minus-loser BCE |
| Huber loss | robust utility regression |
| BCE + Brier | probability head training |
| Multi-label classification | eight-way risk head with weighted BCE |
| Calibration | temperature scaling, Platt scaling, ECE/Brier/NLL |
| Expected-utility decision theory | abstention threshold from utility triple |
| Risk-coverage / AURC | selective prediction and confidence ordering |
| Conformal-like thresholding | frozen dev risk thresholds at fixed FPR budget |
| OOD evaluation | withheld source held out of train/dev |
| Leakage control | group-aware splits, dedup, 13-gram quarantine |
| Deterministic hashing | content IDs, split/config/code fingerprints |
| Entropy | confidence-head feature and distillation gating |
| Distillation | local teacher soft labels, content-addressed cache, KD gate |
| Dynamic quantization | ONNX INT8 weights, preserved FP32 cosine path |
| Mixed precision | FP16 autocast + loss scaling + FP32 loss reductions |
| Gradient clipping | global-norm clipping over all optimized parameters |
| AdamW | decoupled weight decay with parameter groups |
| Warmup + cosine schedule | 5% linear warmup then cosine decay |
| Deterministic algorithms | strict/relaxed process modes, RNG checkpointing |

---

## 12. Commands

```sh
python -m pip install -e '.[dev]' -c requirements.lock
python -c 'import rapport; print(rapport.__version__)'

make probe
make e2e                            # whole pipeline
python -m scripts.e2e --starter --synthetic --steps 4 --runs-dir <tmp>
python -m scripts.e2e --synthetic --acceptance --steps 4 --runs-dir <tmp>

python -m pytest -q
ruff check .
mypy src
```

`--stage` accepts `acquire|clean|adapt|split|sample|train|calibrate|report|
export|bench|data|e2e`; `data` is the acquire+clean+adapt chain. Real-data runs
require `--data-dir` with parquet shards and admitted source manifests.
Acceptance mode without real evidence exits non-zero by design.

---

## 13. Known limits and non-executable declared arms

These are not hidden and are not claimed as measured:

1. The trunk trains **from scratch**; no pretrained `bge-small` loader exists, so
   the A1 pretrained-init comparison cannot run.
2. Mixture ordering is deterministic **fingerprint order**; curriculum ordering
   (A7) is not implemented.
3. KD is implemented as modules/gates, but the default training runner does not
   execute a teacher KD arm.
4. No real corpus is shipped; synthetic fixtures are not acceptance evidence.
5. No CUDA/T4 timing or VRAM measurement was possible in the CPU/MPS validation
   environment. GPU paths were verified structurally and on MPS where possible.
6. FP16 ONNX conversion is refused on the validated ONNX build; FP32 export and
   the INT8 gate remain available.
7. Full Y1–Y14 acceptance requires real dev/test evidence and T4 measurements;
   the current acceptance CLI correctly exits non-zero until those exist.

---

## 14. Verification record

At the repair baseline:

- full suite: **1634 passed, 1 skipped**,
- `ruff check .`: clean,
- `mypy src`: clean (102 source files),
- starter smoke: exit 0, `mode=starter`,
- synthetic acceptance: exit 1, blocking verdicts `not-measured`,
- built wheel includes `src/data/mixture/presets/{starter,full}.yaml`,
- trained-model resume tested bit-exactly for learned weighting,
- GPU blocker reproduced and fixed on MPS where the hardware allowed it.

This document should be updated whenever architecture, training semantics,
evidence rules, or declared limits change.
