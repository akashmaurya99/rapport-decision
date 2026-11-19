# rapport — plan (LARGE)

_Plan of record for the RAPPORT research build. Generated 2026-11-19T13:47:02Z. All paths repo-root-relative. Execution contract: `.archgen/rapport/tasks.yaml`; implementation rationale: `docs/architecture-spec.md`; source of truth for data and API contracts: `architecture.yaml`._

## Profile

- Scope: **LARGE** — a net-new multi-module system (10 modules, 25 tasks) with a public model artifact and measurable external validity targets; no host code exists, so no survey was needed.
- Create: `.archgen/rapport/answers.yaml`, `docs/prd.md`, `architecture.yaml`, `docs/rapport.md`, `docs/mermaid-snippets.md`, `tasks.yaml`, `plans/rapport.md`, plus the research package `docs/research.md`, `docs/architecture-spec.md`, `docs/dataset-strategy.md`, `docs/training-strategy.md`, `docs/t4-optimization.md`, `docs/evaluation.md`, `docs/risks.md`.
- Reuse: none (greenfield).
- Omit: `codebase-map.md` (no host code to survey); `decisions/*.md` (no irreversible, regulated or data-migration trigger — decisions and their falsification tests live in `docs/architecture-spec.md` §2 and §10).
- Escalate if: a hosted paid teacher becomes mandatory; the parameter budget rises above 60M; a regulated or personal-data corpus enters the mixture.

```yaml
scope_class: "LARGE"
intent_class: "GREENFIELD-SYSTEM"
signals: ["new-module", "multi-module", "product-facing", "prototype"]
artifacts:
  - path: ".archgen/rapport/answers.yaml"
    decision: "create"
    reason: "captured requirements and the plan profile"
  - path: ".archgen/rapport/docs/prd.md"
    decision: "create"
    reason: "product-facing system with measurable acceptance targets"
  - path: ".archgen/rapport/architecture.yaml"
    decision: "create"
    reason: "ten modules with cross-module contracts"
  - path: ".archgen/rapport/docs/rapport.md"
    decision: "create"
    reason: "topology, dependency direction and failure domains"
  - path: ".archgen/rapport/docs/mermaid-snippets.md"
    decision: "create"
    reason: "integration and model diagrams"
  - path: ".archgen/rapport/docs/research.md"
    decision: "create"
    reason: "the required research review deliverable"
  - path: ".archgen/rapport/docs/architecture-spec.md"
    decision: "create"
    reason: "the required architecture specification deliverable"
  - path: ".archgen/rapport/docs/dataset-strategy.md"
    decision: "create"
    reason: "the required dataset strategy deliverable"
  - path: ".archgen/rapport/docs/training-strategy.md"
    decision: "create"
    reason: "the required training strategy deliverable"
  - path: ".archgen/rapport/docs/t4-optimization.md"
    decision: "create"
    reason: "the required T4 optimization deliverable"
  - path: ".archgen/rapport/docs/evaluation.md"
    decision: "create"
    reason: "the required evaluation plan deliverable"
  - path: ".archgen/rapport/docs/risks.md"
    decision: "create"
    reason: "the required risk assessment deliverable"
  - path: ".archgen/rapport/tasks.yaml"
    decision: "create"
    reason: "the task contract every plan needs"
  - path: ".archgen/rapport/plans/rapport.md"
    decision: "create"
    reason: "the plan of record"
  - path: ".archgen/rapport/codebase-map.md"
    decision: "omit"
    reason: "not applicable to GREENFIELD-SYSTEM"
  - path: ".archgen/rapport/decisions/0001-<title>.md"
    decision: "omit"
    reason: "no irreversible, regulated or data-migration trigger"
escalate_if: ["a hosted paid teacher becomes a requirement", "the target parameter count rises above 60M", "a regulated or personal-data corpus enters the mixture"]
gate:
  verifier: "general sub-agent, fresh session (round 1 exhaustive) + general sub-agent, fresh session (round 2 delta)"
  rounds: 2
  findings:
    blocking: 5
    nit: 5
    observation: 2
  waived: 0
  residual: 0
  at: "2026-11-19T14:45:51Z"
```

## Context

- Why now: The problem RAPPORT addresses is now a named category. In November 2026, published work shows typed decision models match frontier LLMs on bounded classification at far lower cost, but also that they cluster around 0.6B–27B parameters and fail most often on **abstention** and **option position**. No published result measures the 5M–60M regime. That regime is exactly what a single free T4 can train, and exactly where a bad result is cheap to find and honest to report.
- What exists: Nothing in this repository. The starting point is a 33M-parameter pretrained English encoder (`BAAI/bge-small-en-v1.5`, MIT), a set of licensed public corpora, and the evidence in `docs/research.md`.
- What changes: The repository becomes a reproducible, config-driven pipeline: acquire → clean → split → sample → train → calibrate → evaluate → export → benchmark. Every step writes a manifest; every number in the report resolves to a config hash and a data fingerprint.
- Constraints: T4 sm_75, FP16 + loss scaling (no BF16), 2 vCPU, ~12.7 GB RAM, ephemeral disk, no network during training, no paid service, English only, 512-token training budget with a measured 1024 extension.

## Phases and waves

Wave labels are the floor-derived values (`archgen_plan` re-derives them from the DAG).

- **P0 — probe and scaffold (wave 1):** TASK-01, TASK-02.
- **P1 — contract and data foundation (waves 2–5):** TASK-03, TASK-05, TASK-17, TASK-20 (2) → TASK-04, TASK-07, TASK-09 (3) → TASK-06, TASK-08 (4) → TASK-10, TASK-11 (5).
- **P2 — model (wave 4):** TASK-06 joins P1's tail.
- **P3 — training (waves 5–8):** TASK-11 (5) → TASK-13 (6) → TASK-12 (7) → TASK-14 (8).
- **P4 — Stage C and D (waves 5–8):** TASK-16 (5, after TASK-06) → TASK-15 (8, after TASK-12).
- **P5 — evaluation (waves 2–9):** TASK-17 (2) → TASK-18, TASK-19 (6–7) → TASK-21 (9).
- **P6 — ship (waves 6–10):** TASK-22 (6) → TASK-23 (7) → TASK-24 (8) → TASK-25 (10).

## Per-task sections

### TASK-01 — Config resolution, validation and run manifests

Approach: one pure loader (`src/config/loader.py`) → validated dataclass; one manifest writer (`src/config/manifest.py`) stamping git sha, config hash, environment and library versions. Fail fast on unknown keys. `configs/base.yaml` is the only hand-edited config in the repo root of `configs/`.

Edge cases: missing env override, unknown key, unreadable YAML, dirty git tree (recorded, not refused).

Out of scope: schema for experiment configs beyond what `configs/experiments/*.yaml` needs (delivered by TASK-13).

### TASK-02 — Repository skeleton, toolchain, CI and T4 probe

Approach: minimal `pyproject.toml` with extras `train`, `eval`, `export`, `dev`; `Makefile` exposing `probe data split train calibrate report export bench e2e`, where `probe` is an inline `python -c` one-liner printing capability, VRAM and SDPA backends with no extra module, and every other target delegates to `python -m scripts.e2e --stage <name>`. README points at `.archgen/rapport/`.

Edge cases: wrong GPU allocation, missing CUDA, flash-attn absent (must be absent, not required).

Out of scope: packaging onto PyPI; Docker.

### TASK-03 — DecisionExample contract and normalization

Approach: frozen dataclass + JSON/parquet round-trip; `example_id` = sha256 of (task_family, NFC goal, NFC state, the sorted NFC option texts) so permuting options leaves the id unchanged but distinct states produce distinct ids; validation returns typed error codes, never raises bare exceptions.

Edge cases: empty goal, one option, duplicate option, NFC vs NFD equivalence, out-of-range chosen index.

Out of scope: adapter-specific logic (TASK-09).

### TASK-04 — Tokenizer, special tokens and length bucketing

Approach: start from the init encoder's tokenizer, add `<task> <goal> <state> <opt> </opt>`; `encode_example` returns ids + masks + `truncated` boolean; bucketing by token-count quartile into three batch shapes; padding mask always `attention_mask=0` beyond the real length.

Edge cases: state longer than 512 (truncate with flag), option text longer than 45 tokens, zero-padding present in labels, tokenizer mismatch vs init encoder.

Out of scope: reshuffling the vocabulary; a new BPE.

### TASK-05 — Encoder trunk and parameter accounting

Approach: implement the trunk from `docs/architecture-spec.md` §4 with pre-RMSNorm, RoPE and SwiGLU; `count_parameters` walks the module and asserts against the spec constants; the assert is a build-time failure, not a warning.

Edge cases: a layer that silently adds parameters (e.g. a stray bias), dropout rate mismatch vs spec, empty padding in RoPE frequency cache.

Out of scope: the readout and heads (TASK-06).

### TASK-06 — Marker readout, six heads and model assembly

Approach: gather hidden states at `<goal>` and each `<opt>`; `W_q`, `W_o` projections; cosine score divided by a per-task learned temperature; per-task affine; six micro-heads; `RapportModel` bundles trunk + readout + heads with a single `forward`.

Edge cases: unused `<opt>` slots in a padded batch (masked by `option_mask`), n=2 vs n=8, division by a zero norm (epsilon, and a unit test), a task with no probability label (head masked), gradient flow through the temperature.

### TASK-07 — Dataset acquisition and provenance manifests

Approach: thin streaming downloaders per source (`datasets.load_dataset(..., streaming=True)`), filtered at write time to the columns the adapters need; each download records license from the Hub API, row counts, sha256 of the written shard; runs are resumable by hash.

Edge cases: Hub rate limiting, partial download, license field missing (quarantine), disk exhaustion (fail loudly), corpus that cannot stream (skip it with a recorded reason).

### TASK-08 — Deduplication, quality filtering and contradiction audit

Approach: normalize → sha256 exact dedup → `datasketch` MinHash/LSH near-dedup → named rules Q1–Q6 with counters → contradiction audit grouping by (goal, state, sorted options) and quarantining conflicting groups.

Edge cases: option order differences (dedup must be order-insensitive), near-duplicate gold rationales, "none of the above" options (kept), a source that is 90% near-duplicates (report, do not silently drop its entire family).

### TASK-09 — Six task-family adapters into the unified schema

Approach: one module per family with a shared `Adapter` protocol; each declares the source file, the license, and the mapping; the AbstentionBench adapter records per-subset conversion rate; MMLU-Pro reduces 10→8 options by seeded non-gold removal, reported.

Edge cases: a source with a missing split, a label that cannot map to a family, a row whose options fail validation, extremely long HH-RLHF transcripts (truncate with flag).

### TASK-10 — Deterministic group-aware splits, leakage checks and fingerprints

Approach: stable hash of `group_id` → bucket; stratified by family to hit 70/15/15; leakage check searches for any exact 13-gram or 40-token option substring in two splits and fails the build; writes `SplitManifest`.

Edge cases: a group spanning all three splits, a source too small to stratify (recorded), a withheld source that appears in dev by mistake (fatal).

### TASK-11 — Multi-task loss functions

Approach: pure functions on tensors with explicit masks; each loss has a closed-form test; pairwise BCE for preference samples pairs from the batch; listwise CE for choice; Huber for utility; BCE + 0.5·Brier for probability; BCE for confidence and answerability; pos_weight BCE for risk.

Edge cases: no valid pair in a batch (term = 0.0), all labels masked (term = 0.0, no gradient), Brier weight λ=0.5, a zero temperature (epsilon).

### TASK-12 — FP16 trainer core with checkpoint and resume

Approach: single-process `Trainer` class; FP16 autocast + `GradScaler`; gradient clipping; cosine schedule with warmup; atomic checkpoint via tmp + rename + fsync; RNG states for python/numpy/torch/cuda; the sampler offset.

Edge cases: loss is non-finite (skip step, count), checkpoint half-written (hash mismatch → refuse), resume from a checkpoint whose step ≥ config steps (refuse), CUDA OOM (fail with the memory diagnostics printed).

### TASK-13 — Declarative mixtures and experiment configs

Approach: one YAML per source with (family, rows, weight); `sample_mixture` takes a seed and a fingerprint and reproduces the exact order; `configs/experiments/*.yaml` one per ablation A1–A8, plus `starter.yaml` (55k smoke) and `acceptance.yaml` (the 115.6k Y1–Y14 run, recording the withheld OOD source UltraFeedback-binarized).

Edge cases: a source with fewer rows than its quota (take all, record shortfall), an unknown family (typed error), a fingerprint mismatch against the recorded manifest (fail).

### TASK-14 — Adaptive multi-task weighting and per-task logging

Approach: `TaskWeighter` over per-task losses with `s_t = log σ²`, floor 0.05, cap 4.0; weights and the `WEIGHT_COLLAPSE` event are logged; the fixed-weight arm is one config away.

Edge cases: a task absent from a batch (weight unchanged, term skipped), collapse to the floor (event), a NaN in a term (skip the whole step).

### TASK-15 — Local-teacher soft-label distillation and its gate

Approach: NF4 4-bit Qwen3-1.7B on the T4, greedy; prompt asks for a one-token-per-option probability judgement; cache keyed by hash; gate compares dev accuracy teacher vs supervised student before any student step; gate is a pure function of recorded numbers.

Edge cases: teacher OOM (drop to the next smaller local teacher, record it), empty cache with a missing teacher (abort), cache hash mismatch (refuse to use the entry), teacher slower than budget (stop and record).

### TASK-16 — Per-bucket calibration and the abstention policy

Approach: bucket by (task_family, option-count bucket); 5-fold group-aware CV; grid NLL search over T; a bucket of 50–200 items uses Platt instead of a single temperature; a bucket below 50 inherits the pooled temperature and is flagged; per-category risk thresholds frozen on dev XSTest to FPR ≤0.10 before the test split is ever read; `abstain` derives from the utility triple, not the head alone.

Edge cases: an empty bucket, a bucket that improves on the test split by accident (CV prevents this), risk categories with zero positives (skipped, reported).

### TASK-17 — Metrics implementation

Approach: one `compute_metrics` entry returning a typed dict; NDCG and Spearman from stable implementations; ECE implemented from scratch in 15 bins with an option to pass bin boundaries; every metric tested on a fixture where the answer is known.

Edge cases: NDCG with n=1 (skipped), Spearman with constant utilities (nan → reported as undefined), ECE with n<10 per bin.

### TASK-18 — Classical and frozen-embedding baselines

Approach: TF-IDF char+word n-grams → logistic regression with a fixed C grid on dev; frozen `bge-small-en-v1.5` pooling → cosine choice; majority-class; all from the same split manifest.

Edge cases: features too large (cap vocabulary), a split lacking a family (report, do not fabricate), the held-out source failing to load (recorded).

### TASK-19 — Robustness harness: permutation, option-count sweep, held-out source

Approach: permute options and re-score; sweep option counts 2..8; evaluate the held-out source; apply contradictory and missing-context transforms; report abstention deltas and ECE deltas; run the 512/768/1024-token length sweep, treating a drop above 2 points at 1024 as a failure of the RoPE extension claim.

Edge cases: an example whose family disallows n=8 (skip and record), a source contributing to the sweep (exclude), permutation of an example without a chosen index (metrics skipped, still reported).

### TASK-20 — Structured logging, run registry and reproducibility checks

Approach: `JsonlLogger` writes one line per event; `RunRegistry` indexes manifests; `verify_resume` reruns 50 steps from a fixed seed and compares.

Edge cases: a corrupt JSONL line (registry skips and reports), a manifest with a missing key (excluded from registry), a run id collision (uuid suffix).

### TASK-21 — Ablation arms and the single evaluation report

Approach: one `build_report` that loads runs, computes §3 blocks, §5 ablations, §6 robustness, §7 benchmark, §8 verdicts; refuses to print a claim below the knowledge threshold; writes `report.json` + `report.md` with a negative results section.

Edge cases: a missing metric for an arm (verdict `not-measured`, never `pass`), a failed arm (kept), inconsistent fingerprints across runs compared in one table.

### TASK-22 — The DecisionResult prediction API

Approach: `Predictor` loads config + checkpoint, builds the tokenizer once, exposes `predict`; validates the request, returns a frozen `DecisionResult`, no free text.

Edge cases: empty options, >8 options, unicode input, missing state, a request for an unknown family (typed error and the family list).

### TASK-23 — ONNX export and INT8 dynamic quantization

Approach: export with dynamic axes on batch and option count; verify logits; quantize weights dynamically; per-head drift measurement; reject the artifact if the drift bound is exceeded.

Edge cases: ONNX op not supported at opset 17 (record, fallback to FP16), non-deterministic quantization, INT8 slower on the CI machine (report, do not force).

### TASK-24 — Inference benchmarks: latency, throughput, memory and size

Approach: warmup 50, 1,000 iterations, CUDA events where available; CPU path with pinned thread count; RSS/VRAM sampling; sizes on disk; environment fingerprint in the report.

Edge cases: noisy clocks, a competing process on the GPU (detect and refuse), INT4-looking results (out of scope), the OOD machine (not measured, not claimed).

### TASK-25 — End-to-end acceptance run and report

Approach: `scripts/e2e.py` wires every stage with a `--resume auto` flag; a `--kill-after-step` test path simulates preemption; `--config` picks `acceptance.yaml` (the 115.6k Y1–Y14 run, Y2 included) or `--starter` (the 55k smoke path, no verdicts); prints the Y-verdicts at the end.

Edge cases: a failed stage mid-run (stop, record, exit non-zero), a resume from the wrong config (fingerprint mismatch → refuse), Drive unavailable (local paths only).

## Edge-case matrices

| Case | Input/state | Expected behavior | Covered by |
| --- | --- | --- | --- |
| E1 | empty/null: empty option list or missing state text | predictor returns a typed error, never a traceback or a default answer | TASK-22#1 |
| E2 | boundary: 2 options minimum and 8 maximum | both return n scores; 1 or 9 are refused by validation before the forward pass | TASK-06#1 |
| E3 | invalid input: duplicated option after normalization | example rejected at construction with a distinct error code | TASK-03#1 |
| E4 | leakage: the same example appears in train and dev | group-aware split assertion fails the build before training | TASK-10#2 |
| E5 | training failure: non-finite gradient on one step | optimizer step skipped, counter incremented, weights unchanged | TASK-12#2 |
| E6 | partial failure: checkpoint corrupt mid-write | resume refuses to load it and never uses a partial checkpoint | TASK-12#3 |
| E7 | ordering: option permutation applied to the same input | scores permute exactly within 1e-4, accuracy delta reported | TASK-19#1 |
| E8 | dependency failure: teacher unavailable or below the gate | Stage C aborts and records a negative result in the report | TASK-15#2 |
| E9 | quantization: INT8 drift exceeds the declared bound | export is refused and the FP16 artifact is retained | TASK-23#2 |
| E10 | session failure: Colab preemption mid-epoch | resume reproduces the next-step loss within 1e-4 | TASK-25#1 |

Every row's citation resolves to an acceptance that claims it; no additional rows are claimed.

## Verification steps

1. `pip install -e .[dev] && pytest -q` — all unit tests pass, including the loss-against-fixture and the resume test.
2. `make probe` — prints capability, VRAM and available SDPA backends; values match the manifest.
3. `make data` — acquires the admitted sources, writes parquets, prints the dedup and contradiction counts.
4. `python -m scripts.e2e --stage split` — writes `data/processed/split_manifest.json` with `group_overlap: 0`.
5. `make train config=configs/experiments/starter.yaml` — 55k starter run; the log shows finite losses and a logged weight vector.
6. `make report` — produces `runs/<id>/report.md`; every Y-verdict rendered; negative results section non-empty.
7. `python -m scripts.e2e --config configs/experiments/acceptance.yaml --resume auto --kill-after-step 500` — smoke of the preemption path.
8. `pytest -q tests/unit/training/core -k resume` — the loss-identical-resume test passes.
