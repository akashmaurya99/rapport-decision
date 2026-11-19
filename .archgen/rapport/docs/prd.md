# PRD — RAPPORT

_RAPPORT is a compact, non-autoregressive decision model. It reads a bounded decision state plus a variable set of candidate options and returns a choice, per-option scores, a calibrated probability, a confidence estimate, an abstention decision and a risk vector — in one forward pass, with no generated text._

## Problem statement & goals

- A caller that needs a bounded decision today either pays a generative model whose latency and cost scale with output tokens, or ships a classifier that returns a label with no usable probability. Neither gives a cheap, calibrated, abstaining decision primitive.
- Goals:
  1. One forward pass returns a structurally typed decision for 2–8 options within a 512-token budget.
  2. Decision quality on the curated bench beats a TF-IDF logistic-regression baseline by ≥8 points in-distribution and ≥5 points on a held-out source.
  3. Probabilities are usable: bucket-macro ECE ≤0.05 after calibration, reported next to pooled ECE.
  4. The model abstains deliberately, with a risk–coverage curve and a cost-derived threshold, rather than always answering.
  5. The whole artifact is ≤35M parameters, ≤140 MB FP32, and runs in ≤12 ms per decision on a T4 at batch 1.
- The v1 deliverable is the *plan of record plus a runnable pipeline*; scientific claims are limited to what the pre-registered acceptance run measures.

## Non-goals

- Open-ended generation, chat, summarization, or any output longer than a typed decision record.
- General world knowledge or frontier-LLM parity. RAPPORT is a bounded decision model, not a general assistant.
- Multilingual, multimodal, multi-turn or tool-executing behaviour in v1.
- Reinforcement learning, custom CUDA kernels, multi-GPU training, or a from-scratch tokenizer.
- A web service, UI, or hosted API. The deliverable is a library, a CLI and a model artifact.

## Personas / stakeholders

- **Application engineer** — job: put a decision in a request path with a latency budget. Success: one call returns `DecisionResult` with a calibrated probability and an `abstain` flag; p95 latency inside budget.
- **Research reviewer** — job: decide whether a claim is supported. Success: every number in a report resolves to a config, a seed, a data fingerprint and a run id.
- **Model operator** — job: keep training alive on a free Colab runtime. Success: a killed process resumes and reproduces the loss trajectory; export produces an INT8 artifact whose outputs match PyTorch within tolerance.

## Functional requirements — config

- FR-CONFIG-01: A run loads one YAML config, applies `RAPPORT_*` environment overrides, and fails fast with a typed error on an unknown key or a missing required key.
- FR-CONFIG-02: A run manifest records git sha, config hash, data fingerprint, seed, device name, compute capability, and the versions of Python, PyTorch, transformers and datasets.
- FR-CONFIG-03: Seeds are set for `random`, `numpy`, `torch` and CUDA, and the resolved determinism mode is recorded in the manifest.

## Functional requirements — data

- FR-DATA-01: Every example normalizes to the `DecisionExample` contract; validation rejects an empty goal, fewer than two options, duplicate option text, or a chosen index outside the option range.
- FR-DATA-02: Acquisition writes one manifest per source containing license, declared row count, observed row count, sha256 and acquisition date; a source with no declared license is marked `quarantined` and cannot enter the mixture.
- FR-DATA-03: Exact duplicates are removed by normalized-text hash and near-duplicates by MinHash at a documented Jaccard threshold; the removed counts are reported per source.
- FR-DATA-04: Quality filters apply explicit, named rules (length bounds, markup residue, label contradiction) and write a rejection report with a count per rule.
- FR-DATA-05: Six task-family adapters map source records into `DecisionExample` with a stable `source` and `task_family`; an unmappable record is counted, not silently dropped.
- FR-DATA-06: Splits are deterministic from a seed and grouped by prompt and source so no prompt appears in two splits; a leakage assertion fails the build when a group crosses splits.
- FR-DATA-07: The mixture is declarative (family → weight and cap) and reproduces an identical example order from the recorded fingerprint.

## Functional requirements — tokenization

- FR-TOK-01: The tokenizer adds and round-trips the special tokens `<task>`, `<goal>`, `<state>`, `<opt>` and `</opt>`; encoding then decoding preserves every marker boundary.
- FR-TOK-02: Examples are length-bucketed into fixed batches, with the padding policy and the truncation policy recorded per batch.

## Functional requirements — model

- FR-MODEL-01: The encoder trunk matches the declared layer/width/FFN spec and a parameter counter reports per-component counts within 1% of `docs/architecture-spec.md`.
- FR-MODEL-02: The readout emits exactly one score per option and is permutation-equivariant: permuting options permutes the scores and leaves the sorted score multiset unchanged within 1e-4.
- FR-MODEL-03: The six heads emit `choice_logits[n]`, `scores[n]`, `probability[1]`, `confidence[1]`, `answerable[1]` and `risk[8]` with the declared shapes for any `n` in 2..8.
- FR-MODEL-04: A single forward pass encodes the state and all options together; there is no per-option forward pass in the default path.

## Functional requirements — training

- FR-TRAIN-01: Every loss has a unit test against a hand-computed value within 1e-5, including the masked and degenerate-option cases.
- FR-TRAIN-02: The trainer writes a checkpoint every N steps and resumes to the same next-step loss within 1e-4 on a fixed batch.
- FR-TRAIN-03: Multi-task weighting produces a per-task weight vector that is logged each epoch and is inspectable from the run registry.
- FR-TRAIN-04: FP16 mixed precision uses a gradient scaler and gradient clipping; a non-finite gradient aborts the optimizer step and increments a counter instead of corrupting weights.

## Functional requirements — distill

- FR-DISTILL-01: Teacher soft labels are written to a content-addressed cache keyed by teacher id, prompt template hash and example id; a second run reads the cache and performs zero teacher forwards.
- FR-DISTILL-02: A distillation usefulness gate compares teacher-augmented and label-only validation accuracy and aborts distillation when the margin is below the pre-registered threshold.

## Functional requirements — calibration

- FR-CALIB-01: Temperature maps are fitted per bucket (`task_family` × option-count bucket) with k-fold cross-fitting and stored with the item count per bucket.
- FR-CALIB-02: Evaluation reports pooled ECE and bucket-macro ECE separately, plus Brier score and NLL.
- FR-CALIB-03: The abstention threshold derives from an explicit utility model (correct/wrong/abstain utilities) and emits a risk–coverage curve; per-category risk thresholds are fitted on dev to hold FPR ≤0.10 on dev safe rows (Aegis safe-labeled prompts), frozen before the test split is read; XSTest safe prompts are evaluation-only and are never used for fitting.

## Functional requirements — evaluation

- FR-EVAL-01: The metrics module computes accuracy, macro-F1, NDCG@k, Spearman correlation, Brier, NLL, pooled ECE, bucket-macro ECE, AURC and selective accuracy at fixed coverage.
- FR-EVAL-02: The baselines module runs majority-class, TF-IDF logistic regression, and frozen-embedding cosine scoring from the same split manifest as the model.
- FR-EVAL-03: The robustness harness measures option-permutation delta, accuracy across option counts 2..8, accuracy on a held-out source that contributed no training rows, and accuracy at 512/768/1024 tokens (RoPE extension is a measured extrapolation, not an assumption).
- FR-EVAL-04: The ablation runner emits one report covering every arm, every seed and every declared metric, with no arm omitted for a negative result.
- FR-EVAL-05: The report states a pass/fail verdict for each pre-registered acceptance criterion in `docs/evaluation.md`.

## Functional requirements — inference

- FR-INFER-01: The predictor returns `chosen_index`, `scores[]`, `probabilities[]`, `confidence`, `abstain`, `risk[]` and the calibrated temperature used, for any option count in 2..8.
- FR-INFER-02: ONNX export matches PyTorch logits within 1e-3 on a fixed batch; INT8 dynamic quantization is verified to stay within the declared accuracy drop.
- FR-INFER-03: The benchmark reports p50 and p95 batch-1 latency, batch-32 throughput, peak VRAM, peak RSS and serialized model size.

## Functional requirements — ops

- FR-OPS-01: Every training step, evaluation and benchmark emits one JSONL record with run id, step, split, metric set and wall-clock time.
- FR-OPS-02: A run registry indexes every run id with its manifest path, config hash and the checkpoint paths it produced.
- FR-OPS-03: A reproducibility check reruns a fixed seed for a bounded number of steps and compares the final metrics against the recorded ones within the declared tolerance.

## Non-functional requirements (concrete numbers mandatory; no adjectives)

- Performance: batch-1 p50 latency ≤12 ms on T4 FP16 at 512 tokens and 8 options; batch-32 throughput ≥300 decisions/s.
- Performance: a full 8-epoch training run on 115.6k examples completes in ≤120 minutes on T4.
- Memory: peak training VRAM ≤8 GB at batch 64; peak inference VRAM ≤1 GB at batch 32.
- Size: ≤35M parameters for the primary variant; ≤140 MB FP32, ≤70 MB FP16, ≤35 MB INT8 serialized.
- Reliability: process kill at any checkpoint boundary → resume reproduces the next-step loss within 1e-4; no silent data reordering.
- Reproducibility: same config + seed + data fingerprint → accuracy within 0.5 points across two runs.
- Security: no network call is made during training or evaluation; the optional teacher runs locally and its outputs are cached.
- i18n: English only; Unicode NFC normalization at ingest; no locale-dependent number formatting in reports.
- Accessibility: not applicable (library and CLI only).

## Success metrics

- Choice accuracy (in-distribution): TF-IDF logistic regression baseline → +8 points for RAPPORT-C.
- Choice accuracy (held-out source): frozen bge-small cosine baseline → +5 points for RAPPORT-C.
- Bucket-macro ECE: uncalibrated → ≤0.05 after per-bucket cross-fitted temperature scaling.
- Selective accuracy at 80% coverage: full-coverage accuracy → +5 points.
- Permutation delta: unknown → |Δaccuracy| ≤0.5 points, |Δscore| ≤1e-4.
- Risk recall at FPR ≤0.10 on held-out Aegis rows: rule baseline → ≥0.85.
- Over-refusal on XSTest safe prompts: unmeasured → ≤10%.
- Latency: unknown → p50 ≤12 ms on T4, p95 ≤150 ms on CPU INT8.

## Risks & mitigations

- **Below the capability floor** — published typed decision models start at 0.6B; 33M may lose to TF-IDF on held-out sources. Mitigation: the T18 baseline and the T21 ablation report make this a pass/fail gate, and the 52M variant is the pre-registered fallback. Owned by T21.
- **Construct validity** — MMLU/ARC measure knowledge, not decision quality. Mitigation: report every family separately and never aggregate into a single headline score. Owned by T19.
- **Calibration bucket sparsity** — 18 buckets × small dev sets can overfit the temperature. Mitigation: cross-fitting, per-bucket counts, and a pooled-vs-macro comparison. Owned by T16.
- **Leakage through near-duplicates** — preference corpora share prompts across sources. Mitigation: group-aware splitting plus a leakage assertion that fails the build. Owned by T10.
- **Teacher weakness** — a 1.7B teacher may be no better than labels. Mitigation: the distillation gate aborts the stage when the pre-registered margin is not met. Owned by T15.
- **Colab interruption** — sessions end without warning. Mitigation: checkpoint + resume acceptance, plus a step-budget that fits one session. Owned by T12.
- **INT8 accuracy loss** — dynamic quantization can hurt the score head more than the trunk. Mitigation: quantize and measure per-head drift; refuse the export when drift exceeds the declared bound. Owned by T23.
