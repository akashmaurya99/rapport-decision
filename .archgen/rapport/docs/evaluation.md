# RAPPORT — Evaluation and Benchmark Plan

_Document 6 of the research package. Every criterion here is pre-registered: the report command, not the analyst, decides pass or fail._

## 1. Rules

1. **No headline number.** Results are always reported per family, per split, per seed, as mean ± std. An aggregate is printed only as a secondary table and is never used for a verdict.
2. **Three seeds** {13, 29, 47} for every arm. A claim requires the mean gap to exceed one standard deviation of the per-seed results.
3. **Test is read once**, by T21's report command. Dev is used for early stopping, calibration fitting and all selection. Any peeking at test is a protocol violation and the run is void.
4. **Every arm is reported**, including the arms that hurt. A report with only winners is not a report.
5. **Baselines run from the same split manifest** as the model; they never see data the model does not.

## 2. Baselines (the ladder, weakest to strongest)

| # | Baseline | What it establishes |
|---|---|---|
| B1 | Majority class per family | Trivial floor; any model below this is broken, not merely weak |
| B2 | Random within family | Sanity floor for tie-free families (e.g. risk is multi-label) |
| B3 | TF-IDF (char + word n-grams) + logistic regression | The classical floor that a 33M model **must** beat (arXiv 2611.21393 showed a fine-tuned small decision model losing to this on 3 of 5 fields under a pre-registered design) |
| B4 | Frozen `bge-small-en-v1.5` embeddings + cosine choice | The "System 1, no training" analogue; costs nothing and is a strong odd case |
| B5 | Frozen trunk + trained heads (A5) | Measures how much of the gain is the head vs the trunk |
| B6 | RAPPORT-U / RAPPORT-C / RAPPORT-L | The three registered variants |
| B7 | RAPPORT-C + distillation | Stage C effect, only if the T15 gate passes |
| B8 | Teacher (`Qwen3-1.7B` 4-bit) as reference | Upper-bound context on a 1,000-example subsample only — this is context, not a target |

Acceptance requires beating **B3** and **B4**. B8 is reported, never targeted.

## 3. Metrics, per family

| Family | Metrics | Why these |
|---|---|---|
| choice | accuracy, macro-F1 | Macro-F1 exposes the class imbalance hidden by accuracy |
| preference | pairwise accuracy, NDCG@2, Spearman | Pairwise BCE accuracy and ranking fidelity |
| scoring | Spearman, Pearson, Huber loss, utility MAE | Ordinal and absolute quality are different claims |
| probability | Brier, NLL, ECE (pooled + bucket-macro) | Proper scoring rules; ECE alone hides per-bucket failure |
| abstention | AURC, selective accuracy @ {70, 80, 90}% coverage, abstention precision/recall | The documented weak behaviour; AURC is the summary |
| risk | per-category recall at FPR ≤0.10, micro and macro F1, over-refusal rate on XSTest | Safety needs per-category behaviour, not an average |

Required on every split: pooled ECE **and** bucket-macro ECE, parameter count, serialized size, peak VRAM, p50/p95 latency, throughput, FLOPs.

## 4. Data isolation

- Splits are group-aware (70/15/15), defined by T10, fingerprinted in `SplitManifest`.
- One **entire source** is withheld for the OOD test: **UltraFeedback-binarized** (MIT; zero rows to train and dev). This is the blocking decision recorded in `configs/experiments/acceptance.yaml`; the preference OOD block additionally reports RewardBench trios.
- `test` in-distribution = 12,000 rows; OOD test = 8,000 rows; dev = 12,000. Test never contributes to any training, fitting or selection decision, including calibration.
- RewardBench trios are evaluation-only and are used for the preference OOD check.

## 5. Ablations (T21)

| Arm | Question | Pass condition for the default to stand |
|---|---|---|
| A1 init ∈ {bge-small, random, MiniLM-L6} | does pretraining help at 55k? | bge-small ≥ random by ≥3 points; else the claim is reported as unsupported |
| A2 readout ∈ {cosine marker, joint cross-attention block, CLS-pooled MLP} | is cosine enough? | cosine within 2 points of cross-attention; else revise the spec |
| A3 depth ∈ {6/320, 12/384, 16/448} | what does capacity buy? | RAPPORT-C within 2 points of RAPPORT-L on held-out source; else promote RAPPORT-L |
| A4 weights ∈ {learned, fixed, uniform} | does uncertainty weighting collapse a task? | no task collapsed below the floor for 2 consecutive epochs (`WEIGHT_COLLAPSE`) |
| A5 head-only vs full fine-tune | how much does the trunk learn? | full fine-tune ≥ head-only by ≥3 points; else revisit initialization |
| A6 data ∈ {55k, 115.6k, 115.6k+KD, 115.6k uniform order} | payoff per example | each stage's gain over its predecessor ≥1 point or is dropped |
| A7 ordering ∈ {curriculum, uniform} | is the curriculum earning its complexity? | curriculum ≥ uniform by ≥1 point |
| A8 context ∈ {512 train / 1024 infer, 512/512} | is RoPE extension safe? | no more than 2-point drop at 1024 |

## 6. Robustness and generalization suite (T19)

| Test | Protocol | What it detects |
|---|---|---|
| Option permutation | score all n! permutations for n ≤ 4, 8 random permutations for n > 4 | position bias; scores must permute exactly (|Δscore| ≤ 1e-4) |
| Option-count sweep | accuracy at each n in 2,3,…,8 | whether confidence and calibration hold as n grows (the +0.22 over-confidence bucket in the Laya study) |
| Held-out source | full metric block on the OOD source | Memorialization vs mechanism |
| Length sweep | accuracy at 512/768/1024 tokens | whether 1024 is a safe extension, and whether truncation hurts |
| Contradictory context | state contradicts goal (e.g. requirement + opposing constraint) | whether the model uses the context or pattern-matches the goal |
| Missing information | state with the decisive fact removed | abstention rate should rise; flat accuracy means memorization |
| Adversarial phrasing | same content with negation flipped and label-preserving wording changes | surface-form reliance |
| Premise flip | correct option changed to its plausible distractor | whether the model tracks content or frequency |
| Confidence under OOD | ECE on OOD vs in-distribution | whether confidence is honestly degraded by unfamiliarity |

## 7. Benchmark protocol (T24)

| Measurement | Protocol |
|---|---|
| Warmup | 50 forward passes, discarded |
| Latency | batch 1 and batch 32; 1,000 iterations; report p50, p95, p99 |
| Throughput | decisions/s at max batch that fits 8 GB |
| Peak VRAM | `nvidia-smi` sampled at peak |
| Peak RSS (CPU) | `psutil` sampling at peak |
| Size | serialized bytes on disk: FP32, FP16, INT8 |
| FLOPs | analytic from the layer formula **and** measured with a profiler where practical; the report names which |
| Hardware | record GPU name, compute capability, driver, CUDA, torch version in the manifest |

## 8. Acceptance criteria (pass/fail, pre-registered)

| # | Criterion | Threshold | Direction |
|---|---|---|---|
| Y1 | choice accuracy, in-distribution vs B3 | ≥ +8 points | pass/fail |
| Y2 | choice accuracy, held-out source vs B4 | ≥ +5 points | pass/fail |
| Y3 | bucket-macro ECE (after Stage D) | ≤ 0.05 | pass/fail |
| Y4 | pooled ECE | ≤ 0.05 | pass/fail |
| Y5 | selective accuracy at 80% coverage vs full coverage | ≥ +5 points | pass/fail |
| Y6 | permutation invariance | |Δaccuracy| ≤ 0.5 points, |Δscore| ≤ 1e-4 | pass/fail |
| Y7 | risk recall at FPR ≤ 0.10 (Aegis held-out) | ≥ 0.85 | pass/fail |
| Y8 | over-refusal on XSTest safe prompts | ≤ 10% | pass/fail |
| Y9 | p50 batch-1 latency on T4 FP16 | ≤ 12 ms | pass/fail |
| Y10 | throughput at batch 32 | ≥ 300 decisions/s | pass/fail |
| Y11 | parameters / FP32 size | ≤ 35M / ≤ 140 MB | pass/fail |
| Y12 | INT8 export | accuracy drop ≤ 1 point unless the artifact is rejected | pass/fail |
| Y13 | resume | next-step loss within 1e-4 after simulated kill | pass/fail |
| Y14 | reproducibility | two runs of one seed agree within 0.5 accuracy points | pass/fail |

Every criterion is rendered `pass`, `fail` or `not-measured` by the report. A failed criterion is a **finding**, not a footnote: the report names which Y-number failed and the roadmap is updated.

**Statistical power.** With 12k test items, the standard error of a choice-accuracy estimate near 60% is ≈0.45%. A gap smaller than ~1 point between two arms is not distinguishable; therefore no claim is made below 1 point, and the ±8/+5 margins in Y1/Y2 are deliberately large. With 3 seeds, the standard error of a three-seed mean shrinks by √3; reported intervals are computed over seeds where applicable.

## 9. What is not evaluated

- General generation quality. RAPPORT does not generate.
- Knowledge breadth beyond the bench. MMLU is a proxy and is reported, not extrapolated from.
- Human preference for the model's outputs. No human study is in v1.
- Safety under adversarial jailbreaks. Aegis and XSTest are the safety bars; they are not a guarantee.

## 10. Reporting template

One JSON + one markdown per run id: manifest (config hash, data fingerprint, seeds, environment), the §3 block per family per split, §5 arms, §6 robustness, §7 benchmark, §8 verdicts, and a "negative results" section that cannot be empty.
