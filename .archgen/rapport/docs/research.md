# RAPPORT — Research Review

_Document 1 of the research package. Compiled 2026-11-19._

## 0. Method and its limits

Sources were retrieved on 2026-11-19 through the arXiv API (title/abstract queries, sorted by submission date) and the Hugging Face Hub API (dataset metadata, licenses, row counts). Primary papers were read at abstract level; no result below is attributed to a paper that was not retrieved. **Limitation, stated up front:** a general web search was unavailable in this environment, so this is not an exhaustive survey of every venue. It is a bounded, reproducible query set. Where a claim is an engineering estimate rather than a published result, it is labelled `[ESTIMATE]`. Where a fact is a vendor or library compatibility statement, the primary source is named.

Two consequences follow, and they are load-bearing:

1. Any technique not retrieved here is not automatically wrong — it is simply unverified. The plan therefore contains no dependency on a single paper's number.
2. Every headline claim in the plan is re-measured locally by T21 before it is believed. The literature is used to choose *what to test*, not to assert *what is true*.

## 1. The 2026 context: "typed decision models" are now a named category

The most important development for this project is not an architecture. It is that the problem RAPPORT addresses became a recognized category in late 2026, with published evidence about where it works and where it fails.

| Finding | Source | Why it matters for RAPPORT |
|---|---|---|
| Typed decision models return a chosen option plus an option-level probability distribution in one pass. Across 8 contact-centre use cases, many matched frontier LLMs on classification at the lowest per-request cost, and their confident predictions were more accurate, enabling selective routing. | Laskar et al., *Can Decision Models Replace LLMs in Real-World Contact Center Classification and Judging Tasks?*, arXiv 2611.22949, 2026-11-17 | Confirms the product thesis: a bounded decision model competes on cost and calibration, not on generality. |
| **Their main weakness is abstention — they rarely choose the no-answer option — and they are more prone to position bias. Judging quality drops sharply for smaller models.** | Same, 2026-11-17 | Two of RAPPORT's six heads exist specifically to attack these two failure modes, and T19 measures position bias directly. |
| Head complexity does not affect accuracy: on one backbone every readout reached 81.3–82.0% in-domain, and a **0.5M cosine head matched a 27M joint schema head at half the inference time**. Reading all of a text's questions in one prompt matched one-prompt-per-question at 1.6× faster inference. **Backbone size mattered more:** 0.6B→4B gave +4 in-domain and +11 on held-out sources. Recommendation: *spend effort on the backbone, keep the head minimal*. | *Small Heads, Strong Decisions: A Controlled Study of Typed-Decision Models*, arXiv 2611.15592, 2026-11-11 | Directly selects RAPPORT's readout (cosine, minimal) and its budget allocation (trunk over heads). It also predicts the failure mode: at 33M, capacity is the constraint. |
| With 500 labels plus calibration and prompt optimization, a typed decision model closed a median 72% of the gap to 24 ACL papers' own trained methods; **fine-tuned encoders closed 59%**, and LoRA on a 27B model closed 97%. Zero-shot replacement worked on 12 of 46 test sets. | *Do We Still Need Fine-Tuned Classifiers?*, arXiv 2611.20191, 2026-11-15 | Sets the realistic expectation for a 33M fine-tuned encoder: it is the cheap strong option, not the best possible option. It also supplies the baseline ladder. |
| A **pre-registered** study fine-tuning small typed decision models on one user's email labels **failed to clear a TF-IDF logistic regression** at every label budget on three of five fields, and beat it narrowly on two. | *Train After You Measure*, arXiv 2611.21393, 2026-11-16 | The single most useful negative result in this review. It is why T18 exists and why "beat TF-IDF" is a blocking acceptance criterion rather than an afterthought. |
| Typed decision models in transportation had **lower accuracy than discrete choice models** on travel-mode choice, varied substantially with input representation and question framing, but were competitive in closed-loop control with lower latency. | *When Do Typed Decision Models Work in Transportation?*, arXiv 2611.22926, 2026-11-17 | Frames RAPPORT as distribution-sensitive: framing is a variable to hold fixed and measure, not a nuisance. |
| A 0.8B–27B multimodal decision family reports that **backbone–recipe co-adaptation outweighs raw model recency**, and that targeted data curation lets 4B–9B rival 27B on specialized workflows. | *Piccolo*, arXiv 2611.24651, 2026-11-18 | Supports the data-selection emphasis: at fixed capacity, curation is the lever. |
| A System-1/System-2 harness uses a fast decision model for action selection with an asynchronous larger model building reusable context. | *Think Once, Act Fast*, arXiv 2611.22432, 2026-11-17 | Positions RAPPORT correctly: the fast tier. It also justifies a first-class abstain/escalate output rather than pretending one model decides everything. |
| Inserting a fast non-generative decision model as a gate raised port recall from 75.5% to 89.0% and intercepted decoy traps; chunked batch dispatch recovered 92–100%. | *When Do Decision Models Help LLM-Driven Network Discovery?*, arXiv 2611.23947, 2026-11-18 | Evidence that a small gate model has real value even when it is not the primary actor. |
| Frozen encoder plus two small trained heads reached 33–46% on MedMCQA (chance 25%), described by its authors as "a cheap, fast baseline and **not** a competitive system". | *A System 1 Decision Model for Medical MCQs*, arXiv 2611.23832, 2026-11-18 | Calibrates ambition: a frozen trunk is not enough for knowledge-heavy tasks. RAPPORT fine-tunes the trunk and does not claim medical-grade knowledge. |

**Synthesis.** The 2026 evidence says the category is real, the economics are real, and the open problems are (a) abstention, (b) position bias, (c) small-model judging quality, (d) sensitivity to framing. RAPPORT's design is a direct response to (a)–(d), and its 33M budget is an explicit attempt to find the *lower* edge of the category. That edge is currently unpublished — the smallest backbone in the retrieved literature is 0.6B (Tev1-0.8B and Qwen3-0.6B appear in the fine-tuning studies). **RAPPORT's central risk is therefore known and nameable, not vague: it may be 18× below the smallest validated operating point.**

## 2. Architecture evidence

| Technique | Source (date) | Contribution | Cost | Reported effect | Fit for T4 | Decision |
|---|---|---|---|---|---|---|
| Encoder-only Transformer | BERT, Devlin et al., arXiv 1810.04805 (2018); ModernBERT, arXiv 2412.13663 (2024-12-18) | Bidirectional encoding; ModernBERT adds RoPE, GeGLU, alternating local/global attention, unpadding, trained on 2T tokens at 8192 context with SOTA classification/retrieval | 149M/395M for ModernBERT | Pareto improvement over BERT for encoders | ModernBERT-base exceeds the budget; its *components* are cheap | **Adopt components, not the model**: RoPE, pre-RMSNorm, SwiGLU, unpadded packing |
| MiniLM-style deep self-attention distillation | Wang et al., arXiv 2002.10957 (2020) | Distil the last-layer self-attention of a large teacher into a 6/12-layer student | 22.7M (L6), 33.4M (L12) | Strong quality per parameter for its era | Excellent: the exact size class | **Adopt the shape** (12×384), use modern components instead of 2020 ones |
| DistilBERT | Sanh et al., arXiv 1910.01108 (2019) | 40% fewer parameters, ~60% faster, ~97% of BERT GLUE | 66M | Established distillation recipe | Fits | Reference only; MiniLM is smaller at similar quality |
| TinyBERT | Jiao et al., arXiv 1909.10351 (2019) | Two-stage transformer distillation incl. intermediate representations | 14.5M | Large compression | Fits | Reference for Stage C feature distillation |
| Recurrent depth (TRM / HRM) | Jolicoeur-Martineau, arXiv 2510.04871 (2025-10-06); Wang et al., arXiv 2506.21734 (2025-06-26) | A 2-layer network recursing on a latent state reaches 45% ARC-AGI-1 with **7M parameters**; HRM uses two coupled recurrent modules at 27M with ~1000 training samples | 7M/27M params, but more sequential steps | Large gains on closed-form search puzzles | Latency grows with recursion depth; T4 handles it | **Reject for v1, keep as ablation A6.** The evidence is on puzzle search, not on open-vocabulary decisions, and latency is a stated NFR |
| Set-equivariant pooling (Deep Sets / Set Transformer) | Zaheer et al., arXiv 1703.06114 (2017); Lee et al., arXiv 1810.00825 (2019) | Permutation-invariant/equivariant functions over sets | O(n²) for attention pooling, O(n) for sum/mean | Exact permutation invariance | Trivial at n≤8 | **Adopt the principle** (shared markers + per-option scoring); no pooling block needed |
| Cross-encoder vs bi-encoder | Reimers & Gurevych, arXiv 1908.10084 (2019); Nogueira & Cho, arXiv 1901.04085 (2019) | Joint encoding is more accurate; separate encoding is cacheable and faster | Cross-encoder costs one pass over the joint sequence | Cross-encoders win on ranking quality | One 512-token pass is affordable | **Adopt single-pass joint encoding**, which gets cross-encoder interaction at bi-encoder pass count |
| Listwise learning to rank | Cao et al., ICML 2007 (ListNet); Burges, NIPS 2010 (LambdaRank) | Softmax over scored items; permutation-invariant listwise loss | Free | Handles variable list length | Free | **Adopt listwise softmax CE** as the choice loss |
| ColBERT late interaction | Khattab & Zaharia, arXiv 2004.12832 (2020) | Max-similarity over token embeddings | Multi-vector storage | Strong retrieval quality | Cheap at n≤8 | **Reject**: the option is a short span, so a single marker vector suffices; token-level storage is unnecessary |

**Architecture conclusion.** An encoder-only trunk is the right family, but the specific 2019 recipe is not. RAPPORT-C borrows the 12×384 shape (which is exactly `bge-small-en-v1.5`, enabling initialization), ModernBERT's RoPE/RMSNorm/SwiGLU, and a set-scoring readout. It rejects ModernBERT's 149M size, recurrent depth, and multi-vector late interaction on latency and budget grounds, each with a named ablation that could overturn the rejection.

## 3. Training evidence

| Technique | Source (date) | Reported effect | Limitation | Decision |
|---|---|---|---|---|
| Knowledge distillation (logit) | Hinton et al., arXiv 1503.02531 (2015) | Soft targets carry more information per example than hard labels | Needs a teacher; soft labels can be miscalibrated | **Adopt in Stage C, gated** |
| On-policy / GKD distillation | Agarwal et al., arXiv 2306.13649 (2023) | Student-generated sequences corrected by teacher improve over offline KD | Designed for autoregressive students | **Reject**: RAPPORT is not autoregressive; offline option-distribution distillation is the analogue |
| Multi-task uncertainty weighting | Kendall et al., arXiv 1705.07115 (2018) | Learns per-task weights from homoscedastic uncertainty | Can collapse weights toward easy tasks | **Adopt with a floor**, fixed-weight ablation A4 |
| GradNorm | Chen et al., arXiv 1711.02257 (2018) | Balances gradient magnitudes across tasks | Extra backward passes | **Reject**: 25 tasks × extra passes is wasteful on T4; ablation only |
| PCGrad | Yu et al., arXiv 2006.04120 (2020) | Projects conflicting gradients | Doubles compute per step | **Reject** |
| Mixed precision + loss scaling | Micikevicius et al., arXiv 1710.03740 (2018) | ~2× throughput, ~half activation memory | FP16 range issues without scaling | **Adopt FP16 + GradScaler**; bf16 unavailable on sm_75 |
| AdamW | Loshchilov & Hutter, arXiv 1711.05101 (2019) | Decoupled weight decay | — | **Adopt** |
| Cosine schedule with warmup | Loshchilov & Hutter, arXiv 1608.03983 (2017) | Stable convergence | — | **Adopt** |
| 8-bit optimizer states | Dettmers et al., arXiv 2110.02861 (2022) | ~4× smaller optimizer state | Extra kernel dependency | **Reject for v1**: the full-precision optimizer state is only 268 MB, so the saving is irrelevant at 33M. Keeps the dependency surface small |
| Repeated epochs on a fixed corpus | Muennighoff et al., arXiv 2305.16264 (2023) | Up to ~4 epochs is nearly as valuable as new data; returns decay after that | Value decays with repetition | **Adopt ≤8 epochs with early stopping**, and report the epoch at which dev peaks |
| Data pruning / difficult-example selection | Sorscher et al., arXiv 2206.14486 (2022); Xia et al., arXiv 2402.04333 (2024); Wettig et al., arXiv 2402.09739 (2024) | Pruning and quality scoring can match full-data training at a fraction of the data | Metrics are expensive and model-dependent | **Adopt cheap variants only**: dedup, named quality rules, difficulty stratification. Reject gradient-influence selection (LESS) for v1 — it needs per-example gradients |
| Synthetic data | *Textbooks Are All You Need*, arXiv 2306.11644 (2023) | High-quality synthetic text can substitute for scale | Quality control is the whole problem; contamination risk | **Reject in v1**, except teacher soft labels over *real* inputs. No synthetic inputs are generated |

## 4. Calibration, abstention and selective prediction

| Technique | Source (date) | Reported effect | Limitation | Decision |
|---|---|---|---|---|
| Temperature scaling | Guo et al., arXiv 1706.04599 (2017) | A single scalar fixes most over-confidence | Assumes a single calibration regime | **Adopt, but not globally** |
| Per-bucket cross-fitted temperature | *Recalibrating a Typed-Decision Model in an Afternoon*, arXiv 2611.16499 (2026-11-12) | On a released typed-decision model, **pooled ECE was 0.073 but bucket-macro ECE was 0.143**; the 12-option bucket was over-confident by +0.22 while the 3–5-option bucket was under-confident by 0.05. One global temperature made pooled ECE *worse* (0.173). A cross-fitted temperature **per bucket** brought macro-ECE to 0.073, and to 0.089 with only 10 labelled items per bucket | Needs items per bucket; held-out-set shift raises ECE to 0.099 | **Adopt as the calibration design.** Buckets = task family × option-count bucket; report pooled *and* macro; refuse to fit a bucket below 50 items |
| Platt / isotonic | Platt 1999; Zadrozny & Elkan 2002 | Flexible monotone maps | Need ≥20 items per bucket to approach temperature scaling; isotonic overfits small buckets | **Adopt Platt as a fallback** for buckets with 50–200 items |
| Brier score | Brier 1950 | Proper scoring rule; decomposes into reliability, resolution, uncertainty | Less interpretable than accuracy | **Adopt** as a training term and a reported metric |
| Selective classification / risk–coverage | Geifman & El-Yaniv, arXiv 1705.08500 (2017) | AURC and selective accuracy formalize "know when to abstain" | Requires a cost model to set the threshold | **Adopt** as the abstention evaluation and the threshold derivation |
| AbstentionBench | Kirichenko et al., arXiv 2506.09038 (2025-06-10) | 20 datasets; **abstention is unsolved, scaling helps little, and reasoning fine-tuning degrades abstention by 24% on average** | Benchmark measures generative models | **Adopt as a data source and evaluation reference**, adapted to the typed schema |
| Verbalized confidence | Tian et al., arXiv 2305.14975 (2023) | Asking a model for confidence in text can be well-calibrated | Requires generation | **Reject** (no generation); the confidence head is trained on correctness instead |

## 5. Data evidence

| Source | Retrieved facts | License | Decision |
|---|---|---|---|
| Stanford Human Preferences (SHP) | 385K collective preferences over 18 subject areas; 9.7 GB across configs | HF card declares no SPDX license field; introduced in *Understanding Dataset Difficulty with V-Usable Information* (ICML 2022) | **Excluded by default** — no declared license; the retired 30k slot is covered by HH-RLHF. Readmission only after a recorded license review, never hard-coded |
| `cais/mmlu` | 231,400 total (incl. >99k auxiliary_train) | MIT | **Admit auxiliary_train for training** — the designated training split, disjoint from test by construction; test/dev go to evaluation-only, and T10's leakage assert enforces the boundary mechanically |
| UltraFeedback | 63,967 prompts, >1M GPT-4 judgements (arXiv 2310.01377); 20.7 GB | MIT | **Withheld, evaluation-only**: the designated source-held-out OOD set (zero rows to train/dev). Never the full 20.7 GB — only the binarized subset (~200 MB) |
| HH-RLHF | 169,352 rows | MIT | **Adopt** 23k subset, preference family |
| HelpSteer2 | 21,362 rows, multi-attribute 0–4 ratings | CC-BY-4.0 | **Adopt** fully: 10k preference + 11k scoring |
| MMLU-Pro | 12,102 rows | MIT | **Evaluate only**; never trained on — it is the choice benchmark |
| ARC | 7,787 rows | CC-BY-SA-4.0 | **Conditional**: share-alike is a real constraint on derived data; keep out of the default mixture until reviewed, or use only for evaluation |
| HellaSwag | 59,950 rows | **No license field on the Hub card** | **Quarantine** until verified; do not ship in the default mixture |
| TruthfulQA | 1,634 rows | Apache-2.0 | **Adopt** for probability/abstention |
| Aegis 2.0 | 33,416 rows, 13 risk categories | CC-BY-4.0 | **Adopt** for risk |
| XSTest | 250 safe + 200 unsafe prompts (arXiv 2308.01263, NAACL 2024) | see card | **Adopt** as an over-refusal guard |
| RewardBench | prompt–chosen–rejected trios over chat/reasoning/safety (arXiv 2403.13787) | see card | **Evaluation only**; never train on it |

**Dataset conclusion.** The mixture is built from licensed, verifiable sources with declared row counts. Two apparent staples (HellaSwag, ARC) are *not* automatically admitted: one has no declared license, the other is share-alike. This is the kind of decision that a popularity-driven dataset list gets wrong.

## 6. Hardware and kernel compatibility (verified, not assumed)

| Fact | Source | Consequence |
|---|---|---|
| T4 is sm_75, 16 GB, ~65 TFLOPS FP16 tensor, no bf16 tensor-core path | NVIDIA T4 datasheet; compute capability 7.5 | FP16 + GradScaler is the precision plan. `torch.autocast(dtype=bfloat16)` is **not** the plan |
| **FlashAttention-2 supports Ampere, Ada and Hopper. "For Turing GPUs (T4, RTX 2080), see the separate flash-attention-turing repo." bf16 requires Ampere+** | `Dao-AILab/flash-attention` README, retrieved 2026-11-19 | **Do not depend on `flash-attn`.** Use `torch.nn.functional.scaled_dot_product_attention`, whose backend selection is probed at runtime by T02 and recorded in the manifest |
| At 512 tokens, attention is 4.83 of 27.1 GFLOPs (18%) for RAPPORT-C `[ESTIMATE]` | derived from the layer formula | Even a math-backend attention is not the bottleneck; the linear layers are. Kernel work has low expected value |
| `bge-small-en-v1.5` is 33.4M, 384 hidden, 512 max sequence, MIT, MTEB average 62.17 | HF model card, retrieved 2026-11-19 | Exactly RAPPORT-C's shape; the initialization loads without surgery |

## 7. Optimizations evaluated and their verdicts

| Optimization | Expected gain | Compatibility on T4 | Decision |
|---|---|---|---|
| FP16 autocast + GradScaler | ~2× throughput, ~½ activation memory | Native (sm_75 tensor cores) | **Adopt** |
| BF16 | Simpler numerics | **Not supported** for tensor-core math before sm_80 | **Reject** |
| Gradient checkpointing | ~60–70% activation memory | Works, but costs ~30% step time | **Reject as default**; batch 64 already fits. Available as a flag for the 52M variant |
| Dynamic padding / length bucketing | 20–40% fewer padded tokens `[ESTIMATE]` | Trivial | **Adopt** |
| Sequence packing (unpadding) | 10–25% step time `[ESTIMATE]` | Requires a varlen attention path; SDPA with a padding mask is simpler | **Defer** to T13 if padding waste exceeds 25% |
| 8-bit optimizer | 4× smaller optimizer state | Works | **Reject**: saves 200 MB that is not scarce |
| EMA of weights | Small stability gain | Cheap | **Ablation only** |
| ONNX export + dynamic INT8 | 2–4× CPU speedup for BERT-class models (widely reported; `[ESTIMATE]` for this model) | Works on CPU EP | **Adopt**, with a per-head drift check |
| Static INT8 / QAT | Better than dynamic at high batch | Needs a calibration set and more tooling | **Reject for v1**; revisit if dynamic INT8 drift exceeds the bound |
| `torch.compile` | 10–30% on fwd+bwd `[ESTIMATE]` | Triton supports sm_75; compile time counts against a free session | **Optional flag**, measured by T24, never required for acceptance |

## 8. What this review deliberately does not conclude

- It does **not** conclude that 33M is sufficient. The retrieved evidence points the other way: validated typed-decision backbones start at 0.6B.
- It does **not** conclude that cosine readout is sufficient for constraint-heavy reasoning. The controlled study's tasks are lighter than "choose the action that satisfies three constraints".
- It does **not** conclude that distillation helps. A 1.7B teacher may be worse than labels on a narrow bench; T15 gates it.
- It does **not** conclude that the benchmarks measure decision quality. MMLU and ARC measure knowledge; that is why no aggregate headline score is reported.

## 9. Primary sources

- arXiv 2611.22949 · 2611.15592 · 2611.20191 · 2611.21393 · 2611.22926 · 2611.23832 · 2611.24651 · 2611.23947 · 2611.22432 · 2611.16499 (all retrieved 2026-11-19)
- arXiv 2412.13663 (ModernBERT) · 2510.04871 (TRM) · 2506.21734 (HRM) · 2506.09038 (AbstentionBench) · 2403.13787 (RewardBench) · 2310.01377 (UltraFeedback) · 2308.01263 (XSTest)
- arXiv 2002.10957 (MiniLM) · 1910.01108 (DistilBERT) · 1909.10351 (TinyBERT) · 1810.04805 (BERT) · 1908.10084 (Sentence-BERT) · 1901.04085 (cross-encoder) · 1810.00825 (Set Transformer) · 1703.06114 (Deep Sets) · 2004.12832 (ColBERT)
- arXiv 1503.02531 (KD) · 2306.13649 (GKD) · 1705.07115 (uncertainty weighting) · 1711.02257 (GradNorm) · 2006.04120 (PCGrad) · 1710.03740 (mixed precision) · 1711.05101 (AdamW) · 1608.03983 (cosine) · 2110.02861 (8-bit optimizers) · 2305.16264 (data-constrained scaling)
- arXiv 1706.04599 (calibration) · 1705.08500 (selective classification) · 2305.14975 (verbalized confidence) · 2206.14486 (data pruning) · 2402.04333 (LESS) · 2402.09739 (QuRating) · 2107.06499 (dedup) · 2303.13506 (SemDeDup) · 2103.14749 (label errors)
- arXiv 2205.14135 / 2307.08691 (FlashAttention) · 2104.09864 (RoPE) · 1910.07467 (RMSNorm) · 2002.05202 (GLU variants)
- `Dao-AILab/flash-attention` README (Turing and bf16 constraints) · NVIDIA T4 datasheet · Hugging Face Hub metadata for every dataset and encoder named above
