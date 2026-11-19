# RAPPORT — Architecture Specification

_Document 2 of the research package. Every number here is either derived from the stated formula (and checked by T05's parameter counter) or labelled an estimate. Estimates are hypotheses to be measured, not results._

## 1. Design thesis

A decision is a **set-scoring problem**, not a generation problem. Given a state `s` and options `o_1..o_n`, the model must produce one score per option and a small number of scalar judgements. Three consequences drive the whole design:

1. **One forward pass, all options in context.** The state is encoded once with every option, so each option vector is context-aware. This is the "read all questions in one prompt" result (TDM-Controlled, arXiv 2611.15592), which matched one-prompt-per-question accuracy at 1.6× faster inference and 2.1–2.5× faster training.
2. **A minimal readout.** The same controlled study found that head complexity does not affect accuracy — a 0.5M-parameter cosine head matched a 27M-parameter joint schema head in-domain at half the latency — while backbone capacity moved accuracy by 4 points in-domain and 11 points on held-out sources. RAPPORT therefore spends its budget on the trunk and keeps the readout near-parametric-free.
3. **Permutation invariance by construction.** Published typed decision models are reported to be *more prone to position bias* than chat models (arXiv 2611.22949). RAPPORT's option markers are identical token types with no ordinal embedding, so permuting options permutes scores exactly. This is both a correctness property and a differentiator that T19 can measure.

## 2. Alternatives considered

| # | Alternative | Why it was rejected | Evidence | Falsified if |
|---|---|---|---|---|
| A1 | Decoder-only autoregressive model that emits an answer token | Latency and cost scale with output length; the task needs a typed record, not text; 25–35M is far below the useful generative regime | TDM papers report typed models match LLMs on bounded classification at lower cost (2611.22949, 2611.22926) | a 33M decoder beats RAPPORT-C on the same bench at equal latency |
| A2 | Encoder + heavy joint schema head (27M head) | No measured accuracy gain over a cosine head at equal backbone; doubles inference time | 2611.15592 (five readouts, one protocol) | A2 beats A1 readout by >2 points in T21 |
| A3 | Bi-encoder (state and options encoded separately) | Loses state–option interaction; needs `n` forward passes or a cache; weaker on constraint reasoning | cross-encoder vs bi-encoder trade-off (Reimers & Gurevych 2019, arXiv 1908.10084; Nogueira & Cho 2019, arXiv 1901.04085) | A3 matches A1 readout at lower p95 latency |
| A4 | MLP-Mixer / gated-conv alternative trunk | No pretrained 33M checkpoint to initialize from, and the representative power at this size is unproven for text; loses the initialization benefit | 2611.20191 reports fine-tuned encoders close 59% of the gap to task-specific methods with 500 labels | A4 beats RAPPORT-C with equal data and no pretraining |
| A5 | Frozen encoder + trained heads only | Cheapest, and validated for MCQ (2611.23832), but caps domain adaptation; ablation A1 in T21 keeps it as a control, not the default | 2611.23832 reports 33–46% on MedMCQA — a cheap baseline, explicitly "not a competitive system" | frozen trunk matches fine-tuned within 1 point on held-out sources |
| A6 | Recurrent-depth trunk (TRM/HRM style) | Strong evidence at 7M–27M on puzzle tasks, but those tasks are closed-form search; adds latency and a training-stability risk that the brief asks us to avoid | TRM arXiv 2510.04871 (7M params, 45% ARC-AGI-1); HRM arXiv 2506.21734 (27M params) | a 2× recurrent-depth arm beats RAPPORT-C by >3 points at equal latency |
| A7 | ModernBERT-style trunk at 149M | Out of the 25–35M budget; the upgrade (RoPE, GeGLU, alternating local/global) is borrowed selectively instead | arXiv 2412.13663 | RAPPORT-C fails the latency target while a local-attention trunk passes |

**Selected:** encoder-only trunk initialized from a 33M pretrained encoder + single-pass marker readout + cosine scoring + six micro-heads.

## 3. Variants under evaluation

| Variant | Layers | d_model | Heads (hd) | FFN (SwiGLU) | Vocab | Embeddings | Trunk | Heads | **Total** |
|---|---|---|---|---|---|---|---|---|---|
| RAPPORT-U (ultra) | 6 | 320 | 5 (64) | 896 | 30,522 | factorized 128: 3.948M | 7.622M | 0.343M | **11.91M** |
| RAPPORT-C (compact, recommended) | 12 | 384 | 6 (64) | 1024 | 30,522 | 11.720M | 21.243M | 0.497M | **33.46M** |
| RAPPORT-L (large) | 16 | 448 | 7 (64) | 1192 | 30,522 | 13.674M | 38.492M | 0.666M | **52.83M** |

Per-layer formula:

```
L_layer = 4·d²                 (Q, K, V, O projections)
        + 3·d·ffn              (SwiGLU: gate, up, down)
        + 2·d                  (two RMSNorm gains)
L_total = L·L_layer + V·d_e + d_e·d + heads
```

SwiGLU at `ffn = 8d/3` costs the same as a GELU FFN at `4d` but is consistently stronger per parameter (Shazeer 2020, arXiv 2002.05202; used by PaLM and LLaMA). At `d = 384`, `ffn = 1024` is exactly that ratio.

**Why 33M and not 12M or 53M.** The 53M variant buys an estimated +1–3 points but costs ~1.75× the compute and 60% more parameters, and its own risk is that the gain is inside seed noise. The 12M variant saves 64% of compute but sits at 6 layers, where the published evidence says backbone depth drives held-out generalization — it is the *latency* variant, not the default. The 33M point is the largest model that trains in under 15 minutes per epoch on a T4 at batch 32, which is what makes 3 seeds × 8 ablation arms affordable in one free session. The choice is falsified by T21: if RAPPORT-L beats RAPPORT-C by ≥2 points on held-out sources, the recommendation moves up.

## 4. Layer-by-layer (RAPPORT-C)

| Stage | Tensor in → out | Operation | Params |
|---|---|---|---|
| Embedding | `int32 [B, 512]` → `[B, 512, 384]` | WordPiece embedding, scale by `sqrt(384)` | 11.720M |
| Task conditioning | `[B, 384]` added at the `<task>` position | learned task vector, 6 families + 2 reserved | 3.1K |
| Positions | — | RoPE, base 10,000, applied per layer; **no** learned position table | 0 |
| Encoder block ×12 | `[B, 512, 384]` → same | pre-RMSNorm → MHA (6 heads, hd 64) → residual → pre-RMSNorm → SwiGLU(384→1024→384) → residual | 1.770M each |
| Attention | `[B, 6, 512, 64]` | SDPA, full bidirectional, no causal mask, padding mask only | — |
| Option readout | `[B, 512, 384]` → `[B, n, 384]` | gather hidden states at each `<opt>` marker, project by `W_o` | 147K |
| Query readout | `[B, 512, 384]` → `[B, 384]` | gather hidden state at `<goal>`, project by `W_q` | 147K |
| Scores | `[B, n, 384] × [B, 384]` → `[B, n]` | cosine similarity, divided by learned `τ_task` | n_tasks |
| Probability head | `[B, 384]` (CLS) → `[B, 1]` | Linear 384→192, GELU, Linear 192→1, sigmoid | 74K |
| Confidence head | `[B, 387]` → `[B, 1]` | concat(CLS, top-1 score, margin, entropy) → 128 → 1, sigmoid | 50K |
| Risk head | `[B, 384]` → `[B, 8]` | Linear 384→192, GELU, Linear 192→8, sigmoid | 75K |
| Task affine | `[B, n]` | per-task scale and bias on scores before softmax | 16 |

Total: **33,460,698** parameters. T05's counter must agree within 1%.

## 5. Input format and tokenization

```
<task>choice<goal>Pick the response that better satisfies the request.<state>…</state>
<opt>…</opt><opt>…</opt><opt>…</opt>
```

- `n` options share one `<opt>` marker. There is no `opt_0`, `opt_1`, … token, no ordinal embedding, and no separator that encodes order. This is what makes the readout exactly permutation-equivariant.
- Budget: 512 tokens. 8 options × ~45 tokens + 100 tokens of goal/state ≈ 460. Options longer than 45 tokens are truncated from the end with a recorded flag; T04 counts truncations per source and the report refuses to hide them.
- **Context extension:** trained at 512, extended to 1024 only through RoPE base scaling at inference. This is an extrapolation and is measured by T19's length sweep (accuracy at 512/768/1024). If accuracy degrades by more than 2 points at 1024, the documented maximum stays 512. There is no claim that a 512-trained model is a 1024-token model.

## 6. Decision heads and their mathematics

Let `q ∈ R^384` be the goal vector, `o_i ∈ R^384` the i-th option vector, `c ∈ R^384` the CLS state vector.

```
s_i   = cos(W_q q, W_o o_i) / τ_{task}                     # raw score
p_i   = softmax(s_1..s_n)_i                                  # choice distribution
y     = sigmoid(a_{task} · s_i + b_{task})                   # calibrated utility in [0,1]
P     = sigmoid(w_p · c + b_p)                               # probability the proposition holds
conf  = sigmoid(w_c · [c ; p_max ; p_max - p_2 ; H(p)])      # confidence in the top choice
ans   = sigmoid(w_a · c + b_a)                               # answerability
r_k   = sigmoid(w_{r,k} · c + b_{r,k}), k = 1..8              # risk categories
```

- **Choice** uses listwise softmax cross-entropy `-log p_y`. This is ListNet's top-1 formulation (Cao et al., ICML 2007) and it handles variable `n` natively.
- **Score** uses Huber on the normalized utility, so a single mislabeled extreme does not dominate.
- **Probability** uses BCE plus Brier: `BCE + λ·(P - y)²`, a proper scoring rule that penalizes confident errors quadratically.
- **Confidence** is trained against the indicator "top-1 was correct". It is therefore a *prediction of correctness*, not a re-export of `p_max`; T16 checks that it adds information over `p_max` by comparing AURC.
- **Abstention** is not a head output used directly. `ans` is a feature; the *action* comes from a cost model in T16: answer iff `p̂ > (u_abstain - u_wrong) / (u_correct - u_wrong)`. With the default (1, -3, 0) the threshold is 0.75. This separation is why abstention can be retuned per deployment without retraining.
- **Risk** is multi-label sigmoid over eight categories, used as a veto: any `r_k > θ_k` forces abstain/escalate regardless of `p_max`. XSTest guards against over-refusal.

## 7. Why not a joint cross-attention head

A cross-attention head would let each option attend to the others through a learned block, which is strictly more expressive. It is rejected as the default for three measured reasons: no accuracy gain at equal backbone in the controlled study (2611.15592), 2× inference cost, and it reintroduces an order-sensitive pathway. It remains **ablation arm A2** in T21 with a one-block cross-attention pooler. If A2 wins by >2 points, the recommendation changes and the spec is revised — that is the falsification test, stated before the experiment.

## 8. Compute, memory and latency (estimates, T24 measures them)

| Quantity | RAPPORT-U | RAPPORT-C | RAPPORT-L |
|---|---|---|---|
| Non-embedding params | 7.97M | 21.74M | 39.16M |
| Forward FLOPs at 512 (linear) | 8.16 G | 22.26 G | 40.10 G |
| Forward FLOPs at 512 (attention) | 2.01 G | 4.83 G | 7.52 G |
| **Forward total** | **10.2 G** | **27.1 G** | **47.6 G** |
| Train step, batch 32 (≈3× fwd) | 0.98 T | 2.60 T | 4.57 T |
| Est. p50 batch-1 latency, T4 FP16 | 4–7 ms | 8–14 ms | 15–25 ms |
| Est. throughput, batch 32 | ≥700/s | ≥300/s | ≥150/s |
| Weights FP32 / FP16 / INT8 | 48/24/12 MB | 134/67/34 MB | 211/106/53 MB |

Training memory for RAPPORT-C: fp16 weights 67 MB + fp32 master 134 MB + AdamW moments 268 MB + fp32 grads 134 MB ≈ 0.60 GB; activations at batch 32 × 512 ≈ 1.2–1.8 GB without checkpointing. Batch 64 fits in ~4 GB, leaving the 16 GB T4 ample headroom — which is why the plan spends the headroom on batch size and seeds rather than on gradient checkpointing.

**Utilization caveat.** At `d = 384` the GEMMs are small; 20–30% of the 65 TFLOPS FP16 peak is the realistic band on T4, not 60%. All latency numbers above are estimates until T24 records them.

## 9. Initialization

Initialize the trunk from `BAAI/bge-small-en-v1.5` (33.4M, 12 layers, 384 hidden, MIT license, MTEB average 62.17 at 384 dimensions). Rationale: identical shape to RAPPORT-C (so every trunk weight loads), a strong English text encoder at exactly the target size, and a permissive license. The learned position table is dropped in favour of RoPE (0.2M parameters discarded); every other trunk tensor loads.

Two controls in T21: random initialization, and `sentence-transformers/all-MiniLM-L6-v2` (22.7M). If initialization does not beat random by ≥3 points at 55k examples, the pretraining benefit is not established for this domain and the finding is reported as negative.

## 10. Unresolved technical questions

- **Q1.** Does a 33M encoder clear a TF-IDF logistic regression on a held-out source? The smallest published typed decision backbone is 0.6B. This is the project's central empirical risk and T21 answers it.
- **Q2.** Is cosine scoring sufficient for constraint-heavy choices ("must be under $50 and refundable")? The controlled study used lighter judgement tasks. Ablation A2 tests it.
- **Q3.** Does 512→1024 RoPE scaling hold accuracy? T19 measures; the documented maximum stays 512 unless it does.
- **Q4.** Can a 1.7B 4-bit teacher beat labels by the pre-registered margin? T15 gates it.
- **Q5.** Is 12k dev enough for 18 calibration buckets? Each bucket gets ≈650 items, but the smallest (`risk` × 8 options) may not; T16 records per-bucket counts and refuses to fit below 50 items.
