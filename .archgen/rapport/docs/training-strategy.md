# RAPPORT — Training Strategy

_Document 4 of the research package. Hyperparameters are starting points with a stated search range; the values below are the ones the first acceptance run uses._

## 1. Stages

| Stage | What is trained | Data | Duration (T4 `[ESTIMATE]`) | Gate to the next stage |
|---|---|---|---|---|
| A. Initialization | trunk loaded from `bge-small-en-v1.5`; heads randomly initialized | none | <1 min | every trunk tensor loads except the dropped position table; shape assertions pass |
| B. Supervised decision training | whole model, all six heads | 55k starter, then 115.6k | 25 min / 60 min per 8 epochs | dev accuracy beats TF-IDF baseline; else stop and re-scope |
| C. Distillation (optional) | student, with teacher soft labels | same 115.6k, cached teacher distribution | +35 min teacher cache, +60 min student | teacher ≥ supervised + 3 points on dev, **and** KD student ≥ supervised + 1 point; otherwise abort |
| D. Calibration | 18 per-bucket temperatures + risk thresholds (no weight updates) | dev only, 5-fold cross-fitted | <2 min | bucket-macro ECE ≤0.05 |
| E. Reinforcement learning | **not in v1** | — | — | see §7 |

Stages are separate configs and separate runs. A stage never silently inherits another stage's checkpoint: the config names the `init_from` checkpoint and the manifest records it.

## 2. Stage A — initialization

Load `BAAI/bge-small-en-v1.5` (33.4M, 12×384, MIT) into the RAPPORT-C trunk. Drop the learned position embeddings (512×384 = 0.2M) in favour of RoPE; every other tensor copies by name. Verify with a shape map that fails on any unexpected key rather than silently skipping it.

Controls (ablation A1 in T21): random init, and `all-MiniLM-L6-v2` (22.7M). The claim "pretraining helps at 55k examples" is a hypothesis, not a premise.

## 3. Stage B — supervised multi-task training

### 3.1 Optimizer and schedule

| Setting | Value | Range searched |
|---|---|---|
| Optimizer | AdamW (β₁ 0.9, β₂ 0.98, ε 1e-8) | — |
| Encoder LR | 3e-4 | 1e-4 – 5e-4 |
| Head LR | 1e-3 | 3e-4 – 2e-3 |
| Weight decay | 0.01, excluded on norms, biases and embeddings | 0.0 – 0.1 |
| Schedule | linear warmup 5% then cosine to 1e-5 | warmup 2–10% |
| Batch | 64 examples | 32 – 128 |
| Epochs | up to 8, early stop on dev loss (patience 2) | 3 – 12 |
| Grad clip | global norm 1.0 | 0.5 – 2.0 |
| Precision | FP16 autocast + `GradScaler` | FP32 control |
| Dropout | 0.1 on hidden states and attention | 0.0 – 0.2 |
| Label smoothing | 0.0 | — |

**No label smoothing.** It trades calibration for accuracy, and calibration is a first-class deliverable here. This is a deliberate deviation from the common BERT recipe.

Length bucketing, dynamic padding and a deterministic sampler are on by default. Sequence packing is a flag, enabled only if T13 measures padding waste above 25%.

### 3.2 Total loss

```
L = Σ_t  exp(-s_t)·L_t + s_t          # Kendall et al. 2018, s_t = log σ_t²
      with weight floor exp(-s_t) ≥ 0.05 and a cap ≤ 4.0
```

| Term | Formula | Default weight | Why it exists (distinct failure it fixes) |
|---|---|---|---|
| `L_choice` | listwise softmax CE over scores | 1.0 | The primary objective; variable n handled natively |
| `L_pref` | pairwise BCE on sampled pairs + listwise CE | 1.0 | Preference data often has 2 options; pairwise is denser than a 2-way listwise CE |
| `L_score` | Huber(δ=0.1) on utility | 0.5 | Graded quality, which argmax cannot express |
| `L_prob` | BCE + 0.5·Brier | 0.5 | A proper scoring rule, not a hard label |
| `L_conf` | BCE on "top-1 correct" | 0.5 | Confidence must predict correctness, not restate p_max |
| `L_abstain` | BCE on `answerable` | 1.0 | Attacks the documented weakest behaviour |
| `L_risk` | BCE with per-category `pos_weight` | 0.5 | Multi-label safety veto; class imbalance is severe |

Each term has a degenerate case that must be handled explicitly, not by `nan_to_num`: no positive pair in a batch (`L_pref = 0` and skip), a single option (`L_choice` undefined — the data contract requires ≥2), an example with no risk labels (`L_risk` masked out), and an example with `answerable` unset (`L_abstain` masked out).

The floor and cap on learned weights are the guard against the known failure of uncertainty weighting: it drives weights of hard tasks to zero. T21's A4 compares learned weights against fixed weights; if the learned version collapses a task below the floor for two consecutive epochs, the run records a `WEIGHT_COLLAPSE` event and the report must surface it.

### 3.3 Batch composition

Each batch is mixed across families by the mixture sampler, not by family. A single-family batch makes the task-weighting signal meaningless. Minimum 2 families per batch; the sampler guarantees this by construction and the trainer asserts it.

## 4. Stage C — distillation

- **Teacher:** `Qwen/Qwen3-1.7B`, 4-bit NF4, running locally on the T4 (~1.5 GB VRAM), greedy, one forward per example, temperature 2.0.
- **Signal:** the teacher's distribution over the *same* option strings, obtained by comparing the log-probabilities of the option labels at the answer position. This is a decision distribution, not a generation.
- **Why not GKD/on-policy:** GKD (arXiv 2306.13649) corrects student-generated sequences. RAPPORT does not generate sequences, so the analogue is offline distillation of the option distribution over real inputs. No synthetic inputs are created.
- **Loss:** `L_KD = T² · KL(softmax(t/T) ‖ softmax(s/T))`, weight 0.5, applied only to examples whose teacher distribution has entropy below the 95th percentile (a flat teacher distribution carries no signal and should not be distilled).
- **Cost control:** teacher outputs are cached once, keyed by `(teacher_id, template_hash, example_id)`. The cache is ~2 MB. Re-running the student costs no teacher compute. This is what makes distillation cheap and reproducible rather than a repeated API-style expense.
- **The gate (FR-DISTILL-02):** compute teacher accuracy and student accuracy on the same dev split first. If the 1.7B teacher is not ≥3 points better than the supervised student, distillation is aborted and the negative result is reported. A weak teacher cannot be allowed to "help" by accident through regularization alone; that would be an unexplained effect, and the brief forbids shipping those.

## 5. Stage D — calibration

- **Buckets:** `task_family` × option-count bucket `{2, 3–5, 6–8}` = up to 18 buckets. Empty buckets are skipped and reported.
- **Fit:** grid search over `T ∈ [0.5, 5.0]` at 0.01, minimizing NLL, 5-fold **group-aware** cross-fitting. Grid search is used instead of LBFGS because it is deterministic and trivially reproducible.
- **Guard:** a bucket with fewer than 50 dev items is not fitted; it inherits the pooled temperature and is flagged `inherited: true`. A bucket with 50–200 items uses Platt scaling instead of a single temperature.
- **Reporting:** pooled ECE **and** bucket-macro ECE, always together. The 2026 recalibration study (arXiv 2611.16499) shows a model with pooled ECE 0.073 and macro ECE 0.143 — one number would have hidden a +0.22 error in the 12-option bucket. RAPPORT has the same structure (variable option counts), so the same trap applies.
- **Risk thresholds:** per-category thresholds chosen on dev to hit false-positive rate ≤0.10 on the dev safe rows drawn from Aegis safe-labeled prompts, then frozen. XSTest safe prompts are never used for fitting; they remain the Y8 measurement set, read only by the report.

## 6. Stage E — abstention policy (not a trained stage)

```
answer  iff  p̂_max ≥ θ and max_k r_k < θ_k
θ = (u_abstain - u_wrong) / (u_correct - u_wrong)
```

Default utilities `(u_correct, u_wrong, u_abstain) = (1, -3, 0)` give `θ = 0.75`. The utilities are a *deployment* parameter; the report shows the risk–coverage curve so a caller can choose a different operating point without retraining. The learned `answerable` head is an additional feature, not the decision rule — this keeps the threshold auditable.

## 7. Reinforcement learning — postponed, with reasons

RL is not in v1 because none of its preconditions hold:

- **No environment or transition model.** The decision families are single-step; there is no state transition to learn.
- **No reward beyond the label.** The label already *is* the reward, so RL would be supervised learning with extra variance.
- **No logged propensities.** Offline RL and contextual bandits require the logging policy's action probabilities to correct for distribution shift. They do not exist for these corpora.
- **Scale.** 115.6k labelled examples at 33M parameters is a supervised regime; DPO/KTO-style objectives need preference pairs over *model outputs*, which RAPPORT does not produce.

If a deployment later logs `(state, options, chosen_action, outcome)` with propensities, a contextual bandit over the same encoder is the right first experiment. That is a v2 decision, recorded here so it is not reinvented.

## 8. Checkpoints, resume and Colab survival

- **Cadence:** every 500 steps, at every epoch end, and on `SIGINT`. Keep the last 3 plus the best-dev checkpoint.
- **Contents:** model, optimizer, scaler, scheduler, step, epoch, sampler offset, and the RNG states of `random`, `numpy`, `torch` and CUDA.
- **Atomicity:** write to `*.tmp`, fsync, rename. A partial checkpoint is never loadable.
- **Resume proof (FR-TRAIN-02):** kill the process after step 500, resume, and assert the loss on the next fixed batch matches the uninterrupted run within 1e-4. This is a unit test, not a manual check.
- **Session budget:** 115.6k examples at batch 64 is 1,806 steps per epoch; 8 epochs is 14,450 steps. At **200–300 ms/step** that is **48–72 minutes**, leaving room for evaluation inside one free session. The step budget is a config value, so a session that dies at minute 50 loses at most 500 steps.

## 9. Experimental controls

| Control | Why |
|---|---|
| Fixed seed set {13, 29, 47} for every reported arm | A single run cannot distinguish a 1-point effect from seed noise |
| Dev used for early stopping and calibration, test touched once | Prevents the report from being a test-set tuning artefact |
| Test evaluated only by T21's report command | One code path, one place where test is read |
| Every arm recorded, including failed ones | A report that omits negative arms is a marketing document |
| Data fingerprint in every manifest | A metric without a fingerprint is not reproducible |

## 10. Unresolved questions

- **Optimal LR for a 33M encoder initialized from a sentence encoder** is unknown; sentence encoders are often trained with much smaller LRs. The 1e-4 – 5e-4 range is a guess and T14's first run must record the LR sweep.
- **Whether uncertainty weighting helps or collapses** at seven terms is unknown; the floor/cap plus A4 is the guard.
- **Whether 8 epochs overfits 115.6k examples** is unknown; early stopping decides, and the epoch at which dev peaks is reported.
