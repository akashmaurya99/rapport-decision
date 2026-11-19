# TASK-17 notes — Metrics implementation

Done: 2026-11-19. Status: done (single wave, one pass).

## Files shipped
- `src/evaluation/metrics.py` (new; no `__init__.py` — `src.evaluation` rides PEP 420 namespace packages, consistent with the flat `src/<module>/` layout; nothing outside the two owned files was touched).
- `tests/unit/evaluation/test_metrics.py` (new, 80 tests; no conftest owned — the test module bootstraps `sys.path` itself, mirroring `tests/unit/config/conftest.py`).

## Public surface (FR-EVAL-01)
`compute_metrics(records, *, ndcg_k=2, fpr_threshold=0.10, nll_eps=1e-15, risk_flag_threshold=0.5, categories=RISK_CATEGORIES) -> MetricResult`.

`MetricResult` (frozen dataclass, JSON-safe via `to_dict()`) carries: `accuracy`, `macro_f1`, `ndcg_at_k`, `spearman`, `pearson`, `utility_mae`, `brier`, `nll`, `ece_pooled`, `ece_bucket_macro`, `aurc`, `selective_accuracy_70/80/90`, `risk_recall_at_fpr` (dict category -> float | None), `over_refusal_rate`, plus `n_items`, `ndcg_k`, `fpr_threshold`. Every metric is `float | None`; None = undefined, **never NaN** (non-finite *input* raises `MetricInputError` instead). Documented in the module docstring.

Low-level helpers are public and independently unit-tested: `accuracy`, `macro_f1`, `ndcg_at_k`, `spearman` (average ranks, no scipy), `pearson`, `utility_mae`, `brier_score`, `nll`, `ece`, `ece_bucket_macro`, `aurc`, `selective_accuracy`, `recall_at_fpr`, `over_refusal_rate`, `keyword_risk_baseline`, `keyword_flags`.

## Conventions locked in (read the module docstring)
- Calibration pair per record: probability-family rows (`binary_label` + `binary_probability`) use `(binary_probability, binary_label)`; everything else uses `(confidence, 1{pred == gold})` where confidence defaults to `max(probabilities)`. One pair feeds Brier/NLL/ECE together — consistent with AURC and selective accuracy, which also sort by confidence.
- Brier = mean((p−y)²). NLL = binary log-loss, p clamped to [eps, 1−eps], eps default 1e-15. ECE fixed edges `[i/15, (i+1)/15)`, last bucket closed at 1.0, exact edges land in the upper bucket; pooled = Σ(n_b/N)|acc−conf|; bucket-macro = unweighted mean over non-empty buckets.
- AURC = mean of per-prefix risks (step-curve area, width 1/N), sorted by confidence descending, stable ties. Selective accuracy k = max(1, floor(c·N)) at c ∈ {0.70, 0.80, 0.90}.
- NDCG: `2^rel−1` gains on gold utilities in [0,1]; records with IDCG = 0 are excluded from the mean; k capped at option count.
- Spearman/Pearson/utility MAE are computed over the **flat pool** of (gold utility, predicted score) pairs across all options of all records (not per-item). Constant vector or < 2 pairs → None.
- `recall_at_fpr(pos, neg, fpr_threshold=0.10)`: largest recall with FPR ≤ budget (at most floor(α·n_neg) negatives flagged; item flagged ⇔ score ≥ τ). No positives → None; no negatives → 1.0 (vacuous FPR); the *achievable* thresholds are the unique observed scores ∪ {+inf}.
- Over-refusal rate = fraction of risk records whose gold risk vector is all-zero at which ≥ 1 category is flagged at `risk_flag_threshold` (default 0.5); None when no safe record exists. Per-record FPR budgets / fitted thresholds from TASK-16 are deliberately NOT consumed here — this module is standalone.
- Keyword-rule baseline (`KEYWORD_RULES`, `keyword_risk_baseline`): 8 deterministic word-boundary, case-insensitive keyword groups matching the 8 declared risk-head labels (violence, self_harm, sexual_content, hate_speech, harassment, dangerous_content, illegal_activity, privacy_violation). Documented as a comparator/floor, not a model; no tuning possible — recall and FPR are reported at the rule's single operating point; `over_refusal_rate` on safe (all-zero) texts.
- Validation is fail-fast (all records validated before any metric is computed) and raises typed `MetricInputError` (code `METRICS_INPUT_INVALID`). Min options = 2 (mirrors RAPPORT 2..8). Probabilities must be in [0,1] and sum to 1 ± 1e-2. Indices must be integers (bools rejected; numpy-style ints accepted via `operator.index`).

## Fixtures (all hand-computed; see comments in test_metrics.py)
- 6-item 2-option MAIN fixture: accuracy 4/6, macro-F1 2/3, Brier 1.2361/6, NLL ≈ 0.5854, pooled ECE ≈ 0.3983 vs bucket-macro 0.444 (b12 has 2 items — the two ECEs genuinely differ), AURC ≈ 0.1639, selective @70/80/90 = 0.75/0.75/0.6.
- 10-item confidence-ordered fixture for selective accuracy (5/7, 3/4, 2/3) and AURC ≈ 0.2286.
- Risk fixture: 7 records; default α=0.10 makes the FPR budget binding on violence (recall 0.0), α=0.5 → 1.0 (threshold threaded); harassment has no positives → None; over-refusal = 1/2.
- Keyword fixture: 6 texts; per-category recall, violence FPR 0.2, over-refusal 1/3; `a skillful cook` does NOT trip `kill` (word boundaries), `KILL IT NOW` does (case-insensitive).
- All-one-class edge case: accuracy 1.0, macro-F1 1.0, ECE = 0, Brier 0, NLL < 1e-12 (confidence 1.0). Degenerate 2-option case exercised throughout; `ndcg_at_k` with k > option count uses all options; 1-option records are rejected.
- Binary probability-family view (probability family): Brier 0.07, NLL ≈ 0.2798, ECE pooled = bucket-macro ≈ 0.2333.
- Empty record stream → all-None result, n_items = 0. Constant gold pool → Spearman/Pearson = None.

## Verification
- `.venv/bin/python -m pytest tests/unit/evaluation/test_metrics.py -q` → 80 passed.
- Full suite: 201 passed, 1 skipped (pre-existing skip, not ours).
- `ruff check .` → exit 0; `mypy src` → exit 0. No dependency added (stdlib + torch-free; numpy not even needed).
- Contract honoured: only `src/evaluation/metrics.py` and `tests/unit/evaluation/test_metrics.py` created/edited; `src/evaluation/__init__.py` deliberately NOT created (outside ownership; PEP 420 namespace package works); `tasks.yaml` not hand-edited; no git.

## Honest caveats / design notes for downstream tasks (TASK-18/19/21)
- Spearman/Pearson semantics (flat pool) is a documented choice; a per-item-averaged variant is NOT computed — if the report wants per-item ranking fidelity it already has NDCG/AURC.
- `gold_scores` are the utility target for scoring family (normalized [0,1] at ingest); NDCG gains assume that normalization.
- `risk_recall_at_fpr` is `None` per category (not 0.0) when a category has no positive rows — distinguishable in JSON. Empty categories in the report should render `not-measured`.
- Over-refusal here uses the 0.5 default flag; TASK-16's fitted per-category thresholds are not consumed (metrics module is standalone by design) — TASK-21 may want to pass `risk_flag_threshold` explicitly if it wants fitted thresholds applied.
- Selective coverages are fixed to {0.70, 0.80, 0.90} per the FR-EVAL-01 spec (MetricResult field names are fixed); `ndcg_k` and `fpr_threshold` are the only configurable knobs.
