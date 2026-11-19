# TASK-11 — Multi-task loss functions

Status: done

## What was built

- `src/training/losses.py` — pure functions on tensors, no global state, no
  file I/O. Imports `RapportOutput` from `src.model.head` for types only.
  - `compute_losses(output, option_mask, *, ...) -> LossBreakdown` — computes
    the seven terms of training-strategy.md §3.2 and a weighted total.
  - Terms: `choice` (listwise masked softmax CE), `preference` (pairwise
    BCE / Bradley–Terry on choice-logit differences over explicit
    `(example, winner_slot, loser_slot)` pairs), `score` (Huber δ=0.1 on
    sigmoid utility vs. per-option utility targets), `probability`
    (BCE + 0.5·Brier, soft target), `confidence` (BCE on "top-1 correct",
    argmax over real slots of choice_logits), `abstain` (BCE on answerable),
    `risk` (multi-label BCE with per-category pos_weight).
  - `LossBreakdown` — frozen dataclass; per-term scalars, `total`,
    `active` per-term booleans, `weights` actually applied. Inactive terms
    are exactly `0.0` and excluded from consideration by the caller.
  - `LossNotFiniteError` — typed (`code = "LOSS_NOT_FINITE"`, `component`
    attribute names the offending term or `total`). Raised on any non-finite
    component or total; never swallowed, never `nan_to_num`. TASK-12 counts
    + skips on this error.
  - `DEFAULT_WEIGHTS` — the §3.2 table: choice 1.0, pref 1.0, score 0.5,
    prob 0.5, conf 0.5, abstain 1.0, risk 0.5. Weight overrides validated
    (known keys, finite, non-negative). Kendall weighting with floor/cap is
    TASK-14's, layered on these per-term tensors.

## Interface decisions (for TASK-12)

- `option_mask` [B, n], binary, bool/int/float accepted; each row needs ≥2
  real options (data contract) else `ValueError`.
- `chosen_index` [B] int, `-1` = unset; gates `choice` and `confidence`.
  Out-of-range / padding slot → `ValueError`.
- `preference_pairs` [P, 3] int; pairs referencing padding slots are dropped;
  zero valid pairs → `L_pref = 0.0`, `active["preference"] is False`.
- `score_targets` [B, n] float + optional `score_mask` [B] bool; padding
  slots excluded via option_mask. Huber mean over valid entries.
- `probability_target` [B] float in [0,1] + optional `probability_mask`.
- `answerable_target` [B] float in {0,1} + optional `answerable_mask`
  (answerable absent → term masked out).
- `risk_targets` [B, 8] float in {0,1} + optional `risk_mask` [B] bool;
  examples with no risk labels are masked out. `risk_pos_weight` explicit
  [8], or derived per batch as neg/pos (1.0 neutral when a category has no
  positives). `pos_weight=0` for an all-positive category is the
  neg/pos convention, documented.
- Masking is uniform: per-example mask drops the example from the mean;
  all-masked term → exactly `0.0` and `active=False`; total excludes it via
  the 0.0 term. Verified by tests (abstain-all-unset and risk-all-unset
  totals equal full total minus the term).
- `score_vector` unused here (CE is on `choice_logits`, including MASK_NEG
  padding slots, which are inert — tested).
- Non-finite philosophy: `_bce_from_probs` uses a safe-`where` only to avoid
  `0 * -inf = nan` in the unused branch of a 0/1 target; a genuinely
  infinite branch stays infinite and surfaces as `LossNotFiniteError`.

## Degenerate cases (docs §3.2, handled explicitly)

- single real option → `ValueError` (L_choice undefined; contract requires ≥2)
- no positive pair / all pairs on padding → `L_pref = 0.0`
- example with no risk labels → masked; batch with none → `L_risk = 0.0`
- example with `answerable` unset → masked; batch with none → `L_abstain = 0.0`

## Verification

- `.venv/bin/python -m pytest tests/unit/training/test_losses.py -q` → **43 passed**
- Full repo suite: **680 passed, 1 skipped**
- `.venv/bin/python -m ruff check src/training/losses.py tests/unit/training/test_losses.py` → clean
- `.venv/bin/python -m mypy src/training/losses.py` → no issues
- Hand-computed fixtures (2 × 3, example 1 with one padding slot): L_choice
  1.1388152358130839, L_pref 0.2573374827504877 (invalid padding pair
  dropped), L_score 0.005 (padding target 0.99 ignored), L_prob
  0.46140472997745285, L_conf 0.9830564281864165, L_abstain
  0.2231435513142097 (masked example 1), L_risk 0.4392179969303102 (explicit
  pos_weight), total 2.563635847424871; all asserted at abs=1e-5.
- Fixture D (4 examples, all preds 0.5, uneven category frequency):
  derived pos_weight matches hand-computed 0.6065037829899521 and equals the
  explicit-pos_weight call within 1e-6.
- Non-finite surfaces: nan choice logits, inf/nan risk outputs, and
  confidently-wrong sigmoids (p=0/y=1) for probability/confidence/abstain →
  `LossNotFiniteError` with the right `component`; `code == "LOSS_NOT_FINITE"`.
- Does not exist here: Kendall weighting (TASK-14), trainer counting/skip of
  the step (TASK-12), distillation loss (TASK-15), calibration (TASK-16).

## Notes for downstream

- TASK-12: catch `LossNotFiniteError`, increment the skip counter, skip the
  optimizer step; `LossBreakdown.active` tells which terms had data. The
  total is a scalar tensor with a graph attached — `backward()` verified.
- `answerable` re-emits the probability head output (TASK-06 absorption) —
  probability and abstain terms hit the same parameters, as designed.
- `probabilities` (masked distribution) not needed here; derive from
  `choice_logits` if needed.
