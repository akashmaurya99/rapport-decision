# Experiment configs — mapping to the pre-registered ablations

One YAML per ablation arm declared in `docs/evaluation.md` §5, plus the two
packaged run configs. Every file names `arm`, `init`, `seeds`,
`data_fingerprint` (placeholder `""` — the key exists, the value is resolved
at acquisition time from the split manifest), `variant` / `layers` /
`hidden` (model variant), `epochs`, `batch_size` and `lr`. The `variants:`
list in each arm file enumerates the conditions of that arm, with the arm
under test as the top-level defaults.

Y1–Y14 verdicts are produced by TASK-21's report command; no config here
computes verdicts.

| Arm | Question | Config | Init (default) | Variant (default) | Mixture | Seeds |
|---|---|---|---|---|---|---|
| A1 | does pretraining help at 55k? | `a1-init.yaml` | `bge-small` (alts `random`, `all-MiniLM-L6-v2`) | compact | starter 55k | {13, 29, 47} |
| A2 | is cosine enough? | `a2-readout.yaml` | `bge-small` | compact | starter 55k | {13, 29, 47} |
| A3 | what does capacity buy? | `a3-depth.yaml` | `bge-small` | compact (alts ultra 6/320, large 16/448) | starter 55k | {13, 29, 47} |
| A4 | does uncertainty weighting collapse a task? | `a4-weights.yaml` | `bge-small` | compact | starter 55k | {13, 29, 47} |
| A5 | how much does the trunk learn? | `a5-head-only.yaml` | `bge-small` | compact | starter 55k | {13, 29, 47} |
| A6 | payoff per example | `a6-data.yaml` | `bge-small` | compact | full 115.6k (alt arms 55k / +KD / uniform order) | {13} (subset permitted for A6) |
| A7 | is the curriculum earning its complexity? | `a7-ordering.yaml` | `bge-small` | compact | starter 55k | {13, 29, 47} |
| A8 | is RoPE extension safe? | `a8-context.yaml` | `bge-small` | compact | starter 55k | {13, 29, 47} |
| — | 55k starter run (Stage B) | `starter.yaml` | `bge-small` | compact | starter 55k | {13, 29, 47} |
| — | 115.6k acceptance run (Stage B) | `acceptance.yaml` | `bge-small` | compact | full 115.6k | {13, 29, 47} |

## Arm-specific knobs

- **A1** — `init` ∈ {`bge-small`, `random`, `all-MiniLM-L6-v2`} (`docs/evaluation.md` §5 `MiniLM-L6`; named per `docs/training-strategy.md` §2).
- **A2** — `readout` ∈ {`cosine`, `cross-attn`, `cls-pooled`} (cosine marker | joint cross-attention block | CLS-pooled MLP).
- **A3** — `variant` ∈ {`ultra` 6×320 (11.91M), `compact` 12×384 (33.46M), `large` 16×448 (52.83M)}. Only compact is Y11-eligible.
- **A4** — `weight_mode` ∈ {`learned` (floor 0.05, cap 4.0), `fixed`, `uniform`}; the learned arm watches for `WEIGHT_COLLAPSE`.
- **A5** — `head_only` ∈ {`false` (full fine-tune, default), `true` (frozen trunk, B5 baseline twin)}.
- **A6** — `mixture` ∈ {`starter` 55k, `full` 115.6k, `full` + `kd: true`, `full` + `ordering: uniform`}. The KD arm runs only after the T15 gate passes.
- **A7** — `ordering` ∈ {`curriculum` (epochs 1–2 easy→hard; epochs 3+ uniform, hard oversampled 1.5×), `uniform`}.
- **A8** — `context` ∈ {`512-train-1024-infer` (`max_length: 512`, `infer_max_length: 1024`), `512-512`}.

## Why ablations default to the 55k starter

The arm questions are comparisons, not absolute claims: holding data constant
at 55k keeps each arm at equal cost and inside one free session, and A6 is the
arm that pays out the 55k → 115.6k → +KD → uniform-order ladder. After the
arms are adjudicated, the winning stack is re-run at 115.6k by
`acceptance.yaml`.

## Acceptance run specifics

`acceptance.yaml` records two blocking, non-negotiable facts from
`docs/evaluation.md` §2/§4:

- `ood_source: "UltraFeedback-binarized"` — the designated source-held-out OOD
  set. Zero rows to train/dev; only the binarized subset (~200 MB) is ever
  fetched, solely to build the 8,000-row OOD evaluation. Substitution rule:
  if its cache cannot be built, the OOD set becomes the preference rows of an
  independent HH-RLHF domain cluster never sampled for training, recorded in
  the manifest (`docs/dataset-strategy.md` §6).
- `y2_comparator: "B3"` — TF-IDF (char + word n-grams) + logistic regression,
  `docs/evaluation.md` §2 B3. Acceptance requires beating B3 and B4.

## Fingerprint placeholder

`data_fingerprint: ""` everywhere on purpose: the fingerprint is the
split-manifest fingerprint recorded at acquisition time and cannot be known
when these files are written. The key must exist and be resolved before any
training run; the mixture sampler raises a typed error on mismatch against a
declaration that pins one (see `src/data/mixture/`).

## Config schema

These YAMLs are experiment declarations consumed by the training/eval tooling
(TASK-21), not inputs to `src/config`'s run-config schema; they intentionally
use the `lr` key name and carry arm metadata (`question`, `variants`, …) that
the flat run schema does not model. The six shared keys (`arm`, `init`,
`seeds`, `data_fingerprint`, `variant`, `epochs`, `batch_size`, `lr`) are
asserted by `tests/unit/data/mixture/test_experiments_configs.py`.
