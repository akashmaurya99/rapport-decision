# RAPPORT — Topology and Failure Domains

_Companion to `.archgen/rapport/architecture.yaml`. Diagrams: [`./mermaid-snippets.md`](./mermaid-snippets.md). The architecture contract, not this file, is the source of truth for paths and globs._

## What the system is

RAPPORT is a library plus a pipeline, not a service. Three entry points exist, and they are the only supported ways into the system:

1. `Predictor.predict(request)` — one forward pass, typed decision out (inference path).
2. `make` targets / `python -m scripts.e2e` — the reproducible pipeline (data → train → calibrate → report → export).
3. The run registry under `runs/` — the audit trail of every result.

## Module map and what each module must never do

| Module | Purpose | Must never do |
|---|---|---|
| config | one YAML + env overrides → validated config → run manifest | read network, mutate global state, or proceed on an unknown key |
| data | licensed sources → `DecisionExample` parquet, splits, fingerprints | silently drop a row, or split a group across train/dev/test |
| tokenization | special tokens, encoding, length bucketing | let raw text reach the model unencoded, or pad without a recorded policy |
| model | trunk, marker readout, six heads, parameter accounting | apply an ordinal embedding to option markers (breaks permutation equivariance) |
| training | losses, FP16 trainer, checkpoint/resume, task weighting | commit a gradient after a non-finite check fires, or overwrite a checkpoint |
| distill | local teacher → content-addressed soft-label cache → gated distillation | call a network API, or distill from a distribution that failed the gate |
| calibration | per-bucket cross-fitted temperatures, abstention threshold | fit a bucket with fewer than 50 items, or tune on the test split |
| evaluation | metrics, baselines, robustness, ablations, the single report | read the test split anywhere except T21's report command |
| inference | `DecisionResult` API, ONNX/INT8 export, benchmark | return a text answer, or export an artifact whose drift exceeds the declared bound |
| ops | JSONL logs, run registry, reproducibility check | hide a failure behind a successful exit code |

## Dependency direction and failure domains

Edges point from consumer to provider (`A --> B` means "A depends on B"). The graph is acyclic:

```
config (leaf)
data          -> config
tokenization  -> config, data
model         -> config, tokenization
training      -> config, data, model, ops
distill       -> config, data, model, training
calibration   -> config, model
evaluation    -> config, data, model, calibration, training, ops
inference     -> config, model, calibration, tokenization, ops
ops           -> config
```

`config` is a leaf, `inference` is a sink, and `ops` is a shared dependency of `training`, `evaluation` and `inference`. Option scores flow forward (ids → trunk → readout → heads); nothing flows backward across a failure boundary. A failure is **contained** when it cannot cross a module boundary without an error:

- **Data domain.** A bad row is rejected at T03/T08/T09 with a typed error and a counter; the pipeline stops before training. Nothing bad reaches a tensor.
- **Model domain.** Shape and parameter-count assertions (T05/T06) fail the build rather than let a mis-wired head ship. Permutation equivariance is asserted at construction and re-verified at benchmark time.
- **Training domain.** A non-finite gradient aborts the step and increments a counter; if that counter exceeds 10 in an epoch, training raises. Memory exhaustion is a crash, not a silent spill into swap.
- **Evaluation domain.** The report is a pure function of recorded runs; it cannot re-run training. A failed criterion is printed as failed.
- **Inference domain.** The exported artifact is checked against PyTorch outputs before it is usable; a failed check refuses the export.

The one deliberate cross-cutting dependency is **ops**: `training`, `evaluation` and `inference` write JSONL records and are the only modules that depend on it. Everything else is a DAG with no cycles — `config` is a leaf, `inference` is a sink.

## Boundaries that matter for reproducibility

| Boundary | Enforced by |
|---|---|
| Seed isolation | `RAPPORT_SEED` + per-module RNG spawn; recorded in the manifest |
| Data fingerprint | `SplitManifest.fingerprint` → `RunManifest.data_fingerprint` → every report row |
| Checkpoint → resume | atomic tmp+rename, step + sampler offset in the checkpoint, loss-equivalence unit test |
| Test isolation | test is read only inside `build_report`; the path is not imported elsewhere |
| License boundary | acquisition manifest marks `quarantined`; the mixture refuses quarantined sources |

For the diagrams see [`./mermaid-snippets.md`](./mermaid-snippets.md); for the executable contract see `.archgen/rapport/architecture.yaml`; for the design rationale see `./architecture-spec.md`; for what is not known see `./research.md` §8 and `./risks.md`.
