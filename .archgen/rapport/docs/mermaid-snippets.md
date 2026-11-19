# RAPPORT — Mermaid snippets

Diagrams for the artifacts under `.archgen/rapport/`. Node counts stay at or below 15.

## System context

```mermaid
graph TB
  actor([application caller]) --> rapport[RAPPORT library: Predictor.predict]
  rapport --> local[(local model artifact / Drive)]
  rapport --> hf[Hugging Face Hub at acquisition time only]
  rapport --> ort[ONNX Runtime at CPU inference time]
```

## Pipeline (container view)

```mermaid
graph LR
  raw[raw corpora] --> acquire[T07 acquire] --> parquet[processed parquet]
  parquet --> clean[T08 dedup + quality] --> schema[T09 adapters -> DecisionExample]
  schema --> split[T10 group split] --> mix[T13 mixture sampler] --> train[T12 train]
  train --> resume[T12 checkpoint + resume]
  train --> calib[T16 calibration]
  calib --> eval[T21 ablation + report] --> infer[T22 predictor]
  infer --> export[T23 ONNX + INT8] --> bench[T24 benchmark]
```

## Model dataflow

```mermaid
graph LR
  ids[input ids 512] --> emb[embedding 384] --> block[/12x RMSNorm -> MHA -> SwiGLU/]
  block --> cls[CLS] --> heads[probability / confidence / risk]
  block --> goal[goal marker] --> q[W_q q / cos]
  block --> opts[option markers x N] --> o[W_o o_i]
  q --> cos[cosine / tau_task] --> scores[option scores + softmax]
  cos --> aff[per-task affine]
  scores --> choice[choice head: argmax + distribution]
  aff --> score[score head: calibrated utility]
```

## Integration module DAG

Condensed view — edges run consumer → provider (`A --> B` means "A depends on B"). The graph is acyclic. Edges to `config` are shown for modules that reach it directly.

```mermaid
graph LR
  tokenization --> data --> config
  model --> tokenization
  training --> model
  training --> data
  training --> ops
  distill --> training
  calibration --> model
  evaluation --> calibration
  evaluation --> training
  evaluation --> data
  evaluation --> model
  evaluation --> ops
  inference --> calibration
  inference --> model
  inference --> tokenization
  inference --> ops
  ops --> config
```
