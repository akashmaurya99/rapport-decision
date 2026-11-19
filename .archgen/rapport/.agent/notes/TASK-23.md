# TASK-23 — ONNX export and INT8 dynamic quantization

## What was built

`src/inference/export.py` — the artifact-producing boundary (FR-INFER-02, E9).
`tests/unit/inference/test_export.py` — 45 tests, CPU-only, ~3s.

All 45 tests ran (0 skipped): `onnx` 1.23.2 and `onnxruntime` 1.32.0 ARE installed.
None of the tests skipped; the `onnx_required` marker carries reason
`onnx-not-installed` for environments without the libs. Not pip-installed anything.

Checks: `pytest tests/unit/inference/test_export.py` → 45 passed.
`ruff check` both files → clean. `mypy src/inference/export.py` → clean.
Full `tests/unit/inference` → 85 passed (predictor untouched).

## Real-model verification (not unit tests; one-off shell runs on RAPPORT-C, `torch.manual_seed(0)`)

- FP32 ONNX: 134,245,385 B (≤140 MB PRD bound ✓), opset 17, exported+verified in ~3.2 s.
- **Choice-logits |Δ| vs PyTorch on the fixed 64-row reference batch: 7.36e-08 (tolerance 1e-3)** ✓
  Dynamic-axes smoke check (rows=3, width=4): 3.26e-08 ✓. `sample_count=64`, `tolerance=1e-3`,
  `opset=17`, batch seed/rows/seq_len/width/sha256 recorded in the export report.
- FP16 (`convert_fp32_to_fp16`, keep_io_types): 67,356,340 B (≤70 MB ✓), onnxruntime float16 transformer.
- INT8 dynamic (per-channel QInt8, cosine path excluded): **35,113,111 B ≤ 35*1024*1024 (36,700,160 B) ✓**
  (~1.6 MB headroom). WARNING: if "35 MB" is read as 35,000,000 B exactly, the artifact MISSES by 113,111 B — flag for whoever writes the manifest validator; the bound implemented is 35 MiB (`int8_size_bytes` is reported raw so the manifest can compare either way).
- Cosine path stayed FP32: nodes `/readout/query/MatMul`, `/readout/option/MatMul` excluded from
  quantization; in the INT8 graph both are still op `MatMul` with FLOAT initializers ✓ (per-head rule).
- Per-head drift (max-abs, fp32 vs int8, reference batch): choice_logits 7.9e-04, scores 2.0e-04,
  probability 1.1e-04, confidence 4.8e-03, answerable 1.1e-04, risk 3.4e-02, score_vector 7.9e-04.
- p95 batch=1 seq_len=512 (10 iters, 2 warmup): FP32 93.8 ms, INT8 49.7 ms → **1.88× speedup, below the
  2× bar** → on those measured numbers the gate would refuse INT8 and retain FP16. Honest negative finding;
  the gate working as designed. Tiny/short runs are noisy; the gate is defined on declared report latencies,
  not on wall-clock, so unit tests never flake on this.
- Accuracy-drop ≤1 point and ECE-rise ≤0.02 legs of the gate are NOT measurable here (no dev split /
  calibration harness in this task's scope); they are enforced by `quantization_gate` on caller-supplied
  measurements and covered by unit tests on synthetic reports.

## Decisions the contract left open

- **Exporter**: the dynamo-based `torch.onnx.export` requires the optional `onnxscript` package, which is
  NOT installed (dependency budget). Legacy TorchScript tracing (`dynamo=False`) is used and recorded in
  the report as `exporter`. Two mitigations make the legacy path safe:
  1. the wrapper re-implements the forward with the *same submodules/weights* — `RapportModel.forward`
     is unusable directly (Python loop over rows + `nonzero`, data-dependent) — marker positions are
     computed traceably: `argmax` for the single `<goal>`; a cumsum-rank equality mask (slot j ⇔ marker
     index t where running count of `<opt>` == j+1) for the option slots; padding slots match nothing → 0,
     matching the reference implementation exactly. The wrapper was verified to emit tensor-equal outputs
     to `RapportModel` before export, and the exported graph is re-verified against PyTorch after export
     (any rewrite corruption raises `OnnxExportMismatchError`, never ships silently).
  2. `onnx.shape_inference.infer_shapes` + a custom identity-`Cast` eliminator (`_strip_identity_casts`).
     The legacy tracer emits ~171 float32→float32 casts; onnxruntime's Level-1 rule-based optimizer
     mishandles them and refuses the model ("Node input '…Cast_N_output_0' is not a graph input,
     initializer, or output of a previous node"). Removing identity casts (with transitive rewire and
     recursive If/Loop/Scan subgraph propagation) fixes loading at DEFAULT graph-optimization level.
     ORT_ENABLE_EXTENDED / DISABLE_ALL both also load fine.
- **Fixed reference batch** = `make_reference_batch(seed=20261119, rows=64, width=8, seq_len=64)`; vocab
  from `model.config.vocab_size`. Per-row option counts cycle 2..8 (prefix property holds; trailing
  `PAD_ID` with `attention_mask=0`; exactly one `<task>` at position 0). The batch never leaves the process;
  the report records seed/rows/seq_len/width/vocab_size and `sha256` over the four input tensors
  (`tensor_fingerprint`, dtype/shape/bytes) so the manifest can reproduce it. seq_len is 64, not 512, to
  keep CI fast — rows=64 is the contract constant and is recorded.
- **Signature mapping** (contract → implementation):
  - `export_onnx(model, tokenizer_spec, path, opset=17) -> dict` — `tokenizer_spec` is recorded verbatim
    (JSON-sanitized) in the report; it is informational (the export needs no vocabulary). Asserts
    `|Δchoice_logits| ≤ 1e-3` (raises `OnnxExportMismatchError` carrying the measured diff), plus a
    dynamic-axes smoke check (rows=3, width=4) recorded as `dynamic_axes_check`. Report records
    `opset`, `sample_count=64`, `tolerance=1e-3`, `reference_batch{seed,rows,seq_len,width,vocab_size,sha256}`,
    `dynamic_axes`, `max_abs_logit_diff`, `max_abs_diffs`, `serialized_size_bytes`, versions.
  - `quantize_dynamic_int8(onnx_path, out_path, *, reference_batch=None) -> dict` — per-channel QInt8;
    `readout.query`/`readout.option` (W_q/W_o, matched on node-scope name, `/` or `.` separated, plus
    `w_q`/`w_o`) are excluded from quantization and stay FP32 (`cosine_score_path_fp32` is set from whether
    any were found and excluded — False ⇒ gate fails). Report carries `per_head_drift` (per output tensor
    vs the FP32 session on the reference batch), both sizes, and the batch fingerprint. Reference batch
    defaults to the seeded synthetic batch with vocab inferred from the embedding initializer; pass
    `reference_batch=` to override.
  - `quantization_gate(report) -> GateOutcome` — pure; five conditions with per-condition booleans;
    `artifact="int8"` iff all pass else `"fp16"`; `GateOutcome` is frozen, JSON-serializable
    (`to_dict`/`to_json`), carries `margins` and `limits`; `__post_init__` enforces internal consistency.
    1e-9 epsilon slack on the fraction comparisons so exactly-at-bound passes (0.80−0.79≠0.01 in binary FP).
    Accuracies/ECE must be fractions in [0,1] (percentages rejected with `QuantizationReportError`);
    `cosine_score_path_fp32` must be a real bool; latencies positive finite; missing keys raise
    `QuantizationReportError` naming them.
  - `validate_and_select(report, *, strict=False) -> ExportSelection` — returns kept artifact +
    per-condition breakdown; copies the report and records `decision{artifact,passed,conditions,margins,
    limits,reason}`; never mutates the caller's mapping. `strict=True` raises `QuantizationRejectedError`
    (carrying `.gate`) instead of returning the FP16 selection. Selection is JSON-serializable.
  - `serialized_sizes(path)` — file mode: `{variant: bytes}` classified by filename tokens (`int8`/
    `fp16|f16|float16`, else `fp32`); directory mode: all three keys, first sorted match per variant or
    `None`.
  - `benchmark_p95(session, batch, iters=1000, warmup=50) -> float` — nearest-rank
    (`ceil(0.95·iters)−1`) millisecond p95 over `session.run(None, feed)`; torch tensors in `batch` are
    converted to numpy; same call shape as TASK-24's ORT sessions (fp32/int8) for the 2× condition.
- **Manifest**: the export/quantization reports ARE the manifest fragments for FR-INFER-02 (opset,
  sample_count, tolerance, reference batch, sizes, decision). This task owns no manifest-writer wiring —
  the caller (e2e runner / TASK-25) writes the report into `artifacts/run-manifest.json`. No
  `scripts/` or `src/config/` edits were made.
- **FP16 is the retained artifact** whenever the gate fails (`artifact="fp16"`); the INT8 artifact is
  refused. `convert_fp32_to_fp16` produces `*.fp16.onnx` (weights FP16, I/O dtypes kept) for that path.
- Boundary semantics: drop ≤ 0.01 passes, rise ≤ 0.02 passes, size ≤ 35*1024*1024 B passes, speedup
  ≥ 2.0 passes (all inclusive; epsilon only for binary FP noise).
- Gate speedup compares INT8 p95 vs FP32 ONNX p95 (`fp32_p95_ms / int8_p95_ms`), per contract; risk
  drift on the tiny synthetic model is expected to be larger in relative terms — gate logic does not
  consult drift, only the five declared conditions.

## Contract coverage (verbatim acceptance criteria)

1. "ONNX export matches PyTorch logits within 1e-3 on a fixed 64-row batch and the reference batch and
   tolerance are recorded in the manifest" — `export_onnx` records 64-row sha256-fixed reference batch,
   `sample_count=64`, `tolerance=1e-3`, `opset=17` in the JSON-serializable export report; measured
   |Δlogit| = 7.4e-08 on RAPPORT-C and 3.4e-08..4.8e-08 on the tiny test fixture. Manifest wiring owned
   by the caller (see above). (FR-INFER-02)
2. "INT8 dynamic quantization checked per head and accepted only if accuracy drop ≤ 1 pt, ECE rise
   ≤ 0.02, cosine score path FP32, INT8 size ≤ 35 MB, p95 ≥ 2× FP32; else INT8 refused, FP16 retained" —
   `quantize_dynamic_int8` excludes the cosine projection (per-head drift recorded), `quantization_gate`
   enforces all five bounds (incl. exactly-at-bound acceptance), `validate_and_select` refuses INT8 and
   selects FP16 on any failure and records the decision in the report. Strict mode available. (E9)
   Measured on real RAPPORT-C: size ✓ (35,113,111 B), cosine-FP32 ✓, latency 1.88× (would refuse INT8
   at that measurement), accuracy/ECE legs need the dev split (not in scope) — all enforced and unit-tested.

## Notes for TASK-24/25

- Use `benchmark_p95(session, feed, iters, warmup)` for both FP32 and INT8 sessions; feed =
  `{k: v.numpy() for k, v in batch.items()}` (or pass the torch batch — tensors are auto-converted).
- `serialized_sizes(dir)` expects `*.onnx` with `int8`/`fp16` in the filename for non-FP32 variants.
- If the manifest wants "35 MB" as 35,000,000 B rather than 35 MiB, `int8_size_bytes` (raw) is in the
  quantization report to compare against — do not round.
- onnxruntime Level-1 optimizer mishandles identity casts in legacy exports — any NEW export path must
  run `_strip_identity_casts` (already wired into `export_onnx`) before ORT loads the model; do not bypass.
