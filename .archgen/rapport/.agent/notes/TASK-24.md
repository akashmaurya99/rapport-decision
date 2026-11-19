# TASK-24 — Inference benchmarks: latency, throughput, memory and size

## What was built

`src/inference/bench.py` — `run_benchmark(model, tokenizer, batch_sizes=(1, 32), warmup=50, iters=1000, precision=("fp32", "fp16", "int8"))` → `BenchReport`.
`tests/unit/inference/test_bench.py` — 25 tests, CPU-only. Full inference suite: 110 passed. `ruff` clean, `mypy` clean.

## Acceptance mapping

- warmup=50 / iters=1000 / p50·p95·p99 for batch 1 and 32: module-scope `report` fixture runs the full-default benchmark once (~15 s on the tiny fixture); tests assert `batch == [1, 32]`, `warmup == 50`, `iters == 1000`, positive p50/p95/p99 per batch.
- batch-32 throughput: `throughput_decisions_per_s` (primary precision, batch 32; largest requested batch when 32 is not in `batch_sizes`) — ~17k decisions/s on the tiny fixture, positive, asserted.
- peak VRAM: `torch.cuda.max_memory_allocated` (MiB), peaks reset at bench start; **None on CPU** (asserted when CUDA is absent).
- peak RSS: `max(psutil.Process().memory_info().rss, resource.getrusage(RUSAGE_SELF).ru_maxrss)` in MiB (macOS ru_maxrss = bytes, Linux = KiB; psutil is present at 7.2.2).
- Serialized sizes: `serialized_sizes()` over an artifacts dir when found (`artifacts_dir=` / `RAPPORT_ARTIFACTS_DIR` / `./artifacts`); otherwise a **temporary export** (FP32 → FP16 → INT8) into a `mkdtemp` dir that is `rmtree`d in a `finally`; sizes `{fp32, fp16, int8}` keys always present (int or None). Tiny fixture: 212,134 / 143,348 / 132,974 B. If onnx is missing or the model is not a `RapportModel`, sizes degrade to None with the reason recorded in `precision_note`.
- Environment block (FR-CONFIG-02): exactly the six declared keys — `gpu`, `compute_capability`, `driver`, `cuda_version`, `torch_version`, `python_version` — each in its own try/except; never raises (unit-tested with monkeypatched CUDA failures). `driver` comes from pynvml when importable, else `nvidia-smi` (5 s timeout). `cuda_version` = `torch.version.cuda` (None on the CPU torch 2.15.0 build here).
- Flagging against an earlier run: `BenchReport.flags_against(previous)` (accepts a `BenchReport` or a bare mapping) returns one human-readable flag per changed key, e.g. `"gpu changed: None -> 'NVIDIA T4'"`; empty list when identical. `run_benchmark(..., previous_report=...)` stores them in `flags_against_previous`. Both unit-tested.

## Decisions the contract left open

- **BenchReport shape.** Top-level scalar contract fields are kept verbatim; `p50_ms`/`p95_ms`/`p99_ms` are keyed by batch size (`"1"`, `"32"`) for the *primary precision* (fp32 when requested, else the first requested precision). Per-precision, per-batch detail lives under `measurements[precision][batch]` (p50/p95/p99, throughput, iterations, warmup). Extra fields: `precision`, `serialized_sizes`, `seq_len`, `latency_method`, `quantization_gate`, `flags_against_previous`. `to_dict()`/`to_json()` via `asdict` — report is JSON-serializable (asserted).
- **p95 reuse.** `benchmark_p95` (TASK-23) is used for the reported p95 on *every* precision (it works on any ORT-like session; a `_TorchSession` adapter makes the torch model one too). p50/p99 use the same nearest-rank rule on a shared timed pass; throughput derives from that pass. Two timed passes run per (precision, batch); documented in `latency_method`. `serialized_sizes` and `make_reference_batch(seed=20261119)` are also reused.
- **FP16 on CPU** is emulated: parameters rounded to the fp16 grid, executed in fp32 (CPU Half kernels are limited — the true `.half()` path fails with `RuntimeError: value cannot be converted to type c10::Half`). Latencies are indicative only; stated in `precision_note`. On CUDA, real half kernels are used (untested here, no GPU).
- **INT8**: explicit `int8_artifact=` → `RAPPORT_INT8_ARTIFACT` → artifacts-dir `*int8*.onnx` → temp export. When ORT is available the artifact is timed; the cosine score path stays FP32 per the TASK-23 contract. When no artifact is available, INT8 metrics are `None` and `quantization_gate` is exercised as a **stand-in**: a report with only the measurable keys is fed to it; it refuses for missing choice-accuracy/bucket-macro-ECE keys (not measurable here) and the typed `QuantizationReportError` is caught and recorded as `quantization_gate.status="not_evaluated"` with the missing keys named. `gate_report=` (choice accuracy / bucket-macro ECE / cosine-path bool / int8 size / both p95s) evaluates the real five-condition gate; unit-tested with TASK-23's good-report fixture (`status="passed"`) and a missing-keys report (`status="error"`).
- **RSS** uses psutil (present) with ru_maxrss as the true-peak partner; documented fallback chain for hosts without either (0.0, never raises).
- **Latency** is `time.perf_counter` everywhere except the torch-on-CUDA path (`torch.cuda.Event` + `synchronize`); the ORT path always uses `perf_counter`. Warmup discarded.
- Temp-export path needs a `RapportModel` (the ONNX exporter requires it); other nn.Modules get sizes=None with the reason recorded.

## Real verification (shell, tiny fixture, full defaults)

batch [1, 32], warmup 50, iters 1000 → p50 {'1': 0.417 ms, '32': 1.782 ms}, p95 {'1': 0.494, '32': 2.143}, throughput ≈ 17,382 decisions/s (batch 32), RSS ≈ 313 MB, VRAM None (CPU), sizes {212134, 143348, 132974}, env {gpu None, driver None, cuda_version None, torch 2.15.0, python 3.12.12}, gate `not_evaluated` (as designed). Total bench wall-clock ≈ 14 s. Tracer warnings from the ONNX exporter are inherited from TASK-23 and harmless (artifacts verified against PyTorch).

## Checks

`pytest tests/unit/inference/test_bench.py -q` → 25 passed. `pytest tests/unit/inference -q` → 110 passed. `ruff check` both files → clean. `mypy src/inference/bench.py` → clean.

## Not done / limits

- No CUDA host available: the CUDA branches (half kernels, cuda events, VRAM, gpu/driver/capability fill) are written defensively but were exercised only on CPU (monkeypatched for the failure paths in tests).
- Gate accuracy/ECE legs remain unmeasurable in this harness; `not_evaluated` is recorded honestly, never fabricated.
- Two timed passes per (precision, batch) double the measurement cost; acceptable for an offline benchmark, documented.
