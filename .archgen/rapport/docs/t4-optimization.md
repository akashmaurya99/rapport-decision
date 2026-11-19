# RAPPORT — T4 Optimization Plan

_Document 5 of the research package. Compatibility statements are sourced; performance numbers are `[ESTIMATE]` until T24 records them._

## 1. The machine, stated precisely

| Property | Value | Consequence for RAPPORT |
|---|---|---|
| GPU | NVIDIA Tesla T4, TU104 | single GPU, no NVLink, no multi-GPU plan |
| Compute capability | 7.5 (`sm_75`) | Triton and cuBLAS work; Ampere-only kernels do not |
| VRAM | 16 GB GDDR6 (~15 GB usable under Colab) | batch 128 is the practical ceiling, batch 64 the default |
| FP16 tensor throughput | ~65 TFLOPS | the precision plan is FP16, not BF16 |
| INT8 tensor throughput | ~130 TOPS | INT8 inference is worth exporting |
| Memory bandwidth | ~320 GB/s | attention at 512 tokens is not bandwidth-bound; linear layers dominate |
| BF16 tensor cores | **absent** (requires `sm_80+`) | `torch.autocast(dtype=torch.bfloat16)` is not the plan |
| TF32 | **absent** (requires `sm_80+`) | `allow_tf32` flags are no-ops; do not rely on them |
| Host | Colab Free: ~2 vCPU, ~12.7 GB RAM, ~78 GB ephemeral disk, 12 h session ceiling | `num_workers ≤ 2`; checkpoint often; keep persistent state small |

The T4 datasheet is the vendor source for the throughput figures. **T02's probe re-verifies the runtime facts** (`torch.cuda.get_device_capability()`, `torch.cuda.get_device_properties().total_memory`, available SDPA backends, `torch.version.cuda`) and writes them into the run manifest. No number above is trusted at runtime without that probe.

## 2. Precision

- **FP16 autocast** for all linear and attention ops; **FP32** for RMSNorm, softmax, the loss, temperature parameters, and the cosine score. This is the standard mixed-precision split (Micikevicius et al., arXiv 1710.03740).
- **`GradScaler`** with initial scale 2¹⁶, growth interval 2000, backoff 0.5. Non-finite gradients skip the step and increment a counter (FR-TRAIN-04); they never corrupt weights.
- **Why not BF16:** the FlashAttention README states bf16 requires Ampere, Ada or Hopper, and T4 is Turing. Emulated bf16 is slower than FP16 on this part. This is a compatibility fact, not a preference.
- **Loss-scale instability is the main FP16 risk** at small batch. Mitigation: keep the cosine score and the softmax in FP32, and assert that the reported loss is finite before every optimizer step.

## 3. Attention backend

`torch.nn.functional.scaled_dot_product_attention` is used with an explicit padding mask. The backend is **discovered, not assumed**:

1. T02 runs `torch.backends.cuda.sdp_kernel` capability checks and records which of `flash`, `mem_efficient`, `math` are available.
2. The config accepts `attention_backend: auto | math | mem_efficient`.
3. `auto` selects the fastest available and logs the choice in the manifest.
4. The acceptance run must not require a specific backend; T24 reports latency per backend so the choice is evidence-based.

**Do not install `flash-attn`.** Its README (retrieved 2026-11-19) states FlashAttention-2 supports Ampere/Ada/Hopper and directs Turing users to a separate `flash-attention-turing` repository with a subset of features. Adding it would cost compile time, add a fragile dependency, and buy little: attention is 4.83 of 27.1 GFLOPs (18%) at 512 tokens.

## 4. Memory budget (RAPPORT-C)

| Component | Batch 32 | Batch 64 | Batch 128 |
|---|---|---|---|
| FP16 weights | 67 MB | 67 MB | 67 MB |
| FP32 master weights | 134 MB | 134 MB | 134 MB |
| AdamW moments (m, v) | 268 MB | 268 MB | 268 MB |
| Gradients (FP32) | 134 MB | 134 MB | 134 MB |
| Activations | 1.2–1.8 GB | 2.4–3.6 GB | 4.8–7.2 GB |
| CUDA context + workspace | 0.6–1.0 GB | 0.6–1.0 GB | 0.6–1.0 GB |
| **Peak (estimate)** | **2.4–3.4 GB** | **3.6–5.2 GB** | **6.5–8.8 GB** |

Batch 64 leaves ≥10 GB free on a 16 GB card. That headroom is spent on **batch size and seeds**, not on gradient checkpointing: checkpointing would cut activation memory by ~60–70% but cost ~30% step time, and memory is not the binding constraint at 33M parameters.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set to reduce fragmentation across eval/train transitions.

## 5. Throughput

| Setting | Compute per step | Effective TFLOPS `[ESTIMATE]` | Step time `[ESTIMATE]` | 1 epoch (1,806 steps) `[ESTIMATE]` |
|---|---|---|---|---|
| Batch 32 | 2.6 TFLOPs | 18–22 | 120–145 ms | 4–5 min |
| Batch 64 | 5.2 TFLOPs | 20–26 | 200–260 ms | 7–9 min |
| Batch 128 | 10.4 TFLOPs | 22–28 | 370–470 ms | 12–16 min |

At the default batch 64, 8 epochs is **48–72 minutes** plus evaluation. The 12-hour session ceiling is therefore not the constraint; **session preemption** is, which is why checkpoint cadence is 500 steps rather than per epoch.

Effective TFLOPS is deliberately conservative: the GEMMs at `d=384` are small, and 20–30% of peak is the realistic band for this shape on T4. Any figure above 40% would indicate measurement error.

## 6. Data pipeline on 2 vCPUs

| Setting | Value | Reason |
|---|---|---|
| `DataLoader.num_workers` | 2 | Colab Free has ~2 vCPUs; more workers thrash |
| `pin_memory` | True | cheap host→device speedup |
| `persistent_workers` | True | avoids respawn per epoch |
| `prefetch_factor` | 2 | enough to hide tokenization |
| Length bucketing | on | 20–40% fewer padded tokens `[ESTIMATE]` |
| Tokenized cache | precomputed int32 arrays | CPU tokenization would otherwise be the bottleneck |
| Drive I/O | write checkpoints locally, copy to Drive every 5 checkpoints | Drive mount latency is high and synchronous |

## 7. Inference optimization path

1. **Eager PyTorch, FP16, `eval()`, `torch.inference_mode()`** — the reference path.
2. **SDPA** for attention (backend auto-detected).
3. **`torch.compile`** — an optional flag, measured by T24. Triton supports `sm_75`; compile time (30–120 s `[ESTIMATE]`) counts against a free session, so it is never required for acceptance.
4. **ONNX export** — fixed input signature `(input_ids, attention_mask, option_mask)` with dynamic option count.
5. **Dynamic INT8 quantization** (ONNX Runtime, `QuantizeLinear`/`DequantizeLinear` on `MatMul`/`Gemm` weights).
6. **Thread control** — `OrtSessionOptions.intra_op_num_threads = 4` as the reference setting; the benchmark reports the setting, not just the number.

Excluded: `flash-attn` (see §3), TensorRT (build time and version pinning outweigh the gain at 33M), static QAT (needs a calibration pipeline; revisit only if dynamic INT8 drift exceeds the bound).

## 8. Quantization acceptance

INT8 dynamic quantization is accepted only if:

- choice accuracy drops by ≤1.0 point on dev,
- bucket-macro ECE increases by ≤0.02,
- the cosine score path stays in FP32 (quantizing the score projection is the most likely source of ranking damage),
- serialized size is ≤35 MB,
- p95 CPU latency improves by ≥2× over FP32 ONNX.

If any condition fails, the FP16 ONNX artifact ships instead and the INT8 result is reported as a negative finding.

## 9. What is deliberately not optimized

| Optimization | Why not |
|---|---|
| Gradient checkpointing | memory is not scarce at 33M; it costs 30% step time |
| 8-bit optimizer | saves ~200 MB out of 16 GB; adds a dependency |
| Sequence packing | attention is 18% of FLOPs and padding waste is measured first; enable only if >25% |
| Multi-GPU / DDP | one T4; DDP would add complexity with no throughput |
| Custom kernels | the whole model is small GEMMs; kernel work has low expected value |
| Mixed-precision master weights (FP16 Adam) | known instability at small batch; no memory pressure justifies it |

## 10. Colab operating procedure

1. Mount Drive, symlink `data/` and `runs/` to it, keep `HF_HOME` on local disk (Drive is slow for many small reads).
2. Run `scripts/e2e.py --config configs/experiments/acceptance.yaml --resume auto`.
3. On disconnect: re-run the same command. `--resume auto` finds the newest valid checkpoint, verifies its manifest hash, and continues.
4. Before a long run, record `nvidia-smi` and the probe output in the manifest. A silently downgraded runtime (different GPU, different CUDA) invalidates comparisons and must be visible.

## 11. Unresolved questions

- **Actual p50 latency at batch 1** on T4 FP16 — estimated 8–14 ms, unmeasured.
- **Whether `mem_efficient` SDPA is selected** on this Colab build, and how much it differs from `math` at 512 tokens — T02 and T24 answer.
- **INT8 drift per head** — unknown; the per-head check is the guard.
- **Drive copy throughput** during checkpointing — unknown; if it stalls training, checkpoints stay local and only the best is copied.
