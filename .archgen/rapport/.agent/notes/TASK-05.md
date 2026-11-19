# TASK-05 — Encoder trunk and parameter accounting

Status: done (pending orchestrator snapshot)

## What was built
- `src/model/core/config.py` — `RapportConfig` (U/C/L classmethods), `SpecConstants`, `LayerRow`; `render_layer_table()` reproduces the spec §4 markdown table (verified against `.archgen/rapport/docs/architecture-spec.md` by test). Spec literals stored verbatim; `expected_parameter_counts()` (formula, spec §3) is tested equal to them for all variants.
- `src/model/core/rope.py` — `RotaryEmbedding` + `apply_rope`; base 10,000 (from config), computed on the fly, nothing positional stored as parameter or persistent buffer.
- `src/model/core/norm.py` — `RMSNorm` (gain only, no bias).
- `src/model/core/embedding.py` — `TrunkEmbedding`; scale by sqrt(d_model); factorized 30522×128→128×d for U, plain table for C/L. Int32 ids cast to int64 at the boundary.
- `src/model/core/attention.py` — bias-free MHA, SDPA `is_causal=False` always; `attention_mask` is a padding mask only ([B,T], 1=keep), broadcast additive [B,1,1,T] with `finfo.min` (keeps fully-padded rows finite).
- `src/model/core/ffn.py` — bias-free SwiGLU (gate/up/down).
- `src/model/core/block.py` — pre-RMSNorm → attn → residual → pre-RMSNorm → SwiGLU → residual.
- `src/model/core/trunk.py` — `RapportTrunk`: ids [B,T] → [B,T,d_model]; no positional table anywhere.
- `src/model/core/counting.py` — `count_parameters(module, spec=None, tolerance=0.01)` → `ParameterReport` (per-component + trunk + total); bucket classification by name markers (embed/attn/ffn/norm); `assert_within_spec` raises `ParameterCountMismatchError` listing every component off by >1% — this is the build-failing check.

## Verification
- `pytest tests/unit/model/core -q` → 23 passed.
- `ruff check src/model/core tests/unit/model/core` → clean; `mypy src/model/core` → no issues.
- RAPPORT-C trunk counts **exactly 21,242,880** (spec 21,242,880): attention 7,077,888; swiglu 14,155,776; norms 9,216; embedding 11,720,448. U and L also exact (formula-derived literals match §3).
- Rendered C table matches the spec §4 stage/params cells row-by-row (test parses the spec file; looks in `docs/architecture-spec.md` then `.archgen/rapport/docs/architecture-spec.md`).
- Bidirectionality test (last-token mutation moves position-0 output), padding-mask ≡ unpadded truncation (atol 1e-5), none-mask ≡ all-ones mask, eval-mode determinism (`torch.equal`), no positional parameter/buffer names, RoPE inv_freq derived from base 10,000.
- Failing-build path tested: 11-layer trunk vs C spec raises `ParameterCountMismatchError`; tampered `swiglu_total` raises; tolerance boundary ±1% tested at 0.9%/2%.

## Notes for the orchestrator / downstream
- Source tree has no `src/model/__init__.py` (outside TASK-05 ownership); `src.model.core` imports as a namespace subpackage and works under the repo's pytest rootdir setup — do not add `src/model/__init__.py` without checking TASK-06, which owns `src/model/head/**` and may add one.
- `RapportConfig.spec.heads_total` / `SpecConstants.rows` are for the rendered table and display only; heads implementation is TASK-06. `count_parameters` buckets any foreign parameter under `other` (reported, not checked), so a head reaching the trunk is visible in the report without breaking trunk assertions.
- `RapportTrunk` has no final norm (spec §4 lists none); heads consume raw block outputs.
- FP16 not required at import; everything instantiates on CPU in float32.
- Rendered `Total:` line prints the spec's full total (33,460,698, incl. heads); `ParameterReport.total` for the trunk module is trunk+embedding (32,963,328) — heads are not in this module.
