"""Architecture configuration for the RAPPORT encoder trunk (spec §3 and §4).

One frozen ``RapportConfig`` per variant. The spec is the source of truth for
sizes, so the spec's stated per-component parameter counts are stored verbatim
in ``SpecConstants`` (literals from ``.archgen/rapport/docs/architecture-spec.md``)
instead of being re-derived — the parameter counter checks the implementation
against these literals, and ``expected_parameter_counts`` in ``counting`` checks
the formula against them, so a transcription error in either direction fails a
test rather than shipping silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Spec §4 prints the layer table literally; rows are kept verbatim so the
# rendered table can be diffed against the document by the tests.
@dataclass(frozen=True)
class LayerRow:
    """One row of the spec §4 layer table."""

    stage: str
    tensor: str
    operation: str
    params: str


@dataclass(frozen=True)
class SpecConstants:
    """Spec-stated parameter counts, exact integers (spec §3/§4)."""

    attention_total: int
    swiglu_total: int
    norms_total: int
    trunk: int  # attention + swiglu + norms, non-embedding
    embedding: int
    heads_total: int  # decision heads (TASK-06); carried for the table only
    total: int  # trunk + embedding + heads, as the spec states it
    rows: tuple[LayerRow, ...] = field(compare=False)


@dataclass(frozen=True)
class RapportConfig:
    """Sizes and positions for one RAPPORT variant.

    ``rope_base`` is the only position mechanism the trunk admits (spec §4:
    "RoPE, base 10,000 … no learned position table"). Anything that looked
    like a learned or relative position *table* has no field here on purpose.
    """

    variant: str
    layers: int
    d_model: int
    heads: int
    head_dim: int
    ffn: int
    vocab_size: int
    embedding_dim: int  # factorized bottleneck for U; == d_model for C/L
    spec: SpecConstants
    max_seq_len: int = 512
    rope_base: float = 10_000.0
    rms_eps: float = 1e-6

    @classmethod
    def ultra(cls) -> "RapportConfig":
        rows = (
            LayerRow("Embedding", "`int32 [B, 512]` → `[B, 512, 320]`",
                     "Factorized WordPiece embedding (30522×128 → 128×320),"
                     " scale by `sqrt(320)`", "3.948M"),
            LayerRow("Task conditioning", "`[B, 320]` added at the `<task>` position",
                     "learned task vector, 6 families + 2 reserved", "2.6K"),
            LayerRow("Positions", "—",
                     "RoPE, base 10,000, applied per layer; **no** learned position table", "0"),
            LayerRow("Encoder block ×6", "`[B, 512, 320]` → same",
                     "pre-RMSNorm → MHA (5 heads, hd 64) → residual → pre-RMSNorm →"
                     " SwiGLU(320→896→320) → residual",
                     "1.270M each"),
            LayerRow("Attention", "`[B, 5, 512, 64]`",
                     "SDPA, full bidirectional, no causal mask, padding mask only", "—"),
            LayerRow("Option readout", "`[B, 512, 320]` → `[B, n, 320]`",
                     "gather hidden states at each `<opt>` marker, project by `W_o`", "102K"),
            LayerRow("Query readout", "`[B, 512, 320]` → `[B, 320]`",
                     "gather hidden state at `<goal>`, project by `W_q`", "102K"),
            LayerRow("Scores", "`[B, n, 320] × [B, 320]` → `[B, n]`",
                     "cosine similarity, divided by learned `τ_task`", "n_tasks"),
            LayerRow("Probability head", "`[B, 320]` (CLS) → `[B, 1]`",
                     "Linear 320→192, GELU, Linear 192→1, sigmoid", "62K"),
            LayerRow("Confidence head", "`[B, 323]` → `[B, 1]`",
                     "concat(CLS, top-1 score, margin, entropy) → 128 → 1, sigmoid", "41K"),
            LayerRow("Risk head", "`[B, 320]` → `[B, 8]`",
                     "Linear 320→192, GELU, Linear 192→8, sigmoid", "63K"),
            LayerRow("Task affine", "`[B, n]`",
                     "per-task scale and bias on scores before softmax", "16"),
        )
        spec = SpecConstants(
            attention_total=2_457_600,
            swiglu_total=5_160_960,
            norms_total=3_840,
            trunk=7_622_400,
            embedding=3_947_776,
            heads_total=343_000,
            total=11_913_176,
            rows=rows,
        )
        return cls("RAPPORT-U", 6, 320, 5, 64, 896, 30_522, 128, spec)

    @classmethod
    def compact(cls) -> "RapportConfig":
        rows = (
            LayerRow("Embedding", "`int32 [B, 512]` → `[B, 512, 384]`",
                     "WordPiece embedding, scale by `sqrt(384)`", "11.720M"),
            LayerRow("Task conditioning", "`[B, 384]` added at the `<task>` position",
                     "learned task vector, 6 families + 2 reserved", "3.1K"),
            LayerRow("Positions", "—",
                     "RoPE, base 10,000, applied per layer; **no** learned position table", "0"),
            LayerRow("Encoder block ×12", "`[B, 512, 384]` → same",
                     "pre-RMSNorm → MHA (6 heads, hd 64) → residual → pre-RMSNorm →"
                     " SwiGLU(384→1024→384) → residual",
                     "1.770M each"),
            LayerRow("Attention", "`[B, 6, 512, 64]`",
                     "SDPA, full bidirectional, no causal mask, padding mask only", "—"),
            LayerRow("Option readout", "`[B, 512, 384]` → `[B, n, 384]`",
                     "gather hidden states at each `<opt>` marker, project by `W_o`", "147K"),
            LayerRow("Query readout", "`[B, 512, 384]` → `[B, 384]`",
                     "gather hidden state at `<goal>`, project by `W_q`", "147K"),
            LayerRow("Scores", "`[B, n, 384] × [B, 384]` → `[B, n]`",
                     "cosine similarity, divided by learned `τ_task`", "n_tasks"),
            LayerRow("Probability head", "`[B, 384]` (CLS) → `[B, 1]`",
                     "Linear 384→192, GELU, Linear 192→1, sigmoid", "74K"),
            LayerRow("Confidence head", "`[B, 387]` → `[B, 1]`",
                     "concat(CLS, top-1 score, margin, entropy) → 128 → 1, sigmoid", "50K"),
            LayerRow("Risk head", "`[B, 384]` → `[B, 8]`",
                     "Linear 384→192, GELU, Linear 192→8, sigmoid", "75K"),
            LayerRow("Task affine", "`[B, n]`",
                     "per-task scale and bias on scores before softmax", "16"),
        )
        spec = SpecConstants(
            attention_total=7_077_888,
            swiglu_total=14_155_776,
            norms_total=9_216,
            trunk=21_242_880,
            embedding=11_720_448,
            heads_total=497_370,
            total=33_460_698,
            rows=rows,
        )
        return cls("RAPPORT-C", 12, 384, 6, 64, 1024, 30_522, 384, spec)

    @classmethod
    def large(cls) -> "RapportConfig":
        rows = (
            LayerRow("Embedding", "`int32 [B, 512]` → `[B, 512, 448]`",
                     "WordPiece embedding, scale by `sqrt(448)`", "13.674M"),
            LayerRow("Task conditioning", "`[B, 448]` added at the `<task>` position",
                     "learned task vector, 6 families + 2 reserved", "3.6K"),
            LayerRow("Positions", "—",
                     "RoPE, base 10,000, applied per layer; **no** learned position table", "0"),
            LayerRow("Encoder block ×16", "`[B, 512, 448]` → same",
                     "pre-RMSNorm → MHA (7 heads, hd 64) → residual → pre-RMSNorm →"
                     " SwiGLU(448→1192→448) → residual",
                     "2.406M each"),
            LayerRow("Attention", "`[B, 7, 512, 64]`",
                     "SDPA, full bidirectional, no causal mask, padding mask only", "—"),
            LayerRow("Option readout", "`[B, 512, 448]` → `[B, n, 448]`",
                     "gather hidden states at each `<opt>` marker, project by `W_o`", "201K"),
            LayerRow("Query readout", "`[B, 512, 448]` → `[B, 448]`",
                     "gather hidden state at `<goal>`, project by `W_q`", "201K"),
            LayerRow("Scores", "`[B, n, 448] × [B, 448]` → `[B, n]`",
                     "cosine similarity, divided by learned `τ_task`", "n_tasks"),
            LayerRow("Probability head", "`[B, 448]` (CLS) → `[B, 1]`",
                     "Linear 448→192, GELU, Linear 192→1, sigmoid", "86K"),
            LayerRow("Confidence head", "`[B, 451]` → `[B, 1]`",
                     "concat(CLS, top-1 score, margin, entropy) → 128 → 1, sigmoid", "58K"),
            LayerRow("Risk head", "`[B, 448]` → `[B, 8]`",
                     "Linear 448→192, GELU, Linear 192→8, sigmoid", "87K"),
            LayerRow("Task affine", "`[B, n]`",
                     "per-task scale and bias on scores before softmax", "16"),
        )
        spec = SpecConstants(
            attention_total=12_845_056,
            swiglu_total=25_632_768,
            norms_total=14_336,
            trunk=38_492_160,
            embedding=13_673_856,
            heads_total=666_000,
            total=52_832_016,
            rows=rows,
        )
        return cls("RAPPORT-L", 16, 448, 7, 64, 1192, 30_522, 448, spec)

    def render_layer_table(self) -> str:
        """Render the spec §4 layer table for this variant as markdown."""
        lines = [
            "| Stage | Tensor in → out | Operation | Params |",
            "|---|---|---|---|",
        ]
        lines.extend(
            f"| {row.stage} | {row.tensor} | {row.operation} | {row.params} |"
            for row in self.spec.rows
        )
        lines.append(f"Total: {self.spec.total:,} parameters")
        return "\n".join(lines)
