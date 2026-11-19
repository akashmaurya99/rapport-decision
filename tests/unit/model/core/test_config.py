"""Tests for src/model/core/config.py — variants, spec constants, table render."""

from __future__ import annotations

import re
from pathlib import Path

from src.model.core import RapportConfig, expected_parameter_counts
from src.model.core.config import SpecConstants

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _spec_document() -> str:
    # Acceptance names docs/architecture-spec.md; the archgen slug tree carries it.
    for candidate in (
        _REPO_ROOT / "docs" / "architecture-spec.md",
        _REPO_ROOT / ".archgen" / "rapport" / "docs" / "architecture-spec.md",
    ):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise AssertionError("architecture-spec.md not found in either expected location")


def _spec_layer_rows(document: str) -> list[tuple[str, str]]:
    section = document.split("## 4.", 1)[1].split("## 5.", 1)[0]
    rows = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0].lower() == "stage":
            continue
        rows.append((cells[0], cells[-1]))
    return rows


def test_variants_instantiate_with_expected_shapes():
    ultra, compact, large = RapportConfig.ultra(), RapportConfig.compact(), RapportConfig.large()
    assert (ultra.layers, ultra.d_model, ultra.heads, ultra.ffn) == (6, 320, 5, 896)
    assert (compact.layers, compact.d_model, compact.heads, compact.ffn) == (12, 384, 6, 1024)
    assert (large.layers, large.d_model, large.heads, large.ffn) == (16, 448, 7, 1192)
    assert (ultra.embedding_dim, compact.embedding_dim, large.embedding_dim) == (128, 384, 448)
    assert compact.head_dim == ultra.head_dim == large.head_dim == 64


def test_rope_base_is_10000_for_all_variants():
    for variant in (RapportConfig.ultra(), RapportConfig.compact(), RapportConfig.large()):
        assert variant.rope_base == 10_000.0
        assert variant.max_seq_len == 512


def test_compact_spec_constants_match_the_spec_document_values():
    spec: SpecConstants = RapportConfig.compact().spec
    assert spec.trunk == 21_242_880  # 12 × 1,770,240 — spec §3
    assert spec.attention_total == 7_077_888
    assert spec.swiglu_total == 14_155_776
    assert spec.norms_total == 9_216
    assert spec.embedding == 11_720_448
    assert spec.total == 33_460_698  # spec §4 headline total


def test_formula_reproduces_spec_literals_for_every_variant():
    for variant in (RapportConfig.ultra(), RapportConfig.compact(), RapportConfig.large()):
        expected = expected_parameter_counts(variant)
        assert expected["attention"] == variant.spec.attention_total
        assert expected["swiglu"] == variant.spec.swiglu_total
        assert expected["norms"] == variant.spec.norms_total
        assert expected["trunk"] == variant.spec.trunk
        assert expected["embedding"] == variant.spec.embedding


def test_render_layer_table_matches_spec_document():
    document = _spec_document()
    rows = _spec_layer_rows(document)
    assert rows, "no §4 table rows parsed from the spec document"
    rendered = RapportConfig.compact().render_layer_table()
    for stage, params in rows:
        needle = re.escape(stage)
        line = next((ln for ln in rendered.splitlines() if re.search(needle, ln)), None)
        assert line is not None, f"stage {stage!r} missing from rendered table"
        assert params in line, f"params cell {params!r} missing from rendered row: {line}"
    assert "Total: 33,460,698 parameters" in rendered


def test_render_layer_table_header_matches_spec_document():
    rendered = RapportConfig.compact().render_layer_table()
    assert rendered.splitlines()[:2] == [
        "| Stage | Tensor in → out | Operation | Params |",
        "|---|---|---|---|",
    ]
