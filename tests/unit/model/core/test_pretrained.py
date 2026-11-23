"""Audited BGE pretrained-init loader (no network: synthetic BERT state dict)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.core import RapportConfig, RapportTrunk  # noqa: E402
from src.model.core.pretrained import (  # noqa: E402
    InitReport,
    PretrainedInitError,
    init_trunk_from_bge,
)


def _fake_bge_state(layers: int = 12, d: int = 384, ffn: int = 1536) -> dict:
    state = {
        "embeddings.word_embeddings.weight": torch.randn(30522, d),
        "embeddings.position_embeddings.weight": torch.randn(512, d),
        "embeddings.token_type_embeddings.weight": torch.randn(2, d),
        "embeddings.LayerNorm.weight": torch.ones(d),
        "embeddings.LayerNorm.bias": torch.zeros(d),
        "pooler.dense.weight": torch.randn(d, d),
        "pooler.dense.bias": torch.zeros(d),
    }
    for i in range(layers):
        p = f"encoder.layer.{i}"
        for name in ("query", "key", "value"):
            state[f"{p}.attention.self.{name}.weight"] = torch.randn(d, d)
            state[f"{p}.attention.self.{name}.bias"] = torch.zeros(d)
        state[f"{p}.attention.output.dense.weight"] = torch.randn(d, d)
        state[f"{p}.attention.output.dense.bias"] = torch.zeros(d)
        state[f"{p}.attention.output.LayerNorm.weight"] = torch.ones(d)
        state[f"{p}.attention.output.LayerNorm.bias"] = torch.zeros(d)
        state[f"{p}.intermediate.dense.weight"] = torch.randn(ffn, d)
        state[f"{p}.intermediate.dense.bias"] = torch.zeros(ffn)
        state[f"{p}.output.dense.weight"] = torch.randn(d, ffn)
        state[f"{p}.output.dense.bias"] = torch.zeros(d)
        state[f"{p}.output.LayerNorm.weight"] = torch.ones(d)
        state[f"{p}.output.LayerNorm.bias"] = torch.zeros(d)
    return state


def test_loader_copies_embeddings_and_attention_and_audits_skips():
    cfg = RapportConfig.compact()
    trunk = RapportTrunk(cfg)
    state = _fake_bge_state()
    report = init_trunk_from_bge(trunk, cfg, state=state)
    assert isinstance(report, InitReport)
    assert report.copied["embeddings.word_embeddings.weight"] == "embedding.weight.weight"
    assert len(report.copied) == 1 + 12 * 4
    assert "embeddings.position_embeddings.weight" in report.skipped
    assert all(k not in report.copied for k in state if k.endswith(".bias"))
    # copied tensors actually match
    sd = trunk.state_dict()
    assert torch.equal(sd["embedding.weight.weight"], state["embeddings.word_embeddings.weight"])
    assert torch.equal(sd["blocks.0.attn.q_proj.weight"], state["encoder.layer.0.attention.self.query.weight"])


def test_loader_rejects_unexpected_source_key():
    cfg = RapportConfig.compact()
    trunk = RapportTrunk(cfg)
    state = _fake_bge_state()
    state["encoder.layer.0.mystery.weight"] = torch.randn(3)
    with pytest.raises(PretrainedInitError):
        init_trunk_from_bge(trunk, cfg, state=state)


def test_loader_rejects_wrong_shape():
    cfg = RapportConfig.compact()
    trunk = RapportTrunk(cfg)
    state = _fake_bge_state()
    state["embeddings.word_embeddings.weight"] = torch.randn(30522, 320)
    with pytest.raises(PretrainedInitError):
        init_trunk_from_bge(trunk, cfg, state=state)


def test_loader_requires_compact_trunk():
    cfg = RapportConfig.ultra()
    trunk = RapportTrunk(cfg)
    with pytest.raises(PretrainedInitError):
        init_trunk_from_bge(trunk, cfg, state=_fake_bge_state(layers=6, d=320))
