"""Tests for src/model/core/trunk.py — shapes, positions and masking."""

from __future__ import annotations

import torch

from src.model.core import RapportConfig, RapportTrunk, RotaryEmbedding


def _fixed_batch(vocab: int = 30_522, batch: int = 2, seq_len: int = 512) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20261119)
    return torch.randint(0, vocab, (batch, seq_len), generator=generator, dtype=torch.int32)


def test_trunk_emits_b_512_d_model_for_fixed_batch():
    config = RapportConfig.compact()
    trunk = RapportTrunk(config).eval()
    with torch.no_grad():
        hidden = trunk(_fixed_batch(config.vocab_size))
    assert hidden.shape == (2, 512, 384)
    assert hidden.dtype == torch.float32


def test_all_variants_emit_their_d_model():
    for config in (RapportConfig.ultra(), RapportConfig.compact(), RapportConfig.large()):
        trunk = RapportTrunk(config).eval()
        with torch.no_grad():
            hidden = trunk(_fixed_batch(config.vocab_size, batch=1, seq_len=16))
        assert hidden.shape == (1, 16, config.d_model)


def test_forward_is_deterministic_in_eval_mode():
    config = RapportConfig.compact()
    trunk = RapportTrunk(config).eval()
    ids = _fixed_batch(config.vocab_size)
    with torch.no_grad():
        first = trunk(ids)
        second = trunk(ids)
    assert torch.equal(first, second)


def test_attention_is_bidirectional_not_causal():
    config = RapportConfig.compact()
    trunk = RapportTrunk(config).eval()
    ids = _fixed_batch(config.vocab_size, batch=1, seq_len=64)
    mutated = ids.clone()
    mutated[0, -1] = (mutated[0, -1] + 1) % config.vocab_size
    with torch.no_grad():
        base = trunk(ids)
        changed = trunk(mutated)
    # A causal mask would freeze every position before the last token.
    assert not torch.allclose(base[0, 0], changed[0, 0])


def test_padding_mask_matches_unpadded_truncation():
    config = RapportConfig.compact()
    trunk = RapportTrunk(config).eval()
    ids = _fixed_batch(config.vocab_size, batch=1, seq_len=128)
    valid = 100
    mask = torch.zeros(1, 128, dtype=torch.int64)
    mask[0, :valid] = 1
    with torch.no_grad():
        padded = trunk(ids, attention_mask=mask)
        truncated = trunk(ids[:, :valid])
    assert padded.shape == (1, 128, 384)
    assert torch.allclose(padded[0, :valid], truncated[0], atol=1e-5)


def test_none_mask_equals_all_ones_mask():
    config = RapportConfig.compact()
    trunk = RapportTrunk(config).eval()
    ids = _fixed_batch(config.vocab_size, batch=1, seq_len=32)
    with torch.no_grad():
        no_mask = trunk(ids)
        ones_mask = trunk(ids, attention_mask=torch.ones_like(ids))
    assert torch.allclose(no_mask, ones_mask, atol=1e-6)


def test_no_learned_position_mechanism_anywhere():
    config = RapportConfig.compact()
    trunk = RapportTrunk(config)
    banned = ("position", "pos_embed", "pos_emb", "absolute", "alibi", "relative", "rel_pos", "pe_")
    for name, _ in trunk.named_parameters():
        lowered = name.lower()
        assert not any(m in lowered for m in banned), f"positional parameter found: {name}"
    assert not any(
        marker in name.lower()
        for name, _ in trunk.named_buffers()
        for marker in banned
    )


def test_rope_is_the_only_position_mechanism_and_base_is_10000():
    config = RapportConfig.compact()
    trunk = RapportTrunk(config)
    rotaries = [m for m in trunk.modules() if isinstance(m, RotaryEmbedding)]
    assert len(rotaries) == config.layers
    assert all(rotary.base == 10_000.0 for rotary in rotaries)
    # inv_freq must derive from the base, not from learned state.
    rotary = rotaries[0]
    assert not isinstance(getattr(rotary, "inv_freq", None), torch.nn.Parameter)
    head_dim = rotary.head_dim
    expected = config.rope_base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    positions = torch.arange(8, dtype=torch.float32)
    cos, sin = rotary(8, torch.device("cpu"), torch.float32)
    freqs = torch.outer(positions, expected)
    emb = torch.cat((freqs, freqs), dim=-1)
    assert torch.allclose(cos, emb.cos(), atol=1e-6)
    assert torch.allclose(sin, emb.sin(), atol=1e-6)


def test_trunk_instantiates_on_cpu_without_fp16():
    trunk = RapportTrunk(RapportConfig.compact())
    assert next(trunk.parameters()).dtype == torch.float32


def test_int32_token_ids_are_accepted():
    config = RapportConfig.compact()
    trunk = RapportTrunk(config).eval()
    ids = _fixed_batch(config.vocab_size, batch=1, seq_len=16)
    assert ids.dtype == torch.int32
    with torch.no_grad():
        hidden = trunk(ids)
    assert hidden.shape == (1, 16, config.d_model)
