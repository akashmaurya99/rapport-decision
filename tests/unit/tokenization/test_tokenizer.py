"""FR-TOK-01: the five special tokens exist, round-trip, and stay atomic."""

from __future__ import annotations

import pytest

from src.tokenization import (
    DEFAULT_TOKENIZER,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    RapportTokenizer,
)
from src.tokenization import errors as tok_errors

CANONICAL_SEQUENCE = (
    "<task>choice<goal>Pick the cheapest migration path."
    "<state>Three Postgres primaries across two regions."
    "<opt>Lift and shift both regions.</opt>"
    "<opt>Consolidate to one region.</opt>"
)


def test_the_five_special_tokens_are_single_ids() -> None:
    tok = RapportTokenizer()
    assert list(SPECIAL_TOKENS) == ["<task>", "<goal>", "<state>", "<opt>", "</opt>"]
    for token in SPECIAL_TOKENS:
        ids = tok.encode(token)
        assert ids == [SPECIAL_TOKEN_IDS[token]], token
        assert len(ids) == 1
        assert tok.decode(ids) == token


def test_special_ids_are_stable_and_unique() -> None:
    assert len(set(SPECIAL_TOKEN_IDS.values())) == 5
    assert SPECIAL_TOKEN_IDS == {"<task>": 2, "<goal>": 3, "<state>": 4, "<opt>": 5, "</opt>": 6}
    fresh = RapportTokenizer()
    assert fresh.token_to_id == DEFAULT_TOKENIZER.token_to_id
    assert fresh.vocab_size == DEFAULT_TOKENIZER.vocab_size


def test_encode_decode_roundtrip_preserves_exact_marker_sequence() -> None:
    tok = DEFAULT_TOKENIZER
    ids = tok.encode(CANONICAL_SEQUENCE)
    decoded = tok.decode(ids)
    # Every marker survives as an atomic single token in the decoded text.
    assert decoded == CANONICAL_SEQUENCE


def test_decode_asserts_exact_marker_order_and_boundaries() -> None:
    tok = DEFAULT_TOKENIZER
    ids = tok.encode(CANONICAL_SEQUENCE)
    markers = [
        tok.id_to_token[i]
        for i in ids
        if i in SPECIAL_TOKEN_IDS.values()
    ]
    assert markers == ["<task>", "<goal>", "<state>", "<opt>", "</opt>", "<opt>", "</opt>"]


def test_markers_adjacent_to_words_without_whitespace_stay_atomic() -> None:
    tok = DEFAULT_TOKENIZER
    text = "<goal>Pick<state>St<opt>A</opt>"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_marker_inside_text_is_still_a_single_special_id() -> None:
    # RAPPORT special tokens are added tokens: an occurrence in source text is
    # the marker, not the five ASCII characters. This keeps marker boundaries
    # unambiguous at encode time.
    tok = DEFAULT_TOKENIZER
    ids = tok.encode("before <opt>middle</opt> after")
    assert SPECIAL_TOKEN_IDS["<opt>"] in ids
    assert SPECIAL_TOKEN_IDS["</opt>"] in ids
    assert tok.decode(ids) == "before <opt>middle</opt> after"


def test_roundtrip_plain_text_with_single_spaces() -> None:
    tok = DEFAULT_TOKENIZER
    text = "the quick brown fox jumps over 13 lazy dogs."
    assert tok.decode(tok.encode(text)) == text


def test_whitespace_runs_are_content_preserving() -> None:
    tok = DEFAULT_TOKENIZER
    assert tok.decode(tok.encode("a  b")) == "a  b"  # double space survives
    assert tok.decode(tok.encode("a\nb")) == "a b"  # newlines canonicalize to spaces


def test_unknown_character_maps_to_unk_and_decodes_visibly() -> None:
    tok = DEFAULT_TOKENIZER
    ids = tok.encode("☃")
    assert ids == [1]  # [UNK]
    assert tok.decode(ids) == "[UNK]"


def test_empty_text_encodes_to_empty() -> None:
    tok = DEFAULT_TOKENIZER
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_max_length_truncates_from_the_end() -> None:
    tok = DEFAULT_TOKENIZER
    ids = tok.encode(CANONICAL_SEQUENCE)
    assert tok.encode(CANONICAL_SEQUENCE, max_length=10) == ids[:10]


def test_encode_rejects_invalid_max_length_and_types() -> None:
    tok = DEFAULT_TOKENIZER
    with pytest.raises(tok_errors.BudgetTooSmallError):
        tok.encode("abc", max_length=0)
    with pytest.raises(TypeError):
        tok.encode(123)  # type: ignore[arg-type]


def test_decode_skips_pad_and_tolerates_unknown_ids() -> None:
    tok = DEFAULT_TOKENIZER
    assert tok.decode([0, 123456, 5]) == "[UNK]<opt>"
