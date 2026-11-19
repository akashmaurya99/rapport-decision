"""A small, deterministic, dependency-free tokenizer for RAPPORT unit tests.

RAPPORT initializes its trunk from ``BAAI/bge-small-en-v1.5`` in training
(see ``.archgen/rapport/docs/architecture-spec.md`` §7, 30,522-token WordPiece
vocab, MIT license) — that wiring is TASK-13/TASK-15's concern and requires the
weights to be present, which they are not in the offline unit-test
environment. This module is therefore a **stand-in tokenizer**: it is
deliberately tiny, fully deterministic and offline-safe. It makes no claim
about MTEB, embedding quality, or vocabulary coverage of bge-small; it exists
so the marker machinery, the 512-token budget accounting and the length
bucketing are exercisable without a pretrained vocabulary.

Design (all of it deterministic and auditable below):

* Control tokens ``[PAD]`` (id 0) and ``[UNK]`` (id 1).
* The five RAPPORT special tokens, added as single special-id tokens:
  ``<task>`` (2), ``<goal>`` (3), ``<state>`` (4), ``<opt>`` (5),
  ``</opt>`` (6). They are recognized atomically **before** any other
  splitting, both at encode time and by the joint-sequence builder, so a
  marker can never be fragmented into ordinary pieces.
* WordPiece-ish subwords: printed-ASCII single characters as bare pieces and
  as ``##`` continuations, a small fixed set of multi-character digraph /
  suffix pieces (also with ``##`` continuations), and a single ``Ġ`` piece
  standing for one whitespace character. Any whitespace run is emitted as one
  ``Ġ`` per whitespace character, so ``decode(encode(text))`` is byte-exact
  for texts built from in-vocab characters and single-space runs, and
  content-preserving for other whitespace (each whitespace character maps to
  exactly one ``Ġ`` and back). Words that fail WordPiece longest-match
  decomposition (e.g. runs containing characters outside printed ASCII) map
  to a single ``[UNK]``.
* Encoding first splits out the five special tokens (longest marker first,
  so ``</opt>`` wins over ``<opt>``), then pre-tokenizes the remaining text
  into whitespace runs, word runs (``\\w+``, Unicode-aware) and single
  non-word non-space characters, then applies greedy longest-match WordPiece
  per word.

Decoding concatenates piece strings: ``Ġ`` → one space, ``##x`` → ``x`` with
no space, a special token → its literal marker text, ``[PAD]`` → skipped,
``[UNK]`` → the literal string ``[UNK]``. Because markers are atomic and
whitespace is an explicit piece, every marker boundary survives a
``encode → decode`` round-trip exactly, even when a marker is adjacent to a
word with no whitespace (``<goal>Pick …``).

Non-printed-ASCII characters and any string that cannot be decomposed map to
``[UNK]``; the data pipeline NFC-normalizes and collapses whitespace before
text reaches this tokenizer, which is why that fallback is acceptable here.
"""

from __future__ import annotations

import re
import unicodedata

from .errors import BudgetTooSmallError

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_ID = 0
UNK_ID = 1

# The five RAPPORT special tokens, in id order (2..6). FR-TOK-01 / TASK-04.
SPECIAL_TOKENS: tuple[str, ...] = ("<task>", "<goal>", "<state>", "<opt>", "</opt>")
TASK_ID, GOAL_ID, STATE_ID, OPT_ID, OPT_CLOSE_ID = range(2, 7)
SPECIAL_TOKEN_IDS: dict[str, int] = {
    token: 2 + index for index, token in enumerate(SPECIAL_TOKENS)
}
SPECIAL_ID_TO_TOKEN: dict[int, str] = {value: key for key, value in SPECIAL_TOKEN_IDS.items()}

_SPACE_PIECE = "Ġ"  # one whitespace character

# Small fixed set of multi-character pieces (word-initial and ## continuation
# variants). Chosen from the most common English digraphs and suffixes so the
# greedy longest-match path is exercised by real words; still tiny and fixed.
_MULTI_CHAR_PIECES: tuple[str, ...] = (
    "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
    "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar",
    "st", "to", "nt", "ng", "qu", "ch", "sh",
    "ing", "tion", "est", "ate", "ver",
    "##th", "##he", "##in", "##er", "##an", "##re", "##on", "##at",
    "##en", "##nd", "##ti", "##es", "##ed", "##is", "##it", "##al",
    "##ar", "##st", "##to", "##nt", "##ng", "##ch",
    "##ing", "##tion", "##est", "##ate", "##ver", "##ly", "##ic",
)

# Every printed-ASCII character is a piece, bare (word-initial) and as a
# continuation. This is what guarantees that any word built from printed
# ASCII decomposes without hitting [UNK].
_ASCII_PIECES: tuple[str, ...] = tuple(
    chr(code) for code in range(32, 127) if chr(code) not in " \t\n\r\x0b\x0c"
)


def _build_vocab() -> list[str]:
    pieces = {_SPACE_PIECE, *_MULTI_CHAR_PIECES}
    for char in _ASCII_PIECES:
        pieces.add(char)
        pieces.add("##" + char)
    return [PAD_TOKEN, UNK_TOKEN, *SPECIAL_TOKENS, *sorted(pieces)]

_VOCAB: list[str] = _build_vocab()
_TOKEN_TO_ID: dict[str, int] = {token: index for index, token in enumerate(_VOCAB)}
_ID_TO_TOKEN: dict[int, str] = {index: token for index, token in enumerate(_VOCAB)}

# Longest marker first so "</opt>" is never split into "<opt>" + "/".
_SPECIAL_SPLIT_RE = re.compile(
    "(" + "|".join(
        re.escape(token) for token in sorted(SPECIAL_TOKENS, key=len, reverse=True)
    ) + ")"
)
_PRETOKEN_RE = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)


class RapportTokenizer:
    """The deterministic stand-in tokenizer (see module docstring)."""

    def __init__(self) -> None:
        # Copies: callers may not mutate the shared tables.
        self.token_to_id: dict[str, int] = dict(_TOKEN_TO_ID)
        self.id_to_token: dict[int, str] = dict(_ID_TO_TOKEN)
        self.vocab_size: int = len(_VOCAB)
        self.special_token_ids: dict[str, int] = dict(SPECIAL_TOKEN_IDS)

    # -- encoding -----------------------------------------------------------

    def special_token_id(self, token: str) -> int:
        """Return the single special-id for ``token``; KeyError if not special."""
        return SPECIAL_TOKEN_IDS[self._require_special(token)]

    @staticmethod
    def _require_special(token: str) -> str:
        if token not in SPECIAL_TOKEN_IDS:
            raise KeyError(f"{token!r} is not a RAPPORT special token")
        return token

    def _wordpiece(self, word: str) -> list[int]:
        """Greedy longest-match WordPiece decomposition of one ``\\w+`` run."""
        ids: list[int] = []
        start = 0
        is_first = True
        while start < len(word):
            end = len(word)
            found_id: int | None = None
            while end > start:
                piece = word[start:end] if is_first else "##" + word[start:end]
                piece_id = _TOKEN_TO_ID.get(piece)
                if piece_id is not None:
                    found_id = piece_id
                    break
                end -= 1
            if found_id is None:
                # WordPiece convention: an undecomposable word maps to one UNK.
                return [UNK_ID]
            ids.append(found_id)
            start = end
            is_first = False
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _PRETOKEN_RE.findall(text):
            if piece.isspace():
                ids.extend([_TOKEN_TO_ID[_SPACE_PIECE]] * len(piece))
            elif re.fullmatch(r"\w+", piece) is not None:
                ids.extend(self._wordpiece(piece))
            else:  # single non-word, non-space character
                ids.append(_TOKEN_TO_ID.get(piece, UNK_ID))
        return ids

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        """Encode ``text``; ``max_length`` truncates from the end if set.

        ``max_length`` is a raw cap with no bookkeeping; budget-aware
        truncation with flags and per-source accounting lives in
        ``encode_example``.
        """
        if max_length is not None and max_length < 1:
            raise BudgetTooSmallError("max_length must be a positive integer", field="max_length")
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        text = unicodedata.normalize("NFC", text)
        ids: list[int] = []
        for segment in _SPECIAL_SPLIT_RE.split(text):
            if not segment:
                continue
            special_id = SPECIAL_TOKEN_IDS.get(segment)
            if special_id is not None:
                ids.append(special_id)
            else:
                ids.extend(self._encode_plain(segment))
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    # -- decoding -----------------------------------------------------------

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        """Decode ids back to text; ``[PAD]`` is skipped, unknown ids → ``[UNK]``."""
        out: list[str] = []
        for raw in ids:
            piece = _ID_TO_TOKEN.get(int(raw), UNK_TOKEN)
            if piece == PAD_TOKEN:
                continue
            if piece in SPECIAL_TOKEN_IDS:
                out.append(piece)
            elif piece == _SPACE_PIECE:
                out.append(" ")
            elif piece.startswith("##"):
                out.append(piece[2:])
            else:
                out.append(piece)
        return "".join(out)


DEFAULT_TOKENIZER = RapportTokenizer()
