"""Shared text normalisation, tokenisation and vocabulary handling.

This single module is used by the dataset preparation script, both training
scripts, the evaluation script and the API.  That is what guarantees that the
model sees exactly the same preprocessing in training and in production.

Policy (also documented in documentation/PREPROCESSING.md):
  * Unicode NFKC normalisation, typographic quotes/dashes mapped to ASCII.
  * Lower-casing (suggestions are lower-case; the UI re-applies capitalisation
    only to the text it inserts, never to the user's existing text).
  * Tokens: numbers (``1,000`` / ``3.5``), words (with internal hyphens),
    the clitics ``'s 're 've 'll 'd 'm 't`` as separate tokens, and the
    punctuation marks ``. , ; : ! ? ( ) "``.  Anything else is dropped.
  * A paragraph / document end is represented by ``<eos>``.
  * Out-of-vocabulary tokens become ``<unk>``.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from nltk.tokenize import RegexpTokenizer

PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"
SPECIAL_TOKENS = (PAD, UNK, EOS)
PAD_ID, UNK_ID, EOS_ID = 0, 1, 2

PREPROCESSING_VERSION = "1.0"

TOKEN_PATTERN = (
    r"<unk>"
    r"|\d+(?:[.,]\d+)*(?![^\W\d_])"          # numbers
    r"|[^\W_]+(?:-[^\W_]+)*"                  # words, optional internal hyphens
    r"|'(?:s|re|ve|ll|d|m|t)\b"               # clitics as separate tokens
    r"|[.,;:!?()\"]"                          # punctuation
)
_TOKENIZER = RegexpTokenizer(TOKEN_PATTERN)

_CHAR_MAP = str.maketrans(
    {
        "‘": "'", "’": "'", "‛": "'",
        "“": '"', "”": '"',
        "–": "-", "—": "-", "−": "-",
        " ": " ",
    }
)
_WHITESPACE = re.compile(r"\s+")

SENTENCE_END = {".", "!", "?"}
_NO_SPACE_BEFORE = {".", ",", ";", ":", "!", "?", ")"}
_MAX_PREFIX_CHARS = 40
_PARTIAL_WORD = re.compile(r"([^\W_]+)$")


def normalize(text: str) -> str:
    """NFKC, ASCII quotes/dashes, lower-case, collapsed whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP).lower()
    return _WHITESPACE.sub(" ", text)


def tokenize(text: str) -> list[str]:
    """Normalise and tokenise a piece of text into string tokens."""
    return _TOKENIZER.tokenize(normalize(text))


def is_word_token(token: str) -> bool:
    """True for tokens that may be shown as a *word* suggestion."""
    return token not in SPECIAL_TOKENS and any(ch.isalnum() for ch in token)


# --------------------------------------------------------------------------
# Detokenisation (used for sentence continuation and evaluation examples)
# --------------------------------------------------------------------------
def detokenize(tokens: Iterable[str], *, quote_open: bool = False, keep_unk: bool = False) -> str:
    """Join tokens into readable text (no leading space is added)."""
    out: list[str] = []
    glue_next = True  # nothing before the first token
    for tok in tokens:
        if tok == EOS or tok == PAD or (tok == UNK and not keep_unk):
            continue
        if tok == '"':
            if quote_open:        # closing quote: attach to previous token
                out.append(tok)
                quote_open = False
                glue_next = False
            else:                 # opening quote: space before, none after
                out.append(tok if glue_next else " " + tok)
                quote_open = True
                glue_next = True
            continue
        if tok.startswith("'") or tok in _NO_SPACE_BEFORE:
            out.append(tok)
        elif tok == "(":
            out.append(tok if glue_next else " " + tok)
            glue_next = True
            continue
        else:
            out.append(tok if glue_next else " " + tok)
        glue_next = False
    return "".join(out)


def capitalise_after(prev_text: str, new_text: str) -> str:
    """Capitalise the first letter of ``new_text`` if it starts a sentence."""
    stripped = prev_text.rstrip()
    starts_sentence = (not stripped) or stripped[-1] in SENTENCE_END
    if not starts_sentence:
        return new_text
    for i, ch in enumerate(new_text):
        if ch.isalpha():
            return new_text[:i] + ch.upper() + new_text[i + 1 :]
        if not ch.isspace() and ch not in "\"(":
            break
    return new_text


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
class Vocabulary:
    def __init__(self, itos: Sequence[str]):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        if self.itos[:3] != list(SPECIAL_TOKENS):
            raise ValueError("vocabulary must start with <pad>, <unk>, <eos>")
        self.word_mask = np.array([is_word_token(t) for t in self.itos], dtype=bool)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        get = self.stoi.get
        return [get(t, UNK_ID) for t in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.itos[int(i)] for i in ids]

    @lru_cache(maxsize=2048)
    def prefix_mask(self, prefix: str) -> np.ndarray:
        """Boolean mask of word tokens that start with ``prefix`` (and differ from it)."""
        mask = np.fromiter(
            (t.startswith(prefix) and t != prefix for t in self.itos), dtype=bool, count=len(self.itos)
        )
        mask &= self.word_mask
        mask.setflags(write=False)
        return mask

    @classmethod
    def build(cls, token_stream: Iterable[str], max_size: int, min_freq: int = 2) -> "Vocabulary":
        from collections import Counter

        counts = Counter(t for t in token_stream if t not in SPECIAL_TOKENS)
        kept = [t for t, c in counts.most_common() if c >= min_freq][: max_size - len(SPECIAL_TOKENS)]
        return cls(list(SPECIAL_TOKENS) + kept)

    def to_dict(self) -> dict:
        return {
            "preprocessing_version": PREPROCESSING_VERSION,
            "token_pattern": TOKEN_PATTERN,
            "special_tokens": list(SPECIAL_TOKENS),
            "size": len(self.itos),
            "itos": self.itos,
        }

    def save(self, path: Path, extra: dict | None = None) -> None:
        data = self.to_dict()
        if extra:
            data.update(extra)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError(
                f"vocab was built with preprocessing {data.get('preprocessing_version')!r}, "
                f"code is {PREPROCESSING_VERSION!r} - retrain or regenerate the vocabulary"
            )
        if data.get("token_pattern") != TOKEN_PATTERN:
            raise ValueError("tokenizer pattern differs from the one used at training time")
        return cls(data["itos"])


# --------------------------------------------------------------------------
# Turning raw user text into model input
# --------------------------------------------------------------------------
@dataclass
class ParsedInput:
    context_ids: list[int]
    context_tokens: list[str]
    prefix: str | None        # partially typed word, if the cursor is inside one
    mode: str                 # "next_word" or "completion"
    quote_open: bool


class TextPreprocessor:
    """Converts the text before the cursor into model context ids."""

    def __init__(self, vocab: Vocabulary, max_context: int = 30, max_chars: int = 2000):
        self.vocab = vocab
        self.max_context = max_context
        self.max_chars = max_chars

    def parse(self, text: str, *, whole_word: bool = False) -> ParsedInput:
        """Split ``text`` (everything before the cursor) into model context + optional prefix.

        ``whole_word=True`` is used for explicit "continue my sentence" requests: a trailing
        word that exists in the vocabulary is then treated as finished, not as a prefix.
        """
        text = text[-self.max_chars :]
        prefix: str | None = None
        text_for_context = text
        m = _PARTIAL_WORD.search(text)
        if m and len(m.group(1)) <= _MAX_PREFIX_CHARS:
            before = text[: m.start()]
            # Only a *whole* word being typed counts as a prefix.  Fragments glued to
            # a previous token ("it's", "1,0", "well-kno") are treated as complete.
            if before == "" or before[-1].isspace() or before[-1] in '("':
                candidate = normalize(m.group(1))
                is_vocab_word = candidate in self.vocab.stoi
                no_longer_word = not self.vocab.prefix_mask(candidate).any()
                if is_vocab_word and (whole_word or no_longer_word):
                    pass  # a finished word: predict what follows it
                else:
                    prefix = candidate
                    text_for_context = before

        tokens = tokenize(text_for_context)[-self.max_context :]
        ids = self.vocab.encode(tokens)
        if not ids:                       # empty / unknown context -> sentence start
            ids = [EOS_ID]
            tokens = []
        return ParsedInput(
            context_ids=ids,
            context_tokens=tokens,
            prefix=prefix,
            mode="completion" if prefix else "next_word",
            quote_open=text.count('"') % 2 == 1,
        )
