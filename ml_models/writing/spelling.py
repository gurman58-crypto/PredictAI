"""Dictionary spell checker with language-model-ranked suggestions.

* A word is "known" if it is in the bundled English word list (data/en_words.tsv.gz, ~160k words) or in the
  vocabulary the language models were trained on.
* Candidates for an unknown word are the known words one edit away (two for longer words); they are ranked by
  how common they are and, when a language model is available, by how well they fit the words before them.
* Proper nouns (capitalised mid-sentence), ALL-CAPS acronyms, camelCase, web addresses, e-mails and file names
  are never flagged.
"""
from __future__ import annotations

import gzip
import math
import re
from functools import lru_cache
from pathlib import Path

from ml_models.writing.common import Issue, match_case

_DATA = Path(__file__).parent / "data" / "en_words.tsv.gz"
_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
_SUFFIXES = ("'s", "s'", "n't", "'re", "'ve", "'ll", "'d", "'m")
MAX_EDIT2_PER_CALL = 3
MAX_SPELLING_ISSUES = 40


@lru_cache(maxsize=1)
def _freq() -> dict[str, int]:
    freq: dict[str, int] = {}
    with gzip.open(_DATA, "rt", encoding="utf-8") as fh:
        for line in fh:
            w, _, f = line.rstrip("\n").partition("\t")
            freq[w] = int(f)
    return freq


def add_known_words(words) -> None:
    """Add extra known words (for example the language-model vocabulary)."""
    f = _freq()
    for w in words:
        if re.fullmatch(r"[a-z]{3,}", w) and w not in f:
            f[w] = 60


_BRITISH = [
    (r"our(s|ed|ing|ful|less|able|ite)?$", r"or\1"), (r"is(e|es|ed|ing|ation|ations|er|ers)$", r"iz\1"),
    (r"ys(e|es|ed|ing)$", r"yz\1"), (r"tre(s)?$", r"ter\1"), (r"ence(s)?$", r"ense\1"),
    (r"ll(ed|ing|er|ers)$", r"l\1"), (r"ae", "e"), (r"ogue(s)?$", r"og\1"), (r"ph?aedi", "pedi"),
]


def _is_british_variant(w: str) -> bool:
    """British spellings (colour, organise, centre, travelled, defence) are accepted as correct."""
    f = _freq()
    for pat, rep in _BRITISH:
        v = re.sub(pat, rep, w)
        if v != w and v in f:
            return True
    return False


def is_known(word: str) -> bool:
    w = word.lower().replace("’", "'")
    f = _freq()
    if w in f:
        return True
    if _is_british_variant(w):
        return True
    for suf in _SUFFIXES:
        if w.endswith(suf) and (w[: -len(suf)] in f or w[: -len(suf)] + "e" in f):
            return True
    return False


def _edits1(word: str) -> set[str]:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in _LETTERS]
    inserts = [a + c + b for a, b in splits for c in _LETTERS]
    return set(deletes + transposes + replaces + inserts)


def _rank(word: str, cands: dict[str, int], lm_probs=None, vocab=None) -> list[str]:
    freq = _freq()

    def score(c: str) -> float:
        s = math.log(freq[c.replace("'", "'")] + 1) if c in freq else 0.0
        if c[0] == word[0]:
            s += 2.0                      # people rarely mistype the first letter
        if lm_probs is not None and vocab is not None:
            tid = vocab.stoi.get(c)
            p = float(lm_probs[tid]) if tid is not None else 0.0
            s += 0.7 * math.log(p + 1e-6)  # how well the word fits what comes before it
        return s - 3.0 * (cands[c] - 1)    # edit distance 2 is a much weaker match

    return sorted(cands, key=score, reverse=True)


def suggestions(word: str, lm_probs=None, vocab=None, allow_edit2: bool = True, limit: int = 5) -> list[str]:
    w = word.lower().replace("’", "'")
    freq = _freq()
    e1 = _edits1(w)
    cands = {c: 1 for c in e1 if c in freq}
    if not cands and allow_edit2 and 4 <= len(w) <= 12:
        for e in e1:
            for e2 in _edits1(e):
                if e2 in freq and e2 not in cands:
                    cands[e2] = 2
    if not cands:
        return []
    return [match_case(word, c) for c in _rank(w, cands, lm_probs, vocab)[:limit]]


def _is_skippable(text: str, m: re.Match, sentence_start: bool) -> bool:
    w = m.group(0)
    s, e = m.start(), m.end()
    if len(w) <= 2:
        return True
    if w.isupper():                                       # acronyms: NASA, LSTM
        return True
    if any(c.isupper() for c in w[1:]):                   # camelCase / iPhone
        return True
    if w[0].isupper() and not sentence_start:             # proper noun
        return True
    if text[e:e + 3] == "://" or w.lower() == "www":        # URL scheme / host prefix
        return True
    before = text[s - 1] if s > 0 else ""
    after = text[e] if e < len(text) else ""
    if (before and before in "@/\\_#$%&=+~<>") or (after and after in "@/\\_#=+~<>"):
        return True
    if before.isdigit() or after.isdigit():
        return True
    if after == "." and e + 1 < len(text) and (text[e + 1].isalnum()):   # file.py, example.com
        return True
    if before == "." and s > 1 and text[s - 2].isalnum():
        return True
    return False


def _starts_sentence(text: str, pos: int) -> bool:
    prev = text[:pos].rstrip(" \t\"'([")
    return prev == "" or prev[-1] in ".!?\n"


def check_spelling(text: str, engine=None, model_id: str | None = None, ignore: set[str] | None = None) -> list[Issue]:
    ignore = {w.lower() for w in (ignore or ())}
    out: list[Issue] = []
    edit2_left = MAX_EDIT2_PER_CALL
    vocab = getattr(engine, "vocab", None) if engine is not None else None
    model = None
    if engine is not None:
        try:
            mid = model_id or next((m for m in ("transformer", "ensemble", "lstm", "ngram") if engine.is_ready(m)), None)
            model = engine._get(mid) if mid else None
        except Exception:  # noqa: BLE001 - ranking by context is optional
            model = None
    for m in _TOKEN.finditer(text):
        if len(out) >= MAX_SPELLING_ISSUES:
            break
        w = m.group(0)
        if w.lower() in ignore or is_known(w):
            continue
        if _is_skippable(text, m, _starts_sentence(text, m.start())):
            continue
        probs = None
        if model is not None:
            try:
                ctx = engine.preprocessor.parse(text[: m.start()], whole_word=True).context_ids
                probs = model.next_token_probs(ctx)
            except Exception:  # noqa: BLE001
                probs = None
        sugg = suggestions(w, probs, vocab, allow_edit2=edit2_left > 0)
        if not sugg:
            out.append(Issue("spelling", "Spelling", "error", m.start(), m.end(),
                             f"“{w}” is not in the dictionary.", None, []))
            continue
        edit2_left -= 1 if _edit_distance_one_missing(w, sugg[0]) else 0
        out.append(Issue("spelling", "Spelling", "error", m.start(), m.end(),
                         f"“{w}” is not in the dictionary. Did you mean “{sugg[0]}”?", sugg[0], sugg))
    return out


def _edit_distance_one_missing(word: str, top: str) -> bool:
    return abs(len(word) - len(top)) > 1 or word.lower() not in _edits1(top.lower())
