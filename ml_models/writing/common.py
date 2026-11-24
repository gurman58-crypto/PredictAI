"""Shared helpers: issue records, case handling, sentence splitting."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

MAX_ISSUES = 200


@dataclass
class Issue:
    rule: str                    # stable identifier, e.g. "repeated_word"
    category: str                # Spelling | Grammar | Punctuation | Capitalisation | Style | Readability ...
    severity: str                # "error" | "warning" | "info"
    start: int                   # character offsets into the checked text
    end: int
    message: str
    replacement: str | None = None   # None = no automatic fix; "" = delete the span
    suggestions: list = field(default_factory=list)   # alternatives to offer (first one equals ``replacement``)

    def to_dict(self, text: str) -> dict:
        d = asdict(self)
        d["original"] = text[self.start:self.end]
        if not d["suggestions"] and self.replacement:
            d["suggestions"] = [self.replacement]
        return d


def match_case(original: str, new: str) -> str:
    """Give ``new`` the capitalisation pattern of ``original`` (UPPER, Title or lower)."""
    if not new:
        return new
    if len(original) > 1 and original.isupper():
        return new.upper()
    if original[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


def resolve_overlaps(issues: list[Issue]) -> list[Issue]:
    """Sort by position and drop any issue that overlaps one already kept (earlier wins)."""
    kept: list[Issue] = []
    last_end = -1
    for it in sorted(issues, key=lambda i: (i.start, -(i.end - i.start))):
        if it.start >= last_end or (it.start == it.end and it.start >= last_end):
            kept.append(it)
            last_end = max(last_end, it.end)
    return kept[:MAX_ISSUES]


_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "e.g", "i.e", "no", "fig", "inc", "jr", "sr", "approx", "al"}
_SENT_END = re.compile(r"[.!?]+[\"')\]]*(?=\s+|$)")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans (start, end) of sentences; abbreviations such as 'Dr.' do not end a sentence."""
    spans, start = [], 0
    for m in _SENT_END.finditer(text):
        before = text[start:m.start()].split()
        last = before[-1].lower().rstrip(".") if before else ""
        if m.group().startswith(".") and (last in _ABBREV or re.fullmatch(r"(?:[a-z]\.)+[a-z]", last)):
            continue                                   # "Dr.", "etc.", "U.S."
        end = m.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    # trim leading whitespace
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        out.append((s, e))
    return out


_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def words(text: str) -> list[str]:
    return _WORD.findall(text)
