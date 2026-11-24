"""Lightweight paraphraser: word and phrase substitution, filtered by the trained language model.

This is NOT a neural rewriting system.  It finds slots where a synonym or a shorter/longer phrase could replace
the original, asks the language model how natural each alternative is *in context* (left context, the alternative,
and the next two original words) and only keeps alternatives that score close to the original.  It therefore changes
vocabulary and phrasing, but not sentence structure.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

import numpy as np

from ml_models.preprocessing.text_processing import tokenize
from ml_models.writing.common import match_case
from ml_models.writing.lexicon import PHRASES, WORD_TABLE

MAX_SLOTS_SCORED = 14        # keeps the response time bounded
MARGIN = 2.5                 # an alternative may be at most e^-2.5 (~12x) less likely than the original
MAX_CHANGES = 6


@dataclass
class Slot:
    start: int
    end: int
    original: str
    options: list[str]
    scored: list[tuple[float, str]] = field(default_factory=list)   # (score - original score, option) allowed only


@dataclass
class Variant:
    text: str
    changes: list[dict]
    fluency_delta: float | None


def find_slots(text: str) -> list[Slot]:
    slots: list[Slot] = []
    taken: list[tuple[int, int]] = []

    def free(s, e):
        return all(e <= a or s >= b for a, b in taken)

    for phrase in sorted(PHRASES, key=len, reverse=True):
        for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", text, flags=re.I):
            if free(m.start(), m.end()):
                opts = [match_case(m.group(0), o) for o in PHRASES[phrase]]
                slots.append(Slot(m.start(), m.end(), m.group(0), opts))
                taken.append((m.start(), m.end()))
    for m in re.finditer(r"[A-Za-z]+", text):
        w = m.group(0)
        alts = WORD_TABLE.get(w.lower())
        if alts and free(m.start(), m.end()):
            slots.append(Slot(m.start(), m.end(), w, [match_case(w, a) for a in alts]))
    return sorted(slots, key=lambda s: s.start)


class Scorer:
    """Log-probability of a short word sequence given the left context, using a loaded model."""

    def __init__(self, engine, model_id: str):
        self.engine = engine
        self.model_id = model_id
        self.model = engine._get(model_id)
        self.vocab = engine.vocab
        self._cache: dict = {}

    def context_ids(self, left_text: str) -> list[int]:
        return list(self.engine.preprocessor.parse(left_text, whole_word=True).context_ids)

    def logprob(self, ctx: list[int], phrase: str) -> float:
        tokens = [t for t in tokenize(phrase)]
        ids = self.vocab.encode(tokens)
        key = (tuple(ctx[-12:]), tuple(ids))
        if key in self._cache:
            return self._cache[key]
        total, cur = 0.0, list(ctx)
        for tid in ids:
            p = self.model.next_token_probs(cur)
            total += math.log(max(float(p[tid]), 1e-9))
            cur = (cur + [tid])[-30:]
        self._cache[key] = total
        return total


def _right_words(text: str, end: int, n: int = 2) -> str:
    return " ".join(re.findall(r"[A-Za-z']+", text[end:end + 60])[:n])


def score_slots(text: str, slots: list[Slot], scorer: Scorer) -> None:
    for slot in slots[:MAX_SLOTS_SCORED]:
        left = text[: slot.start]
        ctx = scorer.context_ids(left)
        right = _right_words(text, slot.end)
        base = scorer.logprob(ctx, f"{slot.original} {right}".strip())
        for opt in slot.options:
            s = scorer.logprob(ctx, f"{opt} {right}".strip()) - base
            if s >= -MARGIN:
                slot.scored.append((s, opt))
        slot.scored.sort(key=lambda t: -t[0])


def _apply(text: str, choices: list[tuple[Slot, str, float]]) -> Variant:
    out, pos, changes, delta = [], 0, [], 0.0
    for slot, opt, sc in sorted(choices, key=lambda c: c[0].start):
        out.append(text[pos: slot.start])
        out.append(opt)
        pos = slot.end
        changes.append({"from": slot.original, "to": opt})
        delta += sc
    out.append(text[pos:])
    return Variant("".join(out), changes, round(delta, 2))


def paraphrase(text: str, engine=None, model_id: str | None = None, variants: int = 3, seed: int | None = None) -> tuple[list[Variant], bool]:
    """Return (variants, scored).  ``scored`` is False when no language model was available."""
    rng = random.Random(seed)
    slots = find_slots(text)
    if not slots:
        return [], engine is not None
    scorer = None
    if engine is not None:
        try:
            mid = model_id or next((m for m in ("transformer", "ensemble", "lstm", "ngram") if engine.is_ready(m)), None)
            scorer = Scorer(engine, mid) if mid else None
        except Exception:  # noqa: BLE001 - a model problem must not break the tool
            scorer = None
    if scorer is not None:
        score_slots(text, slots, scorer)
        usable = [s for s in slots if s.scored]
    else:                                    # no model: only the safest, model-free choices (phrases and a few words)
        usable = []
        for s in slots:
            s.scored = [(0.0, o) for o in s.options]
            usable.append(s)
    if not usable:
        return [], scorer is not None

    results: list[Variant] = []
    seen = {text}

    def add(choices):
        if not choices:
            return
        v = _apply(text, choices[:MAX_CHANGES])
        if v.text not in seen:
            seen.add(v.text)
            results.append(v)

    # 1) best alternative everywhere (largest score first, capped)
    ranked = sorted(usable, key=lambda s: -s.scored[0][0])[:MAX_CHANGES]
    add([(s, s.scored[0][1], s.scored[0][0]) for s in ranked])
    # 2) second-best where it exists, on alternate slots
    add([(s, (s.scored[1] if len(s.scored) > 1 else s.scored[0])[1], (s.scored[1] if len(s.scored) > 1 else s.scored[0])[0])
         for i, s in enumerate(usable) if i % 2 == 1 or len(usable) == 1][:MAX_CHANGES])
    # 3+) random subsets
    tries = 0
    while len(results) < variants and tries < 12:
        tries += 1
        k = max(1, min(MAX_CHANGES, round(len(usable) * rng.uniform(0.3, 0.6))))
        pick = rng.sample(usable, min(k, len(usable)))
        add([(s, *(lambda c: (c[1], c[0]))(rng.choice(s.scored[:3]))) for s in pick])
    results.sort(key=lambda v: -(v.fluency_delta or 0))
    return results[:variants], scorer is not None
