"""Rule-based style checker and readability statistics.

Reports measurable things (sentence length, passive voice, filler words, wordy phrases, readability) and, for
formal writing, contractions and slang.  Suggestions are advice, not corrections: style is a matter of taste.
"""
from __future__ import annotations

import re
from collections import Counter

from ml_models.writing.common import Issue, match_case, resolve_overlaps, sentence_spans, words

LONG_SENTENCE = 30
VERY_LONG_SENTENCE = 40

FILLERS = {"very", "really", "just", "quite", "basically", "actually", "literally", "simply", "somewhat",
           "totally", "extremely", "definitely", "probably", "honestly"}

WORDY = {
    "in order to": "to", "due to the fact that": "because", "at this point in time": "now",
    "a large number of": "many", "a majority of": "most", "in the event that": "if", "for the purpose of": "for",
    "with regard to": "about", "with respect to": "about", "has the ability to": "can", "is able to": "can",
    "are able to": "can", "in spite of the fact that": "although", "each and every": "every",
    "it is important to note that": "", "needless to say": "", "at the present time": "currently",
    "in the near future": "soon", "on a daily basis": "daily", "in close proximity to": "near",
    "a great deal of": "much", "make a decision": "decide", "give consideration to": "consider",
    "take into consideration": "consider", "in my opinion": "", "last but not least": "finally",
    "the reason why is that": "because", "whether or not": "whether", "free of charge": "free",
    "past history": "history", "end result": "result", "future plans": "plans", "close proximity": "proximity",
}

CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not", "can't": "cannot", "won't": "will not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not", "haven't": "have not",
    "hasn't": "has not", "hadn't": "had not", "shouldn't": "should not", "wouldn't": "would not",
    "couldn't": "could not", "i'm": "I am", "you're": "you are", "we're": "we are", "they're": "they are",
    "i've": "I have", "we've": "we have", "they've": "they have", "you've": "you have", "i'll": "I will",
    "we'll": "we will", "they'll": "they will", "you'll": "you will", "let's": "let us",
}
INFORMAL = {"gonna": "going to", "wanna": "want to", "gotta": "have to", "kinda": "somewhat", "sorta": "somewhat",
            "lots": "many", "stuff": "things", "guys": "people", "awesome": "excellent", "ok": "acceptable",
            "okay": "acceptable", "yeah": "yes", "nope": "no", "cool": "impressive", "kids": "children"}

_BE = r"(?:am|is|are|was|were|be|been|being)"
_IRREG_PART = r"(?:known|made|done|given|taken|seen|written|shown|built|found|held|kept|led|told|born|chosen|spoken|broken|driven|eaten|gone|grown|hidden|left|lost|met|paid|read|run|sent|set|sold|stolen|thrown|understood|won)"
_PASSIVE = re.compile(rf"\b{_BE}\s+(?:(?:\w+ly)\s+)?(\w+ed|{_IRREG_PART})\b", re.I)
# adjectives that end in -ed and are not passive verbs ("is tired", "was excited")
_NOT_PASSIVE = {"tired", "excited", "worried", "interested", "pleased", "scared", "bored", "surprised", "confused",
                "married", "shocked", "embarrassed", "concerned", "supposed", "used", "need", "indeed", "red", "bed", "speed"}

_STOP = set("""the and for that with this from they have were been their there which would about could should other
into than then them these those what when where while your more most some such only also very over after before
because between through during being does doing each both many much just like will shall can may might must""".split())


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", w)))


def readability(text: str) -> dict:
    ws = words(text)
    sents = sentence_spans(text)
    n_w, n_s = len(ws), max(1, len(sents))
    if n_w == 0:
        return {"words": 0, "sentences": 0, "avg_sentence_words": 0, "reading_ease": None, "grade_level": None,
                "reading_level": None, "reading_time_seconds": 0, "unique_word_ratio": None}
    syl = sum(_syllables(w) for w in ws)
    ease = 206.835 - 1.015 * (n_w / n_s) - 84.6 * (syl / n_w)
    grade = 0.39 * (n_w / n_s) + 11.8 * (syl / n_w) - 15.59
    ease = max(0.0, min(100.0, ease))
    level = ("very easy" if ease >= 90 else "easy" if ease >= 70 else "standard" if ease >= 60 else
             "fairly difficult" if ease >= 50 else "difficult" if ease >= 30 else "very difficult")
    return {
        "words": n_w, "sentences": len(sents), "avg_sentence_words": round(n_w / n_s, 1),
        "reading_ease": round(ease, 1), "grade_level": round(max(0.0, grade), 1), "reading_level": level,
        "reading_time_seconds": round(n_w / 200 * 60), "unique_word_ratio": round(len({w.lower() for w in ws}) / n_w, 2),
    }


def _with_trailing_space(text: str, start: int, end: int) -> int:
    return end + 1 if end < len(text) and text[end] == " " else end


def check_style(text: str, tone: str = "neutral") -> tuple[dict, list[Issue]]:
    issues: list[Issue] = []
    sents = sentence_spans(text)

    for s, e in sents:
        n = len(words(text[s:e]))
        if n > VERY_LONG_SENTENCE:
            issues.append(Issue("long_sentence", "Readability", "warning", s, e,
                                f"This sentence has {n} words. Split it into two or more shorter sentences."))
        elif n > LONG_SENTENCE:
            issues.append(Issue("long_sentence", "Readability", "info", s, e,
                                f"This sentence has {n} words. Shorter sentences are easier to follow."))

    for m in _PASSIVE.finditer(text):
        if m.group(1).lower() in _NOT_PASSIVE:
            continue
        issues.append(Issue("passive_voice", "Style", "info", m.start(), m.end(),
                            f"Possible passive voice (“{m.group(0)}”). Active voice is usually clearer: say who does the action."))

    for m in re.finditer(r"\b(" + "|".join(sorted(FILLERS)) + r")\b", text, flags=re.I):
        end = _with_trailing_space(text, m.start(), m.end())
        issues.append(Issue("filler_word", "Style", "info", m.start(), end,
                            f"“{m.group(1)}” often adds little. Remove it or choose a stronger word.", ""))

    for phrase, short in sorted(WORDY.items(), key=lambda kv: -len(kv[0])):
        for m in re.finditer(r"\b" + re.escape(phrase) + r"\b,?\s?", text, flags=re.I):
            end = m.end()
            rep = match_case(m.group(0), short) + (" " if short and m.group(0).endswith(" ") else "")
            if short == "" :
                rep = ""
            msg = f"Wordy: “{phrase}” can be “{short}”." if short else f"“{phrase}” can usually be cut."
            issues.append(Issue("wordy_phrase", "Style", "info", m.start(), end, msg, rep))

    # three consecutive sentences that start with the same word
    firsts = []
    for s, e in sents:
        m = re.match(r"[\"'(]*([A-Za-z']+)", text[s:e])
        firsts.append((m.group(1).lower() if m else "", s, e))
    for i in range(len(firsts) - 2):
        w = firsts[i][0]
        if w and w == firsts[i + 1][0] == firsts[i + 2][0]:
            issues.append(Issue("repeated_start", "Style", "info", firsts[i + 2][1], firsts[i + 2][2],
                                f"Three sentences in a row start with “{w}”. Vary the openings."))

    # a content word used three times in one sentence
    for s, e in sents:
        counts = Counter(w.lower() for w in words(text[s:e]) if len(w) >= 5 and w.lower() not in _STOP)
        for w, c in counts.items():
            if c >= 3:
                last = [m for m in re.finditer(rf"\b{re.escape(w)}\b", text[s:e], flags=re.I)][-1]
                issues.append(Issue("repeated_word_in_sentence", "Style", "info", s + last.start(), s + last.end(),
                                    f"“{w}” appears {c} times in this sentence. Use a synonym or a pronoun."))

    if tone == "formal":
        for m in re.finditer(r"\b[A-Za-z]+'[a-z]+\b", text):
            low = m.group(0).lower().replace("’", "'")
            if low in CONTRACTIONS:
                issues.append(Issue("contraction", "Tone", "info", m.start(), m.end(),
                                    f"Formal writing usually avoids contractions: “{CONTRACTIONS[low]}”.",
                                    match_case(m.group(0), CONTRACTIONS[low])))
        for m in re.finditer(r"\b(" + "|".join(INFORMAL) + r")\b", text, flags=re.I):
            rep = INFORMAL[m.group(1).lower()]
            issues.append(Issue("informal_word", "Tone", "info", m.start(), m.end(),
                                f"“{m.group(1)}” is informal. Consider “{rep}”.", match_case(m.group(1), rep)))

    stats = readability(text)
    stats["tone"] = tone
    return stats, resolve_overlaps(issues)
