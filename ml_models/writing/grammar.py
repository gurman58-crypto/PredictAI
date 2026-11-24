"""Rule-based grammar, spelling and punctuation checker.

It finds *common, clearly identifiable* mistakes and proposes a fix for each.  It is deliberately narrow
(a few dozen well-tested rules) so that it rarely flags correct text; it is not a full grammar parser and
will miss many errors.
"""
from __future__ import annotations

import re

from ml_models.writing.common import Issue, match_case, resolve_overlaps, sentence_spans, _ABBREV

# ---- spelling: frequent misspellings -> correct form ----------------------------------------------------
MISSPELLINGS = {
    "teh": "the", "recieve": "receive", "recieved": "received", "definately": "definitely", "definatly": "definitely",
    "seperate": "separate", "seperately": "separately", "occured": "occurred", "occurence": "occurrence",
    "untill": "until", "wich": "which", "becuase": "because", "becasue": "because", "beleive": "believe",
    "goverment": "government", "enviroment": "environment", "accomodate": "accommodate", "adress": "address",
    "begining": "beginning", "calender": "calendar", "collegue": "colleague", "comming": "coming",
    "concious": "conscious", "existance": "existence", "foriegn": "foreign", "freind": "friend",
    "grammer": "grammar", "happend": "happened", "independant": "independent", "knowlege": "knowledge",
    "neccessary": "necessary", "necesary": "necessary", "noticable": "noticeable", "occassion": "occasion",
    "persistant": "persistent", "posession": "possession", "prefered": "preferred", "publically": "publicly",
    "realy": "really", "relevent": "relevant", "rythm": "rhythm", "succesful": "successful",
    "successfull": "successful", "tomorow": "tomorrow", "truely": "truly", "wierd": "weird",
    "writting": "writing", "alot": "a lot", "thier": "their", "peice": "piece", "tommorow": "tomorrow",
    "acheive": "achieve", "arguement": "argument", "basicly": "basically", "commited": "committed",
    "dissapoint": "disappoint", "embarass": "embarrass", "excercise": "exercise", "familar": "familiar",
    "finaly": "finally", "gaurd": "guard", "harrass": "harass", "immediatly": "immediately",
    "intresting": "interesting", "libary": "library", "lenght": "length", "mispell": "misspell",
    "oppurtunity": "opportunity", "orignal": "original", "pasttime": "pastime", "probly": "probably",
    "recomend": "recommend", "refered": "referred", "responsability": "responsibility", "sucess": "success",
    "suprise": "surprise", "tecnology": "technology", "wether": "whether", "writen": "written",
}

# ---- confusions and fixed phrases (regex, replacement template) ----------------------------------------
PHRASES = [
    (r"\b(should|could|would|must|might)\s+of\b", lambda m: f"{m.group(1)} have", "Use “have”, not “of”: “{0} have”.", "Grammar"),
    (r"\bto\s+(?:a\s+)?lot\b", None, None, None),  # placeholder, removed below
    (r"\birregardless\b", lambda m: "regardless", "“Irregardless” is non-standard; use “regardless”.", "Grammar"),
    (r"\byour\s+(welcome)\b", lambda m: "you're welcome", "Use “you're” (you are).", "Grammar"),
    (r"\byour\s+(a|an|the|not|going|very|so|being|right|wrong|doing|going)\b", lambda m: f"you're {m.group(1)}", "Use “you're” (you are) here.", "Grammar"),
    (r"\btheir\s+(is|are|was|were)\b", lambda m: f"there {m.group(1)}", "Use “there”, not “their”, to say that something exists.", "Grammar"),
    (r"\b(more|less|better|worse|greater|bigger|smaller|larger|faster|slower|higher|lower|older|younger|rather)\s+then\b",
     lambda m: f"{m.group(1)} than", "Use “than” for comparisons.", "Grammar"),
    (r"\b(a|the)\s+(?:effect)\s+on\s+(?:affect)\b", None, None, None),
    (r"\bcould\s+care\s+less\b", lambda m: "couldn't care less", "The usual phrase is “couldn't care less”.", "Grammar"),
    (r"\bfor\s+all\s+intensive\s+purposes\b", lambda m: "for all intents and purposes", "The phrase is “for all intents and purposes”.", "Grammar"),
    (r"\bone\s+of\s+the\s+most\s+(\w+)\s+(?:thing)\b", None, None, None),
]
PHRASES = [p for p in PHRASES if p[1] is not None]

# ---- subject-verb agreement (simple, high-precision cases) ------------------------------------------------
_AGREE = [
    (r"\b(he|she|it)\s+(are|have|do)\b", {"are": "is", "have": "has", "do": "does"}),
    (r"\b(he|she|it)\s+don't\b", None),
    (r"\b(they|we|you)\s+(is|has)\b", {"is": "are", "has": "have"}),
    (r"\b(they|we|you)\s+was\b", None),
    (r"\bI\s+(is|are|has)\b", {"is": "am", "are": "am", "has": "have"}),
]

_VOWELS = set("aeiou")
_AN_CONSONANT_SOUND = {"university", "universe", "universal", "unique", "united", "union", "unit", "user", "use",
                       "used", "useful", "usual", "usually", "utility", "european", "euro", "one", "once", "uniform",
                       "unicorn", "unanimous", "ubiquitous", "eulogy", "ewe"}
_AN_VOWEL_SOUND = {"hour", "hours", "hourly", "honest", "honestly", "honor", "honour", "honorable", "heir", "heirloom"}


def _needs_an(word: str) -> bool:
    acronym = word.isupper() and len(word) >= 2
    w = word.lower()
    if w in _AN_VOWEL_SOUND:
        return True
    if acronym:                           # read letter by letter: "an FBI agent"; words like NASA: "a NASA team"
        return w[0] in "aefhilmnorsx" and not (len(w) >= 4 and w[1] in _VOWELS)
    if w in _AN_CONSONANT_SOUND or w.startswith(("uni", "use", "usu", "eu")):
        return False
    return w[0] in _VOWELS


def _check_phrases(text: str) -> list[Issue]:
    out = []
    for pattern, fn, msg, cat in PHRASES:
        for m in re.finditer(pattern, text, flags=re.I):
            rep = match_case(m.group(0), fn(m))
            out.append(Issue("common_error", cat, "error", m.start(), m.end(), msg.format(rep), rep))
    return out


def _check_spelling(text: str) -> list[Issue]:
    out = []
    for m in re.finditer(r"[A-Za-z']+", text):
        w = m.group(0)
        fix = MISSPELLINGS.get(w.lower())
        if fix:
            rep = match_case(w, fix)
            out.append(Issue("spelling", "Spelling", "error", m.start(), m.end(), f"Possible misspelling: “{w}”. Did you mean “{rep}”?", rep))
    return out


def _check_repeated(text: str) -> list[Issue]:
    out = []
    for m in re.finditer(r"\b([A-Za-z]+)(\s+)\1\b", text, flags=re.I):
        w = m.group(1).lower()
        if w in {"had", "that"}:            # "had had", "that that" can be correct
            continue
        out.append(Issue("repeated_word", "Grammar", "error", m.start(), m.end(),
                         f"The word “{m.group(1)}” is repeated.", m.group(1)))
    return out


def _check_articles(text: str) -> list[Issue]:
    out = []
    for m in re.finditer(r"\b(a|an)\s+([A-Za-z][A-Za-z'-]*)", text, flags=re.I):
        art, word = m.group(1), m.group(2)
        want = "an" if _needs_an(re.split(r"[-']", word)[0]) else "a"
        if art.lower() != want:
            rep = match_case(art, want)
            out.append(Issue("a_an", "Grammar", "error", m.start(1), m.end(1),
                             f"Use “{rep}” before “{word}”.", rep))
    return out


def _check_agreement(text: str) -> list[Issue]:
    out = []
    for pattern, table in _AGREE:
        for m in re.finditer(pattern, text, flags=re.I):
            subj = m.group(1)
            if table is None:
                verb = m.group(0)[len(subj):].strip().lower()
                rep_verb = {"don't": "doesn't", "was": "were"}[verb]
            else:
                verb = m.group(2).lower()
                rep_verb = table[verb]
            rep = f"{subj} {rep_verb}" if subj else rep_verb
            if pattern.startswith(r"\bI"):
                rep = f"I {rep_verb}"
            out.append(Issue("agreement", "Grammar", "error", m.start(), m.end(),
                             f"Subject–verb agreement: use “{rep}”.", match_case(m.group(0), rep) if not pattern.startswith(r"\bI") else rep))
    return out


def _check_i(text: str) -> list[Issue]:
    out = []
    for m in re.finditer(r"(?<![A-Za-z.'])i(?='(?:m|ll|ve|d)\b|\b)(?!\.[a-z])", text):
        nxt = text[m.end():m.end() + 2]
        if nxt.startswith(".") and text[m.end() + 1:m.end() + 2].lower() == "e":
            continue                         # "i.e."
        out.append(Issue("pronoun_i", "Capitalisation", "error", m.start(), m.end(), "The pronoun “I” is always capitalised.", "I"))
    return out


def _check_capitalisation(text: str) -> list[Issue]:
    out = []
    for s, e in sentence_spans(text):
        m = re.match(r"[\"'(\[]*([a-z])", text[s:e])
        if not m:
            continue
        # a lower-case first word after an abbreviation was already skipped by sentence_spans
        pos = s + m.start(1)
        word = re.match(r"[A-Za-z']+", text[pos:]).group(0)
        if word.lower() in {"i"}:
            continue
        out.append(Issue("sentence_capital", "Capitalisation", "error", pos, pos + 1,
                         "Start a sentence with a capital letter.", text[pos].upper()))
    return out


def _check_spacing(text: str) -> list[Issue]:
    out = []
    for m in re.finditer(r"(?<=\S)[ ]{2,}(?=\S)", text):
        out.append(Issue("double_space", "Typography", "info", m.start(), m.end(), "Use a single space between words.", " "))
    for m in re.finditer(r"(?<=\w)[ ]+(?=(?:[,;:!?]|\.(?=\s|$)))", text):
        out.append(Issue("space_before_punct", "Punctuation", "error", m.start(), m.end(), "Remove the space before the punctuation mark.", ""))
    for m in re.finditer(r"(?<=[A-Za-z ]),(?=[A-Za-z])", text):
        out.append(Issue("space_after_comma", "Punctuation", "error", m.start(), m.end(), "Add a space after the comma.", ", "))
    for m in re.finditer(r"(?<=[a-z]{3})\.(?=[A-Z][a-z]{2})", text):
        out.append(Issue("space_after_period", "Punctuation", "error", m.start(), m.end(), "Add a space after the full stop.", ". "))
    return out


def _check_ending(text: str) -> list[Issue]:
    stripped = text.rstrip()
    if not stripped or len(stripped.split()) < 4:
        return []
    if re.search(r"[A-Za-z0-9]$", stripped):
        last = stripped.split()[-1].lower()
        if last in _ABBREV:
            return []
        return [Issue("end_punctuation", "Punctuation", "info", len(stripped), len(stripped),
                      "The last sentence has no ending punctuation.", ".")]
    return []


def _check_balance(text: str) -> list[Issue]:
    out = []
    if text.count('"') % 2 == 1:
        pos = text.rfind('"')
        out.append(Issue("unbalanced_quote", "Punctuation", "warning", pos, pos + 1, "A quotation mark is not closed."))
    depth, last_open = 0, -1
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
            last_open = i
        elif ch == ")":
            if depth == 0:
                out.append(Issue("unbalanced_bracket", "Punctuation", "warning", i, i + 1, "This closing bracket has no opening bracket."))
            else:
                depth -= 1
    if depth > 0:
        out.append(Issue("unbalanced_bracket", "Punctuation", "warning", last_open, last_open + 1, "This opening bracket is not closed."))
    return out


def check_grammar(text: str, engine=None, ignore=None, spelling: bool = True) -> list[Issue]:
    """All grammar, spelling and punctuation issues.  ``engine`` (optional) lets spelling suggestions be ranked by
    how well they fit the sentence; ``ignore`` is a collection of words the user chose to accept."""
    from ml_models.writing.spelling import check_spelling

    issues: list[Issue] = []
    known = {w.lower() for w in (ignore or ())}
    for fn in (_check_spelling, _check_repeated, _check_articles, _check_agreement, _check_phrases, _check_i,
               _check_capitalisation, _check_spacing, _check_ending, _check_balance):
        issues.extend(fn(text))
    issues = [i for i in issues if not (i.rule == "spelling" and text[i.start:i.end].lower() in known)]
    if spelling:
        issues.extend(check_spelling(text, engine, ignore=known))
    return resolve_overlaps(issues)
