"""Small hand-written substitution lexicon for the paraphraser.

Every entry is a *candidate*: the paraphraser only uses it when the language model finds the result about
as natural as the original in context, which filters out grammatical mismatches.
"""
from __future__ import annotations

import re

# word -> synonyms (same part of speech, invariant forms)
WORDS: dict[str, list[str]] = {
    "big": ["large", "major"], "large": ["big", "substantial"], "huge": ["enormous", "vast", "immense"],
    "small": ["little", "modest"], "good": ["great", "fine", "solid"], "bad": ["poor", "unfavourable"],
    "important": ["significant", "crucial", "vital"], "difficult": ["challenging", "hard"],
    "hard": ["difficult", "demanding"], "easy": ["simple", "straightforward"], "fast": ["quick", "rapid"],
    "quick": ["fast", "rapid"], "quickly": ["rapidly", "swiftly"], "slowly": ["gradually"],
    "many": ["numerous", "several"], "often": ["frequently", "regularly"], "also": ["additionally", "moreover"],
    "however": ["nevertheless", "yet"], "therefore": ["thus", "hence", "consequently"],
    "because": ["since", "as"], "although": ["though", "even though"], "enough": ["sufficient"],
    "main": ["primary", "principal"], "clear": ["obvious", "evident", "apparent"], "happy": ["glad", "pleased"],
    "sad": ["unhappy"], "smart": ["intelligent", "clever"], "strong": ["powerful", "robust"],
    "different": ["distinct", "diverse"], "similar": ["comparable", "alike"], "common": ["widespread", "frequent"],
    "rare": ["uncommon", "scarce"], "whole": ["entire", "complete"], "usually": ["typically", "generally", "normally"],
    "mostly": ["mainly", "largely"], "maybe": ["perhaps", "possibly"], "sometimes": ["occasionally"],
    "nearly": ["almost"], "almost": ["nearly"], "completely": ["entirely", "fully", "totally"],
    "especially": ["particularly", "notably"], "mainly": ["chiefly", "primarily"], "currently": ["presently"],
    "recently": ["lately"], "finally": ["ultimately", "eventually"], "later": ["subsequently", "afterwards"],
    "idea": ["concept", "notion"], "problem": ["issue", "difficulty", "challenge"], "goal": ["objective", "aim"],
    "chance": ["opportunity"], "result": ["outcome"], "benefit": ["advantage"], "people": ["individuals"],
    "answer": ["response", "reply"], "huge": ["enormous", "vast"], "terrible": ["awful", "dreadful"],
    "great": ["excellent", "outstanding"], "interesting": ["fascinating", "engaging"], "useful": ["helpful", "valuable"],
    "obvious": ["clear", "evident"], "needed": ["required", "necessary"], "wrong": ["incorrect", "mistaken"],
    "right": ["correct", "accurate"], "real": ["genuine", "actual"], "enormous": ["huge", "massive"],
    "ability": ["capacity", "capability"], "method": ["approach", "technique"], "reason": ["cause", "motive"],
}

# regular verbs: base -> synonyms (bases); irregular source forms are listed in SOURCE_FORMS
VERBS: dict[str, list[str]] = {
    "help": ["assist", "aid"], "need": ["require"], "buy": ["purchase"], "try": ["attempt"],
    "keep": ["retain", "maintain"], "show": ["demonstrate", "illustrate"], "give": ["provide", "offer"],
    "think": ["believe", "consider"], "use": ["employ", "utilize"], "start": ["begin", "commence"],
    "stop": ["halt", "cease"], "finish": ["complete", "conclude"], "improve": ["enhance", "refine"],
    "increase": ["raise", "boost"], "reduce": ["decrease", "lessen"], "decrease": ["reduce", "lower"],
    "change": ["alter", "modify"], "create": ["produce", "generate"], "build": ["construct"],
    "explain": ["clarify", "describe"], "check": ["verify", "examine"], "find": ["discover", "locate"],
    "tell": ["inform", "notify"], "say": ["state", "mention"], "allow": ["permit", "enable"],
    "decide": ["determine", "resolve"], "choose": ["select", "pick"], "select": ["choose", "pick"],
    "produce": ["generate", "create"], "achieve": ["accomplish", "attain"], "require": ["need", "demand"],
    "begin": ["start", "commence"], "end": ["conclude", "finish"],
    "grow": ["expand", "increase"], "move": ["shift", "relocate"],
    "send": ["transmit", "dispatch"], "receive": ["obtain", "get"], "continue": ["proceed", "persist"],
    "avoid": ["prevent", "evade"], "discuss": ["examine", "debate"], "describe": ["portray", "depict"],
    "include": ["contain", "incorporate"], "consider": ["regard", "weigh"], "suggest": ["propose", "recommend"],
}
# irregular surface forms of source verbs -> (base, kind) where kind in {"s","past","ing"}; past participle == past
SOURCE_FORMS: dict[str, tuple[str, str]] = {
    "bought": ("buy", "past"), "kept": ("keep", "past"), "showed": ("show", "past"), "shown": ("show", "past"),
    "gave": ("give", "past"), "given": ("give", "past"), "thought": ("think", "past"), "built": ("build", "past"),
    "found": ("find", "past"), "told": ("tell", "past"), "said": ("say", "past"), "chose": ("choose", "past"),
    "chosen": ("choose", "past"), "grew": ("grow", "past"),
    "grown": ("grow", "past"), "sent": ("send", "past"), "began": ("begin", "past"), "begun": ("begin", "past"),
}

_DOUBLE = {"stop", "plan", "ship", "drop", "begin", "refer", "occur", "permit", "admit", "commit", "equal"}


def _s(base: str) -> str:
    if re.search(r"(s|x|z|ch|sh|o)$", base):
        return base + "es"
    if re.search(r"[^aeiou]y$", base):
        return base[:-1] + "ies"
    return base + "s"


def _past(base: str) -> str:
    if base.endswith("e"):
        return base + "d"
    if re.search(r"[^aeiou]y$", base):
        return base[:-1] + "ied"
    if base in _DOUBLE:
        return base + base[-1] + "ed"
    return base + "ed"


def _ing(base: str) -> str:
    if base.endswith("ie"):
        return base[:-2] + "ying"
    if base.endswith("e") and not base.endswith(("ee", "ye", "oe")):
        return base[:-1] + "ing"
    if base in _DOUBLE:
        return base + base[-1] + "ing"
    return base + "ing"


def forms(base: str) -> dict[str, str]:
    """Regular inflections of a verb (used for TARGET verbs, which are all regular)."""
    return {"base": base, "s": _s(base), "past": _past(base), "ing": _ing(base)}


def build_verb_table() -> dict[str, list[str]]:
    """Surface form of a source verb -> list of surface-form replacements with the same inflection."""
    table: dict[str, list[str]] = {}
    for src, targets in VERBS.items():
        sf = forms(src)
        # irregular sources replace the regular past/ing guess where needed
        for kind in ("base", "s", "past", "ing"):
            surface = sf[kind]
            if src in {"buy", "keep", "show", "give", "think", "build", "find", "tell", "say", "choose", "get",
                       "make", "grow", "send", "begin"} and kind == "past":
                continue  # regular "buyed" etc. does not exist
            table.setdefault(surface, [])
            for t in targets:
                alt = forms(t)[kind]
                if alt != surface and alt not in table[surface]:
                    table[surface].append(alt)
    for surface, (src, _) in SOURCE_FORMS.items():
        table.setdefault(surface, [])
        for t in VERBS.get(src, []):
            alt = forms(t)["past"]
            if alt not in table[surface]:
                table[surface].append(alt)
    return table


VERB_FORMS = build_verb_table()

# multi-word rewrites: phrase -> alternatives
PHRASES: dict[str, list[str]] = {
    "in order to": ["to"], "a lot of": ["plenty of", "numerous"], "lots of": ["plenty of", "many"],
    "because of": ["due to", "owing to"], "due to": ["because of"], "as well as": ["along with", "in addition to"],
    "in addition": ["additionally", "moreover", "furthermore"], "for example": ["for instance"],
    "such as": ["like"], "in conclusion": ["to conclude", "in summary", "overall"],
    "on the other hand": ["conversely", "in contrast"], "as a result": ["consequently", "therefore", "thus"],
    "in fact": ["indeed"], "at the same time": ["simultaneously", "meanwhile"], "a number of": ["several", "many"],
    "most of the time": ["usually", "typically"], "all the time": ["constantly", "continually"],
    "right now": ["currently", "at present"], "in the end": ["ultimately", "finally"], "at first": ["initially", "originally"],
    "is able to": ["can"], "are able to": ["can"], "is going to": ["will"], "are going to": ["will"],
    "kind of": ["somewhat"], "sort of": ["somewhat"], "a few": ["several", "some"], "get rid of": ["eliminate", "remove"],
    "find out": ["discover", "learn"], "look at": ["examine", "consider"], "look for": ["seek", "search for"],
    "come up with": ["devise", "propose"], "carry out": ["perform", "conduct"], "point out": ["note", "highlight"],
    "deal with": ["handle", "address"], "make sure": ["ensure", "confirm"], "think about": ["consider", "reflect on"],
    "talk about": ["discuss"], "set up": ["establish", "arrange"], "go up": ["rise", "increase"], "go down": ["fall", "decrease"],
}

WORD_TABLE: dict[str, list[str]] = {**WORDS}
for _surface, _alts in VERB_FORMS.items():
    WORD_TABLE.setdefault(_surface, [])
    for _a in _alts:
        if _a not in WORD_TABLE[_surface]:
            WORD_TABLE[_surface].append(_a)
