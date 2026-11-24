import pytest

from ml_models.inference import PredictionEngine
from ml_models.testing import build_toy_saved_dir
from ml_models.writing.common import sentence_spans
from ml_models.writing.grammar import check_grammar
from ml_models.writing.lexicon import WORD_TABLE, forms
from ml_models.writing.paraphrase import find_slots, paraphrase
from ml_models.writing.style import check_style, readability


@pytest.fixture(scope="module")
def toy_engine(tmp_path_factory):
    return PredictionEngine(build_toy_saved_dir(tmp_path_factory.mktemp("writing")))


def fix_all(text, issues):
    """Apply every automatic fix, last one first so offsets stay valid."""
    for i in sorted(issues, key=lambda i: -i.start):
        if i.replacement is not None:
            text = text[:i.start] + i.replacement + text[i.end:]
    return text


def rules(text):
    return {i.rule for i in check_grammar(text)}


@pytest.mark.parametrize("text,rule", [
    ("This is teh best.", "spelling"),
    ("I saw the the cat.", "repeated_word"),
    ("She ate a apple.", "a_an"),
    ("It was an university.", "a_an"),
    ("He don't know.", "agreement"),
    ("They is late.", "agreement"),
    ("You should of gone.", "common_error"),
    ("It is better then that.", "common_error"),
    ("Their is a problem.", "common_error"),
    ("then i left.", "sentence_capital"),
    ("Yesterday i left.", "pronoun_i"),
    ("Hello ,world.", "space_before_punct"),
    ("Yes,no.", "space_after_comma"),
    ("It ended.The next began.", "space_after_period"),
    ('He said "hello.', "unbalanced_quote"),
])
def test_grammar_rules_fire(text, rule):
    assert rule in rules(text)


@pytest.mark.parametrize("text", [
    "Dr. Smith visited the U.S. office. It was an honest mistake.",
    "I had had enough, and he said that that was fine.",
    "An FBI agent and a NASA team saw a unique one.",
    "We met at 5 p.m. yesterday, i.e. after lunch.",
    "The results are clear (see the table).",
    "Is it a hotel or an hour-long tour?",
])
def test_correct_text_is_not_flagged(text):
    assert [i for i in check_grammar(text) if i.severity == "error"] == [], text


def test_fixes_produce_clean_text():
    t = "i saw teh cat  and a apple ,it was better then expected"
    fixed = fix_all(t, check_grammar(t))
    assert fixed.startswith("I saw the cat") and "an apple" in fixed and "better than" in fixed
    assert not [i for i in check_grammar(fixed) if i.severity == "error"]


def test_issue_offsets_match_original_text():
    t = "She ate a apple and teh pie."
    for i in check_grammar(t):
        assert 0 <= i.start <= i.end <= len(t)
        assert i.to_dict(t)["original"] == t[i.start:i.end]


def test_empty_and_whitespace():
    assert check_grammar("") == [] and check_grammar("   \n ") == []
    assert check_style("")[1] == []


def test_sentence_spans_respect_abbreviations():
    t = "Dr. Smith left. He was tired."
    assert len(sentence_spans(t)) == 2


def test_style_rules():
    t = ("It was really a very good day. The report was written by the team in order to show results. "
         "We went. We saw. We won.")
    stats, issues = check_style(t)
    got = {i.rule for i in issues}
    assert {"filler_word", "passive_voice", "wordy_phrase", "repeated_start"} <= got
    assert stats["sentences"] == 5 and stats["words"] > 10


def test_style_long_sentence_and_tone():
    long_s = " ".join(["word"] * 45) + "."
    assert any(i.rule == "long_sentence" and i.severity == "warning" for i in check_style(long_s)[1])
    _, neutral = check_style("We don't care, it's gonna be ok.")
    _, formal = check_style("We don't care, it's gonna be ok.", tone="formal")
    assert not any(i.rule in ("contraction", "informal_word") for i in neutral)
    assert {"contraction", "informal_word"} <= {i.rule for i in formal}


def test_readability_orders_texts_sensibly():
    easy = readability("The cat sat on the mat. The dog ran.")
    hard = readability("Notwithstanding considerable methodological heterogeneity, epidemiological investigations "
                       "demonstrate statistically significant associations.")
    assert easy["reading_ease"] > hard["reading_ease"] and hard["grade_level"] > easy["grade_level"]
    assert readability("")["words"] == 0


def test_lexicon_inflection():
    assert forms("cease")["ing"] == "ceasing" and forms("try")["past"] == "tried" and forms("stop")["past"] == "stopped"
    assert "assisted" in WORD_TABLE["helped"] and "demonstrates" in WORD_TABLE["shows"] and "discovered" in WORD_TABLE["found"]


def test_find_slots_prefers_phrases_and_keeps_case():
    slots = find_slots("In order to win, it is a big goal.")
    texts = [(s.original, s.options) for s in slots]
    assert ("In order to", ["To"]) in texts
    assert any(s.original == "big" for s in slots)


def test_paraphrase_without_model_still_works_and_is_unscored():
    v, scored = paraphrase("It is a big problem.", None, None, 2, seed=1)
    assert scored is False and v and v[0].text != "It is a big problem."


def test_paraphrase_with_model_changes_wording_not_meaning_words(toy_engine):
    text = "the history of the city was a big problem"
    variants, scored = paraphrase(text, toy_engine, "ngram", 3, seed=2)
    assert scored is True
    for v in variants:
        assert v.text != text and v.changes
        assert all(c["from"] and c["to"] for c in v.changes)


def test_paraphrase_nothing_to_change():
    assert paraphrase("Qwxz plmk.", None, None, 3)[0] == []
