import pytest

from ml_models.writing.grammar import check_grammar
from ml_models.writing.spelling import check_spelling, is_known, suggestions


def flagged(text, **kw):
    return {text[i.start:i.end] for i in check_spelling(text, **kw)}


def test_dictionary_flags_unknown_words_and_offers_fixes():
    t = "I recieve the mesage definately."
    got = {text: i for i in check_spelling(t) for text in [t[i.start:i.end]]}
    assert set(got) == {"recieve", "mesage", "definately"}
    assert got["mesage"].replacement == "message" and got["definately"].replacement == "definitely"
    assert got["recieve"].suggestions[0] == "receive"


@pytest.mark.parametrize("text", [
    "The quick brown fox jumps over the lazy dog.",
    "We don't know what they've done, and it's fine.",
    "The colour of the centre was organised well; we travelled and cancelled.",
    "NASA and LSTM models use camelCase and iPhone names.",
    "Antigravity is a name, and so is Delhi.",
    "See https://example.com/path/file.py or mail bob@example.org now.",
    "It costs 5km and 3rd place, 10x faster.",
    "Well-known state-of-the-art results.",
])
def test_correct_text_is_not_flagged(text):
    assert flagged(text) == set(), text


def test_proper_nouns_only_skipped_mid_sentence():
    assert "Ths" in flagged("Ths is wrong.")
    assert flagged("Then Antigravity helped.") == set()


def test_ignore_list_hides_a_word():
    assert flagged("The wrod appears.") == {"wrod"}
    assert flagged("The wrod appears.", ignore={"wrod"}) == set()
    assert not [i for i in check_grammar("The wrod appears.", ignore={"wrod"}) if i.rule == "spelling"]


def test_case_is_preserved_in_suggestions():
    assert suggestions("Teh")[0] == "The" and suggestions("TEH")[0] == "THE"


def test_common_misspelling_list_still_wins_over_dictionary():
    issues = check_grammar("I will do it alot.")
    assert [i.replacement for i in issues if i.rule == "spelling"] == ["a lot"]


def test_context_ranking_uses_the_language_model(tmp_path):
    from ml_models.inference import PredictionEngine
    from ml_models.testing import build_toy_saved_dir

    engine = PredictionEngine(build_toy_saved_dir(tmp_path))
    issues = check_spelling("the history of the citty", engine=engine)
    assert issues and issues[0].suggestions            # works end to end with a loaded model


def test_known_words():
    assert is_known("grammar") and is_known("Don't") and not is_known("grammr")
