import pytest

from ml_models.inference import ModelUnavailableError, PredictionEngine, UnknownModelError
from ml_models.preprocessing.text_processing import EOS, PAD, UNK
from ml_models.testing import build_toy_saved_dir


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    path = build_toy_saved_dir(tmp_path_factory.mktemp("toy"))
    return PredictionEngine(path, enable_lstm=True)


def test_status_reports_trained_and_missing_models(engine):
    statuses = {s.id: s for s in engine.statuses()}
    assert statuses["ngram"].status == "ready" and statuses["ngram"].version
    assert statuses["lstm"].status == "unavailable" and "not trained" in statuses["lstm"].reason
    assert statuses["transformer"].status == "unavailable" and "not trained" in statuses["transformer"].reason


def test_predict_returns_ranked_words_only(engine):
    res = engine.predict("The cat sat on the ", "ngram", top_k=5)
    words = [w for w, _ in res.predictions]
    scores = [s for _, s in res.predictions]
    assert 0 < len(words) <= 5
    assert scores == sorted(scores, reverse=True) and all(0 < s <= 1 for s in scores)
    assert not {PAD, UNK, EOS, ".", ","} & set(words)
    assert res.mode == "next_word" and res.model == "ngram" and res.latency_ms >= 0
    assert words[0] in {"mat", "log"}


def test_top_k_is_respected_and_deterministic(engine):
    a = engine.predict("the dog ", "ngram", top_k=3)
    b = engine.predict("the dog ", "ngram", top_k=3)
    assert len(a.predictions) == 3 and a.predictions == b.predictions


def test_empty_and_unknown_input_fall_back_gracefully(engine):
    for text in ("", "   ", "zzzz qqqq ", "!!! ???", "日本語 "):
        res = engine.predict(text, "ngram")
        assert res.predictions, f"no fallback for {text!r}"


def test_partial_word_completion_filters_by_prefix(engine):
    res = engine.predict("the cat sa", "ngram", top_k=5)
    assert res.mode == "completion" and res.prefix == "sa"
    words = [w for w, _ in res.predictions]
    assert words and all(w.startswith("sa") and w != "sa" for w in words)


def test_completion_prefix_with_no_matches_returns_empty_list(engine):
    assert engine.predict("the xq", "ngram").predictions == []


def test_user_text_is_not_modified_by_prediction(engine):
    text = "  The   CAT sat "
    engine.predict(text, "ngram")
    assert text == "  The   CAT sat "


def test_unknown_and_unavailable_models_raise_clean_errors(engine):
    with pytest.raises(UnknownModelError):
        engine.predict("hello", "gpt")
    with pytest.raises(ModelUnavailableError):
        engine.predict("hello", "lstm")


def test_complete_is_bounded_seeded_and_well_formed(engine):
    kwargs = dict(model_id="ngram", max_tokens=8, seed=7)
    a = engine.complete("The cat sat", **kwargs)
    b = engine.complete("The cat sat", **kwargs)
    assert a.continuation == b.continuation and a.tokens == b.tokens
    assert 0 < len(a.tokens) <= 8
    assert a.continuation.startswith(" ")          # separates from "sat"
    assert a.stopped_because in {"max_tokens", "sentence_end", "end_of_text", "no_candidates"}


def test_complete_stops_at_sentence_end(engine):
    res = engine.complete("The cat sat on the", "ngram", max_tokens=30, stop_at_sentence=True, seed=1)
    assert res.stopped_because in {"sentence_end", "end_of_text", "max_tokens"}
    if res.stopped_because == "sentence_end":
        assert res.tokens[-1] in {".", "!", "?"}


def test_complete_inside_a_word_only_adds_the_remainder(engine):
    res = engine.complete("the cat sa", "ngram", max_tokens=3, seed=3)
    assert res.continuation and not res.continuation.startswith(" ")
    assert "sa" + res.continuation.split(" ")[0] in engine.vocab.stoi   # prefix + remainder = a real word


def test_finished_word_without_trailing_space_is_not_treated_as_a_prefix(engine):
    # "sat" is a complete vocabulary word with no longer completions -> next-word mode
    res = engine.predict("the cat sat", "ngram")
    assert res.mode == "next_word" and res.context_tokens[-1] == "sat"
    assert engine.predict("the cat sa", "ngram").mode == "completion"


def test_complete_after_sentence_end_capitalises(engine):
    res = engine.complete("The cat sat. ", "ngram", max_tokens=3, seed=2)
    assert res.continuation.strip()[0].isupper()


def test_missing_artifacts_do_not_crash_startup(tmp_path):
    engine = PredictionEngine(tmp_path)
    assert not engine.any_ready()
    assert all(s.status == "unavailable" and s.reason for s in engine.statuses())
    with pytest.raises(ModelUnavailableError):
        engine.predict("hello", "ngram")


def test_corrupt_ngram_file_is_reported_not_raised(tmp_path):
    build_toy_saved_dir(tmp_path)
    (tmp_path / "ngram" / "ngram_model.npz").write_bytes(b"not a real file")
    engine = PredictionEngine(tmp_path)
    status = {s.id: s for s in engine.statuses()}["ngram"]
    assert status.status == "unavailable" and "failed to load" in status.reason


def test_lstm_can_be_disabled_by_configuration(tmp_path):
    build_toy_saved_dir(tmp_path)
    engine = PredictionEngine(tmp_path, enable_lstm=False)
    assert "disabled" in {s.id: s for s in engine.statuses()}["lstm"].reason


def test_continuation_capitalises_new_sentences_inside_the_text(engine):
    from ml_models.inference import PredictionEngine as PE

    parsed = engine.preprocessor.parse("x ", whole_word=True)
    out = PE._join_continuation("The cat sat", parsed, ["on", "it", ".", "the", "dog", "ran", "!", "and", "so"])
    assert out == " On it. The dog ran! And so" or out == " on it. The dog ran! And so"


def test_generated_text_never_contains_quotes_or_brackets(engine):
    for seed in range(25):
        res = engine.complete('She said "hello" and (then', "ngram", max_tokens=25, stop_at_sentence=False, seed=seed)
        assert not {'"', "(", ")"} & set(res.tokens)
