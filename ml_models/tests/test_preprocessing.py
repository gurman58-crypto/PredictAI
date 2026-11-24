import json

import pytest

from ml_models.dataset.prepare_dataset import clean_wikitext_line, text_to_tokens
from ml_models.preprocessing.text_processing import (
    EOS,
    EOS_ID,
    PAD,
    UNK,
    UNK_ID,
    TextPreprocessor,
    Vocabulary,
    capitalise_after,
    detokenize,
    is_word_token,
    normalize,
    tokenize,
)


def test_normalize_lowercases_and_maps_typographic_characters():
    assert normalize("It’s  “Great” – NO") == 'it\'s "great" - no'


def test_tokenize_words_and_punctuation():
    assert tokenize("Hello, World!") == ["hello", ",", "world", "!"]


def test_tokenize_numbers_hyphens_and_clitics():
    assert tokenize("It cost 1,000 or 3.5 units") == ["it", "cost", "1,000", "or", "3.5", "units"]
    assert tokenize("a role-playing game") == ["a", "role-playing", "game"]
    assert tokenize("don't") == ["don", "'t"]
    assert tokenize("the dog's bone") == ["the", "dog", "'s", "bone"]
    assert tokenize("3rd place") == ["3rd", "place"]


def test_tokenize_drops_unsupported_symbols_and_handles_empty():
    assert tokenize("a $ b # c") == ["a", "b", "c"]
    assert tokenize("") == []
    assert tokenize("   \n\t ") == []


def test_tokenize_keeps_unk_marker():
    assert tokenize("a <unk> b") == ["a", UNK, "b"]
    assert tokenize("A <UNK> b") == ["a", UNK, "b"]


def test_is_word_token():
    assert is_word_token("hello") and is_word_token("1,000") and is_word_token("'s")
    assert not any(is_word_token(t) for t in (PAD, UNK, EOS, ".", ",", '"'))


def test_vocabulary_specials_first_and_unknown_mapping():
    vocab = Vocabulary.build(["a", "a", "b", "c"], max_size=10, min_freq=1)
    assert vocab.itos[:3] == [PAD, UNK, EOS]
    assert vocab.encode(["a", "zzz"])[1] == UNK_ID
    assert vocab.decode(vocab.encode(["a", "b"])) == ["a", "b"]


def test_vocabulary_min_freq_and_max_size():
    vocab = Vocabulary.build(["a"] * 5 + ["b"] * 3 + ["c"], max_size=5, min_freq=2)
    assert vocab.itos[3:] == ["a", "b"]
    vocab = Vocabulary.build(["a"] * 5 + ["b"] * 3 + ["c"] * 2, max_size=4, min_freq=1)
    assert len(vocab) == 4 and vocab.itos[3] == "a"


def test_vocabulary_save_load_roundtrip_and_version_check(tmp_path):
    vocab = Vocabulary.build(["x", "y", "y"], max_size=10, min_freq=1)
    path = tmp_path / "vocab.json"
    vocab.save(path)
    assert Vocabulary.load(path).itos == vocab.itos
    data = json.loads(path.read_text())
    data["preprocessing_version"] = "0.0"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        Vocabulary.load(path)


def test_prefix_mask_only_matching_words():
    vocab = Vocabulary(["<pad>", "<unk>", "<eos>", "the", "then", "there", ".", "cat"])
    mask = vocab.prefix_mask("the")
    assert [vocab.itos[i] for i in mask.nonzero()[0]] == ["then", "there"]  # "the" itself excluded


@pytest.fixture()
def pre():
    vocab = Vocabulary.build("the cat sat on the mat . hello world".split(), max_size=50, min_freq=1)
    return TextPreprocessor(vocab, max_context=5)


def test_parse_empty_input_falls_back_to_sentence_start(pre):
    for text in ("", "   ", "\n"):
        p = pre.parse(text)
        assert p.context_ids == [EOS_ID] and p.mode == "next_word" and p.prefix is None


def test_parse_next_word_after_space(pre):
    p = pre.parse("the cat ")
    assert p.mode == "next_word" and p.context_tokens == ["the", "cat"]


def test_parse_partial_word_becomes_completion(pre):
    p = pre.parse("the cat sa")
    assert p.mode == "completion" and p.prefix == "sa" and p.context_tokens == ["the", "cat"]
    assert pre.parse("Hel").prefix == "hel"


def test_parse_finished_vocab_word_is_not_a_prefix_when_nothing_longer_exists(pre):
    assert pre.parse("the cat sat").mode == "next_word"        # no vocab word extends "sat"
    assert pre.parse("the cat sat").context_tokens[-1] == "sat"
    assert pre.parse("the cat sa").mode == "completion"          # "sa" is not a word itself


def test_parse_whole_word_flag_for_explicit_continuation():
    vocab = Vocabulary.build("the then there cat".split(), max_size=20, min_freq=1)
    pre = TextPreprocessor(vocab)
    assert pre.parse("a the").mode == "completion"                   # could become "then"/"there"
    assert pre.parse("a the", whole_word=True).mode == "next_word"   # user asked to continue


def test_parse_glued_fragments_are_not_prefixes(pre):
    assert pre.parse("it's").mode == "next_word"
    assert pre.parse("1,0").mode == "next_word"


def test_parse_trailing_punctuation_and_oov_and_truncation(pre):
    p = pre.parse("the cat.")
    assert p.mode == "next_word" and p.context_tokens[-1] == "."
    assert UNK_ID in pre.parse("zzzz qqqq ").context_ids
    assert len(pre.parse("the cat sat on the mat the cat sat on ").context_ids) == 5


def test_detokenize_spacing_rules():
    assert detokenize(["hello", ",", "world", "!"]) == "hello, world!"
    assert detokenize(["don", "'t", "go"]) == "don't go"
    assert detokenize(["she", "said", '"', "hi", '"', "."]) == 'she said "hi".'
    assert detokenize(["a", "(", "b", ")"]) == "a (b)"
    assert detokenize(["a", UNK, "b", EOS]) == "a b"
    assert detokenize([]) == ""


def test_capitalise_after_sentence_boundaries():
    assert capitalise_after("", "hello") == "Hello"
    assert capitalise_after("Done. ", "next one") == "Next one"
    assert capitalise_after("the cat", " sat") == " sat"


def test_training_and_inference_use_the_same_tokenizer():
    """Dataset preparation must produce exactly tokenize(text) + <eos> per paragraph."""
    line = "The cat's tale, in 1,000 words."
    assert text_to_tokens(line) == tokenize(line) + [EOS]


def test_wikitext_cleaning_matches_natural_text():
    wt = "the game 's story did n't end , role @-@ playing and 1 @,@ 000 units"
    assert text_to_tokens(wt) == tokenize("the game's story didn't end, role-playing and 1,000 units") + [EOS]
    assert clean_wikitext_line("a @.@ b") == "a.b"


def test_headings_and_blank_lines_are_skipped():
    raw = " = Title = \n\n The cat sat . \n = = Section = = \n A dog ran . \n"
    toks = text_to_tokens(raw)
    assert toks.count(EOS) == 2 and "title" not in toks
