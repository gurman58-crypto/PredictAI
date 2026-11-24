import numpy as np
import pytest

from ml_models.ngram.ngram_model import NGramModel
from ml_models.preprocessing.text_processing import PAD_ID, UNK_ID, Vocabulary, tokenize

CORPUS = ("the cat sat on the mat . the cat sat on the log . the dog sat on the mat . "
          "a cat saw a dog . the cat ate fish .").split()


@pytest.fixture(scope="module")
def setup():
    vocab = Vocabulary.build(CORPUS, max_size=50, min_freq=1)
    ids = np.array(vocab.encode(CORPUS), dtype=np.int64)
    model = NGramModel(len(vocab), min_count_trigram=1, discount=0.75).fit(ids)
    return vocab, ids, model


def top_word(vocab, probs):
    p = probs.copy()
    p[:3] = 0
    return vocab.itos[int(p.argmax())]


def test_distributions_sum_to_one_for_every_order_and_context(setup):
    vocab, ids, model = setup
    contexts = [[], [vocab.stoi["the"]], [vocab.stoi["the"], vocab.stoi["cat"]],
                [UNK_ID, UNK_ID], [vocab.stoi["fish"], vocab.stoi["the"]]]
    for order in (1, 2, 3):
        for ctx in contexts:
            p = model.next_token_probs(ctx, order=order)
            assert p.shape == (len(vocab),)
            assert p.sum() == pytest.approx(1.0, abs=1e-9)
            assert (p >= 0).all() and p[PAD_ID] == 0


def test_trigram_uses_the_two_previous_words(setup):
    vocab, _, model = setup
    s = vocab.stoi
    assert top_word(vocab, model.next_token_probs([s["cat"], s["sat"]])) == "on"
    assert top_word(vocab, model.next_token_probs([s["sat"], s["on"]])) == "the"


def test_bigram_and_unigram_orders(setup):
    vocab, _, model = setup
    s = vocab.stoi
    # bigram "sat" -> "on"; unigram ignores the context entirely
    assert top_word(vocab, model.next_token_probs([s["sat"]], order=2)) == "on"
    assert np.allclose(model.next_token_probs([s["sat"]], order=1), model.next_token_probs([s["dog"]], order=1))
    assert model.next_token_probs([s["sat"]], order=1).argmax() == model.p_uni.argmax()


def test_backoff_when_trigram_context_is_unseen(setup):
    vocab, _, model = setup
    s = vocab.stoi
    ctx = [s["fish"], s["sat"]]  # ("fish","sat") never occurs -> falls back to bigram ("sat")
    assert np.allclose(model.next_token_probs(ctx, order=3), model.next_token_probs([s["sat"]], order=2))


def test_unknown_words_and_empty_context_still_give_a_distribution(setup):
    vocab, _, model = setup
    p = model.next_token_probs([UNK_ID, UNK_ID])
    assert p.sum() == pytest.approx(1.0) and p.max() > 0
    assert np.allclose(model.next_token_probs([]), model.p_uni)


def test_smoothing_gives_unseen_continuations_nonzero_probability(setup):
    vocab, _, model = setup
    s = vocab.stoi
    p = model.next_token_probs([s["the"], s["cat"]])
    assert p[s["fish"]] > 0  # "the cat fish" is unseen but reachable through backoff


def test_save_load_roundtrip(tmp_path, setup):
    vocab, _, model = setup
    path = tmp_path / "m.npz"
    model.save(path)
    loaded = NGramModel.load(path)
    ctx = [vocab.stoi["the"], vocab.stoi["cat"]]
    assert np.allclose(model.next_token_probs(ctx), loaded.next_token_probs(ctx))
    assert loaded.discount == model.discount and loaded.vocab_size == model.vocab_size


def test_pruning_keeps_distributions_normalised(setup):
    vocab, ids, _ = setup
    pruned = NGramModel(len(vocab), min_count_trigram=2).fit(ids)
    for ctx in ([vocab.stoi["the"], vocab.stoi["cat"]], [vocab.stoi["sat"], vocab.stoi["on"]]):
        assert pruned.next_token_probs(ctx).sum() == pytest.approx(1.0)
    assert len(pruned.trigram) < len(setup[2].trigram)


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        NGramModel(10, order=4)
    with pytest.raises(ValueError):
        NGramModel(10, discount=1.5)
    with pytest.raises(ValueError):
        NGramModel(10).fit(np.array([1, 2]))
    with pytest.raises(ValueError):
        NGramModel(5).fit(np.array([1, 2, 3, 9]))


def test_better_than_unigram_on_training_like_text(setup):
    vocab, ids, model = setup
    def ppl(order):
        nll = sum(-np.log(model.next_token_probs(ids[max(0, t - 2):t], order=order)[ids[t]]) for t in range(2, len(ids)))
        return np.exp(nll / (len(ids) - 2))
    assert ppl(3) < ppl(2) < ppl(1)
