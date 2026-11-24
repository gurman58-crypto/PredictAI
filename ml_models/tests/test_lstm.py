import numpy as np
import pytest

pytest.importorskip("tensorflow")

from ml_models.inference import PredictionEngine  # noqa: E402
from ml_models.lstm.lstm_model import LSTMModel, build_lstm  # noqa: E402
from ml_models.preprocessing.text_processing import PAD_ID  # noqa: E402
from ml_models.testing import build_toy_saved_dir  # noqa: E402


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    return build_toy_saved_dir(tmp_path_factory.mktemp("toy_lstm"), with_lstm=True)


def test_architecture_is_embedding_lstm_dense_softmax():
    model = build_lstm(50, embed_dim=8, lstm_units=8, dropout=0.1)
    kinds = [type(layer).__name__ for layer in model.layers]
    assert kinds == ["InputLayer", "Embedding", "LSTM", "Dropout", "Dense"]
    out = model.predict(np.zeros((2, 7), dtype="int32"), verbose=0)
    assert out.shape == (2, 7, 50) and np.allclose(out.sum(-1), 1.0, atol=1e-4)


def test_saved_model_loads_and_returns_a_valid_distribution(saved):
    lstm = LSTMModel.load(saved / "lstm" / "lstm_model.keras", saved / "lstm" / "lstm_meta.json")
    for ctx in ([2], [3, 4, 5], list(range(3, 40))):          # lengths 1, 3 and > max_context
        p = lstm.next_token_probs(ctx)
        assert p.shape == (lstm.vocab_size,)
        assert p.sum() == pytest.approx(1.0) and p[PAD_ID] == 0 and (p >= 0).all()
    assert lstm.max_context == 10 and lstm.version == "toy"


def test_empty_context_is_handled(saved):
    lstm = LSTMModel.load(saved / "lstm" / "lstm_model.keras")
    assert lstm.next_token_probs([]).sum() == pytest.approx(1.0)


def test_engine_serves_the_lstm_with_same_api_as_ngram(saved):
    engine = PredictionEngine(saved)
    assert engine.is_ready("lstm") and engine.is_ready("ngram")
    res = engine.predict("the cat sat on the ", "lstm", top_k=5)
    assert res.model == "lstm" and res.model_version == "toy"
    assert 0 < len(res.predictions) <= 5
    comp = engine.complete("the cat", "lstm", max_tokens=5, seed=1)
    assert comp.tokens and comp.model == "lstm"


def test_vocab_size_mismatch_is_reported(saved, tmp_path):
    import shutil

    shutil.copytree(saved, tmp_path / "x")
    from ml_models.preprocessing.text_processing import Vocabulary

    Vocabulary(["<pad>", "<unk>", "<eos>", "only"]).save(tmp_path / "x" / "vocab.json")
    status = {s.id: s for s in PredictionEngine(tmp_path / "x").statuses()}
    assert status["lstm"].status == "unavailable" and "mismatch" in status["lstm"].reason
