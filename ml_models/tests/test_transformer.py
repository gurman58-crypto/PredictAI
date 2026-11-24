import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from ml_models.inference import PredictionEngine  # noqa: E402
from ml_models.preprocessing.text_processing import PAD_ID  # noqa: E402
from ml_models.testing import build_toy_saved_dir  # noqa: E402
from ml_models.transformer.transformer_model import TransformerModel, build_transformer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    return build_toy_saved_dir(tmp_path_factory.mktemp("toy_transformer"), with_transformer=True)


def small(**kw):
    args = dict(vocab_size=50, max_len=12, d_model=16, num_heads=2, num_layers=2, ff_dim=32, dropout=0.0)
    args.update(kw)
    return build_transformer(**args)


def test_architecture_has_attention_positions_and_softmax_head():
    model = small()
    kinds = [type(layer).__name__ for layer in model.layers]
    assert kinds.count("MultiHeadAttention") == 2 and kinds.count("PositionalEmbedding") == 1
    assert kinds[1] == "Embedding" and kinds[-1] == "Dense" and "LayerNormalization" in kinds
    out = model.predict(np.zeros((3, 7), dtype="int32"), verbose=0)
    assert out.shape == (3, 7, 50) and np.allclose(out.sum(-1), 1.0, atol=1e-4)


def test_attention_is_causal_future_tokens_cannot_change_earlier_predictions():
    model = small()
    x = np.random.default_rng(0).integers(3, 50, size=(2, 10)).astype("int32")
    changed = x.copy()
    changed[:, 6] = (changed[:, 6] + 1 - 3) % 47 + 3
    a, b = model.predict(x, verbose=0), model.predict(changed, verbose=0)
    assert np.allclose(a[:, :6], b[:, :6], atol=1e-6)         # positions before the change: identical
    assert not np.allclose(a[:, 6:], b[:, 6:])                # the change is visible from its own position on


def test_position_matters_same_tokens_in_another_order_give_different_output():
    model = small()
    x = np.array([[5, 6, 7, 8]], dtype="int32")
    assert not np.allclose(model.predict(x, verbose=0)[:, -1], model.predict(x[:, ::-1], verbose=0)[:, -1])


def test_invalid_head_configuration_is_rejected():
    with pytest.raises(ValueError):
        build_transformer(50, d_model=10, num_heads=4)


def test_saved_model_loads_and_returns_a_valid_distribution(saved):
    model = TransformerModel.load(saved / "transformer" / "transformer_model.keras", saved / "transformer" / "transformer_meta.json")
    assert model.max_context == 10 and model.version == "toy" and model.model_id == "transformer"
    for ctx in ([2], [3, 4, 5], [3 + i % 50 for i in range(60)]):   # lengths 1, 3 and far beyond the position table
        p = model.next_token_probs(ctx)
        assert p.shape == (model.vocab_size,) and p.sum() == pytest.approx(1.0) and p[PAD_ID] == 0 and (p >= 0).all()
    assert model.next_token_probs([]).sum() == pytest.approx(1.0)


def test_loading_works_in_a_fresh_python_process(saved):
    """The custom layer must register itself on load - no import done by the caller is relied upon."""
    code = (
        "import sys; from ml_models.transformer.transformer_model import TransformerModel; "
        "m = TransformerModel.load(sys.argv[1]); p = m.next_token_probs([3, 4, 5]); print(round(float(p.sum()), 4))"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT), "TF_CPP_MIN_LOG_LEVEL": "3"}
    res = subprocess.run([sys.executable, "-c", code, str(saved / "transformer" / "transformer_model.keras")],
                         capture_output=True, text=True, env=env, cwd=ROOT, timeout=240)
    assert res.returncode == 0, res.stderr[-800:]
    assert res.stdout.strip().splitlines()[-1] == "1.0"


def test_engine_serves_the_transformer_like_any_other_model(saved):
    engine = PredictionEngine(saved)
    status = {s.id: s for s in engine.statuses()}
    assert status["transformer"].status == "ready" and status["transformer"].version == "toy"
    assert status["lstm"].status == "unavailable" and "train_lstm" in status["lstm"].reason
    res = engine.predict("the cat sat on the ", "transformer", top_k=5)
    assert res.model == "transformer" and 0 < len(res.predictions) <= 5
    assert [w for w, _ in res.predictions] == [w for w, _ in sorted(res.predictions, key=lambda t: -t[1])]
    assert engine.predict("the cat sa", "transformer").mode == "completion"
    comp = engine.complete("the cat", "transformer", max_tokens=5, seed=1)
    assert comp.model == "transformer" and comp.tokens


def test_transformer_can_be_disabled_independently(saved):
    status = {s.id: s for s in PredictionEngine(saved, enable_transformer=False).statuses()}
    assert status["transformer"].status == "unavailable" and "ENABLE_TRANSFORMER=false" in status["transformer"].reason
    assert status["ngram"].status == "ready"


def test_missing_transformer_names_the_training_script(tmp_path):
    build_toy_saved_dir(tmp_path)
    status = {s.id: s for s in PredictionEngine(tmp_path).statuses()}["transformer"]
    assert status.status == "unavailable" and "train_transformer.py" in status.reason


def test_vocab_mismatch_is_reported(saved, tmp_path):
    import shutil

    from ml_models.preprocessing.text_processing import Vocabulary

    shutil.copytree(saved, tmp_path / "x")
    Vocabulary(["<pad>", "<unk>", "<eos>", "only"]).save(tmp_path / "x" / "vocab.json")
    status = {s.id: s for s in PredictionEngine(tmp_path / "x").statuses()}["transformer"]
    assert status.status == "unavailable" and "mismatch" in status.reason
