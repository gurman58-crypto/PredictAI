import numpy as np
import pytest

pytest.importorskip("tensorflow")

from ml_models.ensemble.ensemble_model import EnsembleModel, fit_weights  # noqa: E402
from ml_models.inference import PredictionEngine  # noqa: E402
from ml_models.testing import build_toy_saved_dir  # noqa: E402


class Fixed:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)

    def next_token_probs(self, ctx):
        return self.p


def test_blend_is_a_weighted_valid_distribution():
    m = EnsembleModel({"a": Fixed([1, 0, 0]), "b": Fixed([0, 0.5, 0.5])}, {"a": 0.25, "b": 0.75}, 3)
    p = m.next_token_probs([1])
    assert np.allclose(p, [0.25, 0.375, 0.375]) and abs(p.sum() - 1) < 1e-9


def test_weights_are_renormalised_over_available_members():
    m = EnsembleModel({"a": Fixed([1, 0]), "b": Fixed([0, 1])}, {"a": 0.2, "b": 0.2, "c": 0.6}, 2)
    assert m.weights == {"a": 0.5, "b": 0.5}


def test_needs_two_members():
    with pytest.raises(ValueError):
        EnsembleModel({"a": Fixed([1, 0])}, {"a": 1.0}, 2)


def test_em_prefers_the_better_model():
    rng = np.random.default_rng(0)
    good = rng.uniform(0.5, 0.9, 2000)
    bad = rng.uniform(0.01, 0.05, 2000)
    w = fit_weights(np.stack([good, bad], axis=1))
    assert w[0] > 0.9 and abs(w.sum() - 1) < 1e-9


def test_engine_serves_the_blend(tmp_path):
    saved = build_toy_saved_dir(tmp_path, with_transformer=True, with_ensemble=True)
    engine = PredictionEngine(saved)
    assert engine.is_ready("ensemble")
    res = engine.predict("the history of the", "ensemble", 5)
    assert res.model == "ensemble" and len(res.predictions) == 5
    assert engine.complete("the history of", "ensemble", max_tokens=5, seed=1).tokens


def test_blend_unavailable_without_tuning_file(tmp_path):
    engine = PredictionEngine(build_toy_saved_dir(tmp_path, with_transformer=True))
    st = {s.id: s for s in engine.statuses()}
    assert st["ensemble"].status == "unavailable" and "train_ensemble.py" in st["ensemble"].reason
