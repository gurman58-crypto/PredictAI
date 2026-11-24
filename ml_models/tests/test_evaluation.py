import numpy as np
import pytest

from ml_models import config
from ml_models.evaluation.evaluate import Scorer, eval_ngram, target_blocks
from ml_models.ngram.ngram_model import NGramModel
from ml_models.preprocessing.text_processing import UNK_ID


def test_scorer_perplexity_and_topk_on_hand_computed_case():
    probs = np.array(
        [
            [0.0, 0.0, 0.1, 0.5, 0.2, 0.1, 0.05, 0.05],   # target 3 -> rank 1
            [0.0, 0.0, 0.4, 0.1, 0.1, 0.1, 0.1, 0.2],     # target 7 -> rank 2 (in top-5, not top-1)
            [0.0, 0.0, 0.3, 0.25, 0.2, 0.15, 0.06, 0.04],  # target 7 -> rank 6, outside top-5
        ]
    )
    targets = np.array([3, 7, 7])
    s = Scorer()
    s.add(probs, targets)
    r = s.result()
    assert r["top1_accuracy"] == pytest.approx(100 / 3, abs=0.01)
    assert r["top5_accuracy"] == pytest.approx(200 / 3, abs=0.01)
    expected_ppl = np.exp(-(np.log(0.5) + np.log(0.2) + np.log(0.04)) / 3)
    assert r["perplexity"] == pytest.approx(expected_ppl, rel=1e-3)


def test_unk_targets_count_for_perplexity_but_not_for_accuracy():
    probs = np.full((2, 6), 1 / 6)
    s = Scorer()
    s.add(probs, np.array([UNK_ID, 3]))
    r = s.result()
    assert r["targets_scored"] == 2 and r["targets_in_accuracy"] == 1


def test_unk_and_pad_are_never_ranked():
    probs = np.array([[0.4, 0.4, 0.05, 0.05, 0.05, 0.03, 0.02]])  # pad/unk hold most mass
    s = Scorer()
    s.add(probs, np.array([2]))
    assert s.result()["top1_accuracy"] == 100.0


def test_target_blocks_cover_the_same_positions_for_every_model():
    t0s = target_blocks(1000)
    targets = (t0s[:, None] + np.arange(config.EVAL_BLOCK)).ravel()
    assert targets.min() == config.EVAL_MIN_CONTEXT and targets.max() < 1000
    assert len(set(targets)) == len(targets)


def test_eval_ngram_runs_and_trigram_beats_unigram_on_structured_data():
    rng = np.random.default_rng(0)
    pattern = np.array([3, 4, 5, 6, 7, 2])
    ids = np.tile(pattern, 400).astype(np.int64)
    ids[rng.integers(0, len(ids), 30)] = 8
    model = NGramModel(10, min_count_trigram=1).fit(ids)
    t0s = target_blocks(len(ids))
    tri = eval_ngram(model, ids, t0s, order=3)
    uni = eval_ngram(model, ids, t0s, order=1)
    assert tri["perplexity"] < uni["perplexity"] and tri["top1_accuracy"] > uni["top1_accuracy"]
    assert tri["targets_scored"] == uni["targets_scored"] == len(t0s) * config.EVAL_BLOCK
