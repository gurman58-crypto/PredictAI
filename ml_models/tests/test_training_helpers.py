import numpy as np
import pytest

pytest.importorskip("tensorflow")

from ml_models.training.keras_common import WindowSequence  # noqa: E402


def test_windows_are_shifted_by_one_and_inside_the_corpus():
    ids = np.arange(1000, dtype=np.int32)
    seq = WindowSequence(ids, seq_len=10, batch_size=8, shuffle=False)
    x, y = seq[0]
    assert x.shape == (8, 10) and y.shape == (8, 10)
    assert np.array_equal(y, x + 1)                  # target = next token
    seen = np.concatenate([seq[i][0].ravel() for i in range(len(seq))])
    assert seen.max() + 1 < len(ids)                 # the last target never runs past the end


def test_every_training_window_is_visited_exactly_once_per_epoch():
    ids = np.arange(5001, dtype=np.int32)
    seq = WindowSequence(ids, seq_len=10, batch_size=7, shuffle=True, seed=1)
    firsts = np.concatenate([seq[i][0][:, 0] for i in range(len(seq))])
    assert len(set(firsts.tolist())) == len(firsts)


def test_shuffling_changes_order_and_offset_between_epochs_but_validation_is_stable():
    ids = np.arange(5001, dtype=np.int32)
    train = WindowSequence(ids, 10, 16, True, seed=3)
    before = train.starts.copy()
    train.on_epoch_end()
    assert not np.array_equal(before, train.starts)
    valid = WindowSequence(ids, 10, 16, False)
    first = valid.starts.copy()
    valid.on_epoch_end()
    assert np.array_equal(first, valid.starts) and np.all(np.diff(first) > 0)
