# tests/test_controls.py

import numpy as np

from geometry.controls import condition_shuffle, trial_split_indices


def _snapshot(labels):
    n = len(labels)
    return {
        'spike_times_ms': np.arange(n, dtype=np.float32),
        'spike_neuron_idx': np.zeros(n, dtype=np.int32),
        'spike_trial_idx': np.arange(n, dtype=np.int32),
        'trial_labels': np.array(labels, dtype=np.int32),
    }


def test_condition_shuffle_preserves_label_multiset():
    snap = _snapshot([0, 0, 1, 1, 2, 2])
    rng = np.random.default_rng(0)
    sh = condition_shuffle(snap, rng)
    # Same counts per direction, but order changed (with high probability).
    np.testing.assert_array_equal(np.sort(sh['trial_labels']), np.sort(snap['trial_labels']))


def test_condition_shuffle_leaves_spikes_untouched():
    snap = _snapshot([0, 1, 2, 3])
    rng = np.random.default_rng(1)
    sh = condition_shuffle(snap, rng)
    np.testing.assert_array_equal(sh['spike_times_ms'], snap['spike_times_ms'])
    np.testing.assert_array_equal(sh['spike_neuron_idx'], snap['spike_neuron_idx'])


def test_trial_split_balanced_and_disjoint():
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 5 per condition
    rng = np.random.default_rng(2)
    idx_a, idx_b = trial_split_indices(labels, rng)
    # Disjoint and covering.
    assert set(idx_a).isdisjoint(set(idx_b))
    assert set(idx_a) | set(idx_b) == set(range(10))
    # Balanced per condition: each fold has both conditions represented.
    for fold in (idx_a, idx_b):
        present = set(labels[fold])
        assert present == {0, 1}


def test_trial_split_is_random():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a1, _ = trial_split_indices(labels, np.random.default_rng(0))
    a2, _ = trial_split_indices(labels, np.random.default_rng(99))
    # Different seeds should usually give different partitions.
    assert not np.array_equal(a1, a2)
