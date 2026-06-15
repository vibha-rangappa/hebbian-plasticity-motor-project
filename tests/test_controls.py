# tests/test_controls.py
#
# Tests for geometry/controls.py, which provides two control manipulations used to
# check that the geometry results are real and not just an artifact of how the data
# is grouped: condition_shuffle (randomly reassigns which trial belongs to which
# direction condition, without touching the actual spikes) and trial_split_indices
# (splits trials into two balanced, non-overlapping halves for cross-validation-style
# checks). These tests confirm both helpers shuffle/split things correctly without
# corrupting the underlying spike data.

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
    """Shuffling the trial labels should not change how many trials belong to each
    direction, just which trial gets which label. So if we sort both the original and
    shuffled labels, they should match exactly (same counts per direction, just
    reordered)."""
    snap = _snapshot([0, 0, 1, 1, 2, 2])
    rng = np.random.default_rng(0)
    sh = condition_shuffle(snap, rng)
    # Same counts per direction, but the order has (most likely) changed.
    np.testing.assert_array_equal(np.sort(sh['trial_labels']), np.sort(snap['trial_labels']))


def test_condition_shuffle_leaves_spikes_untouched():
    """Shuffling the condition labels should only change the labels, not the actual
    spike times or which neuron each spike belongs to. This checks the spike data comes
    back exactly the same as before the shuffle."""
    snap = _snapshot([0, 1, 2, 3])
    rng = np.random.default_rng(1)
    sh = condition_shuffle(snap, rng)
    np.testing.assert_array_equal(sh['spike_times_ms'], snap['spike_times_ms'])
    np.testing.assert_array_equal(sh['spike_neuron_idx'], snap['spike_neuron_idx'])


def test_trial_split_balanced_and_disjoint():
    """With 5 trials in condition 0 and 5 in condition 1, splitting into two halves
    should: put every trial into exactly one half (no trial left out, none counted
    twice), and give each half at least one trial from both conditions, so neither half
    is missing a whole direction."""
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 5 per condition
    rng = np.random.default_rng(2)
    idx_a, idx_b = trial_split_indices(labels, rng)
    # The two halves don't overlap, and together they cover all 10 trials.
    assert set(idx_a).isdisjoint(set(idx_b))
    assert set(idx_a) | set(idx_b) == set(range(10))
    # Each half contains both conditions (0 and 1), not just one of them.
    for fold in (idx_a, idx_b):
        present = set(labels[fold])
        assert present == {0, 1}


def test_trial_split_is_random():
    """Using two different random seeds (0 and 99) should produce different splits of
    the trials into halves, since the split is meant to be randomized each time."""
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a1, _ = trial_split_indices(labels, np.random.default_rng(0))
    a2, _ = trial_split_indices(labels, np.random.default_rng(99))
    # Different seeds should usually give different partitions.
    assert not np.array_equal(a1, a2)
