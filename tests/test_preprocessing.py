# tests/test_preprocessing.py

import numpy as np
import pytest

from geometry.preprocessing import (
    compute_trial_rates,
    preprocess_snapshot,
    condition_mean,
    make_X,
)


def build_snapshot(trains, trial_labels):
    """
    trains: list over trials; each entry a dict {neuron_idx: array_of_times_ms}.
    Returns a snapshot dict in the same shape as plasticity.snapshot.load_snapshot.
    """
    times, neurons, trials = [], [], []
    for tr, d in enumerate(trains):
        for nidx, ts in d.items():
            for t in ts:
                times.append(t); neurons.append(nidx); trials.append(tr)
    return {
        'spike_times_ms': np.array(times, dtype=np.float32),
        'spike_neuron_idx': np.array(neurons, dtype=np.int32),
        'spike_trial_idx': np.array(trials, dtype=np.int32),
        'trial_labels': np.array(trial_labels, dtype=np.int32),
    }


def test_preprocess_shapes():
    # 4 trials, 2 conditions, 10 E neurons. One neuron spikes so rates aren't all zero.
    trains = [{0: [510.0, 530.0]} for _ in range(4)]
    snap = build_snapshot(trains, [0, 0, 1, 1])
    out = preprocess_snapshot(snap, n_exc=10)
    assert out['X_prep'].shape == (10, 50, 2)
    assert out['X_exec'].shape == (10, 50, 2)
    assert out['n_conditions'] == 2


def test_cross_condition_mean_is_zero():
    """After mean-subtraction, summing X over conditions gives ~0 at every (neuron, t)."""
    rng = np.random.default_rng(0)
    trains = []
    for _ in range(6):
        d = {n: np.sort(rng.uniform(0, 1000, rng.integers(5, 20))) for n in range(8)}
        trains.append(d)
    snap = build_snapshot(trains, [0, 1, 2, 0, 1, 2])
    out = preprocess_snapshot(snap, n_exc=8)
    np.testing.assert_allclose(out['X_exec'].sum(axis=2), 0.0, atol=1e-9)
    np.testing.assert_allclose(out['X_prep'].sum(axis=2), 0.0, atol=1e-9)


def test_input_neurons_excluded():
    """Spikes from neurons >= n_exc must not appear (they'd inject imposed tuning)."""
    # neuron 0 is an E neuron; neuron 20 is an "input" neuron, excluded with n_exc=10.
    trains = [{0: [520.0], 20: [520.0] * 50} for _ in range(2)]
    snap = build_snapshot(trains, [0, 1])
    tr = compute_trial_rates(snap, n_exc=10)
    assert tr['trial_rate_exec'].shape[1] == 10  # only 10 neurons retained


def test_rate_magnitude_regular_firing():
    """A neuron firing regularly at 50 Hz -> ~50 Hz smoothed rate in the window interior."""
    spikes = list(np.arange(510.0, 990.0, 20.0))  # 50 Hz across exec window
    trains = [{0: spikes} for _ in range(2)]
    snap = build_snapshot(trains, [0, 1])
    tr = compute_trial_rates(snap, n_exc=4)
    # Interior bin (well inside the window, away from edges): ~50 Hz.
    interior = tr['trial_rate_exec'][0, 0, 25]  # trial 0, neuron 0, mid bin
    assert interior == pytest.approx(50.0, abs=5.0)


def test_identical_across_conditions_centers_to_zero():
    """A neuron with the same response in every condition contributes nothing after centering."""
    spikes = [520.0, 540.0, 560.0]
    # Same spikes in all trials => identical condition averages => centered to ~0.
    trains = [{0: spikes, 1: [100.0 * (i + 1)]} for i in range(4)]
    snap = build_snapshot(trains, [0, 0, 1, 1])
    out = preprocess_snapshot(snap, n_exc=4)
    np.testing.assert_allclose(out['X_exec'][0], 0.0, atol=1e-9)


def test_make_X_subset_matches_manual():
    """make_X with an explicit trial subset reproduces a hand-computed condition average."""
    rng = np.random.default_rng(1)
    n_trials, N, T, C = 6, 5, 4, 2
    trial_rates = rng.uniform(1, 10, (n_trials, N, T))
    labels = np.array([0, 1, 0, 1, 0, 1])
    R = trial_rates.reshape(n_trials, N, T).max(axis=(0, 2)) - \
        trial_rates.min(axis=(0, 2))
    idx = np.array([0, 1, 2, 3])
    X = make_X(trial_rates, labels, R, r_floor=5.0, n_conditions=C, idx=idx)
    # Manual: condition 0 = mean of trials 0,2; condition 1 = mean of trials 1,3.
    cond0 = trial_rates[[0, 2]].mean(axis=0)
    cond1 = trial_rates[[1, 3]].mean(axis=0)
    norm0 = cond0 / (R[:, None] + 5.0)
    norm1 = cond1 / (R[:, None] + 5.0)
    mean_t = (norm0 + norm1) / 2.0
    np.testing.assert_allclose(X[:, :, 0], norm0 - mean_t, atol=1e-9)
    np.testing.assert_allclose(X[:, :, 1], norm1 - mean_t, atol=1e-9)
