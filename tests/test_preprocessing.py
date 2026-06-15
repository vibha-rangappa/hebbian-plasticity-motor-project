# tests/test_preprocessing.py
#
# Tests for geometry/preprocessing.py, which turns raw spike data (a "snapshot" of
# spike times, neuron indices, and trial/condition labels) into the smoothed,
# trial-averaged, mean-subtracted firing-rate arrays (X_prep, X_exec) used by the rest
# of the geometry analyses. These tests check the output shapes, that condition
# averages get properly mean-subtracted (centered), that non-excitatory "input"
# neurons are excluded, that a known firing rate comes out at roughly the right value
# after smoothing, and that make_X with an explicit trial subset matches a manual,
# hand-computed calculation.

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
    """Basic shape check: with 4 trials split into 2 conditions and 10 excitatory
    neurons, the prep and exec output arrays should both come out as (10 neurons, 50
    time bins, 2 conditions), and n_conditions should be reported as 2. One neuron is
    given some spikes so the rates aren't all trivially zero."""
    trains = [{0: [510.0, 530.0]} for _ in range(4)]
    snap = build_snapshot(trains, [0, 0, 1, 1])
    out = preprocess_snapshot(snap, n_exc=10)
    assert out['X_prep'].shape == (10, 50, 2)
    assert out['X_exec'].shape == (10, 50, 2)
    assert out['n_conditions'] == 2


def test_cross_condition_mean_is_zero():
    """The preprocessing subtracts the across-condition average from each neuron's
    response (mean-subtraction / centering). After this, if you add up the activity
    across all conditions for any given neuron and time bin, it should come out to
    ~0. This checks that centering was actually applied correctly to both X_exec and
    X_prep."""
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
    """Only the first n_exc neurons are real excitatory network neurons; anything with
    a higher index is an "input" neuron whose firing pattern is imposed by the task
    and would artificially inject direction tuning if included. Here neuron 0 is a
    real E neuron and neuron 20 is an input neuron; with n_exc=10, only neurons 0-9
    should be kept, so the output should have exactly 10 neurons."""
    trains = [{0: [520.0], 20: [520.0] * 50} for _ in range(2)]
    snap = build_snapshot(trains, [0, 1])
    tr = compute_trial_rates(snap, n_exc=10)
    assert tr['trial_rate_exec'].shape[1] == 10  # only 10 neurons retained


def test_rate_magnitude_regular_firing():
    """If a neuron fires regularly at 50 Hz throughout the exec window, the smoothed
    firing-rate estimate should come out close to 50 Hz, at least in a time bin well
    inside the window (away from the edges, where smoothing can distort the
    estimate)."""
    spikes = list(np.arange(510.0, 990.0, 20.0))  # spikes every 20 ms = 50 Hz, across the exec window
    trains = [{0: spikes} for _ in range(2)]
    snap = build_snapshot(trains, [0, 1])
    tr = compute_trial_rates(snap, n_exc=4)
    # Look at a bin in the middle of the window (trial 0, neuron 0, bin 25): should be ~50 Hz.
    interior = tr['trial_rate_exec'][0, 0, 25]
    assert interior == pytest.approx(50.0, abs=5.0)


def test_identical_across_conditions_centers_to_zero():
    """If a neuron fires exactly the same way in every condition, it carries no
    direction-specific information, so after centering (subtracting the
    across-condition mean) its contribution should be ~0 everywhere. Here neuron 0
    fires identically in all 4 trials, regardless of which condition (0 or 1) the
    trial belongs to, so its condition averages are identical and centering should
    zero it out."""
    spikes = [520.0, 540.0, 560.0]
    # Same spikes in every trial => identical averages per condition => centers to ~0.
    trains = [{0: spikes, 1: [100.0 * (i + 1)]} for i in range(4)]
    snap = build_snapshot(trains, [0, 0, 1, 1])
    out = preprocess_snapshot(snap, n_exc=4)
    np.testing.assert_allclose(out['X_exec'][0], 0.0, atol=1e-9)


def test_make_X_subset_matches_manual():
    """make_X lets you build the condition-averaged, normalized, centered array using
    only a chosen subset of trials (here trials 0-3 out of 6). This test recomputes
    the same thing by hand, step by step (average trials per condition, normalize by
    the firing-rate range R plus a floor of 5.0, then subtract the across-condition
    mean), and checks that make_X's output matches the hand-computed version exactly.
    Condition 0 is the average of trials 0 and 2; condition 1 is the average of trials
    1 and 3."""
    rng = np.random.default_rng(1)
    n_trials, N, T, C = 6, 5, 4, 2
    trial_rates = rng.uniform(1, 10, (n_trials, N, T))
    labels = np.array([0, 1, 0, 1, 0, 1])
    R = trial_rates.reshape(n_trials, N, T).max(axis=(0, 2)) - \
        trial_rates.min(axis=(0, 2))
    idx = np.array([0, 1, 2, 3])
    X = make_X(trial_rates, labels, R, r_floor=5.0, n_conditions=C, idx=idx)
    # Manual version: condition 0 = mean of trials 0 and 2; condition 1 = mean of trials 1 and 3.
    cond0 = trial_rates[[0, 2]].mean(axis=0)
    cond1 = trial_rates[[1, 3]].mean(axis=0)
    norm0 = cond0 / (R[:, None] + 5.0)
    norm1 = cond1 / (R[:, None] + 5.0)
    mean_t = (norm0 + norm1) / 2.0
    np.testing.assert_allclose(X[:, :, 0], norm0 - mean_t, atol=1e-9)
    np.testing.assert_allclose(X[:, :, 1], norm1 - mean_t, atol=1e-9)
