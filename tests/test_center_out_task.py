# tests/test_task.py

import numpy as np
import pytest

from plasticity.center_out_task import (
    assign_preferred_directions,
    rates_for_phase,
    generate_trial_sequence,
    generate_test_trial_sequence,
)


def test_assign_preferred_directions_length_and_unique_count():
    theta_i = assign_preferred_directions(n_input=50, n_directions=8)
    assert len(theta_i) == 50
    assert len(np.unique(theta_i)) == 8


def test_assign_preferred_directions_counts_balanced():
    theta_i = assign_preferred_directions(n_input=50, n_directions=8)
    _, counts = np.unique(theta_i, return_counts=True)
    assert sorted(counts.tolist()) == [6, 6, 6, 6, 6, 6, 7, 7]


def test_assign_preferred_directions_spacing_is_45_degrees():
    theta_i = assign_preferred_directions(n_input=50, n_directions=8)
    directions = np.unique(theta_i)
    spacing = np.diff(directions)
    np.testing.assert_allclose(spacing, np.pi / 4)


def test_rates_for_phase_prep_peak_and_orthogonal():
    theta_i = np.array([0.0, np.pi / 2, np.pi])
    rates = rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='prep')
    np.testing.assert_allclose(rates, [100.0, 0.0, 0.0], atol=1e-10)


def test_rates_for_phase_exec_amplifies_prep_by_1_5():
    theta_i = np.array([0.0, np.pi / 4])
    prep = rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='prep')
    exec_rates = rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='exec')
    np.testing.assert_allclose(exec_rates, prep * 1.5)


def test_rates_for_phase_iti_is_background_regardless_of_cue():
    theta_i = np.array([0.0, np.pi / 2, np.pi])
    rates = rates_for_phase(theta_cue=1.23, theta_i=theta_i, phase='iti')
    np.testing.assert_allclose(rates, [2.0, 2.0, 2.0])


def test_rates_for_phase_invalid_phase_raises():
    theta_i = np.array([0.0])
    with pytest.raises(ValueError):
        rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='bogus')


def test_generate_trial_sequence_reproducible_and_balanced():
    seq1 = generate_trial_sequence(n_per_direction=13, n_directions=8, seed=42)
    seq2 = generate_trial_sequence(n_per_direction=13, n_directions=8, seed=42)
    assert len(seq1) == 13 * 8
    np.testing.assert_array_equal(seq1, seq2)
    _, counts = np.unique(seq1, return_counts=True)
    np.testing.assert_array_equal(counts, np.full(8, 13))


def test_generate_trial_sequence_different_seeds_give_different_order():
    seq1 = generate_trial_sequence(n_per_direction=13, n_directions=8, seed=42)
    seq2 = generate_trial_sequence(n_per_direction=13, n_directions=8, seed=1)
    assert not np.array_equal(seq1, seq2)


def test_generate_test_trial_sequence_default_length_and_balance():
    seq = generate_test_trial_sequence()
    assert len(seq) == 40
    _, counts = np.unique(seq, return_counts=True)
    np.testing.assert_array_equal(counts, np.full(8, 5))
