# tests/test_task.py
#
# Tests for plasticity/center_out_task.py, which sets up the "center-out reaching"
# task: it gives each input neuron a preferred reach direction, works out what firing
# rate each input neuron should have during the prep / exec / inter-trial phases of a
# trial, and builds the sequence of trial directions used for training and testing.
# These tests check the direction assignments are even and correctly spaced, the
# firing rates for each task phase come out as expected, and the trial sequences are
# reproducible and balanced across directions.

import numpy as np
import pytest

from plasticity.center_out_task import (
    assign_preferred_directions,
    rates_for_phase,
    generate_trial_sequence,
    generate_test_trial_sequence,
)


def test_assign_preferred_directions_length_and_unique_count():
    """With 50 input neurons split across 8 directions, every neuron should get a
    preferred direction, and there should be exactly 8 distinct directions in total."""
    theta_i = assign_preferred_directions(n_input=50, n_directions=8)
    assert len(theta_i) == 50
    assert len(np.unique(theta_i)) == 8


def test_assign_preferred_directions_counts_balanced():
    """50 neurons does not divide evenly by 8 directions, so the split should be as
    even as possible: six directions get 6 neurons and two directions get 7 (6*6 + 2*7
    = 50), instead of dumping the leftovers onto just one direction."""
    theta_i = assign_preferred_directions(n_input=50, n_directions=8)
    _, counts = np.unique(theta_i, return_counts=True)
    assert sorted(counts.tolist()) == [6, 6, 6, 6, 6, 6, 7, 7]


def test_assign_preferred_directions_spacing_is_45_degrees():
    """The 8 preferred directions should be evenly spaced around the circle, 45 degrees
    (pi/4 radians) apart from each other, like the 8 spokes of the center-out task."""
    theta_i = assign_preferred_directions(n_input=50, n_directions=8)
    directions = np.unique(theta_i)
    spacing = np.diff(directions)
    np.testing.assert_allclose(spacing, np.pi / 4)


def test_rates_for_phase_prep_peak_and_orthogonal():
    """During the prep phase, a neuron whose preferred direction matches the cue
    direction (both 0.0) should fire at its peak rate of 100 Hz. Neurons tuned 90 or
    180 degrees away from the cue should fire at 0 Hz, since their preferred direction
    is orthogonal or opposite to the cued direction."""
    theta_i = np.array([0.0, np.pi / 2, np.pi])
    rates = rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='prep')
    np.testing.assert_allclose(rates, [100.0, 0.0, 0.0], atol=1e-10)


def test_rates_for_phase_exec_amplifies_prep_by_1_5():
    """The exec phase should just scale up the prep-phase rates by a factor of 1.5,
    representing the extra drive once the movement actually starts."""
    theta_i = np.array([0.0, np.pi / 4])
    prep = rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='prep')
    exec_rates = rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='exec')
    np.testing.assert_allclose(exec_rates, prep * 1.5)


def test_rates_for_phase_exec_sustained_is_default():
    """If exec_mode isn't specified, it should default to 'sustained', which keeps the
    1.5x amplified prep-phase rates. This checks that the default behavior matches
    explicitly asking for 'sustained'."""
    theta_i = np.array([0.0, np.pi / 4])
    prep = rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='prep')
    exec_default = rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='exec')
    exec_sustained = rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='exec',
                                     exec_mode='sustained')
    np.testing.assert_allclose(exec_default, prep * 1.5)
    np.testing.assert_allclose(exec_sustained, prep * 1.5)


def test_rates_for_phase_exec_autonomous_is_background():
    """In 'autonomous' exec mode, the task input is removed entirely and every input
    neuron just falls back to a flat background rate (here 2.0 Hz for all of them),
    regardless of its preferred direction. This represents letting the network drive
    itself instead of being clamped to the cued direction."""
    theta_i = np.array([0.0, np.pi / 4, np.pi])
    rates = rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='exec',
                            r_background=2.0, exec_mode='autonomous')
    np.testing.assert_allclose(rates, [2.0, 2.0, 2.0])


def test_rates_for_phase_iti_is_background_regardless_of_cue():
    """During the inter-trial interval (iti), all neurons should just sit at the
    background rate, no matter what the cue direction was. The cue here is set to an
    arbitrary value (1.23 radians) to confirm it has no effect during iti."""
    theta_i = np.array([0.0, np.pi / 2, np.pi])
    rates = rates_for_phase(theta_cue=1.23, theta_i=theta_i, phase='iti')
    np.testing.assert_allclose(rates, [2.0, 2.0, 2.0])


def test_rates_for_phase_invalid_phase_raises():
    """Asking for a phase name that doesn't exist (here 'bogus') should raise a
    ValueError rather than silently returning something wrong."""
    theta_i = np.array([0.0])
    with pytest.raises(ValueError):
        rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='bogus')


def test_generate_trial_sequence_reproducible_and_balanced():
    """With 13 trials per direction and 8 directions, the sequence should have 13*8
    trials total, with exactly 13 trials of each direction. Using the same seed (42)
    twice should give the exact same sequence both times, so training runs are
    reproducible."""
    seq1 = generate_trial_sequence(n_per_direction=13, n_directions=8, seed=42)
    seq2 = generate_trial_sequence(n_per_direction=13, n_directions=8, seed=42)
    assert len(seq1) == 13 * 8
    np.testing.assert_array_equal(seq1, seq2)
    _, counts = np.unique(seq1, return_counts=True)
    np.testing.assert_array_equal(counts, np.full(8, 13))


def test_generate_trial_sequence_different_seeds_give_different_order():
    """Different random seeds (42 vs 1) should shuffle the trial order differently, so
    the sequences should not be identical."""
    seq1 = generate_trial_sequence(n_per_direction=13, n_directions=8, seed=42)
    seq2 = generate_trial_sequence(n_per_direction=13, n_directions=8, seed=1)
    assert not np.array_equal(seq1, seq2)


def test_generate_test_trial_sequence_default_length_and_balance():
    """The default test sequence should have 40 trials total, split evenly as 5 trials
    per direction across 8 directions (5*8 = 40)."""
    seq = generate_test_trial_sequence()
    assert len(seq) == 40
    _, counts = np.unique(seq, return_counts=True)
    np.testing.assert_array_equal(counts, np.full(8, 5))
