# tests/test_train.py
#
# Tests for plasticity/train.py: the code that runs the center-out reaching
# task trial by trial, takes "snapshots" of the network's spikes/weights at
# given epochs, checks for runaway activity (abort criteria), and saves
# everything to an HDF5 file. The checks here cover timing (does a trial take
# the right amount of simulated time), what gets recorded in a snapshot,
# whether plasticity is correctly frozen during test trials but active during
# training, and whether the synaptic-scaling (weight normalization) keeps
# each neuron's total incoming E->E weight from growing past its target.

import numpy as np
import pytest
from brian2 import start_scope, second, amp, Hz

from circuit.network import build_network, DEFAULT_PARAMS
from plasticity.stdp_network import DEFAULT_PARAMS_PLASTICITY, build_stdp_network
from plasticity.center_out_task import (
    assign_preferred_directions,
    generate_trial_sequence,
    generate_test_trial_sequence,
)
from plasticity.train import (
    run_one_trial,
    extract_snapshot_spikes,
    compute_monitoring_metrics,
    check_abort_criteria,
    run_snapshot,
    run_condition,
)
from plasticity.snapshot import load_snapshot, load_monitoring


def _small_setup(nu_ext=1000.0, seed=1):
    small = {
        **DEFAULT_PARAMS, **DEFAULT_PARAMS_PLASTICITY,
        'N_exc': 20, 'N_inh': 5,
        'P_size': 8, 'X_size': 8,
        'nu_ext': nu_ext,
    }
    net_objs = build_network(small, seed=seed)
    net_objs = build_stdp_network(net_objs, small, p_cross=1.0, seed=seed)
    theta_i = assign_preferred_directions(small['n_input'], small['n_directions'])
    return net_objs, small, theta_i


def test_run_one_trial_advances_time_by_trial_duration():
    # One trial has three phases: preparation (t_prep), movement execution
    # (t_exec), and an inter-trial interval (t_iti). After running one
    # trial, the simulation clock should have moved forward by exactly the
    # sum of those three durations.
    start_scope()
    net_objs, small, theta_i = _small_setup()
    t_before = net_objs['net'].t / second
    run_one_trial(net_objs, small, theta_i, theta_cue=0.0)
    t_after = net_objs['net'].t / second
    expected_dur = small['t_prep'] + small['t_exec'] + small['t_iti']
    assert t_after - t_before == pytest.approx(expected_dur)


def test_run_one_trial_leaves_input_at_background_rate():
    # After a trial finishes, the input neurons' firing rates should be back
    # at the background rate, not stuck at whatever rate was used during the
    # task cue.
    start_scope()
    net_objs, small, theta_i = _small_setup()
    run_one_trial(net_objs, small, theta_i, theta_cue=0.0)
    rates = np.array(net_objs['input_group'].rates[:] / Hz)
    np.testing.assert_allclose(rates, small['r_background'])


def test_extract_snapshot_spikes_keys_and_ranges():
    # Run two trials (for reaching directions 0 and 1), then ask
    # extract_snapshot_spikes() to pull out the spikes recorded during that
    # window. Check that:
    #  - the returned dictionary has exactly the three expected arrays,
    #  - those arrays all have the same length (one entry per spike),
    #  - at least one spike was recorded (nu_ext=1000 is high enough to
    #    guarantee spiking),
    #  - spike times fall within a single trial's duration (0 to trial_dur),
    #  - the trial index for each spike is a valid trial number (0 or 1),
    #  - the neuron index for each spike is a valid neuron, including both
    #    the excitatory population (N_exc) and the input population (n_input).
    start_scope()
    net_objs, small, theta_i = _small_setup()
    t_snapshot_start = net_objs['net'].t / second
    n_test_trials = 2
    for d in (0, 1):
        theta_cue = 2 * np.pi * d / small['n_directions']
        run_one_trial(net_objs, small, theta_i, theta_cue)

    spikes = extract_snapshot_spikes(net_objs, t_snapshot_start, small, n_test_trials)
    assert set(spikes.keys()) == {'spike_times_ms', 'spike_neuron_idx', 'spike_trial_idx'}

    n = len(spikes['spike_times_ms'])
    assert len(spikes['spike_neuron_idx']) == n
    assert len(spikes['spike_trial_idx']) == n
    assert n > 0  # nu_ext=1000 guarantees spikes

    trial_dur_ms = (small['t_prep'] + small['t_exec'] + small['t_iti']) * 1000.0
    assert np.all(spikes['spike_times_ms'] >= 0.0)
    assert np.all(spikes['spike_times_ms'] < trial_dur_ms + 1e-6)
    assert np.all(spikes['spike_trial_idx'] >= 0)
    assert np.all(spikes['spike_trial_idx'] < n_test_trials)
    assert np.all(spikes['spike_neuron_idx'] >= 0)
    assert np.all(spikes['spike_neuron_idx'] < small['N_exc'] + small['n_input'])


def test_compute_monitoring_metrics_keys_and_ranges():
    # After running two trials, compute_monitoring_metrics() should return a
    # dictionary with exactly these four summary numbers: average E firing
    # rate, average E->E weight, fraction of weights near w_max, and average
    # spike-timing irregularity (CV of inter-spike intervals). Check that
    # each value is in a sensible range, and that the reported mean E->E
    # weight actually matches the mean of the current weights.
    start_scope()
    net_objs, small, theta_i = _small_setup()
    t_snapshot_start = net_objs['net'].t / second
    n_test_trials = 2
    for d in (0, 1):
        theta_cue = 2 * np.pi * d / small['n_directions']
        run_one_trial(net_objs, small, theta_i, theta_cue)

    metrics = compute_monitoring_metrics(net_objs, t_snapshot_start, small, n_test_trials)
    assert set(metrics.keys()) == {'mean_rate_E', 'mean_w_EE', 'frac_w_max', 'mean_cv_isi'}
    assert metrics['mean_rate_E'] >= 0.0
    assert 0.0 <= metrics['frac_w_max'] <= 1.0
    w = np.array(net_objs['syn_EE'].w[:] / amp)
    assert metrics['mean_w_EE'] == pytest.approx(np.mean(w))


def test_check_abort_criteria_raises_on_high_rate():
    # If the mean E firing rate is way too high (35 Hz here), this is a sign
    # the network is in a runaway/unstable state, so check_abort_criteria()
    # should raise an error mentioning "mean_rate_E" to flag what went wrong.
    metrics = {'mean_rate_E': 35.0, 'mean_w_EE': 0.0, 'frac_w_max': 0.0, 'mean_cv_isi': 1.0}
    with pytest.raises(RuntimeError, match="mean_rate_E"):
        check_abort_criteria(metrics, epoch=100)


def test_check_abort_criteria_raises_on_high_frac_w_max():
    # If too large a fraction of E->E weights are sitting at w_max (60% here),
    # that's a sign of runaway potentiation, so check_abort_criteria() should
    # raise an error mentioning "frac_w_max".
    metrics = {'mean_rate_E': 5.0, 'mean_w_EE': 0.0, 'frac_w_max': 0.6, 'mean_cv_isi': 1.0}
    with pytest.raises(RuntimeError, match="frac_w_max"):
        check_abort_criteria(metrics, epoch=100)


def test_check_abort_criteria_passes_normal_metrics():
    # With all metrics in a normal, healthy range, check_abort_criteria()
    # should do nothing and not raise any error.
    metrics = {'mean_rate_E': 5.0, 'mean_w_EE': 0.06e-9, 'frac_w_max': 0.05, 'mean_cv_isi': 0.9}
    check_abort_criteria(metrics, epoch=100)  # should not raise


def test_run_snapshot_writes_hdf5_and_restores_plastic(tmp_path):
    # run_snapshot() runs a set of "test" trials (no learning should happen
    # during these) and saves the network state to an HDF5 file. After it's
    # done, check that:
    #  - the E->E synapses are still marked as plastic (plastic == 1), i.e.
    #    run_snapshot() turns plasticity back on afterwards if it turned it
    #    off during the snapshot,
    #  - the saved weight matrix (W_EE_coo, a sparse "coordinate format"
    #    matrix) has one entry per E->E synapse,
    #  - the saved trial labels match the test trial sequence we asked for,
    #  - the monitoring log records that epoch 0 was run.
    start_scope()
    net_objs, small, theta_i = _small_setup()
    h5_path = str(tmp_path / "test.h5")
    test_trial_sequence = generate_test_trial_sequence(
        n_per_direction=1, n_directions=small['n_directions'])

    run_snapshot(net_objs, h5_path, epoch=0,
                  test_trial_sequence=test_trial_sequence,
                  theta_i=theta_i, params=small, check_abort=False)

    syn = net_objs['syn_EE']
    assert np.all(np.array(syn.plastic[:]) == 1)

    snap = load_snapshot(h5_path, epoch=0)
    assert snap['W_EE_coo']['data'].shape[0] == len(syn)
    np.testing.assert_array_equal(snap['trial_labels'], test_trial_sequence)

    mon = load_monitoring(h5_path)
    np.testing.assert_array_equal(mon['epochs'], [0])


def test_run_snapshot_freezes_weights_during_test_trials(tmp_path):
    # Test trials are meant to probe the network's current behavior, not to
    # train it further. So while run_snapshot() is running its test trials,
    # the E->E weights should not change at all: they should come out
    # identical to how they started.
    start_scope()
    net_objs, small, theta_i = _small_setup()
    h5_path = str(tmp_path / "test.h5")
    syn = net_objs['syn_EE']
    w_before = np.array(syn.w[:] / amp).copy()

    test_trial_sequence = generate_test_trial_sequence(
        n_per_direction=1, n_directions=small['n_directions'])
    run_snapshot(net_objs, h5_path, epoch=0,
                  test_trial_sequence=test_trial_sequence,
                  theta_i=theta_i, params=small, check_abort=False)

    w_after = np.array(syn.w[:] / amp)
    np.testing.assert_allclose(w_before, w_after)


def test_run_condition_small_network_writes_snapshots(tmp_path):
    # run_condition() runs a full training condition (burn-in plus training
    # trials) and takes snapshots at the requested epochs (here epochs 0 and
    # 8). Check that the monitoring log records both epochs, that both
    # snapshots have a weight matrix of the same shape (same number of E->E
    # synapses, since connectivity doesn't change), and that the trial
    # labels in the first snapshot match what we asked for.
    start_scope()
    net_objs, small, theta_i = _small_setup()
    small = {**small, 't_burn_in': 0.1}  # keep the test fast

    h5_path = str(tmp_path / "test_condition.h5")
    short_test_sequence = generate_test_trial_sequence(
        n_per_direction=1, n_directions=small['n_directions'])

    run_condition(net_objs, small, h5_path, theta_i,
                   n_per_direction=1, snapshot_epochs={0, 8},
                   seed=1, condition_name='test',
                   check_abort=False, test_trial_sequence=short_test_sequence)

    mon = load_monitoring(h5_path)
    np.testing.assert_array_equal(mon['epochs'], [0, 8])

    snap0 = load_snapshot(h5_path, epoch=0)
    snap8 = load_snapshot(h5_path, epoch=8)
    assert snap0['W_EE_coo']['data'].shape == snap8['W_EE_coo']['data'].shape
    np.testing.assert_array_equal(snap0['trial_labels'], short_test_sequence)


def test_run_condition_frozen_leaves_weights_unchanged(tmp_path):
    """This is the "frozen" control condition (plasticity_on=False): even
    though we drive the network hard (nu_ext=1000, lots of spiking, which
    would normally change weights via STDP), the E->E weights should be
    exactly identical between epoch 0 and epoch 8, because plasticity is
    turned off for this run."""
    start_scope()
    net_objs, small, theta_i = _small_setup()
    small = {**small, 't_burn_in': 0.1}
    h5_path = str(tmp_path / "frozen.h5")
    seq = generate_test_trial_sequence(n_per_direction=1, n_directions=small['n_directions'])

    run_condition(net_objs, small, h5_path, theta_i, n_per_direction=1,
                  snapshot_epochs={0, 8}, seed=1, condition_name='frozen',
                  check_abort=False, test_trial_sequence=seq, plasticity_on=False)

    snap0 = load_snapshot(h5_path, epoch=0)
    snap8 = load_snapshot(h5_path, epoch=8)
    np.testing.assert_array_equal(snap0['W_EE_coo']['data'], snap8['W_EE_coo']['data'])


def test_run_condition_plastic_changes_weights(tmp_path):
    """This is the opposite of the frozen control above: with plasticity_on=True,
    STDP is active during training, so the E->E weights at epoch 8 should be
    different from epoch 0, i.e. the weights actually moved during training."""
    start_scope()
    net_objs, small, theta_i = _small_setup()
    small = {**small, 't_burn_in': 0.1}
    h5_path = str(tmp_path / "plastic.h5")
    seq = generate_test_trial_sequence(n_per_direction=1, n_directions=small['n_directions'])

    run_condition(net_objs, small, h5_path, theta_i, n_per_direction=1,
                  snapshot_epochs={0, 8}, seed=1, condition_name='plastic',
                  check_abort=False, test_trial_sequence=seq, plasticity_on=True)

    snap0 = load_snapshot(h5_path, epoch=0)
    snap8 = load_snapshot(h5_path, epoch=8)
    assert not np.allclose(snap0['W_EE_coo']['data'], snap8['W_EE_coo']['data'], atol=0)


def test_weight_norm_prevents_insum_inflation(tmp_path):
    """This checks the homeostatic synaptic scaling (weight_norm=True), which
    is meant to stop runaway potentiation. After training, for every
    excitatory neuron, the total of its incoming E->E weights ("insum")
    should not exceed the target value it started with (its baseline insum).
    In other words, synaptic scaling can hold a neuron's total input steady
    or bring it down, but it should never let it grow past the target."""
    start_scope()
    net_objs, small, theta_i = _small_setup()
    small = {**small, 't_burn_in': 0.1}
    h5_path = str(tmp_path / "norm.h5")
    seq = generate_test_trial_sequence(n_per_direction=1, n_directions=small['n_directions'])

    run_condition(net_objs, small, h5_path, theta_i, n_per_direction=1,
                  snapshot_epochs=set(), seed=1, condition_name='norm',
                  check_abort=False, test_trial_sequence=seq,
                  plasticity_on=True, weight_norm=True)

    syn = net_objs['syn_EE']
    post = np.asarray(syn.j[:])
    w = np.asarray(syn.w[:] / amp)
    insum = np.bincount(post, weights=w, minlength=small['N_exc'])
    target = net_objs['W_target_EE']
    has = target > 0
    # The "scale then clip" approach can only keep the sum the same or lower
    # it, never push it above the target.
    assert np.all(insum[has] <= target[has] * (1.0 + 1e-6))
    # And the sum should stay reasonably close to the target, not collapse
    # down toward zero.
    assert np.median(insum[has] / target[has]) > 0.8


def test_build_stdp_network_exposes_weight_target():
    # build_stdp_network() should record, for each excitatory neuron, the
    # starting total of its incoming E->E weights, as 'W_target_EE'. This is
    # the target that synaptic scaling tries to maintain later on. Check that
    # this target really does equal the initial per-neuron weight sum.
    start_scope()
    net_objs, small, theta_i = _small_setup()
    assert 'W_target_EE' in net_objs
    syn = net_objs['syn_EE']
    post = np.asarray(syn.j[:])
    w0 = np.asarray(syn.w[:] / amp)
    insum0 = np.bincount(post, weights=w0, minlength=small['N_exc'])
    np.testing.assert_allclose(net_objs['W_target_EE'], insum0, rtol=1e-6)


def test_run_condition_runs_correct_number_of_training_trials():
    # run_condition() should advance the simulation clock by exactly
    # t_burn_in plus (number of trials) x (trial duration). With
    # n_per_direction=1, the number of trials equals n_directions (one trial
    # per reaching direction).
    start_scope()
    net_objs, small, theta_i = _small_setup()
    small = {**small, 't_burn_in': 0.1}
    short_test_sequence = generate_test_trial_sequence(
        n_per_direction=1, n_directions=small['n_directions'])

    t_before = net_objs['net'].t / second

    import tempfile, os as _os
    fd, h5_path = tempfile.mkstemp(suffix='.h5')
    _os.close(fd)
    _os.remove(h5_path)
    try:
        run_condition(net_objs, small, h5_path, theta_i,
                       n_per_direction=1, snapshot_epochs=set(),
                       seed=1, condition_name='test',
                       check_abort=False, test_trial_sequence=short_test_sequence)
    finally:
        if _os.path.exists(h5_path):
            _os.remove(h5_path)

    t_after = net_objs['net'].t / second
    trial_dur = small['t_prep'] + small['t_exec'] + small['t_iti']
    n_trials = small['n_directions'] * 1  # n_per_direction=1
    expected = small['t_burn_in'] + n_trials * trial_dur
    assert (t_after - t_before) == pytest.approx(expected)
