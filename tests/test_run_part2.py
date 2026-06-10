# tests/test_run_part2.py

import numpy as np
import pytest
from brian2 import start_scope, second, amp, Hz

from part1.network import build_network, DEFAULT_PARAMS
from part2.network_part2 import DEFAULT_PARAMS_PART2, build_stdp_network
from part2.task import assign_preferred_directions, generate_test_trial_sequence
from part2.run_part2 import (
    run_one_trial,
    extract_snapshot_spikes,
    compute_monitoring_metrics,
    check_abort_criteria,
    run_snapshot,
)
from part2.snapshot import load_snapshot, load_monitoring


def _small_setup(nu_ext=1000.0, seed=1):
    small = {
        **DEFAULT_PARAMS, **DEFAULT_PARAMS_PART2,
        'N_exc': 20, 'N_inh': 5,
        'P_size': 8, 'X_size': 8,
        'nu_ext': nu_ext,
    }
    net_objs = build_network(small, seed=seed)
    net_objs = build_stdp_network(net_objs, small, p_cross=1.0, seed=seed)
    theta_i = assign_preferred_directions(small['n_input'], small['n_directions'])
    return net_objs, small, theta_i


def test_run_one_trial_advances_time_by_trial_duration():
    start_scope()
    net_objs, small, theta_i = _small_setup()
    t_before = net_objs['net'].t / second
    run_one_trial(net_objs, small, theta_i, theta_cue=0.0)
    t_after = net_objs['net'].t / second
    expected_dur = small['t_prep'] + small['t_exec'] + small['t_iti']
    assert t_after - t_before == pytest.approx(expected_dur)


def test_run_one_trial_leaves_input_at_background_rate():
    start_scope()
    net_objs, small, theta_i = _small_setup()
    run_one_trial(net_objs, small, theta_i, theta_cue=0.0)
    rates = np.array(net_objs['input_group'].rates[:] / Hz)
    np.testing.assert_allclose(rates, small['r_background'])


def test_extract_snapshot_spikes_keys_and_ranges():
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
    metrics = {'mean_rate_E': 35.0, 'mean_w_EE': 0.0, 'frac_w_max': 0.0, 'mean_cv_isi': 1.0}
    with pytest.raises(RuntimeError, match="mean_rate_E"):
        check_abort_criteria(metrics, epoch=100)


def test_check_abort_criteria_raises_on_high_frac_w_max():
    metrics = {'mean_rate_E': 5.0, 'mean_w_EE': 0.0, 'frac_w_max': 0.6, 'mean_cv_isi': 1.0}
    with pytest.raises(RuntimeError, match="frac_w_max"):
        check_abort_criteria(metrics, epoch=100)


def test_check_abort_criteria_passes_normal_metrics():
    metrics = {'mean_rate_E': 5.0, 'mean_w_EE': 0.06e-9, 'frac_w_max': 0.05, 'mean_cv_isi': 0.9}
    check_abort_criteria(metrics, epoch=100)  # should not raise


def test_run_snapshot_writes_hdf5_and_restores_plastic(tmp_path):
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
