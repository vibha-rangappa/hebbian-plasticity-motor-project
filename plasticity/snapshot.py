# plasticity/snapshot.py

"""
HDF5 read/write for STDP training snapshots (spec section 6).

Schema:
    /network, /weights, /validation    — copied from the circuit baseline via
        copy_baseline_provenance(), so each output file is self-contained
    /training_params                   — attrs written by save_training_params():
        p_cross, seed, tau_plus, tau_minus, A_plus, A_minus, w_max, n_input,
        r_max, t_burn_in
    /snapshots/epoch_{N}/
        W_EE/{data, row, col, shape}   — COO, row=postsynaptic, col=presynaptic
        spike_times_ms                 — float32, ms within trial
        spike_neuron_idx                — int32, 0..N_exc-1 = E, N_exc.. = input
        spike_trial_idx                  — int32, 0..n_test_trials-1
        trial_labels                     — int32, direction index per test trial
    /monitoring/
        epochs, mean_rate_E, mean_w_EE, frac_w_max, mean_cv_isi  — resizable,
        one row appended per save_snapshot() call
"""

import h5py
import numpy as np


_MONITORING_KEYS = ('mean_rate_E', 'mean_w_EE', 'frac_w_max', 'mean_cv_isi')


def save_snapshot(h5_path, epoch, W_EE_coo, spike_data, trial_labels, monitoring_metrics):
    """
    Append one training snapshot to h5_path (created if it doesn't exist).

    W_EE_coo : dict with 'data' (amps), 'row' (postsynaptic idx), 'col'
        (presynaptic idx), 'shape' — same convention as circuit/run_baseline.py's
        save_baseline.
    spike_data : dict with 'spike_times_ms', 'spike_neuron_idx', 'spike_trial_idx'.
    trial_labels : array of direction indices (0..n_directions-1), one per
        test trial.
    monitoring_metrics : dict with the four keys in _MONITORING_KEYS.
    """
    with h5py.File(h5_path, 'a') as f:
        grp = f.create_group(f'snapshots/epoch_{epoch}')

        wgrp = grp.create_group('W_EE')
        wgrp.create_dataset('data', data=np.asarray(W_EE_coo['data'], dtype=np.float32))
        wgrp.create_dataset('row', data=np.asarray(W_EE_coo['row'], dtype=np.int32))
        wgrp.create_dataset('col', data=np.asarray(W_EE_coo['col'], dtype=np.int32))
        wgrp.create_dataset('shape', data=np.asarray(W_EE_coo['shape'], dtype=np.int32))

        grp.create_dataset('spike_times_ms',
                            data=np.asarray(spike_data['spike_times_ms'], dtype=np.float32))
        grp.create_dataset('spike_neuron_idx',
                            data=np.asarray(spike_data['spike_neuron_idx'], dtype=np.int32))
        grp.create_dataset('spike_trial_idx',
                            data=np.asarray(spike_data['spike_trial_idx'], dtype=np.int32))
        grp.create_dataset('trial_labels',
                            data=np.asarray(trial_labels, dtype=np.int32))

        _append_monitoring_row(f, epoch, monitoring_metrics)


def _append_monitoring_row(f, epoch, metrics):
    if 'monitoring' not in f:
        mgrp = f.create_group('monitoring')
        mgrp.create_dataset('epochs', data=np.array([epoch], dtype=np.int32),
                             maxshape=(None,))
        for k in _MONITORING_KEYS:
            mgrp.create_dataset(k, data=np.array([metrics[k]], dtype=np.float64),
                                 maxshape=(None,))
        return

    mgrp = f['monitoring']
    n = mgrp['epochs'].shape[0]
    mgrp['epochs'].resize((n + 1,))
    mgrp['epochs'][n] = epoch
    for k in _MONITORING_KEYS:
        mgrp[k].resize((n + 1,))
        mgrp[k][n] = metrics[k]


def load_snapshot(h5_path, epoch):
    """Load one snapshot. Raises KeyError if /snapshots/epoch_{epoch} doesn't exist."""
    with h5py.File(h5_path, 'r') as f:
        grp = f[f'snapshots/epoch_{epoch}']
        return {
            'W_EE_coo': {
                'data':  grp['W_EE/data'][:],
                'row':   grp['W_EE/row'][:],
                'col':   grp['W_EE/col'][:],
                'shape': grp['W_EE/shape'][:],
            },
            'spike_times_ms':   grp['spike_times_ms'][:],
            'spike_neuron_idx': grp['spike_neuron_idx'][:],
            'spike_trial_idx':  grp['spike_trial_idx'][:],
            'trial_labels':     grp['trial_labels'][:],
        }


def load_monitoring(h5_path):
    """Load /monitoring/ as a dict of numpy arrays, keyed by dataset name."""
    with h5py.File(h5_path, 'r') as f:
        mgrp = f['monitoring']
        return {k: mgrp[k][:] for k in mgrp.keys()}


def copy_baseline_provenance(h5_path, baseline_h5_path):
    """
    Copy /network, /weights, /validation from the circuit baseline into
    h5_path, so each training output file is self-contained (spec section 6).
    Call once per file, before any snapshots are saved.
    """
    with h5py.File(baseline_h5_path, 'r') as src, h5py.File(h5_path, 'a') as dst:
        for group_name in ('network', 'weights', 'validation'):
            src.copy(src[group_name], dst, group_name)


def save_training_params(h5_path, params, p_cross, seed):
    """
    Write /training_params attrs (spec section 6): p_cross, STDP params, task
    input params, the trial-sequence seed, and burn-in duration.
    """
    with h5py.File(h5_path, 'a') as f:
        grp = f.require_group('training_params')
        grp.attrs['p_cross'] = float(p_cross)
        grp.attrs['seed'] = int(seed)
        for k in ('tau_plus', 'tau_minus', 'A_plus', 'A_minus', 'w_max',
                  'n_input', 'r_max', 't_burn_in'):
            grp.attrs[k] = float(params[k])
