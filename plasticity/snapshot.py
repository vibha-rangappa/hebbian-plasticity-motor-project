# plasticity/snapshot.py

"""
This file handles reading and writing the HDF5 files that store STDP
training snapshots: the connection weights, spike data, and summary
statistics taken at various points during a training run.

Layout of the HDF5 file:
    /network, /weights, /validation  : copied over from the circuit baseline
        by copy_baseline_provenance(), so each output file is self-contained
        and you can tell what baseline network it came from.
    /training_params  : attributes (metadata) written by save_training_params():
        p_cross, seed, tau_plus, tau_minus, A_plus, A_minus, w_max, n_input,
        r_max, t_burn_in, and more.
    /snapshots/epoch_{N}/  : one group per snapshot, where N is the epoch
        (trial count) the snapshot was taken at.
        W_EE/{data, row, col, shape}  : the E->E weight matrix in sparse
            "COO" format (a list of nonzero entries). row = postsynaptic
            neuron, col = presynaptic neuron.
        spike_times_ms  : float32 array, spike time in milliseconds,
            measured from the start of that spike's trial.
        spike_neuron_idx  : int32 array, which neuron fired. Indices
            0..N_exc-1 are excitatory neurons, N_exc and above are input
            neurons.
        spike_trial_idx  : int32 array, which test trial (0..n_test_trials-1)
            each spike happened in.
        trial_labels  : int32 array, the direction index for each test trial.
    /monitoring/  : a set of arrays (epochs, mean_rate_E, mean_w_EE,
        frac_w_max, mean_cv_isi) that grow over time, one new row added
        each time save_snapshot() is called.
"""

import h5py
import numpy as np


_MONITORING_KEYS = ('mean_rate_E', 'mean_w_EE', 'frac_w_max', 'mean_cv_isi')


def save_snapshot(h5_path, epoch, W_EE_coo, spike_data, trial_labels, monitoring_metrics):
    """
    Add one new training snapshot to h5_path (the file is created first if it
    doesn't exist yet).

    W_EE_coo : dict with 'data' (weights, in amps), 'row' (postsynaptic
        neuron index), 'col' (presynaptic neuron index), and 'shape'. This
        is the same format used by circuit/run_baseline.py's save_baseline.
    spike_data : dict with 'spike_times_ms', 'spike_neuron_idx',
        'spike_trial_idx'.
    trial_labels : array of direction indices (0..n_directions-1), one per
        test trial.
    monitoring_metrics : dict containing the four keys listed in
        _MONITORING_KEYS.
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
    """Load the /monitoring/ group as a dict of numpy arrays, keyed by dataset name."""
    with h5py.File(h5_path, 'r') as f:
        mgrp = f['monitoring']
        return {k: mgrp[k][:] for k in mgrp.keys()}


def copy_baseline_provenance(h5_path, baseline_h5_path):
    """
    Copy the /network, /weights, and /validation groups from the circuit
    baseline file into h5_path, so each training output file is
    self-contained (you can tell what baseline it started from just by
    looking at this one file). Call this once per file, before any snapshots
    are saved.
    """
    with h5py.File(baseline_h5_path, 'r') as src, h5py.File(h5_path, 'a') as dst:
        for group_name in ('network', 'weights', 'validation'):
            src.copy(src[group_name], dst, group_name)


def save_training_params(h5_path, params, p_cross, seed, plasticity_on=True):
    """
    Write the /training_params attributes: p_cross, STDP parameters, task
    input parameters, the trial-sequence seed, and the burn-in duration. Also
    records the execution mode (exec_mode) and whether STDP was on, so each
    output file documents on its own which experimental condition produced
    it.
    """
    with h5py.File(h5_path, 'a') as f:
        grp = f.require_group('training_params')
        grp.attrs['p_cross'] = float(p_cross)
        grp.attrs['seed'] = int(seed)
        grp.attrs['exec_mode'] = str(params.get('exec_mode', 'sustained'))
        grp.attrs['plasticity_on'] = bool(plasticity_on)
        grp.attrs['weight_norm'] = bool(params.get('weight_norm', True))
        grp.attrs['inhibitory_plasticity'] = bool(params.get('inhibitory_plasticity', False))
        # These are the inhibitory-plasticity parameters that get varied across
        # the sweep. We record their values here so the sweep-aggregation code
        # can read each run's settings directly from its own file.
        for k in ('rho0', 'eta_istdp', 'tau_istdp'):
            if k in params:
                grp.attrs[k] = float(params[k])
        for k in ('tau_plus', 'tau_minus', 'A_plus', 'A_minus', 'w_max',
                  'n_input', 'r_max', 't_burn_in'):
            grp.attrs[k] = float(params[k])
