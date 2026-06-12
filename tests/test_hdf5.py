# tests/test_hdf5.py

import os
import numpy as np
import h5py
import scipy.sparse
import pytest
from brian2 import start_scope, second

from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import save_baseline


@pytest.fixture
def small_network(tmp_path):
    """Build a small network, run 0.5 s, and return everything needed for save_baseline."""
    start_scope()
    small = {**DEFAULT_PARAMS, 'N_exc': 80, 'N_inh': 20, 'nu_ext': 15.0}
    objs = build_network(small, seed=7)
    objs['net'].run(0.5 * second)

    validation = {
        'mean_rate_E':        objs['spike_E'].num_spikes / (small['N_exc'] * 0.5),
        'mean_rate_I':        objs['spike_I'].num_spikes / (small['N_inh'] * 0.5),
        'mean_CV_ISI':        0.95,   # placeholder value
        'mean_pairwise_corr': 0.02,
        'raster_times':       np.array([0.1, 0.2, 0.3], dtype=np.float32),
        'raster_indices':     np.array([0, 1, 2],        dtype=np.int32),
    }
    return small, objs, validation, tmp_path / 'test_baseline.h5'


def test_hdf5_required_groups_exist(small_network):
    params, objs, validation, path = small_network
    save_baseline(str(path), params, objs, validation, seed=7)
    with h5py.File(path, 'r') as f:
        for group in ('network', 'weights', 'validation'):
            assert group in f, f"Missing group /{group}"


def test_hdf5_network_scalars(small_network):
    params, objs, validation, path = small_network
    save_baseline(str(path), params, objs, validation, seed=7)
    with h5py.File(path, 'r') as f:
        assert f['network/N_exc'][()] == 80
        assert f['network/N_inh'][()] == 20
        assert f['network/params_neuron/tau_m'][()] == pytest.approx(20e-3)


def test_hdf5_weight_coo_reconstruction(small_network):
    params, objs, validation, path = small_network
    save_baseline(str(path), params, objs, validation, seed=7)
    with h5py.File(path, 'r') as f:
        data  = f['weights/W_EE/data'][:]
        row   = f['weights/W_EE/row'][:]
        col   = f['weights/W_EE/col'][:]
        shape = f['weights/W_EE/shape'][:]
    W = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    assert W.shape == (80, 80)
    assert W.nnz > 0
    assert np.all(W.data > 0), "Weights must be positive"


def test_hdf5_validation_fields(small_network):
    params, objs, validation, path = small_network
    save_baseline(str(path), params, objs, validation, seed=7)
    with h5py.File(path, 'r') as f:
        for field in ('mean_rate_E', 'mean_rate_I', 'mean_CV_ISI',
                      'mean_pairwise_corr', 'raster_times', 'raster_indices',
                      'seed', 'nu_ext_hz', 'g_EI_nA'):
            assert field in f['validation'], f"Missing /validation/{field}"
