# tests/test_network_part2.py

import os

import numpy as np
import pytest
import h5py
from brian2 import start_scope, second, amp

from part1.network import build_network, DEFAULT_PARAMS
from part2.network_part2 import (
    DEFAULT_PARAMS_PART2,
    apply_pool_rescaling,
    load_part1_baseline,
)


def test_apply_pool_rescaling_cross_pool_only():
    # P = {0, 1}, X = {2, 3}, no S.
    # (0,1)=P->P, (0,2)=P->X, (2,0)=X->P, (2,3)=X->X
    i = np.array([0, 0, 2, 2])
    j = np.array([1, 2, 0, 3])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    w_new = apply_pool_rescaling(i, j, w, p_cross=0.2, P_size=2, X_size=2)
    np.testing.assert_allclose(w_new, [1.0, 0.2, 0.2, 1.0])


def test_apply_pool_rescaling_shared_pool_unchanged():
    # P = {0, 1}, X = {2, 3}, S = {4}.
    # (4,0)=S->P, (0,4)=P->S, (2,4)=X->S — none touch the P<->X cross term
    i = np.array([4, 0, 2])
    j = np.array([0, 4, 4])
    w = np.array([1.0, 1.0, 1.0])
    w_new = apply_pool_rescaling(i, j, w, p_cross=0.2, P_size=2, X_size=2)
    np.testing.assert_allclose(w_new, [1.0, 1.0, 1.0])


def test_apply_pool_rescaling_does_not_mutate_input():
    i = np.array([0, 0])
    j = np.array([1, 2])
    w = np.array([1.0, 1.0])
    w_orig = w.copy()
    apply_pool_rescaling(i, j, w, p_cross=0.2, P_size=2, X_size=2)
    np.testing.assert_array_equal(w, w_orig)


BASELINE_H5 = os.path.join(
    os.path.dirname(__file__), '..', 'part1', 'results', 'baseline_network.h5')


# The saved baseline_network.h5 was generated with seed=7 (see
# part1/results/baseline_network.h5:/validation/seed). build_network() is
# deterministic given (params, seed), so seed=7 is required to reproduce its
# connectivity exactly.
BASELINE_SEED = 7


def test_load_part1_baseline_matches_saved_weights():
    start_scope()
    net_objs = load_part1_baseline(BASELINE_H5, DEFAULT_PARAMS, seed=BASELINE_SEED)

    with h5py.File(BASELINE_H5, 'r') as f:
        saved_w = f['weights/W_EE/data'][:]

    actual_w = np.array(net_objs['syn_EE'].w[:] / amp, dtype=np.float32)
    np.testing.assert_allclose(actual_w, saved_w, rtol=1e-5)


def test_load_part1_baseline_returns_expected_keys():
    start_scope()
    net_objs = load_part1_baseline(BASELINE_H5, DEFAULT_PARAMS, seed=BASELINE_SEED)
    expected = {
        'exc', 'inh', 'syn_EE', 'syn_EI', 'syn_IE', 'syn_II',
        'drive_E', 'drive_I', 'spike_E', 'spike_I', 'net',
    }
    assert expected.issubset(net_objs.keys())


def test_load_part1_baseline_raises_on_mismatched_params():
    start_scope()
    bad_params = {**DEFAULT_PARAMS, 'N_exc': 10}
    with pytest.raises(ValueError):
        load_part1_baseline(BASELINE_H5, bad_params, seed=42)
