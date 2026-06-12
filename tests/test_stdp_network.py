# tests/test_stdp_network.py

import os

import numpy as np
import pytest
import h5py
from brian2 import start_scope, second, amp, Hz

from circuit.network import build_network, DEFAULT_PARAMS
from plasticity.stdp_network import (
    DEFAULT_PARAMS_PLASTICITY,
    apply_pool_rescaling,
    load_baseline,
    build_stdp_network,
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
    os.path.dirname(__file__), '..', 'circuit', 'results', 'baseline_network.h5')


# The saved baseline_network.h5 was generated with seed=7 (see
# circuit/results/baseline_network.h5:/validation/seed). build_network() is
# deterministic given (params, seed), so seed=7 is required to reproduce its
# connectivity exactly.
BASELINE_SEED = 7


def test_load_baseline_matches_saved_weights():
    start_scope()
    net_objs = load_baseline(BASELINE_H5, DEFAULT_PARAMS, seed=BASELINE_SEED)

    with h5py.File(BASELINE_H5, 'r') as f:
        saved_w = f['weights/W_EE/data'][:]

    actual_w = np.array(net_objs['syn_EE'].w[:] / amp, dtype=np.float32)
    np.testing.assert_allclose(actual_w, saved_w, rtol=1e-5)


def test_load_baseline_returns_expected_keys():
    start_scope()
    net_objs = load_baseline(BASELINE_H5, DEFAULT_PARAMS, seed=BASELINE_SEED)
    expected = {
        'exc', 'inh', 'syn_EE', 'syn_EI', 'syn_IE', 'syn_II',
        'drive_E', 'drive_I', 'spike_E', 'spike_I', 'net',
    }
    assert expected.issubset(net_objs.keys())


def test_load_baseline_raises_on_mismatched_params():
    start_scope()
    bad_params = {**DEFAULT_PARAMS, 'N_exc': 10}
    with pytest.raises(ValueError):
        load_baseline(BASELINE_H5, bad_params, seed=42)


def _small_params(**overrides):
    """20 E + 5 I neurons, P=[0,8), X=[8,16), S=[16,20)."""
    return {
        **DEFAULT_PARAMS, **DEFAULT_PARAMS_PLASTICITY,
        'N_exc': 20, 'N_inh': 5,
        'P_size': 8, 'X_size': 8,
        **overrides,
    }


def test_build_stdp_network_preserves_connectivity():
    start_scope()
    small = _small_params()
    net_objs = build_network(small, seed=1)
    i_before = np.array(net_objs['syn_EE'].i[:])
    j_before = np.array(net_objs['syn_EE'].j[:])

    result = build_stdp_network(net_objs, small, p_cross=0.2, seed=1)
    syn = result['syn_EE']
    np.testing.assert_array_equal(np.array(syn.i[:]), i_before)
    np.testing.assert_array_equal(np.array(syn.j[:]), j_before)


def test_build_stdp_network_applies_pool_rescaling():
    start_scope()
    small = _small_params()
    net_objs = build_network(small, seed=1)
    i_arr = np.array(net_objs['syn_EE'].i[:])
    j_arr = np.array(net_objs['syn_EE'].j[:])
    w_before = np.array(net_objs['syn_EE'].w[:] / amp)

    result = build_stdp_network(net_objs, small, p_cross=0.2, seed=1)
    w_after = np.array(result['syn_EE'].w[:] / amp)

    expected = apply_pool_rescaling(i_arr, j_arr, w_before, p_cross=0.2,
                                     P_size=small['P_size'], X_size=small['X_size'])
    np.testing.assert_allclose(w_after, expected, rtol=1e-6)


def test_build_stdp_network_clips_initial_weights_to_w_max():
    """
    The circuit's lognormal W_EE is unbounded, but build_stdp_network's
    w_max is enforced by clip() on every STDP event regardless of `plastic`
    -- so a synapse with w > w_max would get silently clamped to w_max on
    its first spike, even during the frozen burn-in. build_stdp_network must
    clip inherited weights to w_max BEFORE pool rescaling, so:
      - initial weights are deterministic (don't depend on burn-in spike
        timing), and
      - the seeded/control cross-pool ratio is exactly p_cross even for
        synapses whose baseline weight exceeded w_max (0.2*w_max for seeded
        vs w_max for control => ratio 0.2, not something else).
    """
    start_scope()
    small = _small_params()
    net_objs = build_network(small, seed=1)
    syn = net_objs['syn_EE']

    i_arr = np.array(syn.i[:])
    j_arr = np.array(syn.j[:])
    in_P = i_arr < small['P_size']
    in_X = (i_arr >= small['P_size']) & (i_arr < small['P_size'] + small['X_size'])
    j_in_P = j_arr < small['P_size']
    j_in_X = (j_arr >= small['P_size']) & (j_arr < small['P_size'] + small['X_size'])
    cross_idx = np.where((in_P & j_in_X) | (in_X & j_in_P))[0][0]

    w = np.array(syn.w[:] / amp)
    w[cross_idx] = small['w_max'] * 1.5  # above w_max, as the baseline init can produce
    syn.w = w * amp

    result = build_stdp_network(net_objs, small, p_cross=0.2, seed=1)
    w_after = np.array(result['syn_EE'].w[:] / amp)

    assert np.all(w_after <= small['w_max'])
    np.testing.assert_allclose(w_after[cross_idx], 0.2 * small['w_max'], rtol=1e-6)


def test_build_stdp_network_has_stdp_state_variables():
    start_scope()
    small = _small_params()
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)
    syn = result['syn_EE']
    assert hasattr(syn, 'plastic')
    assert hasattr(syn, 'apre')
    assert hasattr(syn, 'apost')
    assert np.all(np.array(syn.plastic[:]) == 1)


def test_build_stdp_network_adds_input_neurons():
    start_scope()
    small = _small_params()
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)

    assert len(result['input_group']) == small['n_input']
    assert len(result['syn_input_E']) > 0
    assert len(result['syn_input_I']) > 0

    w_E = np.array(result['syn_input_E'].w[:] / amp)
    w_I = np.array(result['syn_input_I'].w[:] / amp)
    assert np.all(w_E > 0)
    assert np.all(w_I > 0)


def test_build_stdp_network_input_rates_default_to_background():
    start_scope()
    small = _small_params()
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)
    rates = np.array(result['input_group'].rates[:] / Hz)
    np.testing.assert_allclose(rates, small['r_background'])


def test_stdp_plastic_zero_freezes_weights():
    start_scope()
    small = _small_params(nu_ext=1000.0)
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)
    syn = result['syn_EE']
    syn.plastic = 0

    w_before = np.array(syn.w[:] / amp).copy()
    result['net'].run(0.1 * second)
    w_after = np.array(syn.w[:] / amp)
    np.testing.assert_allclose(w_before, w_after)


def test_stdp_plastic_one_changes_weights():
    start_scope()
    small = _small_params(nu_ext=1000.0)
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)
    syn = result['syn_EE']

    w_before = np.array(syn.w[:] / amp).copy()
    result['net'].run(0.5 * second)
    w_after = np.array(syn.w[:] / amp)
    # atol=0: weights are ~1e-11 (amps), so np.allclose's default atol=1e-8
    # would mask any realistic change at this magnitude.
    assert not np.allclose(w_before, w_after, atol=0), \
        "STDP did not change any weights in 0.5 s"


def test_stdp_weights_clipped_to_w_max():
    start_scope()
    small = _small_params(nu_ext=1000.0)
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)
    syn = result['syn_EE']

    result['net'].run(1.0 * second)
    w_after = np.array(syn.w[:] / amp)
    assert np.all(w_after >= 0.0)
    assert np.all(w_after <= small['w_max'] * 1.0000001)
