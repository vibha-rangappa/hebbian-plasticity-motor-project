# tests/test_stdp_network.py
#
# Tests for plasticity/stdp_network.py: the helper functions that set up
# multiplicative synaptic scaling and "pool rescaling" (the seeded cross-pool
# weight pattern used to give STDP a starting bias), plus build_stdp_network
# itself, which turns a plain (non-plastic) circuit into one with E->E STDP,
# optional inhibitory STDP (iSTDP), and the external input neurons that drive
# the network. The checks cover: weight rescaling math, loading a saved
# baseline network, and that STDP actually changes weights (and only when
# turned on), keeps them within bounds, and preserves connectivity.

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
    compute_target_insums,
    rescale_to_target,
)


# ---- weight normalization (multiplicative synaptic scaling) ----

def test_rescale_to_target_restores_per_post_sum():
    # Set up 2 postsynaptic neurons. Post neuron 0 receives synapses [0,1],
    # post neuron 1 receives synapses [2,3].
    post = np.array([0, 0, 1, 1])
    w = np.array([0.6, 0.4, 0.3, 0.3])        # post0 sum=1.0, post1 sum=0.6
    target = np.array([1.0, 1.0])             # we want both sums to become 1.0
    w_max = 10.0
    out = rescale_to_target(post, w, target, w_max)
    # After rescaling, each neuron's total incoming weight should match its target.
    assert out[:2].sum() == pytest.approx(1.0)
    assert out[2:].sum() == pytest.approx(1.0)


def test_rescale_to_target_preserves_relative_pattern():
    """Rescaling multiplies every weight onto the same neuron by the same
    number, so the relative sizes of the weights (their ratios) don't change."""
    post = np.array([0, 0, 0])
    w = np.array([0.2, 0.4, 0.6])             # ratios 1:2:3
    out = rescale_to_target(post, w, target=np.array([3.0]), w_max=10.0)
    np.testing.assert_allclose(out / out[0], [1.0, 2.0, 3.0], rtol=1e-9)


def test_rescale_to_target_clips_to_w_max():
    post = np.array([0, 0])
    w = np.array([1.0, 1.0])
    # A target of 10 would scale each weight up to 5, but w_max caps it at 3.
    out = rescale_to_target(post, w, target=np.array([10.0]), w_max=3.0)
    assert np.all(out <= 3.0)


def test_rescale_to_target_leaves_zero_sum_untouched():
    post = np.array([0, 1, 1])
    w = np.array([0.0, 0.5, 0.5])             # post0 has a zero incoming sum
    out = rescale_to_target(post, w, target=np.array([1.0, 1.0]), w_max=10.0)
    # If a neuron's incoming weights already sum to zero, there's nothing to
    # rescale, so its weight (0) should be left as 0 (scale factor of 1).
    assert out[0] == 0.0


def test_compute_target_insums_groups_by_post():
    # For each postsynaptic neuron (j), sum up the weights of all its
    # incoming synapses. Neuron 0 gets synapses with w=0.1 and 0.2 (sum 0.3),
    # neuron 1 gets w=0.5 (sum 0.5), neuron 2 gets w=0.3 and 0.4 (sum 0.7).
    j = np.array([0, 0, 1, 2, 2])
    w = np.array([0.1, 0.2, 0.5, 0.3, 0.4])
    target = compute_target_insums(j, w, n_exc=3)
    np.testing.assert_allclose(target, [0.3, 0.5, 0.7])


def test_apply_pool_rescaling_cross_pool_only():
    # Two neuron pools: P = {0, 1} and X = {2, 3}, no shared pool S.
    # The four synapses below cover every combination:
    # (0,1) = P->P (within-pool), (0,2) = P->X (cross-pool),
    # (2,0) = X->P (cross-pool), (2,3) = X->X (within-pool).
    i = np.array([0, 0, 2, 2])
    j = np.array([1, 2, 0, 3])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    w_new = apply_pool_rescaling(i, j, w, p_cross=0.2, P_size=2, X_size=2)
    # Only the cross-pool synapses (P->X and X->P) should be scaled down by
    # p_cross=0.2. Within-pool synapses (P->P, X->X) stay at their original
    # weight of 1.0.
    np.testing.assert_allclose(w_new, [1.0, 0.2, 0.2, 1.0])


def test_apply_pool_rescaling_shared_pool_unchanged():
    # Two neuron pools P = {0, 1} and X = {2, 3}, plus a shared pool S = {4}.
    # The synapses below are (4,0)=S->P, (0,4)=P->S, (2,4)=X->S. None of
    # these are a direct P<->X connection, so none should be touched by the
    # cross-pool rescaling.
    i = np.array([4, 0, 2])
    j = np.array([0, 4, 4])
    w = np.array([1.0, 1.0, 1.0])
    w_new = apply_pool_rescaling(i, j, w, p_cross=0.2, P_size=2, X_size=2)
    np.testing.assert_allclose(w_new, [1.0, 1.0, 1.0])


def test_apply_pool_rescaling_does_not_mutate_input():
    # Make sure apply_pool_rescaling returns a new array and doesn't change
    # the weight array we passed in. Mutating inputs in place is an easy way
    # to introduce confusing bugs elsewhere in the code.
    i = np.array([0, 0])
    j = np.array([1, 2])
    w = np.array([1.0, 1.0])
    w_orig = w.copy()
    apply_pool_rescaling(i, j, w, p_cross=0.2, P_size=2, X_size=2)
    np.testing.assert_array_equal(w, w_orig)


BASELINE_H5 = os.path.join(
    os.path.dirname(__file__), '..', 'circuit', 'results', 'baseline_network.h5')


# The saved baseline_network.h5 file was generated with seed=7 (see
# circuit/results/baseline_network.h5:/validation/seed). build_network() is
# deterministic given the same (params, seed), so we have to use seed=7 here
# too if we want to reproduce its connectivity exactly.
BASELINE_SEED = 7


def test_load_baseline_matches_saved_weights():
    # Rebuild the network from the same seed used to make the saved file,
    # then check that the E->E weights we get now match the ones saved in
    # the HDF5 file. This confirms load_baseline() reproduces the saved
    # network rather than building something different.
    start_scope()
    net_objs = load_baseline(BASELINE_H5, DEFAULT_PARAMS, seed=BASELINE_SEED)

    with h5py.File(BASELINE_H5, 'r') as f:
        saved_w = f['weights/W_EE/data'][:]

    actual_w = np.array(net_objs['syn_EE'].w[:] / amp, dtype=np.float32)
    np.testing.assert_allclose(actual_w, saved_w, rtol=1e-5)


def test_load_baseline_returns_expected_keys():
    # load_baseline() should hand back a dictionary containing all the
    # network pieces we expect to use later (neuron groups, synapses,
    # external drives, spike monitors, and the Brian2 Network object itself).
    start_scope()
    net_objs = load_baseline(BASELINE_H5, DEFAULT_PARAMS, seed=BASELINE_SEED)
    expected = {
        'exc', 'inh', 'syn_EE', 'syn_EI', 'syn_IE', 'syn_II',
        'drive_E', 'drive_I', 'spike_E', 'spike_I', 'net',
    }
    assert expected.issubset(net_objs.keys())


def test_load_baseline_raises_on_mismatched_params():
    # If the params we ask for don't match the params the saved network was
    # built with (here we change N_exc), load_baseline() can't honestly claim
    # to reproduce the saved network, so it should refuse with a ValueError
    # instead of silently returning something different.
    start_scope()
    bad_params = {**DEFAULT_PARAMS, 'N_exc': 10}
    with pytest.raises(ValueError):
        load_baseline(BASELINE_H5, bad_params, seed=42)


def _small_params(**overrides):
    """A small test network: 20 excitatory + 5 inhibitory neurons.
    The excitatory neurons are split into pools P=[0,8), X=[8,16), S=[16,20)."""
    return {
        **DEFAULT_PARAMS, **DEFAULT_PARAMS_PLASTICITY,
        'N_exc': 20, 'N_inh': 5,
        'P_size': 8, 'X_size': 8,
        **overrides,
    }


def test_build_stdp_network_preserves_connectivity():
    # build_stdp_network() turns a plain network into a plastic one. It
    # should not add or remove any E->E synapses, just change their weights
    # and add the STDP machinery. So the list of (presynaptic, postsynaptic)
    # pairs should be identical before and after.
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
    # Check that build_stdp_network() actually applies the cross-pool weight
    # rescaling (apply_pool_rescaling) to the initial E->E weights, by
    # comparing its output to calling apply_pool_rescaling() directly on the
    # same starting weights.
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
    The circuit's starting weights (W_EE) are drawn from a lognormal
    distribution, which has no upper limit, so some weights can start out
    above w_max. Once STDP is running, every weight update is passed through
    clip(), so any synapse above w_max gets clamped down to w_max the first
    time it gets an STDP event, even during the frozen burn-in period.

    build_stdp_network must clip these "too big" inherited weights down to
    w_max BEFORE doing the cross-pool rescaling. This matters for two
    reasons:
      - it keeps the initial weights deterministic (they don't end up
        depending on exactly when spikes happen during burn-in), and
      - it keeps the seeded vs control cross-pool ratio exactly equal to
        p_cross, even for synapses whose starting weight was above w_max.
        For example: seeded weight = 0.2*w_max, control weight = w_max,
        giving a ratio of exactly 0.2, not some other number.
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
    # Force this one synapse's weight above w_max, since the lognormal init
    # can occasionally produce values like this.
    w[cross_idx] = small['w_max'] * 1.5
    syn.w = w * amp

    result = build_stdp_network(net_objs, small, p_cross=0.2, seed=1)
    w_after = np.array(result['syn_EE'].w[:] / amp)

    assert np.all(w_after <= small['w_max'])
    np.testing.assert_allclose(w_after[cross_idx], 0.2 * small['w_max'], rtol=1e-6)


def test_inhibitory_plasticity_off_by_default_keeps_syn_IE_static():
    """By default, inhibitory plasticity (iSTDP) is off, so the I->E synapses
    should not have the extra trace variables (apre_i) that iSTDP needs.
    This is a regression check: it makes sure we don't accidentally turn
    iSTDP on by default."""
    start_scope()
    small = _small_params()
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)
    assert result['inhibitory_plasticity'] is False
    assert not hasattr(result['syn_IE'], 'apre_i')


def test_inhibitory_plasticity_preserves_connectivity_and_init_weights():
    # Turning on inhibitory plasticity (iSTDP) should not change which
    # neurons are connected (i, j) or their starting weights. It should just
    # add the extra trace variables (apre_i, apost_i) that the iSTDP rule
    # needs to track recent spike timing.
    start_scope()
    small = _small_params()
    net_objs = build_network(small, seed=1)
    i_before = np.array(net_objs['syn_IE'].i[:])
    j_before = np.array(net_objs['syn_IE'].j[:])
    w_before = np.array(net_objs['syn_IE'].w[:] / amp)

    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1,
                                inhibitory_plasticity=True)
    syn = result['syn_IE']
    np.testing.assert_array_equal(np.array(syn.i[:]), i_before)
    np.testing.assert_array_equal(np.array(syn.j[:]), j_before)
    np.testing.assert_allclose(np.array(syn.w[:] / amp), w_before, rtol=1e-6)
    assert hasattr(syn, 'apre_i') and hasattr(syn, 'apost_i')


def test_inhibitory_plasticity_potentiates_when_E_fires_above_target():
    """iSTDP (Vogels et al. 2011) is a homeostatic rule: it tries to keep each
    excitatory (E) neuron's firing rate near a target rate, rho0, by
    adjusting how strongly inhibitory (I) neurons inhibit it. Here we drive
    the E neurons very hard (well above rho0), so iSTDP should respond by
    making the I->E inhibitory weights stronger, to push the E rate back
    down toward rho0."""
    start_scope()
    small = _small_params(nu_ext=1500.0)   # drive E neurons far above rho0
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1,
                                inhibitory_plasticity=True)
    syn = result['syn_IE']
    w_before = np.array(syn.w[:] / amp).copy()
    result['net'].run(0.5 * second)
    w_after = np.array(syn.w[:] / amp)
    # Homeostatic response: average inhibition should increase, to pull the
    # E firing rate back toward rho0.
    assert w_after.mean() > w_before.mean()
    # Inhibitory weights should never go negative (a negative weight here
    # would not make biological sense).
    assert np.all(w_after >= 0.0)
    assert np.all(w_after <= small['w_max_inh'] * 1.0000001)


def test_build_stdp_network_has_stdp_state_variables():
    # After build_stdp_network(), the E->E synapses should have the extra
    # state variables that the STDP rule needs: "plastic" (a switch that
    # turns plasticity on/off per synapse) and the eligibility traces "apre"
    # and "apost" (recent pre- and post-synaptic spike history). By default
    # plasticity should be turned on for every synapse (plastic == 1).
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
    # build_stdp_network() should add the external "input" neuron population
    # that will later carry the task cue, plus synapses connecting it to both
    # the excitatory (E) and inhibitory (I) neurons. All those input
    # synapse weights should be positive (a zero or negative input weight
    # would mean that input neuron does nothing or has the wrong sign).
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
    # Right after setup, before any task cue is given, the input neurons
    # should be firing at the background rate r_background, not at some
    # task-driven rate.
    start_scope()
    small = _small_params()
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)
    rates = np.array(result['input_group'].rates[:] / Hz)
    np.testing.assert_allclose(rates, small['r_background'])


def test_stdp_plastic_zero_freezes_weights():
    # If we set "plastic" to 0 for every E->E synapse, STDP should have no
    # effect: even after running the network for a while (with plenty of
    # spiking, since nu_ext is high), the weights should come out exactly
    # the same as they started.
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
    # With "plastic" left at its default of 1, running the network for 0.5 s
    # of heavy spiking (nu_ext=1000) should actually change at least some of
    # the E->E weights through STDP.
    start_scope()
    small = _small_params(nu_ext=1000.0)
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)
    syn = result['syn_EE']

    w_before = np.array(syn.w[:] / amp).copy()
    result['net'].run(0.5 * second)
    w_after = np.array(syn.w[:] / amp)
    # We use atol=0 here on purpose. These weights are tiny numbers
    # (around 1e-11, in amps), so np.allclose's default atol of 1e-8 would
    # treat any realistic change at this scale as "no change" and the test
    # would pass even if STDP wasn't doing anything.
    assert not np.allclose(w_before, w_after, atol=0), \
        "STDP did not change any weights in 0.5 s"


def test_stdp_weights_clipped_to_w_max():
    # After running with STDP for 1 s, every E->E weight should still be
    # within its allowed range: not negative, and not above w_max (plus a
    # tiny floating-point tolerance, 1.0000001x, to allow for rounding).
    start_scope()
    small = _small_params(nu_ext=1000.0)
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)
    syn = result['syn_EE']

    result['net'].run(1.0 * second)
    w_after = np.array(syn.w[:] / amp)
    assert np.all(w_after >= 0.0)
    assert np.all(w_after <= small['w_max'] * 1.0000001)
