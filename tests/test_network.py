# tests/test_network.py
#
# Tests for circuit/network.py's build_network function, which builds the spiking
# excitatory/inhibitory (E/I) network in Brian2. These tests check that build_network
# returns all the expected pieces (neuron groups, synapses, monitors), that the
# neuron group sizes and connection probabilities match the requested parameters, that
# there are no self-connections (a neuron synapsing onto itself), that weights come
# out positive and near their target means, and that a small version of the network
# actually runs and produces spikes.

import numpy as np
import pytest
from brian2 import start_scope, second

from circuit.network import build_network, DEFAULT_PARAMS


def test_build_network_returns_required_keys():
    """build_network should hand back a dictionary containing exactly these pieces:
    the excitatory and inhibitory neuron groups (exc, inh), the four synapse
    populations (syn_EE, syn_EI, syn_IE, syn_II), the external drive inputs (drive_E,
    drive_I), the spike monitors (spike_E, spike_I), and the Brian2 Network object
    itself (net). This just checks none are missing or misnamed."""
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    expected = {
        'exc', 'inh',
        'syn_EE', 'syn_EI', 'syn_IE', 'syn_II',
        'drive_E', 'drive_I',
        'spike_E', 'spike_I',
        'net',
    }
    assert set(objs.keys()) == expected


def test_neuron_group_sizes():
    """The excitatory and inhibitory neuron groups should have the sizes requested in
    DEFAULT_PARAMS (N_exc and N_inh)."""
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    assert len(objs['exc']) == DEFAULT_PARAMS['N_exc']
    assert len(objs['inh']) == DEFAULT_PARAMS['N_inh']


def test_no_self_connections_EE():
    """No excitatory neuron should be connected to itself. In Brian2's Synapses object,
    .i lists the presynaptic (sending) neuron indices and .j lists the postsynaptic
    (receiving) ones for each connection, so we just check that no connection has the
    same index on both sides."""
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    syn = objs['syn_EE']
    # Brian2 Synapses.i = presynaptic (sending) neuron indices, .j = postsynaptic (receiving) indices
    pre = np.array(syn.i[:])
    post = np.array(syn.j[:])
    assert np.all(pre != post), "syn_EE contains self-connections"


def test_no_self_connections_II():
    """Same check as above, but for the inhibitory-to-inhibitory connections: no
    inhibitory neuron should synapse onto itself."""
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    syn = objs['syn_II']
    pre = np.array(syn.i[:])
    post = np.array(syn.j[:])
    assert np.all(pre != post), "syn_II contains self-connections"


def test_connectivity_fraction_EE():
    """The excitatory-to-excitatory connections are wired up randomly (Erdos-Renyi
    style) with connection probability p_connect. With no self-connections allowed,
    the possible number of E-to-E connections is Ne*(Ne-1) (every neuron can connect
    to every other neuron, but not itself), so the expected number of actual
    connections is Ne*(Ne-1)*p_connect. Random wiring won't hit that number exactly,
    so we allow up to 5% deviation, which is the expected size of the random
    fluctuation for this kind of random graph."""
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    Ne = DEFAULT_PARAMS['N_exc']
    p = DEFAULT_PARAMS['p_connect']
    expected = Ne * (Ne - 1) * p          # no self-connections allowed -> Ne*(Ne-1) possible pairs
    actual = len(objs['syn_EE'])
    # Allow +/-5% deviation, expected from the randomness of Erdos-Renyi wiring.
    assert abs(actual - expected) / expected < 0.05, \
        f"EE connectivity {actual} far from expected {expected:.0f}"


def test_connectivity_fraction_EI():
    """Same idea as the EE connectivity check, but for excitatory-to-inhibitory
    connections. Here self-connections aren't an issue since E and I are different
    populations, so the expected count is simply Ne*Ni*p_connect, again allowed to be
    off by up to 5% due to the randomness of the wiring."""
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    Ne = DEFAULT_PARAMS['N_exc']
    Ni = DEFAULT_PARAMS['N_inh']
    p = DEFAULT_PARAMS['p_connect']
    expected = Ne * Ni * p
    actual = len(objs['syn_EI'])
    # Allow +/-5% deviation, expected from the randomness of Erdos-Renyi wiring.
    assert abs(actual - expected) / expected < 0.05, \
        f"EI connectivity {actual} far from expected {expected:.0f}"


def test_weights_positive():
    """All synaptic weights, for every connection type (EE, EI, IE, II), should come
    out strictly positive. The sign of excitatory vs inhibitory effect is handled
    elsewhere (e.g. in how the synapse affects the post-synaptic neuron), not by the
    weight value itself being negative."""
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    from brian2 import amp as brian_amp
    for key in ('syn_EE', 'syn_EI', 'syn_IE', 'syn_II'):
        w = np.array(objs[key].w / brian_amp)  # divide out the "amp" unit to get plain numbers
        assert np.all(w > 0), f"{key} has non-positive weights"


def test_weight_mean_EE():
    """The average excitatory-to-excitatory weight should be close to the target value
    w_mean_EE (within 5%). Individual weights are drawn from a distribution, so they
    won't all be exactly w_mean_EE, but the average across all of them should land
    near it."""
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    from brian2 import amp as brian_amp
    w = np.array(objs['syn_EE'].w / brian_amp)
    target = DEFAULT_PARAMS['w_mean_EE']
    assert abs(w.mean() - target) / target < 0.05, \
        f"Mean EE weight {w.mean():.3e} A too far from target {target:.3e} A"


def test_weight_mean_IE():
    """Same idea as the EE weight check, but for inhibitory-to-excitatory weights: the
    average should be close to the target value g_EI (within 5%)."""
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    from brian2 import amp as brian_amp
    w = np.array(objs['syn_IE'].w / brian_amp)
    target = DEFAULT_PARAMS['g_EI']
    assert abs(w.mean() - target) / target < 0.05, \
        f"Mean IE weight {w.mean():.3e} A too far from target {target:.3e} A"


def test_network_runs_and_produces_spikes():
    """
    This is a "smoke test": build a smaller network (80 excitatory, 20 inhibitory
    neurons instead of the usual ~1000) and run it for 200 ms, just to check the whole
    thing compiles and actually runs without errors, and produces at least some
    spikes in both populations. Using a smaller network keeps the test fast (under
    about 10 seconds). Because this small network doesn't have realistic
    excitation/inhibition balance, it needs a much stronger external drive (1000 Hz,
    vs the 20 Hz used in the real spec) just to make sure neurons fire at all in such a
    short run.
    """
    start_scope()
    small = {
        **DEFAULT_PARAMS,
        'N_exc': 80,
        'N_inh': 20,
        'nu_ext': 1000.0,  # 1000 Hz smoke-test drive to guarantee spikes; the spec's 20 Hz does not reliably produce spikes in such short windows with N=100
    }
    objs = build_network(small, seed=0)
    objs['net'].run(0.2 * second)
    assert objs['spike_E'].num_spikes > 0, \
        "No E spikes in 200 ms, network may be silent"
    assert objs['spike_I'].num_spikes > 0, \
        "No I spikes in 200 ms, network may be silent"
