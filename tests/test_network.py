# tests/test_network.py

import numpy as np
import pytest
from brian2 import start_scope, second

from circuit.network import build_network, DEFAULT_PARAMS


def test_build_network_returns_required_keys():
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
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    assert len(objs['exc']) == DEFAULT_PARAMS['N_exc']
    assert len(objs['inh']) == DEFAULT_PARAMS['N_inh']


def test_no_self_connections_EE():
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    syn = objs['syn_EE']
    # Brian2 Synapses.i = presynaptic indices, .j = postsynaptic indices
    pre = np.array(syn.i[:])
    post = np.array(syn.j[:])
    assert np.all(pre != post), "syn_EE contains self-connections"


def test_no_self_connections_II():
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    syn = objs['syn_II']
    pre = np.array(syn.i[:])
    post = np.array(syn.j[:])
    assert np.all(pre != post), "syn_II contains self-connections"


def test_connectivity_fraction_EE():
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    Ne = DEFAULT_PARAMS['N_exc']
    p = DEFAULT_PARAMS['p_connect']
    expected = Ne * (Ne - 1) * p          # no self-connections → Ne*(Ne-1) possible
    actual = len(objs['syn_EE'])
    # Allow ±5% deviation (Erdos-Renyi variance)
    assert abs(actual - expected) / expected < 0.05, \
        f"EE connectivity {actual} far from expected {expected:.0f}"


def test_connectivity_fraction_EI():
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    Ne = DEFAULT_PARAMS['N_exc']
    Ni = DEFAULT_PARAMS['N_inh']
    p = DEFAULT_PARAMS['p_connect']
    expected = Ne * Ni * p
    actual = len(objs['syn_EI'])
    # Allow ±5% deviation (Erdos-Renyi variance)
    assert abs(actual - expected) / expected < 0.05, \
        f"EI connectivity {actual} far from expected {expected:.0f}"


def test_weights_positive():
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    from brian2 import amp as brian_amp
    for key in ('syn_EE', 'syn_EI', 'syn_IE', 'syn_II'):
        w = np.array(objs[key].w / brian_amp)  # strip units → float array in amps
        assert np.all(w > 0), f"{key} has non-positive weights"


def test_weight_mean_EE():
    """Mean E->E weight should be close to w_mean_EE (within 5%)."""
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    from brian2 import amp as brian_amp
    w = np.array(objs['syn_EE'].w / brian_amp)
    target = DEFAULT_PARAMS['w_mean_EE']
    assert abs(w.mean() - target) / target < 0.05, \
        f"Mean EE weight {w.mean():.3e} A too far from target {target:.3e} A"


def test_weight_mean_IE():
    """Mean I->E weight should be close to g_EI (within 5%)."""
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    from brian2 import amp as brian_amp
    w = np.array(objs['syn_IE'].w / brian_amp)
    target = DEFAULT_PARAMS['g_EI']
    assert abs(w.mean() - target) / target < 0.05, \
        f"Mean IE weight {w.mean():.3e} A too far from target {target:.3e} A"


def test_network_runs_and_produces_spikes():
    """
    Build a small network (80 E, 20 I) and run 200 ms.
    Verifies the full assembly compiles and spikes are produced.
    Using N=100 instead of 1000 keeps this test under ~10 s.
    Requires very strong external drive to bypass lack of inhibitory balance.
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
        "No E spikes in 200 ms — network may be silent"
    assert objs['spike_I'].num_spikes > 0, \
        "No I spikes in 200 ms — network may be silent"
