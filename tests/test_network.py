# tests/test_network.py

import numpy as np
import pytest
from brian2 import start_scope, second

from part1.network import build_network, DEFAULT_PARAMS


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
