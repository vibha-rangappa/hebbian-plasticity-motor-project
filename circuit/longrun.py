"""
This script runs the network for 30 seconds at the best operating point found
so far (nu_ext=4.4 Hz, g_EI=0.055 nA). The point of the long run is to let the
CV-ISI (a measure of how irregular the spike timing is) settle down to a
stable value, by checking it over several time windows as the simulation goes
on.
"""
import numpy as np, sys
from brian2 import *
prefs.codegen.target = 'numpy'
from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

# Best candidate so far: nu_ext=4.4, g_EI=0.055 nA
params = {**DEFAULT_PARAMS, 'nu_ext': 4.4, 'g_EI': 0.055e-9}
objs = build_network(params, seed=42)
objs['net'].run(30.0 * second)

trains_E = _extract_spike_trains(objs['spike_E'], 800, 30.0)
rate_E = objs['spike_E'].num_spikes / (800 * 30.0)
rate_I = objs['spike_I'].num_spikes / (200 * 30.0)

for t_start, t_end, label in [
    (1, 5, '1-5s'), (5, 10, '5-10s'), (10, 20, '10-20s'), (20, 30, '20-30s')
]:
    _, cv = compute_cv_isi(trains_E, t_start, t_end, min_spikes=5)
    qualifying = sum(1 for tr in trains_E.values() 
                     if sum(1 for t in tr if t_start <= t < t_end) >= 5)
    print(f'  {label}: CV = {cv:.3f}  (neurons qualifying: {qualifying}/800)')

print(f'\nOverall rates: E={rate_E:.2f} Hz, I={rate_I:.2f} Hz')
