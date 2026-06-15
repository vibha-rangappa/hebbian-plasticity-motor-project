"""
This is a second, narrower one-off calibration scan, following on from
fine_grid.py. It zooms in on the transition between an excitation-dominated
regime (too much E activity) and a balanced regime, scanning over nu_ext
(background input rate) and g_EI (inhibitory weight).

For each (nu_ext, g_EI) pair it runs a short 5-second simulation, then
prints the E and I firing rates, the I/E rate ratio, and the CV-ISI
(irregularity measure), with an "AI" flag marking combinations that land in
the target asynchronous irregular regime (firing rate 2-10 Hz, CV-ISI
between 0.8 and 1.2).
"""
import numpy as np
import sys
from brian2 import *
prefs.codegen.target = 'numpy'
from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

# Zoom in on the transition between an excitation-dominated regime and a balanced regime
nu_vals  = [3.0, 4.0, 5.0, 6.0, 7.0]
gei_vals = np.array([0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070]) * 1e-9

print('nu_ext   g_EI(nA)   nu_E   nu_I    I/E     CV')
print('-'*55)
sys.stdout.flush()
for nu_ext in nu_vals:
    for g_ei in gei_vals:
        params = {**DEFAULT_PARAMS, 'nu_ext': nu_ext, 'g_EI': g_ei}
        objs = build_network(params, seed=42)
        objs['net'].run(5.0 * second)

        rate_E = objs['spike_E'].num_spikes / (800 * 5.0)
        rate_I = objs['spike_I'].num_spikes / (200 * 5.0)
        ratio = rate_I/rate_E if rate_E > 0.05 else float('nan')

        trains_E = _extract_spike_trains(objs['spike_E'], 800, 5.0)
        _, cv = compute_cv_isi(trains_E, 1.0, 5.0, min_spikes=5)

        flag = ' <-- AI' if (2<=rate_E<=10 and 0.8<=cv<=1.2) else ''
        print(f'{nu_ext:6.1f}  {g_ei*1e9:8.3f}  {rate_E:6.2f}  {rate_I:6.2f}  {ratio:5.2f}  {cv:6.3f}{flag}')
        sys.stdout.flush()
