"""
Verify that w_EE = 0.012 nA (EPSP ~0.3 mV) puts the network in the diffusion
regime and enables the AI operating point.

Brunel-equivalent effective g: g_EI = g * (C_I/C_E) * (tau_syn_I/tau_syn_E) * w_EE
  = 5 * (20/80) * (10/5) * 0.012 nA = 0.030 nA
Threshold nu_ext = I_th / (N_ext * w_EE * tau_syn_E)
  = 0.15e-9 / (80 * 0.012e-9 * 5e-3) = 31.25 Hz
AI range: nu_ext ~ 1.5-3 * 31.25 = 47-94 Hz
"""
import numpy as np, sys
from brian2 import *
prefs.codegen.target = 'numpy'
from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

w_new = 0.012e-9  # A (5x smaller)
nu_thr = 0.15e-9 / (80 * w_new * 5e-3)  # = 31.25 Hz
print(f'w_EE = {w_new*1e9:.3f} nA,  EPSP ~ {100e6*w_new*5e-3/20e-3*1e3:.2f} mV,  nu_thr = {nu_thr:.1f} Hz')
print()

# Scan: nu_ext = [40, 50, 60, 70] Hz, g_EI scaled to match Brunel's g=3-7
# g_eff = g_EI/w_EE * (C_I/C_E) * (tau_I/tau_E) = g_EI/w_new * 0.25 * 2 = g_EI/w_new * 0.5
# For g_eff = 3: g_EI = 3*2*w_new = 6*w_new = 0.072 nA
# For g_eff = 5: g_EI = 5*2*w_new = 10*w_new = 0.120 nA
# For g_eff = 7: g_EI = 7*2*w_new = 14*w_new = 0.168 nA
g_scales = [3.0, 4.0, 5.0, 6.0, 7.0]
nu_vals  = [40.0, 50.0, 60.0, 70.0, 80.0]

print('nu_ext   g_EI(nA)  g_eff   nu_E   nu_I    I/E     CV')
print('-'*60)
for nu_ext in nu_vals:
    for g_s in g_scales:
        g_ei = g_s * 2 * w_new  # g_eff = g_ei / (2*w_new)
        params = {**DEFAULT_PARAMS, 'nu_ext': nu_ext, 'g_EI': g_ei, 'w_mean_EE': w_new}
        # Also update sigma_w to 0.5 (unchanged), w_scale_II to 0.2 (unchanged)
        objs = build_network(params, seed=42)
        objs['net'].run(5.0 * second)

        rate_E = objs['spike_E'].num_spikes / (800 * 5.0)
        rate_I = objs['spike_I'].num_spikes / (200 * 5.0)
        ratio = rate_I/rate_E if rate_E > 0.05 else float('nan')

        trains_E = _extract_spike_trains(objs['spike_E'], 800, 5.0)
        _, cv = compute_cv_isi(trains_E, 1.0, 5.0, min_spikes=5)

        flag = ' <--AI' if (2<=rate_E<=10 and 0.8<=cv<=1.2) else ''
        print(f'{nu_ext:6.0f}  {g_ei*1e9:8.3f}  {g_s:5.1f}  {rate_E:6.2f}  {rate_I:6.2f}  '
              f'{ratio:5.2f}  {cv:6.3f}{flag}')
        sys.stdout.flush()
    print()
