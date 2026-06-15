"""
This is a one-off calibration scan. It checks whether using a smaller E-to-E
weight, w_EE = 0.012 nA (which gives an EPSP of about 0.3 mV, a more
realistic single-synapse effect), still lets the network reach the
"diffusion regime" (lots of small inputs adding up, rather than a few big
ones) and the asynchronous irregular (AI) firing pattern we want.

It works through the numbers using Brunel's (2000) way of describing the
network:
The "effective g" (relative strength of inhibition to excitation) is:
  g_EI = g * (C_I/C_E) * (tau_syn_I/tau_syn_E) * w_EE
       = 5 * (20/80) * (10/5) * 0.012 nA = 0.030 nA
The threshold background rate (the rate at which background input alone
would push a neuron to fire) is:
  nu_thr = I_th / (N_ext * w_EE * tau_syn_E)
         = 0.15e-9 / (80 * 0.012e-9 * 5e-3) = 31.25 Hz
The AI regime is expected somewhere in the range nu_ext ~ 1.5 to 3 times
nu_thr, i.e. about 47-94 Hz.

The script below then runs short simulations over a grid of nu_ext and g_EI
values to see which combinations actually land in the AI regime.
"""
import numpy as np, sys
from brian2 import *
prefs.codegen.target = 'numpy'
from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

w_new = 0.012e-9  # A, this is 5x smaller than the default w_mean_EE
nu_thr = 0.15e-9 / (80 * w_new * 5e-3)  # works out to 31.25 Hz
print(f'w_EE = {w_new*1e9:.3f} nA,  EPSP ~ {100e6*w_new*5e-3/20e-3*1e3:.2f} mV,  nu_thr = {nu_thr:.1f} Hz')
print()

# Scan over nu_ext = 40-80 Hz and g_EI, where g_EI is set so the "effective g"
# (g_eff) covers Brunel's range of 3-7.
# g_eff = g_EI/w_EE * (C_I/C_E) * (tau_I/tau_E) = g_EI/w_new * 0.25 * 2 = g_EI/w_new * 0.5
# So for g_eff = 3: g_EI = 3*2*w_new = 6*w_new = 0.072 nA
#    for g_eff = 5: g_EI = 5*2*w_new = 10*w_new = 0.120 nA
#    for g_eff = 7: g_EI = 7*2*w_new = 14*w_new = 0.168 nA
g_scales = [3.0, 4.0, 5.0, 6.0, 7.0]
nu_vals  = [40.0, 50.0, 60.0, 70.0, 80.0]

print('nu_ext   g_EI(nA)  g_eff   nu_E   nu_I    I/E     CV')
print('-'*60)
for nu_ext in nu_vals:
    for g_s in g_scales:
        g_ei = g_s * 2 * w_new  # this gives g_eff = g_ei / (2*w_new) = g_s
        params = {**DEFAULT_PARAMS, 'nu_ext': nu_ext, 'g_EI': g_ei, 'w_mean_EE': w_new}
        # sigma_w (0.5) and w_scale_II (0.2) are left at their default values
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
