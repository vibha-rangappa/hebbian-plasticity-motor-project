"""
This script zooms in on the boundary between the "synchronous irregular" (SI)
regime and the "asynchronous irregular" (AI) regime, by trying a fine grid of
parameter combinations close to that boundary and printing the resulting
firing rates and CV-ISI values for each one.
"""
import numpy as np, sys
from brian2 import *
prefs.codegen.target = 'numpy'
from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

# Very fine scan right in the transition zone.
# Two strategies are tried side by side:
# A) Scan nu_ext finely while holding g_EI fixed at 0.055 nA
# B) Try sigma_w=1.0 (more variability in the weights, which might give a
#    higher CV-ISI since neurons would be less in sync)

combos = [
    # (nu_ext, g_EI, sigma_w)
    (4.25, 0.055, 0.5),
    (4.30, 0.055, 0.5),
    (4.35, 0.055, 0.5),
    (4.40, 0.055, 0.5),
    # More weight variability could mean more heterogeneity, which could mean a higher CV
    (4.30, 0.055, 0.8),
    (4.40, 0.055, 0.8),
    (4.30, 0.055, 1.0),
    (4.40, 0.055, 1.0),
    (4.50, 0.055, 1.0),
    # Slightly weaker g_EI to try to push the firing rate up
    (4.00, 0.053, 0.5),
    (4.00, 0.054, 0.5),
]

print('nu_ext  g_EI(nA) sigma_w   nu_E   nu_I    CV[2-5s]  CV[5-10s]')
print('-'*70)
sys.stdout.flush()
for nu_ext, g_ei, sigma_w in combos:
    params = {**DEFAULT_PARAMS, 'nu_ext': nu_ext, 'g_EI': g_ei*1e-9, 'sigma_w': sigma_w}
    objs = build_network(params, seed=42)
    objs['net'].run(10.0 * second)

    rate_E = objs['spike_E'].num_spikes / (800 * 10.0)
    rate_I = objs['spike_I'].num_spikes / (200 * 10.0)

    trains_E = _extract_spike_trains(objs['spike_E'], 800, 10.0)
    _, cv_mid  = compute_cv_isi(trains_E, 2.0, 5.0, min_spikes=5)
    _, cv_late = compute_cv_isi(trains_E, 5.0, 10.0, min_spikes=5)

    flag = ' <--AI' if (2<=rate_E<=10 and 0.8<=cv_late<=1.2) else ''
    print(f'{nu_ext:5.2f}   {g_ei:7.3f}   {sigma_w:5.1f}  '
          f'{rate_E:6.2f}  {rate_I:6.2f}  {cv_mid:8.3f}  {cv_late:8.3f}{flag}')
    sys.stdout.flush()
