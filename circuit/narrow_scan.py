"""Narrow scan to find the SI->AI transition in detail."""
import numpy as np, sys
from brian2 import *
prefs.codegen.target = 'numpy'
from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

# Very fine scan in the transition zone
# Two parallel strategies:
# A) Scan nu_ext finely at g_EI=0.055 nA 
# B) Try sigma_w=1.0 (more weight variability -> higher CV)

combos = [
    # (nu_ext, g_EI, sigma_w)
    (4.25, 0.055, 0.5),
    (4.30, 0.055, 0.5),
    (4.35, 0.055, 0.5),
    (4.40, 0.055, 0.5),
    # Higher weight variability -> more heterogeneity -> potentially higher CV
    (4.30, 0.055, 0.8),
    (4.40, 0.055, 0.8),
    (4.30, 0.055, 1.0),
    (4.40, 0.055, 1.0),
    (4.50, 0.055, 1.0),
    # Slightly weaker g_EI to push rate up
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
