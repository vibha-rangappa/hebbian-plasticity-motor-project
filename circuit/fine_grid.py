import numpy as np
import sys
from brian2 import *
prefs.codegen.target = 'numpy'
from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

nu_vals  = [4.0, 6.0, 8.0, 10.0, 12.0]
gei_vals = [0.04e-9, 0.06e-9, 0.08e-9, 0.10e-9, 0.12e-9]

print('nu_ext   g_EI(nA)   nu_E   nu_I    I/E     CV')
print('-'*55)
sys.stdout.flush()
for nu_ext in nu_vals:
    for g_ei in gei_vals:
        params = {**DEFAULT_PARAMS, 'nu_ext': nu_ext, 'g_EI': g_ei}
        objs = build_network(params, seed=42)
        objs['net'].run(3.0 * second)

        rate_E = objs['spike_E'].num_spikes / (800 * 3.0)
        rate_I = objs['spike_I'].num_spikes / (200 * 3.0)
        ratio = rate_I/rate_E if rate_E > 0.05 else float('nan')

        trains_E = _extract_spike_trains(objs['spike_E'], 800, 3.0)
        _, cv = compute_cv_isi(trains_E, 0.5, 3.0, min_spikes=5)

        flag = ' <-- AI' if (2<=rate_E<=10 and 0.8<=cv<=1.2) else ''
        print(f'{nu_ext:6.1f}  {g_ei*1e9:8.3f}  {rate_E:6.2f}  {rate_I:6.2f}  {ratio:5.2f}  {cv:6.3f}{flag}')
        sys.stdout.flush()
