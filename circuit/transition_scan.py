import numpy as np
import sys
from brian2 import *
prefs.codegen.target = 'numpy'
from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

# Very fine scan around the SI->AI transition with long run
# Goal: find rate 2-10 Hz AND CV > 0.8
print('nu_ext   g_EI(nA)   nu_E   nu_I    I/E     CV     [1-5s]  [5-10s]')
print('-'*70)
sys.stdout.flush()

combos = [
    (4.2, 0.055), (4.4, 0.055), (4.5, 0.055), (4.6, 0.055), (4.8, 0.055),
    (4.0, 0.057), (4.0, 0.058), (5.0, 0.057), (5.0, 0.058),
    (3.5, 0.055), (3.5, 0.057), (3.5, 0.058),
]

for nu_ext, g_ei in combos:
    g_ei_A = g_ei * 1e-9
    params = {**DEFAULT_PARAMS, 'nu_ext': nu_ext, 'g_EI': g_ei_A}
    objs = build_network(params, seed=42)
    objs['net'].run(10.0 * second)

    rate_E = objs['spike_E'].num_spikes / (800 * 10.0)
    rate_I = objs['spike_I'].num_spikes / (200 * 10.0)
    ratio = rate_I/rate_E if rate_E > 0.05 else float('nan')

    trains_E = _extract_spike_trains(objs['spike_E'], 800, 10.0)
    _, cv_early = compute_cv_isi(trains_E, 1.0, 5.0, min_spikes=5)
    _, cv_late  = compute_cv_isi(trains_E, 5.0, 10.0, min_spikes=5)

    flag = ' <--AI' if (2<=rate_E<=10 and 0.8<=cv_late<=1.2) else ''
    print(f'{nu_ext:6.1f}  {g_ei:8.3f}  {rate_E:6.2f}  {rate_I:6.2f}  '
          f'{ratio:5.2f}  {cv_early:6.3f}  {cv_late:6.3f}{flag}')
    sys.stdout.flush()
