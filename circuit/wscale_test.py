"""
This script tests w_scale_II=1.0 (full-strength inhibitory-to-inhibitory
connections) together with nu_ext=6.25 Hz, to see whether this combination
lands in the balanced AI regime. It also tries a few nearby combinations
(different g_EI, nu_ext, w_scale_II, and seeds) for comparison.
"""
import numpy as np, sys
from brian2 import *
prefs.codegen.target = 'numpy'
from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

combos = [
    # (nu_ext, g_EI, w_scale_II, seed, label)
    (6.25, 0.060, 1.0, 42, 'wscale1.0,nu6.25,g0.06'),
    (6.25, 0.060, 0.5, 42, 'wscale0.5,nu6.25,g0.06'),
    (6.25, 0.055, 1.0, 42, 'wscale1.0,nu6.25,g0.055'),
    (5.5,  0.060, 1.0, 42, 'wscale1.0,nu5.5,g0.06'),
    (7.0,  0.060, 1.0, 42, 'wscale1.0,nu7.0,g0.06'),
    (6.25, 0.060, 1.0, 0,  'wscale1.0,nu6.25,g0.06,seed0'),
]

print(f'{"label":<35}  r[0-5]  r[5-10]  r[10-20]  CV[10-20]  I/E[10-20]')
print('-'*85)
sys.stdout.flush()

for nu_ext, g_ei, wscale, seed, label in combos:
    params = {**DEFAULT_PARAMS, 'nu_ext': nu_ext, 'g_EI': g_ei*1e-9, 'w_scale_II': wscale}
    objs = build_network(params, seed=seed)
    objs['net'].run(20.0 * second)

    t_E = objs['spike_E'].t / second
    t_I = objs['spike_I'].t / second
    
    r05  = sum(1 for t in t_E if t < 5) / (800 * 5)
    r510 = sum(1 for t in t_E if 5 <= t < 10) / (800 * 5)
    r1020 = sum(1 for t in t_E if t >= 10) / (800 * 10)
    
    ri1020 = sum(1 for t in t_I if t >= 10) / (200 * 10)
    ratio = ri1020 / r1020 if r1020 > 0 else float('nan')

    trains_E = _extract_spike_trains(objs['spike_E'], 800, 20.0)
    _, cv = compute_cv_isi(trains_E, 10.0, 20.0, min_spikes=5)
    
    ai_rate = 2 <= r1020 <= 10
    ai_cv   = 0.8 <= cv <= 1.2
    flag = ' AI!' if (ai_rate and ai_cv) else ''
    print(f'{label:<35}  {r05:5.2f}   {r510:5.2f}    {r1020:5.2f}     {cv:5.3f}     {ratio:4.1f}{flag}')
    sys.stdout.flush()
