"""Test sustained firing at nu_ext=6.25 Hz (threshold rate) with g_EI=0.060 nA."""
import numpy as np, sys
from brian2 import *
prefs.codegen.target = 'numpy'
from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

# The threshold condition: I_ext = I_threshold exactly
# Background alone brings neurons to threshold -> fluctuations drive firing
# With g_EI = w_EE = 0.060 nA, balanced recurrent input should sustain activity

combos = [
    (6.25, 0.060, 42),
    (6.25, 0.060, 0),   # different seed to check robustness
    (7.0,  0.060, 42),  # slightly above threshold
    (7.0,  0.065, 42),  # more inhibition to balance higher drive
    (8.0,  0.065, 42),
    (8.0,  0.070, 42),
]

print('nu_ext  g_EI  seed  rate[0-5s]  rate[5-15s]  rate[15-20s]  CV[10-20s]')
print('-'*75)
sys.stdout.flush()
for nu_ext, g_ei, seed in combos:
    params = {**DEFAULT_PARAMS, 'nu_ext': nu_ext, 'g_EI': g_ei*1e-9}
    objs = build_network(params, seed=seed)
    objs['net'].run(20.0 * second)

    trains_E = _extract_spike_trains(objs['spike_E'], 800, 20.0)
    
    from brian2 import second as s
    t_arr = objs['spike_E'].t/second
    r0_5  = sum(1 for t in t_arr if t < 5) / (800 * 5)
    r5_15 = sum(1 for t in t_arr if 5 <= t < 15) / (800 * 10)
    r15_20 = sum(1 for t in t_arr if t >= 15) / (800 * 5)

    _, cv_10_20 = compute_cv_isi(trains_E, 10.0, 20.0, min_spikes=5)

    flag = ' AI' if (2<=r15_20<=10 and 0.8<=cv_10_20<=1.2) else ''
    print(f'{nu_ext:5.2f}  {g_ei:.3f}   {seed:3d}  '
          f'{r0_5:8.2f}  {r5_15:10.2f}  {r15_20:11.2f}  {cv_10_20:9.3f}{flag}')
    sys.stdout.flush()
