"""
This script runs 30-second simulations to check whether w_scale_II=0.5
(inhibitory-to-inhibitory connection strength) gives a stable AI "fixed
point", meaning the firing rate and CV-ISI settle into the AI band and stay
there rather than drifting. It also tries a few nearby values of w_scale_II,
g_EI, and nu_ext for comparison.
"""
import numpy as np, sys
from brian2 import *
prefs.codegen.target = 'numpy'
from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

combos = [
    # (nu_ext, g_EI, w_scale_II, seed)
    (6.25, 0.060, 0.50, 42),
    (6.25, 0.060, 0.40, 42),
    (6.25, 0.060, 0.35, 42),
    (6.25, 0.060, 0.30, 42),
    (6.25, 0.055, 0.50, 42),
    (6.25, 0.065, 0.50, 42),
    (5.5,  0.060, 0.50, 42),
    (7.0,  0.060, 0.50, 42),
]

hdr = f'{"nu  g   ws":<18}  r[0-5]  r[5-10]  r[10-20]  r[20-30]  CV[10-20]  CV[20-30]  I/E[20-30]'
print(hdr)
print('-'*len(hdr))
sys.stdout.flush()

for nu_ext, g_ei, wscale, seed in combos:
    params = {**DEFAULT_PARAMS, 'nu_ext': nu_ext, 'g_EI': g_ei*1e-9, 'w_scale_II': wscale}
    objs = build_network(params, seed=seed)
    objs['net'].run(30.0 * second)

    tE = objs['spike_E'].t / second
    tI = objs['spike_I'].t / second
    
    r05   = sum(1 for t in tE if t < 5)   / (800 * 5)
    r510  = sum(1 for t in tE if 5 <= t < 10)  / (800 * 5)
    r1020 = sum(1 for t in tE if 10 <= t < 20) / (800 * 10)
    r2030 = sum(1 for t in tE if t >= 20) / (800 * 10)
    
    ri2030 = sum(1 for t in tI if t >= 20) / (200 * 10)
    ratio  = ri2030 / r2030 if r2030 > 0 else float('nan')

    trains_E = _extract_spike_trains(objs['spike_E'], 800, 30.0)
    _, cv1020 = compute_cv_isi(trains_E, 10.0, 20.0, min_spikes=5)
    _, cv2030 = compute_cv_isi(trains_E, 20.0, 30.0, min_spikes=5)
    
    label = f'nu={nu_ext} g={g_ei} ws={wscale}'
    ai_r  = 2 <= r2030 <= 10
    ai_cv = 0.8 <= cv2030 <= 1.2
    flag = ' AI!' if (ai_r and ai_cv) else (' rate?' if ai_r else (' cv?' if ai_cv else ''))
    print(f'{label:<18}  {r05:5.2f}   {r510:5.2f}    {r1020:5.2f}     {r2030:5.2f}     '
          f'{cv1020:5.3f}      {cv2030:5.3f}      {ratio:4.1f}{flag}')
    sys.stdout.flush()
