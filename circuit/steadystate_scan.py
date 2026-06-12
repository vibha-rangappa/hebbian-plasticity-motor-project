"""20-second runs to find the point where STEADY-STATE rate >= 2 Hz AND CV >= 0.8."""
import numpy as np, sys
from brian2 import *
prefs.codegen.target = 'numpy'
from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

# Test a range of nu_ext values at g_EI=0.055 nA
combos = [(4.4, 0.055), (5.0, 0.055), (5.5, 0.055), (6.0, 0.055),
          (5.0, 0.057), (5.0, 0.058), (5.5, 0.057)]

print('nu_ext  g_EI  rate[0-5s]  rate[10-20s]  CV[5-10s]  CV[10-20s]')
print('-'*70)
sys.stdout.flush()
for nu_ext, g_ei in combos:
    params = {**DEFAULT_PARAMS, 'nu_ext': nu_ext, 'g_EI': g_ei*1e-9}
    objs = build_network(params, seed=42)
    spike_mon = objs['spike_E']
    spike_mon_I = objs['spike_I']
    objs['net'].run(20.0 * second)

    trains_E = _extract_spike_trains(spike_mon, 800, 20.0)

    # Rates in different windows
    spk_0to5  = sum(1 for t in spike_mon.t/second if t < 5.0)
    spk_10to20 = sum(1 for t in spike_mon.t/second if 10.0 <= t < 20.0)
    rate_early = spk_0to5  / (800 * 5.0)
    rate_late  = spk_10to20 / (800 * 10.0)

    _, cv_5_10  = compute_cv_isi(trains_E, 5.0, 10.0, min_spikes=5)
    _, cv_10_20 = compute_cv_isi(trains_E, 10.0, 20.0, min_spikes=5)

    flag = ' AI' if (2<=rate_late<=10 and 0.8<=cv_10_20<=1.2) else ''
    print(f'{nu_ext:5.1f}  {g_ei:.3f}  {rate_early:8.2f}  {rate_late:11.2f}  '
          f'{cv_5_10:9.3f}  {cv_10_20:9.3f}{flag}')
    sys.stdout.flush()
