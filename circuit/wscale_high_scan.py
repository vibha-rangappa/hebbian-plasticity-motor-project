"""Scan w_scale_II=0.60-0.80 — higher I-I feedback should bring I/E toward 2-3x."""
import numpy as np, sys
from brian2 import *
prefs.codegen.target = 'numpy'
from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

def pairwise_corr(spike_mon, N, t_start, t_end, n_pairs=200, dt=0.002):
    bins = np.arange(t_start, t_end + dt, dt)
    spk_i = spike_mon.i; spk_t = spike_mon.t / second
    mask = (spk_t >= t_start) & (spk_t < t_end)
    trains = np.zeros((N, len(bins)-1), dtype=float)
    for ni, ti in zip(spk_i[mask], spk_t[mask]):
        b = int((ti - t_start) / dt)
        if 0 <= b < trains.shape[1]: trains[ni, b] += 1
    rng = np.random.default_rng(99)
    idx = rng.choice(N, size=(n_pairs, 2), replace=False)
    corrs = [np.corrcoef(trains[a], trains[b])[0,1] for a,b in idx 
             if trains[a].std()>0 and trains[b].std()>0]
    return float(np.mean(corrs)) if corrs else float('nan')

SEEDS = [42, 0, 1, 7]
T_SIM, T_START = 25.0, 15.0

# Focus: ws 0.60-0.80 with most promising g/nu combos
combos = [
    (6.25, 0.065, 0.60), (6.25, 0.065, 0.65), (6.25, 0.065, 0.70),
    (6.25, 0.065, 0.75), (6.25, 0.065, 0.80),
    (6.25, 0.070, 0.65), (6.25, 0.070, 0.70), (6.25, 0.070, 0.75),
    (7.00, 0.070, 0.65), (7.00, 0.070, 0.70), (7.00, 0.070, 0.75),
    (7.00, 0.075, 0.70), (7.00, 0.075, 0.75),
]

print(f'{"nu  g   ws":<22}  n_pass  avg_rate  avg_CV  avg_corr  avg_IE')
print('-' * 72)
sys.stdout.flush()

for nu_ext, g_ei, wscale in combos:
    results = []
    for seed in SEEDS:
        p = {**DEFAULT_PARAMS, 'nu_ext': nu_ext, 'g_EI': g_ei*1e-9, 'w_scale_II': wscale}
        objs = build_network(p, seed=seed)
        objs['net'].run(T_SIM * second)

        tE = objs['spike_E'].t / second; tI = objs['spike_I'].t / second
        r_E = sum(1 for t in tE if t >= T_START) / (800 * (T_SIM - T_START))
        r_I = sum(1 for t in tI if t >= T_START) / (200 * (T_SIM - T_START))
        ratio = r_I / r_E if r_E > 0 else float('nan')

        trains_E = _extract_spike_trains(objs['spike_E'], 800, T_SIM)
        _, cv = compute_cv_isi(trains_E, T_START, T_SIM, min_spikes=5)
        pc = pairwise_corr(objs['spike_E'], 800, T_START, T_SIM)

        passed = (2 <= r_E <= 10) and (0.8 <= cv <= 1.2) and (pc < 0.05)
        results.append((passed, r_E, cv, pc, ratio))

    n_pass = sum(r[0] for r in results)
    avg_rate = np.mean([r[1] for r in results])
    avg_cv   = np.nanmean([r[2] for r in results])
    avg_corr = np.nanmean([r[3] for r in results])
    avg_ie   = np.nanmean([r[4] for r in results])
    flag = ' ★ALL★' if n_pass == 4 else f' ({n_pass}/4)'

    label = f'nu={nu_ext} g={g_ei} ws={wscale}'
    print(f'{label:<22}  {n_pass}/4     {avg_rate:6.2f}    {avg_cv:5.3f}   {avg_corr:6.4f}    {avg_ie:4.1f}{flag}')
    sys.stdout.flush()
