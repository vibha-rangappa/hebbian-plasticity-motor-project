"""
This is a quick check, run with 4 different random seeds, of w_scale_II
values 0.65, 0.70, and 0.75 (how strong inhibitory-to-inhibitory connections
are). We predicted these values should give an inhibitory-to-excitatory
firing rate ratio (I/E) of roughly 2.5 to 3, which is the kind of balance seen
in real cortical circuits.
"""
import numpy as np, sys
from brian2 import *
prefs.codegen.target = 'numpy'
from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

def pairwise_corr(spike_mon, N, t_start, t_end, n_pairs=150, dt=0.002):
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
T_SIM, T_START = 20.0, 10.0  # use shorter 20 s runs for speed, only look at the 10-20 s window

# The main question: does w_scale_II = 0.65-0.75 give an I/E ratio of about 2-3?
combos = [
    (6.25, 0.065, 0.60),
    (6.25, 0.065, 0.65),
    (6.25, 0.065, 0.70),
    (6.25, 0.065, 0.75),
    (6.25, 0.070, 0.65),
    (6.25, 0.070, 0.70),
    (6.25, 0.070, 0.75),
]

print(f'{"nu  g   ws":<22}  n/4  r(E)  CV   corr   I/E  per-seed CVs')
print('-' * 80)
sys.stdout.flush()

for nu_ext, g_ei, wscale in combos:
    results, cvs = [], []
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
        cvs.append(f'{cv:.3f}')
        sys.stdout.flush()

    n_pass = sum(r[0] for r in results)
    avg_r = np.mean([r[1] for r in results])
    avg_cv = np.nanmean([r[2] for r in results])
    avg_pc = np.nanmean([r[3] for r in results])
    avg_ie = np.nanmean([r[4] for r in results])
    flag = ' ★' if n_pass == 4 else ''

    label = f'nu={nu_ext} g={g_ei} ws={wscale}'
    print(f'{label:<22}  {n_pass}/4  {avg_r:.2f}  {avg_cv:.3f}  {avg_pc:.4f}  {avg_ie:.1f}{flag}  [{", ".join(cvs)}]')
    sys.stdout.flush()
