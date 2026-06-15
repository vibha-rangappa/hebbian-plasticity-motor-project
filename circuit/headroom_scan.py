"""
This script checks how much "headroom" we have around the chosen AI operating
point before plasticity (STDP) could push the network out of the AI regime.

It scans (nu_ext, g_EI) over a grid and produces two things:
  1. A corrected heatmap over the right parameter range, with the chosen
     operating point marked. This shows where the AI "corridor" (the band of
     parameters that gives AI-like activity) actually sits, and how narrow it
     is.
  2. A 1D sweep over g_EI at nu_ext=6.25 Hz, w_scale_II=0.50, to map where the
     edges of that AI corridor are. This tells us how much w_EE (the
     excitatory-to-excitatory weight) could grow under STDP before the network
     leaves the AI regime.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import second, prefs
prefs.codegen.target = 'numpy'
from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains, compute_pairwise_corr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
T_SIM, T_START = 20.0, 10.0  # only look at 10-20 s: long enough to be past the startup transient, short enough to run fast

NU_EXT_VALS = [5.0, 5.5, 6.0, 6.25, 6.5, 7.0, 7.5, 8.0]
G_EI_VALS   = [0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.090, 0.100, 0.120]
W_SCALE_II  = 0.50
SEED        = 42

# Operating point marker
OP_NU  = 6.25
OP_GEI = 0.075


def run_point(nu_ext, g_ei):
    p = {**DEFAULT_PARAMS, 'nu_ext': nu_ext, 'g_EI': g_ei * 1e-9, 'w_scale_II': W_SCALE_II}
    objs = build_network(p, seed=SEED)
    objs['net'].run(T_SIM * second)

    tE = objs['spike_E'].t / second
    tI = objs['spike_I'].t / second
    r_E = sum(1 for t in tE if t >= T_START) / (800 * (T_SIM - T_START))
    r_I = sum(1 for t in tI if t >= T_START) / (200 * (T_SIM - T_START))
    ratio = r_I / r_E if r_E > 0 else float('nan')

    trains_E = _extract_spike_trains(objs['spike_E'], 800, T_SIM)
    _, cv = compute_cv_isi(trains_E, T_START, T_SIM, min_spikes=20)
    pc = compute_pairwise_corr(trains_E, T_START, T_SIM, bin_ms=10.0, n_pairs=50, seed=SEED)

    return r_E, cv, pc, ratio


# ---- Run the 2D grid of (nu_ext, g_EI) combinations ----
print(f"Running {len(NU_EXT_VALS) * len(G_EI_VALS)} grid points...")
print(f"{'nu_ext':>7}  {'g_EI(nA)':>9}  {'rate_E':>7}  {'CV':>6}  {'pairr':>7}  {'I/E':>5}")
import sys
sys.stdout.flush()

results = []
for nu in NU_EXT_VALS:
    for gei in G_EI_VALS:
        r_E, cv, pc, ie = run_point(nu, gei)
        ai = (2 <= r_E <= 10) and (not np.isnan(cv)) and (0.8 <= cv <= 1.2) and (not np.isnan(pc)) and (pc < 0.05)
        results.append(dict(nu=nu, gei=gei, r_E=r_E, cv=cv, pc=pc, ie=ie, ai=ai))
        mark = ' AI' if ai else ''
        print(f"{nu:>7.2f}  {gei:>9.3f}  {r_E:>7.2f}  {cv:>6.3f}  {pc:>7.4f}  {ie:>5.1f}{mark}")
        sys.stdout.flush()

# ---- Pack the results into 2D arrays for plotting ----
nu_arr  = np.array(NU_EXT_VALS)
gei_arr = np.array(G_EI_VALS)
rate_grid = np.full((len(nu_arr), len(gei_arr)), np.nan)
cv_grid   = np.full_like(rate_grid, np.nan)
pc_grid   = np.full_like(rate_grid, np.nan)
ai_grid   = np.zeros_like(rate_grid, dtype=bool)

for row in results:
    i = np.argmin(np.abs(nu_arr - row['nu']))
    j = np.argmin(np.abs(gei_arr - row['gei']))
    rate_grid[i, j] = row['r_E']
    cv_grid[i, j]   = row['cv'] if not np.isnan(row['cv']) else 0.0
    pc_grid[i, j]   = row['pc'] if not np.isnan(row['pc']) else 1.0
    ai_grid[i, j]   = row['ai']

# ---- Make the figure ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('AI regime map  (w_scale_II = 0.50, checked in [10–20 s] window, seed=42)', fontsize=11)

ext = [gei_arr[0], gei_arr[-1], nu_arr[0], nu_arr[-1]]

ax = axes[0]
im = ax.imshow(rate_grid, origin='lower', aspect='auto', extent=ext,
               vmin=0, vmax=12, cmap='viridis')
plt.colorbar(im, ax=ax, label='Mean E rate (Hz)')
try:
    ax.contour(gei_arr, nu_arr, rate_grid, levels=[2, 10], colors='white', linewidths=1.5)
except Exception:
    pass
ax.plot(OP_GEI, OP_NU, 'r*', ms=14, label='Operating point')
ax.set_xlabel('g_EI (nA)')
ax.set_ylabel('nu_ext (Hz)')
ax.set_title('Mean E rate\nWhite: 2 Hz, 10 Hz contours')
ax.legend(loc='lower right', fontsize=8)

ax = axes[1]
cv_plot = np.where(np.isnan(cv_grid), 0.0, cv_grid)
im = ax.imshow(cv_plot, origin='lower', aspect='auto', extent=ext,
               vmin=0.5, vmax=1.5, cmap='plasma')
plt.colorbar(im, ax=ax, label='Mean CV-ISI')
try:
    ax.contour(gei_arr, nu_arr, cv_plot, levels=[0.8, 1.2], colors='white', linewidths=1.5)
except Exception:
    pass
ax.plot(OP_GEI, OP_NU, 'r*', ms=14)
ax.set_xlabel('g_EI (nA)')
ax.set_ylabel('nu_ext (Hz)')
ax.set_title('Mean CV-ISI\nWhite: 0.8, 1.2 contours')

ax = axes[2]
pc_plot = np.where(np.isnan(pc_grid), 1.0, pc_grid)
im = ax.imshow(pc_plot, origin='lower', aspect='auto', extent=ext,
               vmin=0, vmax=0.3, cmap='Reds')
plt.colorbar(im, ax=ax, label='Pairwise correlation')
try:
    ax.contour(gei_arr, nu_arr, pc_plot, levels=[0.05], colors='black', linewidths=1.5,
               linestyles='--')
except Exception:
    pass
ax.plot(OP_GEI, OP_NU, 'b*', ms=14, label='Operating point')

# What happens under STDP: as w_EE grows by some factor f, the effective
# inhibition g_EI_eff = g_EI_op / f gets smaller (inhibition is relatively
# weaker compared to the now-bigger excitatory weight). At the fixed
# nu_ext = OP_NU, draw vertical lines showing where the operating point would
# move if w_EE grows by factors of 1.25, 1.50, or 2.0.
for f, label in [(1.25, '×1.25 w_EE'), (1.50, '×1.50'), (2.00, '×2.0')]:
    g_eff = OP_GEI / f
    ax.axvline(g_eff, color='blue', lw=1.2, linestyle=':', alpha=0.8)
    ax.text(g_eff + 0.001, OP_NU + 0.3, label, color='blue', fontsize=7, va='bottom')

ax.set_xlabel('g_EI (nA)')
ax.set_ylabel('nu_ext (Hz)')
ax.set_title('Pairwise corr\nBlue dashes: 0.05 boundary\nBlue verticals: STDP trajectory')
ax.legend(loc='upper right', fontsize=8)

fig.tight_layout()
os.makedirs(RESULTS_DIR, exist_ok=True)
out = os.path.join(RESULTS_DIR, 'figures', 'ai_regime_map.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nHeatmap saved to {out}")

# ---- Print a summary table of the headroom ----
print("\n--- STDP headroom at operating point (nu_ext=6.25, g_EI=0.075, w_EE=0.06) ---")
print("w_EE growth (f)  |  effective g_EI/w_EE  |  g_EI_eff (nA)  |  in AI corridor?")
print("-"*75)

# For each growth factor f, work out what g_EI_eff = 0.075 / f corresponds to
# in our scan, and look up the closest scanned point.
for f in [1.0, 1.1, 1.25, 1.5, 2.0, 3.0]:
    g_eff = OP_GEI / f
    ratio = g_eff / 0.06  # g_EI_eff divided by w_EE
    # Find the closest scan point at nu_ext = 6.25
    i_nu = np.argmin(np.abs(nu_arr - 6.25))
    j_g  = np.argmin(np.abs(gei_arr - g_eff))
    scanned_g = gei_arr[j_g]
    r = rate_grid[i_nu, j_g]
    cv = cv_grid[i_nu, j_g]
    pc = pc_grid[i_nu, j_g]
    ai = ai_grid[i_nu, j_g]
    ai_str = '✓ AI' if ai else '✗ exits AI'
    print(f"  f = {f:.2f}    g_eff = {g_eff:.3f} nA    ratio = {ratio:.2f}    "
          f"(scan: g={scanned_g:.3f}, r={r:.1f}Hz, CV={cv:.3f}, pc={pc:.4f})  {ai_str}")
