# part1/tune_part1.py

"""
2D parameter grid search to find the AI-regime operating point for Part 1.

Scans (nu_ext, g_EI) over a 5×6 grid (30 points), running a 1 s simulation
per point. Saves results to CSV and a two-panel heatmap.

Usage:
    python part1/tune_part1.py
    # → inspect part1/results/tuning_heatmap.png
    # → pick (nu_ext, g_EI) from the overlap region where:
    #       mean_rate_E in [2, 10] Hz  AND  mean_CV_ISI in [0.8, 1.2]
    # → run: python part1/run_part1.py --nu_ext <value> --g_EI <value>
"""

import os
import csv
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import second

from part1.network import build_network, DEFAULT_PARAMS
from part1.run_part1 import compute_cv_isi, _extract_spike_trains

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# ------------------------------------------------------------------
# Grid definition
# nu_ext_grid: 5 values from 5 to 25 Hz
# g_EI_scale:  6 scale factors applied to w_mean_EE
# Total: 30 grid points
# ------------------------------------------------------------------
NU_EXT_VALS = np.linspace(5, 25, 5)           # Hz (per external synapse; total rate = N_ext × nu_ext)
G_EI_SCALES = np.linspace(2.0, 8.0, 6)        # × w_mean_EE; Brunel AI regime ≈ 2–8×
W_MEAN_EE   = DEFAULT_PARAMS['w_mean_EE']      # A
G_EI_VALS   = G_EI_SCALES * W_MEAN_EE         # A

# AI-regime boundaries (for contour overlays on heatmap)
RATE_MIN, RATE_MAX = 2.0, 10.0   # Hz
CV_MIN,   CV_MAX   = 0.8,  1.2


def run_grid_point(nu_ext: float, g_EI: float) -> tuple:
    """
    Build network, run 1 s, return (mean_rate_E_hz, mean_CV_ISI).

    Uses min_spikes=5 for CV-ISI because we only run 1 s — neurons at
    ~5 Hz fire just enough spikes for a rough CV estimate.
    """
    params = {**DEFAULT_PARAMS, 'nu_ext': nu_ext, 'g_EI': g_EI}
    objs = build_network(params, seed=42)
    objs['net'].run(1.0 * second)

    Ne = params['N_exc']
    T_sim = 1.0  # seconds — matches net.run duration above
    mean_rate_E = objs['spike_E'].num_spikes / (Ne * T_sim)

    trains_E = _extract_spike_trains(objs['spike_E'], Ne, 1.0)
    _, mean_cv = compute_cv_isi(trains_E, 0.0, 1.0, min_spikes=5)

    return mean_rate_E, mean_cv


def save_csv(results: list, path: str) -> None:
    """Write grid results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['nu_ext_hz', 'g_EI_nA', 'mean_rate_E_hz', 'mean_CV_ISI'])
        writer.writeheader()
        writer.writerows(results)


def save_heatmap(results: list, path: str) -> None:
    """
    Two-panel heatmap: mean E firing rate (left) and mean CV-ISI (right).
    White contours mark the AI-regime boundaries.
    """
    n_nu  = len(NU_EXT_VALS)
    n_gei = len(G_EI_VALS)

    rate_grid = np.full((n_nu, n_gei), np.nan)
    cv_grid   = np.full((n_nu, n_gei), np.nan)

    for row in results:
        i = np.argmin(np.abs(NU_EXT_VALS - row['nu_ext_hz']))
        j = np.argmin(np.abs(G_EI_VALS / 1e-9 - row['g_EI_nA']))
        rate_grid[i, j] = row['mean_rate_E_hz']
        cv_grid[i, j]   = row['mean_CV_ISI']

    # -1 encodes NaN CV (no qualifying spikes) — convert back before plotting
    cv_grid[cv_grid == -1] = np.nan

    g_ei_nA = G_EI_VALS / 1e-9  # A → nA for axis labels

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(rate_grid, origin='lower', aspect='auto',
                   extent=[g_ei_nA[0], g_ei_nA[-1], NU_EXT_VALS[0], NU_EXT_VALS[-1]],
                   vmin=0, vmax=30, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Mean E rate (Hz)')
    if not np.all(np.isnan(rate_grid)):
        cs = ax.contour(g_ei_nA, NU_EXT_VALS, rate_grid,
                        levels=[RATE_MIN, RATE_MAX], colors='white', linewidths=1.5)
        ax.clabel(cs, fmt='%.0f Hz')
    ax.set_xlabel('g_EI (nA)')
    ax.set_ylabel('nu_ext (Hz)')
    ax.set_title('Mean E firing rate\nWhite contours: 2 Hz, 10 Hz (AI band)')

    ax = axes[1]
    im = ax.imshow(cv_grid, origin='lower', aspect='auto',
                   extent=[g_ei_nA[0], g_ei_nA[-1], NU_EXT_VALS[0], NU_EXT_VALS[-1]],
                   vmin=0, vmax=2, cmap='plasma')
    plt.colorbar(im, ax=ax, label='Mean CV-ISI')
    if not np.all(np.isnan(cv_grid)):
        cs = ax.contour(g_ei_nA, NU_EXT_VALS, cv_grid,
                        levels=[CV_MIN, CV_MAX], colors='white', linewidths=1.5)
        ax.clabel(cs, fmt='%.1f')
    ax.set_xlabel('g_EI (nA)')
    ax.set_ylabel('nu_ext (Hz)')
    ax.set_title('Mean CV-ISI\nWhite contours: 0.8, 1.2 (AI band)\nTarget: overlap with left panel')

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Heatmap saved to {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []
    total = len(NU_EXT_VALS) * len(G_EI_VALS)
    done  = 0

    print(f"Running {total} grid points ({len(NU_EXT_VALS)} nu_ext × "
          f"{len(G_EI_VALS)} g_EI)...")
    print(f"{'nu_ext (Hz)':>12} {'g_EI (nA)':>10} {'rate_E (Hz)':>12} "
          f"{'CV-ISI':>8}")
    print('-' * 46)

    t0 = time.time()
    for nu_ext in NU_EXT_VALS:
        for g_EI in G_EI_VALS:
            rate, cv = run_grid_point(nu_ext, g_EI)
            results.append({
                'nu_ext_hz':      round(float(nu_ext), 2),
                'g_EI_nA':        round(float(g_EI / 1e-9), 4),
                'mean_rate_E_hz': round(float(rate), 3),
                'mean_CV_ISI':    round(float(cv) if not np.isnan(cv) else -1, 4),
            })
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"{nu_ext:>12.1f} {g_EI/1e-9:>10.4f} {rate:>12.3f} "
                  f"{cv:>8.3f}  [{done}/{total}  ETA {eta:.0f}s]")

    csv_path = os.path.join(RESULTS_DIR, 'tuning_results.csv')
    save_csv(results, csv_path)
    print(f"\nCSV saved to {csv_path}")

    heatmap_path = os.path.join(RESULTS_DIR, 'tuning_heatmap.png')
    save_heatmap(results, heatmap_path)

    print('\nNext step: inspect tuning_heatmap.png, find the overlap region '
          'where rate ∈ [2,10] Hz AND CV ∈ [0.8,1.2], then run:')
    print('  python part1/run_part1.py --nu_ext <value> --g_EI <value>')


if __name__ == '__main__':
    main()
