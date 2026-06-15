# circuit/grid_search.py

"""
This script sweeps two parameters of the balanced excitatory/inhibitory (E/I)
network to find a good "asynchronous irregular" (AI) operating point, the
firing regime real cortical networks are thought to sit in (not synchronized,
not silent, somewhat irregular spike timing).

It scans every combination of nu_ext (external input rate) and g_EI
(inhibitory weight) over a 5x6 grid (30 combinations total), running a 1
second simulation for each one. Results are saved as a CSV table and as a
two-panel heatmap image so you can see at a glance which combinations land in
the AI regime.

How to use it:
    python circuit/grid_search.py
    # then look at circuit/results/tuning_heatmap.png
    # pick (nu_ext, g_EI) from the region where both of these are true:
    #       mean_rate_E is between 2 and 10 Hz
    #       mean_CV_ISI is between 0.8 and 1.2
    # then run: python circuit/run_baseline.py --nu_ext <value> --g_EI <value>
"""

import os
import csv
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import second

from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# ------------------------------------------------------------------
# Grid definition
# nu_ext_grid: 5 values from 5 to 25 Hz
# g_EI_scale:  6 scale factors applied to w_mean_EE
# Total: 30 grid points
# ------------------------------------------------------------------
NU_EXT_VALS = np.linspace(10, 60, 5)          # input rate per synapse, in Hz; the effective total rate is N_ext x nu_ext, about 800-4800 Hz
G_EI_SCALES = np.linspace(2.0, 8.0, 6)        # how many times w_mean_EE to use for the inhibitory weight; the classic Brunel AI regime is roughly 2-8x
W_MEAN_EE   = DEFAULT_PARAMS['w_mean_EE']      # amps
G_EI_VALS   = G_EI_SCALES * W_MEAN_EE         # amps

# AI-regime boundaries (used to draw contour lines on the heatmap)
RATE_MIN, RATE_MAX = 2.0, 10.0   # Hz
CV_MIN,   CV_MAX   = 0.8,  1.2


def run_grid_point(nu_ext: float, g_EI: float) -> tuple:
    """
    Build the network with these two parameter values, run it for 1 second,
    and return (mean_rate_E_hz, mean_CV_ISI).

    We use min_spikes=5 for the CV-ISI calculation because the run is only
    1 second long. Neurons firing around 5 Hz will only manage a handful of
    spikes in that time, so 5 is the lowest threshold that still gives a
    rough estimate of how irregular the spiking is.
    """
    params = {**DEFAULT_PARAMS, 'nu_ext': nu_ext, 'g_EI': g_EI}
    objs = build_network(params, seed=42)
    objs['net'].run(1.0 * second)

    Ne = params['N_exc']
    T_sim = 1.0  # seconds, matches the net.run duration above
    mean_rate_E = objs['spike_E'].num_spikes / (Ne * T_sim)

    trains_E = _extract_spike_trains(objs['spike_E'], Ne, 1.0)
    _, mean_cv = compute_cv_isi(trains_E, 0.0, 1.0, min_spikes=5)

    return mean_rate_E, mean_cv


def save_csv(results: list, path: str) -> None:
    """Write the grid results to a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['nu_ext_hz', 'g_EI_nA', 'mean_rate_E_hz', 'mean_CV_ISI'])
        writer.writeheader()
        writer.writerows(results)


def save_heatmap(results: list, path: str) -> None:
    """
    Make a two-panel heatmap: mean E firing rate on the left, mean CV-ISI on
    the right. White contour lines mark the boundaries of the AI regime.
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

    # A value of -1 means "no CV could be computed" (not enough spikes).
    # Convert those back to NaN so they don't get plotted as real numbers.
    cv_grid[cv_grid == -1] = np.nan

    g_ei_nA = G_EI_VALS / 1e-9  # convert amps to nanoamps for the axis labels

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
    print('  python circuit/run_baseline.py --nu_ext <value> --g_EI <value>')


if __name__ == '__main__':
    main()
