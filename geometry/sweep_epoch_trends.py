# geometry/sweep_epoch_trends.py

"""
This script makes figures showing how participation ratio (PR) and orthogonality change
over training, using the results of the 96-point iSTDP sweep (with weight normalization
turned on).

It replaces older figures (pr_vs_epoch.png / orthogonality_vs_epoch.png from Part 2) that
were made before weight normalization existed. Those older figures only had two
conditions and showed a near-straight-line drift over training, which turned out to just
be an artifact of E->E weights growing without bound.

This script groups the 96 sweep conditions by rho0 (the main axis that matters for Q2,
see geometry/sweep_map.py), and for each snapshot epoch it averages PR and orthogonality
over the other three sweep axes (eta_istdp, tau_istdp, and whether E->E plasticity is on
or off), plotting the mean with a shaded band of +/- 1 standard deviation. This makes the
rho0-gated pattern easy to see directly: at low rho0 the trajectories stay flat near the
starting baseline, while at high rho0 they move a lot over the course of training.

Usage:
    PYTHONPATH=. python geometry/sweep_epoch_trends.py \
        --csv geometry/results/geometry_metrics_sweep_full.csv
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = 'geometry/results/figures'


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    rho0_of = {}
    for r in rows:
        cond = r['condition']
        if cond not in rho0_of:
            rho0_of[cond] = float(cond.split('_')[0].replace('rho', ''))
    return rows, rho0_of


def series_by_rho0(rows, rho0_of, observable, metric, window, k):
    """Build a lookup from rho0 to {epoch: [values from the 24 (eta, tau, ee) conditions at that rho0]}."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if (r['observable'] == observable and r['metric'] == metric and
                r['window'] == window and str(r['k']) == str(k)):
            out[rho0_of[r['condition']]][int(r['epoch'])].append(float(r['value']))
    return out


def plot_by_rho0(ax, data, epochs, ylabel, title):
    rho0s = sorted(data.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(rho0s)))
    for rho0, color in zip(rho0s, colors):
        means = np.array([np.mean(data[rho0][ep]) for ep in epochs])
        stds = np.array([np.std(data[rho0][ep]) for ep in epochs])
        ax.plot(epochs, means, 'o-', color=color, label=f'rho0={rho0:g} Hz')
        ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.15)
    ax.set_xlabel('training trials (epoch)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description="Sweep-based epoch-trend figures (Q1/Q2)")
    ap.add_argument('--csv', default='geometry/results/geometry_metrics_sweep_full.csv')
    args = ap.parse_args()

    rows, rho0_of = load(args.csv)
    epochs = sorted({int(r['epoch']) for r in rows})

    # --- PR (prep and exec), grouped by rho0 ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, window in zip(axes, ('prep', 'exec')):
        data = series_by_rho0(rows, rho0_of, 'pr', 'participation_ratio', window, '')
        plot_by_rho0(ax, data, epochs,
                      'participation ratio' if window == 'prep' else '',
                      f'PR ({window})')
    axes[1].legend(fontsize=8)
    fig.suptitle('Effective dimensionality vs training, by rho0 (E-rate setpoint)\n'
                  '96-pt iSTDP sweep, weight-normalized; mean +/-1 SD over '
                  '(eta_istdp, tau_istdp, E->E on/off)')
    fig.tight_layout()
    out1 = f'{FIG_DIR}/pr_vs_epoch.png'
    fig.savefig(out1, dpi=130)
    plt.close(fig)

    # --- orthogonality, grouped by rho0 ---
    fig, ax = plt.subplots(figsize=(7.5, 5))
    data = series_by_rho0(rows, rho0_of, 'orthogonality', 'mean_angle_deg',
                           'prep_vs_exec', 6)
    plot_by_rho0(ax, data, epochs, 'mean principal angle (deg)',
                  'Prep/exec subspace angle vs training, by rho0 (E-rate setpoint)\n'
                  '96-pt iSTDP sweep, weight-normalized -- low rho0 stays near the\n'
                  'pre-training baseline, high rho0 moves toward orthogonal')

    null_means = [float(r['value']) for r in rows
                   if r['observable'] == 'orthogonality' and r['metric'] == 'null_mean_deg'
                   and r['window'] == 'prep_vs_exec' and str(r['k']) == '6']
    null_stds = [float(r['value']) for r in rows
                  if r['observable'] == 'orthogonality' and r['metric'] == 'null_std_deg'
                  and r['window'] == 'prep_vs_exec' and str(r['k']) == '6']
    null_mean, null_std = np.mean(null_means), np.mean(null_stds)
    ax.axhline(90, ls='--', color='k', lw=1, label='orthogonal (90 deg)')
    ax.axhspan(null_mean - 2 * null_std, null_mean + 2 * null_std, color='gray',
               alpha=0.25, label='random-subspace null (+/-2 sd)')
    ax.set_ylim(0, 95)
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    out2 = f'{FIG_DIR}/orthogonality_vs_epoch.png'
    fig.savefig(out2, dpi=130)
    plt.close(fig)

    print(f'[viz] wrote {out1}')
    print(f'[viz] wrote {out2}')


if __name__ == '__main__':
    main()
