# geometry/visualize.py

"""
This script makes the main figures for the Part 3 geometry analysis. It reads the tidy
metrics CSV produced by the geometry pipeline, and for the jPC plane figure it also
re-runs the preprocessing step on one training snapshot. All figures are saved as PNGs to
geometry/results/figures/.

    PYTHONPATH=. python geometry/visualize.py

It produces:
    pr_vs_epoch.png            participation ratio (prep and exec) plotted against
                               training epoch
    orthogonality_vs_epoch.png prep/exec mean principal angle plotted against epoch,
                               along with the random-subspace null band and a 90-degree
                               (orthogonal) reference line. This is the figure that shows
                               whether the network's prep and exec activity stay aligned
                               or become orthogonal over training.
    jpc_plane_<cond>_ep<N>.png exec-epoch trajectories for each reach direction,
                               projected onto the dominant jPC plane
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display available, just write figure files to disk
import matplotlib.pyplot as plt

from plasticity.snapshot import load_snapshot
from geometry.preprocessing import preprocess_snapshot
from geometry.jpca import project_pcs, state_and_derivative, fit_skew, dominant_plane

FIG_DIR = 'geometry/results/figures'


def _load_rows(csv_path):
    return list(csv.DictReader(open(csv_path)))


def _series(rows, condition, observable, metric, window=None, k=''):
    """Pull out one metric's values across epochs and return (epochs, values), sorted by epoch."""
    pts = []
    for r in rows:
        if (r['condition'] == condition and r['observable'] == observable and
                r['metric'] == metric and (window is None or r['window'] == window) and
                str(r['k']) == str(k)):
            pts.append((int(r['epoch']), float(r['value'])))
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def plot_pr_vs_epoch(rows, conditions):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, window in zip(axes, ('prep', 'exec')):
        for cond in conditions:
            ep, pr = _series(rows, cond, 'pr', 'participation_ratio', window=window, k='')
            ax.plot(ep, pr, 'o-', label=cond)
        ax.set_title(f'PR ({window})')
        ax.set_xlabel('training trials (epoch)')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('participation ratio')
    axes[0].legend()
    fig.suptitle('Effective dimensionality vs learning')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'pr_vs_epoch.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_orthogonality_vs_epoch(rows, conditions):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    null_mean = null_std = None
    for cond in conditions:
        ep, ang = _series(rows, cond, 'orthogonality', 'mean_angle_deg',
                          window='prep_vs_exec', k=6)
        ax.plot(ep, ang, 'o-', label=cond)
        if null_mean is None:
            _, nm = _series(rows, cond, 'orthogonality', 'null_mean_deg',
                            window='prep_vs_exec', k=6)
            _, ns = _series(rows, cond, 'orthogonality', 'null_std_deg',
                            window='prep_vs_exec', k=6)
            null_mean, null_std = nm.mean(), ns.mean()
    ax.axhline(90, ls='--', color='k', lw=1, label='orthogonal (90 deg)')
    if null_mean is not None:
        ax.axhspan(null_mean - 2 * null_std, null_mean + 2 * null_std,
                   color='gray', alpha=0.25, label='random-subspace null (+/-2 sd)')
    ax.set_xlabel('training trials (epoch)')
    ax.set_ylabel('mean principal angle (deg)')
    ax.set_title('Prep/exec subspace angle vs learning\n(aligned, not orthogonalizing)')
    ax.set_ylim(0, 95)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'orthogonality_vs_epoch.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_jpc_plane(h5_path, condition, epoch, n_exc=800):
    """Project the exec-epoch trajectories for each condition onto the dominant jPC plane and plot them."""
    snap = load_snapshot(h5_path, epoch)
    pre = preprocess_snapshot(snap, n_exc=n_exc)
    Z, _ = project_pcs(pre['X_exec'], k=6)
    X_all, dX_all = state_and_derivative(Z)
    M = fit_skew(X_all, dX_all)
    _, plane = dominant_plane(M)

    C = Z.shape[1]
    cmap = plt.cm.hsv(np.linspace(0, 1, C, endpoint=False))
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for c in range(C):
        p = Z[:, c, :] @ plane
        ax.plot(p[:, 0], p[:, 1], '-', color=cmap[c], lw=1.4)
        ax.plot(p[0, 0], p[0, 1], 'o', color=cmap[c], ms=5)   # mark the starting point of the trajectory
    ax.set_aspect('equal')
    ax.set_xlabel('jPC 1'); ax.set_ylabel('jPC 2')
    ax.set_title(f'Dominant jPC plane: {condition} epoch {epoch}\n'
                 f'(dots = trajectory start; color = reach direction)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f'jpc_plane_{condition}_ep{epoch}.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="Part 3 geometry figures")
    ap.add_argument('--csv', default='geometry/results/geometry_metrics.csv')
    ap.add_argument('--results_dir', default='plasticity/results')
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    rows = _load_rows(args.csv)
    conditions = sorted({r['condition'] for r in rows})

    written = [plot_pr_vs_epoch(rows, conditions),
               plot_orthogonality_vs_epoch(rows, conditions)]

    # Make the jPC plane figure at the last epoch for each condition.
    epochs = sorted({int(r['epoch']) for r in rows})
    last = epochs[-1]
    for cond in conditions:
        h5_path = os.path.join(args.results_dir, f'training_{cond}.h5')
        if os.path.exists(h5_path):
            written.append(plot_jpc_plane(h5_path, cond, last))

    for w in written:
        print(f"[viz] wrote {w}")


if __name__ == '__main__':
    main()
