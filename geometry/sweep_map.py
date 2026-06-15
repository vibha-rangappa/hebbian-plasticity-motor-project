# geometry/sweep_map.py

"""
This script builds the "plasticity parameters -> geometry results" map for Q2, using the
output of a parameter sweep.

For each sweep run, it reads the inhibitory plasticity parameters (saved as HDF5
metadata/provenance when the run was trained) and joins them with the geometry results
for that run (from the tidy geometry metrics CSV). The output is:

  - a tidy CSV: one row per (run, epoch), with both the plasticity parameter values and
    the geometry observable values
  - heatmaps showing one observable (PR_exec by default) across the (rho0, eta_istdp)
    grid at the final training epoch, with separate panels for each combination of
    tau_istdp and whether E->E plasticity is on or off

Usage (after running the sweep and the geometry analysis on its output):
    PYTHONPATH=. python geometry/sweep_map.py \
        --sweep_dir plasticity/results_sweep \
        --geometry_csv geometry/results/geometry_metrics_sweep.csv
"""

import argparse
import csv
import glob
import os
import re

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = 'geometry/results/figures'


def read_run_coords(sweep_dir):
    """For each run, read its inhibitory plasticity parameters from /training_params attrs in its HDF5 file. Returns a dict mapping run label to a dict of those parameter values."""
    coords = {}
    for path in sorted(glob.glob(os.path.join(sweep_dir, 'training_*.h5'))):
        label = re.match(r'training_(.+)\.h5', os.path.basename(path)).group(1)
        with h5py.File(path, 'r') as f:
            a = f['training_params'].attrs
            coords[label] = {
                'rho0': float(a['rho0']) if 'rho0' in a else np.nan,
                'eta_istdp': float(a['eta_istdp']) if 'eta_istdp' in a else np.nan,
                'tau_istdp': float(a['tau_istdp']) if 'tau_istdp' in a else np.nan,
                'ee_plasticity': bool(a['plasticity_on']) if 'plasticity_on' in a else None,
            }
    return coords


def read_geometry(csv_path):
    """Read the geometry metrics CSV into a dict keyed by (condition, epoch, observable, metric, window, k), with the metric value as the value."""
    out = {}
    for r in csv.DictReader(open(csv_path)):
        key = (r['condition'], int(r['epoch']), r['observable'], r['metric'],
               r['window'], str(r['k']))
        out[key] = float(r['value'])
    return out


def build_map(sweep_dir, geometry_csv):
    coords = read_run_coords(sweep_dir)
    geom = read_geometry(geometry_csv)
    epochs = sorted({k[1] for k in geom})
    rows = []
    for label, c in coords.items():
        for ep in epochs:
            pr_exec = geom.get((label, ep, 'pr', 'participation_ratio', 'exec', ''))
            pr_prep = geom.get((label, ep, 'pr', 'participation_ratio', 'prep', ''))
            ortho = geom.get((label, ep, 'orthogonality', 'mean_angle_deg',
                              'prep_vs_exec', '6'))
            jpca = geom.get((label, ep, 'jpca', 'r2_above_shuffle', 'exec', '6'))
            if pr_exec is None:
                continue
            rows.append({**c, 'label': label, 'epoch': ep,
                         'pr_exec': pr_exec, 'pr_prep': pr_prep,
                         'ortho_deg': ortho, 'jpca_above_shuffle': jpca})
    return rows


def write_map_csv(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cols = ['label', 'rho0', 'eta_istdp', 'tau_istdp', 'ee_plasticity', 'epoch',
            'pr_exec', 'pr_prep', 'ortho_deg', 'jpca_above_shuffle']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    return out_csv


def plot_heatmaps(rows, metric='pr_exec'):
    """Make a heatmap of `metric` over (rho0, eta_istdp) at the final epoch, with a
    separate panel for each combination of (tau_istdp, ee_plasticity)."""
    final_ep = max(r['epoch'] for r in rows)
    sel = [r for r in rows if r['epoch'] == final_ep]
    taus = sorted({r['tau_istdp'] for r in sel})
    ees = sorted({r['ee_plasticity'] for r in sel})
    rho0s = sorted({r['rho0'] for r in sel})
    etas = sorted({r['eta_istdp'] for r in sel})

    ncol = max(len(taus) * len(ees), 1)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4), squeeze=False)
    col = 0
    for ee in ees:
        for tau in taus:
            ax = axes[0][col]
            grid = np.full((len(etas), len(rho0s)), np.nan)
            for r in sel:
                if r['tau_istdp'] == tau and r['ee_plasticity'] == ee and r[metric] is not None:
                    grid[etas.index(r['eta_istdp']), rho0s.index(r['rho0'])] = r[metric]
            im = ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis')
            ax.set_xticks(range(len(rho0s))); ax.set_xticklabels([f"{x:g}" for x in rho0s])
            ax.set_yticks(range(len(etas))); ax.set_yticklabels([f"{x*1e12:g}" for x in etas])
            ax.set_xlabel('rho0 (Hz)'); ax.set_ylabel('eta_istdp (x1e-12 A)')
            ax.set_title(f"{metric}\ntau={tau*1e3:g}ms, EE={'on' if ee else 'off'}")
            fig.colorbar(im, ax=ax, fraction=0.046)
            col += 1
    fig.suptitle(f'{metric} over inhibitory parameter space (epoch {final_ep})')
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f'sweep_map_{metric}.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="Build the plasticity->geometry map (Q2)")
    ap.add_argument('--sweep_dir', default='plasticity/results_sweep')
    ap.add_argument('--geometry_csv', default='geometry/results/geometry_metrics_sweep.csv')
    ap.add_argument('--out_csv', default='geometry/results/sweep_map.csv')
    ap.add_argument('--metrics', nargs='+', default=['pr_exec', 'ortho_deg'])
    args = ap.parse_args()

    rows = build_map(args.sweep_dir, args.geometry_csv)
    if not rows:
        raise SystemExit("No joined rows -- check sweep_dir and geometry_csv.")
    write_map_csv(rows, args.out_csv)
    print(f"[map] wrote {len(rows)} rows -> {args.out_csv}")
    for m in args.metrics:
        print(f"[map] heatmap -> {plot_heatmaps(rows, metric=m)}")


if __name__ == '__main__':
    main()
