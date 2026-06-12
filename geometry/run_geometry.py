# geometry/run_geometry.py

"""
Driver for Part 3 geometry analysis (spec section 6).

Discovers the training_*.h5 snapshot files, runs the preprocessing chokepoint ONCE per
(condition, epoch), computes all three observables with their matched nulls, and writes a
single tidy/long results table. The same code ingests the full 64 x 4 x 8 sweep with no
change -- epochs and conditions are discovered, never hardcoded.

Usage:
    PYTHONPATH=. python geometry/run_geometry.py
    PYTHONPATH=. python geometry/run_geometry.py --n_shuffle 20 --n_cv 20

Output: geometry/results/geometry_metrics.csv (overwritten each run -- idempotent, no
duplicate rows accumulate).
"""

import argparse
import csv
import glob
import os
import re

import numpy as np

from plasticity.snapshot import load_snapshot
from geometry.preprocessing import preprocess_snapshot, make_X
from geometry.dimensionality import participation_ratio
from geometry.jpca import jpca_analysis
from geometry.orthogonality import (
    top_pc_basis, principal_angles, orthogonality_analysis, random_subspace_null,
)
from geometry.controls import trial_split_indices

DOWNSAMPLE_MS = 10
N_EXC = 800


def discover_snapshots(results_dir):
    """Map condition name -> (h5_path, sorted list of epoch ints), from training_*.h5."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, 'training_*.h5'))):
        cond = re.match(r'training_(.+)\.h5', os.path.basename(path)).group(1)
        import h5py
        with h5py.File(path, 'r') as f:
            epochs = sorted(int(k.split('_')[1]) for k in f['snapshots'].keys())
        out[cond] = (path, epochs)
    return out


def jpca_shuffle_floor(pre, k, n_shuffle, seed):
    """
    jPCA R^2 chance floor: re-form condition averages under permuted trial->direction
    labels (reusing the already-smoothed per-trial rates -- no re-smoothing) and refit.
    Returns (mean, std) of the shuffled r2_skew.
    """
    rng = np.random.default_rng(seed)
    rates = pre['trial_rate_exec']
    labels = pre['trial_labels']
    r2s = []
    for _ in range(n_shuffle):
        perm = rng.permutation(labels)
        X = make_X(rates, perm, pre['R'], pre['r_floor'], pre['n_conditions'])
        r2s.append(jpca_analysis(X, k=k)['r2_skew'])
    return float(np.mean(r2s)), float(np.std(r2s))


def cv_subspace_stability(pre, window, k, n_cv, seed):
    """
    Trial-split overfitting guard: mean principal angle (deg) between the top-k subspaces
    estimated from two disjoint trial folds, averaged over n_cv random splits. A small
    angle means the subspace generalizes across trials; an angle near the random-subspace
    null means it is overfit to trial noise.
    """
    rng = np.random.default_rng(seed)
    rates = pre['trial_rate_prep'] if window == 'prep' else pre['trial_rate_exec']
    labels = pre['trial_labels']
    angles = []
    for _ in range(n_cv):
        idx_a, idx_b = trial_split_indices(labels, rng)
        Xa = make_X(rates, labels, pre['R'], pre['r_floor'], pre['n_conditions'], idx=idx_a)
        Xb = make_X(rates, labels, pre['R'], pre['r_floor'], pre['n_conditions'], idx=idx_b)
        ang = np.degrees(np.mean(principal_angles(top_pc_basis(Xa, k), top_pc_basis(Xb, k))))
        angles.append(ang)
    return float(np.mean(angles))


def analyze_snapshot(h5_path, condition, epoch, n_shuffle, n_cv, k_list, seed):
    """Compute every observable for one snapshot; return a list of tidy row dicts."""
    snap = load_snapshot(h5_path, epoch)
    pre = preprocess_snapshot(snap, n_exc=N_EXC, downsample_ms=DOWNSAMPLE_MS)
    rows = []

    def row(observable, window, k, metric, value):
        rows.append({'condition': condition, 'epoch': epoch, 'window': window,
                     'observable': observable, 'k': k, 'metric': metric, 'value': value})

    # --- Participation ratio (both windows) ---
    row('pr', 'prep', '', 'participation_ratio', participation_ratio(pre['X_prep']))
    row('pr', 'exec', '', 'participation_ratio', participation_ratio(pre['X_exec']))

    # --- jPCA (exec window) at each k, with full triangulation ---
    for k in k_list:
        j = jpca_analysis(pre['X_exec'], k=k)
        sh_mean, sh_std = jpca_shuffle_floor(pre, k, n_shuffle, seed + k)
        freq_hz = j['omega'] / (2 * np.pi) * (1000.0 / DOWNSAMPLE_MS)
        for m, v in [('r2_skew', j['r2_skew']), ('r2_full', j['r2_full']),
                     ('omega_rad_per_bin', j['omega']), ('freq_hz', freq_hz),
                     ('direction_consistency', j['direction_consistency']),
                     ('mean_tangling', j['mean_tangling']),
                     ('shuffle_r2_mean', sh_mean), ('shuffle_r2_std', sh_std),
                     ('r2_above_shuffle', j['r2_skew'] - sh_mean)]:
            row('jpca', 'exec', k, m, v)

    # --- Prep/exec subspace orthogonality (k=6) ---
    o = orthogonality_analysis(pre['X_prep'], pre['X_exec'], k=6, n_boot=1000, seed=seed)
    for m, v in [('mean_angle_deg', o['mean_angle_deg']),
                 ('null_mean_deg', o['null_mean_deg']),
                 ('null_std_deg', o['null_std_deg']),
                 ('z_vs_null', o['z_vs_null'])]:
        row('orthogonality', 'prep_vs_exec', 6, m, v)

    # --- Trial-split CV stability (overfitting guard), k=6, both windows ---
    null_ang = np.degrees(random_subspace_null(N_EXC, k=6, n_boot=200, seed=seed).mean())
    for window in ('prep', 'exec'):
        cv_ang = cv_subspace_stability(pre, window, k=6, n_cv=n_cv, seed=seed)
        row('cv_stability', window, 6, 'fold_subspace_angle_deg', cv_ang)
        row('cv_stability', window, 6, 'random_subspace_angle_deg', null_ang)

    return rows


def main():
    ap = argparse.ArgumentParser(description="Part 3 geometry analysis driver")
    ap.add_argument('--results_dir', default='plasticity/results')
    ap.add_argument('--out', default='geometry/results/geometry_metrics.csv')
    ap.add_argument('--n_shuffle', type=int, default=20, help="jPCA shuffle-floor reps")
    ap.add_argument('--n_cv', type=int, default=20, help="trial-split CV reps")
    ap.add_argument('--k_list', type=int, nargs='+', default=[4, 6, 10])
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    found = discover_snapshots(args.results_dir)
    if not found:
        raise SystemExit(f"No training_*.h5 found in {args.results_dir}")

    all_rows = []
    for cond, (path, epochs) in found.items():
        for epoch in epochs:
            print(f"[geometry] {cond} epoch {epoch} ...", flush=True)
            all_rows += analyze_snapshot(path, cond, epoch, args.n_shuffle,
                                         args.n_cv, args.k_list, args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['condition', 'epoch', 'window',
                                          'observable', 'k', 'metric', 'value'])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"[geometry] wrote {len(all_rows)} rows -> {args.out}")

    _print_summary(all_rows)


def _print_summary(rows):
    """Console summary, foregrounding the epoch-0 seeded-vs-control pipeline check."""
    def get(cond, epoch, observable, metric, window=None, k=6):
        for r in rows:
            if (r['condition'] == cond and r['epoch'] == epoch and
                    r['observable'] == observable and r['metric'] == metric and
                    (window is None or r['window'] == window) and
                    (r['k'] == k or r['k'] == '')):
                return r['value']
        return None

    print("\n==== summary ====")
    conds = sorted({r['condition'] for r in rows})
    epochs = sorted({r['epoch'] for r in rows})
    for cond in conds:
        print(f"\n-- {cond} --")
        for ep in epochs:
            pr_e = get(cond, ep, 'pr', 'participation_ratio', window='exec', k='')
            r2 = get(cond, ep, 'jpca', 'r2_skew', k=6)
            r2sh = get(cond, ep, 'jpca', 'r2_above_shuffle', k=6)
            tang = get(cond, ep, 'jpca', 'mean_tangling', k=6)
            dirc = get(cond, ep, 'jpca', 'direction_consistency', k=6)
            ang = get(cond, ep, 'orthogonality', 'mean_angle_deg', window='prep_vs_exec', k=6)
            angnull = get(cond, ep, 'orthogonality', 'null_mean_deg', window='prep_vs_exec', k=6)
            def fmt(x):
                return f"{x:6.3f}" if isinstance(x, float) else str(x)
            print(f"  epoch {ep:>3}: PR_exec={fmt(pr_e)}  jPCA_R2={fmt(r2)} "
                  f"(above_shuffle={fmt(r2sh)})  tangling={fmt(tang)} "
                  f"dir_consist={fmt(dirc)}  ortho={fmt(ang)}deg (null={fmt(angnull)})")

    print("\n==== epoch-0 pipeline check (seeded vs control, learning-independent) ====")
    if 'seeded' in conds and 'control' in conds and 0 in epochs:
        for metric, obs, win, k in [('participation_ratio', 'pr', 'exec', ''),
                                    ('mean_angle_deg', 'orthogonality', 'prep_vs_exec', 6)]:
            s = get('seeded', 0, obs, metric, window=win, k=k)
            c = get('control', 0, obs, metric, window=win, k=k)
            print(f"  {obs}/{metric}: seeded={s:.3f}  control={c:.3f}  diff={s - c:+.3f}")


if __name__ == '__main__':
    main()
