# plasticity/sweep.py

"""
Inhibitory-plasticity parameter sweep.

The Part 3 pilot showed the population geometry (PR, prep/exec orthogonality) is driven by
inhibitory STDP, not E->E STDP. This sweep maps the plasticity-space -> geometry-space
relationship (Q2) over the inhibitory parameters that actually move the geometry:

    rho0       target E rate (the E/I balance setpoint)
    eta_istdp  inhibitory learning rate
    tau_istdp  inhibitory trace window

plus a binary E->E STDP on/off axis to document its null across the inhibitory landscape.

Each grid point is run as an isolated `train.py` subprocess (no Brian2 state bleed) with a
label encoding its coordinates, into a shared results dir. Runs are resumable: a point
whose output .h5 already exists is skipped. Default is serial (--n_jobs 1) to avoid Brian2
cython-cache contention; parallel is available but compiles can collide, so use with care.

Usage:
    PYTHONPATH=. python plasticity/sweep.py --grid quick      # 9-point validation
    PYTHONPATH=. python plasticity/sweep.py --grid full       # full map
    # then:
    PYTHONPATH=. python geometry/run_geometry.py --results_dir plasticity/results_sweep \
        --out geometry/results/geometry_metrics_sweep.csv
"""

import argparse
import itertools
import os
import subprocess
import sys

# Parameter grids. eta in amps, tau in seconds, rho0 in Hz.
GRIDS = {
    # Small validation grid: confirm the inhibitory params produce a readable gradient
    # and estimate per-run cost before committing to the full map.
    'quick': {
        'rho0': [2.0, 3.0, 5.0],
        'eta_istdp': [0.5e-12, 1.0e-12, 2.0e-12],
        'tau_istdp': [20e-3],
        'ee': ['on'],
    },
    # Full map: 4 x 4 x 3 inhibitory grid, crossed with the E->E null axis (on/off).
    'full': {
        'rho0': [1.5, 3.0, 5.0, 8.0],
        'eta_istdp': [0.5e-12, 1.0e-12, 2.0e-12, 4.0e-12],
        'tau_istdp': [10e-3, 20e-3, 40e-3],
        'ee': ['on', 'off'],
    },
}


def point_label(rho0, eta, tau, ee):
    """Filesystem-safe, parseable label encoding a grid point's coordinates."""
    return f"rho{rho0:g}_eta{eta * 1e12:g}_tau{tau * 1e3:g}_ee{ee}"


def grid_points(grid):
    for rho0, eta, tau, ee in itertools.product(
            grid['rho0'], grid['eta_istdp'], grid['tau_istdp'], grid['ee']):
        yield rho0, eta, tau, ee


def run_point(rho0, eta, tau, ee, args):
    label = point_label(rho0, eta, tau, ee)
    out_h5 = os.path.join(args.results_dir, f"training_{label}.h5")
    if os.path.exists(out_h5):
        return label, 'skipped (exists)'

    cmd = [
        sys.executable, 'plasticity/train.py',
        '--condition', 'seeded',
        '--inhibitory_plasticity', 'on',
        '--weight_norm', 'on',
        '--ee_plasticity', ee,
        '--rho0', str(rho0),
        '--eta_istdp', str(eta),
        '--tau_istdp', str(tau),
        '--label', label,
        '--results_dir', args.results_dir,
        '--n_per_direction', str(args.n_per_direction),
        '--snapshot_epochs', *[str(e) for e in args.snapshot_epochs],
    ]
    env = dict(os.environ, PYTHONPATH='.')
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        return label, f"FAILED: {proc.stderr.strip().splitlines()[-1:]}"
    return label, 'done'


def main():
    ap = argparse.ArgumentParser(description="Inhibitory-plasticity parameter sweep")
    ap.add_argument('--grid', choices=list(GRIDS), default='quick')
    ap.add_argument('--results_dir', default='plasticity/results_sweep')
    ap.add_argument('--n_per_direction', type=int, default=13)  # 104 trials, pilot scale
    ap.add_argument('--snapshot_epochs', type=int, nargs='+', default=[0, 50, 100])
    ap.add_argument('--n_jobs', type=int, default=1,
                    help="parallel workers (default 1 serial; >1 risks cython-cache "
                         "collisions)")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    points = list(grid_points(GRIDS[args.grid]))
    print(f"[sweep] {args.grid} grid: {len(points)} points -> {args.results_dir}")

    if args.n_jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(run_point)(*p, args) for p in points)
    else:
        results = []
        for i, p in enumerate(points, 1):
            label, status = run_point(*p, args)
            print(f"[sweep] ({i}/{len(points)}) {label}: {status}", flush=True)
            results.append((label, status))

    done = sum(1 for _, s in results if s == 'done')
    skipped = sum(1 for _, s in results if 'skipped' in s)
    failed = [l for l, s in results if 'FAILED' in s]
    print(f"[sweep] complete: {done} run, {skipped} skipped, {len(failed)} failed")
    for l in failed:
        print(f"  FAILED: {l}")


if __name__ == '__main__':
    main()
