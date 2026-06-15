# plasticity/sweep.py

"""
This script sweeps over the inhibitory plasticity parameters and runs a
training job for each combination.

An earlier pilot run (Part 3) showed that the population-level geometry
(participation ratio, and how separate the "prep" and "exec" subspaces are)
is driven by the inhibitory STDP rule, not by the E->E STDP rule. This sweep
maps out how that geometry changes (Q2) as we vary the inhibitory parameters
that actually seem to matter:

    rho0       the target firing rate for excitatory neurons (sets the E/I balance point)
    eta_istdp  the inhibitory learning rate
    tau_istdp  the time window of the inhibitory STDP trace

It also varies E->E STDP on vs off, just to confirm it stays a null effect
across this whole inhibitory parameter range.

Every run uses --exec_mode autonomous. In 'sustained' mode, the input during
the execution phase clamps the network's state, so jPCA rotation can't show
up at all, by construction (see center_out_task.py). 'autonomous' mode is the
only one where all three measurements (participation ratio, jPCA, subspace
orthogonality) are meaningful, so this sweep covers the full set of questions
(Q1 through Q3), not just participation ratio and orthogonality.

Each grid point runs as its own `train.py` subprocess, so it gets a clean
Brian2 state, with a label encoding its parameter values, writing into a
shared results folder. Runs can be resumed: if a point's output .h5 file
already exists, that point is skipped. By default this runs points one at a
time (--n_jobs 1) to avoid Brian2's compiled-code cache from being written to
by multiple processes at once. Running in parallel is possible but those
cache collisions can happen, so use it carefully.

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

# Parameter grids. eta is in amps, tau is in seconds, rho0 is in Hz.
GRIDS = {
    # A small grid to check that the inhibitory parameters produce a clear,
    # readable trend, and to get a sense of how long each run takes before
    # committing to the full map.
    'quick': {
        'rho0': [2.0, 3.0, 5.0],
        'eta_istdp': [0.5e-12, 1.0e-12, 2.0e-12],
        'tau_istdp': [20e-3],
        'ee': ['on'],
    },
    # The full map: a 4 x 4 x 3 grid over the inhibitory parameters, each
    # combination run with E->E STDP both on and off.
    'full': {
        'rho0': [1.5, 3.0, 5.0, 8.0],
        'eta_istdp': [0.5e-12, 1.0e-12, 2.0e-12, 4.0e-12],
        'tau_istdp': [10e-3, 20e-3, 40e-3],
        'ee': ['on', 'off'],
    },
}


def point_label(rho0, eta, tau, ee):
    """Build a short, file-name-safe label that encodes a grid point's parameter values."""
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
        '--exec_mode', 'autonomous',
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
