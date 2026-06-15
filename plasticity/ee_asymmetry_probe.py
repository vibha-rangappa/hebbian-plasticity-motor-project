# plasticity/ee_asymmetry_probe.py

"""
This script tests whether the balance of E->E STDP potentiation vs depression
matters for the jPCA rotation result.

So far, every jPCA result we've gotten is null (no above-shuffle rotation, no
trend across training epochs), across all 4 combinations tried: {sustained,
autonomous} exec mode x {E->E STDP alone, E->E + iSTDP}. All of those runs used
the DEFAULT E->E STDP amounts (A_plus=2.4e-12 A, A_minus=2.52e-12 A, a pair where
depression is about 5% stronger than potentiation, which is the Song/Miller/Abbott
2000 condition for keeping an UNNORMALIZED network stable).

Our working idea for why jPCA rotation might appear (rotation coming from chain-like
motifs that STDP builds into the connectivity, similar to Ocker/Litwin-Kumar/Doiron
style motif expansion) depends on the *asymmetry* of the STDP rule: how much more
strongly "pre fires before post" strengthens a synapse compared to how strongly
"post fires before pre" weakens it. The default 5%-depression-dominant setting is
nearly balanced on this axis, so this idea has never actually been tested.

This script keeps tau_plus = tau_minus = 20 ms and keeps the total plasticity
"budget" A_plus + A_minus fixed. That way we can change the LTP/LTD *balance*
without changing the overall learning-rate size. It sweeps the ratio
A_plus / A_minus from depression-dominant to potentiation-dominant. Weight
normalization plus iSTDP already prevent runaway weight growth (the Song/Abbott
condition exists to protect an UNNORMALIZED network), so it's safe here to push
the ratio past 1 (potentiation-dominant).

Every run uses --exec_mode autonomous --inhibitory_plasticity on, at the M1-like
iSTDP operating point (rho0=3.0 Hz, eta_istdp=1e-12 A, tau_istdp=20 ms). Per
center_out_task.py's autonomous-mode docstring, this is the one combination (of
the 4 tried) where rotational dynamics could show up at all. The default ratio
(about 0.952) was already run and saved as
plasticity/results_autonomous_istdp/training_autonomous_istdp.h5, so it is not
repeated here.

Each grid point runs as its own `train.py` subprocess, so each gets a clean Brian2
state (no leftover state from a previous run). Runs can be resumed: if a point's
output .h5 file already exists, that point is skipped.

Usage:
    PYTHONPATH=. python plasticity/ee_asymmetry_probe.py
    # then:
    PYTHONPATH=. python geometry/run_geometry.py \
        --results_dir plasticity/results_ee_asymmetry_probe \
        --out geometry/results/geometry_metrics_ee_asymmetry_probe.csv
"""

import argparse
import os
import subprocess
import sys

# A_plus + A_minus stays the same for every point (this is the default
# DEFAULT_PARAMS_PLASTICITY sum, 2.4e-12 + 2.52e-12 A). Keeping the sum fixed while
# changing the A_plus/A_minus ratio lets us vary the LTP/LTD balance without also
# changing the overall STDP learning-rate size.
BUDGET = 2.4e-12 + 2.52e-12

# A_plus/A_minus ratios to test: 0.5 (depression twice as strong as potentiation)
# up to 4.0 (potentiation four times as strong as depression).
RATIOS = [0.5, 1.0, 2.0, 4.0]

RHO0 = 3.0
ETA_ISTDP = 1.0e-12
TAU_ISTDP = 20e-3


def point_label(ratio):
    """Make a short, file-name-safe label that records the A_plus/A_minus ratio."""
    return f"eeasym_r{ratio:g}"


def amplitudes_for_ratio(ratio):
    """Given an A_plus/A_minus ratio, return (A_plus, A_minus) in amps, keeping
    their sum equal to BUDGET."""
    a_plus = ratio / (1.0 + ratio) * BUDGET
    a_minus = 1.0 / (1.0 + ratio) * BUDGET
    return a_plus, a_minus


def run_point(ratio, results_dir, n_per_direction, snapshot_epochs):
    label = point_label(ratio)
    out_h5 = os.path.join(results_dir, f"training_{label}.h5")
    if os.path.exists(out_h5):
        return label, 'skipped (exists)'

    a_plus, a_minus = amplitudes_for_ratio(ratio)
    cmd = [
        sys.executable, 'plasticity/train.py',
        '--condition', 'seeded',
        '--exec_mode', 'autonomous',
        '--inhibitory_plasticity', 'on',
        '--weight_norm', 'on',
        '--rho0', str(RHO0),
        '--eta_istdp', str(ETA_ISTDP),
        '--tau_istdp', str(TAU_ISTDP),
        '--A_plus', str(a_plus),
        '--A_minus', str(a_minus),
        '--label', label,
        '--results_dir', results_dir,
        '--n_per_direction', str(n_per_direction),
        '--snapshot_epochs', *[str(e) for e in snapshot_epochs],
    ]
    env = dict(os.environ, PYTHONPATH='.')
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        return label, f"FAILED: {proc.stderr.strip().splitlines()[-1:]}"
    return label, 'done'


def main():
    ap = argparse.ArgumentParser(description="E->E STDP LTP/LTD-asymmetry probe")
    ap.add_argument('--results_dir', default='plasticity/results_ee_asymmetry_probe')
    ap.add_argument('--n_per_direction', type=int, default=13)  # 104 trials, pilot scale
    ap.add_argument('--snapshot_epochs', type=int, nargs='+', default=[0, 50, 100])
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    print(f"[ee_asymmetry_probe] {len(RATIOS)} points -> {args.results_dir}")
    for i, ratio in enumerate(RATIOS, 1):
        label, status = run_point(ratio, args.results_dir, args.n_per_direction,
                                   args.snapshot_epochs)
        a_plus, a_minus = amplitudes_for_ratio(ratio)
        print(f"[ee_asymmetry_probe] ({i}/{len(RATIOS)}) {label} "
              f"(A_plus={a_plus:.3e}, A_minus={a_minus:.3e}): {status}", flush=True)


if __name__ == '__main__':
    main()
