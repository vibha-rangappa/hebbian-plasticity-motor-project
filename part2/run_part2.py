# part2/run_part2.py

"""
Part 2 trial runner, burn-in/training loop, and CLI entry point (spec
sections 2.3-6).
"""

import argparse
import os

import numpy as np
from brian2 import second, ms, amp, Hz, prefs, NeuronGroup, Network

from part1.network import DEFAULT_PARAMS
from part1.run_part1 import compute_cv_isi
from part2.network_part2 import (
    DEFAULT_PARAMS_PART2,
    load_part1_baseline,
    build_stdp_network,
)
from part2.task import (
    rates_for_phase,
    assign_preferred_directions,
    generate_trial_sequence,
    generate_test_trial_sequence,
)
from part2.snapshot import save_snapshot, copy_part1_provenance, save_part2_params


def run_one_trial(net_objs, params, theta_i, theta_cue):
    """
    Run one trial: prep (t_prep) -> exec (t_exec) -> ITI (t_iti) (spec 2.3).

    Sets net_objs['input_group'].rates from rates_for_phase() for each phase
    and advances net_objs['net'] by that phase's duration. After this
    function returns, the input group is left at the ITI (background) rate.
    """
    for phase in ('prep', 'exec', 'iti'):
        rates = rates_for_phase(
            theta_cue, theta_i, phase,
            r_max=params['r_max'],
            r_background=params['r_background'],
            exec_amplification=params['exec_amplification'],
        )
        net_objs['input_group'].rates = rates * Hz
        net_objs['net'].run(params[f't_{phase}'] * second)


def extract_snapshot_spikes(net_objs, t_snapshot_start, params, n_test_trials):
    """
    Extract spikes recorded during [t_snapshot_start, t_snapshot_start +
    n_test_trials * trial_dur) from spike_E and spike_input, and convert
    absolute simulation time to (trial_idx, time_in_trial_ms).

    Input neurons are offset by N_exc, so spike_neuron_idx in the returned
    dict spans 0..N_exc-1 (E) and N_exc..N_exc+n_input-1 (input), per spec
    section 6.

    Returns dict with 'spike_times_ms' (float32, ms within trial),
    'spike_neuron_idx' (int32), 'spike_trial_idx' (int32).
    """
    trial_dur = params['t_prep'] + params['t_exec'] + params['t_iti']
    t_end = t_snapshot_start + n_test_trials * trial_dur

    t_E = np.array(net_objs['spike_E'].t / second)
    i_E = np.array(net_objs['spike_E'].i[:], dtype=np.int64)
    mask_E = (t_E >= t_snapshot_start) & (t_E < t_end)

    t_in = np.array(net_objs['spike_input'].t / second)
    i_in = np.array(net_objs['spike_input'].i[:], dtype=np.int64) + params['N_exc']
    mask_in = (t_in >= t_snapshot_start) & (t_in < t_end)

    times = np.concatenate([t_E[mask_E], t_in[mask_in]])
    neurons = np.concatenate([i_E[mask_E], i_in[mask_in]])

    rel_t = times - t_snapshot_start
    trial_idx = np.floor(rel_t / trial_dur).astype(np.int32)
    # A spike at the exact end of the window would land in trial n_test_trials;
    # clip it back into the last trial.
    trial_idx = np.clip(trial_idx, 0, n_test_trials - 1)
    time_in_trial_ms = ((rel_t - trial_idx * trial_dur) * 1000.0).astype(np.float32)

    return {
        'spike_times_ms': time_in_trial_ms,
        'spike_neuron_idx': neurons.astype(np.int32),
        'spike_trial_idx': trial_idx,
    }


def compute_monitoring_metrics(net_objs, t_snapshot_start, params, n_test_trials):
    """
    Monitoring metrics over the just-completed test-trial window (spec 2.5):
    mean_rate_E, mean_w_EE, frac_w_max, mean_cv_isi.
    """
    trial_dur = params['t_prep'] + params['t_exec'] + params['t_iti']
    t_end = t_snapshot_start + n_test_trials * trial_dur
    duration = t_end - t_snapshot_start

    spike_trains = {k: np.array(v / second)
                     for k, v in net_objs['spike_E'].spike_trains().items()}

    n_spikes = sum(
        int(np.sum((times >= t_snapshot_start) & (times < t_end)))
        for times in spike_trains.values()
    )
    mean_rate_E = n_spikes / (params['N_exc'] * duration)

    w = np.array(net_objs['syn_EE'].w[:] / amp)
    mean_w_EE = float(np.mean(w))
    frac_w_max = float(np.mean(w >= 0.999 * params['w_max']))

    _, mean_cv = compute_cv_isi(spike_trains, t_snapshot_start, t_end, min_spikes=20)

    return {
        'mean_rate_E': float(mean_rate_E),
        'mean_w_EE': mean_w_EE,
        'frac_w_max': frac_w_max,
        'mean_cv_isi': float(mean_cv),
    }


def check_abort_criteria(metrics, epoch):
    """
    Raise RuntimeError if monitoring metrics indicate the run should stop
    (spec 2.5): mean_rate_E > 30 Hz (runaway potentiation) or frac_w_max > 0.5
    (depression insufficient).
    """
    if metrics['mean_rate_E'] > 30.0:
        raise RuntimeError(
            f"Abort at epoch {epoch}: mean_rate_E={metrics['mean_rate_E']:.2f} Hz "
            f"> 30 Hz (possible runaway potentiation)")
    if metrics['frac_w_max'] > 0.5:
        raise RuntimeError(
            f"Abort at epoch {epoch}: frac_w_max={metrics['frac_w_max']:.3f} "
            f"> 0.5 (depression insufficient)")


def run_snapshot(net_objs, h5_path, epoch, test_trial_sequence, theta_i, params,
                  check_abort=True):
    """
    Snapshot protocol (spec 2.4):
    1. Freeze STDP (plastic=0).
    2. Run the fixed test_trial_sequence (40 trials for Phase A: 5/direction).
    3. Record W_EE (COO) and spike data, save to h5_path.
    4. Unfreeze STDP (plastic=1).
    5. Print a one-line summary and, if check_abort, check abort criteria.

    check_abort=False is for tests that use an unrealistic nu_ext to
    guarantee spiking activity in a tiny network — such networks can
    legitimately exceed the 30 Hz / frac_w_max abort thresholds without that
    meaning anything for the real (validated) network run by run_condition().
    """
    syn = net_objs['syn_EE']
    syn.plastic = 0

    t_snapshot_start = net_objs['net'].t / second
    n_test_trials = len(test_trial_sequence)

    for direction_idx in test_trial_sequence:
        theta_cue = 2 * np.pi * direction_idx / params['n_directions']
        run_one_trial(net_objs, params, theta_i, theta_cue)

    spike_data = extract_snapshot_spikes(net_objs, t_snapshot_start, params, n_test_trials)
    metrics = compute_monitoring_metrics(net_objs, t_snapshot_start, params, n_test_trials)

    i_arr = np.array(syn.i[:], dtype=np.int32)
    j_arr = np.array(syn.j[:], dtype=np.int32)
    w_arr = np.array(syn.w[:] / amp, dtype=np.float32)
    W_EE_coo = {
        'data': w_arr,
        'row': j_arr,   # postsynaptic — matches part1 save_baseline convention
        'col': i_arr,   # presynaptic
        'shape': np.array([params['N_exc'], params['N_exc']], dtype=np.int32),
    }

    save_snapshot(h5_path, epoch, W_EE_coo, spike_data, test_trial_sequence, metrics)

    syn.plastic = 1

    print(f"[epoch {epoch:5d}] mean_rate_E={metrics['mean_rate_E']:6.2f} Hz  "
          f"mean_w_EE={metrics['mean_w_EE'] * 1e9:7.4f} nA  "
          f"frac_w_max={metrics['frac_w_max']:.3f}  "
          f"mean_cv_isi={metrics['mean_cv_isi']:.3f}")

    if check_abort:
        check_abort_criteria(metrics, epoch)


def _select_codegen_backend():
    """
    Switch Brian2 to the cython backend for the full validation run (Task 7
    is ~30x slower on numpy). Must be called AFTER importing from
    part1.network, which sets prefs.codegen.target = 'numpy' at import time
    (see part1/network.py module-level prefs assignment).

    Must ALSO be called AFTER load_part1_baseline()/build_network(): their
    `.connect(p=...)` calls and `rand()`-based v init execute immediately
    under whatever codegen target is active, and Brian2's RNG-consumption
    pattern for those calls differs between 'numpy' and 'cython' even with
    the same seed. baseline_network.h5 was generated under 'numpy', so
    reproducing its connectivity only works if build_network() also runs
    under 'numpy'.

    Falls back to numpy if no working C++ compiler is found.
    """
    prefs.codegen.target = 'cython'
    try:
        test_group = NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1', method='euler')
        test_net = Network(test_group)
        test_net.run(0.1 * ms)
    except Exception as exc:
        print(f"Cython codegen unavailable ({exc!r}); falling back to numpy backend.")
        prefs.codegen.target = 'numpy'


def run_condition(net_objs, params, h5_path, theta_i, n_per_direction, snapshot_epochs,
                   seed=42, condition_name='', check_abort=True, test_trial_sequence=None):
    """
    Burn-in, epoch-0 snapshot, then the training loop with periodic snapshots
    (spec sections 3-4).

    snapshot_epochs : set of int — cumulative trial counts at which to run a
        snapshot, e.g. {0, 50, 100}. Epoch 0 is handled before the training
        loop (the loop's `completed = trial_idx + 1` never equals 0).
    test_trial_sequence : the fixed sequence used at every snapshot. Defaults
        to generate_test_trial_sequence() (40 trials, 5/direction). Tests can
        pass a shorter sequence to keep runtime down.
    """
    if test_trial_sequence is None:
        test_trial_sequence = generate_test_trial_sequence()

    trial_sequence = generate_trial_sequence(n_per_direction, params['n_directions'], seed=seed)
    n_trials = len(trial_sequence)

    # --- Burn-in (spec section 3): STDP frozen, input at background rate,
    # settles the V-initialization transient before any snapshot is taken.
    print(f"[{condition_name}] burn-in: {params['t_burn_in']:.1f} s (plastic=0)")
    syn = net_objs['syn_EE']
    syn.plastic = 0
    net_objs['input_group'].rates = np.full(params['n_input'], params['r_background']) * Hz
    net_objs['net'].run(params['t_burn_in'] * second)
    syn.plastic = 1

    # --- Epoch-0 snapshot (before any training trial)
    if 0 in snapshot_epochs:
        run_snapshot(net_objs, h5_path, epoch=0,
                      test_trial_sequence=test_trial_sequence,
                      theta_i=theta_i, params=params, check_abort=check_abort)

    # --- Training loop
    for trial_idx in range(n_trials):
        direction_idx = trial_sequence[trial_idx]
        theta_cue = 2 * np.pi * direction_idx / params['n_directions']
        run_one_trial(net_objs, params, theta_i, theta_cue)

        completed = trial_idx + 1
        if completed in snapshot_epochs:
            run_snapshot(net_objs, h5_path, epoch=completed,
                          test_trial_sequence=test_trial_sequence,
                          theta_i=theta_i, params=params, check_abort=check_abort)

    return net_objs


def main():
    parser = argparse.ArgumentParser(
        description="Part 2 Phase A: STDP + 8-direction center-out task")
    parser.add_argument('--condition', choices=['seeded', 'control'], required=True,
                         help="seeded: p_cross=0.2 (P/X pool seeding); "
                              "control: p_cross=1.0 (uniform initialization)")
    parser.add_argument('--n_per_direction', type=int, default=13,
                         help="Training trials per direction (default 13 -> "
                              "104 total, Phase A scope)")
    parser.add_argument('--snapshot_epochs', type=int, nargs='+', default=[0, 50, 100],
                         help="Cumulative trial counts at which to snapshot")
    # Must match the seed baseline_network.h5 was generated with (seed=7,
    # see part1/results/baseline_network.h5:/validation/seed) so
    # load_part1_baseline()'s connectivity check passes.
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--baseline_h5', type=str,
                         default='part1/results/baseline_network.h5')
    parser.add_argument('--results_dir', type=str, default='part2/results')
    args = parser.parse_args()

    params = {**DEFAULT_PARAMS, **DEFAULT_PARAMS_PART2}
    p_cross = (params['p_cross_seeded'] if args.condition == 'seeded'
               else params['p_cross_control'])

    os.makedirs(args.results_dir, exist_ok=True)
    h5_path = os.path.join(args.results_dir, f'training_{args.condition}.h5')
    if os.path.exists(h5_path):
        os.remove(h5_path)

    # Self-contained output file (spec section 6): provenance from Part 1,
    # plus this run's STDP/task parameters.
    copy_part1_provenance(h5_path, args.baseline_h5)
    save_part2_params(h5_path, params, p_cross=p_cross, seed=args.seed)

    net_objs = load_part1_baseline(args.baseline_h5, params, seed=args.seed)
    net_objs = build_stdp_network(net_objs, params, p_cross=p_cross, seed=args.seed)
    theta_i = assign_preferred_directions(params['n_input'], params['n_directions'])

    # Codegen target is selected only after the network is fully built (see
    # _select_codegen_backend's docstring) so that load_part1_baseline()'s
    # connectivity reproduction matches baseline_network.h5 exactly.
    _select_codegen_backend()

    run_condition(net_objs, params, h5_path, theta_i,
                   n_per_direction=args.n_per_direction,
                   snapshot_epochs=set(args.snapshot_epochs),
                   seed=args.seed, condition_name=args.condition)

    print(f"Done. Wrote {h5_path}")


if __name__ == '__main__':
    main()
