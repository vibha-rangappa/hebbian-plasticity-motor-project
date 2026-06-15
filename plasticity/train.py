# plasticity/train.py

"""
This is the main training script. It runs the center-out task on the
STDP network: it runs single trials, handles the burn-in period and the
training loop, takes periodic "snapshots" of the network, and provides the
command-line interface used to launch a training run (e.g. with a chosen
condition, execution mode, and plasticity settings).
"""

import argparse
import os

import numpy as np
from brian2 import second, ms, amp, Hz, prefs, NeuronGroup, Network

from circuit.network import DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi
from plasticity.stdp_network import (
    DEFAULT_PARAMS_PLASTICITY,
    load_baseline,
    build_stdp_network,
    normalize_incoming_weights,
)
from plasticity.center_out_task import (
    rates_for_phase,
    assign_preferred_directions,
    generate_trial_sequence,
    generate_test_trial_sequence,
)
from plasticity.snapshot import save_snapshot, copy_baseline_provenance, save_training_params


def run_one_trial(net_objs, params, theta_i, theta_cue):
    """
    Run one trial, which is three phases back to back: prep (lasting
    t_prep), then exec (t_exec), then the inter-trial interval, ITI (t_iti).

    For each phase, this sets net_objs['input_group'].rates using
    rates_for_phase(), then runs net_objs['net'] forward by that phase's
    duration. By the time this function returns, the input group has been
    left at the ITI (background) rate.
    """
    for phase in ('prep', 'exec', 'iti'):
        rates = rates_for_phase(
            theta_cue, theta_i, phase,
            r_max=params['r_max'],
            r_background=params['r_background'],
            exec_amplification=params['exec_amplification'],
            exec_mode=params.get('exec_mode', 'sustained'),
        )
        net_objs['input_group'].rates = rates * Hz
        net_objs['net'].run(params[f't_{phase}'] * second)


def extract_snapshot_spikes(net_objs, t_snapshot_start, params, n_test_trials):
    """
    Pull out all the spikes that happened during the snapshot window, which
    runs from t_snapshot_start to t_snapshot_start + n_test_trials * trial_dur,
    from both spike_E (excitatory neurons) and spike_input (input neurons).
    Converts each spike's absolute simulation time into a (trial number,
    time within that trial in ms) pair.

    Input neuron indices are shifted up by N_exc, so in the returned
    spike_neuron_idx array, values 0..N_exc-1 mean an excitatory neuron and
    values N_exc..N_exc+n_input-1 mean an input neuron.

    Returns a dict with 'spike_times_ms' (float32, time in ms within the
    trial), 'spike_neuron_idx' (int32), and 'spike_trial_idx' (int32).
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
    # A spike that lands exactly at the end of the window would otherwise be
    # counted as belonging to trial n_test_trials (one past the last trial),
    # so clip it back into the last trial.
    trial_idx = np.clip(trial_idx, 0, n_test_trials - 1)
    time_in_trial_ms = ((rel_t - trial_idx * trial_dur) * 1000.0).astype(np.float32)

    return {
        'spike_times_ms': time_in_trial_ms,
        'spike_neuron_idx': neurons.astype(np.int32),
        'spike_trial_idx': trial_idx,
    }


def compute_monitoring_metrics(net_objs, t_snapshot_start, params, n_test_trials):
    """
    Compute the four summary metrics over the test-trial window that was
    just run: mean_rate_E (average firing rate of E neurons), mean_w_EE
    (average E->E weight), frac_w_max (fraction of E->E weights at their
    cap), and mean_cv_isi (average coefficient of variation of inter-spike
    intervals, a measure of spike-timing irregularity).
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
    Stop the run (by raising RuntimeError) if the monitoring metrics show
    something has gone wrong: mean_rate_E above 30 Hz means potentiation is
    running away unchecked, and frac_w_max above 0.5 means depression isn't
    keeping up (too many weights are stuck at their cap).
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
    Take one snapshot of the network's current state. Steps:
    1. Turn STDP off for the duration of the snapshot (plastic = 0), so
       taking the snapshot doesn't itself change the weights.
    2. Run the fixed test_trial_sequence (40 trials: 5 per direction).
    3. Record the E->E weight matrix (in sparse COO form) and the spike data,
       and save both to h5_path.
    4. Put plasticity back to whatever it was before (1 for a normal run, or
       0 for a frozen control, so a frozen run stays frozen the whole time).
    5. Print a one-line summary, and if check_abort is True, check whether
       the run should be aborted.

    check_abort=False is meant for tests. Tests sometimes use an unrealistic
    nu_ext (external input rate) to force spiking activity in a tiny test
    network, and such a network can legitimately go over the 30 Hz / frac_w_max
    abort thresholds without that meaning anything is actually wrong, unlike
    in the real, validated network used by run_condition().
    """
    syn = net_objs['syn_EE']
    # `plastic` is a single shared value for the whole synapse group (not one
    # per synapse), so plastic[:] comes back as a 0-dimensional array.
    prev_plastic = int(np.asarray(syn.plastic[:]))
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
        'row': j_arr,   # postsynaptic neuron index, matches circuit/run_baseline.py's save_baseline convention
        'col': i_arr,   # presynaptic neuron index
        'shape': np.array([params['N_exc'], params['N_exc']], dtype=np.int32),
    }

    save_snapshot(h5_path, epoch, W_EE_coo, spike_data, test_trial_sequence, metrics)

    syn.plastic = prev_plastic

    print(f"[epoch {epoch:5d}] mean_rate_E={metrics['mean_rate_E']:6.2f} Hz  "
          f"mean_w_EE={metrics['mean_w_EE'] * 1e9:7.4f} nA  "
          f"frac_w_max={metrics['frac_w_max']:.3f}  "
          f"mean_cv_isi={metrics['mean_cv_isi']:.3f}")

    if check_abort:
        check_abort_criteria(metrics, epoch)


def _select_codegen_backend():
    """
    Switch Brian2 over to the cython code-generation backend for the full
    run, because the numpy backend is roughly 30x slower for this size of
    simulation.

    This must be called AFTER we import from circuit.network, since that
    module sets prefs.codegen.target = 'numpy' as soon as it's imported (see
    the module-level prefs line in circuit/network.py).

    It must ALSO be called AFTER load_baseline()/build_network() have already
    run. Those functions call `.connect(p=...)` and use `rand()` to set up
    initial membrane potentials, and these run immediately, using whatever
    codegen backend is active at that moment. Brian2's random number
    generator produces a different sequence of draws for 'numpy' vs 'cython'
    even with the same seed, so if we switched backends too early, the
    network's connectivity would come out different from what's stored in
    baseline_network.h5 (which was generated under 'numpy'). So build_network()
    has to run under 'numpy', and only afterward do we switch to 'cython' for
    speed.

    If no working C++ compiler is found, this falls back to the numpy
    backend.
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
                   seed=42, condition_name='', check_abort=True, test_trial_sequence=None,
                   plasticity_on=True, weight_norm=True):
    """
    Run the full training condition: burn-in period, an epoch-0 snapshot,
    then the training loop, taking a snapshot whenever the trial count hits
    one of the requested snapshot epochs.

    snapshot_epochs : a set of ints, the cumulative trial counts at which to
        take a snapshot, e.g. {0, 50, 100}. Epoch 0 is handled separately
        before the training loop starts, since inside the loop
        `completed = trial_idx + 1` can never be 0.
    test_trial_sequence : the fixed sequence of test trials used for every
        snapshot. Defaults to generate_test_trial_sequence() (40 trials, 5
        per direction). Tests can pass in a shorter sequence to run faster.
    """
    if test_trial_sequence is None:
        test_trial_sequence = generate_test_trial_sequence()

    trial_sequence = generate_trial_sequence(n_per_direction, params['n_directions'], seed=seed)
    n_trials = len(trial_sequence)

    # --- Burn-in: STDP is turned off and the input sits at its background
    # rate. This lets the initial membrane-potential transient die down
    # before we take any snapshot.
    print(f"[{condition_name}] burn-in: {params['t_burn_in']:.1f} s (plastic=0)")
    syn = net_objs['syn_EE']
    syn.plastic = 0
    net_objs['input_group'].rates = np.full(params['n_input'], params['r_background']) * Hz
    net_objs['net'].run(params['t_burn_in'] * second)
    # For the frozen control (plasticity_on=False, "Control A"), STDP stays off
    # for the entire run. This isolates what the network's geometry looks like
    # from structure alone, without any learning.
    syn.plastic = 1 if plasticity_on else 0

    # --- Epoch-0 snapshot (taken before any training trial has run)
    if 0 in snapshot_epochs:
        run_snapshot(net_objs, h5_path, epoch=0,
                      test_trial_sequence=test_trial_sequence,
                      theta_i=theta_i, params=params, check_abort=check_abort)

    # --- Training loop
    apply_norm = weight_norm and plasticity_on and 'W_target_EE' in net_objs
    for trial_idx in range(n_trials):
        direction_idx = trial_sequence[trial_idx]
        theta_cue = 2 * np.pi * direction_idx / params['n_directions']
        run_one_trial(net_objs, params, theta_i, theta_cue)

        # After each trial, rescale each neuron's incoming E->E weights so
        # their total stays equal to its baseline value. This is the
        # homeostatic step that has to go along with plain additive STDP: it
        # makes STDP redistribute weight among synapses rather than just
        # adding weight overall, which keeps the network in the
        # asynchronous-irregular (AI) regime instead of runaway excitation.
        # Without this step, the mean firing rate climbed from 1.86 to 9.6 Hz.
        if apply_norm:
            normalize_incoming_weights(net_objs['syn_EE'], net_objs['W_target_EE'],
                                       params['w_max'])

        completed = trial_idx + 1
        if completed in snapshot_epochs:
            run_snapshot(net_objs, h5_path, epoch=completed,
                          test_trial_sequence=test_trial_sequence,
                          theta_i=theta_i, params=params, check_abort=check_abort)

    return net_objs


def main():
    parser = argparse.ArgumentParser(
        description="STDP + 8-direction center-out task: training run")
    # --condition picks the overall experimental condition: how the network is
    # wired up at the start (p_cross, the cross-assembly connection
    # probability) and whether E->E STDP is on.
    parser.add_argument('--condition', choices=['seeded', 'control', 'frozen'],
                         required=True,
                         help="seeded: p_cross=0.2, STDP on; "
                              "control: p_cross=1.0, STDP on; "
                              "frozen: p_cross=0.2, STDP off (Control A, the matched "
                              "structural control for seeded)")
    # --exec_mode controls what happens during the "exec" (movement) phase of
    # each trial: whether the task input keeps driving the network, or is
    # withdrawn so the network evolves on its own.
    parser.add_argument('--exec_mode', choices=['sustained', 'autonomous'],
                         default='sustained',
                         help="sustained: exec input clamps the state (default); "
                              "autonomous: exec input withdrawn, network evolves freely "
                              "from the prep-set initial condition")
    # --weight_norm turns the per-trial synaptic rescaling (the homeostatic
    # step described above) on or off.
    parser.add_argument('--weight_norm', choices=['on', 'off'], default='on',
                         help="on (default): multiplicative synaptic scaling after each "
                              "trial holds total incoming E->E weight constant, keeping "
                              "the network in the AI regime; off: raw additive STDP")
    # --inhibitory_plasticity turns on iSTDP (inhibitory STDP, Vogels et al.
    # 2011) on the I->E synapses, which adjusts inhibition to push each E
    # neuron's firing rate toward a target (rho0).
    parser.add_argument('--inhibitory_plasticity', choices=['on', 'off'], default='off',
                         help="on: Vogels (2011) inhibitory STDP on I->E synapses drives "
                              "each E neuron toward rho0, stabilizing the network gain; "
                              "off (default): static inhibition")
    # --n_per_direction sets how many training trials to run for each of the 8
    # directions (so total training trials = 8 * n_per_direction).
    parser.add_argument('--n_per_direction', type=int, default=13,
                         help="Training trials per direction (default 13 -> "
                              "104 total)")
    # --snapshot_epochs lists the trial counts at which to pause training and
    # save a snapshot of the network's weights and activity.
    parser.add_argument('--snapshot_epochs', type=int, nargs='+', default=[0, 50, 100],
                         help="Cumulative trial counts at which to snapshot")
    # --seed must match the seed that baseline_network.h5 was generated with
    # (seed=7, see circuit/results/baseline_network.h5:/validation/seed),
    # otherwise load_baseline()'s connectivity check will fail.
    parser.add_argument('--seed', type=int, default=7)
    # --baseline_h5, --results_dir: where to read the baseline network from,
    # and where to write this run's output file.
    parser.add_argument('--baseline_h5', type=str,
                         default='circuit/results/baseline_network.h5')
    parser.add_argument('--results_dir', type=str, default='plasticity/results')
    # --label sets the output filename suffix. Use it to give runs that share
    # the same --condition different names, e.g. for an iSTDP-only
    # decomposition control or for individual sweep points.
    parser.add_argument('--label', type=str, default=None,
                         help="output filename suffix (default: condition); use to "
                              "distinguish runs that share a --condition, e.g. an "
                              "iSTDP-only decomposition control or sweep points")
    # The next three arguments override individual iSTDP parameters, used
    # when running a parameter sweep. Leaving them as None means "use the
    # default value".
    parser.add_argument('--rho0', type=float, default=None,
                         help="iSTDP target E rate (Hz)")
    parser.add_argument('--eta_istdp', type=float, default=None,
                         help="iSTDP learning rate (A)")
    parser.add_argument('--tau_istdp', type=float, default=None,
                         help="iSTDP trace time constant (s)")
    # --ee_plasticity lets you turn E->E STDP on or off independently of
    # --condition. This is the sweep's "E->E null axis" (a way to test what
    # happens with E->E plasticity off regardless of which condition you're
    # in). If not given, it just follows --condition as usual.
    parser.add_argument('--ee_plasticity', choices=['on', 'off'], default=None,
                         help="override E->E STDP independent of --condition (the sweep's "
                              "E->E null axis); default follows --condition")
    # --A_plus / --A_minus override the E->E STDP potentiation/depression
    # amplitudes, used by the probe that tests how the LTP/LTD balance
    # affects the network.
    parser.add_argument('--A_plus', type=float, default=None,
                         help="E->E STDP potentiation amplitude (A); overrides default "
                              "(used by the E->E LTP/LTD-asymmetry probe)")
    parser.add_argument('--A_minus', type=float, default=None,
                         help="E->E STDP depression amplitude (A); overrides default "
                              "(used by the E->E LTP/LTD-asymmetry probe)")
    args = parser.parse_args()

    params = {**DEFAULT_PARAMS, **DEFAULT_PARAMS_PLASTICITY}
    params['exec_mode'] = args.exec_mode
    # The 'control' condition starts from uniform connectivity (p_cross=1.0).
    # Both 'seeded' and 'frozen' start from the same seeded connectivity
    # (p_cross=0.2), since 'frozen' is meant as the matched structural control
    # for 'seeded'.
    p_cross = (params['p_cross_control'] if args.condition == 'control'
               else params['p_cross_seeded'])
    plasticity_on = (args.condition != 'frozen')
    weight_norm = (args.weight_norm == 'on')
    params['weight_norm'] = weight_norm
    inhibitory_plasticity = (args.inhibitory_plasticity == 'on')
    params['inhibitory_plasticity'] = inhibitory_plasticity

    # Apply any sweep overrides given on the command line.
    for key in ('rho0', 'eta_istdp', 'tau_istdp', 'A_plus', 'A_minus'):
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)
    if args.ee_plasticity is not None:
        plasticity_on = (args.ee_plasticity == 'on')

    label = args.label or args.condition
    os.makedirs(args.results_dir, exist_ok=True)
    h5_path = os.path.join(args.results_dir, f'training_{label}.h5')
    if os.path.exists(h5_path):
        os.remove(h5_path)

    # Make the output file self-contained: copy over provenance info from the
    # circuit baseline, then write this run's STDP and task parameters.
    copy_baseline_provenance(h5_path, args.baseline_h5)
    save_training_params(h5_path, params, p_cross=p_cross, seed=args.seed,
                         plasticity_on=plasticity_on)

    net_objs = load_baseline(args.baseline_h5, params, seed=args.seed)
    net_objs = build_stdp_network(net_objs, params, p_cross=p_cross, seed=args.seed,
                                  inhibitory_plasticity=inhibitory_plasticity)
    theta_i = assign_preferred_directions(params['n_input'], params['n_directions'])

    # Only switch the codegen backend after the network is fully built (see
    # _select_codegen_backend's docstring for why), so that load_baseline()'s
    # connectivity reproduction matches baseline_network.h5 exactly.
    _select_codegen_backend()

    run_condition(net_objs, params, h5_path, theta_i,
                   n_per_direction=args.n_per_direction,
                   snapshot_epochs=set(args.snapshot_epochs),
                   seed=args.seed, condition_name=label,
                   plasticity_on=plasticity_on, weight_norm=weight_norm)

    print(f"Done. Wrote {h5_path}")


if __name__ == '__main__':
    main()
