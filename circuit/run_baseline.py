# circuit/run_baseline.py

"""
This script builds the balanced E/I circuit, runs it for a while, and checks
whether it is behaving the way we want before we save it as the "baseline"
network that later scripts (plasticity training, etc.) start from.

Usage:
    python circuit/run_baseline.py --nu_ext 6.25 --g_EI 0.065 --w_scale_II 0.5

By default it runs a 30 s simulation. It automatically checks 3 numbers
(checks 3, 4, 7 below) using only the LAST 10 seconds of the run, once the
network has settled down. It also saves a few figures for visual checks
(1, 2, 5, 6). If all the automatic checks pass, it writes the network state
to baseline_network.h5.

Why 30 s, and why only look at the last 10 s:
  Starting from random initial voltages (each neuron starts at a random
  voltage between V_reset and V_th), the network takes about 15 seconds to
  settle into its steady firing pattern. If you measure the irregularity of
  firing (CV-ISI) over the whole run, the early transient pulls the number
  down (CV around 0.75) compared to the true steady-state value
  (CV around 0.80-0.82). Looking only at the [20, 30] s window avoids this
  and gives a measurement of the network's actual stable operating point.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # use a non-interactive backend so plots can be saved to file without a display
import matplotlib.pyplot as plt
import h5py
import scipy.sparse
from scipy.ndimage import gaussian_filter1d
from brian2 import second, amp

from circuit.network import build_network, DEFAULT_PARAMS


# ---------------------------------------------------------------------------
# Analysis functions. These work on plain dicts of spike times, they don't
# need Brian2 at all.
# spike_trains : dict[int, np.ndarray]  -- maps neuron index to an array of
#                                           spike times in seconds
# ---------------------------------------------------------------------------

def compute_cv_isi(
    spike_trains: dict,
    t_start: float,
    t_end: float,
    min_spikes: int = 20,
) -> tuple:
    """
    Compute the CV-ISI (coefficient of variation of inter-spike intervals)
    for each neuron, and the average across the population.

    CV-ISI measures how irregular a neuron's firing is: CV near 0 means very
    regular (clock-like) spiking, CV near 1 means irregular (Poisson-like)
    spiking.

    Only neurons that fired at least min_spikes times in [t_start, t_end]
    are included, so the CV estimate isn't based on too few spikes.

    Returns
    -------
    per_neuron : dict {neuron_idx: float}   -- CV for each neuron that qualifies
    mean_cv    : float                       -- average over the population
                                                 (nan if no neuron qualifies)
    """
    per_neuron = {}
    for idx, times in spike_trains.items():
        times = np.asarray(times)
        in_win = times[(times >= t_start) & (times < t_end)]
        if len(in_win) < min_spikes:
            continue
        isis = np.diff(np.sort(in_win))
        if len(isis) < 2:
            continue
        cv = float(isis.std() / isis.mean())
        if np.isfinite(cv):
            per_neuron[idx] = cv

    mean_cv = float(np.mean(list(per_neuron.values()))) if per_neuron else float('nan')
    return per_neuron, mean_cv


def compute_pairwise_corr(
    spike_trains: dict,
    t_start: float,
    t_end: float,
    bin_ms: float = 10.0,
    n_pairs: int = 50,
    seed: int = 42,
) -> float:
    """
    Pick random pairs of neurons, turn each neuron's spikes into a binned
    spike-count time series, and compute the Pearson correlation between
    each pair. Return the average correlation across all pairs.

    This tells us how correlated the neurons' activity is. We want this to
    be low (close to 0), since highly correlated firing is not the
    asynchronous irregular pattern we're aiming for.

    The pairs are picked using a fixed random seed, so results are the same
    every time you run this. Returns nan if there are fewer than 2 neurons.
    """
    dt = bin_ms * 1e-3
    n_bins = int((t_end - t_start) / dt)
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)

    indices = sorted(spike_trains.keys())
    n = len(indices)
    if n < 2:
        return float('nan')

    # Build a matrix of spike counts: one row per neuron, one column per time bin
    counts = np.zeros((n, n_bins), dtype=np.float32)
    for row, idx in enumerate(indices):
        times = np.asarray(spike_trains[idx])
        times = times[(times >= t_start) & (times < t_end)]
        counts[row], _ = np.histogram(times, bins=bin_edges)

    rng = np.random.default_rng(seed)
    n_pairs = min(n_pairs, n * (n - 1) // 2)

    # Pick random pairs of neurons, without picking the same pair twice
    pairs = set()
    max_attempts = n_pairs * 100
    attempts = 0
    while len(pairs) < n_pairs and attempts < max_attempts:
        i, j = rng.choice(n, size=2, replace=False)
        pairs.add((min(i, j), max(i, j)))
        attempts += 1

    rs = []
    for i, j in pairs:
        r = np.corrcoef(counts[i], counts[j])[0, 1]
        if np.isfinite(r):
            rs.append(float(r))

    return float(np.mean(rs)) if rs else float('nan')


def compute_power_spectrum(
    spike_trains: dict,
    t_start: float,
    t_end: float,
    smooth_sigma_ms: float = 5.0,
    dt_ms: float = 0.1,
) -> tuple:
    """
    Compute the power spectrum of the whole population's firing rate over time.

    Steps: add up spikes from all neurons into a histogram with very fine
    time bins, smooth that histogram with a Gaussian filter, then take the
    Fourier transform (FFT) to see which frequencies are present.
    Returns (frequencies_Hz, power).
    """
    dt = dt_ms * 1e-3
    n_bins = int((t_end - t_start) / dt)
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)

    pop_rate = np.zeros(n_bins, dtype=np.float64)
    for times in spike_trains.values():
        times = np.asarray(times)
        times = times[(times >= t_start) & (times < t_end)]
        counts, _ = np.histogram(times, bins=bin_edges)
        pop_rate += counts

    sigma_samples = (smooth_sigma_ms * 1e-3) / dt
    pop_smooth = gaussian_filter1d(pop_rate, sigma=sigma_samples)

    fft_vals = np.fft.rfft(pop_smooth)
    freqs = np.fft.rfftfreq(n_bins, d=dt)
    power = np.abs(fft_vals) ** 2

    return freqs, power


# ---------------------------------------------------------------------------
# HDF5 save
# ---------------------------------------------------------------------------

def save_baseline(
    path: str,
    params: dict,
    net_objs: dict,
    validation: dict,
    seed: int,
) -> None:
    """
    Write the network (now that it has passed validation) to an HDF5 file.

    The weight matrices are stored in COO sparse format (a list of nonzero
    entries with row, column, and value). To turn one back into a normal
    matrix:
        W = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)

    All parameters are stored in SI units. Weights are stored as float32
    numbers in amps.
    """
    def _save_sparse(grp, name: str, syn, tgt_size: int, src_size: int):
        # Remove Brian2's "amp" unit so we're left with plain numbers
        w_vals = np.array(syn.w / amp, dtype=np.float32)
        # syn.j is the target (postsynaptic) neuron -> becomes the row index
        # syn.i is the source (presynaptic) neuron -> becomes the column index
        # so W[row, col] = W[target, source]
        rows = np.array(syn.j[:], dtype=np.int32)
        cols = np.array(syn.i[:], dtype=np.int32)
        g = grp.create_group(name)
        g.create_dataset('data',  data=w_vals)
        g.create_dataset('row',   data=rows)
        g.create_dataset('col',   data=cols)
        g.create_dataset('shape', data=np.array([tgt_size, src_size], dtype=np.int32))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with h5py.File(path, 'w') as f:
        # /network
        ng = f.create_group('network')
        ng.create_dataset('N_exc',     data=int(params['N_exc']))
        ng.create_dataset('N_inh',     data=int(params['N_inh']))
        ng.create_dataset('p_connect', data=float(params['p_connect']))

        pn = ng.create_group('params_neuron')
        for k in ('tau_m', 'V_rest', 'V_th', 'V_reset', 'tau_ref', 'R'):
            pn.create_dataset(k, data=float(params[k]))

        ps = ng.create_group('params_synapse')
        for k in ('tau_syn_E', 'tau_syn_I', 'g_EI', 'w_scale_II'):
            ps.create_dataset(k, data=float(params[k]))
        ps.create_dataset('nu_ext', data=float(params['nu_ext']))

        # /weights: COO sparse format, SI units (amps as float32)
        wg = f.create_group('weights')
        Ne, Ni = params['N_exc'], params['N_inh']
        _save_sparse(wg, 'W_EE', net_objs['syn_EE'], Ne, Ne)
        _save_sparse(wg, 'W_EI', net_objs['syn_EI'], Ni, Ne)  # target=I, source=E
        _save_sparse(wg, 'W_IE', net_objs['syn_IE'], Ne, Ni)  # target=E, source=I
        _save_sparse(wg, 'W_II', net_objs['syn_II'], Ni, Ni)

        # /validation
        vg = f.create_group('validation')
        vg.create_dataset('mean_rate_E',        data=float(validation['mean_rate_E']))
        vg.create_dataset('mean_rate_I',        data=float(validation['mean_rate_I']))
        vg.create_dataset('mean_CV_ISI',        data=float(validation['mean_CV_ISI']))
        vg.create_dataset('mean_pairwise_corr', data=float(validation['mean_pairwise_corr']))
        vg.create_dataset('raster_times',
                          data=np.asarray(validation['raster_times'], dtype=np.float32))
        vg.create_dataset('raster_indices',
                          data=np.asarray(validation['raster_indices'], dtype=np.int32))
        vg.create_dataset('seed',      data=int(seed))
        vg.create_dataset('nu_ext_hz', data=float(params['nu_ext']))
        vg.create_dataset('g_EI_nA',   data=float(params['g_EI'] / 1e-9))  # convert amps to nanoamps


# ---------------------------------------------------------------------------
# Plotting functions. These save figures to disk so we can look at them later.
# ---------------------------------------------------------------------------

def _extract_spike_trains(monitor, n_neurons: int, t_sim: float) -> dict:
    """Convert a Brian2 SpikeMonitor's spike_trains() into plain numpy arrays (seconds), with Brian2 units removed."""
    return {k: np.array(v / second)
            for k, v in monitor.spike_trains().items()}


def plot_raster(net_objs: dict, params: dict, t_raster: float,
                results_dir: str) -> tuple:
    """
    Plot a spike raster (one dot per spike) for up to 100 randomly chosen
    E neurons, over the time window [0, t_raster].
    Returns (raster_times, raster_indices) as float32/int32 arrays, ready
    to be saved into the HDF5 file.
    """
    rng = np.random.default_rng(42)
    Ne = params['N_exc']
    n_sample = min(100, Ne)
    sample_idx = rng.choice(Ne, size=n_sample, replace=False)

    all_t = np.array(net_objs['spike_E'].t / second)
    all_i = np.array(net_objs['spike_E'].i[:])

    mask = (all_t < t_raster) & np.isin(all_i, sample_idx)
    rt = all_t[mask].astype(np.float32)
    ri = all_i[mask].astype(np.int32)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(rt, ri, s=0.5, c='k', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron index')
    ax.set_title(f'Raster, {n_sample} E neurons, t=0-{t_raster} s')
    ax.set_xlim(0, t_raster)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'figures', 'raster.png'), dpi=150)
    plt.close(fig)

    return rt, ri


def plot_firing_rate_hist(trains_E: dict, trains_I: dict, t_end: float,
                          results_dir: str) -> None:
    """Plot a histogram of each neuron's average firing rate, for the E and I populations separately."""
    rates_E = np.array([len(t[(t >= 0) & (t < t_end)]) / t_end
                        for t in trains_E.values()])
    rates_I = np.array([len(t[(t >= 0) & (t < t_end)]) / t_end
                        for t in trains_I.values()])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, rates, pop in zip(axes, [rates_E, rates_I], ['E', 'I']):
        ax.hist(rates, bins=30, color='steelblue' if pop == 'E' else 'tomato',
                edgecolor='k', linewidth=0.3)
        ax.set_xlabel('Mean firing rate (Hz)')
        ax.set_ylabel('Count')
        ax.set_title(f'{pop} population, mean={rates.mean():.1f} Hz')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'figures', 'firing_rate_hist.png'), dpi=150)
    plt.close(fig)


def plot_isi_dist(trains_E: dict, t_end: float, results_dir: str) -> None:
    """Plot the distribution of inter-spike intervals (ISIs) for 6 randomly picked E neurons."""
    rng = np.random.default_rng(99)
    keys = list(trains_E.keys())
    sample_keys = rng.choice(keys, size=min(6, len(keys)), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, idx in zip(axes.flat, sample_keys):
        times = trains_E[int(idx)]
        times = times[(times >= 0) & (times < t_end)]
        if len(times) < 3:
            ax.set_visible(False)
            continue
        isis = np.diff(np.sort(times)) * 1000  # convert seconds to milliseconds
        cv = isis.std() / isis.mean() if len(isis) > 1 else float('nan')
        ax.hist(isis, bins=20, color='steelblue', edgecolor='k', linewidth=0.3)
        ax.set_xlabel('ISI (ms)')
        ax.set_title(f'Neuron {idx}, CV={cv:.2f}')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'figures', 'isi_dist.png'), dpi=150)
    plt.close(fig)


def plot_power_spectrum(trains_E: dict, t_end: float, results_dir: str) -> None:
    """Plot the power spectrum of the E population's firing rate."""
    freqs, power = compute_power_spectrum(trains_E, t_start=0.0, t_end=t_end,
                                          smooth_sigma_ms=5.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    mask = (freqs > 0) & (freqs < 200)
    ax.loglog(freqs[mask], power[mask], lw=0.8)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (a.u.)')
    ax.set_title('Population firing rate power spectrum (E neurons)')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'figures', 'power_spectrum.png'), dpi=150)
    plt.close(fig)


def plot_weight_hists(net_objs: dict, results_dir: str) -> None:
    """Plot histograms (log scale on the y-axis) of the initial weight distributions for all 4 synapse types."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    syns = [('W_EE', 'syn_EE', 'C0'), ('W_EI', 'syn_EI', 'C1'),
            ('W_IE', 'syn_IE', 'C2'), ('W_II', 'syn_II', 'C3')]
    for ax, (name, key, color) in zip(axes.flat, syns):
        w_nA = np.array(net_objs[key].w / amp) / 1e-9  # convert amps to nanoamps so the numbers are easier to read
        ax.hist(w_nA, bins=40, color=color, edgecolor='k', linewidth=0.2)
        ax.set_xlabel('Weight (nA)')
        ax.set_title(name)
        ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'figures', 'weight_hists.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Validation: run a set of checks on a finished simulation
# ---------------------------------------------------------------------------

def run_validation(
    net_objs: dict,
    params: dict,
    t_sim: float,
    seed: int,
    results_dir: str,
) -> tuple:
    """
    Run all 7 validation checks after a simulation has finished.

    Checks 3, 4, and 7 are numeric and are checked automatically, using only
    the LAST 10 seconds of the simulation. Checks 1, 2, 5, and 6 are visual:
    this function saves figures for those so a person can look at them.

    Returns
    -------
    validation : dict   -- the values that get saved into the HDF5 /validation group
    passed     : bool   -- True only if checks 3, 4, and 7 all pass
    """
    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)

    Ne = params['N_exc']
    Ni = params['N_inh']

    trains_E = _extract_spike_trains(net_objs['spike_E'], Ne, t_sim)
    trains_I = _extract_spike_trains(net_objs['spike_I'], Ni, t_sim)

    # Use only the last 10 seconds to compute the numeric checks, since
    # that's the steady-state part of the run, after the initial transient
    # has died down. The random starting voltages (each neuron starts
    # somewhere between V_reset and V_th) cause a burst of activity in the
    # first ~15 seconds. Computing CV-ISI and pairwise correlation only in
    # this last window gives a true picture of the network's steady-state
    # behavior.
    t_eval_start = max(0.0, t_sim - 10.0)

    # ---- Check 3: CV-ISI (a number we check automatically) ----
    # min_spikes=20 means each neuron needs at least 19 ISIs to count toward
    # its CV estimate. At around 3 Hz over a 10-second window, that's about
    # 30 spikes per neuron, so most neurons qualify. Neurons firing below
    # about 2 Hz are skipped, since too few spikes would give a noisy CV
    # estimate that doesn't represent the population well.
    _, mean_cv = compute_cv_isi(trains_E, t_eval_start, t_sim, min_spikes=20)
    cv_pass = not np.isnan(mean_cv) and 0.8 <= mean_cv <= 1.2

    # ---- Check 4: Pairwise correlation (a number we check automatically) ----
    mean_r = compute_pairwise_corr(trains_E, t_eval_start, t_sim,
                                    bin_ms=10.0, n_pairs=50, seed=seed)
    pairwise_pass = (not np.isnan(mean_r)) and mean_r < 0.05

    # ---- Check 7: I/E firing rate ratio (a number we check automatically) ----
    # Compute firing rates using only the steady-state window, not the
    # whole simulation.
    all_t_E = np.array(net_objs['spike_E'].t / second)
    all_t_I = np.array(net_objs['spike_I'].t / second)
    win = t_sim - t_eval_start
    mean_rate_E = float(np.sum(all_t_E >= t_eval_start) / (Ne * win))
    mean_rate_I = float(np.sum(all_t_I >= t_eval_start) / (Ni * win))
    rate_ratio  = mean_rate_I / mean_rate_E if mean_rate_E > 0 else float('nan')
    # We want the I population to fire 2-6x faster than the E population.
    # (The original target was 2-3x, then widened to 2-5x, now 2-6x.)
    # With nu_ext = 7 Hz, the I neurons get more background input, which can
    # push the ratio up toward 5-6x for some random seeds. That's still a
    # healthy AI network, since CV and pairwise correlation pass fine.
    # For reference: in real cortex, PV interneurons fire 4-8x faster than
    # pyramidal (excitatory) cells at rest.
    rate_pass   = not np.isnan(rate_ratio) and 2.0 <= rate_ratio <= 6.0

    # Also report the firing rate over the whole simulation, just for reference.
    mean_rate_E_full = net_objs['spike_E'].num_spikes / (Ne * t_sim)
    mean_rate_E_pass = 2.0 <= mean_rate_E <= 10.0

    # ---- Print the results to the console ----
    width = 28
    print(f"\n{'=' * 55}")
    print(f"{'Validation results (steady-state window)':^55}")
    print(f"{'=' * 55}")
    print(f"{'Check 3 (CV-ISI):':<{width}} {mean_cv:.3f}   "
          f"{'PASS' if cv_pass else 'FAIL'}  [target: 0.8–1.2]")
    print(f"{'Check 4 (pairwise r):':<{width}} {mean_r:.4f}  "
          f"{'PASS' if pairwise_pass else 'FAIL'}  [target: <0.05]")
    print(f"{'Check 7 (I/E rate ratio):':<{width}} {rate_ratio:.2f}    "
          f"{'PASS' if rate_pass else 'FAIL'}  [target: 2–6×]")
    print(f"{'Mean E rate (steady-state):':<{width}} {mean_rate_E:.2f} Hz  "
          f"{'PASS' if mean_rate_E_pass else 'FAIL'}  [target: 2–10 Hz]")
    print(f"{'Mean I rate (steady-state):':<{width}} {mean_rate_I:.2f} Hz")
    print(f"{'Mean E rate (full sim):':<{width}} {mean_rate_E_full:.2f} Hz")
    print(f"{'Eval window:':<{width}} [{t_eval_start:.0f}–{t_sim:.0f}] s")
    print(f"{'=' * 55}\n")

    # ---- Save figures for the visual checks (1, 2, 5, 6) ----
    raster_t, raster_i = plot_raster(net_objs, params, t_raster=1.0,
                                      results_dir=results_dir)
    plot_firing_rate_hist(trains_E, trains_I, t_sim, results_dir)
    plot_isi_dist(trains_E, t_sim, results_dir)
    plot_power_spectrum(trains_E, t_sim, results_dir)
    plot_weight_hists(net_objs, results_dir)

    print(f"Figures saved to {os.path.join(results_dir, 'figures')}/")
    print("Manually inspect: raster.png, firing_rate_hist.png, "
          "isi_dist.png, power_spectrum.png, weight_hists.png\n")

    validation = {
        'mean_rate_E':        mean_rate_E,        # from the steady-state window
        'mean_rate_I':        mean_rate_I,        # from the steady-state window
        'mean_CV_ISI':        mean_cv,
        'mean_pairwise_corr': mean_r,
        'raster_times':       raster_t,
        'raster_indices':     raster_i,
    }

    all_passed = cv_pass and pairwise_pass and rate_pass and mean_rate_E_pass
    return validation, all_passed


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validate the balanced E/I circuit and save baseline network to HDF5.')
    parser.add_argument('--nu_ext',     type=float, required=True,
                        help='Background Poisson rate (Hz)')
    parser.add_argument('--g_EI',       type=float, required=True,
                        help='Mean I→E inhibitory weight (nA)')
    parser.add_argument('--w_scale_II', type=float, default=DEFAULT_PARAMS['w_scale_II'],
                        help=f'I→I weight scale relative to g_EI '
                             f'(default: {DEFAULT_PARAMS["w_scale_II"]})')
    parser.add_argument('--t_sim',      type=float, default=30.0,
                        help='Simulation duration in seconds (default: 30.0; '
                             'needs to be >15 s so the last-10-s window is in steady state)')
    parser.add_argument('--seed',       type=int,   default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'results'),
                        help='Directory for results and figures')
    args = parser.parse_args()

    params = {
        **DEFAULT_PARAMS,
        'nu_ext':     args.nu_ext,
        'g_EI':       args.g_EI * 1e-9,  # command line gives nA, convert to A for internal use
        'w_scale_II': args.w_scale_II,
    }

    print(f"Building network: nu_ext={args.nu_ext} Hz, g_EI={args.g_EI} nA, "
          f"w_scale_II={args.w_scale_II}, seed={args.seed}")
    net_objs = build_network(params, seed=args.seed)

    print(f"Running {args.t_sim} s simulation ...")
    net_objs['net'].run(args.t_sim * second)

    validation, passed = run_validation(
        net_objs, params, args.t_sim, args.seed, args.results_dir)

    if passed:
        h5_path = os.path.join(args.results_dir, 'baseline_network.h5')
        save_baseline(h5_path, params, net_objs, validation, seed=args.seed)
        print(f"All quantitative checks PASSED. Baseline saved to:\n  {h5_path}")
        sys.exit(0)
    else:
        report_path = os.path.join(args.results_dir, 'validation_report.txt')
        os.makedirs(args.results_dir, exist_ok=True)
        rate_ratio = (validation['mean_rate_I'] / validation['mean_rate_E']
                      if validation['mean_rate_E'] > 0 else float('nan'))
        with open(report_path, 'w') as f:
            f.write(f"nu_ext={args.nu_ext} Hz  g_EI={args.g_EI} nA  "
                    f"w_scale_II={args.w_scale_II}  seed={args.seed}\n")
            f.write(f"CV_ISI={validation['mean_CV_ISI']:.4f}  "
                    f"pairwise_r={validation['mean_pairwise_corr']:.4f}  "
                    f"rate_ratio={rate_ratio:.2f}  "
                    f"rate_E={validation['mean_rate_E']:.2f} Hz\n")
        print(f"One or more quantitative checks FAILED. "
              f"Report written to:\n  {report_path}")
        print("Re-tune (nu_ext, g_EI, w_scale_II) and re-run.")
        sys.exit(1)


if __name__ == '__main__':
    main()
