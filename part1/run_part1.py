# part1/run_part1.py

"""
Validation runner for Part 1 of the Hebbian Plasticity / Manifold Sculptor project.

Usage:
    python part1/run_part1.py --nu_ext 15.0 --g_EI 0.24

Runs a 5 s simulation, auto-evaluates checks 3, 4, 7, saves figures for
visual checks 1, 2, 5, 6, writes baseline_network.h5 if all quantitative
checks pass.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves to file without a display
import matplotlib.pyplot as plt
import h5py
import scipy.sparse
from scipy.ndimage import gaussian_filter1d
from brian2 import second, amp

from part1.network import build_network, DEFAULT_PARAMS


# ---------------------------------------------------------------------------
# Analysis functions — operate on plain dicts of spike times, no Brian2 deps.
# spike_trains : dict[int, np.ndarray]  — neuron_idx → spike times in seconds
# ---------------------------------------------------------------------------

def compute_cv_isi(
    spike_trains: dict,
    t_start: float,
    t_end: float,
    min_spikes: int = 20,
) -> tuple:
    """
    Compute per-neuron CV-ISI and the population mean.

    Only neurons with >= min_spikes spikes in [t_start, t_end] are included.

    Returns
    -------
    per_neuron : dict {neuron_idx: float}   — CV for each qualifying neuron
    mean_cv    : float                       — population mean (nan if none qualify)
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
    Compute mean Pearson correlation of spike-count vectors across random pairs.

    Pairs are drawn with a fixed seed so results are reproducible.
    Returns mean Pearson r (nan if fewer than 2 neurons).
    """
    dt = bin_ms * 1e-3
    n_bins = int((t_end - t_start) / dt)
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)

    indices = sorted(spike_trains.keys())
    n = len(indices)
    if n < 2:
        return float('nan')

    # Build spike-count matrix: shape (n_neurons, n_bins)
    counts = np.zeros((n, n_bins), dtype=np.float32)
    for row, idx in enumerate(indices):
        times = np.asarray(spike_trains[idx])
        times = times[(times >= t_start) & (times < t_end)]
        counts[row], _ = np.histogram(times, bins=bin_edges)

    rng = np.random.default_rng(seed)
    n_pairs = min(n_pairs, n * (n - 1) // 2)

    # Draw unique pairs without replacement
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
    Compute the power spectrum of the summed population firing rate.

    Spikes from all neurons are summed into a fine-bin histogram, smoothed
    with a Gaussian kernel, then FFT'd. Returns (frequencies_Hz, power).
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
    Write the validated baseline network to HDF5.

    Weight matrices stored in COO format — reconstruct with:
        W = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)

    All parameters stored in SI units. Weights as float32 in amps.
    """
    def _save_sparse(grp, name: str, syn, tgt_size: int, src_size: int):
        # Strip Brian2 units: divide by `amp` → float array in amps
        w_vals = np.array(syn.w / amp, dtype=np.float32)
        # .j = postsynaptic (row), .i = presynaptic (col) → W[post, pre]
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
        for k in ('tau_syn_E', 'tau_syn_I', 'g_EI'):
            ps.create_dataset(k, data=float(params[k]))
        ps.create_dataset('nu_ext', data=float(params['nu_ext']))

        # /weights — COO sparse format, SI units (amps as float32)
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
        vg.create_dataset('g_EI_nA',   data=float(params['g_EI'] / 1e-9))  # A → nA


# ---------------------------------------------------------------------------
# Plotting — save figures to disk for visual inspection
# ---------------------------------------------------------------------------

def _extract_spike_trains(monitor, n_neurons: int, t_sim: float) -> dict:
    """Convert Brian2 SpikeMonitor.spike_trains() to plain float arrays (seconds)."""
    return {k: np.array(v / second)
            for k, v in monitor.spike_trains().items()}


def plot_raster(net_objs: dict, params: dict, t_raster: float,
                results_dir: str) -> tuple:
    """
    Plot spike raster for up to 100 random E neurons over [0, t_raster].
    Returns (raster_times, raster_indices) as float32/int32 arrays for HDF5.
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
    ax.set_title(f'Raster — {n_sample} E neurons, t=0–{t_raster} s')
    ax.set_xlim(0, t_raster)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'figures', 'raster.png'), dpi=150)
    plt.close(fig)

    return rt, ri


def plot_firing_rate_hist(trains_E: dict, trains_I: dict, t_end: float,
                          results_dir: str) -> None:
    """Histogram of per-neuron mean firing rates (E and I populations)."""
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
        ax.set_title(f'{pop} population — mean={rates.mean():.1f} Hz')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'figures', 'firing_rate_hist.png'), dpi=150)
    plt.close(fig)


def plot_isi_dist(trains_E: dict, t_end: float, results_dir: str) -> None:
    """ISI distribution for 6 randomly selected E neurons."""
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
        isis = np.diff(np.sort(times)) * 1000  # s → ms
        cv = isis.std() / isis.mean() if len(isis) > 1 else float('nan')
        ax.hist(isis, bins=20, color='steelblue', edgecolor='k', linewidth=0.3)
        ax.set_xlabel('ISI (ms)')
        ax.set_title(f'Neuron {idx} — CV={cv:.2f}')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'figures', 'isi_dist.png'), dpi=150)
    plt.close(fig)


def plot_power_spectrum(trains_E: dict, t_end: float, results_dir: str) -> None:
    """Power spectrum of the E population firing rate."""
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
    """Log-scale histograms of initial weight distributions (all 4 synapse types)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    syns = [('W_EE', 'syn_EE', 'C0'), ('W_EI', 'syn_EI', 'C1'),
            ('W_IE', 'syn_IE', 'C2'), ('W_II', 'syn_II', 'C3')]
    for ax, (name, key, color) in zip(axes.flat, syns):
        w_nA = np.array(net_objs[key].w / amp) / 1e-9  # A → nA for readability
        ax.hist(w_nA, bins=40, color=color, edgecolor='k', linewidth=0.2)
        ax.set_xlabel('Weight (nA)')
        ax.set_title(name)
        ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'figures', 'weight_hists.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Validation workflow
# ---------------------------------------------------------------------------

def run_validation(
    net_objs: dict,
    params: dict,
    t_sim: float,
    seed: int,
    results_dir: str,
) -> tuple:
    """
    Run all 7 validation checks after a completed simulation.

    Auto-evaluates checks 3, 4, 7 (quantitative).
    Saves figures for checks 1, 2, 5, 6 (visual, human-inspected).

    Returns
    -------
    validation : dict   — values for HDF5 /validation group
    passed     : bool   — True iff checks 3, 4, 7 all pass
    """
    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)

    Ne = params['N_exc']
    Ni = params['N_inh']

    trains_E = _extract_spike_trains(net_objs['spike_E'], Ne, t_sim)
    trains_I = _extract_spike_trains(net_objs['spike_I'], Ni, t_sim)

    # ---- Check 3: CV-ISI (quantitative) ----
    _, mean_cv = compute_cv_isi(trains_E, 0.0, t_sim, min_spikes=20)
    cv_pass = not np.isnan(mean_cv) and 0.8 <= mean_cv <= 1.2

    # ---- Check 4: Pairwise correlation (quantitative) ----
    mean_r = compute_pairwise_corr(trains_E, 0.0, t_sim,
                                    bin_ms=10.0, n_pairs=50, seed=seed)
    pairwise_pass = (not np.isnan(mean_r)) and mean_r < 0.05

    # ---- Check 7: I/E rate ratio (quantitative) ----
    mean_rate_E = net_objs['spike_E'].num_spikes / (Ne * t_sim)
    mean_rate_I = net_objs['spike_I'].num_spikes / (Ni * t_sim)
    rate_ratio  = mean_rate_I / mean_rate_E if mean_rate_E > 0 else float('nan')
    rate_pass   = not np.isnan(rate_ratio) and 2.0 <= rate_ratio <= 3.0

    # ---- Print results ----
    width = 26
    print(f"\n{'=' * 50}")
    print(f"{'Validation results':^50}")
    print(f"{'=' * 50}")
    print(f"{'Check 3 (CV-ISI):':<{width}} {mean_cv:.3f}   "
          f"{'PASS' if cv_pass else 'FAIL'}  [target: 0.8–1.2]")
    print(f"{'Check 4 (pairwise r):':<{width}} {mean_r:.4f}  "
          f"{'PASS' if pairwise_pass else 'FAIL'}  [target: <0.05]")
    print(f"{'Check 7 (I/E rate ratio):':<{width}} {rate_ratio:.2f}    "
          f"{'PASS' if rate_pass else 'FAIL'}  [target: 2–3×]")
    print(f"{'Mean E rate:':<{width}} {mean_rate_E:.2f} Hz")
    print(f"{'Mean I rate:':<{width}} {mean_rate_I:.2f} Hz")
    print(f"{'=' * 50}\n")

    # ---- Figures (checks 1, 2, 5, 6) ----
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
        'mean_rate_E':        mean_rate_E,
        'mean_rate_I':        mean_rate_I,
        'mean_CV_ISI':        mean_cv,
        'mean_pairwise_corr': mean_r,
        'raster_times':       raster_t,
        'raster_indices':     raster_i,
    }

    all_passed = cv_pass and pairwise_pass and rate_pass
    return validation, all_passed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run Part 1 validation and save baseline network to HDF5.')
    parser.add_argument('--nu_ext', type=float, required=True,
                        help='Background Poisson rate (Hz)')
    parser.add_argument('--g_EI',   type=float, required=True,
                        help='Mean I→E inhibitory weight (nA)')
    parser.add_argument('--t_sim',  type=float, default=5.0,
                        help='Simulation duration in seconds (default: 5.0)')
    parser.add_argument('--seed',   type=int,   default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'results'),
                        help='Directory for results and figures')
    args = parser.parse_args()

    params = {
        **DEFAULT_PARAMS,
        'nu_ext': args.nu_ext,
        'g_EI':   args.g_EI * 1e-9,  # CLI takes nA; store as A internally
    }

    print(f"Building network: nu_ext={args.nu_ext} Hz, g_EI={args.g_EI} nA, "
          f"seed={args.seed}")
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
            f.write(f"nu_ext={args.nu_ext} Hz  g_EI={args.g_EI} nA  seed={args.seed}\n")
            f.write(f"CV_ISI={validation['mean_CV_ISI']:.4f}  "
                    f"pairwise_r={validation['mean_pairwise_corr']:.4f}  "
                    f"rate_ratio={rate_ratio:.2f}\n")
        print(f"One or more quantitative checks FAILED. "
              f"Report written to:\n  {report_path}")
        print("Re-tune (nu_ext, g_EI) and re-run.")
        sys.exit(1)


if __name__ == '__main__':
    main()
