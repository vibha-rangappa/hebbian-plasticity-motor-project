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
