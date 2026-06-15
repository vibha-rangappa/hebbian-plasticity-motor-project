# geometry/preprocessing.py

"""
This file is the single shared preprocessing step for all of the Part 3 geometry
analyses.

It takes one training snapshot (the raw per-trial spike trains) and turns it into
the standard data matrix X, built the way Churchland et al. (2012) build their data
matrices, that every downstream analysis (participation ratio, jPCA, orthogonality)
then reads. This is done ONCE per snapshot, so PR, jPCA, and orthogonality all look
at exactly the same X, with no re-smoothing or re-normalizing happening separately
in each analysis. That's what guarantees we're not accidentally getting different
numbers from the same data just because of preprocessing differences.

Pipeline (see spec section 1):
    spikes -> 1 ms bin counts -> Gaussian smoothing (sigma = 25 ms) -> convert to Hz
           -> cut into a window and downsample to 10 ms bins
           -> average across trials within each condition (5 trials/direction -> 8
              condition PSTHs, i.e. one average firing-rate curve per direction)
           -> soft-normalize: r / (R_i + 5 Hz)
           -> subtract the across-condition mean at each timepoint

Key design choices baked in here:
  - Only excitatory (E) neurons are used (indices 0..n_exc-1). The 50 task-input
    neurons are deliberately given a cosine-shaped tuning to direction, so including
    them would inject exactly the kind of structure we're trying to test for in the
    rest of the network. (Inhibitory, I, neurons aren't recorded in the snapshots at
    all.)
  - The soft-normalization range R_i (how much each neuron's rate varies) is computed
    ONCE, using the whole task period (prep + exec, all conditions), and then reused
    for both the prep and exec windows. That way prep and exec activity stay on the
    same per-neuron scale, which matters when we later compare the prep and exec
    subspaces for orthogonality.
  - We keep the per-trial firing rates around (not just the condition averages) so
    that trial-split cross-validation (in controls.py) can recompute condition
    averages from a subset of trials, while still using the SAME fixed R_i computed
    from all the data.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


# Trial timing (in ms), matching the center-out task in plasticity/center_out_task.py.
TRIAL_DUR_MS = 1200
PREP_MS = (0, 500)
EXEC_MS = (500, 1000)
# The inter-trial interval [1000, 1200) is just background drive, so we drop it.


def condition_mean(trial_rates, trial_labels, n_conditions):
    """
    Average the per-trial firing rates within each condition (each reach direction).

    trial_rates: (n_trials, N, T)
    trial_labels: (n_trials,) direction index in [0, n_conditions)
    returns: (N, T, n_conditions)
    """
    N, T = trial_rates.shape[1], trial_rates.shape[2]
    out = np.empty((N, T, n_conditions), dtype=np.float64)
    for c in range(n_conditions):
        sel = trial_labels == c
        if not np.any(sel):
            raise ValueError(f"No trials for condition {c}; cannot condition-average.")
        out[:, :, c] = trial_rates[sel].mean(axis=0)
    return out


def make_X(trial_rates, trial_labels, R, r_floor, n_conditions, idx=None):
    """
    Build the final data matrix X (N, T, C): condition-averaged, soft-normalized,
    and centered, from some set of trials.

    Used both to build the full X (idx=None means use all trials) and to build
    trial-split cross-validation folds (idx = the trial indices for that fold). R
    (the soft-normalization range) is always passed in already computed from the
    full dataset, so every fold uses the same normalization. Normalization is a
    preprocessing choice we fix up front, not something we recompute per fold.
    """
    if idx is not None:
        trial_rates = trial_rates[idx]
        trial_labels = trial_labels[idx]
    cond = condition_mean(trial_rates, trial_labels, n_conditions)   # (N, T, C)
    norm = cond / (R[:, None, None] + r_floor)
    # Subtract the across-condition mean at each timepoint. This removes the part of
    # the activity that's the same regardless of reach direction, leaving only the
    # condition-dependent structure. As a side effect, it also makes each neuron's
    # average (over all time x condition samples) zero, so we don't need to do any
    # extra centering before PCA later.
    X = norm - norm.mean(axis=2, keepdims=True)
    return X


def compute_trial_rates(snapshot, n_exc=800, sigma_ms=25.0, bin_ms=1.0,
                        downsample_ms=10):
    """
    Turn each trial's raw spike train into a smoothed, downsampled firing-rate
    estimate (in Hz), and split it into the prep and exec windows.

    Returns dict with:
        trial_rate_prep, trial_rate_exec : (n_trials, n_exc, T) arrays, T = 50
        trial_labels                     : (n_trials,)
        time_prep_ms, time_exec_ms       : (T,) time at the center of each bin
        n_conditions                     : int
    """
    times = np.asarray(snapshot['spike_times_ms'], dtype=np.float64)
    neurons = np.asarray(snapshot['spike_neuron_idx'])
    trials = np.asarray(snapshot['spike_trial_idx'])
    labels = np.asarray(snapshot['trial_labels'])
    n_trials = len(labels)
    n_conditions = int(labels.max()) + 1

    n_bins = int(round(TRIAL_DUR_MS / bin_ms))
    sigma_bins = sigma_ms / bin_ms
    ds = int(round(downsample_ms / bin_ms))
    hz_factor = 1000.0 / bin_ms   # converts spike count per bin into a rate in Hz

    # Keep only excitatory (E) neurons.
    e_mask = neurons < n_exc
    times_e = times[e_mask]
    neurons_e = neurons[e_mask]
    trials_e = trials[e_mask]

    def window_ds(rate_full, bounds):
        lo = int(round(bounds[0] / bin_ms))
        hi = int(round(bounds[1] / bin_ms))
        seg = rate_full[:, lo:hi]
        T = (hi - lo) // ds
        return seg.reshape(rate_full.shape[0], T, ds).mean(axis=2)

    prep_list, exec_list = [], []
    for tr in range(n_trials):
        m = trials_e == tr
        counts = np.zeros((n_exc, n_bins), dtype=np.float64)
        if np.any(m):
            bidx = np.clip((times_e[m] / bin_ms).astype(int), 0, n_bins - 1)
            np.add.at(counts, (neurons_e[m], bidx), 1.0)
        # Smooth the spike counts with a Gaussian (this spreads each spike out over
        # nearby bins, so the total count is preserved). mode='constant' means we
        # treat everything outside the trial as zero rate.
        rate = gaussian_filter1d(counts, sigma=sigma_bins, axis=1, truncate=3.0,
                                 mode='constant') * hz_factor
        prep_list.append(window_ds(rate, PREP_MS))
        exec_list.append(window_ds(rate, EXEC_MS))

    T = (PREP_MS[1] - PREP_MS[0]) // ds
    time_prep = PREP_MS[0] + (np.arange(T) + 0.5) * downsample_ms
    time_exec = EXEC_MS[0] + (np.arange(T) + 0.5) * downsample_ms

    return {
        'trial_rate_prep': np.stack(prep_list),
        'trial_rate_exec': np.stack(exec_list),
        'trial_labels': labels,
        'time_prep_ms': time_prep,
        'time_exec_ms': time_exec,
        'n_conditions': n_conditions,
    }


def preprocess_snapshot(snapshot, n_exc=800, sigma_ms=25.0, downsample_ms=10,
                        r_floor=5.0):
    """
    Run the full preprocessing pipeline on one snapshot to get the standard data
    matrices.

    Returns dict with the full-trial centered matrices X_prep, X_exec (each (N, T, C)),
    plus everything trial-split cross-validation needs to rebuild its own folds: the
    per-trial rates, the fixed soft-normalization range R, r_floor, and trial_labels.
    """
    tr = compute_trial_rates(snapshot, n_exc=n_exc, sigma_ms=sigma_ms,
                             downsample_ms=downsample_ms)
    C = tr['n_conditions']

    # Compute the soft-normalization range using the FULL task period (prep + exec,
    # all conditions) just once, so it can be reused for both windows below.
    cond_prep = condition_mean(tr['trial_rate_prep'], tr['trial_labels'], C)
    cond_exec = condition_mean(tr['trial_rate_exec'], tr['trial_labels'], C)
    full = np.concatenate([cond_prep, cond_exec], axis=1)   # (N, 2T, C)
    R = full.max(axis=(1, 2)) - full.min(axis=(1, 2))       # (N,)

    X_prep = make_X(tr['trial_rate_prep'], tr['trial_labels'], R, r_floor, C)
    X_exec = make_X(tr['trial_rate_exec'], tr['trial_labels'], R, r_floor, C)

    return {
        'X_prep': X_prep,
        'X_exec': X_exec,
        'trial_rate_prep': tr['trial_rate_prep'],
        'trial_rate_exec': tr['trial_rate_exec'],
        'trial_labels': tr['trial_labels'],
        'R': R,
        'r_floor': r_floor,
        'n_conditions': C,
        'time_prep_ms': tr['time_prep_ms'],
        'time_exec_ms': tr['time_exec_ms'],
    }
