# geometry/preprocessing.py

"""
The preprocessing chokepoint for Part 3 geometry analysis.

Turns one training snapshot (per-trial spike trains) into the canonical, Churchland
et al. (2012)-style data matrix X that every downstream observable consumes. Computed
ONCE per snapshot; PR, jPCA, and orthogonality all read the same X without re-smoothing
or re-normalizing (this is what structurally enforces "no duplicate results").

Pipeline (see spec section 1):
    spikes -> 1 ms counts -> Gaussian smooth (sigma=25 ms) -> Hz
           -> window + downsample to 10 ms bins
           -> condition-average (5 trials/direction -> 8 condition PSTHs)
           -> soft-normalize  r / (R_i + 5 Hz)
           -> subtract cross-condition mean at each timepoint

Key design choices baked in here:
  - E neurons only (indices 0..n_exc-1). The 50 task-input neurons carry the imposed
    cosine tuning by construction; including them would inject exactly the structure the
    analysis tests for. (I neurons are not recorded in the snapshots.)
  - The soft-normalization range R_i is computed ONCE over the full task period (prep +
    exec, all conditions) and reused for both windows, so prep and exec stay on the same
    per-neuron scale -- this matters for the prep/exec orthogonality comparison.
  - Per-trial rates are retained so trial-split cross-validation (controls.py) can
    re-form condition averages from a subset of trials using the SAME fixed R_i.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


# Trial structure (ms), matching the center-out task in plasticity/center_out_task.py.
TRIAL_DUR_MS = 1200
PREP_MS = (0, 500)
EXEC_MS = (500, 1000)
# ITI [1000, 1200) is background drive and is dropped.


def condition_mean(trial_rates, trial_labels, n_conditions):
    """
    Average per-trial rates within each condition (reach direction).

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
    Build the centered, soft-normalized condition-averaged data matrix X (N, T, C)
    from a (sub)set of trials.

    Used both for the full X (idx=None -> all trials) and for trial-split CV folds
    (idx = the fold's trial indices). R (the soft-norm range) is passed in fixed, so
    every fold uses the same normalization -- normalization is a preprocessing choice,
    not something refit per fold.
    """
    if idx is not None:
        trial_rates = trial_rates[idx]
        trial_labels = trial_labels[idx]
    cond = condition_mean(trial_rates, trial_labels, n_conditions)   # (N, T, C)
    norm = cond / (R[:, None, None] + r_floor)
    # Subtract the cross-condition mean at each timepoint (removes the
    # condition-independent component). This also zeroes each neuron's mean over all
    # (t, c) samples, so no extra PCA centering is needed downstream.
    X = norm - norm.mean(axis=2, keepdims=True)
    return X


def compute_trial_rates(snapshot, n_exc=800, sigma_ms=25.0, bin_ms=1.0,
                        downsample_ms=10):
    """
    Smooth each trial's spike train into a downsampled firing-rate estimate (Hz),
    split into the prep and exec windows.

    Returns dict with:
        trial_rate_prep, trial_rate_exec : (n_trials, n_exc, T) arrays, T = 50
        trial_labels                     : (n_trials,)
        time_prep_ms, time_exec_ms       : (T,) bin-center times
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
    hz_factor = 1000.0 / bin_ms   # counts-per-bin -> Hz

    # E neurons only.
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
        # Area-preserving Gaussian smoothing; mode='constant' = zero rate outside trial.
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
    Full preprocessing of one snapshot into the canonical data matrices.

    Returns dict with the full-trial centered matrices X_prep, X_exec (each (N, T, C)),
    plus everything trial-split CV needs to re-form folds: the per-trial rates, the
    fixed soft-norm range R, r_floor, and trial_labels.
    """
    tr = compute_trial_rates(snapshot, n_exc=n_exc, sigma_ms=sigma_ms,
                             downsample_ms=downsample_ms)
    C = tr['n_conditions']

    # Soft-norm range over the FULL task period (prep + exec), all conditions, computed
    # once and reused for both windows.
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
