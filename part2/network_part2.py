# part2/network_part2.py

"""
Part 2 network factory: loads the Part 1 baseline, adds pair-based STDP on
E->E synapses (Song, Miller & Abbott 2000), applies P/X/S pool rescaling, and
adds 50 task-input neurons connected to both E and I populations.

See docs/superpowers/specs/2026-06-10-part2-stdp-task-design.md for the full
design and parameter justifications.
"""

import h5py
import numpy as np
from brian2 import (
    Synapses, PoissonGroup, SpikeMonitor, Network,
    second, amp, Hz,
)

from part1.network import build_network, DEFAULT_PARAMS, _lognormal_weights


DEFAULT_PARAMS_PART2 = {
    # Subpopulation sizes (E neuron indices: P=[0,P_size), X=[P_size,P_size+X_size),
    # S=[P_size+X_size, N_exc)). Must satisfy P_size + X_size <= N_exc.
    'P_size': 350,
    'X_size': 350,

    # STDP (spec 2.1). w_max = 4x w_mean_EE; A_plus/A_minus = 0.01/0.0105 x w_max
    # (5% depression-dominant, the Song et al. 2000 stability condition).
    'tau_plus':  20e-3,      # s
    'tau_minus': 20e-3,      # s
    'w_max':     0.24e-9,    # A
    'A_plus':    0.0024e-9,  # A
    'A_minus':   0.00252e-9, # A

    # Task input (spec 2.3)
    'n_input':       50,
    'n_directions':  8,
    'r_max':         100.0,  # Hz
    'r_background':  2.0,    # Hz (ITI level)
    'exec_amplification': 1.5,

    # Trial timing (seconds)
    't_prep': 0.5,
    't_exec': 0.5,
    't_iti':  0.2,

    # Burn-in (seconds) — see spec section 3
    't_burn_in': 15.0,

    # Cross-pool (P<->X) weight scaling for the two conditions
    'p_cross_seeded':  0.2,
    'p_cross_control': 1.0,
}


def apply_pool_rescaling(i, j, w, p_cross, P_size, X_size):
    """
    Rescale E->E weights by P/X/S pool membership (spec 2.2).

    Pools by neuron index: P = [0, P_size), X = [P_size, P_size+X_size),
    S = everything else. Synapses crossing P<->X (in either direction) are
    multiplied by p_cross; all other synapses (within-pool, or touching S)
    are returned unchanged.

    Parameters
    ----------
    i, j : array_like of int   — presynaptic (i) / postsynaptic (j) indices
    w    : array_like of float — weights, same length as i and j
    p_cross : float            — cross-pool scale (0.2 seeded, 1.0 control)
    P_size, X_size : int       — sizes of the P and X pools

    Returns
    -------
    np.ndarray — rescaled copy of w (input is not mutated)
    """
    i = np.asarray(i)
    j = np.asarray(j)
    w_new = np.array(w, dtype=np.float64, copy=True)

    in_P = i < P_size
    in_X = (i >= P_size) & (i < P_size + X_size)
    j_in_P = j < P_size
    j_in_X = (j >= P_size) & (j < P_size + X_size)

    cross = (in_P & j_in_X) | (in_X & j_in_P)
    w_new[cross] *= p_cross
    return w_new


def load_part1_baseline(h5_path, params, seed=42):
    """
    Rebuild the Part 1 network and overwrite weights from the saved HDF5.

    build_network(params, seed=seed) is fully deterministic (Brian2's RNG and
    the numpy weight-init RNG are both seeded), so it reproduces the same
    connectivity and weights as baseline_network.h5. We additionally:

    1. Assert the reproduced (i, j) connectivity matches the saved (row, col)
       COO indices for all four synapse groups — a sanity check that `params`
       and `seed` match what produced the saved file.
    2. Overwrite `.w` from the saved `data` arrays directly, so Part 2 starts
       from the exact validated weights regardless of any future
       floating-point/library-version drift in step 1.

    Returns the same dict shape as build_network().
    """
    net_objs = build_network(params, seed=seed)

    with h5py.File(h5_path, 'r') as f:
        for name, syn in (
            ('W_EE', net_objs['syn_EE']),
            ('W_EI', net_objs['syn_EI']),
            ('W_IE', net_objs['syn_IE']),
            ('W_II', net_objs['syn_II']),
        ):
            saved_row = f[f'weights/{name}/row'][:]
            saved_col = f[f'weights/{name}/col'][:]
            saved_data = f[f'weights/{name}/data'][:]

            # .j = postsynaptic (row), .i = presynaptic (col) — matches the
            # convention in part1/run_part1.py's save_baseline().
            actual_row = np.array(syn.j[:], dtype=np.int32)
            actual_col = np.array(syn.i[:], dtype=np.int32)

            if actual_row.shape != saved_row.shape or not (
                np.array_equal(actual_row, saved_row)
                and np.array_equal(actual_col, saved_col)
            ):
                raise ValueError(
                    f"{name}: connectivity reproduced from build_network(seed="
                    f"{seed}) does not match {h5_path}. Check that `params` "
                    f"matches the params used to generate the baseline.")

            syn.w = saved_data.astype(np.float64) * amp

    return net_objs
