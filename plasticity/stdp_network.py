# plasticity/stdp_network.py

"""
STDP network factory: loads the circuit baseline, adds pair-based STDP on
E->E synapses (Song, Miller & Abbott 2000), applies P/X/S pool rescaling, and
adds 50 task-input neurons connected to both E and I populations.

See docs/superpowers/specs/2026-06-10-stdp-center-out-task-design.md for the
full design and parameter justifications.
"""

import h5py
import numpy as np
from brian2 import (
    Synapses, PoissonGroup, SpikeMonitor, Network,
    second, amp, Hz,
)

from circuit.network import build_network, DEFAULT_PARAMS, _lognormal_weights


DEFAULT_PARAMS_PLASTICITY = {
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

    # Inhibitory STDP (Vogels et al. 2011) on I->E synapses. Off by default;
    # enabled via build_stdp_network(inhibitory_plasticity=True). Stabilizes the
    # network under E->E STDP by driving each E neuron toward rho0.
    'tau_istdp': 20e-3,     # s    inhibitory trace time constant
    'rho0':      3.0,       # Hz   target postsynaptic E rate (the AI operating point)
    'eta_istdp': 1e-12,     # A    inhibitory learning rate
    'w_max_inh': 0.90e-9,   # A    upper bound on inhibitory weight (10x g_EI)
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


def load_baseline(h5_path, params, seed=42):
    """
    Rebuild the circuit network and overwrite weights from the saved HDF5.

    build_network(params, seed=seed) is fully deterministic (Brian2's RNG and
    the numpy weight-init RNG are both seeded), so it reproduces the same
    connectivity and weights as baseline_network.h5. We additionally:

    1. Assert the reproduced (i, j) connectivity matches the saved (row, col)
       COO indices for all four synapse groups — a sanity check that `params`
       and `seed` match what produced the saved file.
    2. Overwrite `.w` from the saved `data` arrays directly, so training starts
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
            # convention in circuit/run_baseline.py's save_baseline().
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


def compute_target_insums(j_arr, w, n_exc):
    """
    Per-postsynaptic-neuron sum of incoming E->E weights: target[j] = sum of w over
    synapses whose postsynaptic index is j. Captured once at build time and held fixed
    as the normalization target.
    """
    return np.bincount(np.asarray(j_arr), weights=np.asarray(w, dtype=np.float64),
                       minlength=n_exc)


def rescale_to_target(post, w, target, w_max):
    """
    Multiplicative synaptic scaling (pure numpy core). For each postsynaptic neuron,
    rescale its incoming weights so they sum to target[j], then clip to [0, w_max].

    Multiplicative (not subtractive) so the *relative* weight pattern STDP learned is
    preserved while the per-neuron total is held constant (Turrigiano-style scaling).
    This converts STDP from mean-weight drift (which drove the runaway we observed) into
    pure redistribution, keeping the network at its balanced AI operating point.

    Neurons whose current incoming sum is 0 are left untouched (factor 1).
    Order: scale, then clip -- clipping can leave a neuron's sum slightly below target
    (only when a weight hits w_max), an acceptable small drift.
    """
    post = np.asarray(post)
    w = np.asarray(w, dtype=np.float64)
    n = len(target)
    cur = np.bincount(post, weights=w, minlength=n)
    factor = np.ones(n)
    nz = cur > 0
    factor[nz] = target[nz] / cur[nz]
    return np.clip(w * factor[post], 0.0, w_max)


def normalize_incoming_weights(syn, target, w_max):
    """Apply rescale_to_target in place to a Brian2 E->E synapse group (weights in amp)."""
    post = np.asarray(syn.j[:])
    w = np.asarray(syn.w[:] / amp)
    syn.w = rescale_to_target(post, w, target, w_max) * amp


def build_inhibitory_stdp_synapse(inh, exc, static_syn_IE, params):
    """
    Build a plastic I->E synapse group implementing the Vogels et al. (2011) inhibitory
    STDP rule, preserving the connectivity (i, j) and initial weights of the static
    baseline syn_IE.

    The rule (see spec): a symmetric Hebbian rule on inhibitory synapses that drives each
    postsynaptic E neuron toward rho0. alpha = 2*rho0*tau_istdp is the depression constant
    that sets the target rate. Weights stay in [0, w_max_inh]; the postsynaptic membrane
    convention matches circuit/network.py (I_inh_post += w, entered as -I_inh in dv/dt).
    """
    p = params
    i_arr = np.array(static_syn_IE.i[:], dtype=np.int32)
    j_arr = np.array(static_syn_IE.j[:], dtype=np.int32)
    w_arr = np.array(static_syn_IE.w[:] / amp, dtype=np.float64)

    alpha = 2.0 * p['rho0'] * p['tau_istdp']     # dimensionless target-rate setpoint
    istdp_ns = {
        'tau_istdp': p['tau_istdp'] * second,
        'eta_istdp': p['eta_istdp'] * amp,
        'w_max_inh': p['w_max_inh'] * amp,
        'alpha': alpha,
    }
    istdp_eqs = '''
    w : amp
    dapre_i/dt  = -apre_i  / tau_istdp : 1 (event-driven)
    dapost_i/dt = -apost_i / tau_istdp : 1 (event-driven)
    '''
    # on_pre = inhibitory neuron spikes; on_post = excitatory neuron spikes.
    on_pre_eqs = '''
    I_inh_post += w
    apre_i += 1
    w = clip(w + eta_istdp * (apost_i - alpha), 0*amp, w_max_inh)
    '''
    on_post_eqs = '''
    apost_i += 1
    w = clip(w + eta_istdp * apre_i, 0*amp, w_max_inh)
    '''
    syn = Synapses(inh, exc, istdp_eqs, on_pre=on_pre_eqs, on_post=on_post_eqs,
                   namespace=istdp_ns, method='euler', name='syn_IE_istdp')
    syn.connect(i=i_arr, j=j_arr)
    syn.w = w_arr * amp
    syn.apre_i = 0
    syn.apost_i = 0
    return syn


def build_stdp_network(net_objs, params, p_cross, seed=42, inhibitory_plasticity=False):
    """
    Replace syn_EE with a plastic STDP synapse group (pool-rescaled initial
    weights, spec 2.1/2.2) and add 50 task-input neurons connected to both E
    and I populations (spec 2.3).

    Returns an updated net_objs dict: same keys as build_network()/
    load_baseline(), with 'syn_EE' replaced by the STDP group, plus
    'input_group', 'syn_input_E', 'syn_input_I', 'spike_input' added, and a
    fresh Network() containing all active components. The original syn_EE
    (and the Network it was part of) is left intact but unused.
    """
    p = params
    old_syn_EE = net_objs['syn_EE']
    exc, inh = net_objs['exc'], net_objs['inh']

    i_arr = np.array(old_syn_EE.i[:], dtype=np.int32)
    j_arr = np.array(old_syn_EE.j[:], dtype=np.int32)
    w_arr = np.array(old_syn_EE.w[:] / amp, dtype=np.float64)

    # The circuit's lognormal W_EE is unbounded, but the STDP on_pre/on_post
    # clip(..., 0, w_max) below applies on every spike regardless of
    # `plastic` -- a synapse with w > w_max would be silently clamped to
    # w_max on its first spike (even during the frozen burn-in). Clip here,
    # before pool rescaling, so initial weights don't depend on burn-in
    # spike timing and the seeded/control cross-pool ratio is exactly
    # p_cross even for synapses whose baseline weight exceeded w_max.
    w_arr = np.clip(w_arr, 0.0, p['w_max'])

    w_rescaled = apply_pool_rescaling(
        i_arr, j_arr, w_arr, p_cross, p['P_size'], p['X_size'])

    stdp_ns = {
        'tau_plus':  p['tau_plus']  * second,
        'tau_minus': p['tau_minus'] * second,
        'A_plus':    p['A_plus']    * amp,
        'A_minus':   p['A_minus']   * amp,
        'w_max':     p['w_max']     * amp,
    }

    # Pair-based STDP, event-driven traces (Song, Miller & Abbott 2000).
    # Depression on presynaptic spike (acausal: post fired recently);
    # potentiation on postsynaptic spike (causal: pre fired recently).
    # `plastic` is a shared flag: 0 freezes weight changes (traces still
    # update) for burn-in and snapshot test trials.
    #
    # NOTE: the traces are named `apre`/`apost` rather than `x_pre`/`x_post`
    # (the spec's naming) because Brian2 reserves any synaptic variable name
    # ending in `_pre`/`_post` for referring to the corresponding pre-/
    # post-synaptic *group* variable (e.g. `v_pre` == `exc.v` of the
    # presynaptic neuron) and raises a ValueError if you try to declare one.
    # `apre`/`apost` is the standard Brian2 STDP-tutorial naming.
    stdp_eqs = '''
    w : amp
    plastic : 1 (shared)
    dapre/dt  = -apre  / tau_plus  : 1 (event-driven)
    dapost/dt = -apost / tau_minus : 1 (event-driven)
    '''
    on_pre_eqs = '''
    I_exc_post += w
    apre += 1
    w = clip(w - plastic * A_minus * apost, 0*amp, w_max)
    '''
    on_post_eqs = '''
    apost += 1
    w = clip(w + plastic * A_plus * apre, 0*amp, w_max)
    '''

    syn_EE_stdp = Synapses(
        exc, exc, stdp_eqs,
        on_pre=on_pre_eqs, on_post=on_post_eqs,
        namespace=stdp_ns, method='euler', name='syn_EE_stdp')
    syn_EE_stdp.connect(i=i_arr, j=j_arr)
    syn_EE_stdp.w = w_rescaled * amp
    syn_EE_stdp.plastic = 1
    syn_EE_stdp.apre = 0
    syn_EE_stdp.apost = 0

    # Task-input neurons: 50 Poisson units, connected to both E and I at the
    # same density as recurrent connectivity (p_connect), with static
    # (non-plastic) lognormal weights. Drawn from a separate RNG stream
    # (seed + 1000) so input-weight draws don't shift the recurrent network's
    # weight draws inside build_network().
    n_input = p['n_input']
    input_group = PoissonGroup(
        n_input, rates=np.full(n_input, p['r_background']) * Hz,
        name='input_group')

    syn_input_E = Synapses(input_group, exc, 'w : amp',
                            on_pre='I_exc_post += w', name='syn_input_E')
    syn_input_I = Synapses(input_group, inh, 'w : amp',
                            on_pre='I_exc_post += w', name='syn_input_I')
    syn_input_E.connect(p=p['p_connect'])
    syn_input_I.connect(p=p['p_connect'])

    input_rng = np.random.default_rng(seed + 1000)
    syn_input_E.w = _lognormal_weights(
        p['w_mean_EE'], p['sigma_w'], len(syn_input_E), input_rng) * amp
    syn_input_I.w = _lognormal_weights(
        p['w_mean_EE'], p['sigma_w'], len(syn_input_I), input_rng) * amp

    spike_input = SpikeMonitor(input_group)

    # Optionally replace the static I->E synapses with a plastic Vogels-rule group.
    if inhibitory_plasticity:
        syn_IE = build_inhibitory_stdp_synapse(exc=exc, inh=inh,
                                               static_syn_IE=net_objs['syn_IE'],
                                               params=p)
    else:
        syn_IE = net_objs['syn_IE']

    net = Network(
        exc, inh,
        syn_EE_stdp, net_objs['syn_EI'], syn_IE, net_objs['syn_II'],
        net_objs['drive_E'], net_objs['drive_I'],
        net_objs['spike_E'], net_objs['spike_I'],
        input_group, syn_input_E, syn_input_I, spike_input,
    )

    result = dict(net_objs)
    result['syn_EE'] = syn_EE_stdp
    result['syn_IE'] = syn_IE
    result['inhibitory_plasticity'] = inhibitory_plasticity
    # Per-postsynaptic-neuron target incoming-weight sum, captured from the initial
    # (rescaled, clipped) weights. Held fixed as the synaptic-scaling target so STDP
    # redistributes rather than inflates each neuron's total excitatory drive.
    result['W_target_EE'] = compute_target_insums(j_arr, w_rescaled, p['N_exc'])
    result['input_group'] = input_group
    result['syn_input_E'] = syn_input_E
    result['syn_input_I'] = syn_input_I
    result['spike_input'] = spike_input
    result['net'] = net
    return result
