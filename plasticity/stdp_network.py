# plasticity/stdp_network.py

"""
This file builds the plastic version of the network used for training.

It takes the static circuit baseline and:
  - adds pair-based STDP to the E->E (excitatory-to-excitatory) synapses
    (the rule from Song, Miller & Abbott 2000),
  - rescales weights between the P, X, and S neuron pools,
  - adds 50 task-input neurons that connect to both the E and I populations.

It also has the option to make the I->E (inhibitory-to-excitatory) synapses
plastic too, using the inhibitory STDP rule from Vogels et al. 2011, which
homeostatically pulls each excitatory neuron's firing rate toward a target
rate (rho0).

See README.md, "Part 2: Task and plasticity", for the full design and the
reasoning behind the parameter choices.
"""

import h5py
import numpy as np
from brian2 import (
    Synapses, PoissonGroup, SpikeMonitor, Network,
    second, amp, Hz,
)

from circuit.network import build_network, DEFAULT_PARAMS, _lognormal_weights


DEFAULT_PARAMS_PLASTICITY = {
    # How the excitatory neurons are split into pools by index:
    # P = neurons [0, P_size), X = neurons [P_size, P_size+X_size), S = everything
    # else. Need P_size + X_size <= N_exc.
    'P_size': 350,
    'X_size': 350,

    # STDP settings (spec 2.1). w_max is set to 4x the mean E->E weight.
    # A_plus and A_minus are 0.01x and 0.0105x w_max, so depression is about
    # 5% stronger than potentiation. This 5%-depression-dominant setting is
    # the Song et al. 2000 condition for keeping the weights stable.
    'tau_plus':  20e-3,      # s
    'tau_minus': 20e-3,      # s
    'w_max':     0.24e-9,    # A
    'A_plus':    0.0024e-9,  # A
    'A_minus':   0.00252e-9, # A

    # Task input settings (spec 2.3)
    'n_input':       50,
    'n_directions':  8,
    'r_max':         100.0,  # Hz
    'r_background':  2.0,    # Hz (rate during the inter-trial interval)
    'exec_amplification': 1.5,

    # Trial timing (seconds)
    't_prep': 0.5,
    't_exec': 0.5,
    't_iti':  0.2,

    # How long to run the network before training starts, in seconds, to let
    # it settle into its steady-state firing pattern (see spec section 3).
    't_burn_in': 15.0,

    # Scaling factor applied to weights crossing between the P and X pools,
    # one value per condition.
    'p_cross_seeded':  0.2,
    'p_cross_control': 1.0,

    # Inhibitory STDP (Vogels et al. 2011) on the I->E synapses. Off by
    # default, turned on via build_stdp_network(inhibitory_plasticity=True).
    # This rule helps keep the network stable under E->E STDP by pushing each
    # excitatory neuron's firing rate toward rho0.
    'tau_istdp': 20e-3,     # s    time constant of the inhibitory STDP trace
    'rho0':      3.0,       # Hz   target firing rate for excitatory neurons (the network's normal operating point)
    'eta_istdp': 1e-12,     # A    inhibitory learning rate (how big each weight step is)
    'w_max_inh': 0.90e-9,   # A    upper limit on an inhibitory weight (10x the baseline I->E weight, g_EI)
}


def apply_pool_rescaling(i, j, w, p_cross, P_size, X_size):
    """
    Scale down E->E weights for synapses that cross between the P and X
    neuron pools (spec 2.2).

    Pools are defined by neuron index: P = [0, P_size), X = [P_size,
    P_size+X_size), and S = everything else. Any synapse that goes from P to
    X or from X to P gets multiplied by p_cross. All other synapses (within a
    pool, or touching the S pool) are left unchanged.

    Parameters
    ----------
    i, j : array_like of int, presynaptic (i) / postsynaptic (j) neuron indices
    w    : array_like of float, weights, same length as i and j
    p_cross : float, scale factor for cross-pool synapses (0.2 for seeded, 1.0 for control)
    P_size, X_size : int, sizes of the P and X pools

    Returns
    -------
    np.ndarray, rescaled copy of w (the input array is not changed)
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
    Rebuild the circuit network from scratch, then overwrite its weights with
    the ones saved in the HDF5 baseline file.

    build_network(params, seed=seed) is fully deterministic (both Brian2's
    random number generator and the numpy one used for weight initialization
    are seeded), so calling it again reproduces the same connectivity and
    weights as baseline_network.h5. On top of that, this function:

    1. Checks that the connectivity (i, j) it just built matches the saved
       (row, col) indices for all four synapse groups. This is a sanity check
       that `params` and `seed` match whatever was used to create the saved
       file.
    2. Overwrites `.w` directly from the saved `data` arrays, so training
       starts from the exact same weights that were validated before, even if
       some future change to floating-point handling or library versions
       would otherwise cause tiny differences in step 1.

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

            # .j is the postsynaptic neuron (saved as "row"), .i is the
            # presynaptic neuron (saved as "col"). This matches the
            # convention used in circuit/run_baseline.py's save_baseline().
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
    For each excitatory neuron, add up the weights of all its incoming E->E
    synapses: target[j] = sum of w over synapses whose postsynaptic neuron is
    j. This is computed once when the network is built and kept fixed as the
    target total for synaptic scaling.
    """
    return np.bincount(np.asarray(j_arr), weights=np.asarray(w, dtype=np.float64),
                       minlength=n_exc)


def rescale_to_target(post, w, target, w_max):
    """
    The plain-numpy core of synaptic scaling. For each postsynaptic neuron,
    rescale its incoming weights so they add up to target[j], then clip the
    result to [0, w_max].

    This is a multiplicative rescale (every incoming weight is multiplied by
    the same factor), not a subtractive one. That way the *pattern* of
    relative weight sizes that STDP has learned is kept, but each neuron's
    total incoming weight stays the same (this is "Turrigiano-style" synaptic
    scaling). Without this, STDP slowly drifts the average weight up, which is
    what caused the runaway growth we saw before. With this rescaling, STDP
    just redistributes weight among synapses instead of inflating the total,
    so the network stays at its normal balanced operating point.

    If a neuron's current incoming sum is 0, it is left alone (scale factor
    of 1). We scale first and then clip, so clipping can occasionally leave a
    neuron's total slightly below its target (only when a weight hits
    w_max). That small amount of drift is fine.
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
    """Apply rescale_to_target in place to a Brian2 E->E synapse group (weights are in amp)."""
    post = np.asarray(syn.j[:])
    w = np.asarray(syn.w[:] / amp)
    syn.w = rescale_to_target(post, w, target, w_max) * amp


def build_inhibitory_stdp_synapse(inh, exc, static_syn_IE, params):
    """
    Build a plastic I->E synapse group that implements the Vogels et al. (2011)
    inhibitory STDP rule. It keeps the same connectivity (i, j) and starting
    weights as the static baseline syn_IE, just makes the weights plastic.

    This rule is a "symmetric Hebbian" rule on inhibitory synapses: it
    nudges each excitatory neuron's firing rate toward the target rate rho0.
    alpha = 2*rho0*tau_istdp is the constant that sets this target rate
    (it appears in the depression term below). Weights are kept in the range
    [0, w_max_inh]. The way inhibition affects the membrane potential matches
    circuit/network.py: each inhibitory spike adds w to I_inh, and I_inh
    enters the voltage equation with a minus sign (so inhibition).
    """
    p = params
    i_arr = np.array(static_syn_IE.i[:], dtype=np.int32)
    j_arr = np.array(static_syn_IE.j[:], dtype=np.int32)
    w_arr = np.array(static_syn_IE.w[:] / amp, dtype=np.float64)

    alpha = 2.0 * p['rho0'] * p['tau_istdp']     # this is a unitless number that sets the target firing rate
    istdp_ns = {
        'tau_istdp': p['tau_istdp'] * second,
        'eta_istdp': p['eta_istdp'] * amp,
        'w_max_inh': p['w_max_inh'] * amp,
        'alpha': alpha,
    }
    # Each synapse has a weight w, plus two "traces" (apre_i and apost_i)
    # that track recent spikes. Each trace just decays exponentially back
    # to 0 with time constant tau_istdp when nothing is spiking.
    istdp_eqs = '''
    w : amp
    dapre_i/dt  = -apre_i  / tau_istdp : 1 (event-driven)
    dapost_i/dt = -apost_i / tau_istdp : 1 (event-driven)
    '''
    # on_pre runs every time the inhibitory (presynaptic) neuron fires;
    # on_post runs every time the excitatory (postsynaptic) neuron fires.
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
    Replace the static syn_EE synapses with a plastic STDP synapse group
    (using initial weights that have already been rescaled by pool, per spec
    2.1/2.2), and add 50 task-input neurons connected to both the E and I
    populations (spec 2.3).

    Returns an updated net_objs dict: same keys as build_network() /
    load_baseline(), but with 'syn_EE' now pointing to the new STDP group,
    plus new entries 'input_group', 'syn_input_E', 'syn_input_I',
    'spike_input', and a fresh Network() containing everything that's
    actually used. The original syn_EE (and the Network it used to belong to)
    is left as-is but is no longer used.
    """
    p = params
    old_syn_EE = net_objs['syn_EE']
    exc, inh = net_objs['exc'], net_objs['inh']

    i_arr = np.array(old_syn_EE.i[:], dtype=np.int32)
    j_arr = np.array(old_syn_EE.j[:], dtype=np.int32)
    w_arr = np.array(old_syn_EE.w[:] / amp, dtype=np.float64)

    # The circuit's starting weights (drawn from a lognormal distribution) are
    # not capped, but the STDP on_pre/on_post rules below always clip the
    # weight to [0, w_max], on every spike, regardless of whether `plastic`
    # is on. So a synapse that starts above w_max would get silently clamped
    # down to w_max the first time it spikes, even during the frozen burn-in.
    # To avoid that surprise, we clip here first, before doing the pool
    # rescaling. This way the starting weights don't depend on the timing of
    # burn-in spikes, and the seeded/control cross-pool ratio comes out to
    # exactly p_cross even for synapses that started above w_max.
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

    # This is the standard pair-based STDP rule (Song, Miller & Abbott 2000),
    # with two trace variables that update only on spikes ("event-driven").
    #
    # Each synapse keeps two traces, apre and apost, that record how recently
    # the presynaptic and postsynaptic neurons fired. Between spikes they
    # just decay back toward 0 (apre with time constant tau_plus, apost with
    # tau_minus).
    #
    # When the presynaptic neuron fires (on_pre):
    #   - it adds its current weight w to the postsynaptic neuron's input
    #     (I_exc_post += w)
    #   - it bumps up its own apre trace by 1
    #   - it weakens the weight (depression) by an amount proportional to
    #     A_minus times the current apost trace. apost being large here means
    #     "the postsynaptic neuron fired recently, before this presynaptic
    #     spike" (i.e. the wrong order for causality), so the synapse gets
    #     weaker.
    #
    # When the postsynaptic neuron fires (on_post):
    #   - it bumps up its own apost trace by 1
    #   - it strengthens the weight (potentiation) by an amount proportional
    #     to A_plus times the current apre trace. apre being large here means
    #     "the presynaptic neuron fired recently, before this postsynaptic
    #     spike" (the causal order), so the synapse gets stronger.
    #
    # In both cases the weight is clipped back into [0, w_max] right away.
    #
    # `plastic` is a shared on/off switch: when it's 0, the weight-change
    # terms are multiplied by 0 so weights stop changing (but the apre/apost
    # traces keep updating). This is used to freeze learning during burn-in
    # and during snapshot test trials.
    #
    # NOTE: the traces are named `apre`/`apost` rather than `x_pre`/`x_post`
    # (which is what the spec calls them) because Brian2 treats any synaptic
    # variable name ending in `_pre`/`_post` as a reference to a variable on
    # the connected neuron group itself (e.g. `v_pre` would mean `exc.v` of
    # the presynaptic neuron), and raises an error if you try to declare your
    # own variable with that kind of name. `apre`/`apost` is just the naming
    # Brian2's own STDP tutorial uses, to avoid this clash.
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

    # Task-input neurons: 50 independent Poisson spike sources, connected to
    # both the E and I populations at the same connection density as the
    # recurrent connections (p_connect). Their weights are fixed (not
    # plastic) and drawn from the same lognormal distribution as the
    # recurrent weights. We use a separate random number generator (seeded
    # with seed + 1000) for these weight draws, so that adding this input
    # doesn't change any of the random draws used inside build_network() for
    # the recurrent network.
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

    # If requested, swap in the plastic Vogels-rule I->E synapses instead of
    # the static ones.
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
    # For each excitatory neuron, this is the total incoming E->E weight at
    # the start (after clipping and pool rescaling). It's kept fixed as the
    # synaptic-scaling target, so that STDP redistributes weight among a
    # neuron's synapses rather than inflating its total excitatory input.
    result['W_target_EE'] = compute_target_insums(j_arr, w_rescaled, p['N_exc'])
    result['input_group'] = input_group
    result['syn_input_E'] = syn_input_E
    result['syn_input_I'] = syn_input_I
    result['spike_input'] = spike_input
    result['net'] = net
    return result
