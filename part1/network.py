# part1/network.py

"""
Shared network factory for the Hebbian Plasticity / Manifold Sculptor project.

All parameter values in DEFAULT_PARAMS are in SI units:
seconds, volts, amps, ohms. Do not pass Brian2 Quantities in params — the
factory function converts them internally so params remain JSON-serialisable.
"""

import numpy as np
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, Network, PoissonInput,
    start_scope,
    seed as brian2_seed,
    second, volt, amp, ohm, Hz,
    prefs,
)

# Use the numpy backend to avoid C compilation overhead across repeated calls
# (e.g., during the grid search in tune_part1.py).
prefs.codegen.target = 'numpy'

DEFAULT_PARAMS = {
    # Neuron — SI units throughout
    'tau_m':    20e-3,    # s    membrane time constant
    'V_rest':  -70e-3,    # V    resting potential
    'V_th':    -55e-3,    # V    spike threshold
    'V_reset': -75e-3,    # V    post-spike reset (mild hyperpolarisation)
    'tau_ref':   2e-3,    # s    absolute refractory period
    'R':        100e6,    # Ω    membrane resistance

    # Synapse
    'tau_syn_E':  5e-3,   # s    AMPA-like decay
    'tau_syn_I': 10e-3,   # s    GABA-A-like decay
    'w_mean_EE': 0.06e-9, # A    mean E→E and E→I weight (0.06 nA)
    'sigma_w':   0.5,     #      log-space std for all weight distributions
    'w_scale_II': 0.2,    #      I→I mean is 0.2× I→E; weaker II preserves AI regime (Brunel 2000)

    # Network topology
    'N_exc':     800,
    'N_inh':     200,
    'p_connect': 0.1,

    # Operating point — override with tune_part1.py results
    'g_EI':   4 * 0.06e-9,  # A    mean I→E weight; starting point = 4× w_mean_EE
    'nu_ext': 10.0,           # Hz   background Poisson rate per neuron
}


def _lognormal_weights(w_mean: float, sigma: float, size: int, rng) -> np.ndarray:
    """
    Draw log-normal weights with E[W] = w_mean and log-space std = sigma.

    mu_log = log(w_mean) - sigma^2/2  →  E[W] = exp(mu_log + sigma^2/2) = w_mean.

    numpy's lognormal(mean, sigma) takes `mean` as the mean of the *underlying*
    normal distribution (mu_log), not the mean of the resulting log-normal.
    The -sigma^2/2 correction makes the log-normal mean equal w_mean.
    """
    mu_log = np.log(w_mean) - 0.5 * sigma ** 2
    return rng.lognormal(mu_log, sigma, size)


def build_network(params: dict, seed: int = 42) -> dict:
    """
    Build a balanced LIF recurrent network with current-based exponential synapses.

    Calls start_scope() to clear all previous Brian2 objects — safe to call
    repeatedly (e.g., in a parameter sweep). All previously returned objects
    become invalid on the next call.

    Parameters
    ----------
    params : dict
        Network parameters in SI units. Build from DEFAULT_PARAMS:
            build_network({**DEFAULT_PARAMS, 'g_EI': 0.30e-9})
    seed : int
        Seeds both Brian2's internal RNG and numpy's weight-init RNG.
        Use the same seed across all runs for reproducibility.

    Returns
    -------
    dict with keys: exc, inh, syn_EE, syn_EI, syn_IE, syn_II,
                    drive_E, drive_I, spike_E, spike_I, net
    """
    start_scope()
    brian2_seed(seed)
    rng = np.random.default_rng(seed)

    p = params

    # ------------------------------------------------------------------
    # Neuron equations
    # I_exc and I_inh are always >= 0; inhibition enters with a minus sign
    # in the membrane equation. This keeps weight signs positive and visible.
    # (unless refractory) means dv/dt is frozen during the refractory period;
    # I_exc and I_inh still decay normally — synaptic inputs are not blocked.
    # ------------------------------------------------------------------
    eqs = '''
    dv/dt     = (-(v - V_rest) + R * (I_exc - I_inh)) / tau_m : volt (unless refractory)
    dI_exc/dt = -I_exc / tau_syn_E : amp
    dI_inh/dt = -I_inh / tau_syn_I : amp
    '''

    ns = {
        'tau_m':     p['tau_m']     * second,
        'V_rest':    p['V_rest']    * volt,
        'V_th':      p['V_th']      * volt,
        'V_reset':   p['V_reset']   * volt,
        'tau_ref':   p['tau_ref']   * second,
        'R':         p['R']         * ohm,
        'tau_syn_E': p['tau_syn_E'] * second,
        'tau_syn_I': p['tau_syn_I'] * second,
    }

    exc = NeuronGroup(
        p['N_exc'], eqs,
        threshold='v > V_th',
        reset='v = V_reset',
        refractory='tau_ref',
        namespace=ns,
        method='euler',
        name='exc',
    )
    inh = NeuronGroup(
        p['N_inh'], eqs,
        threshold='v > V_th',
        reset='v = V_reset',
        refractory='tau_ref',
        namespace=ns,
        method='euler',
        name='inh',
    )

    # Initialise voltages uniformly in [V_reset, V_th] to avoid a long
    # transient where all neurons start at the same potential and fire synchronously.
    exc.v = 'V_reset + rand() * (V_th - V_reset)'
    inh.v = 'V_reset + rand() * (V_th - V_reset)'
    exc.I_exc = 0 * amp
    exc.I_inh = 0 * amp
    inh.I_exc = 0 * amp
    inh.I_inh = 0 * amp

    # ------------------------------------------------------------------
    # Synapses
    # E→target: increments I_exc. I→target: increments I_inh.
    # All weights positive; sign of inhibition is in the membrane equation.
    # ------------------------------------------------------------------
    syn_EE = Synapses(exc, exc, 'w : amp', on_pre='I_exc_post += w', name='syn_EE')
    syn_EI = Synapses(exc, inh, 'w : amp', on_pre='I_exc_post += w', name='syn_EI')
    syn_IE = Synapses(inh, exc, 'w : amp', on_pre='I_inh_post += w', name='syn_IE')
    syn_II = Synapses(inh, inh, 'w : amp', on_pre='I_inh_post += w', name='syn_II')

    p_c = p['p_connect']
    syn_EE.connect(condition='i != j', p=p_c)
    syn_EI.connect(p=p_c)
    syn_IE.connect(p=p_c)
    syn_II.connect(condition='i != j', p=p_c)

    # Log-normal weight init: E[w] = w_mean for each connection type.
    # Store raw float arrays; multiply by `amp` to attach Brian2 units.
    w_ee  = p['w_mean_EE']
    g_ei  = p['g_EI']
    sigma = p['sigma_w']

    syn_EE.w = _lognormal_weights(w_ee,                   sigma, len(syn_EE), rng) * amp
    syn_EI.w = _lognormal_weights(w_ee,                   sigma, len(syn_EI), rng) * amp
    syn_IE.w = _lognormal_weights(g_ei,                   sigma, len(syn_IE), rng) * amp
    syn_II.w = _lognormal_weights(p['w_scale_II'] * g_ei, sigma, len(syn_II), rng) * amp

    # Placeholders for drive/monitors — completed in Task 4
    return {
        'exc': exc, 'inh': inh,
        'syn_EE': syn_EE, 'syn_EI': syn_EI,
        'syn_IE': syn_IE, 'syn_II': syn_II,
        'drive_E': None, 'drive_I': None,
        'spike_E': None, 'spike_I': None,
        'net': None,
    }
