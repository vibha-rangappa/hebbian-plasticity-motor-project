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

    # Network topology
    'N_exc':     800,
    'N_inh':     200,
    'p_connect': 0.1,

    # Operating point — tuned empirically to produce stable AI regime.
    # Tuning procedure: 30-s multi-seed scan over (nu_ext, g_EI, w_scale_II).
    # Target: rate 2–10 Hz, CV-ISI 0.8–1.2, pairwise corr <0.05 in [20–30 s] window.
    # nu_ext = 7.0 Hz (above the 6.25 Hz threshold rate): gives a slightly wider
    #   AI corridor (lower boundary drops to g_EI≈0.070 vs ≈0.075 at nu=6.25).
    # g_EI = 0.090 nA (1.5× w_mean_EE): chosen for STDP headroom — see note below.
    # w_scale_II = 0.50: I→I half of I→E; empirically needed for stable fixed point
    #   (0.2 → E rate decays to <1 Hz; 1.0 → E runaway because I self-cancels).
    # STDP headroom: with g_EI=0.090, w_EE can grow ~29% before CV drops below 0.8
    #   (soft boundary at g_eff=0.070 nA). Hard oscillatory boundary is ~55% away.
    #   2× headroom is architecturally impossible in this network (80 inputs/neuron
    #   + diffusion regime); weight normalization in Part 2 is the primary protection.
    'g_EI':     0.090e-9,   # A    mean I→E inhibitory weight (1.5× w_mean_EE)
    'nu_ext':   7.0,        # Hz   background Poisson rate per neuron (> threshold)
    'w_scale_II': 0.50,     #      I→I mean = 0.50× g_EI
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

    # ------------------------------------------------------------------
    # External Poisson drive
    # N=1 gives each target neuron one INDEPENDENT Poisson process.
    # Brian2's N>1 shares spike trains across all target neurons, introducing
    # correlated background that triggers synchronised inhibition and kills AI.
    # We instead scale the rate by N_ext (≈ recurrent E fan-in) so the mean
    # background current matches C_ext×ν_ext in Brunel's parameterisation:
    #   E[I_bg] = N_ext × nu_ext × w_mean_EE × tau_syn_E
    # Drive goes to I_exc so it decays with tau_syn_E.
    # ------------------------------------------------------------------
    N_ext = int(p['N_exc'] * p['p_connect'])   # = 80 for full network
    drive_E = PoissonInput(exc, 'I_exc', N=1,
                           rate=N_ext * p['nu_ext'] * Hz,
                           weight=p['w_mean_EE'] * amp)
    drive_I = PoissonInput(inh, 'I_exc', N=1,
                           rate=N_ext * p['nu_ext'] * Hz,
                           weight=p['w_mean_EE'] * amp)

    # ------------------------------------------------------------------
    # Spike monitors and Network
    # ------------------------------------------------------------------
    spike_E = SpikeMonitor(exc)
    spike_I = SpikeMonitor(inh)

    net = Network(
        exc, inh,
        syn_EE, syn_EI, syn_IE, syn_II,
        drive_E, drive_I,
        spike_E, spike_I,
    )

    return {
        'exc':    exc,    'inh':    inh,
        'syn_EE': syn_EE, 'syn_EI': syn_EI,
        'syn_IE': syn_IE, 'syn_II': syn_II,
        'drive_E': drive_E, 'drive_I': drive_I,
        'spike_E': spike_E, 'spike_I': spike_I,
        'net':    net,
    }
