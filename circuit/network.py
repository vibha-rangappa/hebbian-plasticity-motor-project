# circuit/network.py

"""
This file builds the spiking network model used everywhere else in the project.

It defines build_network(), which sets up a balanced network of excitatory
(E) and inhibitory (I) leaky integrate-and-fire neurons connected with
current-based exponential synapses, plus a default set of parameters
(DEFAULT_PARAMS) that puts the network in a stable, biologically realistic
firing regime. Other scripts (calibration scans, the baseline runner, the
plasticity training code) all call build_network() with these parameters
(sometimes with a few values changed) to get a fresh copy of the network.

All parameter values in DEFAULT_PARAMS are plain numbers in SI units:
seconds, volts, amps, ohms. Don't pass Brian2 Quantities (numbers with units
attached) into params. build_network() attaches the units internally, which
keeps params as plain numbers that can be saved to JSON.
"""

import numpy as np
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, Network, PoissonInput,
    start_scope,
    seed as brian2_seed,
    second, volt, amp, ohm, Hz,
    prefs,
)

# Use the numpy backend instead of compiling C code each time. This is slower
# per run but avoids the compile step, which adds up when we call build_network()
# many times in a row (e.g., during the grid search in grid_search.py).
prefs.codegen.target = 'numpy'

DEFAULT_PARAMS = {
    # Neuron settings, all in SI units
    'tau_m':    20e-3,    # s    how fast the membrane voltage decays back to rest
    'V_rest':  -70e-3,    # V    resting voltage (no input)
    'V_th':    -55e-3,    # V    voltage at which the neuron fires a spike
    'V_reset': -75e-3,    # V    voltage right after a spike (slightly below rest)
    'tau_ref':   2e-3,    # s    refractory period (can't fire again right after a spike)
    'R':        100e6,    # ohm  membrane resistance (how much voltage a given current produces)

    # Synapse settings
    'tau_syn_E':  5e-3,   # s    how fast excitatory input current decays (AMPA-like)
    'tau_syn_I': 10e-3,   # s    how fast inhibitory input current decays (GABA-A-like)
    'w_mean_EE': 0.06e-9, # A    average weight for E-to-E and E-to-I connections (0.06 nA)
    'sigma_w':   0.5,     #      spread (std dev in log space) of all weight distributions

    # Network size and connectivity
    'N_exc':     800,
    'N_inh':     200,
    'p_connect': 0.1,

    # Operating point: these three values were tuned by hand so the network
    # sits in a stable "asynchronous irregular" (AI) regime, which is the
    # realistic firing pattern we want (similar to real cortex).
    # How we tuned it: ran 30-second simulations with several random seeds,
    # scanning over (nu_ext, g_EI, w_scale_II), and looked for:
    #   firing rate 2-10 Hz, CV-ISI (irregularity measure) 0.8-1.2,
    #   and pairwise correlation < 0.05, all measured in the 20-30 s window.
    # nu_ext = 7.0 Hz: this is above the 6.25 Hz threshold rate, and it gives
    #   a slightly bigger range of g_EI values that still work
    #   (down to g_EI of about 0.070, versus about 0.075 at nu_ext = 6.25).
    # g_EI = 0.090 nA (1.5x w_mean_EE): chosen to leave room for the E-to-E
    #   weights to grow during STDP learning without breaking the AI regime
    #   (see the note below).
    # w_scale_II = 0.50: the I-to-I weight is set to half of I-to-E. This was
    #   needed for the network to settle into a stable firing rate.
    #   At 0.2, the E firing rate decays away to under 1 Hz. At 1.0, the E
    #   rate runs away because the inhibitory neurons end up cancelling
    #   each other out instead of inhibiting the E population.
    # STDP headroom: with g_EI = 0.090, the E-to-E weight (w_EE) can grow
    #   about 29% before the irregularity measure (CV) drops below 0.8
    #   (that's the "soft" boundary, at an effective g of 0.070 nA). The
    #   network only becomes oscillatory (a much worse "hard" boundary)
    #   after about 55% growth. We can't realistically get 2x headroom in
    #   this network just from picking g_EI (because of how it's wired,
    #   with 80 inputs per neuron and the way input currents add up).
    #   So the main thing protecting the network from instability during
    #   STDP training is the weight normalization applied while training.
    'g_EI':     0.090e-9,   # A    average I-to-E inhibitory weight (1.5x w_mean_EE)
    'nu_ext':   7.0,        # Hz   background input rate per neuron (above threshold)
    'w_scale_II': 0.50,     #      I-to-I average weight = 0.50x g_EI
}


def _lognormal_weights(w_mean: float, sigma: float, size: int, rng) -> np.ndarray:
    """
    Draw synaptic weights from a log-normal distribution whose average value
    is exactly w_mean, with sigma controlling the spread (in log space).

    The formula mu_log = log(w_mean) - sigma^2/2 is a correction factor we
    need because numpy's lognormal(mean, sigma) treats "mean" as the mean of
    the underlying normal distribution, not the mean of the resulting
    log-normal values. Without the -sigma^2/2 correction, the average of the
    drawn weights would come out higher than w_mean.
    """
    mu_log = np.log(w_mean) - 0.5 * sigma ** 2
    return rng.lognormal(mu_log, sigma, size)


def build_network(params: dict, seed: int = 42) -> dict:
    """
    Build a balanced recurrent network of leaky integrate-and-fire (LIF)
    neurons, connected with current-based exponential synapses.

    This calls start_scope() first, which wipes out any Brian2 objects from
    a previous call. That makes it safe to call build_network() over and
    over (e.g., once per point in a parameter sweep), but it also means any
    network objects returned from an earlier call stop working once you
    call this again.

    Parameters
    ----------
    params : dict
        Network parameters in SI units. Build from DEFAULT_PARAMS, e.g.:
            build_network({**DEFAULT_PARAMS, 'g_EI': 0.30e-9})
    seed : int
        Seeds both Brian2's internal random number generator and the numpy
        generator used for the initial weights. Use the same seed across
        runs if you want reproducible results.

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
    # I_exc and I_inh are both always >= 0. The minus sign in front of I_inh
    # in the voltage equation is what makes it inhibitory. Keeping all
    # weights positive like this makes it easy to see at a glance whether a
    # weight is excitatory or inhibitory.
    # "(unless refractory)" means the voltage (v) stops updating during the
    # refractory period right after a spike. I_exc and I_inh keep decaying
    # normally during that time, incoming synaptic input is not blocked.
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

    # Start each neuron's voltage at a random value between V_reset and V_th.
    # If every neuron started at the same voltage, they'd all fire their
    # first spike at the same time, creating a long synchronized transient
    # at the start of the simulation. Randomizing the start avoids that.
    exc.v = 'V_reset + rand() * (V_th - V_reset)'
    inh.v = 'V_reset + rand() * (V_th - V_reset)'
    exc.I_exc = 0 * amp
    exc.I_inh = 0 * amp
    inh.I_exc = 0 * amp
    inh.I_inh = 0 * amp

    # ------------------------------------------------------------------
    # Synapses
    # Every synapse from an E neuron adds to the target's I_exc.
    # Every synapse from an I neuron adds to the target's I_inh.
    # All synaptic weights are stored as positive numbers; whether a
    # synapse is excitatory or inhibitory is determined by which current
    # (I_exc or I_inh) it adds to, and I_inh has the minus sign in the
    # voltage equation above.
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

    # Set the initial weights for each connection type by drawing from a
    # log-normal distribution whose average is w_mean for that type.
    # _lognormal_weights() returns plain numbers (floats); multiplying by
    # `amp` attaches Brian2's "amps" unit to them.
    w_ee  = p['w_mean_EE']
    g_ei  = p['g_EI']
    sigma = p['sigma_w']

    syn_EE.w = _lognormal_weights(w_ee,                   sigma, len(syn_EE), rng) * amp
    syn_EI.w = _lognormal_weights(w_ee,                   sigma, len(syn_EI), rng) * amp
    syn_IE.w = _lognormal_weights(g_ei,                   sigma, len(syn_IE), rng) * amp
    syn_II.w = _lognormal_weights(p['w_scale_II'] * g_ei, sigma, len(syn_II), rng) * amp

    # ------------------------------------------------------------------
    # External background input (Poisson drive)
    # Using N=1 gives each neuron its own independent random spike train.
    # If we used N>1, Brian2 would share the same spike train across all
    # target neurons, which would make their background input correlated.
    # That correlated input then drives correlated (synchronized)
    # inhibition, which breaks the asynchronous irregular (AI) firing
    # pattern we want.
    # Instead, we keep N=1 but scale up the rate by N_ext (roughly the
    # number of recurrent E inputs each neuron gets) so the average
    # background current still matches what you'd get with N_ext separate
    # inputs at rate nu_ext each, following Brunel's (2000) notation:
    #   average background current = N_ext * nu_ext * w_mean_EE * tau_syn_E
    # This input current is added to I_exc, so it decays with the
    # excitatory time constant tau_syn_E.
    # ------------------------------------------------------------------
    N_ext = int(p['N_exc'] * p['p_connect'])   # = 80 for the full-size network
    drive_E = PoissonInput(exc, 'I_exc', N=1,
                           rate=N_ext * p['nu_ext'] * Hz,
                           weight=p['w_mean_EE'] * amp)
    drive_I = PoissonInput(inh, 'I_exc', N=1,
                           rate=N_ext * p['nu_ext'] * Hz,
                           weight=p['w_mean_EE'] * amp)

    # ------------------------------------------------------------------
    # Record spikes from both populations, and bundle everything together
    # into a Brian2 Network object so it can be run as one unit.
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
