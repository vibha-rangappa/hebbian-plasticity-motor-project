# Hebbian Plasticity as a Manifold Sculptor

## Overview

Primary motor cortex (M1) population activity during reaching has a few
well documented geometric properties. The effective dimensionality of the
population, measured by the participation ratio (PR), tends to be low for
the size of the recorded population, typically in the range of 5 to 20
(Gallego et al. 2017, using the eigenvalue-based PR measure described in Gao
et al. 2017). During movement, the population trajectory often shows
rotational structure in a low-dimensional subspace, identified by jPCA
(Churchland et al. 2012), with reported R^2 values around 0.87 in their M1
recordings. And the activity associated with planning a movement
(preparatory activity) and with executing it occupies nearly orthogonal
subspaces (Elsayed et al. 2016), consistent with the idea that the motor
system can hold a plan internally without that plan leaking into the output
pathway and causing premature movement (the output-null hypothesis, Kaufman
et al. 2014).

All three properties have mostly been demonstrated in two settings: real
recordings from behaving animals, where connectivity is the product of
development, evolution, and a lifetime of experience-dependent plasticity;
and recurrent neural networks trained by gradient descent on a task, where
connectivity is the product of an explicit, global optimization process with
access to a behavioral error signal.

This project asks what happens in between. We build a generic, biologically
plausible spiking network, balanced and tuned into a realistic firing
regime, and let its recurrent connectivity change only through local,
biologically grounded plasticity rules that have no access to a global error
signal, a behavioral objective, or anything beyond the spike timing and
firing rates each synapse can observe locally. Then we ask whether any of
the three M1-like geometric properties appear anyway.

"Sufficient" is the operative word, not "necessary." The claim under test is
not that the brain produces this geometry through exactly these mechanisms,
but whether a purely local rule, with no access to population-level
information, is enough by itself to produce population-level structure that
is normally explained by appeal to task optimization.

## The conceptual framework

### Three observables

1. **Participation ratio (PR)**, PR = (sum of eigenvalues)^2 / (sum of
   eigenvalues squared), computed from the eigenspectrum of the population
   covariance matrix. PR = 1 means all variance lives along one direction,
   PR = N means it is spread evenly across all N directions. This is the
   measure used in Gao et al. 2017 and reported for M1 in Gallego et al.
   2017.
2. **jPCA rotational structure (R^2)**, the fraction of the population
   trajectory's velocity that is explained by the best skew-symmetric linear
   dynamical system fit to the trajectory in PC space (Churchland et al.
   2012). A value near 1 means the dynamics are close to a pure rotation; a
   value near 0 means rotation explains nothing beyond a generic linear fit.
3. **Preparatory/execution subspace orthogonality**, the mean principal
   angle between the top PCA subspaces of the preparatory-epoch activity and
   the execution-epoch activity (Elsayed et al. 2016). Near 90 degrees means
   the two epochs occupy nearly separate subspaces, the geometric signature
   behind the output-null hypothesis (Kaufman et al. 2014).

### Three questions

- **Q1 (emergence)**: do the three observables move toward M1-like values
  over the course of training, beyond what happens under controls?
- **Q2 (mechanism map)**: which plasticity parameters produce which changes
  in geometry? This is meant to be the durable, reusable contribution: a map
  from plasticity-rule-space to geometry-space.
- **Q3 (mediation)**: if a linear decoder reads out reach direction from
  population activity, is decoding accuracy better predicted by the geometry
  triplet (PR, jPCA R^2, orthogonality) than by the raw amount of weight
  change? A positive answer would suggest local plasticity affects behavior
  mainly through the global geometry it creates, not simply through "more
  weight change is better."

### Three controls

- **Control A, frozen weights**: the same structural starting point as the
  trained network, but with all plasticity turned off. This isolates how
  much of any observed geometry change is due to structure alone versus
  learning.
- **Control B, spike-shuffled plasticity**: a training run where spike
  timing is scrambled before being used to drive plasticity, so the same
  overall amount of weight change happens but its relationship to actual
  spike timing is destroyed. This is a simulation-level control, distinct
  from the analysis-level condition-shuffle used as the jPCA chance floor
  (see Part 3).
- **Control C, symmetric STDP**: removes the causal pre-before-post
  asymmetry from the E->E STDP rule, testing whether the timing-dependent
  (as opposed to merely activity-dependent) part of the rule matters.

## Part 1: Network scaffold (`circuit/`)

### Architecture

The base network is a balanced recurrent network of 1000 leaky
integrate-and-fire (LIF) neurons in Brian2, 800 excitatory (E) and 200
inhibitory (I), connected with current-based exponential synapses
(`circuit/network.py`). Each neuron follows

    dv/dt = (-(v - V_rest) + R * (I_exc - I_inh)) / tau_m

with tau_m = 20 ms, V_rest = -70 mV, V_th = -55 mV, V_reset = -75 mV, and a
2 ms refractory period. I_exc and I_inh each decay exponentially (tau_syn_E
= 5 ms for AMPA-like excitatory input, tau_syn_I = 10 ms for GABA-A-like
inhibitory input) and are always non-negative; the minus sign in front of
I_inh in the voltage equation is what makes inhibitory input inhibitory.
Connectivity is Erdos-Renyi with p_connect = 0.1 (about 80 inputs per
neuron, excluding self-connections), and synaptic weights are drawn from a
log-normal distribution with mean w_mean_EE = 0.06 nA for E-to-E and E-to-I
connections and spread sigma_w = 0.5 in log space.

External drive is a per-neuron independent Poisson input representing
background cortical input from outside the modeled population, scaled
following Brunel (2000): rate = N_ext * nu_ext, where N_ext = N_exc *
p_connect = 80 approximates the number of recurrent E inputs a neuron would
otherwise receive, and the weight matches w_mean_EE.

### Tuning to the asynchronous-irregular (AI) regime

Three parameters were tuned by hand so the network sits in the
asynchronous-irregular (AI) firing regime that characterizes cortex at rest
(Brunel 2000): low single-digit-Hz firing rates, irregular spike timing
(CV-ISI close to 1), and weak pairwise correlations.

- **nu_ext = 7.0 Hz**, the background input rate per neuron, chosen above
  the 6.25 Hz threshold rate at which background drive alone would push a
  neuron toward threshold. This value also gives a slightly wider range of
  workable g_EI values than the originally tried nu_ext = 6.25 Hz.
- **g_EI = 0.090 nA** (1.5x w_mean_EE), the mean I-to-E inhibitory weight.
  This was raised from an earlier value of 0.075 nA after a dedicated
  headroom analysis (see below) showed the lower value left almost no room
  for E-to-E weights to grow under STDP without leaving the AI regime.
- **w_scale_II = 0.50**, the I-to-I mean weight as a fraction of g_EI. This
  turned out to be the most sensitive of the three: at w_scale_II = 0.20,
  I-to-I inhibition is too weak, the I population fires at roughly 3x the E
  rate, and the resulting strong inhibition drives the E rate down to about
  1 Hz at steady state. At w_scale_II = 1.00, I-to-I and I-to-E are equal,
  the I population effectively cancels itself out, fires at the same rate as
  E, and the E rate runs away (observed around 327 Hz). At w_scale_II =
  0.50, the I population settles at roughly 4 to 5 times the E rate and the
  E rate has a stable fixed point near 2.4 Hz.

### Why 2x STDP headroom is not architecturally available

A dedicated scan asked how much the mean E-to-E weight (w_EE) can grow
before the network leaves the AI regime, since this matters for how much
"room" E-to-E STDP has to work with in Part 2. At the operating point above,
the *soft* boundary (where CV-ISI drops below 0.8, corresponding to an
effective inhibitory strength of about 0.070 nA) allows about 29% growth in
w_EE before being crossed. The *hard* boundary (oscillatory runaway, around
an effective inhibitory strength of 0.055 nA) allows about 64% growth. Both
transitions are gradual at nu_ext = 7.0 Hz, unlike the catastrophic
oscillatory jump seen at nu_ext = 6.25 Hz. Reaching a full 2x headroom would
require g_EI = 0.140 nA, which suppresses the E rate below 1 Hz, so it is
not available simply by re-tuning g_EI; the small fan-in (80 inputs per
neuron) and the diffusion-regime dynamics of this network architecturally
limit how far w_EE can move. This is the reason weight normalization
(Part 2) is treated as the primary safeguard against the network drifting
out of the AI regime during STDP, not as an optional extra.

### Validation

The tuned network was validated with 30-second runs, checking the final
10-second window, across four random seeds (42, 0, 1, 7). All four seeds
pass all of the following criteria:

- Mean E firing rate = 2.39 Hz (target 2 to 10 Hz)
- CV-ISI = 0.832 (target 0.8 to 1.2)
- Pairwise correlation = 0.0004 (target below 0.05)
- I/E firing-rate ratio = 4.71 (target widened to 2 to 6x, since at nu_ext =
  7.0 Hz the I population receives more background drive than at the
  originally planned nu_ext = 6.25 Hz, pushing this ratio up; CV-ISI and
  pairwise correlation are the primary AI indicators, with the I/E ratio
  serving as a secondary sanity check)

Roughly 15 to 20 seconds of simulated time are needed to reach this steady
state from the randomized initial membrane voltages, which is why validation
checks a late window rather than the full run. The validated weight matrices
and connectivity are saved to an HDF5 baseline file and reloaded
deterministically (same seed, same parameters) by the plasticity code in
Part 2.

## Part 2: Task and plasticity (`plasticity/`)

### The center-out task

The task follows the 8-direction center-out reaching paradigm familiar from
the M1 literature (Churchland et al. 2012). Fifty Poisson "task-input"
neurons are added to the network, each connected to both the E and I
populations at the same connection density as the recurrent network
(p_connect = 0.1). Each input neuron is assigned a preferred direction
theta_i, evenly spread around the circle. During the preparatory and
execution phases of a trial, an input neuron's firing rate follows a cosine
tuning curve, clipped at zero (Georgopoulos et al. 1982):

    r_i = r_max * max(0, cos(theta_cue - theta_i))

with r_max = 100 Hz. A trial consists of three phases run back to back:
preparation (0.5 s), execution (0.5 s), and an inter-trial interval (0.2 s),
for a total of 1.2 s. During the inter-trial interval every input neuron
fires at a flat background rate, r_background = 2 Hz.

What happens during the execution phase depends on the **execution mode**:

- **Sustained mode**: the cosine tuning curve continues to drive the input
  population during execution, scaled up by exec_amplification = 1.5. The
  task input actively drives the network throughout the trial.
- **Autonomous mode**: the task-specific input drops back to the background
  rate during execution. The preparatory phase has already pushed the
  network into a direction-dependent state, and the recurrent network is then
  left to evolve on its own from that starting point (general background
  drive, nu_ext, stays on throughout, so the network does not go silent;
  only the direction-tuned component is removed). This is the mode in which
  movement-related dynamics such as rotational structure (Churchland et al.
  2012) or transient amplification (Hennequin et al. 2014) could plausibly
  appear, since a network being actively driven throughout the trial cannot
  show this kind of free evolution.

### Subpopulations and conditions

The 800 E neurons are split by index into two pools, P (neurons 0 to 349)
and X (neurons 350 to 699), with the remaining 100 neurons (S) left
unassigned to either pool. At network-build time, E-to-E synapses that cross
between P and X are scaled by a factor p_cross; synapses within a pool, or
touching S, are left unchanged. Three conditions are available from the
command line:

- **seeded**: p_cross = 0.2, so P-X connections start weaker than within-pool
  connections, and both kinds of plasticity (E-to-E STDP, inhibitory STDP)
  are active.
- **control**: p_cross = 1.0, no initial P/X asymmetry, plasticity active.
- **frozen** (Control A): the same p_cross = 0.2 structure as seeded, but
  all plasticity is turned off, so the network is the matched structural
  baseline with no learning.

### E-to-E STDP

E-to-E synapses use the classic pair-based STDP rule (Song, Miller & Abbott
2000). Each synapse keeps two exponentially decaying traces, apre and apost,
that record how recently the presynaptic and postsynaptic neurons fired
(both with time constants tau_plus = tau_minus = 20 ms). When the
presynaptic neuron fires, the synapse delivers its current weight w to the
postsynaptic neuron's excitatory input and is depressed by A_minus times the
current apost trace (a large apost means the postsynaptic neuron fired
recently, before this presynaptic spike, i.e. the acausal order). When the
postsynaptic neuron fires, the synapse is potentiated by A_plus times the
current apre trace (a large apre means the presynaptic neuron fired
recently, before this postsynaptic spike, i.e. the causal order). Weights
are clipped to [0, w_max] after every update.

Default amplitudes are A_plus = 0.0024 nA and A_minus = 0.00252 nA, so
depression is about 5% stronger than potentiation, the Song, Miller & Abbott
(2000) condition for keeping an *unnormalized* network's weights from growing
without bound. w_max = 0.24 nA, four times the mean E-to-E weight. In this
project, weight normalization (below) provides additional protection, which
is what makes it possible to deliberately move away from this near-balanced
default when probing the role of LTP/LTD asymmetry (see "E-to-E STDP
asymmetry probe" below).

### Inhibitory STDP (iSTDP)

When enabled, I-to-E synapses follow the inhibitory STDP rule of Vogels et
al. (2011), implemented as a "symmetric Hebbian" rule that homeostatically
pulls each excitatory neuron's firing rate toward a target rate rho0. Each
synapse keeps two traces (apre_i, apost_i, decaying with time constant
tau_istdp = 20 ms). On a presynaptic (inhibitory) spike, the weight updates
by eta_istdp * (apost_i - alpha), where alpha = 2 * rho0 * tau_istdp is a
constant set by the target rate; on a postsynaptic (excitatory) spike, the
weight updates by eta_istdp * apre_i. Weights are clipped to [0, w_max_inh =
0.90 nA, ten times the baseline I-to-E weight g_EI]. eta_istdp = 1e-12 A sets
the size of each weight step, and rho0 sets the operating point the rule
pushes excitatory rates toward; rho0 = 3.0 Hz matches the network's
validated baseline E rate (about 2.4 Hz), but rho0 was also swept across
1.5, 3, 5, and 8 Hz in Part 2's parameter sweep (see Q2 below).

### Weight normalization

To prevent E-to-E STDP from slowly inflating the mean weight and pushing the
network out of the AI regime (the limited-headroom problem identified in
Part 1), incoming E-to-E weights for each excitatory neuron are
multiplicatively rescaled, after each training step, so their sum matches a
fixed per-neuron target computed once at the start of training (after pool
rescaling). This is Turrigiano-style synaptic scaling: every incoming weight
to a neuron is multiplied by the same factor, so the *pattern* of relative
weight sizes that STDP has produced is preserved, but the *total* incoming
weight stays fixed. STDP can redistribute weight among a neuron's synapses
but cannot inflate the total, which keeps the network at its balanced
operating point over long training runs. Weights are clipped to [0, w_max]
after rescaling.

### Training protocol

Each training run begins by reloading the validated baseline network from
Part 1 (deterministic given the same seed and parameters), applying pool
rescaling for the chosen condition, and adding the 50 task-input neurons.
The network is then run for a 15-second burn-in period with plasticity
active (so the network can settle with its new connections before any
training trials are recorded), followed by a sequence of training trials
drawn from a shuffled, reproducible 8-direction trial order. Periodically (at
specified "snapshot epochs," counted in trials), the network's weights,
spike trains over a fixed set of test trials (5 per direction, in a fixed
order independent of the training sequence), and summary monitoring metrics
(mean E rate, mean E-to-E weight, fraction of weights at w_max, mean CV-ISI)
are saved to HDF5. Two abort criteria stop a run early if something has gone
wrong: mean E rate above 30 Hz (runaway potentiation) or more than half of
E-to-E weights pinned at w_max (insufficient depression).

## Part 3: Geometry analysis pipeline (`geometry/`)

All three observables are computed from the same preprocessed data matrix,
built once per snapshot, so that PR, jPCA, and orthogonality are never
looking at subtly different versions of the data.

### Preprocessing

Following Churchland et al. (2012): spikes from the 800 excitatory neurons
(the 50 task-input neurons are excluded, since their built-in cosine tuning
would inject exactly the kind of direction-related structure the analysis is
trying to test for) are binned at 1 ms, smoothed with a Gaussian kernel
(sigma = 25 ms, truncated at 3 sigma), converted to a rate in Hz, and
downsampled to 10 ms bins. The preparatory window (0 to 500 ms) and execution
window (500 to 1000 ms) of each trial are extracted and averaged across the 5
trials per direction to give 8 condition-averaged PSTHs per neuron per
window. Each neuron's response is then soft-normalized,

    r_norm(t, c) = r(t, c) / (R_i + 5 Hz)

where R_i is that neuron's firing-rate range computed once across the *full*
task period (prep and exec, all conditions), so prep and exec windows remain
on a common per-neuron scale. The 5 Hz floor avoids dividing by near-zero for
weakly driven neurons and prevents a few high-rate neurons from dominating
the subsequent PCA. Finally, the across-condition mean at each timepoint is
subtracted, removing the part of each neuron's response that is the same
regardless of reach direction and leaving only the condition-dependent
(direction-related) variance, which is what jPCA and orthogonality operate
on. Skipping this last step is a known way to artificially inflate apparent
rotational structure.

### Participation ratio

PR is computed from the eigenspectrum of the (T*C) x (T*C) "gram matrix" of
the data, rather than the 800 x 800 neuron covariance matrix. Each window
(prep or exec) has T = 50 timepoints (500 ms at 10 ms resolution) and C = 8
conditions, so M = T*C = 400 samples, far fewer than the 800 neurons, which
would make the neuron-by-neuron covariance matrix rank-deficient. The gram
matrix shares the same nonzero eigenvalues, and the
1/(M-1) normalization that would normally be applied cancels out of the PR
ratio. Because the number of samples (T*C) is much smaller than the number of
neurons (800), and mean-subtraction removes one degree of freedom, the
absolute PR value is biased low; comparisons across training epochs or
conditions are the meaningful quantity, not the absolute number.

### jPCA and the Lebedev safeguards

jPCA looks for the low-dimensional subspace in which the population
trajectory is best explained by a skew-symmetric (pure rotation, no
stretching) linear dynamical system: dX/dt approx X @ M^T with M = -M^T
(Churchland et al. 2012). The pipeline projects the centered data onto its
top 6 PCs (with 4 and 10 PCs checked for sensitivity), computes the
trajectory's velocity by central difference, and fits M_skew in closed form
via least squares over its k(k-1)/2 free parameters (15 for k = 6). The
reported R^2 is

    R2_skew = 1 - ||dX - X @ M_skew^T||^2 / ||dX - mean(dX)||^2

the fraction of velocity variance explained by the rotation, following
Churchland et al. (2012)'s convention. (Note: this differs from a literal
reading of the original analysis brief, which proposed normalizing by the
residual of the *unconstrained* linear fit instead. That version cannot work,
since the rotation-only fit is a restricted special case of the unconstrained
fit, its residual can only be larger, making that ratio always >= 1 and the
resulting "R^2" always <= 0. Instead, the gap between the unconstrained fit's
R^2 and R2_skew is reported separately, as a diagnostic of how much is given
up by forcing the dynamics to be a pure rotation.)

Because Lebedev et al. (2019) showed that jPCA can produce convincing-looking
rotations from any population with a consistent temporal sequence of peak
firing rates across conditions, even purely feedforward, non-dynamical
activity, three additional checks are required before trusting a jPCA result:

1. **Rotation-direction consistency**: all 8 reach-direction trajectories
   should rotate the same way (clockwise or counterclockwise) in the
   dominant jPC plane.
2. **Trajectory tangling** (Russo et al. 2018): Q(t) = max over t' of
   ||dX(t) - dX(t')||^2 / (||X(t) - X(t')||^2 + epsilon). High tangling means
   nearby population states have very different velocities, inconsistent
   with a smooth autonomous dynamical system.
3. **Spike-shuffled chance floor**: trial-to-condition labels are randomly
   permuted (`condition_shuffle`, an analysis-time shuffle, not to be
   confused with the spike-shuffled-plasticity simulation control, Control
   B), and jPCA R^2 is recomputed on the shuffled data. A real result should
   sit clearly above this floor; the metric reported throughout this project
   as "above-shuffle R^2" is the observed R2_skew minus this chance floor.

### Preparatory/execution subspace orthogonality

The top 6 PCs of the prep-window data and the top 6 PCs of the exec-window
data (using the same R_i normalization, computed from `top_pc_basis`) define
two 6-dimensional subspaces of the 800-neuron activity space. The principal
angles between them are computed with `scipy.linalg.subspace_angles`
(Elsayed et al. 2016). Because two random 6-dimensional subspaces of an
800-dimensional space are already close to 90 degrees apart by chance, the
observed mean angle is always compared to a bootstrap null distribution built
from 1000 pairs of random subspaces of the same dimensionality, reported both
as a mean and standard deviation and as a z-score of the observed angle
relative to that null.

### Synthetic validation

Before trusting any of this on real network output, the pipeline is tested
on synthetic data with known properties (`geometry/synthetic.py`): a pure
rotation, where jPCA should recover R^2 close to 1, and a feedforward
sequence with no real dynamics, the Lebedev artifact, which the safeguards
above must catch rather than mistakenly report as a rotation.

## The Q2 sweep: a plasticity-to-geometry map

A 96-point sweep was run over the inhibitory plasticity parameters: rho0 in
{1.5, 3, 5, 8} Hz, eta_istdp at 4 values, tau_istdp at 3 values, crossed with
E-to-E STDP on or off (4 x 4 x 3 x 2 = 96), in autonomous execution mode with
weight normalization on, at pilot scale (about 100 training trials, 13 per
direction, with snapshots at 0, 50, and 100 trials).

**Headline finding: rho0, the iSTDP firing-rate setpoint, is the dominant
lever for both PR (in the execution window) and prep/exec orthogonality, not
the iSTDP learning rate (eta_istdp) or time constant (tau_istdp).** The mean
change in PR_exec from trial 0 to trial 100, by rho0, was +0.6, +0.4, +5.5,
and +16.9 for rho0 = 1.5, 3, 5, and 8 Hz respectively; the corresponding mean
change in prep/exec orthogonality was +0.5, -0.1, +1.6, and +7.3 degrees.

- At low rho0 (1.5 to 3 Hz), the network starts near a ceiling: PR_exec
  around 52 to 54 and orthogonality around 80 to 85 degrees, close to the
  random-subspace null of about 85.8 degrees. A hundred trials of training
  barely move either quantity.
- At rho0 = 8 Hz, the network starts much further from that ceiling, with
  PR_exec around 13 to 30 and orthogonality around 37 to 65 degrees, and
  training closes a large fraction of the gap; some rho0 = 8 configurations
  with E-to-E STDP on nearly triple PR_exec over 100 trials.
- Within the high-rho0 regime, eta_istdp and tau_istdp behave as expected
  secondary modulators: larger eta_istdp and longer tau_istdp both push
  orthogonality further toward 90 degrees.
- E-to-E STDP is not a clean null for orthogonality the way it appeared to be
  for PR in an earlier, smaller pilot. With E-to-E STDP on, the mean change
  in orthogonality is about +3.2 degrees, versus about +1.4 degrees with
  E-to-E STDP off, roughly doubling the effect of inhibitory STDP alone. This
  suggests E-to-E STDP and iSTDP interact rather than acting on independent
  axes of the geometry.
- **jPCA stayed null across the entire sweep.** At trial 100, the
  above-shuffle R^2 had mean -0.022 and standard deviation 0.026 across all
  96 points, with a range of -0.092 to +0.044, only 23 of 96 points positive,
  and no apparent dependence on rho0, eta_istdp, tau_istdp, or whether E-to-E
  STDP was on.

A separate run extended training to 800 trials at rho0 = 3.0 Hz, eta_istdp =
1e-12 A, tau_istdp = 20 ms (the configuration whose iSTDP target rate matches
the network's validated baseline firing rate, with snapshots at 0, 50, 100,
200, 400, and 800 trials). Consistent with rho0 = 3 falling in the
near-ceiling, low-rho0 regime identified above, PR_exec and orthogonality
both stayed close to their starting values across the full 800 trials
(PR_exec from 52.8 to 53.5, orthogonality from 80.8 to 82.9 degrees). The
above-shuffle jPCA R^2 fluctuated between about -0.05 and +0.07 across
epochs with no monotonic trend, an 8x increase in training did not produce a
clear rotational signal at this operating point.

Together, these results reframe what the "mechanism map" in Q2 actually
shows: the durable finding is not a simple "STDP parameters predict geometry
change" relationship, but that rho0, the inhibitory rate setpoint, gates
*whether* STDP has any visible effect on PR and orthogonality at all. jPCA
remains an unresolved null observable across every configuration tried so
far.

## E-to-E STDP asymmetry probe

The working hypothesis for how rotational structure (jPCA) might emerge is
that asymmetric STDP builds chain-like connectivity motifs into the recurrent
weight matrix, in the spirit of the motif-expansion framework of Ocker,
Litwin-Kumar & Doiron (2015), and that such motifs are the substrate for
rotational dynamics. This depends on the *asymmetry* of the E-to-E STDP rule:
how much more strongly the causal (pre-before-post) ordering potentiates a
synapse than the acausal ordering depresses it. Every jPCA run so far,
across all 4 combinations of {sustained, autonomous} execution mode and
{E-to-E STDP alone, E-to-E plus iSTDP}, used the default amplitudes (A_plus =
2.4e-12 A, A_minus = 2.52e-12 A), an LTP/LTD ratio of about 0.952, which is
nearly balanced and so has never actually tested this asymmetry axis.

`plasticity/ee_asymmetry_probe.py` sweeps the A_plus/A_minus ratio over {0.5,
1.0, 2.0, 4.0} while holding tau_plus = tau_minus = 20 ms and the total
plasticity "budget" A_plus + A_minus fixed at the default sum, so the
LTP/LTD balance changes without changing the overall learning-rate
magnitude. Because weight normalization and iSTDP already guard against
runaway growth (the Song, Miller & Abbott 5%-depression-dominant condition
exists specifically to protect an *unnormalized* network), it is safe to push
this ratio well past 1, into potentiation-dominant territory. Each point runs
at rho0 = 3.0 Hz, eta_istdp = 1e-12 A, tau_istdp = 20 ms, autonomous execution
mode, iSTDP and weight normalization on, the same M1-like operating point as
the 800-trial run above (the default ratio of about 0.952 was already covered
by that run and is not repeated).

## Theoretical context

The central hypothesis connecting plasticity to geometry, that asymmetric
STDP sculpts rotational dynamics by strengthening sequential chain-like
connectivity motifs, is currently a plausibility argument rather than a
derived result. The motif-expansion theory of Ocker, Litwin-Kumar & Doiron
(2015) describes, for a given STDP rule and its parameters, which two-synapse
connectivity motifs get amplified or suppressed; chain motifs of the kind
that theory describes are a natural candidate substrate for the kind of
rotational dynamics jPCA looks for. Building the bridge from "what does this
STDP rule do to the distribution of two-synapse motifs in this network" to
"what does that predict for PR, jPCA R^2, and prep/exec orthogonality" would
turn the empirical plasticity-to-geometry map from Q2 into a derived
prediction, and is the main piece of theoretical scaffolding this project
does not yet have.

## References

The following work is referenced throughout the code, the analysis design,
and this document. Citations are reproduced from memory and project notes;
double check exact bibliographic details before using them in any formal
write-up.

- Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory
  and inhibitory spiking neurons. *Journal of Computational Neuroscience*.
- Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian
  learning through spike-timing-dependent synaptic plasticity. *Nature
  Neuroscience*.
- Vogels, T. P., Sprekeler, H., Zenke, F., Clopath, C., & Gerstner, W. (2011).
  Inhibitory plasticity balances excitation and inhibition in sensory
  pathways and memory networks. *Science*.
- Georgopoulos, A. P., Kalaska, J. F., Caminiti, R., & Massey, J. T. (1982).
  On the relations between the direction of two-dimensional arm movements and
  cell discharge in primate motor cortex. *Journal of Neuroscience*.
- Churchland, M. M., Cunningham, J. P., Kaufman, M. T., Foster, J. D.,
  Nuyujukian, P., Ryu, S. I., & Shenoy, K. V. (2012). Neural population
  dynamics during reaching. *Nature*.
- Gao, P., Trautmann, E., Yu, B., Santhanam, G., Ryu, S., Shenoy, K., &
  Ganguli, S. (2017). A theory of multineuronal dimensionality, dynamics and
  measurement.
- Gallego, J. A., Perich, M. G., Miller, L. E., & Solla, S. A. (2017). Neural
  manifolds for the control of movement. *Neuron*.
- Elsayed, G. F., Lara, A. H., Kaufman, M. T., Churchland, M. M., &
  Cunningham, J. P. (2016). Reorganization between preparatory and movement
  population responses in motor cortex. *Nature Communications*.
- Kaufman, M. T., Churchland, M. M., Ryu, S. I., & Shenoy, K. V. (2014).
  Cortical activity in the null space: permitting preparation without
  movement. *Nature Neuroscience*.
- Lebedev, M. A., Ossadtchi, A., Mill, N. A., Urpi, N. A., Rebesco, J. M.,
  Cervera, M. R., & Nicolelis, M. A. L. (2019). Analysis of neuronal ensemble
  activity reveals the pitfalls and shortcomings of rotation dynamics.
  *Scientific Reports*.
- Russo, A. A., Bittner, S. R., Perkins, S. M., Seely, J. S., London, B. M.,
  Lara, A. H., Miri, A., Marshall, N. J., Kohn, A., Jessell, T. M., Abbott, L.
  F., Cunningham, J. P., & Churchland, M. M. (2018). Motor cortex embeds
  muscle-like commands in an untangled population response. *Neuron*.
- Ocker, G. K., Litwin-Kumar, A., & Doiron, B. (2015). Self-organization of
  microcircuits in networks of spiking neurons with plastic synapses. *PLoS
  Computational Biology*.
- Hennequin, G., Vogels, T. P., & Gerstner, W. (2014). Optimal control of
  transient dynamics in balanced networks supports generation of complex
  movements. *Neuron*.

## Repo layout

- `circuit/` - the LIF network definition (`network.py`, including
  `DEFAULT_PARAMS`), the script that runs and saves the validated baseline
  network (`run_baseline.py`, producing `circuit/results/baseline_network.h5`),
  and the calibration scripts used to find and validate the AI-regime operating
  point (`grid_search.py`, `headroom_scan.py`, `wscale_test.py`,
  `wscale05_long.py`, `wscale_high_scan.py`, `fine_grid.py`, `fine_grid2.py`,
  `narrow_scan.py`, `steadystate_scan.py`, `transition_scan.py`,
  `robustness_scan.py`, `quick_ie_check.py`, `sustaintest.py`, `longrun.py`,
  `validate_candidate.py`, `brunel_params.py`).
- `plasticity/` - the center-out task (`center_out_task.py`), the plastic
  network (`stdp_network.py`, E-to-E STDP and inhibitory STDP), the training
  loop and CLI (`train.py`), post-training sanity checks
  (`validate_training.py`), snapshot saving/loading (`snapshot.py`), the
  parameter sweep (`sweep.py`), and the LTP/LTD asymmetry probe
  (`ee_asymmetry_probe.py`).
- `geometry/` - the analysis pipeline: shared preprocessing
  (`preprocessing.py`), the three observables (`dimensionality.py` for PR,
  `jpca.py`, `orthogonality.py`), shared null/control helpers
  (`controls.py`), a small robust-SVD wrapper used throughout
  (`_linalg.py`), the main entry point that ties everything together into one
  results table (`run_geometry.py`, writing `geometry/results/geometry_metrics*.csv`),
  the synthetic validation suite (`synthetic.py`), and the sweep map and
  figure scripts (`sweep_map.py`, `sweep_epoch_trends.py`, `visualize.py`).
- `tests/` - pytest tests for all of the above (network, STDP/iSTDP, center-out
  task, training, snapshot/HDF5 I/O, preprocessing, PR, jPCA, orthogonality,
  controls, and end-to-end analysis). Run with `PYTHONPATH=. python -m
  pytest`; `pytest.ini` restricts collection to this directory so pytest does
  not also try to collect `circuit/wscale_test.py`, a parameter-scan script
  that happens to match pytest's default test-file naming pattern.
- `docs/` - `circuit_tuning_notes.md`, the narrative writeup of the Part 1
  AI-regime tuning process.

Every `.py` file has a plain-English description at the top explaining what
it does and, where relevant, why it exists.
