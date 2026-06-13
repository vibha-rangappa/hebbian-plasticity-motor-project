# Inhibitory STDP (Vogels 2011) — Design

**Date:** 2026-06-12
**Status:** approved, implementing
**Scope:** add inhibitory plasticity on I->E synapses to stabilize the network under E->E
STDP. The pilot showed weight normalization holds the mean E->E weight constant but does
NOT bound network gain — rate still climbed to ~8 Hz because STDP redistributes weight
into recurrent amplifying structure. Inhibition that tracks excitation is the principled
fix, and it is the prerequisite for balanced amplification (rotational dynamics).

## Motivation

Hebbian E->E plasticity is intrinsically destabilizing at the network-gain level; a
homeostatic counterpart is required (Zenke & Gerstner 2017). The canonical local,
biologically plausible mechanism is inhibitory STDP (Vogels, Sprekeler, Zenke, Clopath &
Gerstner 2011, Science): a symmetric Hebbian rule on inhibitory synapses that drives each
postsynaptic excitatory neuron toward a target firing rate. This stays squarely within
the project's thesis — local plasticity only — while controlling the runaway that
per-neuron weight normalization cannot.

This addresses two things at once:
- **Stability**: inhibition scales with excitation, holding E rate near target, removing
  the residual rate-climb confound on the geometry observables.
- **Rotation substrate**: balanced amplification — the mechanism for rotational dynamics —
  lives in the E/I interaction (Murphy & Miller 2009; Hennequin et al. 2014), which the
  current model freezes.

## The rule (Vogels et al. 2011)

Applied to `syn_IE` (inhibitory I -> excitatory E). Inhibitory weight `w >= 0` is a
magnitude; it enters the postsynaptic membrane as `I_inh_post += w` (subtracted as
`-I_inh` in `dv/dt`), matching the existing convention in `circuit/network.py`.

Two eligibility traces, both decaying with `tau_istdp` (~20 ms), event-driven:

    dapre_i/dt  = -apre_i  / tau_istdp     (inhibitory presynaptic trace)
    dapost_i/dt = -apost_i / tau_istdp     (excitatory postsynaptic trace)

On a presynaptic (inhibitory) spike:
    I_inh_post += w
    apre_i += 1
    w = clip(w + eta_istdp * (apost_i - alpha), 0, w_max_inh)

On a postsynaptic (excitatory) spike:
    apost_i += 1
    w = clip(w + eta_istdp * apre_i, 0, w_max_inh)

The depression constant `alpha = 2 * rho0 * tau_istdp` sets the target postsynaptic rate
`rho0`. Intuition: each inhibitory spike depresses by `eta*alpha` and potentiates by
`eta*apost_i`; the fixed point is where the postsynaptic trace averages to `alpha`, i.e.
the E neuron fires at `rho0`. Fire above target -> net potentiation -> more inhibition ->
rate pulled back down. This is the self-correcting homeostat.

`eta_istdp` is dimensionless-scaled to amp via a reference inhibitory weight (the rule
adds multiples of a small current step). Implementation note: `w`, `eta_istdp` carry amp
units; `apre_i`, `apost_i`, `alpha` are dimensionless.

## Parameters (DEFAULT_PARAMS_PLASTICITY additions)

| param | value | rationale |
|---|---|---|
| `tau_istdp` | 20 ms | matches the excitatory STDP window (Vogels default 20 ms) |
| `rho0` | 3.0 Hz | target E rate ~ the validated AI operating point |
| `eta_istdp` | 1e-12 A | inhibitory learning rate; tuned so rate stabilizes within ~100 trials without oscillation (re-pilot will confirm/adjust) |
| `w_max_inh` | 10 x g_EI | generous upper bound; inhibition must be free to grow well above baseline to counter potentiated excitation |

`alpha = 2 * rho0 * tau_istdp` is derived, not stored.

## Integration

- `build_stdp_network` gains `inhibitory_plasticity=False` (default off, preserving
  current behavior and all existing tests). When True, replace the static `syn_IE` from
  the baseline with a plastic Vogels-rule `Synapses`, preserving its connectivity
  (`i`, `j`) and initial weights, and add it to the Network in place of the static one.
- The inhibitory traces start at 0; `w` initialized from the baseline `syn_IE` weights.
- No interaction with the E->E `plastic` flag or weight normalization — iSTDP runs
  whenever `inhibitory_plasticity=True`, independent of E->E plasticity. (For the frozen
  control, both are off.)
- `train.py`: a `--inhibitory_plasticity {on,off}` CLI flag (default off), threaded into
  `build_stdp_network`; recorded in HDF5 provenance.

## Verification

- `PYTHONPATH=. python -m pytest tests/test_stdp_network.py -q`:
  - `build_stdp_network(inhibitory_plasticity=True)` preserves `syn_IE` connectivity and
    initial weights, and exposes plastic `apre_i`/`apost_i`/`w`.
  - With iSTDP on and a high-rate driven E population, inhibitory weights INCREASE
    (homeostatic potentiation when E fires above target); with E held below target they do
    not run away.
  - Default (`inhibitory_plasticity=False`) leaves `syn_IE` static (a regression guard).
- Re-pilot: `--condition seeded --weight_norm on --inhibitory_plasticity on` (sustained),
  104 trials. **Decision metric:** `mean_rate_E` stays near `rho0` (~3 Hz) across epochs
  instead of climbing to ~8 Hz. Compare to the same run without iSTDP.
- Then re-run the frozen/seeded/control geometry pre-check with iSTDP on and read PR /
  orthogonality with the rate-climb confound removed.

## What this does not do

iSTDP fixes stability; it does not by itself create rotation (that also needs the
autonomous regime and adequate sampling) or orthogonality (task-design / output-null).
Those remain separate threads. But a stable network is the precondition for any of them
to be cleanly measurable.
