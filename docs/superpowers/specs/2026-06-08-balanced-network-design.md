# Part 1 Design: Network Scaffold and E/I Balance

**Project:** Hebbian Plasticity as a Manifold Sculptor
**Date:** 2026-06-08
**Status:** Approved

---

## Goal

Build a 1000-neuron LIF network in Brian2, tune it into the asynchronous-irregular (AI) firing
regime, validate it against six pass/fail checks, and save a reproducible baseline to HDF5. This
baseline is the starting point for Part 2 (STDP + task structure). Do not proceed to Part 2 until
all six checks pass.

---

## File Structure

```
part1/
  network.py          # shared: build_network(params, seed) → dict of Brian2 objects
  tune_part1.py       # 2D grid search → tuning_results.csv + tuning_heatmap.png
  run_part1.py        # 5 s validation run → baseline_network.h5 + figures/
  results/
    tuning_results.csv
    tuning_heatmap.png
    baseline_network.h5
    figures/
      raster.png
      firing_rate_hist.png
      isi_dist.png
      pairwise_corr.png
      power_spectrum.png
      weight_hists.png
```

---

## Neuron Model

Leaky integrate-and-fire (LIF). All parameters are explicit named variables, not hardcoded
constants, so they can be perturbed during debugging without hunting through equations.

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `tau_m`   | 20 ms | Standard pyramidal-cell membrane time constant (Koch 1999; Brette & Gerstner 2005) |
| `V_rest`  | -70 mV | Standard resting potential |
| `V_th`    | -55 mV | 15 mV above rest; realistic rheobase |
| `V_reset` | -75 mV | Mild post-spike hyperpolarization; contributes to CV-ISI near 1 without requiring adaptation |
| `tau_ref` | 2 ms | Absolute refractory period; caps firing rate at 500 Hz |
| `R`       | 100 MΩ | Membrane resistance; sets current-to-voltage gain |

**Membrane equation:**
```
dv/dt = (-(v - V_rest) + R * (I_exc - I_inh)) / tau_m : volt (unless refractory)
dI_exc/dt = -I_exc / tau_syn_E : amp
dI_inh/dt = -I_inh / tau_syn_I : amp
```

Sign convention: `I_exc` and `I_inh` are both always ≥ 0. Inhibition enters with a minus sign
in the membrane equation. All synaptic weights are strictly positive (log-normal distributed).
This keeps the sign of each current visible in the equation rather than hidden in weight values.

**Why LIF and not Izhikevich or AdEx?** The scientific question is about the plasticity rule and
population geometry, not single-neuron dynamics. LIF is the minimal model that produces spike
timing, which is all STDP needs. Every additional biophysical parameter is a potential confound.

---

## Network Topology

| Property | Value | Justification |
|----------|-------|---------------|
| N total | 1000 | Participation ratio requires N×N covariance matrix; 1000 balances spectrum fidelity vs. Brian2 runtime (Gao et al. 2017) |
| N_exc | 800 | 80/20 E/I is the canonical cortical ratio (Braitenberg & Schüz 1998) |
| N_inh | 200 | |
| p_connect | 0.1 | ~100 inputs per neuron (80 E, 20 I); consistent with local cortical circuits (Abeles 1991) |
| Topology | Erdős-Rényi random graph | Spatially uniform; no distance dependence in Part 1 |
| Self-connections | None | Explicitly excluded: `connect(condition='i != j', p=p_connect)` |

---

## Synaptic Model

Current-based exponential synapses. The current injected from synapse j into neuron i:

```
I_syn_i(t) = Σ_j w_ij * s_j(t)

tau_syn * ds_j/dt = -s_j
s_j += 1  on each presynaptic spike
```

Brian2 synapse model per connection type:
- E synapses: `on_pre: I_exc_post += w`
- I synapses: `on_pre: I_inh_post += w`
- Variable per synapse: `w : amp`

**Why current-based?** Conductance-based synapses add reversal-potential parameters per type.
For Part 1, goal is to get E/I balance right, and current-based synapses make that tuning
transparent: you can see directly how much current each population drives. Switch to
conductance-based if shunting inhibition becomes relevant to the science later.

### Synaptic time constants

| Connection | τ_syn | Rationale |
|-----------|-------|-----------|
| E (AMPA-like) | 5 ms | Standard AMPA decay |
| I (GABA-A-like) | 10 ms | Standard GABA-A decay |

### Initial weight distributions

All weights drawn from log-normal distributions. **Units: nA (current).** The weight `w` has
units of current because `I_syn = w * s` and `s` is dimensionless; the charge per spike is
`w * tau_syn` (e.g., 0.06 nA × 5 ms = 0.3 pC for E→E).

**Why log-normal?** Cortical synaptic weights follow approximately log-normal distributions
(Buzsáki & Mizuseki 2014, Nat Rev Neurosci). Log-normal is strictly positive (sign is carried by
neuron type, not weight value) and has the heavy tail matching rare-strong-synapse structure.

Log-normal parameterization: `np.random.lognormal(mu_log, sigma)` where
`mu_log = log(w_mean) - sigma²/2`. This gives `E[W] = exp(mu_log + sigma²/2) = w_mean`,
so the distribution *mean* equals `w_mean`. (numpy's `lognormal(mean, sigma)` takes `mean` as
the mean of the underlying normal, not the mean of the log-normal — hence the `- sigma²/2`
correction to target the desired log-normal mean.)

| Connection | w_mean | sigma (log-space) | Notes |
|-----------|--------|-------------------|-------|
| E→E | 0.06 nA | 0.5 | Baseline excitatory weight |
| E→I | 0.06 nA | 0.5 | Same as E→E |
| I→E | g_EI (tuned) | 0.5 | Key free parameter; start at 4 × w_mean_EE = 0.24 nA |
| I→I | 0.2 × g_EI | 0.5 | |

**Key free parameter:** `g_EI` (I→E inhibitory weight scale). This is the primary dial for
reaching the AI regime. Brunel (2000) shows the AI regime requires inhibitory weights roughly
4–5× the excitatory weight.

**Plasticity:** OFF in Part 1. All weights are frozen after initialization. STDP is introduced
in Part 2.

---

## External Drive

Each neuron receives an independent background Poisson spike train at rate `nu_ext`.

Brian2 implementation: `PoissonInput(all_neurons, 'I_exc', N=1, rates=nu_ext, weight=w_mean_EE)`

Each Poisson spike increments `I_exc` by `w_mean_EE` (0.06 nA) — equivalent to a single average
E synapse firing. The drive is applied to `I_exc` so it decays with `tau_syn_E`, consistent with
the synaptic model.

**Why Poisson?** Maximum-entropy spike train with a given mean rate — uncorrelated, no temporal
structure. The right null model for background cortical drive. Avoids introducing artificial
correlations before the task is added.

**Free parameter:** `nu_ext` (tuned alongside `g_EI` in Part 1.5).

---

## module: `network.py`

Single exported function:

```python
def build_network(params: dict, seed: int = 42) -> dict:
```

Sets both `np.random.seed(seed)` and `brian2.seed(seed)` at the top of the function. Returns:

```python
{
    'exc':     NeuronGroup,   # E population (N_exc neurons)
    'inh':     NeuronGroup,   # I population (N_inh neurons)
    'syn_EE':  Synapses,
    'syn_EI':  Synapses,
    'syn_IE':  Synapses,
    'syn_II':  Synapses,
    'drive_E': PoissonInput,  # external drive to E
    'drive_I': PoissonInput,  # external drive to I
    'spike_E': SpikeMonitor,
    'spike_I': SpikeMonitor,
    'net':     Network,       # all objects registered
}
```

`params` is a plain dict. Override one key before passing in:
`build_network({**DEFAULT_PARAMS, 'g_EI': new_val * nA})`. Both tuning and validation scripts
use this interface. Part 2 will call it, then add STDP Synapses to `net` before running.

---

## `tune_part1.py`

**Purpose:** Find the operating point (nu_ext, g_EI) that puts the network in the AI regime.
Run once; pick the operating point from the output; pass chosen values to `run_part1.py`.

**Grid:**
```python
nu_ext_grid  = np.linspace(5, 25, 5) * Hz         # [5, 10, 15, 20, 25] Hz
g_EI_scale   = np.linspace(0.5, 2.0, 6)           # scale × w_mean_EE
g_EI_grid    = g_EI_scale * w_mean_EE             # absolute values in nA
```

30 grid points × ~1 s simulation each. Expected runtime: 2–8 min on a laptop.

**Why these ranges?**
- `nu_ext 5–25 Hz`: brackets from near-silent (too little drive) to high-synchrony (too much)
- `g_EI 0.5–2.0 × w_mean_EE`: Brunel (2000) puts the AI band at ~4–5× the E weight; scanning
  0.5–2.0× the initial estimate (0.24 nA) ensures both boundaries of the band are captured

**Per grid point:** build network → run 1 s → compute mean E firing rate + mean CV-ISI.
CV-ISI only computed for neurons with ≥ 5 spikes in 1 s (lower threshold than the 20-spike
minimum in full validation — this is exploratory).

**Output:**
1. `results/tuning_results.csv` — columns: `nu_ext_hz, g_EI_nA, mean_rate_E_hz, mean_CV_ISI`
2. `results/tuning_heatmap.png` — two-panel heatmap (mean rate | CV-ISI) over the grid, with
   contour lines marking the AI-regime boundaries (rate: 2 Hz, 10 Hz; CV: 0.8, 1.2). The
   overlap region where both criteria pass is the target operating point.
3. Stdout table of all results.

---

## `run_part1.py`

**Usage:**
```
python circuit/run_baseline.py --nu_ext 15 --g_EI 0.24
```

**Steps:**
1. Build network with chosen params, `seed=42`, run 5 s simulation
2. Save all figures to `results/figures/` (required for visual checks 1, 2, 5, 6)
3. Auto-evaluate quantitative checks 3, 4, 7; print `PASS`/`FAIL` with measured value for each
4. **If checks 3, 4, 7 all pass:** write `results/baseline_network.h5`, exit 0
5. **If any of 3, 4, 7 fail:** write `results/validation_report.txt` listing failures, do NOT write HDF5, exit 1
6. After exit 0: inspect the four visual figures manually before starting Part 2

The HDF5 write is gated on all checks passing. If the file exists, it implies the baseline is
valid — Part 2 will trust it without re-running validation.

---

## Validation Checks

All checks use the full 5 s simulation except the raster window (1 s).

Checks are split into two categories:
- **Visual (human-inspected):** script saves the figure; you assess from the plot
- **Quantitative (auto-assessed):** script prints `PASS` or `FAIL` with the measured value

| # | Type | Check | Metric | Pass criterion |
|---|------|-------|--------|----------------|
| 1 | Visual | Raster plot | 100 randomly selected neurons, t=0–1 s | Sparse, irregular; no population bursts or silent stretches |
| 2 | Visual | Firing rate distribution | Per-neuron mean rate histogram (all neurons, 5 s) | Approximately log-normal; not bimodal or uniform |
| 3 | Quantitative | CV-ISI | Mean CV-ISI across neurons with ≥ 20 spikes | 0.8 – 1.2 |
| 4 | Quantitative | Pairwise correlations | Mean Pearson r, spike-count vectors (10 ms bins), 50 E-E pairs + 50 E-I pairs, 5 s | < 0.05 |
| 5 | Visual | Power spectrum | Population firing rate spectrum (5 s), `np.fft.rfft` on smoothed rate | No sharp peaks; roughly 1/f character |
| 6 | Visual | Weight distributions | Histograms of initial W_EE, W_EI, W_IE, W_II | Visually log-normal (sanity check on init) |
| 7 | Quantitative | E vs I rates | Mean I rate / Mean E rate | 2–3× (I fires faster than E in balanced regime) |

The script auto-evaluates checks 3, 4, 7 and exits 1 if any fail. Checks 1, 2, 5, 6 require
human inspection of the saved figures — if any look wrong, re-tune and re-run.

### Analysis functions (all in `run_part1.py`)

```python
compute_cv_isi(spike_mon, t_start, t_end, min_spikes=20) -> (per_neuron_cv, mean_cv)
compute_pairwise_corr(spike_mon, t_start, t_end, bin_ms=10, n_pairs=50, seed=42) -> mean_r
compute_power_spectrum(spike_mon, t_start, t_end, smooth_sigma_ms=5) -> (freqs, power)
```

Pair selection in `compute_pairwise_corr` uses `np.random.default_rng(seed=42)` so the same
neuron pairs are selected on every run — important for comparing results across parameter sweeps.

---

## HDF5 Schema (`baseline_network.h5`)

```
/network/
    N_exc               scalar int
    N_inh               scalar int
    p_connect           scalar float
    params_neuron/      group — each key is a scalar float64 dataset in SI units
        tau_m               seconds
        V_rest              volts
        V_th                volts
        V_reset             volts
        tau_ref             seconds
        R                   ohms
    params_synapse/     group — each key is a scalar float64 dataset in SI units
        tau_syn_E           seconds
        tau_syn_I           seconds
        g_EI                amps
        nu_ext              Hz (dimensionless float; unit noted in dataset attrs)

/weights/
    W_EE/  data (float32), row (int32), col (int32), shape (int32[2])  ← COO format
    W_EI/  same
    W_IE/  same
    W_II/  same

/validation/
    mean_rate_E         scalar float (Hz)
    mean_rate_I         scalar float (Hz)
    mean_CV_ISI         scalar float
    mean_pairwise_corr  scalar float
    raster_times        float32 array (spike times, 100 neurons, 0–1 s)
    raster_indices      int32 array   (neuron indices matching raster_times)
    seed                scalar int
    nu_ext_hz           scalar float
    g_EI_nA             scalar float
```

Sparse weights stored as COO triplets. Reconstruct:
`scipy.sparse.coo_matrix((data, (row, col)), shape=shape)`.

All scalar parameters stored in SI units (seconds, volts, amps, ohms) — not milliseconds or mV.
This matches Brian2's internal unit system and avoids silent scale errors when Part 2 loads the
file and passes values back into Brian2.

---

## How to Run (End-to-End)

```bash
# Step 1: find the operating point
python circuit/grid_search.py
# → inspect results/tuning_heatmap.png, pick (nu_ext, g_EI)

# Step 2: validate and save baseline
python circuit/run_baseline.py --nu_ext 15 --g_EI 0.24
# → if all checks pass, results/baseline_network.h5 is written

# Step 3: confirm the handoff file exists before starting Part 2
ls circuit/results/baseline_network.h5
```

Do not proceed to Part 2 until step 3 succeeds.

---

## Parameters Summary

```python
DEFAULT_PARAMS = {
    # neuron
    'tau_m':   20e-3,   # s
    'V_rest':  -70e-3,  # V
    'V_th':    -55e-3,  # V
    'V_reset': -75e-3,  # V
    'tau_ref':  2e-3,   # s
    'R':        100e6,  # Ohm

    # synapse
    'tau_syn_E': 5e-3,  # s
    'tau_syn_I': 10e-3, # s
    'w_mean_EE': 0.06e-9,  # A  (0.06 nA; charge per spike ≈ 0.3 pC)
    'sigma_w':   0.5,      # log-space std for all weight distributions

    # network
    'N_exc':     800,
    'N_inh':     200,
    'p_connect': 0.1,

    # tuned (override with values from tune_part1.py output)
    'g_EI':   4 * 0.06e-9,  # A  (starting point; tuned empirically)
    'nu_ext': 10.0,          # Hz (starting point; tuned empirically)
}
```

---

## References

- Brunel N (2000). Dynamics of sparsely connected networks of excitatory and inhibitory spiking
  neurons. *J Comput Neurosci* 8(3):183–208.
- Braitenberg V & Schüz A (1998). *Cortex: Statistics and Geometry of Neuronal Connectivity.*
- Buzsáki G & Mizuseki K (2014). The log-dynamic brain. *Nat Rev Neurosci* 15(4):264–278.
- Gao P et al. (2017). A theory of multineuronal dimensionality, dynamics and measurement.
  *bioRxiv.*
- Hromadka T, DeWeese MR & Zador AM (2008). Sparse representation of sounds in the unanesthetized
  auditory cortex. *PLOS Biol* 6(1):e16.
- Koch C (1999). *Biophysics of Computation.* Oxford University Press.
- Softky WR & Koch C (1993). The highly irregular firing of cortical cells is inconsistent with
  temporal integration of random EPSPs. *J Neurosci* 13(1):334–350.
