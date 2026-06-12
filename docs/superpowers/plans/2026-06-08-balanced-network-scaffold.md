# Part 1: Network Scaffold and E/I Balance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a 1000-neuron LIF network in Brian2 that fires in the
asynchronous-irregular (AI) regime, then save a reproducible baseline to HDF5 as the
starting point for Part 2.

**Architecture:** `network.py` is a shared factory (`build_network(params, seed) → dict`).
`tune_part1.py` runs a 30-point 2D parameter grid to find the AI operating point.
`run_part1.py` runs the 5 s validation, auto-evaluates three quantitative checks,
saves figures for four visual checks, and gates the HDF5 write on all quantitative checks
passing.

**Tech Stack:** Python 3.10+, Brian2 ≥ 2.5, NumPy, SciPy, Matplotlib, h5py, pytest

---

## File Map

```
part1/
  __init__.py           empty — makes part1 importable
  network.py            build_network(params, seed) + DEFAULT_PARAMS + _lognormal_weights()
  tune_part1.py         grid search, CSV output, two-panel heatmap
  run_part1.py          analysis functions, save_baseline(), validation workflow, CLI

tests/
  __init__.py           empty
  test_network.py       structural tests for build_network (no full simulation)
  test_analysis.py      unit tests for compute_cv_isi, compute_pairwise_corr
  test_hdf5.py          HDF5 round-trip test

requirements.txt        brian2, numpy, scipy, matplotlib, h5py, pytest
.gitignore              circuit/results/
```

---

## Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `part1/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`
- Create: `circuit/results/` directory (gitignored)

- [ ] **Step 1: Initialise git and create directory structure**

```bash
cd "/Users/vibharangappa/Desktop/Hebbian plasticity- motor project"
git init
mkdir -p circuit/results/figures
mkdir -p tests
touch part1/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

```
brian2>=2.5.0
numpy
scipy
matplotlib
h5py
pytest
```

- [ ] **Step 3: Write `.gitignore`**

```
circuit/results/
__pycache__/
*.pyc
.pytest_cache/
brian_objects/
```

- [ ] **Step 4: Create the conda environment and verify imports**

```bash
conda create -n hebbian-motor python=3.11 -y
conda activate hebbian-motor
pip install -r requirements.txt
python -c "import brian2; import h5py; import scipy; print('imports OK')"
```

Expected output: `imports OK`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore part1/__init__.py tests/__init__.py
git commit -m "chore: project scaffold — requirements, gitignore, directory structure"
```

---

## Task 2: `network.py` — neuron groups

**Files:**
- Create: `circuit/network.py`
- Create: `tests/test_network.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_network.py

import numpy as np
import pytest
from brian2 import start_scope

from circuit.network import build_network, DEFAULT_PARAMS


def test_build_network_returns_required_keys():
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    expected = {
        'exc', 'inh',
        'syn_EE', 'syn_EI', 'syn_IE', 'syn_II',
        'drive_E', 'drive_I',
        'spike_E', 'spike_I',
        'net',
    }
    assert set(objs.keys()) == expected


def test_neuron_group_sizes():
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    assert len(objs['exc']) == DEFAULT_PARAMS['N_exc']
    assert len(objs['inh']) == DEFAULT_PARAMS['N_inh']
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/vibharangappa/Desktop/Hebbian plasticity- motor project"
python -m pytest tests/test_network.py -v
```

Expected: `ModuleNotFoundError: No module named 'circuit.network'` or similar.

- [ ] **Step 3: Implement neuron groups in `network.py`**

```python
# circuit/network.py

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

    # Synapses, external drive, monitors added in later tasks.
    # Return partial dict so Task 2 tests can run without Task 3 code.
    return {
        'exc': exc, 'inh': inh,
        'syn_EE': None, 'syn_EI': None, 'syn_IE': None, 'syn_II': None,
        'drive_E': None, 'drive_I': None,
        'spike_E': None, 'spike_I': None,
        'net': None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_network.py::test_build_network_returns_required_keys \
                 tests/test_network.py::test_neuron_group_sizes -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add circuit/network.py tests/test_network.py
git commit -m "feat: add build_network skeleton with LIF neuron groups"
```

---

## Task 3: `network.py` — synapses and log-normal weight initialisation

**Files:**
- Modify: `circuit/network.py`
- Modify: `tests/test_network.py`

- [ ] **Step 1: Add failing tests for synaptic structure**

Append to `tests/test_network.py`:

```python
def test_no_self_connections_EE():
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    syn = objs['syn_EE']
    # Brian2 Synapses.i = presynaptic indices, .j = postsynaptic indices
    pre = np.array(syn.i[:])
    post = np.array(syn.j[:])
    assert np.all(pre != post), "syn_EE contains self-connections"


def test_no_self_connections_II():
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    syn = objs['syn_II']
    pre = np.array(syn.i[:])
    post = np.array(syn.j[:])
    assert np.all(pre != post), "syn_II contains self-connections"


def test_connectivity_fraction_EE():
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    Ne = DEFAULT_PARAMS['N_exc']
    p = DEFAULT_PARAMS['p_connect']
    expected = Ne * (Ne - 1) * p          # no self-connections → Ne*(Ne-1) possible
    actual = len(objs['syn_EE'])
    # Allow ±20% deviation (Erdos-Renyi variance)
    assert abs(actual - expected) / expected < 0.20, \
        f"EE connectivity {actual} far from expected {expected:.0f}"


def test_weights_positive():
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    from brian2 import amp as brian_amp
    for key in ('syn_EE', 'syn_EI', 'syn_IE', 'syn_II'):
        w = np.array(objs[key].w / brian_amp)  # strip units → float array in amps
        assert np.all(w > 0), f"{key} has non-positive weights"


def test_weight_mean_EE():
    """Mean E->E weight should be close to w_mean_EE (within 20%)."""
    start_scope()
    objs = build_network(DEFAULT_PARAMS, seed=42)
    from brian2 import amp as brian_amp
    w = np.array(objs['syn_EE'].w / brian_amp)
    target = DEFAULT_PARAMS['w_mean_EE']
    assert abs(w.mean() - target) / target < 0.20, \
        f"Mean EE weight {w.mean():.3e} A too far from target {target:.3e} A"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_network.py::test_no_self_connections_EE \
                 tests/test_network.py::test_weights_positive -v
```

Expected: `AttributeError` or `TypeError` because `objs['syn_EE']` is `None`.

- [ ] **Step 3: Add synapses and weight init to `build_network`**

Replace the `# Synapses, external drive, monitors added in later tasks.` comment block
and the `return` statement at the bottom of `build_network` with:

```python
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

    syn_EE.w = _lognormal_weights(w_ee,       sigma, len(syn_EE), rng) * amp
    syn_EI.w = _lognormal_weights(w_ee,       sigma, len(syn_EI), rng) * amp
    syn_IE.w = _lognormal_weights(g_ei,       sigma, len(syn_IE), rng) * amp
    syn_II.w = _lognormal_weights(0.2 * g_ei, sigma, len(syn_II), rng) * amp

    # Placeholders for drive/monitors — completed in Task 4
    return {
        'exc': exc, 'inh': inh,
        'syn_EE': syn_EE, 'syn_EI': syn_EI,
        'syn_IE': syn_IE, 'syn_II': syn_II,
        'drive_E': None, 'drive_I': None,
        'spike_E': None, 'spike_I': None,
        'net': None,
    }
```

- [ ] **Step 4: Run all network tests so far**

```bash
python -m pytest tests/test_network.py -v
```

Expected: all 7 tests PASS. (The two key tests are `test_no_self_connections_EE` and
`test_weights_positive`; the others from Task 2 must still pass.)

- [ ] **Step 5: Commit**

```bash
git add circuit/network.py tests/test_network.py
git commit -m "feat: add synapses and log-normal weight initialisation to build_network"
```

---

## Task 4: `network.py` — external drive and full network assembly

**Files:**
- Modify: `circuit/network.py`
- Modify: `tests/test_network.py`

- [ ] **Step 1: Add failing smoke test**

Append to `tests/test_network.py`:

```python
def test_network_runs_and_produces_spikes():
    """
    Build a small network (80 E, 20 I) and run 200 ms.
    Verifies the full assembly compiles and spikes are produced.
    Using N=100 instead of 1000 keeps this test under ~10 s.
    """
    start_scope()
    small = {
        **DEFAULT_PARAMS,
        'N_exc': 80,
        'N_inh': 20,
        'nu_ext': 20.0,  # higher drive to guarantee spikes in short window
    }
    objs = build_network(small, seed=0)
    objs['net'].run(0.2 * second)  # imported from brian2 via network.py
    assert objs['spike_E'].num_spikes > 0, \
        "No E spikes in 200 ms — network may be silent"
    assert objs['spike_I'].num_spikes > 0, \
        "No I spikes in 200 ms — network may be silent"
```

Note: `second` needs to be importable in the test. Add this import at the top of
`tests/test_network.py`:

```python
from brian2 import start_scope, second
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_network.py::test_network_runs_and_produces_spikes -v
```

Expected: `AttributeError: 'NoneType' object has no attribute 'run'`

- [ ] **Step 3: Complete `build_network` — add drive, monitors, Network**

Replace the `# Placeholders for drive/monitors — completed in Task 4` return block
with:

```python
    # ------------------------------------------------------------------
    # External Poisson drive
    # Each neuron gets 1 independent Poisson process at nu_ext Hz.
    # Each spike adds w_mean_EE to I_exc — equivalent to one background E synapse.
    # Drive goes to I_exc (not I_inh) so it decays with tau_syn_E.
    # ------------------------------------------------------------------
    drive_E = PoissonInput(exc, 'I_exc', N=1,
                           rates=p['nu_ext'] * Hz,
                           weight=p['w_mean_EE'] * amp)
    drive_I = PoissonInput(inh, 'I_exc', N=1,
                           rates=p['nu_ext'] * Hz,
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
```

- [ ] **Step 4: Run all network tests**

```bash
python -m pytest tests/test_network.py -v
```

Expected: all 8 tests PASS. The smoke test may take ~5–20 s.

- [ ] **Step 5: Commit**

```bash
git add circuit/network.py tests/test_network.py
git commit -m "feat: complete build_network with PoissonInput, SpikeMonitors, Network"
```

---

## Task 5: `run_part1.py` — analysis functions

Analysis functions operate on plain Python dicts of spike times — no Brian2 objects.
This makes them unit-testable without running simulations.

**Files:**
- Create: `circuit/run_baseline.py`
- Create: `tests/test_analysis.py`

- [ ] **Step 1: Write failing tests for analysis functions**

```python
# tests/test_analysis.py

import numpy as np
import pytest

from circuit.run_baseline import compute_cv_isi, compute_pairwise_corr, compute_power_spectrum


def _make_poisson_trains(n_neurons: int, rate_hz: float, t_end: float, seed: int) -> dict:
    """Generate independent Poisson spike trains for testing."""
    rng = np.random.default_rng(seed)
    trains = {}
    for i in range(n_neurons):
        n = rng.poisson(rate_hz * t_end)
        trains[i] = np.sort(rng.uniform(0, t_end, n))
    return trains


def _make_regular_trains(n_neurons: int, rate_hz: float, t_end: float) -> dict:
    """Generate perfectly regular (clock-like) spike trains for testing."""
    isi = 1.0 / rate_hz
    times = np.arange(isi, t_end, isi)
    return {i: times.copy() for i in range(n_neurons)}


# ---- compute_cv_isi ----

def test_cv_isi_poisson_near_one():
    """Independent Poisson trains → CV-ISI ≈ 1.0."""
    trains = _make_poisson_trains(n_neurons=200, rate_hz=10.0, t_end=10.0, seed=0)
    _, mean_cv = compute_cv_isi(trains, t_start=0.0, t_end=10.0, min_spikes=20)
    assert abs(mean_cv - 1.0) < 0.15, f"Expected CV≈1 for Poisson, got {mean_cv:.3f}"


def test_cv_isi_regular_near_zero():
    """Regular spike trains → CV-ISI ≈ 0."""
    trains = _make_regular_trains(n_neurons=20, rate_hz=10.0, t_end=10.0)
    _, mean_cv = compute_cv_isi(trains, t_start=0.0, t_end=10.0, min_spikes=20)
    assert mean_cv < 0.05, f"Expected CV≈0 for regular trains, got {mean_cv:.3f}"


def test_cv_isi_excludes_low_spike_neurons():
    """Neurons with fewer than min_spikes spikes should be excluded."""
    trains = {0: np.array([0.1, 0.2, 0.3])}  # only 3 spikes
    per_neuron, mean_cv = compute_cv_isi(trains, t_start=0.0, t_end=5.0, min_spikes=10)
    assert len(per_neuron) == 0
    assert np.isnan(mean_cv)


# ---- compute_pairwise_corr ----

def test_pairwise_corr_uncorrelated_near_zero():
    """Independent Poisson trains → mean pairwise correlation ≈ 0."""
    trains = _make_poisson_trains(n_neurons=100, rate_hz=10.0, t_end=10.0, seed=1)
    r = compute_pairwise_corr(trains, t_start=0.0, t_end=10.0,
                               bin_ms=10.0, n_pairs=50, seed=42)
    assert abs(r) < 0.08, f"Expected r≈0 for independent trains, got {r:.4f}"


def test_pairwise_corr_identical_trains():
    """Identical trains → correlation = 1."""
    times = np.array([0.1, 0.3, 0.5, 0.9, 1.4, 2.0])
    trains = {0: times, 1: times}
    r = compute_pairwise_corr(trains, t_start=0.0, t_end=3.0,
                               bin_ms=10.0, n_pairs=1, seed=0)
    assert abs(r - 1.0) < 1e-6, f"Expected r=1 for identical trains, got {r:.6f}"


# ---- compute_power_spectrum ----

def test_power_spectrum_output_shapes():
    """freqs and power must have equal length; freqs[0] = 0; power >= 0."""
    trains = _make_poisson_trains(n_neurons=50, rate_hz=5.0, t_end=5.0, seed=2)
    freqs, power = compute_power_spectrum(trains, t_start=0.0, t_end=5.0, smooth_sigma_ms=5.0)
    assert len(freqs) == len(power)
    assert freqs[0] == pytest.approx(0.0)
    assert np.all(power >= 0)
    assert len(freqs) > 10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_analysis.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` because `run_part1.py` does not exist.

- [ ] **Step 3: Implement analysis functions in `run_part1.py`**

```python
# circuit/run_baseline.py

"""
Validation runner for Part 1 of the Hebbian Plasticity / Manifold Sculptor project.

Usage:
    python circuit/run_baseline.py --nu_ext 15.0 --g_EI 0.24

Runs a 5 s simulation, auto-evaluates checks 3, 4, 7, saves figures for
visual checks 1, 2, 5, 6, writes baseline_network.h5 if all quantitative
checks pass.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves to file without a display
import matplotlib.pyplot as plt
import h5py
import scipy.sparse
from scipy.ndimage import gaussian_filter1d
from brian2 import second, amp

from circuit.network import build_network, DEFAULT_PARAMS


# ---------------------------------------------------------------------------
# Analysis functions — operate on plain dicts, no Brian2 dependencies.
# spike_trains : dict[int, np.ndarray]  — neuron_idx → spike times in seconds (float)
# ---------------------------------------------------------------------------

def compute_cv_isi(
    spike_trains: dict,
    t_start: float,
    t_end: float,
    min_spikes: int = 20,
) -> tuple:
    """
    Compute per-neuron CV-ISI and the population mean.

    Only neurons with >= min_spikes spikes in [t_start, t_end] are included.

    Returns
    -------
    per_neuron : dict {neuron_idx: float}   — CV for each qualifying neuron
    mean_cv    : float                       — population mean (nan if none qualify)
    """
    per_neuron = {}
    for idx, times in spike_trains.items():
        times = np.asarray(times)
        in_win = times[(times >= t_start) & (times <= t_end)]
        if len(in_win) < min_spikes:
            continue
        isis = np.diff(np.sort(in_win))
        if len(isis) < 2:
            continue
        cv = float(isis.std() / isis.mean())
        per_neuron[idx] = cv

    mean_cv = float(np.mean(list(per_neuron.values()))) if per_neuron else float('nan')
    return per_neuron, mean_cv


def compute_pairwise_corr(
    spike_trains: dict,
    t_start: float,
    t_end: float,
    bin_ms: float = 10.0,
    n_pairs: int = 50,
    seed: int = 42,
) -> float:
    """
    Compute mean Pearson correlation of spike-count vectors across random pairs.

    Pairs are drawn with a fixed seed so results are reproducible.
    Returns mean Pearson r (nan if fewer than 2 neurons).
    """
    dt = bin_ms * 1e-3
    n_bins = int((t_end - t_start) / dt)
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)

    indices = sorted(spike_trains.keys())
    n = len(indices)
    if n < 2:
        return float('nan')

    # Build spike-count matrix: shape (n_neurons, n_bins)
    counts = np.zeros((n, n_bins), dtype=np.float32)
    for row, idx in enumerate(indices):
        times = np.asarray(spike_trains[idx])
        times = times[(times >= t_start) & (times < t_end)]
        counts[row], _ = np.histogram(times, bins=bin_edges)

    rng = np.random.default_rng(seed)
    n_pairs = min(n_pairs, n * (n - 1) // 2)

    # Draw unique pairs without replacement
    pairs = set()
    max_attempts = n_pairs * 100
    attempts = 0
    while len(pairs) < n_pairs and attempts < max_attempts:
        i, j = rng.choice(n, size=2, replace=False)
        pairs.add((min(i, j), max(i, j)))
        attempts += 1

    rs = []
    for i, j in pairs:
        r = np.corrcoef(counts[i], counts[j])[0, 1]
        if np.isfinite(r):
            rs.append(float(r))

    return float(np.mean(rs)) if rs else float('nan')


def compute_power_spectrum(
    spike_trains: dict,
    t_start: float,
    t_end: float,
    smooth_sigma_ms: float = 5.0,
    dt_ms: float = 0.1,
) -> tuple:
    """
    Compute the power spectrum of the summed population firing rate.

    Spikes from all neurons are summed into a fine-bin histogram, smoothed
    with a Gaussian kernel, then FFT'd. Returns (frequencies_Hz, power).
    """
    dt = dt_ms * 1e-3
    n_bins = int((t_end - t_start) / dt)
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)

    pop_rate = np.zeros(n_bins, dtype=np.float64)
    for times in spike_trains.values():
        times = np.asarray(times)
        times = times[(times >= t_start) & (times < t_end)]
        counts, _ = np.histogram(times, bins=bin_edges)
        pop_rate += counts

    sigma_samples = (smooth_sigma_ms * 1e-3) / dt
    pop_smooth = gaussian_filter1d(pop_rate, sigma=sigma_samples)

    fft_vals = np.fft.rfft(pop_smooth)
    freqs = np.fft.rfftfreq(n_bins, d=dt)
    power = np.abs(fft_vals) ** 2

    return freqs, power
```

- [ ] **Step 4: Run analysis tests**

```bash
python -m pytest tests/test_analysis.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add circuit/run_baseline.py tests/test_analysis.py
git commit -m "feat: add analysis functions (CV-ISI, pairwise corr, power spectrum)"
```

---

## Task 6: `run_part1.py` — HDF5 save function

**Files:**
- Modify: `circuit/run_baseline.py`
- Create: `tests/test_hdf5.py`

- [ ] **Step 1: Write the failing HDF5 round-trip test**

```python
# tests/test_hdf5.py

import os
import numpy as np
import h5py
import scipy.sparse
import pytest
from brian2 import start_scope, second

from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import save_baseline


@pytest.fixture
def small_network(tmp_path):
    """Build a small network, run 0.5 s, and return everything needed for save_baseline."""
    start_scope()
    small = {**DEFAULT_PARAMS, 'N_exc': 80, 'N_inh': 20, 'nu_ext': 15.0}
    objs = build_network(small, seed=7)
    objs['net'].run(0.5 * second)

    trains_E = {k: np.array(v / second)
                for k, v in objs['spike_E'].spike_trains().items()}

    validation = {
        'mean_rate_E':        objs['spike_E'].num_spikes / (small['N_exc'] * 0.5),
        'mean_rate_I':        objs['spike_I'].num_spikes / (small['N_inh'] * 0.5),
        'mean_CV_ISI':        0.95,   # placeholder value
        'mean_pairwise_corr': 0.02,
        'raster_times':       np.array([0.1, 0.2, 0.3], dtype=np.float32),
        'raster_indices':     np.array([0, 1, 2],        dtype=np.int32),
    }
    return small, objs, validation, tmp_path / 'test_baseline.h5'


def test_hdf5_required_groups_exist(small_network):
    params, objs, validation, path = small_network
    save_baseline(str(path), params, objs, validation, seed=7)
    with h5py.File(path, 'r') as f:
        for group in ('network', 'weights', 'validation'):
            assert group in f, f"Missing group /{group}"


def test_hdf5_network_scalars(small_network):
    params, objs, validation, path = small_network
    save_baseline(str(path), params, objs, validation, seed=7)
    with h5py.File(path, 'r') as f:
        assert f['network/N_exc'][()] == 80
        assert f['network/N_inh'][()] == 20
        assert f['network/params_neuron/tau_m'][()] == pytest.approx(20e-3)


def test_hdf5_weight_coo_reconstruction(small_network):
    params, objs, validation, path = small_network
    save_baseline(str(path), params, objs, validation, seed=7)
    with h5py.File(path, 'r') as f:
        data  = f['weights/W_EE/data'][:]
        row   = f['weights/W_EE/row'][:]
        col   = f['weights/W_EE/col'][:]
        shape = f['weights/W_EE/shape'][:]
    W = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    assert W.shape == (80, 80)
    assert W.nnz > 0
    assert np.all(W.data > 0), "Weights must be positive"


def test_hdf5_validation_fields(small_network):
    params, objs, validation, path = small_network
    save_baseline(str(path), params, objs, validation, seed=7)
    with h5py.File(path, 'r') as f:
        for field in ('mean_rate_E', 'mean_rate_I', 'mean_CV_ISI',
                      'mean_pairwise_corr', 'raster_times', 'raster_indices',
                      'seed', 'nu_ext_hz', 'g_EI_nA'):
            assert field in f['validation'], f"Missing /validation/{field}"
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_hdf5.py -v
```

Expected: `ImportError: cannot import name 'save_baseline' from 'circuit.run_baseline'`

- [ ] **Step 3: Add `save_baseline` to `run_part1.py`**

Append after the `compute_power_spectrum` function:

```python
# ---------------------------------------------------------------------------
# HDF5 save
# ---------------------------------------------------------------------------

def save_baseline(
    path: str,
    params: dict,
    net_objs: dict,
    validation: dict,
    seed: int,
) -> None:
    """
    Write the validated baseline network to HDF5.

    Weight matrices stored in COO format — reconstruct with:
        W = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)

    All parameters stored in SI units to match Brian2's internal system.
    Weights stored as float32 in SI (amps); shape as int32[2].
    """
    def _save_sparse(grp, name: str, syn, tgt_size: int, src_size: int):
        # Strip Brian2 units before storing: divide by `amp` → float array in amps
        w_vals = np.array(syn.w / amp, dtype=np.float32)
        # .j = postsynaptic (row), .i = presynaptic (col) → W[post, pre]
        rows = np.array(syn.j[:], dtype=np.int32)
        cols = np.array(syn.i[:], dtype=np.int32)
        g = grp.create_group(name)
        g.create_dataset('data',  data=w_vals)
        g.create_dataset('row',   data=rows)
        g.create_dataset('col',   data=cols)
        g.create_dataset('shape', data=np.array([tgt_size, src_size], dtype=np.int32))

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    with h5py.File(path, 'w') as f:
        # /network
        ng = f.create_group('network')
        ng.create_dataset('N_exc',     data=int(params['N_exc']))
        ng.create_dataset('N_inh',     data=int(params['N_inh']))
        ng.create_dataset('p_connect', data=float(params['p_connect']))

        pn = ng.create_group('params_neuron')
        for k in ('tau_m', 'V_rest', 'V_th', 'V_reset', 'tau_ref', 'R'):
            pn.create_dataset(k, data=float(params[k]))

        ps = ng.create_group('params_synapse')
        for k in ('tau_syn_E', 'tau_syn_I', 'g_EI'):
            ps.create_dataset(k, data=float(params[k]))
        ps.create_dataset('nu_ext', data=float(params['nu_ext']))

        # /weights — COO sparse format, SI units (amps as float32)
        wg = f.create_group('weights')
        Ne, Ni = params['N_exc'], params['N_inh']
        _save_sparse(wg, 'W_EE', net_objs['syn_EE'], Ne, Ne)
        _save_sparse(wg, 'W_EI', net_objs['syn_EI'], Ni, Ne)  # target=I, source=E
        _save_sparse(wg, 'W_IE', net_objs['syn_IE'], Ne, Ni)  # target=E, source=I
        _save_sparse(wg, 'W_II', net_objs['syn_II'], Ni, Ni)

        # /validation
        vg = f.create_group('validation')
        vg.create_dataset('mean_rate_E',        data=float(validation['mean_rate_E']))
        vg.create_dataset('mean_rate_I',        data=float(validation['mean_rate_I']))
        vg.create_dataset('mean_CV_ISI',        data=float(validation['mean_CV_ISI']))
        vg.create_dataset('mean_pairwise_corr', data=float(validation['mean_pairwise_corr']))
        vg.create_dataset('raster_times',
                          data=np.asarray(validation['raster_times'], dtype=np.float32))
        vg.create_dataset('raster_indices',
                          data=np.asarray(validation['raster_indices'], dtype=np.int32))
        vg.create_dataset('seed',      data=int(seed))
        vg.create_dataset('nu_ext_hz', data=float(params['nu_ext']))
        vg.create_dataset('g_EI_nA',   data=float(params['g_EI'] / 1e-9))  # A → nA
```

- [ ] **Step 4: Run HDF5 tests**

```bash
python -m pytest tests/test_hdf5.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run all tests to check no regressions**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add circuit/run_baseline.py tests/test_hdf5.py
git commit -m "feat: add save_baseline with COO sparse HDF5 format"
```

---

## Task 7: `run_part1.py` — plotting, validation workflow, and CLI

**Files:**
- Modify: `circuit/run_baseline.py`

No new tests — the validation workflow is itself the integration test (run on real
simulation output). A manual smoke test on a small network is included.

- [ ] **Step 1: Add plotting functions to `run_part1.py`**

Append after `save_baseline`:

```python
# ---------------------------------------------------------------------------
# Plotting — save figures to disk for visual inspection
# ---------------------------------------------------------------------------

def _extract_spike_trains(monitor, n_neurons: float, t_sim: float) -> dict:
    """Convert Brian2 SpikeMonitor spike_trains() to plain float arrays."""
    return {k: np.array(v / second)
            for k, v in monitor.spike_trains().items()}


def plot_raster(net_objs: dict, params: dict, t_raster: float,
                results_dir: str) -> tuple:
    """
    Plot spike raster for 100 random E neurons over [0, t_raster].
    Returns (raster_times, raster_indices) for HDF5 storage.
    """
    rng = np.random.default_rng(42)
    Ne = params['N_exc']
    n_sample = min(100, Ne)
    sample_idx = rng.choice(Ne, size=n_sample, replace=False)

    all_t = np.array(net_objs['spike_E'].t / second)
    all_i = np.array(net_objs['spike_E'].i[:])

    mask = (all_t < t_raster) & np.isin(all_i, sample_idx)
    rt = all_t[mask].astype(np.float32)
    ri = all_i[mask].astype(np.int32)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(rt, ri, s=0.5, c='k', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron index')
    ax.set_title(f'Raster — {n_sample} E neurons, t=0–{t_raster} s')
    ax.set_xlim(0, t_raster)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'figures', 'raster.png'), dpi=150)
    plt.close(fig)

    return rt, ri


def plot_firing_rate_hist(trains_E: dict, trains_I: dict, t_end: float,
                          results_dir: str) -> None:
    """Histogram of per-neuron mean firing rates (E and I populations)."""
    rates_E = np.array([len(t[(t >= 0) & (t <= t_end)]) / t_end
                        for t in trains_E.values()])
    rates_I = np.array([len(t[(t >= 0) & (t <= t_end)]) / t_end
                        for t in trains_I.values()])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, rates, pop in zip(axes, [rates_E, rates_I], ['E', 'I']):
        ax.hist(rates, bins=30, color='steelblue' if pop == 'E' else 'tomato',
                edgecolor='k', linewidth=0.3)
        ax.set_xlabel('Mean firing rate (Hz)')
        ax.set_ylabel('Count')
        ax.set_title(f'{pop} population — mean={rates.mean():.1f} Hz')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'figures', 'firing_rate_hist.png'), dpi=150)
    plt.close(fig)


def plot_isi_dist(trains_E: dict, t_end: float, results_dir: str) -> None:
    """ISI distribution for 6 randomly selected E neurons."""
    rng = np.random.default_rng(99)
    keys = rng.choice(list(trains_E.keys()), size=min(6, len(trains_E)), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, idx in zip(axes.flat, keys):
        times = trains_E[idx]
        times = times[(times >= 0) & (times <= t_end)]
        if len(times) < 3:
            ax.set_visible(False)
            continue
        isis = np.diff(np.sort(times)) * 1000  # convert to ms
        cv = isis.std() / isis.mean() if len(isis) > 1 else float('nan')
        ax.hist(isis, bins=20, color='steelblue', edgecolor='k', linewidth=0.3)
        ax.set_xlabel('ISI (ms)')
        ax.set_title(f'Neuron {idx} — CV={cv:.2f}')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'figures', 'isi_dist.png'), dpi=150)
    plt.close(fig)


def plot_pairwise_corr(trains_E: dict, t_end: float, mean_r: float,
                       results_dir: str) -> None:
    """Distribution of pairwise correlations for 200 random E-E pairs."""
    rng = np.random.default_rng(7)
    dt = 10e-3  # 10 ms bins
    n_bins = int(t_end / dt)
    bin_edges = np.linspace(0, t_end, n_bins + 1)

    indices = sorted(trains_E.keys())
    counts = np.zeros((len(indices), n_bins), dtype=np.float32)
    for row, idx in enumerate(indices):
        t = trains_E[idx]
        t = t[(t >= 0) & (t < t_end)]
        counts[row], _ = np.histogram(t, bins=bin_edges)

    n = len(indices)
    n_pairs = min(200, n * (n - 1) // 2)
    pairs = set()
    while len(pairs) < n_pairs:
        i, j = rng.choice(n, size=2, replace=False)
        pairs.add((min(i, j), max(i, j)))

    rs = [np.corrcoef(counts[i], counts[j])[0, 1] for i, j in pairs
          if np.isfinite(np.corrcoef(counts[i], counts[j])[0, 1])]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rs, bins=30, color='purple', edgecolor='k', linewidth=0.3)
    ax.axvline(mean_r, color='r', linewidth=1.5, label=f'mean r={mean_r:.4f}')
    ax.set_xlabel('Pearson r')
    ax.set_ylabel('Count')
    ax.set_title('Pairwise spike-count correlation (E-E, 200 pairs)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'figures', 'pairwise_corr.png'), dpi=150)
    plt.close(fig)


def plot_power_spectrum(trains_E: dict, t_end: float, results_dir: str) -> None:
    """Power spectrum of the E population firing rate."""
    freqs, power = compute_power_spectrum(trains_E, t_start=0.0, t_end=t_end,
                                          smooth_sigma_ms=5.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    # Plot only up to 200 Hz; lower frequencies dominate
    mask = (freqs > 0) & (freqs < 200)
    ax.loglog(freqs[mask], power[mask], lw=0.8)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (a.u.)')
    ax.set_title('Population firing rate power spectrum (E neurons)')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'figures', 'power_spectrum.png'), dpi=150)
    plt.close(fig)


def plot_weight_hists(net_objs: dict, results_dir: str) -> None:
    """Log-scale histograms of initial weight distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    syns = [('W_EE', 'syn_EE', 'C0'), ('W_EI', 'syn_EI', 'C1'),
            ('W_IE', 'syn_IE', 'C2'), ('W_II', 'syn_II', 'C3')]
    for ax, (name, key, color) in zip(axes.flat, syns):
        w_nA = np.array(net_objs[key].w / amp) / 1e-9  # amps → nA
        ax.hist(w_nA, bins=40, color=color, edgecolor='k', linewidth=0.2)
        ax.set_xlabel('Weight (nA)')
        ax.set_title(name)
        ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'figures', 'weight_hists.png'), dpi=150)
    plt.close(fig)
```

- [ ] **Step 2: Add validation workflow and `main` to `run_part1.py`**

Append to the end of `run_part1.py`:

```python
# ---------------------------------------------------------------------------
# Validation workflow
# ---------------------------------------------------------------------------

def run_validation(
    net_objs: dict,
    params: dict,
    t_sim: float,
    seed: int,
    results_dir: str,
) -> tuple:
    """
    Run all 7 validation checks after a completed simulation.

    Auto-evaluates checks 3, 4, 7 (quantitative); saves figures for
    checks 1, 2, 5, 6 (visual).

    Returns
    -------
    validation : dict   — values for HDF5 /validation group
    passed     : bool   — True iff checks 3, 4, 7 all pass
    """
    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)

    Ne = params['N_exc']
    Ni = params['N_inh']

    trains_E = _extract_spike_trains(net_objs['spike_E'], Ne, t_sim)
    trains_I = _extract_spike_trains(net_objs['spike_I'], Ni, t_sim)

    # ---- Check 3: CV-ISI (quantitative) ----
    _, mean_cv = compute_cv_isi(trains_E, 0.0, t_sim, min_spikes=20)
    cv_pass = 0.8 <= mean_cv <= 1.2

    # ---- Check 4: Pairwise correlation (quantitative) ----
    mean_r = compute_pairwise_corr(trains_E, 0.0, t_sim,
                                    bin_ms=10.0, n_pairs=50, seed=seed)
    pairwise_pass = (not np.isnan(mean_r)) and mean_r < 0.05

    # ---- Check 7: I/E rate ratio (quantitative) ----
    mean_rate_E = net_objs['spike_E'].num_spikes / (Ne * t_sim)
    mean_rate_I = net_objs['spike_I'].num_spikes / (Ni * t_sim)
    rate_ratio  = mean_rate_I / mean_rate_E if mean_rate_E > 0 else float('nan')
    rate_pass   = 2.0 <= rate_ratio <= 3.0

    # ---- Print results ----
    width = 26
    print(f"\n{'=' * 50}")
    print(f"{'Validation results':^50}")
    print(f"{'=' * 50}")
    print(f"{'Check 3 (CV-ISI):':<{width}} {mean_cv:.3f}   "
          f"{'PASS' if cv_pass else 'FAIL'}  [target: 0.8–1.2]")
    print(f"{'Check 4 (pairwise r):':<{width}} {mean_r:.4f}  "
          f"{'PASS' if pairwise_pass else 'FAIL'}  [target: <0.05]")
    print(f"{'Check 7 (I/E rate ratio):':<{width}} {rate_ratio:.2f}    "
          f"{'PASS' if rate_pass else 'FAIL'}  [target: 2–3×]")
    print(f"{'Mean E rate:':<{width}} {mean_rate_E:.2f} Hz")
    print(f"{'Mean I rate:':<{width}} {mean_rate_I:.2f} Hz")
    print(f"{'=' * 50}\n")

    # ---- Figures (checks 1, 2, 5, 6 — human-inspected) ----
    raster_t, raster_i = plot_raster(net_objs, params, t_raster=1.0,
                                      results_dir=results_dir)
    plot_firing_rate_hist(trains_E, trains_I, t_sim, results_dir)
    plot_isi_dist(trains_E, t_sim, results_dir)
    plot_pairwise_corr(trains_E, t_sim, mean_r, results_dir)
    plot_power_spectrum(trains_E, t_sim, results_dir)
    plot_weight_hists(net_objs, results_dir)

    print(f"Figures saved to {os.path.join(results_dir, 'figures')}/")
    print("Manually inspect: raster.png, firing_rate_hist.png, "
          "power_spectrum.png, weight_hists.png\n")

    validation = {
        'mean_rate_E':        mean_rate_E,
        'mean_rate_I':        mean_rate_I,
        'mean_CV_ISI':        mean_cv,
        'mean_pairwise_corr': mean_r,
        'raster_times':       raster_t,
        'raster_indices':     raster_i,
    }

    all_passed = cv_pass and pairwise_pass and rate_pass
    return validation, all_passed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run Part 1 validation and save baseline network to HDF5.')
    parser.add_argument('--nu_ext', type=float, required=True,
                        help='Background Poisson rate (Hz)')
    parser.add_argument('--g_EI',   type=float, required=True,
                        help='Mean I→E inhibitory weight (nA)')
    parser.add_argument('--t_sim',  type=float, default=5.0,
                        help='Simulation duration in seconds (default: 5.0)')
    parser.add_argument('--seed',   type=int,   default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'results'),
                        help='Directory for results and figures')
    args = parser.parse_args()

    params = {
        **DEFAULT_PARAMS,
        'nu_ext': args.nu_ext,
        'g_EI':   args.g_EI * 1e-9,  # CLI input in nA; store in A
    }

    print(f"Building network: nu_ext={args.nu_ext} Hz, g_EI={args.g_EI} nA, "
          f"seed={args.seed}")
    net_objs = build_network(params, seed=args.seed)

    print(f"Running {args.t_sim} s simulation ...")
    net_objs['net'].run(args.t_sim * second)

    validation, passed = run_validation(
        net_objs, params, args.t_sim, args.seed, args.results_dir)

    if passed:
        h5_path = os.path.join(args.results_dir, 'baseline_network.h5')
        save_baseline(h5_path, params, net_objs, validation, seed=args.seed)
        print(f"All quantitative checks PASSED. Baseline saved to:\n  {h5_path}")
        sys.exit(0)
    else:
        report_path = os.path.join(args.results_dir, 'validation_report.txt')
        os.makedirs(args.results_dir, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(f"nu_ext={args.nu_ext} Hz  g_EI={args.g_EI} nA  seed={args.seed}\n")
            f.write(f"CV_ISI={validation['mean_CV_ISI']:.4f}  "
                    f"pairwise_r={validation['mean_pairwise_corr']:.4f}  "
                    f"rate_ratio="
                    f"{validation['mean_rate_I']/(validation['mean_rate_E'] or 1):.2f}\n")
        print(f"One or more quantitative checks FAILED. "
              f"Report written to:\n  {report_path}")
        print("Re-tune (nu_ext, g_EI) and re-run.")
        sys.exit(1)


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Smoke test on a small network**

This does NOT save an HDF5 (the tiny network will almost certainly fail rate/CV checks,
which is expected and correct behaviour).

```bash
python circuit/run_baseline.py --nu_ext 15 --g_EI 0.24 \
       --t_sim 1.0 --results_dir /tmp/hebbian_smoke_test
```

Expected output: prints check results and either "PASSED" or "One or more checks FAILED."
No crash. Figures appear in `/tmp/hebbian_smoke_test/figures/`.

Verify figures exist:
```bash
ls /tmp/hebbian_smoke_test/figures/
```

Expected: `raster.png  firing_rate_hist.png  isi_dist.png  pairwise_corr.png  power_spectrum.png  weight_hists.png`

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS (the new plotting/validation code has no new unit tests — it was
exercised by the smoke test above).

- [ ] **Step 5: Commit**

```bash
git add circuit/run_baseline.py
git commit -m "feat: add validation workflow, plotting functions, and CLI to run_part1.py"
```

---

## Task 8: `tune_part1.py` — grid search, CSV, and heatmap

**Files:**
- Create: `circuit/grid_search.py`

- [ ] **Step 1: Write `tune_part1.py`**

```python
# circuit/grid_search.py

"""
2D parameter grid search to find the AI-regime operating point for Part 1.

Scans (nu_ext, g_EI) over a 5×6 grid (30 points), running a 1 s simulation
per point. Saves results to CSV and a two-panel heatmap.

Usage:
    python circuit/grid_search.py
    # → inspect circuit/results/tuning_heatmap.png
    # → pick (nu_ext, g_EI) from the overlap region where:
    #       mean_rate_E in [2, 10] Hz  AND  mean_CV_ISI in [0.8, 1.2]
    # → run: python circuit/run_baseline.py --nu_ext <value> --g_EI <value>
"""

import os
import csv
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import second

from circuit.network import build_network, DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi, _extract_spike_trains

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# ------------------------------------------------------------------
# Grid definition
# ------------------------------------------------------------------
# nu_ext_grid: 5 values from 5 to 25 Hz
# g_EI_scale:  6 scale factors from 0.5 to 2.0 × w_mean_EE
# Total: 30 grid points
# ------------------------------------------------------------------
NU_EXT_VALS = np.linspace(5, 25, 5)           # Hz
G_EI_SCALES = np.linspace(0.5, 2.0, 6)        # × w_mean_EE
W_MEAN_EE   = DEFAULT_PARAMS['w_mean_EE']      # A
G_EI_VALS   = G_EI_SCALES * W_MEAN_EE         # A

# AI-regime target boundaries (for contour overlays and pass/fail)
RATE_MIN, RATE_MAX = 2.0, 10.0   # Hz
CV_MIN,   CV_MAX   = 0.8,  1.2


def run_grid_point(nu_ext: float, g_EI: float) -> tuple:
    """
    Build network, run 1 s, return (mean_rate_E_hz, mean_CV_ISI).

    Uses a lower min_spikes threshold (5) for CV-ISI because we only have 1 s;
    neurons at 5 Hz fire ~5 spikes, just enough for a rough CV estimate.
    """
    params = {**DEFAULT_PARAMS, 'nu_ext': nu_ext, 'g_EI': g_EI}
    objs = build_network(params, seed=42)
    objs['net'].run(1.0 * second)

    Ne = params['N_exc']
    mean_rate_E = objs['spike_E'].num_spikes / (Ne * 1.0)

    trains_E = _extract_spike_trains(objs['spike_E'], Ne, 1.0)
    _, mean_cv = compute_cv_isi(trains_E, 0.0, 1.0, min_spikes=5)

    return mean_rate_E, mean_cv


def save_csv(results: list, path: str) -> None:
    """Write grid results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['nu_ext_hz', 'g_EI_nA', 'mean_rate_E_hz', 'mean_CV_ISI'])
        writer.writeheader()
        writer.writerows(results)


def save_heatmap(results: list, path: str) -> None:
    """
    Two-panel heatmap: mean E firing rate (left) and mean CV-ISI (right).
    Contours mark the AI-regime boundaries. The overlap region is the target.
    """
    n_nu  = len(NU_EXT_VALS)
    n_gei = len(G_EI_VALS)

    rate_grid = np.full((n_nu, n_gei), np.nan)
    cv_grid   = np.full((n_nu, n_gei), np.nan)

    for row in results:
        i = np.argmin(np.abs(NU_EXT_VALS - row['nu_ext_hz']))
        j = np.argmin(np.abs(G_EI_VALS / 1e-9 - row['g_EI_nA']))
        rate_grid[i, j] = row['mean_rate_E_hz']
        cv_grid[i, j]   = row['mean_CV_ISI']

    g_ei_nA = G_EI_VALS / 1e-9  # convert A → nA for axis labels

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: mean E firing rate
    ax = axes[0]
    im = ax.imshow(rate_grid, origin='lower', aspect='auto',
                   extent=[g_ei_nA[0], g_ei_nA[-1], NU_EXT_VALS[0], NU_EXT_VALS[-1]],
                   vmin=0, vmax=30, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Mean E rate (Hz)')
    cs = ax.contour(g_ei_nA, NU_EXT_VALS, rate_grid,
                    levels=[RATE_MIN, RATE_MAX], colors='white', linewidths=1.5)
    ax.clabel(cs, fmt='%.0f Hz')
    ax.set_xlabel('g_EI (nA)')
    ax.set_ylabel('nu_ext (Hz)')
    ax.set_title('Mean E firing rate\nWhite contours: 2 Hz, 10 Hz (AI band)')

    # Right panel: mean CV-ISI
    ax = axes[1]
    im = ax.imshow(cv_grid, origin='lower', aspect='auto',
                   extent=[g_ei_nA[0], g_ei_nA[-1], NU_EXT_VALS[0], NU_EXT_VALS[-1]],
                   vmin=0, vmax=2, cmap='plasma')
    plt.colorbar(im, ax=ax, label='Mean CV-ISI')
    cs = ax.contour(g_ei_nA, NU_EXT_VALS, cv_grid,
                    levels=[CV_MIN, CV_MAX], colors='white', linewidths=1.5)
    ax.clabel(cs, fmt='%.1f')
    ax.set_xlabel('g_EI (nA)')
    ax.set_ylabel('nu_ext (Hz)')
    ax.set_title('Mean CV-ISI\nWhite contours: 0.8, 1.2 (AI band)\nTarget: overlap with left panel')

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Heatmap saved to {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []
    total = len(NU_EXT_VALS) * len(G_EI_VALS)
    done  = 0

    print(f"Running {total} grid points ({len(NU_EXT_VALS)} nu_ext × "
          f"{len(G_EI_VALS)} g_EI)...")
    print(f"{'nu_ext (Hz)':>12} {'g_EI (nA)':>10} {'rate_E (Hz)':>12} "
          f"{'CV-ISI':>8}")
    print('-' * 46)

    t0 = time.time()
    for nu_ext in NU_EXT_VALS:
        for g_EI in G_EI_VALS:
            rate, cv = run_grid_point(nu_ext, g_EI)
            results.append({
                'nu_ext_hz':      round(float(nu_ext), 2),
                'g_EI_nA':        round(float(g_EI / 1e-9), 4),
                'mean_rate_E_hz': round(float(rate), 3),
                'mean_CV_ISI':    round(float(cv) if not np.isnan(cv) else -1, 4),
            })
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"{nu_ext:>12.1f} {g_EI/1e-9:>10.4f} {rate:>12.3f} "
                  f"{cv:>8.3f}  [{done}/{total}  ETA {eta:.0f}s]")

    csv_path = os.path.join(RESULTS_DIR, 'tuning_results.csv')
    save_csv(results, csv_path)
    print(f"\nCSV saved to {csv_path}")

    heatmap_path = os.path.join(RESULTS_DIR, 'tuning_heatmap.png')
    save_heatmap(results, heatmap_path)

    print('\nNext step: inspect tuning_heatmap.png, find the overlap region '
          'where rate ∈ [2,10] Hz AND CV ∈ [0.8,1.2], then run:')
    print('  python circuit/run_baseline.py --nu_ext <value> --g_EI <value>')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke test on a 2×2 grid**

Edit `tune_part1.py` temporarily — override the grid at the top of `main()` before
the loop, just to check it runs without errors:

```python
    # SMOKE TEST ONLY — remove after verifying
    NU_EXT_VALS_test = np.array([10.0, 20.0])
    G_EI_VALS_test   = np.array([0.12e-9, 0.24e-9])
```

Then run:

```bash
python circuit/grid_search.py
```

Expected: prints 4 rows of results, saves CSV and heatmap to `circuit/results/`.
Verify:

```bash
ls circuit/results/
# → tuning_results.csv  tuning_heatmap.png
head circuit/results/tuning_results.csv
# → nu_ext_hz,g_EI_nA,mean_rate_E_hz,mean_CV_ISI
```

- [ ] **Step 3: Revert smoke-test override**

Remove the two `_test` lines you added to `main()`. The script should use the
module-level `NU_EXT_VALS` and `G_EI_VALS` constants again.

- [ ] **Step 4: Run full test suite (no regressions)**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add circuit/grid_search.py
git commit -m "feat: add tune_part1.py with 30-point grid search and heatmap"
```

---

## Task 9: Integration run — tune then validate

This task produces the final `baseline_network.h5` that Part 2 will load.

- [ ] **Step 1: Run the full parameter grid search**

```bash
python circuit/grid_search.py
```

Expected runtime: 5–20 min on a laptop. Prints a table of 30 results.

- [ ] **Step 2: Inspect the heatmap and choose the operating point**

```bash
open circuit/results/tuning_heatmap.png   # macOS
```

Find the overlap region where both panels show values inside the AI band
(rate 2–10 Hz AND CV 0.8–1.2). Pick a point comfortably inside the overlap —
not near either boundary. Record the `(nu_ext, g_EI)` values from the printed table.

If there is no overlap region:
- If all rates are > 10 Hz: reduce `nu_ext` or increase `g_EI`
- If all rates are < 2 Hz: increase `nu_ext` or decrease `g_EI`
- If CV is everywhere < 0.8: the network is too regular — decrease `g_EI`
- Widen the grid ranges in `tune_part1.py` and re-run

- [ ] **Step 3: Run full 5 s validation**

Replace `<nu_ext>` and `<g_EI>` with your chosen values:

```bash
python circuit/run_baseline.py --nu_ext <nu_ext> --g_EI <g_EI>
```

Example (adjust based on your heatmap):

```bash
python circuit/run_baseline.py --nu_ext 15.0 --g_EI 0.24
```

Expected stdout (values will differ):
```
Building network: nu_ext=15.0 Hz, g_EI=0.24 nA, seed=42
Running 5.0 s simulation ...

==================================================
              Validation results
==================================================
Check 3 (CV-ISI):          0.943   PASS  [target: 0.8–1.2]
Check 4 (pairwise r):      0.0183  PASS  [target: <0.05]
Check 7 (I/E rate ratio):  2.41    PASS  [target: 2–3×]
Mean E rate:               5.21 Hz
Mean I rate:               12.57 Hz
==================================================

Figures saved to circuit/results/figures/
...
All quantitative checks PASSED. Baseline saved to:
  circuit/results/baseline_network.h5
```

- [ ] **Step 4: Inspect visual checks manually**

```bash
open circuit/results/figures/raster.png
open circuit/results/figures/firing_rate_hist.png
open circuit/results/figures/power_spectrum.png
open circuit/results/figures/weight_hists.png
```

Confirm:
- Raster: sparse, irregular, no bands of synchronous bursts
- Firing rate histogram: right-skewed, approximately log-normal
- Power spectrum: no sharp peaks; monotonically decreasing
- Weight histograms: log-normal shape (right-skewed, all positive)

If any look wrong, return to tuning.

- [ ] **Step 5: Confirm the handoff file exists**

```bash
ls -lh circuit/results/baseline_network.h5
```

Expected: file exists, size ~1–5 MB.

Quick integrity check:

```bash
python -c "
import h5py, scipy.sparse, numpy as np
with h5py.File('circuit/results/baseline_network.h5', 'r') as f:
    print('N_exc:', f['network/N_exc'][()])
    print('N_inh:', f['network/N_inh'][()])
    print('mean_rate_E:', f['validation/mean_rate_E'][()], 'Hz')
    print('mean_CV_ISI:', f['validation/mean_CV_ISI'][()])
    d = f['weights/W_EE/data'][:]
    r = f['weights/W_EE/row'][:]
    c = f['weights/W_EE/col'][:]
    s = f['weights/W_EE/shape'][:]
    W = scipy.sparse.coo_matrix((d, (r, c)), shape=s)
    print('W_EE shape:', W.shape, '  nnz:', W.nnz)
"
```

Expected output (values vary):
```
N_exc: 800
N_inh: 200
mean_rate_E: 5.21 Hz
mean_CV_ISI: 0.943
W_EE shape: (800, 800)   nnz: 63847
```

- [ ] **Step 6: Final commit**

```bash
git add circuit/results/tuning_results.csv circuit/results/tuning_heatmap.png
git add circuit/results/baseline_network.h5
git commit -m "data: Part 1 tuning results and validated baseline network (seed=42)"
```

> Note: normally large data files should not go in git. For this project, the HDF5
> baseline is the reproducibility artifact that Part 2 depends on — committing it once
> is appropriate. If it grows large in later parts, move to git-lfs or store separately.

---

## Self-Review

**Spec coverage check:**

| Spec section | Covered by task(s) |
|---|---|
| LIF neuron equations + parameters | Task 2–3 |
| Network topology (N=1000, 80/20, p=0.1, no self-connections) | Task 3 |
| Current-based exponential synapses | Task 3 |
| Log-normal weight init, units in nA | Task 3 |
| External Poisson drive | Task 4 |
| `build_network(params, seed)` interface | Tasks 2–4 |
| `tune_part1.py` grid search + CSV + heatmap | Task 8 |
| `run_part1.py` CLI with `--nu_ext`, `--g_EI` | Task 7 |
| 7 validation checks (3 auto, 4 visual) | Tasks 5, 7 |
| HDF5 save gated on quantitative pass | Tasks 6–7 |
| COO sparse weight storage | Task 6 |
| All figures saved to `results/figures/` | Task 7 |
| `validation_report.txt` on failure | Task 7 |
| Integration: tune → validate → h5 | Task 9 |

All spec requirements covered. No gaps found.

**Placeholder scan:** No TBD, TODO, or "similar to above" found.

**Type consistency:**
- `build_network` returns dict with keys: `exc, inh, syn_EE, syn_EI, syn_IE, syn_II, drive_E, drive_I, spike_E, spike_I, net` — consistent across Tasks 2–9.
- `compute_cv_isi` signature: `(spike_trains: dict, t_start: float, t_end: float, min_spikes: int) → (dict, float)` — consistent between Task 5 (implementation) and Task 8 (caller).
- `save_baseline` signature: `(path: str, params: dict, net_objs: dict, validation: dict, seed: int) → None` — consistent between Task 6 (implementation) and Task 7 (caller in `run_validation`).
- `_extract_spike_trains(monitor, n_neurons, t_sim) → dict` — defined in Task 7, called in Tasks 7 and 8.
