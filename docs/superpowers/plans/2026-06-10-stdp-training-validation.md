# Part 2 Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Part 2 infrastructure (STDP synapses on E→E, P/X/S pool seeding,
50-neuron task input, trial loop with burn-in, HDF5 snapshot I/O, monitoring/abort
checks) and validate it with a 104-trial run in both the seeded and control
conditions.

**Architecture:** Four new modules under `part2/`: `network_part2.py` (Brian2
object construction — loads the Part 1 baseline, adds STDP synapses and task
input neurons), `task.py` (pure-numpy tuning curves and trial sequences, no
Brian2 dependency), `snapshot.py` (HDF5 read/write for `/snapshots/epoch_N/...`
and `/monitoring/...`), and `run_part2.py` (orchestrates burn-in, the trial loop,
and snapshots; CLI entry point). Each module is built and unit-tested with a
small synthetic network before being wired together.

**Tech Stack:** Python, Brian2 (numpy backend for tests, cython for the full
validation run), h5py, numpy, pytest. Builds on `circuit/network.py`'s
`build_network()`, `_lognormal_weights()`, and `DEFAULT_PARAMS`, and
`circuit/run_baseline.py`'s `compute_cv_isi()`.

---

## Reference: spec

Full design at `docs/superpowers/specs/2026-06-10-stdp-center-out-task-design.md`.
Key numbers used throughout this plan:

- Pools (E neuron indices): P = [0, 350), X = [350, 700), S = [700, 800)
- STDP: `tau_plus = tau_minus = 20 ms`, `w_max = 0.24 nA`, `A_plus = 0.0024 nA`,
  `A_minus = 0.00252 nA`
- Task input: 50 neurons, 8 directions (45° spacing), `r_max = 100 Hz`,
  `r_background = 2 Hz`, exec amplification ×1.5
- Trial timing: prep 500 ms, exec 500 ms, ITI 200 ms (1.2 s/trial)
- Burn-in: 15 s, STDP frozen, input at background rate
- Phase A: 104 trials (13/direction), snapshot epochs `{0, 50, 100}`, both
  conditions (`p_cross = 0.2` seeded, `p_cross = 1.0` control)
- Abort: `mean_rate_E > 30 Hz` or `frac_w_max > 0.5`

---

## Task 1: `plasticity/stdp_network.py` — params, pool rescaling, baseline loading

**Files:**
- Create: `part2/__init__.py` (empty)
- Create: `plasticity/stdp_network.py`
- Test: `tests/test_network_part2.py`

- [ ] **Step 1: Create the package `__init__.py`**

```bash
mkdir -p part2
touch part2/__init__.py
```

- [ ] **Step 2: Write the failing test for `apply_pool_rescaling`**

Create `tests/test_network_part2.py`:

```python
# tests/test_network_part2.py

import os

import numpy as np
import pytest
import h5py
from brian2 import start_scope, second, amp

from circuit.network import build_network, DEFAULT_PARAMS
from plasticity.stdp_network import (
    DEFAULT_PARAMS_PLASTICITY,
    apply_pool_rescaling,
    load_baseline,
)


def test_apply_pool_rescaling_cross_pool_only():
    # P = {0, 1}, X = {2, 3}, no S.
    # (0,1)=P->P, (0,2)=P->X, (2,0)=X->P, (2,3)=X->X
    i = np.array([0, 0, 2, 2])
    j = np.array([1, 2, 0, 3])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    w_new = apply_pool_rescaling(i, j, w, p_cross=0.2, P_size=2, X_size=2)
    np.testing.assert_allclose(w_new, [1.0, 0.2, 0.2, 1.0])


def test_apply_pool_rescaling_shared_pool_unchanged():
    # P = {0, 1}, X = {2, 3}, S = {4}.
    # (4,0)=S->P, (0,4)=P->S, (2,4)=X->S — none touch the P<->X cross term
    i = np.array([4, 0, 2])
    j = np.array([0, 4, 4])
    w = np.array([1.0, 1.0, 1.0])
    w_new = apply_pool_rescaling(i, j, w, p_cross=0.2, P_size=2, X_size=2)
    np.testing.assert_allclose(w_new, [1.0, 1.0, 1.0])


def test_apply_pool_rescaling_does_not_mutate_input():
    i = np.array([0, 0])
    j = np.array([1, 2])
    w = np.array([1.0, 1.0])
    w_orig = w.copy()
    apply_pool_rescaling(i, j, w, p_cross=0.2, P_size=2, X_size=2)
    np.testing.assert_array_equal(w, w_orig)
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
PYTHONPATH=. python -m pytest tests/test_network_part2.py -v
```

Expected: `ModuleNotFoundError: No module named 'plasticity.stdp_network'` (or
`ImportError: cannot import name 'apply_pool_rescaling'`).

- [ ] **Step 4: Implement `DEFAULT_PARAMS_PLASTICITY` and `apply_pool_rescaling`**

Create `plasticity/stdp_network.py`:

```python
# plasticity/stdp_network.py

"""
Part 2 network factory: loads the Part 1 baseline, adds pair-based STDP on
E->E synapses (Song, Miller & Abbott 2000), applies P/X/S pool rescaling, and
adds 50 task-input neurons connected to both E and I populations.

See docs/superpowers/specs/2026-06-10-stdp-center-out-task-design.md for the full
design and parameter justifications.
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
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
PYTHONPATH=. python -m pytest tests/test_network_part2.py -v
```

Expected: the three `apply_pool_rescaling` tests PASS, the `load_baseline`
import still fails (not yet defined) — confirm by checking the failure is now
an `ImportError`/`AttributeError` for `load_baseline`, not for
`apply_pool_rescaling`.

- [ ] **Step 6: Write the failing test for `load_baseline`**

Append to `tests/test_network_part2.py`:

```python
BASELINE_H5 = os.path.join(
    os.path.dirname(__file__), '..', 'part1', 'results', 'baseline_network.h5')


def test_load_baseline_matches_saved_weights():
    start_scope()
    net_objs = load_baseline(BASELINE_H5, DEFAULT_PARAMS, seed=42)

    with h5py.File(BASELINE_H5, 'r') as f:
        saved_w = f['weights/W_EE/data'][:]

    actual_w = np.array(net_objs['syn_EE'].w[:] / amp, dtype=np.float32)
    np.testing.assert_allclose(actual_w, saved_w, rtol=1e-5)


def test_load_baseline_returns_expected_keys():
    start_scope()
    net_objs = load_baseline(BASELINE_H5, DEFAULT_PARAMS, seed=42)
    expected = {
        'exc', 'inh', 'syn_EE', 'syn_EI', 'syn_IE', 'syn_II',
        'drive_E', 'drive_I', 'spike_E', 'spike_I', 'net',
    }
    assert expected.issubset(net_objs.keys())


def test_load_baseline_raises_on_mismatched_params():
    start_scope()
    bad_params = {**DEFAULT_PARAMS, 'N_exc': 10}
    with pytest.raises(ValueError):
        load_baseline(BASELINE_H5, bad_params, seed=42)
```

Add `start_scope` to the existing `from brian2 import ...` line at the top of
the test file (it's already imported in Step 2's version above — if not,
add it).

- [ ] **Step 7: Run the test to verify it fails**

```bash
PYTHONPATH=. python -m pytest tests/test_network_part2.py -v -k load_baseline
```

Expected: `ImportError: cannot import name 'load_baseline'`.

- [ ] **Step 8: Implement `load_baseline`**

Append to `plasticity/stdp_network.py`:

```python
def load_baseline(h5_path, params, seed=42):
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
```

- [ ] **Step 9: Run the test to verify it passes**

```bash
PYTHONPATH=. python -m pytest tests/test_network_part2.py -v
```

Expected: all 6 tests PASS (3 from Step 2/4, 3 from Step 6/8).

- [ ] **Step 10: Commit**

```bash
git add part2/__init__.py plasticity/stdp_network.py tests/test_network_part2.py
git commit -m "feat(part2): pool rescaling + Part 1 baseline loading"
```

---

## Task 2: `plasticity/stdp_network.py` — STDP synapses + task input neurons

**Files:**
- Modify: `plasticity/stdp_network.py`
- Test: `tests/test_network_part2.py`

All tests in this task use a small synthetic network (built directly via
`build_network()`, not `load_baseline()`) so they run in well under a
second.

- [ ] **Step 1: Write the failing tests for `build_stdp_network`**

Append to `tests/test_network_part2.py` (add `Hz` to the brian2 import line:
`from brian2 import start_scope, second, amp, Hz`):

```python
from plasticity.stdp_network import (
    DEFAULT_PARAMS_PLASTICITY,
    apply_pool_rescaling,
    load_baseline,
    build_stdp_network,
)


def _small_params(**overrides):
    """20 E + 5 I neurons, P=[0,8), X=[8,16), S=[16,20)."""
    return {
        **DEFAULT_PARAMS, **DEFAULT_PARAMS_PLASTICITY,
        'N_exc': 20, 'N_inh': 5,
        'P_size': 8, 'X_size': 8,
        **overrides,
    }


def test_build_stdp_network_preserves_connectivity():
    start_scope()
    small = _small_params()
    net_objs = build_network(small, seed=1)
    i_before = np.array(net_objs['syn_EE'].i[:])
    j_before = np.array(net_objs['syn_EE'].j[:])

    result = build_stdp_network(net_objs, small, p_cross=0.2, seed=1)
    syn = result['syn_EE']
    np.testing.assert_array_equal(np.array(syn.i[:]), i_before)
    np.testing.assert_array_equal(np.array(syn.j[:]), j_before)


def test_build_stdp_network_applies_pool_rescaling():
    start_scope()
    small = _small_params()
    net_objs = build_network(small, seed=1)
    i_arr = np.array(net_objs['syn_EE'].i[:])
    j_arr = np.array(net_objs['syn_EE'].j[:])
    w_before = np.array(net_objs['syn_EE'].w[:] / amp)

    result = build_stdp_network(net_objs, small, p_cross=0.2, seed=1)
    w_after = np.array(result['syn_EE'].w[:] / amp)

    expected = apply_pool_rescaling(i_arr, j_arr, w_before, p_cross=0.2,
                                     P_size=small['P_size'], X_size=small['X_size'])
    np.testing.assert_allclose(w_after, expected, rtol=1e-6)


def test_build_stdp_network_has_stdp_state_variables():
    start_scope()
    small = _small_params()
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)
    syn = result['syn_EE']
    assert hasattr(syn, 'plastic')
    assert hasattr(syn, 'x_pre')
    assert hasattr(syn, 'x_post')
    assert np.all(np.array(syn.plastic[:]) == 1)


def test_build_stdp_network_adds_input_neurons():
    start_scope()
    small = _small_params()
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)

    assert len(result['input_group']) == small['n_input']
    assert len(result['syn_input_E']) > 0
    assert len(result['syn_input_I']) > 0

    w_E = np.array(result['syn_input_E'].w[:] / amp)
    w_I = np.array(result['syn_input_I'].w[:] / amp)
    assert np.all(w_E > 0)
    assert np.all(w_I > 0)


def test_build_stdp_network_input_rates_default_to_background():
    start_scope()
    small = _small_params()
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)
    rates = np.array(result['input_group'].rates[:] / Hz)
    np.testing.assert_allclose(rates, small['r_background'])


def test_stdp_plastic_zero_freezes_weights():
    start_scope()
    small = _small_params(nu_ext=1000.0)
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)
    syn = result['syn_EE']
    syn.plastic = 0

    w_before = np.array(syn.w[:] / amp).copy()
    result['net'].run(0.1 * second)
    w_after = np.array(syn.w[:] / amp)
    np.testing.assert_allclose(w_before, w_after)


def test_stdp_plastic_one_changes_weights():
    start_scope()
    small = _small_params(nu_ext=1000.0)
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)
    syn = result['syn_EE']

    w_before = np.array(syn.w[:] / amp).copy()
    result['net'].run(0.5 * second)
    w_after = np.array(syn.w[:] / amp)
    assert not np.allclose(w_before, w_after), \
        "STDP did not change any weights in 0.5 s"


def test_stdp_weights_clipped_to_w_max():
    start_scope()
    small = _small_params(nu_ext=1000.0)
    net_objs = build_network(small, seed=1)
    result = build_stdp_network(net_objs, small, p_cross=1.0, seed=1)
    syn = result['syn_EE']

    result['net'].run(1.0 * second)
    w_after = np.array(syn.w[:] / amp)
    assert np.all(w_after >= 0.0)
    assert np.all(w_after <= small['w_max'] * 1.0000001)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=. python -m pytest tests/test_network_part2.py -v -k stdp_network
```

Expected: `ImportError: cannot import name 'build_stdp_network'`.

- [ ] **Step 3: Implement `build_stdp_network`**

First, extend the brian2 import at the top of `plasticity/stdp_network.py`:

```python
from brian2 import (
    Synapses, PoissonGroup, SpikeMonitor, Network,
    second, amp, Hz,
)
```

(replace the existing `from brian2 import (...)` line from Task 1 with this).

Then append:

```python
def build_stdp_network(net_objs, params, p_cross, seed=42):
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
    stdp_eqs = '''
    w : amp
    plastic : 1 (shared)
    dx_pre/dt  = -x_pre  / tau_plus  : 1 (event-driven)
    dx_post/dt = -x_post / tau_minus : 1 (event-driven)
    '''
    on_pre_eqs = '''
    I_exc_post += w
    x_pre += 1
    w = clip(w - plastic * A_minus * x_post, 0*amp, w_max)
    '''
    on_post_eqs = '''
    x_post += 1
    w = clip(w + plastic * A_plus * x_pre, 0*amp, w_max)
    '''

    syn_EE_stdp = Synapses(
        exc, exc, stdp_eqs,
        on_pre=on_pre_eqs, on_post=on_post_eqs,
        namespace=stdp_ns, method='euler', name='syn_EE_stdp')
    syn_EE_stdp.connect(i=i_arr, j=j_arr)
    syn_EE_stdp.w = w_rescaled * amp
    syn_EE_stdp.plastic = 1
    syn_EE_stdp.x_pre = 0
    syn_EE_stdp.x_post = 0

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

    net = Network(
        exc, inh,
        syn_EE_stdp, net_objs['syn_EI'], net_objs['syn_IE'], net_objs['syn_II'],
        net_objs['drive_E'], net_objs['drive_I'],
        net_objs['spike_E'], net_objs['spike_I'],
        input_group, syn_input_E, syn_input_I, spike_input,
    )

    result = dict(net_objs)
    result['syn_EE'] = syn_EE_stdp
    result['input_group'] = input_group
    result['syn_input_E'] = syn_input_E
    result['syn_input_I'] = syn_input_I
    result['spike_input'] = spike_input
    result['net'] = net
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH=. python -m pytest tests/test_network_part2.py -v
```

Expected: all tests PASS (the 6 from Task 1 plus 8 new ones). The
`test_stdp_plastic_one_changes_weights` and `test_stdp_weights_clipped_to_w_max`
tests use `nu_ext=1000.0` to guarantee spikes within the short run — if either
is flaky (no weight change in 0.5 s), increase to `nu_ext=2000.0` or the run
duration to `1.0 * second`.

- [ ] **Step 5: Commit**

```bash
git add plasticity/stdp_network.py tests/test_network_part2.py
git commit -m "feat(part2): STDP synapses on E->E + task input neurons"
```

---

## Task 3: `plasticity/center_out_task.py` — tuning curves and trial sequences

**Files:**
- Create: `plasticity/center_out_task.py`
- Test: `tests/test_task.py`

Pure numpy, no Brian2 dependency — this module only computes input rates and
trial orderings (spec 2.3).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_task.py`:

```python
# tests/test_task.py

import numpy as np
import pytest

from plasticity.center_out_task import (
    assign_preferred_directions,
    rates_for_phase,
    generate_trial_sequence,
    generate_test_trial_sequence,
)


def test_assign_preferred_directions_length_and_unique_count():
    theta_i = assign_preferred_directions(n_input=50, n_directions=8)
    assert len(theta_i) == 50
    assert len(np.unique(theta_i)) == 8


def test_assign_preferred_directions_counts_balanced():
    theta_i = assign_preferred_directions(n_input=50, n_directions=8)
    _, counts = np.unique(theta_i, return_counts=True)
    assert sorted(counts.tolist()) == [6, 6, 6, 6, 6, 6, 7, 7]


def test_assign_preferred_directions_spacing_is_45_degrees():
    theta_i = assign_preferred_directions(n_input=50, n_directions=8)
    directions = np.unique(theta_i)
    spacing = np.diff(directions)
    np.testing.assert_allclose(spacing, np.pi / 4)


def test_rates_for_phase_prep_peak_and_orthogonal():
    theta_i = np.array([0.0, np.pi / 2, np.pi])
    rates = rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='prep')
    np.testing.assert_allclose(rates, [100.0, 0.0, 0.0], atol=1e-10)


def test_rates_for_phase_exec_amplifies_prep_by_1_5():
    theta_i = np.array([0.0, np.pi / 4])
    prep = rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='prep')
    exec_rates = rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='exec')
    np.testing.assert_allclose(exec_rates, prep * 1.5)


def test_rates_for_phase_iti_is_background_regardless_of_cue():
    theta_i = np.array([0.0, np.pi / 2, np.pi])
    rates = rates_for_phase(theta_cue=1.23, theta_i=theta_i, phase='iti')
    np.testing.assert_allclose(rates, [2.0, 2.0, 2.0])


def test_rates_for_phase_invalid_phase_raises():
    theta_i = np.array([0.0])
    with pytest.raises(ValueError):
        rates_for_phase(theta_cue=0.0, theta_i=theta_i, phase='bogus')


def test_generate_trial_sequence_reproducible_and_balanced():
    seq1 = generate_trial_sequence(n_per_direction=13, n_directions=8, seed=42)
    seq2 = generate_trial_sequence(n_per_direction=13, n_directions=8, seed=42)
    assert len(seq1) == 13 * 8
    np.testing.assert_array_equal(seq1, seq2)
    _, counts = np.unique(seq1, return_counts=True)
    np.testing.assert_array_equal(counts, np.full(8, 13))


def test_generate_trial_sequence_different_seeds_give_different_order():
    seq1 = generate_trial_sequence(n_per_direction=13, n_directions=8, seed=42)
    seq2 = generate_trial_sequence(n_per_direction=13, n_directions=8, seed=1)
    assert not np.array_equal(seq1, seq2)


def test_generate_test_trial_sequence_default_length_and_balance():
    seq = generate_test_trial_sequence()
    assert len(seq) == 40
    _, counts = np.unique(seq, return_counts=True)
    np.testing.assert_array_equal(counts, np.full(8, 5))
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=. python -m pytest tests/test_task.py -v
```

Expected: `ModuleNotFoundError: No module named 'plasticity.center_out_task'`.

- [ ] **Step 3: Implement `plasticity/center_out_task.py`**

```python
# plasticity/center_out_task.py

"""
Task structure for the 8-direction center-out reaching task (spec section 2.3).

Pure numpy — no Brian2 dependency. run_part2.py converts the rate arrays
returned here to Brian2 Quantities (`* Hz`) before assigning to the input
PoissonGroup.
"""

import numpy as np


def assign_preferred_directions(n_input=50, n_directions=8):
    """
    Assign each of n_input task-input neurons a preferred direction theta_i
    (radians), evenly spaced around the circle.

    With n_input=50, n_directions=8: 50 = 6*8 + 2, so 2 directions get 7
    neurons and 6 directions get 6 neurons (spec: "6-7 neurons per direction").
    """
    base = n_input // n_directions
    remainder = n_input % n_directions
    counts = np.array([base + 1] * remainder + [base] * (n_directions - remainder))
    directions = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    return np.repeat(directions, counts)


def rates_for_phase(theta_cue, theta_i, phase, r_max=100.0, r_background=2.0,
                     exec_amplification=1.5):
    """
    Firing rates (Hz) for each input neuron during one trial phase.

    - 'prep': half-wave rectified cosine tuning curve (Georgopoulos et al.
      1982): r_i = r_max * max(0, cos(theta_cue - theta_i))
    - 'exec': the prep tuning curve scaled by exec_amplification (stronger
      drive during movement)
    - 'iti': flat r_background for every neuron, independent of theta_cue

    Returns an array the same shape as theta_i.
    """
    if phase == 'iti':
        return np.full_like(theta_i, r_background, dtype=np.float64)

    tuning = r_max * np.maximum(0.0, np.cos(theta_cue - theta_i))
    if phase == 'prep':
        return tuning
    elif phase == 'exec':
        return tuning * exec_amplification
    else:
        raise ValueError(f"Unknown phase '{phase}', expected 'prep', 'exec', or 'iti'")


def generate_trial_sequence(n_per_direction, n_directions=8, seed=42):
    """
    A shuffled sequence of direction indices in [0, n_directions), with
    exactly n_per_direction trials per direction. Reproducible for a given
    seed (spec 2.3: "randomly interleaved").
    """
    rng = np.random.default_rng(seed)
    sequence = np.repeat(np.arange(n_directions), n_per_direction)
    rng.shuffle(sequence)
    return sequence


def generate_test_trial_sequence(n_per_direction=5, n_directions=8, seed=12345):
    """
    The fixed pseudorandom test-trial sequence used for every snapshot (spec
    2.4: "5 per direction, fixed pseudorandom order"). A different seed from
    generate_trial_sequence's training default (42) so test trials are not
    accidentally a prefix of the training sequence.
    """
    return generate_trial_sequence(n_per_direction, n_directions, seed)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH=. python -m pytest tests/test_task.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add plasticity/center_out_task.py tests/test_task.py
git commit -m "feat(part2): tuning curves and trial sequences for center-out task"
```

---

## Task 4: `plasticity/snapshot.py` — HDF5 snapshot and monitoring I/O

**Files:**
- Create: `plasticity/snapshot.py`
- Test: `tests/test_snapshot.py`

Follows the COO convention from `circuit/run_baseline.py`'s `save_baseline()`:
`row` = postsynaptic index (`syn.j`), `col` = presynaptic index (`syn.i`),
weights as float32 in amps (spec section 6).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_snapshot.py`:

```python
# tests/test_snapshot.py

import h5py
import numpy as np
import pytest

from plasticity.stdp_network import DEFAULT_PARAMS_PLASTICITY
from plasticity.snapshot import (
    save_snapshot, load_snapshot, load_monitoring,
    copy_baseline_provenance, save_training_params,
)


def _dummy_snapshot_data(epoch, n_exc=20):
    rng = np.random.default_rng(epoch + 1)
    n_syn = 30
    W_EE_coo = {
        'data': rng.uniform(0, 0.24e-9, size=n_syn).astype(np.float32),
        'row': rng.integers(0, n_exc, size=n_syn).astype(np.int32),
        'col': rng.integers(0, n_exc, size=n_syn).astype(np.int32),
        'shape': np.array([n_exc, n_exc], dtype=np.int32),
    }
    spike_data = {
        'spike_times_ms': np.array([1.0, 2.5, 100.0], dtype=np.float32),
        'spike_neuron_idx': np.array([0, 5, 21], dtype=np.int32),
        'spike_trial_idx': np.array([0, 0, 1], dtype=np.int32),
    }
    trial_labels = np.arange(8) % 8
    monitoring_metrics = {
        'mean_rate_E': 2.5 + epoch * 0.01,
        'mean_w_EE': 0.06e-9,
        'frac_w_max': 0.01,
        'mean_cv_isi': 0.9,
    }
    return W_EE_coo, spike_data, trial_labels, monitoring_metrics


def test_save_and_load_snapshot_round_trip(tmp_path):
    h5_path = str(tmp_path / "test.h5")
    W_EE_coo, spike_data, trial_labels, metrics = _dummy_snapshot_data(epoch=0)

    save_snapshot(h5_path, epoch=0, W_EE_coo=W_EE_coo, spike_data=spike_data,
                   trial_labels=trial_labels, monitoring_metrics=metrics)

    loaded = load_snapshot(h5_path, epoch=0)
    np.testing.assert_array_equal(loaded['W_EE_coo']['data'], W_EE_coo['data'])
    np.testing.assert_array_equal(loaded['W_EE_coo']['row'], W_EE_coo['row'])
    np.testing.assert_array_equal(loaded['W_EE_coo']['col'], W_EE_coo['col'])
    np.testing.assert_array_equal(loaded['W_EE_coo']['shape'], W_EE_coo['shape'])
    np.testing.assert_array_equal(loaded['spike_times_ms'], spike_data['spike_times_ms'])
    np.testing.assert_array_equal(loaded['spike_neuron_idx'], spike_data['spike_neuron_idx'])
    np.testing.assert_array_equal(loaded['spike_trial_idx'], spike_data['spike_trial_idx'])
    np.testing.assert_array_equal(loaded['trial_labels'], trial_labels)


def test_save_snapshot_creates_monitoring_row(tmp_path):
    h5_path = str(tmp_path / "test.h5")
    W_EE_coo, spike_data, trial_labels, metrics = _dummy_snapshot_data(epoch=0)
    save_snapshot(h5_path, epoch=0, W_EE_coo=W_EE_coo, spike_data=spike_data,
                   trial_labels=trial_labels, monitoring_metrics=metrics)

    mon = load_monitoring(h5_path)
    np.testing.assert_array_equal(mon['epochs'], [0])
    np.testing.assert_allclose(mon['mean_rate_E'], [metrics['mean_rate_E']])
    np.testing.assert_allclose(mon['mean_w_EE'], [metrics['mean_w_EE']])
    np.testing.assert_allclose(mon['frac_w_max'], [metrics['frac_w_max']])
    np.testing.assert_allclose(mon['mean_cv_isi'], [metrics['mean_cv_isi']])


def test_save_snapshot_appends_monitoring_across_epochs(tmp_path):
    h5_path = str(tmp_path / "test.h5")
    for epoch in (0, 50):
        W_EE_coo, spike_data, trial_labels, metrics = _dummy_snapshot_data(epoch=epoch)
        save_snapshot(h5_path, epoch=epoch, W_EE_coo=W_EE_coo, spike_data=spike_data,
                       trial_labels=trial_labels, monitoring_metrics=metrics)

    mon = load_monitoring(h5_path)
    np.testing.assert_array_equal(mon['epochs'], [0, 50])
    assert mon['mean_rate_E'].shape == (2,)

    snap0 = load_snapshot(h5_path, epoch=0)
    snap50 = load_snapshot(h5_path, epoch=50)
    assert not np.array_equal(snap0['W_EE_coo']['data'], snap50['W_EE_coo']['data'])


def test_load_snapshot_missing_epoch_raises_keyerror(tmp_path):
    h5_path = str(tmp_path / "test.h5")
    W_EE_coo, spike_data, trial_labels, metrics = _dummy_snapshot_data(epoch=0)
    save_snapshot(h5_path, epoch=0, W_EE_coo=W_EE_coo, spike_data=spike_data,
                   trial_labels=trial_labels, monitoring_metrics=metrics)

    with pytest.raises(KeyError):
        load_snapshot(h5_path, epoch=999)


def test_copy_baseline_provenance_copies_groups(tmp_path):
    baseline_path = str(tmp_path / "baseline.h5")
    with h5py.File(baseline_path, 'w') as f:
        ng = f.create_group('network')
        ng.create_dataset('N_exc', data=20)
        wg = f.create_group('weights')
        eeg = wg.create_group('W_EE')
        eeg.create_dataset('data', data=np.array([1.0, 2.0], dtype=np.float32))
        vg = f.create_group('validation')
        vg.create_dataset('mean_rate_E', data=2.5)

    h5_path = str(tmp_path / "test.h5")
    copy_baseline_provenance(h5_path, baseline_path)

    with h5py.File(h5_path, 'r') as f:
        assert f['network/N_exc'][()] == 20
        np.testing.assert_array_equal(f['weights/W_EE/data'][:], [1.0, 2.0])
        assert f['validation/mean_rate_E'][()] == pytest.approx(2.5)


def test_save_training_params_writes_attrs(tmp_path):
    h5_path = str(tmp_path / "test.h5")
    save_training_params(h5_path, DEFAULT_PARAMS_PLASTICITY, p_cross=0.2, seed=42)

    with h5py.File(h5_path, 'r') as f:
        attrs = f['training_params'].attrs
        assert attrs['p_cross'] == pytest.approx(0.2)
        assert attrs['seed'] == 42
        assert attrs['tau_plus'] == pytest.approx(DEFAULT_PARAMS_PLASTICITY['tau_plus'])
        assert attrs['w_max'] == pytest.approx(DEFAULT_PARAMS_PLASTICITY['w_max'])
        assert attrs['t_burn_in'] == pytest.approx(DEFAULT_PARAMS_PLASTICITY['t_burn_in'])
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=. python -m pytest tests/test_snapshot.py -v
```

Expected: `ModuleNotFoundError: No module named 'plasticity.snapshot'`.

- [ ] **Step 3: Implement `plasticity/snapshot.py`**

```python
# plasticity/snapshot.py

"""
HDF5 read/write for Part 2 training snapshots (spec section 6).

Schema:
    /network, /weights, /validation    — copied from the Part 1 baseline via
        copy_baseline_provenance(), so each output file is self-contained
    /training_params                      — attrs written by save_training_params():
        p_cross, seed, tau_plus, tau_minus, A_plus, A_minus, w_max, n_input,
        r_max, t_burn_in
    /snapshots/epoch_{N}/
        W_EE/{data, row, col, shape}   — COO, row=postsynaptic, col=presynaptic
        spike_times_ms                 — float32, ms within trial
        spike_neuron_idx                — int32, 0..N_exc-1 = E, N_exc.. = input
        spike_trial_idx                  — int32, 0..n_test_trials-1
        trial_labels                     — int32, direction index per test trial
    /monitoring/
        epochs, mean_rate_E, mean_w_EE, frac_w_max, mean_cv_isi  — resizable,
        one row appended per save_snapshot() call
"""

import h5py
import numpy as np


_MONITORING_KEYS = ('mean_rate_E', 'mean_w_EE', 'frac_w_max', 'mean_cv_isi')


def save_snapshot(h5_path, epoch, W_EE_coo, spike_data, trial_labels, monitoring_metrics):
    """
    Append one training snapshot to h5_path (created if it doesn't exist).

    W_EE_coo : dict with 'data' (amps), 'row' (postsynaptic idx), 'col'
        (presynaptic idx), 'shape' — same convention as part1's save_baseline.
    spike_data : dict with 'spike_times_ms', 'spike_neuron_idx', 'spike_trial_idx'.
    trial_labels : array of direction indices (0..n_directions-1), one per
        test trial.
    monitoring_metrics : dict with the four keys in _MONITORING_KEYS.
    """
    with h5py.File(h5_path, 'a') as f:
        grp = f.create_group(f'snapshots/epoch_{epoch}')

        wgrp = grp.create_group('W_EE')
        wgrp.create_dataset('data', data=np.asarray(W_EE_coo['data'], dtype=np.float32))
        wgrp.create_dataset('row', data=np.asarray(W_EE_coo['row'], dtype=np.int32))
        wgrp.create_dataset('col', data=np.asarray(W_EE_coo['col'], dtype=np.int32))
        wgrp.create_dataset('shape', data=np.asarray(W_EE_coo['shape'], dtype=np.int32))

        grp.create_dataset('spike_times_ms',
                            data=np.asarray(spike_data['spike_times_ms'], dtype=np.float32))
        grp.create_dataset('spike_neuron_idx',
                            data=np.asarray(spike_data['spike_neuron_idx'], dtype=np.int32))
        grp.create_dataset('spike_trial_idx',
                            data=np.asarray(spike_data['spike_trial_idx'], dtype=np.int32))
        grp.create_dataset('trial_labels',
                            data=np.asarray(trial_labels, dtype=np.int32))

        _append_monitoring_row(f, epoch, monitoring_metrics)


def _append_monitoring_row(f, epoch, metrics):
    if 'monitoring' not in f:
        mgrp = f.create_group('monitoring')
        mgrp.create_dataset('epochs', data=np.array([epoch], dtype=np.int32),
                             maxshape=(None,))
        for k in _MONITORING_KEYS:
            mgrp.create_dataset(k, data=np.array([metrics[k]], dtype=np.float64),
                                 maxshape=(None,))
        return

    mgrp = f['monitoring']
    n = mgrp['epochs'].shape[0]
    mgrp['epochs'].resize((n + 1,))
    mgrp['epochs'][n] = epoch
    for k in _MONITORING_KEYS:
        mgrp[k].resize((n + 1,))
        mgrp[k][n] = metrics[k]


def load_snapshot(h5_path, epoch):
    """Load one snapshot. Raises KeyError if /snapshots/epoch_{epoch} doesn't exist."""
    with h5py.File(h5_path, 'r') as f:
        grp = f[f'snapshots/epoch_{epoch}']
        return {
            'W_EE_coo': {
                'data':  grp['W_EE/data'][:],
                'row':   grp['W_EE/row'][:],
                'col':   grp['W_EE/col'][:],
                'shape': grp['W_EE/shape'][:],
            },
            'spike_times_ms':   grp['spike_times_ms'][:],
            'spike_neuron_idx': grp['spike_neuron_idx'][:],
            'spike_trial_idx':  grp['spike_trial_idx'][:],
            'trial_labels':     grp['trial_labels'][:],
        }


def load_monitoring(h5_path):
    """Load /monitoring/ as a dict of numpy arrays, keyed by dataset name."""
    with h5py.File(h5_path, 'r') as f:
        mgrp = f['monitoring']
        return {k: mgrp[k][:] for k in mgrp.keys()}


def copy_baseline_provenance(h5_path, baseline_h5_path):
    """
    Copy /network, /weights, /validation from the Part 1 baseline into
    h5_path, so each Part 2 output file is self-contained (spec section 6).
    Call once per file, before any snapshots are saved.
    """
    with h5py.File(baseline_h5_path, 'r') as src, h5py.File(h5_path, 'a') as dst:
        for group_name in ('network', 'weights', 'validation'):
            src.copy(src[group_name], dst, group_name)


def save_training_params(h5_path, params, p_cross, seed):
    """
    Write /training_params attrs (spec section 6): p_cross, STDP params, task
    input params, the trial-sequence seed, and burn-in duration.
    """
    with h5py.File(h5_path, 'a') as f:
        grp = f.require_group('training_params')
        grp.attrs['p_cross'] = float(p_cross)
        grp.attrs['seed'] = int(seed)
        for k in ('tau_plus', 'tau_minus', 'A_plus', 'A_minus', 'w_max',
                  'n_input', 'r_max', 't_burn_in'):
            grp.attrs[k] = float(params[k])
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH=. python -m pytest tests/test_snapshot.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add plasticity/snapshot.py tests/test_snapshot.py
git commit -m "feat(part2): HDF5 snapshot and monitoring I/O"
```

---

## Task 5: `plasticity/train.py` — trial runner and snapshot logic

**Files:**
- Create: `plasticity/train.py`
- Test: `tests/test_run_part2.py`

This task builds the per-trial simulation step (`run_one_trial`), the spike
extraction and metrics used at snapshot time, and `run_snapshot` itself
(freeze STDP, run the 40-trial test sequence, save to HDF5, unfreeze). The
training loop and CLI (`run_condition`, `main`) are Task 6.

All tests use the small synthetic network from Task 2
(`N_exc=20, N_inh=5, P_size=8, X_size=8`) with `nu_ext=1000.0` so neurons spike
copiously within a couple of trials (~2.4 s of simulated time).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_part2.py`:

```python
# tests/test_run_part2.py

import numpy as np
import pytest
from brian2 import start_scope, second, amp, Hz

from circuit.network import build_network, DEFAULT_PARAMS
from plasticity.stdp_network import DEFAULT_PARAMS_PLASTICITY, build_stdp_network
from plasticity.center_out_task import assign_preferred_directions, generate_test_trial_sequence
from plasticity.train import (
    run_one_trial,
    extract_snapshot_spikes,
    compute_monitoring_metrics,
    check_abort_criteria,
    run_snapshot,
)
from plasticity.snapshot import load_snapshot, load_monitoring


def _small_setup(nu_ext=1000.0, seed=1):
    small = {
        **DEFAULT_PARAMS, **DEFAULT_PARAMS_PLASTICITY,
        'N_exc': 20, 'N_inh': 5,
        'P_size': 8, 'X_size': 8,
        'nu_ext': nu_ext,
    }
    net_objs = build_network(small, seed=seed)
    net_objs = build_stdp_network(net_objs, small, p_cross=1.0, seed=seed)
    theta_i = assign_preferred_directions(small['n_input'], small['n_directions'])
    return net_objs, small, theta_i


def test_run_one_trial_advances_time_by_trial_duration():
    start_scope()
    net_objs, small, theta_i = _small_setup()
    t_before = net_objs['net'].t / second
    run_one_trial(net_objs, small, theta_i, theta_cue=0.0)
    t_after = net_objs['net'].t / second
    expected_dur = small['t_prep'] + small['t_exec'] + small['t_iti']
    assert t_after - t_before == pytest.approx(expected_dur)


def test_run_one_trial_leaves_input_at_background_rate():
    start_scope()
    net_objs, small, theta_i = _small_setup()
    run_one_trial(net_objs, small, theta_i, theta_cue=0.0)
    rates = np.array(net_objs['input_group'].rates[:] / Hz)
    np.testing.assert_allclose(rates, small['r_background'])


def test_extract_snapshot_spikes_keys_and_ranges():
    start_scope()
    net_objs, small, theta_i = _small_setup()
    t_snapshot_start = net_objs['net'].t / second
    n_test_trials = 2
    for d in (0, 1):
        theta_cue = 2 * np.pi * d / small['n_directions']
        run_one_trial(net_objs, small, theta_i, theta_cue)

    spikes = extract_snapshot_spikes(net_objs, t_snapshot_start, small, n_test_trials)
    assert set(spikes.keys()) == {'spike_times_ms', 'spike_neuron_idx', 'spike_trial_idx'}

    n = len(spikes['spike_times_ms'])
    assert len(spikes['spike_neuron_idx']) == n
    assert len(spikes['spike_trial_idx']) == n
    assert n > 0  # nu_ext=1000 guarantees spikes

    trial_dur_ms = (small['t_prep'] + small['t_exec'] + small['t_iti']) * 1000.0
    assert np.all(spikes['spike_times_ms'] >= 0.0)
    assert np.all(spikes['spike_times_ms'] < trial_dur_ms + 1e-6)
    assert np.all(spikes['spike_trial_idx'] >= 0)
    assert np.all(spikes['spike_trial_idx'] < n_test_trials)
    assert np.all(spikes['spike_neuron_idx'] >= 0)
    assert np.all(spikes['spike_neuron_idx'] < small['N_exc'] + small['n_input'])


def test_compute_monitoring_metrics_keys_and_ranges():
    start_scope()
    net_objs, small, theta_i = _small_setup()
    t_snapshot_start = net_objs['net'].t / second
    n_test_trials = 2
    for d in (0, 1):
        theta_cue = 2 * np.pi * d / small['n_directions']
        run_one_trial(net_objs, small, theta_i, theta_cue)

    metrics = compute_monitoring_metrics(net_objs, t_snapshot_start, small, n_test_trials)
    assert set(metrics.keys()) == {'mean_rate_E', 'mean_w_EE', 'frac_w_max', 'mean_cv_isi'}
    assert metrics['mean_rate_E'] >= 0.0
    assert 0.0 <= metrics['frac_w_max'] <= 1.0
    w = np.array(net_objs['syn_EE'].w[:] / amp)
    assert metrics['mean_w_EE'] == pytest.approx(np.mean(w))


def test_check_abort_criteria_raises_on_high_rate():
    metrics = {'mean_rate_E': 35.0, 'mean_w_EE': 0.0, 'frac_w_max': 0.0, 'mean_cv_isi': 1.0}
    with pytest.raises(RuntimeError, match="mean_rate_E"):
        check_abort_criteria(metrics, epoch=100)


def test_check_abort_criteria_raises_on_high_frac_w_max():
    metrics = {'mean_rate_E': 5.0, 'mean_w_EE': 0.0, 'frac_w_max': 0.6, 'mean_cv_isi': 1.0}
    with pytest.raises(RuntimeError, match="frac_w_max"):
        check_abort_criteria(metrics, epoch=100)


def test_check_abort_criteria_passes_normal_metrics():
    metrics = {'mean_rate_E': 5.0, 'mean_w_EE': 0.06e-9, 'frac_w_max': 0.05, 'mean_cv_isi': 0.9}
    check_abort_criteria(metrics, epoch=100)  # should not raise


def test_run_snapshot_writes_hdf5_and_restores_plastic(tmp_path):
    start_scope()
    net_objs, small, theta_i = _small_setup()
    h5_path = str(tmp_path / "test.h5")
    test_trial_sequence = generate_test_trial_sequence(
        n_per_direction=1, n_directions=small['n_directions'])

    run_snapshot(net_objs, h5_path, epoch=0,
                  test_trial_sequence=test_trial_sequence,
                  theta_i=theta_i, params=small, check_abort=False)

    syn = net_objs['syn_EE']
    assert np.all(np.array(syn.plastic[:]) == 1)

    snap = load_snapshot(h5_path, epoch=0)
    assert snap['W_EE_coo']['data'].shape[0] == len(syn)
    np.testing.assert_array_equal(snap['trial_labels'], test_trial_sequence)

    mon = load_monitoring(h5_path)
    np.testing.assert_array_equal(mon['epochs'], [0])


def test_run_snapshot_freezes_weights_during_test_trials(tmp_path):
    start_scope()
    net_objs, small, theta_i = _small_setup()
    h5_path = str(tmp_path / "test.h5")
    syn = net_objs['syn_EE']
    w_before = np.array(syn.w[:] / amp).copy()

    test_trial_sequence = generate_test_trial_sequence(
        n_per_direction=1, n_directions=small['n_directions'])
    run_snapshot(net_objs, h5_path, epoch=0,
                  test_trial_sequence=test_trial_sequence,
                  theta_i=theta_i, params=small, check_abort=False)

    w_after = np.array(syn.w[:] / amp)
    np.testing.assert_allclose(w_before, w_after)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=. python -m pytest tests/test_run_part2.py -v
```

Expected: `ModuleNotFoundError: No module named 'plasticity.train'`.

- [ ] **Step 3: Implement `plasticity/train.py`**

Create `plasticity/train.py`:

```python
# plasticity/train.py

"""
Part 2 trial runner and snapshot logic (spec sections 2.4, 3-5).

run_condition() and the CLI entry point (main) are added in a later task —
this module currently provides the per-trial step and the snapshot routine
used by both the burn-in and training loop.
"""

import numpy as np
from brian2 import second, amp, Hz

from circuit.run_baseline import compute_cv_isi
from plasticity.center_out_task import rates_for_phase
from plasticity.snapshot import save_snapshot


def run_one_trial(net_objs, params, theta_i, theta_cue):
    """
    Run one trial: prep (t_prep) -> exec (t_exec) -> ITI (t_iti) (spec 2.3).

    Sets net_objs['input_group'].rates from rates_for_phase() for each phase
    and advances net_objs['net'] by that phase's duration. After this
    function returns, the input group is left at the ITI (background) rate.
    """
    for phase in ('prep', 'exec', 'iti'):
        rates = rates_for_phase(
            theta_cue, theta_i, phase,
            r_max=params['r_max'],
            r_background=params['r_background'],
            exec_amplification=params['exec_amplification'],
        )
        net_objs['input_group'].rates = rates * Hz
        net_objs['net'].run(params[f't_{phase}'] * second)


def extract_snapshot_spikes(net_objs, t_snapshot_start, params, n_test_trials):
    """
    Extract spikes recorded during [t_snapshot_start, t_snapshot_start +
    n_test_trials * trial_dur) from spike_E and spike_input, and convert
    absolute simulation time to (trial_idx, time_in_trial_ms).

    Input neurons are offset by N_exc, so spike_neuron_idx in the returned
    dict spans 0..N_exc-1 (E) and N_exc..N_exc+n_input-1 (input), per spec
    section 6.

    Returns dict with 'spike_times_ms' (float32, ms within trial),
    'spike_neuron_idx' (int32), 'spike_trial_idx' (int32).
    """
    trial_dur = params['t_prep'] + params['t_exec'] + params['t_iti']
    t_end = t_snapshot_start + n_test_trials * trial_dur

    t_E = np.array(net_objs['spike_E'].t / second)
    i_E = np.array(net_objs['spike_E'].i[:], dtype=np.int64)
    mask_E = (t_E >= t_snapshot_start) & (t_E < t_end)

    t_in = np.array(net_objs['spike_input'].t / second)
    i_in = np.array(net_objs['spike_input'].i[:], dtype=np.int64) + params['N_exc']
    mask_in = (t_in >= t_snapshot_start) & (t_in < t_end)

    times = np.concatenate([t_E[mask_E], t_in[mask_in]])
    neurons = np.concatenate([i_E[mask_E], i_in[mask_in]])

    rel_t = times - t_snapshot_start
    trial_idx = np.floor(rel_t / trial_dur).astype(np.int32)
    # A spike at the exact end of the window would land in trial n_test_trials;
    # clip it back into the last trial.
    trial_idx = np.clip(trial_idx, 0, n_test_trials - 1)
    time_in_trial_ms = ((rel_t - trial_idx * trial_dur) * 1000.0).astype(np.float32)

    return {
        'spike_times_ms': time_in_trial_ms,
        'spike_neuron_idx': neurons.astype(np.int32),
        'spike_trial_idx': trial_idx,
    }


def compute_monitoring_metrics(net_objs, t_snapshot_start, params, n_test_trials):
    """
    Monitoring metrics over the just-completed test-trial window (spec 2.5):
    mean_rate_E, mean_w_EE, frac_w_max, mean_cv_isi.
    """
    trial_dur = params['t_prep'] + params['t_exec'] + params['t_iti']
    t_end = t_snapshot_start + n_test_trials * trial_dur
    duration = t_end - t_snapshot_start

    spike_trains = {k: np.array(v / second)
                     for k, v in net_objs['spike_E'].spike_trains().items()}

    n_spikes = sum(
        int(np.sum((times >= t_snapshot_start) & (times < t_end)))
        for times in spike_trains.values()
    )
    mean_rate_E = n_spikes / (params['N_exc'] * duration)

    w = np.array(net_objs['syn_EE'].w[:] / amp)
    mean_w_EE = float(np.mean(w))
    frac_w_max = float(np.mean(w >= 0.999 * params['w_max']))

    _, mean_cv = compute_cv_isi(spike_trains, t_snapshot_start, t_end, min_spikes=20)

    return {
        'mean_rate_E': float(mean_rate_E),
        'mean_w_EE': mean_w_EE,
        'frac_w_max': frac_w_max,
        'mean_cv_isi': float(mean_cv),
    }


def check_abort_criteria(metrics, epoch):
    """
    Raise RuntimeError if monitoring metrics indicate the run should stop
    (spec 2.5): mean_rate_E > 30 Hz (runaway potentiation) or frac_w_max > 0.5
    (depression insufficient).
    """
    if metrics['mean_rate_E'] > 30.0:
        raise RuntimeError(
            f"Abort at epoch {epoch}: mean_rate_E={metrics['mean_rate_E']:.2f} Hz "
            f"> 30 Hz (possible runaway potentiation)")
    if metrics['frac_w_max'] > 0.5:
        raise RuntimeError(
            f"Abort at epoch {epoch}: frac_w_max={metrics['frac_w_max']:.3f} "
            f"> 0.5 (depression insufficient)")


def run_snapshot(net_objs, h5_path, epoch, test_trial_sequence, theta_i, params,
                  check_abort=True):
    """
    Snapshot protocol (spec 2.4):
    1. Freeze STDP (plastic=0).
    2. Run the fixed test_trial_sequence (40 trials for Phase A: 5/direction).
    3. Record W_EE (COO) and spike data, save to h5_path.
    4. Unfreeze STDP (plastic=1).
    5. Print a one-line summary and, if check_abort, check abort criteria.

    check_abort=False is for tests that use an unrealistic nu_ext to
    guarantee spiking activity in a tiny network — such networks can
    legitimately exceed the 30 Hz / frac_w_max abort thresholds without that
    meaning anything for the real (validated) network run by run_condition().
    """
    syn = net_objs['syn_EE']
    syn.plastic = 0

    t_snapshot_start = net_objs['net'].t / second
    n_test_trials = len(test_trial_sequence)

    for direction_idx in test_trial_sequence:
        theta_cue = 2 * np.pi * direction_idx / params['n_directions']
        run_one_trial(net_objs, params, theta_i, theta_cue)

    spike_data = extract_snapshot_spikes(net_objs, t_snapshot_start, params, n_test_trials)
    metrics = compute_monitoring_metrics(net_objs, t_snapshot_start, params, n_test_trials)

    i_arr = np.array(syn.i[:], dtype=np.int32)
    j_arr = np.array(syn.j[:], dtype=np.int32)
    w_arr = np.array(syn.w[:] / amp, dtype=np.float32)
    W_EE_coo = {
        'data': w_arr,
        'row': j_arr,   # postsynaptic — matches part1 save_baseline convention
        'col': i_arr,   # presynaptic
        'shape': np.array([params['N_exc'], params['N_exc']], dtype=np.int32),
    }

    save_snapshot(h5_path, epoch, W_EE_coo, spike_data, test_trial_sequence, metrics)

    syn.plastic = 1

    print(f"[epoch {epoch:5d}] mean_rate_E={metrics['mean_rate_E']:6.2f} Hz  "
          f"mean_w_EE={metrics['mean_w_EE'] * 1e9:7.4f} nA  "
          f"frac_w_max={metrics['frac_w_max']:.3f}  "
          f"mean_cv_isi={metrics['mean_cv_isi']:.3f}")

    if check_abort:
        check_abort_criteria(metrics, epoch)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH=. python -m pytest tests/test_run_part2.py -v
```

Expected: all 9 tests PASS. If `test_extract_snapshot_spikes_keys_and_ranges` or
`test_compute_monitoring_metrics_keys_and_ranges` fails with `n == 0` (no
spikes), increase `nu_ext` in `_small_setup` to `2000.0` — at `nu_ext=1000`
each E neuron's background drive alone should already exceed threshold for
this 20-neuron network, but the margin is small.

- [ ] **Step 5: Commit**

```bash
git add plasticity/train.py tests/test_run_part2.py
git commit -m "feat(part2): trial runner and snapshot protocol"
```

---

## Task 6: `plasticity/train.py` — burn-in, training loop, CLI

**Files:**
- Modify: `plasticity/train.py`
- Test: `tests/test_run_part2.py`

This task adds `_select_codegen_backend()`, `run_condition()` (burn-in →
epoch-0 snapshot → training loop with periodic snapshots), and the CLI
`main()`. `main()` itself is exercised end-to-end in Task 7 (the real
104-trial validation run) rather than unit-tested — it requires the full
800-neuron baseline and the cython backend, both too heavy for a unit test.

- [ ] **Step 1: Write the failing test for `run_condition`**

Append to `tests/test_run_part2.py` (add `generate_trial_sequence` to the
existing `from plasticity.center_out_task import ...` line, and add `run_condition` to the
existing `from plasticity.train import ...` line):

```python
from plasticity.center_out_task import (
    assign_preferred_directions,
    generate_trial_sequence,
    generate_test_trial_sequence,
)
from plasticity.train import (
    run_one_trial,
    extract_snapshot_spikes,
    compute_monitoring_metrics,
    check_abort_criteria,
    run_snapshot,
    run_condition,
)
```

Then append the test itself:

```python
def test_run_condition_small_network_writes_snapshots(tmp_path):
    start_scope()
    net_objs, small, theta_i = _small_setup()
    small = {**small, 't_burn_in': 0.1}  # keep the test fast

    h5_path = str(tmp_path / "test_condition.h5")
    short_test_sequence = generate_test_trial_sequence(
        n_per_direction=1, n_directions=small['n_directions'])

    run_condition(net_objs, small, h5_path, theta_i,
                   n_per_direction=1, snapshot_epochs={0, 8},
                   seed=1, condition_name='test',
                   check_abort=False, test_trial_sequence=short_test_sequence)

    mon = load_monitoring(h5_path)
    np.testing.assert_array_equal(mon['epochs'], [0, 8])

    snap0 = load_snapshot(h5_path, epoch=0)
    snap8 = load_snapshot(h5_path, epoch=8)
    assert snap0['W_EE_coo']['data'].shape == snap8['W_EE_coo']['data'].shape
    np.testing.assert_array_equal(snap0['trial_labels'], short_test_sequence)


def test_run_condition_runs_correct_number_of_training_trials():
    start_scope()
    net_objs, small, theta_i = _small_setup()
    small = {**small, 't_burn_in': 0.1}
    short_test_sequence = generate_test_trial_sequence(
        n_per_direction=1, n_directions=small['n_directions'])

    t_before = net_objs['net'].t / second

    import tempfile, os as _os
    fd, h5_path = tempfile.mkstemp(suffix='.h5')
    _os.close(fd)
    _os.remove(h5_path)
    try:
        run_condition(net_objs, small, h5_path, theta_i,
                       n_per_direction=1, snapshot_epochs=set(),
                       seed=1, condition_name='test',
                       check_abort=False, test_trial_sequence=short_test_sequence)
    finally:
        if _os.path.exists(h5_path):
            _os.remove(h5_path)

    t_after = net_objs['net'].t / second
    trial_dur = small['t_prep'] + small['t_exec'] + small['t_iti']
    n_trials = small['n_directions'] * 1  # n_per_direction=1
    expected = small['t_burn_in'] + n_trials * trial_dur
    assert (t_after - t_before) == pytest.approx(expected)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=. python -m pytest tests/test_run_part2.py -v -k run_condition
```

Expected: `ImportError: cannot import name 'run_condition'`.

- [ ] **Step 3: Implement `_select_codegen_backend`, `run_condition`, and `main`**

Replace the import block at the top of `plasticity/train.py` with:

```python
import argparse
import os

import numpy as np
from brian2 import second, ms, amp, Hz, prefs, NeuronGroup, Network

from circuit.network import DEFAULT_PARAMS
from circuit.run_baseline import compute_cv_isi
from plasticity.stdp_network import (
    DEFAULT_PARAMS_PLASTICITY,
    load_baseline,
    build_stdp_network,
)
from plasticity.center_out_task import (
    rates_for_phase,
    assign_preferred_directions,
    generate_trial_sequence,
    generate_test_trial_sequence,
)
from plasticity.snapshot import save_snapshot, copy_baseline_provenance, save_training_params
```

Append to `plasticity/train.py`:

```python
def _select_codegen_backend():
    """
    Switch Brian2 to the cython backend for the full validation run (Task 7
    is ~30x slower on numpy). Must be called AFTER importing from
    circuit.network, which sets prefs.codegen.target = 'numpy' at import time
    (see circuit/network.py module-level prefs assignment).

    Falls back to numpy if no working C++ compiler is found.
    """
    prefs.codegen.target = 'cython'
    try:
        test_group = NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1', method='euler')
        test_net = Network(test_group)
        test_net.run(0.1 * ms)
    except Exception as exc:
        print(f"Cython codegen unavailable ({exc!r}); falling back to numpy backend.")
        prefs.codegen.target = 'numpy'


def run_condition(net_objs, params, h5_path, theta_i, n_per_direction, snapshot_epochs,
                   seed=42, condition_name='', check_abort=True, test_trial_sequence=None):
    """
    Burn-in, epoch-0 snapshot, then the training loop with periodic snapshots
    (spec sections 3-4).

    snapshot_epochs : set of int — cumulative trial counts at which to run a
        snapshot, e.g. {0, 50, 100}. Epoch 0 is handled before the training
        loop (the loop's `completed = trial_idx + 1` never equals 0).
    test_trial_sequence : the fixed sequence used at every snapshot. Defaults
        to generate_test_trial_sequence() (40 trials, 5/direction). Tests can
        pass a shorter sequence to keep runtime down.
    """
    if test_trial_sequence is None:
        test_trial_sequence = generate_test_trial_sequence()

    trial_sequence = generate_trial_sequence(n_per_direction, params['n_directions'], seed=seed)
    n_trials = len(trial_sequence)

    # --- Burn-in (spec section 3): STDP frozen, input at background rate,
    # settles the V-initialization transient before any snapshot is taken.
    print(f"[{condition_name}] burn-in: {params['t_burn_in']:.1f} s (plastic=0)")
    syn = net_objs['syn_EE']
    syn.plastic = 0
    net_objs['input_group'].rates = np.full(params['n_input'], params['r_background']) * Hz
    net_objs['net'].run(params['t_burn_in'] * second)
    syn.plastic = 1

    # --- Epoch-0 snapshot (before any training trial)
    if 0 in snapshot_epochs:
        run_snapshot(net_objs, h5_path, epoch=0,
                      test_trial_sequence=test_trial_sequence,
                      theta_i=theta_i, params=params, check_abort=check_abort)

    # --- Training loop
    for trial_idx in range(n_trials):
        direction_idx = trial_sequence[trial_idx]
        theta_cue = 2 * np.pi * direction_idx / params['n_directions']
        run_one_trial(net_objs, params, theta_i, theta_cue)

        completed = trial_idx + 1
        if completed in snapshot_epochs:
            run_snapshot(net_objs, h5_path, epoch=completed,
                          test_trial_sequence=test_trial_sequence,
                          theta_i=theta_i, params=params, check_abort=check_abort)

    return net_objs


def main():
    parser = argparse.ArgumentParser(
        description="Part 2 Phase A: STDP + 8-direction center-out task")
    parser.add_argument('--condition', choices=['seeded', 'control'], required=True,
                         help="seeded: p_cross=0.2 (P/X pool seeding); "
                              "control: p_cross=1.0 (uniform initialization)")
    parser.add_argument('--n_per_direction', type=int, default=13,
                         help="Training trials per direction (default 13 -> "
                              "104 total, Phase A scope)")
    parser.add_argument('--snapshot_epochs', type=int, nargs='+', default=[0, 50, 100],
                         help="Cumulative trial counts at which to snapshot")
    # Must match the seed baseline_network.h5 was generated with (seed=7,
    # see circuit/results/baseline_network.h5:/validation/seed) so
    # load_baseline()'s connectivity check passes.
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--baseline_h5', type=str,
                         default='circuit/results/baseline_network.h5')
    parser.add_argument('--results_dir', type=str, default='part2/results')
    args = parser.parse_args()

    _select_codegen_backend()

    params = {**DEFAULT_PARAMS, **DEFAULT_PARAMS_PLASTICITY}
    p_cross = (params['p_cross_seeded'] if args.condition == 'seeded'
               else params['p_cross_control'])

    os.makedirs(args.results_dir, exist_ok=True)
    h5_path = os.path.join(args.results_dir, f'training_{args.condition}.h5')
    if os.path.exists(h5_path):
        os.remove(h5_path)

    # Self-contained output file (spec section 6): provenance from Part 1,
    # plus this run's STDP/task parameters.
    copy_baseline_provenance(h5_path, args.baseline_h5)
    save_training_params(h5_path, params, p_cross=p_cross, seed=args.seed)

    net_objs = load_baseline(args.baseline_h5, params, seed=args.seed)
    net_objs = build_stdp_network(net_objs, params, p_cross=p_cross, seed=args.seed)
    theta_i = assign_preferred_directions(params['n_input'], params['n_directions'])

    run_condition(net_objs, params, h5_path, theta_i,
                   n_per_direction=args.n_per_direction,
                   snapshot_epochs=set(args.snapshot_epochs),
                   seed=args.seed, condition_name=args.condition)

    print(f"Done. Wrote {h5_path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH=. python -m pytest tests/test_run_part2.py -v
```

Expected: all 11 tests PASS. `test_run_condition_small_network_writes_snapshots`
runs 8 training trials + two 8-trial snapshot sequences (24 trials total,
~29s simulated) on the numpy backend — it may take 20-60s wall time, which is
expected for this backend on a small network.

- [ ] **Step 5: Run the full test suite**

```bash
PYTHONPATH=. python -m pytest -v
```

Expected: all tests across `tests/test_network.py`, `tests/test_hdf5.py`,
`tests/test_analysis.py` (Part 1) and `tests/test_network_part2.py`,
`tests/test_task.py`, `tests/test_snapshot.py`, `tests/test_run_part2.py`
(Part 2) PASS.

- [ ] **Step 6: Commit**

```bash
git add plasticity/train.py tests/test_run_part2.py
git commit -m "feat(part2): burn-in, training loop, and CLI entry point"
```

---

## Task 7: Phase A validation run

**Files:**
- Modify: `.gitignore` (add `plasticity/results/`)
- Create: `plasticity/validate_training.py`
- Test: `tests/test_inspect_phase_a.py`

This task implements the spec section 7 checks as small, independently
testable functions (using synthetic snapshots, like Task 4), then runs the
real 104-trial validation for both conditions and applies those checks to the
real output.

- [ ] **Step 1: Add `plasticity/results/` to `.gitignore`**

`circuit/results/` is already gitignored (large HDF5 output); add the Part 2
equivalent. Edit `.gitignore`:

```
circuit/results/
plasticity/results/
__pycache__/
*.pyc
.pytest_cache/
brian_objects/
```

- [ ] **Step 2: Write the failing tests for the check functions**

Create `tests/test_inspect_phase_a.py`:

```python
# tests/test_inspect_phase_a.py

import numpy as np
import pytest

from plasticity.snapshot import save_snapshot, load_snapshot, load_monitoring
from plasticity.validate_training import (
    check_no_nans,
    check_monitoring_band,
    check_weight_movement,
    check_pool_rescaling,
)


def _make_snapshot_h5(tmp_path, name, epochs_data):
    """epochs_data: dict {epoch: (W_EE_coo, spike_data, trial_labels, metrics)}"""
    h5_path = str(tmp_path / name)
    for epoch, (W_EE_coo, spike_data, trial_labels, metrics) in epochs_data.items():
        save_snapshot(h5_path, epoch, W_EE_coo, spike_data, trial_labels, metrics)
    return h5_path


def _basic_coo(row, col, data, n=4):
    return {'data': np.array(data, dtype=np.float32),
            'row': np.array(row, dtype=np.int32),
            'col': np.array(col, dtype=np.int32),
            'shape': np.array([n, n], dtype=np.int32)}


def _basic_spikes():
    return {'spike_times_ms': np.array([1.0, 2.0], dtype=np.float32),
            'spike_neuron_idx': np.array([0, 1], dtype=np.int32),
            'spike_trial_idx': np.array([0, 0], dtype=np.int32)}


def _basic_metrics(rate=5.0, w=0.06e-9, frac=0.05, cv=0.9):
    return {'mean_rate_E': rate, 'mean_w_EE': w, 'frac_w_max': frac, 'mean_cv_isi': cv}


def test_check_no_nans_passes_on_clean_snapshot(tmp_path):
    coo = _basic_coo([0, 1], [1, 0], [0.05e-9, 0.06e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5",
                                  {0: (coo, _basic_spikes(), [0, 1], _basic_metrics())})
    snap = load_snapshot(h5_path, epoch=0)
    check_no_nans(snap, epoch=0)  # should not raise


def test_check_no_nans_raises_on_nan_weight(tmp_path):
    coo = _basic_coo([0, 1], [1, 0], [np.nan, 0.06e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5",
                                  {0: (coo, _basic_spikes(), [0, 1], _basic_metrics())})
    snap = load_snapshot(h5_path, epoch=0)
    with pytest.raises(AssertionError, match="NaN in W_EE"):
        check_no_nans(snap, epoch=0)


def test_check_monitoring_band_passes_in_range(tmp_path):
    coo = _basic_coo([0, 1], [1, 0], [0.05e-9, 0.06e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5", {
        0:  (coo, _basic_spikes(), [0, 1], _basic_metrics(rate=3.0, frac=0.02)),
        50: (coo, _basic_spikes(), [0, 1], _basic_metrics(rate=8.0, frac=0.10)),
    })
    monitoring = load_monitoring(h5_path)
    check_monitoring_band(monitoring, "test")  # should not raise


def test_check_monitoring_band_raises_on_runaway_rate(tmp_path):
    coo = _basic_coo([0, 1], [1, 0], [0.05e-9, 0.06e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5", {
        0: (coo, _basic_spikes(), [0, 1], _basic_metrics(rate=50.0)),
    })
    monitoring = load_monitoring(h5_path)
    with pytest.raises(AssertionError, match="mean_rate_E"):
        check_monitoring_band(monitoring, "test")


def test_check_weight_movement_raises_if_unchanged(tmp_path):
    coo = _basic_coo([0, 1], [1, 0], [0.05e-9, 0.06e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5", {
        0:   (coo, _basic_spikes(), [0, 1], _basic_metrics()),
        100: (coo, _basic_spikes(), [0, 1], _basic_metrics()),
    })
    snap0 = load_snapshot(h5_path, epoch=0)
    snap100 = load_snapshot(h5_path, epoch=100)
    with pytest.raises(AssertionError, match="identical"):
        check_weight_movement(snap0, snap100, epoch_n=100)


def test_check_weight_movement_passes_if_changed(tmp_path):
    coo0 = _basic_coo([0, 1], [1, 0], [0.05e-9, 0.06e-9])
    coo100 = _basic_coo([0, 1], [1, 0], [0.07e-9, 0.04e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5", {
        0:   (coo0, _basic_spikes(), [0, 1], _basic_metrics()),
        100: (coo100, _basic_spikes(), [0, 1], _basic_metrics()),
    })
    snap0 = load_snapshot(h5_path, epoch=0)
    snap100 = load_snapshot(h5_path, epoch=100)
    check_weight_movement(snap0, snap100, epoch_n=100)  # should not raise


def test_check_pool_rescaling_passes_for_correctly_scaled_weights(tmp_path):
    # P_size=2, X_size=2. row=postsynaptic, col=presynaptic.
    # synapse 0: pre=0,post=1 -> P->P (not cross)
    # synapse 1: pre=0,post=2 -> P->X (cross)
    # synapse 2: pre=2,post=0 -> X->P (cross)
    # synapse 3: pre=2,post=3 -> X->X (not cross)
    row = [1, 2, 0, 3]
    col = [0, 0, 2, 2]
    w_control = [1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9]
    w_seeded  = [1.0e-9, 0.2e-9, 0.2e-9, 1.0e-9]  # cross terms x0.2

    h5_seeded = _make_snapshot_h5(
        tmp_path, "seeded.h5",
        {0: (_basic_coo(row, col, w_seeded), _basic_spikes(), [0, 1], _basic_metrics())})
    h5_control = _make_snapshot_h5(
        tmp_path, "control.h5",
        {0: (_basic_coo(row, col, w_control), _basic_spikes(), [0, 1], _basic_metrics())})

    snap_seeded = load_snapshot(h5_seeded, epoch=0)
    snap_control = load_snapshot(h5_control, epoch=0)

    check_pool_rescaling(snap_seeded, snap_control, p_cross=0.2, P_size=2, X_size=2)


def test_check_pool_rescaling_raises_on_connectivity_mismatch(tmp_path):
    coo_seeded = _basic_coo([1, 2], [0, 0], [1.0e-9, 0.2e-9])
    coo_control = _basic_coo([1, 3], [0, 0], [1.0e-9, 1.0e-9])  # row[1] differs

    h5_seeded = _make_snapshot_h5(
        tmp_path, "seeded.h5",
        {0: (coo_seeded, _basic_spikes(), [0, 1], _basic_metrics())})
    h5_control = _make_snapshot_h5(
        tmp_path, "control.h5",
        {0: (coo_control, _basic_spikes(), [0, 1], _basic_metrics())})

    snap_seeded = load_snapshot(h5_seeded, epoch=0)
    snap_control = load_snapshot(h5_control, epoch=0)

    with pytest.raises(AssertionError, match="connectivity"):
        check_pool_rescaling(snap_seeded, snap_control, p_cross=0.2, P_size=2, X_size=2)
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
PYTHONPATH=. python -m pytest tests/test_inspect_phase_a.py -v
```

Expected: `ModuleNotFoundError: No module named 'plasticity.validate_training'`.

- [ ] **Step 4: Implement `plasticity/validate_training.py`**

```python
# plasticity/validate_training.py

"""
Phase A validation checks (spec section 7). Run after both conditions have
completed:

    PYTHONPATH=. python plasticity/train.py --condition seeded
    PYTHONPATH=. python plasticity/train.py --condition control
    PYTHONPATH=. python plasticity/validate_training.py

Each check_* function takes loaded snapshot/monitoring dicts (from
plasticity.snapshot) and either returns None (pass) or raises AssertionError with
a descriptive message.
"""

import sys

import numpy as np

from plasticity.stdp_network import DEFAULT_PARAMS_PLASTICITY
from plasticity.snapshot import load_snapshot, load_monitoring


def check_no_nans(snapshot, epoch):
    """W_EE weights and spike times must not contain NaNs."""
    w = snapshot['W_EE_coo']['data']
    assert not np.any(np.isnan(w)), f"epoch {epoch}: NaN in W_EE data"
    assert not np.any(np.isnan(snapshot['spike_times_ms'])), \
        f"epoch {epoch}: NaN in spike_times_ms"


def check_monitoring_band(monitoring, condition_name, rate_max=30.0, frac_w_max_max=0.5):
    """Abort criteria (spec 2.5) should hold at every recorded epoch."""
    for epoch, rate, frac in zip(monitoring['epochs'],
                                  monitoring['mean_rate_E'],
                                  monitoring['frac_w_max']):
        assert rate <= rate_max, \
            f"{condition_name} epoch {epoch}: mean_rate_E={rate:.2f} > {rate_max}"
        assert frac <= frac_w_max_max, \
            f"{condition_name} epoch {epoch}: frac_w_max={frac:.3f} > {frac_w_max_max}"
    assert not np.any(np.isnan(monitoring['mean_cv_isi'])), \
        f"{condition_name}: NaN in mean_cv_isi (no neuron had >=20 spikes in a snapshot window)"


def check_weight_movement(snap_epoch0, snap_epoch_n, epoch_n):
    """STDP should have changed at least some E->E weights by epoch_n."""
    w0 = snap_epoch0['W_EE_coo']['data']
    wn = snap_epoch_n['W_EE_coo']['data']
    assert w0.shape == wn.shape, "W_EE sparsity pattern changed between snapshots"
    assert not np.allclose(w0, wn), \
        f"W_EE identical between epoch 0 and epoch {epoch_n} -- STDP had no effect"


def check_pool_rescaling(snap_seeded_epoch0, snap_control_epoch0, p_cross, P_size, X_size,
                          atol=1e-6):
    """
    At epoch 0, seeded and control W_EE should have identical connectivity
    (row, col) -- both come from load_baseline(seed=7) -- and differ
    only on P<->X cross-pool synapses, by exactly a factor of p_cross
    (spec 2.2).
    """
    row_s, col_s, w_s = (snap_seeded_epoch0['W_EE_coo'][k] for k in ('row', 'col', 'data'))
    row_c, col_c, w_c = (snap_control_epoch0['W_EE_coo'][k] for k in ('row', 'col', 'data'))

    assert np.array_equal(row_s, row_c) and np.array_equal(col_s, col_c), \
        "seeded and control W_EE have different connectivity at epoch 0 -- " \
        "did both conditions use the same seed?"

    # row = postsynaptic, col = presynaptic (part1 save_baseline convention)
    pre, post = col_s, row_s
    in_P_pre = pre < P_size
    in_X_pre = (pre >= P_size) & (pre < P_size + X_size)
    in_P_post = post < P_size
    in_X_post = (post >= P_size) & (post < P_size + X_size)
    cross = (in_P_pre & in_X_post) | (in_X_pre & in_P_post)

    ratio = w_s[cross] / w_c[cross]
    np.testing.assert_allclose(
        ratio, p_cross, atol=atol,
        err_msg="P<->X cross-pool weights are not scaled by p_cross")
    np.testing.assert_allclose(
        w_s[~cross], w_c[~cross], atol=atol,
        err_msg="non-cross-pool weights differ between seeded and control")


def main():
    params = DEFAULT_PARAMS_PLASTICITY
    snapshot_epochs = (0, 50, 100)

    results = {}
    for condition in ('seeded', 'control'):
        h5_path = f'plasticity/results/training_{condition}.h5'
        monitoring = load_monitoring(h5_path)
        snapshots = {epoch: load_snapshot(h5_path, epoch) for epoch in snapshot_epochs}
        results[condition] = (monitoring, snapshots)

    all_ok = True

    for condition, (monitoring, snapshots) in results.items():
        for epoch, snap in snapshots.items():
            try:
                check_no_nans(snap, epoch)
            except AssertionError as e:
                print(f"FAIL [{condition}] {e}")
                all_ok = False

        try:
            check_monitoring_band(monitoring, condition)
        except AssertionError as e:
            print(f"FAIL [{condition}] {e}")
            all_ok = False

        try:
            check_weight_movement(snapshots[0], snapshots[100], epoch_n=100)
        except AssertionError as e:
            print(f"FAIL [{condition}] {e}")
            all_ok = False

    try:
        check_pool_rescaling(
            results['seeded'][1][0], results['control'][1][0],
            p_cross=params['p_cross_seeded'],
            P_size=params['P_size'], X_size=params['X_size'])
    except AssertionError as e:
        print(f"FAIL [pool rescaling] {e}")
        all_ok = False

    print("\n=== Monitoring summary ===")
    for condition, (monitoring, _) in results.items():
        print(f"\n{condition}:")
        for epoch, rate, w, frac, cv in zip(
                monitoring['epochs'], monitoring['mean_rate_E'],
                monitoring['mean_w_EE'], monitoring['frac_w_max'],
                monitoring['mean_cv_isi']):
            print(f"  epoch {epoch:5d}: rate={rate:6.2f} Hz  "
                  f"w_EE={w * 1e9:7.4f} nA  frac_w_max={frac:.3f}  cv_isi={cv:.3f}")

    if all_ok:
        print("\nAll Phase A checks PASSED.")
        return 0
    else:
        print("\nSome Phase A checks FAILED -- see above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
PYTHONPATH=. python -m pytest tests/test_inspect_phase_a.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Commit the validation script**

```bash
git add .gitignore plasticity/validate_training.py tests/test_inspect_phase_a.py
git commit -m "feat(part2): Phase A validation checks (spec section 7)"
```

- [ ] **Step 7: Run the seeded condition (104 trials)**

```bash
PYTHONPATH=. python plasticity/train.py --condition seeded
```

Expected: prints `[test] burn-in: 15.0 s (plastic=0)` (well, `[seeded]
burn-in...`), then one `[epoch ...]` summary line at epochs 0, 50, and 100,
each showing `mean_rate_E`, `mean_w_EE`, `frac_w_max`, `mean_cv_isi`. Ends
with `Done. Wrote plasticity/results/training_seeded.h5`.

With the cython backend this should take roughly 5-10 minutes (15s burn-in +
104 training trials * 1.2s + 3 snapshots * 40 test trials * 1.2s ≈ 327s of
simulated time). If `_select_codegen_backend()` fell back to numpy (printed a
warning), expect proportionally longer (numpy is roughly an order of
magnitude slower for this network size) -- run in the background if so.

If this run raises `RuntimeError: Abort at epoch ...`, do NOT proceed to Step
8. Instead inspect the printed `mean_rate_E`/`frac_w_max` values: per spec
2.5, a runaway `mean_rate_E` suggests reducing `A_plus` or increasing
`A_minus` in `DEFAULT_PARAMS_PLASTICITY` (plasticity/stdp_network.py) and re-running
from this step.

- [ ] **Step 8: Run the control condition (104 trials)**

```bash
PYTHONPATH=. python plasticity/train.py --condition control
```

Same expectations as Step 7, writing `plasticity/results/training_control.h5`.

- [ ] **Step 9: Run the validation checks on the real output**

```bash
PYTHONPATH=. python plasticity/validate_training.py
```

Expected: `All Phase A checks PASSED.` followed by the monitoring summary for
both conditions. Read the summary against spec section 7 / 2.5:

- `mean_rate_E` at epoch 0 should be in roughly the Part 1 band (2-10 Hz,
  possibly higher under task drive) for both conditions, and not wildly
  different between epoch 0 and epoch 100 (spec check 3: "within 2x of the
  Part 1 baseline").
- `mean_w_EE` and `frac_w_max` may drift slightly upward from epoch 0 to 100
  (spec 2.5: "should drift upward slightly early, then stabilize") -- 104
  trials is early, so a small increase is expected and a large jump (toward
  `frac_w_max` near 0.5) would be a warning sign worth flagging even though
  the hard `check_monitoring_band` threshold (0.5) wasn't crossed.
- `mean_cv_isi` should stay roughly within [0.5, 1.5] -- a value collapsing
  toward 0 would indicate the network becoming pathologically synchronous.

If `check_pool_rescaling` passes, the P/X pool seeding (spec 2.2) was applied
correctly: at epoch 0, the seeded condition's P<->X weights are exactly 0.2x
the control condition's, and all other E->E weights are identical between
conditions.

This step is the Phase A acceptance gate: if all checks pass and the
monitoring summary looks reasonable per the bullets above, Phase A
infrastructure is validated and ready for the full Phase B run (3200 trials,
spec section 8). Report the printed summary back for review -- the "drift
upward slightly" and CV-ISI bullets are scientific judgment calls, not hard
pass/fail thresholds, and are worth a second look before committing to the
3200-trial run.
