# Part 2: STDP and Task Structure — Design

## Goal

Add pair-based STDP to the recurrent E→E synapses of the Part 1 network, introduce
an 8-direction center-out reaching task as structured input, and run a training
protocol that produces snapshots of the weight matrix and spike trains for later
(Part 3/4) analysis. Two conditions are run: a **seeded** condition with weak
preparatory/execution cross-pool coupling at initialization, and a **control**
with uniform coupling.

This is one project executed in two phases:

- **Phase A** (this plan): build all infrastructure (STDP synapses, pool seeding,
  task input, trial loop, snapshot I/O, monitoring/abort logic) and validate it
  with a short run (100 trials, both conditions, snapshots at {0, 50, 100}).
- **Phase B** (follow-up plan, not covered by this doc's implementation plan):
  the full 3200-trial × 2-condition training runs, snapshots at all 8 epochs, and
  the section "Phase B validation checks" below.

## Prerequisite

Part 1 is complete and validated (`circuit/results/baseline_network.h5`, operating
point `nu_ext=7.0 Hz`, `g_EI=0.090 nA`, `w_scale_II=0.50`; see
`docs/circuit_tuning_notes.md`). Part 2 loads this baseline rather than building a
network from scratch.

**Critical constraint carried over from Part 1:** w_EE can grow at most ~29%
before the network exits the AI corridor (CV-ISI < 0.8). Part 2's STDP parameters
are depression-dominant (per Song et al. 2000) to keep the *mean* weight roughly
stable while individual synapses redistribute toward 0 or `w_max`. This is
monitored at every snapshot (see "Monitoring and abort criteria") rather than
solved with explicit weight normalization — Phase A's job is partly to confirm
this is sufficient.

---

## Architecture

```
part2/
    network_part2.py   # load Part 1 baseline, build STDP synapses, pool rescaling, task input
    task.py             # tuning curves, trial sequences, per-epoch input rates
    snapshot.py         # HDF5 snapshot save/load
    run_part2.py        # CLI: training loop, burn-in, monitoring, abort checks
    results/
        figures/
        training_seeded.h5
        training_control.h5
```

Each module has one job: `network_part2.py` only builds Brian2 objects,
`task.py` only computes rates/sequences (pure numpy, no Brian2 dependency),
`snapshot.py` only does HDF5 I/O, `run_part2.py` orchestrates.

---

## 1. Network construction (`network_part2.py`)

### 1.1 Loading the Part 1 baseline

`load_baseline(h5_path, params, seed=42)`:

1. Calls `build_network(params, seed=seed)` from `circuit/network.py` with the
   same `DEFAULT_PARAMS` and seed used to produce `baseline_network.h5`. Because
   both Brian2's RNG (`brian2_seed`) and the numpy RNG used for weight
   initialization are seeded identically, this **deterministically reproduces**
   the same connectivity and weights as the saved file.
2. Sanity-checks this reproduction against the saved HDF5: asserts
   `(syn_EE.i[:], syn_EE.j[:])` matches the saved `(row, col)` COO indices for
   `W_EE` (and similarly for `syn_EI`, `syn_IE`, `syn_II`).
3. Overwrites `.w` on all four synapse groups from the HDF5-saved `data` arrays
   directly — this guarantees Part 2 starts from the *exact validated* weights,
   independent of any future floating-point or library-version drift in step 1.
4. Returns the same `net_objs` dict shape as `build_network()` (`exc`, `inh`,
   `syn_EE`, `syn_EI`, `syn_IE`, `syn_II`, `drive_E`, `drive_I`, `spike_E`,
   `spike_I`, `net`), without yet adding it to a `Network()`.

### 1.2 Subpopulation seeding (spec 2.2)

Partition the 800 E neurons by index:

| Pool | Indices | Size |
|------|---------|------|
| Preparatory (P) | 0–349 | 350 |
| Execution (X) | 350–699 | 350 |
| Shared (S) | 700–799 | 100 |

`apply_pool_rescaling(i, j, w, p_cross)` — pure numpy, operates on the loaded
`syn_EE` arrays `(i=pre, j=post, w)`:

- If `(i in P and j in X)` or `(i in X and j in P)`: multiply `w` by `p_cross`.
- All other pairs (P↔P, X↔X, S↔S, S↔P, S↔X, P↔S, X↔S): unchanged (×1.0).

`p_cross = 0.2` for the **seeded** condition, `p_cross = 1.0` for **control**.
This function only touches the numpy weight array — it does not call any Brian2
or RNG function, so it cannot perturb Brian2's random stream.

### 1.3 STDP synapses (spec 2.1)

The loaded `syn_EE` (static weights, no plasticity) is **discarded** and replaced
by a new `Synapses` group with the same `(i, j)` connectivity and the
pool-rescaled `w` as initial values:

```python
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

syn_EE_stdp = Synapses(exc, exc, stdp_eqs,
                        on_pre=on_pre_eqs, on_post=on_post_eqs,
                        namespace=stdp_ns, method='euler', name='syn_EE_stdp')
syn_EE_stdp.connect(i=i_arr, j=j_arr)
syn_EE_stdp.w = w_rescaled * amp
syn_EE_stdp.plastic = 1
syn_EE_stdp.x_pre = 0
syn_EE_stdp.x_post = 0
```

`(event-driven)` on the trace ODEs gives exact analytic integration between
spikes and avoids per-timestep updates for ~64,000 synapses.

`plastic` is a **shared** synapse-level variable (one value for the whole
group). Setting `syn_EE_stdp.plastic = 0` freezes weight changes (traces still
update) — used during burn-in and snapshot test trials. `syn_EI`, `syn_IE`,
`syn_II` are carried over from `load_baseline()` unchanged — STDP applies
only to E→E.

**STDP parameters:**

| Parameter | Value | Derivation |
|-----------|-------|------------|
| `tau_plus` | 20 ms | Spec baseline (Bi & Poo 1998) |
| `tau_minus` | 20 ms | Spec baseline, symmetric |
| `w_max` | 0.24 nA | 4× `w_mean_EE` (0.06 nA) |
| `A_plus` | 0.0024 nA | `0.01 × w_max` |
| `A_minus` | 0.00252 nA | `0.0105 × w_max` (5% depression-dominant, per Song et al. 2000 stability condition `A_minus·tau_minus > A_plus·tau_plus`) |

Note on `w_max`: with the Part 1 lognormal weight distribution
(`mu_log = ln(0.06) - 0.125`, `sigma=0.5`), roughly 0.1% of E→E synapses
(~60-70 out of ~64,000) already exceed 0.24 nA at initialization and will be
clipped to `w_max` on their first STDP event. This is negligible and not worth
special-casing.

### 1.4 Task input neurons (spec 2.3, connectivity choice from design discussion)

50 additional neurons, **not part of the recurrent E/I populations**:

```python
input_group = PoissonGroup(50, rates=np.full(50, 2.0) * Hz)  # init at ITI rate
syn_input_E = Synapses(input_group, exc, 'w : amp', on_pre='I_exc_post += w', name='syn_input_E')
syn_input_I = Synapses(input_group, inh, 'w : amp', on_pre='I_exc_post += w', name='syn_input_I')
syn_input_E.connect(p=0.1)
syn_input_I.connect(p=0.1)
syn_input_E.w = _lognormal_weights(w_mean_EE, sigma_w, len(syn_input_E), rng) * amp
syn_input_I.w = _lognormal_weights(w_mean_EE, sigma_w, len(syn_input_I), rng) * amp
spike_input = SpikeMonitor(input_group)
```

- Connects to **both** E and I populations at `p=0.1`, matching recurrent
  connectivity density.
- Weights drawn from the same lognormal(`w_mean_EE=0.06nA`, `sigma=0.5`)
  distribution as Part 1's E→E/E→I weights — same statistics, just a different
  source population. Drawn from a **separate** `np.random.default_rng(seed +
  1000)` (not the `rng` consumed inside `build_network()`), so input-weight
  draws don't shift and don't depend on the recurrent network's weight draws.
- **Static** (`w : amp`, no STDP equations) — not subject to plasticity.
- `input_group.rates` is reassigned (a plain array of 50 `Hz` values) before
  each epoch's `net.run()` call; Brian2 supports changing `PoissonGroup.rates`
  between runs.

### 1.5 Assembling the network

`build_stdp_network(net_objs, params, p_cross)` returns a new `net_objs` dict
with `syn_EE` replaced by `syn_EE_stdp`, plus `input_group`, `syn_input_E`,
`syn_input_I`, `spike_input` added, and a fresh `Network(...)` containing all of
these plus everything from `load_baseline()` (`exc`, `inh`, `syn_EI`,
`syn_IE`, `syn_II`, `drive_E`, `drive_I`, `spike_E`, `spike_I`).

Background Poisson drive (`drive_E`, `drive_I`, `nu_ext=7.0 Hz`) **continues
unchanged** throughout Part 2 — the task input is additive on top of it, exactly
as it was additive on top of nothing in Part 1.

### 1.6 Codegen backend

Part 1 used `prefs.codegen.target = 'numpy'` (good for many short repeated runs
during tuning). Part 2's training runs are long (3840s × 2 conditions for Phase
B); switch to Brian2's default `'cython'` backend, which compiles once and then
runs the simulation loop natively. `run_part2.py` checks for a working C++
compiler at startup and falls back to `'numpy'` with a warning if compilation
fails (so Phase A can still run on a machine without a compiler, just slower).

---

## 2. Task structure (`task.py`)

### 2.1 Input neuron preferred directions

50 neurons assigned preferred directions evenly spaced at 45° (8 directions):
2 directions get 7 neurons, 6 directions get 6 neurons (`2×7 + 6×6 = 50`).

```python
def assign_preferred_directions(n_input=50, n_directions=8):
    base = n_input // n_directions       # 6
    extra = n_input % n_directions        # 2
    counts = [base + 1 if d < extra else base for d in range(n_directions)]
    thetas = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
    theta_i = np.repeat(thetas, counts)
    return theta_i  # shape (50,)
```

### 2.2 Tuning curve and per-epoch rates

```python
def rates_for_epoch(theta_cue, theta_i, epoch, r_max=100.0):
    base = r_max * np.maximum(0.0, np.cos(theta_cue - theta_i))
    if epoch == 'prep':
        return base
    elif epoch == 'exec':
        return 1.5 * base
    elif epoch == 'iti':
        return np.full_like(theta_i, 2.0)
```

### 2.3 Trial sequences

`generate_trial_sequence(n_per_direction, n_directions=8, seed)`: returns an
array of cue-direction indices (0-7), `n_per_direction` of each, shuffled with
the given seed. For Phase B, `n_per_direction=400` → 3200 trials. For Phase A,
`n_per_direction` is chosen so the total is ~100 trials (12-13 per direction).

**Both conditions (seeded, control) use the same trial sequence** (same seed) —
this isolates `p_cross` as the only difference between runs.

`generate_test_trial_sequence(n_per_direction=5, n_directions=8, seed)`: a
separate fixed 40-trial sequence, used identically at every snapshot in both
conditions.

### 2.4 Trial timing

| Epoch | Duration | Input rate |
|-------|----------|------------|
| Preparatory | 500 ms | `r_max · max(0, cos(θ_cue - θ_i))` |
| Execution | 500 ms | `1.5 ×` preparatory rate |
| ITI | 200 ms | flat 2 Hz |

Total = 1.2 s/trial.

---

## 3. Burn-in (new — addresses Part 1's "transient on load" warning)

Part 1's tuning notes warn: "If Part 2 loads the HDF5 weights but reinitializes
V randomly, you'll see the same 15-second transient." `build_network()`
initializes `V ~ U[V_reset, V_th]`, so `load_baseline()` reproduces this
random initial condition.

For the full 3840s Phase B run, a 15-20s transient is <0.5% of total time and
negligible. For Phase A's ~100-trial (~120s) run, it would be 12-17% of the run
— enough to contaminate the epoch-0 and epoch-50 snapshots.

**Procedure:** after building the full Part 2 network (§1.5) with
`syn_EE_stdp.plastic = 0` and `input_group.rates` at the ITI/background level
(2 Hz for all 50 neurons), run `net.run(15*second)`. Then proceed to the epoch-0
snapshot (§4) with STDP still frozen. Spikes generated during burn-in are not
included in any snapshot — `run_part2.py` records `spike_E.num_spikes` (etc.)
immediately after burn-in as the baseline index for slicing.

This applies identically to both conditions.

---

## 4. Training loop and snapshot protocol (`run_part2.py`)

```python
# after burn-in:
spike_idx = {'E': spike_E.num_spikes, 'I': spike_I.num_spikes, 'input': spike_input.num_spikes}

# epoch 0: post-burn-in baseline, before any training trials
run_snapshot(net_objs, h5_path, epoch=0, test_trial_sequence, theta_i, params)

for trial_idx, cue in enumerate(trial_sequence):
    input_group.rates = rates_for_epoch(cue, theta_i, 'prep') * Hz
    net.run(0.5 * second)
    input_group.rates = rates_for_epoch(cue, theta_i, 'exec') * Hz
    net.run(0.5 * second)
    input_group.rates = rates_for_epoch(cue, theta_i, 'iti') * Hz
    net.run(0.2 * second)

    epoch = trial_idx + 1
    if epoch in snapshot_epochs:
        run_snapshot(net_objs, h5_path, epoch, test_trial_sequence, theta_i, params)
```

### 4.1 `run_snapshot`

1. `syn_EE_stdp.plastic = 0`.
2. Record spike counts (`spike_E.num_spikes` etc.) as the start index.
3. Run the 40-trial test sequence (5/direction, fixed order) using the same
   prep/exec/ITI loop as training, but without checking `snapshot_epochs`.
4. Slice `spike_E.t[:]`, `spike_E.i[:]` (and `spike_input`) from the recorded
   start index to the end. Convert absolute times to
   `(spike_trial_idx, spike_time_ms)` using the known per-trial 1.2s boundaries
   relative to the snapshot's start time.
5. Read current `syn_EE_stdp.w[:]` (with `syn_EE_stdp.i[:]`, `.j[:]` as the COO
   row/col — these are fixed since training time, only `w` changes).
6. Compute monitoring metrics from the 40-trial spike data (§5).
7. Call `snapshot.save_snapshot(h5_path, epoch, W_EE_coo, spike_data,
   trial_labels, monitoring_metrics)`.
8. Check abort conditions (§5). If triggered, raise/exit with a clear message —
   do not continue training.
9. `syn_EE_stdp.plastic = 1`, continue training.

Spike indices for the combined `spike_neuron_idx` array: 0-799 = E neurons
(`exc`), 800-849 = input neurons (`input_group`). I neurons are simulated
(needed for network dynamics) but **not** included in snapshots, per the
approved scope.

---

## 5. Monitoring and abort criteria (spec 2.5)

At every snapshot, computed from the 40 test-trial spike data:

- **Mean E firing rate** (Hz), averaged over all 800 E neurons and the 48s of
  test-trial time.
- **Mean E→E weight** (`syn_EE_stdp.w[:]` mean, in nA).
- **Fraction of E→E weights at `w_max`** (within floating-point tolerance,
  e.g. `w >= 0.999 * w_max`).
- **Mean CV-ISI** across E neurons with `>=20` spikes in the test-trial window
  (same `min_spikes=20` convention as Part 1).

These four values are appended to `/monitoring/` in the HDF5 file (one row per
snapshot epoch).

**Abort conditions** (checked after every snapshot):
- `mean_rate_E > 30 Hz`, OR
- `frac_w_max > 0.5`

If either triggers, `run_part2.py` prints a diagnostic summary (current values,
which condition, which epoch) and exits without continuing training. Per the
spec, the most likely fix is adjusting `A_minus`/`A_plus` — that is a Phase A
follow-up if it occurs, not handled automatically.

---

## 6. HDF5 schema

Two files, fully self-contained (each copies `/network`, `/weights`,
`/validation` from `baseline_network.h5` for provenance):

- `plasticity/results/training_seeded.h5` (`p_cross=0.2`)
- `plasticity/results/training_control.h5` (`p_cross=1.0`)

```
/network        — copied from Part 1 (params, seed, etc.)
/weights        — copied from Part 1 (original W_EE, W_EI, W_IE, W_II, pre-rescaling)
/validation     — copied from Part 1 (Part 1's own validation results)

/snapshots/epoch_{N}/
    W_EE/data, W_EE/row, W_EE/col, W_EE/shape   — COO, current syn_EE_stdp.w
    spike_times_ms      — 1D float array, time within trial (0-1200 ms)
    spike_neuron_idx    — 1D int array, 0-799 = E, 800-849 = input
    spike_trial_idx     — 1D int array, 0-39
    trial_labels        — (40,) int array, cue direction index 0-7

/monitoring/
    epochs              — (n_snapshots,) int array
    mean_rate_E         — (n_snapshots,) float, Hz
    mean_w_EE           — (n_snapshots,) float, nA
    frac_w_max          — (n_snapshots,) float
    mean_cv_isi         — (n_snapshots,) float

/training_params       — attrs: p_cross, tau_plus, tau_minus, A_plus, A_minus,
                       w_max, n_input, r_max, trial sequence seed, burn-in duration
```

---

## 7. Phase A scope (this implementation plan)

- Build all of `network_part2.py`, `task.py`, `snapshot.py`, `run_part2.py`.
- Run **both** conditions (seeded `p_cross=0.2`, control `p_cross=1.0`) for a
  short sequence of **104 trials (13 per direction × 8)**, snapshot epochs
  `{0, 50, 100}`. (Trials 101-104 continue training after the last snapshot
  with no further snapshot — kept only so `n_per_direction` divides evenly;
  they don't need separate handling.)
- Approximate cost: 15s burn-in + ~125s training + 3×48s test trials ≈ 4.7 min
  simulated time per condition, ~9.4 min total — cheap enough to iterate on.

**Phase A checks:**
1. Network stays in a reasonable AI-like band (rate 2-10 Hz region; CV not
   collapsing) at every snapshot — the "did we break it" check. At 100 trials
   we do *not* expect the full Song et al. bimodal weight distribution yet.
2. `W_EE` at epoch 0 differs between conditions **only** in P↔X cross-pool
   entries, by exactly the `0.2/1.0` ratio — validates §1.2.
3. Weight values show *some* movement between epoch 0 and epoch 100 (not
   frozen) — confirms STDP is wired correctly (sign conventions, traces,
   `plastic` flag).
4. Snapshots round-trip correctly through HDF5 (`save_snapshot` →
   `load_snapshot` → identical arrays).
5. `/monitoring/` values are sane (no NaNs, rate/CV in plausible ranges).

**Explicitly expected to need retuning:** input neuron weight scale and/or
`A_plus`/`A_minus`, in the same way Part 1's `g_EI` needed empirical correction.
If Phase A's monitoring shows runaway rate or excessive `frac_w_max` growth even
at 100 trials, that's the signal to retune before Phase B — not a Phase A
failure per se, but something to flag and resolve before committing to the full
3200-trial runs.

---

## 8. Phase B (follow-up, not part of this plan)

Full training: both conditions, 3200 trials (400/direction), snapshot epochs
`{0, 50, 100, 200, 400, 800, 1600, 3200}`. After completion, run the spec's
section 2.6 validation checks:

1. **Weight distribution at epoch 3200** — histogram of all `W_EE` values;
   expect bimodal (Song et al. 2000) with mass near 0 and near `w_max`.
2. **Weight matrix structure over training** — mean P→X vs. mean P→P (and
   X→X) weight as a function of epoch; if seeding matters, within-pool should
   grow faster than cross-pool, and the seeded condition should diverge from
   control.
3. **Network stability at epoch 3200** — mean rate and CV-ISI within 2× of the
   Part 1 baseline (rate 2.39 Hz, CV 0.832).
4. **Trial-to-trial reproducibility** — pairwise correlation of smoothed
   (25 ms Gaussian) population rate vectors across the 5 same-direction test
   trials at epoch 3200, compared to across-direction pairs; same-direction
   should be substantially higher.

---

## Open questions / future refinements (not blocking Phase A)

- **Common random numbers across conditions:** running both conditions with
  identical Brian2 RNG streams (background Poisson, input Poisson draws) would
  make any seeded-vs-control divergence attributable purely to `p_cross`. Not
  implemented in Phase A to keep the two conditions as independent, simpler
  runs; revisit if Phase B results are ambiguous and noise-vs-effect is hard to
  separate.
