# Geometry Analysis (Part 3) — Design

**Date:** 2026-06-12
**Status:** approved, implementing
**Prerequisite:** Plasticity training snapshots saved (`plasticity/results/training_{seeded,control}.h5`) and validated.

## Goal

Compute the three geometric observables — participation ratio (PR), jPCA rotational
structure, and preparatory/execution subspace orthogonality — from the snapshot spike
trains collected in the plasticity training stage. Pure analysis; no new simulations.

This is the measurement layer for Q1 (emergence) and the substrate for the Q2
plasticity-space → geometry-space map. See `memory/project_overall_goal.md` for the
north star. The decoder/mediation analysis (Q3) is **out of scope** — it is Part 4,
because the mediation claim is a regression *across networks* and is near content-free
on a single-parameter pilot.

## Scope: pilot, built sweep-ready

The original Part 3 brief assumed 8 epochs × 64 STDP parameter sets × 4 controls. What
exists is a pilot: **2 conditions (seeded p_cross=0.2, control p_cross=1.0) × 3 epochs
(0, 50, 100) × 1 parameter set**, 40 test trials per snapshot (5 reps × 8 directions),
1200 ms/trial (prep 0–500, exec 500–1000, ITI 1000–1200).

The code is written **control-agnostic and sweep-ready**: the driver discovers
`training_*.h5` files and their available epochs rather than hardcoding them, so the
full sweep (64 × 4 × 8) ingests through the identical pipeline with no code change. We
validate and report on the pilot now.

**Controls.** The simulation-level controls — A frozen weights, B spike-shuffled
*plasticity*, C symmetric STDP — are future `training_*.h5` files; not available yet,
same pipeline later. The only control computable from existing data is the
**analysis-level condition-label spike-shuffle**, which is the jPCA chance floor (§3).
Do not conflate it with the simulation-level Control B.

## Two user constraints, enforced structurally

**No duplicate results.** Part 2 files are read-only; spike trains and weights are never
copied into new files. `preprocessing.py` is the single chokepoint that produces the
canonical data matrix `X` once per (condition, epoch, window); all three observables
consume it without re-smoothing or re-normalizing. One output file, keyed by
(condition, epoch, window, n_pcs), written idempotently (re-running overwrites the same
keys, never appends duplicates).

**No false positives / over- or under-fitting.** Every observable is reported against a
matched null, never as a raw number:
- jPCA: Lebedev triple guard — R² above spike-shuffle floor, rotation-direction
  consistency across all 8 conditions, low tangling. A rotation must clear all three.
- Orthogonality: principal angles vs a bootstrap random-subspace null.
- PR: reported with its sample-size ceiling; absolute PR is rank-biased at 40 trials, so
  we lean on relative comparisons (epoch-to-epoch, seeded-vs-control).
- Universal overfitting guard: **trial-split cross-validation** — estimate geometry on
  one half of the 5 trials/condition, measure on the held-out half. Real structure
  generalizes across the split; artifacts of averaging few noisy trials do not.
- **Synthetic jPCA validation is a hard prerequisite** (§3): the jPCA implementation
  must pass a pure-rotation fixture (R²→1) and a Lebedev feedforward-sequence fixture
  (high naive R² but caught by the triangulation) before it is trusted on real data.

## File structure

New top-level `geometry/` package, mirroring `circuit/` and `plasticity/`:

| File | Responsibility |
|---|---|
| `geometry/preprocessing.py` | Snapshot spikes → canonical data matrix `X`. Single source. |
| `geometry/dimensionality.py` | Participation ratio via gram-matrix eigenspectrum. |
| `geometry/jpca.py` | Skew-symmetric fit, R²_skew, jPC planes, rotation consistency, tangling. |
| `geometry/orthogonality.py` | Prep/exec principal angles + bootstrap null. |
| `geometry/controls.py` | Shared nulls: condition-label spike-shuffle, trial-split CV indices. |
| `geometry/synthetic.py` | Synthetic fixtures: pure rotation, Lebedev sequence (for validation). |
| `geometry/run_geometry.py` | Driver: discover files, preprocess once, compute all, write tidy results. |
| `tests/test_*` | One per module; `test_jpca.py` carries the synthetic-validation gate. |

Decomposition principle: `preprocessing.py` is the chokepoint — everything downstream
consumes its output and nothing recomputes it. That structurally enforces "no copies."

## 1. Preprocessing pipeline (`preprocessing.py`)

Applied identically to every snapshot. Follows Churchland et al. (2012).

| Stage | Decision | Rationale |
|---|---|---|
| Neurons | E only, indices 0–799 (N=800). Drop input neurons 800–849. | Input neurons carry the imposed cosine tuning by construction — including them injects the structure we test for. I not recorded. |
| Rate estimate | Bin each trial to 1 ms; `scipy.ndimage.gaussian_filter1d` σ=25 ms, truncate=3.0, `mode='constant'` (zero outside trial). | Churchland smoothing. Exec window [500,1000] is fully interior (clean); prep first ~75 ms mildly edge-deflated. |
| Condition-average | Average the 5 trials per direction → 8 condition PSTHs `r_i(t,c)`. Retain the 5 raw trials for CV. | Churchland operates on condition-averaged rates. |
| Downsample | After smoothing, downsample to **10 ms bins** → 50 timepoints per 500 ms window. | A 25 ms Gaussian makes 1 ms samples near-perfectly autocorrelated; 500 redundant rows bloat the gram matrix and make the jPCA derivative tiny/noisy. 10 ms is standard. |
| Soft-norm | `r_i / (R_i + 5 Hz)`, `R_i` = range over all t and all 8 conditions. | Churchland; the +5 Hz floor down-weights silent/high-rate neurons. Not z-scoring. |
| Mean-subtract | Subtract the cross-condition mean at each t: `r_centered(t,c) = r_norm(t,c) − mean_c r_norm(t,c)`. | Removes the condition-independent component. Skipping this inflates rotational R² — the most common analysis error. |
| Windows | prep = [0,500), exec = [500,1000); ITI dropped. | PR per-window; jPCA on exec; orthogonality compares the two subspaces. |

Output per (condition, epoch): `X` of shape **(N=800, T=50, C=8)** per window, plus
condition labels, the time axis, and per-trial arrays for CV. Computed once.

Note: cross-condition mean subtraction also zeroes each neuron's mean over all (t,c)
samples (since `Σ_c r_centered(t,c)=0` for every t), so no additional PCA centering is
needed — the gram matrix on `X` is already the centered covariance.

## 2. Participation ratio (`dimensionality.py`)

Form `D` of shape (M=T·C, N) from the mean-subtracted `X`. Compute the M×M gram matrix
`G = D Dᵀ` (here 400×400, avoiding the rank-deficient 800×800 covariance), take its
nonnegative eigenvalues `λ_i`:

```
PR = (Σ_i λ_i)² / Σ_i λ_i²
```

The `1/(M−1)` covariance scaling cancels in the ratio. Computed per window (prep, exec).

Sample-size ceiling: with M=400 samples and the mean-subtraction removing one df per
timepoint, PR is bounded well below N=800; absolute PR is biased. Report it, but read
**relative** changes (epoch, condition) as the signal. Prediction (Q1): PR decreases
over learning under asymmetric STDP.

## 3. jPCA rotational structure (`jpca.py`)

1. **Project to top-k PCs** of the exec-window `X` (k=6 default; sensitivity at 4, 10).
   PCs via SVD of `D`; scores reshaped to per-condition trajectories `X_c(t)` (T, k).
2. **Derivative** by central difference within each condition,
   `dX_c(t) = (X_c(t+1) − X_c(t−1)) / 2`; drop endpoints. Stack to `X_all`, `dX_all`.
3. **Skew-symmetric fit.** `M_skew` (k×k, `M=−Mᵀ`, k(k−1)/2 free params). Build the
   design matrix mapping the upper-triangle params to `vec(X_all @ Mᵀ)` and solve with
   `numpy.linalg.lstsq` (closed form — not generic optimization).
4. **R²_skew** relative to the unconstrained fit `M_full = lstsq(X_all, dX_all)ᵀ`:
   ```
   R²_skew = 1 − ‖dX_all − X_all M_skewᵀ‖² / ‖dX_all − X_all M_fullᵀ‖²
   ```
   (fraction of *explainable* derivative variance that is specifically rotational). Also
   record R² vs total `dX` variance for comparison against the shuffle floor.
5. **jPC planes.** Eigendecompose `M_skew` → imaginary conjugate pairs `±iω`; the
   largest |ω| pair is the dominant plane (real/imag parts of the eigenvector).

**Lebedev triangulation (mandatory).** A positive jPCA result must clear all three:
- **Rotation-direction consistency** — project each condition onto the dominant jPC
  plane; the signed angular velocity (rotation sense) must agree across all 8 conditions.
- **Spike-shuffle floor** — R² on condition-label-shuffled data (`controls.py`) must be
  substantially below the real R².
- **Tangling** (Russo et al. 2018):
  `Q(t) = max_{t'} ‖dX(t) − dX(t')‖² / (‖X(t) − X(t')‖² + ε)`, ε a small fraction of
  total state variance. Report mean tangling; compare to the shuffle control. High
  tangling ⇒ inconsistent with an autonomous dynamical system (sequence artifact).

**Synthetic validation gate (hard prerequisite, in `test_jpca.py`).**
- Pure rotation: trajectories generated by a known skew `M` ⇒ R²_skew ≈ 1, recovered ω
  matches, rotation-direction consistent, low tangling.
- Lebedev sequence: staggered Gaussian bumps with consistent cross-condition ordering ⇒
  high *naive* R² but the triangulation flags it (inconsistent direction and/or high
  tangling and/or no lift above shuffle). This tests that the **guards** work, not just
  the fit. No real-data jPCA result is reported unless both fixtures pass.

## 4. Prep/exec subspace orthogonality (`orthogonality.py`)

Separately PCA the prep-window and exec-window `X`; take the top-6 PCs of each →
`P_prep`, `P_exec` (each an N×6 orthonormal basis). Principal angles via
`scipy.linalg.subspace_angles(P_prep, P_exec)` → six angles in [0, π/2]; summarize as the
mean. Near π/2 = orthogonal, near 0 = aligned.

Null: random 6-D subspaces in N=800-D already sit near 90°, so the raw angle is
meaningless alone. Bootstrap (≥1000 draws) the mean-principal-angle distribution for
random 6-D subspaces; report the observed angle relative to that null. Prediction (Q1):
STDP drives the two subspaces toward orthogonality, i.e. above the structural baseline.

## 5. Trial-split cross-validation (`controls.py`)

Universal overfitting guard. Split the 5 trials/condition into two folds; recompute the
geometry on each and measure generalization (principal angle between the two folds'
subspaces; jPCA `M` fit on fold A scored on fold B's `dX`). Repeated random splits, since
5 is small. A geometry that is an artifact of averaging few noisy trials will not
generalize across the split.

## 6. Driver and output (`run_geometry.py`)

Discover `plasticity/results/training_*.h5`; for each file read available
`/snapshots/epoch_*`. For each (condition, epoch): preprocess once per window, compute PR
(both windows), jPCA (exec, k∈{4,6,10}) with full triangulation, orthogonality (prep vs
exec) with null, and the CV generalization metrics. Write one **tidy/long** results file
`geometry/results/geometry_metrics.h5` (or `.csv`): one row per
(condition, epoch, window, observable, n_pcs) with the value and its matched null/shuffle.
At full scale this same schema holds 64 × 4 × 8 rows; the Q2 map is then a groupby, never
a re-analysis. Idempotent: re-running overwrites keys.

## Verification

- `PYTHONPATH=. python -m pytest tests/test_*.py` — all geometry tests pass, **including
  the two synthetic jPCA fixtures** (the gate).
- `PYTHONPATH=. python geometry/run_geometry.py` — produces the tidy results file; sanity
  reads:
  - PR in a plausible range (single-digit to low-tens), exec ≤ prep not assumed.
  - jPCA: report whether the real R² clears the shuffle floor *and* direction-consistency
    *and* low tangling — expect this to possibly fail on the thin pilot; that is an honest
    negative, not a bug.
  - Orthogonality: observed mean angle vs the random-subspace null.
  - The **epoch-0 seeded-vs-control** comparison is the primary pipeline-correctness
    check: the p_cross=0.2 seeding imposes a P↔X weight asymmetry at init, so the two
    should differ structurally *before* any learning. If they don't, suspect the pipeline.
- Trends across epoch 0→100 are **exploratory**, not confirmatory: weight change was ~11%
  and N=40 trials/epoch, so epoch differences may be sampling-noise-dominated. The result
  tells us whether the next move is more training, more test trials, or STDP retuning.
