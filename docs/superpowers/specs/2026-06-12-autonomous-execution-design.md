# Autonomous Execution + Frozen Control — Design

**Date:** 2026-06-12
**Status:** approved, implementing
**Scope:** a small, targeted change to the plasticity/training stage to enable a cheap,
decisive pre-check before any long (3200-trial) run.

## Motivation

The Part 3 geometry pilot found no rotational structure (jPCA at/below the shuffle floor)
and aligned — not orthogonal — prep/exec subspaces. The dominant cause is a property of
the **task**, not of STDP: the current execution epoch delivers a *sustained* directional
drive (`exec_amplification = 1.5x` for 500 ms), which **clamps the network state**.

All three target signatures are properties of **autonomous recurrent dynamics**, not of an
input-driven relay:
- Rotational dynamics occur while motor-cortex activity is largely autonomous — movement
  unfolds from a prepared initial condition (Churchland et al. 2012; Sussillo et al. 2015).
- The concrete spiking-network mechanism is **non-normal transient amplification**: a brief
  perturbation, then recurrent E/I structure generates a rotating, decaying trajectory
  (Murphy & Miller 2009; Hennequin, Vogels & Gerstner 2014).

A state held by constant input has no transient to amplify and no autonomous trajectory to
rotate, *regardless of what STDP did to the connectivity*. So the signature is currently
unobservable in principle. The fix is to let the network evolve autonomously during
execution.

This change stays entirely within the project's thesis — **pure local pair-STDP**. It does
not add a readout, reward, or supervision (that would abandon the sufficiency question).

## The change

### 1. Autonomous execution (`center_out_task.py`, `train.py`)

Add `exec_mode` to `rates_for_phase`:
- `exec_mode='sustained'` (default): current behavior — exec input = prep tuning x 1.5.
- `exec_mode='autonomous'`: during exec the task input is **withdrawn to background**
  (`r_background`, like the ITI). The direction cue during prep sets a direction-dependent
  initial condition; the recurrent network then evolves freely during exec under constant
  background drive. The background (`nu_ext`) keeps the network at its AI fixed point, so
  it does not go silent — only the *task* drive is removed.

`run_one_trial` threads `params['exec_mode']` (default `'sustained'`) into `rates_for_phase`,
so both training trials and snapshot test trials use the same regime. No change to the
snapshot schema or the geometry code: the exec window [500, 1000) ms now contains the
autonomous transient instead of the clamped response.

Default stays `'sustained'` so existing behavior, the saved pilot, and all current tests
are unchanged.

### 2. Frozen-weight control (Control A) (`train.py`)

The frozen control runs the identical architecture and task with **STDP off throughout**
(weights never change), isolating geometry that comes from network structure alone. Any
learning effect must exceed this baseline.

- `run_snapshot`: restore the **prior** plastic state after the (always-frozen) test
  trials, instead of hardcoding `plastic = 1`. So a frozen run stays frozen and a plastic
  run stays plastic.
- `run_condition`: add `plasticity_on=True`; after burn-in set `syn.plastic = 1 if
  plasticity_on else 0`.
- `main`: add `'frozen'` to `--condition` and a `--exec_mode {sustained,autonomous}` flag.
  `frozen` uses `p_cross = p_cross_seeded` (0.2) so it is the matched structural control
  for the `seeded` condition. `plasticity_on = (condition != 'frozen')`.

### 3. Output separation (no overwrite, no duplicate results)

The autonomous pre-check writes to a separate directory via `--results_dir
plasticity/results_autonomous`, leaving the existing sustained pilot
(`plasticity/results/`) intact. Filenames stay `training_{condition}.h5`, so the geometry
driver discovers them automatically when pointed at the new dir.

## The pre-check experiment

Two 100-trial runs (snapshots {0, 50, 100}), autonomous exec, into
`plasticity/results_autonomous/`:

| Run | condition | p_cross | STDP | exec_mode |
|---|---|---|---|---|
| seeded-auto | seeded | 0.2 | on | autonomous |
| frozen-auto | frozen | 0.2 | off | autonomous |

Then `geometry/run_geometry.py --results_dir plasticity/results_autonomous`.

**Decision rule:**
- If `seeded-auto` jPCA R² rises above the shuffle floor **and** above `frozen-auto` (and
  the prep/exec angle moves toward orthogonality relative to frozen) → the regime is real;
  the long 3200-trial run is justified.
- If still null → strong evidence the bottleneck is plastic inhibition (next step: add
  inhibitory STDP, Vogels et al. 2011) or that pure E→E STDP is genuinely insufficient (an
  informative §7 null). Either way, learned cheaply, before the long run.

The `frozen-auto` run is the load-bearing control: it shows whether any geometry is
structural (architecture + autonomous dynamics) versus *sculpted by STDP*.

## Verification

- `PYTHONPATH=. python -m pytest tests/test_center_out_task.py tests/test_train.py -q`
  — exec_mode and frozen-control behavior covered; existing tests unchanged (defaults
  preserve current behavior).
  - New: `rates_for_phase(..., exec_mode='autonomous')` returns background for exec.
  - New: a frozen `run_condition` leaves E→E weights unchanged across the run.
- `PYTHONPATH=. python plasticity/train.py --condition seeded --exec_mode autonomous
  --results_dir plasticity/results_autonomous`
  and `--condition frozen --exec_mode autonomous ...` — both complete, abort criteria not
  tripped.
- Sanity: `frozen-auto` `mean_w_EE` is identical at epochs 0/50/100 (weights frozen);
  `seeded-auto` `mean_w_EE` drifts.
