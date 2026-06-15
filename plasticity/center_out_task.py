# plasticity/center_out_task.py

"""
This file defines the 8-direction center-out reaching task: which direction
each "trial" is for, and what firing rate each input neuron should have during
each part (phase) of a trial.

It only uses numpy, not Brian2. train.py takes the rate arrays this file
returns and converts them to Brian2 Quantities (by multiplying by `Hz`)
before handing them to the input PoissonGroup.
"""

import numpy as np


def assign_preferred_directions(n_input=50, n_directions=8):
    """
    Give each of the n_input task-input neurons a "preferred direction"
    theta_i (in radians), spread evenly around the circle.

    With n_input=50 and n_directions=8: 50 = 6*8 + 2, so 2 of the 8 directions
    get 7 neurons each and the other 6 directions get 6 neurons each (about
    6-7 neurons per direction).
    """
    base = n_input // n_directions
    remainder = n_input % n_directions
    counts = np.array([base + 1] * remainder + [base] * (n_directions - remainder))
    directions = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    return np.repeat(directions, counts)


def rates_for_phase(theta_cue, theta_i, phase, r_max=100.0, r_background=2.0,
                     exec_amplification=1.5, exec_mode='sustained'):
    """
    Work out the firing rate (Hz) for each input neuron during one phase of
    a trial.

    - 'prep' (preparation): each neuron's rate follows a cosine tuning curve,
      clipped at zero (Georgopoulos et al. 1982):
      r_i = r_max * max(0, cos(theta_cue - theta_i))
      So a neuron fires fastest when its preferred direction theta_i matches
      the cued direction theta_cue, and not at all if it's pointing more than
      90 degrees away.
    - 'exec' (execution/movement): depends on exec_mode, see below.
    - 'iti' (inter-trial interval): every neuron just fires at the flat
      background rate r_background, regardless of theta_cue.

    exec_mode controls what happens during the 'exec' phase:
    - 'sustained' (default): rates are the same prep tuning curve but scaled
      up by exec_amplification (stronger drive during the "movement"). The
      input is still actively driving the network during this phase.
    - 'autonomous': the task input drops back down to r_background during
      exec. The prep phase has already pushed the network into a
      direction-dependent state, and then during exec the recurrent network
      is left to evolve on its own from that starting point. This is the
      setting where movement-like dynamics (rotational dynamics, Churchland
      2012; transient amplification, Hennequin et al. 2014) can show up,
      since a network that's being actively driven the whole time can't
      show this kind of free evolution. The general background drive
      (nu_ext) stays on the whole time, so the network doesn't go silent;
      only the task-specific input is removed.

    Returns an array the same shape as theta_i.
    """
    if phase == 'iti':
        return np.full_like(theta_i, r_background, dtype=np.float64)
    if phase == 'exec' and exec_mode == 'autonomous':
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
    Build a shuffled list of direction indices (each between 0 and
    n_directions - 1), with exactly n_per_direction trials for each
    direction. The shuffle is reproducible for a given seed, so directions
    are "randomly interleaved" but the order can be regenerated exactly.
    """
    rng = np.random.default_rng(seed)
    sequence = np.repeat(np.arange(n_directions), n_per_direction)
    rng.shuffle(sequence)
    return sequence


def generate_test_trial_sequence(n_per_direction=5, n_directions=8, seed=12345):
    """
    The fixed test-trial sequence used for every snapshot: 5 trials per
    direction, in a fixed pseudorandom order. This uses a different seed
    (12345) than generate_trial_sequence's training default (42), so the
    test trials don't accidentally end up matching the start of the training
    sequence.
    """
    return generate_trial_sequence(n_per_direction, n_directions, seed)
