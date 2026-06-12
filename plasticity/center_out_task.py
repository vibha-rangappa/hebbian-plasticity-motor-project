# plasticity/center_out_task.py

"""
Task structure for the 8-direction center-out reaching task (spec section 2.3).

Pure numpy — no Brian2 dependency. train.py converts the rate arrays
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
