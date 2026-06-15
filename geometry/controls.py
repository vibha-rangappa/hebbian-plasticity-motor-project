# geometry/controls.py

"""
This file holds shared "null" and cross-validation helpers used by all three geometry
observables (PR, jPCA, orthogonality), as described in spec sections 3 and 5. The idea is
to define these checks once here, instead of writing slightly different versions of them
in each observable's code.

Two helpers live here:

  condition_shuffle    -- builds the chance-level baseline ("null") for jPCA. It randomly
                          shuffles which trial belongs to which reach direction, so any
                          real link between condition and direction is destroyed, but each
                          trial's actual spike data is untouched (so each neuron still
                          fires at its normal overall rate). This is a shuffle done during
                          ANALYSIS, after training. It is a different thing from the
                          "spike-shuffled plasticity" control, which is a separate training
                          run where spikes were shuffled DURING simulation. The two should
                          not be confused with each other.

  trial_split_indices  -- a general-purpose check against overfitting. It splits the
                          trials into two halves (folds), keeping the same number of
                          trials per condition in each half where possible. You compute
                          your geometry measure on one half and check it still holds on
                          the other half. Real structure in the data should show up in
                          both halves; something that only shows up because you averaged
                          a few noisy trials together will not.
"""

import numpy as np


def condition_shuffle(snapshot, rng):
    """
    Return a shallow copy of the snapshot, but with the trial labels (which direction
    each trial belongs to) randomly shuffled. The number of trials per direction stays
    the same (still 5 each), but which trial goes with which direction is now random.
    This wipes out any real condition-related structure while leaving each neuron's
    overall firing rate unchanged.
    """
    shuffled = dict(snapshot)
    labels = np.asarray(snapshot['trial_labels'])
    shuffled['trial_labels'] = rng.permutation(labels)
    return shuffled


def trial_split_indices(trial_labels, rng):
    """
    Split trial indices into two groups (folds), keeping each condition (direction)
    balanced across the two folds.

    With 5 trials per condition, each condition gets split 2 vs 3 (which trials go in
    which fold is picked randomly each time this is called). Returns (idx_a, idx_b) as
    arrays of integer indices. If a condition only has 1 trial, that trial goes into
    fold A and fold B ends up with no trials for that condition. So if you plan to
    average over conditions using both folds, you need at least 2 trials per condition
    (the real data has 5, so this is fine in practice).
    """
    labels = np.asarray(trial_labels)
    idx_a, idx_b = [], []
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        members = rng.permutation(members)
        half = len(members) // 2
        idx_b.extend(members[:half])      # smaller fold
        idx_a.extend(members[half:])      # larger fold
    return np.sort(np.array(idx_a, dtype=int)), np.sort(np.array(idx_b, dtype=int))
