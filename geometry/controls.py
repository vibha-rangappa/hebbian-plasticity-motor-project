# geometry/controls.py

"""
Shared null / cross-validation machinery used by every observable (spec sections 3, 5).
Kept in one place so the nulls are defined once, not re-implemented per observable.

Two facilities:

  condition_shuffle    -- the jPCA chance floor. Permutes trial->direction labels,
                          breaking the condition/direction mapping while preserving each
                          trial's spike content (so each neuron's marginal firing is
                          intact). This is the ANALYSIS-level shuffle; it is distinct
                          from the simulation-level "spike-shuffled plasticity" control
                          (a separate training run), which this project does not conflate.

  trial_split_indices  -- the universal overfitting guard. Splits trials into two folds,
                          balanced per condition, so geometry estimated on one fold can be
                          measured on the held-out fold. Real structure generalizes across
                          the split; artifacts of averaging few noisy trials do not.
"""

import numpy as np


def condition_shuffle(snapshot, rng):
    """
    Return a shallow copy of the snapshot with trial_labels randomly permuted. A
    permutation preserves the per-direction trial count (still 5 each) but scrambles
    which trials belong to which direction, destroying genuine cross-condition geometry
    while leaving marginal firing statistics untouched.
    """
    shuffled = dict(snapshot)
    labels = np.asarray(snapshot['trial_labels'])
    shuffled['trial_labels'] = rng.permutation(labels)
    return shuffled


def trial_split_indices(trial_labels, rng):
    """
    Split trial indices into two folds, balanced within each condition.

    With 5 trials/condition the split is 2 vs 3 per condition (assigned at random each
    call). Returns (idx_a, idx_b) as int arrays. Conditions with a single trial put it in
    fold A and leave fold B without that condition -- callers using both folds for a
    condition-average should use >= 2 trials/condition (the real data has 5).
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
