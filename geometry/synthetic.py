# geometry/synthetic.py

"""
This file generates fake ("synthetic") datasets with known properties, used to test the
jPCA pipeline before we trust it on real data (spec section 3). The idea is: if we feed
the analysis data where we already know the right answer, we can check that it gives that
answer back.

The two main generators are:

  make_pure_rotation    -- builds trajectories from a system we know rotates (a
                           skew-symmetric dynamical system). This is a genuine rotation:
                           jPCA should find a high R^2, the rotation direction should be
                           the same across all conditions, and the trajectories should
                           have LOW "tangling" (different conditions don't cross paths
                           and reverse direction).

  make_lebedev_sequence -- builds a "traveling bump" sequence: each neuron turns on at a
                           different time, in a fixed order, and each condition just
                           scales this same sequence up or down, with a sign that flips
                           across conditions (Lebedev et al. 2019). When you look at this
                           in PC space it LOOKS like rotation, but it is NOT a true
                           dynamical rotation: the rotation direction is inconsistent
                           across conditions, and the trajectories have HIGH tangling.

The second generator exists to test our "is this really rotation" safety checks
(the triangulation guards). We want to make sure the analysis can correctly call out this
known fake-rotation pattern as not being real rotation. If the analysis can't catch this
known artifact in synthetic data, we can't trust a "yes, it's rotating" answer on real
data either.

Both generators return X with shape (N, T, C) (neurons x timepoints x conditions), already
with the cross-condition mean subtracted at each timepoint, exactly like a real
preprocessed snapshot, so jpca.py treats them the same way it treats real data.
"""

import numpy as np


def _center_conditions(X):
    """Subtract the average across conditions at each timepoint, the same way the real preprocessing pipeline does."""
    return X - X.mean(axis=2, keepdims=True)


def make_pure_rotation(N=30, T=50, C=8, omega=0.15, radius=1.0, seed=0, noise=0.0):
    """
    Build a genuine 2D rotation, embedded inside N neurons. Each of the C conditions
    starts at a different point evenly spaced around a circle, and all conditions rotate
    in the same direction.

    omega is the rotation speed in radians per time-bin. Returns (X, omega).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    phases = 2 * np.pi * np.arange(C) / C
    # z holds the 2D rotating trajectory for each condition c, shape (2, T, C)
    z = np.empty((2, T, C))
    for c in range(C):
        z[0, :, c] = radius * np.cos(omega * t + phases[c])
        z[1, :, c] = radius * np.sin(omega * t + phases[c])
    W, _ = np.linalg.qr(rng.normal(size=(N, 2)))   # random orthonormal map from 2D to N neurons
    X = np.einsum('nk,ktc->ntc', W, z)
    if noise > 0:
        X = X + rng.normal(scale=noise, size=X.shape)
    return _center_conditions(X), omega


def make_inconsistent_rotation(N=30, T=50, C=8, omega=0.15, radius=1.0, seed=0):
    """
    Build a rotation where half the conditions spin one way and the other half spin the
    opposite way (omega has its sign flipped for half the conditions). Each individual
    condition still looks like a clean rotation (low tangling), but there is no single
    shared rotation rule across conditions. This is meant to make the
    rotation-direction-consistency check fail on purpose, so we can confirm that check
    works. It is a different kind of artifact than the feedforward sequence below, which
    is instead caught by the tangling check.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    phases = 2 * np.pi * np.arange(C) / C
    senses = np.where(np.arange(C) < C // 2, 1.0, -1.0)   # first half spins +1, second half spins -1
    z = np.empty((2, T, C))
    for c in range(C):
        z[0, :, c] = radius * np.cos(senses[c] * omega * t + phases[c])
        z[1, :, c] = radius * np.sin(senses[c] * omega * t + phases[c])
    W, _ = np.linalg.qr(rng.normal(size=(N, 2)))
    X = np.einsum('nk,ktc->ntc', W, z)
    return _center_conditions(X)


def _direction_tuned(U, a, g):
    """Build activity that only lives within the directions in U: X[n,t,c] = sum_j U[n,j] a[j,c] g[t]. Result is mean-centered across conditions."""
    X = np.einsum('nj,jc,t->ntc', U, a, g)
    return _center_conditions(X)


def make_shared_code_subspaces(N=60, T=20, C=8, k=6, seed=0):
    """
    Build prep and exec activity that both use the SAME k neural tuning directions
    (a shared "which direction am I reaching" code), so their PC subspaces end up
    aligned with each other. This matches the situation in our current task: the
    preparatory cue and the execution drive both carry the same reach-direction
    information, so the two epochs naturally share a subspace, and there is nothing
    pushing them to become orthogonal. Returns (X_prep, X_exec).
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(N, k)))     # shared set of tuning directions
    g = np.linspace(0.2, 1.0, T)
    X_prep = _direction_tuned(U, rng.normal(size=(k, C)), g)
    X_exec = _direction_tuned(U, rng.normal(size=(k, C)), g)
    return X_prep, X_exec


def make_output_null_subspaces(N=60, T=20, C=8, k=6, seed=0):
    """
    Build prep and exec activity that use DIFFERENT, mutually orthogonal sets of neural
    directions, so their principal angles come out near 90 degrees. This is the
    "output-null" scenario described by Kaufman et al. 2014 and Elsayed et al. 2016,
    representing what you'd see in a task that specifically requires prep and exec
    activity to be kept separate. Returns (X_prep, X_exec).
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(N, 2 * k)))
    U_prep, U_exec = Q[:, :k], Q[:, k:2 * k]         # two non-overlapping sets of directions
    g = np.linspace(0.2, 1.0, T)
    X_prep = _direction_tuned(U_prep, rng.normal(size=(k, C)), g)
    X_exec = _direction_tuned(U_exec, rng.normal(size=(k, C)), g)
    return X_prep, X_exec


def make_lebedev_sequence(N=30, T=50, C=8, width=3.0, seed=0, noise=0.0):
    """
    Build a feedforward "traveling bump" sequence: neuron i has its activity peak at time
    t_i, so as time goes on, the bump of activity sweeps across the population in a fixed
    order. Each condition c then scales this whole sequence up or down by a signed gain
    g_c = cos(2*pi*c/C). Because the gain changes sign across conditions, opposite
    conditions trace out the sequence in opposite directions, so the
    rotation-direction-consistency check should fail. Because the underlying pattern is a
    one-way sweep (not real rotation), the tangling check should also flag it as high
    tangling. This is the "Lebedev artifact" pattern (Lebedev et al. 2019).

    Returns X of shape (N, T, C).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    peak_times = np.linspace(0, T - 1, N)
    bumps = np.exp(-((t[None, :] - peak_times[:, None]) ** 2) / (2 * width ** 2))  # (N, T) activity bump for each neuron over time
    gains = np.cos(2 * np.pi * np.arange(C) / C)                                    # (C,) signed scale factor per condition
    X = bumps[:, :, None] * gains[None, None, :]                                    # (N, T, C) combine bumps and gains
    if noise > 0:
        X = X + rng.normal(scale=noise, size=X.shape)
    return _center_conditions(X)
