# geometry/synthetic.py

"""
Synthetic data fixtures for validating the jPCA pipeline BEFORE trusting it on real data
(spec section 3). Two generators:

  make_pure_rotation   -- trajectories from a known skew-symmetric system. Genuine
                          autonomous rotation: high jPCA R^2, consistent rotation
                          direction across conditions, LOW trajectory tangling.

  make_lebedev_sequence -- a feedforward traveling-bump sequence with signed per-condition
                          gains (Lebedev et al. 2019). Produces a rotational-looking
                          picture but is NOT a dynamical rotation: inconsistent rotation
                          direction across conditions and HIGH tangling.

The point of the second fixture is to prove the triangulation GUARDS work -- that the
analysis can tell a real rotation from the known artifact. If it can't catch the
artifact on synthetic data, no real-data jPCA result is trustworthy.

Both return X of shape (N, T, C), already cross-condition-mean-subtracted, so jpca.py
processes them identically to a real preprocessed snapshot.
"""

import numpy as np


def _center_conditions(X):
    """Subtract the cross-condition mean at each timepoint, as the real pipeline does."""
    return X - X.mean(axis=2, keepdims=True)


def make_pure_rotation(N=30, T=50, C=8, omega=0.15, radius=1.0, seed=0, noise=0.0):
    """
    A genuine 2D rotation embedded in N neurons. Each condition starts at a different
    phase evenly spaced around the circle; all share the same rotation sense.

    omega is in radians per time-bin. Returns (X, omega).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    phases = 2 * np.pi * np.arange(C) / C
    # latent z_c(t): (2, T, C)
    z = np.empty((2, T, C))
    for c in range(C):
        z[0, :, c] = radius * np.cos(omega * t + phases[c])
        z[1, :, c] = radius * np.sin(omega * t + phases[c])
    W, _ = np.linalg.qr(rng.normal(size=(N, 2)))   # orthonormal embedding (N, 2)
    X = np.einsum('nk,ktc->ntc', W, z)
    if noise > 0:
        X = X + rng.normal(scale=noise, size=X.shape)
    return _center_conditions(X), omega


def make_inconsistent_rotation(N=30, T=50, C=8, omega=0.15, radius=1.0, seed=0):
    """
    A rotation where half the conditions rotate one way and half the other (sign-flipped
    omega). Genuinely rotational per-condition (low tangling) but NOT a shared dynamical
    rule -- the rotation-direction-consistency guard must fire. This validates that guard
    against the artifact it is meant to catch (distinct from the feedforward sequence,
    which the tangling guard catches).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    phases = 2 * np.pi * np.arange(C) / C
    senses = np.where(np.arange(C) < C // 2, 1.0, -1.0)   # half +, half -
    z = np.empty((2, T, C))
    for c in range(C):
        z[0, :, c] = radius * np.cos(senses[c] * omega * t + phases[c])
        z[1, :, c] = radius * np.sin(senses[c] * omega * t + phases[c])
    W, _ = np.linalg.qr(rng.normal(size=(N, 2)))
    X = np.einsum('nk,ktc->ntc', W, z)
    return _center_conditions(X)


def _direction_tuned(U, a, g):
    """Activity confined to span(U): X[n,t,c] = sum_j U[n,j] a[j,c] g[t]. Centered."""
    X = np.einsum('nj,jc,t->ntc', U, a, g)
    return _center_conditions(X)


def make_shared_code_subspaces(N=60, T=20, C=8, k=6, seed=0):
    """
    Prep and exec activity built from the SAME neural tuning axes (a shared direction
    code) -> their subspaces are aligned. This is the regime of the CURRENT task: the
    prep cue and the exec drive carry the same direction signal, so the two epochs share
    a subspace and there is no pressure to orthogonalize. Returns (X_prep, X_exec).
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(N, k)))     # shared axes
    g = np.linspace(0.2, 1.0, T)
    X_prep = _direction_tuned(U, rng.normal(size=(k, C)), g)
    X_exec = _direction_tuned(U, rng.normal(size=(k, C)), g)
    return X_prep, X_exec


def make_output_null_subspaces(N=60, T=20, C=8, k=6, seed=0):
    """
    Prep and exec activity in ORTHOGONAL neural subspaces (the output-null regime of
    Kaufman et al. 2014 / Elsayed et al. 2016) -> near-90-degree principal angles. This
    is what a task WITH an output-null constraint would produce. Returns (X_prep, X_exec).
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(N, 2 * k)))
    U_prep, U_exec = Q[:, :k], Q[:, k:2 * k]         # mutually orthogonal axes
    g = np.linspace(0.2, 1.0, T)
    X_prep = _direction_tuned(U_prep, rng.normal(size=(k, C)), g)
    X_exec = _direction_tuned(U_exec, rng.normal(size=(k, C)), g)
    return X_prep, X_exec


def make_lebedev_sequence(N=30, T=50, C=8, width=3.0, seed=0, noise=0.0):
    """
    A feedforward traveling-bump sequence. Neuron i peaks at time t_i (a sweep across the
    population); each condition c scales the whole sequence by a signed gain g_c =
    cos(2*pi*c/C). The signed gains make opposite conditions trace the sequence with
    opposite orientation -> rotation-direction consistency must fail; the feedforward
    structure -> high tangling. This is the Lebedev artifact.

    Returns X of shape (N, T, C).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    peak_times = np.linspace(0, T - 1, N)
    bumps = np.exp(-((t[None, :] - peak_times[:, None]) ** 2) / (2 * width ** 2))  # (N, T)
    gains = np.cos(2 * np.pi * np.arange(C) / C)                                    # (C,)
    X = bumps[:, :, None] * gains[None, None, :]                                    # (N, T, C)
    if noise > 0:
        X = X + rng.normal(scale=noise, size=X.shape)
    return _center_conditions(X)
