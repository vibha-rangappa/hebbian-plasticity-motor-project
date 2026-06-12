# geometry/orthogonality.py

"""
Preparatory/execution subspace orthogonality (spec section 4).

Elsayed et al. (2016) showed prep and movement activity in M1 occupy near-orthogonal
subspaces (the output-null hypothesis, Kaufman et al. 2014). We take the top-6 PC
subspace of the prep-window activity and of the exec-window activity, then measure the
principal angles between them.

The raw mean angle is meaningless on its own: random 6-D subspaces in N=800-D space
already sit near 90 degrees purely by dimensionality. So every observed angle is reported
against a bootstrap random-subspace null. "Orthogonal beyond chance" means the observed
mean angle is NOT below the null (i.e. the subspaces are at least as orthogonal as random
-- the interesting direction is when learning pushes them toward, and not away from,
orthogonality relative to that baseline).
"""

import numpy as np
from scipy.linalg import subspace_angles


def top_pc_basis(X, k=6):
    """
    Orthonormal basis (N, k) for the top-k PC subspace of centered data X (N, T, C).
    Columns are the leading left singular vectors of the (N, M) neuron-by-sample matrix.
    """
    N, T, C = X.shape
    D = X.reshape(N, T * C)               # neurons x samples
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    return U[:, :k]


def principal_angles(basis_a, basis_b):
    """Principal angles (radians, ascending) between two subspaces given as N x k bases."""
    return subspace_angles(basis_a, basis_b)


def mean_principal_angle(X_prep, X_exec, k=6):
    """Mean principal angle (radians) between the prep and exec top-k PC subspaces."""
    Pa = top_pc_basis(X_prep, k)
    Pe = top_pc_basis(X_exec, k)
    return float(np.mean(principal_angles(Pa, Pe)))


def random_subspace_null(n_neurons, k=6, n_boot=1000, seed=0):
    """
    Bootstrap null for the mean principal angle between two independent random k-D
    subspaces in n_neurons-dimensional space. Returns the array of mean angles (radians).
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        A, _ = np.linalg.qr(rng.normal(size=(n_neurons, k)))
        B, _ = np.linalg.qr(rng.normal(size=(n_neurons, k)))
        out[b] = np.mean(subspace_angles(A[:, :k], B[:, :k]))
    return out


def orthogonality_analysis(X_prep, X_exec, k=6, n_boot=1000, seed=0):
    """
    Returns a dict:
        mean_angle_rad / mean_angle_deg : observed prep-vs-exec mean principal angle
        null_mean_deg, null_std_deg     : random-subspace null (same N, k)
        z_vs_null                       : (observed - null_mean) / null_std
        principal_angles_deg            : all k angles, ascending
    """
    N = X_prep.shape[0]
    obs = mean_principal_angle(X_prep, X_exec, k)
    angles = np.degrees(principal_angles(top_pc_basis(X_prep, k), top_pc_basis(X_exec, k)))
    null = random_subspace_null(N, k=k, n_boot=n_boot, seed=seed)
    null_mean, null_std = float(null.mean()), float(null.std())
    obs_deg = np.degrees(obs)
    z = (obs - null_mean) / null_std if null_std > 0 else float('nan')
    return {
        'mean_angle_rad': obs,
        'mean_angle_deg': obs_deg,
        'null_mean_deg': np.degrees(null_mean),
        'null_std_deg': np.degrees(null_std),
        'z_vs_null': float(z),
        'principal_angles_deg': angles,
        'k': k,
    }
