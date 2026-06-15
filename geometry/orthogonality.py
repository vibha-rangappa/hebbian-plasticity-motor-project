# geometry/orthogonality.py

"""
This file measures how separate the "preparatory" and "execution" activity patterns are
(spec section 4), our third main observable.

Elsayed et al. (2016) found that in real M1, the activity during movement preparation and
the activity during movement execution live in nearly separate (orthogonal, i.e. roughly
90 degrees apart) subspaces (the "output-null" idea from Kaufman et al. 2014). Here we
take the top 6 principal-component directions of the prep-window activity, and the top 6
of the exec-window activity, and measure the angles between those two 6-dimensional
subspaces (the "principal angles").

On its own, the raw angle doesn't mean much: in a space with N = 800 neurons, two random
6-dimensional subspaces are already close to 90 degrees apart just by chance, simply
because the space is so big. So we always compare the observed angle to a "null"
distribution built from random subspaces of the same size. If the observed angle is at
least as large as (not smaller than) this random baseline, we say the subspaces are
"orthogonal beyond chance." The interesting question is whether learning pushes the prep
and exec subspaces toward this random-like separation, or pulls them closer together
(more aligned) than random.
"""

import numpy as np
from scipy.linalg import subspace_angles

from geometry._linalg import robust_svd


def top_pc_basis(X, k=6):
    """
    Find the top-k principal component directions for the centered data X (shape N, T, C),
    and return them as an orthonormal (N, k) basis (a set of k unit vectors in N-dimensional
    neuron space, all at right angles to each other). These come from the leading left
    singular vectors of the (N, M) neuron-by-sample matrix, where M = T*C.
    """
    N, T, C = X.shape
    D = X.reshape(N, T * C)               # reshape to neurons x samples
    U, S, Vt = robust_svd(D, full_matrices=False)
    return U[:, :k]


def principal_angles(basis_a, basis_b):
    """Compute the principal angles (in radians, smallest first) between two subspaces, each given as an N x k basis."""
    return subspace_angles(basis_a, basis_b)


def mean_principal_angle(X_prep, X_exec, k=6):
    """Compute the average principal angle (in radians) between the prep and exec top-k PC subspaces."""
    Pa = top_pc_basis(X_prep, k)
    Pe = top_pc_basis(X_exec, k)
    return float(np.mean(principal_angles(Pa, Pe)))


def random_subspace_null(n_neurons, k=6, n_boot=1000, seed=0):
    """
    Build a "null" distribution for the mean principal angle by repeatedly drawing two
    random, independent k-dimensional subspaces in n_neurons-dimensional space and
    measuring the angle between them. Returns an array of n_boot mean angles (in
    radians), one per random draw. This tells us what angle to expect by chance alone.
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
    Run the full prep-vs-exec orthogonality analysis and return a dict with:
        mean_angle_rad / mean_angle_deg : the observed mean principal angle between
                                           prep and exec subspaces
        null_mean_deg, null_std_deg     : mean and standard deviation of the
                                           random-subspace null (same N and k)
        z_vs_null                       : how many null-standard-deviations the
                                           observed angle is from the null mean,
                                           i.e. (observed - null_mean) / null_std
        principal_angles_deg            : all k individual angles, smallest first
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
