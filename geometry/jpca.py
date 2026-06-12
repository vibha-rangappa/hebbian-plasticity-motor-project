# geometry/jpca.py

"""
jPCA rotational-structure analysis (spec section 3).

Finds the skew-symmetric linear dynamical system that best fits the population
trajectory derivative, and reports how rotational the dynamics are -- together with the
three Lebedev triangulation guards (rotation-direction consistency, spike-shuffle floor,
trajectory tangling) that distinguish a genuine rotation from a sequence artifact.

R^2 convention (IMPORTANT): the primary jPCA R^2 here is variance-explained relative to
the TOTAL derivative variance:

    R2_skew = 1 - ||dX - X @ M_skew.T||^2 / ||dX - mean(dX)||^2

This is the Churchland et al. (2012) convention and is ~1 for genuine rotation, ~0 for
non-rotational dynamics. Note this differs from the literal formula in the original Part
3 brief, which divided by the unconstrained-fit residual; because the skew fit is a
constrained subset of the unconstrained fit, that ratio is always >= 1 and the formula
yields <= 0 -- mathematically it cannot be "near 1 for rotation". We instead report the
skew-vs-full gap (R2_full - R2_skew) as a separate diagnostic of how much the rotation
constraint costs.
"""

import numpy as np

from geometry._linalg import robust_svd


def project_pcs(X, k):
    """
    Project the centered data matrix X (N, T, C) onto its top-k PCs.

    Returns:
        Z : (T, C, k) per-condition trajectories in PC space
        V : (N, k) principal axes (neuron-space loadings)
    """
    N, T, C = X.shape
    D = np.transpose(X, (1, 2, 0)).reshape(T * C, N)     # (M, N)
    # Economy SVD; right singular vectors are the PCs in neuron space.
    U, S, Vt = robust_svd(D, full_matrices=False)
    V = Vt[:k].T                                          # (N, k)
    scores = D @ V                                        # (M, k)
    Z = scores.reshape(T, C, k)
    return Z, V


def state_and_derivative(Z):
    """
    Central-difference derivative within each condition; endpoints dropped.

    Z : (T, C, k). Returns X_all (M', k), dX_all (M', k) with M' = (T-2)*C.
    """
    T, C, k = Z.shape
    X_list, dX_list = [], []
    for c in range(C):
        traj = Z[:, c, :]                                # (T, k)
        dtraj = (traj[2:] - traj[:-2]) / 2.0             # (T-2, k)
        X_list.append(traj[1:-1])
        dX_list.append(dtraj)
    return np.concatenate(X_list, 0), np.concatenate(dX_list, 0)


def fit_full(X_all, dX_all):
    """Unconstrained least-squares M_full with dX ~ X @ M_full.T."""
    B, *_ = np.linalg.lstsq(X_all, dX_all, rcond=None)   # solves X_all @ B = dX_all
    return B.T                                           # M_full = B.T


def fit_skew(X_all, dX_all):
    """
    Best skew-symmetric M (M = -M.T) with dX ~ X @ M.T, via the closed-form linear
    least-squares over the k(k-1)/2 upper-triangle parameters.

    Prediction P = X @ M.T = -X @ M, and for a<b: M[a,b]=p_ab, M[b,a]=-p_ab. Then
        P[s, a] = sum_{b>a} X[s,b] p_ab  -  sum_{b<a} X[s,b] p_ba
    so each parameter p_(i,j) loads onto output dim i with +X[:,j] and onto output dim j
    with -X[:,i]. We stack that into a design matrix and solve with lstsq.
    """
    M_, k = X_all.shape
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    n_params = len(pairs)
    Phi = np.zeros((M_ * k, n_params))
    for q, (i, j) in enumerate(pairs):
        Phi[i::k, q] += X_all[:, j]      # rows for output dim a=i
        Phi[j::k, q] += -X_all[:, i]     # rows for output dim a=j
    target = dX_all.reshape(-1)          # row-major (s, a) matches Phi's i::k indexing
    p, *_ = np.linalg.lstsq(Phi, target, rcond=None)
    M = np.zeros((k, k))
    for q, (i, j) in enumerate(pairs):
        M[i, j] = p[q]
        M[j, i] = -p[q]
    return M


def r2_variance_explained(X_all, dX_all, M):
    """R^2 = 1 - ||dX - X@M.T||^2 / ||dX - mean(dX)||^2 (Churchland convention)."""
    resid = dX_all - X_all @ M.T
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((dX_all - dX_all.mean(axis=0)) ** 2).sum())
    if ss_tot == 0.0:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def dominant_plane(M_skew):
    """
    Dominant rotation: eigendecompose the skew matrix (eigenvalues +/- i*omega) and take
    the conjugate pair with the largest |omega|.

    Returns (omega, plane) where omega is in radians per time-bin and plane is an
    orthonormal (k, 2) basis spanning the rotation plane.
    """
    evals, evecs = np.linalg.eig(M_skew)
    omegas = np.abs(evals.imag)
    idx = int(np.argmax(omegas))
    omega = float(omegas[idx])
    v = evecs[:, idx]
    plane = np.column_stack([v.real, v.imag])
    # Orthonormalize the 2D plane basis.
    q, _ = np.linalg.qr(plane)
    return omega, q[:, :2]


def rotation_direction_signs(Z, plane):
    """
    Signed rotation sense of each condition in the given plane.

    For each condition, project its trajectory onto the 2D plane and accumulate the
    signed area swept, sum_t (x*dy - y*dx). The sign is the rotation direction. Returns
    an array of +/-1 (or 0) per condition.
    """
    T, C, k = Z.shape
    signs = np.zeros(C)
    for c in range(C):
        p = Z[:, c, :] @ plane                # (T, 2)
        x, y = p[:, 0], p[:, 1]
        dx = np.gradient(x)
        dy = np.gradient(y)
        signed_area = np.sum(x * dy - y * dx)
        signs[c] = np.sign(signed_area)
    return signs


def rotation_consistency(Z, plane):
    """
    Fraction of conditions sharing the majority rotation direction. 1.0 = all conditions
    rotate the same way (a requirement for a genuine rotation; the Lebedev guard).
    """
    signs = rotation_direction_signs(Z, plane)
    nonzero = signs[signs != 0]
    if len(nonzero) == 0:
        return 0.0
    majority = np.sign(nonzero.sum()) or 1.0
    return float(np.mean(nonzero == majority))


def mean_tangling(X_all, dX_all, eps_frac=0.1):
    """
    Trajectory tangling (Russo et al. 2018):

        Q(t) = max_{t'} ||dX(t) - dX(t')||^2 / (||X(t) - X(t')||^2 + eps)

    High tangling = nearby states with very different velocities = inconsistent with an
    autonomous dynamical system (the signature of a feedforward sequence). eps is a small
    fraction of the total state variance, stabilizing the denominator.

    Returns the mean of Q(t) over timepoints.
    """
    eps = eps_frac * float((X_all ** 2).sum(axis=1).mean())
    dx2 = ((dX_all[:, None, :] - dX_all[None, :, :]) ** 2).sum(-1)   # (M', M')
    x2 = ((X_all[:, None, :] - X_all[None, :, :]) ** 2).sum(-1)
    Q = (dx2 / (x2 + eps)).max(axis=1)
    return float(Q.mean())


def jpca_analysis(X, k=6, eps_frac=0.1):
    """
    Full jPCA on a centered exec-window data matrix X (N, T, C).

    Returns a dict:
        r2_skew      : primary jPCA R^2 (rotation), Churchland total-variance convention
        r2_full      : R^2 of the unconstrained linear fit
        skew_gap     : r2_full - r2_skew (cost of the rotation constraint)
        omega        : dominant rotation frequency (rad/time-bin)
        direction_consistency : fraction of conditions rotating the same way (1.0 = all)
        mean_tangling: mean trajectory tangling
        k            : number of PCs used
    """
    Z, V = project_pcs(X, k)
    X_all, dX_all = state_and_derivative(Z)
    M_skew = fit_skew(X_all, dX_all)
    M_full = fit_full(X_all, dX_all)
    omega, plane = dominant_plane(M_skew)
    return {
        'r2_skew': r2_variance_explained(X_all, dX_all, M_skew),
        'r2_full': r2_variance_explained(X_all, dX_all, M_full),
        'skew_gap': (r2_variance_explained(X_all, dX_all, M_full)
                     - r2_variance_explained(X_all, dX_all, M_skew)),
        'omega': omega,
        'direction_consistency': rotation_consistency(Z, plane),
        'mean_tangling': mean_tangling(X_all, dX_all, eps_frac=eps_frac),
        'k': k,
    }
