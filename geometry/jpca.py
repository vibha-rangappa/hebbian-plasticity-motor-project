# geometry/jpca.py

"""
jPCA analysis: checks whether the population activity rotates in a structured way
over time, the way Churchland et al. (2012) found in motor cortex during reaching
(spec section 3).

The idea: take the population trajectory in PC space, and ask "is there a single
rotation (a skew-symmetric linear dynamical system) that predicts how the state
changes from one timestep to the next, across all conditions?" If yes, that's a
genuine rotation. We also run three extra checks ("Lebedev triangulation guards":
rotation-direction consistency, a spike-shuffle chance floor, and trajectory
tangling) to make sure what we're seeing is a real rotation and not just an artifact
of, say, neurons firing in a fixed sequence one after another.

R^2 convention (IMPORTANT): the main jPCA R^2 number reported here is the fraction of
the total variance in the trajectory's derivative (velocity) that the rotation
explains:

    R2_skew = 1 - ||dX - X @ M_skew.T||^2 / ||dX - mean(dX)||^2

This is the same convention Churchland et al. (2012) used: it's close to 1 for a
genuine rotation and close to 0 if the dynamics aren't rotational. Note this is
different from the literal formula written in the original Part 3 brief, which
divided by the residual of the unconstrained fit instead. That version can't work
mathematically: since the rotation-only fit is a restricted special case of the
unconstrained fit, its residual is always at least as large, so that ratio is always
>= 1 and the resulting "R^2" would always be <= 0, it could never be "near 1 for
rotation". Instead, we separately report the gap between the unconstrained R^2 and
the rotation-only R^2 (R2_full - R2_skew) as a diagnostic of how much we "give up" by
forcing the dynamics to be a pure rotation.
"""

import numpy as np

from geometry._linalg import robust_svd


def project_pcs(X, k):
    """
    Project the centered data matrix X (N, T, C) onto its top k principal
    components (PCs), to reduce from N neurons down to k dimensions.

    Returns:
        Z : (T, C, k) each condition's trajectory, in PC space
        V : (N, k) the PC directions, expressed as weights on each neuron
    """
    N, T, C = X.shape
    D = np.transpose(X, (1, 2, 0)).reshape(T * C, N)     # (M, N)
    # Run SVD on the reshaped data. Its right singular vectors give us the PC
    # directions in neuron space directly, no separate covariance matrix needed.
    U, S, Vt = robust_svd(D, full_matrices=False)
    V = Vt[:k].T                                          # (N, k)
    scores = D @ V                                        # (M, k)
    Z = scores.reshape(T, C, k)
    return Z, V


def state_and_derivative(Z):
    """
    Compute the trajectory's velocity (derivative) at each timepoint using a
    central difference (the average of the step before and the step after). The
    first and last timepoint of each condition are dropped since they don't have
    both neighbors.

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
    """Fit the best linear map M_full (no constraints) so that dX ~ X @ M_full.T.
    This is the unconstrained baseline we compare the rotation-only fit against."""
    B, *_ = np.linalg.lstsq(X_all, dX_all, rcond=None)   # solves X_all @ B = dX_all
    return B.T                                           # M_full = B.T


def fit_skew(X_all, dX_all):
    """
    Find the best skew-symmetric matrix M (meaning M = -M.T, so it represents a pure
    rotation with no stretching or shrinking) such that dX ~ X @ M.T. We do this as a
    closed-form linear least-squares fit over the k(k-1)/2 free parameters in the
    upper triangle of M (the lower triangle is just the negative, by definition of
    skew-symmetric).

    Here's the trick: writing P = X @ M.T = -X @ M, and using M[a,b]=p_ab,
    M[b,a]=-p_ab for a<b, you can show that
        P[s, a] = sum_{b>a} X[s,b] p_ab  -  sum_{b<a} X[s,b] p_ba
    In other words, each free parameter p_(i,j) contributes +X[:,j] to output
    dimension i, and -X[:,i] to output dimension j. We build a big design matrix
    encoding exactly that relationship, then solve for all the p_(i,j) at once with
    a standard least-squares solve.
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
    """Compute R^2 = 1 - ||dX - X@M.T||^2 / ||dX - mean(dX)||^2, i.e. how much of the
    velocity variance the fitted matrix M explains (Churchland et al. 2012 convention)."""
    resid = dX_all - X_all @ M.T
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((dX_all - dX_all.mean(axis=0)) ** 2).sum())
    if ss_tot == 0.0:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def dominant_plane(M_skew):
    """
    Find the dominant rotation plane: a skew-symmetric matrix's eigenvalues always
    come in +/- i*omega conjugate pairs (where omega is a rotation frequency), each
    pair corresponding to a 2D plane of rotation. We pick the pair with the largest
    |omega|, i.e. the fastest/strongest rotation.

    Returns (omega, plane) where omega is in radians per time-bin, and plane is an
    orthonormal (k, 2) basis (two perpendicular unit vectors in k-dim PC space)
    spanning that rotation plane.
    """
    evals, evecs = np.linalg.eig(M_skew)
    omegas = np.abs(evals.imag)
    idx = int(np.argmax(omegas))
    omega = float(omegas[idx])
    v = evecs[:, idx]
    plane = np.column_stack([v.real, v.imag])
    # Make the two plane axes orthonormal (perpendicular, unit length).
    q, _ = np.linalg.qr(plane)
    return omega, q[:, :2]


def rotation_direction_signs(Z, plane):
    """
    Work out which way (clockwise or counterclockwise) each condition's trajectory
    rotates within the dominant plane.

    For each condition, project its trajectory onto the 2D plane and add up the
    signed area it sweeps out, sum_t (x*dy - y*dx). The sign of that total tells you
    the rotation direction. Returns an array of +1, -1, or 0 (one value per
    condition).
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
    Fraction of conditions that rotate in the same direction as the majority.
    1.0 means every condition rotates the same way, which is what you'd expect for a
    genuine shared rotation (this is one of the Lebedev triangulation guards).
    """
    signs = rotation_direction_signs(Z, plane)
    nonzero = signs[signs != 0]
    if len(nonzero) == 0:
        return 0.0
    majority = np.sign(nonzero.sum()) or 1.0
    return float(np.mean(nonzero == majority))


def mean_tangling(X_all, dX_all, eps_frac=0.1):
    """
    Trajectory tangling (Russo et al. 2018): a check for whether nearby points in
    state space have wildly different velocities.

        Q(t) = max_{t'} ||dX(t) - dX(t')||^2 / (||X(t) - X(t')||^2 + eps)

    For each timepoint t, we look at the other timepoint t' whose state is most
    similar to t but whose velocity is most different, and compute that ratio. High
    tangling means two nearby states have very different velocities, which can't
    happen in a clean autonomous dynamical system (a rotation, say), it's the
    signature of a feedforward sequence instead (neurons just firing one after
    another in a fixed order). eps is a small fraction of the total state variance,
    added to the denominator so we don't divide by something near zero.

    Returns the average of Q(t) over all timepoints.
    """
    eps = eps_frac * float((X_all ** 2).sum(axis=1).mean())
    dx2 = ((dX_all[:, None, :] - dX_all[None, :, :]) ** 2).sum(-1)   # (M', M')
    x2 = ((X_all[:, None, :] - X_all[None, :, :]) ** 2).sum(-1)
    Q = (dx2 / (x2 + eps)).max(axis=1)
    return float(Q.mean())


def jpca_analysis(X, k=6, eps_frac=0.1):
    """
    Run the full jPCA pipeline on a centered exec-window data matrix X (N, T, C).

    Returns a dict:
        r2_skew      : main jPCA R^2 (how rotational the dynamics are), Churchland
                       total-variance convention
        r2_full      : R^2 of the unconstrained linear fit (no rotation constraint)
        skew_gap     : r2_full - r2_skew (how much we lose by forcing a rotation)
        omega        : dominant rotation frequency (radians per time-bin)
        direction_consistency : fraction of conditions rotating the same way (1.0 = all)
        mean_tangling: mean trajectory tangling (see mean_tangling above)
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
