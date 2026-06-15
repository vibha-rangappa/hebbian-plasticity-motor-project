# geometry/dimensionality.py

"""
This file computes the "participation ratio" (PR), which is our measure of effective
dimensionality (spec section 2). It is one of the three main observables we track.

    PR = (sum_i lambda_i)^2 / sum_i (lambda_i^2)

where lambda_i are the eigenvalues of the population covariance matrix. PR tells you how
many dimensions the population activity is actually spread across: PR = 1 means all the
variance is in a single mode (everything moves together along one axis), PR = N means the
variance is spread evenly across all N modes.

Instead of building the full neuron-by-neuron covariance matrix, we compute the same
eigenvalues from the "gram matrix" (sample-by-sample, D @ D.T). Here N = 800 neurons but
only M = T*C = 400 samples (T timepoints times C conditions), so the N x N covariance
matrix would be rank-deficient (most of its eigenvalues would be exactly zero) and slower
to compute. The M x M gram matrix has the same nonzero eigenvalues, so we use that
instead. The usual 1/(M-1) scaling factor for covariance matrices cancels out in the PR
ratio anyway, so we don't bother applying it.

One caveat about sample size: because we only have M samples, and mean-subtraction at
each timepoint removes one degree of freedom, PR will always come out well below N, and
the raw number is biased low. So don't read too much into the absolute PR value, look at
how it changes (across training epochs, or seeded vs control conditions) instead.
"""

import numpy as np


def _data_matrix(X):
    """Reshape the centered (N, T, C) array into an (M=T*C, N) sample-by-neuron matrix."""
    N, T, C = X.shape
    # For each condition c, we get a (T, N) block of activity over time. Stack all
    # conditions on top of each other along the sample axis to get one big data matrix.
    return np.transpose(X, (1, 2, 0)).reshape(T * C, N)


def eigenspectrum(X):
    """
    Compute the eigenvalues of the population covariance matrix (via the gram matrix
    trick described above), sorted from largest to smallest. Eigenvalues can't be
    negative in theory, but tiny negative values can appear from floating-point
    rounding error, so those get clipped to 0.
    """
    D = _data_matrix(X)
    G = D @ D.T                      # (M, M) gram matrix, has the same nonzero eigenvalues as the covariance
    w = np.linalg.eigvalsh(G)        # eigenvalues in ascending order, real since G is symmetric and positive semi-definite
    w = np.clip(w, 0.0, None)[::-1]  # flip to descending order and clip tiny negatives to 0
    return w


def participation_ratio(X):
    """
    Compute PR for the centered data matrix X (shape N, T, C). Returns a single float.
    If X has no variance at all (every eigenvalue is zero), returns NaN instead of
    dividing by zero.
    """
    w = eigenspectrum(X)
    s1 = w.sum()
    s2 = (w ** 2).sum()
    if s2 == 0.0:
        return float('nan')
    return float(s1 * s1 / s2)
