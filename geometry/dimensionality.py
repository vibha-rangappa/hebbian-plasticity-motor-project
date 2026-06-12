# geometry/dimensionality.py

"""
Effective dimensionality via the participation ratio (spec section 2).

    PR = (sum_i lambda_i)^2 / sum_i (lambda_i^2)

where lambda_i are the eigenvalues of the population covariance. PR is the effective
number of dimensions carrying variance: 1 if all variance is in one mode, N if spread
uniformly across N modes.

We compute it in the gram-matrix (sample x sample) space rather than the
neuron x neuron covariance space. With N=800 neurons but only M = T*C = 400 samples,
the N x N covariance is rank-deficient and slow; the M x M gram matrix D @ D.T has the
same nonzero eigenvalues. The 1/(M-1) covariance scaling cancels in the PR ratio, so we
skip it.

Sample-size caveat: with only M samples (and mean-subtraction removing one df per
timepoint), PR is bounded well below N and the absolute value is biased low. Read
relative changes (epoch-to-epoch, seeded-vs-control), not the absolute number.
"""

import numpy as np


def _data_matrix(X):
    """(N, T, C) centered matrix -> (M=T*C, N) sample-by-neuron data matrix."""
    N, T, C = X.shape
    # Each column c is a (T, N) block of states; stack conditions along the sample axis.
    return np.transpose(X, (1, 2, 0)).reshape(T * C, N)


def eigenspectrum(X):
    """
    Nonnegative eigenvalues of the population covariance, via the gram matrix.
    Returned in descending order. Tiny negative values from round-off are clipped to 0.
    """
    D = _data_matrix(X)
    G = D @ D.T                      # (M, M), same nonzero spectrum as the covariance
    w = np.linalg.eigvalsh(G)        # ascending, real (G is symmetric PSD)
    w = np.clip(w, 0.0, None)[::-1]  # descending, nonnegative
    return w


def participation_ratio(X):
    """
    PR for the centered data matrix X (N, T, C). Returns a float; NaN if X has no
    variance (all eigenvalues zero).
    """
    w = eigenspectrum(X)
    s1 = w.sum()
    s2 = (w ** 2).sum()
    if s2 == 0.0:
        return float('nan')
    return float(s1 * s1 / s2)
