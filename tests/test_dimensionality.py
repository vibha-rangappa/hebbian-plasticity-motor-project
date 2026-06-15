# tests/test_dimensionality.py
#
# Tests for geometry/dimensionality.py, which computes the participation ratio (PR,
# a single number summarizing how many "effective dimensions" the population activity
# uses) and the underlying eigenspectrum (variance carried by each dimension). These
# tests build small data matrices where we know the answer ahead of time (one mode,
# several equal-variance modes, a graded spectrum, all-zero data) and check that PR
# and the eigenspectrum come out matching what the math predicts.

import numpy as np
import pytest

from geometry.dimensionality import participation_ratio, eigenspectrum, _data_matrix


def _X_from_data_matrix(D, C):
    """Goes the opposite direction from _data_matrix: turns a (M=T*C, N) matrix back
    into an (N, T, C) array, so we can build simple test inputs and convert them into
    the shape the real functions expect."""
    M, N = D.shape
    T = M // C
    return np.transpose(D.reshape(T, C, N), (2, 0, 1))


def test_pr_single_mode_is_one():
    """If all the variance in the data comes from just one underlying pattern (one
    "mode"), the participation ratio should be exactly 1, since there is only one
    effective dimension."""
    rng = np.random.default_rng(0)
    M, N, C = 40, 8, 4
    scores = rng.normal(size=(M, 1))          # one latent
    loading = rng.normal(size=(1, N))
    D = scores @ loading
    X = _X_from_data_matrix(D, C)
    assert participation_ratio(X) == pytest.approx(1.0, abs=1e-6)


def test_pr_isotropic_equals_rank():
    """If the data has exactly k independent directions (modes) that all carry equal
    variance, the participation ratio should equal k. Here we build data with exactly
    5 equal-sized modes (singular value 3.0 each) and expect PR = 5."""
    M, N, C = 40, 12, 4
    # Build D with exactly 5 equal singular values, using orthonormal (independent) columns.
    k = 5
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.normal(size=(M, k)))   # M x k orthonormal
    V, _ = np.linalg.qr(rng.normal(size=(N, k)))   # N x k orthonormal
    D = Q @ (np.eye(k) * 3.0) @ V.T                # all singular values equal (=3)
    X = _X_from_data_matrix(D, C)
    assert participation_ratio(X) == pytest.approx(float(k), abs=1e-6)


def test_pr_between_one_and_rank():
    """If the data has 6 modes but they carry unequal amounts of variance (a "graded"
    spectrum, here singular values 10, 5, 2, 1, 0.5, 0.1), the participation ratio
    should land strictly between 1 (all variance in one mode) and 6 (all modes equal).
    This is the realistic in-between case."""
    M, N, C = 40, 12, 4
    k = 6
    rng = np.random.default_rng(2)
    Q, _ = np.linalg.qr(rng.normal(size=(M, k)))
    V, _ = np.linalg.qr(rng.normal(size=(N, k)))
    svals = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1])
    D = Q @ np.diag(svals) @ V.T
    X = _X_from_data_matrix(D, C)
    pr = participation_ratio(X)
    assert 1.0 < pr < k


def test_eigenspectrum_matches_covariance_eigs():
    """The eigenspectrum function works on a "Gram matrix" (a TC x TC matrix), but the
    more familiar covariance matrix is N x N. Mathematically, the nonzero eigenvalues
    of these two matrices should match. This test checks that the top N eigenvalues
    from eigenspectrum agree with the eigenvalues computed directly from the N x N
    covariance matrix."""
    rng = np.random.default_rng(3)
    M, N, C = 24, 6, 4
    D = rng.normal(size=(M, N))
    D = D - D.mean(axis=0)  # center the data, same as the real analysis pipeline does
    X = _X_from_data_matrix(D, C)
    w_gram = eigenspectrum(X)
    cov = D.T @ D
    w_cov = np.sort(np.linalg.eigvalsh(cov))[::-1]
    # Compare the top N eigenvalues (the Gram matrix has M of them, the covariance
    # matrix has N; only the top N are shared and nonzero).
    np.testing.assert_allclose(np.sort(w_gram)[::-1][:N], w_cov, atol=1e-8)


def test_pr_zero_variance_is_nan():
    """If the data is all zeros, there is no variance at all, so the participation
    ratio is undefined (0/0). The function should return NaN (not a number) rather
    than crashing or returning a misleading 0."""
    X = np.zeros((5, 4, 3))
    assert np.isnan(participation_ratio(X))
