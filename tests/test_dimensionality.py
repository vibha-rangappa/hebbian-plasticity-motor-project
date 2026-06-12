# tests/test_dimensionality.py

import numpy as np
import pytest

from geometry.dimensionality import participation_ratio, eigenspectrum, _data_matrix


def _X_from_data_matrix(D, C):
    """Inverse of _data_matrix: (M=T*C, N) -> (N, T, C), for building test inputs."""
    M, N = D.shape
    T = M // C
    return np.transpose(D.reshape(T, C, N), (2, 0, 1))


def test_pr_single_mode_is_one():
    """All variance in one direction => PR = 1."""
    rng = np.random.default_rng(0)
    M, N, C = 40, 8, 4
    scores = rng.normal(size=(M, 1))          # one latent
    loading = rng.normal(size=(1, N))
    D = scores @ loading
    X = _X_from_data_matrix(D, C)
    assert participation_ratio(X) == pytest.approx(1.0, abs=1e-6)


def test_pr_isotropic_equals_rank():
    """k equal-variance orthogonal modes => PR = k."""
    M, N, C = 40, 12, 4
    # Build D with exactly 5 equal singular values via orthonormal columns.
    k = 5
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.normal(size=(M, k)))   # M x k orthonormal
    V, _ = np.linalg.qr(rng.normal(size=(N, k)))   # N x k orthonormal
    D = Q @ (np.eye(k) * 3.0) @ V.T                # all singular values equal (=3)
    X = _X_from_data_matrix(D, C)
    assert participation_ratio(X) == pytest.approx(float(k), abs=1e-6)


def test_pr_between_one_and_rank():
    """A graded spectrum gives PR strictly between 1 and the number of modes."""
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
    """Gram-matrix spectrum == covariance spectrum (nonzero part)."""
    rng = np.random.default_rng(3)
    M, N, C = 24, 6, 4
    D = rng.normal(size=(M, N))
    D = D - D.mean(axis=0)  # center, as the real pipeline does
    X = _X_from_data_matrix(D, C)
    w_gram = eigenspectrum(X)
    cov = D.T @ D
    w_cov = np.sort(np.linalg.eigvalsh(cov))[::-1]
    # Compare the top N eigenvalues (gram has M, covariance has N; shared nonzero part).
    np.testing.assert_allclose(np.sort(w_gram)[::-1][:N], w_cov, atol=1e-8)


def test_pr_zero_variance_is_nan():
    X = np.zeros((5, 4, 3))
    assert np.isnan(participation_ratio(X))
