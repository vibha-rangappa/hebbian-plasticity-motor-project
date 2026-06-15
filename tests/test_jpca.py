# tests/test_jpca.py
#
# Tests for geometry/jpca.py, which implements jPCA (Churchland et al. 2012): a method
# for finding rotational structure in population activity. Before trusting any jPCA
# result on real data, the method has to pass a "synthetic validation gate" (see spec
# section 3): it must (a) correctly recover a rotation when one is actually there, and
# (b) NOT be fooled by the kind of feedforward, non-rotational activity sequence
# described by Lebedev et al, which can look rotational by R^2 alone. If jPCA can't
# tell these two cases apart on synthetic data where we know the ground truth, we
# can't trust what it says about the real data either.

import numpy as np
import pytest

from geometry.jpca import (
    fit_skew,
    fit_full,
    r2_variance_explained,
    dominant_plane,
    jpca_analysis,
    project_pcs,
    state_and_derivative,
)
from geometry.synthetic import (
    make_pure_rotation,
    make_lebedev_sequence,
    make_inconsistent_rotation,
)


# ---- closed-form skew fit ----
# A "skew" (skew-symmetric) matrix M satisfies M = -M^T. jPCA fits this kind of matrix
# to the data because skew matrices produce pure rotations (no growth or decay), which
# is the signature jPCA is looking for.

def test_fit_skew_recovers_known_matrix():
    """If we generate the rate-of-change data (dX) using a known skew matrix M_true,
    fitting should recover that same M_true (to high precision). This checks the
    fitting math itself is correct."""
    rng = np.random.default_rng(0)
    k = 6
    A = rng.normal(size=(k, k))
    M_true = A - A.T                       # construct a skew-symmetric matrix
    X_all = rng.normal(size=(200, k))
    dX_all = X_all @ M_true.T
    M_hat = fit_skew(X_all, dX_all)
    np.testing.assert_allclose(M_hat, M_true, atol=1e-8)


def test_fit_skew_output_is_skew_symmetric():
    """No matter what data goes in, fit_skew's output must always be a valid skew
    matrix, i.e. M = -M^T. Here the input data is pure random noise (no real
    structure), but the fitted matrix should still satisfy this property exactly."""
    rng = np.random.default_rng(1)
    X_all = rng.normal(size=(100, 5))
    dX_all = rng.normal(size=(100, 5))
    M = fit_skew(X_all, dX_all)
    np.testing.assert_allclose(M, -M.T, atol=1e-12)


def test_full_fit_recovers_general_matrix():
    """fit_full is the unconstrained version: it should be able to recover ANY matrix
    M_true (not just skew-symmetric ones), given data generated from that matrix."""
    rng = np.random.default_rng(2)
    k = 4
    M_true = rng.normal(size=(k, k))       # a general matrix, not required to be skew
    X_all = rng.normal(size=(300, k))
    dX_all = X_all @ M_true.T
    M_hat = fit_full(X_all, dX_all)
    np.testing.assert_allclose(M_hat, M_true, atol=1e-8)


# ---- fixture 1: a genuine rotation must score high and pass the guards ----
# make_pure_rotation builds synthetic data that really is a rotation in a 2D plane,
# embedded in N neurons, with all 8 conditions rotating the same way. jPCA should
# recognize this for what it is.

def test_pure_rotation_high_r2():
    """For data that is a genuine, noise-free rotation, jPCA's r2_skew (how well a
    pure-rotation model explains the dynamics) should be very high, above 0.95 (out of
    a max of 1.0)."""
    X, _ = make_pure_rotation(N=30, T=50, C=8, omega=0.15, noise=0.0)
    out = jpca_analysis(X, k=2)
    assert out['r2_skew'] > 0.95


def test_pure_rotation_direction_consistent():
    """For a genuine rotation, all 8 conditions rotate in the same direction, so the
    direction_consistency score should be exactly 1.0 (fully consistent)."""
    X, _ = make_pure_rotation(N=30, T=50, C=8, omega=0.15, noise=0.0)
    out = jpca_analysis(X, k=2)
    assert out['direction_consistency'] == pytest.approx(1.0)


def test_pure_rotation_recovers_omega():
    """The synthetic rotation was built with a known angular speed, omega = 0.12
    radians per time bin. After projecting onto the top 2 PCs and fitting the skew
    matrix M, the recovered angular speed (omega_hat) should match the true omega to
    within 0.01."""
    X, omega = make_pure_rotation(N=30, T=50, C=8, omega=0.12, noise=0.0)
    Z, _ = project_pcs(X, k=2)
    X_all, dX_all = state_and_derivative(Z)
    M = fit_skew(X_all, dX_all)
    omega_hat, _ = dominant_plane(M)
    assert omega_hat == pytest.approx(omega, abs=0.01)


# ---- fixture 2: the Lebedev feedforward sequence, caught by TANGLING ----
# make_lebedev_sequence builds a feedforward "traveling bump" pattern (not a real
# rotation) that is known to fool naive jPCA: it gives a deceptively HIGH R^2 (~0.87,
# right around the value seen in real M1 data) and even passes the
# direction-consistency check (because of how its signed gains work out, the
# direction signal ends up sign-independent). So R^2 and the direction guard alone do
# NOT catch this artifact. The "tangling" measure does catch it. This is exactly the
# point made by Russo et al. (2018): high R^2 alone doesn't prove real rotational
# dynamics.

def test_lebedev_sequence_has_deceptively_high_r2():
    """This feedforward (non-rotational) sequence should still produce a high r2_skew,
    above 0.6, fooling a naive "high R^2 means rotation" reading. This is exactly why
    R^2 by itself isn't a sufficient check, and why the tangling guard exists."""
    X = make_lebedev_sequence(N=30, T=50, C=8, width=3.0)
    out = jpca_analysis(X, k=6)
    assert out['r2_skew'] > 0.6   # high enough to fool a naive reading based on R^2 alone


def test_lebedev_sequence_caught_by_tangling():
    """Even though the Lebedev sequence has a deceptively high R^2, its "tangling"
    (a measure of how much the trajectory crosses or doubles back on itself) should be
    more than 3 times higher than for a genuine rotation. This is the check that
    actually catches the artifact."""
    X_seq = make_lebedev_sequence(N=30, T=50, C=8, width=3.0)
    X_rot, _ = make_pure_rotation(N=30, T=50, C=8, omega=0.15, noise=0.0)
    seq = jpca_analysis(X_seq, k=6)
    rot = jpca_analysis(X_rot, k=6)
    assert seq['mean_tangling'] > 3.0 * rot['mean_tangling']


# ---- fixture 3: opposite-rotating conditions, caught by the DIRECTION guard ----

def test_inconsistent_rotation_caught_by_direction_guard():
    """
    Here, half the conditions rotate clockwise and half rotate counterclockwise. Each
    condition on its own still looks like a clean rotation (low tangling), but the
    conditions don't share one common rotational rule. The direction-consistency check
    is designed to catch exactly this case, so direction_consistency should drop below
    0.9 (compared to 1.0 for a fully consistent rotation).
    """
    X = make_inconsistent_rotation(N=30, T=50, C=8, omega=0.15)
    out = jpca_analysis(X, k=2)
    assert out['direction_consistency'] < 0.9


def test_jpca_analysis_keys():
    """Basic check that jpca_analysis returns a dictionary with exactly the expected
    set of result keys: the skew and full-fit R^2 values, the gap between them, the
    recovered rotation speed (omega), the direction-consistency score, the mean
    tangling, and k (the number of PCs used)."""
    X, _ = make_pure_rotation(N=30, T=50, C=8)
    out = jpca_analysis(X, k=6)
    assert set(out.keys()) == {
        'r2_skew', 'r2_full', 'skew_gap', 'omega',
        'direction_consistency', 'mean_tangling', 'k',
    }
