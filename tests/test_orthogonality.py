# tests/test_orthogonality.py
#
# Tests for geometry/orthogonality.py, which checks how aligned or "orthogonal"
# (independent) the preparatory (prep) and execution (exec) activity subspaces are,
# following the prep/exec subspace logic of Elsayed et al. 2016. The main tool is the
# "principal angle" between the top-PC subspaces of prep and exec activity: 0 degrees
# means the subspaces are identical, 90 degrees means they share no common directions.
# A "null" distribution from random subspaces gives a baseline for what angle would
# happen just by chance. These tests check the building blocks (PC basis, principal
# angle, random null) on cases with a known right answer, and then check that the full
# orthogonality_analysis correctly tells apart a task design where prep and exec share
# a direction code (aligned, the current task's regime) from one with an output-null
# constraint (orthogonal, Kaufman et al. 2014 / Elsayed et al. 2016).

import numpy as np
import pytest

from geometry.orthogonality import (
    top_pc_basis,
    mean_principal_angle,
    random_subspace_null,
    orthogonality_analysis,
)
from geometry.synthetic import (
    make_shared_code_subspaces,
    make_output_null_subspaces,
)


def test_identical_subspaces_zero_angle():
    """If prep and exec activity are exactly the same data, their subspaces should be
    perfectly aligned, so the principal angle between them should be 0."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 10, 4))
    angle = mean_principal_angle(X, X, k=6)
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_disjoint_neuron_subspaces_are_orthogonal():
    """If prep activity only involves one set of neurons and exec activity only
    involves a completely different, non-overlapping set of neurons, the two
    subspaces share no directions at all, so the principal angle should be 90 degrees
    (fully orthogonal)."""
    N, T, C, k = 20, 10, 4, 6
    rng = np.random.default_rng(1)
    X_prep = np.zeros((N, T, C))
    X_exec = np.zeros((N, T, C))
    X_prep[0:k] = rng.normal(size=(k, T, C))     # only neurons 0..5 are active in prep
    X_exec[k:2 * k] = rng.normal(size=(k, T, C))  # only neurons 6..11 are active in exec (disjoint from prep)
    angle = np.degrees(mean_principal_angle(X_prep, X_exec, k=k))
    assert angle == pytest.approx(90.0, abs=1e-6)


def test_top_pc_basis_orthonormal():
    """The top-PC basis (the top k principal component directions) should have the
    right shape (15 neurons x 6 components) and should be orthonormal, meaning its
    columns are at right angles to each other and each has length 1. That's checked by
    confirming B^T @ B equals the identity matrix."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(15, 8, 3))
    B = top_pc_basis(X, k=6)
    assert B.shape == (15, 6)
    np.testing.assert_allclose(B.T @ B, np.eye(6), atol=1e-9)


def test_random_subspace_null_range():
    """For random 6-dimensional subspaces (no real structure, just chance), the
    average principal angle should fall strictly between 0 and 90 degrees, and on
    average should be fairly large (between 40 and 90 degrees), since two random
    subspaces are unlikely to be either perfectly aligned or perfectly orthogonal."""
    null = np.degrees(random_subspace_null(n_neurons=50, k=6, n_boot=100, seed=3))
    assert np.all(null > 0) and np.all(null < 90)
    assert 40.0 < null.mean() < 90.0


def test_orthogonality_analysis_disjoint_above_null():
    """When prep and exec are built to use disjoint sets of neurons (so they should be
    fully orthogonal), the full analysis should report a mean angle of 90 degrees, a
    z-score above 0 (meaning more orthogonal than the random-chance baseline), and an
    array of per-dimension principal angles with shape (k,) = (6,)."""
    N, T, C, k = 60, 10, 4, 6
    rng = np.random.default_rng(4)
    X_prep = np.zeros((N, T, C))
    X_exec = np.zeros((N, T, C))
    X_prep[0:k] = rng.normal(size=(k, T, C))
    X_exec[k:2 * k] = rng.normal(size=(k, T, C))
    out = orthogonality_analysis(X_prep, X_exec, k=k, n_boot=200, seed=5)
    assert out['mean_angle_deg'] == pytest.approx(90.0, abs=1e-4)
    assert out['z_vs_null'] > 0
    assert out['principal_angles_deg'].shape == (k,)


# ---- task-design controls: the analysis must tell apart the two possible regimes ----
# In the real data, prep and exec subspaces come out aligned (not orthogonal). The
# next two tests show this is a property of the MODEL/task design (there's no
# "output-null" constraint forcing them apart), not a blind spot in the analysis
# itself. When a task shares one direction code between prep and exec, the analysis
# should report alignment. When a task is built so prep and exec use separate,
# orthogonal subspaces, the analysis should report ~90 degrees.

def test_shared_direction_code_reads_as_aligned():
    """If prep and exec are built from the same underlying tuning directions (the
    regime used by the current task), the analysis should report them as aligned: a
    mean angle clearly below 45 degrees, and a negative z-score, meaning MORE aligned
    than you'd expect from random chance."""
    X_prep, X_exec = make_shared_code_subspaces(N=80, T=20, C=8, k=6, seed=0)
    out = orthogonality_analysis(X_prep, X_exec, k=6, n_boot=200, seed=1)
    assert out['mean_angle_deg'] < 45.0      # clearly aligned, not orthogonal
    assert out['z_vs_null'] < 0              # more aligned than random chance


def test_output_null_task_reads_as_orthogonal():
    """If prep and exec are built to use separate, orthogonal directions (the
    "output-null" task design), the analysis should report them as close to 90
    degrees apart, with a positive z-score, meaning more orthogonal than random
    chance."""
    X_prep, X_exec = make_output_null_subspaces(N=80, T=20, C=8, k=6, seed=0)
    out = orthogonality_analysis(X_prep, X_exec, k=6, n_boot=200, seed=1)
    assert out['mean_angle_deg'] > 80.0
    assert out['z_vs_null'] > 0
