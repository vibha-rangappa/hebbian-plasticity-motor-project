# tests/test_orthogonality.py

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
    """Prep and exec with identical activity => 0 principal angle (fully aligned)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 10, 4))
    angle = mean_principal_angle(X, X, k=6)
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_disjoint_neuron_subspaces_are_orthogonal():
    """Activity confined to disjoint neuron sets => 90 degree principal angles."""
    N, T, C, k = 20, 10, 4, 6
    rng = np.random.default_rng(1)
    X_prep = np.zeros((N, T, C))
    X_exec = np.zeros((N, T, C))
    X_prep[0:k] = rng.normal(size=(k, T, C))     # neurons 0..5
    X_exec[k:2 * k] = rng.normal(size=(k, T, C))  # neurons 6..11 (disjoint)
    angle = np.degrees(mean_principal_angle(X_prep, X_exec, k=k))
    assert angle == pytest.approx(90.0, abs=1e-6)


def test_top_pc_basis_orthonormal():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(15, 8, 3))
    B = top_pc_basis(X, k=6)
    assert B.shape == (15, 6)
    np.testing.assert_allclose(B.T @ B, np.eye(6), atol=1e-9)


def test_random_subspace_null_range():
    """Null mean angles for random 6-D subspaces lie strictly inside (0, 90) degrees."""
    null = np.degrees(random_subspace_null(n_neurons=50, k=6, n_boot=100, seed=3))
    assert np.all(null > 0) and np.all(null < 90)
    assert 40.0 < null.mean() < 90.0


def test_orthogonality_analysis_disjoint_above_null():
    """Constructed-orthogonal subspaces sit above the random-subspace null (z > 0)."""
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


# ---- task-design controls: the analysis must DISTINGUISH the two regimes ----
# The real-data pilot shows prep/exec are aligned (no orthogonalization). These two
# tests prove that is a property of the MODEL (no output-null constraint), not a blind
# spot of the analysis: when a task shares a direction code the analysis reports
# alignment, and when a task imposes orthogonal prep/exec subspaces it reports ~90 deg.

def test_shared_direction_code_reads_as_aligned():
    """Shared tuning axes (the current task's regime) -> aligned, below the chance null."""
    X_prep, X_exec = make_shared_code_subspaces(N=80, T=20, C=8, k=6, seed=0)
    out = orthogonality_analysis(X_prep, X_exec, k=6, n_boot=200, seed=1)
    assert out['mean_angle_deg'] < 45.0      # clearly aligned, not orthogonal
    assert out['z_vs_null'] < 0              # more aligned than random chance


def test_output_null_task_reads_as_orthogonal():
    """Orthogonal prep/exec axes (an output-null task) -> ~90 deg, above the null."""
    X_prep, X_exec = make_output_null_subspaces(N=80, T=20, C=8, k=6, seed=0)
    out = orthogonality_analysis(X_prep, X_exec, k=6, n_boot=200, seed=1)
    assert out['mean_angle_deg'] > 80.0
    assert out['z_vs_null'] > 0
