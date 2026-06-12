# geometry/_linalg.py

"""
Numerically robust SVD.

numpy's default SVD uses LAPACK's 'gesdd' (divide-and-conquer), which is fast but
occasionally fails to converge on degenerate inputs -- e.g. a near-silent, low-rank
firing-rate matrix from a frozen, autonomous-exec snapshot, or a thin trial-split fold.
When that happens we retry with the slower but more robust 'gesvd' driver, which
converges on these cases.
"""

import numpy as np
import scipy.linalg


def robust_svd(D, full_matrices=False):
    """np.linalg.svd, falling back to LAPACK gesvd on non-convergence."""
    try:
        return np.linalg.svd(D, full_matrices=full_matrices)
    except np.linalg.LinAlgError:
        return scipy.linalg.svd(D, full_matrices=full_matrices, lapack_driver='gesvd')
