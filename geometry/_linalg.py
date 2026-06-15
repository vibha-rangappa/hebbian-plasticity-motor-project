# geometry/_linalg.py

"""
A small helper that wraps numpy's SVD (singular value decomposition) so it doesn't
crash on tricky inputs. The rest of the geometry code uses SVD a lot (it's the core
of PCA), so this one fix lives here and everything else just imports it.

numpy's default SVD uses a fast method ('gesdd') that can occasionally fail to
converge on awkward inputs, for example a near-silent, low-rank firing-rate matrix
from a frozen exec window, or a small trial-split fold with very few trials. If that
happens, we just retry with a slower but more robust method ('gesvd'), which handles
these cases fine.
"""

import numpy as np
import scipy.linalg


def robust_svd(D, full_matrices=False):
    """Same as np.linalg.svd, but if that fails to converge, retry with the slower,
    more robust 'gesvd' driver instead of crashing."""
    try:
        return np.linalg.svd(D, full_matrices=full_matrices)
    except np.linalg.LinAlgError:
        return scipy.linalg.svd(D, full_matrices=full_matrices, lapack_driver='gesvd')
