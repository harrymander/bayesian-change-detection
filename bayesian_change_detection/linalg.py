import numpy as np
from scipy.linalg import lapack

try:
    # matrix_transpose is added in numpy v2
    from numpy import matrix_transpose  # type: ignore
except ImportError:

    def matrix_transpose(a: np.ndarray) -> np.ndarray:  # type: ignore
        return np.swapaxes(a, -1, -2)


def inv_positive_definite(a: np.ndarray) -> np.ndarray:
    squeeze = False
    if a.ndim == 2:
        squeeze = True
        a = np.expand_dims(a, 0)
    elif a.ndim != 3:
        raise ValueError("a must be 2- or 3-D")

    m, n = a.shape[1:]
    if m != n:
        raise ValueError("a must be square")

    if m == 1:
        inv = 1 / a
    elif m == 2:
        inv = _inv_2x2(a)
    else:
        inv = _inv_cholesky(a)

    return inv[0] if squeeze else inv


def _inv_2x2(a: np.ndarray) -> np.ndarray:
    """Inverts t arrays stacked along the first dimension such that a has shape
    (t, m, m)."""
    dets = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
    if np.any(dets == 0):
        raise ValueError("at least one matrix is singular")

    flat = np.vstack((a[:, 1, 1], -a[:, 0, 1], -a[:, 1, 0], a[:, 0, 0])) / dets
    return flat.T.reshape(-1, 2, 2)


def _inv_cholesky(a: np.ndarray) -> np.ndarray:
    """Invert a positive-definite matrix using the Cholesky decomposition.
    Assumes a is a 3D array of square matrices across the first axis."""
    uinv = np.empty_like(a)
    for i in range(len(a)):
        # u is the upper triangular matrix of the Cholesky decomposition
        # a[i] = u.T @ u
        u, info = lapack.dpotrf(a[i])
        if info != 0:
            raise ValueError("matrix is not positive-definite")

        # dpotri only returns the upper triangular part of the inverse
        uinv[i], info = lapack.dpotri(u)
        if info != 0:
            raise ValueError("matrix is singular")

    # Not sure if lapack sets the lower diagonal to zero or if it leaves it
    # unset, so explicitly set the lower diagonal indices, which seems to be
    # slightly slower than simply adding the transpose of the upper triangular
    # matrix.
    idx = np.tril_indices(a.shape[1], -1)
    uinv[:, *idx] = matrix_transpose(uinv)[:, *idx]
    return uinv
