"""Linear algebra helpers."""

import numpy as np
from scipy.linalg import lapack

try:
    # matrix_transpose is added in numpy v2
    from numpy import matrix_transpose  # type: ignore
except ImportError:

    def matrix_transpose(a: np.ndarray) -> np.ndarray:  # type: ignore
        return np.swapaxes(a, -1, -2)


class PositiveDefiniteError(ValueError):
    def __init__(self):
        super().__init__("at least one matrix is not positive-definite")


def inv_positive_definite(a: np.ndarray) -> np.ndarray:
    """
    Inverts a positive definite matrix or array of matrices.

    Args:
        a: (m, m) or (n, m, m) array of positive-definite matrices.

    Returns:
        (m, m) or (n, m, m) array of inverses of matrix/matrices in a.

    Raises:
        PositiveDefiniteError: If any of the matrices are not
            positive-definite.
    """
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
        if np.any(a <= 0):
            raise PositiveDefiniteError()
        inv = 1 / a
    elif m == 2:
        inv = _inv_2x2_positive_definite(a)
    else:
        inv = _inv_cholesky(a)

    return inv[0] if squeeze else inv


def _inv_2x2_positive_definite(a: np.ndarray) -> np.ndarray:
    """Inverts t arrays stacked along the first dimension such that a has shape
    (t, m, m)."""

    # Check leading principals are >0
    if np.any(a[:, 0, 0] <= 0):
        raise PositiveDefiniteError()

    # Check matrices are symmetric
    if not np.isclose(a[:, 0, 1], a[:, 1, 0]).all():
        raise PositiveDefiniteError()

    # Check determinants are >0
    dets = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
    if np.any(dets <= 0):
        raise PositiveDefiniteError()

    flat = np.vstack((a[:, 1, 1], -a[:, 0, 1], -a[:, 1, 0], a[:, 0, 0])) / dets
    return flat.T.reshape(-1, 2, 2)


def _inv_cholesky(a: np.ndarray) -> np.ndarray:
    dtype = a.dtype
    if dtype == np.float32:
        return _inv_cholesky_generic(
            a, potrf=lapack.spotrf, potri=lapack.spotri
        )

    if dtype == np.float64:
        return _inv_cholesky_generic(
            a, potrf=lapack.dpotrf, potri=lapack.dpotri
        )

    raise TypeError(f"unsupported dtype {dtype}")


def _inv_cholesky_generic(a: np.ndarray, *, potrf, potri) -> np.ndarray:
    """Invert a positive-definite matrix using the Cholesky decomposition.
    Assumes a is a 3D array of square matrices across the first axis."""
    if not np.allclose(a, matrix_transpose(a)):
        raise PositiveDefiniteError()

    uinv = np.empty_like(a)
    for i in range(len(a)):
        # u is the upper triangular matrix of the Cholesky decomposition
        # a[i] = u.T @ u
        u, info = potrf(a[i])
        if info != 0:
            raise PositiveDefiniteError()

        # dpotri only returns the upper triangular part of the inverse
        uinv[i], info = potri(u)
        if info != 0:
            raise PositiveDefiniteError()

    # Not sure if lapack sets the lower diagonal to zero or if it leaves it
    # unset, so explicitly set the lower diagonal indices, which seems to be
    # slightly slower than simply adding the transpose of the upper triangular
    # matrix.
    idx = np.tril_indices(a.shape[1], -1)
    uinv[:, *idx] = matrix_transpose(uinv)[:, *idx]
    return uinv
