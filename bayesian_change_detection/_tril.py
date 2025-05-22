"""
The following are functions for working with efficient representations of
lower-triangular matrices, where the lower-triangular elements are stored
contiguously in a 1D array. For example, the 2D matrix

  a = [[a, 0, 0]
       [b, c, 0]
       [d, e, f]]

Would be stored as:

  b = [a, b, c, d, e, f]

The `TrilArray` class provides a simple interface for indexing into rows of
such an array. For example:

  >>> b[1]
  [b, c]
"""

from typing import Any, Protocol, TypeVar

import numpy as np
from numpy.typing import DTypeLike

_IntegerT = TypeVar("_IntegerT", np.integer, int)


def tril_size(n: _IntegerT) -> _IntegerT:
    """Returns the number of lower-triangular elements in an (n, n) matrix."""
    return n * (n + 1) // 2


class TrilArray:
    """
    The lower-triangular elements of an (n, n) matrix stored contiguously.

    Use with care. Many checks (e.g. boundary checking on indexing) are not
    performed.
    """

    upper_val: Any
    """
    Value of the upper-triangular elements as returned by `full`. If `None`,
    will be the zero value of the dtype of `data`.
    """

    __slots__ = ["_data", "_n", "upper_val"]

    def __init__(self, n: int, data: np.ndarray, upper_val: Any = None):
        """Do not instantiate directly - use tril_full"""
        self._n = n
        self._data = data
        self.upper_val = upper_val

    @property
    def n(self) -> int:
        """Number of rows (also the number of columns)."""
        return self._n

    @property
    def data(self) -> np.ndarray:
        """A 1D array of the lower-triangular elements of the matrix."""
        return self._data

    def _row_slice(self, i: _IntegerT) -> np.ndarray:
        n = tril_size(i)
        return self._data[n : n + i + 1]

    def __getitem__(self, i: _IntegerT) -> np.ndarray:
        """
        Returns a view into the lower-triangular elements of the ith row.

        `i` must be in [0, n - 1]. Negative indexing not supported. Boundary
        checking is not necessarily performed.
        """
        return self._row_slice(i)

    def __setitem__(self, i: _IntegerT, v: Any) -> None:
        self._row_slice(i)[:] = v

    def full(self) -> np.ndarray:
        """
        Returns a full (n, n) 2D lower-triangular matrix.

        The upper-triangular elements are set to `self.upper_val`.
        """
        data = self._data
        dtype = data.dtype
        upper_val = self.upper_val
        if upper_val is None:
            upper_val = np.zeros((), dtype=dtype)

        full = np.empty((self._n, self._n), dtype=dtype)
        full[np.tril_indices_from(full)] = data
        full[np.triu_indices_from(full, 1)] = upper_val
        return full

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._n}, {self._data!r})"

    def __bool__(self) -> bool:
        if self._n == 1:
            return bool(self._data)

        # Similar to the message raised by numpy.ndarray.
        msg = (
            "The truth value of an array with more than one element "
            "is ambiguous. Use a.data.any(), a.data.all(), a.full().any(), "
            "or a.full().all()"
        )
        raise ValueError(msg)


class _ArrayCreator(Protocol):
    def __call__(self, size: int, /, *, dtype: DTypeLike) -> np.ndarray: ...


def _new_tril(
    n: int,
    f: _ArrayCreator,
    upper_val: Any,
    dtype: DTypeLike,
) -> TrilArray:
    if n <= 0:
        raise ValueError("n must be > 0")
    return TrilArray(n, f(tril_size(n), dtype=dtype), upper_val)


def tril_full(
    n: int,
    val: Any,
    *,
    upper_val: Any = None,
    dtype: DTypeLike = np.float64,
) -> TrilArray:
    return _new_tril(
        n,
        lambda size, dtype: np.full(size, val, dtype=dtype),
        upper_val,
        dtype,
    )


def tril_empty(
    n: int,
    *,
    upper_val: Any = None,
    dtype: DTypeLike = np.float64,
) -> TrilArray:
    return _new_tril(n, np.empty, upper_val, dtype)
