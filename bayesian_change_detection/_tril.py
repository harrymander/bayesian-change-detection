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


class TrilArray(Protocol):
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

    @property
    def n(self) -> int:
        """Number of rows (also the number of columns)."""
        ...

    @property
    def data(self) -> np.ndarray:
        """1D array storing the lower diagonal elements of the matrix."""
        ...

    def __getitem__(self, i: _IntegerT) -> np.ndarray:
        """
        Returns a view into the lower-triangular elements of the ith row.

        `i` must be in [0, n - 1]. Negative indexing not supported. Boundary
        checking is not necessarily performed.
        """
        ...

    def __setitem__(self, i: _IntegerT, v: Any) -> None: ...

    def full(self) -> np.ndarray:
        """Returns a full (n, n) 2D lower-triangular matrix.

        The upper-triangular elements are set to `self.upper_val`."""
        ...


def tril_size(n: _IntegerT) -> _IntegerT:
    """Returns the number of lower-triangular elements in an (n, n) matrix."""
    return n * (n + 1) // 2


class _TrilArray:
    """Implementation of _TrilArray."""

    __slots__ = ["data", "n", "upper_val"]

    def __init__(self, n: int, data: np.ndarray, upper_val: Any = None):
        """Do not instantiate directly - use tril_full"""
        self.n = n
        self.data = data
        self.upper_val = upper_val

    def _row_slice(self, i: _IntegerT) -> np.ndarray:
        n = tril_size(i)
        return self.data[n : n + i + 1]

    def __getitem__(self, i: _IntegerT) -> np.ndarray:
        return self._row_slice(i)

    def __setitem__(self, i: _IntegerT, v: Any) -> None:
        self._row_slice(i)[:] = v

    def full(self) -> np.ndarray:
        data = self.data
        dtype = data.dtype
        upper_val = self.upper_val
        if upper_val is None:
            upper_val = np.zeros((), dtype=dtype)

        full = np.empty((self.n, self.n), dtype=dtype)
        full[np.tril_indices_from(full)] = data
        full[np.triu_indices_from(full, 1)] = upper_val
        return full

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.n}, {self.data!r})"


def tril_full(
    n: int,
    val: Any,
    *,
    upper_val: Any = None,
    dtype: DTypeLike = np.float64,
) -> TrilArray:
    if n <= 0:
        raise ValueError("n must be > 0")
    return _TrilArray(n, np.full(tril_size(n), val, dtype=dtype), upper_val)
