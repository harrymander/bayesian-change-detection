"""
The following are functions for working with efficient representations of
lower-triangular matrices, where the lower-triangular elements are stored
contiguously in a 1D array. For example, the 2D matrix

  a = [[a, 0, 0]
       [b, c, 0]
       [d, e, f]]

Would be stored as:

  b = [a, b, c, d, e, f]

`tril_expand` converts the 1D representation in `b` back into the 2D
representation in `a`. `tril_row_slice` can be used to efficiently index into
rows of the matrix; for example:

  >>> b[tril_row_slice(1)]
  [b, c]
"""

from typing import Any

import numpy as np


def tril_size(n: int) -> int:
    """Returns the number of lower-triangular elements in an (n, n) matrix."""
    return n * (n + 1) // 2


def tril_row_slice(i: int, *, col_start: int = 0) -> slice:
    """
    Returns slice into the ith row of a lower-triangular matrix that is stored
    in a 1D row-wise concatenation of the lower-triangular elements. Does not
    do any boundary checking.

    If `a` is a 1D array of lower-triangular elements, then

      a[tril_row_slice(i, col_start=j)]  # returns view of size `i + 1 - j`

    is equivalent to the following indexing on a 2D array `b`:

      b[i, j : i + 1]  # returns view of size `i + 1 - j`
    """
    n = tril_size(i)
    return slice(n + col_start, n + i + 1)


def tril_expand(a: np.ndarray, *, upper_val: Any = None) -> np.ndarray:
    """Expand a 1D array representing row-wise concatenation of the
    lower-triangular elements of a matrix to its full 2D representation."""
    dtype = a.dtype
    if upper_val is None:
        upper_val = np.zeros((), dtype=dtype)

    n = (int(np.sqrt(1 + 8 * a.size)) - 1) // 2
    full = np.empty((n, n), dtype=dtype)
    full[np.tril_indices_from(full)] = a
    full[np.triu_indices_from(full, 1)] = upper_val
    return full
