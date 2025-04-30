import numpy as np
import pytest

from bayesian_change_detection._tril import (
    tril_expand,
    tril_row_slice,
    tril_size,
)
from tests.utils import assert_array_equal_strict


@pytest.mark.parametrize("upper_val", (None, np.inf))
def test_tril(upper_val) -> None:
    n = 5
    lower_val = -np.inf
    compressed = np.full(tril_size(n), lower_val, dtype=float)

    compressed[tril_row_slice(0)] = 1
    compressed[tril_row_slice(1)] = [10, 20]
    compressed[tril_row_slice(3, col_start=2)] = [-1, -2]
    compressed[tril_row_slice(4, col_start=1)] = [1, 2, 3, 4]

    U = upper_val or 0
    L = lower_val
    # fmt: off
    expected = np.asarray([
        [1,  U,  U,  U,  U],
        [10, 20, U,  U,  U],
        [L,  L,  L,  U,  U],
        [L,  L,  -1, -2, U],
        [L,  1,  2,  3,  4],
    ])
    # fmt: on
    expanded = tril_expand(compressed, upper_val=upper_val)
    assert_array_equal_strict(expanded, expected)
