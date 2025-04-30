import numpy as np
import pytest

from bayesian_change_detection._tril import tril_full
from tests.utils import assert_array_equal_strict


@pytest.mark.parametrize("upper_val", (None, np.inf))
def test_tril(upper_val) -> None:
    n = 5
    lower_val = -np.inf
    kw = {} if upper_val is None else {"upper_val": upper_val}
    tril = tril_full(n, lower_val, **kw)

    tril[0] = 1
    tril[1] = [10, 20]
    tril[3][2:] = [-1, -2]
    tril[4][1:] = [1, 2, 3, 4]

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

    assert_array_equal_strict(tril.full(), expected)
