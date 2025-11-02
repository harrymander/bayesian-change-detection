import re

import numpy as np
import pytest

from bayesian_change_detection.tril import tril_full
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


@pytest.mark.parametrize("upper_val", (None, np.inf))
def test_tril_less_cols_than_rows(upper_val) -> None:
    n = 5
    m = 3
    lower_val = -np.inf
    kw = {} if upper_val is None else {"upper_val": upper_val}

    tril = tril_full(n, lower_val, m, **kw)

    tril[0] = 1
    tril[1] = [10, 20]
    tril[2][1:2] = -1
    tril[3][2:] = [-2]
    tril[4][1:] = [-10, -20]

    U = upper_val or 0
    L = lower_val
    # fmt: off
    expected = np.asarray([
        [1,  U,   U],
        [10, 20,  U],
        [L,  -1,  L],
        [L,  L,   -2],
        [L,  -10, -20],
    ])
    # fmt: on

    assert_array_equal_strict(tril.full(), expected)


@pytest.mark.parametrize("upper_val", (None, np.inf))
def test_tril_more_cols_than_rows(upper_val) -> None:
    n = 5
    m = 7
    lower_val = -np.inf
    kw = {} if upper_val is None else {"upper_val": upper_val}

    tril = tril_full(n, lower_val, m, **kw)

    tril[0] = 1
    tril[1] = [10, 20]
    tril[2][2] = -1
    tril[3][2:] = [-2, -3]
    tril[4][1:-1] = [-10, -20, -30]

    U = upper_val or 0
    L = lower_val
    # fmt: off
    expected = np.asarray([
        [1,  U,   U,   U,   U, U, U],
        [10, 20,  U,   U,   U, U, U],
        [L,  L,   -1,  U,   U, U, U],
        [L,  L,   -2,  -3,  U, U, U],
        [L,  -10, -20, -30, L, U, U],
    ])
    # fmt: on

    assert_array_equal_strict(tril.full(), expected)


def test_single_element_nonzero_tril_is_truthy() -> None:
    a = tril_full(1, 1)
    assert a


def test_single_element_zero_tril_is_falsy() -> None:
    a = tril_full(1, 0)
    assert not a


def test_bool_of_tril_with_more_than_one_element_raises_error() -> None:
    b = tril_full(2, 1)
    m = "The truth value of an array with more than one element is ambiguous."
    with pytest.raises(ValueError, match=f"^{re.escape(m)}"):
        bool(b)


@pytest.mark.parametrize("rows,cols", [(3, None), (3, 3), (3, 2), (3, 4)])
def test_tril_len_is_num_rows(rows: int, cols: int | None) -> None:
    a = tril_full(rows, np.nan, cols)
    assert len(a) == rows


def test_tril_set_item() -> None:
    a = tril_full(3, 1, 4, upper_val=9, dtype=int)

    a[1] = 2
    a[2][-2:] = [7, 8]

    # fmt: off
    expected = np.array([
        [1, 9, 9, 9],
        [2, 2, 9, 9],
        [1, 7, 8, 9],
    ])
    # fmt: on

    assert_array_equal_strict(a.full(), expected)
