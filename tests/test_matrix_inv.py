import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from bayesian_change_detection.linalg import (
    PositiveDefiniteError,
    inv_positive_definite,
    matrix_transpose,
)


def random_positive_definite_matrices(p: int) -> np.ndarray:
    rng = np.random.default_rng(p)
    a = rng.normal(10, 5, size=(50, p, p))
    return a @ matrix_transpose(a)


def parametrize_array_size(f):
    return pytest.mark.parametrize(
        "p",
        range(1, 5),
        ids=[f"{d}x{d}" for d in range(1, 5)],
    )(f)


@parametrize_array_size
def test_inv_positive_definite(p: int):
    a = random_positive_definite_matrices(p)
    inv = np.linalg.inv(a)
    assert_equal(np.isfinite(inv), True)
    assert_allclose(inv_positive_definite(a), inv)


@parametrize_array_size
def test_inv_positive_definite_fails_with_non_positive_definite_matrix(p: int):
    a = random_positive_definite_matrices(p)

    if p == 1:
        a[len(a) // 2] *= -1
    else:
        # Make middle matrix non-symmetric
        a[len(a) // 2, 0, 1] += 0.1

    with pytest.raises(PositiveDefiniteError):
        inv_positive_definite(a)
