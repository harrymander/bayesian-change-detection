import numpy as np
import scipy.special
from numpy.testing import assert_array_equal

from bayesian_change_detection.array_utils import (
    logsumexp_sparse,
    masked_argpartition,
)


def _masked_argsort(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Helper function for comparing to masked_argpartition"""
    return np.where(mask)[0][np.argsort(x[mask])]


def test_masked_argsort() -> None:
    """Test helper function"""
    a = np.array([5, 4, 3, 2, 1])
    mask = np.array([True, False, True, False, True])
    expected = np.array([4, 2, 0])
    assert_array_equal(_masked_argsort(a, mask), expected)


def test_masked_argsort_with_full_mask_is_equivalent_to_argsort() -> None:
    """Test helper function"""
    rng = np.random.default_rng(123)
    a = rng.random(100)
    mask = np.ones_like(a, dtype=bool)
    assert_array_equal(np.argsort(a), _masked_argsort(a, mask))


def test_masked_argpartition() -> None:
    a = np.array([5, 4, 3, 2, 1])
    mask = np.array([False, True, False, True, False])

    # The second-highest element in `a[mask]` is at position
    # 3 in the original array `a` (i.e. the value is 2)
    assert masked_argpartition(a, mask, -2)[-2] == 3

    # The second-highest element in `a` (i.e. ignoring the mask) is at position
    # 1
    assert np.argpartition(a, -2)[-2] == 1


def test_masked_argpartition_with_full_mask_is_equivalent_to_argpartition():
    rng = np.random.default_rng(123)
    a = rng.random(100)
    mask = np.ones_like(a, dtype=bool)
    k = 23
    assert_array_equal(np.argpartition(a, k), masked_argpartition(a, mask, k))


def test_masked_argpartition_equivalent_to_argsort() -> None:
    rng = np.random.default_rng(42)
    a = rng.random(100)
    mask = rng.random(100) > 0.5
    assert_array_equal(
        np.sort(_masked_argsort(a, mask)[-20:]),
        np.sort(masked_argpartition(a, mask, -20)[-20:]),
    )


def test_logsumexp_sparse() -> None:
    rng = np.random.default_rng(42)
    a = np.log(rng.random(10000))
    a[rng.random(a.size) > 0.5] = -np.inf
    assert scipy.special.logsumexp(a) == logsumexp_sparse(a)
