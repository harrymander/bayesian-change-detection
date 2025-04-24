import numpy as np
from numpy.testing import assert_array_equal

from bayesian_change_detection.array_utils import (
    masked_argpartition,
    masked_argsort,
)


def test_masked_argsort() -> None:
    a = np.array([5, 4, 3, 2, 1])
    mask = np.array([True, False, True, False, True])
    expected = np.array([4, 2, 0])
    assert_array_equal(masked_argsort(a, mask), expected)


def test_masked_argsort_with_full_mask_is_equivalent_to_argsort() -> None:
    rng = np.random.default_rng(123)
    a = rng.random(100)
    mask = np.ones_like(a, dtype=bool)
    assert_array_equal(np.argsort(a), masked_argsort(a, mask))


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
        np.sort(masked_argsort(a, mask)[-20:]),
        np.sort(masked_argpartition(a, mask, -20)[-20:]),
    )
