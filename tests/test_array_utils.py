import numpy as np

from bayesian_change_detection.array_utils import masked_argmin


def test_masked_argmin() -> None:
    a = np.array([5, 4, 3, 2, 1])
    mask = np.array([False, True, False, True, False])

    # The lowest element in `a[mask]` is at position 3 in the original array
    # `a` (i.e. the value is 2)
    assert masked_argmin(a, mask) == 3
