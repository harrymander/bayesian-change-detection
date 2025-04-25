import numpy as np
import scipy.special


def logsumexp_sparse(x: np.ndarray) -> np.ndarray:
    """Log-sum-exp. More efficient for arrays where most elements are -inf."""
    return scipy.special.logsumexp(x[np.isfinite(x)])


def masked_argpartition(
    x: np.ndarray, mask: np.ndarray, kth: int
) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError("mask must be 1D")
    idx = np.where(mask)[0]
    return idx[np.argpartition(x[mask], kth)]
