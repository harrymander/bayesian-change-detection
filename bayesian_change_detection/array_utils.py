import numpy as np


def masked_argpartition(
    x: np.ndarray, mask: np.ndarray, kth: int
) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError("mask must be 1D")
    idx = np.where(mask)[0]
    return idx[np.argpartition(x[mask], kth)]
