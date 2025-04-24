import numpy as np


def _where_1d(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError("mask must be 1D")
    return np.where(mask)[0]


def masked_argsort(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return _where_1d(x, mask)[np.argsort(x[mask])]


def masked_argpartition(
    x: np.ndarray, mask: np.ndarray, kth: int
) -> np.ndarray:
    return _where_1d(x, mask)[np.argpartition(x[mask], kth)]
