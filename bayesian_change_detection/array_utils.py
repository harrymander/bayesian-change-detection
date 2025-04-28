import numpy as np


def masked_argmin(x: np.ndarray, mask: np.ndarray) -> int:
    idx = np.where(mask)[0]
    return idx[x[idx].argmin()]
