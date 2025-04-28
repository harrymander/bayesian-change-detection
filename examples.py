import numpy as np


def generate_random_piecewise_data(
    *,
    rng: np.random.Generator | None,
    var: float,
    mean_mean: float,
    mean_var: float,
    hazard: float,
    num_samples: int,
) -> tuple[np.ndarray, list[int]]:
    """
    Generate random piecewise data of length `num_samples`.

    Changepoints are randomly generated at a rate defined by `hazard`. Within
    each segment, data are drawn from a normal distribution with variance `var`
    and a mean drawn from a normal distribution with `mean_mean` mean and
    `mean_var` variance.

    Returns:
        data: The generated data.
        changepoints: List of changepoint indices.
    """
    rng = rng or np.random.default_rng()
    data = np.empty(num_samples)
    changepoints = []
    mean = rng.normal(mean_mean, mean_var)
    for t in range(num_samples):
        if rng.random() < hazard:
            mean = rng.normal(mean_mean, mean_var)
            changepoints.append(t)
        data[t] = rng.normal(mean, var)

    return data, changepoints
