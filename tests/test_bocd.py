import numpy as np
import pytest
from numpy.testing import assert_allclose

from bayesian_change_detection import multivariate_bcdm
from tests.conftest import NDArraySnapshot

RandomData = tuple[np.ndarray, np.ndarray]


def generate_random_data(
    seed: int,
    n: int,
    w0: float,
    w1: float,
) -> RandomData:
    """
    Using random seed, generate n paired observations according to the linear
    model: y = w0 + x*w1.
    """
    rng = np.random.default_rng(seed)
    x: np.ndarray = np.linspace(0, 10, n)
    y = w0 + x * w1 + rng.normal(0, 2, size=n)
    x = np.c_[np.ones_like(x), x]
    assert x.shape == (n, 2)
    return x, y


@pytest.fixture
def random_data() -> tuple[np.ndarray, np.ndarray]:
    return generate_random_data(42, 500, 2, 1.5)


def test_1d_change_detection_snapshot(ndarray_snapshot: NDArraySnapshot):
    samples = 100
    hazard = 0.1  # Constant prior on changepoint probability.
    mean0 = 0.0  # The prior mean on the mean parameter.
    var0 = 2.0  # The prior variance for mean parameter.
    varx = 1.0  # The known variance of the data.

    rng = np.random.default_rng(42)

    # Generate random piecewise data
    data = []
    meanx = mean0
    for _ in range(samples):
        if rng.random() < hazard:  # new changepoint
            meanx = rng.normal(mean0, var0)
        data.append(rng.normal(meanx, varx))

    y = np.asarray(data)
    res = multivariate_bcdm(np.ones_like(y), y, prior_cov=var0, hazard=hazard)

    log_posterior = res.log_posterior
    assert_allclose(log_posterior, ndarray_snapshot(log_posterior))


def test_2d_change_detection_snapshot(ndarray_snapshot: NDArraySnapshot):
    rng = np.random.default_rng(42)

    # Create input and outputs.
    samples = 100
    x = np.linspace(0, 3 * 2 * np.pi, samples)
    noise = 0.1 * rng.standard_normal(samples)
    y = -np.arcsin(np.sin(x)) * 2 / np.pi + noise
    res = multivariate_bcdm(
        np.c_[np.ones_like(x), x],
        y,
        hazard=0.02,
        prior_cov=1e6,
        prior_shape=1e-3,
        prior_scale=1e-6,
    )

    log_posterior = res.log_posterior
    assert_allclose(log_posterior, ndarray_snapshot(log_posterior))
