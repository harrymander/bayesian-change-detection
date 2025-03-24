import numpy as np
import pytest
from numpy.testing import assert_allclose

from bayesian_change_detection import (
    MultivariateBcdm,
    MultivariateNormalInverseGamma,
)
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


@pytest.fixture
def distribution() -> MultivariateNormalInverseGamma:
    return MultivariateNormalInverseGamma(
        mean=np.zeros(2),
        cov=np.eye(2),
        shape=1,
        scale=1,
    )


@pytest.mark.parametrize("attrname", ("mean", "cov", "shape", "scale"))
def test_multivariate_normal_inv_gamma_parameters_snapshot(
    distribution: MultivariateNormalInverseGamma,
    random_data: RandomData,
    attrname: str,
    ndarray_snapshot: NDArraySnapshot,
):
    for xt, yt in zip(*random_data, strict=True):
        distribution.predict_and_update(xt, yt)

    attr = getattr(distribution, attrname)
    assert_allclose(attr, ndarray_snapshot(attr))


def test_predictive_log_density_snapshot(
    distribution: MultivariateNormalInverseGamma,
    random_data: RandomData,
    ndarray_snapshot: NDArraySnapshot,
):
    log_density = np.empty(len(random_data[0]))
    for i, (x, y) in enumerate(zip(*random_data, strict=True)):
        log_density[i] = distribution.predict_and_update(x, y)

    assert_allclose(log_density, ndarray_snapshot(log_density))


def test_change_detection_snapshot(ndarray_snapshot: NDArraySnapshot):
    rng = np.random.default_rng(42)

    # Create input and outputs.
    samples = 100
    x = np.linspace(0, 3 * 2 * np.pi, samples)
    noise = 0.1 * rng.standard_normal(samples)
    y = -np.arcsin(np.sin(x)) * 2 / np.pi + noise
    model = MultivariateBcdm(
        2,
        hazard=0.02,
        prior_cov=1e6,
        prior_shape=1e-3,
        prior_scale=1e-6,
    )

    # Update the segment length hypotheses given the data.
    for xt, yt in zip(x, y, strict=True):
        model.update(np.array([1, xt]), yt)

    log_posterior = model.log_posterior()
    assert_allclose(log_posterior, ndarray_snapshot(log_posterior))
