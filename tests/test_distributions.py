import numpy as np
import pytest
from numpy.testing import assert_allclose

from bayesian_change_detection import MultivariateNormalInverseGamma
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
        distribution.update(xt, yt)

    attr = getattr(distribution, attrname)
    assert_allclose(attr, ndarray_snapshot(attr))


def test_predictive_log_density_snapshot(
    distribution: MultivariateNormalInverseGamma,
    random_data: RandomData,
    ndarray_snapshot: NDArraySnapshot,
):
    log_density = np.empty(len(random_data[0]))
    for i, (xt, yt) in enumerate(zip(*random_data, strict=True)):
        log_density[i] = distribution.log_density(xt, yt)
        distribution.update(xt, yt)

    assert_allclose(log_density, ndarray_snapshot(log_density))
