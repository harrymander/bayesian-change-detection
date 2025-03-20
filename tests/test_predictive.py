import numpy as np
import pytest
from numpy.testing import assert_allclose

from bayesian_change_detection import _MultivariateNormalInverseGamma
from tests.conftest import NDArraySnapshot

PARAMETER_ATTRS = ("mean", "cov", "shape", "scale")


def _generate_random_data(
    seed: int,
    n: int,
    w0: float,
    w1: float,
) -> tuple[np.ndarray, np.ndarray]:
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
    return _generate_random_data(42, 500, 2, 1.5)


def _new_model() -> _MultivariateNormalInverseGamma:
    return _MultivariateNormalInverseGamma(
        mean=np.zeros(2),
        cov=np.eye(2),
        shape=1,
        scale=1,
    )


def _iterative_update_model(model, data):
    for xt, yt in zip(*data, strict=True):
        assert xt.shape == (2,)
        assert isinstance(yt, float)
        model.update(xt, yt)


@pytest.mark.parametrize("attr", PARAMETER_ATTRS)
def test_predictive_parameters_batch_and_iterative_are_equivalent(
    attr: str, random_data
) -> None:
    """
    Updating the model iterative or in batch with the same data should give
    the same results.
    """
    iterative = _new_model()
    _iterative_update_model(iterative, random_data)
    batch = _new_model()
    batch.update(*random_data)
    assert_allclose(
        getattr(iterative, attr),
        getattr(batch, attr),
        err_msg=f"iterative.{attr} ≉ batch.{attr}",
    )


@pytest.mark.parametrize("attrname", PARAMETER_ATTRS)
def test_predictive_parameters_snapshot(
    random_data,
    attrname: str,
    ndarray_snapshot: NDArraySnapshot,
):
    model = _new_model()
    _iterative_update_model(model, random_data)

    attr = getattr(model, attrname)
    assert_allclose(attr, ndarray_snapshot(attr))


def test_predictive_log_density_snapshot(
    random_data,
    ndarray_snapshot: NDArraySnapshot,
):
    model = _new_model()
    _iterative_update_model(model, random_data)

    x, y = _generate_random_data(123, 50, 2.5, 1.2)
    log_density = np.fromiter(
        (model.log_density(xt, yt) for xt, yt in zip(x, y, strict=True)),
        dtype=y.dtype,
        count=y.size,
    )
    assert_allclose(log_density, ndarray_snapshot(log_density))
