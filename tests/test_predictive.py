import numpy as np
import pytest
from numpy.testing import assert_allclose

from bayesian_change_detection import _MultivariateNormalInverseGamma

PARAMETER_ATTRS = ("mean", "cov", "shape", "scale")


@pytest.fixture
def random_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    n = 500
    x: np.ndarray = np.linspace(0, 10, n)
    y = x * 1.5 + 2 + rng.normal(0, 2, size=n)
    x = np.c_[np.ones_like(x), x]
    assert x.shape == (n, 2)
    return x, y


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
