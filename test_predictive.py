import numpy as np
import pytest
from numpy.testing import assert_allclose

from bayesian_change_detection import _MultivariateNormalInverseGamma

Models = tuple[
    _MultivariateNormalInverseGamma,
    _MultivariateNormalInverseGamma,
]


def _new_model(p: int) -> _MultivariateNormalInverseGamma:
    return _MultivariateNormalInverseGamma(
        mean=np.zeros(p),
        cov=np.eye(p),
        shape=1,
        scale=1,
    )


@pytest.fixture(scope="module")
def models() -> Models:
    rng = np.random.default_rng(42)

    n = 500
    x: np.ndarray = np.linspace(0, 10, n)
    y = x * 1.5 + 2 + rng.normal(0, 2, size=n)
    x = np.c_[np.ones_like(x), x]
    assert x.shape == (n, 2)

    p = 2
    model_iterative = _new_model(p)
    for xt, yt in zip(x, y, strict=True):
        assert xt.shape == (2,)
        assert isinstance(yt, float)
        model_iterative.update(xt, yt)

    model_batch = _new_model(p)
    model_batch.update(x, y)

    return model_iterative, model_batch


@pytest.mark.parametrize("attr", ("mean", "cov", "shape", "scale"))
def test_predictive_distribution(attr: str, models: Models) -> None:
    """
    Updating the model iterative or in batch with the same data should give
    the same results.
    """
    iterative, batch = models
    assert_allclose(
        getattr(iterative, attr),
        getattr(batch, attr),
        err_msg=f"iterative.{attr} ≉ batch.{attr}",
    )
