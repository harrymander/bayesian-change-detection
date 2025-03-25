from abc import ABC, abstractmethod

import numpy as np
import pytest
from numpy.testing import assert_allclose

from bayesian_change_detection import (
    MultivariateBcdmResults,
    multivariate_bcdm,
)
from tests.conftest import JsonSnapshot, NDArraySnapshot

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


class BocdTester(ABC):
    results: MultivariateBcdmResults

    @classmethod
    @abstractmethod
    def run_bocd(cls) -> MultivariateBcdmResults:
        pass

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def _setup_bocd(cls) -> None:
        cls.results = cls.run_bocd()

    def test_log_posterior_snapshot(self, ndarray_snapshot: NDArraySnapshot):
        log_posterior = self.results.log_posterior
        assert_allclose(log_posterior, ndarray_snapshot(log_posterior))

    @pytest.mark.parametrize("attrname", ("mean", "cov", "shape", "scale"))
    def test_parameters_snapshot(
        self,
        attrname: str,
        ndarray_snapshot: NDArraySnapshot,
    ):
        parameters = getattr(self.results, attrname)
        assert parameters.ndim <= 3
        if parameters.ndim == 3:
            b, m, n = parameters.shape
            parameters_2d = parameters.reshape(-1, m * n)
            snapshot = ndarray_snapshot(parameters_2d)
            snapshot = snapshot.reshape(b, m, n)
        else:
            snapshot = ndarray_snapshot(parameters).reshape(*parameters.shape)

        assert_allclose(parameters, snapshot)

    def test_changepoints_snapshot(self, json_snapshot: JsonSnapshot):
        changepoints = self.results.changepoints()
        assert changepoints == json_snapshot(changepoints)

    def test_changepoints_are_in_ascending_order(self) -> None:
        changepoints = self.results.changepoints()
        assert changepoints == sorted(changepoints)


class Test1DChangeDetection(BocdTester):
    @classmethod
    def run_bocd(cls) -> MultivariateBcdmResults:
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
        return multivariate_bcdm(
            np.ones_like(y),
            y,
            prior_cov=var0,
            hazard=hazard,
        )


class Test2DChangeDetection(BocdTester):
    @classmethod
    def run_bocd(cls) -> MultivariateBcdmResults:
        rng = np.random.default_rng(42)

        # Create input and outputs.
        samples = 100
        x = np.linspace(0, 3 * 2 * np.pi, samples)
        noise = 0.1 * rng.standard_normal(samples)
        y = -np.arcsin(np.sin(x)) * 2 / np.pi + noise
        return multivariate_bcdm(
            np.c_[np.ones_like(x), x],
            y,
            hazard=0.02,
            prior_cov=1e6,
            prior_shape=1e-3,
            prior_scale=1e-6,
        )
