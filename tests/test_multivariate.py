from abc import ABC, abstractmethod

import numpy as np
import numpy.testing
import pytest
import scipy

from bayesian_change_detection import (
    MultivariateBcdmResults,
    multivariate_bcdm,
)
from bayesian_change_detection.multivariate import NigParams
from tests.conftest import JsonSnapshot, NDArraySnapshot

RandomData = tuple[np.ndarray, np.ndarray]


def assert_allclose(a, b, **kwargs) -> None:
    """Same as numpy.testing.assert_allclose, but NaNs do not compare equal by
    default."""
    __tracebackhide__ = True
    kwargs = {"equal_nan": False} | kwargs
    numpy.testing.assert_allclose(a, b, **kwargs)


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


def new_params(t: int, p: int) -> NigParams:
    """
    Generate new params for t NIG distributions of dimension p.
    """
    cov = np.stack([np.eye(p) * i for i in np.linspace(1, 4, t)])
    assert cov.shape == (t, p, p)
    return NigParams(
        mean=np.linspace(0, 5, t).repeat(p).reshape(-1, p),
        cov=cov,
        prec=np.linalg.inv(cov),
        shape=np.linspace(2, 5, t),
        scale=np.linspace(2, 5, t),
    )


class TestNigParams:
    iteratively_updated_params: NigParams
    batch_updated_params: NigParams
    p: int  # dimensionality of independent variable
    t = 5  # number of distributions

    @pytest.fixture(
        scope="class",
        autouse=True,
        params=range(1, 4),
        ids=[f"{i}D" for i in range(1, 4)],
    )
    @classmethod
    def _update_params(cls, request: pytest.FixtureRequest) -> None:
        n = 500  # number of observations
        p: int = request.param

        rng = np.random.default_rng(p)
        x = rng.normal(size=(n, p))
        y = rng.normal(size=n)

        cls.iteratively_updated_params = new_params(cls.t, p)
        for xt, yt in zip(x, y, strict=True):
            cls.iteratively_updated_params.update(
                xt.reshape(1, p), np.atleast_1d(yt)
            )

        cls.batch_updated_params = new_params(cls.t, p)
        cls.batch_updated_params.update(x, y)
        cls.p = p

    @pytest.mark.parametrize(
        "attrname", ("mean", "cov", "prec", "shape", "scale")
    )
    def test_batch_and_iteratively_updated_params_are_equivalent(
        self, attrname: str
    ) -> None:
        assert_allclose(
            getattr(self.batch_updated_params, attrname),
            getattr(self.iteratively_updated_params, attrname),
        )

    @classmethod
    def _mvt_logpdf_iterative(
        cls, params: NigParams, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        logpdf = np.empty((cls.t, len(x)))
        for i, (xt, yt) in enumerate(zip(x, y, strict=True)):
            pdf = params.mvt_logpdf(xt.reshape(1, -1), np.atleast_1d(yt))
            assert pdf.shape == (cls.t, 1)
            logpdf[:, i] = pdf.ravel()
        return logpdf

    def test_mvt_logpdf_snapshot(self, ndarray_snapshot: NDArraySnapshot):
        n = 500
        rng = np.random.default_rng(1234)
        x: np.ndarray = rng.normal(size=(n, self.p))
        y: np.ndarray = rng.normal(size=(n,))
        logpdf = self._mvt_logpdf_iterative(
            self.iteratively_updated_params, x, y
        )
        assert_allclose(logpdf, ndarray_snapshot(logpdf))

    def test_iteratively_and_batch_computed_mvt_logpdf_are_equivalent(
        self,
    ) -> None:
        n = 500
        rng = np.random.default_rng(1234)
        x: np.ndarray = rng.normal(size=(n, self.p))
        y: np.ndarray = rng.normal(size=(n,))
        iterative_logpdf = self._mvt_logpdf_iterative(
            self.iteratively_updated_params, x, y
        )
        batch_logpdf = self.iteratively_updated_params.mvt_logpdf(x, y)
        assert_allclose(batch_logpdf, iterative_logpdf)


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

    def test_log_posteriors_are_normalised(self) -> None:
        col_sums = scipy.special.logsumexp(self.results.log_posterior, axis=0)
        assert_allclose(col_sums, 0, atol=1e-12, rtol=1e-12)

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


class Test3DChangeDetection(BocdTester):
    @classmethod
    def run_bocd(cls) -> MultivariateBcdmResults:
        rng = np.random.default_rng(42)

        # Generate random piecewise data
        samples = 100
        w = rng.random(3)
        hazard = 0.1
        x = np.linspace(0, 5, samples)
        x = np.stack((np.ones_like(x), x, x * 2), axis=1)
        data = []
        for xt in x:
            if rng.random() <= hazard:
                w = rng.random(3)  # new changepoint
            data.append(np.dot(xt, w))

        assert x.shape == (samples, 3)
        return multivariate_bcdm(
            x,
            np.asarray(data),
            hazard=hazard,
            prior_mean=np.full_like(w, 0.5),
        )
