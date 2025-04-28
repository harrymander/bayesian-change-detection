from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import numpy.testing
import pytest
import scipy

from bayesian_change_detection import (
    MultivariateBcdmResults,
    multivariate_bcdm,
)
from bayesian_change_detection.multivariate import NigParams, NigPrior
from tests.conftest import JsonSnapshot, NDArraySnapshot

RandomData = tuple[np.ndarray, np.ndarray]


class _NumpyAssertFunction(Protocol):
    def __call__(
        self,
        actual,
        desired,
        *args: Any,
        err_msg: str | None = ...,
        **kwargs: Any,
    ) -> None: ...


def _make_numpy_assert_function(
    assert_func: Callable,
    default_err_msg: str,
    **default_kwargs,
) -> _NumpyAssertFunction:
    def _assert(
        actual, desired, *args, err_msg: str | None = None, **kwargs
    ) -> None:
        __tracebackhide__ = True
        kwargs = default_kwargs | kwargs
        if not err_msg:
            err_msg = default_err_msg
        try:
            assert_func(actual, desired, *args, err_msg=err_msg, **kwargs)
        except AssertionError as e:
            raise AssertionError(f"{err_msg}{e}") from None

    return _assert


assert_allclose = _make_numpy_assert_function(
    numpy.testing.assert_allclose, "Arrays are not close"
)
assert_array_equal_strict = _make_numpy_assert_function(
    numpy.testing.assert_array_equal, "Arrays are not equal", strict=True
)


def assert_array_less_strict(actual, desired, *args, **kwargs) -> None:
    __tracebackhide__ = True

    # strict only supported on numpy v2
    if int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
        default_kwargs = {"strict": True}
    else:
        default_kwargs = {}
        assert actual.shape == desired.shape, (
            f"Shapes are not equal: {actual.shape} != {desired.shape}"
        )
        assert actual.dtype is desired.dtype, (
            f"Dtypes are not equivalent: {actual.dtype} is not {desired.dtype}"
        )

    _make_numpy_assert_function(
        numpy.testing.assert_array_less,
        "Arrays are not strictly ordered `x < y`",
        **default_kwargs,
    )(actual, desired, *args, **kwargs)


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


def parametrize_attrs(name: str):
    return pytest.mark.parametrize(name, ("mean", "cov", "shape", "scale"))


class BcdmTester(ABC):
    results: MultivariateBcdmResults

    @classmethod
    @abstractmethod
    def run_bcdm(cls) -> MultivariateBcdmResults:
        pass

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def _setup_bcdm(cls) -> None:
        cls.results = cls.run_bcdm()

    def test_log_posterior_snapshot(self, ndarray_snapshot: NDArraySnapshot):
        log_posterior = self.results.log_posterior()
        assert_allclose(log_posterior, ndarray_snapshot(log_posterior))

    def test_log_posteriors_are_normalised(self) -> None:
        log_posterior = self.results.log_posterior()
        col_sums = scipy.special.logsumexp(log_posterior, axis=0)
        assert_allclose(col_sums, 0, atol=1e-12, rtol=1e-12)

    def test_changepoints_snapshot(self, json_snapshot: JsonSnapshot):
        changepoints = self.results.changepoints()
        assert changepoints == json_snapshot(changepoints)

    def test_changepoints_are_in_ascending_order(self) -> None:
        changepoints = self.results.changepoints()
        assert changepoints == sorted(changepoints)

    @parametrize_attrs("attrname")
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

    @parametrize_attrs("attrname")
    def test_parameters_all_nan_outside_support(self, attrname: str):
        parameters = getattr(self.results, attrname)
        nans = np.isnan(parameters).reshape((parameters.shape[0], -1))
        assert_array_equal_strict(
            nans.all(axis=1),
            nans.any(axis=1),
            err_msg=f"{attrname} has mixture of NaNs and normal numbers",
        )
        assert_array_equal_strict(
            self.results.final_support_mask(),
            ~nans.all(axis=1),
            err_msg="Support mask does not match position of NaNs",
        )

    def test_final_predictive_is_zero_outside_support(self) -> None:
        zeros = np.isneginf(self.results.log_predictive[:, -1])
        assert_array_equal_strict(zeros, ~self.results.final_support_mask())

    def test_final_joint_is_zero_outside_support(self) -> None:
        zeros = np.isneginf(self.results.log_joint[:, -1])
        assert_array_equal_strict(zeros, ~self.results.final_support_mask())


class BcdmWithoutSupportTrimmingTester(BcdmTester):
    def test_final_support_mask_is_all_true(self) -> None:
        support = self.results.final_support_mask()
        assert_array_equal_strict(
            support,
            np.ones_like(support),
            err_msg="Support not all True",
        )

    def test_full_support_mask_is_upper_triangular_bool_matrix(self) -> None:
        expected = np.triu(np.ones(self.results.log_joint.shape, dtype=bool))
        assert_array_equal_strict(self.results.full_support_mask(), expected)


class BcdmWithSupportTrimmingTester(BcdmTester):
    max_num_probs: int

    def test_final_support_max_size(self) -> None:
        support_size = self.results.final_support_mask().sum()
        assert support_size <= self.max_num_probs

    def test_full_support_mask_max_size(self) -> None:
        mask = self.results.full_support_mask()
        max_sizes = np.full(mask.shape[0], self.max_num_probs)
        max_sizes[: self.max_num_probs] = np.arange(self.max_num_probs) + 1
        assert_array_less_strict(mask.sum(axis=0), max_sizes + 1)


class Test1DChangeDetection(BcdmWithoutSupportTrimmingTester):
    @classmethod
    def run_bcdm(cls) -> MultivariateBcdmResults:
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
            cov=var0,
            hazard=hazard,
        )


class Test1DChangeDetectionWithSupportTrimming(BcdmWithSupportTrimmingTester):
    max_num_probs = 10

    @classmethod
    def run_bcdm(cls) -> MultivariateBcdmResults:
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
            cov=var0,
            hazard=hazard,
            max_num_probs=cls.max_num_probs,
            min_prob=1e-12,
        )


class Test2DChangeDetection(BcdmWithoutSupportTrimmingTester):
    @classmethod
    def run_bcdm(cls) -> MultivariateBcdmResults:
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
            cov=1e6,
            shape=1e-3,
            scale=1e-6,
        )


class Test3DChangeDetection(BcdmWithoutSupportTrimmingTester):
    @classmethod
    def run_bcdm(cls) -> MultivariateBcdmResults:
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
            mean=np.full_like(w, 0.5),
        )


def test_params_index_with_scalar_maintains_shape() -> None:
    params = NigParams.from_prior(NigPrior(5), 10)
    param = params[0]
    assert param.mean.shape == (1, 5)
    assert param.cov.shape == (1, 5, 5)
    assert param.prec.shape == (1, 5, 5)
    assert param.shape.shape == (1,)
    assert param.scale.shape == (1,)


def assert_nig_prior_equal(a: NigPrior, b: NigPrior):
    __tracebackhide__ = True
    for field in NigParams.__dataclass_fields__:
        numpy.testing.assert_array_equal(
            getattr(a, field),
            getattr(b, field),
            err_msg=f"NigParams field '{field}' not equal",
        )


def test_scalar_priors_equivalent_to_array() -> None:
    array_priors = NigPrior(
        3,
        mean=np.full(3, 2.0),
        cov=np.eye(3) * 3.0,
    )
    scalar_priors = NigPrior(3, mean=2, cov=3)
    assert_nig_prior_equal(array_priors, scalar_priors)


def test_1d_cov_prior_equivalent_to_2d() -> None:
    twod_array_priors = NigPrior(3, cov=np.eye(3) * 3.0)
    oned_array_priors = NigPrior(3, cov=np.full(3, 3.0))
    assert_nig_prior_equal(twod_array_priors, oned_array_priors)


def test_arraylike_priors_equivalent_to_array() -> None:
    array_priors = NigPrior(
        3,
        mean=np.array([1.0, 2.0, 3.0]),
        cov=np.diag([1.0, 2.0, 3.0]),
    )
    arraylike_priors = NigPrior(
        3,
        mean=[1, 2, 3],
        cov=[
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3],
        ],
    )
    assert_nig_prior_equal(array_priors, arraylike_priors)


def test_arraylike_1d_cov_prior_equivalent_to_2d() -> None:
    array_priors = NigPrior(3, cov=np.diag([1.0, 2.0, 3.0]))
    arraylike_priors = NigPrior(3, cov=[1, 2, 3])
    assert_nig_prior_equal(array_priors, arraylike_priors)


def test_arraylike_prior_fails_if_not_convertible_to_float() -> None:
    with pytest.raises(TypeError):
        NigParams.from_prior(3, mean=[1, 2, 3j])  # type: ignore
