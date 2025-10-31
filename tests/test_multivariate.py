from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.testing
import pytest
import scipy
from numpy.typing import NDArray

from bayesian_change_detection import (
    MultivariateBcdmResults,
    multivariate_bcdm,
)
from bayesian_change_detection.multivariate import NigParams, NigPrior
from bayesian_change_detection.tril import TrilArray
from tests.conftest import JsonSnapshot, NDArraySnapshot
from tests.utils import (
    assert_allclose,
    assert_array_equal_strict,
    assert_array_less_strict,
)


def new_params(t: int, p: int) -> NigParams:
    """
    Generate new params for t NIG distributions of dimension p.
    """
    cov = np.stack([np.eye(p) * i for i in np.linspace(1, 4, t)])
    assert cov.shape == (t, p, p)
    return NigParams(
        mean=np.linspace(0, 5, t, dtype=np.float64).repeat(p).reshape(-1, p),
        cov=cov,
        prec=np.linalg.inv(cov),
        shape=np.linspace(2, 5, t, dtype=np.float64),
        scale=np.linspace(2, 5, t, dtype=np.float64),
    )


class TestNigParams:
    params_updated_with_2d_regressors: NigParams
    params_updated_with_1d_regressor: NigParams
    n: int
    p: int

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

        cls.params_updated_with_2d_regressors = new_params(n, p)
        cls.params_updated_with_1d_regressor = new_params(n, p)

        for xt, yt in zip(x, y, strict=True):
            assert xt.shape == (p,)
            assert np.isscalar(yt)
            cls.params_updated_with_2d_regressors.update(
                np.tile(xt.reshape(1, p), (n, 1)),
                np.tile(np.atleast_1d(yt), (n,)),
            )
            cls.params_updated_with_1d_regressor.update(
                cast(NDArray[np.floating], xt),
                cast(np.floating, yt),
            )

        cls.n = n
        cls.p = p

    @pytest.mark.parametrize(
        "attrname", ("mean", "cov", "prec", "shape", "scale")
    )
    def test_params_updated_with_1d_and_2d_params_are_equivalent(
        self, attrname: str
    ) -> None:
        assert_array_equal_strict(
            getattr(self.params_updated_with_2d_regressors, attrname),
            getattr(self.params_updated_with_1d_regressor, attrname),
        )

    def test_predictive_logpdf_snapshot(
        self, ndarray_snapshot: NDArraySnapshot
    ):
        rng = np.random.default_rng(1234)
        x: np.ndarray = rng.normal(size=(self.n, self.p))
        y: np.ndarray = rng.normal(size=(self.n,))
        logpdf = self.params_updated_with_1d_regressor.predictive_logpdf(x, y)
        assert_allclose(logpdf, ndarray_snapshot(logpdf))

    def test_predictive_logpdf_with_1d_and_2d_repeated_params_are_equvalent(
        self,
    ) -> None:
        rng = np.random.default_rng(1234)
        x = rng.normal(size=self.p)
        y = rng.normal()

        logpdf_1d = self.params_updated_with_1d_regressor.predictive_logpdf(
            x, y
        )

        xx = np.tile(x.reshape(1, -1), (self.n, 1))
        assert xx.shape == (self.n, self.p)
        yy = np.repeat(y, self.n)
        assert yy.shape == (self.n,)
        logpdf_2d = self.params_updated_with_1d_regressor.predictive_logpdf(
            xx, yy
        )

        assert_array_equal_strict(logpdf_1d, logpdf_2d)


def parametrize_nig_params_attrs(name: str):
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

    def test_log_predictive_snapshot(self, ndarray_snapshot: NDArraySnapshot):
        log_predictive = self.results.log_predictive.full()
        assert_allclose(log_predictive, ndarray_snapshot(log_predictive))

    def test_log_posteriors_are_normalised(self) -> None:
        log_posterior = self.results.log_posterior()
        col_sums = scipy.special.logsumexp(log_posterior, axis=1)
        assert_allclose(col_sums, 0, atol=1e-12, rtol=1e-12)

    def test_changepoints_snapshot(self, json_snapshot: JsonSnapshot):
        changepoints = self.results.changepoints()
        assert changepoints == json_snapshot(changepoints)

    def test_changepoints_are_in_ascending_order(self) -> None:
        changepoints = self.results.changepoints()
        assert changepoints == sorted(changepoints)

    @parametrize_nig_params_attrs("attrname")
    def test_parameters_snapshot(
        self,
        attrname: str,
        ndarray_snapshot: NDArraySnapshot,
    ):
        parameters = getattr(self.results, attrname)
        assert_allclose(parameters, ndarray_snapshot(parameters))

    @parametrize_nig_params_attrs("attrname")
    def test_parameters_no_mixed_nans_across_first_axis(self, attrname: str):
        parameters = getattr(self.results, attrname)
        nans = np.isnan(parameters).reshape((parameters.shape[0], -1))
        assert_array_equal_strict(
            nans.all(axis=1),
            nans.any(axis=1),
            err_msg=(
                f"{attrname} has mixture of NaNs and normal numbers along "
                "the first axis"
            ),
        )

    def test_joint_support_snapshot(self, ndarray_snapshot: NDArraySnapshot):
        support = self.results.joint_support.full()
        assert_array_equal_strict(
            support,
            ndarray_snapshot(support, fmt="%d").astype(bool),
        )

    @parametrize_nig_params_attrs("attrname")
    def test_parameters_all_nan_outside_final_support(self, attrname: str):
        parameters = getattr(self.results, attrname)
        nans = np.isnan(parameters).reshape((parameters.shape[0], -1))
        support = self.results.joint_support
        assert_array_equal_strict(
            support[support.shape[0] - 1],
            ~nans.all(axis=1),
            err_msg="Support mask does not match position of NaNs",
        )

    def _assert_array_is_zero_outside_support(self, arr: TrilArray) -> None:
        __tracebackhide__ = True
        assert_array_equal_strict(
            np.isneginf(arr.full()),
            ~self.results.joint_support.full(),
            err_msg="Array is not all-zero outside of the support",
        )

    def test_predictive_is_zero_outside_support(self) -> None:
        self._assert_array_is_zero_outside_support(self.results.log_predictive)

    def test_joint_is_zero_outside_support(self) -> None:
        self._assert_array_is_zero_outside_support(self.results.log_joint)


class BcdmWithoutSupportTrimmingTester(BcdmTester):
    def test_support_mask_is_lower_triangular_constant(self) -> None:
        support_2d = self.results.joint_support.full()
        expected = np.tril(np.ones(support_2d.shape, dtype=bool))
        assert_array_equal_strict(support_2d, expected)


class BcdmWithSupportTrimmingTester(BcdmTester):
    def test_support_mask_is_lower_triangular(self) -> None:
        support = self.results.joint_support.full()
        support_triu = support[np.triu_indices_from(support, 1)]
        assert_array_equal_strict(
            support_triu,
            np.zeros_like(support_triu),
            err_msg="Support is not lower-triangular",
        )


def _run_1d_bcdm(**kwargs) -> MultivariateBcdmResults:
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
        **kwargs,
    )


class Test1DChangeDetection(BcdmWithoutSupportTrimmingTester):
    @classmethod
    def run_bcdm(cls) -> MultivariateBcdmResults:
        return _run_1d_bcdm()


class Test1DChangeDetectionWithMaxNumProbsAndMinProb(
    BcdmWithSupportTrimmingTester
):
    max_num_probs = 10

    @classmethod
    def run_bcdm(cls) -> MultivariateBcdmResults:
        return _run_1d_bcdm(max_num_probs=cls.max_num_probs, min_prob=1e-6)

    def test_support_mask_max_size_less_equal_than_max_num_probs(self) -> None:
        mask = self.results.joint_support.full()
        max_sizes = np.full(mask.shape[0], self.max_num_probs)
        max_sizes[: self.max_num_probs] = np.arange(self.max_num_probs) + 1
        assert_array_less_strict(mask.sum(axis=1), max_sizes + 1)


class Test1DChangeDetectionWithMaxNumProbs(BcdmWithSupportTrimmingTester):
    max_num_probs = 10

    @classmethod
    def run_bcdm(cls) -> MultivariateBcdmResults:
        return _run_1d_bcdm(max_num_probs=cls.max_num_probs)

    def test_support_mask_max_size_equal_to_max_num_probs(self) -> None:
        mask = self.results.joint_support.full()
        max_sizes = np.full(mask.shape[0], self.max_num_probs)
        max_sizes[: self.max_num_probs] = np.arange(self.max_num_probs) + 1
        assert_array_equal_strict(mask.sum(axis=1), max_sizes)


class Test1DChangeDetectionWithMaxRunLength(BcdmTester):
    max_run_length = 10

    @classmethod
    def run_bcdm(cls) -> MultivariateBcdmResults:
        return _run_1d_bcdm(max_run_length=cls.max_run_length)

    def test_support_mask_is_limited_to_max_run_length(self) -> None:
        mask = self.results.joint_support.full()

        expected_mask = np.zeros_like(mask)
        for i in range(self.max_run_length):
            expected_mask[i, : i + 1] = True
        expected_mask[self.max_run_length :, : self.max_run_length] = True

        assert_array_equal_strict(mask, expected_mask)


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
        x: np.ndarray = np.linspace(0, 5, samples)
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


def assert_nig_prior_equal(a: NigPrior, b: NigPrior):
    __tracebackhide__ = True
    for field in NigPrior.__slots__:
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
    with pytest.raises(TypeError, match=r"^Cannot cast array data from dtype"):
        NigPrior(3, mean=cast(Sequence[float], [1, 2, 3j]))


def test_hook_function_called_with_consecutive_indices() -> None:
    indices = []

    def hook(index: int, *_):
        indices.append(index)

    rng = np.random.default_rng(42)
    n = 5
    multivariate_bcdm(
        rng.normal(size=(n, 3)),
        rng.normal(size=n),
        hazard=0.5,
        update_hook=hook,
    )
    assert indices == list(range(n))


def test_hook_function_called_with_correct_data_shape() -> None:
    p = 3

    def hook(t: int, x: np.ndarray, y: np.ndarray, _):
        assert x.shape == (1, p), f"x has wrong shape in hook at t={t}"
        assert y.shape == (1,), f"y has wrong shape in hook at t={t}"

    rng = np.random.default_rng(42)
    n = 5
    multivariate_bcdm(
        rng.normal(size=(n, p)),
        y=rng.normal(size=n),
        hazard=0.5,
        update_hook=hook,
    )


def test_hook_function_called_with_correct_input_data() -> None:
    p = 3
    n = 5
    X_hook = np.empty((n, p))
    Y_hook = np.empty(n)

    def hook(t: int, x: np.ndarray, y: np.ndarray, _):
        X_hook[t] = x
        Y_hook[t] = y.item()

    rng = np.random.default_rng(42)
    X = rng.normal(size=(5, 3))
    Y = rng.normal(size=5)
    multivariate_bcdm(X, Y, hazard=0.5, update_hook=hook)
    assert_array_equal_strict(X_hook, X)
    assert_array_equal_strict(Y_hook, Y)


def test_bcdm_called_with_invalid_x_dtype_fails() -> None:
    x = np.arange(10)
    assert x.dtype == np.dtype(np.int_)
    y = np.random.normal(size=x.size)  # noqa: NPY002
    with pytest.raises(TypeError, match=r"^x must be of type float64"):
        multivariate_bcdm(
            x,  # type: ignore
            y,
            hazard=0.5,
        )


class TestNigBayesianLinearRegression:
    model: NigParams
    x: np.ndarray

    n = 50
    coeffs = np.array([3.4, 2.4])

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def _fit_regression(cls) -> None:
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, cls.n) + rng.normal(size=cls.n)
        cls.x = np.c_[np.ones_like(x), x]
        y = cls.x @ cls.coeffs + rng.normal(size=cls.n)
        cls.model = NigPrior(p=2).fit_regression(cls.x, y)

    def test_model_dimensions(self) -> None:
        assert self.model.mean.shape == (self.n, 2)

    def test_fitted_coefficients_are_close_to_actual(self) -> None:
        assert_allclose(self.model.mean[-1], self.coeffs, atol=0, rtol=0.1)

    @parametrize_nig_params_attrs("attr")
    def test_parameters_snapshot(
        self, attr: str, ndarray_snapshot: NDArraySnapshot
    ):
        param = getattr(self.model, attr)
        assert_allclose(param, ndarray_snapshot(param))

    def test_mvt_mean_snapshot(self, ndarray_snapshot: NDArraySnapshot):
        mean = self.model.mvt_mean(self.x)
        assert mean.shape == (self.n,)
        assert_allclose(mean, ndarray_snapshot(mean))

    def test_mvt_variance_snapshot(self, ndarray_snapshot: NDArraySnapshot):
        variance = self.model.mvt_variance(self.x)
        assert variance.shape == (self.n,)
        assert_allclose(variance, ndarray_snapshot(variance))
