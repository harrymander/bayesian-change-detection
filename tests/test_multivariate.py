from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.testing
import pytest
import scipy

from bayesian_change_detection import (
    MultivariateBcdmResults,
    multivariate_bcdm,
)
from bayesian_change_detection._tril import TrilArray
from bayesian_change_detection.multivariate import NigParams, NigPrior
from tests.conftest import JsonSnapshot, NDArraySnapshot
from tests.utils import (
    assert_allclose,
    assert_array_equal_strict,
    assert_array_less_strict,
)


def new_params(t: int, p: int, *, dtype=np.float64) -> NigParams:
    """
    Generate new params for t NIG distributions of dimension p.
    """
    cov = np.stack([np.eye(p) * i for i in np.linspace(1, 4, t)]).astype(dtype)
    assert cov.shape == (t, p, p)
    params = NigParams(
        mean=np.linspace(0, 5, t, dtype=dtype).repeat(p).reshape(-1, p),
        cov=cov,
        prec=np.linalg.inv(cov),
        shape=np.linspace(2, 5, t, dtype=dtype),
        scale=np.linspace(2, 5, t, dtype=dtype),
    )
    for attr in params.__dataclass_fields__:
        attr_dtype = getattr(params, attr).dtype
        msg = f"invalid dtype for {attr}: {attr_dtype} != {dtype}"
        assert attr_dtype == dtype, msg
    return params


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


def test_mvt_logpdf_with_float32_args() -> None:
    p = 3
    exp_dtype = np.float32
    params = new_params(5, p, dtype=exp_dtype)
    n = 500
    rng = np.random.default_rng(5431)
    x = rng.normal(size=(n, p)).astype(exp_dtype)
    y = rng.normal(size=(n,)).astype(exp_dtype)
    logpdf = params.mvt_logpdf(x, y)
    assert logpdf.dtype == exp_dtype


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

    @parametrize_attrs("attrname")
    def test_parameters_all_nan_outside_final_support(self, attrname: str):
        parameters = getattr(self.results, attrname)
        nans = np.isnan(parameters).reshape((parameters.shape[0], -1))
        support = self.results.joint_support
        assert_array_equal_strict(
            support[support.n - 1],
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
    max_num_probs: int

    def test_support_mask_is_lower_triangular(self) -> None:
        support = self.results.joint_support.full()
        support_triu = support[np.triu_indices_from(support, 1)]
        assert_array_equal_strict(
            support_triu,
            np.zeros_like(support_triu),
            err_msg="Support is not lower-triangular",
        )

    def test_support_mask_max_size(self) -> None:
        mask = self.results.joint_support.full()
        max_sizes = np.full(mask.shape[0], self.max_num_probs)
        max_sizes[: self.max_num_probs] = np.arange(self.max_num_probs) + 1
        assert_array_less_strict(mask.sum(axis=1), max_sizes + 1)


def _generate_1d_data(*, dtype) -> np.ndarray:
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

    return np.asarray(data, dtype=dtype)


def _1d_multivariate_bcdm(*, dtype=np.float64, **kwargs):
    kwargs = dict(cov=2, hazard=0.1) | kwargs
    data = _generate_1d_data(dtype=dtype)
    assert data.dtype == dtype
    return multivariate_bcdm(np.ones_like(data), data, **kwargs)


class Test1DChangeDetection(BcdmWithoutSupportTrimmingTester):
    @classmethod
    def run_bcdm(cls) -> MultivariateBcdmResults:
        return _1d_multivariate_bcdm()


class Test1DChangeDetectionWithSupportTrimming(BcdmWithSupportTrimmingTester):
    max_num_probs = 10

    @classmethod
    def run_bcdm(cls) -> MultivariateBcdmResults:
        return _1d_multivariate_bcdm(
            max_num_probs=cls.max_num_probs,
            min_prob=1e-12,
        )


def _2d_multivariate_bcdm(*, dtype=np.float64) -> MultivariateBcdmResults:
    rng = np.random.default_rng(42)

    # Create input and outputs.
    samples = 100
    x = np.linspace(0, 3 * 2 * np.pi, samples, dtype=dtype)
    assert x.dtype == dtype
    noise = 0.1 * rng.standard_normal(samples, dtype=dtype)
    y = -np.arcsin(np.sin(x)) * 2 / np.pi + noise
    assert y.dtype == dtype
    return multivariate_bcdm(
        np.c_[np.ones_like(x), x],
        y,
        hazard=0.02,
        cov=1e6,
        shape=1e-3,
        scale=1e-6,
    )


class Test2DChangeDetection(BcdmWithoutSupportTrimmingTester):
    @classmethod
    def run_bcdm(cls) -> MultivariateBcdmResults:
        return _2d_multivariate_bcdm()


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


def _xfail_param(*args, reason: str = ""):
    return pytest.param(*args, marks=pytest.mark.xfail(reason=reason))


def parametrize_results_generators(name: str):
    generators = [
        ("1d", _1d_multivariate_bcdm),
        ("2d", _xfail_param(_2d_multivariate_bcdm, reason="Unstable")),
    ]
    return pytest.mark.parametrize(
        name,
        [g[1] for g in generators],
        ids=[g[0] for g in generators],
    )


float32_fields = list(MultivariateBcdmResults.__dataclass_fields__)
float32_fields.pop(float32_fields.index("joint_support"))


@parametrize_results_generators("generator")
@pytest.mark.parametrize("attr", float32_fields)
def test_bcdm_with_float32_has_float32_result(generator, attr: str) -> None:
    results = generator(dtype=np.float32)
    dtype = getattr(results, attr).dtype
    assert dtype == np.float32


@parametrize_results_generators("generator")
def test_bcdm_with_float32_has_float32_log_posterior(generator) -> None:
    results = generator(dtype=np.float32)
    dtype = results.log_posterior().dtype
    assert dtype == np.float32


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
