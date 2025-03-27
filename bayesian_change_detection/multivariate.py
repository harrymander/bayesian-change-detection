from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict, Unpack, cast, overload

import numpy as np
import scipy

from bayesian_change_detection.linalg import (
    inv_positive_definite,
    matrix_transpose,
)

_LOG_2PI = np.log(2 * np.pi)


def _check_array_shape(name: str, x, exp: Sequence[int]) -> np.ndarray:
    if x.dtype != np.float64:
        raise TypeError(f"{name} must be of type float64, got {x.dtype}")

    s = x.shape
    if s != exp:
        raise ValueError(f"invalid shape for {name}: expected {exp}, got {s}")
    return x


class NigPriorKwargs(TypedDict, total=False):
    mean: float | np.ndarray
    cov: float | np.ndarray
    shape: float
    scale: float


@overload
def multivariate_bcdm(
    x: np.ndarray,
    y: np.ndarray,
    hazard: float,
    prior: None = None,
    **prior_kwargs: Unpack[NigPriorKwargs],
) -> "MultivariateBcdmResults": ...


@overload
def multivariate_bcdm(
    x: np.ndarray,
    y: np.ndarray,
    hazard: float,
    prior: "NigParams" = ...,
) -> "MultivariateBcdmResults": ...


def multivariate_bcdm(
    x: np.ndarray,
    y: np.ndarray,
    hazard: float,
    prior: "None | NigParams" = None,
    **prior_kwargs: Unpack[NigPriorKwargs],
) -> "MultivariateBcdmResults":
    """
    Bayesian change detection model for univariate response data.

    Args:
        x: (n, p) or (n,) predictor variables.
        y: (n,) response variables.
        hazard: Hazard rate.
        prior: NIG prior parameters
        **prior_kwargs: Parameters for NIG prior if prior is None.
    """
    if x.ndim == 1:
        n = len(x)
        p = 1
        x = x.reshape(-1, 1)
    elif x.ndim != 2:
        raise ValueError("x must be 1- or 2-D")
    else:
        n, p = x.shape

    _check_array_shape("y", y, (n,))

    if prior is None:
        prior = NigParams.from_priors(p, t=1, **prior_kwargs)
    elif prior_kwargs:
        raise ValueError(
            "cannot pass prior keyword arguments if a prior object is passed"
        )
    elif prior.mean.shape[0] != 1:
        raise ValueError("prior must have a single distribution")

    worker = _MultivariateBcdmWorker(x, y, prior=prior, hazard=hazard)
    return worker.fit()


@dataclass
class MultivariateBcdmResults:
    mean: np.ndarray
    cov: np.ndarray
    shape: np.ndarray
    scale: np.ndarray

    log_posterior: np.ndarray
    """
    An upper-triangular matrix where each column contains the
    natural logarithm of the posterior probability of the segment running
    length at that time point.

    E.g. for 4 samples at time points `a`, `b`, `c`, and `d`, returns a
    matrix:

      ```
      a0 b0 c0 d0
      -  b1 c1 d1
      -  -  c2 d2
      -  -  -  d3
      ```

    where `c2` is the log probability of the point at time `c` belonging to
    a segment that began 2 time points before, for example. The elements on
    the lower diagonal are `-inf`, corresponding to a zero probability
    (i.e. a point cannot belong to a segment longer than the number of
    points observed so far).
    """

    def changepoints(self) -> list[int]:
        """Compute indices of changepoints in ascending order. Returned list
        does not include first and last indices, which may be changepoints."""
        changepoints: list[int] = []
        run_length = self.log_posterior[:, -1].argmax()
        i = len(self.log_posterior) - 1 - run_length
        while i > 0:
            changepoints.append(i.item())
            i -= 1 + self.log_posterior[: i + 1, i].argmax()

        changepoints.reverse()
        return changepoints


class _MultivariateBcdmWorker:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        prior: "NigParams",
        hazard: float,
    ):
        # Assumes parameters have been validated
        self.n, self.p = x.shape
        self.x = x
        self.y = y
        self.prior = prior
        self.params = NigParams.empty(self.n, self.p)
        self.log_hazard = np.log(hazard)
        self.log_1mhazard = np.log1p(-hazard)
        self.prev_log_joint = np.array([0.0])
        self.log_joint = np.full((self.n, self.n), -np.inf)

    def _add_new_hypothesis(self, t: int) -> None:
        self.params.mean[t] = self.prior.mean[0]
        self.params.cov[t] = self.prior.cov[0]
        self.params.shape[t] = self.prior.shape[0]
        self.params.scale[t] = self.prior.scale[0]
        self.params.prec[t] = self.prior.prec[0]

    def _log_posterior(self) -> np.ndarray:
        """Convert joint probabilities to a triangular matrix of posterior
        probabilities. See docstring in MultivariateBcdmResults for description
        of the matrix."""
        log_joint_trans = self.log_joint.T
        col_sums = scipy.special.logsumexp(log_joint_trans, axis=0)
        return log_joint_trans - col_sums

    def fit(self) -> MultivariateBcdmResults:
        for t, (x, y) in enumerate(zip(self.x, self.y, strict=True)):
            self._step(t, x, y)

        return MultivariateBcdmResults(
            mean=self.params.mean,
            cov=self.params.cov,
            shape=self.params.shape,
            scale=self.params.scale,
            log_posterior=self._log_posterior(),
        )

    def _step(self, t: int, x: np.ndarray, y: float) -> None:
        self._add_new_hypothesis(t)
        log_pred = self.params[: t + 1].mvt_logpdf(
            x.reshape(1, -1),
            np.atleast_1d(y),
        )
        assert log_pred.shape == (t + 1, 1)
        log_pred = log_pred.ravel()

        # Compute the (t + 1) changepoint probabilities
        log_joint = self.log_joint[t, : t + 1]
        log_joint[0] = scipy.special.logsumexp(  # reset probability
            log_pred[-1] + self.log_hazard + self.prev_log_joint
        )
        log_joint[1:] = (  # growth probabilities
            log_pred[:-1][::-1] + self.log_1mhazard + self.prev_log_joint
        )

        self.prev_log_joint = log_joint
        self.params[: t + 1].update(x.reshape(1, -1), np.asarray((y,)))


@dataclass
class NigParams:
    """
    Parameters of t independent p-dimensional normal-inverse-gamma
    distributions.

    Recommended to use from_priors to initialise, rather than constructing
    directly.
    """

    mean: np.ndarray
    """
    (t, p) array of means.
    """

    cov: np.ndarray
    """
    (t, p, p) array of covariances.
    """

    shape: np.ndarray
    """
    (t,) array of shape parameters.
    """

    scale: np.ndarray
    """
    (t,) array of scale parameters.
    """

    prec: np.ndarray
    """
    (t, p, p) array of precisions.
    """

    def __post_init__(self) -> None:
        if self.mean.ndim != 2:
            raise ValueError("mean must be a 2D array")

        t, p = self.mean.shape

        def _validate_param(name: str, ndims: int) -> None:
            param = getattr(self, name)
            if param.ndim != ndims + 1:
                raise ValueError(f"{name} must be a {ndims + 1}D array")

            expected_shape = (t, *(p for _ in range(ndims)))
            if param.shape != expected_shape:
                raise ValueError(
                    f"expected {name} to be array of shape {expected_shape}, "
                    f"got {param.shape}"
                )

        _validate_param("cov", 2)
        _validate_param("shape", 0)
        _validate_param("scale", 0)
        _validate_param("prec", 2)

    @classmethod
    def from_priors(
        cls,
        p: int,
        t: int = 1,
        *,
        mean: float | np.ndarray = 0,
        cov: float | np.ndarray = 1,
        shape: float = 1,
        scale: float = 1,
    ):
        if np.isscalar(mean):
            mean = np.full(p, mean, dtype=np.float64)
        else:
            _check_array_shape("mean", mean, (p,))

        if np.isscalar(cov):
            cov = np.eye(p, dtype=np.float64) * cast(float, cov)
        else:
            _check_array_shape("cov", cov, (p, p))

        if shape <= 0:
            raise ValueError("shape must be > 0")
        if scale <= 0:
            raise ValueError("scale must be > 0")

        prec = inv_positive_definite(cast(np.ndarray, cov))
        return cls(
            mean=np.expand_dims(mean, 0).repeat(t, axis=0),
            cov=np.expand_dims(cov, 0).repeat(t, axis=0),
            prec=np.expand_dims(prec, 0).repeat(t, axis=0),
            shape=np.full(t, shape, dtype=np.float64),
            scale=np.full(t, scale, dtype=np.float64),
        )

    @classmethod
    def empty(cls, t: int, p: int) -> "NigParams":
        return cls(
            mean=np.empty((t, p)),
            cov=np.empty((t, p, p)),
            prec=np.empty((t, p, p)),
            shape=np.empty(t),
            scale=np.empty(t),
        )

    def __getitem__(self, i) -> "NigParams":
        return NigParams(
            mean=self.mean[i],
            cov=self.cov[i],
            prec=self.prec[i],
            shape=self.shape[i],
            scale=self.scale[i],
        )

    def _validate_x_arg(self, x: np.ndarray) -> None:
        if x.ndim != 2:
            raise ValueError("x must be 2-D")
        xp = x.shape[1]
        p = self.mean.shape[1]
        if xp != p:
            raise ValueError(f"expected {p}-dimensional predictors, got {xp}")

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Update sufficient statistics given n new observations (x, y).

        Args:
            x: (n, p) array of independent (predictor) variables.
            y: (n,) array of dependent (response) variables.
        """
        self._validate_x_arg(x)
        n = x.shape[0]
        _check_array_shape("y", y, (n,))
        t, p = self.mean.shape

        cov0 = self.cov  # (t, p, p)
        prec0 = self.prec  # (t, p, p)
        mean0 = np.expand_dims(self.mean, -1)  # (t, p, 1)

        # Add dimension for broadcasting
        x = np.expand_dims(x, 0)  # (1, n, p)

        # (1, p, n) @ (1, n, p) -> (1, p, p)
        xx = matrix_transpose(x) @ x
        assert xx.shape == (1, p, p)

        # Eq. 30, first equality
        # (t, p, p) + (1, p, p)
        new_prec = prec0 + xx
        assert new_prec.shape == (t, p, p)

        # (t, p, p) @ (1, p, n) -> (t, p, n)
        if p >= n:
            # If p >= n, we can use the Woodbury matrix identity to avoid
            # inverting the larger (p, p) matrix. See Eq. 30, second equality.

            vx = cov0 @ matrix_transpose(x)
            assert vx.shape == (t, p, n)

            # (1, n, p) @ (t, p, n) -> (t, n, n)
            xvx = x @ vx
            assert xvx.shape == (t, n, n)

            if n == 1:
                # (t, p, n) @ (t, n, p) -> (t, p, p)
                vxxv = vx @ matrix_transpose(vx)
                assert vxxv.shape == (t, p, p)

                # (t, p, p) - (t, p, p) / (t, 1, 1) -> (t, p, p)
                new_cov = cov0 - vxxv / (xvx + 1)
            else:
                # Matrix may not be positive definite
                xvx_p1_inv = np.linalg.inv(xvx + np.eye(n))

                # (t, p, p) - (t, p, n) @ (t, n, n) @ (t, n, p) -> (t, p, p)
                new_cov = cov0 - vx @ xvx_p1_inv @ matrix_transpose(vx)
        else:
            # Just invert the (t, p, p) matrix. See Eq. 30, first equality.
            new_cov = inv_positive_definite(new_prec)

        assert new_cov.shape == (t, p, p)

        # (1, p, n) * (1, n, 1) -> (1, p, 1)
        xy = matrix_transpose(x) @ y.reshape(1, -1, 1)
        assert xy.shape == (1, p, 1)

        # Eq. 31
        # (t, p, p) @ ((t, p, p) @ (t, p, 1) + (1, p, 1))
        # -> (t, p, p) @ (t, p, 1)
        # -> (t, p, 1)
        new_mean = new_cov @ (prec0 @ mean0 + xy)
        assert new_mean.shape == (t, p, 1)

        # (t, 1, p) @ (t, p, p) @ (t, p, 1) -> (t, 1, 1)
        mean_prec_mean0 = matrix_transpose(mean0) @ prec0 @ mean0
        assert mean_prec_mean0.shape == (t, 1, 1)

        # as above
        mean_prec_mean_new = matrix_transpose(new_mean) @ new_prec @ new_mean
        assert mean_prec_mean_new.shape == (t, 1, 1)

        # Eq. 33
        new_scale = (
            mean_prec_mean0.ravel() + np.dot(y, y) - mean_prec_mean_new.ravel()
        ) / 2
        assert new_scale.shape == (t,)

        self.mean[:] = new_mean.squeeze(-1)
        self.cov[:] = new_cov
        self.prec[:] = new_prec
        self.shape += n / 2  # Eq. 32
        self.scale += new_scale
        assert np.all(self.scale >= 0), "got negative scale"

    def _mvt_mean(self, x: np.ndarray) -> np.ndarray:
        """
        Mean of the t-distribution. Assumes has valid shape.

          x(j)^T @ μ(i) for i = 1...t and j=1...n
        """
        mean = np.einsum("np,tp->tn", x, self.mean)
        assert mean.shape == (len(self.mean), len(x))
        return mean

    def _xvx_product(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the quadratic form in the t-distribution shape param (Eq. 26):

          x(j)^T V(i) x(j) for i = 1...t and j=1...n

        where V(i) is the (p, p) covariance matrix of the i-th distribution.

        Returns: (t, n) array
        """
        res = np.einsum("tij,nj,ni->tn", self.cov, x, x)
        assert res.shape == (self.mean.shape[0], len(x))
        return res

    def mvt_logpdf(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the log PDFs of the predictive distributions of t independent
        Bayesian linear regression models using self as the NIG priors.
        (Logarithm of Eq. 27.)

        Args:
            x: (n, p) array of predictor variables.
            y: (n,) response variable.

        Returns:
            (t, n) array of log predictive probabilities, the i-th row contains
            the probabilities over (x, y) given the i-th model.
        """
        self._validate_x_arg(x)
        n = x.shape[0]
        _check_array_shape("y", y, (n,))
        t = self.mean.shape[0]

        # Σ from Eq. 27
        # Product across columns: (t, 1) * (t, n) -> (t, n)
        sigma = self.scale.reshape(-1, 1) * (self._xvx_product(x) + 1)
        assert sigma.shape == (t, n)

        # Quadratic term in PDF
        # (n,) - (t, n) -> (t, n)
        dev_squared = np.square(y - self._mvt_mean(x))
        assert dev_squared.shape == (t, n)

        # 0.5 * degrees of freedom - reshape for broadcasting
        a = self.shape.reshape(-1, 1)
        a_plus_half = a + 0.5
        logpdf = (
            scipy.special.gammaln(a_plus_half)
            - scipy.special.gammaln(a)
            - (_LOG_2PI + np.log(sigma)) / 2
            - a_plus_half * np.log(1 + dev_squared / sigma / 2)
        )

        assert logpdf.shape == (t, n)
        return logpdf

    def _validate_axis_t_arg(self, x: np.ndarray) -> None:
        t = len(self.mean)
        n = len(x)
        if n != t:
            msg = (
                f"in axis='t' mode, x, which has {n} rows, must have the same"
                f" number of rows as there are models ({t})"
            )
            raise ValueError(msg)

    def mvt_mean(
        self,
        x: np.ndarray,
        axis: Literal["n", "t"] = "n",
    ) -> np.ndarray:
        """
        Calculate the expected value of y given x using t independent Bayesian
        linear regression models with NIG priors.

        This is the mean of the t-distribution, the PDF of which can be
        obtained from mvt_logpdf.

        Args:
            x: array of predictor variables; (n, p) if axis == 'n', otherwise
                must be shape (t, p) if axis == 't' (i.e. have the same number
                of rows as models).
            axis: Axis along which to calculate the expected values. If 'n',
                returns the expected value for each model across all
                observations. If 't', returns the expected value of each
                observation given the corresponding model index.

        Returns: array of expected values; has shape (t, n) if axis == 'n' or
            shape (t,) if axis == 't'.
        """
        if axis not in "nt":
            raise ValueError("axis must be 'n' or 't'")

        self._validate_x_arg(x)
        if axis == "n":
            return self._mvt_mean(x)

        self._validate_axis_t_arg(x)

        # The dot products between the rows of x and self.mean
        return np.einsum("tp,tp->t", x, self.mean)

    def mvt_variance(
        self,
        x: np.ndarray,
        axis: Literal["n", "t"] = "n",
    ) -> np.ndarray:
        """
        Calculate the variances of y given x using t independent Bayesian
        linear regression models with NIG priors.

        This the variance of the t-distribution, the PDF of which can be
        obtained from mvt_logpdf. The variance of a t-distribution with v
        degrees of freedom and Σ shape parameter is

          v * Σ / (v - 2)

        The degrees of freedom are 2*self.shape; Σ is equivalent to the term in
        Eq. 26.

        Args:
            x: array of predictor variables; (n, p) if axis == 'n', otherwise
                must be shape (t, p) if axis == 't' (i.e. have the same number
                of rows as models).
            axis: Axis along which to calculate the variances. If 'n', returns
                the variance for each model across all observations. If 't',
                returns the variance of each observation given the
                corresponding model index.

        Returns: array of variances; has shape (t, n) if axis == 'n' or shape
            (t,) if axis == 't'.
        """
        if axis not in "nt":
            raise ValueError("axis must be 'n' or 't'")

        self._validate_x_arg(x)

        coeff = self.scale / (self.shape - 1)

        if axis == "n":
            # Product across columns: (t, 1) * (t, n) -> (t, n)
            var = coeff.reshape(-1, 1) * (1 + self._xvx_product(x))
            assert var.shape == (self.mean.shape[0], len(x))
            return var

        self._validate_axis_t_arg(x)

        # Compute x[i]^T @ V[i] @ x[i] for i = 1...t
        # This is equivalent to the diagonal of the value when axis == "n"
        xvx = np.einsum("tij,tj,ti->t", self.cov, x, x)
        assert xvx.shape == (len(x),)
        return coeff / (1 + xvx)

    def fit_regression(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> "NigParams":
        """
        Fit a Bayesian linear regression model using this NIG prior.

        Returns the NIG parameters after each observation.

        ```python
        prior = NigPrior.from_priors(2)
        # x.shape == is (n, 2) and y.shape == (n,)
        model = prior.fit_regression(x, y)

        # The MAP estimate for the regression coefficients is the mean of the
        # final model:
        beta = model.mean[-1]

        # Compute the mean and variance after each observation
        mean = model.mvt_mean(x, axis="t")
        var = model.mvt_variance(x, axis="t")
        ```

        Note that if the prior shape if small enough (< 1), then the variance
        may be negative for the first observation or two.

        Args:
            x: (n, p) array of predictor variables.
            y: (n,) array of response variables.

        Returns: NigParams object with n distributions, representing the model
            parameters after observing each (x, y).
        """
        self._validate_x_arg(x)
        n, p = x.shape
        _check_array_shape("y", y, (n,))

        if self.mean.shape[0] != 1:
            msg = "can only fit regression model from a single prior"
            raise ValueError(msg)

        params = NigParams.from_priors(
            p=p,
            t=n,
            cov=self.cov[0],
            mean=self.mean[0],
            shape=self.shape[0],
            scale=self.scale[0],
        )
        for t, (xt, yt) in enumerate(zip(x, y, strict=True)):
            assert xt.ndim == 1, xt.shape
            assert np.isscalar(yt), yt
            params[t:].update(xt.reshape(1, -1), np.atleast_1d(yt))

        return params
