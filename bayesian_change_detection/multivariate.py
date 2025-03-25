from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import scipy
from numpy.testing import assert_allclose
from scipy.linalg import lapack

try:
    # matrix_transpose is added in numpy v2
    from numpy import matrix_transpose  # type: ignore
except ImportError:

    def matrix_transpose(a: np.ndarray) -> np.ndarray:  # type: ignore
        return np.swapaxes(a, -1, -2)


_LOG_2PI = np.log(2 * np.pi)


def _check_array_shape(name: str, x, exp: Sequence[int]) -> np.ndarray:
    if x.dtype != np.float64:
        raise TypeError(f"{name} must be of type float64, got {x.dtype}")

    s = x.shape
    if s != exp:
        raise ValueError(f"invalid shape for {name}: expected {exp}, got {s}")
    return x


def multivariate_bcdm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    prior_mean: float | np.ndarray = 0,
    prior_cov: float | np.ndarray = 1,
    prior_shape: float = 1,
    prior_scale: float = 1,
    hazard: float = 0.1,
) -> "MultivariateBcdmResults":
    """
    Bayesian change detection model for univariate response data.

    Args:
        x: (n, p) or (n,) predictor variables.
        y: (n,) response variables.
        prior_mean: Prior mean (Lambda) of the model. If a scalar, it is
            a constant vector of that value.
        prior_cov: Prior covariance matrix (Omega^-1) of the model. If a
            scalar, it is set to be a diagonal matrix with that value.
        prior_shape: Shape of the prior noise distribution.
        prior_scale: Scale of the prior noise distribution.
        hazard: Hazard rate.
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

    if np.isscalar(prior_mean):
        prior_mean = np.full(p, prior_mean, dtype=np.float64)
    else:
        prior_mean = _check_array_shape("prior_mean", prior_mean, (p,))

    if np.isscalar(prior_cov):
        prior_cov = np.eye(p, dtype=np.float64) * cast(float, prior_cov)
    else:
        prior_cov = _check_array_shape("prior_cov", prior_cov, (p, p))
        if np.any(np.linalg.eigvals(prior_cov) <= 0):
            raise ValueError("prior_cov must be positive-definite")

    if prior_shape <= 0:
        raise ValueError("prior_shape must be > 0")
    prior_shape = float(prior_shape)

    if prior_scale <= 0:
        raise ValueError("prior_scale must be > 0")
    prior_scale = float(prior_scale)

    worker = _MultivariateBcdmWorker(
        x,
        y,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        prior_shape=prior_shape,
        prior_scale=prior_scale,
        hazard=hazard,
    )
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
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        prior_shape: float,
        prior_scale: float,
        hazard: float,
    ):
        self.n, self.p = x.shape
        self.x = x
        self.y = y
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.prior_prec = _positive_definite_inv(prior_cov)
        self.prior_shape = prior_shape
        self.prior_scale = prior_scale

        self.params = NigParams.empty(self.n, self.p)
        self.log_joint = np.array([0.0])
        self.log_posteriors: list[np.ndarray] = []
        self.log_hazard = np.log(hazard)
        self.log_1mhazard = np.log1p(-hazard)

    def _add_new_hypothesis(self, t: int) -> None:
        self.params.mean[t] = self.prior_mean
        self.params.cov[t] = self.prior_cov
        self.params.shape[t] = self.prior_shape
        self.params.scale[t] = self.prior_scale
        self.params.prec[t] = self.prior_prec

    def fit(self) -> MultivariateBcdmResults:
        for t, (x, y) in enumerate(zip(self.x, self.y, strict=True)):
            self._step(t, x, y)

        n = len(self.log_posteriors)
        log_posterior: np.ndarray = np.empty((n, n))
        log_posterior[np.tril_indices(n)] = np.concatenate(self.log_posteriors)
        log_posterior[np.triu_indices(n, 1)] = -np.inf
        log_posterior = log_posterior.T
        assert_allclose(
            scipy.special.logsumexp(log_posterior, axis=0),
            0,
            atol=1e-10,
        )
        return MultivariateBcdmResults(
            mean=self.params.mean,
            cov=self.params.cov,
            shape=self.params.shape,
            scale=self.params.scale,
            log_posterior=log_posterior,
        )

    def _step(self, t: int, x: np.ndarray, y: float) -> None:
        self._add_new_hypothesis(t)
        log_predictive_probs = self.params[: t + 1].mvt_logpdf(x, y)
        assert log_predictive_probs.shape == (t + 1,)

        # Compute the (t + 1) changepoint probabilities
        log_joint = np.empty(t + 1)
        log_joint[0] = scipy.special.logsumexp(  # reset proabability
            log_predictive_probs[-1] + self.log_hazard + self.log_joint
        )
        log_joint[1:] = (  # growth probabilities
            log_predictive_probs[:-1][::-1]
            + self.log_1mhazard
            + self.log_joint
        )

        # Normalise to get the posterior
        log_posterior = log_joint - scipy.special.logsumexp(log_joint)
        self.log_posteriors.append(log_posterior)
        self.log_joint = log_joint
        self.params[: t + 1].update(x.reshape(1, -1), np.asarray((y,)))


@dataclass
class NigParams:
    """
    Parameters of t independent p-dimensional normal-inverse-gamma
    distributions.
    """

    mean: np.ndarray
    """
    (t, p) array of means.
    """

    cov: np.ndarray
    """
    (t, p, p) array of covariances.
    """

    prec: np.ndarray
    """
    (t, p, p) array of precisions.
    """

    shape: np.ndarray
    """
    (t,) array of shape parameters.
    """

    scale: np.ndarray
    """
    (t,) array of scale parameters.
    """

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

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Update sufficient statistics given n new observations (x, y).

        Args:
            x: (n, p) array of independent (predictor) variables.
            y: (n,) array of dependent (response) variables.
        """
        if x.ndim != 2:
            raise ValueError("x must be 2-D")
        n, xp = x.shape
        _check_array_shape("y", y, (n,))
        t, p = self.mean.shape
        if xp != p:
            raise ValueError(f"expected {p}-dimensional predictors, got {xp}")

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
        vx = cov0 @ matrix_transpose(x)
        assert vx.shape == (t, p, n)

        # (1, n, p) @ (t, p, n) -> (t, n, n)
        xvx = x @ vx
        assert xvx.shape == (t, n, n)

        # Eq. 30, second equality
        if n == 1:
            # (t, p, n) @ (t, n, p) -> (t, p, p)
            vxxv = vx @ matrix_transpose(vx)
            assert vxxv.shape == (t, p, p)

            # (t, p, p) - (t, p, p) / (t, 1, 1) -> (t, p, p)
            new_cov = cov0 - vxxv / (xvx + 1)
        else:
            # TODO: use more efficient matrix inversion code from below?
            xvx_p1_inv = np.linalg.inv(xvx + np.eye(n))

            # (t, p, p) - (t, p, n) @ (t, n, n) @ (t, n, p) -> (t, p, p)
            new_cov = cov0 - vx @ xvx_p1_inv @ matrix_transpose(vx)

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

    def mvt_logpdf(self, x: np.ndarray, y: float) -> np.ndarray:
        """
        Calculate the log PDFs of the predictive distributions of n multiple
        independent Bayesian linear regression models using self as the NIG
        priors. (Logarithm of Eq. 27.)

        Args:
            x: (p,) array of predictor variables.
            y: response variable.

        Returns:
            (n,) array of log predictive probabilities.
        """

        assert x.ndim == 1
        assert np.isscalar(y)
        n = len(self.mean)

        x = x.reshape(1, -1, 1)  # (1, p, 1)

        # Σ from Eq. 27
        # (1, 1, p) @ (n, p, p) @ (1, p, 1) -> (n, 1, 1)
        xvx = matrix_transpose(x) @ self.cov @ x
        assert xvx.shape == (n, 1, 1)
        sigma = self.scale * (xvx.ravel() + 1)
        assert sigma.shape == (n,)

        # (1, 1, p) @ (n, p, 1) -> (n, 1, 1)
        loc = matrix_transpose(x) @ np.expand_dims(self.mean, -1)
        assert loc.shape == (n, 1, 1)

        # Quadratic term in PDF
        dev_squared = np.square(y - loc.ravel())
        assert dev_squared.shape == (n,)

        a = self.shape  # 0.5 * degrees of freedom
        a_plus_half = a + 0.5
        logpdf = (
            scipy.special.gammaln(a_plus_half)
            - scipy.special.gammaln(a)
            - (_LOG_2PI + np.log(sigma)) / 2
            - a_plus_half * np.log(1 + dev_squared / sigma / 2)
        )
        assert logpdf.shape == (n,)
        return logpdf


def _positive_definite_inv(a: np.ndarray) -> np.ndarray:
    n = len(a)
    if n == 1:
        return 1 / a
    if n == 2:
        return _2x2_inv(a)
    return _cholesky_inv(a)


def _2x2_inv(a: np.ndarray) -> np.ndarray:
    det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    if det == 0:
        raise ValueError("matrix is singular")

    return (
        np.array(
            (
                (a[1, 1], -a[0, 1]),
                (-a[1, 0], a[0, 0]),
            )
        )
        / det
    )


def _cholesky_inv(a: np.ndarray) -> np.ndarray:
    """Invert a positive-definite matrix using the Cholesky decomposition."""

    # u is the upper triangular matrix of the Cholesky decomposition
    # a = u.T @ u
    u, info = lapack.dpotrf(a)
    if info != 0:
        raise ValueError("matrix is not positive-definite")

    # dpotri only returns the upper triangular part of the inverse
    uinv, info = lapack.dpotri(u)
    if info != 0:
        raise ValueError("matrix is singular")

    # Not sure if lapack sets the lower diagonal to zero or if it leaves it
    # unset: explicitly set the lower diagonal indices, which is slightly
    # slower than `uinv += np.triu(uinv, 1).T`.
    idx = np.tril_indices_from(uinv, -1)
    uinv[idx] = uinv.T[idx]
    return uinv
