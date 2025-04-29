from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    Protocol,
    TypedDict,
    TypeVar,
    Unpack,
    cast,
    overload,
)

import numpy as np
import scipy
from numpy.typing import NDArray

from bayesian_change_detection._tril import TrilArray, tril_full
from bayesian_change_detection.array_utils import masked_argmin
from bayesian_change_detection.linalg import (
    inv_positive_definite,
    matrix_transpose,
)

_LOG_2PI = np.log(2 * np.pi)
_Floating_T = TypeVar("_Floating_T", bound=np.floating)
FloatArray = NDArray[_Floating_T]


class _HasDtype(Protocol):
    @property
    def dtype(self) -> np.dtype: ...


def _check_dtype(name: str, arr: _HasDtype, dtype):
    if arr.dtype != dtype:
        raise TypeError(f"{name} must be of type {dtype}, got {arr.dtype}")


def _check_array_shape(name: str, x: np.ndarray, exp: Sequence[int]) -> None:
    s = x.shape
    if s != exp:
        raise ValueError(f"invalid shape for {name}: expected {exp}, got {s}")


PriorMeanArg = float | Sequence[float] | np.ndarray
PriorCovArg = float | Sequence[float] | Sequence[Sequence[float]] | np.ndarray


class NigPriorKwargs(TypedDict, total=False):
    # See NigPrior.from_priors for default values
    mean: PriorMeanArg
    cov: PriorCovArg
    shape: float
    scale: float


class HookData(Protocol):
    @property
    def joint_support(self) -> TrilArray: ...

    @property
    def log_joint(self) -> TrilArray: ...

    @property
    def log_pred(self) -> TrilArray: ...

    @property
    def params(self) -> "NigParams": ...


class UpdateHook(Protocol):
    def __call__(
        self,
        t: int,
        x: np.ndarray,
        y: np.ndarray,
        data: HookData,
        /,
    ) -> Any:
        pass


@overload
def multivariate_bcdm(
    x: FloatArray,
    y: FloatArray,
    hazard: float,
    prior: None = None,
    *,
    init_prob: float = ...,
    min_prob: float = ...,
    max_num_probs: int | None = ...,
    update_hook: UpdateHook | None = None,
    **prior_kwargs: Unpack[NigPriorKwargs],
) -> "MultivariateBcdmResults": ...


@overload
def multivariate_bcdm(
    x: FloatArray,
    y: FloatArray,
    hazard: float,
    prior: "NigPrior" = ...,
    *,
    init_prob: float = ...,
    min_prob: float = ...,
    max_num_probs: int | None = ...,
    update_hook: UpdateHook | None = None,
) -> "MultivariateBcdmResults": ...


def multivariate_bcdm(
    x: FloatArray,
    y: FloatArray,
    hazard: float,
    prior: "None | NigPrior" = None,
    *,
    init_prob: float = 1,
    min_prob: float = 0,
    max_num_probs: int | None = None,
    update_hook: UpdateHook | None = None,
    **prior_kwargs: Unpack[NigPriorKwargs],
) -> "MultivariateBcdmResults":
    """
    Bayesian change detection model for univariate response data.

    Args:
        x: (n, p) or (n,) predictor variables.
        y: (n,) response variables.
        hazard: Hazard rate.
        prior: NIG prior parameters. Cannot be passed if any `prior_kwargs` are
            already passed.
        init_prob: The initial reset probability, i.e., the probability of a
            new segment beginning at the first datapoint.
        min_prob: Minimum changepoint posterior probability. If greater than
            zero, posterior probabilities less than this value will be
            zeroed-out after each time step. This can significantly reduce the
            processing time for large data.
        max_num_probs: Maximum number of changepoint probabilities to keep
            after each time step. If provided, will zero out all but the top
            `max_num_probs` posterior probabilities after each time step. This
            can significantly reduce the processing time for large data.
        update_hook: Optional function to call after each datapoint is
            processed. If provided, will be called with the following
            arguments after processing each datapoint in (x, y):
              - `t`: The index of the current datapoint (0 to `n - 1`).
              - `x`: Predictor variable that was just processed.
              - `y`: Response variable that was just processed.
              - `data`: Current state of the model. See `HookData`.
            This can be used, for example, for progress indication (see
                multivariate_examples.py for an example of such usage).
        **prior_kwargs: Parameters for NIG prior if prior is None. See
            `NigPrior` for details. Cannot be passed if `prior` is already
            passed.

    Returns:
        Change detection results.

    Raises:
        ValueError: If any arguments have the wrong dimension, shape, or are
            otherwise invalid.
        TypeError: If dtypes of `x` and `y` are not floating point or they are
            not of the same type.
    """
    if not (0 < hazard < 1):
        raise ValueError("hazard must be in (0, 1)")
    if not (0 <= init_prob <= 1):
        raise ValueError("init_prob must be in [0, 1]")
    if not (0 <= min_prob < 1):
        raise ValueError("min_prob must be in [0, 1)")
    if max_num_probs is not None and max_num_probs <= 0:
        raise ValueError("max_num_probs must be > 0")

    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError(f"x must have a floating dtype, got {x.dtype}")

    if x.ndim == 1:
        n = len(x)
        p = 1
        x = x.reshape(-1, 1)
    elif x.ndim != 2:
        raise ValueError("x must be 1- or 2-D")
    else:
        n, p = x.shape

    dtype = x.dtype
    _check_dtype("y", y, dtype)
    _check_array_shape("y", y, (n,))

    if prior:
        if prior_kwargs:
            msg = "cannot pass prior kwargs if a prior object is passed"
            raise ValueError(msg)
        if prior.p != p:
            raise ValueError(
                f"dimensionality of prior ({prior.p}) does not match "
                f"that of the data ({p})"
            )
        _check_dtype("prior", prior, dtype)
    else:
        prior = NigPrior(p, **prior_kwargs, dtype=dtype)

    worker = _MultivariateBcdmWorker(
        x,
        y,
        init_prob=init_prob,
        min_prob=min_prob,
        max_num_probs=max_num_probs,
        prior=prior,
        hazard=hazard,
        update_hook=update_hook,
    )
    return worker.fit()


@dataclass
class MultivariateBcdmResults:
    """Results of running the multivariate Bayesian change detection model over
    `n` datapoints with `p`-dimensional predictor (independent) variables."""

    mean: np.ndarray
    """`(n, p)` array of means. Is `np.nan` outside of the support."""

    cov: np.ndarray
    """`(n, p, p)` array of covariances. Is `np.nan` outside of the support."""

    shape: np.ndarray
    """`(n,)` array of shape parameters. Is `np.nan` outside of the support."""

    scale: np.ndarray
    """`(n,)` array of scale parameters. Is `np.nan` outside of the support."""

    log_joint: TrilArray
    """
    `(n, n)` lower triangular matrix of log joint probabilities of the
    run-length at each time point.

    A lower-triangular matrix where each row contains the natural logarithm of
    the posterior probability of the segment running length at that time point.
    E.g. the element in row `i` and column `j` is the log probability of the
    observation at time `i` belonging to a segment that began `j` observations
    before.

    The upper triangular elements are `-inf`, corresponding to a zero
    probability (i.e. a point cannot belong to a segment longer than the number
    of points observed so far).

    The log posterior probabilities are obtained by normalising the joint at
    each time point and are returned by `log_posterior()`.
    """

    log_predictive: TrilArray
    """
    `(n, n)` lower triangular matrix of log probabilities of the predictive
    distributions at each time point.

    Has a similar structure to `log_joint`, i.e., an lower-triangular matrix
    where the rows corresponds to the time points. In each row, the elements
    are the log density of the predictive distribution of the model given the
    elements in the segment up to that timepoint. E.g., the element in row `i`
    and column `j` is the log predictive probability of the `i`th observation
    given the model for the segment containing the `j` previous observations.
    """

    joint_support: TrilArray

    def log_posterior(self) -> np.ndarray:
        """Convert joint probabilities to a 2D lower triangular matrix of
        posterior probabilities. Obtained by normalising the joint probabilites
        across the rows. See docs for `log_joint` for structure."""
        log_joint = self.log_joint.full()
        norm = scipy.special.logsumexp(log_joint, axis=1)
        return log_joint - norm.reshape(-1, 1)

    def changepoints(self) -> list[int]:
        """Compute indices of changepoints in ascending order.

        Returns:
            List of (0-indexed) changepoint indices. Does not include first and
            last indices, which may be changepoints.
        """
        probs = self.log_joint
        run_length = probs[probs.n - 1].argmax()
        i = probs.n - 1 - run_length
        changepoints = []
        while i > 0:
            changepoints.append(i.item())
            i -= 1 + probs[i].argmax()

        changepoints.reverse()
        return changepoints


class _MultivariateBcdmWorker:
    __slots__ = [
        "init_log_joint",
        "joint_support",
        "log_1mhazard",
        "log_hazard",
        "log_joint",
        "log_pred",
        "max_num_probs",
        "min_log_prob",
        "n",
        "p",
        "params",
        "update_hook",
        "x",
        "y",
    ]

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        prior: "NigPrior",
        hazard: float,
        init_prob: float,
        min_prob: float,
        max_num_probs: int | None,
        update_hook: UpdateHook | None,
    ):
        # Assumes parameters have been validated - see multivariate_bcdm
        self.n, self.p = x.shape
        self.x = x
        self.y = y
        self.params = NigParams.from_prior(prior, self.n)
        self.log_hazard: float = np.log(hazard)
        self.log_1mhazard: float = np.log1p(-hazard)
        self.init_log_joint = np.log(init_prob) if init_prob else -np.inf
        self.min_log_prob: float | None = (
            np.log(min_prob) if min_prob else None
        )
        self.max_num_probs = max_num_probs or 0
        self.update_hook = update_hook

        # To reduce memory usage, store only the lower triangular part of these
        # matrices
        self.joint_support = tril_full(self.n, True, dtype=bool)
        self.log_joint = tril_full(self.n, -np.inf, upper_val=-np.inf)
        self.log_pred = tril_full(self.n, -np.inf, upper_val=-np.inf)

    def fit(self) -> MultivariateBcdmResults:
        X = self.x[:, np.newaxis, :]
        Y = self.y[:, np.newaxis]

        # Compute the initial reset probability (Eq. 28).
        log_pred = self.params[:1].mvt_logpdf(X[0], Y[0])[0]
        log_joint = log_pred + self.log_hazard + self.init_log_joint
        self.log_joint[0] = log_joint[0]
        self.log_pred[0] = log_pred[0]
        self.params[:1].update(X[0], Y[0])
        if self.update_hook:
            self.update_hook(0, X[0], Y[0], self)

        for t, (x, y) in enumerate(zip(X[1:], Y[1:], strict=True), start=1):
            self._update(t, x, y)
            if self.update_hook:
                self.update_hook(t, x, y, self)

        joint_support = self.joint_support
        nan_mask = ~joint_support[joint_support.n - 1]

        def _mask_to_nan(a: np.ndarray) -> np.ndarray:
            a[nan_mask] = np.nan
            return a

        return MultivariateBcdmResults(
            mean=_mask_to_nan(self.params.mean),
            cov=_mask_to_nan(self.params.cov),
            shape=_mask_to_nan(self.params.shape),
            scale=_mask_to_nan(self.params.scale),
            log_joint=self.log_joint,
            joint_support=joint_support,
            log_predictive=self.log_pred,
        )

    def _update(self, t: int, x: np.ndarray, y: np.ndarray) -> None:
        # Note that the variable `t` is 0-indexed here whereas in the notes it
        # starts from 1.

        # self.params is stored in the opposite order to `prev_log_joint`, so
        # it is reversed. Therefore params_view[0] corresponds to Theta_t in
        # Eq. 25, and params_view[-1] corresponds to Theta_1.
        params_view = self.params[: t + 1][::-1]

        prev_log_joint = self.log_joint[t - 1]

        # Copy the previous support to the current one, shifted right by one,
        # since `self.joint_support[t, 0]` must be True since it corresponds to
        # the 'hypothesis' that a new segment begins after this time step.
        new_mask = self.joint_support[t]
        mask = new_mask[1:]
        mask[:] = self.joint_support[t - 1]

        # Mask anything lower than the kth largest value
        max_probs = self.max_num_probs
        if max_probs and t >= max_probs:
            mask[masked_argmin(prev_log_joint, mask)] = False

        # Mask anything lower than min_log_prob
        if self.min_log_prob:
            # Normalise the joint to get the posterior
            log_normaliser = scipy.special.logsumexp(prev_log_joint[mask])
            mask[(prev_log_joint - log_normaliser) < self.min_log_prob] = False

        mask = new_mask

        # Same as above, log_pred[0] corresponds to Theta_t etc. Use the mask
        # to avoid expensive PDF computation outside of the support
        log_pred = self.log_pred[t]  # all -np.inf
        log_pred[mask] = params_view[mask].mvt_logpdf(x, y).ravel()

        # The (t + 1) changepoint probabilities
        log_joint = self.log_joint[t]

        # Reset probability.
        log_joint[0] = scipy.special.logsumexp(
            log_pred[0] + self.log_hazard + prev_log_joint[mask[1:]]
        )

        # Growth probabilities. Will be -inf where log_pred is -inf.
        # TODO(?): could also mask out over the -inf points, but probably
        # faster to just perform the computation and let the -inf come out in
        # the addition. Need to profile.
        log_joint[1:] = log_pred[1:] + self.log_1mhazard + prev_log_joint

        # Update the model parameters (Eqs. 30-33). Avoid expensive computation
        # outside of the support.
        params_view.update(x, y, mask=mask)


def _as_array(name: str, x, dtype) -> np.ndarray:
    if isinstance(x, np.ndarray):
        _check_dtype(name, x, dtype)
        return x
    return np.asarray(x).astype(dtype, casting="safe")


class NigPrior:
    """
    Parameters of p-dimensional normal-inverse-gamma distributions.
    """

    __slots__ = [
        "cov",
        "mean",
        "prec",
        "scale",
        "shape",
    ]

    mean: np.ndarray
    """
    (p) mean array.
    """

    cov: np.ndarray
    """
    (p, p) covariance array.
    """

    shape: float
    """
    Shape parameter.
    """

    scale: float
    """
    Scale parameters.
    """

    prec: np.ndarray
    """
    (p, p) precision.
    """

    @property
    def p(self) -> int:
        """Dimensionality of the distribution (i.e. the dimensionality of the
        independent variable)"""
        return self.mean.shape[0]

    @property
    def dtype(self) -> np.dtype:
        return self.mean.dtype

    def __init__(
        self,
        p: int,
        *,
        mean: PriorMeanArg = 0,
        cov: PriorCovArg = 1,
        shape: float = 1,
        scale: float = 1,
        dtype=None,
    ):
        """
        A `p`-dimensional NIG distribution.

        Args:
            p: Dimensionality of the NIG distributions.
            mean: Mean of the distributions. Must be a scalar or (p,)
                array-like.
            cov: Covariance of the distributions. Can be either:
                (1) A scalar, in which case the covariance is a diagonal
                    matrix with that value on the diagonal.
                (2) A (p,) array-like, in which case the covariance is a
                    diagonal matrix with the diagonal being the array.
                (3) A (p, p) array-like representing a positive-definite
                    matrix.
            dtype: numpy dtype of the parameters. If provided, ndarray
                arguments must match this dtype, otherwise will try to convert
                values to this type. If `mean` is an `np.ndarray`, defaults to
                its dtype, otherwise defaults to `np.float64`.
            shape: Shape parameter.
            scale: Scale parameter.
        """
        if isinstance(mean, np.ndarray):
            if dtype is None:
                dtype = mean.dtype
                if not np.issubdtype(dtype, np.floating):
                    msg = f"mean must have floating type, got {dtype}"
                    raise TypeError(msg)
            else:
                _check_dtype("mean", mean, dtype)
        elif dtype is None:
            dtype = np.float64

        if np.isscalar(mean):
            mean = np.full(p, mean, dtype=dtype)
        else:
            mean = _as_array("mean", mean, dtype)
            _check_array_shape("mean", mean, (p,))
        self.mean = mean

        if np.isscalar(cov):
            cov = np.eye(p, dtype=dtype) * cov
        else:
            cov = _as_array("cov", cov, dtype)
            if cov.ndim == 1:
                cov = np.diag(cov)
            _check_array_shape("cov", cov, (p, p))
        self.cov = cov

        self.shape = shape
        if self.shape <= 0:
            raise ValueError("shape must be > 0")

        self.scale = scale
        if self.scale <= 0:
            raise ValueError("scale must be > 0")

        self.prec = inv_positive_definite(cast(np.ndarray, cov))


@dataclass
class NigParams:
    """
    Parameters of t independent p-dimensional normal-inverse-gamma
    distributions.

    Use `from_priors` to initialise, rather than constructing directly.
    """

    __slots__ = [
        "cov",
        "mean",
        "prec",
        "scale",
        "shape",
    ]

    mean: FloatArray
    """
    (t, p) array of means.
    """

    cov: FloatArray
    """
    (t, p, p) array of covariances.
    """

    shape: FloatArray
    """
    (t,) array of shape parameters.
    """

    scale: FloatArray
    """
    (t,) array of scale parameters.
    """

    prec: FloatArray
    """
    (t, p, p) array of precisions.
    """

    @property
    def dtype(self) -> np.dtype:
        return self.mean.dtype

    @classmethod
    def from_prior(cls, priors: NigPrior, t: int) -> "NigParams":
        """
        Generate `t` independent NIG distributions from priors.

        Args:
            priors: Priors of the distributions.
            t: Number of distributions.
        """
        if t < 1:
            raise ValueError("t must be > 0")

        return NigParams(
            mean=np.expand_dims(priors.mean, 0).repeat(t, axis=0),
            cov=np.expand_dims(priors.cov, 0).repeat(t, axis=0),
            prec=np.expand_dims(priors.prec, 0).repeat(t, axis=0),
            shape=np.full(t, priors.shape, dtype=priors.dtype),
            scale=np.full(t, priors.scale, dtype=priors.dtype),
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
        _check_dtype("x", x, self.dtype)
        if x.ndim != 2:
            raise ValueError("x must be 2-D")
        xp = x.shape[1]
        p = self.mean.shape[1]
        if xp != p:
            raise ValueError(f"expected {p}-dimensional predictors, got {xp}")

    def update(self, x: np.ndarray, y: np.ndarray, *, mask=None) -> None:
        """
        Update sufficient statistics given n new observations (x, y).

        Args:
            x: (n, p) array of independent (predictor) variables.
            y: (n,) array of dependent (response) variables.
            mask: any object that can be used to index into the parameters
                (e.g. a boolean or index array); sufficient statistics will
                only be updated for the selected distributions.
        """
        if mask is None:
            mask = ...

        self._validate_x_arg(x)
        n = x.shape[0]
        _check_array_shape("y", y, (n,))
        _check_dtype("y", y, self.dtype)

        mean = self.mean[mask]
        t, p = mean.shape

        cov0 = self.cov[mask]  # (t, p, p)
        prec0 = self.prec[mask]  # (t, p, p)
        mean0 = np.expand_dims(mean, -1)  # (t, p, 1)

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

        self.mean[mask] = new_mean.squeeze(-1)
        self.cov[mask] = new_cov
        self.prec[mask] = new_prec
        self.shape[mask] += n / 2  # Eq. 32
        self.scale[mask] += new_scale
        assert np.all(self.scale[mask] >= 0), "got negative scale"

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

        Returns:
            (t, n) array
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
        _check_dtype("y", y, self.dtype)
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

        Returns:
            Array of expected values; has shape (t, n) if `axis == 'n'` or
            shape (t,) if `axis == 't'`.
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

        Returns:
            Array of variances; has shape (t, n) if `axis == 'n'` or shape
            (t,) if `axis == 't'`.
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

        Returns:
            NigParams object with n distributions, representing the model
            parameters after observing each (x, y).
        """
        self._validate_x_arg(x)
        n, p = x.shape
        _check_array_shape("y", y, (n,))
        _check_dtype("y", y, self.dtype)

        if self.mean.shape[0] != 1:
            msg = "can only fit regression model from a single prior"
            raise ValueError(msg)

        prior = NigPrior(
            p=p,
            cov=self.cov[0],
            mean=self.mean[0],
            shape=self.shape[0],
            scale=self.scale[0],
        )
        params = NigParams.from_prior(prior, n)
        for t, (xt, yt) in enumerate(zip(x, y, strict=True)):
            assert xt.ndim == 1, xt.shape
            assert np.isscalar(yt), yt
            params[t:].update(xt.reshape(1, -1), np.atleast_1d(yt))

        return params
