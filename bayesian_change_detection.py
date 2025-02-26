"""Bayesian model-based change detection for input-output sequence data

The Bayesian change-point detection model (BCDM) class implements a recursive
algorithm for partitioning a sequence of real-valued input-output data into
non-overlapping segments. The segment boundaries are chosen under the
assumption that, within each segment, the data follow a multi-variate linear
model.

Segmentation is carried out in an online fashion by recursively updating a set
of hypotheses. The hypotheses capture the belief about the current segment,
e.g. its duration and the linear relationship between inputs and outputs, given
all the data so far. Each time a new pair of data is received, the hypotheses
are propagated and re-weighted to reflect this new knowledge.

.. codeauthor:: Gabriel Agamennoni <gabriel.agamennoni@mavt.ethz.ch>
.. codeauthor:: Asher Bender <a.bender@acfr.usyd.edu.au>

"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import overload

import numpy as np
from numpy import linalg
from scipy import special
from scipy.stats import t as StudentsT


class MatrixVariateNormalInvGamma:
    """Matrix-variate normal, matrix-variate inverse gamma distribution

    The matrix-variate normal, inverse-gamma distribution is the conjugate
    prior for a matrix-variate normal distribution. As a result the
    distribution can be used in Bayesian estimation of the location and scale
    parameters of the matrix-variate normal distribution.

    """

    def __init__(
        self,
        mu,
        omega,
        sigma,
        eta,
        *,
        rng: np.random.Generator | None = None,
    ):
        self.rng = rng or np.random.default_rng()

        # Get size of data.
        m, n = np.shape(mu)
        self.m, self.n = m, n

        # Check that the location parameter is a matrix of finite numbers.
        if not (
            np.ndim(mu) == 2
            and np.shape(mu) == (m, n)
            and not np.isnan(mu).any()
            and np.isfinite(mu).all()
        ):
            raise ValueError(
                "The location parameter must be a matrix of finite numbers."
            )

        # Check that the scale parameter is a symmetric, positive-definite
        # matrix.
        if not (
            np.ndim(omega) == 2
            and np.shape(omega) == (m, m)
            and not np.isnan(omega).any()
            and np.isfinite(omega).all()
            and np.allclose(np.transpose(omega), omega)
            and linalg.det(omega) > 0.0
        ):
            raise ValueError(
                "The scale parameter must be a symmetric, "
                "positive-definite matrix"
            )

        # Check that the dispersion parameter is a symmetric, positive-definite
        # matrix.
        if not (
            np.ndim(sigma) == 2
            and np.shape(sigma) == (n, n)
            and not np.isnan(sigma).any()
            and np.isfinite(sigma).all()
            and np.allclose(np.transpose(sigma), sigma)
            and linalg.det(sigma) > 0.0
        ):
            raise ValueError(
                "The noise parameter must be a symmetric, positive-definite "
                "matrix."
            )

        # Check that the shape parameter is a number greater than one minus the
        # number of degrees of freedom.
        if not (
            np.isscalar(eta)
            and not np.isnan(eta)
            and np.isfinite(eta)
            and eta > n - 1.0
        ):
            raise ValueError(
                "The shape parameter must be greater than one minus the "
                "degrees of freedom."
            )

        # Allocate space for storing the matrix of product statistics.
        self.prod = np.zeros([m + n, m + n])

        # Initialize the statistics with the parameters of the prior
        # distribution.
        x = np.dot(omega, mu)
        self.prod = np.block(
            [[omega, x], [x.T, np.dot(mu.T, x) + eta * sigma]]
        )
        self.weight = eta

    def update(self, X, Y):
        """Update the sufficient statistics given observed data.

        The sufficient statistics are the only parameters required to describe
        the shape of the distribution. Initially, the sufficient statistics
        contain no information apart from that implied by the prior
        distribution. As data arrive, the statistics are updated incrementally
        in order to reflect this new knowledge. Performing updates allows the
        sufficient statistics to summarise all information contained in the
        data observed so far.

        """

        # (Equation 5a, b)
        #
        #     | XX    XY |
        #     | YX    YY |
        #
        if np.ndim(X) > 1:
            k, m = np.shape(X)
            x = np.dot(X.T, Y)

            # Update the statistics given a block of data (in the following
            # order: XX, XY, YX, YY)
            self.prod[:m, :m] += np.dot(X.T, X)
            self.prod[:m, m:] += x
            self.prod[m:, :m] += x.T
            self.prod[m:, m:] += np.dot(Y.T, Y)
            self.weight += k

        else:
            m = np.size(X)
            x = np.outer(X, Y)

            # Update the statistics given a single datum.
            self.prod[:m, :m] += np.outer(X, X)
            self.prod[:m, m:] += x
            self.prod[m:, :m] += x.T
            self.prod[m:, m:] += np.outer(Y, Y)
            self.weight += 1

    def log_constant(self):
        m, n = self.m, self.n

        # Note usage of the log-determinant 'trick':
        #
        #     log(det(A)) = 2*sum(log(diag(chol(A))))
        #
        d = np.diag(linalg.cholesky(self.prod))
        w = self.weight

        # Evaluate the log-normalization constant.
        # (Equation 8)
        return (
            special.gammaln(0.5 * (w - np.arange(n))).sum()
            - n * np.log(d[:m]).sum()
            - w * np.log(d[m:] / np.sqrt(w)).sum()
            - n * (0.5 * w) * np.log(0.5 * w)
        )

    def parameters(self):
        """Return the posterior parameters.

        All the information content of the data is summarized by the sufficient
        statistics. As a result the posterior parameters are a function of the
        sufficient statistics. This is a consequence of the conjugacy of the
        matrix-variate Gaussian-inverse-Gamma distribution.

        """

        m = self.m
        s = linalg.cholesky(self.prod).transpose()
        w = self.weight

        # Compute the parameters of the posterior distribution.
        return (
            linalg.solve(s[:m, :m], s[:m, m:]),
            np.dot(s[:m, :m].transpose(), s[:m, :m]),
            np.dot(s[m:, m:].transpose(), s[m:, m:]) / w,
            w,
        )

    def rand(self):
        m, n = self.m, self.n

        s = linalg.cholesky(self.prod).transpose()
        w = self.weight

        # Compute the parameters of the posterior distribution.
        mu = linalg.solve(s[:m, :m], s[:m, m:])
        omega = np.dot(s[:m, :m].transpose(), s[:m, :m])
        sigma = np.dot(s[m:, m:].transpose(), s[m:, m:]) / w
        eta = w

        # Simulate the marginal Wishart distribution.
        f = linalg.solve(
            np.diag(np.sqrt(2.0 * self.rng.gamma((eta - np.arange(n)) / 2.0)))
            + np.tril(self.rng.standard_normal((n, n)), -1),
            np.sqrt(eta) * linalg.cholesky(sigma).transpose(),
        )
        b = np.dot(f.transpose(), f)

        # Simulate the conditional Gauss distribution.
        a = mu + linalg.solve(
            linalg.cholesky(omega).transpose(),
            np.dot(
                self.rng.standard_normal((m, n)),
                linalg.cholesky(b).transpose(),
            ),
        )

        return a, b


def _log_sum_exp(x: float, y: float) -> float:
    """
    Numerically stable form of:

      log(exp(x) + exp(y))
    """
    return max(x, y) + np.log1p(np.exp(-abs(x - y)))


def _identity_basis(x: np.ndarray) -> np.ndarray:
    return x


BasisFunction = Callable[[np.ndarray], np.ndarray]
HazardFunction = Callable[[int], float]


@dataclass
class _Hypothesis:
    count: int
    basis: BasisFunction
    distribution: MatrixVariateNormalInvGamma
    log_constant: float
    maxprod: float
    sumprod: float


@dataclass
class RegressionParameters:
    mu: np.ndarray
    omega: np.ndarray
    sigma: np.ndarray
    eta: float

    def var(self, X: np.ndarray) -> np.ndarray:
        return (
            self.sigma
            / (self.eta - 1)
            * (1 + (np.dot(X, self.omega) * X).sum(axis=1))
        )

    def confidence_interval(
        self, X: np.ndarray, sig: float = 0.05
    ) -> np.ndarray:
        t = StudentsT(2 * self.eta).ppf(1 - sig / 2)
        return t * np.sqrt(self.var(X))


class Bcdm:
    """Bayesian change detection model.

    Args:
        numpred (int): Number of predictor variables, M
                       (dimensionality of input).
        numresp (int): Number of response variables, N
                       (dimensionality of output).
        mu (numpy.array): (M x N) location parameters of the prior
                          distribution.
        omega (numpy.array): (M x M) scale parameters of the prior
                             distribution.
        sigma (numpy.array): (N x N) dispersion parameters of the prior
                             distribution.
        eta (float): shape parameter of the prior distribution.
        hazardfunc (float): Relative chance of a new segments being generated.
                            ``hazardfunc`` is a value between 0 and 1. Segments
                            are MORE likely to be created with values closer to
                            zero. Segments are LESS likely to form with values
                            closer to 1. Alternatively, hazardfunc can be set
                            to an executable hazard function. The hazard
                            function must accept non-negative integers and
                            return non-negative floating-point numbers.
        minprob (float): Minimum probability required for a
                         hypothesis. Hypotheses with insignificant support
                         (probabilities below this value) will be pruned.
        maxhypot (int): Maximum number of segmentation hypotheses to
                        consider. After each update, pruning will take place to
                        limit the number of hypotheses. If set to ``None``, no
                        pruning will NOT take place after updates, however,
                        pruning can be initiated manually by calling
                        :py:meth:`.trim`.

    Raises:
        ValueError: If the any of the inputs are invalid.

    """

    def _validate_shape(self, name: str, exp: Sequence[int]):
        shape = getattr(self, name).shape
        if shape != exp:
            raise ValueError(
                f"invalid shape for {name}: expected {exp}, got {shape}"
            )

    def __init__(
        self,
        numpred: int,
        numresp: int,
        *,
        mu: np.ndarray | float | None = None,
        omega: np.ndarray | float | None = None,
        sigma: np.ndarray | float | None = None,
        eta: float | None = None,
        hazardfunc: HazardFunction | float = 0.1,
        minprob: float = 1.0e-6,
        maxhypot: int | None = 20,
        rng: np.random.Generator | None = None,
    ):
        self.rng = rng

        if numpred <= 0:
            raise ValueError("numpred must be > 0")
        self.numpred = numpred

        if numresp <= 0:
            raise ValueError("numresp must be > 0")
        self.numresp = numresp

        # Set uninformative prior for the location parameter.
        self.mu: np.ndarray
        if mu is None:
            self.mu = np.zeros((numpred, numresp))
        else:
            self.mu = np.atleast_2d(mu)
            self._validate_shape("mu", (numpred, numresp))

        # Set uninformative prior for the scale parameter.
        self.omega: np.ndarray
        if omega is None:
            self.omega = np.eye(numpred)
        else:
            self.omega = np.atleast_2d(omega)
            self._validate_shape("omega", (numpred, numpred))

        # Set uninformative prior for the dispersion/noise parameter.
        self.sigma: np.ndarray
        if sigma is None:
            self.sigma = np.eye(numresp)
        else:
            self.sigma = np.atleast_2d(sigma)
            self._validate_shape("sigma", (numresp, numresp))

        # Set uninformative prior for the shape parameter.
        self.eta: float
        if eta is None:
            self.eta = numresp
        else:
            if eta < 1:
                raise ValueError("eta must be >= 1")
            self.eta = eta

        # If 'maxhypot' is set to none, no hypotheses will be trimmed.
        if maxhypot is None or maxhypot > 0:
            self.maximum_hypotheses = maxhypot
        else:
            raise ValueError("maxhypot must be > 0")

        if minprob > 0:
            self.min_probability = minprob
        else:
            raise ValueError(
                "The input 'minprob' must be a float greater than zero."
            )

        # Variables for tracking segments.
        self.hypotheses: list[_Hypothesis] = []
        self.log_likelihoods: list[list[tuple[int, float]]] = []
        self.maxprod_maxinds: list[int] = []

        self.hazardfunc: HazardFunction = (
            hazardfunc if callable(hazardfunc) else lambda _: hazardfunc
        )

    def _linear_model(self) -> MatrixVariateNormalInvGamma:
        """Create new Bayesian linear model (using supplied priors)."""
        return MatrixVariateNormalInvGamma(
            mu=self.mu,
            omega=self.omega,
            sigma=self.sigma,
            eta=self.eta,
            rng=self.rng,
        )

    def _add_new_hypothesis(
        self,
        sumprod: float,
        maxprod: float,
        basis: BasisFunction,
    ) -> None:
        # Add a new hypothesis, which states that a new segment is about to
        # begin.
        distribution = self._linear_model()
        new = _Hypothesis(
            count=0,
            basis=basis,
            distribution=distribution,
            log_constant=distribution.log_constant(),
            sumprod=sumprod,
            maxprod=maxprod,
        )
        self.hypotheses.append(new)

    @staticmethod
    def _num_samples(x: np.ndarray) -> int:
        if x.ndim == 1:
            return 1
        if x.ndim == 2:
            return x.shape[0]
        raise ValueError("input must be 1- or 2-D array")

    @dataclass
    class _LogProbabilities:
        prob: float = -np.inf
        sum: float = -np.inf

    @dataclass
    class _MaxLogProbabilities(_LogProbabilities):
        max: float = -np.inf
        argmax: int = -1

    @overload
    def regression_parameters(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        segmentations: Sequence[int],
        basis: BasisFunction | None = ...,
    ) -> list[RegressionParameters]: ...

    @overload
    def regression_parameters(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        basis: BasisFunction | None = ...,
        segmentations: None = ...,
    ) -> RegressionParameters: ...

    def regression_parameters(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        basis: BasisFunction | None = None,
        segmentations: Sequence[int] | None = None,
    ) -> RegressionParameters | list[RegressionParameters]:
        if self.numresp != 1:
            raise NotImplementedError(
                "Currently only supports single response variable"
            )
        if Y.shape[-1] != self.numresp:
            raise ValueError(
                f"Y must have {self.numresp} response variables(s), "
                f"got {Y.shape[-1]}"
            )

        if basis:
            X = basis(X)

        if X.shape[-1] != self.numpred:
            raise ValueError(
                f"X must have {self.numpred} predictor(s), got {X.shape[-1]}"
            )

        indices = segmentations or (0, len(X) - 1)
        params = []
        for i0, i1 in pairwise(indices):
            x = X[i0 : i1 + 1]
            y = Y[i0 : i1 + 1]

            omega_inv = self.omega + (x.T @ x)
            omega = np.linalg.inv(omega_inv)
            mu = omega @ ((self.omega @ self.mu) + (x.T @ y))
            sigma = (
                self.sigma
                + (
                    (self.mu.T @ self.omega @ self.mu)
                    + (y.T @ y)
                    - (mu.T @ omega_inv @ mu)
                )
                / 2
            )
            eta = self.eta + len(x) / 2

            # if (sigma < 0).any():
            #     breakpoint()

            params.append(
                RegressionParameters(
                    mu=mu,
                    omega=omega,
                    sigma=sigma,
                    eta=eta,
                )
            )

        if segmentations:
            return params

        assert len(params) == 1
        return params[0]

    def update(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        basis: BasisFunction | None = None,
    ) -> None:
        """Update model with a single observation.

        When new input-output data is available, the model can be updated using
        this method. As more and more data are collected, the number of
        hypotheses grows, increasing the computational complexity. By default
        hypotheses are pruned at the end of each update (see
        :py:meth:`.trim`.). To disable hypotheses trimming, initialise the
        class with ``maxhypot`` set to ``None``.

        Args:
            X (numpy.array): Observed (k x M) or (M) input data
                             (predictor variable).
            Y (numpy.array): Observed (k x N) or (N) output data
                             (response variable).
            basis (callable | None): Basis function. If None, defaults to the
                                     identity function.
        """

        basis = basis or _identity_basis
        if not self.hypotheses:
            self._add_new_hypothesis(0, 0, basis)

        k = self._num_samples(X)
        k_y = self._num_samples(Y)
        if k != k_y:
            raise ValueError("X and Y must have the same number of samples")

        Y = np.atleast_2d(Y)

        # Update hypotheses by updating each matrix variate, normal inverse
        # gamma distribution over the linear models.
        sumprod = self._LogProbabilities()
        maxprod = self._MaxLogProbabilities()
        for hypothesis in self.hypotheses:
            # Update the sufficient statistics.
            hypothesis.distribution.update(hypothesis.basis(X), Y)

            # Compute the log-normalization constant after the update
            # (posterior parameter distribution) (Equation 8)
            n_o = hypothesis.log_constant
            hypothesis.log_constant = hypothesis.distribution.log_constant()
            n_k = hypothesis.log_constant

            # Evaluate the log-density of the predictive distribution.
            # (Equation 16)
            log_density = (
                n_k
                - n_o
                - k * (self.numpred * self.numresp / 2) * np.log(2 * np.pi)
            )

            # Accumulate the log-likelihood of the data (Equation 17)
            hypothesis.count += 1
            hazard = self.hazardfunc(hypothesis.count)
            log_hazard = np.log(hazard)

            # Keep track of the posterior probability.
            sumprod.prob = _log_sum_exp(
                sumprod.prob, log_hazard + log_density + hypothesis.sumprod
            )

            # Keep track of the highest log-likelihood.
            maxprod_aux = log_hazard + log_density + hypothesis.maxprod
            maxprod.prob = _log_sum_exp(maxprod.prob, maxprod_aux)
            if maxprod_aux > maxprod.max:
                maxprod.max = maxprod_aux
                maxprod.argmax = hypothesis.count

            # Update and accumulate the log-probabilities.
            hypoth_log_prob = np.log1p(-hazard) + log_density
            hypothesis.maxprod += hypoth_log_prob
            maxprod.sum = _log_sum_exp(maxprod.sum, hypothesis.maxprod)
            hypothesis.sumprod += hypoth_log_prob
            sumprod.sum = _log_sum_exp(sumprod.sum, hypothesis.sumprod)

        # Keep track of the most likely hypotheses.
        maxprod.prob = maxprod.max
        self.maxprod_maxinds.append(maxprod.argmax)

        # Add a new hypothesis, which states that the next segment is about to
        # begin.
        self._add_new_hypothesis(sumprod.prob, maxprod.prob, basis)

        # Normalize the hypotheses so that their probabilities sum to one.
        maxprod.sum = _log_sum_exp(maxprod.sum, maxprod.prob)
        sumprod.sum = _log_sum_exp(sumprod.sum, sumprod.prob)
        for hypothesis in self.hypotheses:
            hypothesis.maxprod -= maxprod.sum
            hypothesis.sumprod -= sumprod.sum

        # Automatically trim hypothesis on each update if requested.
        if self.maximum_hypotheses is not None:
            self.trim_hypotheses(
                minprob=self.min_probability,
                maxhypot=self.maximum_hypotheses,
            )

        # Keep track of the probabilities.
        self.log_likelihoods.append(
            [(h.count, h.sumprod) for h in self.hypotheses]
        )

    def trim_hypotheses(self, minprob=1.0e-6, maxhypot=20) -> None:
        """Prune hypotheses to limit computational complexity.

        The computational complexity of the algorithm can be managed by
        limiting the number of hypotheses maintained. This method limits the
        number of hypotheses maintained by the model by:

            1) Removing any hypotheses with a support (probability) less than
               ``minprob``.

            2) Preserving the first ``maxhypot`` likely hypotheses and
               discarding the rest.

        """

        # Skip pruning if less hypotheses exist than the maximum allowed.
        if len(self.hypotheses) <= maxhypot:
            return

        # Sort the hypotheses in decreasing log probability order.
        self.hypotheses.sort(key=lambda h: -h.sumprod)

        # Store the indices of likely hypotheses.
        minprob = np.log(minprob)
        index = [
            i
            for i, hypot in enumerate(self.hypotheses)
            if hypot.sumprod > minprob
        ]

        # Trim the hypotheses.
        index = index[:maxhypot] if len(index) >= maxhypot else index
        self.hypotheses = [self.hypotheses[i] for i in index]

        # NOTE: This final ordering can preserve the original order of the
        #       hypotheses. Interestingly, the algorithm specified in update
        #       does not require that the hypotheses be ordered! This sort can
        #       safely be ignored.
        # self.hypotheses.sort(key=lambda h: h.count)

        # Normalize the hypotheses so that their probabilities sum to one.
        logsum = -np.inf
        for hypot in self.hypotheses:
            logsum = _log_sum_exp(logsum, hypot.sumprod)
        for hypot in self.hypotheses:
            hypot.sumprod -= logsum

    def posterior_probabilities(self) -> np.ndarray:
        """
        Calculate posterior probabilities of the segmentation hypotheses.
        """
        k = len(self.log_likelihoods)
        segment_probabilities = np.zeros((k + 1, k + 1))

        # Update hypotheses probabilities.
        for i in range(k):
            for j, p in self.log_likelihoods[i]:
                segment_probabilities[j, i + 1] = np.exp(p)

        # A segment always occurs at the beginning of the dataset.
        segment_probabilities[0, 0] = 1.0
        return segment_probabilities

    def segmentations(self) -> list[int]:
        """
        Calculate the most likely sequence segmentation as a list of
        integers. Each integer in the list marks where a segment begins.
        """
        # The most likely hypotheses are tracked. Recover the most likely
        # segment boundaries by performing a back-trace.

        if any(ind < 0 for ind in self.maxprod_maxinds):
            raise AssertionError("maxprod_maxinds contains negative indices")

        # Find the best sequence segmentation given all the data so far.
        max_hypothesis = max(self.hypotheses, key=lambda h: h.maxprod)
        segment_boundaries = [len(self.maxprod_maxinds) - 1]
        index = segment_boundaries[0] - 1
        count = max_hypothesis.count - 1
        while index > 0:
            index -= count
            segment_boundaries.insert(0, index)
            count = self.maxprod_maxinds[index - 1]

        return segment_boundaries
