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

import numpy as np
from numpy import linalg, random
from scipy import special


class MatrixVariateNormalInvGamma:
    """Matrix-variate normal, matrix-variate inverse gamma distribution

    The matrix-variate normal, inverse-gamma distribution is the conjugate
    prior for a matrix-variate normal distribution. As a result the
    distribution can be used in Bayesian estimation of the location and scale
    parameters of the matrix-variate normal distribution.

    """

    def __init__(self, mu, omega, sigma, eta):
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
            np.diag(np.sqrt(2.0 * random.gamma((eta - np.arange(n)) / 2.0)))
            + np.tril(random.randn(n, n), -1),
            np.sqrt(eta) * linalg.cholesky(sigma).transpose(),
        )
        b = np.dot(f.transpose(), f)

        # Simulate the conditional Gauss distribution.
        a = mu + linalg.solve(
            linalg.cholesky(omega).transpose(),
            np.dot(random.randn(m, n), linalg.cholesky(b).transpose()),
        )

        return a, b


class Bcdm:
    """Bayesian change detection model.

    Args:
        mu (numpy.array): (M x N) location parameters of the prior
                          distribution.
        omega (numpy.array): (M x M) scale parameters of the prior
                             distribution.
        sigma (numpy.array): (N x N) dispersion parameters of the prior
                             distribution.
        eta (float): shape parameter of the prior distribution.
        alg (string): Specifies the algorithm to use. Choose either 'sumprod'
                      for the sum-product algorithm or 'maxprod' for the
                      max-product algorithm. If the sum-product algorithm is
                      selected, the posterior probabilities of the segmentation
                      hypotheses will be calculated. If the max-product
                      algorithm is selected, the most likely sequence
                      segmentation will be calculated.
        hazardfunc (float): Relative chance of a new segments being generated.
                            ``hazardfunc`` is a value between 0 and 1. Segments
                            are MORE likely to be created with values closer to
                            zero. Segments are LESS likely to form with values
                            closer to 1. Alternatively, hazardfunc can be set
                            to an executable hazard function. The hazard
                            function must accept non-negative integers and
                            return non-negative floating-point numbers.
        basisfunc (callable): Feature functions for basis function
                              expansion. Feature functions provide additional
                              flexibility by mapping the predictor variables to
                              an intermmediate feature space, thus allowing the
                              user to model non-linear relationships.
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

    def __init__(
        self,
        mu=None,
        omega=None,
        sigma=None,
        eta=None,
        alg="sumprod",
        hazardfunc=0.1,
        basisfunc=None,
        minprob=1.0e-6,
        maxhypot=20,
    ):
        alg = alg.lower()
        algs = ("sumprod", "maxprod")
        if alg not in algs:
            raise ValueError(f"The input 'alg' must be in {algs}.")

        self.alg = alg
        self.numpred = None
        self.numresp = None
        self.mu = mu  # location parameter
        self.omega = omega  # scale parameter
        self.sigma = sigma  # dispersion/noise parameter
        self.eta = eta  # shape parameter

        # Ensure algorithm initialises on first call to update.
        self.initialised = False

        # If 'maxhypot' is set to none, no hypotheses will be trimmed.
        if maxhypot is None or maxhypot > 0:
            self.maximum_hypotheses = maxhypot
        else:
            raise ValueError(
                "The input 'maxhypot' must be an integer greater than zero."
            )

        if minprob > 0:
            self.min_probability = minprob
        else:
            raise ValueError(
                "The input 'minprob' must be a float greater than zero."
            )

        # Variables for tracking segments.
        self.hypotheses = []
        self.counts = []
        self.probabilities = []

        # Store basis and hazard function.
        self.basisfunc = basisfunc if callable(basisfunc) else lambda x: x
        self.hazardfunc = (
            hazardfunc if callable(hazardfunc) else lambda _: hazardfunc
        )

    def _init_algorithm(self, numpred, numresp):
        """Initialise the Bcdm algorithm."""

        # Ensure input dimensions are consistent.
        if self.numpred is None:
            self.numpred = numpred
        elif self.numpred != numpred:
            raise ValueError(
                f"Expected {numpred} dimensions in the predictor variable."
            )

        # Ensure output dimensions are consistent.
        if self.numresp is None:
            self.numresp = numresp
        elif self.numresp != numresp:
            raise ValueError(
                f"Expected {numresp} dimensions in the response variable."
            )

        # Set uninformative prior for the location parameter.
        if self.mu is None:
            self.mu = np.zeros((numpred, numresp))

        # Set uninformative prior for the scale parameter.
        if self.omega is None:
            self.omega = np.eye(numpred)

        # Set uninformative prior for the dispersion/noise parameter.
        if self.sigma is None:
            self.sigma = np.eye(numresp)

        # Set uninformative prior for the shape parameter.
        if self.eta is None:
            self.eta = numresp

        # Create the initial hypothesis, which states that the first segment is
        # about to begin.
        self._add_new_hypothesis(0.0)

    def _softmax(self, x, y):
        return max(x, y) + np.log1p(np.exp(-abs(x - y)))

    def _add_new_hypothesis(self, log_likelihood, basisfunc=None):
        """Function for spawning new hypothesis"""

        # Set basis function.
        if basisfunc is None:
            basisfunc = self.basisfunc

        # Create new Bayesian linear model (using supplied priors).
        stat = MatrixVariateNormalInvGamma(
            self.mu, self.omega, self.sigma, self.eta
        )

        # Add a new hypothesis, which states that a new segment is about to
        # begin.
        self.hypotheses.append(
            {
                "count": 0,
                "log_probability": log_likelihood,
                "distribution": stat,
                "log_constant": stat.log_constant(),
                "basisfunc": basisfunc,
            }
        )

    def update(self, X, Y, basisfunc=None):
        """Update model with a single observation.

        When new input-output data is available, the model can be updated using
        this method. As more and more data are collected, the number of
        hypotheses grows, increasing the computational complexity. By default
        hypotheses are pruned at the end of each update (see
        :py:meth:`.trim`.). To disable hypotheses trimming, initialise the
        class with ``maxhypot`` set to ``None``.

        Args:
            X (numpy.array): Observed (1 x M) input data (predictor variable).
            Y (numpy.array): Observed (1 x N) output data (response variable).

        """

        # Initialise algorithm on first call to update. This allows the
        # algorithm to configure itself to the size of the first input/output
        # data if no hyper-parameters have been specified.
        if not self.initialised:
            init_basis = self.basisfunc if basisfunc is None else basisfunc
            x = init_basis(X)
            m = x.shape[1] if np.ndim(x) > 1 else X.size
            n = Y.shape[1] if np.ndim(Y) > 1 else Y.size
            self._init_algorithm(m, n)
            self.initialised = True

        # Get size of data.
        k = X.shape[0] if np.ndim(X) > 1 else X.size
        m, n = self.numpred, self.numresp

        # Allocate variables for the dynamic programming pass.
        loglik = -np.inf
        logmax = -np.inf
        logsum = -np.inf
        ind = np.nan

        # Update hypotheses by updating each matrix variate, normal inverse
        # gamma distribution over the linear models.
        for hypothesis in self.hypotheses:
            # Update the sufficient statistics.
            hypothesis["distribution"].update(hypothesis["basisfunc"](X), Y)

            # Compute the log-normalization constant after the update
            # (posterior parameter distribution).
            # (Equation 8)
            n_o = hypothesis["log_constant"]
            n_k = hypothesis["log_constant"] = hypothesis[
                "distribution"
            ].log_constant()

            # Evaluate the log-density of the predictive distribution.
            # (Equation 16)
            log_density = n_k - n_o - k * (0.5 * m * n) * np.log(2.0 * np.pi)

            # Increment the counter.
            hypothesis["count"] += 1

            # Accumulate the log-likelihood of the data.
            # (Equation 17)
            hazard = self.hazardfunc(hypothesis["count"])
            aux = np.log(hazard) + log_density + hypothesis["log_probability"]
            loglik = self._softmax(loglik, aux)

            # Keep track of the highest, log-likelihood.
            if aux > logmax:
                logmax, ind = aux, hypothesis["count"]

            # Update and accumulate the log-probabilities.
            hypothesis["log_probability"] += np.log1p(-hazard) + log_density
            logsum = self._softmax(logsum, hypothesis["log_probability"])

        # In the max-product algorithm, keep track of the most likely
        # hypotheses.
        if self.alg == "maxprod":
            loglik = logmax
            self.counts.append(ind)

        # Add a new hypothesis, which states that the next segment is about to
        # begin.
        self._add_new_hypothesis(loglik, basisfunc)

        # Normalize the hypotheses so that their probabilities sum to one.
        logsum = self._softmax(logsum, loglik)
        for hypothesis in self.hypotheses:
            hypothesis["log_probability"] -= logsum

        # Automatically trim hypothesis on each update if requested.
        if self.maximum_hypotheses is not None:
            self.trim_hypotheses(
                minprob=self.min_probability,
                maxhypot=self.maximum_hypotheses,
            )

        # In the sum-product algorithm, keep track of the probabilities.
        if self.alg == "sumprod":
            iteration = list()
            for hypothesis in self.hypotheses:
                iteration.append(
                    (hypothesis["count"], hypothesis["log_probability"])
                )

            self.probabilities.append(iteration)

    def trim_hypotheses(self, minprob=1.0e-6, maxhypot=20):
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
        self.hypotheses.sort(key=lambda dct: -dct["log_probability"])

        # Store the indices of likely hypotheses.
        minprob = np.log(minprob)
        index = [
            i
            for i, hypot in enumerate(self.hypotheses)
            if hypot["log_probability"] > minprob
        ]

        # Trim the hypotheses.
        index = index[:maxhypot] if len(index) >= maxhypot else index
        self.hypotheses = [self.hypotheses[i] for i in index]

        # NOTE: This final ordering can preserve the original order of the
        #       hypotheses. Interestingly, the algorithm specified in update
        #       does not require that the hypotheses be ordered! This sort can
        #       safely be ignored.
        # self.__hypotheses.sort(key=lambda dct: dct['index'])

        # Normalize the hypotheses so that their probabilities sum to one.
        logsum = -np.inf
        for hypot in self.hypotheses:
            logsum = self._softmax(logsum, hypot["log_probability"])
        for hypot in self.hypotheses:
            hypot["log_probability"] -= logsum

    def infer(self):
        """Return posterior probabilities OR sequence segmentation.

        If the MAX-PRODUCT algorithm is selected, this method returns the most
        likely sequence segmentation as a list of integers. Each integer in the
        list marks where a segment begins.

        If the SUM-PRODUCT algorithm is selected, this method returns the
        posterior probabilities of the segmentation hypotheses as a numpy
        array. Rows in the array represent hypotheses and columns in the array
        represent data points in the time-series.

        Returns:
            object: This method returns the inference results. In the case of
                    the MAX-PRODUCT algorithm, the method returns the most
                    likely segmentation. In the case of the SUM-PRODUCT
                    algorithm, this method returns the posterior probabilities
                    of the segmentation hypotheses.

        """

        # In the max-product algorithm, the most likely hypotheses are
        # tracked. Recover the most likely segment boundaries by performing a
        # back-trace.
        if self.alg == "maxprod":
            # Find the most likely hypothesis.
            max_hypothesis = max(
                self.hypotheses, key=lambda dct: dct["log_probability"]
            )

            # Find the best sequence segmentation given all the data so far.
            segment_boundaries = [
                len(self.counts) - 1,
            ]
            index = segment_boundaries[0] - 1
            count = max_hypothesis["count"] - 1
            while index > 0:
                index -= count
                segment_boundaries.insert(0, index)
                count = self.counts[index - 1]

            return segment_boundaries

        # In the sum-product algorithm, the segment probabilities are
        # tracked. Recover the segment probabilities by formatting the stored
        # history.
        else:
            k = len(self.probabilities)
            segment_probabilities = np.zeros((k + 1, k + 1))
            segment_probabilities[0, 0] = 1.0

            # Update hypotheses probabilities.
            for i in range(len(self.probabilities)):
                for j, probability in self.probabilities[i]:
                    segment_probabilities[j, i + 1] = np.exp(probability)

            # A segment always occurs at the beginning of the dataset.
            segment_probabilities[0, 0] = 1.0

            return segment_probabilities
