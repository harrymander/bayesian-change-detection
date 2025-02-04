#!/usr/bin/python3

"""Examples of the Bayesian model-based change detection model in action

This script runs a set of examples that demonstrate the Bayesian change
detection model on different datasets. Some of these datasets are synthetic,
while others are real.

.. codeauthor:: Gabriel Agamennoni <gabriel.agamennoni@mavt.ethz.ch>
.. codeauthor:: Asher Bender <a.bender@acfr.usyd.edu.au>

"""

import logging
from itertools import batched, pairwise
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from change_detec import Bcdm, MatrixVariateNormalInvGamma

# Use same random data for repeatability.
np.random.seed(seed=1729)

DATA_DIR = Path(__file__).parent / "data"


def gen_random_data(
    numpred,
    numresp,
    numpoint,
    numseg,
    mu=None,
    omega=None,
    sigma=None,
    eta=None,
    featfun=None,
):
    """Randomly generate multi-variate input-output data

    First, generate a sequence of segments at random. Then, for each segment,
    generate a random coefficient matrix and a random noise covariance matrix,
    and use these to generate input-output data.
    """

    # Set default prior for the location parameter.
    if mu is None:
        mu = np.zeros((numpred, numresp))

    # Set default prior for the scale parameter.
    if omega is None:
        omega = np.eye(numpred)

    # Set default prior for the dispersion/noise parameter.
    if sigma is None:
        sigma = np.eye(numresp)

    # Set default prior for the shape parameter.
    if eta is None:
        eta = numresp

    featfun = featfun or (lambda x: x)

    # Generate the segment boundaries.
    changepoints = np.sort(
        np.random.choice(np.arange(numpoint - 1), numseg - 1, replace=False)
        + 1
    )
    boundaries = np.r_[0, changepoints, numpoint]

    # For each segment generate a set of predictor-response data.
    X = np.random.rand(numpoint, numpred)
    Y = np.empty((numpoint, numresp))
    for i0, i1 in pairwise(boundaries):
        # Generate the coefficient matrix and the noise covariance matrix.
        norm_inv_gamma = MatrixVariateNormalInvGamma(mu, omega, sigma, eta)
        coeff, noise = norm_inv_gamma.rand()
        fact = np.linalg.cholesky(noise).transpose()

        # Given a set of predictor data, generate a corresponding set of
        # response data.
        for i in range(i0, i1):
            eps = np.dot(np.random.randn(numresp), fact)
            Y[i] = featfun(np.dot(X[i], coeff)) + eps

    # Adjust the last element of the true boundaries for python's zero
    # indexing.
    boundaries[-1] -= 1

    return boundaries, X, Y


def plot_probability(axes, prob, scale=None, **arg):
    """Plot hypotheses probabilities as a raster image."""

    if scale is None:
        scale = lambda x: x  # noqa: E731

    k = max(np.shape(prob)) - 1
    (ind,) = np.nonzero(prob.max(axis=1) > 0)
    j = ind.max()

    # Plot the posterior probabilities over segment length hypotheses.
    axes.imshow(
        1.0 - prob[: j + 1],
        origin="lower",
        aspect="auto",
        extent=[scale(-0.5), scale(k + 0.5), -0.5, j + 0.5],
        interpolation="none",
        **arg,
    )


def batched_exact(it, n):
    for batch in batched(it, n):
        if len(batch) < n:
            break
        yield batch


def plot_segment_span(x, segments=None, **arg):
    """Plot segments as alternating vertical spans (rectangles)."""

    indices = range(len(x)) if segments is None else segments
    for i in batched_exact(indices, 2):
        plt.axvspan(x[i[0]], x[i[1]], **arg)


def plot_segment_boundaries(x, segments=None, **args):
    """Plot segment boundaries as vertical lines."""

    indices = range(len(x)) if segments is None else segments
    for i in indices:
        plt.axvline(x[i], **args)


def random_data():
    """Simple example with synthetic data."""

    # Set the size of the problem.
    numpred = 2
    numresp = 2
    numpoint = 200
    numseg = 5

    # Set parameters for generating the data.
    coeffparam = 0.5
    noiseparam = 5.0

    # Generate a sequence of segments and, for each segment, generate a set of
    # predictor-response data.
    segbound, X, Y = gen_random_data(
        numpred,
        numresp,
        numpoint,
        numseg,
        omega=coeffparam * np.eye(numpred),
        eta=noiseparam,
    )

    rate = float(numseg) / float(numpoint - numseg)

    # Compute the posterior probabilities over segment length hypotheses. Then,
    # find the most likely segmentation of the sequence.
    bcdm_probabilities = Bcdm(alg="sumprod", hazardfunc=rate)
    bcdm_segments = Bcdm(alg="maxprod", hazardfunc=rate)

    # Update the segment length hypotheses given the data.
    for x, y in zip(X, Y, strict=False):
        bcdm_probabilities.update(x, y)
        bcdm_segments.update(x, y)

    # Recover the hypothesis probabilities and back-trace to find the most
    # likely segmentation of the sequence.
    hypotheses_probability = bcdm_probabilities.infer()
    segments = bcdm_segments.infer()

    # Create subplots with shared X-axis.
    fig, (upperaxes, loweraxes) = plt.subplots(2, sharex=True)
    fig.subplots_adjust(hspace=0)

    # Plot the response data.
    t = np.arange(1, numpoint + 1)
    for i in range(numresp):
        upperaxes.plot(t, Y[:, i])

    # Plot the posterior probabilities over segment length hypotheses.
    plot_probability(loweraxes, hypotheses_probability, cmap=plt.cm.gray)

    # Plot the changes detected by the segmentation algorithm as alternating
    # coloured spans. Plot the true segment boundaries as vertical lines.
    for ax in (upperaxes, loweraxes):
        plt.sca(ax)
        plot_segment_span(
            t, segments, facecolor="y", alpha=0.2, edgecolor="none"
        )
        plot_segment_boundaries(t, segbound, color="k", linestyle=":")
        ax.set_xlim([0, numpoint])

    fig.canvas.manager.set_window_title("Randomly generated data")
    upperaxes.set_title("Randomly generated data")
    upperaxes.set_ylabel("Output values")
    loweraxes.set_xlabel("Observation")
    loweraxes.set_ylabel("Hypothesis probability")

    return fig


def square_wave(x):
    return np.sign(np.sin(x))


def sawtooth_wave(a, x):
    return 2 * ((x / a) - np.floor(0.5 + (x / a)))


def triangle_wave(a, x):
    return 2 * np.abs(sawtooth_wave(a, x)) - 1


def non_sinusoidal():
    """Simple example with triangular wave data."""

    rate = 0.001
    omega = 1.0e-3 * np.eye(2)
    sigma = 1.0e-6 * np.eye(3)
    samples = 1000

    def basis(x):
        return np.r_[1.0, x].reshape(1, -1)

    # Create input and outputs.
    X = np.linspace(0, 3 * 2 * np.pi, samples).reshape(samples, 1)
    Y = np.hstack(
        [
            square_wave(X),
            triangle_wave(2 * np.pi, X - np.pi / 2),
            sawtooth_wave(2 * np.pi, X + np.pi / 3),
        ]
    )

    # Create Gaussian noise.
    Y += np.vstack(
        [
            0.025 * np.random.randn(samples),
            0.1 * np.random.randn(samples),
            0.05 * np.random.randn(samples),
        ]
    ).T

    # Determine location of true boundaries.
    true_boundaries = np.hstack(
        (
            np.pi * np.arange(0, 7),
            np.pi * np.arange(0, 6) + np.pi / 2,
            2 * np.pi * np.arange(0, 4) + np.pi - np.pi / 3,
        )
    )

    true_boundaries = np.sort(true_boundaries[true_boundaries <= max(X)])

    # Compute the posterior probabilities over segment length hypotheses. Then,
    # find the most likely segmentation of the sequence.
    bcdm_probabilities = Bcdm(
        alg="sumprod",
        hazardfunc=rate,
        basisfunc=basis,
        omega=omega,
        sigma=sigma,
    )

    bcdm_segments = Bcdm(
        alg="maxprod",
        hazardfunc=rate,
        basisfunc=basis,
        omega=omega,
        sigma=sigma,
    )

    # Update the segment length hypotheses given the data.
    for x, y in zip(X, Y, strict=False):
        y = np.array([y])
        basis_t = lambda xt: basis(xt - x)  # noqa: E731,B023
        bcdm_probabilities.update(x, y, basisfunc=basis_t)
        bcdm_segments.update(x, y, basisfunc=basis_t)

    # Recover the hypothesis probabilities and back-trace to find the most
    # likely segmentation of the sequence.
    hypotheses_probability = bcdm_probabilities.infer()
    segments = bcdm_segments.infer()

    # Create subplots with shared X-axis.
    fig, (upperaxes, loweraxes) = plt.subplots(2, sharex=False)

    # Plot the response data.
    for i in range(Y.shape[1]):
        upperaxes.plot(X, Y[:, i])

    # Plot the posterior probabilities over segment length hypotheses.
    plot_probability(loweraxes, hypotheses_probability, cmap=plt.cm.gray)

    # Plot the changes detected by the segmentation algorithm as alternating
    # coloured spans. Plot the true segment boundaries as vertical lines.
    plt.sca(upperaxes)
    plot_segment_span(
        X.ravel(), segments, facecolor="y", alpha=0.2, edgecolor="none"
    )
    plot_segment_boundaries(true_boundaries, color="k", linestyle=":")

    plt.sca(loweraxes)
    plot_segment_span(segments, facecolor="y", alpha=0.2, edgecolor="none")
    plot_segment_boundaries(
        samples * true_boundaries / max(X), color="k", linestyle=":"
    )

    upperaxes.set_xlim([0, max(X)])
    loweraxes.set_xlim([0, len(X)])

    fig.canvas.manager.set_window_title("Triangular wave data")
    upperaxes.set_title("Triangular wave data")
    upperaxes.set_ylabel("Signal values")
    loweraxes.set_xlabel("Observation")
    loweraxes.set_ylabel("Hypothesis probability")

    return fig


def well_data():
    """Simple example with nuclear response data collected a well drilling

    Segment the well log data used in Fearnhead and Clifford (1996). This data
    consist of measurements of the nuclear magnetic response of underground
    rocks, collected during the drilling of a well bore. The data are composed
    of piecewise constant segments, each segment relating to a stratum with a
    single type of rock. The jump discontinuities between segments occur at the
    boundaries between rock strata.

    P. Fearnhead and P. Clifford, "Online Inference for Hidden Markov Models
    via Particle Filters," Journal of the Royal Statistical Society: Series B
    (Statistical Methodology), Vol. 65, Issue 4, pp. 887-889, November 2003.
    """

    loc = 1.0e5
    scale = 1.0e4
    rate = 1.0e-2

    # Format the data.
    Y = np.loadtxt(DATA_DIR / "well-data.txt", comments="#").reshape(-1, 1)
    X = np.ones_like(Y)

    loc = np.array([(loc,)])
    scale = np.array([(scale,)])

    # Compute the posterior probabilities over segment length hypotheses. Then,
    # find the most likely sequence segmentation.
    kwargs = {"hazardfunc": rate, "mu": loc, "sigma": scale}
    bcdm_probabilities = Bcdm(alg="sumprod", **kwargs)
    bcdm_segments = Bcdm(alg="maxprod", **kwargs)

    # Update the segment length hypotheses given the data.
    for x, y in zip(X, Y, strict=False):
        bcdm_probabilities.update(x, y)
        bcdm_segments.update(x, y)

    # Recover the hypothesis probabilities and back-trace to find the most
    # likely segmentation of the sequence.
    hypotheses_probability = bcdm_probabilities.infer()
    segments = bcdm_segments.infer()

    # Create subplots with shared X-axis.
    fig, (upperaxes, loweraxes) = plt.subplots(2, sharex=True)
    fig.subplots_adjust(hspace=0)

    # Plot the response data.
    t = np.arange(1, Y.size + 1)
    upperaxes.plot(t, Y[:])

    # Plot the posterior probabilities over segment length hypotheses.
    plot_probability(loweraxes, hypotheses_probability, cmap=plt.cm.gray)

    # Plot the changes detected by the segmentation algorithm as alternating
    # coloured spans. Plot the true segment boundaries as vertical lines.
    for ax in (upperaxes, loweraxes):
        plt.sca(ax)
        plot_segment_span(
            t, segments, facecolor="y", alpha=0.2, edgecolor="none"
        )
        ax.set_xlim([0, Y.size])

    fig.canvas.manager.set_window_title("Well log data")
    upperaxes.set_title("Well log data")
    upperaxes.set_ylabel("Nuclear magnetic response")
    loweraxes.set_xlabel("Measurement number")
    loweraxes.set_ylabel("Hypothesis probability")

    return fig


def index_data():
    """Simple example with equity index return data

    Segment the daily rates of return of a pair of equity indices between April
    23rd, 1993 and July 14th, 2003. The indices are the Cotation Assistee en
    Continu (CAC) and the Deutscher Aktienindex (DAX). The rates of return are
    computed based on the daily closing price of each index.
    """

    # Load data
    val = np.genfromtxt(
        DATA_DIR / "equity-index-data.csv", delimiter=",", names=True
    )

    # Select daily returns from CAC and DAX.
    index_names = ["cac", "dax"]
    X = np.ones((len(val), 1))
    Y = np.c_[*(val[name] for name in index_names)]
    assert Y.shape[1] == len(index_names)

    # Compute the posterior probabilities over segment length hypotheses. Then,
    # find the most likely sequence segmentation.
    kwargs = {
        "mu": np.zeros([1, Y.shape[1]]),  # 0% expected rate of return
        "sigma": 1.0e-4 * np.eye(Y.shape[1]),  # 1% expected volatility
        "maxhypot": 50,
        "hazardfunc": 1.0e-2,  # 1% expected hazard rate
        "minprob": 1.0e-16,
    }
    bcdm_probabilities = Bcdm(alg="sumprod", **kwargs)
    bcdm_segments = Bcdm(alg="maxprod", **kwargs)

    # Update the segment length hypotheses given the data.
    for x, y in zip(X, Y, strict=False):
        bcdm_probabilities.update(x, y)
        bcdm_segments.update(x, y)

    # Recover the hypothesis probabilities and back-trace to find the most
    # likely segmentation of the sequence.
    hypotheses_probability = bcdm_probabilities.infer()
    segments = bcdm_segments.infer()

    # Create subplots with shared X-axis.
    fig, (upperaxes, loweraxes) = plt.subplots(2, sharex=True)
    fig.subplots_adjust(hspace=0)

    # Plot the response data.
    t = np.arange(1, len(val) + 1)
    upperaxes.plot(t, Y[:])

    # Plot the posterior probabilities over segment length hypotheses.
    plot_probability(loweraxes, hypotheses_probability, cmap=plt.cm.gray)

    # Plot the changes detected by the segmentation algorithm as alternating
    # coloured spans. Plot the true segment boundaries as vertical lines.
    for ax in (upperaxes, loweraxes):
        plt.sca(ax)
        plot_segment_span(
            t, segments, facecolor="y", alpha=0.2, edgecolor="none"
        )
        ax.set_xlim([0, len(val)])

    fig.canvas.manager.set_window_title("Equity index data")
    upperaxes.set_title("Equity index data")
    upperaxes.set_ylabel("Rate of return")
    loweraxes.set_xlabel("Trading day")
    loweraxes.set_ylabel("Hypothesis probability")
    upperaxes.legend([name.upper() for name in index_names])

    return fig


def save_fig(name, fig):
    fig.savefig(f"{name}.png")


EXAMPLES = {
    "random": random_data,
    "non-sinusoidal": non_sinusoidal,
    "well": well_data,
    "index": index_data,
}


def parse_examples(value):
    examples = {}
    for item in value.split(","):
        if item == "all":
            return EXAMPLES
        example = EXAMPLES.get(item)
        if not example:
            raise click.UsageError(f"Invalid example '{item}'")
        examples[item] = example

    return examples


def get_plot_image_paths(prefix, names, overwrite):
    prefix = Path(prefix)
    if prefix.is_dir():
        parent = prefix
        filename_prefix = ""
    else:
        parent = prefix.parent
        if not parent.exists():
            raise click.ClickException(f"Directory does not exist: {parent}")
        filename_prefix = prefix.name

    paths = {}
    for name in names:
        path = parent / f"{filename_prefix}{name}.png"
        if not overwrite and path.exists():
            raise click.ClickException(
                f"File exists, re-run with --overwrite/-f to overwrite: {path}"
            )
        paths[name] = path

    return paths


@click.command()
@click.option(
    "--prefix",
    "-p",
    help=f"""Write plots images to paths with the format '<prefix><name>.png'
    where <name> is the name of the example
    (e.g. '{next(iter(EXAMPLES))}').""",
)
@click.option("--overwrite/--no-overwrite", "-f/", default=False)
@click.option(
    "--show",
    is_flag=True,
    help="""Whether to show the plots when --prefix/-p is passed.""",
)
@click.option(
    "--examples",
    "-e",
    default="all",
    help=f"""Comma-separated values of examples to run, or "all" to run all
    (the default). Available examples are: {",".join(EXAMPLES)}.""",
    callback=lambda c, p, value: parse_examples(value),
)
def main(prefix, overwrite, show, examples):
    image_paths = (
        get_plot_image_paths(prefix, examples.keys(), overwrite)
        if prefix
        else None
    )

    # Create a basic console logger.
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    for name, example in examples.items():
        logger.info(f"Running {name} data example...")
        fig = example()
        if image_paths:
            path = image_paths[name]
            fig.savefig(path)
            logger.info("Saved %s data plot to '%s'", name, path)
            if not show:
                plt.close(fig)

    if show or not image_paths:
        plt.show()


if __name__ == "__main__":
    main()
