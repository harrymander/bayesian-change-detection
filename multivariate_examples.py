from itertools import batched, chain, pairwise
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.image import AxesImage

from bayesian_change_detection import multivariate_bcdm
from bayesian_change_detection.multivariate import (
    NigParams,
)

FIGSIZE = (20, 10)
DATA_DIR = Path(__file__).parent / "data"


def plot_posterior_probabilities(
    ax: Axes,
    log_posterior: np.ndarray,
    *,
    xlim=None,
    trim_zero: bool = True,
    **kwargs,
) -> tuple[np.ndarray, AxesImage]:
    samples = len(log_posterior)
    if xlim is None:
        x0, x1 = (0, samples - 1)
    else:
        x0, x1 = xlim

    posterior = np.exp(log_posterior)
    if trim_zero:
        zeros = np.where(np.isclose(posterior, 0).all(axis=1))[0]
        if zeros.size:
            zero_row = zeros[0]
            posterior = posterior[:zero_row]

    ax.set_ylabel("Run length")
    kwargs = {
        "aspect": "auto",
        "origin": "lower",
        "cmap": "gray_r",
        "norm": LogNorm(vmin=1e-4, vmax=1),
        "extent": (x0, x1, 0, len(posterior)),
    } | kwargs
    return posterior, ax.imshow(posterior, **kwargs)  # type: ignore


def triangular() -> None:
    """Simple example with triangular wave data."""
    rng = np.random.default_rng(42)

    samples = 500

    # Create input and outputs.
    t = np.linspace(0, 3 * 2 * np.pi, samples)
    noise = 0.1 * rng.standard_normal(samples)
    Y = -np.arcsin(np.sin(t)) * 2 / np.pi + noise
    true_changepoints = np.pi * np.arange(0, 6) + np.pi / 2

    X = np.c_[np.ones_like(t), t]  # add bias (intercept) term
    prior = NigParams.from_priors(
        2,
        cov=1e6,
        shape=1e-3,
        scale=1e-6,
    )
    res = multivariate_bcdm(X, Y, prior=prior, hazard=0.02)

    axes = plt.subplots(2, sharex=True, figsize=FIGSIZE)[1]
    axes[0].plot(t, Y, "-o")
    posterior, im = plot_posterior_probabilities(
        axes[1], res.log_posterior, xlim=t[[0, -1]]
    )
    axes[1].plot(t, posterior.argmax(axis=0), color="green")
    changepoints = res.changepoints()
    for ax in axes:
        for cp in true_changepoints:
            ax.axvline(cp, color="red", linestyle="--")
        for idx in changepoints:
            ax.axvline(t[idx], color="green", linestyle="--")
    for ax in axes[1:]:
        ax.set_ylabel("Run length")

    plot_segment_predictions(axes[0], prior, changepoints, X, Y, t)

    plt.tight_layout()
    plt.colorbar(im, ax=axes)


def generate_random_piecewise_data(rng, varx, mean0, var0, T, cp_prob):
    """Generate partitioned data of T observations according to constant
    changepoint probability `cp_prob` with hyperpriors `mean0` and `prec0`.
    """
    data = []
    cps = []
    meanx = mean0
    for t in range(0, T):
        if rng.random() < cp_prob:
            meanx = rng.normal(mean0, var0)
            cps.append(t)
        data.append(rng.normal(meanx, varx))
    return data, cps


def random_piecewise() -> None:
    rng = np.random.default_rng(42)

    T = 500  # Number of observations.
    hazard = 1 / 100  # Constant prior on changepoint probability.
    mean0 = 0  # The prior mean on the mean parameter.
    var0 = 2  # The prior variance for mean parameter.
    varx = 1  # The known variance of the data.

    data, true_changepoints = generate_random_piecewise_data(
        rng, varx, mean0, var0, T, hazard
    )
    y = np.asarray(data)
    prior = NigParams.from_priors(1, cov=var0)
    res = multivariate_bcdm(np.ones_like(y), y, prior=prior, hazard=hazard)

    axes = plt.subplots(2, 1, sharex=True, figsize=FIGSIZE)[1]
    axes[0].plot(data, "-o")
    changepoints = res.changepoints()
    for ax in axes:
        for cp in true_changepoints:
            ax.axvline(cp, color="red", linestyle="--")
        for cp in changepoints:
            ax.axvline(cp, color="green", linestyle="--")

    plot_segment_predictions(axes[0], prior, changepoints, np.ones_like(y), y)
    posterior, im = plot_posterior_probabilities(
        axes[1],
        res.log_posterior,
        trim_zero=False,
    )
    axes[1].plot(posterior.argmax(axis=0), color="green")
    plt.tight_layout()
    plt.colorbar(im, ax=axes)


def well_data() -> None:
    # Format the data.
    y = np.loadtxt(DATA_DIR / "well-data.txt", comments="#")
    assert y.ndim == 1
    x = np.ones_like(y)

    prior = NigParams.from_priors(1, mean=1e5, scale=1e4)
    hazard = 0.01
    results = multivariate_bcdm(x, y, hazard, prior)

    axes = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)[1]
    axes[0].plot(y)
    axes[0].set_ylabel("Nuclear magnetic response")
    axes[-1].set_xlabel("Time")
    plot_posterior_probabilities(axes[1], results.log_posterior)

    changepoints = results.changepoints()
    for cp in changepoints:
        for ax in axes:
            ax.axvline(cp, color="black", linewidth=0.5, linestyle="--")

    for segment in batched(chain((0,), changepoints, (len(x) - 1,)), 2):
        if len(segment) < 2:
            break
        for ax in axes:
            ax.axvspan(
                *segment,
                facecolor="yellow",
                edgecolor="none",
                alpha=0.3,
            )


def predict_segment(
    params: NigParams,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model = params.fit_regression(x, y)
    var = model.mvt_variance(x, axis="t")
    var[var < 0] = np.nan
    return model.mvt_mean(x, axis="t"), var


def atleast_2d_col(x: np.ndarray) -> np.ndarray:
    return np.expand_dims(x, 1) if x.ndim < 2 else x


def plot_segment_predictions(
    ax,
    prior: NigParams,
    changepoints: list[int],
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray | None = None,
) -> None:
    n = len(x)
    if t is None:
        t = np.arange(n)

    for i1, i2 in pairwise(chain((0,), changepoints, (n,))):
        xseg = x[i1:i2]
        yseg = y[i1:i2]
        tseg = t[i1:i2]
        pmean, pvar = predict_segment(prior, atleast_2d_col(xseg), yseg)
        ax.plot(tseg, pmean, color="green")
        sd = np.sqrt(pvar)
        ax.plot(tseg, pmean + sd, color="green", linestyle="--")
        ax.plot(tseg, pmean - sd, color="green", linestyle="--")


EXAMPLES = {
    "random": random_piecewise,
    "triangular": triangular,
    "well": well_data,
}


@click.command()
@click.argument(
    "examples",
    type=click.Choice(list(EXAMPLES.keys())),
    nargs=-1,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def main(examples, output: Path | None) -> None:
    if output and len(examples) != 1:
        raise click.UsageError(
            "--output can only be specified with a single example"
        )

    if not examples:
        examples = EXAMPLES.keys()
    for example in examples:
        click.echo(f"Running '{example}' example")
        EXAMPLES[example]()

    if output:
        plt.savefig(output, format=None if output.suffix else "png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
