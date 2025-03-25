from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.image import AxesImage

from bayesian_change_detection import multivariate_bcdm

FIGSIZE = (20, 10)


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

    kwargs = {
        "aspect": "auto",
        "origin": "lower",
        "cmap": "gray_r",
        "norm": LogNorm(vmin=1e-4, vmax=1),
        "extent": (x0, x1, 0, len(posterior)),
    } | kwargs
    return posterior, ax.imshow(posterior, **kwargs)  # type: ignore


def triangular(rng: np.random.Generator):
    """Simple example with triangular wave data."""

    samples = 500

    # Create input and outputs.
    X = np.linspace(0, 3 * 2 * np.pi, samples)
    noise = 0.1 * rng.standard_normal(samples)
    Y = -np.arcsin(np.sin(X)) * 2 / np.pi + noise
    true_changepoints = np.pi * np.arange(0, 6) + np.pi / 2

    res = multivariate_bcdm(
        np.c_[np.ones_like(X), X],
        Y,
        hazard=0.02,
        prior_cov=1e6,
        prior_shape=1e-3,
        prior_scale=1e-6,
    )

    axes = plt.subplots(2, sharex=True, figsize=FIGSIZE)[1]
    axes[0].plot(X, Y, "-o")
    posterior, im = plot_posterior_probabilities(
        axes[1], res.log_posterior, xlim=X[[0, -1]]
    )
    axes[1].plot(X, posterior.argmax(axis=0), color="red")
    changepoints = res.changepoints()
    for ax in axes:
        for cp in true_changepoints:
            ax.axvline(cp, color="red", linestyle="--")
        for idx in changepoints:
            ax.axvline(X[idx], color="green", linestyle="--")
    for ax in axes[1:]:
        ax.set_ylabel("Run length")

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


def random_piecewise(rng: np.random.Generator):
    T = 500  # Number of observations.
    hazard = 1 / 100  # Constant prior on changepoint probability.
    mean0 = 0  # The prior mean on the mean parameter.
    var0 = 2  # The prior variance for mean parameter.
    varx = 1  # The known variance of the data.

    data, true_changepoints = generate_random_piecewise_data(
        rng, varx, mean0, var0, T, hazard
    )
    y = np.asarray(data)
    res = multivariate_bcdm(
        np.ones_like(y),
        y,
        prior_cov=var0,
        hazard=hazard,
    )

    axes = plt.subplots(2, 1, sharex=True, figsize=FIGSIZE)[1]
    axes[0].plot(data, "-o")
    change_points = res.changepoints()
    for ax in axes:
        for cp in true_changepoints:
            ax.axvline(cp, color="red", linestyle="--")
        for cp in change_points:
            ax.axvline(cp, color="green", linestyle="--")

    posterior, im = plot_posterior_probabilities(
        axes[1],
        res.log_posterior,
        trim_zero=False,
    )
    axes[1].plot(posterior.argmax(axis=0))
    plt.tight_layout()
    plt.colorbar(im, ax=axes)


EXAMPLES = {
    "random": random_piecewise,
    "triangular": triangular,
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
        EXAMPLES[example](np.random.default_rng(42))

    if output:
        plt.savefig(output, format=None if output.suffix else "png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
